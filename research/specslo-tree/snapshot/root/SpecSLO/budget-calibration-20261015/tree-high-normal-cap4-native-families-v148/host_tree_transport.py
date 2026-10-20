"""Tree control messages on the existing all-rank CPU coordination group.

Model forwards, device completion fences, finite checks and KV preflight are
retained. The complete structural/numerical consensus still runs on all ranks.
"""
import ast
from collections import Counter
import inspect
import math
import textwrap

import torch
import torch.distributed as dist


def group(engine):
    cached=getattr(engine,'_dynamic_tree_cpu_group_object',None)
    if cached is not None:
        return cached
    topology=engine.topology
    if len(topology.draft_ranks)!=1 or tuple(sorted(topology.verification_ranks))!=tuple(range(dist.get_world_size())):
        raise RuntimeError('Tree CPU control requires one draft rank and the complete verification world')
    selected=getattr(engine.groups,'verification_coordination_group',None)
    if selected is None or str(dist.get_backend(selected))!='gloo':
        raise RuntimeError('Tree CPU control requires the existing Gloo coordination group')
    engine._dynamic_tree_cpu_group_object=selected
    return selected


def count(engine,kind):
    counters=getattr(engine,'_dynamic_tree_cpu_counts',None)
    if counters is None:
        counters=engine._dynamic_tree_cpu_counts=Counter()
    counters[kind]+=1


def broadcast(engine,tensor,*,src,kind):
    if tensor.device.type!='cpu':
        raise RuntimeError('CPU control broadcast requires a host tensor')
    if kind in ('arrival_clock','prefill_clock','cycle_clock') and engine.device.type=='npu':
        # The previous device clock publication also completed earlier local
        # KV work. Preserve that ordering without an NPU control collective.
        torch.npu.current_stream().synchronize()
        count(engine,'clock_local_fence')
    dist.broadcast(tensor,src=src,group=group(engine))
    count(engine,kind)


def reduce(engine,tensor,*,op,kind):
    # Copy the actual device flags, rather than replacing or skipping a vote.
    host=tensor.cpu()
    dist.all_reduce(host,op=op,group=group(engine))
    count(engine,kind)
    return host


def barrier(engine):
    dist.barrier(group=group(engine))
    count(engine,'barrier')


def verdict(engine,output,plans,target_compute_ms=0.0):
    if not plans:
        return [],[],0.0
    if not math.isfinite(target_compute_ms) or target_compute_ms<0:
        raise ValueError('target tree compute time must be finite and non-negative')
    depth=max(int(plan.depth) for plan in plans)
    batch=len(plans)
    size=batch*(depth+1+depth)+1
    leader=engine.topology.target_leader_rank
    if engine.rank==leader:
        if output is None:
            raise RuntimeError('Target leader did not produce a tree verdict')
        if tuple(output.token_ids.shape)!=(batch,depth+1):
            raise RuntimeError('Tree verifier output has an invalid token shape')
        if tuple(output.accepted_node_indices.shape)!=(batch,depth):
            raise RuntimeError('Tree verifier output has an invalid acceptance shape')
        message=torch.cat((output.token_ids.to(device='cpu',dtype=torch.long).reshape(-1),
                           output.accepted_node_indices.to(device='cpu',dtype=torch.long).reshape(-1),
                           torch.tensor([round(target_compute_ms*1000.0)],dtype=torch.long)))
    else:
        message=torch.empty(size,dtype=torch.long)
    broadcast(engine,message,src=leader,kind='verdict')
    token_end=batch*(depth+1)
    return (message[:token_end].reshape(batch,depth+1).tolist(),
            message[token_end:-1].reshape(batch,depth).tolist(),
            int(message[-1])/1000.0)


class Rewrite(ast.NodeTransformer):
    def __init__(self,broadcasts,*,reductions=None,change_barrier=False,cpu_verdict=False):
        self.broadcasts=broadcasts
        self.reductions=reductions or {}
        self.change_barrier=change_barrier
        self.cpu_verdict=cpu_verdict
        self.changes=Counter()

    def visit_Assign(self,node):
        self.generic_visit(node)
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in self.broadcasts:
            if not isinstance(node.value,ast.Call):
                raise AssertionError('Expected a standalone control tensor constructor')
            devices=[kw for kw in node.value.keywords if kw.arg=='device']
            assert len(devices)==1 and ast.unparse(devices[0].value)=='self.device'
            devices[0].value=ast.Constant('cpu')
            self.changes['host_constructor']+=1
        return node

    def visit_Call(self,node):
        self.generic_visit(node)
        if self.cpu_verdict and isinstance(node.func,ast.Attribute) and ast.unparse(node.func)=='self._verify_tree_outputs_host':
            assert ast.unparse(node.args[-1])=='self.device'
            node.args[-1]=ast.Call(func=ast.Attribute(value=ast.Name('torch',ast.Load()),attr='device',ctx=ast.Load()),
                                  args=[ast.Constant('cpu')],keywords=[])
            self.changes['host_verdict_output']+=1
        return node

    def visit_Expr(self,node):
        self.generic_visit(node)
        call=node.value
        if not isinstance(call,ast.Call):
            return node
        name=ast.unparse(call.func)
        variable=call.args[0].id if call.args and isinstance(call.args[0],ast.Name) else None
        if name=='dist.broadcast' and variable in self.broadcasts:
            assert len(call.args)==1 and {kw.arg for kw in call.keywords}=={'src'}
            call.func=ast.Name('_dt_cpu_broadcast',ast.Load())
            call.args.insert(0,ast.Name('self',ast.Load()))
            call.keywords.append(ast.keyword('kind',ast.Constant(self.broadcasts[variable])))
            self.changes['broadcast']+=1
        elif name=='dist.all_reduce' and variable in self.reductions:
            assert len(call.args)==1 and {kw.arg for kw in call.keywords}=={'op'}
            call.func=ast.Name('_dt_cpu_reduce',ast.Load())
            call.args.insert(0,ast.Name('self',ast.Load()))
            call.keywords.append(ast.keyword('kind',ast.Constant(self.reductions[variable])))
            self.changes['reduce']+=1
            return ast.Assign(targets=[ast.Name(variable,ast.Store())],value=call)
        elif name=='dist.barrier' and self.change_barrier:
            assert not call.args and not call.keywords
            call.func=ast.Name('_dt_cpu_barrier',ast.Load())
            call.args=[ast.Name('self',ast.Load())]
            self.changes['barrier']+=1
        return node


def patch_decode_source(source):
    rewrite=Rewrite({'now_tensor':'arrival_clock','refill_elapsed':'prefill_clock',
                     'elapsed_tensor':'cycle_clock'},
                    reductions={'preflight_status':'commit_consensus','rank_flags':'failure_detail'},
                    change_barrier=True,cpu_verdict=True)
    tree=rewrite.visit(ast.parse(source))
    assert rewrite.changes=={'host_constructor':5,'broadcast':5,'reduce':2,
                             'barrier':1,'host_verdict_output':1},rewrite.changes
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'


def install(native):
    cls=native.NativePearlEngine
    namespace=vars(native)
    namespace.update(_dt_cpu_broadcast=broadcast,_dt_cpu_reduce=reduce,_dt_cpu_barrier=barrier)
    source=textwrap.dedent(inspect.getsource(cls._exchange_spec_rhythm_tree_candidates))
    rewrite=Rewrite({'message':'proposal'})
    tree=rewrite.visit(ast.parse(source))
    assert rewrite.changes=={'host_constructor':2,'broadcast':1},rewrite.changes
    exec(compile(ast.fix_missing_locations(tree),__file__+':proposal','exec'),namespace)
    cls._exchange_spec_rhythm_tree_candidates=namespace['_exchange_spec_rhythm_tree_candidates']
    cls._broadcast_spec_rhythm_tree_verdict=verdict
    previous_seal=cls.seal_graph_cache
    previous_metrics=cls.graph_metrics
    baseline={}
    def sealed(engine):
        result=previous_seal(engine)
        baseline.clear();baseline.update(getattr(engine,'_dynamic_tree_cpu_counts',{}))
        return result
    def metrics(engine):
        values=getattr(engine,'_dynamic_tree_cpu_counts',{})
        return previous_metrics(engine)|{'dynamic_tree_CPU_transport_enabled':1,
            **{'dynamic_tree_CPU_'+key:int(value) for key,value in values.items()},
            **{'dynamic_tree_measured_CPU_'+key:int(value-baseline.get(key,0))
               for key,value in values.items()}}
    cls.seal_graph_cache=sealed
    cls.graph_metrics=metrics
