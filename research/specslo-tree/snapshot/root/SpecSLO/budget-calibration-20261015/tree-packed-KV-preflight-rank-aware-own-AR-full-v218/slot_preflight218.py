"""Keep exact preflight checks; read original packed target mapping once.

Original per-row device dtype conversions/cat only reconstruct a flat tensor
already supplied by target_output. Draft/staged-eager paths stay original.
Actual original/new results are compared during every component invocation;
shadow execution ends before any warm/zero/full performance generation.
"""
import ast,copy,json,textwrap,types
from collections import Counter
from pathlib import Path

B=Path('/root/SpecSLO/budget-calibration-20261015')
NATIVE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_engine.py')
CANONICAL=B/'tree-high-normal-cap4-native-families-v148/canonical_eager.py'
NAME='_preflight_spec_rhythm_tree_cache_commit'
_STATE=None


def need(value,message):
    if value is not True:raise RuntimeError('ROOT218 preflight: '+message)


def canonical_tree():
    source=NATIVE.read_text();tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='NativePearlEngine')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name==NAME)
    # Same complete source lines as inspect.getsource, then original dedent.
    block='\n'.join(source.splitlines()[fn.lineno-1:fn.end_lineno])+'\n'
    tree=ast.parse(textwrap.dedent(block))
    module=ast.parse(CANONICAL.read_text())
    install=next(n for n in module.body if isinstance(n,ast.FunctionDef)and n.name=='install')
    begin=next(i for i,n in enumerate(install.body)if isinstance(n,ast.Assign)
        and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='count')
    end=next(i for i,n in enumerate(install.body[begin:],begin)if isinstance(n,ast.Assert))
    selected=copy.deepcopy(install.body[begin:end+1]);namespace=dict(ast=ast,tree=tree)
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(CANONICAL)+'::original-transform','exec'),namespace)
    need(namespace['count']==1,'exact original canonical-eager deferred validation transform')
    return tree


def optimized_tree(tree):
    tree=copy.deepcopy(tree);changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign)and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)and node.targets[0].id=='slot_values':
            expected='torch.cat([mapping.to(dtype=torch.long) for _, mapping, _, _ in checked]).detach().cpu().tolist()'
            need(ast.unparse(node.value)==expected,'exact original per-row conversion/concatenation expression')
            node.value=ast.IfExp(test=ast.parse('not self.is_draft and target_output is not None',mode='eval').body,
                body=ast.parse("target_output['cache_slot_mapping'].detach().cpu().tolist()",mode='eval').body,
                orelse=node.value);changed+=1
    need(changed==1,'one explicit target-only mapping-read delta')
    return ast.fix_missing_locations(tree)


def functions(namespace):
    tree=canonical_tree();oldns=dict(namespace);newns=dict(namespace)
    exec(compile(ast.fix_missing_locations(copy.deepcopy(tree)),str(CANONICAL)+':preflight','exec'),oldns)
    exec(compile(optimized_tree(tree),str(Path(__file__))+'::packed-target-mapping','exec'),newns)
    return oldns[NAME],newns[NAME]


def fingerprint(code):
    return (code.co_code,code.co_names,code.co_varnames,code.co_freevars,code.co_cellvars,
        code.co_argcount,code.co_kwonlyargcount,code.co_posonlyargcount,code.co_flags,
        tuple(fingerprint(v)if isinstance(v,types.CodeType)else v for v in code.co_consts))


def compare_results(original,current):
    before,old_host=original;after,new_host=current
    need(old_host==new_host and before.keys()==after.keys(),'all actual HOST slots and pending owner sets exact')
    for owner in before:
        a,b=before[owner],after[owner]
        need(a.dtype==b.dtype and a.device==b.device and a.shape==b.shape
            and a.data_ptr()==b.data_ptr()and a.storage_offset()==b.storage_offset(),
            'same actual pending device slice and physical storage')
    return True


def install(native):
    global _STATE
    need(_STATE is None,'one explicit preflight hook per fresh worker')
    cls=native.NativePearlEngine;current=getattr(cls,NAME)
    base,candidate=functions(current.__globals__)
    need(fingerprint(base.__code__)==fingerprint(current.__code__),
        'actual installed preflight exactly original canonical-eager body before one mapping-read delta')
    need(getattr(cls,'_dynamic_tree_canonical_eager',False)is True,'current canonical eager already installed')
    state=dict(shadow=True,qualified=False,shadow_pairs=0,shadow_target_pairs=0,shadow_draft_pairs=0,
        actual_target_dtype_counts=Counter(),by_generation={})
    _STATE=state
    def preflight(self,controller,payloads,cache_mappings,target_indices,target_plans,target_output,*,return_host_mappings=False):
        args=(self,controller,payloads,cache_mappings,target_indices,target_plans,target_output)
        if state['shadow']:
            original=current(*args,return_host_mappings=True)
            value=candidate(*args,return_host_mappings=True)
            compare_results(original,value);state['shadow_pairs']+=1
            state['shadow_draft_pairs'if self.is_draft else'shadow_target_pairs']+=1
            if not self.is_draft and target_output is not None:
                state['actual_target_dtype_counts'][str(target_output['cache_slot_mapping'].dtype)]+=1
            return value if return_host_mappings else value[0]
        generation=getattr(self,'_dynamic_tree_generation',None)
        need(type(generation)is int and generation>=0,'actual current generation for execution accounting')
        counts=state['by_generation'].setdefault(str(generation),Counter());counts['calls']+=1
        if not self.is_draft and target_output is not None and target_indices:
            counts['direct_packed_target_calls']+=1
            counts['target_rows']+=len(target_indices)
        else:counts['unchanged_other_calls']+=1
        return candidate(*args,return_host_mappings=return_host_mappings)
    preflight.__name__=current.__name__;preflight._ROOT218_original=current
    setattr(cls,NAME,preflight)


def end_shadow(engine):
    need(_STATE is not None and _STATE['shadow']is True and _STATE['shadow_pairs']>0,
         'actual current-engine component preflight comparisons required')
    need(_STATE['shadow_draft_pairs'if engine.is_draft else'shadow_target_pairs']>0,
         'actual role original/new preflight values compared before performance')
    _STATE['shadow']=False;_STATE['qualified']=True
    return snapshot()


def snapshot():
    need(_STATE is not None,'installed preflight required')
    return json.loads(json.dumps(_STATE))


def audit_snapshot(state,rank,trace):
    need(state['shadow']is False and state['qualified']is True and state['shadow_pairs']>0,
         'actual original/new component comparison completed')
    role='shadow_draft_pairs'if rank==0 else'shadow_target_pairs'
    need(state[role]==state['shadow_pairs'],'only this actual model role in shadow comparison')
    row=state['by_generation'][str(trace['source_generation'])]
    calls=len(trace['normal_target_calls'])
    need(row['calls']>=calls and(row.get('direct_packed_target_calls',0)==(0 if rank==0 else calls)),
         'original all positive target calls use direct packed mapping, draft path unchanged')
    need(row['calls']==row.get('direct_packed_target_calls',0)+row.get('unchanged_other_calls',0),
         'exact original versus direct-path accounting')
    return True
