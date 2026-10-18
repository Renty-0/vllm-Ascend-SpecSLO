"""Real Native allocator/metadata/forward protocol; CPU arithmetic model double."""
import ast,copy,importlib.util,json,sys,textwrap,types
from dataclasses import dataclass,replace
from pathlib import Path
from typing import Any,Sequence
P=Path(__file__).resolve().parent;B=P.parent
N=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
def need(v,m):
    if not v:raise AssertionError(m)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
def main():
    import torch
    torch.set_num_threads(1)
    for name in('vllm_ascend','vllm_ascend.spec_decode','vllm_ascend.spec_decode.pearl'):
        module=types.ModuleType(name);module.__path__=[];sys.modules[name]=module
    tree=load('vllm_ascend.spec_decode.pearl.tree',N/'tree.py');cache=load('_ROOT231_real_native_cache',N/'native_cache.py')
    ns=dict(torch=torch,dataclass=dataclass,replace=replace,Any=Any,Sequence=Sequence,NativeCacheAllocation=cache.NativeCacheAllocation)
    names={'make_attention_metadata','_is_contiguous_linear_tree_plan','make_tree_fia_mask'}
    nodes=[copy.deepcopy(n)for n in ast.walk(ast.parse((N/'native_model.py').read_text()))if isinstance(n,ast.FunctionDef)and n.name in names or isinstance(n,ast.ClassDef)and n.name=='NativeAttentionMetadata']
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(N/'native_model.py')+'::CPU-original','exec'),ns)
    module=types.ModuleType('vllm_ascend.spec_decode.pearl.native_model');module.__dict__.update(ns);sys.modules[module.__name__]=module
    methods={'_allocate_cache','_activate_cache_sequence','_ensure_cache_capacity','_release_cache','_cache_slot_mapping','_prepare_attention_metadata'}
    engine_source=(N/'native_engine.py').read_text();nodes=[copy.deepcopy(n)for n in ast.walk(ast.parse(engine_source))if isinstance(n,ast.FunctionDef)and n.name in methods]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(N/'native_engine.py')+'::CPU-original-cache','exec'),ns)
    sys.path[:0]=[str(P),str(B/'tree-static-q5-long-prefix-origins-node-math-v246'),str(B/'tree-causal80-small-cohort-role-aware-audit-full-v209'),str(B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'),str(B/'tree-canonical-contiguous-full463-own-AR-v235'),str(B/'tree-high-normal-cap4-native-families-v148'),str(B/'tree-static-q5-projection-host-v243')]
    import host_target_metadata as host
    host._uses_fused_attention=lambda device,attention:True # CPU ABI fixture predicate only; no NPU claim.
    class ArithmeticModel:
        def __init__(self):
            self.embed_tokens=types.SimpleNamespace(weight=torch.zeros(1));self.max_model_len=4096
            self.layers=[types.SimpleNamespace(self_attn=types.SimpleNamespace(block_table=torch.zeros((527,32),dtype=torch.int32),blocks_per_sequence=32,block_size=128,uses_paged_attention=True,key_cache=torch.zeros((544,128,1,1)),value_cache=torch.zeros((544,128,1,1)),_fused_infer_attention=lambda q,m:q))]
            self.attention_mask=torch.zeros((2048,2048),dtype=torch.int8)
            self.make_attention_metadata=types.MethodType(ns['make_attention_metadata'],self)
        def __call__(self,inputs,positions,metadata):
            self.last_metadata=metadata
            return torch.stack([inputs.float(),positions.float()],dim=1)
        def compute_greedy_tokens(self,h,vocabulary_size):return h[:,0].long()
    engine=types.SimpleNamespace(device=torch.device('cpu'),model=ArithmeticModel(),prefix_cache=cache.NativePrefixCache(544,32,128),
        _cache_storage_sequence_capacity=527,_tree_cache_capacity=544*128,_tree_graph_scratch_sequence_ids=tuple(range(463,527)),
        config=types.SimpleNamespace(kvcache_block_size=128,max_model_len=4096,enforce_eager=False,spec_rhythm_tree_width=2,spec_rhythm_tree_depth=4),cache_allocation=None,cache_block_tables=None,
        is_draft=False,target_vocab_size=151936,draft_vocab_size=151936)
    for name in methods:setattr(engine,name,types.MethodType(ns[name],engine))
    from bank231 import allocate
    prompts,evidence=allocate(engine,[[103+i]*127 for i in range(16)])
    need(evidence['unique_pages']==160 and len(prompts)==48,'actual real48 plus scratch64 bank allocation')
    from root_padding_forward231 import project
    from padding231 import pad,allowed_dummy,root_only
    node=next(n for n in ast.walk(ast.parse(engine_source))if isinstance(n,ast.FunctionDef)and n.name=='target_tree_forward')
    source=textwrap.dedent('\n'.join(engine_source.splitlines()[node.lineno-1:node.end_lineno]))
    before=('self.model.make_tree_attention_metadata(\n        plan_list,\n        roots,\n        padded_candidates,\n        self.cache_allocation.block_tables,')
    after=('_dynamic_tree_make_host_metadata(\n        self,\n        plan_list,\n        roots,\n        padded_candidates,')
    need(source.count(before)==1,'same exact original host metadata rewrite')
    source=source.replace(before,after)
    namespace=dict(ns,envs=types.SimpleNamespace(VLLM_ASCEND_SPECRHYTHM_TREE_GRAPH=True),
        dist=types.SimpleNamespace(is_initialized=lambda:False),pack_selected_tree_plan=tree.pack_selected_tree_plan,
        _root_only_padding_allowed231=allowed_dummy,attention_backend_identity=lambda metadata:'CPU-protocol-double',
        _dynamic_tree_make_host_metadata=lambda e,*args,**kwargs:host.make_host_tree_attention_metadata(e,*args,**dict(kwargs,allow_causal_fast_path=False)))
    native=types.ModuleType('_ROOT246_CPU_actual_forward');native.__dict__.update(namespace)
    exec(compile('from __future__ import annotations\n'+project(source),str(P/'check_adapter246_HOST.py')+'::CPU-actual-forward','exec'),native.__dict__)
    class Engine:pass
    Engine.target_tree_forward=native.target_tree_forward;native.NativePearlEngine=Engine
    active=Engine();vars(active).update(vars(engine));engine=active
    class Runner:
        last_target_execution=types.SimpleNamespace(used_aclgraph=True,mode='CPU_protocol_double',fallback_reason=None)
        def run_target_greedy(self,xs,ps,ms,vocabulary_size):
            need(ms[0].actual_seq_lengths_q==tuple(range(5,81,5))and ms[0].sequence_lens==(2560,)*16,'actual CPU arithmetic physical input ABI')
            return (engine.model.compute_greedy_tokens(engine.model(xs[0],ps[0],ms[0]),vocabulary_size),)
    engine.graph_runner=Runner()
    from static_state246 import install
    install(engine,native);state=engine._branch_tail_state
    forward=native.NativePearlEngine.target_tree_forward;cases=[]
    for budgets in([1],[5],[4]*15+[3],[4]*16,[1]*16,[5]*8):
        n=len(budgets);raw=tree.cached_cpu_tree_speculation_plan(2,4,127,4096,candidate_budget=8)
        plans=[tree.pack_selected_tree_plan(raw,list(range(min(b,4)))+([7]if b==5 else[]))for b in budgets]
        roots=list(range(41,41+n));rows=[[103+17*i+j for j in range(b)]for i,b in enumerate(budgets)]
        pp,rr,dd,ss,real=pad(engine,plans,roots,rows,list(range(n)),[],[])
        out=forward(engine,pp,rr,dd,sequence_ids=ss,return_logits=False,real_tree_count=real)
        Q=n+sum(budgets)
        need(state.static_current['physical'][2] is engine.model.last_metadata,'actual physical metadata passed to CPU arithmetic model')
        need(state.last[2].tree_attention_mask.shape[2]==(10 if 5 in budgets else 9),'declared no-copy/copy graph mask capacity')
        need(out['query_count']==80 and out['logical_query_count']==Q and out['graph_padding_query_count']==80-Q,'exact physical/logical query conservation')
        need(out['num_draft_tokens']==budgets and out['tree_count']==n,'original real budgets and owners unchanged')
        need(out['target_query_token_ids'].tolist()==[v for root,row in zip(roots,rows,strict=True)for v in [root]+row]and out['target_token_ids'].numel()==sum(budgets),'all real model outputs scattered to original Native order')
        need(out['cache_slot_mapping'].numel()==Q and out['cache_slot_mapping'].unique().numel()==Q,'actual real KV coordinate count and uniqueness')
        cases.append(dict(N=n,budgets=budgets,real_queries=Q,physical_segments=len(pp),root_only_padding=any(p.candidate_budget==0 for p in pp)))
    dummy=root_only(4096);need(allowed_dummy(dummy,row_id=16,real_count=16)and not allowed_dummy(dummy,row_id=15,real_count=16),'typed dummy is never a real request')
    try:forward(engine,[dummy],[0],[[]],sequence_ids=[463],return_logits=False,real_tree_count=1)
    except ValueError:real_zero_rejected=True
    else:raise AssertionError('real zero-budget plan must fail')
    fake=types.SimpleNamespace(**vars(dummy));need(not allowed_dummy(fake,row_id=16,real_count=16),'untyped zero-budget object rejected')
    need('torch_npu'not in sys.modules,'strictly CPU only')
    result=dict(status='HOST_actual_Native_forward_static_projection_and_original_real_outputs_passed',post_Native_padding_metadata_identity_checked=True,actual_constructor_tree_width2_depth4=True,unique_pages=evidence['unique_pages'],cases=cases,
        original_real_positive_budget_guard_preserved=real_zero_rejected,untyped_dummy_rejected=True,root_only_excluded_from_real_output_and_budget=True,
        original_Native_forward_executed=True,model_double='CPU input/position identity arithmetic and explicit graph protocol double; no neural/attention/ACLGraph computation',isolated_FIA_device_predicate_stub=True,
        NPU_used=False,runtime_installed=False,neural_qualified=False,performance_qualified=False)
    from static_reference246 import pack_q5 as own_pack
    from audit_reference245 import expected_static_mask_sha256
    import hashlib,inspect
    boundary_cases=[]
    for root_position in (0,127,128,1505,2555,2559):
        frame=dict(tokens=[41,0,0,0,0]+[0]*75,ids=[16]*5+[v for i in range(15)for v in [463+i]*5],
                   positions=list(range(root_position,root_position+5))+list(range(5))*15,
                   selected=[16]+list(range(463,478)))
        engine._prepare_attention_metadata(frame['ids'],frame['positions'],True)
        _,_,metadata=own_pack(engine,frame,full=True)
        raw=metadata.tree_attention_mask.contiguous().view(torch.uint8).numpy().tobytes()
        need(hashlib.sha256(raw).hexdigest()==expected_static_mask_sha256(frame['positions']),'actual Native reference mask equals independent HOST reconstruction')
        need(tuple(metadata.sequence_lens)==(2560,)*16,'exact static reference extent')
        boundary_cases.append(root_position)
    frame['positions']=list(range(2560,2565))+list(range(5))*15
    engine._prepare_attention_metadata(frame['ids'],frame['positions'],True)
    try:own_pack(engine,frame,full=True)
    except RuntimeError:out_of_extent_rejected=True
    else:raise AssertionError('live reference prefix beyond2560 must fail')
    import trace245,trace209
    namespace=trace245.namespace()
    need(all(inspect.signature(namespace[name])==inspect.signature(getattr(trace209,name))for name in ('warm_to_fixedpoint','run_windows')),'all original positional and keyword-only defaults preserved')
    from runtime245 import project_source
    original_runtime=(B/'tree-causal80-small-cohort-role-aware-audit-full-v209/runtime209.py').read_text()
    projected,changes=project_source(original_runtime)
    need(bool(projected)and len(changes)==11,'complete original runtime projection and inverse')
    result.update(static_own_reference_boundaries=boundary_cases,actual_mask_hash_matches_independent_reconstruction=True,
                  out_of_extent_reference_rejected=out_of_extent_rejected,all_original_trace_defaults_preserved=True,
                  original_runtime_projection_inverse_passed=True,full463_ran=False)
    with(P/'HOST_FULL245_PROTOCOL_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')

    print(json.dumps(result))
if __name__=='__main__':main()
