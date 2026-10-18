"""Real Native allocator/metadata/forward protocol; CPU arithmetic model double."""
import ast,copy,importlib.util,json,sys,textwrap,types
from dataclasses import dataclass,replace
from pathlib import Path
from typing import Any,Sequence
P=Path(__file__).resolve().parent;B=P.parent
sys.path.append(str(B/'tree-static-q5-M120-full463-own-AR-v256'))
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
    sys.path[:0]=[str(P),str(B/'tree-static-q5-M128-projection-host-v249'),str(B/'tree-static-q5-immutable-index-cache-host-v247'),str(B/'tree-static-q5-long-prefix-origins-node-math-v246'),str(B/'tree-causal80-small-cohort-role-aware-audit-full-v209'),str(B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'),str(B/'tree-canonical-contiguous-full463-own-AR-v235'),str(B/'tree-high-normal-cap4-native-families-v148'),str(B/'tree-static-q5-projection-host-v243')]
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
    sys.path.append(str(B/'tree-static-q5-M120-node-math-v255'))
    sys.path.append(str(B/'tree-static-q5-M128-full463-own-AR-v251'))
    from bank251 import allocate
    prompts,evidence=allocate(engine,[[103+i]*127 for i in range(24)])
    need(evidence['unique_pages']==208 and len(prompts)==72,'actual real72 plus scratch64 bank allocation')
    from root_padding_forward231 import project
    from padding255 import pad
    from padding231 import allowed_dummy,root_only
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
            need(ms[0].actual_seq_lengths_q==tuple(range(5,121,5))and ms[0].sequence_lens==(2560,)*24,'actual CPU arithmetic physical input ABI')
            return (engine.model.compute_greedy_tokens(engine.model(xs[0],ps[0],ms[0]),vocabulary_size),)
    engine.graph_runner=Runner()
    from static_state257 import install
    install(engine,native);state=engine._branch_tail_state
    forward=native.NativePearlEngine.target_tree_forward
    extension_source=ast.parse((P/'extension257.py').read_text())
    observer=copy.deepcopy(next(n for n in ast.walk(extension_source)if isinstance(n,ast.FunctionDef)and n.name=='observed_forward'))
    observer_ns=dict(original_forward=forward)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[observer],type_ignores=[])),str(P/'extension257.py')+'::HOST-actual-observer','exec'),observer_ns)
    forward=observer_ns['observed_forward']
    engine._canonical257_phase='qualified';engine._dynamic_tree_generation=1;engine._canonical257_actual_calls={}
    cases=[]
    from oracle257 import cases as actual_cases
    for case in actual_cases():
        budgets=case['budgets']
        n=len(budgets);raw=tree.cached_cpu_tree_speculation_plan(2,4,127,4096,candidate_budget=8)
        plans=[tree.pack_selected_tree_plan(raw,list(range(min(b,4)))+([4+case['anchor']]if b==5 else[]))for b in budgets]
        roots=list(range(41,41+n));rows=[[103+17*i+j for j in range(b)]for i,b in enumerate(budgets)]
        pp,rr,dd,ss,real=pad(engine,plans,roots,rows,list(range(n)),[],[])
        out=forward(engine,pp,rr,dd,sequence_ids=ss,return_logits=False,real_tree_count=real)
        Q=n+sum(budgets)
        need(state.static_current['physical'][2] is engine.model.last_metadata,'actual physical metadata passed to CPU arithmetic model')
        need(state.last[2].tree_attention_mask.shape[2]==(10 if 5 in budgets else 9),'declared no-copy/copy graph mask capacity')
        need(out['query_count']==120 and out['logical_query_count']==Q and out['graph_padding_query_count']==120-Q,'exact physical/logical query conservation')
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
    from metadata257 import pack_q5,checked
    boundary_positions=[]
    for position in (0,127,128,1505,2555,2559):
        frame=dict(tokens=[41,0,0,0,0]+[0]*115,
            ids=[24]*5+[v for i in range(23)for v in [463+i]*5],
            positions=list(range(position,position+5))+list(range(5))*23,
            selected=[24]+list(range(463,486)))
        engine._prepare_attention_metadata(frame['ids'],frame['positions'],True)
        x,pos,meta=pack_q5(engine,frame,full=True);checked(meta)
        expected=torch.ones((24,1,10,4096),dtype=torch.bool)
        expected[0,0,0,:position+1]=False;expected[:,0,1:5,0]=False;expected[1:,0,0,0]=False
        need(torch.equal(meta.tree_attention_mask,expected),'complete actual own-reference masks including last real2559')
        need(x.numel()==120 and len(meta.actual_seq_lengths_q)==24,'no nonreal short tail')
        boundary_positions.append(position)
    result['actual_own_reference_boundaries']=boundary_positions
    import trace257,trace209,inspect
    namespace=trace257.namespace()
    need(all(inspect.signature(namespace[name])==inspect.signature(getattr(trace209,name))for name in('warm_to_fixedpoint','run_windows')),'all original positional/keyword-only defaults preserved')
    from runtime257 import project_source
    projected,changes=project_source((B/'tree-causal80-small-cohort-role-aware-audit-full-v209/runtime209.py').read_text())
    need(bool(projected)and len(changes)==13,'complete original runtime projection and inverse')
    need(len(engine._canonical257_actual_calls['1'])==227,'actual full observer recorded every real Native target')
    draft=types.SimpleNamespace(_canonical257_phase='qualified',is_draft=True,_branch_tail_state=types.SimpleNamespace(static_graph_calls=[]))
    draft_ns=dict(original_forward=lambda *args,**kwargs:None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(observer)],type_ignores=[])),str(P/'extension257.py')+'::HOST-draft-observer','exec'),draft_ns)
    need(draft_ns['observed_forward'](draft)is None,'original draft None protocol preserved')
    from reference257 import frame
    from audit_reference257 import expected_static_mask_sha256,expected_allocated_table
    prompts=[[11]*127,[12]*1505,[13]*2555];outputs=[[19]]*3
    for indices in ([0],[1],[2],[0,1,2]):
        data=frame(prompts,outputs,indices,future=0)
        need(len(data['tokens'])==120 and len(data['sequence_lens'])==24,'complete independent same-M120 frame')
        engine._prepare_attention_metadata(data['ids'],data['positions'],True)
        _,_,m=pack_q5(engine,dict(data,selected=data['ids'][:120:5]),full=True)
        import hashlib
        need(hashlib.sha256(m.tree_attention_mask.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==expected_static_mask_sha256(data['positions']),'actual full reference mask equals independent byte reconstruction')
    result.update(original_trace_defaults_preserved=True,complete_runtime_inverse=True,
        actual_observer_cases227=True,draft_None_protocol=True,independent_AR_mask_reconstruction=True)
    with(P/'HOST_FULL257_PROTOCOL_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],cases=len(result['cases']),boundaries=result['actual_own_reference_boundaries'])))
if __name__=='__main__':main()
