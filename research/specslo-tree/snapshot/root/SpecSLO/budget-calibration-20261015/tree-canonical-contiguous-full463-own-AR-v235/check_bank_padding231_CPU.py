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
    sys.path[:0]=[str(P),str(B/'tree-high-normal-cap4-native-families-v148')]
    import host_target_metadata as host
    host._uses_fused_attention=lambda device,attention:True # CPU ABI fixture predicate only; no NPU claim.
    class ArithmeticModel:
        def __init__(self):
            self.embed_tokens=types.SimpleNamespace(weight=torch.zeros(1));self.max_model_len=4096
            self.layers=[types.SimpleNamespace(self_attn=types.SimpleNamespace(block_table=torch.zeros((527,32),dtype=torch.int32),blocks_per_sequence=32,block_size=128,uses_paged_attention=True))]
            self.attention_mask=torch.zeros((2048,2048),dtype=torch.int8)
            self.make_attention_metadata=types.MethodType(ns['make_attention_metadata'],self)
        def __call__(self,inputs,positions,metadata):return torch.arange(inputs.numel()*2,dtype=torch.float32).reshape(-1,2)
        def compute_greedy_tokens(self,h,vocabulary_size):return h[:,0].long()
    engine=types.SimpleNamespace(device=torch.device('cpu'),model=ArithmeticModel(),prefix_cache=cache.NativePrefixCache(544,32,128),
        _cache_storage_sequence_capacity=527,_tree_cache_capacity=544*128,_tree_graph_scratch_sequence_ids=tuple(range(463,527)),
        config=types.SimpleNamespace(kvcache_block_size=128,max_model_len=4096,enforce_eager=True),cache_allocation=None,cache_block_tables=None,
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
    namespace=dict(ns,envs=types.SimpleNamespace(VLLM_ASCEND_SPECRHYTHM_TREE_GRAPH=False),
        dist=types.SimpleNamespace(is_initialized=lambda:False),pack_selected_tree_plan=tree.pack_selected_tree_plan,
        _root_only_padding_allowed231=allowed_dummy,attention_backend_identity=lambda metadata:'CPU-protocol-double',
        _dynamic_tree_make_host_metadata=lambda e,*args,**kwargs:host.make_host_tree_attention_metadata(e,*args,**dict(kwargs,allow_causal_fast_path=False)))
    exec(compile('from __future__ import annotations\n'+project(source),str(P/'root_padding_forward231.py')+'::CPU-actual-forward','exec'),namespace)
    forward=namespace['target_tree_forward'];cases=[]
    for budgets in([4]*15+[3],[4]*16,[1]*16,[5]*12+[3,1,1]):
        n=len(budgets);raw=tree.cached_cpu_tree_speculation_plan(2,4,127,4096,candidate_budget=8)
        plans=[tree.pack_selected_tree_plan(raw,list(range(min(b,4)))+([7]if b==5 else[]))for b in budgets]
        roots=list(range(41,41+n));rows=[[103+17*i+j for j in range(b)]for i,b in enumerate(budgets)]
        pp,rr,dd,ss,real=pad(engine,plans,roots,rows,list(range(n)),[],[])
        out=forward(engine,pp,rr,dd,sequence_ids=ss,return_logits=False,real_tree_count=real)
        Q=n+sum(budgets)
        need(out['query_count']==80 and out['logical_query_count']==Q and out['graph_padding_query_count']==80-Q,'exact physical/logical query conservation')
        need(out['num_draft_tokens']==budgets and out['tree_count']==n,'original real budgets and owners unchanged')
        need(out['target_query_token_ids'].tolist()==list(range(0,2*Q,2))and out['target_token_ids'].numel()==sum(budgets),'dummy contributes no real prediction or candidate')
        need(out['cache_slot_mapping'].numel()==Q and out['cache_slot_mapping'].unique().numel()==Q,'actual real KV coordinate count and uniqueness')
        cases.append(dict(N=n,budgets=budgets,real_queries=Q,physical_segments=len(pp),root_only_padding=any(p.candidate_budget==0 for p in pp)))
    dummy=root_only(4096);need(allowed_dummy(dummy,row_id=16,real_count=16)and not allowed_dummy(dummy,row_id=15,real_count=16),'typed dummy is never a real request')
    try:forward(engine,[dummy],[0],[[]],sequence_ids=[463],return_logits=False,real_tree_count=1)
    except ValueError:real_zero_rejected=True
    else:raise AssertionError('real zero-budget plan must fail')
    fake=types.SimpleNamespace(**vars(dummy));need(not allowed_dummy(fake,row_id=16,real_count=16),'untyped zero-budget object rejected')
    need('torch_npu'not in sys.modules,'strictly CPU only')
    result=dict(status='CPU_Native_allocator_metadata_forward_protocol_passed',unique_pages=evidence['unique_pages'],cases=cases,
        original_real_positive_budget_guard_preserved=real_zero_rejected,untyped_dummy_rejected=True,root_only_excluded_from_real_output_and_budget=True,
        original_Native_forward_executed=True,model_double='CPU arange arithmetic; no neural/attention computation',isolated_FIA_device_predicate_stub=True,
        NPU_used=False,runtime_installed=False,neural_qualified=False,performance_qualified=False)
    with(P/'CPU_BANK_PADDING231_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
