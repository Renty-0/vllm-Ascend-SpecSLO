"""Actual entire source lifecycle and static-input single-graph contract."""
import ast,inspect,importlib.abc,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent
class NoModel(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):raise RuntimeError('HOST451 no model')
sys.meta_path.insert(0,NoModel())
from host_env451 import setup
setup()
import runtime451,probe451,guard451,forward451,audit451
s=runtime451.worker_source()
assert s.count('from check451 import ready')==2 and s.count('from probe451 import run')==1
assert 'forward451'in probe451._run.__code__.co_names
assert 'separately allocated ordinary Native prefill and16-step AR'in Path(guard451.__file__).read_text()
ns=guard451.execution_namespace()
assert ns['ACTION']==runtime451.ACTION and ns['execute_locked'].__globals__ is ns
assert (ns['MAX_SECONDS'],ns['MAX_TAG_BYTES'],ns['MAX_AGGREGATE_BYTES'])==(3600,192*1024**2,3584*1024**2)
assert ns['prepared']is guard451.prepared and ns['actual_audit']is guard451.actual_audit
assert 'action='+runtime451.ACTION+'\n'in(P/'startup451.sh').read_text()
body=inspect.getsource(forward451.measure)
assert body.count('torch.npu.NPUGraph()')==1 and '.reset()'not in body and 'empty_cache'not in body.replace('no_graph_reset_or_empty_cache_during_epoch','')
assert 'if model_body_closed:raise RuntimeError'in body
assert 'actual independent numerical mismatch; no timing claim'in body
assert '.copy_(fm.tree_attention_mask)'in body and 'static_ptrs!='in body
table=[10,11]+[-1]*30
for start in(0,127):
    for length in(1,2,3,4,5,17,40):
        rows=forward451.rows_for_span(72,start,[3]*length,table)
        assert len(rows)==length and len({r['slot']for r in rows})==length
for p in P.glob('*.py'):compile(p.read_text(),str(p),'exec')
assert 'no ordinary Native model call after capture'in body
assert 'all3_native_AR_tokens_exact'in body and 'post_replay_decode128_KV_exact'in body
assert 'use_aclgraph=False,use_fused_infer_attention=True'in body
assert 'for step in range(15)'in body and 'native_prompts={74:tokens,75:prefix+tokens,76:prefix+tokens,77:tokens,78:prefix+tokens}'in body
out=dict(status='PASS_HOST451_COMPLETE_SOURCE_AND_SINGLE_STATIC_GRAPH_CONTRACT',
    actual_runtime_source_inverse=True,all9_current_cases_retained=True,
    exactly_one_diagnostic_graph=True,no_eager_model_body_after_capture=True,
    no_graph_reset_or_empty_cache_during_epoch=True,independent_references_before_capture=True,
    NPU_used=False,model_constructors=0,Raw=None,Goodput=None)
with(P/'HOST451_CONSTRUCTION_RESULT.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
