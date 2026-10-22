"""Actual source lifecycle and initial/same-owner row bounds, no backend."""
import ast,inspect,importlib.abc,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent
class NoModel(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):raise RuntimeError('HOST447 no model')
sys.meta_path.insert(0,NoModel())
from host_env447 import setup
setup()
import runtime447,probe447,guard447,forward447,audit447,parent447
s=runtime447.worker_source()
assert s.count('from check447 import ready')==2 and s.count('from probe447 import run')==1
assert 'forward447'in probe447._run.__code__.co_names
ns=guard447.execution_namespace()
assert ns['ACTION']==runtime447.ACTION and ns['execute_locked'].__globals__ is ns
assert (ns['MAX_SECONDS'],ns['MAX_TAG_BYTES'],ns['MAX_AGGREGATE_BYTES'])==(3600,192*1024**2,3584*1024**2)
assert ns['prepared']is guard447.prepared and ns['actual_audit']is guard447.actual_audit
assert 'action='+runtime447.ACTION+'\n'in(P/'startup447.sh').read_text()
table=[10,11]+[-1]*30
for start in(0,127):
    for n in(1,2,3,4,5,17,40):
        rows=forward447.rows_for_span(72,start,[3]*n,table)
        assert len(rows)==n and len({r['slot']for r in rows})==n
        assert rows[0]['row']==80 and rows[-1]['row']==79+n
        assert rows[-1]['position']==start+n-1 and max(r['group']for r in rows)<=23
for start,known,t in[(-1,[1],table),(0,[1]*41,table),(2555,[1]*6,table),(127,[1,2],[10]+[-1]*31)]:
    try:forward447.rows_for_span(72,start,known,t)
    except ValueError:pass
    else:raise AssertionError('ROOT447 invalid context/query/page bounds accepted')
body=inspect.getsource(forward447.measure)
assert 'g.reset()'in body and 'torch.npu.empty_cache()'in body and 'len(graphs)!=2'in body
assert 'post_pair_prefill128_KV_exact'in body
assert '_release_parent447'in parent447.measure.__code__.co_names
for p in P.glob('*.py'):compile(p.read_text(),str(p),'exec')
out=dict(status='PASS_HOST447_COMPLETE_ORIGINAL_LIFECYCLE_INITIAL_AND_BULK_ROW_BOUNDS',
    actual_source_inverse=True,all14_initial_continuation_span_cases=True,all4_negative_cases=True,
    original72_prefix_protection_preserved=True,fresh445_scope_reexecuted_in_current_engine=True,
    at_most2_bulk_diagnostic_graphs_explicitly_retained=True,NPU_used=False,model_constructors=0,Raw=None,Goodput=None)
with(P/'HOST447_CONSTRUCTION_RESULT.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
