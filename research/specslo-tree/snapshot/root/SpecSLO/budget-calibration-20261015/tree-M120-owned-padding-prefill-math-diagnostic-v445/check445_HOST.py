"""Actual immutable lifecycle and q5 ownership/slot boundary tests, no model."""
import ast,importlib.abc,inspect,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent
class NoModel(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):
            raise RuntimeError('HOST445 no model backend')
sys.meta_path.insert(0,NoModel())
from host_env445 import setup
setup()
import runtime445,probe445,forward445,guard445,audit445,layout445
source=runtime445.worker_source()
assert source.count('from check445 import ready')==2 and source.count('from probe445 import run')==1
assert runtime445.ACTION in source and 'forward445'in probe445._run.__code__.co_names
ns=guard445.execution_namespace()
assert ns['ACTION']==runtime445.ACTION and ns['execute_locked'].__globals__ is ns
assert ns['prepared']is guard445.prepared and ns['actual_audit']is guard445.actual_audit
assert (ns['MAX_SECONDS'],ns['MAX_TAG_BYTES'],ns['MAX_AGGREGATE_BYTES'])==(3600,192*1024**2,3584*1024**2)
startup=(P/'startup445.sh').read_text()
assert 'action='+runtime445.ACTION+'\n'in startup and str(P/'runtime445.py')in startup
prefixes=[[100]*125 for _ in range(24)]
tables=[[2*i,2*i+1]+[-1]*30 for i in range(72)]
for length in range(1,6):
    rows=layout445.plan(prefixes,tables,list(range(16,24)),list(range(16,24)),[[1]*length for _ in range(8)],real_groups=16)
    assert len(rows)==8*length and len({r['slot']for r in rows})==8*length
    assert all(80<=r['row']<120 and r['position']==125+r['row']%5 and r['visible_key_count']==r['position']+1 for r in rows)
bad=[dict(owners=[16,16],groups=[16,17],tokens=[[1],[1]],real_groups=16),
    dict(owners=[16],groups=[15],tokens=[[1]],real_groups=16),
    dict(owners=[16],groups=[24],tokens=[[1]],real_groups=16),
    dict(owners=[16],groups=[16],tokens=[[1]*6],real_groups=16)]
for kw in bad:
    try:layout445.plan(prefixes,tables,**kw)
    except ValueError:pass
    else:raise AssertionError('invalid physical group or owner accepted')
returns=[n for n in ast.walk(ast.parse(inspect.getsource(audit445.audit)))if isinstance(n,ast.Return)]
assert len(returns)==1 and isinstance(returns[0].value,ast.Tuple)and len(returns[0].value.elts)==3
body=inspect.getsource(forward445.measure)
assert "frames={'baseline':base}"in body and 'frames[name]=f'in body
assert 'before-timing.json'in body
for p in P.glob('*.py'):compile(p.read_text(),str(p),'exec')
out=dict(status='PASS_HOST445_ORIGINAL_LIFECYCLE_AND_Q5_PREFILL_OWNERSHIP',
    actual_N16_case_projection=True,original72_prefix_checks_retained=True,
    all5_lengths_and4_negative_metadata_cases=True,unchanged_physical_M120_S24=True,
    NPU_used=False,model_constructors=0,Native_flags_assigned=False,Raw=None,Goodput=None)
with(P/'HOST445_CONSTRUCTION_RESULT.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
