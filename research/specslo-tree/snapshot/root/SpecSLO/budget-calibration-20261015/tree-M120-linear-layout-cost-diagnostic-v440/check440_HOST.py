"""Actual source construction, API shapes and supervisor wiring, no model."""
import ast,hashlib,importlib.abc,inspect,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent
class NoModel(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):raise RuntimeError('HOST440 cannot import a model '+fullname)
sys.meta_path.insert(0,NoModel())
from host_env440 import setup
setup()
import runtime440,probe440,guard440,linear440,probe385,audit440
text=runtime440.worker_source();tree=ast.parse(text)
assert len([n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in('main','worker')])==2
assert 'from probe440 import run'in text and 'from probe385 import run'not in text
assert runtime440.ACTION in text and 'real-M120-deferred-chain-original-graph-qualifier'not in text
inspect.signature(probe440.run).bind(None,None,None,None,None)
inspect.signature(linear440.measure).bind(None,None,None)
assert 'linear440'in probe440.run.__code__.co_names
assert probe440.run.__globals__['__file__']==probe385.__file__
ns=guard440.execution_namespace()
assert ns['execute_locked'].__globals__ is ns
assert ns['MAX_SECONDS']==3600 and ns['MAX_TAG_BYTES']==192*1024**2 and ns['MAX_AGGREGATE_BYTES']==3584*1024**2
assert ns['ACTION']==runtime440.ACTION
assert ns['prepared']is guard440.prepared and ns['actual_audit']is guard440.actual_audit and ns['storage']is guard440.storage
startup=(P/'startup440.sh').read_text()
assert 'action='+runtime440.ACTION+'\n'in startup
assert str(P/'runtime440.py')in startup and str(P/'check440.py')in startup
assert 'VLLM_ASCEND_PEARL_TARGET_NZ_MODE=14 VLLM_ASCEND_PEARL_TARGET_LARGE_M_NZ_MIN_ROWS=161'in startup
assert len(linear440.VARIANTS)==4
returns=[n for n in ast.walk(ast.parse(inspect.getsource(audit440.audit)))if isinstance(n,ast.Return)]
assert len(returns)==1 and isinstance(returns[0].value,ast.Tuple)and len(returns[0].value.elts)==3
for f in P.glob('*.py'):compile(f.read_text(),str(f),'exec')
result=dict(status='PASS_HOST440_ACTUAL_FROZEN_PARENT_LIFECYCLE_AND_LOCAL_OPERATOR_DIAGNOSTIC_CONSTRUCTION',
    original385_whole_worker_and_main_inverse=True,actual_probe_signature_bound=True,
    actual_genuine_capture_neural_call_unchanged=True,genuine_extra_gate_up_capture_explicit=True,
    actual_supervisor_callbacks_own=True,runtime_seconds=3600,tag_MiB=192,aggregate_MiB=3584,
    no_model_or_communication_profile_installed=True,no_production_permission=True,
    NPU_used=False,model_constructors=0,Raw=None,Goodput=None,
    source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in P.glob('*.py')})
with(P/'HOST440_CONSTRUCTION_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items()if k!='source_sha256'}))
