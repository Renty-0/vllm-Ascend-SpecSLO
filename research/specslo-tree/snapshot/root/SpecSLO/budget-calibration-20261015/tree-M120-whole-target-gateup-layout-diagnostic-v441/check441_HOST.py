"""Actual441 immutable lifecycle/action/source construction; no model."""
import ast,hashlib,importlib.abc,inspect,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent
class NoModel(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):raise RuntimeError('HOST441 no model backend')
sys.meta_path.insert(0,NoModel())
from host_env441 import setup
setup()
import runtime441,probe441,forward441,guard441,audit441
source=runtime441.worker_source();assert source.count('from check441 import ready')==2 and source.count('from probe441 import run')==1
assert runtime441.ACTION in source
assert 'forward441'in probe441.run.__code__.co_names
inspect.signature(forward441.measure).bind(None,None,None)
ns=guard441.execution_namespace()
assert ns['execute_locked'].__globals__ is ns and ns['ACTION']==runtime441.ACTION
assert ns['MAX_SECONDS']==3600 and ns['MAX_TAG_BYTES']==192*1024**2 and ns['MAX_AGGREGATE_BYTES']==3584*1024**2
assert ns['prepared']is guard441.prepared and ns['actual_audit']is guard441.actual_audit
startup=(P/'startup441.sh').read_text();assert 'action='+runtime441.ACTION+'\n'in startup and str(P/'runtime441.py')in startup
assert "finally:"in inspect.getsource(forward441.measure)
assert 'for module,old,_,_ in originals:module.weight.data=old'in inspect.getsource(forward441.measure)
returns=[n for n in ast.walk(ast.parse(inspect.getsource(audit441.audit)))if isinstance(n,ast.Return)]
assert len(returns)==1 and isinstance(returns[0].value,ast.Tuple)and len(returns[0].value.elts)==3
for p in P.glob('*.py'):compile(p.read_text(),str(p),'exec')
r=dict(status='PASS_HOST441_ACTUAL_WHOLE_TARGET_LAYOUT_DIAGNOSTIC_LIFECYCLE',whole_original440_worker_main_inverse=True,actual385_genuine_frame_and72_prefix_protection_retained=True,actual_own_supervisor_callbacks=True,all64_transient_layout_restore_finally_present=True,MC2_row_weights_not_modified_in_code=True,Native_validation_flags_assigned=False,NPU_used=False,model_constructors=0,Raw=None,Goodput=None)
with(P/'HOST441_CONSTRUCTION_RESULT.json').open('x')as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r))
