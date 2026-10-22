"""Cold source/ABI imports only. Never execute model guard/prepare/main."""
import ast,hashlib,importlib.abc,importlib.util,json,sys,time
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-bounded-parent4-shared4-eager-diagnostic-v199';G=D/'v199-root-execution'
sys.path.insert(0,str(P))
class Firewall(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend','transformers','safetensors'):raise ImportError('cold HOST backend denied '+name)
sys.meta_path.insert(0,Firewall())
from base199 import sha,require,read
start=time.perf_counter();before={str(p):sha(p)for p in P.glob('*.py')}
for name in('target199','protocol199','next199','runtime199','extension199','audit199'):__import__(name)
spec=importlib.util.spec_from_file_location('_v199_actual_guard_cold',G/'guard199.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
state=guard.source_state(archive=False);inputs=guard.guard_inputs();storage=guard.storage()
from check199 import require_ready
try:require_ready()
except FileNotFoundError as error:
 require(str(P/'FROZEN_SOURCE199.json')in str(error),'only absent own ROOT freeze blocks cold entry');blocked=dict(type=type(error).__name__,reason=str(error))
else:raise RuntimeError('unfrozen entry unexpectedly enabled')
require(before=={str(p):sha(p)for p in P.glob('*.py')},'no own source mutation')
require(not any(n.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend')for n in sys.modules),'no backend library imported')
row=dict(status='passed_source_only',candidate=str(P),source_state=state,guard_inputs=inputs,storage=storage,actual_new_module_imports=['target199','protocol199','next199','runtime199','extension199','audit199'],entry_blocked_without_ROOT_freeze=blocked,elapsed_seconds=time.perf_counter()-start,Torch_imported=False,NPU_used=False,model_constructed=False,old_suites_rerun=False,parent_archive_member_audit_performed=False,ROOT_prepare_must_verify_dependency_archives=True,production_permission=False,full463=False)
p=D/'V199_COLD_ACTUAL_GUARD_SOURCE_STATE_HOST_RESULT.json'
with p.open('x')as stream:json.dump(row,stream,indent=2,sort_keys=True);stream.write('\n')
print(json.dumps(dict(path=str(p),sha256=sha(p),elapsed_seconds=row['elapsed_seconds'],status=row['status'])))
