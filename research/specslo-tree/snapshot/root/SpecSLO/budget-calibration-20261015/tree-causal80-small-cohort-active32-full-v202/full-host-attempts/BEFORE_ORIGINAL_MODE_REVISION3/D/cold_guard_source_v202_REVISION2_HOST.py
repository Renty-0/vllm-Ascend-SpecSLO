"""Cold actual source/checker/guard bindings; no prepare/freeze/model/entry."""
import ast,hashlib,importlib.abc,importlib.util,json,sys,time
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-causal80-small-cohort-active32-full-v202';G=D/'v202-root-execution'
sys.path.insert(0,str(P))
class Firewall(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend','transformers','safetensors'):raise ImportError('cold HOST backend denied '+name)
sys.meta_path.insert(0,Firewall())
from base202 import sha,read,require
start=time.perf_counter();before={str(p):sha(p)for p in P.glob('*.py')}
for name in('mapper202','bank202','target202','actor202','audit_usage202','component202','runtime202','scope202','extension202','audit202','trace202'):__import__(name)
from check202 import require_parent,require_ready
parents=require_parent(archive=False)
spec=importlib.util.spec_from_file_location('_actual202_guard_source_only',G/'guard202.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
state=g.source_state(archive=True);guard=g.guard_inputs();storage=g.storage()
try:require_ready()
except FileNotFoundError as error:
 require(str(P/'FROZEN_SOURCE202.json')in str(error),'only absent ownROOT freeze blocks entry');blocked=dict(type=type(error).__name__,reason=str(error))
else:raise RuntimeError('unfrozen entry unexpectedly enabled')
require(before=={str(p):sha(p)for p in P.glob('*.py')},'no source mutation')
require(not any(n.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend')for n in sys.modules),'no backend import')
r=dict(status='passed_source_only',candidate=str(P),source_state=state,guard_inputs=guard,storage=storage,parent_source_checks=parents,entry_blocked_without_ROOT_freeze=blocked,elapsed_seconds=time.perf_counter()-start,Torch_imported=False,NPU_used=False,model_constructed=False,old_suites_rerun=False,parent_archive_member_audit_performed=True,ROOT_prepare_must_verify_dependency_archives=True)
out=D/'V202_COLD_ACTUAL_GUARD_SOURCE_STATE_HOST_RESULT_REVISION2.json'
with out.open('x')as f:json.dump(r,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(path=str(out),sha256=sha(out),elapsed_seconds=r['elapsed_seconds'],status=r['status'])))
