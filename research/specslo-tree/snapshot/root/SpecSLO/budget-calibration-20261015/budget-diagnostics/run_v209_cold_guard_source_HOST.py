"""Cold source-only guard verification. No preparation, freeze, or child launch."""
import importlib.abc,importlib.util,json,sys,time,traceback
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');P=B/'tree-causal80-small-cohort-role-aware-audit-full-v209';G=B/'budget-diagnostics/v209-root-execution'
sys.path.insert(0,str(P))
class Firewall(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0]in('torch','torch_npu','triton','vllm','vllm_ascend','transformers','safetensors'):raise ImportError('V209 cold SOURCE forbids '+name)
sys.meta_path.insert(0,Firewall())
from base209 import read,sha,require
from check209 import local_files,external_files
start=time.perf_counter();before=local_files();extbefore=external_files();guards={str(p):sha(p)for p in G.iterdir()if p.is_file()}
result=dict(status='failed',candidate=str(P),Torch_imported=False,NPU_used=False,model_constructed=False,old_suites_rerun=False)
try:
 spec=importlib.util.spec_from_file_location('_V209_COLD_original_self_guard',G/'guard209.py');guard=importlib.util.module_from_spec(spec);sys.modules[spec.name]=guard;spec.loader.exec_module(guard)
 state=guard.source_state(archive=True);inputs=guard.guard_inputs();require(inputs==guards,'exact current guard inputs')
 require(not guard.FREEZE.exists()and not guard.ARCHIVE.exists()and not(P/'actual').exists(),'SOURCE only; own freeze/archive/actual not created')
 try:guard.prepared()
 except FileNotFoundError as e:
  require(Path(e.filename)==guard.FREEZE,'cold prepared fails only missing own ROOT freeze after all current source checks')
  refusal=dict(type=type(e).__name__,reason=str(e),missing_path=e.filename)
 else:raise RuntimeError('V209 prepared() must reject missing ROOT freeze')
 require(before==local_files()and extbefore==external_files()and guards=={str(p):sha(p)for p in G.iterdir()if p.is_file()},'current source pre/post exact')
 require(not any(n.split('.')[0]in('torch','torch_npu','triton','vllm','vllm_ascend')for n in sys.modules),'no backend imported')
 result.update(status='passed',source_state=state,guard_inputs=inputs,storage=guard.storage(),entry_blocked_without_ROOT_freeze=refusal,
  parent_archive_member_audit_performed=True,parent205_donor_failed_run_not_qualified=True,own3current_HOST=True,old205seven_suites_rerun=False,
  local_before=before,local_after=local_files(),external_before=extbefore,external_after=external_files(),guard_before=guards,guard_after={str(p):sha(p)for p in G.iterdir()if p.is_file()})
except BaseException:
 result['error']=traceback.format_exc()
result['elapsed_seconds']=time.perf_counter()-start;out=B/'budget-diagnostics/V209_COLD_ACTUAL_GUARD_SOURCE_STATE_HOST_RESULT.json'
require(not out.exists(),'preserve every cold result');out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(path=str(out),sha256=sha(out),status=result['status'],elapsed_seconds=result['elapsed_seconds'],error=result.get('error'))))
raise SystemExit(result['status']!='passed')
