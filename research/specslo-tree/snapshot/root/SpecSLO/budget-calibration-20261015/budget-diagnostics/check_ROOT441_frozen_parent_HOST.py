"""Cold actual441 parent/runtime/owned callbacks; no model or device."""
import importlib.abc,json,sys,hashlib,datetime
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-M120-whole-target-gateup-layout-diagnostic-v441';sys.path.insert(0,str(P))
class NoModel(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):raise RuntimeError('HOST441 cold parent no model '+fullname)
sys.meta_path.insert(0,NoModel())
from host_env441 import setup
setup()
import guard441,runtime441
r=json.loads((P/'SOURCE441_RECEIPT.json').read_text());guard441.EXPECTED_FREEZE=r['freeze_sha256'];guard441.EXPECTED_ARCHIVE=r['archive_sha256']
_,sources,_=guard441.prepared();ns=guard441.execution_namespace();runtime=runtime441.build_runtime()
assert runtime['worker'].__globals__['worker']is runtime441.worker
assert ns['execute_locked'].__globals__ is ns
argv=guard441.launch_argv('HOST441-NOT-LAUNCHED');assert str(P/'startup441.sh')in argv
assert any(x=='SPECSLO_V441_ROOT_FREEZE_SHA='+r['freeze_sha256']for x in argv)
assert ns['ACTION']==runtime441.ACTION
storage=guard441.storage()
assert storage['aggregate_private_bytes']+256*1024**2<=3584*1024**2
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_HOST441_FROZEN_FULL_PARENT_ACTUAL_WORKER_AND_SUPERVISOR',source_members=len(sources),source_freeze=r['freeze_sha256'],source_archive=r['archive_sha256'],private_storage=storage,actual_model_constructors=0,NPU_used=False,all_actual_owned_callbacks_retained=True,only_local_operator_diagnostic=True,new_NN_permission=False)
with(D/'ROOT441_FROZEN_PARENT_AND_WORKER_HOST_RESULT.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
