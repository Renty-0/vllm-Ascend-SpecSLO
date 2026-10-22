"""Cold actual445 parent/runtime/owned callbacks; no model or device."""
import importlib.abc,json,sys,hashlib,datetime
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-M120-owned-padding-prefill-math-diagnostic-v445';sys.path.insert(0,str(P))
class NoModel(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):raise RuntimeError('HOST445 cold parent no model '+fullname)
sys.meta_path.insert(0,NoModel())
from host_env445 import setup
setup()
import guard445,runtime445
r=json.loads((P/'SOURCE445_RECEIPT.json').read_text());guard445.EXPECTED_FREEZE=r['freeze_sha256'];guard445.EXPECTED_ARCHIVE=r['archive_sha256']
_,sources,_=guard445.prepared();ns=guard445.execution_namespace();runtime=runtime445.build_runtime()
assert runtime['worker'].__globals__['worker']is runtime445.worker
assert ns['execute_locked'].__globals__ is ns
argv=guard445.launch_argv('HOST445-NOT-LAUNCHED');assert str(P/'startup445.sh')in argv
assert any(x=='SPECSLO_V445_ROOT_FREEZE_SHA='+r['freeze_sha256']for x in argv)
assert ns['ACTION']==runtime445.ACTION
storage=guard445.storage()
assert storage['aggregate_private_bytes']+256*1024**2<=3584*1024**2
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_HOST445_FROZEN_FULL_PARENT_ACTUAL_WORKER_AND_SUPERVISOR',source_members=len(sources),source_freeze=r['freeze_sha256'],source_archive=r['archive_sha256'],private_storage=storage,actual_model_constructors=0,NPU_used=False,all_actual_owned_callbacks_retained=True,only_local_operator_diagnostic=True,new_NN_permission=False)
with(D/'ROOT445_FROZEN_PARENT_AND_WORKER_HOST_RESULT.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
