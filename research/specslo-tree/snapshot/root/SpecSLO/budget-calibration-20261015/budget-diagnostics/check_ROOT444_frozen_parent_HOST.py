"""Cold actual444 parent/runtime/owned callbacks; no model or device."""
import importlib.abc,json,sys,hashlib,datetime
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-M120-padding-prefill-math-diagnostic-v444';sys.path.insert(0,str(P))
class NoModel(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):raise RuntimeError('HOST444 cold parent no model '+fullname)
sys.meta_path.insert(0,NoModel())
from host_env444 import setup
setup()
import guard444,runtime444
r=json.loads((P/'SOURCE444_RECEIPT.json').read_text());guard444.EXPECTED_FREEZE=r['freeze_sha256'];guard444.EXPECTED_ARCHIVE=r['archive_sha256']
_,sources,_=guard444.prepared();ns=guard444.execution_namespace();runtime=runtime444.build_runtime()
assert runtime['worker'].__globals__['worker']is runtime444.worker
assert ns['execute_locked'].__globals__ is ns
argv=guard444.launch_argv('HOST444-NOT-LAUNCHED');assert str(P/'startup444.sh')in argv
assert any(x=='SPECSLO_V444_ROOT_FREEZE_SHA='+r['freeze_sha256']for x in argv)
assert ns['ACTION']==runtime444.ACTION
storage=guard444.storage()
assert storage['aggregate_private_bytes']+256*1024**2<=3584*1024**2
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_HOST444_FROZEN_FULL_PARENT_ACTUAL_WORKER_AND_SUPERVISOR',source_members=len(sources),source_freeze=r['freeze_sha256'],source_archive=r['archive_sha256'],private_storage=storage,actual_model_constructors=0,NPU_used=False,all_actual_owned_callbacks_retained=True,only_local_operator_diagnostic=True,new_NN_permission=False)
with(D/'ROOT444_FROZEN_PARENT_AND_WORKER_HOST_RESULT.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
