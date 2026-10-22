"""Cold457 complete bytes, actual built worker/main, supervisor and storage."""
import argparse,datetime,hashlib,importlib.abc,json,os,sys
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-adaptive-own-geometry-node-math-v459'
class NoModel(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):raise RuntimeError('HOST459 cold no backend')
sys.meta_path.insert(0,NoModel());sys.path.insert(0,str(P))
from host_env459 import setup
setup()
import guard459,runtime459
r=json.loads((P/'SOURCE459_RECEIPT.json').read_text());guard459.EXPECTED_FREEZE=r['freeze_sha256'];guard459.EXPECTED_ARCHIVE=r['archive_sha256'];os.environ['SPECSLO_V459_ROOT_FREEZE_SHA']=r['freeze_sha256']
_,sources,readiness=guard459.prepared();ns=guard459.execution_namespace();runtime=runtime459.build_runtime()
assert ns['execute_locked'].__globals__ is ns and runtime['worker'].__globals__['worker'] is runtime459.worker
argv=guard459.launch_argv('HOST459-NOT-LAUNCHED');assert str(P/'startup459.sh')in argv
assert'SPECSLO_V459_ROOT_FREEZE_SHA='+r['freeze_sha256']in argv
assert ns['ACTION']==runtime459.ACTION
class StopAfterParse(Exception):pass
old_parse=argparse.ArgumentParser.parse_args;old_argv=sys.argv[:];seen=[]
def stop(parser,*a,**k):
 value=old_parse(parser,*a,**k);seen.append(vars(value));raise StopAfterParse()
try:
 argparse.ArgumentParser.parse_args=stop;sys.argv=['runtime459.py','--action',runtime459.ACTION,'--tag','HOST459-FROZEN-ACTUAL-MAIN']
 try:runtime['main']()
 except StopAfterParse:pass
finally:argparse.ArgumentParser.parse_args=old_parse;sys.argv=old_argv
assert seen==[dict(action=runtime459.ACTION,tag='HOST459-FROZEN-ACTUAL-MAIN')]
storage=guard459.storage();assert storage['aggregate_private_bytes']+256*1024**2<=3584*1024**2
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_HOST459_FROZEN_COMPLETE_CLOSURE_AND_ACTUAL_WORKER_MAIN',source_members=len(sources),
 source_freeze=r['freeze_sha256'],source_archive=r['archive_sha256'],private_storage=storage,
 all_actual_owned_callbacks_retained=True,actual_constructed_main_parser_accepted=True,unchanged_original_profile=True,
 original_cross_M120_failure_preserved=True,actual_model_constructors=0,NPU_used=False,NN_permission=False,Raw=None,Goodput=None)
with(D/'ROOT459_FROZEN_PARENT_AND_WORKER_HOST_RESULT.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
