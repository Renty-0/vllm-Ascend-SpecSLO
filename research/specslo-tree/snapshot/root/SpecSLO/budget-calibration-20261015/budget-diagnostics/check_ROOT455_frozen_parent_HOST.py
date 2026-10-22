"""Cold actual455 parent/runtime/owned callbacks; no model or device."""
import importlib.abc,json,sys,hashlib,datetime
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-M160-qualified-profile-whole-model-v455';sys.path.insert(0,str(P))
class NoModel(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):raise RuntimeError('HOST455 cold parent no model '+fullname)
sys.meta_path.insert(0,NoModel())
from host_env455 import setup
setup()
import guard455,runtime455
r=json.loads((P/'SOURCE455_RECEIPT.json').read_text());guard455.EXPECTED_FREEZE=r['freeze_sha256'];guard455.EXPECTED_ARCHIVE=r['archive_sha256']
_,sources,_=guard455.prepared();ns=guard455.execution_namespace();runtime=runtime455.build_runtime()
assert runtime['worker'].__globals__['worker']is runtime455.worker
assert ns['execute_locked'].__globals__ is ns
argv=guard455.launch_argv('HOST455-NOT-LAUNCHED');assert str(P/'startup455.sh')in argv
assert any(x=='SPECSLO_V455_ROOT_FREEZE_SHA='+r['freeze_sha256']for x in argv)
assert ns['ACTION']==runtime455.ACTION
# Exercise the actual constructed main, not only its module ACTION value.
import argparse
class StopAfterActualParser(Exception):pass
old_parse=argparse.ArgumentParser.parse_args;old_argv=sys.argv[:];seen=[]
def parse_and_stop(parser,*a,**k):
 value=old_parse(parser,*a,**k);assert value.action==runtime455.ACTION and value.tag=='HOST455-ACTUAL-MAIN'
 seen.append(vars(value));raise StopAfterActualParser()
try:
 argparse.ArgumentParser.parse_args=parse_and_stop
 sys.argv=['runtime455.py','--action',runtime455.ACTION,'--tag','HOST455-ACTUAL-MAIN']
 try:runtime['main']()
 except StopAfterActualParser:pass
finally:
 argparse.ArgumentParser.parse_args=old_parse;sys.argv=old_argv
assert len(seen)==1
storage=guard455.storage()
assert storage['aggregate_private_bytes']+256*1024**2<=3584*1024**2
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_HOST455_FROZEN_FULL_PARENT_ACTUAL_WORKER_AND_SUPERVISOR',source_members=len(sources),source_freeze=r['freeze_sha256'],source_archive=r['archive_sha256'],private_storage=storage,actual_model_constructors=0,NPU_used=False,all_actual_owned_callbacks_retained=True,actual_constructed_main_parser_accepted=True,only_local_operator_diagnostic=False,only_local_whole_model_diagnostic=True,new_NN_permission=False)
with(D/'ROOT455_FROZEN_PARENT_AND_WORKER_HOST_RESULT.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
