"""Bounded true whole-target layout diagnostic; no serving claim or Git."""
import datetime,json,os,sys
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-M160-qualified-profile-whole-model-v455';sys.path.insert(0,str(P))
from host_env455 import setup
setup()
r=json.loads((P/'SOURCE455_RECEIPT.json').read_text())
assert r['freeze_sha256']=='ad040025cafaf5d2f6b1e22d5a7952154cd9bab579b8bfa64896ff82c280809f'and r['archive_sha256']=='64a6236daac945c331ac9275cf12f4e382e88ff21f186bdc1c8821349ff974f5'
check=json.loads((D/'ROOT455_FROZEN_PARENT_AND_WORKER_HOST_RESULT.json').read_text());assert check['status'].startswith('PASS_HOST455')and check['NPU_used']is False
prior=json.loads((D/'ROOT454_TERMINAL_INDEPENDENT_REAL32_MC2_AUDIT_HOST.json').read_text());assert prior['original_supervisor_status']=='complete'and prior['all4idle']is True and prior['all6_original_operator_gates_accepted']is True 
import guard455
guard455.EXPECTED_FREEZE=r['freeze_sha256'];guard455.EXPECTED_ARCHIVE=r['archive_sha256'];guard455.prepared();storage=guard455.storage();assert storage['aggregate_private_bytes']+256*1024**2<=3584*1024**2
TAG='M160-qualified-profile-root-v455-unfused-fullmodel-and120-cost-v1'
argv=[sys.executable,'-B',str(P/'guard455.py'),'--tag',TAG,'--expected-freeze-sha256',r['freeze_sha256'],'--expected-archive-sha256',r['archive_sha256']]
request=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tag=TAG,argv=argv,source_freeze=r['freeze_sha256'],source_archive=r['archive_sha256'],private_storage=storage,maximum_seconds=3600,private_tag_MiB=192,aggregate_MiB=3584,scope='strictly measured160 profile declared before actual constructor; compare full32B ordinary unfused control and optimized160, current120; independent full hidden/token/128KV and post-timed replay checks,9 balanced device event windows; no full463 serving claim',Native_flags_assigned=False,model_parameter_storages_never_swapped=True,no_new_production_permission=True,Raw=None,Goodput=None,Git_pushes=0)
with(D/'ROOT455_WHOLE_MODEL_LAUNCH_REQUEST.json').open('x')as f:json.dump(request,f,indent=2);f.write('\n')
os.execve(sys.executable,argv,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
