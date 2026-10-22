"""Launch a source-bound local-cost diagnostic; no production result or Git."""
from pathlib import Path
import datetime,hashlib,json,os,sys
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-M120-linear-layout-cost-diagnostic-v440';sys.path.insert(0,str(P))
from host_env440 import setup
setup()
record=json.loads((P/'SOURCE440_RECEIPT.json').read_text())
assert record['freeze_sha256']=='bf6b9b0d2704ff7c3973b028bc71f6f5b5ac8481db8bd3f62571f00ab745a3ad'
assert record['archive_sha256']=='6a46a565f534cf5b7f345e77f1487cfa0d6297be1c1c81492e359d21de491c1e'
control=json.loads((D/'ROOT440_FROZEN_PARENT_AND_WORKER_HOST_RESULT.json').read_text())
assert control['status'].startswith('PASS_HOST440')and control['NPU_used']is False
archive=json.loads((D/'ROOT440_NEGATIVE438_PRIVATE_ARCHIVAL_RESULT.json').read_text())
assert archive['public_storage_used']is False
import guard440
guard440.EXPECTED_FREEZE=record['freeze_sha256'];guard440.EXPECTED_ARCHIVE=record['archive_sha256'];guard440.prepared()
storage=guard440.storage();assert storage['aggregate_private_bytes']+256*1024**2<=3584*1024**2
TAG='M120-real-layout-padding-root-v440-local-operator-v1'
argv=[sys.executable,'-B',str(P/'guard440.py'),'--tag',TAG,'--expected-freeze-sha256',record['freeze_sha256'],'--expected-archive-sha256',record['archive_sha256']]
r=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tag=TAG,argv=argv,source_freeze=record['freeze_sha256'],source_archive=record['archive_sha256'],private_storage=storage,maximum_seconds=3600,private_tag_MiB=192,aggregate_MiB=3584,scope='actual real layer0 local matmul layout and pad128 device-event costs only',no_new_production_capability=True,no_model_or_communication_profile_replacement=True,Raw=None,Goodput=None,Git_pushes=0)
with(D/'ROOT440_LOCAL_OPERATOR_LAUNCH_REQUEST.json').open('x')as f:json.dump(r,f,indent=2);f.write('\n')
os.execve(sys.executable,argv,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
