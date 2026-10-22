"""Bounded true whole-target layout diagnostic; no serving claim or Git."""
import datetime,json,os,sys
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-M120-initial-and-bulk-prefill-math-diagnostic-v446';sys.path.insert(0,str(P))
from host_env446 import setup
setup()
r=json.loads((P/'SOURCE446_RECEIPT.json').read_text())
assert r['freeze_sha256']=='b5ab08de22a892d6d8ecc71e62e91781de246a81292659e17272484cff490eb3'and r['archive_sha256']=='073c18e842b9e7acccf62e5904c541082b974225452f6cc23e36d9c93083f74a'
check=json.loads((D/'ROOT446_FROZEN_PARENT_AND_WORKER_HOST_RESULT.json').read_text());assert check['status'].startswith('PASS_HOST446')and check['NPU_used']is False
prior=json.loads((D/'ROOT445_TERMINAL_INDEPENDENT_PADDING_PREFILL_AUDIT_HOST.json').read_text());assert prior['original_supervisor_status']=='complete'and prior['all4idle']is True and prior['all3_all5_lengths_math_exact']is True
import guard446
guard446.EXPECTED_FREEZE=r['freeze_sha256'];guard446.EXPECTED_ARCHIVE=r['archive_sha256'];guard446.prepared();storage=guard446.storage();assert storage['aggregate_private_bytes']+256*1024**2<=3584*1024**2
TAG='M120-initial-and-bulk-prefill-root-v446-nine-cases-math-cost-v1'
argv=[sys.executable,'-B',str(P/'guard446.py'),'--tag',TAG,'--expected-freeze-sha256',r['freeze_sha256'],'--expected-archive-sha256',r['archive_sha256']]
request=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tag=TAG,argv=argv,source_freeze=r['freeze_sha256'],source_archive=r['archive_sha256'],private_storage=storage,maximum_seconds=3600,private_tag_MiB=192,aggregate_MiB=3584,scope='fresh445 scope then initial and continuation same-owner multigroup known spans1..40, independent serial references and whole-target device costs',Native_flags_assigned=False,model_parameter_storages_never_swapped=True,no_new_production_permission=True,Raw=None,Goodput=None,Git_pushes=0)
with(D/'ROOT446_PADDING_PREFILL_LAUNCH_REQUEST.json').open('x')as f:json.dump(request,f,indent=2);f.write('\n')
os.execve(sys.executable,argv,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
