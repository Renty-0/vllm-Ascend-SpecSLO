"""Bounded true whole-target layout diagnostic; no serving claim or Git."""
import datetime,json,os,sys
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-M120-owned-padding-prefill-math-diagnostic-v445';sys.path.insert(0,str(P))
from host_env445 import setup
setup()
r=json.loads((P/'SOURCE445_RECEIPT.json').read_text())
assert r['freeze_sha256']=='3fece48e7ed37ae80ebac4d9744e3a46cbd50e03fbb8da1c8f3fee47a6afd8db'and r['archive_sha256']=='dc84c9c7f099cabf01bc706b6f5de1791dfa914e1ebbae22b43d1c08ec88eb69'
check=json.loads((D/'ROOT445_FROZEN_PARENT_AND_WORKER_HOST_RESULT.json').read_text());assert check['status'].startswith('PASS_HOST445')and check['NPU_used']is False
prior=json.loads((D/'ROOT441_TERMINAL_INDEPENDENT_WHOLE_TARGET_AUDIT_HOST.json').read_text());assert prior['original_supervisor_status']=='complete'and prior['all4_idle']is True and prior['full463']is False
import guard445
guard445.EXPECTED_FREEZE=r['freeze_sha256'];guard445.EXPECTED_ARCHIVE=r['archive_sha256'];guard445.prepared();storage=guard445.storage();assert storage['aggregate_private_bytes']+256*1024**2<=3584*1024**2
TAG='M120-N16-owned-padding-prefill-root-v445-lengths1to5-math-cost-v1'
argv=[sys.executable,'-B',str(P/'guard445.py'),'--tag',TAG,'--expected-freeze-sha256',r['freeze_sha256'],'--expected-archive-sha256',r['archive_sha256']]
request=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tag=TAG,argv=argv,source_freeze=r['freeze_sha256'],source_archive=r['archive_sha256'],private_storage=storage,maximum_seconds=3600,private_tag_MiB=192,aggregate_MiB=3584,scope='actual N16 decode plus eight known prompt chunks in wholly unused q5 groups, independent serial references, lengths1..5 and unchanged full-target graph device costs',Native_flags_assigned=False,model_parameter_storages_never_swapped=True,no_new_production_permission=True,Raw=None,Goodput=None,Git_pushes=0)
with(D/'ROOT445_PADDING_PREFILL_LAUNCH_REQUEST.json').open('x')as f:json.dump(request,f,indent=2);f.write('\n')
os.execve(sys.executable,argv,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
