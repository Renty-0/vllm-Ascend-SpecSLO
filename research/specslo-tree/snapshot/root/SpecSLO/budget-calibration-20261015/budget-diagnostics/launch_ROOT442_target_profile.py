"""Bounded true whole-target layout diagnostic; no serving claim or Git."""
import datetime,json,os,sys
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-M120-target-kernel-profile-diagnostic-v442';sys.path.insert(0,str(P))
from host_env442 import setup
setup()
r=json.loads((P/'SOURCE442_RECEIPT.json').read_text())
assert r['freeze_sha256']=='9fb100603aaff42108a77b2d4e13f6b91455798e620a7c1b8b93f0b6edd21e17'and r['archive_sha256']=='bb71d9dae09f093f4b6090b5f2cceeebee56285d56e4846abbcc3ba07c1916b7'
check=json.loads((D/'ROOT442_FROZEN_PARENT_AND_WORKER_HOST_RESULT.json').read_text());assert check['status'].startswith('PASS_HOST442')and check['NPU_used']is False
prior=json.loads((D/'ROOT441_TERMINAL_INDEPENDENT_WHOLE_TARGET_AUDIT_HOST.json').read_text());assert prior['original_supervisor_status']=='complete'and prior['all4_idle']is True and prior['full463']is False
import guard442
guard442.EXPECTED_FREEZE=r['freeze_sha256'];guard442.EXPECTED_ARCHIVE=r['archive_sha256'];guard442.prepared();storage=guard442.storage();assert storage['aggregate_private_bytes']+256*1024**2<=3584*1024**2
TAG='M120-unchanged-target-kernel-profile-root-v442-one-replay-v1'
argv=[sys.executable,'-B',str(P/'guard442.py'),'--tag',TAG,'--expected-freeze-sha256',r['freeze_sha256'],'--expected-archive-sha256',r['archive_sha256']]
request=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tag=TAG,argv=argv,source_freeze=r['freeze_sha256'],source_archive=r['archive_sha256'],private_storage=storage,maximum_seconds=3600,private_tag_MiB=192,aggregate_MiB=3584,scope='unchanged actual M120/S24 target graph, nine unprofiled event windows and one NPU-only profile replay per rank',Native_flags_assigned=False,model_parameter_storages_never_swapped=True,no_new_production_permission=True,Raw=None,Goodput=None,Git_pushes=0)
with(D/'ROOT442_TARGET_PROFILE_LAUNCH_REQUEST.json').open('x')as f:json.dump(request,f,indent=2);f.write('\n')
os.execve(sys.executable,argv,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
