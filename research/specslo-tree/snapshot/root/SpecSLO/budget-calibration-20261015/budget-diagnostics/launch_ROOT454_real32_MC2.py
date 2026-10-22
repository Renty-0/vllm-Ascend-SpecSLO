"""Bounded true whole-target layout diagnostic; no serving claim or Git."""
import datetime,json,os,sys
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-M160-real32-inmemory-MC2-qualification-v454';sys.path.insert(0,str(P))
from host_env454 import setup
setup()
r=json.loads((P/'SOURCE454_RECEIPT.json').read_text())
assert r['freeze_sha256']=='430c7b17ae1dc4730efe8c826130b61c65b61b5504488278355621e9d4b61cbd'and r['archive_sha256']=='1590532f720f6e52df868ff20a6b68bc6c83941160a5f2932b13db04d81bd456'
check=json.loads((D/'ROOT454_FROZEN_PARENT_AND_WORKER_HOST_RESULT.json').read_text());assert check['status'].startswith('PASS_HOST454')and check['NPU_used']is False
prior=json.loads((D/'ROOT453_TERMINAL_INDEPENDENT_NATIVE_PREFILL_AUDIT_HOST.json').read_text());assert prior['original_supervisor_status']=='complete'and prior['all4idle']is True 
import guard454
guard454.EXPECTED_FREEZE=r['freeze_sha256'];guard454.EXPECTED_ARCHIVE=r['archive_sha256'];guard454.prepared();storage=guard454.storage();assert storage['aggregate_private_bytes']+256*1024**2<=3584*1024**2
TAG='M160-real32-native-epilogue-root-v454-six-layers-and1024-change-v1'
argv=[sys.executable,'-B',str(P/'guard454.py'),'--tag',TAG,'--expected-freeze-sha256',r['freeze_sha256'],'--expected-archive-sha256',r['archive_sha256']]
request=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tag=TAG,argv=argv,source_freeze=r['freeze_sha256'],source_archive=r['archive_sha256'],private_storage=storage,maximum_seconds=3600,private_tag_MiB=192,aggregate_MiB=3584,scope='genuine32 separately prefilled owners and all160 real B4 queries; observed layers0/31/63 attention/down; unchanged original1024 changed inputs and31 paired p95 samples; no tensors saved or profiles installed',Native_flags_assigned=False,model_parameter_storages_never_swapped=True,no_new_production_permission=True,Raw=None,Goodput=None,Git_pushes=0)
with(D/'ROOT454_REAL32_MC2_LAUNCH_REQUEST.json').open('x')as f:json.dump(request,f,indent=2);f.write('\n')
os.execve(sys.executable,argv,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
