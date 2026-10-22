"""Bounded true whole-target layout diagnostic; no serving claim or Git."""
import datetime,json,os,sys
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-M120-paired-memory-prefill-math-diagnostic-v447';sys.path.insert(0,str(P))
from host_env447 import setup
setup()
r=json.loads((P/'SOURCE447_RECEIPT.json').read_text())
assert r['freeze_sha256']=='e7c99bc4cb5543a9781d6556149c94052f9d46e52b10358300ec51a78a2b42ed'and r['archive_sha256']=='a36b8bdd89ca7c9caed48bf046c6435920467a8d789a4c422953f7c823925f0a'
check=json.loads((D/'ROOT447_FROZEN_PARENT_AND_WORKER_HOST_RESULT.json').read_text());assert check['status'].startswith('PASS_HOST447')and check['NPU_used']is False
prior=json.loads((D/'ROOT445_TERMINAL_INDEPENDENT_PADDING_PREFILL_AUDIT_HOST.json').read_text());assert prior['original_supervisor_status']=='complete'and prior['all4idle']is True and prior['all3_all5_lengths_math_exact']is True
import guard447
guard447.EXPECTED_FREEZE=r['freeze_sha256'];guard447.EXPECTED_ARCHIVE=r['archive_sha256'];guard447.prepared();storage=guard447.storage();assert storage['aggregate_private_bytes']+256*1024**2<=3584*1024**2
TAG='M120-paired-memory-prefill-root-v447-nine-cases-math-cost-v1'
argv=[sys.executable,'-B',str(P/'guard447.py'),'--tag',TAG,'--expected-freeze-sha256',r['freeze_sha256'],'--expected-archive-sha256',r['archive_sha256']]
request=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tag=TAG,argv=argv,source_freeze=r['freeze_sha256'],source_archive=r['archive_sha256'],private_storage=storage,maximum_seconds=3600,private_tag_MiB=192,aggregate_MiB=3584,scope='fresh445 NN and timings with explicit teardown, then all9 initial/bulk cases with2 live diagnostic graphs per pair and actual post-replay output/KV checks',Native_flags_assigned=False,model_parameter_storages_never_swapped=True,no_new_production_permission=True,Raw=None,Goodput=None,Git_pushes=0)
with(D/'ROOT447_PADDING_PREFILL_LAUNCH_REQUEST.json').open('x')as f:json.dump(request,f,indent=2);f.write('\n')
os.execve(sys.executable,argv,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
