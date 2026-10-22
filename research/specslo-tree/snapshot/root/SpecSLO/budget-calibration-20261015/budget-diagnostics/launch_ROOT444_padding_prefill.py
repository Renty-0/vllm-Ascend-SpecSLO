"""Bounded true whole-target layout diagnostic; no serving claim or Git."""
import datetime,json,os,sys
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-M120-padding-prefill-math-diagnostic-v444';sys.path.insert(0,str(P))
from host_env444 import setup
setup()
r=json.loads((P/'SOURCE444_RECEIPT.json').read_text())
assert r['freeze_sha256']=='ee9ce37bfce0fb87b485e5f2ee26916bbf795f5a5e5176a90cfd5b8662b39521'and r['archive_sha256']=='dc3952c16cdc9f7f662ea4d45accc796773734a541b9e5309d528006a01e9e14'
check=json.loads((D/'ROOT444_FROZEN_PARENT_AND_WORKER_HOST_RESULT.json').read_text());assert check['status'].startswith('PASS_HOST444')and check['NPU_used']is False
prior=json.loads((D/'ROOT441_TERMINAL_INDEPENDENT_WHOLE_TARGET_AUDIT_HOST.json').read_text());assert prior['original_supervisor_status']=='complete'and prior['all4_idle']is True and prior['full463']is False
import guard444
guard444.EXPECTED_FREEZE=r['freeze_sha256'];guard444.EXPECTED_ARCHIVE=r['archive_sha256'];guard444.prepared();storage=guard444.storage();assert storage['aggregate_private_bytes']+256*1024**2<=3584*1024**2
TAG='M120-N16-padding-prefill-root-v444-lengths1to5-math-cost-v1'
argv=[sys.executable,'-B',str(P/'guard444.py'),'--tag',TAG,'--expected-freeze-sha256',r['freeze_sha256'],'--expected-archive-sha256',r['archive_sha256']]
request=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tag=TAG,argv=argv,source_freeze=r['freeze_sha256'],source_archive=r['archive_sha256'],private_storage=storage,maximum_seconds=3600,private_tag_MiB=192,aggregate_MiB=3584,scope='actual N16 decode plus eight known prompt chunks in wholly unused q5 groups, independent serial references, lengths1..5 and unchanged full-target graph device costs',Native_flags_assigned=False,model_parameter_storages_never_swapped=True,no_new_production_permission=True,Raw=None,Goodput=None,Git_pushes=0)
with(D/'ROOT444_PADDING_PREFILL_LAUNCH_REQUEST.json').open('x')as f:json.dump(request,f,indent=2);f.write('\n')
os.execve(sys.executable,argv,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
