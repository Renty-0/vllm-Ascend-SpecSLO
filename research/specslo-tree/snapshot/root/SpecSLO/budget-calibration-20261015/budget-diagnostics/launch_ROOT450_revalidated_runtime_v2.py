"""Bounded true whole-target layout diagnostic; no serving claim or Git."""
import datetime,json,os,sys
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-M120-single-graph-prefill-config-v450';sys.path.insert(0,str(P))
from host_env450 import setup
setup()
r=json.loads((P/'SOURCE450_RECEIPT.json').read_text())
assert r['freeze_sha256']=='1f401de775125382cc1cfc7bc0e474192d42b01a7ddfa6064c7e1ab26c04ae00'and r['archive_sha256']=='e78dd7bc591ae8818c081f219a4e120f33af58ba9dad834cdeb4b9b41c2068d1'
check=json.loads((D/'ROOT450_FROZEN_PARENT_AND_WORKER_HOST_RESULT.json').read_text());assert check['status'].startswith('PASS_HOST450')and check['NPU_used']is False
prior=json.loads((D/'ROOT445_TERMINAL_INDEPENDENT_PADDING_PREFILL_AUDIT_HOST.json').read_text());assert prior['original_supervisor_status']=='complete'and prior['all4idle']is True and prior['all3_all5_lengths_math_exact']is True
import guard450
guard450.EXPECTED_FREEZE=r['freeze_sha256'];guard450.EXPECTED_ARCHIVE=r['archive_sha256'];guard450.prepared();storage=guard450.storage();assert storage['aggregate_private_bytes']+256*1024**2<=3584*1024**2
TAG='M120-single-graph-prefill-root-v450-nine-cases-isolation-v2'
argv=[sys.executable,'-B',str(P/'guard450.py'),'--tag',TAG,'--expected-freeze-sha256',r['freeze_sha256'],'--expected-archive-sha256',r['archive_sha256']]
request=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tag=TAG,argv=argv,source_freeze=r['freeze_sha256'],source_archive=r['archive_sha256'],private_storage=storage,maximum_seconds=3600,private_tag_MiB=192,aggregate_MiB=3584,scope='all independent eager references before one persistent graph capture, static input copies only afterward, page/query alias and full parameter-byte fingerprints; negative cases are not timed',Native_flags_assigned=False,model_parameter_storages_never_swapped=True,no_new_production_permission=True,Raw=None,Goodput=None,Git_pushes=0)
with(D/'ROOT450_PADDING_PREFILL_RELAUNCH_V2_REQUEST.json').open('x')as f:json.dump(request,f,indent=2);f.write('\n')
os.execve(sys.executable,argv,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
