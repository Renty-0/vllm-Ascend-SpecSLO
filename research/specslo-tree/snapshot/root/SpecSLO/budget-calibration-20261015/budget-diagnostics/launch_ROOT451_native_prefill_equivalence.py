"""Bounded true whole-target layout diagnostic; no serving claim or Git."""
import datetime,json,os,sys
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-M120-native-prefill-equivalence-v451';sys.path.insert(0,str(P))
from host_env451 import setup
setup()
r=json.loads((P/'SOURCE451_RECEIPT.json').read_text())
assert r['freeze_sha256']=='ba7212482c26ae0e18ecf562abce523c49da93e75b4e92ccbf0c8f70e83748d8'and r['archive_sha256']=='051f9d56911c23723ddc6a5068cfb3cda087a4c9fbe0e10ed5f9408cd52a8ce0'
check=json.loads((D/'ROOT451_FROZEN_PARENT_AND_WORKER_HOST_RESULT.json').read_text());assert check['status'].startswith('PASS_HOST451')and check['NPU_used']is False
prior=json.loads((D/'ROOT450_TERMINAL_INDEPENDENT_SINGLE_GRAPH_AUDIT_HOST.json').read_text());assert prior['original_supervisor_status']=='complete'and prior['all4idle']is True and prior['all3_all9_cases_math_exact']is True
import guard451
guard451.EXPECTED_FREEZE=r['freeze_sha256'];guard451.EXPECTED_ARCHIVE=r['archive_sha256'];guard451.prepared();storage=guard451.storage();assert storage['aggregate_private_bytes']+256*1024**2<=3584*1024**2
TAG='M120-native-prefill-equivalence-root-v451-nine-cases-plus16AR-v1'
argv=[sys.executable,'-B',str(P/'guard451.py'),'--tag',TAG,'--expected-freeze-sha256',r['freeze_sha256'],'--expected-archive-sha256',r['archive_sha256']]
request=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tag=TAG,argv=argv,source_freeze=r['freeze_sha256'],source_archive=r['archive_sha256'],private_storage=storage,maximum_seconds=3600,private_tag_MiB=192,aggregate_MiB=3584,scope='retain nine same-frame cases and add ordinary Native prompt/full-prompt and16-step true AR references in distinct owners, all before one graph capture; explicit post-replay incumbent checks',Native_flags_assigned=False,model_parameter_storages_never_swapped=True,no_new_production_permission=True,Raw=None,Goodput=None,Git_pushes=0)
with(D/'ROOT451_NATIVE_PREFILL_LAUNCH_REQUEST.json').open('x')as f:json.dump(request,f,indent=2);f.write('\n')
os.execve(sys.executable,argv,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
