"""Bounded true whole-target layout diagnostic; no serving claim or Git."""
import datetime,json,os,sys
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-M120-native-prefill-action-routing-v453';sys.path.insert(0,str(P))
from host_env453 import setup
setup()
r=json.loads((P/'SOURCE453_RECEIPT.json').read_text())
assert r['freeze_sha256']=='40284d76c73ac3a7c67edb73eebdfe1c8feb17279bba6efb7e97340a0008dedd'and r['archive_sha256']=='58a28d70c50f56f33710fad4c7200183cec9f725ceba7f6da8723e0c0663ddff'
check=json.loads((D/'ROOT453_FROZEN_PARENT_AND_WORKER_HOST_RESULT.json').read_text());assert check['status'].startswith('PASS_HOST453')and check['NPU_used']is False
prior=json.loads((D/'ROOT450_TERMINAL_INDEPENDENT_SINGLE_GRAPH_AUDIT_HOST.json').read_text());assert prior['original_supervisor_status']=='complete'and prior['all4idle']is True and prior['all3_all9_cases_math_exact']is True
import guard453
guard453.EXPECTED_FREEZE=r['freeze_sha256'];guard453.EXPECTED_ARCHIVE=r['archive_sha256'];guard453.prepared();storage=guard453.storage();assert storage['aggregate_private_bytes']+256*1024**2<=3584*1024**2
TAG='M120-native-prefill-equivalence-root-v453-nine-cases-plus16AR-v1'
argv=[sys.executable,'-B',str(P/'guard453.py'),'--tag',TAG,'--expected-freeze-sha256',r['freeze_sha256'],'--expected-archive-sha256',r['archive_sha256']]
request=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tag=TAG,argv=argv,source_freeze=r['freeze_sha256'],source_archive=r['archive_sha256'],private_storage=storage,maximum_seconds=3600,private_tag_MiB=192,aggregate_MiB=3584,scope='retain nine same-frame cases and add ordinary Native prompt/full-prompt and16-step true AR references in distinct owners, all before one graph capture; explicit post-replay incumbent checks',Native_flags_assigned=False,model_parameter_storages_never_swapped=True,no_new_production_permission=True,Raw=None,Goodput=None,Git_pushes=0)
with(D/'ROOT453_NATIVE_PREFILL_LAUNCH_REQUEST.json').open('x')as f:json.dump(request,f,indent=2);f.write('\n')
os.execve(sys.executable,argv,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
