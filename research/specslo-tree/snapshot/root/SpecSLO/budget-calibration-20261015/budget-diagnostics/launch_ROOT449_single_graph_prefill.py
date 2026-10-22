"""Bounded true whole-target layout diagnostic; no serving claim or Git."""
import datetime,json,os,sys
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-M120-single-graph-prefill-scope-v449';sys.path.insert(0,str(P))
from host_env449 import setup
setup()
r=json.loads((P/'SOURCE449_RECEIPT.json').read_text())
assert r['freeze_sha256']=='e8e9bee937308a60ce9a1c2e139f5d497f6ffaa43b6e0fd27dca3dfbaca13cb9'and r['archive_sha256']=='0854940cbcda603a7de1a8bfedf95ca5a80d1de6b36a3f70a8123665998f8aa6'
check=json.loads((D/'ROOT449_FROZEN_PARENT_AND_WORKER_HOST_RESULT.json').read_text());assert check['status'].startswith('PASS_HOST449')and check['NPU_used']is False
prior=json.loads((D/'ROOT445_TERMINAL_INDEPENDENT_PADDING_PREFILL_AUDIT_HOST.json').read_text());assert prior['original_supervisor_status']=='complete'and prior['all4idle']is True and prior['all3_all5_lengths_math_exact']is True
import guard449
guard449.EXPECTED_FREEZE=r['freeze_sha256'];guard449.EXPECTED_ARCHIVE=r['archive_sha256'];guard449.prepared();storage=guard449.storage();assert storage['aggregate_private_bytes']+256*1024**2<=3584*1024**2
TAG='M120-single-graph-prefill-root-v449-nine-cases-isolation-v1'
argv=[sys.executable,'-B',str(P/'guard449.py'),'--tag',TAG,'--expected-freeze-sha256',r['freeze_sha256'],'--expected-archive-sha256',r['archive_sha256']]
request=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tag=TAG,argv=argv,source_freeze=r['freeze_sha256'],source_archive=r['archive_sha256'],private_storage=storage,maximum_seconds=3600,private_tag_MiB=192,aggregate_MiB=3584,scope='all independent eager references before one persistent graph capture, static input copies only afterward, page/query alias and full parameter-byte fingerprints; negative cases are not timed',Native_flags_assigned=False,model_parameter_storages_never_swapped=True,no_new_production_permission=True,Raw=None,Goodput=None,Git_pushes=0)
with(D/'ROOT449_PADDING_PREFILL_LAUNCH_REQUEST.json').open('x')as f:json.dump(request,f,indent=2);f.write('\n')
os.execve(sys.executable,argv,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
