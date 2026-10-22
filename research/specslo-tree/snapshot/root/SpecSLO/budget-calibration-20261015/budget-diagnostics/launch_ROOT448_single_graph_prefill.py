"""Bounded true whole-target layout diagnostic; no serving claim or Git."""
import datetime,json,os,sys
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-M120-single-graph-prefill-isolation-v448';sys.path.insert(0,str(P))
from host_env448 import setup
setup()
r=json.loads((P/'SOURCE448_RECEIPT.json').read_text())
assert r['freeze_sha256']=='4611321dae535bc6db1f57680c3254504b1cd214e1cd52d5bcc0dbf28d1c1728'and r['archive_sha256']=='65e7f4fed325fd3e3da3e42ebf19eff6b426b6e17fa6400e5c6ec59a7e8ad901'
check=json.loads((D/'ROOT448_FROZEN_PARENT_AND_WORKER_HOST_RESULT.json').read_text());assert check['status'].startswith('PASS_HOST448')and check['NPU_used']is False
prior=json.loads((D/'ROOT445_TERMINAL_INDEPENDENT_PADDING_PREFILL_AUDIT_HOST.json').read_text());assert prior['original_supervisor_status']=='complete'and prior['all4idle']is True and prior['all3_all5_lengths_math_exact']is True
import guard448
guard448.EXPECTED_FREEZE=r['freeze_sha256'];guard448.EXPECTED_ARCHIVE=r['archive_sha256'];guard448.prepared();storage=guard448.storage();assert storage['aggregate_private_bytes']+256*1024**2<=3584*1024**2
TAG='M120-single-graph-prefill-root-v448-nine-cases-isolation-v1'
argv=[sys.executable,'-B',str(P/'guard448.py'),'--tag',TAG,'--expected-freeze-sha256',r['freeze_sha256'],'--expected-archive-sha256',r['archive_sha256']]
request=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tag=TAG,argv=argv,source_freeze=r['freeze_sha256'],source_archive=r['archive_sha256'],private_storage=storage,maximum_seconds=3600,private_tag_MiB=192,aggregate_MiB=3584,scope='all independent eager references before one persistent graph capture, static input copies only afterward, page/query alias and full parameter-byte fingerprints; negative cases are not timed',Native_flags_assigned=False,model_parameter_storages_never_swapped=True,no_new_production_permission=True,Raw=None,Goodput=None,Git_pushes=0)
with(D/'ROOT448_PADDING_PREFILL_LAUNCH_REQUEST.json').open('x')as f:json.dump(request,f,indent=2);f.write('\n')
os.execve(sys.executable,argv,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
