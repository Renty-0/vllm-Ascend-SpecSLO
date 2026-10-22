"""Actual own-M80 local model diagnostic; no serving/AR permission or Git."""
from pathlib import Path
import datetime,json,os,sys
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-adaptive-own-geometry-node-math-v459';sys.path.insert(0,str(P))
from host_env459 import setup
setup()
r=json.loads((P/'SOURCE459_RECEIPT.json').read_text())
assert r['freeze_sha256']=='74ffbd335bd6223f6e318dfb0de864ecf11f34e41324820426f638dc0810c4fb'and r['archive_sha256']=='f00ccf518eb842ca8b69f3dd5cae638a304af7e2fc47fd1f816c565fb4240f88'
host=json.loads((D/'ROOT459_FROZEN_PARENT_AND_WORKER_HOST_RESULT.json').read_text());assert host['status'].startswith('PASS_HOST459')and host['NPU_used']is False
prior=json.loads((D/'ROOT458_RECONCILED457_WHOLE_MODEL_AUDIT_HOST.json').read_text());assert prior['original_supervisor_status']=='failed'and prior['all4idle']is True and prior['all3_eager_and_graph_math_exact']is True and prior['NN_permission_issued']is False
import guard459
guard459.EXPECTED_FREEZE=r['freeze_sha256'];guard459.EXPECTED_ARCHIVE=r['archive_sha256'];guard459.prepared();storage=guard459.storage();assert storage['aggregate_private_bytes']+256*1024**2<=3584*1024**2
TAG='adaptive-own-physical-root-v459-693-node-and-paired-cost-v1'
argv=[sys.executable,'-B',str(P/'guard459.py'),'--tag',TAG,'--expected-freeze-sha256',r['freeze_sha256'],'--expected-archive-sha256',r['archive_sha256']]
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tag=TAG,argv=argv,source_freeze=r['freeze_sha256'],source_archive=r['archive_sha256'],private_storage=storage,
 maximum_seconds=3600,private_tag_MiB=192,aggregate_MiB=3584,global_active_ceiling=64,
 scope='fresh693 independent same-physical M80/M120 spine/side hidden/token/all128KV cases over3 independent prefix banks;20 paired costs for6 arms;all original comparators and Native validation; no serving or ownAR permission',
 ordinary_prompt_prefill_unchanged=True,Native_flags_assigned=False,new_production_permission=False,Raw=None,Goodput=None,Git_pushes=0)
with(D/'ROOT459_WHOLE_MODEL_LAUNCH_REQUEST.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
os.execve(sys.executable,argv,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
