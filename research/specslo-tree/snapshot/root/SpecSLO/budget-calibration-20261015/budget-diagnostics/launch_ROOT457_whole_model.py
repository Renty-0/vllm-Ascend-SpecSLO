"""Actual own-M80 local model diagnostic; no serving/AR permission or Git."""
from pathlib import Path
import datetime,json,os,sys
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-M80-own-geometry-whole-model-v457';sys.path.insert(0,str(P))
from host_env457 import setup
setup()
r=json.loads((P/'SOURCE457_RECEIPT.json').read_text())
assert r['freeze_sha256']=='6046ff35cab90a252560b2be0bb757b3395174c8fa5b6e33a4225afbb0966228'and r['archive_sha256']=='e89132e0e60825586ce0bcd94bcf9fdb2f62ca92af45352de40c3541bfbd63d4'
host=json.loads((D/'ROOT457_FROZEN_PARENT_AND_WORKER_HOST_RESULT.json').read_text());assert host['status'].startswith('PASS_HOST457')and host['NPU_used']is False
prior=json.loads((D/'ROOT455_TERMINAL_INDEPENDENT_WHOLE_MODEL_AUDIT_HOST.json').read_text());assert prior['original_supervisor_status']=='complete'and prior['all4idle']is True
import guard457
guard457.EXPECTED_FREEZE=r['freeze_sha256'];guard457.EXPECTED_ARCHIVE=r['archive_sha256'];guard457.prepared();storage=guard457.storage();assert storage['aggregate_private_bytes']+256*1024**2<=3584*1024**2
TAG='M80-own-geometry-root-v457-unfused-fullmodel-and120-cost-v1'
argv=[sys.executable,'-B',str(P/'guard457.py'),'--tag',TAG,'--expected-freeze-sha256',r['freeze_sha256'],'--expected-archive-sha256',r['archive_sha256']]
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tag=TAG,argv=argv,source_freeze=r['freeze_sha256'],source_archive=r['archive_sha256'],private_storage=storage,
 maximum_seconds=3600,private_tag_MiB=192,aggregate_MiB=3584,global_active_ceiling=64,
 scope='trueM80 against independent ordinary M80 modelbody and original M120; cross-M120 discrepancies retained, all128 query KV, all parameters SHA, last-timed checks; no new tree or full463 own-AR permission',
 ordinary_prompt_prefill_unchanged=True,Native_flags_assigned=False,new_production_permission=False,Raw=None,Goodput=None,Git_pushes=0)
with(D/'ROOT457_WHOLE_MODEL_LAUNCH_REQUEST.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
os.execve(sys.executable,argv,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
