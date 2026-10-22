"""Bounded true whole-target layout diagnostic; no serving claim or Git."""
import datetime,json,os,sys
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-M120-whole-target-gateup-layout-diagnostic-v441';sys.path.insert(0,str(P))
from host_env441 import setup
setup()
r=json.loads((P/'SOURCE441_RECEIPT.json').read_text())
assert r['freeze_sha256']=='5819985a78cec1901fd2afc32c58065085a144d25d879810bee211288fd84709'and r['archive_sha256']=='0c90d64172a1fce194483d615dc26d44b6a15da46e2829eade5552eda4e0b472'
check=json.loads((D/'ROOT441_FROZEN_PARENT_AND_WORKER_HOST_RESULT.json').read_text());assert check['status'].startswith('PASS_HOST441')and check['NPU_used']is False
prior=json.loads((D/'ROOT440_TERMINAL_INDEPENDENT_OPERATOR_AUDIT_HOST.json').read_text());assert prior['original_supervisor_status']=='complete'and prior['all4_idle']is True and prior['whole_workload_AR_qualified']is False
import guard441
guard441.EXPECTED_FREEZE=r['freeze_sha256'];guard441.EXPECTED_ARCHIVE=r['archive_sha256'];guard441.prepared();storage=guard441.storage();assert storage['aggregate_private_bytes']+256*1024**2<=3584*1024**2
TAG='M120-whole-target-gateup-layout-root-v441-one-frame-v1'
argv=[sys.executable,'-B',str(P/'guard441.py'),'--tag',TAG,'--expected-freeze-sha256',r['freeze_sha256'],'--expected-archive-sha256',r['archive_sha256']]
request=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tag=TAG,argv=argv,source_freeze=r['freeze_sha256'],source_archive=r['archive_sha256'],private_storage=storage,maximum_seconds=3600,private_tag_MiB=192,aggregate_MiB=3584,scope='actual whole M120/S24 target frame, transient gate-up layout variants all64 layers, hidden/queryKV and balanced target-forward device events',Native_flags_assigned=False,all_parameter_storages_restored_after_diagnostic=True,no_new_production_permission=True,Raw=None,Goodput=None,Git_pushes=0)
with(D/'ROOT441_WHOLE_TARGET_DIAGNOSTIC_LAUNCH_REQUEST.json').open('x')as f:json.dump(request,f,indent=2);f.write('\n')
os.execve(sys.executable,argv,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
