"""Fresh SLO-aware refill experiment; no previous model permission inherited."""
import datetime,hashlib,json,os,re,shlex,sys
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-slo-aware-refill-complete-interface-full-v438';DONOR=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='slo-aware-refill-parent4-E4-global64-root-v438-fresh75-plus6-full463-ownAR-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 terminal=json.loads((D/'ROOT437_TERMINAL_INDEPENDENT_REPORT_EXPORT_FAILURE.json').read_text())
 assert terminal['errors']==[]and terminal['supervisor_status']=='failed'and terminal['metrics']is None
 assert terminal['original_receipt_sha256']=='778a71e50e4bab9f95fb88afb16e66031a808465b95e895a5ace9b7614b310ec'
 previous_AR=json.loads((D/'ROOT437_COMPLETED_FULL463_INDEPENDENT_OWN_AR_AUDIT_HOST.json').read_text())
 assert previous_AR['same_backend_exact_output_qualified']is True
 bound={}
 for path in(P/'HOST438_POLICY_RESULT.json',P/'HOST438_COMPLETE_INTERFACE_RESULT.json',P/'HOST438_CLEAN_STARTUP_RUN.json',D/'ROOT438_FROZEN_ACTUAL_TRACE_PRECONSTRUCTOR_HOST.json',D/'ROOT438_FROZEN_COLD_PARENT_RESULT.json',D/'ROOT438_TERMINAL_ADMISSION_HOST_RESULT.json'):
  row=json.loads(path.read_text());assert row['status'].startswith('PASS_HOST438')and row['NPU_used']is False
  bound[str(path)]=sha(path)
 policy=json.loads((P/'HOST438_POLICY_RESULT.json').read_text())
 assert policy['actual_CPU_cases']==185 and policy['actual_live64_observed_in_CPU_controls']is True
 interface=json.loads((P/'HOST438_COMPLETE_INTERFACE_RESULT.json').read_text())
 assert len(interface['referenced_attributes'])==13 and interface['all_actual_call_signatures_bound']==14
 assert json.loads((D/'ROOT438_TERMINAL_ADMISSION_HOST_RESULT.json').read_text())['terminal_auditor_sha256']==sha(D/'audit_ROOT438_terminal_full_HOST.py')
 record=json.loads((P/'SOURCE438_RECEIPT.json').read_text())
 assert record['freeze_sha256']=='95460e17ad93cfe1aa521ceb200e8f2c7999c13176012e811cf2f5117d44bc40'
 assert record['archive_sha256']=='129ce5c270dd1d34fd2b7b79d6063f9c42aa5baedbfc8ce6d5944900be27800d'
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
 for line in(P/'launch438.sh').read_text().splitlines():
  for value in shlex.split(line.rstrip(' \\')):
   if re.fullmatch(r'SPECSLO_V[0-9]+_ROOT_FREEZE_SHA=[a-f0-9]{64}',value):
    key,setting=value.split('=',1);env[key]=setting;os.environ[key]=setting
 env['SPECSLO_V257_ROOT_FREEZE_SHA']='b352ac6bb9f3ae182d33d2f11dc278f7e49c2b8dff9d9d514c0b32d7a14a01f2'
 os.environ['SPECSLO_V257_ROOT_FREEZE_SHA']=env['SPECSLO_V257_ROOT_FREEZE_SHA']
 line=next(l for l in(P/'startup438.sh').read_text().splitlines()if l.startswith('export PYTHONPATH='))
 path=line.removeprefix('export PYTHONPATH=');F=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1')
 for old,new in(('$repo',str(F/'source/vllm-ascend-hust')),('$frozen',str(F)),('$task_root',str(DONOR)),('${PYTHONPATH:-}','')):path=path.replace(old,new)
 assert '$'not in path
 env['PYTHONPATH']=path;sys.path[:0]=path.split(':')
 import guard438
 guard438.EXPECTED_FREEZE=record['freeze_sha256'];guard438.EXPECTED_ARCHIVE=record['archive_sha256'];guard438.prepared()
 storage=guard438.storage()
 assert storage['aggregate_private_bytes']+320*1024**2<=3584*1024**2,'Original256MiB tag plus64MiB overhead reserve'
 argv=[sys.executable,'-B',str(P/'guard438.py'),'--tag',TAG,'--expected-freeze-sha256',record['freeze_sha256'],'--expected-archive-sha256',record['archive_sha256']]
 request=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),argv=argv,tag=TAG,source_freeze=record['freeze_sha256'],source_archive=record['archive_sha256'],HOST_prerequisites=bound,exact_same437_neural_callbacks=True,changes=['current-SLO-aware nonempty refill threshold: strict16 otherwise32','restore exact original433 preflight_snapshot report API'],all_original227_681_75_plus6_warm_twozero_seal_full463_all3ownAR_fresh_required=True,prior_NN_permission_inherited=False,Native_flags_assigned=False,global_active_limit=64,prelaunch_private_storage=storage,private_tag_MiB=256,aggregate_MiB=3584,original_run_seconds=18000,new_Git_commits=0,new_Git_pushes=0,Git_only_after_confirmed_complete_effective_improvement=True)
 with(D/'ROOT438_FRESH_FULL_LAUNCH_REQUEST.json').open('x')as f:json.dump(request,f,indent=2);f.write('\n')
 os.execve(sys.executable,argv,env)
if __name__=='__main__':main()
