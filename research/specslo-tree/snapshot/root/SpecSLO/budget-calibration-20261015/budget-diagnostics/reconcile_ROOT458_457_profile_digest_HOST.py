"""Re-read all original457 checks with the Native canonical profile digest."""
from pathlib import Path
import datetime,hashlib,importlib.abc,json,sys
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-M80-own-geometry-whole-model-v457';R=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='M80-own-geometry-root-v457-unfused-fullmodel-and120-cost-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class NoModel(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):raise RuntimeError('HOST458 reconciliation no backend')
sys.meta_path.insert(0,NoModel());sys.path.insert(0,str(P))
from host_env457 import setup
setup()
import guard457,audit457
source=json.loads((P/'SOURCE457_RECEIPT.json').read_text());guard457.EXPECTED_FREEZE=source['freeze_sha256'];guard457.EXPECTED_ARCHIVE=source['archive_sha256'];_,closure,_=guard457.prepared()
f=R/f'receipt-{TAG}.json';receipt=json.loads(f.read_text())
assert receipt['status']=='failed'and len(receipt['errors'])==1 and receipt['errors'][0].rstrip().endswith('RuntimeError: ROOT457 unchanged original MC2 profile')
assert receipt['sources_unchanged']is True and receipt['owned_remaining']==[]and receipt['devices_after']['all_idle']is True
for phase in('pre_verifier','post_verifier'):
 row=receipt[phase];assert row['exit_code']==0 and row['actual_result']['status']=='pass'and sha(row['path'])==row['sha256']
proof_path=D/'ROOT458_ACTUAL_NATIVE_ORIGINAL_PROFILE_DIGEST_HOST.json';proof=json.loads(proof_path.read_text())
assert proof['status']=='PASS_HOST458_ACTUAL_NATIVE_ORIGINAL_PROFILE_FILE_AND_CANONICAL_DIGEST'and proof['NPU_initialized']is False and proof['actual_model_constructors']==0
assert sha(proof['profile_path'])==proof['file_sha256']=='ad9e0648810ed697679ee84b8bbc2939ea14bfb687d0215d27f403ad5526db1c'
normalized=hashlib.sha256(json.dumps(proof['normalized_document'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert normalized==proof['normalized_profile_sha256']=='67feca4c924ce95b252812d2db606a65a43942c472c429a71d12e5270442deb7'
from projection376 import project_function
audit=project_function(audit457.audit,((repr(proof['file_sha256']),repr(normalized)),))
actual,scope,_=audit(R/'actual'/TAG)
assert actual['all3_eager_and_graph_math_exact']is True
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_HOST458_RECONCILED457_FULL_NUMERICAL_AND_DEVICE_COST_RECORDS',
 original_supervisor_status='failed',original_failure='auditor compared raw profile file SHA against Native canonical normalized-profile SHA',
 original_receipt=str(f),original_receipt_sha256=sha(f),original_receipt_not_rewritten=True,
 original_auditor=str(P/'audit457.py'),original_auditor_sha256=sha(P/'audit457.py'),
 only_audit_delta='compare the reported Native canonical profile digest with the independently normalized Native digest; every other old predicate retained',
 actual_Native_digest_proof=str(proof_path),actual_Native_digest_proof_sha256=sha(proof_path),
 file_sha256=proof['file_sha256'],normalized_profile_sha256=normalized,
 source_members=len(closure),source_freeze=source['freeze_sha256'],source_archive=source['archive_sha256'],
 all3_eager_and_graph_math_exact=True,actual_whole_model_costs=actual['actual_whole_model_costs'],
 cross_M120_observations=[v['cross_M120_observation']for v in actual['reports'][0]['by_target_rank']],
 all72_committed_prefixes_protected=True,all_model_parameter_SHA_unchanged=True,all4idle=True,owned_remaining=[],
 NPU_reexecuted=False,NN_permission_issued=False,whole_workload_AR_qualified=False,full463=False,
 production_permission=False,Raw=None,Goodput=None,adopted=False,Git_pushes=0)
destination=D/'ROOT458_RECONCILED457_WHOLE_MODEL_AUDIT_HOST.json'
with destination.open('x')as stream:json.dump(out,stream,indent=2);stream.write('\n')
print(json.dumps({k:v for k,v in out.items()if k!='cross_M120_observations'}))
