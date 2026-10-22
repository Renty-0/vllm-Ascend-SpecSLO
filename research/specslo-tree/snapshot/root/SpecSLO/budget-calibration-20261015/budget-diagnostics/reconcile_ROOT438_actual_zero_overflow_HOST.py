"""Reconcile finished438 records with a strict observed-coverage HOST audit.

Original receipt/source/results are immutable. This does not make the failed
supervisor complete, issue production permission, or adopt poor performance.
"""
import datetime,hashlib,importlib.abc,json,os,re,shlex,sys
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-slo-aware-refill-complete-interface-full-v438';H=B/'tree-observed-normal-history-audit-host-v439';DONOR=B/'tree-causal80-small-cohort-role-aware-audit-full-v209';TAG='slo-aware-refill-parent4-E4-global64-root-v438-fresh75-plus6-full463-ownAR-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class NoModel(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):raise RuntimeError('HOST439 reconciliation imports no model '+fullname)
def main():
 sys.meta_path.insert(0,NoModel())
 receipt=DONOR/f'receipt-{TAG}.json';assert sha(receipt)=='31f493e2d3b1ee3b5fdf057242bd2172067c6b71ee2b130eac1774834bf8dd09'
 actual=DONOR/'actual'/TAG
 for line in(P/'launch438.sh').read_text().splitlines():
  for item in shlex.split(line.rstrip(' \\')):
   if re.fullmatch(r'SPECSLO_V[0-9]+_ROOT_FREEZE_SHA=[a-f0-9]{64}',item):k,v=item.split('=',1);os.environ[k]=v
 os.environ['SPECSLO_V257_ROOT_FREEZE_SHA']='b352ac6bb9f3ae182d33d2f11dc278f7e49c2b8dff9d9d514c0b32d7a14a01f2'
 line=next(x for x in(P/'startup438.sh').read_text().splitlines()if x.startswith('export PYTHONPATH='));value=line.removeprefix('export PYTHONPATH=');F=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1')
 for x,y in[('$repo',str(F/'source/vllm-ascend-hust')),('$frozen',str(F)),('$task_root',str(DONOR)),('${PYTHONPATH:-}','')]:value=value.replace(x,y)
 assert '$'not in value;sys.path[:0]=[str(H),*value.split(':')]
 import guard438
 source=json.loads((P/'SOURCE438_RECEIPT.json').read_text());guard438.EXPECTED_FREEZE=source['freeze_sha256'];guard438.EXPECTED_ARCHIVE=source['archive_sha256'];guard438.prepared()
 import audit425,audit438
 from history439 import verify
 from projection425 import project_function
 old="need(all(value == normal_snapshots[0] for value in normal_snapshots)\n         and normal_audits[0]['overflow_calls'] > 0, 'all4 actual class/age normal24 histories and actual overflow')"
 new="need(_observed_history439(normal_snapshots, normal_audits), 'all4 actual class/age normal24 histories and observed overflow coverage')"
 assert old in Path(audit425.__file__).read_text()
 original=audit425.audit
 projected=project_function(original,((old,new),),dict(_observed_history439=verify))
 # Only this isolated HOST audit process uses the corrected audit function;
 # neither the original file nor any worker/model/backend is changed.
 try:
  audit425.audit=projected
  computation=audit438.audit(actual,(64,))
 finally:audit425.audit=original
 result=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_HOST439_RECONCILED_EXISTING438_COMPLETED_COMPUTATION_ONLY',source_receipt_sha256=sha(receipt),original_supervisor_status='failed',original_receipt_unchanged=True,original_source_files_unchanged=True,complete_computation_audit=computation,all_original_math_full463_and_ownAR_checks_retained=True,only_assertion_change='require actual overflow iff full raw-N histogram has overflow; otherwise prove all narrow owners admitted unchanged',coverage_helper_sha256=sha(H/'history439.py'),old_audit_source_sha256=sha(audit425.__file__),HOST_control_result_sha256=sha(H/'HOST439_HISTORY_RESULT.json'),NPU_used=False,new_NN_permission_issued=False,original_job_adopted=False,formal_Goodput=False,Raw=None,Goodput=None)
 p=D/'ROOT439_RECONCILED438_COMPLETE_COMPUTATION_AUDIT_HOST.json'
 with p.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:result[k]for k in('status','original_supervisor_status','all_original_math_full463_and_ownAR_checks_retained','NPU_used','original_job_adopted')}))
if __name__=='__main__':main()
