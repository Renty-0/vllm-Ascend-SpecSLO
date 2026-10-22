"""Independent source/receipt and event arithmetic; no backend import."""
from pathlib import Path
import datetime,hashlib,importlib.abc,json,math,statistics,sys
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-M120-linear-layout-cost-diagnostic-v440';R=B/'tree-causal80-small-cohort-role-aware-audit-full-v209';TAG='M120-real-layout-padding-root-v440-local-operator-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class NoModel(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):raise RuntimeError('HOST440 terminal auditor imports no model')
sys.meta_path.insert(0,NoModel());sys.path.insert(0,str(P))
from host_env440 import setup
setup()
import guard440
source=json.loads((P/'SOURCE440_RECEIPT.json').read_text());guard440.EXPECTED_FREEZE=source['freeze_sha256'];guard440.EXPECTED_ARCHIVE=source['archive_sha256'];_,closure,_=guard440.prepared()
p=R/f'receipt-{TAG}.json';receipt=json.loads(p.read_text())
assert receipt['status']=='complete'and receipt['errors']==[]and receipt['sources_unchanged']is True and receipt['owned_remaining']==[]and receipt['devices_after']['all_idle']is True
for phase in('pre_verifier','post_verifier'):
 v=receipt[phase];assert v['exit_code']==0 and v['actual_result']['status']=='pass'and sha(v['path'])==v['sha256']
from audit440 import audit
actual=R/'actual'/TAG;audited,_,_=audit(actual)
reports=[json.loads((actual/f'rank{k}-linear-layout-cost440.json').read_text())for k in(1,2,3)];rows=[]
for i,role in enumerate(reports[0]['roles']):
 gathered=role['by_target_rank'];cost={}
 for name in('ND120','NZ120','NDpad128','NZpad128'):
  samples=[max(x['device_event_ms_per_local_matmul'][name][j]for x in gathered)for j in range(9)]
  stored=role['balanced_max_rank_costs'][name]
  assert samples==stored['max_TP_rank_samples_ms']and statistics.median(samples)==stored['p50_ms']
  cost[name]=dict(p50_ms=statistics.median(samples),samples_ms=samples)
 base=cost['ND120']['p50_ms']
 rows.append(dict(projection=role['projection'],local_cost=cost,NZ120_over_ND120_speedup=base/cost['NZ120']['p50_ms'],NZ120_bitwise_equal_all3=all(x['comparison']['NZ120']['bitwise_equal_to_ND120']for x in gathered),pad128_bitwise_equal_all3=all(x['comparison']['NDpad128']['bitwise_equal_to_ND120']for x in gathered)))
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_HOST440_TERMINAL_SOURCE_AND_ACTUAL_DEVICE_EVENT_ARITHMETIC',source_members=len(closure),receipt=str(p),receipt_sha256=sha(p),source_freeze=source['freeze_sha256'],source_archive=source['archive_sha256'],original_supervisor_status='complete',original_audit=audited,summary=rows,all4_idle=True,owned_remaining=[],operator_only=True,full463=False,new_NN_permission_issued=False,NPU_used_by_this_auditor=False,whole_workload_AR_qualified=False,Raw=None,Goodput=None,adopted=False,new_Git_pushes=0)
with(D/'ROOT440_TERMINAL_INDEPENDENT_OPERATOR_AUDIT_HOST.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(dict(status=out['status'],summary=[{k:v for k,v in r.items()if k!='local_cost'}for r in rows],whole_model_speedup_claimed=False)))
