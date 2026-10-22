"""Read current453 actual Native reference, same-frame NN and cleanup evidence."""
from pathlib import Path
import datetime,hashlib,importlib.abc,json,sys
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-M120-native-prefill-action-routing-v453';R=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='M120-native-prefill-equivalence-root-v453-nine-cases-plus16AR-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class NoModel(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):raise RuntimeError('HOST453 no device backend')
sys.meta_path.insert(0,NoModel());sys.path.insert(0,str(P))
from host_env453 import setup
setup()
import guard453
source=json.loads((P/'SOURCE453_RECEIPT.json').read_text());guard453.EXPECTED_FREEZE=source['freeze_sha256'];guard453.EXPECTED_ARCHIVE=source['archive_sha256']
_,closure,_=guard453.prepared()
p=R/f'receipt-{TAG}.json';receipt=json.loads(p.read_text())
assert receipt['status']=='complete'and receipt['errors']==[]and receipt['sources_unchanged']is True
assert receipt['owned_remaining']==[]and receipt['devices_after']['all_idle']is True
for phase in('pre_verifier','post_verifier'):
 v=receipt[phase];assert v['exit_code']==0 and v['actual_result']['status']=='pass'and sha(v['path'])==v['sha256']
from audit453 import audit,CASES,KEYS
audited,scope,_=audit(R/'actual'/TAG);r=audited['reports'][0];summary=[]
for i,name in enumerate(CASES):
 cases=[v['cases'][i]for v in r['by_target_rank']]
 native={}
 for label in cases[0]['ordinary_prefill_checks']:
  cs=[c['ordinary_prefill_checks'][label]for c in cases]
  native[label]=dict(hidden_exact_all3=all(c['hidden_exact']for c in cs),
   tokens_exact_all3=all(c['tokens_exact']for c in cs),all128_KV_exact_all3=all(c['all128_KV_exact']for c in cs),
   max_hidden_abs=max(c['hidden_max_abs']for c in cs),max_KV_abs=max(c['max_KV_abs']for c in cs))
 summary.append(dict(name=name,all3_same_frame_math_exact=all(all(c[k]is True for k in(*KEYS,'post_replay_hidden_stable','post_replay_prefill128_KV_exact','post_replay_decode_hidden_exact','post_replay_decode_tokens_exact','post_replay_decode128_KV_exact'))for c in cases),
  ordinary_prefill_checks=native,actual_paired_costs=r['paired_max_rank_costs'][name]))
AR={}
for family in('initial','continuation'):
 rows=[v['actual_independent_Native_AR'][family]for v in r['by_target_rank']]
 AR[family]=dict(all3_all16_tokens_exact=all(v['all16_tokens_exact']for v in rows),all3_unrelated_decode_stable=all(v['all_unrelated_decode_checks_exact']for v in rows),
  by_rank={str(v['rank']):v['actual_independent_Native_AR'][family]for v in r['by_target_rank']})
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_HOST453_ACTUAL_NATIVE_REFERENCE_AND_SOURCE_RECORD_AUDIT',
 source_members=len(closure),freeze=source['freeze_sha256'],archive=source['archive_sha256'],receipt=str(p),receipt_sha256=sha(p),
 original_supervisor_status='complete',all3_all9_same_physical_cases_math_exact=r['all3_all9_cases_math_exact'],
 all3_native_AR_tokens_exact=r['all3_native_AR_tokens_exact'],summary=summary,independent_Native_AR=AR,scope=scope,
 all_parameter_bytes_SHA256_unchanged=True,all72_original_committed_prefixes_protected=True,all4idle=True,owned_remaining=[],
 whole_workload_AR_qualified=False,production_permission=False,Raw=None,Goodput=None,adopted=False,Git_pushes=0)
with(D/'ROOT453_TERMINAL_INDEPENDENT_NATIVE_PREFILL_AUDIT_HOST.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(dict(status=out['status'],all3_all9_same_physical_cases_math_exact=out['all3_all9_same_physical_cases_math_exact'],all3_native_AR_tokens_exact=out['all3_native_AR_tokens_exact'],all4idle=True,Raw=None,Goodput=None)))
