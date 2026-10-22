"""Read only current455 whole-model numerical and device-event progress."""
from pathlib import Path
import datetime,json
B=Path('/root/SpecSLO/budget-calibration-20261015');R=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='M160-qualified-profile-root-v455-unfused-fullmodel-and120-cost-v1';A=R/'actual'/TAG
out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_exists':A.exists()}
f=R/f'receipt-{TAG}.json'
if f.exists():
 j=json.loads(f.read_text());out['receipt']={k:j.get(k)for k in('status','errors','sources_unchanged','owned_remaining')};out['child_pid']=(j.get('child')or{}).get('pid')
 if j.get('pre_verifier'):out['pre_verifier_exit']=j['pre_verifier']['exit_code']
if A.exists():
 out['file_count']=len(list(A.iterdir()));out['dispatch_ranks']=[f.name.split('-')[0]for f in A.glob('rank*-installed-dispatch-before-model.json')]
 out['parameter_fingerprints_before']=[f.name for f in A.glob('rank*-whole455-weights-before.json')]
 for f in A.glob('rank*-whole455-eager-before-capture.json'):
  j=json.loads(f.read_text());out.setdefault('eager',{})[f.name]={k:j[k]for k in('hidden_exact','tokens_exact','all128_query_KV_exact','hidden_max_abs')}
 for f in A.glob('rank*-whole455-final.json'):
  j=json.loads(f.read_text());out.setdefault('final',{})[f.name]={k:j[k]for k in('all3_eager_and_graph_math_exact','actual_whole_model_costs')}
 for f in A.glob('rank*-last-error.json'):out.setdefault('worker_errors',{})[f.name]=json.loads(f.read_text())
f=R/f'local-{TAG}.log'
if f.exists():
 with f.open('rb')as stream:stream.seek(max(0,f.stat().st_size-14000));tail=stream.read().decode(errors='replace')
 out['log_bytes']=f.stat().st_size;out['log_tail']=[line[:300]for line in tail.splitlines()[-6:]]
 out['fatal_tail_indicators']=[line[:350]for line in tail.splitlines()if any(v in line for v in('NPU out of memory.','Traceback','RuntimeError:','ValueError:','FileExistsError:'))][-10:]
print(json.dumps(out))
