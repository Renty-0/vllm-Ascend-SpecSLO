"""Read only current450 receipts, narrow NN progress and owned log tail."""
from pathlib import Path
import json,datetime
B=Path('/root/SpecSLO/budget-calibration-20261015');R=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='M120-single-graph-prefill-root-v450-nine-cases-isolation-v2';A=R/'actual'/TAG
out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_exists':A.exists()}
f=R/f'receipt-{TAG}.json'
if f.exists():
 j=json.loads(f.read_text());out['receipt']={k:j.get(k)for k in('status','errors','sources_unchanged','owned_remaining')};out['child_pid']=(j.get('child')or{}).get('pid')
 if j.get('pre_verifier'):out['pre_verifier_exit']=j['pre_verifier']['exit_code']
if A.exists():
 out['files']=[f.name for f in sorted(A.iterdir())]
 out['references']={}
 for f in A.glob('rank*-single-graph448-reference-progress.json'):
  j=json.loads(f.read_text());out['references'][f.name]={k:dict(steps=len(v),failed=sum(x is not True for x in v))for k,v in j['decode_reference_checks'].items()}
 out['cases']={}
 for f in A.glob('rank*-single-graph448-case-progress.json'):
  j=json.loads(f.read_text());out['cases'][f.name]=[dict(name=c['name'],baseline=c['graph_baseline_matches_eager'],eager_decode=c['eager_reference_decode_stable'],prefill=c['prefill_hidden_exact'],decode=c['decode_hidden_exact'],timing_skipped=c['timing_skipped'])for c in j['cases']]
 for f in A.glob('rank*-single-graph-prefill448.json'):
  j=json.loads(f.read_text());out.setdefault('final_math',{})[f.name]=j['all3_all9_cases_math_exact']
f=R/f'local-{TAG}.log'
if f.exists():
 with f.open('rb')as s:
  s.seek(max(0,f.stat().st_size-9000));tail=s.read().decode(errors='replace')
 out['log_bytes']=f.stat().st_size;out['log_tail']=[line[:300]for line in tail.splitlines()[-6:]]
 out['fatal_tail_indicators']=[line[:350]for line in tail.splitlines()if any(v in line for v in('NPU out of memory.','Traceback','RuntimeError:','ValueError:'))][-8:]
print(json.dumps(out))
