"""Read only Native451 live progress; no model imports or mutations."""
from pathlib import Path
import json,datetime
B=Path('/root/SpecSLO/budget-calibration-20261015');R=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='M120-native-prefill-equivalence-root-v453-nine-cases-plus16AR-v1';A=R/'actual'/TAG
out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_exists':A.exists()}
f=R/f'receipt-{TAG}.json'
if f.exists():
 j=json.loads(f.read_text());out['receipt']={k:j.get(k)for k in('status','errors','sources_unchanged','owned_remaining')};out['child_pid']=(j.get('child')or{}).get('pid')
 if j.get('pre_verifier'):out['pre_verifier_exit']=j['pre_verifier']['exit_code']
if A.exists():
 out['file_count']=len(list(A.iterdir()));out['dispatch_ranks']=[f.name.split('-')[0]for f in A.glob('rank*-installed-dispatch-before-model.json')]
 out['ordinary_references']={f.name:json.loads(f.read_text())for f in A.glob('rank*-native-prefill451-ordinary-reference-progress.json')}
 out['eager_references']={}
 for f in A.glob('rank*-native-prefill451-reference-progress.json'):
  j=json.loads(f.read_text());out['eager_references'][f.name]={k:dict(steps=len(v),failed=sum(x is not True for x in v))for k,v in j['decode_reference_checks'].items()}
 out['cases']={}
 for f in A.glob('rank*-native-prefill451-case-progress.json'):
  j=json.loads(f.read_text());out['cases'][f.name]=[dict(name=c['name'],same_frame_pre=all(v['exact_before']for v in c['actual_all3_pre_timing_gate']),native_tokens={k:v['tokens_exact']for k,v in c['ordinary_prefill_checks'].items()},post_decode=c['post_replay_decode128_KV_exact'])for c in j['cases']]
 out['actual_AR']={}
 for f in A.glob('rank*-native-prefill451-actual-AR-progress.json'):
  j=json.loads(f.read_text());out['actual_AR'][f.name]={k:dict(all16_tokens_exact=v['all16_tokens_exact'],first_mismatch=v['first_mismatch'],decode_stable=v['all_unrelated_decode_checks_exact'])for k,v in j.items()}
 for f in A.glob('rank*-native-prefill-equivalence451.json'):
  j=json.loads(f.read_text());out.setdefault('final_math',{})[f.name]={k:j[k]for k in('all3_all9_cases_math_exact','all3_native_AR_tokens_exact')}
f=R/f'local-{TAG}.log'
if f.exists():
 with f.open('rb')as s:s.seek(max(0,f.stat().st_size-9000));tail=s.read().decode(errors='replace')
 out['log_bytes']=f.stat().st_size;out['log_tail']=[line[:300]for line in tail.splitlines()[-5:]]
 out['fatal_tail_indicators']=[line[:350]for line in tail.splitlines()if any(v in line for v in('NPU out of memory.','Traceback','RuntimeError:','ValueError:'))][-8:]
print(json.dumps(out))
