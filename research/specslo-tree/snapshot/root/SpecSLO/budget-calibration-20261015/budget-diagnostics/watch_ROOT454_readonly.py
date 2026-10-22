"""Read only current454 process and exact operator qualification progress."""
from pathlib import Path
import json,datetime
B=Path('/root/SpecSLO/budget-calibration-20261015');R=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='M160-real32-native-epilogue-root-v454-six-layers-and1024-change-v1';A=R/'actual'/TAG
out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_exists':A.exists()}
f=R/f'receipt-{TAG}.json'
if f.exists():
 j=json.loads(f.read_text());out['receipt']={k:j.get(k)for k in('status','errors','sources_unchanged','owned_remaining')};out['child_pid']=(j.get('child')or{}).get('pid')
 if j.get('pre_verifier'):out['pre_verifier_exit']=j['pre_verifier']['exit_code']
if A.exists():
 out['file_count']=len(list(A.iterdir()));out['dispatch_ranks']=[f.name.split('-')[0]for f in A.glob('rank*-installed-dispatch-before-model.json')]
 out['real32_captures']={}
 for f in A.glob('rank*-M160454-real32-capture.json'):
  j=json.loads(f.read_text());out['real32_captures'][f.name]={k:j[k]for k in('physical_M','physical_S','real_requests','real_query_rows','all_real_slots_disjoint')}
 out['progress']={}
 for f in A.glob('rank*-M160454-qualification-progress.json'):
  j=json.loads(f.read_text());out['progress'][f.name]=[{k:v[k]for k in('layer','projection','production_row_accepted','original_qualifier_admission_rejection')}for v in j]
 out['raw_measurements']={}
 for f in A.glob('live-M160454/*/layer-*/m160/measurement.json'):
  j=json.loads(f.read_text());out['raw_measurements'][str(f.relative_to(A))]={'correctness':j['correctness'],'summary':j['summary']}
 for f in A.glob('rank*-M160454-final.json'):
  j=json.loads(f.read_text());out.setdefault('final',{})[f.name]=j['all6_original_operator_gates_accepted']
f=R/f'local-{TAG}.log'
if f.exists():
 with f.open('rb')as stream:stream.seek(max(0,f.stat().st_size-14000));tail=stream.read().decode(errors='replace')
 out['log_bytes']=f.stat().st_size;out['log_tail']=[line[:300]for line in tail.splitlines()[-6:]]
 out['fatal_tail_indicators']=[line[:350]for line in tail.splitlines()if any(v in line for v in('NPU out of memory.','Traceback','RuntimeError:','ValueError:','FileExistsError:'))][-10:]
print(json.dumps(out))
