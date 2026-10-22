"""Read only; no NPU interaction or claim from the existence of a file."""
import datetime,json
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');O=B/'tree-causal80-small-cohort-role-aware-audit-full-v209';T='dynamic-root-alternate-root-v282-B1to8-paired-full463-ownAR-v1';A=O/'actual'/T
r=json.loads((O/f'receipt-{T}.json').read_text())
print(datetime.datetime.now(datetime.timezone.utc).isoformat(),r['status'],r.get('elapsed_seconds'),r.get('errors'))
for q in sorted((A/'startup-progress').glob('*.json')):
 if 'rank1' in q.name or q.name=='finite-draft-rank0.json':
  v=json.loads(q.read_text());print('startup',q.name,{k:v.get(k)for k in('status','completed','total','next_M','last_M')})
print('component cycle files',{q.name:len(list(q.glob('*.json.gz')))for q in A.iterdir()if q.is_dir()and 'process'in q.name})
print('actual cases',{p:len(list(A.rglob(p)))for p in('rank1-canonical-case*.json','rank2-canonical-case*.json','rank3-canonical-case*.json','draft-case*.json','rank0-branch-followup*.json')})
print('window files',[str(q.relative_to(A))for q in A.glob('windows/window48/*')])
for q in sorted(A.glob('*last-error.json')):print('ERROR',q.read_text()[-8000:])
if r['status']!='running':print('terminal',json.dumps(r)[-5000:])
