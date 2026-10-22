"""Read-only qualification progress; no model/NPU APIs or performance inference."""
import datetime,json,time
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015')
D=B/'budget-diagnostics'
P=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='nonempty-refill-parent4-E4-global64-root-v436-current75-plus6-full463-ownAR-v1'
A=P/'actual'/TAG
R=P/f'receipt-{TAG}.json'
start=time.monotonic();previous=None
while time.monotonic()-start<18600:
 receipt=json.loads(R.read_text()) if R.exists() else {}
 full=A/'windows/window64/full463'
 value=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tag=TAG,
  supervisor_status=receipt.get('status'),elapsed_seconds=receipt.get('elapsed_seconds'),
  errors=[p.name for p in A.glob('rank*-last-error.json')],NPU_API_called=False,
  full463_directory_exists=full.exists(),whole_results_qualified=False)
 if not full.exists():
  value.update(canonical227_cases=len(list((A/'canonical-independent-node-math').glob('rank1-canonical-case*.json'))),
   uniform_banks={name:len(list((A/f'target-uniform-KV-bucket-math425/{name}').glob('rank1-canonical-case*.json')))for name in ('mixed','mid','short')},
   parent4_cases_each_rank=[len(list((A/'parent4-current-component425/bounded199').glob(f'rank{rank}-pass*.gz')))for rank in range(4)],
   warm_trace_files=len(list((A/'windows').glob('**/*actual-trace.json.gz'))))
 # Preserve transient running receipts without inventing a score or Native qualification.
 f=D/'ROOT436_READONLY_PROGRESS.json';tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(f)
 milestone={k:v for k,v in value.items()if k not in ('UTC','elapsed_seconds')}
 for k in ('canonical227_cases',):
  if k in milestone:milestone[k]=milestone[k]//25
 if 'uniform_banks' in milestone:milestone['uniform_banks']={k:v//25 for k,v in milestone['uniform_banks'].items()}
 key=json.dumps(milestone,sort_keys=True)
 if key!=previous:print(json.dumps(value),flush=True);previous=key
 if receipt.get('status') in ('complete','failed') and receipt.get('finished_UTC'):break
 time.sleep(10)
