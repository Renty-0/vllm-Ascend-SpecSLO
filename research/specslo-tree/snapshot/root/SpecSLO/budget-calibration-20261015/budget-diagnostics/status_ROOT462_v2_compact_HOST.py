"""Read-only filesystem status; no model imports, mutations or maintenance."""
from pathlib import Path
import datetime,json
B=Path('/root/SpecSLO/budget-calibration-20261015/tree-causal80-small-cohort-role-aware-audit-full-v209')
TAG='adaptive-own-history-global64-root-v462-qualified-wait-full463-ownAR-v2'
R=B/'actual'/TAG
p=R/'parent4-current-component425/bounded199'
receipt=json.loads((B/f'receipt-{TAG}.json').read_text())
value=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    status=receipt['status'],lifecycle=[len(list(p.glob(f'rank{rank}-pass*.json.gz')))for rank in range(4)],
    full_traces=len(list((R/'windows/window64/full463').glob('rank*-full463-actual-trace.json.gz'))),
    errors=[dict(file=str(f.relative_to(R)),error=f.read_text()[-3500:])for f in R.glob('rank*-last-error.json')])
value['incumbent_prefill_cases']=[len(list((R/'multiincumbent32-prefill427-own-current-process').glob(f'rank{rank}-N*-pass*.json.gz')))for rank in range(4)]
value['prewarm_files']=len(list((R/'prewarm').glob('*')))
value['postfull_AR_files']=len(list((R/'postfull-independent-FIA-AR').glob('*')))
value['root_diagnostics']=len(list(R.glob('rank*-local-diagnostic.json')))
value['window_entries']=[f.name for f in(R/'windows/window64').glob('*')]
if receipt['status']!='running':
    value['receipt_keys']=list(receipt)
    value['receipt']=str(B/f'receipt-{TAG}.json')
print(json.dumps(value,ensure_ascii=False))
