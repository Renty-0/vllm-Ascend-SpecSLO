"""Bounded read-only report for the owned full299 process; no NPU calls."""
import datetime
import json
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
O = B / 'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG = 'capacity-risk-span-root-v299-B1to8-full463-ownAR-v1'
A = O / 'actual' / TAG


def main():
    now = datetime.datetime.now(datetime.timezone.utc)
    receipt = json.loads((O / f'receipt-{TAG}.json').read_text())
    report = dict(UTC=now.isoformat(), tag=TAG, status=receipt['status'],
                  elapsed_seconds=receipt.get('elapsed_seconds'), errors=receipt.get('errors'))
    startup = {}
    for rank in range(4):
        path = A / f'startup-progress/prefill-rank{rank}.json'
        if path.exists():
            row = json.loads(path.read_text())
            startup[str(rank)] = {key: row.get(key) for key in ('status', 'completed', 'total', 'next_M')}
            startup[str(rank)]['age_seconds'] = round(now.timestamp() - path.stat().st_mtime)
    report['startup_prefill'] = startup
    finite = {}
    for rank in range(4):
        path = A / f'startup-progress/finite-inventory-rank{rank}.jsonl'
        if path.exists():
            lines = path.read_text().splitlines()
            if lines:
                row = json.loads(lines[-1])
                finite[str(rank)] = {key: row.get(key) for key in
                                    ('phase', 'completed', 'total', 'local_errors')}
                finite[str(rank)]['age_seconds'] = round(now.timestamp() - path.stat().st_mtime)
    report['startup_finite_inventory'] = finite
    report['current_component_directories'] = sorted(path.name for path in A.iterdir() if path.is_dir()) if A.exists() else []
    report['target_numeric_case_files'] = {
        str(rank): len(list((A / 'canonical-independent-node-math').glob(f'rank{rank}-canonical-case*.json')))
        for rank in (1, 2, 3)}
    path = A / 'rank0-online-state-latest.json'
    if path.exists():
        state = json.loads(path.read_text())
        report['online_snapshot'] = {key: state.get(key) for key in
                                     ('generation', 'observations_since_process_start', 'last_verdict')}
        report['online_snapshot']['age_seconds'] = round(now.timestamp() - path.stat().st_mtime)
    report['full_trace_files'] = len(list((A / 'windows/window48/full463').glob('rank*-full463-actual-trace.json.gz')))
    report['AR_summaries'] = {}
    for rank in (1, 2, 3):
        path = A / f'postfull-independent-FIA-AR/rank{rank}-summary.json'
        if path.exists():
            summary = json.loads(path.read_text())
            report['AR_summaries'][str(rank)] = summary.get('comparison')
    actual_errors = []
    for path in A.glob('rank*-last-error.json'):
        actual_errors.append(dict(file=path.name, tail=path.read_text()[-2500:]))
    report['actual_error_files'] = actual_errors
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
