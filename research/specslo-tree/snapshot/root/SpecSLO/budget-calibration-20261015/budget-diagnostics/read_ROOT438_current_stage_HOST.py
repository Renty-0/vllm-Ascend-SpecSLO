"""Read actual checkpoint counts without inferring full NN qualification."""
import datetime
import json
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
TAG = 'slo-aware-refill-parent4-E4-global64-root-v438-fresh75-plus6-full463-ownAR-v1'
ROOT = B / 'tree-causal80-small-cohort-role-aware-audit-full-v209'


def main():
    receipt_path = ROOT / f'receipt-{TAG}.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {'status':'prelaunch_source_verification'}
    actual = ROOT / 'actual' / TAG
    rows = []
    for name in ('startup-progress', 'variable-B4-own-full-process',
                 'causal80-own-current-process', 'multiowner-eager-prefill-own-current-process',
                 'causal80-seven-real-N-current-process', 'canonical-independent-node-math',
                 'target-uniform-KV-bucket-math425', 'parent4-current-component425',
                 'multiincumbent32-prefill427-own-current-process',
                 'windows', 'postfull-independent-FIA-AR'):
        folder = actual / name
        files = [p for p in folder.rglob('*') if p.is_file()] if folder.exists() else []
        if not files:
            continue
        latest = max(files, key=lambda p: p.stat().st_mtime)
        stamp = latest.stat().st_mtime
        rows.append(dict(directory=name, observed_files=len(files),
            latest=str(latest.relative_to(actual)),
            latest_UTC=datetime.datetime.fromtimestamp(stamp, datetime.timezone.utc).isoformat(),
            latest_epoch=stamp))
    rows.sort(key=lambda row: row['latest_epoch'])
    value = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        tag=TAG, supervisor_status=receipt['status'],
        elapsed_seconds=receipt.get('elapsed_seconds'), phases=rows,
        lifecycle_cases_each_rank=[len(list((actual / 'parent4-current-component425' /
            'bounded199').glob(f'rank{rank}-pass*.json.gz'))) for rank in range(4)],
        incumbent_prefill_case_files_each_rank=[len(list((actual / 'multiincumbent32-prefill427-own-current-process').glob(f'rank{rank}-*pass*.json.gz'))) for rank in range(4)],
        worker_errors=[p.name for p in actual.glob('rank*-last-error.json')],
        checkpoint_presence_does_not_replace_terminal_audit=True,
        NPU_API_called=False, Torch_imported=False, Raw=None, Goodput=None)
    output = D / 'ROOT438_EXPANDED_PROGRESS.json'
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(output)
    print(json.dumps(value))


if __name__ == '__main__':
    main()
