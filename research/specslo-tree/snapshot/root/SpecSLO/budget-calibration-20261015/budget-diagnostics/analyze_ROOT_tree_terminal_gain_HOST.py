"""Read retained, independently audited final gain curves without a backend."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(audit_path, version):
    audit = json.loads(audit_path.read_text())
    assert audit['status'] == 'terminal_completed_compute_audited'
    assert audit['supervisor_status'] == 'complete' and not audit['errors']
    assert audit['whole_workload_AR_qualified'] is True
    manifest = audit['source_and_raw_manifest']
    reports = sorted(Path(p) for p in manifest if p.endswith('/rank0-local-diagnostic.json'))
    assert len(reports) == 1 and sha(reports[0]) == manifest[str(reports[0])]['sha256']
    report = json.loads(reports[0].read_text())
    geometry = report[f'ROOT{version}_actual_geometry']
    risk = geometry['actual_paired_gain']
    generation = str(risk['generation'])
    current = risk['by_generation'][generation]
    classes = collections.defaultdict(list)
    for owner, row in current['requests'].items():
        cache = row['cache']
        if cache is None:
            continue
        g = cache['g']
        assert len(g) == 9 and g[0] == 0 and g[4] > 0
        assert all(0 <= a <= b <= 5 for a, b in zip(g, g[1:]))
        classes[row['slo_class']].append(dict(owner=int(owner), g=g,
            observations=row['observations'], B8_to_B4=g[8]/g[4],
            alternate_trials=row['raw']['samples'][4:]))
    summary = {}
    for cls, rows in sorted(classes.items()):
        summary[cls] = dict(requests=len(rows),
            median_final_cached_g=[statistics.median(r['g'][b] for r in rows) for b in range(9)],
            median_final_cached_B8_to_B4=statistics.median(r['B8_to_B4'] for r in rows),
            median_verdict_observations=statistics.median(r['observations'] for r in rows),
            requests_without_any_alternate_trials=sum(not any(r['alternate_trials']) for r in rows))
    original = audit['metrics']
    result = dict(schema='ROOT.tree-terminal-gain-descriptive.v1', version=version,
        terminal_audit=dict(path=str(audit_path), sha256=sha(audit_path)),
        actual_report=dict(path=str(reports[0]), sha256=sha(reports[0])),
        generation=int(generation), classes=summary,
        actual_budget_owner_rounds=original['budget_owner_rounds'],
        actual_target_calls=original['target_calls'], actual_mean_N=original['mean_N'],
        actual_bonus_accounting=original['bonus_accounting'],
        limitation='Final cached request predictions describe the predictor; they are neither held-out accuracy nor counterfactual gains from different budgets.',
        independent_AR_qualified=True, NPU_used=False, performance_change_inferred=False)
    completed = audit.get('completed_compute_audit', {})
    cost = completed.get(f'ROOT{version}_actual_current_target_cost')
    if cost is not None:
        result['independently_recomputed_current_cost'] = cost
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--version', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    result = analyze(args.audit, args.version)
    with args.output.open('x') as saved:
        saved.write(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
