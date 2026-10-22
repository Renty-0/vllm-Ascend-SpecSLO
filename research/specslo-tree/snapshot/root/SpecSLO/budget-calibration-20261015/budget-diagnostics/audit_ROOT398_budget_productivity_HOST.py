"""Describe qualified observed budgets; never infer a counterfactual speedup."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
RUNS = {
    'v376_historical_active48': (
        'static-M120-uniform-KV-root-v376-full463-ownAR-v1', 48),
    'v398_actual_active64': (
        'active64-whole-ticket-S24-segments-root-v398-full463-ownAR-v1', 64),
}


def describe(tag, active_limit):
    path = (B / 'tree-causal80-small-cohort-role-aware-audit-full-v209'
            / 'actual' / tag / 'windows' / f'window{active_limit}' / 'full463'
            / 'rank1-full463-actual-trace.json.gz')
    raw = path.read_bytes()
    trace = json.loads(gzip.decompress(raw))
    assert trace['configured_capacity'] == 64
    assert trace['admission_window160'] == active_limit
    assert trace['active_peak'] == active_limit
    requests = {r['request_index']: r for r in trace['requests']}
    first = set()
    commits = []
    roots_before = collections.defaultdict(list)
    all_output = 0
    for event in trace['stream_events']:
        all_output += len(event['token_ids'])
        owner = event['request_index']
        if owner in first:
            commits.append(event)
        else:
            assert len(event['token_ids']) == 1
            first.add(owner)
            roots_before[len(commits)].append(owner)
    assert len(first) == len(requests) == 463
    assert all_output == 463 * 256
    groups = collections.defaultdict(collections.Counter)
    active = set()
    active_at_target = collections.Counter()
    active_and_selected = collections.Counter()
    cursor = 0
    calls = trace['normal_target_calls']
    for call in calls:
        for owner in roots_before[cursor]:
            assert owner not in active
            active.add(owner)
        assert not any(roots_before[k] for k in range(cursor + 1, cursor + call['N']))
        active_at_target[len(active)] += 1
        active_and_selected[(len(active), call['N'])] += 1
        owners = call['sequence_ids'][:call['N']]
        assert set(owners) <= active
        events = commits[cursor:cursor + call['N']]
        cursor += call['N']
        assert [e['request_index'] for e in events] == owners
        assert len(call['budgets']) == len(events)
        for owner, budget, event in zip(owners, call['budgets'], events, strict=True):
            delivered = len(event['token_ids'])
            assert 1 <= delivered <= budget + 1
            for label in ['all', requests[owner]['slo_class']]:
                stats = groups[(label, budget)]
                stats['owner_rounds'] += 1
                stats['committed_tokens'] += delivered
                stats['allocated_candidates'] += budget
                stats['terminal_owner_rounds'] += int(event['finished'])
                stats[f'delivered_{delivered}'] += 1
            if event['finished']:
                active.remove(owner)
    assert cursor == len(commits) == sum(c['N'] for c in calls)
    assert not active and max(active_at_target) == active_limit
    assert sum(v['committed_tokens'] for (label, _), v in groups.items()
               if label == 'all') == all_output - 463
    result = []
    for (label, budget), counts in sorted(groups.items()):
        n = counts['owner_rounds']
        result.append(dict(slo_class=label, budget=budget, counts=dict(counts),
                           observed_mean_committed_tokens=counts['committed_tokens'] / n,
                           observed_committed_per_candidate=(
                               counts['committed_tokens'] / counts['allocated_candidates'])))
    return dict(source=str(path), source_sha256=hashlib.sha256(raw).hexdigest(),
                active_limit=active_limit, target_calls=len(calls),
                mean_owners_per_target=cursor / len(calls),
                mean_real_queries_per_target=sum(c['real_query_count'] for c in calls) / len(calls),
                useful_query_fraction=sum(c['real_query_count'] for c in calls) / (120 * len(calls)),
                active_owner_histogram=dict(sorted(active_at_target.items())),
                active_and_selected_histogram=[dict(active=a, selected=n, calls=count)
                                               for (a, n), count in sorted(active_and_selected.items())],
                calls_active_gt24=sum(count for (a, n), count in active_and_selected.items() if a > 24),
                calls_active_gt24_selected_lt24=sum(count for (a, n), count in active_and_selected.items()
                                                   if a > 24 and n < 24),
                single_owner_calls_at_single_active=active_and_selected[(1, 1)],
                owner_rounds=cursor, output_tokens=all_output, by_class_budget=result)


def main():
    audit_path = D / 'ROOT398_TERMINAL_INDEPENDENT_FULL_AUDIT.json'
    audit = json.loads(audit_path.read_text())
    assert audit['status'] == 'terminal_completed_compute_audited'
    assert audit['whole_workload_AR_qualified'] is True and not audit['errors']
    result = dict(
        status='PASS_COMPLETE_EVENT_ALIGNMENT_AND_OBSERVED_BUDGET_COUNTS',
        qualification_audit=str(audit_path),
        qualification_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        runs={name: describe(*scope) for name, scope in RUNS.items()},
        claim_limits=[
            'Budgets were chosen by the live policy, not randomly assigned.',
            'Conditional means are observed productivity, not causal g(i,n) curves.',
            'Tokens include correction/bonus tokens and terminal output truncation.',
            'Comparisons change active limit and segment packing together.',
            'Phase clocks and replay cost do not independently measure overlap.',
            'No new model execution, performance result, or production qualification.',
        ],
        NPU_used=False, production_permission=False)
    path = D / 'ROOT398_OBSERVED_BUDGET_PRODUCTIVITY.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    for name, run in result['runs'].items():
        print(name, json.dumps({k: run[k] for k in [
            'target_calls', 'mean_owners_per_target', 'useful_query_fraction']}))
        print(json.dumps([dict(B=r['budget'], rounds=r['counts']['owner_rounds'],
                               mean=r['observed_mean_committed_tokens'])
                          for r in run['by_class_budget'] if r['slo_class'] == 'all']))


if __name__ == '__main__':
    main()
