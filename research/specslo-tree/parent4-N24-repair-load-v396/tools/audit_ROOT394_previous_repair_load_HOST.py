"""Infer linear-leaf repair pressure from an immutable, completed old trace.

This reads output events only. It neither executes models nor changes admission,
budgets, timing, qualifications, or the active device experiment. B5 predecessor
branches are deliberately unknown, rather than inferred from output length.
"""
import collections
import gzip
import hashlib
import json
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
TRACE = B / (
    'tree-causal80-small-cohort-role-aware-audit-full-v209/actual/'
    'static-M120-uniform-KV-root-v376-full463-ownAR-v1/windows/window48/'
    'full463/rank1-full463-actual-trace.json.gz'
)


def main():
    compressed = TRACE.read_bytes()
    trace = json.loads(gzip.decompress(compressed))
    calls = trace['normal_target_calls']
    initial_owners = set()
    commits = []
    for event in trace['stream_events']:
        owner = event['request_index']
        if owner not in initial_owners:
            initial_owners.add(owner)
        else:
            commits.append(event)
    assert len(initial_owners) == 463
    assert len(commits) == sum(call['N'] for call in calls)
    position = 0
    pending = collections.defaultdict(int)
    all_hist = collections.Counter()
    n24_hist = collections.Counter()
    unknown_calls = 0
    unknown_rows = 0
    known_repair_rows = 0
    inferred_root_seed_calls = 0
    inferred_root_seed_rows = 0
    inferred_head_bucket_rows = 0
    for call in calls:
        n = call['N']
        owners = call['sequence_ids'][:n]
        events = commits[position:position + n]
        position += n
        assert [e['request_index'] for e in events] == owners
        flags = [pending[i] for i in owners]
        unknown = sum(f is None for f in flags)
        repairs = sum(f for f in flags if f is not None)
        known_repair_rows += repairs
        if unknown:
            unknown_calls += 1
            unknown_rows += unknown
        else:
            all_hist[repairs] += 1
            if n == 24:
                n24_hist[repairs] += 1
            if repairs:
                inferred_root_seed_calls += 1
                inferred_root_seed_rows += n
                inferred_head_bucket_rows += next(
                    size for size in (1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64)
                    if size >= n
                )
        steps = min(4, max(call['budgets']))
        for owner, budget, event in zip(owners, call['budgets'], events, strict=True):
            assert 1 <= len(event['token_ids']) <= budget + 1
            if budget == 5:
                pending[owner] = None
            else:
                pending[owner] = int(
                    budget == steps
                    and len(event['token_ids']) == budget + 1
                    and not event['finished']
                )
    assert position == len(commits)
    measured = json.loads((D / 'ROOT393_CURRENT_BEST_NORMAL_ROOT_SEED_SCREEN.json').read_text())[
        'formal_after_minus_before'
    ]
    n24 = sum(n24_hist.values())
    result = {
        'status': 'PASS_EVENT_GROUPING_WITH_EXPLICIT_BRANCH_UNCERTAINTY',
        'source': str(TRACE),
        'source_sha256': hashlib.sha256(compressed).hexdigest(),
        'historical_source_active_limit': 48,
        'required_future_active_limit': 64,
        'target_calls': len(calls),
        'initial_root_events': len(initial_owners),
        'commit_events': len(commits),
        'all_call_owner_orders_match_events': True,
        'unknown_calls_with_B5_predecessor': unknown_calls,
        'unknown_rows_with_B5_predecessor': unknown_rows,
        'fully_unambiguous_calls': len(calls) - unknown_calls,
        'linear_inferred_repair_rows': known_repair_rows,
        'fully_unambiguous_repair_histogram': dict(sorted(all_hist.items())),
        'fully_unambiguous_N24_repair_histogram': dict(sorted(n24_hist.items())),
        'N24_calls_unambiguous': n24,
        'N24_R0_or_R1_calls': n24_hist[0] + n24_hist[1],
        'N24_R0_or_R1_fraction': (n24_hist[0] + n24_hist[1]) / n24,
        'inferred_root_seed_calls_on_fully_unambiguous_groups': inferred_root_seed_calls,
        'inferred_root_seed_rows_on_fully_unambiguous_groups': inferred_root_seed_rows,
        'inferred_head_bucket_rows_on_fully_unambiguous_groups': inferred_head_bucket_rows,
        'measured_RootSeed_repair_queries': measured['root_seed_committed_repair_queries'],
        'measured_RootSeed_calls': measured['root_seed_calls'],
        'claim_limit': (
            'Inferred repair demand at subsequent target-owner grouping, not a directly '
            'measured per-draft timing or complete B5 repair reconstruction. Historical '
            'active48 trace cannot qualify active64 or any new eager shape.'
        ),
        'next_component_scope': 'After N16/P16 B4 protocol, include realistic N24 repair loads above R1.',
        'new_NPU_used': False,
        'production_permission': False,
        'Raw': None,
        'Goodput': None,
    }
    output = D / 'ROOT394_PREVIOUS_LINEAR_REPAIR_LOAD_INFERENCE.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in (
        'status', 'target_calls', 'fully_unambiguous_calls', 'linear_inferred_repair_rows',
        'measured_RootSeed_repair_queries', 'N24_calls_unambiguous', 'N24_R0_or_R1_fraction'
    )}))


if __name__ == '__main__':
    main()
