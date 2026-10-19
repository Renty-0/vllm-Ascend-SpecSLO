"""Describe measured admission/occupancy costs; no counterfactual gain claims."""
import argparse
import collections
import gzip
import hashlib
import json
import statistics
from pathlib import Path


def summarize(path):
    raw = Path(path).read_bytes()
    trace = json.loads(gzip.decompress(raw) if str(path).endswith('.gz') else raw)
    assert trace['scope_count'] == 463
    calls = trace['normal_target_calls']
    assert calls and all(row['N'] == len(row['budgets']) for row in calls)
    assert all(row['physical_M'] == 120 for row in calls)
    first, last = {}, {}
    seen = set()
    for event in trace['stream_events']:
        index = event['request_index']
        if index not in seen:
            assert 'first_token_wall_seconds160' in event
            first[index] = event['first_token_wall_seconds160']
            seen.add(index)
        last[index] = event['elapsed_seconds']
    assert set(first) == set(last) == set(range(463))
    classes = {}
    for cls in ('tight', 'normal', 'loose'):
        indices = [i for i, row in enumerate(trace['requests']) if row['slo_class'] == cls]
        delays = [first[i] - trace['absolute_arrivals'][i] for i in indices]
        assert all(value >= 0 for value in delays)
        classes[cls] = dict(requests=len(indices), TTFT_median_seconds=statistics.median(delays),
                            TTFT_max_seconds=max(delays), last_publication_seconds=max(last[i] for i in indices))
    sizes = [len(row['sequence_ids']) for row in trace['standalone_prefill_calls']]
    assert sum(sizes) == 463
    counters = trace['decode_counters']
    reported_refill = counters.get('spec_rhythm_online_refill_ms')
    return dict(source=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                elapsed_seconds=trace['elapsed_seconds'], target_calls=len(calls),
                mean_N=statistics.mean(row['N'] for row in calls),
                N_histogram=dict(sorted(collections.Counter(row['N'] for row in calls).items())),
                fraction_N24=sum(row['N'] == 24 for row in calls) / len(calls),
                useful_logical_query_fraction=statistics.mean(
                    (row['N'] + sum(row['budgets'])) / 120 for row in calls),
                primary_and_side_group_fraction=statistics.mean(
                    (row['N'] + sum(b > 4 for b in row['budgets'])) / 24 for row in calls),
                actual_branch_target_calls=sum(any(b > 4 for b in row['budgets']) for row in calls),
                prefill_groups=len(sizes), prefill_group_sizes=sizes,
                prefill_size_histogram=dict(sorted(collections.Counter(sizes).items())),
                original_reported_online_refill_ms=reported_refill,
                phase_seconds=trace['phase_seconds'], by_class=classes,
                phase_sums_not_device_overlap_measurement=True,
                individual_prefill_cost_not_inferred=True,
                causal_attribution_proven=False, NPU_used=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    value = dict(status='retained_measured_trace_descriptive_only', measured=summarize(args.trace))
    if args.reference:
        value['reference'] = summarize(args.reference)
        value['changes_are_joint_not_isolated_ablation'] = True
    with args.output.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
