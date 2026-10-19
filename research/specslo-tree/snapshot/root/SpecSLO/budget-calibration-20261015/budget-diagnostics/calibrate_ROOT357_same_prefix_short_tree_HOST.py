"""Conditional depth3 curves from retained, AR-qualified raw8 observations.

These are exact clipped token-count counterfactuals within the old adaptive B8
cohort. They are not a new Q4 model run or an unconditional calibration curve.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from riskset_from_ROOT284_histograms_HOST import recover


def clipped(row, budget):
    if type(budget) is not int or not 1 <= budget <= 6:
        raise ValueError('new primary3/alternate3 retained B1..6')
    recovered = recover({'8': row})['by_budget']['8']
    primary = recovered['primary_path_counts']
    alternate = recovered['alternate_path_counts']
    result = Counter()
    if budget <= 3:
        for length, count in enumerate(primary):
            result[min(length, budget) + 1] += count
    else:
        failed = primary[0] - sum(alternate[1:])
        assert failed >= 0
        result[1] += failed
        for length, count in enumerate(primary[1:], 1):
            result[1 + min(length, 3)] += count
        for length, count in enumerate(alternate[1:], 1):
            result[1 + min(length, budget - 3)] += count
    assert sum(result.values()) == row['samples']
    assert all(1 <= length <= 4 and count >= 0 for length, count in result.items())
    return sum(length * count for length, count in result.items()), dict(result)


def synthetic_check():
    primary_paths = list(range(5))
    side_paths = list(range(1, 5))
    observations = [('primary', n, n + 1) for n in primary_paths]
    observations += [('side', n, n + 2) for n in side_paths]
    row = dict(samples=0, delivered_sum=0, accepted_side_nodes=0, shadow_B4_sum=0,
               delivered_histogram=[0] * 6, side_histogram=[0] * 5)
    for family, length, copies in observations:
        side = length if family == 'side' else 0
        row['samples'] += copies
        row['delivered_sum'] += (length + 1) * copies
        row['accepted_side_nodes'] += side * copies
        row['shadow_B4_sum'] += ((length + 1) if family == 'primary' else 1) * copies
        row['delivered_histogram'][length + 1] += copies
        row['side_histogram'][side] += copies
    for budget in range(1, 7):
        expected = Counter()
        for family, length, copies in observations:
            accepted = min(length, min(3, budget)) if family == 'primary' else min(length, max(0, budget - 3))
            expected[accepted + 1] += copies
        total, histogram = clipped(row, budget)
        assert total == sum(k * v for k, v in expected.items())
        assert {k: v for k, v in histogram.items() if v} == dict(expected)
    return dict(path_types=len(observations), weighted_observations=row['samples'], new_budgets=6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    qualification_raw = args.qualification.read_bytes()
    qualification = json.loads(qualification_raw)
    assert qualification['whole463_AR_qualified'] is True and qualification['measured_B'] == 8
    snapshot = Path(qualification['source_snapshot'])
    snapshot_raw = snapshot.read_bytes()
    assert hashlib.sha256(snapshot_raw).hexdigest() == qualification['source_sha256']
    data = json.loads(snapshot_raw)
    assert data['generation'] == 5
    grouped = {name: [] for name in ('tight', 'normal', 'loose')}
    for owner, request in data['request_gain_observations']['5'].items():
        row = request['by_budget'].get('8')
        if not row:
            continue
        values = [clipped(row, budget) for budget in range(1, 7)]
        grouped[request['slo_class']].append(dict(owner=owner, samples=row['samples'],
            sums=[v[0] for v in values], histograms=[v[1] for v in values]))
    curves = {}
    for ordinal, (name, rows) in enumerate(grouped.items()):
        samples = sum(row['samples'] for row in rows)
        assert samples == qualification['curves'][name]['B8_untruncated_verifications']
        means = [sum(row['sums'][i] for row in rows) / samples for i in range(6)]
        rng = random.Random(357 + ordinal)
        draws = [[] for _ in range(6)]
        for _ in range(500):
            selected = [rows[rng.randrange(len(rows))] for _ in rows]
            total = sum(row['samples'] for row in selected)
            for i in range(6):
                draws[i].append(sum(row['sums'][i] for row in selected) / total)
        assert all(1 <= value <= 4 for value in means) and all(a <= b for a, b in zip(means, means[1:]))
        old = qualification['curves'][name]['mean_delivered']
        assert means[:3] == old[:3]
        curves[name] = dict(n=list(range(1, 7)), mean_delivered=means,
            observed_requests=len(rows), observed_untruncated_verifications=samples,
            request_cluster_bootstrap95=[[sorted(v)[12] for v in draws], [sorted(v)[487] for v in draws]],
            old_B4_same_cohort=old[3], new_B3_over_old_B4=means[2] / old[3],
            new_B6_over_old_B4=means[5] / old[3], requests=rows)
    result = dict(status='HOST_exact_depth3_same_prefix_clipped_counts_in_old_adaptive_B8_cohort',
        qualification=str(args.qualification), qualification_sha256=hashlib.sha256(qualification_raw).hexdigest(),
        snapshot=str(snapshot), snapshot_sha256=hashlib.sha256(snapshot_raw).hexdigest(),
        synthetic_check=synthetic_check(), curves=curves,
        source_workload_whole463_AR_qualified=True, new_Q4_neural_math_qualified=False,
        source_periodic_snapshot_not_final_complete_eventlog=True,
        adaptive_cohort_selection_bias_not_removed=True,
        deployment_curve=False, new_geometry_latency_measured=False,
        g_scope='intrinsic token yield before per-request output-tail clipping; conditional on observed old B8 cohort',
        NPU_used=False, new_tree_full463_qualified=False, Raw=None, Goodput=None)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: ({c: {key: value for key, value in row.items() if key != 'requests'}
                          for c, row in v.items()} if k == 'curves' else v) for k, v in result.items()}))


if __name__ == '__main__':
    main()
