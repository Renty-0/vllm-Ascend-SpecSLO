"""Independently recheck real296 model outputs, rotated costs and cleanup."""
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
P = B / 'tree-concrete-device-span-dual-cache-target-cost-v296'
O = B / 'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG = 'span-prefix-dual-cache-concrete-device-v296-B1to8-408-math-cost-v1'
A = O / 'actual' / TAG
sys.path.insert(0, str(P))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    pins = read(P / 'SOURCE296_RECEIPT.json')
    receipt_path = O / f'receipt-{TAG}.json'
    receipt = read(receipt_path)
    assert receipt['status'] == 'complete' and not receipt['errors']
    assert receipt['sources_unchanged'] is True and receipt['owned_remaining'] == []
    assert receipt['devices_after']['all_idle'] is True
    assert receipt['child']['exit_code'] == 0
    import guard296
    guard296.EXPECTED_FREEZE = pins['freeze_sha256']
    guard296.EXPECTED_ARCHIVE = pins['archive_sha256']
    guard296.prepared()
    actual, _, _ = guard296.actual_audit(A)
    assert actual == receipt['actual_component_audit']
    reports = [read(A / f'rank{rank}-numeric.json') for rank in range(4)]
    candidates = [[row['candidate_tokens'] for row in report['math']['cases']]
                  for report in reports[1:]]
    assert candidates[0] == candidates[1] == candidates[2]
    costs = []
    for rank, report in enumerate(reports[1:], 1):
        assert all(row['all128_KV_original_matcher_passed'] is True
                   and row['candidate_KV_copied'] is False for row in report['math']['cases'])
        for row in report['paired_target_cost']['rows']:
            counts = {b: 0 for b in range(4, 9)}
            values = {b: [] for b in range(4, 9)}
            for pair in row['pairs']:
                assert sorted(pair['order']) == list(range(4, 9))
                counts[pair['order'][0]] += 1
                for b in values:
                    value = pair['milliseconds'][str(b)]
                    assert isinstance(value, (float, int)) and math.isfinite(value) and value > 0
                    values[b].append(value)
            assert set(counts.values()) == {4}
            costs.append(dict(rank=rank, N=row['N'], branch_owners=row['branch_owners'],
                rounds=20, median_ms={str(b): statistics.median(v) for b, v in values.items()},
                within_pair_ratio_to_B4={str(b): statistics.median(
                    pair['milliseconds'][str(b)] / pair['milliseconds']['4'] for pair in row['pairs'])
                    for b in range(5, 9)}))
    for name in ('pre_verifier', 'post_verifier'):
        item = receipt[name]
        assert item['exit_code'] == 0 and sha(item['path']) == item['sha256']
        evidence = item['actual_result']
        assert evidence['status'] == 'pass' and evidence['errors'] == []
        assert (evidence['checked_frozen_files'], evidence['checked_external_files'],
                evidence['checked_packages']) == (10496, 15, 212)
    result = dict(status='PASS_actual296_target408_all3_rawKV_hidden_greedy_and_rotated_cost',
        receipt=str(receipt_path), receipt_sha256=sha(receipt_path), source_pins=pins,
        target_cases_per_rank=408, target_ranks=[1, 2, 3],
        all_target_greedy_outputs_equal=True, all128KV_original_matchers=True,
        independent_committed_prefixes_unchanged=True,
        paired_cost_configurations_per_rank=11, paired_rotations=20, target_costs=costs,
        original_environment_prepost_10496_15_212_verified=True,
        owned_cleanup_complete=True, all4_idle_at_terminal=True,
        full463_own_AR_qualified=False, real_draft_cost_measured=False,
        end_to_end_performance_qualified=False, Goodput=None, adopted=False)
    output = D / 'ROOT296_INDEPENDENT_TARGET_MATH_AND_COST_AUDIT.json'
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'target_costs'}))


if __name__ == '__main__':
    main()
