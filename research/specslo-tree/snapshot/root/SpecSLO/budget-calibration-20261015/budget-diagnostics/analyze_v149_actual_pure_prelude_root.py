"""Independent retained-byte arithmetic for the completed V149 local probe."""
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

B = Path('/root/SpecSLO/budget-calibration-20261015')
P = B / 'tree-repair-frontier-fusion-local-v149'
D = B / 'budget-diagnostics'
TAG = 'repair-frontier-union-v149-fixedprefix-ABBA-v1'
RECEIPT_SHA = 'a1b5025e46f95ad99ef08e3898453c3aa0a09a949f4c17af5397f9aab88672ad'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    sys.path.insert(0, str(D / 'v149-root-execution'))
    import common_guard
    common_guard.prepared()
    receipt_path = P / f'receipt-{TAG}.json'
    assert sha(receipt_path) == RECEIPT_SHA
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'complete' and receipt['errors'] == []
    assert receipt['owned_remaining'] == [] and receipt['sources_unchanged'] is True
    assert receipt['devices_before']['all_idle'] is receipt['devices_after']['all_idle'] is True
    for key in ('pre_verifier', 'post_verifier'):
        v = receipt[key]['actual_result']
        assert receipt[key]['exit_code'] == 0 and v['status'] == 'pass' and not v['errors']
        assert (v['checked_frozen_files'], v['checked_external_files'], v['checked_packages']) == (10496, 15, 212)
    directory = P / 'actual' / TAG
    reports = [json.loads(gzip.decompress((directory / f'rank{i}-component.json.gz').read_bytes())) for i in range(4)]
    native_checks = 0
    inputs = {str(receipt_path): RECEIPT_SHA}
    for i, report in enumerate(reports):
        inputs[str(directory / f'rank{i}-component.json.gz')] = sha(directory / f'rank{i}-component.json.gz')
        assert report['rank'] == i and report['completed'] is report['sealed'] is report['cache_released'] is True
        assert report['complete_local_zero_sweeps'] == 2 and len(report['local_zeros']) == 2
        assert report['adopted'] is report['formal_Goodput'] is report['compiled_protocol_qualified'] is False
        for row in [*report['local_zeros'], report['sealed_cost_Native']]:
            before, after, delta = row['before'], row['after'], row['delta']
            assert set(before) == set(after) == set(delta) and len(before) == 21
            for key in before:
                assert type(before[key]) is type(after[key]) is type(delta[key]) is int
                assert before[key] >= 0 and after[key] - before[key] == delta[key] == 0
                native_checks += 1
        inv = report['final_inventory']
        assert [len(inv[k]) for k in ('entries', 'draft_entries', 'target_entries')] == ([0, 2, 6] if i == 0 else [0, 0, 0])
        assert all(row['runtime_validated'] is True for rows in inv['natural_state'].values() for row in rows)
    draft = reports[0]
    assert draft['sameM_pass'] is True
    endpoints = []
    for case, size in (('R1-E1', 1), ('R32-E32', 32)):
        value = next(v for v in draft['cost'] if v['case'] == case)
        cost_path = directory / f'rank0-{case}-cost.json.gz'
        assert value == json.loads(gzip.decompress(cost_path.read_bytes()))
        inputs[str(cost_path)] = sha(cost_path)
        rows, summary = value['rows'], value['summary']
        assert len(rows) == 22 and summary['samples_per_arm'] == 11
        raw = {'old': [], 'new': []}
        paired = []
        tokens = {}
        for pair in range(11):
            batch = rows[2 * pair:2 * pair + 2]
            order = ['old', 'new'] if pair % 2 == 0 else ['new', 'old']
            assert [r['arm'] for r in batch] == order
            sample = {}
            for row in batch:
                arm, m = row['arm'], row['measurement']
                assert row['pair'] == pair and row['order'] == order
                assert math.isfinite(m['elapsed_ms']) and m['elapsed_ms'] > 0
                assert len(m['tokens']) == size
                assert m['model_calls'] == (2 if arm == 'old' else 1)
                assert m['model_M'] == ([size, size] if arm == 'old' else [2 * size])
                assert m['head_M'] == size and m['pure_device_time'] is False
                assert m['contains_diagnostic_hidden_KV_clone_or_hash'] is False
                assert tokens.setdefault(arm, m['tokens']) == m['tokens']
                raw[arm].append(m['elapsed_ms'])
                sample[arm] = m['elapsed_ms']
            paired.append(sample['old'] / sample['new'])
        medians = {k: statistics.median(v) for k, v in raw.items()}
        ratio = medians['old'] / medians['new']
        assert raw == summary['raw_ms'] and medians == summary['median_ms']
        assert paired == summary['paired_old_over_new'] and ratio == summary['median_old_over_new']
        assert summary['cost_advantage'] is (ratio > 1)
        assert tokens['old'] == summary['old_tokens'] and tokens['new'] == summary['new_tokens']
        cross = []
        for q in draft['qualification']:
            if q['case'] != case:
                continue
            assert q['sameM_pass'] is True and q['target_acceptance_observed'] is False
            c = q['cross_M']
            assert (c['old_tokens'] == c['new_tokens']) is c['actual_frontier_tokens_equal']
            assert c['old_tokens'] == tokens['old'] and c['new_tokens'] == tokens['new']
            cross.append(dict(pass_index=q['pass_index'],frontier_tokens_equal=c['actual_frontier_tokens_equal'],
                real_hidden_match=[r['original_matcher'] for r in c['real_hidden']],
                real_hidden_max_abs_difference=[r['max_abs_difference'] for r in c['real_hidden']],
                all56_real_write_KV_original_matcher=c['all56_real_write_KV_original_matcher']))
        assert len(cross) == 3
        endpoints.append(dict(case=case,raw_ms=raw,median_ms=medians,old_over_new=ratio,
            latency_reduction_fraction=1-medians['new']/medians['old'],paired_old_over_new=paired,
            all_pairs_faster=all(r>1 for r in paired),frontier_tokens_equal=tokens['old']==tokens['new'],
            cross_shape_observations=cross))
    assert native_checks == 252 and 'torch' not in sys.modules and 'torch_npu' not in sys.modules
    result=dict(schema='ROOT.V149.actual-pure-prelude-cost-arithmetic.v1',status='passed',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),inputs=inputs,script_SHA256=sha(__file__),
        original_full_guard_completed=True,independent_Native_zero_subtractions=native_checks,endpoints=endpoints,
        sameM_numerical_receipts_passed=True,hidden_KV_recomputed_by_HOST=False,
        crossM_behavior_field_means_frontier_tokens_only=True,fixed_prefix_cost_only=True,
        actual_TTU_tasks_or_skips_retained=False,dynamic_prefix_latency_covered=False,
        timing='whole prelude metadata/model/head/real-token readback; excludes preparation, full prefix hashes and outer stage rendezvous',
        compiled_protocol_qualified=False,full_draft_qualified=False,formal_Goodput=False,adopted=False,NPU_used=False)
    output=D/'ROOT_V149_ACTUAL_PURE_PRELUDE_COST_AUDIT.json'
    with output.open('x') as stream:
        json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(dict(output=str(output),sha256=sha(output),endpoints=[{k:v[k] for k in ('case','median_ms','old_over_new','frontier_tokens_equal')} for v in endpoints])))


if __name__ == '__main__':
    main()
