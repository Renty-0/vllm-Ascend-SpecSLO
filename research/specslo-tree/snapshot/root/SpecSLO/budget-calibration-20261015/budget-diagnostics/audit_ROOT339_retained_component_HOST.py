"""Read completed339 without changing its failed supervisor receipt."""
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import sys

B = Path('/root/SpecSLO/budget-calibration-20261015')
P = B/'tree-Q4-closure-aware-host-builder-adapter-v339'
sys.path.insert(0, str(P))
import guard339

TAG = 'linear-Q4-Q5-closure-aware-host-builder-v339-36-math-paired-cost-v1'
OUTPUT = guard339.DONOR/'actual'/TAG


def cold_calls(report, cases):
    # Native's unchanged qualification wrapper calls actual/changed/restored
    # inputs at each first new graph shape. All three calls are observed.
    sequence = []
    seen = set()
    for case in cases:
        layout = case['layout']
        sequence.extend([layout]*(1 if layout in seen else 3))
        seen.add(layout)
    calls = report['cold_native_calls']
    guard339.require(len(sequence) == 42 and [c['layout'] for c in calls] == sequence,
                     'all36 finite calls plus exactly6 retained first-shape qualification calls')
    before, after = report['original_engine_metrics_before'], report['original_engine_metrics_after']
    guard339.require(after['aclgraph_captures']-before['aclgraph_captures'] == 3,
                     'exact three current target graph captures')
    guard339.require(after['aclgraph_runtime_validation_replays']-before['aclgraph_runtime_validation_replays'] == 3,
                     'exact three changed-input validations completed')


def corrected_reader():
    original = guard339.numerical_guard.actual_audit
    source = inspect.getsource(original)
    before = "require(len(report['cold_native_calls']) == 36, 'one actual Native target invocation per finite case')"
    after = '_ROOT339_cold_calls(report, expected)'
    if source.count(before) != 1:
        raise RuntimeError('one exact original observation-count check required')
    changed = source.replace(before, after)
    assert changed.replace(after, before) == source
    namespace = dict(original.__globals__, _ROOT339_cold_calls=cold_calls)
    exec(compile(changed, __file__+'::one-original-audit-count-correction', 'exec'), namespace)
    return namespace['actual_audit']


def main():
    receipt_path = guard339.DONOR/f'receipt-{TAG}.json'
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    assert receipt['status'] == 'failed'
    assert len(receipt['errors']) == 1 and 'one actual Native target invocation per finite case' in receipt['errors'][0]
    assert receipt['sources_unchanged'] is True and receipt['owned_remaining'] == []
    assert receipt['child']['exit_code'] == 0
    assert receipt['devices_after']['all_idle'] is True
    assert receipt['pre_verifier']['actual_result']['status'] == receipt['post_verifier']['actual_result']['status'] == 'pass'
    result, status, evidence = corrected_reader()(OUTPUT)
    for rank in range(4):
        report = json.loads((OUTPUT/f'rank{rank}-numeric.json').read_text())
        checkpoint = json.loads((OUTPUT/f'rank{rank}-independent-bank-before-math339.json').read_text())
        assert checkpoint['rank'] == rank and checkpoint['evidence'] == report['bank']
        if rank:
            assert report['activation']['all_original_wrapper_code_and_signatures_preserved'] is True
    names = list(result['costs'][0]['median_ms'])
    median_cost = {name: statistics.median(row['median_ms'][name] for row in result['costs']) for name in names}
    saved = dict(status='completed_component_independently_audited_original_supervisor_failed_count_check',
        original_receipt=str(receipt_path), original_receipt_SHA=hashlib.sha256(receipt_bytes).hexdigest(),
        original_receipt_unchanged=True, original_supervisor_status='failed',
        original_child_exit=0, owned_cleanup=True, all_four_idle=True,
        finite_case_and_original_qualification_calls=42,
        strict_original_math_and_cost_reader_status=status, numerical=result, evidence=evidence,
        median_of_three_rank_medians_ms=median_cost,
        actual_dynamic_B_or_tree_branch_qualified=False, Goodput=None, adopted=False)
    assert receipt_path.read_bytes() == receipt_bytes
    target = B/'budget-diagnostics/ROOT339_INDEPENDENT_LINEAR_MATH_AND_COST_AUDIT.json'
    target.write_text(json.dumps(saved, indent=2)+'\n')
    print(json.dumps(saved))


if __name__ == '__main__':
    main()
