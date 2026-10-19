"""Normalize complete measured traces; require terminal evidence for qualification."""
import argparse
import collections
import json
from pathlib import Path

from read_common_publication_metrics import tree, read, descriptor
from final_common_comparison import FIXED_FLOOR, OFFICIAL_FLOOR

B = Path(__file__).resolve().parent.parent
TAG = 'M120-B8-coalesced-refill4-root-v346-current-cost-full463-ownAR-v1'


def summarize(path):
    metrics, _, contract = tree(Path(path))
    trace = read(path)
    calls = trace['normal_target_calls']
    budgets = collections.Counter(b for row in calls for b in row['budgets'])
    assert calls and all(1 <= b <= 8 for b in budgets)
    n = [row['N'] for row in calls]
    gp = metrics['all']['Goodput']
    return dict(status='measured_trace_arithmetic_only', source=descriptor(path),
        input_contract=contract, all=metrics['all'], by_class=metrics['by_class'],
        elapsed_seconds=metrics['elapsed_seconds'], TPOT_definition=metrics['TPOT_definition'],
        target_calls=len(calls), mean_N=sum(n)/len(n), maximum_N=max(n),
        budget_owner_rounds=dict(sorted(budgets.items())),
        actual_branch_target_calls=sum(any(b>4 for b in row['budgets']) for row in calls),
        phase_seconds=trace['phase_seconds'],
        historical_fixed_gamma4_GP=FIXED_FLOOR, historical_strong_official_MC2_GP=OFFICIAL_FLOOR,
        GP_over_fixed_gamma4=gp/FIXED_FLOOR, GP_over_strong_official_MC2=gp/OFFICIAL_FLOOR,
        historical_1p5_threshold=1.5*OFFICIAL_FLOOR,
        measured_above_historical_floors=gp>FIXED_FLOOR and gp>1.5*OFFICIAL_FLOOR,
        best_clean_tree_GP=554.7604454409803,
        GP_over_best_clean_tree=gp/554.7604454409803,
        all_final_stream_tokens_reconstructed=True,
        whole463_own_AR_qualified=None, terminal_supervisor_complete=None,
        matched_repeats_completed=False, adopted=False, goal_achieved=False, NPU_used=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--terminal-audit', type=Path)
    args = parser.parse_args()
    assert not args.output.exists() and TAG in args.trace.parts
    value = summarize(args.trace)
    trace = read(args.trace)
    assert trace['decode_counters']['spec_rhythm_ROOT346_group_budget_enabled'] == 1
    import sys
    sys.path[:0] = [str(B/'tree-coalesced-live-refill-M120-B8-full-v346'),
                   str(B/'tree-causal80-small-cohort-role-aware-audit-full-v209')]
    from empty346 import audit_trace
    value['actual_refill_cohorts'] = audit_trace(trace)
    assert value['actual_refill_cohorts']['actual_inflight_refill_groups'] > 0
    if args.terminal_audit is not None:
        audit = read(args.terminal_audit)
        value['terminal_audit'] = descriptor(args.terminal_audit)
        value['terminal_supervisor_complete'] = audit['supervisor_status'] == 'complete'
        value['whole463_own_AR_qualified'] = audit['whole_workload_AR_qualified']
        assert audit['source_and_raw_manifest'][str(args.trace)]['sha256'] == value['source']['sha256']
        assert audit['status'] == 'terminal_completed_compute_audited' and not audit['errors']
        if value['terminal_supervisor_complete'] and value['whole463_own_AR_qualified']:
            value['status'] = 'complete_independently_audited_single_run'
    args.output.write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps({k:value[k] for k in ('status','all','by_class','GP_over_fixed_gamma4',
        'GP_over_strong_official_MC2','whole463_own_AR_qualified','goal_achieved')}))


if __name__ == '__main__': main()
