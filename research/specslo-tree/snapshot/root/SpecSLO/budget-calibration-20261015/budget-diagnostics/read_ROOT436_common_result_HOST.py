"""Read independently reconstructed full clocks; no NPU or new metric formula."""
import argparse
import hashlib
import json
from pathlib import Path

FIXED = 501.63560407964184
OFFICIAL = 429.8442987424407
BEST_TREE = 609.0250374874602


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT425 result: ' + message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--terminal-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(not args.output.exists(), 'new immutable result artifact')
    a = json.loads(args.terminal_audit.read_text())
    need(a['schema'] == 'ROOT436.independent-terminal-compute-audit.v1'
         and a['status'] == 'terminal_completed_compute_audited' and a['errors'] == []
         and a['supervisor_status'] == 'complete' and a['supervisor_errors'] == [],
         'original supervisor and independent terminal audit complete')
    need(a['whole_workload_AR_qualified'] is True
         and a['completed_compute_audit']['whole_workload_AR_qualified'] is True,
         'all463 complete independent own-history checks')
    need(a['completed_compute_audit']['ROOT425_additional_current_model_cases'] == 681
         and a['completed_compute_audit']['ROOT425_original227_and_complete463_own_AR_preserved'] is True
         and a['completed_compute_audit']['ROOT425_original_best257_draft_and_allocator_unchanged'] is False
         and a['completed_compute_audit']['ROOT425_best257_draft_neural_math_unchanged'] is False
         and a['completed_compute_audit']['ROOT425_current_parent4_same_engine_cases'] == 75
         and a['completed_compute_audit']['ROOT425_current_parent4_full_natural_coverage'] is True
         and a['completed_compute_audit']['ROOT425_normal_work_policy_limit'] == 24
         and len(a['completed_compute_audit']['ROOT425_actual_normal_admission']) == 4,
         'fresh same-instance uniform KV math and actual best257 optimized draft retained')
    extra = a['completed_compute_audit']['ROOT432_current_original427_NN_proof']
    need(extra['passed'] is True
         and extra['all_old_allocated_KV_pages_and_draft_ledger_protected'] is True
         and extra['original_new_prefix_reference_root_next_and_twozero_gates_retained'] is True,
         'fresh actual six-case incumbent KV and ledger protection proof')
    refill = a['metrics']['actual_admission_cohort_replay']
    need(refill['nonempty_cohorts'] > 0
         and refill['source_clocks_unchanged'] is True
         and refill['exact_original_incumbent_prefill_charge'] is True,
         'actual nonempty refill observed, exact original incumbent waiting retained')
    metrics = a['metrics']['metrics']
    actual = metrics['host_stream_paper']
    raw = metrics['raw_tokens_per_second']
    gp = actual['Goodput']
    need(0 < gp <= raw and 0 <= actual['attained'] <= 463, 'actual stream arithmetic')
    classes = {name: dict(requests=value['requests'], attained=value['host_stream_paper']['attained'],
                 SLO_ms=value['SLO_ms'], TPOT_median_ms=value['host_stream_paper']['median_ms'],
                 TPOT_P90_ms=value['host_stream_paper']['P90_ms'], Goodput=value['host_stream_paper']['Goodput'])
               for name, value in metrics['by_class'].items()}
    need(sum(v['requests'] for v in classes.values()) == 463
         and sum(v['attained'] for v in classes.values()) == actual['attained'], 'all original classes')
    result = dict(status='complete_independently_audited_single_run',
        source_terminal_audit=str(args.terminal_audit),
        source_terminal_audit_sha256=hashlib.sha256(args.terminal_audit.read_bytes()).hexdigest(),
        source_freeze_sha256=a['source_freeze_sha256'], source_archive_sha256=a['source_archive_sha256'],
        Raw=raw, Goodput=gp, attained=actual['attained'], requests=463,
        fresh_current_incumbent_prefill_six_cases_qualified=True,
        actual_nonempty_refill=refill,
        actual_prefill_helpers=len(refill['cohorts']),
        actual_prefill_seconds=sum(row['prefill_ms'] for row in refill['cohorts'])/1000.,
        incumbent_prefill_pause_memberships=sum(row['incumbent_count'] for row in refill['cohorts']),
        total_incumbent_prefill_charge_ms=sum(row['incumbent_count']*row['prefill_ms'] for row in refill['cohorts']),
        actual_normal_work_policy_limit=24,
        actual_class_age_admission=a['completed_compute_audit']['ROOT425_actual_normal_admission'],
        prior_global64_control_GP=214.96187487135404,
        GP_over_prior_global64_control=gp/214.96187487135404,
        active_request_ceiling=64, actual_active=a['actual_active64'], historical_best_tree_active_ceiling=48,
        actual_target_segment_packing=a['actual_target_segment_packing425'],
        TPOT_median_ms=actual['median_ms'], TPOT_P90_ms=actual['P90_ms'],
        TPOT_definition=actual['clock'], TPOT_denominator=actual['TPOT_denominator'], by_class=classes,
        observed_TPOT_parallel_definition=metrics['host_stream_observed'],
        target_calls=a['metrics']['target_calls'], mean_N=a['metrics']['mean_N'],
        budget_owner_rounds=a['metrics']['budget_owner_rounds'],
        whole463_own_AR_qualified=True, fresh_target227_and_all_original_components_qualified=True,
        fresh_target681_uniform_KV_qualified=True, original_best257_optimized_draft_preserved=False, original_RootSeed_legacy_Fast_retained=True,
        current75_new_dispatcher_cases=True, actual_full_modes=a['completed_compute_audit']['ROOT425_full463_actual_modes'],
        E4_mode_observed=a['completed_compute_audit']['ROOT425_E4_mode_observed'],
        original_global_warm_twozero_sealing_retained=True,
        historical_fixed_gamma4_GP=FIXED, historical_strong_official_MC2_GP=OFFICIAL,
        historical_best_tree_GP=BEST_TREE, GP_over_fixed_gamma4=gp / FIXED,
        GP_over_strong_official_MC2=gp / OFFICIAL, GP_over_historical_best_tree=gp / BEST_TREE,
        historical_1p5_threshold=1.5 * OFFICIAL,
        measured_above_historical_floors=(gp > FIXED and gp > 1.5 * OFFICIAL),
        matched_repeats_completed=0, adopted=False, goal_achieved=False, NPU_used=False)
    with args.output.open('x') as saved:
        json.dump(result, saved, indent=2)
        saved.write('\n')
    print(json.dumps({key: result[key] for key in ('status', 'Raw', 'Goodput', 'attained',
                     'GP_over_fixed_gamma4', 'GP_over_strong_official_MC2', 'measured_above_historical_floors')}))


if __name__ == '__main__':
    main()
