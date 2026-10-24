"""Read a closed, fully qualified private run; never starts or modifies workers."""
import argparse
import hashlib
import json
from pathlib import Path

from final_common_comparison import collect, FIXED_FLOOR, OFFICIAL_FLOOR
from read_common_publication_metrics import read, descriptor


def need(ok, message):
    if ok is not True:
        raise ValueError(message)


def collect_closed(root, tag, *, expected_freeze=None):
    root = Path(root)
    receipt_path = root / ('receipt-' + tag + '.json')
    receipt = read(receipt_path)
    need(receipt['tag'] == tag and receipt['status'] == 'complete'
         and receipt['errors'] == [] and receipt['sources_unchanged'] is True
         and receipt['actual_execution_complete'] is True
         and receipt['actual_full_status'] == 'passed_current_canonical_full_AR',
         'Complete original supervisor and unchanged source required')
    need(receipt['owned_remaining'] == [] and receipt['devices_after']['all_idle'] is True
         and receipt['post_verifier']['exit_code'] == 0,
         'Owned cleanup, all-card idle and post-verifier required')
    audit = receipt['actual_full_audit']
    need(audit['passed'] is True and audit['math_independent_qualified'] is True
         and audit['whole_workload_AR_qualified'] is True
         and audit['original_arrivals_and_outer_wall'] is True
         and audit['hidden_KV_recomputed_by_HOST'] is False
         and audit['global_active_limit'] == 64,
         'All original neural/AR checks, global64 and actual clocks required')
    freeze = receipt['readiness']['own_freeze_sha256']
    if expected_freeze is not None:
        need(freeze == expected_freeze, 'The preselected source freeze is required')
    directory = root / 'actual' / tag
    trace_path = directory / 'windows/window64/full463/rank1-full463-actual-trace.json.gz'
    trace = read(trace_path)
    trace_sha = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    ids = {row['request_id'] for row in trace['requests']}
    need(len(ids) == 463, 'All 463 distinct original request identities required')
    references = []
    for rank in (1, 2, 3):
        path = directory / 'postfull-independent-FIA-AR' / f'rank{rank}-summary.json'
        reference = read(path)
        comparison = reference['comparison']
        need(reference['rank'] == rank and reference['completed'] is True
             and reference['same_prefill_cohorts'] is True
             and reference['reference_trajectories_generated'] == 463
             and reference['reference_only_prompt_plus_own_outputs'] is True
             and reference['candidate_KV_or_tokens_seeded'] is False
             and reference['performance_clock_already_finished'] is True,
             'All three independent own-history reference trajectories required')
        rank_trace = trace_path.with_name(f'rank{rank}-full463-actual-trace.json.gz')
        need(reference['source_artifact'] == str(rank_trace)
             and reference['source_artifact_sha256'] == descriptor(rank_trace)['sha256']
             and reference['candidate_result_owner_rank'] == 1
             and reference['candidate_result_artifact'] == descriptor(trace_path)
             and reference['reference_geometry_source_artifact'] == str(trace_path)
             and reference['reference_geometry_source_sha256'] == trace_sha,
             'Independent reference must compare these exact measured output bytes')
        need(comparison['requests'] == comparison['equal_requests'] == 463
             and comparison['mismatching_requests'] == 0
             and comparison['exact_all_tokens_equal'] is True
             and len(comparison['rows']) == 463
             and {r['request_id'] for r in comparison['rows']} == ids
             and all(r['equal'] is True and r['first_mismatch_zero_based'] is None
                     for r in comparison['rows']),
             '463/463 exact-token equalities on every target rank required')
        references.append(dict(rank=rank, artifact=descriptor(path), requests=463,
                               equal_requests=463, mismatching_requests=0))
    metrics = collect('qualified_dynamic_tree', trace_path)
    measured = metrics['all']
    return dict(schema='SpecSLO.closed-qualified-tree-result.HOST.v1', tag=tag,
        receipt=descriptor(receipt_path), candidate_freeze_sha256=freeze,
        references=references, metrics=metrics,
        Goodput_to_historical_fixed=measured['Goodput'] / FIXED_FLOOR,
        Goodput_to_strong_historical_MC2=measured['Goodput'] / OFFICIAL_FLOOR,
        historical_1p5_MC2_target=1.5 * OFFICIAL_FLOOR,
        single_run_exceeds_fixed=measured['Goodput'] > FIXED_FLOOR,
        single_run_exceeds_1p5_MC2=measured['Goodput'] > 1.5 * OFFICIAL_FLOOR,
        single_run_SLO_at_least_80_percent=measured['attained'] / 463 >= .8,
        matched_nine_cell_comparison_inferred=False,
        performance_stability_inferred=False, HOST_readonly_arithmetic=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--expected-freeze')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(not args.output.exists(), 'A fresh result artifact is required')
    result = collect_closed(args.root, args.tag, expected_freeze=args.expected_freeze)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(tag=args.tag, **result['metrics']['all'],
        references=[r['equal_requests'] for r in result['references']],
        Goodput_to_strong_MC2=result['Goodput_to_strong_historical_MC2'])))


if __name__ == '__main__':
    main()
