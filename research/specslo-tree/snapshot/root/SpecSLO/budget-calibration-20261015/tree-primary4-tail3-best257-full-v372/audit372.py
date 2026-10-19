"""Entire original257 full audit plus fresh draft math and actual LIVE use."""
import gzip
import json
from pathlib import Path

import audit257 as original


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT372 full audit: ' + message)


def audit(directory, selected_windows):
    directory = Path(directory)
    previous = original.audit(directory, selected_windows)
    need(previous['passed'] is True and previous['whole_workload_AR_qualified'] is True,
         'all original startup/target227/full463/independent own-AR gates')
    from audit_draft372 import audit as draft_audit
    math = draft_audit(directory / 'draft-tail3-same-instance')
    reports = [json.loads((directory / f'rank{rank}-local-diagnostic.json').read_text()) for rank in range(4)]
    for rank, report in enumerate(reports):
        need(report['standalone_scope']['fresh_same_instance_draft306_required'] is True
             and report['standalone_scope']['original_target257_and_all_prior_obligations_preserved'] is True
             and report['standalone_scope']['Native_qualification_flag_assignment'] is False,
             'truthful extended scope with all original obligations')
        view = report['ROOT372_draft_tail']
        need(view['is_draft'] is (rank == 0), 'actual original draft/target roles')
        if rank:
            need(view['changed_target_model_or_sampler'] is False, 'target numerical path unchanged')
    view = reports[0]['ROOT372_draft_tail']
    need(view['private_entry_count'] == 64 and view['proof']['cases'] == 306
         and view['proof']['fresh_same_model_instance'] is True
         and view['standalone_component_qualification_inherited'] is False
         and view['Native_qualification_flags_assigned'] is False,
         'same-instance naturally qualified draft backend')
    trace_path = directory / 'windows/window48/full463/rank0-full463-actual-trace.json.gz'
    trace = json.loads(gzip.decompress(trace_path.read_bytes()))
    counts = view['counts_by_generation'][str(trace['source_generation'])]
    need(counts['tail_applied'] > 0 and counts['calls'] == counts['tail_applied'] + counts['fallback'],
         'real formal full463 exercised the new tail')
    need(counts['actual_tail_graphs'] == counts['tail_applied']
         and counts['actual_four_primary_steps'] == 4 * counts['tail_applied']
         and counts['actual_graph_calls'] >= 2 * counts['tail_applied'] + counts['original_frontier_graphs']
         and counts['actual_physical_queries'] > 0 and counts['logical_owners'] > 0,
         'truthful graph submissions, same four steps, and actual query charges')
    return dict(previous, ROOT372_fresh_draft_math=math, ROOT372_formal_draft_counts=counts,
                ROOT372_original_full_audit_preserved=True)
