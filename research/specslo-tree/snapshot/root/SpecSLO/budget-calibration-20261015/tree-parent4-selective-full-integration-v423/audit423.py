"""Preserve every original full trace and own-AR audit, add strict bucket math."""
import json
import hashlib
from pathlib import Path
import audit_base423 as original
from projection423 import project_function
from bucket423 import valid_record


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT423 audit: ' + message)


def projected_original_audit():
    return project_function(original.audit, (
        ("and row['fixed_KV_extent']==[2560]*24", "and _ROOT423_valid_record(row)"),
    ), {'_ROOT423_valid_record': valid_record})


def audit(directory, selected_windows):
    result = projected_original_audit()(directory, selected_windows)
    from oracle257 import cases
    from dispatch423 import audit_calls
    target_heads = []
    normal_snapshots = []
    normal_audits = []
    E4_modes = []
    from audit419 import audit as audit_parent4
    parent4 = audit_parent4(Path(directory)/'parent4-current-component423')
    need(parent4['passed'] is True and parent4['full_required_natural_coverage'] is True,
         'all75 same-process original parent4 lifecycle histories')
    for rank in range(4):
        path = Path(directory) / f'rank{rank}-local-diagnostic.json'
        report = json.loads(path.read_text())
        from audit_normal423 import verify_snapshot
        import gzip
        trace = json.loads(gzip.decompress((Path(directory) / 'windows/window64/full463'
                         / f'rank{rank}-full463-actual-trace.json.gz').read_bytes()))
        normal_snapshots.append(report['ROOT423_normal_admission']['normal_admission'])
        normal_audits.append(verify_snapshot(report['ROOT423_normal_admission'], trace['source_generation']))
        current = report['ROOT257_actual_geometry']['selective_parent4_E4_current_proof']
        need(current['issued'] is True and current['same_engine_protocol_cases'] == 75
             and current['repeated_zero_cases'] == 50
             and current['natural_mixed_and_leaf_coverage'] is True
             and current['Native_flags_assigned'] is False,
             'full production uses current explicitly issued E4 proof')
        modes = report['ROOT423_normal_admission']['modes_by_generation'][str(trace['source_generation'])]
        need(all(type(v) is int and v >= 0 for v in modes.values())
             and sum(modes.values()) == normal_audits[-1]['calls'],
             'actual full463 typed calls and mode accounting')
        E4_modes.append(modes)
        proof = report['ROOT423_uniform_KV_math']
        need(proof['cases'] == 681 and proof['same_model_instance'] is True
             and len(proof['banks']) == 3 and proof['original227_FULL_math_preserved'] is True
             and proof['native_flags_assigned'] is False, 'all additional same-model math')
        heads = []
        for descriptor in proof['banks']:
            path = Path(descriptor['path']).resolve()
            need(path.is_relative_to(Path(directory).resolve()) and descriptor['cases'] == 227
                 and hashlib.sha256(path.read_bytes()).hexdigest() == descriptor['sha256'],
                 'complete current retained bank proof and exact digest')
            bank = json.loads(path.read_text())
            need(len(bank['math']['cases']) == 227, 'all original227 shapes at each context bank')
            if rank == 0:
                need(bank['bank'] is None and bank['dispatch'] is None
                     and all(v is None for v in bank['math']['cases']), 'draft makes no target neural claim')
                continue
            need(bank['bank']['independent_neural_prefills'] == 72
                 and bank['bank']['all_triplet_roots_equal'] is True
                 and bank['bank']['all_triplet_prefix_KV_equal'] is True
                 and bank['math']['all_independent_prefixes_unchanged'] is True,
                 'all72 independent roots and committed prefixes exact')
            # Reconstruct every extent from independently retained prefixes and
            # original logical budgets, rather than trusting a call's extent.
            prefixes = [list(range(n)) for n in bank['prefix_lengths']]
            audit_calls(cases(), bank['actual_graph_calls'], prefixes)
            for expected, observed in zip(cases(), bank['math']['cases'], strict=True):
                need(observed['case'] == expected and observed['exact_greedy'] is True
                     and observed['hidden_bitwise_equal'] is True and observed['KV_bitwise_equal'] is True
                     and observed['all128_KV_original_matcher_passed'] is True
                     and observed['candidate_tokens'] == observed['reference_tokens']
                     and observed['candidate_KV_copied'] is False,
                     'original227 node matcher and strict hidden/KV bits')
                heads.append(observed['candidate_tokens'])
        if rank:
            target_heads.append(heads)
    need(target_heads[0] == target_heads[1] == target_heads[2], 'all3 target heads agree')
    need(all(value == normal_snapshots[0] for value in normal_snapshots)
         and normal_audits[0]['overflow_calls'] > 0, 'all4 actual class/age normal24 histories and actual overflow')
    need(all(v == E4_modes[0] for v in E4_modes), 'all4 actual allocation modes agree')
    return dict(result, ROOT423_additional_current_model_cases=681,
        ROOT423_original227_and_complete463_own_AR_preserved=True,
        ROOT423_original_best257_draft_and_allocator_unchanged=False,
        ROOT423_best257_draft_neural_math_unchanged=False,
        ROOT423_original_RootSeed_legacy_Fast_fallback_preserved=True,
        ROOT423_current_parent4_same_engine_cases=75,
        ROOT423_current_parent4_full_natural_coverage=True,
        ROOT423_full463_actual_modes=E4_modes[0],
        ROOT423_E4_mode_observed=E4_modes[0].get('proposed_exact_normal24_B4_future4_B4', 0) > 0,
        ROOT423_normal_work_policy_limit=24,
        ROOT423_actual_normal_admission=normal_audits)
