"""Preserve every original full trace and own-AR audit, add strict bucket math."""
import json
import hashlib
from pathlib import Path
import audit257 as original
from projection376 import project_function
from bucket376 import valid_record


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT376 audit: ' + message)


def projected_original_audit():
    return project_function(original.audit, (
        ("and row['fixed_KV_extent']==[2560]*24", "and _ROOT376_valid_record(row)"),
    ), {'_ROOT376_valid_record': valid_record})


def audit(directory, selected_windows):
    result = projected_original_audit()(directory, selected_windows)
    from oracle257 import cases
    from dispatch376 import audit_calls
    target_heads = []
    for rank in range(4):
        path = Path(directory) / f'rank{rank}-local-diagnostic.json'
        report = json.loads(path.read_text())
        proof = report['ROOT376_uniform_KV_math']
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
    return dict(result, ROOT376_additional_current_model_cases=681,
        ROOT376_original227_and_complete463_own_AR_preserved=True,
        ROOT376_original_best257_draft_and_allocator_unchanged=True)
