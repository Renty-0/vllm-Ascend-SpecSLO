"""HOST-only inventory and actual supervisor audit-ABI regression controls."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import statistics
import sys
from unittest.mock import patch

from probe370 import HERE, configurations
import audit370
import guard370


def main():
    output = HERE / 'HOST370_RESULT.json'
    assert not output.exists()
    rows = []
    for n, origin, profile in configurations():
        physical = 5 * (1 << n.bit_length())
        count = [0 if profile == 'none' else (4 if profile == 'all4' else i % 5) for i in range(n)]
        for phase in ('A1', 'B', 'A2'):
            rows.append(dict(N=n, origin=origin, profile=profile, phase=phase,
                catchup_lengths=[k + 1 for k in count], all8_tokens_exact=True,
                all8_FP32_probabilities_bitwise=True, mean_confidence_bitwise=True,
                physical_queries_equal=True, prefix_preserved=True, all_KV_tensors=56,
                original_graph_calls=4, candidate_graph_calls=2,
                original_actual_model_steps=4, candidate_actual_model_steps=4,
                query_count=4 * physical, tail=dict(query_count=3 * physical,
                    graph_calls=1, model_calls=1, actual_model_steps=3, logical_owners=n,
                    graph_padding_query_count=3 * (physical - n),
                    original_first_step_preserved=True, final_leaf_KV_materialized=False)))
    costs = [dict(N=n, profile=profile, pair=pair, mode=mode, seconds=.02 + pair / 1000000)
        for n in (4, 12, 24, 32) for profile in ('none', 'staggered') for pair in range(20)
        for mode in (('generic4', 'first_plus_tail3') if pair % 2 == 0 else ('first_plus_tail3', 'generic4'))]
    summary = {f'N{n}_{profile}': {mode: statistics.median([r['seconds'] for r in costs
                    if r['N'] == n and r['profile'] == profile and r['mode'] == mode])
                    for mode in ('generic4', 'first_plus_tail3')}
                    for n in (4, 12, 24, 32) for profile in ('none', 'staggered')}
    fixture = dict(status='complete', source_sha256=hashlib.sha256(b'HOST').hexdigest(),
        models_loaded=['Qwen3-0.6B'], target_model_loaded=False, TP=1, dtype='BF16', quantized=False,
        independent_prefills=96, layers=28, math_rows=rows, costs=costs, median_seconds=summary,
        all_committed_prefix_KV_protected=True, native_qualification_bodies_unchanged=True,
        graph_stats=dict(unvalidated_entries=0, disabled_entries=0),
        Raw=None, Goodput=None, adopted=False, goal_achieved=False, serving_installed=False)
    def run(value):
        with patch.object(Path, 'read_text', return_value=json.dumps(value)), \
             patch.object(Path, 'read_bytes', return_value=b'HOST'):
            return audit370.audit('/HOST-only-fixture')
    actual = run(fixture)
    assert isinstance(actual, tuple) and len(actual) == 3
    assert actual[0]['math_rows'] == 99 and actual[0]['timing_calls'] == 320
    with patch.object(audit370, 'audit', return_value=actual):
        assert guard370.actual_audit('/HOST-only-fixture') == actual
    with patch.object(audit370, 'audit', return_value=actual[0]):
        try: guard370.actual_audit('/HOST-only-fixture')
        except RuntimeError: pass
        else: raise AssertionError('incorrect supervisor dict ABI accepted')
    negatives = []
    for label in ('missing_math', 'missing_catchup', 'probabilities', 'KV', 'graph_fallback',
                  'dropped_query_charge', 'timing_order', 'prefix_corruption', 'unvalidated', 'invented_Raw'):
        value = copy.deepcopy(fixture)
        if label == 'missing_math': value['math_rows'].pop()
        if label == 'missing_catchup': value['math_rows'][-1]['catchup_lengths'][0] = 1
        if label == 'probabilities': value['math_rows'][0]['all8_FP32_probabilities_bitwise'] = False
        if label == 'KV': value['math_rows'][0]['all_KV_tensors'] = 55
        if label == 'graph_fallback': value['math_rows'][0]['candidate_graph_calls'] = 1
        if label == 'dropped_query_charge': value['math_rows'][0]['query_count'] -= 1
        if label == 'timing_order': value['costs'][0], value['costs'][1] = value['costs'][1], value['costs'][0]
        if label == 'prefix_corruption': value['all_committed_prefix_KV_protected'] = False
        if label == 'unvalidated': value['graph_stats']['unvalidated_entries'] = 1
        if label == 'invented_Raw': value['Raw'] = 1000
        try: run(value)
        except RuntimeError: negatives.append(label)
        else: raise AssertionError('corrupt result accepted: ' + label)
    for path in HERE.glob('*.py'): ast.parse(path.read_text(), filename=str(path))
    assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
    result = dict(status='PASS_HOST_inventory_independent_auditor_and_actual_three_item_guard_ABI',
        math_rows=99, timing_calls=320, rejected_corruptions=negatives,
        invalid_dict_supervisor_ABI_rejected=True, original_qualification_flag_assignments=False,
        probe_sha256=hashlib.sha256((HERE / 'probe370.py').read_bytes()).hexdigest(),
        NPU_used=False, model_qualified=False, Raw=None, Goodput=None)
    with output.open('x') as saved:
        json.dump(result, saved, indent=2)
        saved.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
