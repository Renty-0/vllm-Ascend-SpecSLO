"""HOST-only protocol inventory, independent auditor negative controls and pins."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import statistics
import sys
from unittest.mock import patch

from probe365 import HERE, records, digest, SOURCE
import audit365


def main():
    output = HERE / 'HOST365_RESULT.json'
    assert not output.exists()
    assert len(records()) == 11 and len(set(records())) == 11
    for n, origin in records():
        assert 1 <= n <= 32 and 0 <= origin and origin + n <= 32
        assert 5 * (1 << n.bit_length()) <= 512
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    tree = ast.parse((HERE / 'probe365.py').read_text())
    forbidden = {'runtime_validated', 'validated_real_row_count', 'disabled_entry_keys',
                 '_capture_budget_used', 'graph_cache_sealed', 'enabled'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            assert node.attr not in forbidden
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr in ('init_process_group', 'NativePearlEngine')
                   for node in ast.walk(tree))
    # Match actual original Native declaration and every directly bound method.
    original_tree = ast.parse((SOURCE / 'vllm_ascend/spec_decode/pearl/native_engine.py').read_text())
    original_class = next(node for node in original_tree.body
                          if isinstance(node, ast.ClassDef) and node.name == 'NativePearlEngine')
    methods = {node.name for node in original_class.body if isinstance(node, ast.FunctionDef)}
    required = {'_allocate_cache', '_activate_cache_sequence', '_ensure_cache_capacity',
                '_release_cache', '_cache_slot_mapping', '_prepare_attention_metadata',
                '_pack_tree_draft_level', '_pad_tree_draft_graph_requests', 'draft_tree_forward'}
    assert required <= methods
    rows = []
    for n, origin in records():
        for phase in ('A1', 'B', 'A2'):
            candidates = []
            for final in (False, True):
                q = 5 * (1 << n.bit_length())
                candidates.append(dict(final_kv_only=final, all8_tokens_exact=True,
                    all8_FP32_probabilities_bitwise=True, all_KV_tensors=56,
                    physical_queries_per_step=q, physical_model_queries=q * (5 if final else 4),
                    real_model_queries=n * (5 if final else 4), original_execution=dict(mode='replay'),
                    entry_key=[f'fixture|ROOT364:raw-spine4-top2-float32-bits|logical:{n}', q]))
            rows.append(dict(N=n, origin=origin, phase=phase, candidates=candidates))
    costs = [dict(N=n, pair=i, mode=mode, seconds=0.002 + i / 1000000)
             for n in (4, 12, 24, 32) for i in range(20) for mode in ('generic4', 'chain4', 'chain5')]
    summary = {f'N{n}': {mode: statistics.median([r['seconds'] for r in costs if r['N'] == n and r['mode'] == mode])
                        for mode in ('generic4', 'chain4', 'chain5')} for n in (4, 12, 24, 32)}
    fixture = dict(status='complete', source_sha256=digest(HERE / 'probe365.py'),
        models_loaded=['Qwen3-0.6B'], target_model_loaded=False, dtype='BF16', quantized=False,
        TP=1, independent_prefills=96, layers=28, math_rows=rows, costs=costs,
        median_seconds=summary, all_committed_prefix_KV_protected=True,
        native_qualification_bodies_unchanged=True,
        graph_stats=dict(unvalidated_entries=0, disabled_entries=0),
        Raw=None, Goodput=None, adopted=False, goal_achieved=False)
    def run(value):
        with patch.object(Path, 'read_text', return_value=json.dumps(value)), \
             patch.object(audit365, 'digest', return_value=fixture['source_sha256']):
            return audit365.audit(Path('/HOST-fixture-only'))
    assert run(fixture)['status'] == 'complete_real_draft_component_audited'
    negative = []
    for label in ('missing_math', 'KV_tolerance', 'probability_bits', 'fallback', 'uncharged_fifth',
                  'unqualified', 'missing_timing', 'invented_Raw', 'target_loaded'):
        value = copy.deepcopy(fixture)
        if label == 'missing_math': value['math_rows'].pop()
        if label == 'KV_tolerance': value['math_rows'][0]['candidates'][0]['all_KV_tensors'] = 55
        if label == 'probability_bits': value['math_rows'][0]['candidates'][0]['all8_FP32_probabilities_bitwise'] = False
        if label == 'fallback': value['math_rows'][0]['candidates'][0]['original_execution']['mode'] = 'eager'
        if label == 'uncharged_fifth': value['math_rows'][0]['candidates'][1]['real_model_queries'] -= 1
        if label == 'unqualified': value['graph_stats']['unvalidated_entries'] = 1
        if label == 'missing_timing': value['costs'].pop()
        if label == 'invented_Raw': value['Raw'] = 1000
        if label == 'target_loaded': value['target_model_loaded'] = True
        try:
            run(value)
        except RuntimeError:
            negative.append(label)
        else:
            raise AssertionError('auditor accepted corruption: ' + label)
    assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
    result = dict(status='PASS_HOST_inventory_original_bindings_and_independent_auditor',
        probe_sha256=digest(HERE / 'probe365.py'), math_rows=len(rows), timing_calls=len(costs),
        rejected_corruptions=negative, original_method_bindings=sorted(required),
        model_loaded=False, NPU_used=False, model_qualified=False, Raw=None, Goodput=None)
    with output.open('x') as saved:
        json.dump(result, saved, indent=2)
        saved.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
