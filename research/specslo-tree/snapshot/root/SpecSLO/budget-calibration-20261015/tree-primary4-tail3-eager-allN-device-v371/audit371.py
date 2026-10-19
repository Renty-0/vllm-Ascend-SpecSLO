"""Independent all-N/mixed-eager inventory and original three-item guard ABI."""
import hashlib
import json
import math
from pathlib import Path
import statistics

from probe371 import HERE, configurations


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT371 audit: ' + message)


def audit(directory):
    path = Path(directory) / 'result.json'
    a = json.loads(path.read_text())
    need(a['status'] == 'complete' and a['source_sha256'] == hashlib.sha256((HERE / 'probe371.py').read_bytes()).hexdigest(), 'source')
    need(a['models_loaded'] == ['Qwen3-0.6B'] and a['target_model_loaded'] is False and a['TP'] == 1
         and a['dtype'] == 'BF16' and a['quantized'] is False, 'real draft-only BF16 scope')
    need(a['independent_prefills'] == 96 and a['layers'] == 28, 'independent prefix banks and28 layers')
    expected = {(n, origin, mode, phase) for n, origin, mode in configurations() for phase in ('A1', 'B', 'A2')}
    rows = a['math_rows']
    need(len(rows) == len(expected) == 306 and {(r['N'], r['origin'], r['mode'], r['phase']) for r in rows} == expected,
         'all306 distinct all-N/normal/mixed/eager/ABA cases')
    for r in rows:
        n = r['N']
        eager = 0 if r['mode'] == 'normal' else (n if r['mode'] == 'all_eager' else (n + 1) // 2)
        need(r['eager_owners'] == eager, 'actual eager-owner inventory')
        need(all(r[k] is True for k in ('all8_tokens_exact', 'all8_FP32_probabilities_bitwise',
             'mean_confidence_bitwise', 'original_eager_frontier_roots_exact', 'eager_promotion_map_exact',
             'physical_queries_equal')), 'all original exact output and promotion contracts')
        need(r['all_KV_tensors'] == 56 and r['A2_restored'] is (True if r['phase'] == 'A2' else None), 'all56 KV and ABA')
        need(r['original_graph_calls'] == 4 + bool(eager) and r['candidate_graph_calls'] == 2 + bool(eager)
             and r['original_actual_model_steps'] == r['candidate_actual_model_steps'] == 4 + bool(eager), 'actual work')
        physical = 5 * (1 << n.bit_length())
        frontier = 5 * (1 << eager.bit_length()) if eager else 0
        need(r['query_count'] == 4 * physical + frontier, 'all physical frontier and primary queries charged')
        tail = r['tail']
        need(tail['graph_calls'] == tail['model_calls'] == 1 and tail['actual_model_steps'] == 3
             and tail['query_count'] == 3 * physical and tail['graph_padding_query_count'] == 3 * (physical - n)
             and tail['logical_owners'] == n and tail['original_first_step_preserved'] is True
             and tail['final_leaf_KV_materialized'] is False, 'original first/frontier and three-step tail')
    costs = a['costs']
    order = [(n, mode, pair, name) for n in (4, 12, 24, 32) for mode in ('mixed_eager', 'all_eager') for pair in range(20)
             for name in (('generic', 'first_plus_tail3') if pair % 2 == 0 else ('first_plus_tail3', 'generic'))]
    need(len(costs) == 320 and [(r['N'], r['mode'], r['pair'], r['operation']) for r in costs] == order,
         'all320 repeated measurements in balanced order')
    need(all(type(r['seconds']) is float and math.isfinite(r['seconds']) and r['seconds'] > 0 for r in costs), 'real positive times')
    summary = {f'N{n}_{mode}': {name: statistics.median([r['seconds'] for r in costs
                if r['N'] == n and r['mode'] == mode and r['operation'] == name])
                for name in ('generic', 'first_plus_tail3')}
                for n in (4, 12, 24, 32) for mode in ('mixed_eager', 'all_eager')}
    need(summary == a['median_seconds'], 'recomputed medians')
    need(all(a[k] is True for k in ('all_committed_prefix_KV_protected', 'native_qualification_bodies_unchanged',
         'all_logical_N_1_to32_exercised', 'normal_mixed_and_all_eager_exercised')), 'all committed prefixes and coverage')
    need(a['graph_stats']['unvalidated_entries'] == a['graph_stats']['disabled_entries'] == 0, 'all original Native qualification')
    need(a['serving_installed'] is False and a['Raw'] is None and a['Goodput'] is None and a['adopted'] is False
         and a['goal_achieved'] is False, 'component-only scope')
    result = dict(status='completed_real_draft_allN_mixed_eager_tail3_component', math_rows=306, timing_calls=320,
        median_seconds=summary, result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        full463_qualified=False, Raw=None, Goodput=None, adopted=False)
    return result, 'completed_draft_eager_component_no_serving_qualification', dict(
        all306_frontier_probability_promotion_and56KV_cases_recounted=True, balanced320_timings_recounted=True)
