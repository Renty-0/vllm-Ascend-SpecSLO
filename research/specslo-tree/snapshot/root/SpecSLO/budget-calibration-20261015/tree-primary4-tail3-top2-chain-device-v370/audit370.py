"""Recount all real draft math and balanced timing cells; original guard ABI."""
import hashlib
import json
import math
from pathlib import Path
import statistics

from probe370 import HERE, configurations


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT370 audit: ' + message)


def audit(directory):
    path = Path(directory) / 'result.json'
    a = json.loads(path.read_text())
    need(a['status'] == 'complete' and a['source_sha256'] == hashlib.sha256((HERE / 'probe370.py').read_bytes()).hexdigest(),
         'exact completed source')
    need(a['models_loaded'] == ['Qwen3-0.6B'] and a['target_model_loaded'] is False
         and a['TP'] == 1 and a['dtype'] == 'BF16' and a['quantized'] is False, 'actual model scope')
    need(a['independent_prefills'] == 96 and a['layers'] == 28, 'three original independent banks')
    expected = {(n, origin, profile, phase) for n, origin, profile in configurations()
                for phase in ('A1', 'B', 'A2')}
    need(len(a['math_rows']) == len(expected) == 99 and
         {(r['N'], r['origin'], r['profile'], r['phase']) for r in a['math_rows']} == expected,
         'all99 exact shape/origin/catchup/ABA cells')
    for r in a['math_rows']:
        n = r['N']
        count = [0 if r['profile'] == 'none' else (4 if r['profile'] == 'all4' else j % 5) for j in range(n)]
        need(r['catchup_lengths'] == [k + 1 for k in count], 'all genuine committed catch-up queries present')
        need(all(r[k] is True for k in ('all8_tokens_exact', 'all8_FP32_probabilities_bitwise',
             'mean_confidence_bitwise', 'physical_queries_equal', 'prefix_preserved')), 'all exact original contracts')
        need(r['all_KV_tensors'] == 56 and r['original_graph_calls'] == 4 and r['candidate_graph_calls'] == 2,
             'all-layer KV and real graph submissions')
        need(r['original_actual_model_steps'] == r['candidate_actual_model_steps'] == 4,
             'same four dependent model steps')
        physical = 5 * (1 << n.bit_length())
        need(r['query_count'] == 4 * physical, 'all actual physical queries charged')
        detail = r['tail']
        need(detail['query_count'] == 3 * physical and detail['graph_calls'] == detail['model_calls'] == 1
             and detail['actual_model_steps'] == 3 and detail['logical_owners'] == n,
             'one actual tail graph with all three model steps')
        need(detail['graph_padding_query_count'] == 3 * (physical - n)
             and detail['original_first_step_preserved'] is True
             and detail['final_leaf_KV_materialized'] is False, 'original first/catch-up and deferred leaf policy')
    costs = a['costs']
    expected_costs = [(n, profile, pair, mode) for n in (4, 12, 24, 32)
        for profile in ('none', 'staggered') for pair in range(20)
        for mode in (('generic4', 'first_plus_tail3') if pair % 2 == 0 else ('first_plus_tail3', 'generic4'))]
    need(len(costs) == 320 and [(r['N'], r['profile'], r['pair'], r['mode']) for r in costs] == expected_costs,
         'all balanced same-instance timings in predeclared order')
    need(all(type(r['seconds']) is float and math.isfinite(r['seconds']) and r['seconds'] > 0 for r in costs),
         'real positive measured durations')
    summary = {f'N{n}_{profile}': {mode: statistics.median([r['seconds'] for r in costs
                    if r['N'] == n and r['profile'] == profile and r['mode'] == mode])
                    for mode in ('generic4', 'first_plus_tail3')}
                    for n in (4, 12, 24, 32) for profile in ('none', 'staggered')}
    need(summary == a['median_seconds'], 'recomputed component medians')
    need(a['all_committed_prefix_KV_protected'] is True and a['native_qualification_bodies_unchanged'] is True,
         'all96 protected committed prefixes and unchanged Native checks')
    need(a['graph_stats']['unvalidated_entries'] == a['graph_stats']['disabled_entries'] == 0,
         'all resident graph qualification intact')
    need(a['Raw'] is None and a['Goodput'] is None and a['adopted'] is False and a['goal_achieved'] is False
         and a['serving_installed'] is False, 'component-only result scope')
    result = dict(status='completed_real_draft_first_catchup_plus_tail3_component',
        math_rows=99, timing_calls=320, median_seconds=summary,
        result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        model_component_qualified=True, full463_qualified=False, Raw=None, Goodput=None, adopted=False)
    # The original v154 supervisor unpacks exactly this three-item ABI.
    return result, 'completed_draft_component_no_serving_qualification', dict(
        all99_original_probability_and_all56KV_cases_recounted=True,
        balanced320_timing_order_recounted=True, no_serving_metrics_claimed=True)
