"""Independent inventory and timing-contract audit of real draft-only results."""
import json
import math
from pathlib import Path
import statistics

from probe365 import records, TIMED_PAIRS, digest, HERE


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT365 audit: ' + message)


def audit(directory):
    p = Path(directory) / 'result.json'
    a = json.loads(p.read_text())
    need(a['status'] == 'complete' and a['source_sha256'] == digest(HERE / 'probe365.py'), 'own complete source')
    need(a['models_loaded'] == ['Qwen3-0.6B'] and a['target_model_loaded'] is False,
         '0.6B-only model scope')
    need(a['dtype'] == 'BF16' and a['quantized'] is False and a['TP'] == 1, 'real BF16 TP1')
    need(a['independent_prefills'] == 96 and a['layers'] == 28, 'all three independent banks and28 layers')
    expected = {(n, origin, phase) for n, origin in records() for phase in ('A1', 'B', 'A2')}
    actual = a['math_rows']
    need(len(actual) == len(expected) and {(r['N'], r['origin'], r['phase']) for r in actual} == expected,
         'all original shape/origin/ABA records exactly once')
    for row in actual:
        need([r['final_kv_only'] for r in row['candidates']] == [False, True], 'both explicit chain modes')
        for r in row['candidates']:
            steps = 5 if r['final_kv_only'] else 4
            need(r['all8_tokens_exact'] is True and r['all8_FP32_probabilities_bitwise'] is True
                 and r['all_KV_tensors'] == 56, 'all token/probability/KV exact checks')
            need(r['physical_queries_per_step'] == 5 * (1 << row['N'].bit_length())
                 and r['physical_model_queries'] == steps * r['physical_queries_per_step']
                 and r['real_model_queries'] == steps * row['N'], 'all actual queries charged including fifth step')
            need(r['original_execution']['mode'] in ('capture_replay', 'replay'), 'actual original Native graph')
            need('|ROOT364:raw-spine4-top2-float32-bits' in r['entry_key'][0]
                 and r['entry_key'][0].endswith('|logical:' + str(row['N'])), 'private logical graph key')
    costs = a['costs']
    need(len(costs) == 4 * TIMED_PAIRS * 3, 'all balanced repeated timing cells')
    expected_costs = {(n, pair, mode) for n in (4, 12, 24, 32) for pair in range(TIMED_PAIRS)
                      for mode in ('generic4', 'chain4', 'chain5')}
    need({(r['N'], r['pair'], r['mode']) for r in costs} == expected_costs, 'no omitted or fastest-selected timings')
    for r in costs:
        need(type(r['seconds']) is float and math.isfinite(r['seconds']) and r['seconds'] > 0, 'finite real duration')
    summary = {f'N{n}': {mode: statistics.median([r['seconds'] for r in costs if r['N'] == n and r['mode'] == mode])
                        for mode in ('generic4', 'chain4', 'chain5')} for n in (4, 12, 24, 32)}
    need(a['median_seconds'] == summary, 'recomputed medians')
    need(a['all_committed_prefix_KV_protected'] is True and a['native_qualification_bodies_unchanged'] is True,
         'all committed prefix protection and unchanged checks')
    need(a['graph_stats']['unvalidated_entries'] == 0 and a['graph_stats']['disabled_entries'] == 0,
         'no unqualified or disabled resident graphs')
    need(a['Raw'] is None and a['Goodput'] is None and a['adopted'] is False and a['goal_achieved'] is False,
         'component-only performance scope')
    return dict(status='complete_real_draft_component_audited', result_sha256=digest(p),
                math_rows=len(actual), timing_calls=len(costs), median_seconds=summary,
                Raw=None, Goodput=None, adopted=False)
