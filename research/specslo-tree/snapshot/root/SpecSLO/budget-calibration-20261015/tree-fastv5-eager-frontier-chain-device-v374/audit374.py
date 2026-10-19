"""Recount actual optimized-baseline frontier fusion, no serving metric claim."""
import hashlib
import json
import math
from pathlib import Path
import statistics
from probe374 import HERE, configurations


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT374 audit: ' + message)


def physical_bucket(n):
    return next(v for v in (1, 2, 4, 8, 12, 16, 20, 24, 28, 32) if v >= n)


def audit(directory):
    path = Path(directory) / 'result.json'
    a = json.loads(path.read_text())
    need(a['status'] == 'complete' and a['source_sha256'] == hashlib.sha256((HERE / 'probe374.py').read_bytes()).hexdigest(), 'own complete source')
    need(a['models_loaded'] == ['Qwen3-0.6B'] and a['target_model_loaded'] is False and a['TP'] == 1
         and a['dtype'] == 'BF16' and a['quantized'] is False and a['model_weights_or_precision_changed'] is False,
         'original BF16 draft model')
    need(a['comparison_baseline'] == 'actual installed RootSeed over FastV5, not bare Native'
         and a['original_confidence_encoding_retained'] is True, 'actual optimized comparison')
    need(a['independent_prefills'] == 96 and a['layers'] == 28, 'independent neural banks/all28 layers')
    expected = {(n, origin, budget, phase) for n, origin, budget in configurations() for phase in ('A1', 'B', 'A2')}
    rows = a['math_rows']
    need(len(rows) == len(expected) == 510
         and {(r['N'], r['origin'], r['budget'], r['phase']) for r in rows} == expected, 'all510 predeclared cases')
    for r in rows:
        steps = min(r['budget'], 4)
        physical = physical_bucket(r['N'])
        need(r['chain_steps'] == steps and r['original_graph_submissions'] == 2 and r['fused_graph_submissions'] == 1,
             'same original stages with one fewer graph submission')
        need(all(r[k] is True for k in ('all_tokens_and_roots_exact', 'all_original_confidence_bits_exact',
             'canonical_ledger_equal', 'physical_queries_equal')) and r['all_KV_tensors'] == 56,
             'all token/root/confidence/ledger/all56 KV contracts exact')
        need(r['A2_restored'] is (True if r['phase'] == 'A2' else None), 'full ABA restoration')
        v = r['actual']
        need(v['logical'] == r['N'] and v['physical'] == physical and v['chain_steps'] == steps
             and v['model_steps'] == steps + 1 and v['graph_submissions'] == 1
             and v['actual_physical_queries'] == (steps + 1) * physical
             and v['frontier_CPU_readback_eliminated'] is True
             and v['original_weight_precision_unchanged'] is True, 'all actual work charged')
    order = [(n, budget, pair, name) for n in (4, 12, 24, 32) for budget in (4, 5) for pair in range(20)
             for name in (('optimized_existing', 'fused_frontier_chain') if pair % 2 == 0 else ('fused_frontier_chain', 'optimized_existing'))]
    costs = a['costs']
    need(len(costs) == 320 and [(r['N'], r['budget'], r['pair'], r['operation']) for r in costs] == order,
         'all320 balanced optimized-baseline paired timings')
    need(all(type(r['seconds']) is float and math.isfinite(r['seconds']) and r['seconds'] > 0 for r in costs), 'positive real durations')
    summary = {f'N{n}_B{budget}': {name: statistics.median([r['seconds'] for r in costs
                if r['N'] == n and r['budget'] == budget and r['operation'] == name])
                for name in ('optimized_existing', 'fused_frontier_chain')}
                for n in (4, 12, 24, 32) for budget in (4, 5)}
    need(summary == a['median_seconds'], 'recomputed medians')
    need(a['all_committed_prefix_KV_protected'] is True and a['native_qualification_bodies_unchanged'] is True
         and a['graph_stats']['unvalidated_entries'] == a['graph_stats']['disabled_entries'] == 0,
         'all96 committed prefixes and unmodified natural graph qualification')
    need(a['serving_installed'] is False and a['Raw'] is None and a['Goodput'] is None and a['adopted'] is False,
         'component-only evidence')
    return dict(status='completed_actual_FastV5_frontier_fusion_component', math_rows=510, timing_calls=320,
        median_seconds=summary, result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        full463_qualified=False, Raw=None, Goodput=None, adopted=False), 'completed_optimized_draft_component', dict(
        actual_optimized_baseline_used=True, all510_root_confidence_ledger_KV_checks_recounted=True,
        all320_balanced_pairs_recounted=True)
