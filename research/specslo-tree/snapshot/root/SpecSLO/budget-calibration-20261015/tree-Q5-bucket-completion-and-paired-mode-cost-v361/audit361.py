"""All original344 gates, plus independent paired FULL/bucket cost accounting."""
import json
import math
from pathlib import Path
import statistics

import audit344 as original
from benchmark361 import arms
from bucket344 import case_extent


need = original.need


def mode_panel(row, prefixes, rank):
    target = rank != 0
    panel = row['paired_KV_modes361']
    calls = row['actual_KV_mode_calls361']
    cases = list(arms())
    modes = ['full2560', 'uniform_bucket']
    need(panel['rank'] == rank and panel['target_math_on_rank'] is target,
         'new paired-mode actual role')
    need(panel['arms'] == cases and panel['modes'] == modes and panel['rounds'] == 20,
         'all declared new mode arms and20 rounds')
    need(panel['no_new_capture_validation_or_fallback'] is panel['outputs_equal_between_modes'] is True,
         'mode output equality and no extra timed graph work')
    need(panel['complete_serving'] is panel['real_draft_generation'] is False
         and panel['Raw'] is panel['Goodput'] is None, 'target-only paired component')
    keys = ('aclgraph_captures', 'aclgraph_capture_attempts', 'aclgraph_runtime_validation_replays',
            'aclgraph_failed_captures', 'aclgraph_capacity_fallbacks', 'aclgraph_shape_fallbacks')
    need(all(panel['graph_counters_before'][key] == panel['graph_counters_after'][key] for key in keys),
         'new timed mode counters unchanged')
    expected_calls = [(arm, mode) for arm in cases for mode in modes]
    samples = {arm['name']: {mode: [] for mode in modes} for arm in cases}
    need(len(panel['pairs']) == 20, 'all20 paired mode rotations retained')
    for iteration, pair in enumerate(panel['pairs']):
        order = cases[iteration % len(cases):] + cases[:iteration % len(cases)]
        need(pair['iteration'] == iteration and pair['arm_order'] == [a['name'] for a in order]
             and len(pair['samples']) == 5, 'balanced new arm rotation')
        for index, (arm, sample) in enumerate(zip(order, pair['samples'], strict=True)):
            mode_order = modes if (iteration + index) % 2 == 0 else modes[::-1]
            need(sample['arm'] == arm['name'] and sample['mode_order'] == mode_order
                 and set(sample['milliseconds']) == set(modes), 'balanced mode order')
            expected_calls.extend((arm, mode) for mode in mode_order)
            for mode, elapsed in sample['milliseconds'].items():
                if target:
                    need(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > 0,
                         'positive actual new paired elapsed')
                    samples[arm['name']][mode].append(elapsed)
                else:
                    need(elapsed is None, 'draft has no new target timing')
    if not target:
        need(calls == [] and row['all72_prefixes_after_KV_modes361'] is None,
             'draft has no new target calls or prefix claim')
        return []
    need(row['all72_prefixes_after_KV_modes361'] is True, 'all72 prefixes after new mode costs')
    need(len(calls) == len(expected_calls) == 210, 'all10 warm and200 timed Native mode calls')
    for call, (arm, mode) in zip(calls, expected_calls, strict=True):
        n = arm['N']
        branch = arm['budget'] > 4
        extent = 2560 if mode == 'full2560' else case_extent(dict(budgets=arm['budgets']), prefixes)
        copied = sum(len(prefixes[i]) % 128 + 5 for i in range(n)) if branch else 0
        grid = next(v for v in (128, 256, 512, 1024, 1792) if v >= copied) if copied else 0
        need(call['N'] == n and call['real_queries'] == n + sum(arm['budgets']) and call['M'] == 120
             and call['attention_segments'] == 24 and call['actual_seq_lengths_q'] == list(range(5, 121, 5))
             and call['sequence_lens'] == [extent] * 24
             and call['mask_shape'] == [24, 1, 11 if branch else 10, 4096]
             and call['copy_enabled'] is branch and call['copy_grid'] == grid
             and call['primary_side_groups'] == n * (2 if branch else 1)
             and call['used_aclgraph'] is True and call['execution_mode'] == 'replay'
             and call['fallback_reason'] is None, 'actual new mode M120/copy/replay geometry')
        detail = call['uniform_KV_bucket344']
        need(detail['enabled'] is (mode == 'uniform_bucket') and detail['extent'] == extent
             and 0 < detail['original_max_context'] <= extent, 'actual mode and full untruncated context')
        key = call['actual_capture_key']
        need((f'|ROOT296:copy-grid:{grid}' in key) if branch else ('|ROOT296:copy-grid:' not in key),
             'original mode copy-family key')
    return [dict(rank=rank, bank=row['name'], arm=arm['name'], N=arm['N'], B=arm['budget'],
                 bucket_extent=case_extent(dict(budgets=arm['budgets']), prefixes),
                 full2560_median_ms=statistics.median(samples[arm['name']]['full2560']),
                 bucket_median_ms=statistics.median(samples[arm['name']]['uniform_bucket']),
                 full_over_bucket=statistics.median(samples[arm['name']]['full2560'])
                 / statistics.median(samples[arm['name']]['uniform_bucket'])) for arm in cases]


def audit(directory):
    base, reason, evidence = original.audit(directory)
    directory = Path(directory)
    summaries = []
    for rank in range(4):
        report = json.loads((directory / f'rank{rank}-numeric.json').read_text())
        for row in report['banks']:
            prefixes = [[0] * length for length in row['prefix_lengths']]
            summaries.extend(mode_panel(row, prefixes, rank))
            saved = json.loads((directory / row['name'] / f'rank{rank}-cost-and-protection361.json').read_text())
            need(saved['rank'] == rank and saved['bank'] == row['name']
                 and saved['original_budget_cost'] == row['paired_target_cost']
                 and saved['paired_KV_modes361'] == row['paired_KV_modes361']
                 and saved['actual_KV_mode_calls361'] == row['actual_KV_mode_calls361']
                 and saved['all72_prefixes_after_KV_modes361'] == row['all72_prefixes_after_KV_modes361'],
                 'each completed cost panel durably retained before next bank')
    value = dict(base, original344_audit_status=base['status'],
                 status='PASS_current_M120_bucket_branch594_and_same_instance_KV_mode_costs361',
                 paired_KV_mode_costs361=summaries,
                 TP3_ranks_are_not_independent_experiments=True)
    return value, reason + '_and_paired_KV_mode_costs361', dict(evidence, paired_mode_calls_recomputed=True)
