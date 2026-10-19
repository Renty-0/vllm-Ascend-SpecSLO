"""Source-bound finite synthetic diagnostic, never a model qualification."""
import hashlib
import json
import math
from pathlib import Path

from probe363 import configurations


P = Path(__file__).resolve().parent


def need(value, message):
    if not value:
        raise RuntimeError('ROOT363 audit: ' + message)


def audit(directory):
    directory = Path(directory)
    path = directory / 'result.json'
    value = json.loads(path.read_text())
    need(value['schema'] == 'ROOT363.synthetic-FIA-multi-owner-page-offset.v1' and value['seed'] == 363,
         'actual synthetic schema and seed')
    need(value['source_sha256'] == hashlib.sha256((P / 'probe363.py').read_bytes()).hexdigest(),
         'actual source-bound diagnostic')
    need(value['models_loaded'] is value['model_qualified'] is value['performance'] is value['adopted'] is False
         and value['Raw'] is value['Goodput'] is None, 'no model or serving result')
    need(value['logical_device'] == 1 and value['physical_cards_requested'] == [2, 5, 6, 7]
         and len(value['visible_runtime_ids'].split(',')) == len(set(value['visible_runtime_ids'].split(','))) == 4,
         'explicit original four-card map and owned logical target device')
    cases = json.loads(json.dumps(configurations()))
    rows = value['rows']
    need(value['complete_cases'] == len(cases) == 640 and len(rows) == 1920,
         'all640 cases with complete ABA phases')
    persisted = [json.loads(line) for line in (directory / 'completed-cases363.jsonl').read_text().splitlines()]
    need(persisted == rows, 'every final row matches its durable completed-case checkpoint')
    summaries = {}
    for ordinal, row in enumerate(rows):
        case = cases[ordinal // 3]
        phase = ('A1', 'B', 'A2')[ordinal % 3]
        need(row['case_ordinal'] == ordinal // 3 and row['case'] == case and row['phase'] == phase,
             'exact declared case/phase order')
        compared = 24 * sum(case['sizes'])
        need(row['compared_queries'] == compared and row['physical_M'] == 120
             and row['groups'] == 24 and row['width'] == 5
             and row['identical_ordered_real_KV_and_original_query'] is True,
             'original physical Q5/S24 and complete real query subset')
        indices = row['different_query_indices']
        differences = row['max_abs_by_query']
        need(indices == sorted(set(indices)) and all(type(i) is int and 0 <= i < compared for i in indices),
             'valid raw16 disagreement indices')
        need(len(differences) == compared and all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                                                for v in differences), 'finite actual numeric differences')
        need(all(value == 0 for i, value in enumerate(differences) if i not in indices)
             and row['all_compared_raw16_equal'] is (not indices), 'actual bit-equality/difference arithmetic')
        offset = row['same_extent_page_offset_raw16_equal']
        extent = row['unshifted_extent_vs_FULL2560_raw16_equal']
        need(type(offset) is bool and type(extent) is bool,
             'two independent unshifted-reference comparison flags')
        need(not (offset and extent) or row['all_compared_raw16_equal'], 'transitive raw16 equality')
        need(sum((offset, extent, row['all_compared_raw16_equal'])) != 2,
             'all three pairwise equality flags form a consistent equivalence relation')
        need(row['whole_output_restored'] is (True if phase == 'A2' else None), 'complete packed-output ABA restoration')
        key = f"KV{case['KV_heads']}-extent{case['extent']}-sizes" + '-'.join(map(str, case['sizes']))
        summary = summaries.setdefault(key, dict(rows=0, noninvariant=0, offset_noninvariant=0,
                                                  unshifted_extent_noninvariant=0))
        summary['rows'] += 1
        summary['noninvariant'] += int(not row['all_compared_raw16_equal'])
        summary['offset_noninvariant'] += int(not offset)
        summary['unshifted_extent_noninvariant'] += int(not extent)
    result = dict(status='completed_synthetic_FIA_multi_owner_page_offset_diagnostic',
                  cases=640, rows=1920, summaries=summaries,
                  all_page_offsets_raw16_invariant=all(row['same_extent_page_offset_raw16_equal'] for row in rows),
                  all_match_original_FULL2560=all(row['all_compared_raw16_equal'] for row in rows),
                  all_unshifted_extents_match_FULL2560=all(row['unshifted_extent_vs_FULL2560_raw16_equal'] for row in rows),
                  result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  models_loaded=False, model_qualified=False, performance=False,
                  full463_qualified=False, Raw=None, Goodput=None, adopted=False)
    return result, 'completed_synthetic_no_model_qualification', dict(all640_cases_and_ABA_recounted=True,
        full2560_and_same_extent_controls_recounted=True, Native_counters_claimed=False)
