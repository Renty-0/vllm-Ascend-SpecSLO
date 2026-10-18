"""Original HOST clock formulas, plus explicitly measured first-token walls."""
import ast
import copy
import gzip
import json
import math
import statistics
from pathlib import Path
from window160 import BASE, P, checked, require, sha


def _percentiles(values):
    require(bool(values), 'nonempty actual timing samples')
    values = sorted(values)
    at = (len(values)-1)*.9
    lo, hi = math.floor(at), math.ceil(at)
    return dict(median_ms=statistics.median(values),
                P90_ms=values[lo]+(values[hi]-values[lo])*(at-lo))


def full_metrics(info):
    path = Path(info['path']).resolve()
    require(path.is_relative_to(P) and sha(path) == info['sha256'], 'own completed trace artifact SHA')
    trace = json.loads(gzip.decompress(path.read_bytes()))
    if trace['generate_results'] is None:
        return None
    source = checked(P.parent / 'budget-diagnostics/analyze_completed_tree_full463_cost.py')
    tree = ast.parse(source.read_text())
    nodes = [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ('need', 'clocks')]
    require(len(nodes) == 2, 'two original HOST clock functions')
    ns = dict(math=math, statistics=statistics, CLASSES=('tight', 'normal', 'loose'))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(source) + '::V160-unchanged-clocks', 'exec'), ns)
    result = ns['clocks'](trace)  # Unchanged Native/stream TPOT and Goodput.
    first = {}
    for event in trace['stream_events']:
        if 'first_token_wall_seconds160' in event:
            owner = event['request_index']
            require(owner not in first, 'one actual first-token wall per request')
            first[owner] = event['first_token_wall_seconds160']
    rows = trace['generate_results']
    require(set(first) == set(range(463)), 'all463 actual first-token callback wall samples')
    ttft = [(first[i]-row['arrival_ts'])*1000 for i, row in enumerate(rows)]
    require(all(math.isfinite(v) and v >= 0 for v in ttft), 'actual nonnegative arrival-to-first-token wall')
    result['TTFT160'] = dict(_percentiles(ttft),
        clock='first original token callback time.time minus unchanged rebased arrival_ts',
        extra_wall_samples=463, original_TPOT_clock_unchanged=True,
        by_class={name: _percentiles([ttft[i] for i, row in enumerate(rows) if row['slo_class'] == name])
                  for name in ('tight', 'normal', 'loose')})
    result['admission_window160'] = trace['admission_window160']
    result['actual_peak'] = trace['active_peak']
    result['actual_window_covered160'] = trace['actual_window_covered160']
    result['arrival_offsets_unchanged'] = True
    return result
