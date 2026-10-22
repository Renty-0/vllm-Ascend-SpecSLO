"""Read complete retained window traces; never imports a backend or edits a run."""
import collections
import ast
import copy
import gzip
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

B = Path(__file__).resolve().parent.parent
P = B / 'tree-active-window-full-v166'
sys.path.insert(0, str(P))
from base166 import ACTIVE, checked, functions


def metrics(info):
    # Offline arithmetic uses an explicit complete namespace. The frozen
    # trace166.metrics lacked sha, after all raw inference traces were saved.
    namespace = dict(P=P, Path=Path, gzip=gzip, json=json, ast=ast, copy=copy,
                     math=math, statistics=statistics, checked=checked,
                     sha=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                     require=lambda ok, message: None if ok is True else _reject(message))
    functions(ACTIVE/'metrics160.py', namespace, names=('_percentiles', 'full_metrics'))
    return namespace['full_metrics'](info)


def _reject(message):
    raise ValueError(message)


def analyze(tag):
    actual = P / 'actual' / tag
    results = []
    for cap in (32, 24, 16):
        path = actual / 'windows' / f'window{cap}' / 'full463' / 'rank1-full463-actual-trace.json.gz'
        if not path.exists():
            continue
        raw = path.read_bytes()
        trace = json.loads(gzip.decompress(raw))
        info = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())
        value = metrics(info)
        assert trace['scope_count'] == len(trace['generate_results']) == 463
        assert trace['sealed_before'] is trace['sealed_after'] is True
        assert len(trace['native']['delta']) == 21
        assert all(v == trace['native']['after'][k] - trace['native']['before'][k] == 0
                   for k, v in trace['native']['delta'].items())
        calls = trace['normal_target_calls']
        budgets = collections.Counter(b for row in calls for b in row['budgets'])
        n_counts = collections.Counter(row['N'] for row in calls)
        families = collections.Counter(row['family'] for row in calls)
        geometries = collections.Counter((row['N'], row['real_query_count'], row['family'],
                                         row['physical_M'], row['physical_S']) for row in calls)
        assert sum(budgets.values()) == sum(row['N'] for row in calls)
        first_counts = collections.Counter(e['request_index'] for e in trace['stream_events']
                                           if 'first_token_wall_seconds160' in e)
        assert first_counts == collections.Counter(range(463))
        results.append(dict(window=cap, artifact=info, metrics=value,
            normal_target_calls=len(calls), normal_budget_owner_rounds=dict(sorted(budgets.items())),
            normal_batch_N_counts=dict(sorted(n_counts.items())), attention_family_counts=dict(families),
            actual_geometries=[dict(N=k[0], real_Q=k[1], family=k[2], M=k[3], S=k[4], calls=n)
                               for k, n in sorted(geometries.items())],
            original_decode_counters=trace['decode_counters'],
            original_phase_seconds=trace['phase_seconds'],
            final_policy_snapshot=trace['policy_after'],
            original_arrivals_unchanged=True, original_outer_generate_denominator=True,
            source_generation=trace['source_generation'],
            all21_sealed_measurement_counter_subtractions_zero=True))
    receipt_path = P / f'receipt-{tag}.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    return dict(schema='ROOT.V166.retained-window-full-trace-analysis.v1', candidate=str(P),
                tag=tag, completed_full_windows=results,
                all4_worker_audit_written=(actual / 'completed.json').exists(),
                supervisor_terminal_complete=receipt.get('status') == 'complete',
                phase_wrappers_are_not_device_only_costs=True,
                per_family_device_costs_unavailable=True,
                formal_Goodput=False, adopted=False)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: root_analyze_v166_windows.py <tag>')
    print(json.dumps(analyze(sys.argv[1]), ensure_ascii=False, allow_nan=False, indent=2))
