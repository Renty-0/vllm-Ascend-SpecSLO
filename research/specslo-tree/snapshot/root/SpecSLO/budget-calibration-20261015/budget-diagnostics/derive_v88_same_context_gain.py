"""HOST-only acceptance counterfactuals from retained, actual B5 predictions.

These are two coding owners in one instrumented run, not held-out class
curves or fresh cap1..4 executions. No model/tensor import is required.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path('/root/SpecSLO/budget-calibration-20261015')
RUN = ROOT / 'tree-candidate-hidden-capture-v88/local-output/candidate-hidden-kv48-v2'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def emitted(tokens, parents, predictions):
    assert len(tokens) == len(parents) and len(predictions) == len(tokens) + 1
    current, result = -1, []
    while True:
        prediction = predictions[current + 1]
        matches = [i for i, (token, parent) in enumerate(zip(tokens, parents, strict=True))
                   if parent == current and token == prediction]
        assert len(matches) <= 1
        result.append(prediction)
        if not matches:
            return result
        current = matches[0]


def main():
    paths = [RUN / f'rank{rank}-candidate-actual-evidence.json.gz' for rank in (1, 2, 3)]
    data = [json.loads(gzip.decompress(path.read_bytes())) for path in paths]
    assert all(row['candidate_cache_released'] is True for row in data)
    assert all(row['candidate_capture_count'] == 164 for row in data)
    manifest_path = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/manifest.jsonl')
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    curves, rows, excluded = defaultdict(lambda: defaultdict(list)), [], Counter()
    for calls in zip(*(row['candidate_captures'] for row in data), strict=True):
        leader = calls[0]
        assert all(call['call_number'] == leader['call_number'] and call['context'] == leader['context']
                   for call in calls)
        by_owner = defaultdict(dict)
        for row in leader['rows']:
            key = (row['owner'], row['query_index'])
            for call in calls[1:]:
                other = [v for v in call['rows'] if (v['owner'], v['query_index']) == key]
                assert len(other) == 1 and other[0]['selected_token_id'] == row['selected_token_id']
                assert other[0]['input_prefix_sha256'] == row['input_prefix_sha256']
            assert row['query_index'] not in by_owner[row['owner']]
            by_owner[row['owner']][row['query_index']] = row
        for owner, observed in by_owner.items():
            plans = [plan for plan in leader['context']['plans'] if plan['owner'] == owner and plan['real']]
            assert len(plans) == 1
            plan = plans[0]
            if plan['candidate_budget'] != 5 or set(observed) != set(range(6)):
                excluded['not_complete_B5'] += 1
                continue
            assert plan['parents'][:4] == [-1, 0, 1, 2]
            assert len(plan['parents']) == len(plan['candidate_token_ids']) == 5
            first = observed[0]['output_index']
            if 256 - first < 5:
                excluded['terminal_horizon'] += 1
                continue
            predictions = [observed[q]['selected_token_id'] for q in range(6)]
            actual = emitted(plan['candidate_token_ids'], plan['parents'], predictions)
            result = data[0]['generate_results'][owner]
            assert actual == result['completion_token_ids'][first:first + len(actual)]
            gains = {b: len(emitted(plan['candidate_token_ids'][:b], plan['parents'][:b], predictions[:b+1]))
                     for b in range(1, 5)}
            gains[5] = len(actual)
            assert all(gains[b] <= gains[b+1] for b in range(1, 5))
            category = manifest[owner]['category']
            for b, gain in gains.items():
                curves[category][b].append(gain)
            rows.append(dict(call_number=leader['call_number'], path=leader['context']['path'],
                             owner=owner, output_index=first, prefix_len=plan['prefix_len'],
                             physical_M=leader['context']['physical_M'],
                             physical_S=leader['context']['physical_S'],
                             branch_parent=plan['parents'][4], gains=gains))
    assert rows and set(curves) == {'coding'}
    coding = curves['coding']
    output = dict(schema='ROOT.V88.same-actual-call-prediction-gain-counterfactual.v1',
                  UTC=datetime.now(timezone.utc).isoformat(),
                  source_script_sha256=digest(Path(__file__)),
                  actual_input_sha256={str(path): digest(path) for path in paths},
                  manifest_sha256=digest(manifest_path),
                  scope='two coding requests, same-context truncated backbone using actual B5 target predictions',
                  actual_paired_contexts=len(rows), owners=sorted({row['owner'] for row in rows}),
                  excluded=dict(excluded),
                  empirical_counterfactual_gain={category: {str(b): statistics.mean(values)
                      for b, values in budgets.items()} for category, budgets in curves.items()},
                  B5_minus_B4_mean=statistics.mean(a-b for a, b in zip(coding[5], coding[4], strict=True)),
                  B5_positive_side_gain_contexts=sum(a > b for a, b in zip(coding[5], coding[4], strict=True)),
                  actual_B5_output_slice_checks_passed=len(rows),
                  three_target_rank_prediction_checks_passed=True,
                  independent_owners=len({row['owner'] for row in rows}),
                  contexts_are_correlated_within_owners=True,
                  new_cap1_to4_device_executions=False,
                  counterfactual_uses_same_B5_hidden_and_compute_geometry=True,
                  B1_to4_own_causal_geometry_numerical_equivalence_proved=False,
                  heldout_class_model_validated=False,
                  performance_or_full_workload_gain_proved=False,
                  model_or_NPU_initialized=False, rows=rows)
    destination = ROOT / 'budget-diagnostics/V88_ACTUAL_B5_SAME_CONTEXT_GAIN_COUNTERFACTUAL.json'
    with destination.open('x') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: output[key] for key in ('actual_paired_contexts', 'owners',
          'empirical_counterfactual_gain', 'B5_minus_B4_mean', 'B5_positive_side_gain_contexts',
          'actual_B5_output_slice_checks_passed', 'excluded')}))


if __name__ == '__main__':
    main()
