"""Bound incumbent refill charges from retained serial callback gaps; no models."""
from pathlib import Path
import gzip
import hashlib
import json
import statistics

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
INPUTS = {
    173: B / 'tree-causal80-real16-bankfix-full-v173/actual/causal80-bankfix-v173-full463-v1-overlay177/windows/window32/full463/rank1-full463-actual-trace.json.gz',
    171: B / 'tree-normal128-eager64-global192-full-v171/actual/normal128-v171-full463-v1-overlay177/windows/window64/full463/rank1-full463-actual-trace.json.gz',
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def derive(path):
    data = json.loads(gzip.decompress(path.read_bytes()))
    events = data['stream_events']
    first = {}
    for j, event in enumerate(events):
        first.setdefault(event['request_index'], j)
    active = set()
    groups = []
    cursor = 0
    previous = 0.0
    for call in data['standalone_prefill_calls']:
        ids = call['sequence_ids']
        j = first[ids[0]]
        assert j >= cursor and ids
        while cursor < j:
            event = events[cursor]
            if event['finished']:
                active.discard(event['request_index'])
            previous = event['elapsed_seconds']
            cursor += 1
        assert [events[j+k]['request_index'] for k in range(len(ids))] == ids
        assert all(first[index] == j+k for k, index in enumerate(ids))
        upper = events[j+len(ids)-1]['elapsed_seconds'] - previous
        assert upper >= 0.0
        groups.append((upper, frozenset(active)))
        for k, index in enumerate(ids):
            assert not events[j+k]['finished']
            active.add(index)
        cursor = j+len(ids)
        previous = events[cursor-1]['elapsed_seconds']
    total = data['decode_counters']['spec_rhythm_online_refill_ms'] / 1000.0
    total_upper = sum(upper for upper, _ in groups)
    assert 0.0 <= total <= total_upper
    assert len(groups) == data['decode_counters']['spec_rhythm_prefill_batches']
    rows = []
    for index, row in enumerate(data['generate_results']):
        assert len(row['completion_token_ids']) == 256
        inside = sum(upper for upper, owners in groups if index in owners)
        outside = total_upper-inside
        lower = max(0.0, total-outside)
        upper = min(total, inside)
        assert 0.0 <= lower <= upper
        rows.append(dict(request_index=index, request_id=row['request_id'],
            slo_class=row['slo_class'], Native_charged_TPOT_ms=row['paper_tpot_ms'],
            incumbent_refill_charge_lower_seconds=lower,
            incumbent_refill_charge_upper_seconds=upper,
            charged_TPOT_contribution_lower_ms=lower*1000/256,
            charged_TPOT_contribution_upper_ms=upper*1000/256))
    classes = {}
    for label in ('tight', 'normal', 'loose'):
        selected = [r for r in rows if r['slo_class'] == label]
        classes[label] = dict(requests=len(selected),
            Native_charged_TPOT_median_ms=statistics.median(r['Native_charged_TPOT_ms'] for r in selected),
            refill_contribution_lower_bound_median_ms=statistics.median(r['charged_TPOT_contribution_lower_ms'] for r in selected),
            refill_contribution_upper_bound_median_ms=statistics.median(r['charged_TPOT_contribution_upper_ms'] for r in selected))
    assert len(rows) == 463
    return dict(input=dict(path=str(path), sha256=sha(path)), prefill_groups=len(groups),
        total_original_leader_refill_HOST_seconds=total,
        sum_previous_publication_to_group_last_root_upper_seconds=total_upper,
        unassigned_gap_slack_seconds=total_upper-total, classes=classes, per_request_bounds=rows)

source = B / 'tree-high-normal-cap4-native-families-v148/runtime_tree_decode.py'
output = D / 'COMPLETED_REFILL_CHARGE_CALLBACK_GAP_BOUNDS_ROOT.json'
record = dict(schema='ROOT.completed-refill-charge.callback-gap-bounds.v1',
    status='completed_trace_conditional_timing_bounds',
    script=dict(path=str(Path(__file__).resolve()), sha256=sha(Path(__file__).resolve())),
    source=dict(path=str(source), sha256=sha(source)),
    assumptions=[
        'Original admit_available invokes the synchronous wholeprompt helper after previous callbacks, measures its elapsed HOST interval, then publishes this group roots after the helper returns.',
        'Consequently each nonnegative helper duration d_j is at most U_j, the gap from the previous callback to the group last root callback, and sum(d_j) is the retained original leader refill HOST total.',
        'The original active membership reconstruction is shared across the whole refill group; new owners do not receive the incumbent charge.',
        'For owner i, min possible charge=max(0,total_refill-sum(U_j outside i)); max possible charge=min(total_refill,sum(U_j inside i)).',
        'Convert charges using the historical Native charged denominator256; no client or common-publication denominator is substituted.'
    ], limitations=[
        'These are constraints on completed-run HOST charge attribution, not measured individual prefill durations or NPU kernel times.',
        'Subtracting a bound from observed TPOT is not a prediction for a different admission policy: batching, tail sizes, padding, total wall and TTFT may change.',
        'Class medians describe distributions of per-request bounds; the bounds need not be attained simultaneously by one duration assignment.',
        'No new workload, Goodput, production qualification, numerical-losslessness result or performance adoption is created.'
    ], completed={str(epoch):derive(path) for epoch, path in INPUTS.items()},
    Torch_imported=False, NPU_used=False, new_model_run=False,
    formal_Goodput=False, adopted=False)
with output.open('x') as stream:
    json.dump(record, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps(dict(path=str(output), sha256=sha(output),
    completed={epoch:row['classes'] for epoch,row in record['completed'].items()})))
