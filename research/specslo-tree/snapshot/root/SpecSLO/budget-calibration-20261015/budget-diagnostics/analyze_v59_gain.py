"""Describe actual current-pool gains; epochs are clustered by request.

This does not fit a production policy, estimate Goodput, or grant numerical
qualification. The independent runtime audit supplies those scope checks.
"""
import collections
import datetime
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
W = ROOT / 'tree-terminal-pool-context-gain-v59'
RAW = W / 'context-cost-sweep-v1/terminal-pool-context-gain-v1/rank0-context-gain.json'
OUT = ROOT / 'budget-diagnostics/V59_ACTUAL_G_CLASS_SPLIT_OFFSET_ANALYSIS.json'


def mean(values):
    return statistics.mean(values)


def main():
    report = json.loads(RAW.read_text())
    receipt_path = W/'cost-receipt-terminal-pool-context-gain-v1.json'
    receipt = json.loads(receipt_path.read_text())
    assert len(report['qualification']) == 30 and not report['errors']
    assert report['pure_numerical_context_gain']
    assert not report['fixed_point'] and not report['cost'] and not report['sealed']
    labels = []
    for scope in report['qualification']:
        meta = scope['cohort']
        assert len(meta['contexts']) == len(scope['g']) == 12
        for context, g, packed_path, selection, emitted in zip(
                meta['contexts'], scope['g'], scope['accepted'],
                scope['selected_original_nodes'], scope['emitted']):
            path = [selection[n] for n in packed_path if n >= 0]
            tokens = [n for n in emitted if n >= 0]
            assert g == len(path) + 1 and 1 <= g <= 5
            assert g == len(tokens)
            side = any(n >= 4 for n in path)
            assert sum(n >= 4 for n in path) <= 1
            labels.append(dict(**context, B=scope['B'], g=g,
                selected_original_nodes=selection, accepted_original_nodes=path,
                emitted=tokens,
                terminal_side_accepted=side,
                same_M_primary_only_counterfactual_g=g-int(side)))
    assert len(labels) == 360
    assert len({(x['request_id'], x['output_offset'], x['B']) for x in labels}) == 360
    old_pilot = json.loads((ROOT/'tree-terminal-pool-gain-v58/inputs-v58.json').read_text())
    previously_seen = {x['request_id'] for x in old_pilot['contexts']}
    overlap = {}
    for cls in ('tight', 'normal', 'loose'):
        overlap[cls] = {split: len({x['request_id'] for x in labels
            if x['slo_class'] == cls and x['split'] == split
            and x['request_id'] in previously_seen}) for split in ('train', 'heldout')}
    overall = {}
    per_class = {}
    split_rows = {}
    offset_rows = {}
    heldout_error = {}
    for b in range(4, 9):
        rows = [x for x in labels if x['B'] == b]
        overall[str(b)] = dict(contexts=len(rows), independent_requests=24,
            g_sum=sum(x['g'] for x in rows), mean_g=mean(x['g'] for x in rows),
            side_accepted_contexts=sum(x['terminal_side_accepted'] for x in rows))
        per_class[str(b)] = {}
        split_rows[str(b)] = {}
        offset_rows[str(b)] = {}
        heldout_error[str(b)] = {}
        for cls in ('tight', 'normal', 'loose'):
            selected = [x for x in rows if x['slo_class'] == cls]
            per_class[str(b)][cls] = dict(mean_g=mean(x['g'] for x in selected),
                context_epochs=24, independent_requests=8,
                g_range=[min(x['g'] for x in selected), max(x['g'] for x in selected)])
            training = [x for x in selected if x['split'] == 'train']
            heldout = [x for x in selected if x['split'] == 'heldout']
            assert len(training) == len(heldout) == 12
            prediction = mean(x['g'] for x in training)
            split_rows[str(b)][cls] = dict(train_mean=prediction,
                heldout_mean=mean(x['g'] for x in heldout), requests_per_split=4,
                correlated_epochs_per_split=12)
            grouped = collections.defaultdict(list)
            for x in heldout:
                grouped[x['request_id']].append(x['g'])
            assert len(grouped) == 4 and all(len(v) == 3 for v in grouped.values())
            heldout_error[str(b)][cls] = dict(train_class_mean_prediction=prediction,
                heldout_context_MAE=mean(abs(x['g']-prediction) for x in heldout),
                heldout_context_RMSE=mean((x['g']-prediction)**2 for x in heldout)**.5,
                heldout_request_mean_MAE=mean(abs(mean(v)-prediction) for v in grouped.values()),
                error_is_descriptive_not_confidence_interval=True)
        for offset in (32, 96, 192):
            selected = [x for x in rows if x['output_offset'] == offset]
            offset_rows[str(b)][str(offset)] = dict(contexts=24,
                mean_g=mean(x['g'] for x in selected),
                side_accepted_contexts=sum(x['terminal_side_accepted'] for x in selected))
    baseline = {(x['request_id'], x['output_offset']): x['g'] for x in labels if x['B'] == 4}
    paired = {(x['request_id'], x['output_offset'], x['B']): x for x in labels}
    common_prefix_conflicts = []
    for x in labels:
        if x['B'] == 4:
            continue
        y = paired[x['request_id'], x['output_offset'], 4]
        length = min(len(x['emitted']), len(y['emitted']))
        if x['emitted'][:length] != y['emitted'][:length]:
            common_prefix_conflicts.append(dict(request_id=x['request_id'],
                output_offset=x['output_offset'], B=x['B'],
                actual_B4_tokens=y['emitted'], other_layout_tokens=x['emitted']))
    b7_b8_conflicts = []
    for x in labels:
        if x['B'] != 8:
            continue
        y = paired[x['request_id'], x['output_offset'], 7]
        length = min(len(x['emitted']), len(y['emitted']))
        if x['emitted'][:length] != y['emitted'][:length]:
            b7_b8_conflicts.append(dict(request_id=x['request_id'],
                output_offset=x['output_offset'], actual_B7_tokens=y['emitted'],
                actual_B8_tokens=x['emitted'], B7_g=y['g'], B8_g=x['g']))
    request_changes = []
    for b in range(5, 9):
        clustered = collections.defaultdict(list)
        for x in labels:
            if x['B'] != b:
                continue
            clustered[x['request_id']].append(x['g']-baseline[x['request_id'], x['output_offset']])
        assert len(clustered) == 24 and all(len(v) == 3 for v in clustered.values())
        request_changes.append(dict(B=b,
            request_mean_deltas={k: mean(v) for k, v in sorted(clustered.items())},
            positive_request_mean_deltas=sum(mean(v)>0 for v in clustered.values()),
            negative_request_mean_deltas=sum(mean(v)<0 for v in clustered.values())))
    out = dict(status='ACTUAL_72_CONTEXT_DESCRIPTIVE_GAIN_NO_PERFORMANCE_CLAIM',
        created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source=str(RAW), source_sha256=hashlib.sha256(RAW.read_bytes()).hexdigest(),
        supervisor_receipt=str(receipt_path), supervisor_status=receipt['status'],
        numerical_records_complete=True, whole_experiment_completed=False,
        final_reply_failure="After all30 numerical scopes: calibrate_base.worker expects report['gain']; V59 omitted it. Original failed receipt preserved.",
        independent_request_clusters=24, correlated_context_epochs=72,
        original_request_split_preserved=True, B_specific_labels=360,
        previously_observed_V58_request_overlap_by_class_split=overlap,
        overall=overall, per_class=per_class, split_means=split_rows,
        by_offset=offset_rows, training_mean_heldout_error=heldout_error,
        request_mean_deltas_vs_actual_B4=request_changes, labels=labels,
        cross_layout_common_prefix_conflicts_vsB4=common_prefix_conflicts,
        cross_layout_common_prefix_conflicts_B7_vsB8=b7_b8_conflicts,
        counterfactual_scope='g minus a terminal side accepted is primary-only under THAT same-M target prediction; it is not measured B4 execution.',
        limits=['V58 pilot requests overlap these24; this is sample expansion, not a fresh holdout.',
            'The old heldout split was partly observed inV58; descriptive errors are not a pristine unseen-test result.',
            'Three epochs of one request are correlated; no72-independent-sample inference.',
            'ActualB4/5/6/7/8 use differentphysicalM/packing/family; crossM target token equality unqualified.',
            'Actual cross-layout common-prefix token conflicts exist; neither layout is proved equal to a denseAR reference.',
            'Class means are a descriptive pilot, not a validated per-request g(i,n,t) guarantee.',
            'No newcost/zero/seal/online or Goodput qualification fromV59.'],
        whole_workload_AR_equality_qualified=False, Goodput=None)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(output=str(OUT), overall=overall, per_class=per_class), ensure_ascii=False))


if __name__ == '__main__':
    main()
