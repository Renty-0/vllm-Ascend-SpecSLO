"""Grouped exploratory counterfactual accounting; never a measured new g/GP."""
import datetime
import hashlib
import json
from pathlib import Path

from target_aligned_selector_prototype_ROOT import LAYOUTS, SPINE, choose, make_observations

D = Path(__file__).resolve().parent
SOURCE = D / 'V206_COMPLETED_BOUNDED_GAIN_COST_AND_OLD_HEAD_OPPORTUNITY_HOST_ANALYSIS.json'
COUNTS = D / 'V206_CONDITIONAL_OLD_HEAD_SELECTOR_LIMITS_AND_DESIGN_JUDGMENT.json'
PINS = {
    SOURCE: '4ab768297ecc5d18bce630d3bd93e2d7e36953ea8e22dd02af143bb7ee253d5a',
    COUNTS: '3abfa81725a4bb04cda9097ae9c617c504ee6d92f7d83d1940880cb58e91bfbb',
}


def ref(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    for path, expected in PINS.items():
        assert ref(path)['sha256'] == expected
    rows = json.loads(SOURCE.read_text())['measured_pairs']
    counts = json.loads(COUNTS.read_text())['per_prefix_conditional_rows']
    key = lambda r: (r['category'], r['request_id'], r['prefix_offset'])
    lookup = {key(row): row for row in counts}
    assert len(rows) == len(lookup) == 36 and len(LAYOUTS) == 8
    classes = sorted({row['category'] for row in rows})
    identities = {c: sorted({row['request_id'] for row in rows if row['category'] == c}) for c in classes}
    assert all(len(values) == 4 for values in identities.values())
    results = []
    folds = []
    for fold in range(4):
        heldout = {values[fold] for values in identities.values()}
        training = [row for row in rows if row['request_id'] not in heldout]
        testing = [row for row in rows if row['request_id'] in heldout]
        observations = make_observations(training)
        assert len(training) == 27 and len(testing) == 9
        assert all(o['request_id'] not in heldout for o in observations)
        folds.append(dict(fold=fold, heldout_request_ids=sorted(heldout),
                          training_rows=len(training), testing_rows=len(testing),
                          previous_decision_labels=len(observations)))
        for row in testing:
            # No current target outcome reaches the selector interface.
            nodes, predicted = choose(row['category'], row['raw_conditional_probabilities'], observations)
            conditional_count = lookup[key(row)]['conditional_counts_by_layout'][','.join(map(str, nodes))]
            assert conditional_count <= 5
            results.append(dict(
                category=row['category'], request_id=row['request_id'], prefix_offset=row['prefix_offset'],
                fold=fold, selected_nodes=nodes, prediction=predicted,
                actual_old_spine_count=row['old_g'], actual_old_joint_count=row['joint_g'],
                conditional_old_head_selected_count=conditional_count,
                conditional_proxy_delta=conditional_count-row['old_g'],
            ))
    assert len({key(row) for row in results}) == 36
    summary = []
    for category in classes:
        selected = [row for row in results if row['category'] == category]
        summary.append(dict(category=category, rows=len(selected),
                            old_spine_actual=sum(r['actual_old_spine_count'] for r in selected),
                            old_joint_actual=sum(r['actual_old_joint_count'] for r in selected),
                            grouped_conditional_proxy=sum(r['conditional_old_head_selected_count'] for r in selected),
                            selected_alternate_layout_rows=sum(tuple(r['selected_nodes']) != SPINE for r in selected)))
    result = dict(
        schema='ROOT.target-aligned-selector.grouped-exploratory-HOST.v1',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='completed_exploratory_conditional_proxy_only',
        inputs=[ref(path) for path in PINS],
        source=[ref(Path(__file__)), ref(D/'target_aligned_selector_prototype_ROOT.py')],
        fixed_design=dict(prior_strength=4, other_class_weight=.25,
                          primary_confidence_bin_edges=[.6,.9], hyperparameter_search=False,
                          depth_and_category_conditioning=True, exact_B4_ancestor_closure=True),
        folds=folds, per_class=summary, rows=results,
        limitations=[
            'Only 12 request identities, three correlated prefixes each. Grouping prevents same-identity label leakage but is exploratory, not new independent validation.',
            'Counterfactual selected counts reuse old causal target parent-head decisions. Unrun FULL9/side-head numerical behavior and KV repair are unproved.',
            'Prototype consumes only current draft probabilities/category and previous training labels. Evaluator alone reads heldout outcomes after selection.',
            'Fixed design was proposed after descriptive V206 inspection. This is not preregistered or confirmatory evidence.',
            'No family latency penalty, production selector, g curve, model run, Native qualification, workload Goodput, or adoption is created.',
        ],
        new_g_installed=False, production_selector_installed=False,
        measured_new_delivery=False, Goodput=None, NPU_used=False, Torch_imported=False,
    )
    output = D/'ROOT_TARGET_ALIGNED_SELECTOR_GROUPED_EXPLORATORY_HOST.json'
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=ref(output), per_class=summary)))


if __name__ == '__main__':
    main()
