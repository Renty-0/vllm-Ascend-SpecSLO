"""Read-only synthetic policy examples; never installs a candidate/runtime."""
import ast
from collections import Counter
import copy
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import types

ROOT = Path('/root/SpecSLO/budget-calibration-20261015')
P = ROOT/'tree-standalone-prefill-control-v143'
NATIVE = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/spec_rhythm.py')


class Firewall(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend'):
            raise RuntimeError('HOST construction forbids device/model import '+fullname)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, Firewall())
    sys.path.insert(0, str(P))
    from budget_policy import TreeBudgetPolicy
    import split_shape_v132 as actual_shape
    from normal_admission_v140 import reset
    from target_layout import envelope

    tree = ast.parse(NATIVE.read_text())
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)
        and node.module == 'vllm_ascend.spec_decode.pearl.roofline']
    if len(imports) != 1:
        raise RuntimeError('Actual Native HOST state capsule import anchor differs')
    tree.body.remove(imports[0])
    native = types.ModuleType('_CAP4_ACTUAL_NATIVE_HOST_ONLY')
    sys.modules[native.__name__] = native
    native.ProfiledRoofline = type('UnusedProfiledRoofline', (), {})
    exec(compile(tree, str(NATIVE)+':HOST-state-capsule', 'exec'), vars(native))

    # One expression only, in a private HOST namespace. Counter label strings
    # remain original and are deliberately not published as candidate metrics.
    original = ast.parse((P/'split_shape_v132.py').read_text())
    modified = copy.deepcopy(original)
    anchors = [node for node in ast.walk(modified) if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'limit' for t in node.targets)]
    if len(anchors) != 1 or ast.unparse(anchors[0].value) != 'DraftWorkLimit(normal_per_request=3 if len(normal) > 8 else 5)':
        raise RuntimeError('Actual high-N cap3 expression differs')
    anchors[0].value.keywords[0].value.body.value = 4
    namespace = dict(vars(actual_shape))
    exec(compile(ast.fix_missing_locations(modified), str(P/'split_shape_v132.py')+':private-HOST-cap4-expression', 'exec'), namespace)
    cap4 = namespace['shape']
    reverse = copy.deepcopy(modified)
    assignment = next(node for node in ast.walk(reverse) if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'limit' for t in node.targets))
    assignment.value.keywords[0].value.body.value = 3
    if ast.dump(reverse, include_attributes=False) != ast.dump(original, include_attributes=False):
        raise RuntimeError('Private HOST expression changed additional algorithm AST')

    def construct(name, gaps, *, flat_loose_horizon=False, learned_normal_B3=False, eager=0):
        classes = ['tight']*19+['normal']*6+['loose']*7+['normal']*eager
        states = {}
        for owner, category in enumerate(classes):
            deadline = dict(tight=40., normal=50., loose=150.)[category]
            state = native.SpecRhythmRuntimeState(request_index=owner,
                home_batch_id=owner%2, slo_tpot_ms=deadline, slo_class=category,
                max_gamma=5, output_token_budget=256)
            state.delivered_tokens = 255 if flat_loose_horizon and category == 'loose' else 1
            gap = gaps[category] if owner < 32 else 1.0
            state.decode_elapsed_ms = (state.delivered_tokens-1+gap)*deadline
            states[owner] = state
        out = dict(name=name, synthetic_states_only=True,
            synthetic_gap_by_class=gaps, loose_remaining_output_one=flat_loose_horizon,
            synthetic_observed_normal_B3=[12,23] if learned_normal_B3 else None,
            synthetic_eager_gap=1.0 if eager else None,
            admitted_eager_indices=list(range(32,32+eager)), variants=[])
        for label, shape in (('original_highN_cap3', actual_shape.shape), ('private_HOST_highN_cap4', cap4)):
            policy = TreeBudgetPolicy(json.loads((P/'tree-gain-model.json').read_text()))
            if learned_normal_B3:
                policy.observed[('normal',3)] = (12,23)
            reset(policy, 1)
            shaper = native.SpecRhythmBudgetShaper(min_gamma=1, max_gamma=8, verification_budget=96)
            plan = shape(policy, shaper, plan_id=0, normal_request_indices=tuple(range(32)),
                eager_request_indices=tuple(range(32,32+eager)), states=copy.deepcopy(states),
                projected_wait_ms=0., context_len=128, draft_token_budget=192,
                batch_size=64, eager_token_cap=3, eager_reserve_tokens=32, verification_roof=96)
            budgets = [plan.normal_budgets.get(i,0) for i in range(32)]
            n = len(plan.normal_budgets);c = sum(budgets);q = n+c
            physical = envelope(n,q,'causal')
            out['variants'].append(dict(label=label,normal_budgets=budgets,N=n,C=c,Q=q,
                class_budget_histograms={category:{str(k):v for k,v in sorted(Counter(
                    budgets[i] for i in range(32) if classes[i]==category).items())}
                    for category in ('tight','normal','loose')},
                physical_M=physical[0],physical_S=physical[1],dummy_query_lengths=physical[2],
                raw_eager_budgets=dict(plan.raw_eager_budgets),
                retained_eager_budgets=dict(plan.eager_budgets),
                shared_steps4_eager_deferred_rows=policy.counts['eager_deferred_shared_steps4_rows'],
                need_by_owner={str(i):plan.needs[i] for i in range(32)},
                no_new_runtime_installed=True))
        return out

    inputs = {str(P/name):sha(P/name) for name in ('split_shape_v132.py','split_budget_v132.py',
        'budget_policy.py','tree-gain-model.json','normal_admission_v140.py','target_layout.py')}
    inputs[str(NATIVE)] = sha(NATIVE)
    document = dict(schema='HOST.actual-policy-synthetic-highN-cap4-examples.v1',
        scope='Private AST cap-expression construction only; no current actual trace was read',
        input_sha256=inputs,normal_classes=dict(tight=19,normal=6,loose=7),
        candidate_epoch_created=False,runtime_installed=False,Torch_imported=False,
        device_queried=False,model_initialized=False,device_qualification_created=False,
        old_priority_need_gain_heap_and_eager_defer_AST_unchanged=True,
        original_source_counter_labels_not_adopted_as_new_candidate_metrics=True,
        cases=[construct('prior_gap_zero',dict(tight=0.,normal=0.,loose=0.)),
            construct('initial_need_T4_N2_L1_then_heap',dict(tight=3.5,normal=2.2,loose=1.)),
            construct('C95_unaffordable_positive_jump',dict(tight=3.5,normal=2.2,loose=.5),
                      flat_loose_horizon=True,learned_normal_B3=True),
            construct('eight_postwindow_eager_inputs',dict(tight=0.,normal=0.,loose=0.),eager=8)],
        performance_prediction=None,formal_Goodput=False)
    if any(sha(path)!=wanted for path,wanted in inputs.items()):
        raise RuntimeError('Readonly actual source changed during construction')
    destination=ROOT/'budget-diagnostics/HIGH_NORMAL_CAP4_N32_SYNTHETIC_HOST_EXAMPLE.json'
    with destination.open('x') as stream:
        json.dump(document,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(path=str(destination),sha256=sha(destination),
        cases=[dict(name=row['name'],variants=[{k:v[k] for k in ('label','C','Q','physical_M',
            'physical_S','class_budget_histograms','shared_steps4_eager_deferred_rows')} for v in row['variants']])
            for row in document['cases']]),indent=2))


if __name__ == '__main__':main()
