"""Actual typed allocator source and callback, without models or NPU imports."""
import ast
import copy
import json
from pathlib import Path
import random
import sys
from types import MethodType, SimpleNamespace

HERE = Path(__file__).resolve().parent
B = HERE.parent
BASE = B / 'tree-high-normal-cap4-native-families-v148'


def callback_factory(original_shape):
    path = BASE / 'candidate_controls_v132.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CandidateControls')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    callback = next(n for n in init.body if isinstance(n, ast.FunctionDef) and n.name == 'candidate_shape')
    factory = ast.parse('def factory(policy):\n    pass\n').body[0]
    factory.body = [copy.deepcopy(callback), ast.Return(value=ast.Name(id='candidate_shape', ctx=ast.Load()))]
    namespace = dict(shape=original_shape)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[])),
                 str(path), 'exec', dont_inherit=True), namespace)
    return namespace['factory']


def main():
    sys.path.insert(0, str(B / 'tree-frugal-S32-paid-budget-control-host-v401'))
    from check401_HOST import original_control_sources, state_set, BACKENDS
    rhythm = original_control_sources()
    sys.path.insert(0, str(BASE))
    import budget_policy
    import normal_admission_v140
    import split_shape_v132
    import shape405
    import admission405
    from audit_normal405 import verify_snapshot
    sys.path.insert(0, str(B / 'tree-M120-S32-frugal-independent-target-host-v399'))
    from bootstrap399 import install
    install()
    from callback405 import constructor_callback
    from mapper209 import controls_class
    from types import FunctionType
    Controls = controls_class()
    callback_code, callback_namespace = constructor_callback(Controls)
    def factory(policy):
        return FunctionType(callback_code, callback_namespace,
                            closure=((lambda: policy).__closure__[0],))
    model = json.loads((BASE / 'tree-gain-model.json').read_text())
    rng = random.Random(0)
    exact = capped = 0
    for trial in range(800):
        n = rng.randrange(1, 65)
        normal_count = rng.randrange(1, n + 1)
        eager = tuple(range(normal_count, min(n, normal_count + 4)))
        states = state_set(rhythm, n, rng)
        states_before = copy.deepcopy(states)
        policy = budget_policy.TreeBudgetPolicy(model)
        for label in ['tight', 'normal', 'loose']:
            for budget in range(1, 6):
                policy.observed[label, budget] = (17, rng.randrange(17, 17 * (min(4, budget) + 1) + 1))
        expected_policy = copy.deepcopy(policy)
        shaper = rhythm.SpecRhythmBudgetShaper(min_gamma=1, max_gamma=8, verification_budget=96)
        expected_shaper = rhythm.SpecRhythmBudgetShaper(min_gamma=1, max_gamma=8, verification_budget=96)
        shaper.shape = MethodType(factory(policy), shaper)
        expected_shaper.shape = MethodType(factory(expected_policy), expected_shaper)
        normal_admission_v140.reset(policy, 1)
        normal_admission_v140.reset(expected_policy, 1)
        engine = SimpleNamespace(_dynamic_tree_generation=1)
        control = object.__new__(Controls)
        control.engine, control.mode, control.policy = engine, 'candidate', policy
        control.candidate_shape = shaper.shape.__func__
        engine._root405_controls = control
        kwargs = dict(plan_id=trial, normal_request_indices=tuple(range(normal_count)),
                      eager_request_indices=eager, states=states, projected_wait_ms=20.,
                      context_len=2048, batch_size=n, draft_token_budget=192,
                      eager_token_cap=3, eager_reserve_tokens=32, verification_roof=96)
        expected = expected_shaper.shape(**kwargs)
        result = shape405.shape(engine, shaper, **kwargs)
        record = admission405.snapshot(policy)
        verify_snapshot(shape405.snapshot(engine), 1)
        assert record['call_count'] == 1
        if normal_count <= 24:
            assert result == expected and policy.counts == expected_policy.counts
            assert record['ages'] == normal_admission_v140.snapshot(expected_policy)['ages']
            exact += 1
        else:
            assert len(result.requested_normal) == 24
            assert len(result.normal_budgets) <= 24
            event = record['calls'][0]
            assert event['raw_N'] == normal_count and event['admitted_N'] == 24
            assert event['actual_work_normal_row_limit'] == 24
            assert event['explicit_active_batch_size'] == n
            assert set(event['all_deferred_normal']) | set(result.normal_budgets) == set(range(normal_count))
            assert set(event['all_deferred_normal']).isdisjoint(result.normal_budgets)
            assert all(event['ages_after'][i] == (0 if i in result.normal_budgets else 1)
                       for i in range(normal_count))
            priority = budget_policy.PRIORITY
            selected = set(result.requested_normal)
            assert min(priority[states[i].slo_class] for i in selected) >= max(
                priority[states[i].slo_class] for i in range(normal_count) if i not in selected)
            capped += 1
        assert states == states_before and policy.observed == expected_policy.observed
        assert len(result.normal_budgets) <= 24
        assert sum(result.normal_budgets.values()) <= 96
        assert sum(result.eager_budgets.values()) <= 96
        assert result.allocated_selected_candidates <= 192
        assert all(b == 3 for b in result.eager_budgets.values())
        assert engine._normal405_binding.policy is policy
    # Actual original phase generation remains the authority, no reset-by-flag.
    normal_admission_v140.reset(policy, 2)
    engine._dynamic_tree_generation = 2
    shape405.shape(engine, shaper, **kwargs)
    assert admission405.snapshot(policy)['generation'] == 2
    engine._dynamic_tree_generation = 3
    try:
        shape405.shape(engine, shaper, **kwargs)
    except RuntimeError as error:
        assert 'phase generation' in str(error)
    else:
        raise AssertionError('unissued phase accepted')
    bad_snapshot = copy.deepcopy(shape405.snapshot(engine))
    bad_snapshot['normal_work_limit'] = 32
    try:
        verify_snapshot(bad_snapshot, 2)
    except RuntimeError:
        pass
    else:
        raise AssertionError('false normal work limit accepted by terminal auditor')
    assert not any(name.split('.')[0] in BACKENDS for name in sys.modules)
    output = dict(status='PASS_HOST405_ACTUAL_TYPED_ALLOCATOR_AND_CLASS_AGE_ADMISSION24',
                  actual_original_typed_callback_source_used=True,
                  exact_at_most24_plan_counter_age_cases=exact,
                  actual_class_priority_deferred_age_cases=capped,
                  per_role96_global192_paid_limits_retained=True,
                  original_model_and_online_observations_not_changed=True,
                  original_phase_generation_authoritative=True,
                  independent_retained_admission_audit_and_corruption_control=True,
                  admission=admission405.descriptor(), global_active_limit=64,
                  production_runtime_installed=False, NPU_used=False, models_constructed=0,
                  model_qualification=False, production_permission=False, Raw=None, Goodput=None)
    (HERE / 'HOST405_TYPED_ALLOCATION_RESULT.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({k: output[k] for k in ['status', 'exact_at_most24_plan_counter_age_cases',
                     'actual_class_priority_deferred_age_cases']}))


if __name__ == '__main__':
    main()
