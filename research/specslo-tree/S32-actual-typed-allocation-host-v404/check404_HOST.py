"""Exercise the real typed shape body and exact callback source, no backend."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
from types import MethodType, SimpleNamespace

HERE = Path(__file__).resolve().parent
B = HERE.parent


def main():
    sys.path[:0] = [str(B / 'tree-global64-normal24-typed-admission-v402'),
                    str(B / 'tree-frugal-S32-paid-budget-control-host-v401')]
    from check401_HOST import BACKENDS, original_control_sources, state_set
    from check402_typed_HOST import callback_factory
    rhythm = original_control_sources()
    sys.path.insert(0, str(B / 'tree-high-normal-cap4-native-families-v148'))
    import budget_policy
    import normal_admission_v140 as admission
    import split_shape_v132
    import shape404
    factory = callback_factory(split_shape_v132.shape)
    model = json.loads((B / 'tree-high-normal-cap4-native-families-v148/tree-gain-model.json').read_text())
    rng = random.Random(0)
    narrow = wide = eager_positive = 0
    for trial in range(1200):
        n = rng.randrange(1, 65)
        normal_n = rng.randrange(1, n + 1)
        eager = tuple(range(normal_n, min(n, normal_n + 8)))
        states = state_set(rhythm, n, rng)
        before = copy.deepcopy(states)
        policy = budget_policy.TreeBudgetPolicy(model)
        for label in ('tight', 'normal', 'loose'):
            for budget in range(1, 6):
                policy.observed[label, budget] = (17, rng.randrange(17, 17 * (min(4, budget) + 1) + 1))
        expected_policy = copy.deepcopy(policy)
        shaper = rhythm.SpecRhythmBudgetShaper(min_gamma=1, max_gamma=8, verification_budget=96)
        expected_shaper = copy.deepcopy(shaper)
        shaper.shape = MethodType(factory(policy), shaper)
        expected_shaper.shape = MethodType(factory(expected_policy), expected_shaper)
        admission.reset(policy, 1)
        admission.reset(expected_policy, 1)
        for i in range(n):
            age = rng.randrange(4)
            admission._state(policy)['ages'][i] = age
            admission._state(expected_policy)['ages'][i] = age
        engine = SimpleNamespace(_dynamic_tree_generation=1)
        kwargs = dict(plan_id=trial, normal_request_indices=tuple(range(normal_n)),
                      eager_request_indices=eager, states=states, projected_wait_ms=20.,
                      context_len=2048, batch_size=n, draft_token_budget=192,
                      eager_token_cap=3, eager_reserve_tokens=32,
                      verification_roof=rng.choice([0, 16, 64, 88, 96]))
        expected = expected_shaper.shape(**kwargs)
        admitted = expected.requested_normal
        got = shape404.shape(engine, shaper, **kwargs)
        assert engine._typed404_binding.policy is policy
        assert admission._state(policy)['call_count'] == 1
        if len(admitted) <= 24:
            assert got == expected and policy.counts == expected_policy.counts
            assert admission.snapshot(policy) == admission.snapshot(expected_policy)
            narrow += 1
        else:
            assert got.requested_normal == admitted
            assert len(got.normal_budgets) <= 32 and sum(got.normal_budgets.values()) <= 88
            assert sum(b == 3 for b in got.normal_budgets.values()) <= 24
            assert all(1 <= b <= 3 for b in got.normal_budgets.values())
            assert all(b == 3 for b in got.eager_budgets.values())
            assert set(got.normal_budgets) | set(got.deferred_normal_request_indices) == set(admitted)
            event = admission.snapshot(policy)['calls']
            if event:
                event = event[0]
                assert event['explicit_active_batch_size'] == n
                assert set(event['positive_normal_grants'][j]['owner'] for j in range(len(event['positive_normal_grants']))) == set(got.normal_budgets)
            wide += 1
            eager_positive += bool(got.eager_budgets)
        assert states == before and policy.observed == expected_policy.observed
        assert got.allocated_selected_candidates <= 192 and sum(got.eager_budgets.values()) <= 96
    # The large q4 slots follow strict class priority, even if loose requests
    # have larger urgency. Within a class original deferred age breaks ties.
    states = state_set(rhythm, 32, rng)
    states = {i: replace(s, max_gamma=5, slo_class='tight' if i < 24 else 'loose',
                        slo_tpot_ms=40. if i < 24 else 150.) for i, s in states.items()}
    caps = shape404.caps(engine._typed404_binding, shaper, tuple(range(32)), states, 20.)
    assert all(caps[i] == (3 if i < 24 else 2) for i in range(32))
    age_states = {i: replace(s, slo_class='tight', slo_tpot_ms=40.) for i, s in states.items()}
    for i in range(32):
        admission._state(policy)['ages'][i] = int(i >= 8)
    aged = shape404.caps(engine._typed404_binding, shaper, tuple(range(32)), age_states, 20.)
    assert all(aged[i] == (3 if i >= 8 else 2) for i in range(32))
    engine._dynamic_tree_generation = 2
    try:
        shape404.shape(engine, shaper, **kwargs)
    except RuntimeError as error:
        assert 'issued phase' in str(error)
    else:
        raise AssertionError('unissued phase accepted')
    admission.reset(policy, 2)
    shape404.shape(engine, shaper, **kwargs)
    assert shape404.snapshot(engine)['generation'] == 2
    try:
        shape404.shape(engine, shaper, **dict(kwargs, batch_size=65))
    except ValueError:
        pass
    else:
        raise AssertionError('global65 accepted')
    assert not any(n.split('.')[0] in BACKENDS for n in sys.modules)
    assert wide > 0 and narrow > 0 and eager_positive > 0
    output = dict(status='PASS_HOST404_ACTUAL_TYPED_S32_PAID_CAPS',
                  exact_narrow_typed_cases=narrow, wide_typed_cases=wide,
                  wide_cases_with_original_B3_eager=eager_positive,
                  original_policy_closure_and_callback_source=True,
                  original_admission32_and_global64_retained=True,
                  class_and_deferred_age_large_slot_priority=True,
                  original_gain_observations_and_request_states_preserved=True,
                  original_phase_generation_authoritative=True,
                  actual_LIVE_serving_binding_implemented=False,
                  fresh_S32_math_and_cost_required=True,
                  NPU_used=False, models_constructed=0,
                  model_qualification=False, production_permission=False,
                  Raw=None, Goodput=None)
    (HERE / 'HOST404_RESULT.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
