"""Real typed allocator checks for weak, strong and priority-mixed gains."""
import json
import sys
import types
from pathlib import Path

P = Path(__file__).resolve().parent
B = P.parent
sys.path[:0] = [str(B / 'tree-capacity-request-risk-span-full-v299'),
               str(B / 'tree-spare-capacity-budget-host-v290')]
import check299


def main():
    from check_cost299_HOST import N, load
    for name in ('vllm_ascend', 'vllm_ascend.spec_decode', 'vllm_ascend.spec_decode.pearl'):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    load('vllm_ascend.spec_decode.pearl.roofline', N / 'roofline.py')
    rhythm = load('vllm_ascend.spec_decode.pearl.spec_rhythm', N / 'spec_rhythm.py')
    sys.path[:0] = [str(B / 'tree-high-normal-cap4-native-families-v148'),
                   str(B / 'tree-spare-branch-group-budget-full-v260')]
    from budget_policy import TreeBudgetPolicy
    from normal_admission_v140 import reset
    from budget309 import eligible, prepared_shape
    import budget290
    from cost309 import envelope, ratio
    model = json.loads((B / 'tree-high-normal-cap4-native-families-v148/tree-gain-model.json').read_text())
    original_eligible = budget290.prepared_shape().__globals__['shape_draft_work'].__globals__['eligible']
    rows = []
    for n in (1, 4, 8, 12, 16, 20, 23, 24):
        for kind in ('weak', 'strong'):
            policy = TreeBudgetPolicy(model)
            reset(policy, 1)
            curve = ([1.5, 2.1, 2.8, 3.1, 3.2, 3.3, 3.35, 3.4] if kind == 'weak'
                     else [1.1, 1.3, 1.6, 2., 2.5, 3.5, 4.5, 5.])
            for cls in ('tight', 'normal', 'loose'):
                for budget, gain in enumerate(curve, 1):
                    policy.observed[cls, budget] = (1000000, round(1000000 * gain))
            states = {i: rhythm.SpecRhythmRuntimeState(
                i, i % 2, slo_tpot_ms=40, slo_class='tight', max_gamma=8,
                output_token_budget=256, delivered_tokens=1, decode_elapsed_ms=140)
                for i in range(n)}
            chosen = eligible(policy, states, tuple(states))
            if kind == 'weak' or n >= 23:
                assert not chosen
            elif n <= 12:
                assert chosen == frozenset(range(n))
            else:
                assert chosen == frozenset(range(24 - n))
            plan = prepared_shape()(
                policy, types.SimpleNamespace(min_gamma=1, max_gamma=8),
                plan_id=1, normal_request_indices=tuple(states), eager_request_indices=(),
                states=states, projected_wait_ms=10., context_len=127,
                draft_token_budget=192, batch_size=n, eager_token_cap=3,
                eager_reserve_tokens=0, verification_roof=96)
            assert dict(plan.normal_budgets) == {i: 8 if i in chosen else 4 for i in states}
            assert plan.allocated_selected_candidates <= 192
            assert sum(plan.normal_budgets.values()) <= 96
            assert n + sum(b > 4 for b in plan.normal_budgets.values()) <= 24
            rows.append(dict(N=n, gain_kind=kind, branches=len(chosen),
                             actual_normal_candidates=sum(plan.normal_budgets.values())))
    policy = TreeBudgetPolicy(model)
    reset(policy, 1)
    for cls in ('tight', 'normal', 'loose'):
        for budget, gain in enumerate([1.1, 1.3, 1.6, 2., 2.5, 3.5, 4.5, 5.], 1):
            policy.observed[cls, budget] = (1000000, round(1000000 * gain))
    states = {i: rhythm.SpecRhythmRuntimeState(
        i, i % 2, slo_tpot_ms=40 if i >= 8 else 150,
        slo_class='tight' if i >= 8 else 'loose', max_gamma=8,
        output_token_budget=256, delivered_tokens=1, decode_elapsed_ms=140) for i in range(16)}
    assert eligible(policy, states, tuple(states)) == frozenset(range(8, 16))
    for invalid in (True, -1, 13, 1.5):
        try:
            ratio(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid branch count accepted')
    assert original_eligible is budget290.eligible
    assert budget290.prepared_shape().__globals__['shape_draft_work'].__globals__['eligible'] is original_eligible
    assert all(ratio(i) < ratio(i + 1) for i in range(12))
    assert not any(name.split('.')[0] in ('torch', 'torch_npu') for name in sys.modules)
    result = dict(status='HOST_actual_typed_cost_gate_weak_strong_and_priority_cases_passed',
                  cases=rows, measured_ratio_envelope=list(envelope()),
                  original_allocator_not_mutated=True, original_priority_order_preserved=True,
                  C96_future96_global192_and24_groups_preserved=True,
                  synthetic_gains_not_new_machine_observations=True,
                  real_draft_cost_included_in_proxy=False,
                  NPU_used=False, runtime_installed=False, Goodput=None)
    with (P / 'HOST309_RESULT.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
