"""Real typed allocator reacts to causal request-specific branch benefit."""
import json
import sys
import types
from pathlib import Path

import check299


def main():
    from check_cost299_HOST import N, load
    for name in ('vllm_ascend', 'vllm_ascend.spec_decode', 'vllm_ascend.spec_decode.pearl'):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    load('vllm_ascend.spec_decode.pearl.roofline', N / 'roofline.py')
    rhythm = load('vllm_ascend.spec_decode.pearl.spec_rhythm', N / 'spec_rhythm.py')
    base = Path(__file__).resolve().parent.parent
    sys.path.extend([str(base / 'tree-high-normal-cap4-native-families-v148'),
                     str(base / 'tree-spare-branch-group-budget-full-v260')])
    from budget_policy import TreeBudgetPolicy
    from normal_admission_v140 import reset
    from budget299 import eligible, prepared_shape
    from risk299 import begin_generation, observe, snapshot, verify_snapshot
    model = json.loads((base / 'tree-high-normal-cap4-native-families-v148/tree-gain-model.json').read_text())
    policy = TreeBudgetPolicy(model)
    reset(policy, 1)
    begin_generation(policy, 1)
    states = {i: rhythm.SpecRhythmRuntimeState(i, i % 2, slo_tpot_ms=40,
                 slo_class='tight', max_gamma=8, output_token_budget=256,
                 delivered_tokens=1, decode_elapsed_ms=140) for i in range(16)}
    reports = []
    for phase in range(2):
        hard = set(range(8, 16)) if phase == 0 else set(range(8))
        for _ in range(96):
            for index, request in states.items():
                path = [4, 5, 6, 7] if index in hard else [0, 1, 2, 3]
                observe(policy, request, parents=[-1, 0, 1, 2, -1, 4, 5, 6],
                        accepted=path, outputs=[100, 101, 102, 103, 104], remaining_before=255)
        chosen = eligible(policy, states, tuple(states))
        assert chosen == hard
        plan = prepared_shape()(policy, types.SimpleNamespace(min_gamma=1, max_gamma=8),
            plan_id=phase, normal_request_indices=tuple(states), eager_request_indices=(),
            states=states, projected_wait_ms=10., context_len=127, draft_token_budget=192,
            batch_size=16, eager_token_cap=3, eager_reserve_tokens=0, verification_roof=96)
        assert dict(plan.normal_budgets) == {i: 8 if i in hard else 4 for i in states}
        assert sum(plan.normal_budgets.values()) == 96
        assert len(plan.normal_budgets) + sum(b > 4 for b in plan.normal_budgets.values()) == 24
        reports.append(dict(phase=phase, hard_owners=sorted(hard),
                            actual_branch_eligible=sorted(chosen), budgets=dict(plan.normal_budgets)))
    audit = verify_snapshot(snapshot(policy), 1)
    assert audit['observed_verdicts'] == 2 * 96 * 16
    assert not any(name.split('.')[0] in ('torch', 'torch_npu') for name in sys.modules)
    value = dict(status='HOST_actual_typed_allocator_request_conditional_spare_branch_owners_and_phase_change_passed',
                 phases=reports, causal_risk_audit=audit, same_class_priority_unchanged=True,
                 C96_and24_groups_preserved=True, synthetic_not_machine_measurement=True,
                 NPU_used=False, Goodput=None)
    target = Path(__file__).resolve().parent / 'HOST_RISK_ALLOCATION299_RESULT.json'
    with target.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
