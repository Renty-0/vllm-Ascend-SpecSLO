"""Actual typed allocator/controller under proposed role caps; HOST only."""
import dataclasses
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
    from check_cost299_HOST import load, N
    for name in ('vllm_ascend', 'vllm_ascend.spec_decode', 'vllm_ascend.spec_decode.pearl'):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    load('vllm_ascend.spec_decode.pearl.roofline', N / 'roofline.py')
    rhythm = load('vllm_ascend.spec_decode.pearl.spec_rhythm', N / 'spec_rhythm.py')
    sys.path[:0] = [str(B / 'tree-high-normal-cap4-native-families-v148'),
                   str(B / 'tree-spare-branch-group-budget-full-v260')]
    import split_budget_v132 as original
    from normal_admission_v140 import reset
    from budget_policy import TreeBudgetPolicy
    from budget328 import limit_classes, prepared_shape, build_plan
    target, work = limit_classes()
    assert dataclasses.asdict(target()) == dict(candidates=128, queries=160, rows=32)
    assert dataclasses.asdict(work()) == dict(dataclasses.asdict(original.DraftWorkLimit()),
                                            normal_candidates=128, future_eager_candidates=64,
                                            normal_per_request=8)
    assert dataclasses.asdict(original.TargetAdmissionLimit()) == dict(candidates=96, queries=128, rows=32)
    rejected = []
    for constructor, kwargs in ((target, dict(candidates=129)), (target, dict(queries=161)),
                                 (target, dict(rows=33)), (work, dict(normal_candidates=129)),
                                 (work, dict(future_eager_candidates=65)),
                                 (work, dict(global_selected_candidates=193)),
                                 (work, dict(normal_per_request=9))):
        try:
            constructor(**kwargs)
        except ValueError:
            rejected.append(kwargs)
        else:
            raise AssertionError('Invalid explicit limit accepted')
    model = json.loads((B / 'tree-high-normal-cap4-native-families-v148/tree-gain-model.json').read_text())
    rows = []
    for n in range(1, 33):
        for eager_count in (0, 8, 16, 32):
            policy = TreeBudgetPolicy(model, target_queries=160, max_target_requests=32)
            reset(policy, 1)
            for cls in ('tight', 'normal', 'loose'):
                for budget, gain in enumerate([1.1, 1.3, 1.6, 2., 2.5, 3.5, 4.5, 5.], 1):
                    policy.observed[cls, budget] = (1000000, round(gain * 1000000))
            states = {i: rhythm.SpecRhythmRuntimeState(
                i, i % 2, slo_tpot_ms=40 if i < n else 150,
                slo_class='tight' if i < n else 'loose', max_gamma=8,
                output_token_budget=256, delivered_tokens=1, decode_elapsed_ms=140)
                for i in range(n + eager_count)}
            result = prepared_shape()(
                policy, types.SimpleNamespace(min_gamma=1, max_gamma=8), plan_id=1,
                normal_request_indices=tuple(range(n)),
                eager_request_indices=tuple(range(n, n + eager_count)), states=states,
                projected_wait_ms=10., context_len=127, draft_token_budget=192,
                batch_size=n + eager_count, eager_token_cap=3,
                eager_reserve_tokens=64, verification_roof=128)
            normal, eager = dict(result.normal_budgets), dict(result.eager_budgets)
            assert sum(normal.values()) <= 128 and sum(eager.values()) <= 64
            assert result.allocated_selected_candidates <= 192
            assert len(normal) <= 32 and len(normal) + sum(b > 4 for b in normal.values()) <= 32
            assert len(normal) + sum(normal.values()) <= 160
            assert sum(b > 4 for b in normal.values()) <= 12
            if n == 32:
                assert normal == dict.fromkeys(range(32), 4)
                assert sum(normal.values()) == 128
            rows.append(dict(N=n, eager_requested=eager_count, normal=sum(normal.values()),
                             future=sum(eager.values()), selected=result.allocated_selected_candidates))
    controller_rows = []
    for n in (4, 8, 12, 16, 20, 24, 28, 31, 32):
        for budget in (5, 6, 7, 8):
            states = {i: rhythm.SpecRhythmRuntimeState(
                i, i % 2, slo_tpot_ms=40 if i >= n // 2 else 150,
                slo_class='tight' if i >= n // 2 else 'loose', max_gamma=8,
                output_token_budget=256, delivered_tokens=1, decode_elapsed_ms=140)
                for i in range(n)}
            controller = rhythm.SpecRhythmPipelineController(states)
            controller.publish([controller.new_ticket(i, gamma=budget if i >= n // 2 else 4, eager=False)
                                for i in states])
            before = {i: (ticket.proposal_id, ticket.gamma, ticket.required_prefix_epoch)
                      for i, ticket in controller.ready.items()}
            result = build_plan(controller, list(states), active_indices_registered_unique=True,
                                priority=True, target_slo_first=True, merge_ready_homes=True,
                                max_target_requests=32, verification_budget=128)
            values = result.target_candidate_budgets
            assert len(values) + sum(b > 4 for b in values.values()) <= 32
            assert sum(values.values()) <= 128 and len(values) + sum(values.values()) <= 160
            assert {i: (ticket.proposal_id, ticket.gamma, ticket.required_prefix_epoch)
                    for i, ticket in controller.ready.items()} == before
            if any(i < n // 2 for i in values):
                assert set(range(n // 2, n)) <= set(values)
            controller_rows.append(dict(N=n, B=budget, selected=len(values),
                                        groups=len(values) + sum(b > 4 for b in values.values()),
                                        candidates=sum(values.values())))
    assert not any(name.split('.')[0] in ('torch', 'torch_npu') for name in sys.modules)
    output = dict(status='HOST_actual_typed_normal128_future64_global192_and_groups32_passed',
                  allocator_cases=rows, actual_controller_cases=controller_rows,
                  rejected_limits=rejected, original_DTOs_and_controller_not_mutated=True,
                  global_budget192_preserved=True, priority_preserved=True,
                  future_original_B3_only_gate_preserved=True,
                  synthetic_gains_only=True, actual218_bound_repair_not_yet_tested=True,
                  raw_draft_and_target_model_limits_not_yet_qualified=True,
                  NPU_used=False, Goodput=None, adopted=False)
    with (P / 'HOST328_RESULT.json').open('x') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: value for key, value in output.items()
                      if key not in ('allocator_cases', 'actual_controller_cases')}))


if __name__ == '__main__':
    main()
