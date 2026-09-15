# SPDX-License-Identifier: Apache-2.0

import random
import threading
import time

import pytest

from vllm_ascend.spec_decode.pearl.runtime import PearlDualModelScheduler
from vllm_ascend.spec_decode.pearl.spec_rhythm import (
    PipelinePhase,
    ProposalLifecycle,
    SpecRhythmBudgetShaper,
    SpecRhythmPipelineController,
    SpecRhythmRuntimeState,
    SpecRhythmScheduler,
)


def _states(count=4):
    return {
        index: SpecRhythmRuntimeState(
            request_index=index,
            home_batch_id=index % 2,
            slo_tpot_ms=40.0 if index < 2 else 150.0,
        )
        for index in range(count)
    }


def test_progress_gap_matches_paper_equation():
    state = SpecRhythmRuntimeState(
        request_index=0,
        home_batch_id=0,
        slo_tpot_ms=40.0,
        delivered_tokens=4,
        decode_elapsed_ms=150.0,
    )
    assert state.projected_progress_gap(90.0) == 2
    assert state.projected_progress_gap(0.0) == 0


def test_arrival_wait_is_scheduler_debt_but_not_decode_tpot():
    state = SpecRhythmRuntimeState(
        request_index=0,
        home_batch_id=0,
        slo_tpot_ms=40.0,
        arrival_wait_ms=120.0,
        delivered_tokens=4,
        decode_elapsed_ms=80.0,
    )
    assert state.effective_elapsed_ms == 200.0
    assert state.observed_tpot_ms == 20.0
    assert state.projected_progress_gap(0.0) == 1


def test_budget_shaper_honors_batch_roof_and_draft_window():
    states = _states()
    states[0].decode_elapsed_ms = 300.0
    states[1].decode_elapsed_ms = 300.0
    shaper = SpecRhythmBudgetShaper(
        min_gamma=1,
        max_gamma=5,
        roofline={"batch:2": 7},
    )
    plan = shaper.shape(
        plan_id=3,
        normal_request_indices=[1, 3],
        eager_request_indices=[0, 2],
        states=states,
        projected_wait_ms=80.0,
        context_len=256,
        draft_token_budget=9,
    )
    assert sum(plan.normal_budgets.values()) <= 7
    assert sum(plan.eager_budgets.values()) <= 7
    assert plan.allocated_draft_tokens <= 9
    assert plan.eager_priorities[0] > plan.eager_priorities[2]
    assert plan.eager_budgets.get(0, 0) >= plan.eager_budgets.get(2, 0)


def test_fixed_verification_budget_is_independent_of_gamma():
    states = _states(4)
    shaper_gamma2 = SpecRhythmBudgetShaper(
        min_gamma=1,
        max_gamma=2,
        verification_budget=12,
    )
    shaper_gamma5 = SpecRhythmBudgetShaper(
        min_gamma=1,
        max_gamma=5,
        verification_budget=12,
    )
    assert shaper_gamma2.verification_roof(64, 2048) == 12
    assert shaper_gamma5.verification_roof(64, 2048) == 12
    plan = shaper_gamma5.shape(
        plan_id=0,
        normal_request_indices=[1, 3],
        eager_request_indices=[0, 2],
        states=states,
        projected_wait_ms=0.0,
        context_len=2048,
        batch_size=64,
    )
    assert plan.verification_roof == 12
    # B is an independent upper bound, not a mandate to start gap-zero eager
    # work. The two normal requests reach their caps; two slots remain unused.
    assert plan.allocated_draft_tokens == 10
    assert plan.verification_input_tokens == 12  # Legacy envelope-size alias, not actual packed input.
    assert plan.unused_verification_tokens == 2
    assert plan.eager_budgets == {}


def test_measured_zero_roof_defers_speculation_without_inventing_gamma_budget():
    from vllm_ascend.spec_decode.pearl.roofline import ProfiledRoofline

    roofline = ProfiledRoofline(
        {"2:1": 0},
        {
            "attention_backend": "fused_infer_attention_tree_v1",
            "ar_attention_backend": "paged_attention_v1",
        },
        (),
    )
    shaper = SpecRhythmBudgetShaper(min_gamma=1, max_gamma=4, roofline=roofline)
    states = {
        index: SpecRhythmRuntimeState(request_index=index, home_batch_id=index)
        for index in range(2)
    }
    plan = shaper.shape(
        plan_id=3,
        normal_request_indices=[0, 1],
        eager_request_indices=[],
        states=states,
        projected_wait_ms=10.0,
        context_len=128,
        batch_size=2,
    )
    assert plan.verification_roof == 0
    assert plan.normal_budgets == {}
    assert plan.deferred_normal_request_indices == (0, 1)


def test_execution_plan_roof_is_reused_without_a_second_lookup():
    states = _states(2)
    shaper = SpecRhythmBudgetShaper(
        min_gamma=1,
        max_gamma=8,
        roofline={"2:1": 7, "batch:1": 1},
    )
    plan = shaper.shape(
        plan_id=4,
        normal_request_indices=[0],
        eager_request_indices=[],
        states=states,
        projected_wait_ms=0.0,
        context_len=32,
        # The execution controller has already looked up B for the complete
        # active state.  A one-row drafting window must not re-key the table.
        batch_size=1,
        verification_roof=7,
    )
    assert plan.verification_roof == 7
    assert plan.normal_budgets == {0: 7}


def test_execution_plan_roof_must_be_non_negative():
    states = _states(1)
    shaper = SpecRhythmBudgetShaper(min_gamma=1, max_gamma=8)
    with pytest.raises(ValueError, match="execution-plan verification roof"):
        shaper.shape(
            plan_id=0,
            normal_request_indices=[0],
            eager_request_indices=[],
            states=states,
            projected_wait_ms=0.0,
            context_len=32,
            verification_roof=-1,
        )


def test_fixed_budget_does_not_bypass_low_acceptance_eager_gate():
    states = _states(2)
    states[0].acceptance_ema = 0.01
    states[0].decode_elapsed_ms = 1000.0
    shaper = SpecRhythmBudgetShaper(
        min_gamma=1,
        max_gamma=8,
        verification_budget=8,
    )
    plan = shaper.shape(
        plan_id=0,
        normal_request_indices=[],
        eager_request_indices=[0],
        states=states,
        projected_wait_ms=100.0,
        context_len=32,
    )
    assert plan.eager_budgets == {}
    assert plan.allocated_draft_tokens == 0
    assert plan.unused_verification_tokens == 8


def test_zero_benefit_normal_keeps_minimum_but_does_not_fill_fixed_budget():
    states = _states(1)
    states[0].acceptance_ema = 0.0
    shaper = SpecRhythmBudgetShaper(
        min_gamma=1,
        max_gamma=8,
        verification_budget=8,
    )
    plan = shaper.shape(
        plan_id=0,
        normal_request_indices=[0],
        eager_request_indices=[],
        states=states,
        projected_wait_ms=100.0,
        context_len=32,
    )
    assert plan.normal_budgets == {0: 1}
    assert plan.unused_verification_tokens == 7


def test_small_fixed_budget_defers_normal_rows_without_starvation():
    states = _states(4)
    states[0].decode_elapsed_ms = 1e6
    shaper = SpecRhythmBudgetShaper(
        min_gamma=1,
        max_gamma=8,
        verification_budget=1,
    )
    selected = []
    for plan_id in range(4):
        plan = shaper.shape(
            plan_id=plan_id,
            normal_request_indices=list(states),
            eager_request_indices=[],
            states=states,
            projected_wait_ms=100.0,
            context_len=32,
        )
        assert sum(plan.normal_budgets.values()) == 1
        assert len(plan.deferred_normal_request_indices) == 3
        selected.extend(plan.normal_budgets)
    assert set(selected) == set(states)


def test_normal_deferral_age_survives_opposite_home_draft_window():
    states = _states(3)
    states[0].decode_elapsed_ms = 1e6
    shaper = SpecRhythmBudgetShaper(
        min_gamma=1,
        max_gamma=8,
        verification_budget=1,
    )

    def shape(indices, plan_id):
        return shaper.shape(
            plan_id=plan_id,
            normal_request_indices=indices,
            eager_request_indices=[],
            states=states,
            projected_wait_ms=100.0,
            context_len=32,
        )

    assert shape([0, 2], 0).normal_budgets == {0: 1}
    assert shape([1], 1).normal_budgets == {1: 1}
    assert shape([0, 2], 2).normal_budgets == {2: 1}


def test_zero_draft_window_defers_normal_rows_without_failure():
    states = _states(2)
    shaper = SpecRhythmBudgetShaper(
        min_gamma=1,
        max_gamma=8,
        verification_budget=8,
    )
    plan = shaper.shape(
        plan_id=0,
        normal_request_indices=[0, 1],
        eager_request_indices=[],
        states=states,
        projected_wait_ms=100.0,
        context_len=32,
        draft_token_budget=0,
    )
    assert plan.normal_budgets == {}
    assert plan.deferred_normal_request_indices == (0, 1)


def test_budget_shaper_preserves_prefix_by_allocating_integer_depths():
    states = _states(2)
    states[0].decode_elapsed_ms = 500.0
    shaper = SpecRhythmBudgetShaper(min_gamma=1, max_gamma=4)
    plan = shaper.shape(
        plan_id=0,
        normal_request_indices=[1],
        eager_request_indices=[0],
        states=states,
        projected_wait_ms=100.0,
        context_len=32,
        draft_token_budget=5,
    )
    assert 1 <= plan.normal_budgets[1] <= 4
    assert 1 <= plan.eager_budgets[0] <= 4
    assert plan.allocated_draft_tokens == 5


def test_fixed_gamma_reserves_complete_eager_proposals_inside_global_b():
    states = {
        index: SpecRhythmRuntimeState(
            request_index=index,
            home_batch_id=index % 2,
            slo_tpot_ms=40.0,
            delivered_tokens=1,
            decode_elapsed_ms=1000.0,
        )
        for index in range(40)
    }
    shaper = SpecRhythmBudgetShaper(
        min_gamma=4,
        max_gamma=4,
        verification_budget=120,
    )

    plan = shaper.shape(
        plan_id=0,
        normal_request_indices=range(30),
        eager_request_indices=range(30, 40),
        states=states,
        projected_wait_ms=100.0,
        context_len=32,
        eager_reserve_tokens=24,
    )

    assert plan.eager_budgets == {index: 4 for index in range(30, 36)}
    assert len(plan.normal_budgets) == 24
    assert set(plan.normal_budgets.values()) == {4}
    assert len(plan.deferred_normal_request_indices) == 6
    assert sum(plan.normal_budgets.values()) + sum(plan.eager_budgets.values()) == 120


def test_eager_reserve_does_not_withhold_b_without_eligible_eager_work():
    states = {
        index: SpecRhythmRuntimeState(
            request_index=index,
            home_batch_id=index % 2,
            slo_tpot_ms=40.0,
            delivered_tokens=1,
            decode_elapsed_ms=1000.0,
            acceptance_ema=(1.0 if index < 30 else 0.0),
        )
        for index in range(40)
    }
    shaper = SpecRhythmBudgetShaper(
        min_gamma=4,
        max_gamma=4,
        verification_budget=120,
    )

    plan = shaper.shape(
        plan_id=0,
        normal_request_indices=range(30),
        eager_request_indices=range(30, 40),
        states=states,
        projected_wait_ms=100.0,
        context_len=32,
        eager_reserve_tokens=24,
    )

    assert plan.eager_budgets == {}
    assert plan.normal_budgets == {index: 4 for index in range(30)}
    assert plan.deferred_normal_request_indices == ()
    assert plan.allocated_draft_tokens == 120


def test_budget_plan_rejects_normal_plus_eager_global_roof_overrun():
    from vllm_ascend.spec_decode.pearl.spec_rhythm import SpecRhythmBudgetPlan

    with pytest.raises(ValueError, match="global target verification roofline"):
        SpecRhythmBudgetPlan(
            plan_id=0,
            normal_budgets={0: 3},
            eager_budgets={1: 3},
            progress_gaps={0: 0, 1: 0},
            eager_priorities={1: 0.0},
            verification_roof=5,
            draft_token_budget=6,
            allocated_draft_tokens=6,
        )


def test_dual_batch_pipeline_warms_up_then_alternates():
    states = _states()
    controller = SpecRhythmPipelineController(states)
    warmup = controller.build_plan([0, 1, 2, 3])
    assert warmup.phase is PipelinePhase.WARMUP
    assert warmup.target_request_indices == ()
    assert warmup.normal_draft_request_indices == (0, 2)

    tickets = [controller.new_ticket(index, gamma=4, eager=False) for index in (0, 2)]
    controller.publish(tickets)
    first = controller.build_plan([0, 1, 2, 3])
    assert first.phase is PipelinePhase.STEADY
    assert first.target_request_indices == (0, 2)
    assert first.normal_draft_request_indices == (1, 3)
    assert first.eager_candidate_indices == (0, 2)


def test_slo_pipeline_can_merge_ready_home_batches():
    states = _states()
    controller = SpecRhythmPipelineController(states)
    tickets = [controller.new_ticket(index, gamma=4, eager=False) for index in (0, 2)]
    controller.publish(tickets)
    # Publish the opposite home so both logical batches are ready at once.
    opposite = [controller.new_ticket(index, gamma=4, eager=False) for index in (1, 3)]
    controller.publish(opposite)
    merged = controller.build_plan([0, 1, 2, 3], merge_ready_homes=True)
    assert set(merged.target_request_indices) == {0, 1, 2, 3}


def test_slo_pipeline_keeps_alternating_home_by_default():
    states = _states()
    controller = SpecRhythmPipelineController(states)
    controller.publish([controller.new_ticket(index, gamma=4, eager=False) for index in (0, 2)])
    controller.publish([controller.new_ticket(index, gamma=4, eager=False) for index in (1, 3)])

    plan = controller.build_plan([0, 1, 2, 3])

    assert plan.target_request_indices == (0, 2)
    assert plan.normal_draft_request_indices == ()
    assert plan.eager_candidate_indices == (0, 2)


def test_promoted_eager_joins_next_opposite_home_without_merging_old_ready_rows():
    states = _states(3)
    controller = SpecRhythmPipelineController(states)

    # Promote request 2's ahead-of-turn proposal from home 0.
    controller.publish([controller.new_ticket(2, gamma=4, eager=False)])
    promoted = controller.new_ticket(2, gamma=4, eager=True)
    controller.publish([promoted])
    assert _finish_simulated_proposal(controller, 2) is promoted

    # Home 1 owns the next normal verification. Home 0 also has an older
    # normal ticket, which must not be swept in with the promoted continuation.
    controller.publish(
        [
            controller.new_ticket(0, gamma=4, eager=False),
            controller.new_ticket(1, gamma=4, eager=False),
        ]
    )
    plan = controller.build_plan(
        [0, 1, 2],
        verification_budget=8,
    )

    assert plan.target_home_batch_id == 1
    assert plan.target_request_indices == (1, 2)
    assert plan.target_candidate_budgets == {1: 4, 2: 4}
    assert 0 not in plan.target_request_indices

    # Per-request completion order is mixed-home; the cycle owner controls
    # the next normal home after all tickets have committed.
    for index in plan.target_request_indices:
        _finish_simulated_proposal(controller, index)
    controller.finish_cycle(plan.target_home_batch_id)
    assert controller.next_target_home_batch_id == 0


def _finish_simulated_proposal(controller, index, *, fully_accepted=True):
    proposed = controller.ready[index].gamma
    accepted = proposed if fully_accepted else proposed // 2
    return controller.finish_verification(
        index,
        fully_accepted=fully_accepted,
        proposed_tokens=proposed,
        accepted_tokens=accepted,
        delivered_tokens=accepted + 1,
        draft_confidence=1.0,
        ema_alpha=0.2,
    )


def test_actual_ready_budget_bounds_retained_eager_plus_new_normal():
    """Audit regression: independently legal 4 + 8 proposals must not verify as 12."""
    controller = SpecRhythmPipelineController(_states(), next_target_home_batch_id=1)
    controller.publish(
        [
            controller.new_ticket(1, gamma=3, eager=False),
            controller.new_ticket(3, gamma=5, eager=False),
        ]
    )
    first = controller.build_plan(range(4), verification_budget=8)
    assert first.target_candidate_budgets == {1: 3, 3: 5}
    retained = controller.new_ticket(1, gamma=4, eager=True)
    controller.publish(
        [
            retained,
            controller.new_ticket(0, gamma=3, eager=False),
            controller.new_ticket(2, gamma=1, eager=False),
        ]
    )
    assert _finish_simulated_proposal(controller, 1) is retained
    _finish_simulated_proposal(controller, 3, fully_accepted=False)
    second = controller.build_plan(range(4), verification_budget=8)
    # The promoted request joins the next opposite-home target batch instead
    # of lingering for a full additional rotation.
    assert second.target_candidate_budgets == {0: 3, 1: 4, 2: 1}
    assert second.normal_draft_request_indices == (3,)
    fresh = controller.new_ticket(3, gamma=8, eager=False)
    controller.publish([fresh])
    for index in second.target_request_indices:
        _finish_simulated_proposal(controller, index, fully_accepted=False)

    third = controller.build_plan(range(4), verification_budget=8)
    assert sum(third.target_candidate_budgets.values()) <= 8
    assert third.target_request_indices == (3,)
    assert third.target_candidate_budgets == {3: 8}
    assert controller.ready[3] is fresh
    assert fresh.lifecycle is ProposalLifecycle.AVAILABLE
    assert fresh.gamma == 8  # No unsafe truncation of the tree or dependency.


@pytest.mark.parametrize("merge_homes", [False, True])
@pytest.mark.parametrize("priority", [False, True])
def test_actual_ready_budget_holds_under_rolling_continuations(merge_homes, priority):
    random_source = random.Random(0)
    states = _states()
    controller = SpecRhythmPipelineController(states)
    shaper = SpecRhythmBudgetShaper(
        min_gamma=1,
        max_gamma=8,
        verification_budget=8,
    )
    completed_rounds = {index: 0 for index in states}
    promoted_count = 0
    for _ in range(120):
        for state in states.values():
            state.add_decode_time(50.0)
        execution = controller.build_plan(
            tuple(states),
            priority=priority,
            projected_wait_ms=100.0,
            merge_ready_homes=merge_homes,
            verification_budget=8,
        )
        assert sum(execution.target_candidate_budgets.values()) <= 8
        assert execution.target_candidate_budgets == {
            index: controller.ready[index].gamma for index in execution.target_request_indices
        }
        budget = shaper.shape(
            plan_id=execution.plan_id,
            normal_request_indices=execution.normal_draft_request_indices,
            eager_request_indices=execution.eager_candidate_indices,
            states=states,
            projected_wait_ms=100.0,
            context_len=32,
        )
        assert budget.allocated_draft_tokens <= 8
        controller.publish(
            [
                controller.new_ticket(index, gamma=gamma, eager=eager)
                for budgets, eager in ((budget.normal_budgets, False), (budget.eager_budgets, True))
                for index, gamma in budgets.items()
            ]
        )
        for index in execution.target_request_indices:
            promoted = _finish_simulated_proposal(
                controller,
                index,
                fully_accepted=random_source.random() < 0.7,
            )
            promoted_count += int(promoted is not None)
            completed_rounds[index] += 1
    assert promoted_count > 0
    assert min(completed_rounds.values()) > 5


@pytest.mark.parametrize("request_cap", [None, 1])
def test_budget_deferral_does_not_starve_large_relaxed_request(request_cap):
    states = _states(2)
    states[0].decode_elapsed_ms = 1e6
    states[1].slo_tpot_ms = 150.0
    states[1].home_batch_id = 0
    controller = SpecRhythmPipelineController(states)
    controller.publish(
        [
            controller.new_ticket(0, gamma=1, eager=False),
            controller.new_ticket(1, gamma=8, eager=False),
        ]
    )
    first = controller.build_plan(
        [0, 1],
        verification_budget=8,
        max_target_requests=request_cap,
    )
    assert first.target_request_indices == (0,)
    _finish_simulated_proposal(controller, 0)
    controller.publish([controller.new_ticket(0, gamma=1, eager=False)])
    second = controller.build_plan(
        [0, 1],
        verification_budget=8,
        max_target_requests=request_cap,
    )
    assert second.target_candidate_budgets == {1: 8}


def test_slo_ready_budget_prioritizes_a_need_before_deferral_age():
    states = _states(2)
    for state in states.values():
        state.home_batch_id = 0
        state.delivered_tokens = 1
        state.decode_elapsed_ms = 200.0
    states[0].slo_tpot_ms = 40.0
    states[1].slo_tpot_ms = 150.0
    controller = SpecRhythmPipelineController(states)
    controller.publish(
        [controller.new_ticket(index, gamma=4, eager=False) for index in states]
    )
    # Even a previously deferred relaxed request cannot displace the tight
    # request while the latter has the larger section-4.3 progress gap.
    controller._ready_wait_cycles[1] = 10

    plan = controller.build_plan(
        [0, 1],
        priority=True,
        projected_wait_ms=100.0,
        merge_ready_homes=True,
        verification_budget=4,
    )

    assert plan.target_request_indices == (0,)
    assert plan.deferred_target_request_indices == (1,)


def test_ready_budget_accepts_explicit_candidate_counts():
    states = _states(2)
    states[1].home_batch_id = 0
    controller = SpecRhythmPipelineController(states)
    controller.publish([controller.new_ticket(index, gamma=8, eager=False) for index in states])
    plan = controller.build_plan(
        [0, 1],
        verification_budget=6,
        ready_candidate_counts={0: 3, 1: 3},
    )
    assert plan.target_candidate_budgets == {0: 3, 1: 3}


def test_oversized_ready_proposal_is_rejected_without_payload_mutation():
    controller = SpecRhythmPipelineController(_states(1))
    ticket = controller.new_ticket(0, gamma=9, eager=False)
    controller.publish([ticket])
    with pytest.raises(ValueError, match="rebuild or ancestor-safely prune"):
        controller.build_plan([0], verification_budget=8)
    assert controller.ready[0] is ticket
    assert ticket.gamma == 9
    assert ticket.lifecycle is ProposalLifecycle.AVAILABLE


def test_generic_scheduler_enforces_existing_ready_budget():
    scheduler = SpecRhythmScheduler(
        SpecRhythmBudgetShaper(min_gamma=1, max_gamma=8, verification_budget=8),
    )
    for index in range(2):
        scheduler.admit(index, home_batch_id=0)
    scheduler.controller.publish(
        [
            scheduler.controller.new_ticket(0, gamma=4, eager=False),
            scheduler.controller.new_ticket(1, gamma=8, eager=False),
        ]
    )
    schedule = scheduler.schedule(projected_wait_ms=0.0, context_len=32)
    assert sum(schedule.execution.target_candidate_budgets.values()) <= 8
    assert schedule.execution.deferred_target_request_indices


def test_full_acceptance_promotes_matching_eager_continuation():
    states = _states(2)
    controller = SpecRhythmPipelineController(states)
    normal = controller.new_ticket(0, gamma=3, eager=False)
    controller.publish([normal])
    eager = controller.new_ticket(0, gamma=2, eager=True)
    controller.publish([eager])

    promoted = controller.finish_verification(
        0,
        fully_accepted=True,
        proposed_tokens=3,
        accepted_tokens=3,
        delivered_tokens=3,
        draft_confidence=0.8,
        ema_alpha=0.2,
    )
    assert promoted is eager
    assert eager.lifecycle is ProposalLifecycle.AVAILABLE
    assert controller.ready[0] is eager
    assert states[0].prefix_epoch == eager.required_prefix_epoch == 1


def test_dual_model_scheduler_runs_both_runners_and_returns_schedule():
    scheduler = SpecRhythmScheduler(SpecRhythmBudgetShaper(min_gamma=1, max_gamma=2), max_num_seqs=2)
    scheduler.admit(0)
    calls = []

    def runner(schedule, role):
        calls.append((role, schedule.execution.phase))
        return role

    adapter = PearlDualModelScheduler(scheduler, runner, runner)
    result = adapter.execute_step(projected_wait_ms=0.0, context_len=8)
    assert {value[0] for value in calls} == {"draft", "target"}
    assert result.draft_result in {"draft", "target"}
    assert result.target_result in {"draft", "target"}
    assert result.elapsed_seconds >= 0.0


def test_dual_model_scheduler_records_real_worker_overlap():
    scheduler = SpecRhythmScheduler(SpecRhythmBudgetShaper(min_gamma=1, max_gamma=2), max_num_seqs=2)
    scheduler.admit(0)
    barrier = threading.Barrier(2)

    def runner(schedule, role):
        barrier.wait(timeout=1.0)
        time.sleep(0.01)
        return role

    result = PearlDualModelScheduler(scheduler, runner, runner).execute_step(projected_wait_ms=0.0, context_len=8)
    assert result.overlapped
    assert result.overlap_seconds > 0.0
    assert result.draft_elapsed_seconds > 0.0
    assert result.target_elapsed_seconds > 0.0


def test_scheduler_attaches_tree_plans_to_the_fixed_budget():
    scheduler = SpecRhythmScheduler(
        SpecRhythmBudgetShaper(
            min_gamma=1,
            max_gamma=4,
            verification_budget=4,
        ),
        max_num_seqs=2,
        tree_width=2,
        tree_max_depth=2,
        tree_max_model_len=16,
    )
    scheduler.admit(0)
    schedule = scheduler.schedule(
        projected_wait_ms=0.0,
        context_len=4,
        draft_token_budget=2,
        prefix_lengths={0: 4},
    )
    assert schedule.tree_enabled
    assert set(schedule.tree_plans) == {0}
    tree_plan = schedule.tree_plans[0]
    assert tree_plan.candidate_budget == schedule.budget.normal_budgets[0]
    assert tree_plan.candidate_budget <= schedule.budget.verification_roof


def test_rejection_discards_eager_continuation_and_advances_epoch():
    states = _states(2)
    controller = SpecRhythmPipelineController(states)
    normal = controller.new_ticket(0, gamma=4, eager=False)
    controller.publish([normal])
    eager = controller.new_ticket(0, gamma=3, eager=True)
    controller.publish([eager])

    promoted = controller.finish_verification(
        0,
        fully_accepted=False,
        proposed_tokens=4,
        accepted_tokens=1,
        delivered_tokens=2,
        draft_confidence=0.7,
        ema_alpha=0.2,
    )
    assert promoted is None
    assert eager.lifecycle is ProposalLifecycle.INVALIDATED
    assert 0 not in controller.ready
    assert 0 not in controller.staged_eager
    assert states[0].prefix_epoch == 1


def test_stale_mailbox_is_rejected_before_state_mutation():
    states = _states(1)
    controller = SpecRhythmPipelineController(states)
    ticket = controller.new_ticket(0, gamma=2, eager=False)
    controller.publish([ticket])
    states[0].prefix_epoch += 1
    with pytest.raises(RuntimeError, match="stale proposal"):
        controller.finish_verification(
            0,
            fully_accepted=True,
            proposed_tokens=2,
            accepted_tokens=2,
            delivered_tokens=2,
            draft_confidence=None,
            ema_alpha=0.2,
        )
    assert controller.ready[0] is ticket
    assert ticket.lifecycle is ProposalLifecycle.AVAILABLE
    assert states[0].delivered_tokens == 0
    assert states[0].verification_rounds == 0


def test_validate_verification_checks_routing_before_mutation():
    controller = SpecRhythmPipelineController(_states(1))
    ticket = controller.new_ticket(0, gamma=2, eager=False)
    controller.publish([ticket])
    assert controller.validate_verification(0) is ticket
    ticket.home_batch_id = 1
    with pytest.raises(RuntimeError, match="routing/lifecycle"):
        controller.validate_verification(0)
    assert controller.ready[0] is ticket
    assert ticket.lifecycle is ProposalLifecycle.AVAILABLE


def test_inactive_request_invalidates_ready_and_staged_payloads():
    states = _states(2)
    controller = SpecRhythmPipelineController(states)
    ready = controller.new_ticket(0, gamma=2, eager=False)
    controller.publish([ready])
    controller.build_plan([1])
    assert ready.lifecycle is ProposalLifecycle.INVALIDATED
    assert controller.ready == {}
