# SPDX-License-Identifier: Apache-2.0
"""SpecRhythm 4.3/5.3 eager gap closure and residual-window priority."""

import math

import pytest

from tests.ut.spec_decode.test_spec_rhythm_tree_loop import _TreeLoopHarness
from vllm_ascend.spec_decode.pearl import native_engine as native
from vllm_ascend.spec_decode.pearl.spec_rhythm import (
    ProposalLifecycle,
    SpecRhythmBudgetShaper,
    SpecRhythmPipelineController,
    SpecRhythmRuntimeState,
    SpecRhythmScheduler,
)


def _states():
    return {
        index: SpecRhythmRuntimeState(
            index,
            index % 2,
            slo_tpot_ms=40,
            delivered_tokens=10,
            decode_elapsed_ms=250,
        )
        for index in (0, 1)
    }


def _shape(shaper, states, plan_id=0):
    return shaper.shape(
        plan_id=plan_id,
        normal_request_indices=[0],
        eager_request_indices=[1],
        states=states,
        projected_wait_ms=100,
        context_len=128,
        batch_size=2,
        eager_token_cap=4,
    )


@pytest.mark.parametrize("elapsed", [250, 300, 300.001, 340, 500])
def test_progress_gap_matches_paper_token_units_at_boundary(elapsed):
    state = _states()[1]
    state.decode_elapsed_ms = elapsed
    # Paper 4.3: a_need = ceil(max(0, (L + T_next) / tau - N)).
    # L, T_next and tau are all milliseconds; the result is token count.
    expected = math.ceil(max(0, (elapsed + 100) / 40 - 10))
    assert state.arrival_wait_ms == 0
    assert state.projected_progress_gap(100) == expected


@pytest.mark.parametrize("fixed_budget", [None, 8])
def test_closed_gap_cannot_gain_new_eager_from_residual_budget(fixed_budget):
    states = _states()
    shaper = SpecRhythmBudgetShaper(min_gamma=1, max_gamma=4, verification_budget=fixed_budget)
    assert states[1].projected_progress_gap(100) == 0
    assert states[1].urgency(100) == pytest.approx(0.875)
    plan = _shape(shaper, states)
    assert plan.eager_budgets == {}
    assert plan.normal_budgets == {0: 4}  # Normal residual-Goodput stage remains.
    assert plan.unused_verification_tokens == 4


def test_positive_gap_closing_stops_only_subsequent_eager_allocation():
    states = _states()
    states[1].decode_elapsed_ms = 500
    shaper = SpecRhythmBudgetShaper(min_gamma=1, max_gamma=4, verification_budget=8)
    before = _shape(shaper, states)
    assert before.eager_budgets == {1: 4}
    states[1].delivered_tokens += 5
    after = _shape(shaper, states, plan_id=1)
    assert states[1].projected_progress_gap(100) == 0
    assert after.eager_budgets == {}
    assert after.normal_budgets == {0: 4}


def test_gap_closure_does_not_cancel_ready_or_valid_staged_promotion():
    states = _states()
    states[1].decode_elapsed_ms = 500
    shaper = SpecRhythmBudgetShaper(min_gamma=1, max_gamma=4, verification_budget=8)
    assert _shape(shaper, states).eager_budgets[1] == 4
    controller = SpecRhythmPipelineController(states)
    ready = controller.new_ticket(1, gamma=2, eager=False)
    controller.publish([ready])
    staged = controller.new_ticket(1, gamma=4, eager=True)
    controller.publish([staged])
    states[1].delivered_tokens += 5
    assert _shape(shaper, states, plan_id=1).eager_budgets == {}
    assert controller.ready[1] is ready
    assert controller.staged_eager[1] is staged
    promoted = controller.finish_verification(
        1,
        fully_accepted=True,
        proposed_tokens=2,
        accepted_tokens=2,
        delivered_tokens=3,
        draft_confidence=1.0,
        ema_alpha=0.2,
    )
    assert promoted is staged
    assert promoted.lifecycle is ProposalLifecycle.AVAILABLE
    assert controller.ready[1] is staged


def test_generic_scheduler_excludes_closed_gap_only_from_new_eager_plan():
    scheduler = SpecRhythmScheduler(
        SpecRhythmBudgetShaper(min_gamma=1, max_gamma=4, verification_budget=8), max_eager_tokens=4
    )
    for index in (0, 1):
        state = scheduler.admit(index, slo_tpot_ms=40)
        state.delivered_tokens = 10
        state.decode_elapsed_ms = 250
    ready = scheduler.controller.new_ticket(0, gamma=2, eager=False)
    scheduler.controller.publish([ready])
    schedule = scheduler.schedule(projected_wait_ms=100, context_len=128)
    assert schedule.execution.target_request_indices == (0,)
    assert schedule.execution.eager_candidate_indices == ()
    assert schedule.budget.eager_budgets == {}
    assert schedule.budget.normal_budgets == {1: 4}
    assert scheduler.controller.ready[0] is ready


def _control_state_after_warmup(monkeypatch, harness, configure):
    original = native.SpecRhythmPipelineController.build_plan

    def build(controller, *args, **kwargs):
        if len(harness.active_snapshots) == 1:
            configure(controller.request_states)
        return original(controller, *args, **kwargs)

    monkeypatch.setattr(native.SpecRhythmPipelineController, "build_plan", build)


def test_real_tree_loop_closes_eager_gate_before_window_accounting(monkeypatch):
    harness = _TreeLoopHarness(monkeypatch, requests=4, capacity=4, max_tokens=32, slo=40)

    def close_gap(states):
        for state in states.values():
            state.delivered_tokens = 100
            state.decode_elapsed_ms = 0

    _control_state_after_warmup(monkeypatch, harness, close_gap)
    original_estimate = native.DraftWindowEstimator.estimate
    window_inputs = []

    def estimate(estimator, **kwargs):
        window_inputs.append(kwargs)
        return original_estimate(estimator, **kwargs)

    monkeypatch.setattr(native.DraftWindowEstimator, "estimate", estimate)
    eager_requests = []
    original_draft = harness.draft

    def draft(*args, **kwargs):
        eager_requests.append(tuple((kwargs.get("eager_parent_sources") or {}).keys()))
        return original_draft(*args, **kwargs)

    harness.engine.draft_tree_forward = draft
    harness.run(max_rounds=2)
    assert len(window_inputs) == 2
    assert window_inputs[1] == {
        "normal_tokens": 8,
        "max_draft_tokens": 8,
        "eager_work": False,
    }
    assert eager_requests == [(), ()]


def test_single_eager_window_prioritizes_gap_benefit_not_tpot_ratio(monkeypatch):
    # Eight milliseconds of normal work plus the conservatively estimated
    # four-millisecond eager fixed cost must fit before the four-node eager
    # tree itself is admitted.
    harness = _TreeLoopHarness(monkeypatch, requests=4, capacity=4, max_tokens=32, slo=40, draft_window_ms=16)
    measured_priority = {}

    def configure(states):
        # Home 0's requests 0 and 2 are ready after warmup. With one spare
        # four-node tree, 2 has larger a_need despite 0's much higher ratio.
        for state in states.values():
            state.delivered_tokens = 100
            state.decode_elapsed_ms = 0
        states[0].delivered_tokens = 1
        states[0].decode_elapsed_ms = 400
        states[2].decode_elapsed_ms = 4400

    _control_state_after_warmup(monkeypatch, harness, configure)
    original_shape = native.SpecRhythmBudgetShaper.shape

    def shape(shaper, **kwargs):
        if kwargs["eager_request_indices"]:
            states, wait = kwargs["states"], kwargs["projected_wait_ms"]
            measured_priority.update(
                admitted=tuple(kwargs["eager_request_indices"]),
                gap0=states[0].projected_progress_gap(wait),
                gap2=states[2].projected_progress_gap(wait),
                ratio0=states[0].urgency(wait),
                ratio2=states[2].urgency(wait),
            )
        plan = original_shape(shaper, **kwargs)
        if kwargs["eager_request_indices"]:
            measured_priority["eager_budgets"] = dict(plan.eager_budgets)
        return plan

    monkeypatch.setattr(native.SpecRhythmBudgetShaper, "shape", shape)
    harness.run(max_rounds=2)
    assert measured_priority["gap2"] > measured_priority["gap0"]
    assert measured_priority["ratio0"] > measured_priority["ratio2"]
    assert measured_priority["admitted"] == (2,)
    assert measured_priority["eager_budgets"] == {2: 4}
