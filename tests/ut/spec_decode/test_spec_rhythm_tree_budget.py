# SPDX-License-Identifier: Apache-2.0

import math
import random

import pytest

from vllm_ascend.spec_decode.pearl.spec_rhythm import SpecRhythmBudgetShaper, SpecRhythmRuntimeState
from vllm_ascend.spec_decode.pearl.tree_budget import (
    DraftWindowEstimator,
    TreeCandidateRequest,
    select_global_tree_candidates,
)


def _chain(confidences, **kwargs):
    return TreeCandidateRequest(
        parents=[-1, *range(len(confidences) - 1)],
        token_ids=list(range(len(confidences))),
        conditional_confidences=confidences,
        max_candidates=len(confidences),
        **kwargs,
    )


def test_global_stage1_concentrates_candidates_on_urgent_progress_gap():
    requests = {
        0: _chain([0.9] * 3, progress_gap=3, urgency=2),
        1: _chain([0.9] * 3),
        2: _chain([0.9] * 3),
    }
    selected = select_global_tree_candidates(requests, 5)
    assert selected.candidate_counts == {0: 3, 1: 1, 2: 1}
    assert selected.expected_progress[0] == pytest.approx(0.9 + 0.81 + 0.729)
    assert sum(selected.stage1_candidate_counts.values()) == 5
    assert sum(selected.stage2_candidate_counts.values()) == 0


def test_global_stage2_redistributes_after_urgent_gap_is_closed():
    selected = select_global_tree_candidates(
        {
            0: _chain([0.9] * 3, progress_gap=1, urgency=2),
            1: _chain([0.99] * 3),
        },
        4,
    )
    assert selected.candidate_counts == {0: 2, 1: 2}
    assert selected.stage1_candidate_counts == {0: 2, 1: 1}
    assert selected.stage2_candidate_counts == {0: 0, 1: 1}


def test_candidate_selection_uses_joint_not_last_token_confidence():
    request = TreeCandidateRequest(
        parents=[-1, 0, -1],
        token_ids=[10, 11, 12],
        conditional_confidences=[0.1, 1.0, 0.8],
        max_candidates=3,
    )
    selected = select_global_tree_candidates({0: request}, 1)
    assert selected.selected_indices == {0: (2,)}
    assert selected.expected_progress == {0: 0.8}


def test_refinement_redistributes_beyond_equal_provisional_scalar_splits():
    selected = select_global_tree_candidates(
        {
            0: _chain([0.99] * 3),
            1: _chain([0.1] * 3),
        },
        4,
    )
    assert selected.candidate_counts == {0: 3, 1: 1}


def test_zero_value_eager_does_not_force_fill_global_budget():
    selected = select_global_tree_candidates(
        {
            0: _chain([0.0] * 3),
            1: _chain([0.9] * 3, minimum_candidates=0, acceptance_rate=0.0, progress_gap=10),
        },
        6,
    )
    assert selected.candidate_counts == {0: 1, 1: 0}
    assert selected.total_candidates == 1


def test_cost_roof_and_atomic_normal_minimum():
    selected = select_global_tree_candidates(
        {
            0: _chain([0.9] * 2, minimum_candidates=2, candidate_costs=[2.0, 2.0], urgency=10),
            1: _chain([0.9] * 3, candidate_costs=[1.0] * 3),
        },
        4,
        max_total_cost=3.0,
    )
    assert selected.selected_indices[0] == ()
    assert selected.deferred_request_indices == (0,)
    assert selected.candidate_counts[1] == 3
    assert selected.total_cost == 3.0


def test_aged_normal_minimum_prevents_urgent_request_starvation():
    selected = select_global_tree_candidates(
        {
            0: _chain([0.9], urgency=1e6),
            1: _chain([0.9], waiting_age=1),
        },
        1,
    )
    assert selected.candidate_counts == {0: 0, 1: 1}


def test_crossrequest_ties_are_independent_of_mapping_insertion_order():
    requests = {index: _chain([0.9] * 3, minimum_candidates=0) for index in range(4)}
    forward = select_global_tree_candidates(requests, 7)
    backward = select_global_tree_candidates(dict(reversed(list(requests.items()))), 7)
    assert forward == backward


def test_random_global_selections_preserve_ancestors_caps_counts_and_costs():
    random_source = random.Random(5)
    for _ in range(100):
        requests = {}
        for index in range(5):
            count = random_source.randint(1, 12)
            requests[index] = TreeCandidateRequest(
                parents=[random_source.randrange(-1, node) for node in range(count)],
                token_ids=list(range(count)),
                conditional_confidences=[random_source.random() for _ in range(count)],
                max_candidates=random_source.randint(1, count),
                minimum_candidates=random_source.randint(0, 1),
                progress_gap=random_source.randint(0, 5),
                urgency=random_source.random() * 3,
                candidate_costs=[random_source.uniform(0.5, 2.0) for _ in range(count)],
            )
        budget = random_source.randint(1, 20)
        cost_roof = random_source.uniform(1, 20)
        selected = select_global_tree_candidates(requests, budget, max_total_cost=cost_roof)
        cost = 0.0
        for index, nodes in selected.selected_indices.items():
            request = requests[index]
            assert list(nodes) == sorted(set(nodes))
            assert len(nodes) <= request.max_candidates
            assert all(request.parents[node] == -1 or request.parents[node] in nodes for node in nodes)
            cost += sum(request.candidate_costs[node] for node in nodes)
        assert selected.total_candidates == sum(selected.candidate_counts.values()) <= budget
        assert selected.total_cost == pytest.approx(cost)
        assert cost <= cost_roof
        assert (
            sum(selected.stage1_candidate_counts.values()) + sum(selected.stage2_candidate_counts.values())
            == selected.total_candidates
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"parents": [-1, 1]},
        {"conditional_confidences": [math.nan, 0.5]},
        {"conditional_confidences": [1.1, 0.5]},
        {"candidate_costs": [0.0, 1.0]},
        {"minimum_candidates": 3},
    ],
)
def test_malformed_exploratory_tree_is_rejected(kwargs):
    values = dict(parents=[-1, 0], token_ids=[1, 2], conditional_confidences=[0.5, 0.5], max_candidates=2)
    values.update(kwargs)
    with pytest.raises(ValueError):
        TreeCandidateRequest(**values)


def test_draft_window_bootstrap_only_admits_normal_work():
    predicted = DraftWindowEstimator().estimate(normal_tokens=3, max_draft_tokens=12)
    assert not predicted.calibrated
    assert predicted.draft_token_budget == 3
    assert predicted.eager_token_budget == 0


def test_measured_draft_window_reserves_normal_work_before_eager():
    estimator = DraftWindowEstimator(ema_alpha=0.5)
    estimator.observe(draft_compute_ms=8.0, drafted_tokens=4, target_verify_ms=20.0, communication_ms=2.0)
    predicted = estimator.estimate(normal_tokens=4, max_draft_tokens=16)
    assert predicted.calibrated
    assert predicted.draft_window_ms == 18.0
    assert predicted.draft_ms_per_token == 2.0
    assert predicted.draft_token_budget == 9
    assert predicted.eager_token_budget == 5
    assert predicted.residual_window_ms == 10.0
    assert predicted.predicted_exposed_draft_ms == 0.0
    estimator.observe(draft_compute_ms=12.0, drafted_tokens=4, target_verify_ms=10.0, communication_ms=4.0)
    updated = estimator.estimate(normal_tokens=4, max_draft_tokens=16)
    assert updated.draft_window_ms == 12.0
    assert updated.draft_ms_per_token == 2.5
    assert updated.draft_token_budget == 4
    assert updated.eager_token_budget == 0


def test_normal_overflow_is_reported_but_does_not_admit_eager():
    estimator = DraftWindowEstimator()
    estimator.observe(draft_compute_ms=10.0, drafted_tokens=2, target_verify_ms=8.0)
    predicted = estimator.estimate(normal_tokens=3, max_draft_tokens=20)
    assert predicted.draft_token_budget == 3
    assert predicted.eager_token_budget == 0
    assert predicted.residual_window_ms == 0.0
    assert predicted.predicted_exposed_draft_ms == 7.0


def test_draft_window_charges_and_learns_eager_batch_fixed_overhead():
    estimator = DraftWindowEstimator(ema_alpha=1.0)
    estimator.observe(
        draft_compute_ms=8.0,
        drafted_tokens=4,
        target_verify_ms=12.0,
    )

    # Before the first eager sample, one normal cycle is the conservative
    # fixed-cost proxy: 8 ms normal + 8 ms extra cannot fit a 12 ms W.
    blocked = estimator.estimate(
        normal_tokens=4,
        max_draft_tokens=12,
        eager_work=True,
    )
    assert blocked.eager_fixed_overhead_ms == 8.0
    assert blocked.eager_fixed_overhead_hidden is False
    assert blocked.eager_token_budget == 0

    # A measured eager cycle isolates 14 - (4 * 2) = 6 ms of batch-level
    # frontier/materialization overhead without changing the normal slope.
    estimator.observe(
        draft_compute_ms=14.0,
        drafted_tokens=4,
        target_verify_ms=20.0,
        eager_work=True,
    )
    predicted = estimator.estimate(
        normal_tokens=4,
        max_draft_tokens=12,
        eager_work=True,
    )
    assert estimator.draft_ms_per_token == 2.0
    assert estimator.eager_fixed_overhead_ms == 6.0
    assert predicted.eager_fixed_overhead_ms == 6.0
    assert predicted.eager_fixed_overhead_hidden is True
    assert predicted.eager_token_budget == 3
    assert predicted.residual_window_ms == 6.0


def test_profiled_roof_is_not_silently_increased_to_active_batch_size():
    shaper = SpecRhythmBudgetShaper(min_gamma=1, max_gamma=8, roofline={"batch:8": 3})
    assert shaper.verification_roof(8, 1024) == 3


def test_shaper_uses_measured_window_for_combined_normal_and_eager_work():
    states = {
        index: SpecRhythmRuntimeState(index, index % 2, slo_tpot_ms=40.0, decode_elapsed_ms=500.0) for index in range(2)
    }
    shaper = SpecRhythmBudgetShaper(min_gamma=1, max_gamma=8, verification_budget=12)
    plan = shaper.shape(
        plan_id=0,
        normal_request_indices=[0],
        eager_request_indices=[1],
        states=states,
        projected_wait_ms=100.0,
        context_len=32,
        draft_window_ms=7.0,
        draft_ms_per_token=2.0,
    )
    assert plan.allocated_draft_tokens == 3
    assert plan.draft_token_budget == 3
    assert plan.normal_budgets[0] >= 1


def test_shaper_preserves_only_normal_minimum_when_hidden_window_is_zero():
    states = {index: SpecRhythmRuntimeState(index, 0) for index in range(2)}
    shaper = SpecRhythmBudgetShaper(min_gamma=1, max_gamma=8, verification_budget=8)
    plan = shaper.shape(
        plan_id=0,
        normal_request_indices=[0],
        eager_request_indices=[1],
        states=states,
        projected_wait_ms=100.0,
        context_len=32,
        draft_window_ms=0.0,
        draft_ms_per_token=2.0,
    )
    assert plan.normal_budgets == {0: 1}
    assert plan.eager_budgets == {}
