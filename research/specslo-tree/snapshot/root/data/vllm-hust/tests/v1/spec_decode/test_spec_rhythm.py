# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm.v1.spec_decode.spec_rhythm import SpecRhythmPolicy


class _SamplingParams:
    def __init__(self, extra_args):
        self.extra_args = extra_args


def _request(*, extra_args=None, arrival_time=100.0, num_tokens=128):
    return SimpleNamespace(
        request_id="req-0",
        arrival_time=arrival_time,
        num_tokens=num_tokens,
        sampling_params=_SamplingParams(extra_args),
    )


def test_spec_rhythm_budget_is_bounded_by_roofline():
    policy = SpecRhythmPolicy(
        min_gamma=1,
        max_gamma=5,
        roofline={"8:1": 3},
    )
    request = _request(
        extra_args={"spec_rhythm": {"slo_tpot_ms": 50}},
        arrival_time=99.0,
    )

    plan = policy.plan_for(
        request,
        batch_size=8,
        context_len=128,
        plan_id=1,
        now=100.0,
    )

    assert 1 <= plan.gamma <= 3
    assert plan.eager_tokens == 0


def test_spec_rhythm_urgent_request_gets_eager_continuation():
    policy = SpecRhythmPolicy(
        min_gamma=1,
        max_gamma=5,
        max_eager_tokens=2,
        urgency_threshold=0.75,
    )
    request = _request(
        extra_args={"spec_rhythm": {"slo_tpot_ms": 50}},
        arrival_time=99.0,
    )

    plan = policy.plan_for(
        request,
        batch_size=8,
        context_len=128,
        plan_id=1,
        now=100.0,
    )

    assert plan.urgency == 1.0
    assert plan.gamma == 5
    assert plan.eager_tokens == 2


def test_low_acceptance_reduces_budget_and_epoch_is_carried():
    policy = SpecRhythmPolicy(min_gamma=1, max_gamma=5)
    request = _request(
        extra_args={"spec_rhythm": {"slo_tpot_ms": 50}},
        arrival_time=99.0,
    )
    first = policy.plan_for(
        request,
        batch_size=8,
        context_len=128,
        plan_id=1,
        now=100.0,
    )
    for _ in range(5):
        policy.record_verification(
            "req-0", accepted_tokens=0, proposed_tokens=5
        )
    state = policy.state_for(request)
    state.invalidate_continuation()
    second = policy.plan_for(
        request,
        batch_size=8,
        context_len=128,
        plan_id=2,
        now=100.0,
    )

    assert second.gamma < first.gamma
    assert second.continuation_epoch == 1
    assert second.parent_prefix_epoch == 1


def test_step_plan_partitions_home_batches_and_marks_eager_requests():
    policy = SpecRhythmPolicy(min_gamma=1, max_gamma=4, max_eager_tokens=1)
    requests = [
        _request(
            extra_args={"spec_rhythm": {"slo_tpot_ms": 10}},
            arrival_time=99.0,
        ),
        SimpleNamespace(
            request_id="req-1",
            arrival_time=100.0,
            num_tokens=128,
            sampling_params=_SamplingParams(None),
        ),
    ]
    plans = policy.plan_batch(requests, plan_id=3, now=100.0)
    step_plan = policy.build_step_plan(requests, plans, plan_id=3)

    assert step_plan.target_home_batch_id == 1
    assert step_plan.draft_home_batch_id == 0
    assert set(step_plan.target_request_ids).isdisjoint(step_plan.draft_request_ids)
    assert set(step_plan.target_request_ids) | set(step_plan.draft_request_ids) == {
        "req-0",
        "req-1",
    }
    assert step_plan.execution_mode == "metadata_only"
    assert not step_plan.is_execution_ready
