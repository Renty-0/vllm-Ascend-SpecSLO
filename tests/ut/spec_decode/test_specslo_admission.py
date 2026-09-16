# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_ascend.spec_decode.pearl.admission import decide_prefill_coalesce


def _decision(**overrides):
    values = {
        "ready_count": 1,
        "remaining_count": 8,
        "active_request_count": 3,
        "request_limit": 64,
        "minimum_requests": 4,
        "maximum_wait_ms": 1100.0,
        "now": 2.0,
        "ready_arrival_times": [1.5],
        "token_budget_saturated": False,
    }
    values.update(overrides)
    return decide_prefill_coalesce(**values)


def test_prefill_coalesce_defers_until_size_or_timeout():
    deferred = _decision()
    assert not deferred.release
    assert deferred.reason == "defer"
    assert deferred.waited_ms == pytest.approx(500.0)

    size = _decision(
        ready_count=4,
        ready_arrival_times=[1.5, 1.6, 1.7, 1.8],
    )
    assert size.release
    assert size.reason == "size"

    timeout = _decision(now=2.7)
    assert timeout.release
    assert timeout.reason == "timeout"
    assert timeout.waited_ms == pytest.approx(1200.0)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"active_request_count": 0}, "empty_service"),
        ({"token_budget_saturated": True}, "token_budget"),
        ({"remaining_count": 3}, "tail"),
        ({"ready_arrival_times": [None]}, "missing_arrival"),
        ({"minimum_requests": 1}, "disabled"),
        ({"maximum_wait_ms": 0.0}, "disabled"),
    ],
)
def test_prefill_coalesce_forced_release_conditions(overrides, reason):
    decision = _decision(**overrides)
    assert decision.release
    assert decision.reason == reason


def test_prefill_coalesce_uses_available_request_limit_as_effective_minimum():
    decision = _decision(
        ready_count=2,
        remaining_count=8,
        request_limit=2,
        ready_arrival_times=[1.5, 1.6],
    )
    assert decision.release
    assert decision.reason == "size"
    assert decision.minimum_requests == 2


def test_prefill_coalesce_rejects_inconsistent_ready_metadata():
    with pytest.raises(ValueError, match="one arrival timestamp"):
        _decision(ready_arrival_times=[])

    with pytest.raises(ValueError, match="include every ready"):
        _decision(remaining_count=0)
