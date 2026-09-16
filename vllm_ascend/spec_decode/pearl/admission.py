# SPDX-License-Identifier: Apache-2.0
"""Shared online-prefill admission decisions for SpecSLO benchmarks.

The production SpecSLO loop and the native vLLM target-only comparison use
this device-free policy so a coalesced baseline cannot silently apply a
different release rule.  Arrival timestamps stay in their original clock
domain; callers remain responsible for charging admission delay to E2E time.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PrefillCoalesceDecision:
    """One deterministic admission decision for currently ready requests."""

    release: bool
    reason: str
    minimum_requests: int
    waited_ms: float | None = None


def decide_prefill_coalesce(
    *,
    ready_count: int,
    remaining_count: int,
    active_request_count: int,
    request_limit: int,
    minimum_requests: int,
    maximum_wait_ms: float,
    now: float,
    ready_arrival_times: Sequence[float | None],
    token_budget_saturated: bool = False,
) -> PrefillCoalesceDecision:
    """Decide whether an online-prefill cohort should be released now.

    ``remaining_count`` includes the ready cohort and every future request.
    The policy never idles an empty service, never waits for an impossible
    cohort, and bounds a ready request's admission delay.  A minimum of one or
    a non-positive timeout disables coalescing and releases immediately.
    """

    integer_fields = {
        "ready_count": ready_count,
        "remaining_count": remaining_count,
        "active_request_count": active_request_count,
        "request_limit": request_limit,
        "minimum_requests": minimum_requests,
    }
    if any(not isinstance(value, int) for value in integer_fields.values()):
        raise TypeError("Prefill coalescing count fields must be integers.")
    if ready_count < 0 or remaining_count < 0 or active_request_count < 0:
        raise ValueError("Prefill coalescing request counts must be non-negative.")
    if request_limit <= 0 or minimum_requests <= 0:
        raise ValueError("Prefill coalescing limits must be positive.")
    if remaining_count < ready_count:
        raise ValueError("remaining_count must include every ready request.")
    if len(ready_arrival_times) != ready_count:
        raise ValueError("Expected one arrival timestamp per ready request.")
    if not math.isfinite(maximum_wait_ms) or maximum_wait_ms < 0:
        raise ValueError("Prefill coalescing maximum wait must be finite and non-negative.")
    if not math.isfinite(now):
        raise ValueError("Prefill coalescing current time must be finite.")

    minimum = min(minimum_requests, request_limit)
    if ready_count == 0:
        return PrefillCoalesceDecision(False, "no_ready", minimum)
    if minimum_requests <= 1 or maximum_wait_ms <= 0:
        return PrefillCoalesceDecision(True, "disabled", minimum)
    if active_request_count == 0:
        return PrefillCoalesceDecision(True, "empty_service", minimum)
    if ready_count >= minimum:
        return PrefillCoalesceDecision(True, "size", minimum)
    if token_budget_saturated:
        return PrefillCoalesceDecision(True, "token_budget", minimum)
    if remaining_count < minimum:
        return PrefillCoalesceDecision(True, "tail", minimum)
    if any(value is None for value in ready_arrival_times):
        return PrefillCoalesceDecision(True, "missing_arrival", minimum)

    arrivals = [float(value) for value in ready_arrival_times if value is not None]
    if any(not math.isfinite(value) for value in arrivals):
        raise ValueError("Prefill coalescing arrival timestamps must be finite.")
    waited_ms = max(0.0, (now - min(arrivals)) * 1000.0)
    if waited_ms >= maximum_wait_ms:
        return PrefillCoalesceDecision(True, "timeout", minimum, waited_ms)
    return PrefillCoalesceDecision(False, "defer", minimum, waited_ms)
