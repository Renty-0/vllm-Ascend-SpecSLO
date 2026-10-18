# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SLO-aware speculative budget planning for SpecRhythm.

The policy in this module is deliberately CPU-only and side-effect free during
planning.  It produces a per-request plan that can be consumed by a scheduler
or a draft proposer without changing the target model's verification semantics.
Device execution, dual-batch overlap, and KV commits are intentionally kept out
of this module so that the policy can be tested independently.
"""

from __future__ import annotations

import time
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _read_request_options(request: Any) -> dict[str, Any]:
    """Read request-level options from SamplingParams.extra_args.

    The nested form is preferred::

        extra_args={"spec_rhythm": {"slo_tpot_ms": 50}}

    Flat keys are accepted as a compatibility convenience for clients that
    already use vLLM's ``extra_args`` extension point.
    """

    sampling_params = getattr(request, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None)
    if not isinstance(extra_args, Mapping):
        return {}

    options: dict[str, Any] = {}
    nested = extra_args.get("spec_rhythm")
    if isinstance(nested, Mapping):
        options.update(nested)
    for key in (
        "slo_tpot_ms",
        "slo_class",
        "initial_gamma",
        "max_gamma",
    ):
        flat_key = f"spec_rhythm_{key}"
        if flat_key in extra_args and key not in options:
            options[key] = extra_args[flat_key]
    return options


def get_request_spec_rhythm_options(request: Any) -> dict[str, Any]:
    """Return validated-at-use request options for scheduler integrations."""

    return _read_request_options(request)


@dataclass
class SpecRhythmRequestState:
    """Scheduler-side state used by the SLO controller."""

    request_id: str
    arrival_time: float
    slo_tpot_ms: float | None = None
    slo_class: str | None = None
    per_request_gamma: int | None = None
    accepted_tokens: int = 0
    proposed_tokens: int = 0
    decode_elapsed_ms: float = 0.0
    projected_wait_ms: float = 0.0
    acceptance_ema: float = 1.0
    continuation_epoch: int = 0
    last_plan_id: int = -1
    home_batch_id: int = 0

    @classmethod
    def from_request(cls, request: Any) -> "SpecRhythmRequestState":
        options = _read_request_options(request)
        slo = options.get("slo_tpot_ms")
        if slo is not None:
            try:
                slo = float(slo)
            except (TypeError, ValueError) as exc:
                raise ValueError("spec_rhythm.slo_tpot_ms must be numeric") from exc
            if slo <= 0:
                raise ValueError("spec_rhythm.slo_tpot_ms must be greater than zero")

        per_request_gamma = options.get("initial_gamma", options.get("max_gamma"))
        if per_request_gamma is not None:
            try:
                per_request_gamma = int(per_request_gamma)
            except (TypeError, ValueError) as exc:
                raise ValueError("spec_rhythm gamma override must be an integer") from exc
            if per_request_gamma <= 0:
                raise ValueError("spec_rhythm gamma override must be greater than zero")

        return cls(
            request_id=str(request.request_id),
            arrival_time=float(getattr(request, "arrival_time", time.time())),
            slo_tpot_ms=slo,
            slo_class=(
                str(options["slo_class"])
                if options.get("slo_class") is not None
                else None
            ),
            per_request_gamma=per_request_gamma,
        )

    @property
    def observed_tpot_ms(self) -> float:
        if self.accepted_tokens <= 0:
            return 0.0
        return self.decode_elapsed_ms / self.accepted_tokens

    def record_verification(
        self,
        *,
        accepted_tokens: int,
        proposed_tokens: int,
        decode_elapsed_ms: float = 0.0,
        ema_alpha: float = 0.2,
    ) -> None:
        """Update acceptance and optional decode-time measurements."""

        accepted = max(0, int(accepted_tokens))
        proposed = max(0, int(proposed_tokens))
        self.accepted_tokens += accepted
        self.proposed_tokens += proposed
        self.decode_elapsed_ms += max(0.0, float(decode_elapsed_ms))
        if proposed:
            sample = _clamp(accepted / proposed)
            alpha = _clamp(float(ema_alpha), 1e-6, 1.0)
            self.acceptance_ema = (
                alpha * sample + (1.0 - alpha) * self.acceptance_ema
            )

    def invalidate_continuation(self) -> None:
        """Invalidate an eager continuation after target rejection."""

        self.continuation_epoch += 1


@dataclass(frozen=True)
class SpecRhythmPlan:
    """Per-request plan carried alongside a SchedulerOutput."""

    request_id: str
    plan_id: int
    gamma: int
    eager_tokens: int
    urgency: float
    progress_gap: float
    acceptance_ema: float
    slo_tpot_ms: float | None
    continuation_epoch: int
    parent_prefix_epoch: int

    @property
    def has_eager_continuation(self) -> bool:
        return self.eager_tokens > 0


@dataclass(frozen=True)
class SpecRhythmStepPlan:
    """Logical two-batch assignment for one scheduler cycle.

    ``execution_mode`` is explicitly metadata-only until the executor has a
    device-side transport and independent target/draft streams. Keeping this
    distinction in the contract prevents a scheduler-only rollout from being
    mistaken for the paper's overlapped execution pipeline.
    """

    plan_id: int
    target_home_batch_id: int
    draft_home_batch_id: int
    target_request_ids: tuple[str, ...]
    draft_request_ids: tuple[str, ...]
    eager_request_ids: tuple[str, ...]
    execution_mode: str = "metadata_only"

    @property
    def is_execution_ready(self) -> bool:
        return self.execution_mode == "dual_batch"


class SpecRhythmPolicy:
    """Deterministic SLO-aware budget controller.

    ``roofline`` is an optional mapping from ``"batch:context_bucket"`` (or
    ``"default"``) to a maximum gamma.  The table is deliberately a cap, not
    an untrusted request override, so every plan remains within the global
    speculative token limit.
    """

    def __init__(
        self,
        *,
        min_gamma: int,
        max_gamma: int,
        max_eager_tokens: int = 0,
        urgency_threshold: float = 0.75,
        acceptance_floor: float = 0.4,
        acceptance_ema_alpha: float = 0.2,
        roofline: Mapping[str, int] | None = None,
    ) -> None:
        if min_gamma < 1:
            raise ValueError("spec_rhythm min_gamma must be at least 1")
        if max_gamma < min_gamma:
            raise ValueError("spec_rhythm max_gamma must be >= min_gamma")
        if max_eager_tokens < 0:
            raise ValueError("spec_rhythm max_eager_tokens must be non-negative")
        if max_eager_tokens > max_gamma - 1:
            raise ValueError(
                "spec_rhythm max_eager_tokens must be smaller than max_gamma"
            )
        self.min_gamma = int(min_gamma)
        self.max_gamma = int(max_gamma)
        self.max_eager_tokens = int(max_eager_tokens)
        self.urgency_threshold = _clamp(float(urgency_threshold))
        self.acceptance_floor = _clamp(float(acceptance_floor))
        self.acceptance_ema_alpha = _clamp(float(acceptance_ema_alpha), 1e-6, 1.0)
        self.roofline = {str(k): int(v) for k, v in (roofline or {}).items()}
        self._states: dict[str, SpecRhythmRequestState] = {}

    @classmethod
    def from_config(cls, config: Any) -> "SpecRhythmPolicy":
        max_gamma = getattr(config, "spec_rhythm_max_gamma", None)
        if max_gamma is None:
            max_gamma = getattr(config, "num_speculative_tokens")
        return cls(
            min_gamma=int(getattr(config, "spec_rhythm_min_gamma", 1)),
            max_gamma=int(max_gamma),
            max_eager_tokens=int(
                getattr(config, "spec_rhythm_max_eager_tokens", 0)
            ),
            urgency_threshold=float(
                getattr(config, "spec_rhythm_urgency_threshold", 0.75)
            ),
            acceptance_floor=float(
                getattr(config, "spec_rhythm_acceptance_floor", 0.4)
            ),
            acceptance_ema_alpha=float(
                getattr(config, "spec_rhythm_acceptance_ema_alpha", 0.2)
            ),
            roofline=getattr(config, "spec_rhythm_roofline", None),
        )

    def register_request(self, request: Any) -> SpecRhythmRequestState:
        request_id = str(request.request_id)
        state = self._states.get(request_id)
        if state is None:
            state = SpecRhythmRequestState.from_request(request)
            state.home_batch_id = len(self._states) % 2
            self._states[request_id] = state
        return state

    def state_for(self, request: Any) -> SpecRhythmRequestState:
        return self.register_request(request)

    def forget(self, request_id: str) -> None:
        self._states.pop(str(request_id), None)

    def record_verification(
        self,
        request_id: str,
        *,
        accepted_tokens: int,
        proposed_tokens: int,
        decode_elapsed_ms: float = 0.0,
    ) -> None:
        state = self._states.get(str(request_id))
        if state is None:
            return
        state.record_verification(
            accepted_tokens=accepted_tokens,
            proposed_tokens=proposed_tokens,
            decode_elapsed_ms=decode_elapsed_ms,
            ema_alpha=self.acceptance_ema_alpha,
        )

    def _roofline_cap(self, batch_size: int, context_len: int) -> int:
        context_bucket = max(1, math.ceil(max(1, int(context_len)) / 512))
        keys = (
            f"{int(batch_size)}:{context_bucket}",
            f"batch:{int(batch_size)}",
            "default",
        )
        cap = self.max_gamma
        for key in keys:
            if key in self.roofline:
                cap = min(cap, int(self.roofline[key]))
                break
        return max(self.min_gamma, min(self.max_gamma, cap))

    def plan_for(
        self,
        request: Any,
        *,
        batch_size: int,
        context_len: int,
        plan_id: int,
        now: float | None = None,
    ) -> SpecRhythmPlan:
        state = self.state_for(request)
        now = time.time() if now is None else float(now)
        cap = self._roofline_cap(batch_size, context_len)
        if state.per_request_gamma is not None:
            cap = min(cap, state.per_request_gamma)
        cap = max(self.min_gamma, cap)

        if state.slo_tpot_ms is None:
            gamma = cap
            urgency = 0.0
            progress_gap = 0.0
            eager_tokens = 0
        else:
            slo = state.slo_tpot_ms
            queue_wait_ms = max(0.0, (now - state.arrival_time) * 1000.0)
            observed_tpot = state.observed_tpot_ms
            urgency = _clamp(
                max(queue_wait_ms, state.projected_wait_ms, observed_tpot) / slo
            )
            progress_gap = _clamp(
                max(0.0, queue_wait_ms - state.decode_elapsed_ms) / slo
            )
            confidence = _clamp(state.acceptance_ema)
            score = 0.55 * urgency + 0.25 * progress_gap + 0.20 * confidence
            if confidence < self.acceptance_floor:
                score *= confidence / max(self.acceptance_floor, 1e-6)
            gamma = int(round(self.min_gamma + min(1.0, score) * (cap - self.min_gamma)))
            gamma = max(self.min_gamma, min(cap, gamma))
            eager_tokens = (
                min(self.max_eager_tokens, max(0, gamma - 1))
                if urgency >= self.urgency_threshold
                and confidence >= self.acceptance_floor
                else 0
            )

        state.last_plan_id = int(plan_id)
        return SpecRhythmPlan(
            request_id=str(request.request_id),
            plan_id=int(plan_id),
            gamma=int(gamma),
            eager_tokens=int(eager_tokens),
            urgency=float(urgency),
            progress_gap=float(progress_gap),
            acceptance_ema=float(state.acceptance_ema),
            slo_tpot_ms=state.slo_tpot_ms,
            continuation_epoch=state.continuation_epoch,
            parent_prefix_epoch=state.continuation_epoch,
        )

    def plan_batch(
        self,
        requests: list[Any],
        *,
        plan_id: int,
        now: float | None = None,
    ) -> dict[str, SpecRhythmPlan]:
        batch_size = max(1, len(requests))
        plans: dict[str, SpecRhythmPlan] = {}
        for request in requests:
            context_len = int(getattr(request, "num_tokens", 0))
            plan = self.plan_for(
                request,
                batch_size=batch_size,
                context_len=context_len,
                plan_id=plan_id,
                now=now,
            )
            plans[plan.request_id] = plan
        return plans

    def build_step_plan(
        self,
        requests: list[Any],
        plans: Mapping[str, SpecRhythmPlan],
        *,
        plan_id: int,
    ) -> SpecRhythmStepPlan:
        target_home_batch_id = int(plan_id) % 2
        draft_home_batch_id = 1 - target_home_batch_id
        target_request_ids: list[str] = []
        draft_request_ids: list[str] = []
        eager_request_ids: list[str] = []
        for request in requests:
            state = self.state_for(request)
            request_id = str(request.request_id)
            if state.home_batch_id == target_home_batch_id:
                target_request_ids.append(request_id)
            else:
                draft_request_ids.append(request_id)
            plan = plans.get(request_id)
            if plan is not None and plan.has_eager_continuation:
                eager_request_ids.append(request_id)
        return SpecRhythmStepPlan(
            plan_id=int(plan_id),
            target_home_batch_id=target_home_batch_id,
            draft_home_batch_id=draft_home_batch_id,
            target_request_ids=tuple(target_request_ids),
            draft_request_ids=tuple(draft_request_ids),
            eager_request_ids=tuple(eager_request_ids),
        )


__all__ = [
    "get_request_spec_rhythm_options",
    "SpecRhythmPlan",
    "SpecRhythmPolicy",
    "SpecRhythmRequestState",
    "SpecRhythmStepPlan",
]
