# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SpecRhythm control plane and guarded two-batch mailbox state.

The paper separates the mechanism that creates a draft window from the policy
that allocates it.  This module follows that boundary: it contains no model or
distributed calls, and owns only deterministic request accounting, budget
shaping, and proposal lifecycle transitions.  The native PEARL engine consumes
the resulting plans and keeps token/KV mutation on the owning model workers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum

from vllm_ascend.spec_decode.pearl.roofline import ProfiledRoofline


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass
class SpecRhythmRuntimeState:
    """Online state used by the paper's urgency and acceptance policy."""

    request_index: int
    home_batch_id: int
    slo_tpot_ms: float | None = None
    slo_class: str | None = None
    max_gamma: int | None = None
    # ``N`` in the paper is the number of tokens delivered before the first
    # verification round.  The denominator is guarded with ``max(1, N)``
    # where it is used, so a fresh request must start at zero rather than one.
    delivered_tokens: int = 0
    decode_elapsed_ms: float = 0.0
    # Elapsed decode time at which the first *measured* token was committed.
    # The initial target token is produced by prefill and is intentionally not
    # part of TPOT, matching vLLM's first-token/last-token decode interval.
    decode_start_elapsed_ms: float | None = None
    # Arrival delay is scheduling debt, not decoding time.  Keeping it
    # separate prevents TPOT from charging a request for time before it was
    # admitted to the decode stage while still allowing urgency to catch up.
    arrival_wait_ms: float = 0.0
    acceptance_ema: float = 1.0
    # Probability that the *entire* proposal window is accepted.  Rolling
    # Eager continuations are useful only in that event, so the per-token
    # acceptance ratio above is an optimistic score for this decision.
    full_acceptance_ema: float = 1.0
    draft_confidence_ema: float = 1.0
    prefix_epoch: int = 0
    verification_rounds: int = 0

    def __post_init__(self) -> None:
        if self.request_index < 0:
            raise ValueError("SpecRhythm request indices must be non-negative.")
        if self.home_batch_id not in (0, 1):
            raise ValueError("SpecRhythm home_batch_id must be zero or one.")
        if self.slo_tpot_ms is not None and self.slo_tpot_ms <= 0:
            raise ValueError("SpecRhythm TPOT SLO must be positive when supplied.")
        if self.max_gamma is not None and self.max_gamma <= 0:
            raise ValueError("SpecRhythm per-request max gamma must be positive.")
        if (
            self.delivered_tokens < 0
            or self.decode_elapsed_ms < 0
            or self.arrival_wait_ms < 0
            or (self.decode_start_elapsed_ms is not None and self.decode_start_elapsed_ms < 0)
        ):
            raise ValueError("SpecRhythm progress counters must be non-negative.")
        self.acceptance_ema = _clamp(float(self.acceptance_ema))
        self.full_acceptance_ema = _clamp(float(self.full_acceptance_ema))
        self.draft_confidence_ema = _clamp(float(self.draft_confidence_ema))

    @property
    def expected_acceptance_benefit(self) -> float:
        return self.acceptance_ema * self.draft_confidence_ema

    @property
    def expected_continuation_benefit(self) -> float:
        """Expected value of a dependency-exact Rolling Eager continuation."""

        return self.full_acceptance_ema * self.draft_confidence_ema

    @property
    def observed_tpot_ms(self) -> float:
        return self.decode_elapsed_ms / max(1, self.delivered_tokens)

    @property
    def effective_elapsed_ms(self) -> float:
        """Elapsed time used by the scheduler, including admission debt."""
        return self.decode_elapsed_ms + self.arrival_wait_ms

    def projected_progress_gap(self, projected_wait_ms: float) -> int:
        """Return ``a_need`` from section 4.3 of the SpecRhythm paper."""

        if self.slo_tpot_ms is None:
            return 0
        projected_tokens = (self.effective_elapsed_ms + max(0.0, float(projected_wait_ms))) / self.slo_tpot_ms
        return int(math.ceil(max(0.0, projected_tokens - self.delivered_tokens)))

    def urgency(self, projected_wait_ms: float) -> float:
        if self.slo_tpot_ms is None:
            return 0.0
        projected_tpot = (self.effective_elapsed_ms + max(0.0, float(projected_wait_ms))) / max(
            1, self.delivered_tokens
        )
        return max(0.0, projected_tpot / self.slo_tpot_ms)

    def add_decode_time(self, elapsed_ms: float) -> None:
        self.decode_elapsed_ms += max(0.0, float(elapsed_ms))

    def record_verification(
        self,
        *,
        proposed_tokens: int,
        accepted_tokens: int,
        delivered_tokens: int,
        draft_confidence: float | None,
        ema_alpha: float,
    ) -> None:
        proposed = max(0, int(proposed_tokens))
        accepted = max(0, min(int(accepted_tokens), proposed))
        delivered = max(0, int(delivered_tokens))
        alpha = _clamp(float(ema_alpha), 1e-6, 1.0)
        if proposed:
            sample = accepted / proposed
            self.acceptance_ema = alpha * sample + (1.0 - alpha) * self.acceptance_ema
            full_sample = float(accepted == proposed)
            self.full_acceptance_ema = alpha * full_sample + (1.0 - alpha) * self.full_acceptance_ema
        if draft_confidence is not None:
            confidence = _clamp(float(draft_confidence))
            self.draft_confidence_ema = alpha * confidence + (1.0 - alpha) * self.draft_confidence_ema
        self.delivered_tokens += delivered
        self.verification_rounds += 1
        self.prefix_epoch += 1


@dataclass(frozen=True)
class SpecRhythmBudgetPlan:
    """Per-window allocation constrained by draft work and target rooflines."""

    plan_id: int
    normal_budgets: Mapping[int, int]
    eager_budgets: Mapping[int, int]
    progress_gaps: Mapping[int, int]
    eager_priorities: Mapping[int, float]
    verification_roof: int
    draft_token_budget: int
    allocated_draft_tokens: int
    deferred_normal_request_indices: tuple[int, ...] = ()

    @property
    def unused_verification_tokens(self) -> int:
        """Candidate slots left in a fixed-B envelope after allocation."""
        return max(0, int(self.verification_roof) - int(self.allocated_draft_tokens))

    @property
    def verification_input_tokens(self) -> int:
        """The target-side envelope size, independent of nominal gamma."""
        return int(self.verification_roof)

    def __post_init__(self) -> None:
        normal = dict(self.normal_budgets)
        eager = dict(self.eager_budgets)
        if set(normal).intersection(eager):
            raise ValueError("Normal and eager SpecRhythm draft sets must be disjoint.")
        if set(self.deferred_normal_request_indices).intersection(normal):
            raise ValueError("Deferred normal requests cannot also receive a draft budget.")
        if any(value <= 0 for value in (*normal.values(), *eager.values())):
            raise ValueError("Every allocated SpecRhythm proposal must contain a token.")
        normal_tokens = sum(normal.values())
        eager_tokens = sum(eager.values())
        if normal_tokens > self.verification_roof:
            raise ValueError("Normal proposals exceed the target verification roofline.")
        if eager_tokens > self.verification_roof:
            raise ValueError("Eager proposals exceed the target verification roofline.")
        if normal_tokens + eager_tokens > self.verification_roof:
            raise ValueError("Normal and eager proposals exceed the global target verification roofline.")
        if self.allocated_draft_tokens != sum(normal.values()) + sum(eager.values()):
            raise ValueError("SpecRhythm allocated draft-token accounting is inconsistent.")
        if self.allocated_draft_tokens > self.draft_token_budget:
            raise ValueError("SpecRhythm proposals exceed the available draft window.")


class SpecRhythmBudgetShaper:
    """Two-stage SLO-aware candidate allocator from paper section 4.4."""

    def __init__(
        self,
        *,
        min_gamma: int,
        max_gamma: int,
        verification_budget: int | None = None,
        acceptance_floor: float = 0.4,
        acceptance_ema_alpha: float = 0.2,
        roofline: Mapping[str, int] | None = None,
    ) -> None:
        if min_gamma <= 0 or max_gamma < min_gamma:
            raise ValueError("SpecRhythm requires 1 <= min_gamma <= max_gamma.")
        self.min_gamma = int(min_gamma)
        self.max_gamma = int(max_gamma)
        if verification_budget is not None and verification_budget <= 0:
            raise ValueError("SpecRhythm verification budget must be positive.")
        self.verification_budget = int(verification_budget) if verification_budget is not None else None
        self.acceptance_floor = _clamp(float(acceptance_floor))
        self.acceptance_ema_alpha = _clamp(float(acceptance_ema_alpha), 1e-6, 1.0)
        self.roofline = (
            roofline
            if isinstance(roofline, ProfiledRoofline)
            else {str(key): int(value) for key, value in (roofline or {}).items()}
        )
        if isinstance(self.roofline, ProfiledRoofline) and self.verification_budget is not None:
            raise ValueError("A strict measured roofline cannot be overridden by a fixed verification budget")
        if any(value < 0 for value in self.roofline.values()):
            raise ValueError("SpecRhythm roofline entries must be non-negative token budgets.")
        self._normal_wait_cycles: dict[int, int] = {}

    def verification_roof(self, batch_size: int, context_len: int) -> int:
        """Look up the profiled batch-level candidate-token roofline."""

        if self.verification_budget is not None:
            return self.verification_budget

        if isinstance(self.roofline, ProfiledRoofline):
            return self.roofline.lookup(batch_size, context_len)

        batch_size = max(1, int(batch_size))
        context_bucket = max(1, math.ceil(max(1, int(context_len)) / 512))
        for key in (
            f"{batch_size}:{context_bucket}",
            f"batch:{batch_size}",
            "default",
        ):
            if key in self.roofline:
                return self.roofline[key]
        return batch_size * self.max_gamma

    def shape(
        self,
        *,
        plan_id: int,
        normal_request_indices: Sequence[int],
        eager_request_indices: Sequence[int],
        states: Mapping[int, SpecRhythmRuntimeState],
        projected_wait_ms: float,
        context_len: int,
        draft_token_budget: int | None = None,
        batch_size: int | None = None,
        eager_token_cap: int | None = None,
        eager_reserve_tokens: int = 0,
        draft_window_ms: float | None = None,
        draft_ms_per_token: float | None = None,
        verification_roof: int | None = None,
    ) -> SpecRhythmBudgetPlan:
        normal = tuple(dict.fromkeys(int(index) for index in normal_request_indices))
        eager = tuple(dict.fromkeys(int(index) for index in eager_request_indices))
        if set(normal).intersection(eager):
            raise ValueError("A request cannot be both a normal and eager draft in one step.")
        requested = (*normal, *eager)
        if any(index not in states for index in requested):
            raise ValueError("SpecRhythm cannot shape an unregistered request.")

        if verification_roof is None:
            roof = self.verification_roof(
                max(len(requested), 1) if batch_size is None else batch_size,
                context_len,
            )
        else:
            roof = int(verification_roof)
            if roof < 0:
                raise ValueError("SpecRhythm execution-plan verification roof must be non-negative.")
        if eager_token_cap is not None and eager_token_cap < 0:
            raise ValueError("SpecRhythm eager-token cap must be non-negative.")
        if eager_reserve_tokens < 0:
            raise ValueError("SpecRhythm eager-token reserve must be non-negative.")
        if (draft_window_ms is None) != (draft_ms_per_token is None):
            raise ValueError("A measured draft window requires a measured per-token draft cost.")
        if draft_window_ms is not None and (
            not math.isfinite(draft_window_ms)
            or draft_window_ms < 0.0
            or not math.isfinite(draft_ms_per_token)
            or draft_ms_per_token <= 0.0
        ):
            raise ValueError("Draft window must be finite/non-negative and token cost positive.")
        available_draft = 2 * roof if draft_token_budget is None else max(0, int(draft_token_budget))
        hidden_tokens: int | None = None
        if draft_window_ms is not None:
            assert draft_ms_per_token is not None
            hidden_tokens = int(draft_window_ms / draft_ms_per_token)
        normal_caps = {index: min(self.max_gamma, states[index].max_gamma or self.max_gamma) for index in normal}
        minimum_budgets = {index: min(self.min_gamma, normal_caps[index]) for index in normal}
        if roof == 0:
            for index in normal:
                self._normal_wait_cycles[index] = self._normal_wait_cycles.get(index, 0) + 1
            gaps = {index: states[index].projected_progress_gap(projected_wait_ms) for index in requested}
            return SpecRhythmBudgetPlan(
                plan_id=int(plan_id),
                normal_budgets={},
                eager_budgets={},
                progress_gaps=gaps,
                eager_priorities={index: gaps[index] * states[index].expected_acceptance_benefit for index in eager},
                verification_roof=0,
                draft_token_budget=0,
                allocated_draft_tokens=0,
                deferred_normal_request_indices=normal,
            )
        if any(value > roof for value in minimum_budgets.values()):
            raise ValueError("The SpecRhythm roofline cannot fit the minimum normal proposal budget.")

        # A fixed-gamma normal home can otherwise consume all of B before the
        # rolling-eager stage is considered.  The reserve is an explicit
        # *upper bound*, not a quota: only requests that already passed the
        # caller's W admission and still have positive a_need/acceptance value
        # receive a complete minimum proposal.  If none qualifies, no budget
        # is withheld from normal progress.
        initial_gaps = {index: states[index].projected_progress_gap(projected_wait_ms) for index in requested}
        eager_budgets: dict[int, int] = {}
        reserve_limit = min(int(eager_reserve_tokens), roof, available_draft)
        if hidden_tokens is not None:
            reserve_limit = min(reserve_limit, hidden_tokens)
        reserve_remaining = reserve_limit
        for index in sorted(
            eager,
            key=lambda index: (
                initial_gaps[index] * states[index].expected_acceptance_benefit,
                states[index].urgency(projected_wait_ms),
                -index,
            ),
            reverse=True,
        ):
            state = states[index]
            cap = min(self.max_gamma, state.max_gamma or self.max_gamma)
            if eager_token_cap is not None:
                cap = min(cap, eager_token_cap)
            minimum = min(self.min_gamma, cap)
            if (
                minimum <= 0
                or initial_gaps[index] <= 0
                or state.expected_acceptance_benefit < self.acceptance_floor
                or minimum > reserve_remaining
            ):
                continue
            eager_budgets[index] = minimum
            reserve_remaining -= minimum
        reserved_eager_tokens = sum(eager_budgets.values())

        # A fixed B can be smaller than the number of ready-to-draft requests.
        # Admit a bounded subset instead of failing the whole service cycle.
        # Retain age for deferred rows so a tight-SLO stream cannot permanently
        # exclude a relaxed row before it ever acquires a ready proposal.
        normal_candidates = normal
        normal_admission_budget = min(
            roof - reserved_eager_tokens,
            available_draft - reserved_eager_tokens,
        )
        if hidden_tokens is not None and reserved_eager_tokens:
            # Reserved work is optional and therefore must fit W.  Normal work
            # may retain the legacy mandatory-overflow behavior only when no
            # eager reservation was actually made.
            normal_admission_budget = min(
                normal_admission_budget,
                hidden_tokens - reserved_eager_tokens,
            )
        normal_admission_budget = max(0, normal_admission_budget)
        if sum(minimum_budgets.values()) > normal_admission_budget:
            remaining_minimum = normal_admission_budget
            admitted = []
            for index in sorted(
                normal,
                key=lambda index: (
                    self._normal_wait_cycles.get(index, 0),
                    states[index].urgency(projected_wait_ms),
                    -index,
                ),
                reverse=True,
            ):
                if minimum_budgets[index] <= remaining_minimum:
                    admitted.append(index)
                    remaining_minimum -= minimum_budgets[index]
            admitted_set = set(admitted)
            normal = tuple(index for index in normal if index in admitted_set)
        deferred_normal = tuple(index for index in normal_candidates if index not in normal)
        for index in normal:
            self._normal_wait_cycles.pop(index, None)
        for index in deferred_normal:
            self._normal_wait_cycles[index] = self._normal_wait_cycles.get(index, 0) + 1
        for index in set(self._normal_wait_cycles) - set(states):
            self._normal_wait_cycles.pop(index)
        requested = (*normal, *eager)
        normal_budgets = {index: minimum_budgets[index] for index in normal}
        minimum_needed = sum(normal_budgets.values())
        if draft_window_ms is not None:
            assert hidden_tokens is not None
            # Normal minimum progress is mandatory; only its predicted
            # overflow is exposed. Optional eager/deeper work shares the
            # measured residual W and cannot extend that mandatory overflow.
            available_draft = min(available_draft, max(minimum_needed, hidden_tokens))
        remaining_draft = available_draft - minimum_needed - reserved_eager_tokens
        # The roofline is a target-side *global* candidate budget.  Normal and
        # eager proposals share it because both are verified in the same target
        # step.  Keeping one counter also makes the invariant explicit for
        # future tree-shaped allocations.
        remaining_roof = roof - minimum_needed - reserved_eager_tokens
        gaps = {index: initial_gaps[index] for index in requested}
        priorities = {index: gaps[index] * states[index].expected_acceptance_benefit for index in eager}

        # Stage 1: close projected progress gaps. Prefix depth is allocated one
        # token at a time, so no child candidate can exist without its parent.
        urgent = sorted(
            requested,
            key=lambda index: (
                gaps[index] * states[index].expected_acceptance_benefit,
                states[index].urgency(projected_wait_ms),
                -index,
            ),
            reverse=True,
        )
        for index in urgent:
            state = states[index]
            if gaps[index] <= 0 or state.expected_acceptance_benefit < self.acceptance_floor:
                continue
            cap = min(self.max_gamma, state.max_gamma or self.max_gamma)
            if index in normal_budgets:
                wanted = max(0, min(cap, gaps[index]) - normal_budgets[index])
                grant = min(wanted, remaining_roof, remaining_draft)
                normal_budgets[index] += grant
                remaining_roof -= grant
            else:
                eager_cap = cap if eager_token_cap is None else min(cap, eager_token_cap)
                current = eager_budgets.get(index, 0)
                wanted = max(
                    0,
                    min(eager_cap, max(self.min_gamma, gaps[index])) - current,
                )
                grant = min(wanted, remaining_roof, remaining_draft)
                if grant > 0:
                    eager_budgets[index] = current + grant
                    remaining_roof -= grant
            remaining_draft -= grant
            if remaining_draft <= 0:
                break

        # Stage 2: spend residual capacity on the highest marginal expected
        # progress. Deeper tokens decay by the measured acceptance benefit.
        while remaining_draft > 0:
            candidates: list[tuple[float, int, str]] = []
            for kind, budgets, indices in (
                ("normal", normal_budgets, normal),
                ("eager", eager_budgets, eager),
            ):
                if remaining_roof <= 0:
                    continue
                for index in indices:
                    state = states[index]
                    current = budgets.get(index, 0)
                    cap = min(self.max_gamma, state.max_gamma or self.max_gamma)
                    if kind == "eager" and eager_token_cap is not None:
                        cap = min(cap, eager_token_cap)
                    if current >= cap:
                        continue
                    benefit = state.expected_acceptance_benefit
                    if benefit <= 0.0:
                        continue
                    # Section 5.3 stops *new* eager work once a_need closes.
                    # Residual Goodput allocation must not reopen a request
                    # skipped by the urgent stage merely to consume spare B.
                    # Normal drafts still use the residual capacity below.
                    if kind == "eager" and (gaps[index] <= 0 or benefit < self.acceptance_floor):
                        continue
                    urgency_weight = 1.0 + state.urgency(projected_wait_ms)
                    marginal = urgency_weight * (benefit ** (current + 1))
                    candidates.append((marginal, -index, kind))
            if not candidates:
                break
            _, negative_index, kind = max(candidates)
            index = -negative_index
            if kind == "normal":
                normal_budgets[index] += 1
            else:
                eager_budgets[index] = eager_budgets.get(index, 0) + 1
            remaining_roof -= 1
            remaining_draft -= 1

        # B is an upper bound, not a requirement to fill otherwise unhelpful
        # candidates. In particular, do not bypass the eager acceptance gate
        # merely because the roof was supplied explicitly.
        allocated = sum(normal_budgets.values()) + sum(eager_budgets.values())
        return SpecRhythmBudgetPlan(
            plan_id=int(plan_id),
            normal_budgets=normal_budgets,
            eager_budgets=eager_budgets,
            progress_gaps=gaps,
            eager_priorities=priorities,
            verification_roof=roof,
            draft_token_budget=available_draft,
            allocated_draft_tokens=allocated,
            deferred_normal_request_indices=deferred_normal,
        )


class ProposalLifecycle(str, Enum):
    AVAILABLE = "available"
    STAGED_EAGER = "staged_eager"
    CONSUMED = "consumed"
    INVALIDATED = "invalidated"


@dataclass
class SpecRhythmProposalTicket:
    """Guard metadata carried with a device-resident proposal payload."""

    proposal_id: int
    request_index: int
    home_batch_id: int
    gamma: int
    required_prefix_epoch: int
    eager: bool = False
    lifecycle: ProposalLifecycle = ProposalLifecycle.AVAILABLE

    def __post_init__(self) -> None:
        if self.proposal_id < 0 or self.request_index < 0 or self.gamma <= 0:
            raise ValueError("SpecRhythm proposal identifiers and gamma must be positive.")
        if self.home_batch_id not in (0, 1) or self.required_prefix_epoch < 0:
            raise ValueError("SpecRhythm proposal routing metadata is invalid.")


class PipelinePhase(str, Enum):
    WARMUP = "warmup"
    STEADY = "steady"
    DRAIN = "drain"
    COMPLETE = "complete"


@dataclass(frozen=True)
class SpecRhythmExecutionPlan:
    plan_id: int
    phase: PipelinePhase
    target_home_batch_id: int | None
    draft_home_batch_id: int | None
    target_request_indices: tuple[int, ...]
    normal_draft_request_indices: tuple[int, ...]
    eager_candidate_indices: tuple[int, ...]
    # These counts describe this cycle's *existing* ready proposals. They are
    # distinct from the draft-window allocation, which may produce proposals
    # consumed in different future cycles.
    target_candidate_budgets: Mapping[int, int] = field(default_factory=dict)
    verification_candidate_budget: int | None = None
    deferred_target_request_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.target_candidate_budgets:
            if set(self.target_candidate_budgets) != set(self.target_request_indices):
                raise ValueError("SpecRhythm target candidate counts do not match the plan.")
            if any(value <= 0 for value in self.target_candidate_budgets.values()):
                raise ValueError("SpecRhythm target proposals must contain candidates.")
        if self.verification_candidate_budget is not None:
            if self.verification_candidate_budget <= 0:
                raise ValueError("SpecRhythm verification candidate budget must be positive.")
            if set(self.target_candidate_budgets) != set(self.target_request_indices):
                raise ValueError("A bounded SpecRhythm target plan needs every candidate count.")
            if sum(self.target_candidate_budgets.values()) > self.verification_candidate_budget:
                raise ValueError("SpecRhythm ready proposals exceed the verification budget.")


@dataclass
class SpecRhythmPipelineController:
    """Own two alternating logical batches and guarded proposal lifecycles."""

    request_states: dict[int, SpecRhythmRuntimeState]
    next_target_home_batch_id: int = 0
    plan_id: int = 0
    _next_proposal_id: int = 0
    ready: dict[int, SpecRhythmProposalTicket] = field(default_factory=dict)
    staged_eager: dict[int, SpecRhythmProposalTicket] = field(default_factory=dict)
    _priority_home_id: int | None = None
    _priority_streak: int = 0
    _ready_wait_cycles: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.next_target_home_batch_id not in (0, 1):
            raise ValueError("SpecRhythm next target batch must be zero or one.")

    def build_plan(
        self,
        active_request_indices: Sequence[int],
        *,
        priority: bool = False,
        projected_wait_ms: float = 0.0,
        priority_burst: int = 2,
        merge_ready_homes: bool = False,
        max_target_requests: int | None = None,
        verification_budget: int | None = None,
        ready_candidate_counts: Mapping[int, int] | None = None,
    ) -> SpecRhythmExecutionPlan:
        active = tuple(dict.fromkeys(int(index) for index in active_request_indices))
        if any(index not in self.request_states for index in active):
            raise ValueError("SpecRhythm cannot schedule an unregistered request.")
        if priority_burst <= 0:
            raise ValueError("SpecRhythm priority burst must be positive.")
        if max_target_requests is not None and max_target_requests <= 0:
            raise ValueError("SpecRhythm target request limit must be positive.")
        if verification_budget is not None and verification_budget <= 0:
            raise ValueError("SpecRhythm verification candidate budget must be positive.")
        active_set = set(active)
        self._discard_inactive_payloads(active_set)
        ready_homes = {self.request_states[index].home_batch_id for index in active if index in self.ready}
        # A promoted rolling-eager ticket belongs to the home that was just
        # verified, but paper section 4.3 requires it to join the *next*
        # cycle's opposite-home verification.  It must not take ownership of
        # that cycle or pull every old ready row from its home along with it.
        normal_ready_homes = {
            self.request_states[index].home_batch_id
            for index in active
            if index in self.ready and not self.ready[index].eager
        }
        base_ready_homes = normal_ready_homes or ready_homes
        target_home: int | None
        if self.next_target_home_batch_id in base_ready_homes:
            target_home = self.next_target_home_batch_id
        elif base_ready_homes:
            target_home = min(base_ready_homes)
        else:
            target_home = None
        if priority and len(base_ready_homes) > 1:
            urgency_by_home = {
                home: max(
                    self.request_states[index].urgency(projected_wait_ms)
                    for index in active
                    if self.request_states[index].home_batch_id == home
                    and index in self.ready
                    and (not normal_ready_homes or not self.ready[index].eager)
                )
                for home in base_ready_homes
            }
            urgent_home = max(
                urgency_by_home,
                key=lambda home: (urgency_by_home[home], -home),
            )
            if urgency_by_home[urgent_home] >= 1.0:
                if self._priority_home_id == urgent_home and self._priority_streak >= priority_burst:
                    alternatives = [home for home in base_ready_homes if home != urgent_home]
                    target_home = min(alternatives) if alternatives else urgent_home
                else:
                    target_home = urgent_home
                if target_home == self._priority_home_id:
                    self._priority_streak += 1
                else:
                    self._priority_home_id = target_home
                    self._priority_streak = 1
            else:
                self._priority_home_id = None
                self._priority_streak = 0
        # The paper's hidden draft window can leave both logical homes with a
        # ready proposal.  In that state, verifying both homes together avoids
        # charging each request for an otherwise idle alternating round.  The
        # legacy PEARL rotation remains the default for unconstrained traffic.
        target_homes = (
            frozenset(ready_homes)
            if merge_ready_homes and len(ready_homes) > 1
            else frozenset((target_home,))
            if target_home is not None
            else frozenset()
        )
        draft_home = 1 - target_home if target_home is not None else self.next_target_home_batch_id
        if target_home is None and not any(self.request_states[index].home_batch_id == draft_home for index in active):
            draft_home = 1 - draft_home
        target_candidates = tuple(
            index
            for index in active
            if index in self.ready
            and (
                self.request_states[index].home_batch_id in target_homes
                or (not merge_ready_homes and target_home is not None and self.ready[index].eager)
            )
        )
        candidate_counts = {
            index: int(self.ready[index].gamma if ready_candidate_counts is None else ready_candidate_counts[index])
            for index in target_candidates
        }
        if any(value <= 0 for value in candidate_counts.values()):
            raise ValueError("SpecRhythm ready proposals must contain candidates.")
        if verification_budget is not None and any(value > verification_budget for value in candidate_counts.values()):
            # Never silently truncate a tree or an eager dependency path.
            # A caller changing its roof must rebuild/prune that payload first.
            raise ValueError(
                "A ready SpecRhythm proposal exceeds the verification budget; "
                "rebuild or ancestor-safely prune it before scheduling."
            )
        constrained = (max_target_requests is not None and len(target_candidates) > max_target_requests) or (
            verification_budget is not None and sum(candidate_counts.values()) > verification_budget
        )
        if constrained:
            # Pack whole proposals so deferral cannot invalidate an eager
            # continuation's exact parent path.  For unconstrained traffic,
            # age wins after deferral to prevent starvation.  Under an SLO
            # policy, section 4.3's a_need must be the primary ordering key:
            # putting age first turns fixed-B scheduling into round-robin and
            # prevents tight requests from catching up.  Relaxed requests are
            # still starvation-safe because their a_need grows while waiting.
            def target_priority(index: int):
                age = self._ready_wait_cycles.get(index, 0)
                urgency = self.request_states[index].urgency(projected_wait_ms)
                if priority:
                    return (
                        self.request_states[index].projected_progress_gap(projected_wait_ms),
                        urgency,
                        age,
                        self.request_states[index].home_batch_id == self.next_target_home_batch_id,
                        -index,
                    )
                return (
                    age,
                    urgency,
                    self.request_states[index].home_batch_id == self.next_target_home_batch_id,
                    -index,
                )

            ordered = sorted(
                target_candidates,
                key=target_priority,
                reverse=True,
            )
            selected: list[int] = []
            selected_tokens = 0
            for index in ordered:
                if max_target_requests is not None and len(selected) >= max_target_requests:
                    break
                if verification_budget is not None and selected_tokens + candidate_counts[index] > verification_budget:
                    continue
                selected.append(index)
                selected_tokens += candidate_counts[index]
            target = tuple(selected)
        else:
            target = target_candidates
        target_set = set(target)
        deferred = tuple(index for index in target_candidates if index not in target_set)
        for index in active:
            if index in self.ready:
                self._ready_wait_cycles[index] = 0 if index in target_set else self._ready_wait_cycles.get(index, 0) + 1
        normal = tuple(
            index
            for index in active
            if self.request_states[index].home_batch_id == draft_home
            and index not in self.ready
            and index not in self.staged_eager
        )
        eager = tuple(index for index in target if index not in self.staged_eager)
        if target:
            phase = PipelinePhase.STEADY
        elif normal:
            phase = PipelinePhase.WARMUP
        elif self.ready:
            phase = PipelinePhase.DRAIN
        else:
            phase = PipelinePhase.COMPLETE
        plan = SpecRhythmExecutionPlan(
            plan_id=self.plan_id,
            phase=phase,
            target_home_batch_id=target_home,
            draft_home_batch_id=draft_home if normal or eager else None,
            target_request_indices=target,
            normal_draft_request_indices=normal,
            eager_candidate_indices=eager,
            target_candidate_budgets={index: candidate_counts[index] for index in target},
            verification_candidate_budget=verification_budget,
            deferred_target_request_indices=deferred,
        )
        self.plan_id += 1
        return plan

    def build_single_batch_plan(
        self,
        active_request_indices: Sequence[int],
        *,
        overlap: bool,
        max_target_requests: int | None = None,
        verification_budget: int | None = None,
        ready_candidate_counts: Mapping[int, int] | None = None,
    ) -> SpecRhythmExecutionPlan:
        """Build a one-logical-batch serial or PEARL-overlap plan.

        This is an ablation companion to :meth:`build_plan`, not another SLO
        policy.  ``overlap=False`` emits mutually exclusive draft and target
        cycles.  ``overlap=True`` lets the draft group build dependency-exact
        next-window proposals while the target group verifies the current
        windows for the same logical batch.  Whole proposals are always
        selected, so a target row is never truncated merely to fit ``B``.
        """

        active = tuple(dict.fromkeys(int(index) for index in active_request_indices))
        if any(index not in self.request_states for index in active):
            raise ValueError("SpecRhythm cannot schedule an unregistered request.")
        if max_target_requests is not None and max_target_requests <= 0:
            raise ValueError("SpecRhythm target request limit must be positive.")
        if verification_budget is not None and verification_budget <= 0:
            raise ValueError("SpecRhythm verification candidate budget must be positive.")
        active_set = set(active)
        self._discard_inactive_payloads(active_set)

        target_candidates = tuple(index for index in active if index in self.ready)
        candidate_counts = {
            index: int(self.ready[index].gamma if ready_candidate_counts is None else ready_candidate_counts[index])
            for index in target_candidates
        }
        if any(value <= 0 for value in candidate_counts.values()):
            raise ValueError("SpecRhythm ready proposals must contain candidates.")
        if verification_budget is not None and any(value > verification_budget for value in candidate_counts.values()):
            raise ValueError(
                "A ready SpecRhythm proposal exceeds the verification budget; "
                "rebuild or ancestor-safely prune it before scheduling."
            )

        constrained = (max_target_requests is not None and len(target_candidates) > max_target_requests) or (
            verification_budget is not None and sum(candidate_counts.values()) > verification_budget
        )
        if constrained:
            # Age first keeps a fixed-cap one-batch ablation starvation-free.
            ordered = sorted(
                target_candidates,
                key=lambda index: (self._ready_wait_cycles.get(index, 0), -index),
                reverse=True,
            )
            selected: list[int] = []
            selected_tokens = 0
            for index in ordered:
                if max_target_requests is not None and len(selected) >= max_target_requests:
                    break
                if verification_budget is not None and selected_tokens + candidate_counts[index] > verification_budget:
                    continue
                selected.append(index)
                selected_tokens += candidate_counts[index]
            target = tuple(selected)
        else:
            target = target_candidates

        target_set = set(target)
        deferred = tuple(index for index in target_candidates if index not in target_set)
        for index in active:
            if index in self.ready:
                self._ready_wait_cycles[index] = 0 if index in target_set else self._ready_wait_cycles.get(index, 0) + 1

        draftable = tuple(index for index in active if index not in self.ready and index not in self.staged_eager)
        if target and not overlap:
            normal = ()
            eager = ()
        else:
            normal = draftable
            eager = tuple(index for index in target if index not in self.staged_eager) if overlap else ()
        if target:
            phase = PipelinePhase.STEADY
        elif normal:
            phase = PipelinePhase.WARMUP
        elif self.ready:
            phase = PipelinePhase.DRAIN
        else:
            phase = PipelinePhase.COMPLETE
        plan = SpecRhythmExecutionPlan(
            plan_id=self.plan_id,
            phase=phase,
            target_home_batch_id=None,
            draft_home_batch_id=0 if normal or eager else None,
            target_request_indices=target,
            normal_draft_request_indices=normal,
            eager_candidate_indices=eager,
            target_candidate_budgets={index: candidate_counts[index] for index in target},
            verification_candidate_budget=verification_budget,
            deferred_target_request_indices=deferred,
        )
        self.plan_id += 1
        return plan

    def new_ticket(
        self,
        request_index: int,
        *,
        gamma: int,
        eager: bool,
    ) -> SpecRhythmProposalTicket:
        state = self.request_states[request_index]
        ticket = SpecRhythmProposalTicket(
            proposal_id=self._next_proposal_id,
            request_index=request_index,
            home_batch_id=state.home_batch_id,
            gamma=int(gamma),
            required_prefix_epoch=state.prefix_epoch + int(eager),
            eager=bool(eager),
            lifecycle=(ProposalLifecycle.STAGED_EAGER if eager else ProposalLifecycle.AVAILABLE),
        )
        self._next_proposal_id += 1
        return ticket

    def publish(self, tickets: Sequence[SpecRhythmProposalTicket]) -> None:
        for ticket in tickets:
            index = ticket.request_index
            if ticket.eager:
                if index in self.staged_eager:
                    raise RuntimeError("SpecRhythm request already has a staged eager continuation.")
                self.staged_eager[index] = ticket
            else:
                if index in self.ready:
                    raise RuntimeError("SpecRhythm request already has an available proposal.")
                if ticket.required_prefix_epoch != self.request_states[index].prefix_epoch:
                    raise RuntimeError("SpecRhythm normal proposal was built from a stale prefix.")
                self.ready[index] = ticket

    def finish_verification(
        self,
        request_index: int,
        *,
        fully_accepted: bool,
        proposed_tokens: int,
        accepted_tokens: int,
        delivered_tokens: int,
        draft_confidence: float | None,
        ema_alpha: float,
    ) -> SpecRhythmProposalTicket | None:
        current = self.validate_verification(request_index)
        self.ready.pop(request_index)
        self._ready_wait_cycles.pop(request_index, None)
        state = self.request_states[request_index]
        current.lifecycle = ProposalLifecycle.CONSUMED
        state.record_verification(
            proposed_tokens=proposed_tokens,
            accepted_tokens=accepted_tokens,
            delivered_tokens=delivered_tokens,
            draft_confidence=draft_confidence,
            ema_alpha=ema_alpha,
        )
        eager = self.staged_eager.pop(request_index, None)
        if eager is None:
            promoted = None
        elif fully_accepted and eager.required_prefix_epoch == state.prefix_epoch:
            eager.lifecycle = ProposalLifecycle.AVAILABLE
            self.ready[request_index] = eager
            promoted = eager
        else:
            eager.lifecycle = ProposalLifecycle.INVALIDATED
            promoted = None
        self.next_target_home_batch_id = 1 - state.home_batch_id
        return promoted

    def finish_cycle(self, target_home_batch_id: int | None) -> None:
        """Advance home rotation after every request in one target cycle.

        A rolling-eager verification may contain tickets from both homes.
        Per-request completion order therefore cannot define the next normal
        home.  Native executors call this once after the complete target batch;
        the per-request update above remains for backward-compatible callers
        that verify only one home at a time.
        """

        if target_home_batch_id is None:
            return
        if target_home_batch_id not in (0, 1):
            raise ValueError("SpecRhythm target batch must be zero or one.")
        self.next_target_home_batch_id = 1 - target_home_batch_id

    def validate_verification(self, request_index: int) -> SpecRhythmProposalTicket:
        """Check a ready ticket without mutating state, before device KV commit."""

        current = self.ready.get(request_index)
        if current is None:
            raise RuntimeError("SpecRhythm target attempted to consume a missing proposal.")
        state = self.request_states[request_index]
        if current.required_prefix_epoch != state.prefix_epoch:
            raise RuntimeError("SpecRhythm target attempted to verify a stale proposal.")
        if (
            current.request_index != request_index
            or current.home_batch_id != state.home_batch_id
            or current.lifecycle is not ProposalLifecycle.AVAILABLE
        ):
            raise RuntimeError("SpecRhythm target proposal routing/lifecycle is invalid.")
        return current

    def invalidate_request(self, request_index: int) -> None:
        self._ready_wait_cycles.pop(request_index, None)
        for mapping in (self.ready, self.staged_eager):
            ticket = mapping.pop(request_index, None)
            if ticket is not None:
                ticket.lifecycle = ProposalLifecycle.INVALIDATED

    def _discard_inactive_payloads(self, active: set[int]) -> None:
        for index in set(self.ready).union(self.staged_eager) - active:
            self.invalidate_request(index)


@dataclass(frozen=True)
class SpecRhythmSchedule:
    """Portable scheduler output for a generic vLLM service adapter."""

    execution: SpecRhythmExecutionPlan
    budget: SpecRhythmBudgetPlan
    # Fixed-shape tree plans are created from the same scalar budget that is
    # used by the linear path.  Keeping them in the schedule prevents a
    # worker from silently inventing a second, gamma-derived budget.
    tree_plans: Mapping[int, object] = field(default_factory=dict)

    @property
    def tree_enabled(self) -> bool:
        return bool(self.tree_plans)


class SpecRhythmScheduler:
    """Request admission/preemption adapter shared by non-native frontends.

    The scheduler owns metadata and lifecycle only. A worker integration can
    consume :class:`SpecRhythmSchedule.budget`, execute its own draft/target
    forwards, then call :meth:`finish_verification` to commit the guarded
    prefix. This keeps generic request admission independent from the Ascend
    HCCL transport.
    """

    def __init__(
        self,
        shaper: SpecRhythmBudgetShaper,
        *,
        max_num_seqs: int = 512,
        tree_width: int = 1,
        tree_max_depth: int = 1,
        tree_max_model_len: int | None = None,
        tree_device: object | None = None,
        priority_mode: bool | None = None,
        priority_burst: int = 2,
        merge_ready_homes: bool = False,
        max_target_requests: int | None = None,
        max_eager_tokens: int = 0,
        eager_reserve_tokens: int = 0,
        urgency_threshold: float = 0.75,
    ) -> None:
        if max_num_seqs <= 0:
            raise ValueError("SpecRhythm scheduler max_num_seqs must be positive")
        self.shaper = shaper
        self.max_num_seqs = int(max_num_seqs)
        if tree_width <= 0 or tree_max_depth <= 0:
            raise ValueError("SpecRhythm tree width and depth must be positive")
        self.tree_width = int(tree_width)
        self.tree_max_depth = int(tree_max_depth)
        self.tree_max_model_len = tree_max_model_len
        self.tree_device = tree_device
        if priority_burst <= 0:
            raise ValueError("SpecRhythm priority burst must be positive")
        if max_target_requests is not None and max_target_requests <= 0:
            raise ValueError("SpecRhythm target request limit must be positive")
        if max_eager_tokens < 0:
            raise ValueError("SpecRhythm eager-token cap must be non-negative")
        if eager_reserve_tokens < 0:
            raise ValueError("SpecRhythm eager-token reserve must be non-negative")
        if urgency_threshold < 0:
            raise ValueError("SpecRhythm urgency threshold must be non-negative")
        self.priority_mode = priority_mode
        self.priority_burst = int(priority_burst)
        self.merge_ready_homes = bool(merge_ready_homes)
        self.max_target_requests = max_target_requests
        self.max_eager_tokens = int(max_eager_tokens)
        self.eager_reserve_tokens = int(eager_reserve_tokens)
        self.urgency_threshold = float(urgency_threshold)
        self.request_states: dict[int, SpecRhythmRuntimeState] = {}
        self._active: list[int] = []
        self.controller = SpecRhythmPipelineController(self.request_states)

    @property
    def active_request_indices(self) -> tuple[int, ...]:
        return tuple(self._active)

    def admit(
        self,
        request_index: int,
        *,
        home_batch_id: int | None = None,
        slo_tpot_ms: float | None = None,
        slo_class: str | None = None,
        max_gamma: int | None = None,
    ) -> SpecRhythmRuntimeState:
        """Register and admit one request into the alternating batch set."""

        index = int(request_index)
        if index in self.request_states:
            raise ValueError(f"SpecRhythm request {index} is already registered")
        if len(self._active) >= self.max_num_seqs:
            raise RuntimeError("SpecRhythm scheduler has no free request slots")
        if home_batch_id is None:
            counts = [
                sum(self.request_states[current].home_batch_id == home for current in self._active) for home in (0, 1)
            ]
            home_batch_id = 0 if counts[0] <= counts[1] else 1
        state = SpecRhythmRuntimeState(
            request_index=index,
            home_batch_id=int(home_batch_id),
            slo_tpot_ms=slo_tpot_ms,
            slo_class=slo_class,
            max_gamma=max_gamma,
        )
        self.request_states[index] = state
        self._active.append(index)
        return state

    def preempt(self, request_index: int) -> None:
        """Remove a request from active scheduling while retaining accounting."""

        index = int(request_index)
        if index not in self.request_states:
            raise KeyError(index)
        self.controller.invalidate_request(index)
        self._active = [current for current in self._active if current != index]

    def reactivate(self, request_index: int) -> None:
        index = int(request_index)
        if index not in self.request_states:
            raise KeyError(index)
        if index in self._active:
            return
        if len(self._active) >= self.max_num_seqs:
            raise RuntimeError("SpecRhythm scheduler has no free request slots")
        self._active.append(index)

    def remove(self, request_index: int) -> None:
        index = int(request_index)
        self.preempt(index)
        del self.request_states[index]

    def schedule(
        self,
        *,
        projected_wait_ms: float,
        context_len: int,
        draft_token_budget: int | None = None,
        batch_size: int | None = None,
        prefix_lengths: Mapping[int, int] | None = None,
        draft_window_ms: float | None = None,
        draft_ms_per_token: float | None = None,
    ) -> SpecRhythmSchedule:
        """Build a dual-batch execution plan and one global roofline budget."""

        has_slo_constraints = any(
            self.request_states[index].slo_tpot_ms is not None or self.request_states[index].slo_class is not None
            for index in self._active
        )
        priority = has_slo_constraints if self.priority_mode is None else self.priority_mode
        execution = self.controller.build_plan(
            self._active,
            priority=priority,
            projected_wait_ms=projected_wait_ms,
            priority_burst=self.priority_burst,
            merge_ready_homes=self.merge_ready_homes and has_slo_constraints,
            max_target_requests=self.max_target_requests,
            verification_budget=self.shaper.verification_roof(
                len(self._active) if batch_size is None else batch_size,
                context_len,
            ),
        )
        eager_indices = list(execution.eager_candidate_indices)
        effective_eager_cap = self.max_eager_tokens or (self.shaper.max_gamma if has_slo_constraints else 0)
        if effective_eager_cap:
            eager_indices = [
                index
                for index in eager_indices
                if self.request_states[index].projected_progress_gap(projected_wait_ms) > 0
                and self.request_states[index].urgency(projected_wait_ms) >= self.urgency_threshold
                and self.request_states[index].expected_acceptance_benefit >= self.shaper.acceptance_floor
            ]
        else:
            eager_indices = []
        if tuple(eager_indices) != execution.eager_candidate_indices:
            execution = replace(
                execution,
                eager_candidate_indices=tuple(eager_indices),
            )
        budget = self.shaper.shape(
            plan_id=execution.plan_id,
            normal_request_indices=execution.normal_draft_request_indices,
            eager_request_indices=eager_indices,
            states=self.request_states,
            projected_wait_ms=projected_wait_ms,
            context_len=context_len,
            draft_token_budget=draft_token_budget,
            batch_size=batch_size,
            eager_token_cap=self.max_eager_tokens or None,
            eager_reserve_tokens=self.eager_reserve_tokens,
            draft_window_ms=draft_window_ms,
            draft_ms_per_token=draft_ms_per_token,
            verification_roof=execution.verification_candidate_budget,
        )
        tree_plans: dict[int, object] = {}
        if self.tree_width > 1 or self.tree_max_depth > 1:
            # Lazy import keeps the scalar control plane usable in minimal
            # CPU-only environments that do not import the tree kernels.
            from vllm_ascend.spec_decode.pearl.tree import SpecRhythmTreeCoordinator

            coordinator = SpecRhythmTreeCoordinator(
                width=self.tree_width,
                max_depth=self.tree_max_depth,
            )
            prefixes = prefix_lengths or {}
            max_model_len = self.tree_max_model_len or (
                max(1, int(context_len)) + self.tree_width * self.tree_max_depth + 1
            )
            allocated = set(budget.normal_budgets).union(budget.eager_budgets)
            for index in sorted(allocated):
                prefix_len = int(prefixes.get(index, context_len))
                tree_plan = coordinator.for_request(
                    budget,
                    index,
                    prefix_len=prefix_len,
                    max_model_len=max_model_len,
                    device=self.tree_device,
                )
                if tree_plan is not None:
                    tree_plans[index] = tree_plan
        return SpecRhythmSchedule(
            execution=execution,
            budget=budget,
            tree_plans=tree_plans,
        )

    def advance_cycle(self, elapsed_ms: float) -> None:
        """Charge active requests for one completed service cycle."""
        elapsed = max(0.0, float(elapsed_ms))
        for index in self._active:
            self.request_states[index].add_decode_time(elapsed)

    def finish_verification(self, request_index: int, **kwargs) -> SpecRhythmProposalTicket | None:
        return self.controller.finish_verification(request_index, **kwargs)

    def finish_cycle(self, target_home_batch_id: int | None) -> None:
        self.controller.finish_cycle(target_home_batch_id)


__all__ = [
    "PipelinePhase",
    "ProposalLifecycle",
    "SpecRhythmBudgetPlan",
    "SpecRhythmBudgetShaper",
    "SpecRhythmExecutionPlan",
    "SpecRhythmPipelineController",
    "SpecRhythmSchedule",
    "SpecRhythmScheduler",
    "SpecRhythmProposalTicket",
    "SpecRhythmRuntimeState",
]
