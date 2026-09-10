# SPDX-License-Identifier: Apache-2.0
"""Pure control-plane tree refinement and measured hidden-draft capacity.

The candidate selector implements the two-stage policy in SpecRhythm section
4.4. Confidence is an estimate of expected progress, not a correctness guard;
target verification still decides every committed token. All inputs/outputs
are ordinary Python values so one authoritative plan can be broadcast to every
worker without depending on device-local tensor state.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TreeCandidateRequest:
    """An exploratory tree plus the request's current scheduling constraints.

    ``parents`` use topological indices and -1 for a child of the virtual
    committed root. ``conditional_confidences`` are draft probabilities given
    the parent path; the selector multiplies along that path. Costs default to
    one per node, but may use a profiled common unit such as milliseconds.
    Normal requests normally require one candidate; optional eager requests
    should pass ``minimum_candidates=0``.
    """

    parents: Sequence[int]
    token_ids: Sequence[int]
    conditional_confidences: Sequence[float]
    max_candidates: int
    progress_gap: float = 0.0
    urgency: float = 0.0
    acceptance_rate: float = 1.0
    minimum_candidates: int = 1
    candidate_costs: Sequence[float] | None = None
    waiting_age: int = 0

    def __post_init__(self) -> None:
        parents = tuple(int(value) for value in self.parents)
        tokens = tuple(int(value) for value in self.token_ids)
        confidence = tuple(float(value) for value in self.conditional_confidences)
        costs = (
            (1.0,) * len(parents)
            if self.candidate_costs is None
            else tuple(float(value) for value in self.candidate_costs)
        )
        if not (len(parents) == len(tokens) == len(confidence) == len(costs)):
            raise ValueError("Exploratory tree parents, tokens, confidence and costs must align.")
        if any(parent < -1 or parent >= index for index, parent in enumerate(parents)):
            raise ValueError("Exploratory tree parents must precede their children.")
        if any(token < 0 for token in tokens):
            raise ValueError("Exploratory tree token IDs must be non-negative.")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in confidence):
            raise ValueError("Draft confidence must be finite and in [0, 1].")
        if any(not math.isfinite(value) or value <= 0.0 for value in costs):
            raise ValueError("Candidate costs must be finite and strictly positive.")
        if not math.isfinite(self.acceptance_rate) or not 0.0 <= self.acceptance_rate <= 1.0:
            raise ValueError("Acceptance rate must be finite and in [0, 1].")
        if any(not math.isfinite(value) or value < 0.0 for value in (self.progress_gap, self.urgency)):
            raise ValueError("Progress gap and urgency must be finite and non-negative.")
        if self.waiting_age < 0:
            raise ValueError("Candidate waiting age must be non-negative.")
        if not 0 <= self.minimum_candidates <= min(self.max_candidates, len(parents)):
            raise ValueError("Candidate minimum must fit the per-request cap and tree.")
        object.__setattr__(self, "parents", parents)
        object.__setattr__(self, "token_ids", tokens)
        object.__setattr__(self, "conditional_confidences", confidence)
        object.__setattr__(self, "candidate_costs", costs)


@dataclass(frozen=True)
class GlobalTreeCandidateSelection:
    selected_indices: Mapping[int, tuple[int, ...]]
    expected_progress: Mapping[int, float]
    stage1_candidate_counts: Mapping[int, int]
    stage2_candidate_counts: Mapping[int, int]
    total_candidates: int
    total_cost: float
    verification_budget: int
    deferred_request_indices: tuple[int, ...] = ()

    @property
    def candidate_counts(self) -> dict[int, int]:
        return {index: len(nodes) for index, nodes in self.selected_indices.items()}


def select_global_tree_candidates(
    requests: Mapping[int, TreeCandidateRequest],
    verification_budget: int,
    *,
    max_total_cost: float | None = None,
) -> GlobalTreeCandidateSelection:
    """Refine exploratory trees under one actual candidate count/cost roof.

    First reserve each admitted normal request's minimum ancestor-closed tree,
    then close projected progress gaps with high-confidence reachable nodes.
    Finally distribute residual capacity across requests using joint path
    probability weighted by residual urgency. B remains an upper bound: zero
    expected-value nodes are not added merely to fill it. A deferred minimum
    retains no partial proposal, so callers can retry it with increased age.
    """

    if verification_budget <= 0:
        raise ValueError("Global verification budget must be positive.")
    if max_total_cost is not None and (not math.isfinite(max_total_cost) or max_total_cost < 0.0):
        raise ValueError("Global candidate cost budget must be finite and non-negative.")
    indices = sorted(requests)
    if any(index < 0 for index in indices):
        raise ValueError("Candidate request IDs must be non-negative.")
    cost_roof = math.inf if max_total_cost is None else float(max_total_cost)
    selected: dict[int, set[int]] = {index: set() for index in indices}
    progress = {index: 0.0 for index in indices}
    stage1 = {index: 0 for index in indices}
    stage2 = {index: 0 for index in indices}
    children: dict[int, list[list[int]]] = {}
    gains: dict[int, list[float]] = {}
    frontiers: dict[int, list[tuple[float, int]]] = {}
    total_count = 0
    total_cost = 0.0
    deferred: set[int] = set()

    for index in indices:
        request = requests[index]
        joint: list[float] = []
        child_rows: list[list[int]] = [[] for _ in request.parents]
        frontier = []
        for node, parent in enumerate(request.parents):
            probability = request.conditional_confidences[node]
            joint.append(probability * (joint[parent] if parent >= 0 else 1.0))
            if parent >= 0:
                child_rows[parent].append(node)
            else:
                heapq.heappush(frontier, (-joint[node] / request.candidate_costs[node], node))
        children[index] = child_rows
        gains[index] = [value * request.acceptance_rate for value in joint]
        # Acceptance is request-constant, so it does not change within-request
        # ordering. Multiplication here also makes zero acceptance explicit.
        frontiers[index] = [(score * request.acceptance_rate, node) for score, node in frontier]
        heapq.heapify(frontiers[index])

    def add_candidate(index: int, node: int, *, urgent_stage: bool) -> None:
        nonlocal total_count, total_cost
        request = requests[index]
        selected[index].add(node)
        progress[index] += gains[index][node]
        total_count += 1
        total_cost += request.candidate_costs[node]
        (stage1 if urgent_stage else stage2)[index] += 1
        for child in children[index][node]:
            heapq.heappush(
                frontiers[index],
                (-gains[index][child] / request.candidate_costs[child], child),
            )

    # Normal minima keep target progress live even for zero-confidence drafts.
    # Under overload an older postponed request receives the next capacity.
    for index in sorted(
        indices,
        key=lambda current: (
            requests[current].waiting_age,
            requests[current].urgency,
            requests[current].progress_gap * requests[current].acceptance_rate,
            -current,
        ),
        reverse=True,
    ):
        request = requests[index]
        minimum = request.minimum_candidates
        if not minimum:
            continue
        preview_frontier = list(frontiers[index])
        preview_nodes: list[int] = []
        preview_cost = 0.0
        if total_count + minimum <= verification_budget:
            while preview_frontier and len(preview_nodes) < minimum:
                _, node = heapq.heappop(preview_frontier)
                if total_cost + preview_cost + request.candidate_costs[node] > cost_roof:
                    continue
                preview_nodes.append(node)
                preview_cost += request.candidate_costs[node]
                for child in children[index][node]:
                    heapq.heappush(
                        preview_frontier,
                        (-gains[index][child] / request.candidate_costs[child], child),
                    )
        if len(preview_nodes) < minimum:
            deferred.add(index)
            continue
        for node in preview_nodes:
            add_candidate(index, node, urgent_stage=True)
        # The preview has already removed selected/unaffordable nodes and
        # inserted children. Replacing the frontier avoids duplicate entries.
        frontiers[index] = preview_frontier

    for urgent_stage in (True, False):
        while total_count < verification_budget:
            choices: list[tuple[float, int, int]] = []
            for index in indices:
                request = requests[index]
                if index in deferred or len(selected[index]) >= request.max_candidates:
                    continue
                gap = max(0.0, request.progress_gap - progress[index])
                if urgent_stage and gap <= 0.0:
                    continue
                frontier = frontiers[index]
                while frontier and (total_cost + request.candidate_costs[frontier[0][1]] > cost_roof):
                    # Remaining capacity only decreases, so an unaffordable
                    # candidate cannot become eligible later in this step.
                    heapq.heappop(frontier)
                if not frontier:
                    continue
                negative_gain_per_cost, node = frontier[0]
                if negative_gain_per_cost >= 0.0:
                    continue
                residual_urgency = request.urgency * gap / max(1.0, request.progress_gap)
                weight = gap * (1.0 + request.urgency) if urgent_stage else 1.0 + residual_urgency
                choices.append((-negative_gain_per_cost * weight, -index, -node))
            if not choices:
                break
            _, negative_index, negative_node = max(choices)
            index, node = -negative_index, -negative_node
            heapq.heappop(frontiers[index])
            add_candidate(index, node, urgent_stage=urgent_stage)

    return GlobalTreeCandidateSelection(
        selected_indices={index: tuple(sorted(selected[index])) for index in indices},
        expected_progress=progress,
        stage1_candidate_counts=stage1,
        stage2_candidate_counts=stage2,
        total_candidates=total_count,
        total_cost=total_cost,
        verification_budget=int(verification_budget),
        deferred_request_indices=tuple(sorted(deferred)),
    )


@dataclass(frozen=True)
class DraftWindowBudget:
    """A measured prediction, not a guarantee of hidden execution latency."""

    draft_window_ms: float
    draft_ms_per_token: float | None
    normal_tokens: int
    draft_token_budget: int
    eager_token_budget: int
    residual_window_ms: float
    predicted_exposed_draft_ms: float
    calibrated: bool


@dataclass
class DraftWindowEstimator:
    """EMA model of W and the residual drafting capacity after normal work.

    Feed role-local compute durations, not a host span including a wait for
    the other model. A shared/broadcast observation keeps every rank's plan
    identical. Drafted token counts must include exploratory/discarded work,
    not just the refined candidates eventually passed to target verification.
    """

    ema_alpha: float = 0.2
    draft_ms_per_token: float | None = field(default=None, init=False)
    target_verify_ms: float | None = field(default=None, init=False)
    communication_ms: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.ema_alpha) or not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("Draft-window EMA alpha must be in (0, 1].")

    def observe(
        self,
        *,
        draft_compute_ms: float,
        drafted_tokens: int,
        target_verify_ms: float,
        communication_ms: float = 0.0,
    ) -> None:
        values = (draft_compute_ms, target_verify_ms, communication_ms)
        if any(not math.isfinite(value) or value < 0.0 for value in values) or drafted_tokens < 0:
            raise ValueError("Draft-window observations must be finite and non-negative.")

        def update(previous: float | None, sample: float) -> float:
            return sample if previous is None else self.ema_alpha * sample + (1.0 - self.ema_alpha) * previous

        if drafted_tokens and draft_compute_ms > 0.0:
            self.draft_ms_per_token = update(self.draft_ms_per_token, draft_compute_ms / drafted_tokens)
        if target_verify_ms > 0.0:
            self.target_verify_ms = update(self.target_verify_ms, target_verify_ms)
        self.communication_ms = update(self.communication_ms, communication_ms)

    def estimate(self, *, normal_tokens: int, max_draft_tokens: int) -> DraftWindowBudget:
        if normal_tokens < 0 or max_draft_tokens < 0:
            raise ValueError("Draft-window token counts must be non-negative.")
        normal = min(int(normal_tokens), int(max_draft_tokens))
        calibrated = self.draft_ms_per_token is not None and self.target_verify_ms is not None
        window_ms = max(0.0, (self.target_verify_ms or 0.0) - (self.communication_ms or 0.0))
        if not calibrated:
            # Bootstrap mandatory normal work; without a measured overlap
            # window there is no evidence for admitting optional eager work.
            return DraftWindowBudget(window_ms, self.draft_ms_per_token, normal, normal, 0, 0.0, 0.0, False)
        assert self.draft_ms_per_token is not None
        normal_ms = normal * self.draft_ms_per_token
        hidden_tokens = int(window_ms / self.draft_ms_per_token)
        capacity = min(int(max_draft_tokens), max(normal, hidden_tokens))
        return DraftWindowBudget(
            draft_window_ms=window_ms,
            draft_ms_per_token=self.draft_ms_per_token,
            normal_tokens=normal,
            draft_token_budget=capacity,
            eager_token_budget=max(0, capacity - normal),
            residual_window_ms=max(0.0, window_ms - normal_ms),
            predicted_exposed_draft_ms=max(0.0, normal_ms - window_ms),
            calibrated=True,
        )
