# SPDX-License-Identifier: Apache-2.0
"""Tree-aware SpecRhythm planning and Ascend attention-mask construction.

The regular vLLM tree verifier owns the final token comparison.  This module
owns the parts that are specific to PEARL scheduling: translating a scalar
roofline allocation into a bounded tree and selecting candidates without
breaking parent-prefix dependencies.  Masks use the vLLM convention where
``True`` means blocked attention and ``False`` means visible attention.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm_ascend.spec_decode.pearl.spec_rhythm import SpecRhythmBudgetPlan


def make_spine_first_parents(width: int, depth: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Return contiguous spine-first parent indices for a uniform tree."""

    if width < 1 or depth < 1:
        raise ValueError("tree width and depth must be positive")
    parents = [-1] + list(range(depth - 1))
    for level in range(depth):
        parent = -1 if level == 0 else level - 1
        parents.extend([parent] * (width - 1))
    return torch.tensor(parents, dtype=torch.int32, device=device)


def _node_levels(width: int, depth: int) -> list[int]:
    return list(range(1, depth + 1)) + [level for level in range(1, depth + 1) for _ in range(width - 1)]


def build_tree_attention_mask(
    width: int,
    depth: int,
    prefix_len: int,
    max_model_len: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build one root-plus-tree causal mask for CANN attention.

    The returned shape is ``[1 + width * depth, max_model_len]``.  Prefix
    tokens are visible to every query, the virtual root sees itself, and each
    draft node sees only its prefix, root and ancestor chain.  Invalid columns
    remain blocked so the tensor can be reused in a fixed-shape graph bucket.
    """

    if prefix_len < 0 or max_model_len <= 0:
        raise ValueError("prefix_len must be non-negative and max_model_len positive")
    query_len = 1 + width * depth
    if prefix_len + query_len > max_model_len:
        raise ValueError("tree query positions exceed max_model_len")
    parents = make_spine_first_parents(width, depth).tolist()
    levels = _node_levels(width, depth)
    query_len = 1 + width * depth
    mask = torch.ones((query_len, max_model_len), dtype=torch.bool, device=device)
    mask[:, :prefix_len] = False
    mask[0, prefix_len] = False
    for node, parent in enumerate(parents):
        row = node + 1
        mask[row, prefix_len] = False
        current = node
        while current >= 0:
            mask[row, prefix_len + current + 1] = False
            current = parents[current]
    # ``levels`` is intentionally computed above as a validation of the
    # spine-first layout; positions are exposed by TreeSpeculationPlan.
    assert len(levels) == len(parents)
    return mask


@dataclass(frozen=True)
class TreeSpeculationPlan:
    """Fixed-shape tree metadata consumed by an Ascend target worker."""

    width: int
    depth: int
    candidate_budget: int
    prefix_len: int
    max_model_len: int
    parent_indices: torch.Tensor
    positions: torch.Tensor
    attention_mask: torch.Tensor
    # Logical depth positions are useful to schedulers, but KV cache writes
    # need a unique slot per sibling branch.  Native target forward consumes
    # this separate physical-position vector.
    cache_positions: torch.Tensor | None = None

    def __post_init__(self) -> None:
        node_count = self.parent_indices.numel()
        if self.candidate_budget < 1 or self.candidate_budget > node_count:
            raise ValueError("candidate_budget must fit inside the configured tree")
        if not 1 <= node_count <= self.width * self.depth:
            raise ValueError("parent_indices do not match tree shape")
        if self.positions.numel() != node_count + 1:
            raise ValueError("positions must include root and every tree node")
        if tuple(self.attention_mask.shape) != (node_count + 1, self.max_model_len):
            raise ValueError("attention_mask does not match tree shape")
        if self.cache_positions is None:
            object.__setattr__(self, "cache_positions", self.positions.clone())
        assert self.cache_positions is not None
        if self.cache_positions.numel() != node_count + 1:
            raise ValueError("cache_positions must include root and every tree node")
        if self.cache_positions.ndim != 1 or self.cache_positions.unique().numel() != self.cache_positions.numel():
            raise ValueError("cache_positions must be unique within one tree")


def pack_selected_tree_plan(
    plan: TreeSpeculationPlan,
    selected_indices: Sequence[int] | torch.Tensor,
) -> TreeSpeculationPlan:
    """Pack an ancestor-closed subset into exactly root + selected KV rows.

    Logical RoPE depths are retained; physical sibling slots and parent IDs
    are remapped. No inactive candidates are sent through the target model.
    """
    indices = (
        selected_indices.detach().cpu().tolist()
        if isinstance(selected_indices, torch.Tensor)
        else list(selected_indices)
    )
    indices = [int(value) for value in indices]
    parents = plan.parent_indices.detach().cpu().tolist()
    if not indices or indices != sorted(set(indices)) or indices[0] < 0 or indices[-1] >= len(parents):
        raise ValueError("selected tree indices must be unique and topologically ordered")
    remap = {old: new for new, old in enumerate(indices)}
    if any(parents[index] != -1 and parents[index] not in remap for index in indices):
        raise ValueError("selected tree is not ancestor closed")
    new_parents = [-1 if parents[index] == -1 else remap[parents[index]] for index in indices]
    device = plan.parent_indices.device
    query_indices = torch.tensor([0, *[index + 1 for index in indices]], dtype=torch.long, device=device)
    positions = plan.positions.index_select(0, query_indices)
    mask = torch.ones((len(indices) + 1, plan.max_model_len), dtype=torch.bool, device=device)
    mask[:, : plan.prefix_len + 1] = False
    for index in range(len(indices)):
        ancestor = index
        while ancestor >= 0:
            mask[index + 1, plan.prefix_len + ancestor + 1] = False
            ancestor = new_parents[ancestor]
    return TreeSpeculationPlan(
        width=plan.width,
        depth=plan.depth,
        candidate_budget=len(indices),
        prefix_len=plan.prefix_len,
        max_model_len=plan.max_model_len,
        parent_indices=torch.tensor(new_parents, dtype=torch.int32, device=device),
        positions=positions,
        attention_mask=mask,
        cache_positions=torch.arange(
            plan.prefix_len, plan.prefix_len + len(indices) + 1, dtype=torch.int32, device=device
        ),
    )


def tree_primary_path(plan: TreeSpeculationPlan) -> list[int]:
    """Return the first-child dependency path in topological candidate order."""
    parents = plan.parent_indices[: plan.candidate_budget].detach().cpu().tolist()
    path: list[int] = []
    parent = -1
    for index, value in enumerate(parents):
        if value == parent:
            path.append(index)
            parent = index
    return path


def build_tree_speculation_plan(
    width: int,
    depth: int,
    prefix_len: int,
    max_model_len: int,
    *,
    candidate_budget: int | None = None,
    device: torch.device | str | None = None,
) -> TreeSpeculationPlan:
    """Construct parent, position and mask tensors for one request."""

    if width < 1 or depth < 1:
        raise ValueError("tree width and depth must be positive")
    budget = width * depth if candidate_budget is None else int(candidate_budget)
    if not 1 <= budget <= width * depth:
        raise ValueError("candidate_budget must be in [1, width * depth]")
    parents = make_spine_first_parents(width, depth, device=device)
    levels = torch.tensor([0] + _node_levels(width, depth), dtype=torch.int32, device=device)
    positions = levels + int(prefix_len)
    mask = build_tree_attention_mask(width, depth, prefix_len, max_model_len, device=device)
    return TreeSpeculationPlan(
        width=width,
        depth=depth,
        candidate_budget=budget,
        prefix_len=int(prefix_len),
        max_model_len=int(max_model_len),
        parent_indices=parents,
        positions=positions,
        attention_mask=mask,
        cache_positions=torch.arange(
            prefix_len,
            prefix_len + width * depth + 1,
            dtype=torch.int32,
            device=device,
        ),
    )


@dataclass(frozen=True)
class TreeCandidateSelection:
    """Top-scoring tree candidates with all required ancestors included."""

    indices: torch.Tensor
    token_ids: torch.Tensor


@dataclass(frozen=True)
class TreeVerificationOutput:
    """Greedy tree output and request-local accepted node indices."""

    token_ids: torch.Tensor
    accepted_node_indices: torch.Tensor


def _verify_greedy_tree_uniform(
    draft_token_ids: torch.Tensor,
    parent_indices: torch.Tensor,
    target_token_ids: torch.Tensor,
    bonus_token_id: torch.Tensor,
    max_depth: int,
    placeholder_token_id: int,
) -> TreeVerificationOutput:
    if draft_token_ids.ndim != 1 or parent_indices.shape != draft_token_ids.shape:
        raise ValueError("draft tokens and parent indices must be 1-D and aligned")
    if target_token_ids.numel() not in (draft_token_ids.numel(), draft_token_ids.numel() + 1):
        raise ValueError(
            "target_token_ids must contain one prediction per draft node, "
            "or those predictions plus the root-query output"
        )
    if bonus_token_id.numel() != 1:
        raise ValueError("one bonus token is required per tree")
    if max_depth < 1:
        raise ValueError("max_depth must be positive")

    device = draft_token_ids.device
    node_count = draft_token_ids.numel()
    has_root_query = target_token_ids.numel() == node_count + 1
    node_ids = torch.arange(node_count, dtype=torch.int32, device=device)
    no_match = torch.full_like(node_ids, node_count)
    output = torch.full((max_depth + 1,), placeholder_token_id, dtype=draft_token_ids.dtype, device=device)
    accepted = torch.full((max_depth,), -1, dtype=torch.int32, device=device)
    current_parent = torch.tensor(-1, dtype=torch.int32, device=device)
    prediction_index = torch.tensor(0, dtype=torch.long, device=device)
    active = torch.tensor(True, dtype=torch.bool, device=device)

    # Fixed iteration count keeps the traversal device-side and makes the
    # helper suitable for NPU eager mode and future ACLGraph capture.
    for depth in range(max_depth):
        # Variable-budget trees may expose fewer target rows than the
        # configured maximum depth.  Once the accepted branch reaches the
        # active frontier, keep the traversal device-side but stop indexing
        # beyond the row; the inactive prediction is ignored by ``active``.
        target_count = int(target_token_ids.numel())
        safe_index = prediction_index.clamp(max=max(0, target_count - 1))
        indexed_prediction = target_token_ids[safe_index]
        prediction = torch.where(
            prediction_index < target_count,
            indexed_prediction,
            torch.as_tensor(placeholder_token_id, dtype=draft_token_ids.dtype, device=device),
        )
        output[depth] = torch.where(
            active,
            prediction,
            torch.as_tensor(placeholder_token_id, dtype=output.dtype, device=device),
        )
        matches = (parent_indices == current_parent) & (draft_token_ids == prediction) & active
        selected = torch.where(matches, node_ids, no_match).amin()
        matched = selected < node_count
        accepted[depth] = torch.where(
            matched,
            selected,
            torch.as_tensor(-1, dtype=accepted.dtype, device=device),
        )
        prediction_index = torch.where(matched, selected.to(torch.long) + 1, prediction_index)
        current_parent = torch.where(matched, selected, current_parent)
        active = matched

    if has_root_query:
        bonus_index = prediction_index.clamp(max=max(0, target_token_ids.numel() - 1))
        bonus = target_token_ids[bonus_index].to(dtype=output.dtype)
    else:
        bonus = bonus_token_id.reshape(()).to(dtype=output.dtype)
    output[max_depth] = torch.where(
        active,
        bonus,
        torch.as_tensor(placeholder_token_id, dtype=output.dtype, device=device),
    )
    return TreeVerificationOutput(output, accepted)


def verify_greedy_tree(
    draft_token_ids: torch.Tensor,
    parent_indices: torch.Tensor,
    target_token_ids: torch.Tensor,
    bonus_token_id: torch.Tensor | int,
    max_depth: int,
    placeholder_token_id: int = -1,
) -> TreeVerificationOutput:
    """Verify one request-local greedy tree without host-side traversal.

    ``target_token_ids`` contains the target argmax for each draft-node query;
    ``bonus_token_id`` is the target argmax at the accepted path frontier.
    Nodes are expected in topological order and use ``-1`` for root parents.
    """
    bonus = torch.as_tensor(bonus_token_id, dtype=draft_token_ids.dtype, device=draft_token_ids.device)
    return _verify_greedy_tree_uniform(
        draft_token_ids,
        parent_indices,
        target_token_ids,
        bonus,
        max_depth,
        placeholder_token_id,
    )


def verify_sampled_tree(
    draft_token_ids: torch.Tensor,
    parent_indices: torch.Tensor,
    sampled_target_token_ids: torch.Tensor,
    sampled_bonus_token_id: torch.Tensor | int,
    max_depth: int,
    placeholder_token_id: int = -1,
) -> TreeVerificationOutput:
    """Traverse proposals using samples already drawn from the target.

    Unlike linear rejection sampling, tree verification does not need the
    draft distribution: the target independently samples the successor of
    every queried parent.  A matching proposal only selects the already
    sampled target branch; a miss commits that same target sample.  Therefore
    the emitted token at every depth has the target sampling distribution,
    regardless of how the draft candidate set was constructed.
    """

    return _verify_greedy_tree_uniform(
        draft_token_ids,
        parent_indices,
        sampled_target_token_ids,
        torch.as_tensor(
            sampled_bonus_token_id,
            dtype=draft_token_ids.dtype,
            device=draft_token_ids.device,
        ),
        max_depth,
        placeholder_token_id,
    )


def verify_greedy_tree_batch(
    draft_token_ids: torch.Tensor,
    parent_indices: torch.Tensor,
    num_draft_tokens: Sequence[int] | None,
    target_token_ids: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    max_depth: int,
    placeholder_token_id: int = -1,
) -> TreeVerificationOutput:
    """Verify uniform or variable-width request-major tree tensors.

    Flat tensors are accepted for direct vLLM metadata use. Variable-width
    batches use a short Python loop over requests, while every token traversal
    remains on the originating device.
    """
    if draft_token_ids.ndim == 2:
        batch_size, width = draft_token_ids.shape
        if parent_indices.numel() != draft_token_ids.numel() or target_token_ids.numel() not in (
            draft_token_ids.numel(),
            draft_token_ids.numel() + batch_size,
        ):
            raise ValueError("2-D tree tensors must have matching node counts")
        if bonus_token_ids.numel() != batch_size:
            raise ValueError("bonus_token_ids must have one value per request")
        drafts = draft_token_ids
        parents = parent_indices.reshape(batch_size, width)
        target_width = width + 1 if target_token_ids.numel() == batch_size * (width + 1) else width
        targets = target_token_ids.reshape(batch_size, target_width)
        counts = [width] * batch_size if num_draft_tokens is None else [int(value) for value in num_draft_tokens]
        if any(value != width for value in counts):
            raise ValueError("2-D tree tensors require uniform num_draft_tokens")
    elif draft_token_ids.ndim == 1:
        if num_draft_tokens is None:
            raise ValueError("num_draft_tokens is required for flat tree tensors")
        counts = [int(value) for value in num_draft_tokens]
        if any(value <= 0 for value in counts) or sum(counts) != draft_token_ids.numel():
            raise ValueError("num_draft_tokens must partition the flat draft tensor")
        if parent_indices.shape != draft_token_ids.shape:
            raise ValueError("flat tree tensors must have matching draft/parent shapes")
        expected_target_counts = (sum(counts), sum(counts) + len(counts))
        if target_token_ids.numel() not in expected_target_counts:
            raise ValueError("flat tree tensors must have matching target rows")
        if len(set(counts)) == 1:
            batch_size, width = len(counts), counts[0]
            drafts = draft_token_ids.reshape(batch_size, width)
            parents = parent_indices.reshape(batch_size, width)
            target_width = width + 1 if target_token_ids.numel() == batch_size * (width + 1) else width
            targets = target_token_ids.reshape(batch_size, target_width)
        else:
            if bonus_token_ids.numel() != len(counts):
                raise ValueError("bonus_token_ids must have one value per request")
            rows: list[TreeVerificationOutput] = []
            cursor = 0
            target_cursor = 0
            full_target_rows = target_token_ids.numel() == sum(counts) + len(counts)
            for row, count in enumerate(counts):
                target_count = count + int(full_target_rows)
                rows.append(
                    verify_greedy_tree(
                        draft_token_ids[cursor : cursor + count],
                        parent_indices[cursor : cursor + count],
                        target_token_ids[target_cursor : target_cursor + target_count],
                        bonus_token_ids[row],
                        max_depth,
                        placeholder_token_id,
                    )
                )
                cursor += count
                target_cursor += target_count
            return TreeVerificationOutput(
                torch.stack([row.token_ids for row in rows]),
                torch.stack([row.accepted_node_indices for row in rows]),
            )
    else:
        raise ValueError("draft_token_ids must be a 1-D or 2-D tensor")

    if len(counts) != drafts.shape[0] or bonus_token_ids.numel() != drafts.shape[0]:
        raise ValueError("tree batch dimensions do not match")
    rows = [
        verify_greedy_tree(
            drafts[index],
            parents[index],
            targets[index],
            bonus_token_ids[index],
            max_depth,
            placeholder_token_id,
        )
        for index in range(drafts.shape[0])
    ]
    return TreeVerificationOutput(
        torch.stack([row.token_ids for row in rows]),
        torch.stack([row.accepted_node_indices for row in rows]),
    )


def select_tree_candidates(
    candidate_token_ids: torch.Tensor,
    parent_indices: torch.Tensor,
    scores: torch.Tensor,
    budget: int,
) -> TreeCandidateSelection:
    """Select a dependency-closed candidate subset deterministically.

    ``scores`` is one scalar per node.  A high-scoring descendant is admitted
    only when its complete ancestor chain fits in ``budget``; selected indices
    are returned in original topological order for direct target packing.
    """

    if candidate_token_ids.ndim != 1 or parent_indices.shape != candidate_token_ids.shape:
        raise ValueError("candidate_token_ids and parent_indices must be aligned 1-D tensors")
    if scores.ndim != 1 or scores.numel() != candidate_token_ids.numel():
        raise ValueError("scores must contain one value per candidate")
    if not 1 <= int(budget) <= candidate_token_ids.numel():
        raise ValueError("budget must be in [1, number of candidates]")
    parents = [int(value) for value in parent_indices.detach().cpu().tolist()]
    count = len(parents)
    for index, parent in enumerate(parents):
        if parent < -1 or parent >= index:
            raise ValueError("parent indices must reference an earlier node or the virtual root")
    score_values = scores.detach().float().cpu().tolist()
    order = sorted(range(count), key=lambda index: (-score_values[index], index))
    selected: set[int] = set()
    for index in order:
        chain: list[int] = []
        current = index
        while current >= 0 and current not in selected:
            chain.append(current)
            current = parents[current]
        if len(selected) + len(chain) <= int(budget):
            selected.update(chain)
    selected_indices = torch.tensor(sorted(selected), dtype=torch.long, device=candidate_token_ids.device)
    return TreeCandidateSelection(
        indices=selected_indices,
        token_ids=torch.index_select(candidate_token_ids, 0, selected_indices),
    )


def tree_budget_from_spec_rhythm(
    plan: SpecRhythmBudgetPlan,
    request_index: int,
    width: int,
    max_depth: int,
) -> int:
    """Map one scalar SpecRhythm request allocation to tree node capacity."""

    if width < 1 or max_depth < 1:
        raise ValueError("tree width and max_depth must be positive")
    index = int(request_index)
    allocated = int(plan.normal_budgets.get(index, 0)) + int(plan.eager_budgets.get(index, 0))
    return max(0, min(allocated, width * max_depth))


class SpecRhythmTreeCoordinator:
    """Turn per-request SpecRhythm budgets into fixed-shape tree plans."""

    def __init__(self, *, width: int, max_depth: int) -> None:
        if width < 1 or max_depth < 1:
            raise ValueError("tree width and max_depth must be positive")
        self.width = int(width)
        self.max_depth = int(max_depth)

    def for_request(
        self,
        budget_plan: SpecRhythmBudgetPlan,
        request_index: int,
        *,
        prefix_len: int,
        max_model_len: int,
        device: torch.device | str | None = None,
    ) -> TreeSpeculationPlan | None:
        """Build a request tree, returning ``None`` for an unallocated row."""

        budget = tree_budget_from_spec_rhythm(budget_plan, request_index, self.width, self.max_depth)
        if budget == 0:
            return None
        depth = min(self.max_depth, max(1, math.ceil(budget / self.width)))
        return build_tree_speculation_plan(
            self.width,
            depth,
            prefix_len,
            max_model_len,
            candidate_budget=min(budget, self.width * depth),
            device=device,
        )


__all__ = [
    "TreeCandidateSelection",
    "TreeSpeculationPlan",
    "TreeVerificationOutput",
    "SpecRhythmTreeCoordinator",
    "build_tree_attention_mask",
    "build_tree_speculation_plan",
    "make_spine_first_parents",
    "select_tree_candidates",
    "tree_budget_from_spec_rhythm",
    "verify_greedy_tree",
    "verify_greedy_tree_batch",
]
