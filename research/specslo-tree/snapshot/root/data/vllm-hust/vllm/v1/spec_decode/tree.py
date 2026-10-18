# SPDX-License-Identifier: Apache-2.0
"""Tree verification primitives for greedy speculative decoding."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TreeVerificationOutput:
    token_ids: torch.Tensor
    accepted_node_indices: torch.Tensor


def verify_greedy_tree(
    draft_token_ids: torch.Tensor,
    parent_indices: torch.Tensor,
    target_token_ids: torch.Tensor,
    max_depth: int,
    placeholder_token_id: int = -1,
) -> TreeVerificationOutput:
    """Accept the longest draft-tree path matching target greedy predictions.

    ``target_token_ids`` has one prediction for the virtual root followed by
    one prediction per draft node. The final prediction is therefore also the
    recovery/bonus token for the selected path. Parent indices are local to a
    request and use -1 for children of the virtual root.
    """
    if draft_token_ids.ndim != 1 or parent_indices.shape != draft_token_ids.shape:
        raise ValueError("draft tokens and parent indices must be 1-D and aligned")
    if target_token_ids.shape != (draft_token_ids.numel() + 1,):
        raise ValueError("target_token_ids must contain root and per-node predictions")
    if max_depth < 1:
        raise ValueError("max_depth must be positive")

    device = draft_token_ids.device
    output = torch.full(
        (max_depth + 1,), placeholder_token_id, dtype=draft_token_ids.dtype, device=device
    )
    accepted = torch.full(
        (max_depth,), -1, dtype=torch.int32, device=device
    )
    parent = -1
    prediction_index = 0
    output_len = 0

    # Tree mode is currently Eager-only. Scalar reads keep the traversal
    # simple and deterministic while the NPU kernels remain responsible for
    # the expensive target forward and KV movement.
    for depth in range(max_depth):
        prediction = target_token_ids[prediction_index]
        children = torch.nonzero(parent_indices == parent, as_tuple=False).flatten()
        if children.numel() == 0:
            break
        matches = children[draft_token_ids[children] == prediction]
        if matches.numel() == 0:
            break
        node = int(matches[0].item())
        output[output_len] = prediction
        accepted[depth] = node
        output_len += 1
        parent = node
        prediction_index = node + 1

    output[output_len] = target_token_ids[prediction_index]
    return TreeVerificationOutput(output, accepted)


def make_full_tree_parents(width: int, depth: int, device: torch.device | None = None) -> torch.Tensor:
    """Return breadth-first parent indices for a full ``width`` x ``depth`` tree."""
    if width < 1 or depth < 1:
        raise ValueError("tree width and depth must be positive")
    parents: list[int] = []
    previous_level = [-1]
    next_node = 0
    for _ in range(depth):
        current_level: list[int] = []
        for parent in previous_level:
            parents.extend([parent] * width)
            current_level.extend(range(next_node, next_node + width))
            next_node += width
        previous_level = current_level
    return torch.tensor(parents, dtype=torch.int32, device=device)


def make_spine_tree_parents(
    width: int, depth: int, device: torch.device | None = None
) -> torch.Tensor:
    """Return parents for top-k siblings along a top-1 continuation spine."""
    if width < 1 or depth < 1:
        raise ValueError("tree width and depth must be positive")
    parents = [-1] * width
    for level in range(1, depth):
        parents.extend([(level - 1) * width] * width)
    return torch.tensor(parents, dtype=torch.int32, device=device)


def make_spine_first_tree_parents(
    width: int, depth: int, device: torch.device | None = None
) -> torch.Tensor:
    """Return a contiguous top-1 spine followed by per-level leaf siblings."""
    if width < 1 or depth < 1:
        raise ValueError("tree width and depth must be positive")
    parents = [-1] + list(range(depth - 1))
    for level in range(depth):
        parent = -1 if level == 0 else level - 1
        parents.extend([parent] * (width - 1))
    return torch.tensor(parents, dtype=torch.int32, device=device)


def verify_greedy_tree_batch(
    draft_token_ids: torch.Tensor,
    parent_indices: torch.Tensor,
    num_draft_tokens: list[int],
    target_token_ids: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    max_depth: int,
    placeholder_token_id: int = -1,
) -> TreeVerificationOutput:
    """Batched wrapper for request-major flattened tree tensors."""
    batch_size = len(num_draft_tokens)
    outputs = torch.full(
        (batch_size, max_depth + 1),
        placeholder_token_id,
        dtype=draft_token_ids.dtype,
        device=draft_token_ids.device,
    )
    accepted = torch.full(
        (batch_size, max_depth),
        -1,
        dtype=torch.int32,
        device=draft_token_ids.device,
    )
    draft_start = 0
    target_start = 0
    for req_idx, count in enumerate(num_draft_tokens):
        request_predictions = torch.cat(
            (target_token_ids[target_start : target_start + count], bonus_token_ids[req_idx].reshape(1))
        )
        result = verify_greedy_tree(
            draft_token_ids[draft_start : draft_start + count],
            parent_indices[draft_start : draft_start + count],
            request_predictions,
            max_depth,
            placeholder_token_id,
        )
        outputs[req_idx] = result.token_ids
        accepted[req_idx] = result.accepted_node_indices
        draft_start += count
        target_start += count
    return TreeVerificationOutput(outputs, accepted)
