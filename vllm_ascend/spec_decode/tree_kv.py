# SPDX-License-Identifier: Apache-2.0
"""Token-granular KV compaction used after tree verification."""

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch_npu


@dataclass(frozen=True)
class TreeKVCompactionPlan:
    """Device-indexed source/destination slots for an accepted tree path."""

    source_slots: torch.Tensor
    destination_slots: torch.Tensor
    accepted_node_indices: torch.Tensor

    def __post_init__(self) -> None:
        if self.source_slots.ndim != 1 or self.destination_slots.shape != self.source_slots.shape:
            raise ValueError("tree KV compaction slot tensors must be aligned 1-D tensors")
        if self.accepted_node_indices.shape != self.source_slots.shape:
            raise ValueError("accepted node indices must align with source slots")


def build_tree_kv_compaction_plan(
    node_slots: torch.Tensor,
    accepted_node_indices: torch.Tensor,
    *,
    destination_start: torch.Tensor | int | None = None,
) -> TreeKVCompactionPlan:
    """Build a compact path move without reading NPU indices on the host.

    ``node_slots`` is the physical KV slot for each request-local draft node;
    ``accepted_node_indices`` contains ``-1`` for rejected/unused depths. The
    destination sequence is contiguous and starts at the first node slot,
    allowing callers to overwrite a linear PEARL prefix after verification.
    """
    if node_slots.ndim != 1 or accepted_node_indices.ndim != 1:
        raise ValueError("tree KV compaction inputs must be 1-D")
    if accepted_node_indices.numel() == 0:
        return TreeKVCompactionPlan(
            node_slots.new_empty((0,)),
            node_slots.new_empty((0,)),
            accepted_node_indices.to(dtype=torch.long),
        )
    if (accepted_node_indices >= node_slots.numel()).any() or (accepted_node_indices < -1).any():
        raise ValueError("accepted tree node index is outside node_slots")
    valid = accepted_node_indices >= 0
    selected = accepted_node_indices[valid].to(dtype=torch.long)
    source = torch.index_select(node_slots, 0, selected)
    first_destination = node_slots[:1] if destination_start is None else torch.as_tensor(
        destination_start, dtype=node_slots.dtype, device=node_slots.device
    ).reshape(1)
    destination = first_destination + torch.arange(
        selected.numel(), dtype=node_slots.dtype, device=node_slots.device
    )
    return TreeKVCompactionPlan(source, destination, selected)


def verify_greedy_tree_device(
    draft_token_ids: torch.Tensor,
    parent_indices: torch.Tensor,
    target_token_ids: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    batch_size: int,
    num_nodes: int,
    max_depth: int,
    placeholder_token_id: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verify a uniform greedy draft tree without device-to-host transfers."""
    if draft_token_ids.numel() != batch_size * num_nodes:
        raise ValueError("draft token count does not match the uniform tree shape")
    if parent_indices.numel() != draft_token_ids.numel():
        raise ValueError("parent indices must align with draft tokens")
    if target_token_ids.numel() != draft_token_ids.numel():
        raise ValueError("target token count must align with draft tokens")
    if bonus_token_ids.numel() != batch_size:
        raise ValueError("one bonus token is required per request")

    drafts = draft_token_ids.reshape(batch_size, num_nodes)
    targets = target_token_ids.reshape(batch_size, num_nodes)
    parents = parent_indices.reshape(batch_size, num_nodes)
    predictions = torch.cat(
        (targets, bonus_token_ids.reshape(batch_size, 1)), dim=1
    )
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

    node_ids = torch.arange(
        num_nodes, dtype=torch.int32, device=draft_token_ids.device
    ).unsqueeze(0)
    no_match = torch.full_like(node_ids, num_nodes)
    current_parent = torch.full(
        (batch_size,), -1, dtype=torch.int32, device=draft_token_ids.device
    )
    prediction_indices = torch.zeros(
        batch_size, dtype=torch.long, device=draft_token_ids.device
    )
    active = torch.ones(
        batch_size, dtype=torch.bool, device=draft_token_ids.device
    )
    batch_indices = torch.arange(
        batch_size, dtype=torch.long, device=draft_token_ids.device
    )

    for depth in range(max_depth):
        prediction = predictions[batch_indices, prediction_indices]
        outputs[:, depth] = torch.where(
            active,
            prediction,
            torch.full_like(prediction, placeholder_token_id),
        )
        matches = (
            (parents == current_parent.unsqueeze(1))
            & (drafts == prediction.unsqueeze(1))
            & active.unsqueeze(1)
        )
        selected = torch.where(matches, node_ids, no_match).amin(dim=1)
        matched = selected < num_nodes
        accepted[:, depth] = torch.where(
            matched, selected, torch.full_like(selected, -1)
        )
        prediction_indices = torch.where(
            matched, selected.to(torch.long) + 1, prediction_indices
        )
        current_parent = torch.where(matched, selected, current_parent)
        active = matched

    final_prediction = predictions[batch_indices, prediction_indices]
    outputs[:, max_depth] = torch.where(
        active,
        final_prediction,
        torch.full_like(final_prediction, placeholder_token_id),
    )
    return outputs, accepted


def _move_tensor_slots(
    cache: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
    packed_kv: bool,
) -> None:
    if cache.ndim < 2:
        raise ValueError(f"unsupported KV cache shape: {tuple(cache.shape)}")
    if packed_kv:
        if cache.shape[0] != 2 or cache.ndim < 3:
            raise ValueError(f"invalid packed K/V cache shape: {tuple(cache.shape)}")
        for plane in cache.unbind(0):
            flat = plane.flatten(0, 1)
            source_values = torch.index_select(flat, 0, source_slots)
            if flat.device.type == "npu":
                torch_npu.npu_scatter_nd_update_(
                    flat,
                    destination_slots.to(torch.int32).view(-1, 1),
                    source_values,
                )
            else:
                flat.index_copy_(0, destination_slots, source_values)
    else:
        flat = cache.flatten(0, 1)
        source_values = torch.index_select(flat, 0, source_slots)
        if flat.device.type == "npu":
            torch_npu.npu_scatter_nd_update_(
                flat,
                destination_slots.to(torch.int32).view(-1, 1),
                source_values,
            )
        else:
            flat.index_copy_(0, destination_slots, source_values)


def _move_npu_kv_pair_slots(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
) -> None:
    """Gather an overlapping-safe K/V pair and scatter it with one NPU op."""
    key_values = torch.index_select(key_cache.flatten(0, 1), 0, source_slots)
    value_values = torch.index_select(value_cache.flatten(0, 1), 0, source_slots)
    torch_npu._npu_reshape_and_cache(
        key=key_values,
        value=value_values,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_indices=destination_slots.to(torch.int32),
    )


def _is_npu_kv_pair(layer_cache: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> bool:
    return len(layer_cache) == 2 and all(
        isinstance(cache, torch.Tensor) and cache.device.type == "npu"
        for cache in layer_cache
    )


def move_kv_cache_slots(
    kv_caches: Iterable[torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]],
    source_slots: torch.Tensor,
    destination_slots: torch.Tensor,
) -> None:
    """Move token slots in every layer, safely handling overlapping paths."""
    if source_slots.shape != destination_slots.shape:
        raise ValueError("source and destination slot tensors must have equal shapes")
    if source_slots.numel() == 0:
        return
    source_slots = source_slots.to(dtype=torch.long)
    destination_slots = destination_slots.to(dtype=torch.long)
    for layer_cache in kv_caches:
        if isinstance(layer_cache, torch.Tensor):
            _move_tensor_slots(
                layer_cache, source_slots, destination_slots, packed_kv=True
            )
        elif _is_npu_kv_pair(layer_cache):
            _move_npu_kv_pair_slots(
                layer_cache[0],
                layer_cache[1],
                source_slots,
                destination_slots,
            )
        else:
            for cache in layer_cache:
                _move_tensor_slots(
                    cache, source_slots, destination_slots, packed_kv=False
                )
