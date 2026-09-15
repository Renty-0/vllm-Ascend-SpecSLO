# SPDX-License-Identifier: Apache-2.0
"""Token-granular KV compaction used after tree verification."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

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


@dataclass
class _TreeKVCompactionGraphEntry:
    """One address-stable compaction graph for an exact move count."""

    source_slots: torch.Tensor
    destination_slots: torch.Tensor
    graph: Any
    validated: bool = False


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
        slot_indices=destination_slots,
    )


def _is_npu_kv_pair(layer_cache: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> bool:
    return len(layer_cache) == 2 and all(
        isinstance(cache, torch.Tensor) and cache.device.type == "npu"
        for cache in layer_cache
    )


class TreeKVCompactionGraphRunner:
    """Replay all-layer NPU KV movement graphs keyed by exact move count.

    An ``NPUGraph`` binds the addresses of every captured KV tensor.  A runner
    therefore belongs to one model/worker cache allocation for its entire
    lifetime; it must never be shared by engines merely because their move
    counts match.  Calls are expected to be serialized on one worker's current
    stream, like the model ACLGraph runner.

    The first call for a count warms up and captures an identity movement, then
    copies the real indices into graph-owned buffers and replays exactly once.
    Running the real overlapping movement eagerly before capture validation
    would apply it twice and can destroy a later source slot.
    """

    def __init__(
        self,
        kv_caches: Iterable[tuple[torch.Tensor, torch.Tensor]],
        *,
        enabled: bool = True,
        max_graph_entries: int = 16,
        validate_first_replay: bool = True,
    ) -> None:
        if max_graph_entries <= 0:
            raise ValueError("tree KV compaction max_graph_entries must be positive")
        supplied_caches = tuple(kv_caches)
        if not supplied_caches or not all(
            isinstance(pair, (tuple, list)) and _is_npu_kv_pair(pair)
            for pair in supplied_caches
        ):
            raise ValueError("tree KV compaction graphs require non-empty NPU K/V pairs")
        layer_caches = tuple((pair[0], pair[1]) for pair in supplied_caches)
        devices = {cache.device for pair in layer_caches for cache in pair}
        if len(devices) != 1:
            raise ValueError("tree KV compaction graph caches must share one NPU device")
        for key_cache, value_cache in layer_caches:
            if key_cache.ndim < 2 or value_cache.shape != key_cache.shape:
                raise ValueError("tree KV compaction graph K/V cache shapes must align")
            if value_cache.dtype != key_cache.dtype:
                raise ValueError("tree KV compaction graph K/V cache dtypes must align")

        self.layer_caches = layer_caches
        self.device = next(iter(devices))
        self.enabled = bool(enabled)
        self.max_graph_entries = int(max_graph_entries)
        self.validate_first_replay = bool(validate_first_replay)
        self.entries: dict[int, _TreeKVCompactionGraphEntry] = {}
        self.disabled_move_counts: set[int] = set()
        self.capture_count = 0
        self.capture_attempt_count = 0
        self.replay_count = 0
        self.failed_capture_count = 0
        self.failed_validation_count = 0
        self.capacity_fallback_count = 0

    def _eager_move(
        self,
        source_slots: torch.Tensor,
        destination_slots: torch.Tensor,
    ) -> None:
        for key_cache, value_cache in self.layer_caches:
            _move_npu_kv_pair_slots(
                key_cache,
                value_cache,
                source_slots,
                destination_slots,
            )

    def _capture_entry(self, move_count: int) -> _TreeKVCompactionGraphEntry:
        # Identity slots make both allocation warmup and capture harmless to
        # live model state. They are mutable graph inputs, not captured host
        # literals; later replays only change their values, never addresses.
        source_slots = torch.arange(
            move_count,
            dtype=torch.long,
            device=self.device,
        )
        destination_slots = source_slots.to(dtype=torch.int32)
        minimum_capacity = min(
            int(cache.shape[0]) * int(cache.shape[1]) for pair in self.layer_caches for cache in pair
        )
        if move_count > minimum_capacity:
            raise ValueError("tree KV compaction identity warmup exceeds cache capacity")

        # Initialize lazy operator resources before entering capture. Capture
        # itself is not assumed to produce a consumable first result.
        self._eager_move(source_slots, destination_slots)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            self._eager_move(source_slots, destination_slots)
        return _TreeKVCompactionGraphEntry(
            source_slots=source_slots,
            destination_slots=destination_slots,
            graph=graph,
        )

    def _source_snapshots(
        self,
        source_slots: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (
                torch.index_select(key_cache.flatten(0, 1), 0, source_slots).clone(),
                torch.index_select(value_cache.flatten(0, 1), 0, source_slots).clone(),
            )
            for key_cache, value_cache in self.layer_caches
        ]

    def _validate_or_repair_first_replay(
        self,
        move_count: int,
        entry: _TreeKVCompactionGraphEntry,
        destination_slots: torch.Tensor,
        expected: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> bool:
        # A single host synchronization covers every layer. Exact copies must
        # be bit-identical; accepting a tolerance could conceal a stale-index
        # graph replay and commit the wrong branch KV.
        matches = torch.ones((), dtype=torch.bool, device=self.device)
        destination_long = destination_slots.to(dtype=torch.long)
        for (key_cache, value_cache), (expected_key, expected_value) in zip(
            self.layer_caches,
            expected,
        ):
            actual_key = torch.index_select(key_cache.flatten(0, 1), 0, destination_long)
            actual_value = torch.index_select(value_cache.flatten(0, 1), 0, destination_long)
            matches = matches & torch.all(actual_key == expected_key)
            matches = matches & torch.all(actual_value == expected_value)
        if bool(matches.cpu().item()):
            entry.validated = True
            return True

        # The graph only writes destination slots. Repair them from snapshots
        # rather than rerunning a gather from possibly-overwritten sources.
        torch.npu.synchronize()
        self.entries.pop(move_count, None)
        entry.graph.reset()
        self.disabled_move_counts.add(move_count)
        self.failed_validation_count += 1
        for (key_cache, value_cache), (expected_key, expected_value) in zip(
            self.layer_caches,
            expected,
        ):
            torch_npu._npu_reshape_and_cache(
                key=expected_key,
                value=expected_value,
                key_cache=key_cache,
                value_cache=value_cache,
                slot_indices=destination_slots,
            )
        return False

    def move(
        self,
        source_slots: torch.Tensor,
        destination_slots: torch.Tensor,
    ) -> bool:
        """Move slots and return whether an ACLGraph replay performed it."""
        if source_slots.ndim != 1 or destination_slots.shape != source_slots.shape:
            raise ValueError("tree KV compaction graph slots must be aligned 1-D tensors")
        move_count = int(source_slots.numel())
        if move_count == 0:
            return False
        source_slots = source_slots.to(device=self.device, dtype=torch.long)
        destination_slots = destination_slots.to(
            device=self.device,
            dtype=torch.int32,
        )
        if not self.enabled or move_count in self.disabled_move_counts:
            self._eager_move(source_slots, destination_slots)
            return False

        entry = self.entries.get(move_count)
        if entry is None:
            if len(self.entries) >= self.max_graph_entries:
                self.capacity_fallback_count += 1
                self._eager_move(source_slots, destination_slots)
                return False
            self.capture_attempt_count += 1
            try:
                entry = self._capture_entry(move_count)
            except Exception:
                # Capture is an optimization, not part of the correctness
                # contract.  Identity warmup/capture cannot have changed the
                # requested destinations, so this real move is still safe to
                # execute exactly once through the eager path.
                self.failed_capture_count += 1
                self.disabled_move_counts.add(move_count)
                self._eager_move(source_slots, destination_slots)
                return False
            self.entries[move_count] = entry
            self.capture_count += 1

        expected = self._source_snapshots(source_slots) if self.validate_first_replay and not entry.validated else None
        # Copies and replay intentionally use the current stream. This orders
        # changed indices after the previous replay and before this replay
        # without a per-cycle host synchronization or auxiliary-stream race.
        entry.source_slots.copy_(source_slots)
        entry.destination_slots.copy_(destination_slots)
        entry.graph.replay()
        self.replay_count += 1
        if expected is not None:
            return self._validate_or_repair_first_replay(
                move_count,
                entry,
                destination_slots,
                expected,
            )
        return True

    def release(self) -> int:
        """Synchronously release resident CANN graph resources."""
        if not self.entries:
            return 0
        torch.npu.synchronize()
        entries = tuple(self.entries.values())
        self.entries.clear()
        for entry in entries:
            entry.graph.reset()
        return len(entries)


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
    destination_slots_int32 = destination_slots.to(dtype=torch.int32)
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
                destination_slots_int32,
            )
        else:
            for cache in layer_cache:
                _move_tensor_slots(
                    cache, source_slots, destination_slots, packed_kv=False
                )
