# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ACLGraph capture and paged-attention graph-task updates for native PEARL."""

from __future__ import annotations

import json
import operator
import os
import time
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import torch
import torch_npu

from vllm_ascend import envs

DEFAULT_MAX_ACLGRAPH_ENTRIES = 16
TREE_FIA_SPARSE_MODE = 1
TARGET_FIA_TASK_EVENT_GROUP_SIZES = (1, 2, 4)
# ``inner_precise=1`` is bit-identical to mode 2 for the production FULL-mask
# tree contract (including sparse scratch positions and graph-task updates) on
# 910B2, while avoiding the slower high-precision update path.  Keep this
# local to SpecRhythm tree FIA; ordinary vLLM attention retains its defaults.
TREE_FIA_INNER_PRECISE = 1


@dataclass(frozen=True)
class NativeGraphExecution:
    """Outcome of one target call, rather than the graph configuration flag.

    A failed validation can execute a replay but return the eager reference.
    ``used_aclgraph`` describes the returned output; ``replay_executed`` records
    the actual work, including a replay that failed validation.
    """

    mode: str = "not_run"
    fallback_reason: str | None = None
    capture_attempted: bool = False
    replay_executed: bool = False

    @property
    def used_aclgraph(self) -> bool:
        return self.mode in ("capture_replay", "replay")


def bucket_tree_attention_metadata(metadata, *, block_size: int):
    """Make the Python dense-attention KV extent stable across decode steps.

    All queries share a power-of-two physical extent. The request-local mask
    still determines their *actual* visible KV positions, and is refreshed on
    every replay. Unallocated pages in the padded tail map to a valid dummy
    page; dense attention zeros masked K/V before SDPA so stale/NaN cache data
    in those pages cannot influence the result. This is context bucketing, not
    token-budget padding: no new query/candidate rows are added.
    """
    if metadata.use_fused_infer_attention or metadata.attention_mask is None:
        return metadata
    if block_size <= 0 or metadata.block_tables.ndim != 2:
        raise ValueError("Tree graph metadata requires positive block size and 2-D block tables")
    lengths = metadata.context_lens
    if lengths.device.type != "cpu":
        raise ValueError("Tree graph context lengths must remain CPU-resident")
    if lengths.numel() == 0 or bool((lengths <= 0).any()):
        raise ValueError("Tree graph context lengths must be non-empty and positive")
    mask = metadata.attention_mask
    if mask.ndim != 2 or mask.shape[0] != lengths.numel():
        raise ValueError("Tree graph mask rows must match context lengths")
    maximum_length = int(lengths.max())
    capacity = min(int(mask.shape[1]), int(metadata.block_tables.shape[1]) * block_size)
    if maximum_length > capacity:
        raise ValueError("Tree graph context length exceeds its mask or block-table capacity")
    bucket = min(1 << (maximum_length - 1).bit_length(), capacity)
    physical_positions = torch.arange(mask.shape[1], device=mask.device)
    tail_mask = physical_positions.unsqueeze(0) >= lengths.to(device=mask.device).unsqueeze(1)
    return replace(
        metadata,
        context_lens=torch.full_like(lengths, bucket),
        block_tables=metadata.block_tables.clamp_min(0),
        attention_mask=mask.to(torch.bool) | tail_mask,
    )


def bucket_tree_fia_attention_metadata(metadata, *, block_size: int):
    """Stabilize FULL-mask FIA KV lengths without adding query rows.

    FIA receives sequence lengths as host literals embedded in every captured
    attention task. Updating those literals one layer at a time dominates tiny
    tree replays. Within a power-of-two context bucket the request-local FULL
    mask is already the source of truth, so masked tail pages may safely share
    a valid dummy page and every request can use one stable KV extent.
    """
    if not metadata.use_fused_infer_attention or not getattr(metadata, "tree_attention", False):
        return metadata
    mask = getattr(metadata, "tree_attention_mask", None)
    tables = metadata.request_block_tables
    lengths = tuple(int(value) for value in metadata.sequence_lens)
    if (
        block_size <= 0
        or mask is None
        or mask.ndim != 4
        or mask.dtype != torch.bool
        or tables is None
        or tables.ndim != 2
        or not lengths
        or len(lengths) != mask.shape[0]
        or tables.shape[0] != len(lengths)
        or any(value <= 0 for value in lengths)
    ):
        raise ValueError("Tree FIA bucketing requires aligned positive lengths, FULL mask and page tables")
    capacity = min(int(mask.shape[-1]), int(tables.shape[1]) * block_size)
    maximum_length = max(lengths)
    if maximum_length > capacity:
        raise ValueError("Tree FIA context length exceeds its mask or block-table capacity")
    bucket = min(1 << (maximum_length - 1).bit_length(), capacity)
    positions = torch.arange(mask.shape[-1], device=mask.device)
    actual_lengths = torch.tensor(lengths, device=mask.device)
    tail_mask = positions.view(1, 1, 1, -1) >= actual_lengths.view(-1, 1, 1, 1)
    return replace(
        metadata,
        sequence_lens=(bucket,) * len(lengths),
        request_block_tables=tables.clamp_min(0),
        tree_attention_mask=mask | tail_mask,
    )


@dataclass
class NativePagedAttentionGraphTask:
    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    num_kv_heads: int
    num_heads: int
    scale: float
    block_table: torch.Tensor
    context_lens: torch.Tensor
    output: torch.Tensor
    workspace: torch.Tensor
    handle: Any
    event: torch.npu.ExternalEvent
    # Interned by the owning runner on first replay.  Workspace compatibility
    # depends on tensor structure, not layer identity; retaining the compact
    # id avoids rebuilding and hashing the same large shape/dtype tuple for
    # every one of 28 layers on every gamma step.
    workspace_structure_id: int | None = None


@dataclass
class NativeFusedInferAttentionGraphTask:
    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    num_kv_heads: int
    num_heads: int
    scale: float
    block_table: torch.Tensor
    attention_mask: torch.Tensor
    output: torch.Tensor
    softmax_lse: torch.Tensor
    block_size: int
    workspace: torch.Tensor
    handle: Any
    event: torch.npu.ExternalEvent
    tree_attention: bool = False


@dataclass
class NativeACLGraphEntry:
    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    request_block_tables: torch.Tensor | None
    actual_seq_lengths_q: tuple[int, ...]
    sequence_lens: tuple[int, ...]
    graph: torch.npu.NPUGraph
    output: torch.Tensor
    tasks: list[NativePagedAttentionGraphTask | NativeFusedInferAttentionGraphTask]
    attention_mask: torch.Tensor | None = None
    runtime_validated: bool = False
    validated_real_row_count: int = 0
    # Recorded after generic graph-owned input copies.  The target update
    # stream waits on this event before rebuilding captured attention tasks.
    replay_first_copy_done_event: Any | None = None
    # Exact stable-FIA service reuses these buffers after changed-input
    # qualification and graph-cache sealing.  Keeping the host/device staging
    # tensors on the graph entry avoids rebuilding positions, slot mappings,
    # and request block tables as temporary device tensors on every decode.
    stable_staged_metadata: Any | None = None
    stable_positions_cpu: torch.Tensor | None = None
    stable_slot_mapping_cpu: torch.Tensor | None = None
    stable_segment_ids_cpu: torch.Tensor | None = None
    stable_segment_ids_device: torch.Tensor | None = None
    attention_mask_source_id: int | None = None
    attention_mask_source_version: int | None = None
    # Exact target-verification FIA graphs may share one ExternalEvent across
    # a small consecutive set of independently captured single-op handles.
    # The default of one preserves the production per-layer event contract.
    task_event_group_size: int = 1
    # A shorter first group lets replay start before the steady-state FIA
    # update grain without changing any single-op graph-task handle.
    task_event_prefix_group_size: int = 0
    # The generic graph family can also serve stepwise draft calls.  Only an
    # explicitly captured device-position PagedAttention entry may omit CANN
    # graph-task handles.
    taskless_device_attention: bool = False


@dataclass
class NativeDraftACLGraphEntry:
    input_ids: torch.Tensor
    positions: tuple[torch.Tensor, ...]
    slot_mappings: tuple[torch.Tensor, ...]
    context_lens: tuple[torch.Tensor, ...]
    block_tables: tuple[torch.Tensor, ...]
    request_block_tables: tuple[torch.Tensor | None, ...]
    actual_seq_lengths_q: tuple[tuple[int, ...], ...]
    sequence_lens: tuple[tuple[int, ...], ...]
    graph: torch.npu.NPUGraph
    output: torch.Tensor
    tasks: list[NativePagedAttentionGraphTask | NativeFusedInferAttentionGraphTask]
    tasks_per_step: int
    attention_masks: tuple[torch.Tensor | None, ...] = ()
    tree_attention_masks: tuple[torch.Tensor | None, ...] = ()
    tree_attention_modes: tuple[bool, ...] = ()
    runtime_validated: bool = False
    validated_real_row_count: int = 0
    # The event is recorded after current-replay input copies.  Since draft
    # replays use the same current stream, waiting for it also proves that the
    # preceding replay has completed and reset every captured ExternalEvent.
    replay_first_copy_done_event: Any | None = None
    # A fixed-gamma common-KV FIA graph may put one external wait/reset before
    # its first attention task instead of one gate per decoder layer.  The
    # optimization is fail-closed: its captured host length signature must
    # remain unchanged for the complete lifetime of this entry.
    stable_task_barrier: bool = False
    stable_task_barrier_event: Any | None = None
    # Empty captured attention-task lists are valid only for the explicit
    # device-position PagedAttention backend.  Record that fact at capture so
    # a collector regression cannot later masquerade as a taskless replay and
    # silently skip required CANN graph-task updates.
    taskless_device_attention: bool = False


@dataclass
class NativeTargetACLGraphEntry:
    input_ids: tuple[torch.Tensor, ...]
    positions: tuple[torch.Tensor, ...]
    slot_mappings: tuple[torch.Tensor, ...]
    context_lens: tuple[torch.Tensor, ...]
    block_tables: tuple[torch.Tensor, ...]
    request_block_tables: tuple[torch.Tensor | None, ...]
    # Tree verification uses an explicit request-local ancestor mask.  Keep a
    # graph-owned copy so changing the prefix/tree contents between replays
    # cannot leave the graph reading the mask captured by the first request.
    attention_masks: tuple[torch.Tensor | None, ...]
    actual_seq_lengths_q: tuple[tuple[int, ...], ...]
    sequence_lens: tuple[tuple[int, ...], ...]
    graph: torch.npu.NPUGraph
    outputs: tuple[torch.Tensor, ...]
    tasks: list[NativePagedAttentionGraphTask | NativeFusedInferAttentionGraphTask]
    tasks_per_step: int
    runtime_validated: bool = False
    tree_attention_masks: tuple[torch.Tensor | None, ...] = ()
    tree_attention_modes: tuple[bool, ...] = ()


_CAPTURED_TASKS: list[NativePagedAttentionGraphTask | NativeFusedInferAttentionGraphTask] | None = None
_CAPTURED_PA_WORKSPACES: dict[tuple[Any, ...], torch.Tensor] | None = None
_CAPTURED_FIA_WORKSPACES: dict[tuple[Any, ...], torch.Tensor] | None = None
_CAPTURED_STABLE_TASK_BARRIER = False
_CAPTURED_STABLE_TASK_BARRIER_EVENT: Any | None = None
_CAPTURED_TASK_EVENT_GROUP_SIZE = 1
_CAPTURED_TASK_EVENT_PREFIX_GROUP_SIZE = 0
_CAPTURED_TASK_EVENT_GROUP_INDEX = 0
_CAPTURED_TASK_EVENT_GROUP_EVENT: Any | None = None


@contextmanager
def _collect_graph_tasks(
    pa_workspaces: dict[tuple[Any, ...], torch.Tensor] | None = None,
    fia_workspaces: dict[tuple[Any, ...], torch.Tensor] | None = None,
    *,
    stable_task_barrier: bool = False,
    task_event_group_size: int = 1,
    task_event_prefix_group_size: int = 0,
):
    global _CAPTURED_FIA_WORKSPACES, _CAPTURED_PA_WORKSPACES, _CAPTURED_STABLE_TASK_BARRIER
    global _CAPTURED_STABLE_TASK_BARRIER_EVENT, _CAPTURED_TASKS
    global _CAPTURED_TASK_EVENT_GROUP_EVENT, _CAPTURED_TASK_EVENT_GROUP_INDEX
    global _CAPTURED_TASK_EVENT_GROUP_SIZE, _CAPTURED_TASK_EVENT_PREFIX_GROUP_SIZE
    if _CAPTURED_TASKS is not None:
        raise RuntimeError("Nested native PEARL ACLGraph capture is not supported.")
    if task_event_group_size not in TARGET_FIA_TASK_EVENT_GROUP_SIZES:
        raise ValueError("ACLGraph task event group size must be 1, 2, or 4.")
    if task_event_prefix_group_size not in (0, 1, 2):
        raise ValueError("ACLGraph task event prefix group size must be 0, 1, or 2.")
    if task_event_prefix_group_size and task_event_prefix_group_size >= task_event_group_size:
        raise ValueError("ACLGraph task event prefix group must be smaller than the steady group.")
    if stable_task_barrier and (task_event_group_size != 1 or task_event_prefix_group_size):
        raise ValueError("A stable-task barrier cannot also use grouped task events.")
    tasks: list[NativePagedAttentionGraphTask | NativeFusedInferAttentionGraphTask] = []
    _CAPTURED_TASKS = tasks
    # Graph entries on one model runner replay serially.  FIA can therefore
    # keep its large max-workspace buffers in a runner-lifetime pool.  PA stays
    # capture-local because its workspace/tiling depends on exact sequence
    # lengths and is refreshed before every task update.
    _CAPTURED_PA_WORKSPACES = {} if pa_workspaces is None else pa_workspaces
    _CAPTURED_FIA_WORKSPACES = {} if fia_workspaces is None else fia_workspaces
    _CAPTURED_STABLE_TASK_BARRIER = bool(stable_task_barrier)
    _CAPTURED_STABLE_TASK_BARRIER_EVENT = None
    _CAPTURED_TASK_EVENT_GROUP_SIZE = task_event_group_size
    _CAPTURED_TASK_EVENT_PREFIX_GROUP_SIZE = task_event_prefix_group_size
    _CAPTURED_TASK_EVENT_GROUP_INDEX = 0
    _CAPTURED_TASK_EVENT_GROUP_EVENT = None
    try:
        yield tasks
    finally:
        _CAPTURED_TASKS = None
        _CAPTURED_PA_WORKSPACES = None
        _CAPTURED_FIA_WORKSPACES = None
        _CAPTURED_STABLE_TASK_BARRIER = False
        _CAPTURED_STABLE_TASK_BARRIER_EVENT = None
        _CAPTURED_TASK_EVENT_GROUP_SIZE = 1
        _CAPTURED_TASK_EVENT_PREFIX_GROUP_SIZE = 0
        _CAPTURED_TASK_EVENT_GROUP_INDEX = 0
        _CAPTURED_TASK_EVENT_GROUP_EVENT = None


def _capture_graph_task_event(stream: torch.npu.Stream) -> Any:
    """Insert the ExternalEvent gate for one captured attention task.

    Legacy graphs retain one event per task.  The opt-in stable-task draft
    graph inserts exactly one wait/reset pair before its first attention task.
    Exact target-verification FIA graphs may instead share one event across a
    bounded group while retaining one CANN single-op handle per layer.
    """

    global _CAPTURED_STABLE_TASK_BARRIER_EVENT
    global _CAPTURED_TASK_EVENT_GROUP_EVENT, _CAPTURED_TASK_EVENT_GROUP_INDEX
    if _CAPTURED_STABLE_TASK_BARRIER:
        event = _CAPTURED_STABLE_TASK_BARRIER_EVENT
        if event is None:
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            _CAPTURED_STABLE_TASK_BARRIER_EVENT = event
        return event
    task_index = _CAPTURED_TASK_EVENT_GROUP_INDEX
    prefix_size = _CAPTURED_TASK_EVENT_PREFIX_GROUP_SIZE
    if _CAPTURED_TASK_EVENT_GROUP_SIZE == 1:
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        _CAPTURED_TASK_EVENT_GROUP_INDEX += 1
        return event
    relative_index = task_index if not prefix_size or task_index < prefix_size else task_index - prefix_size
    starts_group = task_index == 0 or (
        task_index >= prefix_size and relative_index % _CAPTURED_TASK_EVENT_GROUP_SIZE == 0
    )
    if starts_group:
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        _CAPTURED_TASK_EVENT_GROUP_EVENT = event
    event = _CAPTURED_TASK_EVENT_GROUP_EVENT
    if event is None:
        raise RuntimeError("ACLGraph task event group has no capture event.")
    _CAPTURED_TASK_EVENT_GROUP_INDEX += 1
    return event


def run_native_paged_attention(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    num_heads: int,
    scale: float,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Run eager paged attention or record its replay-time host parameters."""
    if _CAPTURED_TASKS is None:
        torch_npu._npu_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=num_kv_heads,
            num_heads=num_heads,
            scale_value=scale,
            block_table=block_table,
            context_lens=context_lens,
            out=output,
        )
        return

    assert _CAPTURED_PA_WORKSPACES is not None
    workspace_key = (
        tuple(query.shape),
        tuple(key_cache.shape),
        tuple(value_cache.shape),
        query.dtype,
        key_cache.dtype,
        value_cache.dtype,
        num_kv_heads,
        num_heads,
        tuple(block_table.shape),
        tuple(int(value) for value in context_lens.tolist()),
        tuple(output.shape),
    )
    workspace = _CAPTURED_PA_WORKSPACES.get(workspace_key)
    if workspace is None:
        workspace = torch_npu._npu_paged_attention_get_workspace(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=num_kv_heads,
            num_heads=num_heads,
            scale_value=scale,
            block_table=block_table,
            context_lens=context_lens,
            out=output,
        )
        _CAPTURED_PA_WORKSPACES[workspace_key] = workspace
    stream = torch.npu.current_stream()
    event = _capture_graph_task_event(stream)
    torch.npu.graph_task_group_begin(stream)
    torch_npu._npu_paged_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        num_kv_heads=num_kv_heads,
        num_heads=num_heads,
        scale_value=scale,
        block_table=block_table,
        context_lens=context_lens,
        out=output,
        workspace=workspace,
    )
    handle = torch.npu.graph_task_group_end(stream)
    _CAPTURED_TASKS.append(
        NativePagedAttentionGraphTask(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=num_kv_heads,
            num_heads=num_heads,
            scale=scale,
            block_table=block_table,
            context_lens=context_lens,
            output=output,
            workspace=workspace,
            handle=handle,
            event=event,
        )
    )


def _validate_tree_fia_mask(
    query_count: int,
    attention_mask: torch.Tensor | None,
    block_table: torch.Tensor | None,
    actual_seq_lengths_q,
    actual_seq_lengths_kv,
    block_size: int,
) -> None:
    """Validate the FULL-mask ABI using host metadata, without device sync."""
    if (
        attention_mask is None
        or attention_mask.ndim != 4
        or attention_mask.dtype != torch.bool
        or attention_mask.shape[1] != 1
    ):
        raise ValueError("Tree FIA requires a 4-D boolean [requests, 1, maxQ, max_context] FULL mask")
    request_count = len(actual_seq_lengths_q)
    if (
        not request_count
        or len(actual_seq_lengths_kv) != request_count
        or attention_mask.shape[0] != request_count
        or block_table is None
        or block_table.ndim != 2
        or block_table.shape[0] != request_count
        or block_size <= 0
    ):
        raise ValueError("Tree FIA request lengths, FULL mask and block tables must align")
    prior = 0
    for end, kv_length in zip(actual_seq_lengths_q, actual_seq_lengths_kv):
        if end <= prior or end - prior > attention_mask.shape[2]:
            raise ValueError("Tree FIA actual query counts must be positive and fit the FULL mask")
        if not 0 < kv_length <= min(attention_mask.shape[3], block_table.shape[1] * block_size):
            raise ValueError("Tree FIA actual KV length exceeds its FULL mask or page-table capacity")
        prior = end
    if prior != query_count:
        raise ValueError("Tree FIA actual query lengths must consume every query row without padding")


def run_native_fused_infer_attention(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    num_heads: int,
    scale: float,
    block_table: torch.Tensor,
    attention_mask: torch.Tensor,
    actual_seq_lengths_q: list[int],
    actual_seq_lengths_kv: list[int],
    block_size: int,
    output: torch.Tensor,
    tree_attention: bool = False,
) -> None:
    """Run or capture the FIA task used by speculative verification."""
    softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
    args = {
        "query": query,
        "key": key_cache,
        "value": value_cache,
        "atten_mask": attention_mask,
        "block_table": block_table,
        "input_layout": "TND",
        "block_size": block_size,
        "actual_seq_lengths": actual_seq_lengths_q,
        "actual_seq_lengths_kv": actual_seq_lengths_kv,
        "num_key_value_heads": num_kv_heads,
        "num_heads": num_heads,
        "scale": scale,
    }
    if tree_attention:
        _validate_tree_fia_mask(
            query.shape[0],
            attention_mask,
            block_table,
            actual_seq_lengths_q,
            actual_seq_lengths_kv,
            block_size,
        )
        args.update(sparse_mode=TREE_FIA_SPARSE_MODE, inner_precise=TREE_FIA_INNER_PRECISE)
    else:
        # Match vLLM-Ascend's production TND causal FIA contract.  In
        # particular, keep this identical to the graph-task refresh below:
        # capturing with an unbounded future window and later refreshing the
        # same task with ``next_tokens=0`` changes the operator semantics after
        # the first dynamic-length replay.
        args.update(sparse_mode=3, next_tokens=0)
    if _CAPTURED_TASKS is None:
        torch_npu.npu_fused_infer_attention_score.out(
            **args,
            out=[output, softmax_lse],
        )
        return

    assert _CAPTURED_FIA_WORKSPACES is not None
    if tree_attention:
        # FULL-mask tree entries differ frequently only in the number of live
        # requests/candidates.  FIA workspaces are scratch storage and graph
        # entries on one runner never replay concurrently, so a workspace from
        # an equal-or-larger envelope is reusable.  Keep tensor/head/cache
        # layout in the structural key and compare the three dynamic extents
        # separately.  This turns O(number-of-tree-shapes) HBM growth into a
        # small monotonic capacity pool.
        workspace_structure = (
            "tree-fia",
            str(query.device),
            tuple(query.shape[1:]),
            tuple(key_cache.shape),
            tuple(value_cache.shape),
            query.dtype,
            num_kv_heads,
            num_heads,
            block_size,
            int(attention_mask.shape[3]),
            int(block_table.shape[1]),
        )
        workspace_capacity = (
            int(query.shape[0]),
            int(attention_mask.shape[0]),
            int(attention_mask.shape[2]),
        )
        workspace_key = (*workspace_structure, workspace_capacity)
        compatible = [
            (key[-1], value)
            for key, value in _CAPTURED_FIA_WORKSPACES.items()
            if key[:-1] == workspace_structure
            and all(owned >= needed for owned, needed in zip(key[-1], workspace_capacity))
        ]
        workspace = min(compatible, key=lambda item: item[0])[1] if compatible else None
    else:
        # Causal FIA graphs owned by one runner execute serially.  Their
        # workspace is scratch storage, and CANN accepts an equal-or-larger
        # max-workspace buffer for a smaller query envelope.  Keying this pool
        # by the exact T dimension retained one ~322 MiB allocation per mixed
        # prefill/verification graph on 910B and exhausted HBM before the full
        # P128 graph family could be captured.  Capture is largest-first, so
        # reuse the smallest already-owned compatible T capacity while keeping
        # every layout/dtype/cache attribute that defines workspace safety in
        # the structural key.
        workspace_structure = (
            "causal-fia",
            str(query.device),
            tuple(query.shape[1:]),
            tuple(key_cache.shape),
            tuple(value_cache.shape),
            query.dtype,
            num_kv_heads,
            num_heads,
            block_size,
            False,
        )
        workspace_capacity = int(query.shape[0])
        workspace_key = (*workspace_structure, workspace_capacity)
        compatible = [
            (int(key[-1]), value)
            for key, value in _CAPTURED_FIA_WORKSPACES.items()
            if key[:-1] == workspace_structure and int(key[-1]) >= workspace_capacity
        ]
        workspace = min(compatible, key=lambda item: item[0])[1] if compatible else None
    if workspace is None:
        workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(**args)
        _CAPTURED_FIA_WORKSPACES[workspace_key] = workspace
    stream = torch.npu.current_stream()
    event = _capture_graph_task_event(stream)
    torch.npu.graph_task_group_begin(stream)
    torch_npu.npu_fused_infer_attention_score.out(
        **args,
        workspace=workspace,
        out=[output, softmax_lse],
    )
    handle = torch.npu.graph_task_group_end(stream)
    _CAPTURED_TASKS.append(
        NativeFusedInferAttentionGraphTask(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=num_kv_heads,
            num_heads=num_heads,
            scale=scale,
            block_table=block_table,
            attention_mask=attention_mask,
            output=output,
            softmax_lse=softmax_lse,
            block_size=block_size,
            workspace=workspace,
            handle=handle,
            event=event,
            tree_attention=tree_attention,
        )
    )


class NativeACLGraphRunner:
    """Capture native Qwen model decode graphs keyed by packed token count."""

    def __init__(
        self,
        model,
        enabled: bool = True,
        max_graph_tokens: int = 512,
        max_graph_entries: int = DEFAULT_MAX_ACLGRAPH_ENTRIES,
        update_stream_priority: int | None = None,
        is_target_worker: bool = False,
    ) -> None:
        if max_graph_entries <= 0:
            raise ValueError("Native PEARL max_graph_entries must be positive.")
        self.model = model
        self.enabled = enabled
        self.max_graph_tokens = max_graph_tokens
        self.max_graph_entries = max_graph_entries
        self.entries: dict[tuple[str, int], NativeACLGraphEntry] = {}
        self.draft_entries: dict[tuple[str, int], NativeDraftACLGraphEntry] = {}
        self.target_entries: dict[tuple[str, tuple[int, ...]], NativeTargetACLGraphEntry] = {}
        self._fia_workspace_pool: dict[tuple[Any, ...], torch.Tensor] = {}
        if update_stream_priority is None:
            update_stream_priority = envs.VLLM_ASCEND_PEARL_GRAPH_UPDATE_STREAM_PRIORITY
        if update_stream_priority not in (-1, 0):
            raise ValueError("VLLM_ASCEND_PEARL_GRAPH_UPDATE_STREAM_PRIORITY must be -1 or 0.")
        target_fia_task_event_group_size = int(envs.VLLM_ASCEND_PEARL_TARGET_FIA_TASK_EVENT_GROUP_SIZE)
        if target_fia_task_event_group_size not in TARGET_FIA_TASK_EVENT_GROUP_SIZES:
            raise ValueError("VLLM_ASCEND_PEARL_TARGET_FIA_TASK_EVENT_GROUP_SIZE must be 1, 2, or 4.")
        target_fia_task_prefix_event_group_size = int(envs.VLLM_ASCEND_PEARL_TARGET_FIA_TASK_PREFIX_EVENT_GROUP_SIZE)
        if target_fia_task_prefix_event_group_size not in (0, 1, 2):
            raise ValueError("VLLM_ASCEND_PEARL_TARGET_FIA_TASK_PREFIX_EVENT_GROUP_SIZE must be 0, 1, or 2.")
        if (
            target_fia_task_prefix_event_group_size
            and target_fia_task_prefix_event_group_size >= target_fia_task_event_group_size
        ):
            raise ValueError(
                "VLLM_ASCEND_PEARL_TARGET_FIA_TASK_PREFIX_EVENT_GROUP_SIZE "
                "must be smaller than TARGET_FIA_TASK_EVENT_GROUP_SIZE."
            )
        self.update_stream = torch.npu.Stream(priority=update_stream_priority) if enabled else None
        target_fia_task_update_workers = int(envs.VLLM_ASCEND_PEARL_TARGET_FIA_TASK_UPDATE_WORKERS)
        if target_fia_task_update_workers not in (1, 2, 4):
            raise ValueError("VLLM_ASCEND_PEARL_TARGET_FIA_TASK_UPDATE_WORKERS must be 1, 2, or 4.")
        self.target_fia_task_update_workers = target_fia_task_update_workers if enabled and is_target_worker else 1
        self._target_fia_parallel_streams = (
            [self.update_stream]
            + [
                torch.npu.Stream(priority=update_stream_priority)
                for _ in range(self.target_fia_task_update_workers - 1)
            ]
            if self.update_stream is not None
            else []
        )
        self._target_fia_update_executor = (
            ThreadPoolExecutor(
                max_workers=self.target_fia_task_update_workers,
                thread_name_prefix="pearl-target-fia-update",
            )
            if self.target_fia_task_update_workers > 1
            else None
        )
        self.shared_graph_pool_enabled = bool(enabled and envs.VLLM_ASCEND_PEARL_SHARED_GRAPH_POOL)
        self.graph_pool = torch.npu.graph_pool_handle() if self.shared_graph_pool_enabled else None
        self.update_stream_priority = update_stream_priority
        self.target_fia_task_event_group_size = target_fia_task_event_group_size
        self.target_fia_task_prefix_event_group_size = target_fia_task_prefix_event_group_size
        self.target_fia_parallel_update_replays = 0
        self.target_fia_parallel_update_tasks = 0
        # ``capture_attempt_count`` is cumulative telemetry.  Capacity is a
        # separate lifetime guard so an offline profiler can explicitly
        # release one measured shape and reopen exactly those resident slots
        # without falsifying capture metrics.
        self._capture_budget_used = 0
        self.capture_attempt_count = 0
        self.capture_count = 0
        self.replay_count = 0
        self.failed_capture_count = 0
        self.capacity_fallback_count = 0
        self.shape_fallback_count = 0
        self.task_update_replay_count = 0
        self.task_update_skip_replay_count = 0
        self.task_update_replay_counts = {kind: 0 for kind in ("generic", "draft", "target")}
        self.task_update_task_counts = {kind: 0 for kind in ("generic", "draft", "target")}
        # PagedAttention replay telemetry distinguishes the optimized
        # host-metadata key from the compatibility fallback that materializes
        # the CPU context-length tensor. Fine-grained PA/FIA clock reads are
        # opt-in so normal benchmark results do not include profiling overhead.
        self.pa_workspace_host_key_tasks = 0
        self.pa_workspace_tensor_key_tasks = 0
        self.pa_workspace_get_calls = 0
        self.pa_workspace_cache_hits = 0
        self.draft_step_major_pa_replays = 0
        self.draft_step_major_pa_fallback_replays = 0
        self.generic_taskless_replays = 0
        self.draft_taskless_replays = 0
        self.draft_stable_task_barrier_records = 0
        self._pa_workspace_structure_ids: dict[tuple[Any, ...], int] = {}
        self.pa_task_update_profiled_tasks = 0
        self.pa_task_update_host_ns = {phase: 0 for phase in ("key", "get_workspace", "task_update")}
        self.fia_task_update_profiled_tasks = 0
        self.fia_task_update_host_ns = {phase: 0 for phase in ("key", "submit", "event")}
        self.profile_pa_task_update = envs.VLLM_ASCEND_PEARL_PROFILE_PA_TASK_UPDATE
        self.runtime_validation_replay_count = 0
        self.execution_counters: dict[str, dict[str, int]] = {
            kind: {
                "total_calls": 0,
                "capture_replay_calls": 0,
                "replay_calls": 0,
                "eager_fallback_calls": 0,
                "disabled_entry_calls": 0,
                "unclassified_calls": 0,
                "runtime_validation_calls": 0,
                "runtime_validation_failures": 0,
                "changed_input_validation_calls": 0,
                "logical_row_expansion_validation_calls": 0,
            }
            for kind in ("generic", "draft", "target")
        }
        # Keep replay-shape telemetry in host-only dictionaries.  Generic
        # graphs are the steady target path used by SpecRhythm, while the
        # dedicated target family is still used by multi-step target calls.
        # ``token_rows`` is the physical model input M seen by every replayed
        # graph step (capture-size padding included), which is the dimension
        # needed to assess shape-gated kernels such as MC2.
        self.replay_entry_key_histograms: dict[str, dict[Any, int]] = {kind: {} for kind in ("generic", "target")}
        self.replay_token_rows_histograms: dict[str, dict[int, int]] = {kind: {} for kind in ("generic", "target")}
        self.disabled_entry_keys: set[tuple[str, int]] = set()
        self.expected_fia_batch_size: int | None = None
        self.last_fia_shape: tuple[int, ...] = ()
        self.last_fia_expected_batch_size: int | None = None
        self.last_target_execution = NativeGraphExecution()
        self.last_draft_execution = NativeGraphExecution()
        self.last_generic_execution = NativeGraphExecution()
        self.last_draft_entry_key: tuple[str, int] | None = None
        self.last_target_validation_error: dict[str, Any] | None = None
        # A benchmark may lazily discover and changed-input qualify only the
        # shapes exercised by its real request trace, then seal that resident
        # set before starting the measured window.  Sealing is deliberately a
        # fail-closed audit mode: an unseen or unqualified shape is an error,
        # never an implicit capture, validation replay, or eager fallback that
        # would contaminate timed throughput.
        self.graph_cache_sealed = False
        self.pruned_unvalidated_entry_counts = {kind: 0 for kind in ("generic", "draft", "target")}

    def graph_qualification_status(self) -> dict[str, int]:
        """Return the resident graph inventory used by strict benchmarks."""

        entry_groups = {
            "generic": self.entries,
            "draft": self.draft_entries,
            "target": self.target_entries,
        }
        status = {f"{kind}_entries": len(entries) for kind, entries in entry_groups.items()}
        status["shared_memory_pool"] = int(self.shared_graph_pool_enabled)
        status.update(
            {
                f"{kind}_unvalidated_entries": sum(not bool(entry.runtime_validated) for entry in entries.values())
                for kind, entries in entry_groups.items()
            }
        )
        status["unvalidated_entries"] = sum(status[f"{kind}_unvalidated_entries"] for kind in entry_groups)
        status["disabled_entries"] = len(self.disabled_entry_keys)
        status.update(
            {
                f"{kind}_pruned_unvalidated_entries": count
                for kind, count in self.pruned_unvalidated_entry_counts.items()
            }
        )
        status["pruned_unvalidated_entries"] = sum(self.pruned_unvalidated_entry_counts.values())
        status["sealed"] = int(self.graph_cache_sealed)
        return status

    def seal_graph_cache(self) -> dict[str, int]:
        """Forbid graph discovery and validation after qualification.

        Capture validation proves only the values used to create an entry.
        Every resident entry must also have passed its first changed-input
        replay before it can enter this mode.  Diagnostic validate-every-replay
        mode is incompatible with a validation-free measured interval.
        """

        if envs.VLLM_ASCEND_PEARL_VALIDATE_GRAPH_REPLAYS:
            raise RuntimeError("Cannot seal the ACLGraph cache while validate-every-replay diagnostics are enabled.")
        status = self.graph_qualification_status()
        if status["disabled_entries"]:
            raise RuntimeError(
                f"Cannot seal an ACLGraph cache containing disabled entries: disabled={status['disabled_entries']}."
            )
        if status["unvalidated_entries"]:
            details = ", ".join(
                f"{kind}={status[f'{kind}_unvalidated_entries']}"
                for kind in ("generic", "draft", "target")
                if status[f"{kind}_unvalidated_entries"]
            )
            raise RuntimeError(f"Cannot seal an ACLGraph cache with unqualified resident entries: {details}.")
        self.graph_cache_sealed = True
        return self.graph_qualification_status()

    def unseal_graph_cache(self) -> dict[str, int]:
        """Reopen lazy graph discovery for the next untimed workload point."""

        self.graph_cache_sealed = False
        return self.graph_qualification_status()

    def prune_unvalidated_graph_entries(self) -> dict[str, int]:
        """Release cold-only capture entries that never reached qualification.

        This operation is legal only before sealing.  A strict benchmark calls
        it after a complete trace performed no new capture.  If a removed key
        is actually needed by a later hot trace, that trace must capture it
        again and therefore cannot satisfy the final zero-capture fixed point.
        """

        if self.graph_cache_sealed:
            raise RuntimeError("Cannot prune unvalidated entries from a sealed ACLGraph cache.")
        entry_groups = {
            "generic": self.entries,
            "draft": self.draft_entries,
            "target": self.target_entries,
        }
        stale_keys = {
            kind: [key for key, entry in entries.items() if not bool(entry.runtime_validated)]
            for kind, entries in entry_groups.items()
        }
        stale_count = sum(len(keys) for keys in stale_keys.values())
        if not stale_count:
            return self.graph_qualification_status()
        if stale_count > self._capture_budget_used:
            raise RuntimeError("ACLGraph resident capture accounting is smaller than the unvalidated prune set.")
        torch.npu.synchronize()
        for kind, keys in stale_keys.items():
            entries = entry_groups[kind]
            for key in keys:
                entry = entries.pop(key)
                self._reset_graph_entry(entry)
            self.pruned_unvalidated_entry_counts[kind] += len(keys)
        self._capture_budget_used -= stale_count
        return self.graph_qualification_status()

    def _raise_if_graph_cache_sealed(
        self,
        kind: str,
        reason: str,
        entry_key: Any | None = None,
    ) -> None:
        if not self.graph_cache_sealed:
            return
        suffix = "" if entry_key is None else f"; entry_key={entry_key!r}"
        raise RuntimeError(
            f"A sealed ACLGraph cache refused a non-replay execution: kind={kind}, reason={reason}{suffix}."
        )

    def _record_execution(
        self,
        kind: str,
        execution: NativeGraphExecution,
        output: Any,
        *,
        entry_key: Any | None = None,
        token_rows: Sequence[int] = (),
    ) -> Any:
        """Record the returned execution path exactly once per public call."""
        if kind not in self.execution_counters:
            raise ValueError(f"Unknown native ACLGraph execution kind: {kind!r}")
        setattr(self, f"last_{kind}_execution", execution)
        counters = self.execution_counters[kind]
        counters["total_calls"] += 1
        if execution.mode == "capture_replay":
            counters["capture_replay_calls"] += 1
        elif execution.mode == "replay":
            counters["replay_calls"] += 1
            if entry_key is not None and kind in self.replay_entry_key_histograms:
                entry_histogram = self.replay_entry_key_histograms[kind]
                entry_histogram[entry_key] = entry_histogram.get(entry_key, 0) + 1
                token_histogram = self.replay_token_rows_histograms[kind]
                for rows in token_rows:
                    token_histogram[rows] = token_histogram.get(rows, 0) + 1
        elif execution.mode == "eager":
            counters["eager_fallback_calls"] += 1
            if execution.fallback_reason == "disabled_entry":
                counters["disabled_entry_calls"] += 1
        else:
            counters["unclassified_calls"] += 1
        return output

    def _record_runtime_validation(
        self,
        kind: str,
        *,
        changed_input: bool,
        logical_row_expansion: bool = False,
        failed: bool = False,
    ) -> None:
        counters = self.execution_counters[kind]
        counters["runtime_validation_calls"] += 1
        counters["changed_input_validation_calls"] += int(changed_input)
        counters["logical_row_expansion_validation_calls"] += int(logical_row_expansion)
        counters["runtime_validation_failures"] += int(failed)
        self.runtime_validation_replay_count += 1

    def graph_execution_metrics(self) -> dict[str, Any]:
        """Return stable per-path counters and replay histograms for auditing."""
        metrics = {
            f"{kind}_{name}": value
            for kind, counters in self.execution_counters.items()
            for name, value in counters.items()
        }
        metrics.update(
            {
                f"{kind}_task_update_replays": self.task_update_replay_counts[kind]
                for kind in self.task_update_replay_counts
            }
        )
        metrics.update(
            {f"{kind}_task_update_tasks": self.task_update_task_counts[kind] for kind in self.task_update_task_counts}
        )
        for kind in self.replay_entry_key_histograms:
            metrics[f"{kind}_replay_entry_key_histogram"] = {
                repr(entry_key): count
                for entry_key, count in sorted(
                    self.replay_entry_key_histograms[kind].items(),
                    key=lambda item: repr(item[0]),
                )
            }
            metrics[f"{kind}_replay_token_rows_histogram"] = {
                str(rows): count for rows, count in sorted(self.replay_token_rows_histograms[kind].items())
            }
        metrics.update(
            {
                "pa_workspace_host_key_tasks": self.pa_workspace_host_key_tasks,
                "pa_workspace_tensor_key_tasks": self.pa_workspace_tensor_key_tasks,
                "pa_workspace_get_calls": self.pa_workspace_get_calls,
                "pa_workspace_cache_hits": self.pa_workspace_cache_hits,
                "draft_step_major_pa_replays": (self.draft_step_major_pa_replays),
                "draft_step_major_pa_fallback_replays": (self.draft_step_major_pa_fallback_replays),
                "generic_taskless_replays": self.generic_taskless_replays,
                "draft_taskless_replays": self.draft_taskless_replays,
                "draft_stable_task_barrier_records": (self.draft_stable_task_barrier_records),
                "pa_task_update_profiled_tasks": self.pa_task_update_profiled_tasks,
                "graph_update_stream_priority": self.update_stream_priority,
                "target_fia_task_event_group_size": (self.target_fia_task_event_group_size),
                "target_fia_task_prefix_event_group_size": (self.target_fia_task_prefix_event_group_size),
                "target_fia_task_update_workers": self.target_fia_task_update_workers,
                "target_fia_parallel_update_replays": self.target_fia_parallel_update_replays,
                "target_fia_parallel_update_tasks": self.target_fia_parallel_update_tasks,
                **{
                    f"pa_task_update_host_{phase}_ns": elapsed_ns
                    for phase, elapsed_ns in self.pa_task_update_host_ns.items()
                },
                "fia_task_update_profiled_tasks": self.fia_task_update_profiled_tasks,
                **{
                    f"fia_task_update_host_{phase}_ns": elapsed_ns
                    for phase, elapsed_ns in self.fia_task_update_host_ns.items()
                },
            }
        )
        return metrics

    def _record_task_update(self, kind: str, task_count: int) -> None:
        """Record one complete attention-task refresh for a steady replay."""
        if kind not in self.task_update_replay_counts:
            raise ValueError(f"Unknown native ACLGraph task-update kind: {kind!r}")
        if task_count < 0:
            raise ValueError("Native ACLGraph task-update count cannot be negative.")
        self.task_update_replay_count += 1
        self.task_update_replay_counts[kind] += 1
        self.task_update_task_counts[kind] += task_count

    def _draft_replay_first_copy_event(
        self,
        entry: NativeDraftACLGraphEntry,
    ) -> Any:
        """Validate and return the per-entry replay-first dependency event.

        The event is recorded on the current stream after graph-owned inputs
        are copied.  The preceding replay is ordered on that same stream, so
        the event protects both the previous ExternalEvent reset and the new
        input buffers without a host or whole-stream synchronization.
        """
        if self.update_stream is None or not callable(getattr(self.update_stream, "wait_event", None)):
            raise RuntimeError(
                "Draft ACLGraph replay-first task update requires an auxiliary stream with wait_event support."
            )
        expected_task_count = entry.tasks_per_step * len(entry.positions)
        if expected_task_count != len(entry.tasks):
            raise RuntimeError(
                "Draft ACLGraph replay-first task metadata is incomplete: "
                f"expected {expected_task_count} tasks but found {len(entry.tasks)}."
            )
        for task_index, task in enumerate(entry.tasks):
            if not callable(getattr(task.event, "record", None)):
                raise RuntimeError(
                    "Draft ACLGraph replay-first task update requires a "
                    f"recordable ExternalEvent for task {task_index}."
                )
        event = entry.replay_first_copy_done_event
        if event is None:
            event_factory = getattr(torch.npu, "Event", None)
            if not callable(event_factory):
                raise RuntimeError("Draft ACLGraph replay-first task update requires torch.npu.Event support.")
            try:
                event = event_factory()
            except Exception as exc:
                raise RuntimeError(
                    "Draft ACLGraph replay-first task update could not create its input-readiness event."
                ) from exc
            if not callable(getattr(event, "record", None)):
                raise RuntimeError("Draft ACLGraph replay-first task update created an event without record support.")
            entry.replay_first_copy_done_event = event
        return event

    def _record_draft_stable_task_barrier(
        self,
        entry: NativeDraftACLGraphEntry,
        stream: torch.npu.Stream,
    ) -> None:
        """Release one entry-wide stable-task graph barrier exactly once."""

        if getattr(entry, "stable_task_barrier", False) is not True:
            raise RuntimeError("A draft stable-task barrier record was requested for a legacy per-task event entry.")
        event = getattr(entry, "stable_task_barrier_event", None)
        if event is None or not callable(getattr(event, "record", None)):
            raise RuntimeError("A draft stable-task barrier entry has no recordable ExternalEvent.")
        event.record(stream)
        self.draft_stable_task_barrier_records += 1

    def _refresh_draft_stable_task_barrier(
        self,
        entry: NativeDraftACLGraphEntry,
        *,
        stream: torch.npu.Stream | None = None,
    ) -> None:
        """Refresh every handle, then release the single capture barrier."""

        update_stream = stream or self.update_stream
        assert update_stream is not None
        self._update_draft_attention_tasks(
            entry,
            stream=update_stream,
            record_events=False,
        )
        self._record_draft_stable_task_barrier(entry, update_stream)

    def _generic_replay_first_copy_event(
        self,
        entry: NativeACLGraphEntry,
    ) -> Any:
        """Validate the generic target replay-first ExternalEvent contract."""
        if self.update_stream is None or not callable(getattr(self.update_stream, "wait_event", None)):
            raise RuntimeError(
                "Generic target ACLGraph replay-first task update requires an auxiliary stream with wait_event support."
            )
        for task_index, task in enumerate(entry.tasks):
            if not callable(getattr(task.event, "record", None)):
                raise RuntimeError(
                    "Generic target ACLGraph replay-first task update requires "
                    f"a recordable ExternalEvent for task {task_index}."
                )
        event = entry.replay_first_copy_done_event
        if event is None:
            event_factory = getattr(torch.npu, "Event", None)
            if not callable(event_factory):
                raise RuntimeError("Generic target ACLGraph replay-first task update requires torch.npu.Event support.")
            try:
                event = event_factory()
            except Exception as exc:
                raise RuntimeError(
                    "Generic target ACLGraph replay-first task update could not create its input-readiness event."
                ) from exc
            if not callable(getattr(event, "record", None)):
                raise RuntimeError(
                    "Generic target ACLGraph replay-first task update created an event without record support."
                )
            entry.replay_first_copy_done_event = event
        return event

    @staticmethod
    def _tensor_changed(captured: torch.Tensor, incoming: torch.Tensor) -> bool:
        return (
            captured.shape != incoming.shape or captured.dtype != incoming.dtype or not torch.equal(captured, incoming)
        )

    @staticmethod
    def _reset_graph_entry(entry: Any) -> None:
        """Release CANN graph resources held by one no-longer-reachable entry."""
        entry.graph.reset()

    def release_target_graph_entries(self) -> int:
        """Synchronously release every resident target graph.

        Production serving normally retains its small stable shape set.  The
        section 5.3 roofline collector deliberately visits many mutually
        exclusive shapes in one model lifecycle; it calls this method after
        finishing all warmup/timed replays for a shape so graphs do not
        accumulate until HBM exhaustion.  Cumulative capture/replay telemetry
        is preserved, while the released resident capacity becomes reusable.
        """
        if not self.target_entries:
            return 0
        torch.npu.synchronize()
        entries = tuple(self.target_entries.values())
        self.target_entries.clear()
        for entry in entries:
            self._reset_graph_entry(entry)
        self._capture_budget_used = max(0, self._capture_budget_used - len(entries))
        return len(entries)

    def set_expected_fia_batch_size(self, batch_size: int | None) -> None:
        """Set or clear the optional fixed FIA request-count guard."""
        if batch_size is not None and batch_size <= 0:
            raise ValueError("Native PEARL FIA batch size must be positive.")
        self.expected_fia_batch_size = batch_size

    def __call__(self, input_ids: torch.Tensor, positions: torch.Tensor, attention_metadata) -> torch.Tensor:
        return self._run(
            input_ids,
            positions,
            attention_metadata,
            output_kind="hidden",
            output_transform=None,
        )

    def run_stable_fia_hidden(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata,
        *,
        graph_key: str,
        expected_tokens: int,
        expected_request_segments: int,
    ) -> torch.Tensor:
        return self._run_stable_fia(
            input_ids,
            positions,
            attention_metadata,
            graph_key=graph_key,
            expected_tokens=expected_tokens,
            expected_request_segments=expected_request_segments,
            output_kind="hidden",
            output_transform=None,
        )

    def run_stable_fia_greedy(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata,
        vocabulary_size: int,
        *,
        graph_key: str,
        expected_tokens: int,
        expected_request_segments: int,
    ) -> torch.Tensor:
        """Run fixed-Q FIA verification and greedy selection in one graph."""

        return self._run_stable_fia(
            input_ids,
            positions,
            attention_metadata,
            graph_key=graph_key,
            expected_tokens=expected_tokens,
            expected_request_segments=expected_request_segments,
            output_kind=f"greedy:{vocabulary_size}",
            output_transform=lambda hidden_states: self.model.compute_greedy_tokens(
                hidden_states,
                vocabulary_size,
            ),
        )

    def run_stable_fia_greedy_staged(
        self,
        input_ids: torch.Tensor,
        vocabulary_size: int,
        *,
        positions: Any,
        slot_mapping: Any,
        actual_seq_lengths_q: Any,
        sequence_lens: Any,
        segment_sequence_ids: Any,
        cache_block_tables: torch.Tensor,
        attention_mask: torch.Tensor | None,
        graph_key: str,
        expected_tokens: int,
        expected_request_segments: int,
        real_tokens: int | None = None,
    ) -> torch.Tensor:
        """Replay one qualified stable-FIA graph from persistent buffers.

        This is deliberately narrower than :meth:`run_stable_fia_greedy`.
        It is a sealed-service fast path for the exact decode-only
        ``stable-target-verify`` family, never a capture or qualification
        path.  Qualification still performs an ordinary changed-input replay;
        only after that replay has validated the entry and the whole graph
        cache has been sealed may service write directly into the graph-owned
        tensors.  Any missing proof fails closed rather than falling back.

        ``cache_block_tables`` remains the authoritative live page table.  A
        persistent segment-index tensor gathers its current request rows into
        the captured request table, so service does not first allocate device
        metadata in the engine and then copy it again in ``_run``.
        """

        if not envs.VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH:
            raise RuntimeError("Exact stable-FIA staging requires the explicit mixed-target graph opt-in.")
        if not isinstance(graph_key, str) or not graph_key.startswith("stable-target-verify-hidden|"):
            raise ValueError(
                "Persistent stable-FIA staging is restricted to the exact target-verification service family."
            )
        if expected_tokens <= 0 or expected_request_segments <= 0:
            raise ValueError("Stable-FIA token and request counts must be positive.")
        if real_tokens is None:
            real_tokens = expected_tokens
        real_tokens = operator.index(real_tokens)
        if not 0 < real_tokens <= expected_tokens:
            raise ValueError("Stable-FIA real token count must fit the captured token capacity.")
        output_kind = f"greedy:{vocabulary_size}"
        entry_key = (
            f"{output_kind}|fia-stable:{graph_key}",
            expected_tokens,
        )
        if not self.enabled:
            raise RuntimeError("Persistent stable-FIA staging requires an enabled ACLGraph runner.")
        if self.graph_cache_sealed is not True:
            raise RuntimeError("Persistent stable-FIA staging is legal only after graph-cache sealing.")
        if entry_key in self.disabled_entry_keys:
            raise RuntimeError(f"Persistent stable-FIA staging refused a disabled graph entry: {entry_key!r}.")
        entry = self.entries.get(entry_key)
        if entry is None:
            raise RuntimeError(f"Persistent stable-FIA staging requires a resident qualified entry: {entry_key!r}.")
        if not bool(entry.runtime_validated):
            raise RuntimeError(f"Persistent stable-FIA staging cannot bypass changed-input validation: {entry_key!r}.")
        if entry.validated_real_row_count != expected_tokens:
            raise RuntimeError(
                "Persistent stable-FIA staging requires exact validated rows: "
                f"entry={entry.validated_real_row_count}, "
                f"expected={expected_tokens}."
            )

        position_values = self._index_tuple(
            positions,
            expected_tokens,
            "positions",
        )
        slot_values = self._index_tuple(
            slot_mapping,
            expected_tokens,
            "slot_mapping",
        )
        actual_q = self._index_tuple(
            actual_seq_lengths_q,
            expected_request_segments,
            "actual_seq_lengths_q",
        )
        kv_lengths = self._index_tuple(
            sequence_lens,
            expected_request_segments,
            "sequence_lens",
        )
        segment_ids = self._index_tuple(
            segment_sequence_ids,
            expected_request_segments,
            "segment_sequence_ids",
        )
        if (
            any(value < 0 for value in position_values)
            or any(value < 0 for value in slot_values)
            or not actual_q
            or actual_q[-1] != expected_tokens
            or any(current <= previous for previous, current in zip((0, *actual_q[:-1]), actual_q))
            or any(value <= 0 for value in kv_lengths)
            or any(value < 0 for value in segment_ids)
        ):
            raise ValueError(
                "Exact stable-FIA staged routing contains invalid positions, "
                "slots, query partitions, KV lengths, or sequence ids."
            )
        if (
            input_ids.ndim != 1
            or tuple(input_ids.shape) != (real_tokens,)
            or input_ids.dtype != entry.input_ids.dtype
            or input_ids.device != entry.input_ids.device
            or tuple(entry.positions.shape) != (expected_tokens,)
            or tuple(entry.slot_mapping.shape) != (expected_tokens,)
            or tuple(entry.context_lens.shape) != (expected_tokens,)
            or entry.request_block_tables is None
            or entry.request_block_tables.ndim != 2
            or entry.request_block_tables.shape[0] != expected_request_segments
        ):
            raise ValueError("Stable-FIA staged tensors do not match the qualified entry.")
        if (
            not torch.is_tensor(cache_block_tables)
            or cache_block_tables.ndim != 2
            or cache_block_tables.dtype != entry.request_block_tables.dtype
            or cache_block_tables.device != entry.request_block_tables.device
            or cache_block_tables.shape[1] != entry.request_block_tables.shape[1]
            or any(value >= cache_block_tables.shape[0] for value in segment_ids)
        ):
            raise ValueError("Exact stable-FIA live block tables do not match the qualified request-table buffer.")
        if (
            entry.stable_staged_metadata is None
            or entry.stable_positions_cpu is None
            or entry.stable_slot_mapping_cpu is None
            or entry.stable_segment_ids_cpu is None
            or entry.stable_segment_ids_device is None
        ):
            raise RuntimeError(
                "Exact stable-FIA entry was not provisioned with persistent staging before graph-cache sealing."
            )
        self._validate_stable_attention_mask_source(entry, attention_mask)

        # All structural and seal checks above precede the first mutation.
        # Once staging starts, any copy/gather failure propagates: replay must
        # not run with a partially updated graph-owned input set.
        self._write_cpu_indices(
            entry.stable_positions_cpu,
            position_values,
            "positions",
        )
        self._write_cpu_indices(
            entry.stable_slot_mapping_cpu,
            slot_values,
            "slot_mapping",
        )
        self._write_cpu_indices(
            entry.context_lens,
            tuple(value + 1 for value in position_values),
            "context_lens",
        )
        self._write_cpu_indices(
            entry.stable_segment_ids_cpu,
            segment_ids,
            "segment_sequence_ids",
        )
        # Qualification leaves valid dummy token ids in the suffix.  Service
        # updates only the real prefix, avoiding a padding allocation/cat on
        # every replay while scratch KV routing isolates the retained suffix.
        entry.input_ids[:real_tokens].copy_(input_ids)
        entry.positions.copy_(entry.stable_positions_cpu, non_blocking=True)
        entry.slot_mapping.copy_(
            entry.stable_slot_mapping_cpu,
            non_blocking=True,
        )
        entry.stable_segment_ids_device.copy_(
            entry.stable_segment_ids_cpu,
            non_blocking=True,
        )
        torch.index_select(
            cache_block_tables,
            0,
            entry.stable_segment_ids_device,
            out=entry.request_block_tables,
        )
        staged_metadata = entry.stable_staged_metadata
        staged_metadata.actual_seq_lengths_q = actual_q
        staged_metadata.sequence_lens = kv_lengths

        return self._run(
            entry.input_ids,
            entry.positions,
            staged_metadata,
            output_kind=output_kind,
            output_transform=lambda hidden_states: self.model.compute_greedy_tokens(
                hidden_states,
                vocabulary_size,
            ),
            stable_fia_key=graph_key,
            stable_fia_capture_size=expected_tokens,
            _graph_owned_entry=entry,
        )

    @staticmethod
    def _index_tuple(values: Any, expected_size: int, name: str) -> tuple[int, ...]:
        try:
            result = tuple(operator.index(value) for value in values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Exact stable-FIA {name} must contain integer values.") from exc
        if len(result) != expected_size:
            raise ValueError(f"Exact stable-FIA {name} must contain {expected_size} values, got {len(result)}.")
        return result

    @staticmethod
    def _write_cpu_indices(
        destination: torch.Tensor,
        values: tuple[int, ...],
        name: str,
    ) -> None:
        if destination.device.type != "cpu" or destination.numel() != len(values):
            raise RuntimeError(f"Exact stable-FIA persistent {name} staging has an invalid device or shape.")
        # Assign through the resident CPU tensor's NumPy view.  Per-element
        # ``Tensor.__setitem__`` is roughly two orders of magnitude slower for
        # the 128-row exact route, while ``torch.tensor(values)`` would recreate
        # a temporary tensor on every replay.  The view keeps the storage (and
        # pinned-memory registration on NPU service) unchanged.
        destination.numpy().reshape(-1)[:] = values

    @staticmethod
    def _tensor_mutation_version(tensor: torch.Tensor) -> int | None:
        try:
            return int(tensor._version)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None

    @classmethod
    def _validate_stable_attention_mask_source(
        cls,
        entry: NativeACLGraphEntry,
        attention_mask: torch.Tensor | None,
    ) -> None:
        captured = entry.attention_mask
        if captured is None:
            if attention_mask is not None:
                raise RuntimeError("Exact stable-FIA attention-mask structure changed after qualification.")
            return
        version = cls._tensor_mutation_version(attention_mask) if torch.is_tensor(attention_mask) else None
        if (
            not torch.is_tensor(attention_mask)
            or id(attention_mask) != entry.attention_mask_source_id
            or version is None
            or version != entry.attention_mask_source_version
            or attention_mask.shape != captured.shape
            or attention_mask.dtype != captured.dtype
            or attention_mask.device != captured.device
        ):
            raise RuntimeError(
                "Exact stable-FIA attention mask changed identity, mutation version, or structure after qualification."
            )

    @classmethod
    def _provision_stable_fia_staging(
        cls,
        entry: NativeACLGraphEntry,
        captured_metadata: Any,
    ) -> None:
        """Allocate persistent host/device staging before an entry is sealed."""

        if entry.request_block_tables is None:
            raise RuntimeError("Stable-FIA persistent staging requires request block tables.")
        pin_memory = entry.positions.device.type == "npu"
        entry.stable_positions_cpu = torch.empty(
            entry.positions.shape,
            dtype=entry.positions.dtype,
            device="cpu",
            pin_memory=pin_memory,
        )
        entry.stable_slot_mapping_cpu = torch.empty(
            entry.slot_mapping.shape,
            dtype=entry.slot_mapping.dtype,
            device="cpu",
            pin_memory=pin_memory,
        )
        segment_shape = (entry.request_block_tables.shape[0],)
        entry.stable_segment_ids_cpu = torch.empty(
            segment_shape,
            dtype=torch.long,
            device="cpu",
            pin_memory=pin_memory,
        )
        entry.stable_segment_ids_device = torch.empty(
            segment_shape,
            dtype=torch.long,
            device=entry.request_block_tables.device,
        )
        entry.stable_staged_metadata = SimpleNamespace(
            slot_mapping=entry.slot_mapping,
            context_lens=entry.context_lens,
            block_tables=entry.block_tables,
            actual_seq_lengths_q=entry.actual_seq_lengths_q,
            sequence_lens=entry.sequence_lens,
            request_block_tables=entry.request_block_tables,
            attention_mask=entry.attention_mask,
            use_fused_infer_attention=True,
            tree_attention=False,
            tree_attention_mask=None,
        )
        source_mask = getattr(captured_metadata, "attention_mask", None)
        if source_mask is not None:
            version = cls._tensor_mutation_version(source_mask)
            if version is None:
                raise RuntimeError("Stable-FIA attention-mask mutation version is unavailable.")
            entry.attention_mask_source_id = id(source_mask)
            entry.attention_mask_source_version = version

    def _run_stable_fia(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata,
        *,
        graph_key: str,
        expected_tokens: int,
        expected_request_segments: int,
        output_kind: str,
        output_transform: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> torch.Tensor:
        """Run a fixed-Q FIA graph whose request partitions may change.

        Ordinary generic FIA entries include the complete cumulative query
        partition in their key.  Mixed target verification/prefill instead
        pads only scratch segments so the total query count and request count
        are fixed while real prompt partitions retain their exact lengths.
        CANN task updates own the dynamic q/K host literals on replay.

        Validate the complete structural contract before calling ``_run`` so
        a malformed layout cannot partially copy graph-owned input buffers.
        """

        if not envs.VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH:
            raise RuntimeError(
                "Stable mixed-target FIA ACLGraph requires the explicit "
                "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH=1 opt-in."
            )
        if not isinstance(graph_key, str) or not graph_key:
            raise ValueError("Stable FIA graph key must be a non-empty string.")
        if expected_tokens <= 0 or expected_request_segments <= 0:
            raise ValueError("Stable FIA token and request-segment counts must be positive.")
        if not getattr(attention_metadata, "use_fused_infer_attention", False):
            raise ValueError("Stable FIA graph requires fused-infer-attention metadata.")
        actual_q = tuple(int(value) for value in getattr(attention_metadata, "actual_seq_lengths_q", ()))
        sequence_lens = tuple(int(value) for value in getattr(attention_metadata, "sequence_lens", ()))
        request_tables = getattr(
            attention_metadata,
            "request_block_tables",
            None,
        )
        slot_mapping = getattr(attention_metadata, "slot_mapping", None)
        if (
            input_ids.ndim != 1
            or positions.ndim != 1
            or tuple(input_ids.shape) != (expected_tokens,)
            or tuple(positions.shape) != (expected_tokens,)
            or not torch.is_tensor(slot_mapping)
            or tuple(slot_mapping.shape) != (expected_tokens,)
            or len(actual_q) != expected_request_segments
            or len(sequence_lens) != expected_request_segments
            or not actual_q
            or actual_q[-1] != expected_tokens
            or any(current <= previous for previous, current in zip((0, *actual_q[:-1]), actual_q))
            or any(length <= 0 for length in sequence_lens)
            or not torch.is_tensor(request_tables)
            or request_tables.ndim != 2
            or request_tables.shape[0] != expected_request_segments
        ):
            raise ValueError(
                "Stable FIA inputs must have fixed token/table shapes and aligned positive query/KV partitions."
            )
        return self._run(
            input_ids,
            positions,
            attention_metadata,
            output_kind=output_kind,
            output_transform=output_transform,
            stable_fia_key=graph_key,
            stable_fia_capture_size=expected_tokens,
        )

    def run_greedy(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata,
        vocabulary_size: int,
    ) -> torch.Tensor:
        return self._run(
            input_ids,
            positions,
            attention_metadata,
            output_kind=f"greedy:{vocabulary_size}",
            output_transform=lambda hidden_states: self.model.compute_greedy_tokens(
                hidden_states,
                vocabulary_size,
            ),
        )

    def run_draft_greedy(
        self,
        input_ids: torch.Tensor,
        positions: list[torch.Tensor],
        attention_metadatas: list[Any],
        vocabulary_size: int,
        *,
        valid_row_count: int | None = None,
        final_kv_only: bool = False,
        return_confidence: bool = False,
        graph_lane: int | None = None,
        stable_task_barrier: bool = False,
    ) -> torch.Tensor:
        """Replay all draft proposal steps inside one ACLGraph."""
        self.last_draft_execution = NativeGraphExecution()
        self.last_draft_entry_key = None
        if not positions or len(positions) != len(attention_metadatas):
            raise ValueError("A draft ACLGraph requires matching non-empty step metadata.")
        if final_kv_only and len(positions) < 2:
            raise ValueError("A KV-only final draft step requires at least one proposal step.")
        if graph_lane is not None and graph_lane not in (0, 1):
            raise ValueError("A draft ACLGraph lane must be zero or one.")
        input_row_count = int(input_ids.shape[0])
        if valid_row_count is None:
            valid_row_count = input_row_count
        else:
            valid_row_count = int(valid_row_count)
        if valid_row_count <= 0 or valid_row_count > input_row_count:
            raise ValueError(
                f"A draft ACLGraph valid row count must be in [1, {input_row_count}], got {valid_row_count}."
            )
        if not self.enabled:
            self._raise_if_graph_cache_sealed("draft", "disabled")
            execution = NativeGraphExecution("eager", "disabled")
            output = self._execute_draft(
                input_ids,
                positions,
                attention_metadatas,
                vocabulary_size,
                final_kv_only=final_kv_only,
                return_confidence=return_confidence,
            )
            return self._record_execution("draft", execution, output)
        attention_modes = [metadata.use_fused_infer_attention for metadata in attention_metadatas]
        if any(attention_modes) != all(attention_modes):
            raise ValueError("Every step in a draft ACLGraph must use the same attention backend.")
        if (
            all(attention_modes)
            and self.expected_fia_batch_size is not None
            and input_ids.shape[0] != self.expected_fia_batch_size
        ):
            self._raise_if_graph_cache_sealed("draft", "shape")
            self.shape_fallback_count += 1
            execution = NativeGraphExecution("eager", "shape")
            output = self._execute_draft(
                input_ids,
                positions,
                attention_metadatas,
                vocabulary_size,
                final_kv_only=final_kv_only,
                return_confidence=return_confidence,
            )
            return self._record_execution("draft", execution, output)
        tree_fia_modes = [
            bool(metadata.use_fused_infer_attention and getattr(metadata, "tree_attention", False))
            for metadata in attention_metadatas
        ]
        if any(tree_fia_modes) != all(tree_fia_modes):
            raise ValueError("Every draft graph step must use the same tree or linear FIA contract.")
        if stable_task_barrier:
            if not all(attention_modes):
                raise ValueError("A draft stable-task barrier is restricted to fused infer attention entries.")
            length_signatures = tuple(
                (
                    tuple(metadata.actual_seq_lengths_q),
                    tuple(metadata.sequence_lens),
                )
                for metadata in attention_metadatas
            )
            if not length_signatures or any(signature != length_signatures[0] for signature in length_signatures[1:]):
                raise ValueError(
                    "A draft stable-task barrier requires one fixed common-KV "
                    "FIA length signature across every proposal step."
                )
            stable_sequence_lens = length_signatures[0][1]
            if not stable_sequence_lens or any(
                length != stable_sequence_lens[0] for length in stable_sequence_lens[1:]
            ):
                raise ValueError("A draft stable-task barrier requires one common KV capacity across every graph row.")
        if all(tree_fia_modes):
            block_size = self.model.layers[0].self_attn.block_size
            for metadata in attention_metadatas:
                _validate_tree_fia_mask(
                    input_row_count,
                    getattr(metadata, "tree_attention_mask", None),
                    metadata.request_block_tables,
                    metadata.actual_seq_lengths_q,
                    metadata.sequence_lens,
                    block_size,
                )
            shape_signature = tuple(
                (
                    tuple(metadata.tree_attention_mask.shape),
                    tuple(metadata.request_block_tables.shape),
                )
                for metadata in attention_metadatas
            )
            attention_key = f"tree-fia-full:{shape_signature}"
        elif all(attention_modes):
            query_shape = ",".join(str(length) for length in attention_metadatas[0].actual_seq_lengths_q)
            attention_key = f"fia:{query_shape}"
        else:
            attention_key = "paged"
        final_kv_key = "|final-kv" if final_kv_only else ""
        confidence_key = "|confidence" if return_confidence else ""
        lane_key = "" if graph_lane is None else f"|lane:{graph_lane}"
        barrier_key = f"|stable-task-barrier|kv-cap:{stable_sequence_lens[0]}" if stable_task_barrier else ""
        entry_key = (
            f"draft-greedy:{vocabulary_size}|steps:{len(positions)}|"
            f"{attention_key}{final_kv_key}{confidence_key}{lane_key}{barrier_key}",
            input_ids.shape[0],
        )
        self.last_draft_entry_key = entry_key
        if entry_key in self.disabled_entry_keys:
            self._raise_if_graph_cache_sealed("draft", "disabled_entry", entry_key)
            execution = NativeGraphExecution("eager", "disabled_entry")
            output = self._execute_draft(
                input_ids,
                positions,
                attention_metadatas,
                vocabulary_size,
                final_kv_only=final_kv_only,
                return_confidence=return_confidence,
            )
            return self._record_execution("draft", execution, output)
        entry = self.draft_entries.get(entry_key)
        if entry is None:
            self._raise_if_graph_cache_sealed("draft", "missing_entry", entry_key)
            if (
                len(self.entries) + len(self.draft_entries) + len(self.target_entries) >= self.max_graph_entries
                or self._capture_budget_used >= self.max_graph_entries
            ):
                self.capacity_fallback_count += 1
                execution = NativeGraphExecution("eager", "entry_capacity")
                output = self._execute_draft(
                    input_ids,
                    positions,
                    attention_metadatas,
                    vocabulary_size,
                    final_kv_only=final_kv_only,
                    return_confidence=return_confidence,
                )
                return self._record_execution("draft", execution, output)
            self.capture_attempt_count += 1
            self._capture_budget_used += 1
            output = self._capture_draft(
                entry_key,
                input_ids,
                positions,
                attention_metadatas,
                vocabulary_size,
                valid_row_count=valid_row_count,
                final_kv_only=final_kv_only,
                return_confidence=return_confidence,
                stable_task_barrier=stable_task_barrier,
            )
            return self._record_execution("draft", self.last_draft_execution, output)
        changed_input = not entry.runtime_validated and self._draft_inputs_changed(
            entry, input_ids, positions, attention_metadatas
        )
        task_lengths_unchanged = entry.actual_seq_lengths_q == tuple(
            metadata.actual_seq_lengths_q for metadata in attention_metadatas
        ) and entry.sequence_lens == tuple(metadata.sequence_lens for metadata in attention_metadatas)
        entry_uses_stable_task_barrier = getattr(entry, "stable_task_barrier", False) is True
        if entry_uses_stable_task_barrier != bool(stable_task_barrier):
            raise RuntimeError("Draft ACLGraph stable-task barrier mode changed for a resident entry.")
        if entry_uses_stable_task_barrier and not task_lengths_unchanged:
            raise RuntimeError(
                "Draft ACLGraph stable-task barrier rejected changed attention "
                "task lengths before input copy or graph submission."
            )
        validated_real_row_count = getattr(entry, "validated_real_row_count", input_row_count)
        if not isinstance(validated_real_row_count, int):
            validated_real_row_count = input_row_count
        logical_row_expansion = valid_row_count > validated_real_row_count
        if not entry.runtime_validated:
            self._raise_if_graph_cache_sealed("draft", "unvalidated_entry", entry_key)
        if logical_row_expansion:
            self._raise_if_graph_cache_sealed("draft", "logical_row_expansion", entry_key)
        current_stream = torch.npu.current_stream()
        inline_update = envs.VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE
        replay_first_update = envs.VLLM_ASCEND_PEARL_DRAFT_REPLAY_FIRST_TASK_UPDATE
        taskless_device_attention = self._validate_draft_attention_task_contract(entry)
        if not taskless_device_attention and inline_update and replay_first_update:
            raise RuntimeError(
                "VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE and "
                "VLLM_ASCEND_PEARL_DRAFT_REPLAY_FIRST_TASK_UPDATE are "
                "mutually exclusive."
            )
        if not taskless_device_attention:
            assert self.update_stream is not None
        copy_done_event = (
            self._draft_replay_first_copy_event(entry)
            if replay_first_update and not taskless_device_attention
            else None
        )
        self._copy_draft_inputs(entry, input_ids, positions, attention_metadatas)
        if taskless_device_attention:
            # Device-position PagedAttention has no host-valued CANN task ABI
            # and therefore captures no ExternalEvent/handle pairs.  Input
            # copies and graph replay share the current stream, so routing an
            # empty update through the auxiliary stream only adds a needless
            # cross-stream dependency to every serial draft window.
            entry.graph.replay()
            self.draft_taskless_replays += 1
        elif replay_first_update:
            assert copy_done_event is not None
            try:
                copy_done_event.record(current_stream)
                # Queue the dependency before graph submission, but deliberately
                # submit every task update afterwards.  The captured graph can
                # execute its prefix and then waits on each task's ExternalEvent,
                # matching vLLM-Ascend's production ACLGraph ordering.
                self.update_stream.wait_event(copy_done_event)
            except Exception as exc:
                raise RuntimeError(
                    "Draft ACLGraph replay-first dependency setup failed before graph submission."
                ) from exc
            try:
                entry.graph.replay()
            except Exception as exc:
                raise RuntimeError(
                    "Draft ACLGraph replay-first graph submission failed before attention-task update."
                ) from exc
            try:
                if task_lengths_unchanged:
                    if entry_uses_stable_task_barrier:
                        self._record_draft_stable_task_barrier(
                            entry,
                            self.update_stream,
                        )
                    else:
                        for task in entry.tasks:
                            task.event.record(self.update_stream)
                    self.task_update_skip_replay_count += 1
                else:
                    self._update_draft_attention_tasks(entry)
            except Exception as exc:
                # The submitted graph may be waiting on one of its captured
                # ExternalEvents.  Continuing with eager execution could expose
                # partial KV writes or stale output, so fail the worker loudly.
                raise RuntimeError(
                    "Draft ACLGraph replay-first attention-task update failed "
                    "after graph submission; the runner cannot safely fall back "
                    "for this replay."
                ) from exc
        elif inline_update:
            if task_lengths_unchanged:
                if entry_uses_stable_task_barrier:
                    self._record_draft_stable_task_barrier(
                        entry,
                        current_stream,
                    )
                else:
                    for task in entry.tasks:
                        task.event.record(current_stream)
                self.task_update_skip_replay_count += 1
            else:
                self._update_draft_attention_tasks(entry, stream=current_stream)
        else:
            self.update_stream.wait_stream(current_stream)
            if task_lengths_unchanged:
                if entry_uses_stable_task_barrier:
                    self._record_draft_stable_task_barrier(
                        entry,
                        self.update_stream,
                    )
                else:
                    for task in entry.tasks:
                        task.event.record(self.update_stream)
                self.task_update_skip_replay_count += 1
            else:
                self._update_draft_attention_tasks(entry)
            # Match the target replay contract: graph replay must not race an
            # unfinished task update on the auxiliary stream.  The captured
            # ExternalEvents order each attention task, while this dependency
            # also protects CANN's graph-task metadata between consecutive
            # multi-step draft replays.
            current_stream.wait_stream(self.update_stream)
        if not replay_first_update and not taskless_device_attention:
            entry.graph.replay()
        if not task_lengths_unchanged and not taskless_device_attention:
            self._record_task_update("draft", len(entry.tasks))
        self.replay_count += 1
        execution = NativeGraphExecution("replay", replay_executed=True)
        if changed_input or logical_row_expansion:
            torch.npu.synchronize()
            graph_output = entry.output.clone()
            reference_output = self._execute_draft(
                input_ids,
                positions,
                attention_metadatas,
                vocabulary_size,
                final_kv_only=final_kv_only,
                return_confidence=return_confidence,
            )
            torch.npu.synchronize()
            validation_failed = not self._draft_outputs_match(
                graph_output,
                reference_output,
                (attention_metadatas[:-1] if final_kv_only else attention_metadatas),
                valid_row_count,
            )
            self._record_runtime_validation(
                "draft",
                changed_input=changed_input,
                logical_row_expansion=logical_row_expansion,
                failed=validation_failed,
            )
            if validation_failed:
                failed_entry = self.draft_entries.pop(entry_key)
                self._reset_graph_entry(failed_entry)
                self.disabled_entry_keys.add(entry_key)
                self.failed_capture_count += 1
                execution = NativeGraphExecution(
                    "eager",
                    "runtime_validation",
                    replay_executed=True,
                )
                warnings.warn(
                    "Native PEARL disabled a gamma-step draft ACLGraph whose "
                    "changed-input replay did not match eager execution: "
                    f"{entry_key!r}; "
                    f"{self._mismatch_summary(graph_output, reference_output)}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self._record_execution("draft", execution, reference_output)
            entry.runtime_validated = True
            entry.validated_real_row_count = max(
                validated_real_row_count,
                valid_row_count,
            )
        return self._record_execution("draft", execution, entry.output)

    def run_target_greedy(
        self,
        input_ids: list[torch.Tensor],
        positions: list[torch.Tensor],
        attention_metadatas: list[Any],
        vocabulary_size: int,
    ) -> tuple[torch.Tensor, ...]:
        """Replay fixed draft-token inputs as causal one-token target steps."""
        return self._run_target(input_ids, positions, attention_metadatas, vocabulary_size)

    def run_target_logits(
        self,
        input_ids: list[torch.Tensor],
        positions: list[torch.Tensor],
        attention_metadatas: list[Any],
        vocabulary_size: int,
    ) -> tuple[torch.Tensor, ...]:
        """Replay a target-only step while retaining logits for sampling."""
        return self._run_target(
            input_ids,
            positions,
            attention_metadatas,
            vocabulary_size,
            output_kind="logits",
        )

    def run_tree_logits(self, input_ids, positions, metadata, vocabulary_size: int) -> torch.Tensor:
        """Capture a batched draft tree level, including its vocabulary head."""
        return self._run_target([input_ids], [positions], [metadata], vocabulary_size, output_kind="logits")[0]

    def run_tree_hidden(self, input_ids, positions, metadata) -> torch.Tensor:
        """Capture the final tree KV materialization without an unused head."""
        return self._run_target([input_ids], [positions], [metadata], 0, output_kind="hidden")[0]

    def _run_target(
        self,
        input_ids,
        positions,
        attention_metadatas,
        vocabulary_size: int,
        *,
        output_kind: str = "greedy",
    ) -> tuple[torch.Tensor, ...]:
        self.last_target_execution = NativeGraphExecution()
        if not input_ids or not (len(input_ids) == len(positions) == len(attention_metadatas)):
            raise ValueError("A target ACLGraph requires matching non-empty step inputs and metadata.")
        if not self.enabled:
            self._raise_if_graph_cache_sealed("target", "disabled")
            execution = NativeGraphExecution("eager", "disabled")
            output = self._execute_target(
                input_ids,
                positions,
                attention_metadatas,
                vocabulary_size,
                output_kind=output_kind,
            )
            return self._record_execution("target", execution, output)
        attention_modes = [metadata.use_fused_infer_attention for metadata in attention_metadatas]
        if any(attention_modes) != all(attention_modes):
            raise ValueError("Every step in a target ACLGraph must use the same attention backend.")
        step_sizes = tuple(int(value.shape[0]) for value in input_ids)
        if max(step_sizes) > self.max_graph_tokens:
            self._raise_if_graph_cache_sealed("target", "token_capacity")
            self.shape_fallback_count += 1
            execution = NativeGraphExecution("eager", "token_capacity")
            output = self._execute_target(
                input_ids,
                positions,
                attention_metadatas,
                vocabulary_size,
                output_kind=output_kind,
            )
            return self._record_execution("target", execution, output)
        reference_metadatas = attention_metadatas
        tree_fia_modes = [
            bool(metadata.use_fused_infer_attention and getattr(metadata, "tree_attention", False))
            for metadata in attention_metadatas
        ]
        if any(tree_fia_modes) != all(tree_fia_modes):
            raise ValueError("Every target graph step must use the same tree or linear FIA contract")
        if all(tree_fia_modes):
            block_size = self.model.layers[0].self_attn.block_size
            for size, metadata in zip(step_sizes, attention_metadatas):
                _validate_tree_fia_mask(
                    size,
                    getattr(metadata, "tree_attention_mask", None),
                    metadata.request_block_tables,
                    metadata.actual_seq_lengths_q,
                    metadata.sequence_lens,
                    block_size,
                )
            shape_signature = tuple(
                (
                    tuple(metadata.tree_attention_mask.shape),
                    tuple(metadata.request_block_tables.shape),
                )
                for metadata in attention_metadatas
            )
            # Query/KV *values* and both exact length lists are updated through
            # the captured FIA tasks.  Do not round KV lengths to a power-of-two
            # bucket: on Ascend BF16 FULL-mask FIA, appending masked KV columns
            # can change reduction order enough to change a greedy token.  The
            # query partition is a host literal rather than a tensor extent, so
            # it must not split otherwise identical graphs either.  Fixed input
            # and mask shapes are the complete capture contract; changed query
            # partitions or KV lengths take the task-update path below.
            attention_key = f"tree-fia-full:exact-length-task-update|buffers:{shape_signature}"
        elif all(attention_modes):
            # Packed causal FIA can have the same total query-token count but
            # a different number of request segments.  Those segments own one
            # page-table row each, so reusing an entry keyed only by token
            # count tries to copy e.g. [32, blocks] into a captured
            # [8, blocks] buffer.  Length values remain task-update arguments;
            # only graph-owned buffer shapes belong in this key.
            shape_signature = tuple(
                (
                    tuple(metadata.request_block_tables.shape) if metadata.request_block_tables is not None else None,
                    tuple(metadata.block_tables.shape),
                    tuple(metadata.attention_mask.shape) if metadata.attention_mask is not None else None,
                )
                for metadata in attention_metadatas
            )
            attention_key = f"fia:exact-length-task-update|buffers:{shape_signature}"
        elif any(metadata.attention_mask is not None for metadata in attention_metadatas):
            if not all(metadata.attention_mask is not None for metadata in attention_metadatas):
                raise ValueError("Every target tree ACLGraph step must provide an attention mask")
            # The dense model reads host context lengths when tracing arange.
            # Change both its actual physical extent and the mask before
            # simplifying the key; removing lengths from the key alone is
            # incorrect. Model-side masked-KV sanitization makes tail padding
            # safe even when uninitialized cache storage contains NaNs.
            block_size = self.model.layers[0].self_attn.block_size
            attention_metadatas = [
                bucket_tree_attention_metadata(metadata, block_size=block_size) for metadata in attention_metadatas
            ]
            context_signature = tuple(int(metadata.context_lens[0]) for metadata in attention_metadatas)
            shape_signature = tuple(
                (tuple(metadata.attention_mask.shape), tuple(metadata.block_tables.shape))
                for metadata in attention_metadatas
            )
            attention_key = f"tree-bucket:{context_signature}|buffers:{shape_signature}"
        else:
            attention_key = "paged"
        entry_key = (
            f"target-{output_kind}:{vocabulary_size}|steps:{len(input_ids)}|{attention_key}",
            step_sizes,
        )
        if entry_key in self.disabled_entry_keys:
            self._raise_if_graph_cache_sealed("target", "disabled_entry", entry_key)
            execution = NativeGraphExecution("eager", "disabled_entry")
            output = self._execute_target(
                input_ids,
                positions,
                reference_metadatas,
                vocabulary_size,
                output_kind=output_kind,
            )
            return self._record_execution("target", execution, output)
        entry = self.target_entries.get(entry_key)
        if entry is None:
            self._raise_if_graph_cache_sealed("target", "missing_entry", entry_key)
            total_entries = len(self.entries) + len(self.draft_entries) + len(self.target_entries)
            if total_entries >= self.max_graph_entries or self._capture_budget_used >= self.max_graph_entries:
                self.capacity_fallback_count += 1
                execution = NativeGraphExecution("eager", "entry_capacity")
                output = self._execute_target(
                    input_ids,
                    positions,
                    reference_metadatas,
                    vocabulary_size,
                    output_kind=output_kind,
                )
                return self._record_execution("target", execution, output)
            self.capture_attempt_count += 1
            self._capture_budget_used += 1
            output = self._capture_target(
                entry_key,
                input_ids,
                positions,
                attention_metadatas,
                vocabulary_size,
                reference_metadatas=reference_metadatas,
                output_kind=output_kind,
            )
            return self._record_execution(
                "target",
                self.last_target_execution,
                output,
                entry_key=entry_key,
                token_rows=step_sizes,
            )
        if not entry.runtime_validated:
            self._raise_if_graph_cache_sealed("target", "unvalidated_entry", entry_key)
        task_lengths_unchanged = entry.actual_seq_lengths_q == tuple(
            metadata.actual_seq_lengths_q for metadata in attention_metadatas
        ) and entry.sequence_lens == tuple(metadata.sequence_lens for metadata in attention_metadatas)
        changed_input = not entry.runtime_validated and self._target_inputs_changed(
            entry, input_ids, positions, attention_metadatas
        )
        self._copy_target_inputs(entry, input_ids, positions, attention_metadatas)
        assert self.update_stream is not None
        current_stream = torch.npu.current_stream()
        if task_lengths_unchanged:
            # Graph-owned input buffers were copied on current_stream. FIA's
            # host literal lengths are unchanged, so rebuilding every layer's
            # task is redundant. Refresh only the captured external events.
            for task in entry.tasks:
                task.event.record(current_stream)
            self.task_update_skip_replay_count += 1
        else:
            self.update_stream.wait_stream(current_stream)
            self._update_target_attention_tasks(entry)
            current_stream.wait_stream(self.update_stream)
            self._record_task_update("target", len(entry.tasks))
        entry.graph.replay()
        self.replay_count += 1
        execution = NativeGraphExecution("replay", replay_executed=True)
        if changed_input:
            torch.npu.synchronize()
            graph_outputs = tuple(value.clone() for value in entry.outputs)
            reference_outputs = self._execute_target(
                input_ids,
                positions,
                reference_metadatas,
                vocabulary_size,
                output_kind=output_kind,
            )
            torch.npu.synchronize()
            validation_failed = not self._target_outputs_match(graph_outputs, reference_outputs)
            self._record_runtime_validation(
                "target",
                changed_input=True,
                failed=validation_failed,
            )
            if validation_failed:
                reference_outputs = self._diagnose_target_mismatch(
                    graph_outputs,
                    reference_outputs,
                    input_ids,
                    positions,
                    attention_metadatas,
                    reference_metadatas,
                    vocabulary_size,
                    output_kind=output_kind,
                    entry_key=entry_key,
                )
                failed_entry = self.target_entries.pop(entry_key)
                self._reset_graph_entry(failed_entry)
                self.disabled_entry_keys.add(entry_key)
                self.failed_capture_count += 1
                execution = NativeGraphExecution("eager", "runtime_validation", replay_executed=True)
                warnings.warn(
                    "Native PEARL disabled a target ACLGraph whose runtime replay did not match eager execution: "
                    f"{entry_key!r}; diagnostics={json.dumps(self.last_target_validation_error)}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self._record_execution(
                    "target",
                    execution,
                    reference_outputs,
                    entry_key=entry_key,
                    token_rows=step_sizes,
                )
            entry.runtime_validated = True
        return self._record_execution(
            "target",
            execution,
            entry.outputs,
            entry_key=entry_key,
            token_rows=step_sizes,
        )

    def _execute_target(
        self,
        input_ids: list[torch.Tensor] | tuple[torch.Tensor, ...],
        positions: list[torch.Tensor] | tuple[torch.Tensor, ...],
        attention_metadatas: list[Any],
        vocabulary_size: int,
        *,
        output_kind: str = "greedy",
    ) -> tuple[torch.Tensor, ...]:
        outputs = []
        for step_input, step_positions, metadata in zip(input_ids, positions, attention_metadatas):
            hidden_states = self.model(step_input, step_positions, metadata)
            if output_kind == "hidden":
                outputs.append(hidden_states)
            elif output_kind == "logits":
                outputs.append(self.model.compute_logits(hidden_states)[:, :vocabulary_size])
            else:
                outputs.append(self.model.compute_greedy_tokens(hidden_states, vocabulary_size))
        return tuple(outputs)

    def _capture_target(
        self,
        entry_key: tuple[str, tuple[int, ...]],
        input_ids: list[torch.Tensor],
        positions: list[torch.Tensor],
        attention_metadatas: list[Any],
        vocabulary_size: int,
        *,
        reference_metadatas: list[Any] | None = None,
        output_kind: str = "greedy",
    ) -> tuple[torch.Tensor, ...]:
        reference_outputs = tuple(
            value.clone()
            for value in self._execute_target(
                input_ids,
                positions,
                reference_metadatas if reference_metadatas is not None else attention_metadatas,
                vocabulary_size,
                output_kind=output_kind,
            )
        )
        torch.npu.synchronize()
        captured_input_ids = tuple(value.clone() for value in input_ids)
        captured_positions = tuple(value.clone() for value in positions)
        captured_metadatas = []
        captured_attention_masks: list[torch.Tensor | None] = []
        captured_tree_attention_masks: list[torch.Tensor | None] = []
        for metadata in attention_metadatas:
            captured_request_block_tables = (
                metadata.request_block_tables.clone()
                if metadata.use_fused_infer_attention and metadata.request_block_tables is not None
                else None
            )
            tree_mask = getattr(metadata, "tree_attention_mask", None)
            clone_fields = {
                "slot_mapping": metadata.slot_mapping.clone(),
                "context_lens": metadata.context_lens.clone(),
                "block_tables": metadata.block_tables.clone(),
                "request_block_tables": captured_request_block_tables,
                "attention_mask": metadata.attention_mask.clone() if metadata.attention_mask is not None else None,
            }
            if hasattr(metadata, "tree_attention_mask"):
                clone_fields["tree_attention_mask"] = tree_mask.clone() if tree_mask is not None else None
            # Preserve mode/other immutable fields, but never let a graph task
            # keep a caller-owned literal mask whose next contents may change.
            captured_metadatas.append(replace(metadata, **clone_fields))
            captured_attention_masks.append(captured_metadatas[-1].attention_mask)
            captured_tree_attention_masks.append(getattr(captured_metadatas[-1], "tree_attention_mask", None))
        graph = torch.npu.NPUGraph()
        with (
            _collect_graph_tasks(
                fia_workspaces=self._fia_workspace_pool,
            ) as tasks,
            torch.npu.graph(graph, pool=self.graph_pool),
        ):
            outputs = self._execute_target(
                captured_input_ids,
                captured_positions,
                captured_metadatas,
                vocabulary_size,
                output_kind=output_kind,
            )
        if len(tasks) % len(captured_metadatas):
            raise RuntimeError("Target ACLGraph attention tasks do not divide evenly across verification steps.")
        tasks_per_step = len(tasks) // len(captured_metadatas)
        entry = NativeTargetACLGraphEntry(
            input_ids=captured_input_ids,
            positions=captured_positions,
            slot_mappings=tuple(metadata.slot_mapping for metadata in captured_metadatas),
            context_lens=tuple(metadata.context_lens for metadata in captured_metadatas),
            block_tables=tuple(metadata.block_tables for metadata in captured_metadatas),
            request_block_tables=tuple(metadata.request_block_tables for metadata in captured_metadatas),
            attention_masks=tuple(captured_attention_masks),
            actual_seq_lengths_q=tuple(metadata.actual_seq_lengths_q for metadata in captured_metadatas),
            sequence_lens=tuple(metadata.sequence_lens for metadata in captured_metadatas),
            graph=graph,
            outputs=outputs,
            tasks=tasks,
            tasks_per_step=tasks_per_step,
            tree_attention_masks=tuple(captured_tree_attention_masks),
            tree_attention_modes=tuple(
                bool(metadata.use_fused_infer_attention and getattr(metadata, "tree_attention", False))
                for metadata in captured_metadatas
            ),
            # Capture validation covers only the captured values.  The first
            # replay whose graph-owned inputs actually change is qualified
            # independently before this entry becomes a production hot path.
            runtime_validated=False,
        )
        self.target_entries[entry_key] = entry
        current_stream = torch.npu.current_stream()
        current_stream.synchronize()
        assert self.update_stream is not None
        self.update_stream.wait_stream(current_stream)
        self._update_target_attention_tasks(entry)
        # FULL-mask FIA may enqueue thousands of target tasks.  Relying only
        # on the ExternalEvent embedded in every captured attention task can
        # let the first replay race the task-update stream on CANN/TP3.  The
        # steady-state replay path already carries this reverse dependency;
        # make capture validation obey the identical ordering contract.
        current_stream.wait_stream(self.update_stream)
        entry.graph.replay()
        torch.npu.synchronize()
        if not self._target_outputs_match(outputs, reference_outputs):
            reference_outputs = self._diagnose_target_mismatch(
                tuple(value.clone() for value in outputs),
                reference_outputs,
                input_ids,
                positions,
                attention_metadatas,
                reference_metadatas if reference_metadatas is not None else attention_metadatas,
                vocabulary_size,
                output_kind=output_kind,
                entry_key=entry_key,
            )
            failed_entry = self.target_entries.pop(entry_key)
            self._reset_graph_entry(failed_entry)
            self.disabled_entry_keys.add(entry_key)
            self.failed_capture_count += 1
            self.last_target_execution = NativeGraphExecution(
                "eager", "capture_validation", capture_attempted=True, replay_executed=True
            )
            warnings.warn(
                "Native PEARL disabled a target ACLGraph whose first replay did not match eager execution: "
                f"{entry_key!r}; diagnostics={json.dumps(self.last_target_validation_error)}",
                RuntimeWarning,
                stacklevel=2,
            )
            return reference_outputs
        self.capture_count += 1
        self.replay_count += 1
        self.last_target_execution = NativeGraphExecution(
            "capture_replay", capture_attempted=True, replay_executed=True
        )
        return outputs

    def _diagnose_target_mismatch(
        self,
        graph_outputs,
        reference_outputs,
        input_ids,
        positions,
        graph_metadatas,
        reference_metadatas,
        vocabulary_size,
        *,
        output_kind,
        entry_key,
    ):
        """Separate capture faults from context-padding numerical changes.

        This is only a failed-validation slow path. Keep the original strict
        tolerance: matching a bucketed reference does not authorize serving a
        result which differs from the unbucketed reference. Finally rerun the
        original eager path so fallback restores its KV writes as well as its
        returned output (returning an old hidden tensor alone is insufficient).
        """
        bucket_outputs = tuple(
            value.clone()
            for value in self._execute_target(
                input_ids,
                positions,
                graph_metadatas,
                vocabulary_size,
                output_kind=output_kind,
            )
        )
        torch.npu.synchronize()
        restored_outputs = self._execute_target(
            input_ids,
            positions,
            reference_metadatas,
            vocabulary_size,
            output_kind=output_kind,
        )
        torch.npu.synchronize()
        self.last_target_validation_error = {
            "rank": int(os.environ.get("RANK", "-1")),
            "output_kind": output_kind,
            "entry_key": repr(entry_key),
            "rtol": 1e-3,
            "atol": 1e-3,
            "original_context_lens": [metadata.context_lens.tolist() for metadata in reference_metadatas],
            "graph_context_lens": [metadata.context_lens.tolist() for metadata in graph_metadatas],
            "graph_vs_original": self._target_difference_summary(graph_outputs, reference_outputs),
            "graph_vs_same_bucket_eager": self._target_difference_summary(graph_outputs, bucket_outputs),
            "bucket_eager_vs_original": self._target_difference_summary(bucket_outputs, reference_outputs),
            "restored_eager_vs_original": self._target_difference_summary(restored_outputs, reference_outputs),
            "eager_kv_restored": True,
        }
        return restored_outputs

    @staticmethod
    def _target_difference_summary(outputs, references):
        summaries = []
        for output, reference in zip(outputs, references):
            summary = {
                "shape": list(output.shape),
                "reference_shape": list(reference.shape),
                "dtype": str(output.dtype),
                "reference_dtype": str(reference.dtype),
            }
            if output.shape == reference.shape:
                lhs, rhs = output.detach().float(), reference.detach().float()
                error = (lhs - rhs).abs()
                close = torch.isclose(lhs, rhs, rtol=1e-3, atol=1e-3)
                summary.update(
                    {
                        "elements": output.numel(),
                        "mismatch_elements": int((~close).sum().item()),
                        "nonfinite_output": int((~torch.isfinite(lhs)).sum().item()),
                        "nonfinite_reference": int((~torch.isfinite(rhs)).sum().item()),
                        "max_abs_error": float(error.max().item()) if error.numel() else 0.0,
                        "mean_abs_error": float(error.mean().item()) if error.numel() else 0.0,
                        "rmse": float(error.square().mean().sqrt().item()) if error.numel() else 0.0,
                    }
                )
                if error.ndim >= 2:
                    summary["per_row_max_abs_error_first32"] = error.flatten(1).amax(1)[:32].cpu().tolist()
            summaries.append(summary)
        return summaries

    @staticmethod
    def _target_inputs_changed(
        entry: NativeTargetACLGraphEntry,
        input_ids: list[torch.Tensor],
        positions: list[torch.Tensor],
        attention_metadatas: list[Any],
    ) -> bool:
        if not (len(entry.input_ids) == len(input_ids) == len(positions) == len(attention_metadatas)):
            return True
        for step, (step_input, step_positions, metadata) in enumerate(zip(input_ids, positions, attention_metadatas)):
            if (
                NativeACLGraphRunner._tensor_changed(entry.input_ids[step], step_input)
                or NativeACLGraphRunner._tensor_changed(entry.positions[step], step_positions)
                or NativeACLGraphRunner._tensor_changed(entry.slot_mappings[step], metadata.slot_mapping)
            ):
                return True
            captured_attention_mask = entry.attention_masks[step]
            incoming_attention_mask = getattr(metadata, "attention_mask", None)
            if (captured_attention_mask is None) != (incoming_attention_mask is None):
                return True
            if captured_attention_mask is not None and NativeACLGraphRunner._tensor_changed(
                captured_attention_mask, incoming_attention_mask
            ):
                return True
            tree_masks = getattr(entry, "tree_attention_masks", ())
            captured_tree_mask = tree_masks[step] if tree_masks else None
            incoming_tree_mask = getattr(metadata, "tree_attention_mask", None)
            if (captured_tree_mask is None) != (incoming_tree_mask is None):
                return True
            if captured_tree_mask is not None and NativeACLGraphRunner._tensor_changed(
                captured_tree_mask, incoming_tree_mask
            ):
                return True
            if entry.request_block_tables[step] is not None:
                if metadata.request_block_tables is None or NativeACLGraphRunner._tensor_changed(
                    entry.request_block_tables[step],
                    metadata.request_block_tables,
                ):
                    return True
            elif NativeACLGraphRunner._tensor_changed(
                entry.context_lens[step], metadata.context_lens
            ) or NativeACLGraphRunner._tensor_changed(entry.block_tables[step], metadata.block_tables):
                return True
            if (
                entry.actual_seq_lengths_q[step] != metadata.actual_seq_lengths_q
                or entry.sequence_lens[step] != metadata.sequence_lens
            ):
                return True
        return False

    @staticmethod
    def _copy_target_inputs(
        entry: NativeTargetACLGraphEntry,
        input_ids: list[torch.Tensor],
        positions: list[torch.Tensor],
        attention_metadatas: list[Any],
    ) -> None:
        # Check the entire mask contract before copying the first buffer.
        for step, metadata in enumerate(attention_metadatas):
            modes = getattr(entry, "tree_attention_modes", ())
            mode = bool(metadata.use_fused_infer_attention and getattr(metadata, "tree_attention", False))
            if modes and modes[step] != mode:
                raise RuntimeError("Target ACLGraph tree/linear FIA contract changed between replays")
            masks = getattr(entry, "tree_attention_masks", ())
            captured_mask = masks[step] if masks else None
            incoming_mask = getattr(metadata, "tree_attention_mask", None)
            if (captured_mask is None) != (incoming_mask is None):
                raise RuntimeError("Target ACLGraph FULL tree mask appeared or disappeared between replays")
            if captured_mask is not None and (
                captured_mask.shape != incoming_mask.shape or captured_mask.dtype != incoming_mask.dtype
            ):
                raise RuntimeError("Target ACLGraph FULL tree mask shape or dtype changed between replays")
        for step, (step_input, step_positions, metadata) in enumerate(zip(input_ids, positions, attention_metadatas)):
            entry.input_ids[step].copy_(step_input)
            entry.positions[step].copy_(step_positions)
            entry.slot_mappings[step].copy_(metadata.slot_mapping)
            if entry.attention_masks[step] is not None:
                if metadata.attention_mask is None:
                    raise RuntimeError("Target ACLGraph requires a tree attention mask")
                if entry.attention_masks[step].shape != metadata.attention_mask.shape:
                    raise RuntimeError("Target ACLGraph tree mask shape changed between replays")
                entry.attention_masks[step].copy_(metadata.attention_mask)
            masks = getattr(entry, "tree_attention_masks", ())
            if masks and masks[step] is not None:
                masks[step].copy_(metadata.tree_attention_mask)
            if entry.request_block_tables[step] is not None:
                assert metadata.request_block_tables is not None
                entry.request_block_tables[step].copy_(metadata.request_block_tables)
            else:
                entry.context_lens[step].copy_(metadata.context_lens)
                entry.block_tables[step].copy_(metadata.block_tables)
        entry.actual_seq_lengths_q = tuple(metadata.actual_seq_lengths_q for metadata in attention_metadatas)
        entry.sequence_lens = tuple(metadata.sequence_lens for metadata in attention_metadatas)

    @staticmethod
    def _target_outputs_match(
        outputs: tuple[torch.Tensor, ...],
        reference_outputs: tuple[torch.Tensor, ...],
    ) -> bool:
        return len(outputs) == len(reference_outputs) and all(
            output.shape == reference.shape
            and output.dtype == reference.dtype
            and (
                torch.allclose(output, reference, rtol=1e-3, atol=1e-3)
                if output.dtype.is_floating_point
                else torch.equal(output, reference)
            )
            for output, reference in zip(outputs, reference_outputs)
        )

    def _execute_draft(
        self,
        input_ids: torch.Tensor,
        positions: list[torch.Tensor] | tuple[torch.Tensor, ...],
        attention_metadatas: list[Any],
        vocabulary_size: int,
        *,
        final_kv_only: bool = False,
        return_confidence: bool = False,
    ) -> torch.Tensor:
        outputs = []
        confidence_outputs = []
        step_input = input_ids
        final_step = len(positions) - 1
        for step, (step_positions, metadata) in enumerate(zip(positions, attention_metadatas)):
            hidden_states = self.model(step_input, step_positions, metadata)
            if final_kv_only and step == final_step:
                # Standard speculative decoding may need one extra draft
                # forward solely to materialize d_gamma in KV before a target
                # bonus token is committed.  Its logits are never consumed.
                # Keep that transformer/KV write in the graph but avoid a
                # 151K-vocabulary LM-head projection and argmax.
                continue
            if return_confidence:
                step_input, step_confidence = self.model.compute_greedy_tokens_with_confidence(
                    hidden_states,
                    vocabulary_size,
                )
                # A single long tensor keeps ACLGraph capture/replay and the
                # compact HCCL proposal on one fixed-shape output contract.
                # One-part-per-million precision is ample for scheduler
                # ranking and avoids a second graph output/collective.
                confidence_outputs.append(
                    torch.round(step_confidence.float().clamp(0.0, 1.0) * 1_000_000).to(torch.long)
                )
            else:
                step_input = self.model.compute_greedy_tokens(hidden_states, vocabulary_size)
            outputs.append(step_input)
        token_output = torch.stack(outputs, dim=1)
        if not return_confidence:
            return token_output
        return torch.stack(
            (token_output, torch.stack(confidence_outputs, dim=1)),
            dim=-1,
        )

    @staticmethod
    def _draft_outputs_match(
        graph_output: torch.Tensor,
        reference_output: torch.Tensor,
        attention_metadatas: list[Any],
        valid_row_count: int,
    ) -> bool:
        """Compare the caller-declared real-row prefix at every draft step.

        ``NativeACLGraphRunner._pad_inputs`` appends dummy rows and marks their
        slot mappings negative.  The logical row count must come from the
        caller rather than being inferred from those mappings: otherwise a
        real row that is consistently (and incorrectly) mapped to ``-1``
        could be hidden from validation.  Every step therefore has to expose
        an exact non-negative prefix followed by a negative padding suffix.
        """
        if (
            graph_output.shape != reference_output.shape
            or graph_output.dtype != reference_output.dtype
            or graph_output.ndim not in (2, 3)
            or (graph_output.ndim == 3 and graph_output.shape[2] != 2)
            or len(attention_metadatas) != graph_output.shape[1]
        ):
            return False
        row_count = graph_output.shape[0]
        if valid_row_count <= 0 or valid_row_count > row_count:
            return False
        for metadata in attention_metadatas:
            slot_mapping = getattr(metadata, "slot_mapping", None)
            if not torch.is_tensor(slot_mapping) or slot_mapping.ndim != 1 or slot_mapping.numel() != row_count:
                return False
            if not bool(torch.all(slot_mapping[:valid_row_count] >= 0).item()):
                return False
            if valid_row_count < row_count and not bool(torch.all(slot_mapping[valid_row_count:] < 0).item()):
                return False
        graph_valid = graph_output[:valid_row_count]
        reference_valid = reference_output[:valid_row_count]
        if graph_valid.dtype.is_floating_point:
            return torch.allclose(graph_valid, reference_valid, rtol=1e-3, atol=1e-3)
        return torch.equal(graph_valid, reference_valid)

    def _capture_draft(
        self,
        entry_key: tuple[str, int],
        input_ids: torch.Tensor,
        positions: list[torch.Tensor],
        attention_metadatas: list[Any],
        vocabulary_size: int,
        *,
        valid_row_count: int,
        final_kv_only: bool = False,
        return_confidence: bool = False,
        stable_task_barrier: bool = False,
    ) -> torch.Tensor:
        reference_output = self._execute_draft(
            input_ids,
            positions,
            attention_metadatas,
            vocabulary_size,
            final_kv_only=final_kv_only,
            return_confidence=return_confidence,
        ).clone()
        torch.npu.synchronize()
        captured_input_ids = input_ids.clone()
        captured_positions = tuple(value.clone() for value in positions)
        captured_metadatas = []
        captured_attention_masks: list[torch.Tensor | None] = []
        captured_tree_attention_masks: list[torch.Tensor | None] = []
        captured_request_table_aliases: dict[int, torch.Tensor] = {}
        for metadata in attention_metadatas:
            captured_request_block_tables = None
            if metadata.use_fused_infer_attention and metadata.request_block_tables is not None:
                incoming_table_id = id(metadata.request_block_tables)
                captured_request_block_tables = captured_request_table_aliases.get(incoming_table_id)
                if captured_request_block_tables is None:
                    captured_request_block_tables = metadata.request_block_tables.clone()
                    captured_request_table_aliases[incoming_table_id] = captured_request_block_tables
            captured_attention_mask = metadata.attention_mask.clone() if metadata.attention_mask is not None else None
            tree_attention_mask = getattr(
                metadata,
                "tree_attention_mask",
                None,
            )
            captured_tree_attention_mask = tree_attention_mask.clone() if tree_attention_mask is not None else None
            clone_fields = {
                "slot_mapping": metadata.slot_mapping.clone(),
                "context_lens": metadata.context_lens.clone(),
                "block_tables": metadata.block_tables.clone(),
                "actual_seq_lengths_q": metadata.actual_seq_lengths_q,
                "sequence_lens": metadata.sequence_lens,
                "request_block_tables": captured_request_block_tables,
                "attention_mask": captured_attention_mask,
                "use_fused_infer_attention": metadata.use_fused_infer_attention,
            }
            if hasattr(metadata, "tree_attention"):
                clone_fields["tree_attention"] = bool(metadata.tree_attention)
            if hasattr(metadata, "tree_attention_mask"):
                clone_fields["tree_attention_mask"] = captured_tree_attention_mask
            captured_metadatas.append(
                replace(metadata, **clone_fields)
                if hasattr(metadata, "__dataclass_fields__")
                else type(metadata)(**clone_fields)
            )
            captured_attention_masks.append(captured_attention_mask)
            captured_tree_attention_masks.append(captured_tree_attention_mask)
        graph = torch.npu.NPUGraph()
        with (
            _collect_graph_tasks(
                fia_workspaces=self._fia_workspace_pool,
                stable_task_barrier=stable_task_barrier,
            ) as tasks,
            torch.npu.graph(graph, pool=self.graph_pool),
        ):
            output = self._execute_draft(
                captured_input_ids,
                list(captured_positions),
                captured_metadatas,
                vocabulary_size,
                final_kv_only=final_kv_only,
                return_confidence=return_confidence,
            )
        if len(tasks) % len(captured_metadatas):
            raise RuntimeError("Draft ACLGraph attention tasks do not divide evenly across proposal steps.")
        tasks_per_step = len(tasks) // len(captured_metadatas)
        taskless_device_attention = self._draft_uses_device_position_attention(captured_metadatas)
        if taskless_device_attention and tasks:
            graph.reset()
            raise RuntimeError(
                "Device-position PagedAttention draft capture unexpectedly produced refreshable attention tasks."
            )
        stable_task_barrier_event = None
        if stable_task_barrier:
            if not tasks:
                raise RuntimeError("A draft stable-task barrier capture produced no attention tasks.")
            stable_task_barrier_event = tasks[0].event
            if any(task.event is not stable_task_barrier_event for task in tasks):
                raise RuntimeError(
                    "A draft stable-task barrier capture did not share exactly "
                    "one ExternalEvent across its attention tasks."
                )
        entry = NativeDraftACLGraphEntry(
            input_ids=captured_input_ids,
            positions=captured_positions,
            slot_mappings=tuple(metadata.slot_mapping for metadata in captured_metadatas),
            context_lens=tuple(metadata.context_lens for metadata in captured_metadatas),
            block_tables=tuple(metadata.block_tables for metadata in captured_metadatas),
            request_block_tables=tuple(metadata.request_block_tables for metadata in captured_metadatas),
            actual_seq_lengths_q=tuple(metadata.actual_seq_lengths_q for metadata in captured_metadatas),
            sequence_lens=tuple(metadata.sequence_lens for metadata in captured_metadatas),
            graph=graph,
            output=output,
            tasks=tasks,
            tasks_per_step=tasks_per_step,
            attention_masks=tuple(captured_attention_masks),
            tree_attention_masks=tuple(captured_tree_attention_masks),
            tree_attention_modes=tuple(
                bool(metadata.use_fused_infer_attention and getattr(metadata, "tree_attention", False))
                for metadata in captured_metadatas
            ),
            validated_real_row_count=valid_row_count,
            stable_task_barrier=stable_task_barrier,
            stable_task_barrier_event=stable_task_barrier_event,
            taskless_device_attention=taskless_device_attention,
        )
        self.draft_entries[entry_key] = entry
        current_stream = torch.npu.current_stream()
        current_stream.synchronize()
        if not taskless_device_attention:
            assert self.update_stream is not None
            if envs.VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE:
                if stable_task_barrier:
                    self._refresh_draft_stable_task_barrier(
                        entry,
                        stream=current_stream,
                    )
                else:
                    self._update_draft_attention_tasks(entry, stream=current_stream)
            else:
                self.update_stream.wait_stream(current_stream)
                if stable_task_barrier:
                    self._refresh_draft_stable_task_barrier(entry)
                else:
                    self._update_draft_attention_tasks(entry)
                current_stream.wait_stream(self.update_stream)
        entry.graph.replay()
        if taskless_device_attention:
            self.draft_taskless_replays += 1
        torch.npu.synchronize()
        if not self._draft_outputs_match(
            output,
            reference_output,
            (captured_metadatas[:-1] if final_kv_only else captured_metadatas),
            valid_row_count,
        ):
            failed_entry = self.draft_entries.pop(entry_key)
            self._reset_graph_entry(failed_entry)
            self.disabled_entry_keys.add(entry_key)
            self.failed_capture_count += 1
            self.last_draft_execution = NativeGraphExecution(
                "eager",
                "capture_validation",
                capture_attempted=True,
                replay_executed=True,
            )
            warnings.warn(
                "Native PEARL disabled a gamma-step draft ACLGraph whose replay did not match eager execution: "
                f"{entry_key!r}; {self._mismatch_summary(output, reference_output)}",
                RuntimeWarning,
                stacklevel=2,
            )
            return reference_output
        self.capture_count += 1
        self.replay_count += 1
        self.last_draft_execution = NativeGraphExecution(
            "capture_replay",
            capture_attempted=True,
            replay_executed=True,
        )
        return output

    @staticmethod
    def _draft_inputs_changed(
        entry: NativeDraftACLGraphEntry,
        input_ids: torch.Tensor,
        positions: list[torch.Tensor],
        attention_metadatas: list[Any],
    ) -> bool:
        if NativeACLGraphRunner._tensor_changed(entry.input_ids, input_ids):
            return True
        if len(entry.positions) != len(positions):
            return True
        for step, (step_positions, metadata) in enumerate(zip(positions, attention_metadatas)):
            if NativeACLGraphRunner._tensor_changed(
                entry.positions[step], step_positions
            ) or NativeACLGraphRunner._tensor_changed(entry.slot_mappings[step], metadata.slot_mapping):
                return True
            if entry.request_block_tables[step] is not None:
                if metadata.request_block_tables is None or NativeACLGraphRunner._tensor_changed(
                    entry.request_block_tables[step], metadata.request_block_tables
                ):
                    return True
            elif NativeACLGraphRunner._tensor_changed(
                entry.context_lens[step], metadata.context_lens
            ) or NativeACLGraphRunner._tensor_changed(entry.block_tables[step], metadata.block_tables):
                return True
            attention_masks = getattr(entry, "attention_masks", ())
            captured_mask = attention_masks[step] if attention_masks else None
            incoming_mask = getattr(metadata, "attention_mask", None)
            if (captured_mask is None) != (incoming_mask is None):
                return True
            if captured_mask is not None and NativeACLGraphRunner._tensor_changed(captured_mask, incoming_mask):
                return True
            tree_modes = getattr(entry, "tree_attention_modes", ())
            incoming_tree_mode = bool(
                getattr(metadata, "use_fused_infer_attention", False) and getattr(metadata, "tree_attention", False)
            )
            if tree_modes and tree_modes[step] != incoming_tree_mode:
                return True
            tree_masks = getattr(entry, "tree_attention_masks", ())
            captured_tree_mask = tree_masks[step] if tree_masks else None
            incoming_tree_mask = getattr(metadata, "tree_attention_mask", None)
            if (captured_tree_mask is None) != (incoming_tree_mask is None):
                return True
            if captured_tree_mask is not None and NativeACLGraphRunner._tensor_changed(
                captured_tree_mask,
                incoming_tree_mask,
            ):
                return True
            if (
                entry.actual_seq_lengths_q[step] != metadata.actual_seq_lengths_q
                or entry.sequence_lens[step] != metadata.sequence_lens
            ):
                return True
        return False

    def _draft_uses_device_position_attention(
        self,
        attention_metadatas: Sequence[Any],
    ) -> bool:
        """Identify the only draft graph contract allowed to own zero tasks."""

        if not attention_metadatas or any(
            bool(getattr(metadata, "use_fused_infer_attention", False))
            or getattr(metadata, "attention_mask", None) is not None
            for metadata in attention_metadatas
        ):
            return False
        try:
            layers = tuple(self.model.layers)
        except (AttributeError, TypeError):
            return False
        if not layers:
            return False
        attention_layers = tuple(getattr(layer, "self_attn", None) for layer in layers)
        return all(
            attention is not None
            and bool(getattr(attention, "uses_paged_attention", False))
            and bool(getattr(attention, "use_device_paged_attention", False))
            for attention in attention_layers
        )

    @staticmethod
    def _validate_draft_attention_task_contract(entry: NativeDraftACLGraphEntry) -> bool:
        """Fail closed when an empty task list lacks an explicit device PA origin."""

        taskless_device_attention = getattr(entry, "taskless_device_attention", False) is True
        task_count = len(entry.tasks)
        step_count = len(entry.positions)
        tasks_per_step = int(entry.tasks_per_step)
        if taskless_device_attention:
            if task_count or tasks_per_step:
                raise RuntimeError(
                    "Taskless device-position PagedAttention draft entry unexpectedly owns attention tasks."
                )
            return True
        if tasks_per_step <= 0 or task_count != step_count * tasks_per_step:
            raise RuntimeError(
                "Draft ACLGraph has missing attention-task metadata and is not an explicit "
                "device-position PagedAttention entry."
            )
        return False

    @staticmethod
    def _copy_draft_inputs(
        entry: NativeDraftACLGraphEntry,
        input_ids: torch.Tensor,
        positions: list[torch.Tensor],
        attention_metadatas: list[Any],
    ) -> None:
        def require_tensor_contract(
            captured: Any,
            incoming: Any,
            label: str,
        ) -> None:
            if not torch.is_tensor(captured) or not torch.is_tensor(incoming):
                raise RuntimeError(f"Draft ACLGraph {label} must remain a tensor between replays.")
            if (
                captured.shape != incoming.shape
                or captured.dtype != incoming.dtype
                or captured.device != incoming.device
            ):
                raise RuntimeError(f"Draft ACLGraph {label} shape, dtype or device changed between replays.")

        def require_optional_tensor_contract(
            captured: Any,
            incoming: Any,
            label: str,
        ) -> None:
            if (captured is None) != (incoming is None):
                raise RuntimeError(f"Draft ACLGraph {label} appeared or disappeared between replays.")
            if captured is not None:
                require_tensor_contract(captured, incoming, label)

        def require_host_length_tuple(value: Any, label: str) -> tuple[int, ...]:
            if not isinstance(value, tuple) or any(
                not isinstance(item, int) or isinstance(item, bool) for item in value
            ):
                raise RuntimeError(f"Draft ACLGraph {label} must remain a tuple of integers.")
            return value

        step_count = len(attention_metadatas)
        if step_count <= 0 or len(positions) != step_count:
            raise RuntimeError("Draft ACLGraph replay inputs must preserve the captured step count.")
        captured_step_fields = {
            "positions": entry.positions,
            "slot mappings": entry.slot_mappings,
            "context lengths": entry.context_lens,
            "block tables": entry.block_tables,
            "request block tables": entry.request_block_tables,
            "query partitions": entry.actual_seq_lengths_q,
            "KV lengths": entry.sequence_lens,
            "attention masks": getattr(entry, "attention_masks", ()),
            "FULL masks": getattr(entry, "tree_attention_masks", ()),
            "tree/FIA modes": getattr(entry, "tree_attention_modes", ()),
        }
        malformed_fields = [
            label
            for label, values in captured_step_fields.items()
            if not isinstance(values, tuple) or len(values) != step_count
        ]
        if malformed_fields:
            raise RuntimeError(
                "Draft ACLGraph captured step metadata is incomplete for: " + ", ".join(malformed_fields) + "."
            )

        # Preflight every step before the first ``copy_``.  A late failure is
        # otherwise observable by the captured graph as a mixture of old and
        # new replay inputs, which cannot be rolled back without synchronizing
        # and cloning every graph-owned tensor.
        require_tensor_contract(entry.input_ids, input_ids, "input IDs")
        if input_ids.ndim != 1:
            raise RuntimeError("Draft ACLGraph input IDs must remain a one-dimensional row buffer.")
        input_row_count = int(input_ids.numel())
        new_query_partitions: list[tuple[int, ...]] = []
        new_kv_lengths: list[tuple[int, ...]] = []
        shared_request_table_inputs: dict[int, torch.Tensor] = {}
        for step, (step_positions, metadata) in enumerate(zip(positions, attention_metadatas)):
            require_tensor_contract(entry.positions[step], step_positions, f"step {step} positions")
            require_tensor_contract(
                entry.slot_mappings[step],
                getattr(metadata, "slot_mapping", None),
                f"step {step} slot mapping",
            )
            require_tensor_contract(
                entry.context_lens[step],
                getattr(metadata, "context_lens", None),
                f"step {step} context lengths",
            )
            require_tensor_contract(
                entry.block_tables[step],
                getattr(metadata, "block_tables", None),
                f"step {step} block table",
            )
            if entry.positions[step].shape != input_ids.shape or entry.slot_mappings[step].shape != input_ids.shape:
                raise RuntimeError(
                    f"Draft ACLGraph step {step} positions and slot mapping must cover every captured input row."
                )

            captured_attention_mask = entry.attention_masks[step]
            incoming_attention_mask = getattr(metadata, "attention_mask", None)
            require_optional_tensor_contract(
                captured_attention_mask,
                incoming_attention_mask,
                f"step {step} attention mask",
            )
            captured_tree_mask = entry.tree_attention_masks[step]
            incoming_tree_mask = getattr(metadata, "tree_attention_mask", None)
            require_optional_tensor_contract(
                captured_tree_mask,
                incoming_tree_mask,
                f"step {step} FULL mask",
            )

            captured_request_table = entry.request_block_tables[step]
            incoming_request_table = getattr(metadata, "request_block_tables", None)
            require_optional_tensor_contract(
                captured_request_table,
                incoming_request_table,
                f"step {step} request block table",
            )
            if captured_request_table is not None:
                captured_table_id = id(captured_request_table)
                previous_incoming_table = shared_request_table_inputs.get(captured_table_id)
                if previous_incoming_table is None:
                    shared_request_table_inputs[captured_table_id] = incoming_request_table
                elif previous_incoming_table is not incoming_request_table:
                    raise RuntimeError(
                        "Draft ACLGraph steps sharing one captured request block "
                        "table must also share one replay input table."
                    )
            incoming_fia = bool(getattr(metadata, "use_fused_infer_attention", False))
            captured_fia = captured_request_table is not None
            if incoming_fia != captured_fia:
                raise RuntimeError(f"Draft ACLGraph step {step} FIA/paged-attention contract changed between replays.")
            incoming_tree_mode = bool(incoming_fia and getattr(metadata, "tree_attention", False))
            if entry.tree_attention_modes[step] != incoming_tree_mode:
                raise RuntimeError(f"Draft ACLGraph step {step} tree/linear FIA contract changed between replays.")
            if incoming_tree_mode and incoming_tree_mask is None:
                raise RuntimeError(f"Draft ACLGraph step {step} tree FIA requires a FULL mask.")

            actual_q = require_host_length_tuple(
                getattr(metadata, "actual_seq_lengths_q", None),
                f"step {step} query partition",
            )
            sequence_lens = require_host_length_tuple(
                getattr(metadata, "sequence_lens", None),
                f"step {step} KV lengths",
            )
            if actual_q:
                prior = 0
                for end in actual_q:
                    if end <= prior or end > input_row_count:
                        raise RuntimeError(
                            f"Draft ACLGraph step {step} query partition must "
                            "be strictly increasing within the captured rows."
                        )
                    prior = end
            elif sequence_lens:
                raise RuntimeError(f"Draft ACLGraph step {step} KV lengths require a query partition.")

            if incoming_fia:
                if (
                    not actual_q
                    or actual_q[-1] != input_row_count
                    or len(sequence_lens) != len(actual_q)
                    or any(length <= 0 for length in sequence_lens)
                    or incoming_request_table.ndim != 2
                    or incoming_request_table.shape[0] != len(actual_q)
                ):
                    raise RuntimeError(
                        f"Draft ACLGraph step {step} FIA query/KV lengths and request page-table rows do not align."
                    )
                if incoming_tree_mode:
                    query_segments = (
                        actual_q[0],
                        *(current - previous for previous, current in zip(actual_q, actual_q[1:])),
                    )
                    if (
                        incoming_tree_mask.ndim != 4
                        or incoming_tree_mask.dtype != torch.bool
                        or incoming_tree_mask.shape[0] != len(actual_q)
                        or incoming_tree_mask.shape[1] != 1
                        or incoming_tree_mask.shape[2] < max(query_segments)
                        or incoming_tree_mask.shape[3] < max(sequence_lens)
                    ):
                        raise RuntimeError(
                            f"Draft ACLGraph step {step} FULL mask does not cover its FIA query/KV lengths."
                        )
            elif actual_q:
                # Paged attention appends finite one-token graph padding
                # rows.  They reuse a live page table but keep slot_mapping
                # at -1, so replay can read a valid KV prefix without writing
                # request state.  Zero-length rows are deliberately rejected:
                # Ascend PA may emit NaN for them at large graph buckets.
                # Packed requests retain one KV length per query segment;
                # one-token/request draft batches instead retain one length
                # per captured row, including that read-only suffix.
                taskless_device_attention = getattr(entry, "taskless_device_attention", False) is True
                if taskless_device_attention and (
                    actual_q != tuple(range(1, len(actual_q) + 1)) or len(sequence_lens) != input_row_count
                ):
                    raise RuntimeError(
                        f"Draft ACLGraph step {step} device-position PagedAttention "
                        "requires one KV length per real or padded row."
                    )
                if len(sequence_lens) not in (
                    len(actual_q),
                    input_row_count,
                ) or any(length <= 0 for length in sequence_lens[: len(actual_q)]):
                    raise RuntimeError(f"Draft ACLGraph step {step} paged-attention query/KV lengths do not align.")
                if any(length != 1 for length in sequence_lens[len(actual_q) :]):
                    raise RuntimeError(f"Draft ACLGraph step {step} padded KV lengths must be one.")
            new_query_partitions.append(actual_q)
            new_kv_lengths.append(sequence_lens)

        entry.input_ids.copy_(input_ids)
        copied_request_tables: set[int] = set()
        for step, (step_positions, metadata) in enumerate(zip(positions, attention_metadatas)):
            entry.positions[step].copy_(step_positions)
            entry.slot_mappings[step].copy_(metadata.slot_mapping)
            attention_masks = getattr(entry, "attention_masks", ())
            if attention_masks and attention_masks[step] is not None:
                if metadata.attention_mask is None:
                    raise RuntimeError("Draft ACLGraph attention mask disappeared between replays")
                attention_masks[step].copy_(metadata.attention_mask)
            tree_masks = getattr(entry, "tree_attention_masks", ())
            if tree_masks and tree_masks[step] is not None:
                tree_masks[step].copy_(metadata.tree_attention_mask)
            if entry.request_block_tables[step] is not None:
                assert metadata.request_block_tables is not None
                captured_table_id = id(entry.request_block_tables[step])
                if captured_table_id not in copied_request_tables:
                    entry.request_block_tables[step].copy_(metadata.request_block_tables)
                    copied_request_tables.add(captured_table_id)
            else:
                entry.context_lens[step].copy_(metadata.context_lens)
                entry.block_tables[step].copy_(metadata.block_tables)
        entry.actual_seq_lengths_q = tuple(new_query_partitions)
        entry.sequence_lens = tuple(new_kv_lengths)

    def _run(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata,
        *,
        output_kind: str,
        output_transform: Callable[[torch.Tensor], torch.Tensor] | None,
        stable_fia_key: str | None = None,
        stable_fia_capture_size: int | None = None,
        _graph_owned_entry: NativeACLGraphEntry | None = None,
    ) -> torch.Tensor:
        self.last_generic_execution = NativeGraphExecution()
        if _graph_owned_entry is not None and self.graph_cache_sealed is not True:
            raise RuntimeError("Graph-owned stable-FIA replay cannot bypass graph-cache sealing.")
        num_tokens = input_ids.shape[0]
        if not self.enabled or num_tokens > self.max_graph_tokens:
            reason = "disabled" if not self.enabled else "token_capacity"
            self._raise_if_graph_cache_sealed("generic", reason)
            if reason == "token_capacity":
                self.shape_fallback_count += 1
            execution = NativeGraphExecution("eager", reason)
            output = self._execute(input_ids, positions, attention_metadata, output_transform)
            return self._record_execution("generic", execution, output)
        if attention_metadata.use_fused_infer_attention:
            self.last_fia_shape = tuple(attention_metadata.actual_seq_lengths_q)
            self.last_fia_expected_batch_size = self.expected_fia_batch_size
            if stable_fia_key is None and not self._is_reusable_fia_shape(attention_metadata.actual_seq_lengths_q):
                self._raise_if_graph_cache_sealed("generic", "shape")
                self.shape_fallback_count += 1
                execution = NativeGraphExecution("eager", "shape")
                output = self._execute(input_ids, positions, attention_metadata, output_transform)
                return self._record_execution("generic", execution, output)
            if stable_fia_key is None:
                query_shape = ",".join(str(length) for length in attention_metadata.actual_seq_lengths_q)
                output_kind = f"{output_kind}|fia:{query_shape}"
                capture_size = num_tokens
            else:
                if (
                    stable_fia_capture_size is None
                    or stable_fia_capture_size <= 0
                    or num_tokens != stable_fia_capture_size
                ):
                    raise ValueError("Stable FIA graph input does not match its fixed capture size.")
                output_kind = f"{output_kind}|fia-stable:{stable_fia_key}"
                capture_size = stable_fia_capture_size
        else:
            if stable_fia_key is not None or stable_fia_capture_size is not None:
                raise ValueError("Stable FIA graph keys require FIA metadata.")
            capture_size = min(
                (size for kind, size in self.entries if kind == output_kind and size >= num_tokens),
                default=num_tokens,
            )
        padded_inputs = self._pad_inputs(input_ids, positions, attention_metadata, capture_size)
        padded_input_ids, padded_positions, padded_metadata = padded_inputs
        entry_key = (output_kind, capture_size)
        if entry_key in self.disabled_entry_keys:
            self._raise_if_graph_cache_sealed("generic", "disabled_entry", entry_key)
            execution = NativeGraphExecution("eager", "disabled_entry")
            output = self._execute(input_ids, positions, attention_metadata, output_transform)
            return self._record_execution("generic", execution, output)
        entry = self.entries.get(entry_key)
        if entry is None:
            self._raise_if_graph_cache_sealed("generic", "missing_entry", entry_key)
            if len(self.entries) >= self.max_graph_entries or self._capture_budget_used >= self.max_graph_entries:
                self.capacity_fallback_count += 1
                execution = NativeGraphExecution("eager", "entry_capacity")
                output = self._execute(input_ids, positions, attention_metadata, output_transform)
                return self._record_execution("generic", execution, output)
            self.capture_attempt_count += 1
            self._capture_budget_used += 1
            output = self._capture(
                entry_key,
                padded_input_ids,
                padded_positions,
                padded_metadata,
                output_transform,
                valid_row_count=num_tokens,
            )
            return self._record_execution(
                "generic",
                self.last_generic_execution,
                output[:num_tokens],
                entry_key=entry_key,
                token_rows=(capture_size,),
            )
        if _graph_owned_entry is not None:
            if entry is not _graph_owned_entry:
                raise RuntimeError("Graph-owned stable-FIA replay resolved a different graph entry.")
            if not bool(entry.runtime_validated):
                raise RuntimeError("Graph-owned stable-FIA replay cannot bypass changed-input validation.")
            if (
                padded_input_ids is not entry.input_ids
                or padded_positions is not entry.positions
                or padded_metadata is not entry.stable_staged_metadata
                or padded_metadata.slot_mapping is not entry.slot_mapping
                or padded_metadata.context_lens is not entry.context_lens
                or padded_metadata.block_tables is not entry.block_tables
                or padded_metadata.request_block_tables is not entry.request_block_tables
                or padded_metadata.attention_mask is not entry.attention_mask
            ):
                raise RuntimeError("Graph-owned stable-FIA replay received non-resident input or metadata buffers.")
            changed_input = False
        else:
            changed_input = not entry.runtime_validated and self._generic_inputs_changed(
                entry,
                padded_input_ids,
                padded_positions,
                padded_metadata,
            )
        validated_real_row_count = getattr(entry, "validated_real_row_count", capture_size)
        if not isinstance(validated_real_row_count, int):
            validated_real_row_count = capture_size
        logical_row_expansion = num_tokens > validated_real_row_count
        if not entry.runtime_validated:
            self._raise_if_graph_cache_sealed("generic", "unvalidated_entry", entry_key)
        if logical_row_expansion:
            self._raise_if_graph_cache_sealed("generic", "logical_row_expansion", entry_key)
        task_lengths_unchanged = (
            entry.actual_seq_lengths_q == padded_metadata.actual_seq_lengths_q
            and entry.sequence_lens == padded_metadata.sequence_lens
        )
        current_stream = torch.npu.current_stream()
        inline_update = envs.VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE
        replay_first_update = envs.VLLM_ASCEND_PEARL_TARGET_REPLAY_FIRST_TASK_UPDATE
        taskless_device_attention = self._validate_generic_attention_task_contract(entry)
        if not taskless_device_attention and inline_update and replay_first_update:
            raise RuntimeError(
                "VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE and "
                "VLLM_ASCEND_PEARL_TARGET_REPLAY_FIRST_TASK_UPDATE are "
                "mutually exclusive."
            )
        copy_done_event = (
            self._generic_replay_first_copy_event(entry)
            if replay_first_update and not taskless_device_attention
            else None
        )
        if _graph_owned_entry is None:
            self._copy_inputs(
                entry,
                padded_input_ids,
                padded_positions,
                padded_metadata,
            )
        else:
            # The exact service method has already filled the resident tensor
            # buffers.  Preserve only the dynamic host literals consumed by
            # the attention-task updater; a second tensor copy here would
            # defeat persistent staging.
            entry.actual_seq_lengths_q = padded_metadata.actual_seq_lengths_q
            entry.sequence_lens = padded_metadata.sequence_lens
        if envs.VLLM_ASCEND_PEARL_SYNC_GRAPH_INPUTS:
            current_stream.synchronize()
        if taskless_device_attention:
            # Device-position PA consumes graph-owned tensors on the current
            # stream and owns no CANN host-literal task handles or events.
            entry.graph.replay()
            self.generic_taskless_replays += 1
        else:
            # Order task updates after the previous replay and the current
            # input copies. CANN releases a replay early when updates are issued
            # on an auxiliary stream, so inline mode remains available where
            # ExternalEvent visibility is insufficient for multi-token PA.
            assert self.update_stream is not None
            if replay_first_update:
                assert copy_done_event is not None
                try:
                    copy_done_event.record(current_stream)
                    # The update stream cannot touch graph-owned metadata until
                    # this replay's input copies and preceding replay complete.
                    self.update_stream.wait_event(copy_done_event)
                except Exception as exc:
                    raise RuntimeError(
                        "Generic target ACLGraph replay-first dependency setup failed before graph submission."
                    ) from exc
                try:
                    entry.graph.replay()
                except Exception as exc:
                    raise RuntimeError(
                        "Generic target ACLGraph replay-first graph submission failed before attention-task update."
                    ) from exc
                try:
                    if task_lengths_unchanged:
                        self._record_attention_task_events(
                            entry,
                            self.update_stream,
                        )
                        self.task_update_skip_replay_count += 1
                    else:
                        self._update_attention_tasks(entry)
                except Exception as exc:
                    # The submitted graph may already be waiting on one of its
                    # captured ExternalEvents.  Eager fallback could observe
                    # partial KV writes or stale outputs, so terminate this
                    # worker instead of returning unknown-provenance output.
                    raise RuntimeError(
                        "Generic target ACLGraph replay-first attention-task "
                        "update failed after graph submission; the runner cannot "
                        "safely fall back for this replay."
                    ) from exc
            elif not inline_update:
                self.update_stream.wait_stream(current_stream)
                if task_lengths_unchanged:
                    self._record_attention_task_events(
                        entry,
                        self.update_stream,
                    )
                    self.task_update_skip_replay_count += 1
                else:
                    self._update_attention_tasks(entry)
            else:
                if task_lengths_unchanged:
                    self._record_attention_task_events(entry, current_stream)
                    self.task_update_skip_replay_count += 1
                else:
                    self._update_attention_tasks(entry, stream=current_stream)
            if not inline_update and not replay_first_update:
                # Attention task updates are issued on the auxiliary stream.
                # The reverse dependency prevents replay from consuming old
                # metadata.
                current_stream.wait_stream(self.update_stream)
            if not inline_update and envs.VLLM_ASCEND_PEARL_SYNC_GRAPH_TASK_UPDATE:
                self.update_stream.synchronize()
            if not task_lengths_unchanged:
                self._record_task_update("generic", len(entry.tasks))
            if not replay_first_update:
                entry.graph.replay()
        self.replay_count += 1
        execution = NativeGraphExecution("replay", replay_executed=True)
        if envs.VLLM_ASCEND_PEARL_SYNC_GRAPH_REPLAY:
            torch.npu.synchronize()
        runtime_validation_performed = False
        if changed_input or logical_row_expansion:
            torch.npu.synchronize()
            graph_output = entry.output[:num_tokens].clone()
            reference_output = self._execute(
                input_ids,
                positions,
                attention_metadata,
                output_transform,
            )
            torch.npu.synchronize()
            validation_failed = not self._outputs_match(graph_output, reference_output)
            self._record_runtime_validation(
                "generic",
                changed_input=changed_input,
                logical_row_expansion=logical_row_expansion,
                failed=validation_failed,
            )
            runtime_validation_performed = True
            if validation_failed:
                failed_entry = self.entries.pop(entry_key)
                self._reset_graph_entry(failed_entry)
                self.disabled_entry_keys.add(entry_key)
                self.failed_capture_count += 1
                execution = NativeGraphExecution("eager", "runtime_validation", replay_executed=True)
                warnings.warn(
                    "Native PEARL disabled an ACLGraph entry whose runtime replay did not match eager execution: "
                    f"{entry_key!r}; {self._mismatch_summary(graph_output, reference_output)}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self._record_execution(
                    "generic",
                    execution,
                    reference_output,
                    entry_key=entry_key,
                    token_rows=(capture_size,),
                )
            entry.runtime_validated = True
            entry.validated_real_row_count = max(
                validated_real_row_count,
                num_tokens,
            )
        if envs.VLLM_ASCEND_PEARL_VALIDATE_GRAPH_REPLAYS and not runtime_validation_performed:
            torch.npu.synchronize()
            replay_output = entry.output[:num_tokens].clone()
            replay_reference = self._execute(
                input_ids,
                positions,
                attention_metadata,
                output_transform,
            )
            torch.npu.synchronize()
            replay_differs = not self._outputs_match(replay_output, replay_reference)
            self._record_runtime_validation(
                "generic",
                changed_input=changed_input,
                logical_row_expansion=logical_row_expansion,
                failed=replay_differs,
            )
            if replay_differs:
                failed_entry = self.entries.pop(entry_key)
                self._reset_graph_entry(failed_entry)
                self.disabled_entry_keys.add(entry_key)
                self.failed_capture_count += 1
                execution = NativeGraphExecution("eager", "runtime_validation", replay_executed=True)
                warnings.warn(
                    "Native PEARL disabled an ACLGraph entry whose diagnostic "
                    "runtime replay did not match eager execution: "
                    f"{entry_key!r}; "
                    f"{self._mismatch_summary(replay_output, replay_reference)}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self._record_execution(
                    "generic",
                    execution,
                    replay_reference,
                    entry_key=entry_key,
                    token_rows=(capture_size,),
                )
        return self._record_execution(
            "generic",
            execution,
            entry.output[:num_tokens],
            entry_key=entry_key,
            token_rows=(capture_size,),
        )

    def _is_reusable_fia_shape(self, actual_seq_lengths_q: tuple[int, ...]) -> bool:
        if not actual_seq_lengths_q:
            return False
        if self.expected_fia_batch_size is not None and len(actual_seq_lengths_q) != self.expected_fia_batch_size:
            return False
        segment_lengths = (actual_seq_lengths_q[0],) + tuple(
            current - previous for previous, current in zip(actual_seq_lengths_q, actual_seq_lengths_q[1:])
        )
        # The full request set has stable KV ownership even when target
        # verification mixes one-token pre-verify rows with gamma-token rows.
        # Its exact cumulative shape is part of the graph key. Dynamic tails
        # remain eager because their request count fails the guard above.
        return all(length > 0 for length in segment_lengths)

    def _execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata,
        output_transform: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> torch.Tensor:
        output = self.model(input_ids, positions, attention_metadata)
        return output if output_transform is None else output_transform(output)

    @staticmethod
    def _pad_inputs(input_ids, positions, attention_metadata, capture_size: int):
        pad_size = capture_size - input_ids.shape[0]
        if pad_size == 0:
            return input_ids, positions, attention_metadata
        padded_input_ids = torch.zeros(capture_size, dtype=input_ids.dtype, device=input_ids.device)
        padded_positions = torch.zeros(capture_size, dtype=positions.dtype, device=positions.device)
        padded_slot_mapping = torch.full(
            (capture_size,),
            -1,
            dtype=attention_metadata.slot_mapping.dtype,
            device=attention_metadata.slot_mapping.device,
        )
        # A zero-length PA row is not a harmless no-op on every Ascend
        # attention kernel: large graph buckets can produce NaN for those
        # dummy rows, which then (correctly) trips the model-wide finite
        # guards even though all logical rows are healthy.  Give every dummy
        # row a finite, read-only one-token context instead.  Reusing the
        # first live page is safe because ``slot_mapping`` stays -1, so the
        # padding rows never write KV state.
        padded_context_lens = torch.ones(capture_size, dtype=attention_metadata.context_lens.dtype)
        padded_block_tables = torch.zeros(
            (capture_size, attention_metadata.block_tables.shape[1]),
            dtype=attention_metadata.block_tables.dtype,
            device=attention_metadata.block_tables.device,
        )
        num_tokens = input_ids.shape[0]
        padded_input_ids[:num_tokens].copy_(input_ids)
        padded_positions[:num_tokens].copy_(positions)
        padded_slot_mapping[:num_tokens].copy_(attention_metadata.slot_mapping)
        padded_context_lens[:num_tokens].copy_(attention_metadata.context_lens)
        padded_block_tables[:num_tokens].copy_(attention_metadata.block_tables)
        if pad_size and num_tokens:
            padded_block_tables[num_tokens:].copy_(attention_metadata.block_tables[0])
        metadata_updates = {
            "slot_mapping": padded_slot_mapping,
            "context_lens": padded_context_lens,
            "block_tables": padded_block_tables,
        }
        # A fixed-gamma PA draft has exactly one query token per request, so
        # its request-level ``sequence_lens`` tuple is also the exact host-side
        # value of the per-token ``context_lens`` tensor.  Preserve that tuple
        # and append finite, read-only one-token dummy requests when graph
        # bucketing pads rows.
        # Packed multi-token PA does not satisfy this invariant and deliberately
        # retains its original request-level tuple for the safe fallback below.
        sequence_lens = tuple(getattr(attention_metadata, "sequence_lens", ()))
        if (
            not getattr(
                attention_metadata,
                "use_fused_infer_attention",
                False,
            )
            and len(sequence_lens) == num_tokens
        ):
            metadata_updates["sequence_lens"] = (
                *sequence_lens,
                *((1,) * pad_size),
            )
        if hasattr(attention_metadata, "__dataclass_fields__"):
            padded_metadata = replace(attention_metadata, **metadata_updates)
        else:
            padded_metadata = type(attention_metadata)(**metadata_updates)
        return padded_input_ids, padded_positions, padded_metadata

    def _capture(
        self,
        entry_key: tuple[str, int],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata,
        output_transform: Callable[[torch.Tensor], torch.Tensor] | None,
        *,
        valid_row_count: int,
    ) -> torch.Tensor:
        if not 0 < valid_row_count <= int(input_ids.shape[0]):
            raise ValueError(
                f"A generic ACLGraph valid row count must be in [1, {int(input_ids.shape[0])}], got {valid_row_count}."
            )
        # Warm up allocations and collectives before entering stream capture.
        reference_output = self._execute(
            input_ids,
            positions,
            attention_metadata,
            output_transform,
        ).clone()
        torch.npu.synchronize()
        # The eager reference above may leave hundreds of MiB of inactive
        # operator scratch in the caching allocator.  ACLGraph capture uses a
        # separate pool and cannot borrow those cached blocks, so on a
        # memory-tight TP3 target the first otherwise-small capture can fail
        # even though the inactive cache is larger than its allocation.  This
        # path runs only while creating a new graph (never on replay); release
        # unoccupied scratch before entering the graph pool without touching
        # any live model, KV-cache, graph-entry, or FIA-workspace tensor.
        torch.npu.empty_cache()
        captured_input_ids = input_ids.clone()
        captured_positions = positions.clone()
        captured_slot_mapping = attention_metadata.slot_mapping.clone()
        captured_context_lens = attention_metadata.context_lens.clone()
        captured_block_tables = attention_metadata.block_tables.clone()
        captured_request_block_tables = (
            attention_metadata.request_block_tables.clone()
            if attention_metadata.use_fused_infer_attention and attention_metadata.request_block_tables is not None
            else None
        )
        captured_attention_mask = (
            attention_metadata.attention_mask.clone() if attention_metadata.attention_mask is not None else None
        )
        captured_metadata = type(attention_metadata)(
            slot_mapping=captured_slot_mapping,
            context_lens=captured_context_lens,
            block_tables=captured_block_tables,
            actual_seq_lengths_q=attention_metadata.actual_seq_lengths_q,
            sequence_lens=attention_metadata.sequence_lens,
            request_block_tables=captured_request_block_tables,
            attention_mask=captured_attention_mask,
            use_fused_infer_attention=attention_metadata.use_fused_infer_attention,
        )
        exact_target_verification = "|fia-stable:stable-target-verify-hidden|" in entry_key[0]
        task_event_group_size = self.target_fia_task_event_group_size if exact_target_verification else 1
        task_event_prefix_group_size = self.target_fia_task_prefix_event_group_size if exact_target_verification else 0
        graph = torch.npu.NPUGraph()
        with (
            _collect_graph_tasks(
                fia_workspaces=self._fia_workspace_pool,
                task_event_group_size=task_event_group_size,
                task_event_prefix_group_size=task_event_prefix_group_size,
            ) as tasks,
            torch.npu.graph(graph, pool=self.graph_pool),
        ):
            output = self._execute(
                captured_input_ids,
                captured_positions,
                captured_metadata,
                output_transform,
            )
        self._validate_task_event_groups(
            tasks,
            task_event_group_size,
            task_event_prefix_group_size,
        )
        taskless_device_attention = self._draft_uses_device_position_attention((captured_metadata,))
        if taskless_device_attention and tasks:
            graph.reset()
            raise RuntimeError(
                "Device-position PagedAttention generic capture unexpectedly produced refreshable attention tasks."
            )
        entry = NativeACLGraphEntry(
            input_ids=captured_input_ids,
            positions=captured_positions,
            slot_mapping=captured_slot_mapping,
            context_lens=captured_context_lens,
            block_tables=captured_block_tables,
            request_block_tables=captured_request_block_tables,
            actual_seq_lengths_q=attention_metadata.actual_seq_lengths_q,
            sequence_lens=attention_metadata.sequence_lens,
            graph=graph,
            output=output,
            tasks=tasks,
            attention_mask=captured_attention_mask,
            validated_real_row_count=valid_row_count,
            task_event_group_size=task_event_group_size,
            task_event_prefix_group_size=task_event_prefix_group_size,
            taskless_device_attention=taskless_device_attention,
        )
        if exact_target_verification:
            self._provision_stable_fia_staging(entry, attention_metadata)
        self.entries[entry_key] = entry
        # Capture execution only records the graph; CANN does not guarantee
        # that its output buffers contain a valid inference result. Rebind the
        # host metadata and replay once before serving the triggering request.
        current_stream = torch.npu.current_stream()
        current_stream.synchronize()
        if not taskless_device_attention:
            assert self.update_stream is not None
            if envs.VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE:
                self._update_attention_tasks(entry, stream=current_stream)
            else:
                self.update_stream.wait_stream(current_stream)
                self._update_attention_tasks(entry)
                current_stream.wait_stream(self.update_stream)
        entry.graph.replay()
        if taskless_device_attention:
            self.generic_taskless_replays += 1
        torch.npu.synchronize()
        graph_matches_eager = self._outputs_match(output, reference_output)
        if not graph_matches_eager:
            failed_entry = self.entries.pop(entry_key)
            self._reset_graph_entry(failed_entry)
            self.disabled_entry_keys.add(entry_key)
            self.failed_capture_count += 1
            self.last_generic_execution = NativeGraphExecution(
                "eager",
                "capture_validation",
                capture_attempted=True,
                replay_executed=True,
            )
            warnings.warn(
                "Native PEARL disabled an ACLGraph entry whose first replay did not match eager execution: "
                f"{entry_key!r}; {self._mismatch_summary(output, reference_output)}",
                RuntimeWarning,
                stacklevel=2,
            )
            return reference_output
        self.capture_count += 1
        self.replay_count += 1
        self.last_generic_execution = NativeGraphExecution(
            "capture_replay",
            capture_attempted=True,
            replay_executed=True,
        )
        return output

    @staticmethod
    def _outputs_match(graph_output: torch.Tensor, reference_output: torch.Tensor) -> bool:
        if graph_output.shape != reference_output.shape or graph_output.dtype != reference_output.dtype:
            return False
        if graph_output.dtype.is_floating_point:
            return torch.allclose(graph_output, reference_output, rtol=1e-3, atol=1e-3)
        return torch.equal(graph_output, reference_output)

    @staticmethod
    def _validate_generic_attention_task_contract(entry: NativeACLGraphEntry) -> bool:
        """Return the explicit taskless mode or reject a missing task list."""

        taskless_device_attention = getattr(entry, "taskless_device_attention", False) is True
        if taskless_device_attention:
            if entry.tasks:
                raise RuntimeError(
                    "Taskless device-position PagedAttention generic entry unexpectedly owns attention tasks."
                )
            return True
        if not entry.tasks:
            raise RuntimeError(
                "Generic ACLGraph has no attention-task metadata and is not an explicit "
                "device-position PagedAttention entry."
            )
        return False

    @staticmethod
    def _mismatch_summary(graph_output: torch.Tensor, reference_output: torch.Tensor) -> str:
        if graph_output.dtype.is_floating_point:
            max_error = (graph_output.float() - reference_output.float()).abs().max().item()
            return f"shape={tuple(graph_output.shape)}, max_abs_error={max_error:g}"
        mismatches = torch.count_nonzero(graph_output != reference_output).item()
        return (
            f"mismatches={mismatches}/{reference_output.numel()}, "
            f"eager={reference_output.flatten()[:16].cpu().tolist()}, "
            f"graph={graph_output.flatten()[:16].cpu().tolist()}"
        )

    @staticmethod
    def _generic_inputs_changed(
        entry: NativeACLGraphEntry,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata: Any,
    ) -> bool:
        if (
            NativeACLGraphRunner._tensor_changed(entry.input_ids, input_ids)
            or NativeACLGraphRunner._tensor_changed(entry.positions, positions)
            or NativeACLGraphRunner._tensor_changed(entry.slot_mapping, attention_metadata.slot_mapping)
        ):
            return True
        if entry.request_block_tables is not None:
            if attention_metadata.request_block_tables is None or NativeACLGraphRunner._tensor_changed(
                entry.request_block_tables,
                attention_metadata.request_block_tables,
            ):
                return True
        elif NativeACLGraphRunner._tensor_changed(
            entry.context_lens, attention_metadata.context_lens
        ) or NativeACLGraphRunner._tensor_changed(entry.block_tables, attention_metadata.block_tables):
            return True
        captured_attention_mask = getattr(entry, "attention_mask", None)
        incoming_attention_mask = getattr(attention_metadata, "attention_mask", None)
        if (captured_attention_mask is None) != (incoming_attention_mask is None):
            return True
        if captured_attention_mask is not None and NativeACLGraphRunner._tensor_changed(
            captured_attention_mask, incoming_attention_mask
        ):
            return True
        return (
            entry.actual_seq_lengths_q != attention_metadata.actual_seq_lengths_q
            or entry.sequence_lens != attention_metadata.sequence_lens
        )

    @staticmethod
    def _copy_inputs(entry: NativeACLGraphEntry, input_ids, positions, attention_metadata) -> None:
        entry.input_ids.copy_(input_ids)
        entry.positions.copy_(positions)
        entry.slot_mapping.copy_(attention_metadata.slot_mapping)
        captured_attention_mask = getattr(entry, "attention_mask", None)
        if captured_attention_mask is not None:
            if attention_metadata.attention_mask is None:
                raise RuntimeError("ACLGraph attention mask disappeared between replays")
            captured_attention_mask.copy_(attention_metadata.attention_mask)
            entry.attention_mask_source_id = id(attention_metadata.attention_mask)
            entry.attention_mask_source_version = NativeACLGraphRunner._tensor_mutation_version(
                attention_metadata.attention_mask
            )
        if entry.request_block_tables is not None:
            assert attention_metadata.request_block_tables is not None
            entry.request_block_tables.copy_(attention_metadata.request_block_tables)
        else:
            entry.context_lens.copy_(attention_metadata.context_lens)
            entry.block_tables.copy_(attention_metadata.block_tables)
        entry.actual_seq_lengths_q = attention_metadata.actual_seq_lengths_q
        entry.sequence_lens = attention_metadata.sequence_lens

    @staticmethod
    def _task_event_group_size(entry: NativeACLGraphEntry) -> int:
        group_size = getattr(entry, "task_event_group_size", 1)
        # A few CPU-only compatibility tests use a MagicMock graph entry.  Such
        # legacy entries predate grouped events and retain group size one.
        if not isinstance(group_size, int):
            return 1
        if group_size not in TARGET_FIA_TASK_EVENT_GROUP_SIZES:
            raise RuntimeError(f"An ACLGraph entry has an unsupported task event group size: {group_size}.")
        return group_size

    @staticmethod
    def _task_event_prefix_group_size(
        entry: NativeACLGraphEntry,
        group_size: int,
    ) -> int:
        prefix_size = getattr(entry, "task_event_prefix_group_size", 0)
        if not isinstance(prefix_size, int):
            return 0
        if prefix_size not in (0, 1, 2) or (prefix_size and prefix_size >= group_size):
            raise RuntimeError(f"An ACLGraph entry has an unsupported task event prefix group size: {prefix_size}.")
        return prefix_size

    @staticmethod
    def _task_event_group_ranges(
        task_count: int,
        group_size: int,
        prefix_size: int = 0,
    ) -> list[range]:
        if task_count < 0 or group_size not in TARGET_FIA_TASK_EVENT_GROUP_SIZES:
            raise RuntimeError("Invalid ACLGraph task-event grouping contract.")
        if prefix_size not in (0, 1, 2) or (prefix_size and prefix_size >= group_size):
            raise RuntimeError("Invalid ACLGraph task-event prefix grouping contract.")
        groups: list[range] = []
        start = 0
        if prefix_size and task_count:
            groups.append(range(0, min(prefix_size, task_count)))
            start = min(prefix_size, task_count)
        groups.extend(
            range(offset, min(offset + group_size, task_count)) for offset in range(start, task_count, group_size)
        )
        return groups

    @staticmethod
    def _validate_task_event_groups(
        tasks: list[NativePagedAttentionGraphTask | NativeFusedInferAttentionGraphTask],
        group_size: int,
        prefix_size: int = 0,
    ) -> None:
        if group_size == 1 and not prefix_size:
            return
        if not tasks or any(not isinstance(task, NativeFusedInferAttentionGraphTask) for task in tasks):
            raise RuntimeError("Grouped ACLGraph task events require a non-empty FIA-only task list.")
        previous_event = None
        for group in NativeACLGraphRunner._task_event_group_ranges(
            len(tasks),
            group_size,
            prefix_size,
        ):
            event = tasks[group.start].event
            if any(tasks[index].event is not event for index in group):
                raise RuntimeError("ACLGraph FIA tasks in one event group do not share their captured ExternalEvent.")
            if event is previous_event:
                raise RuntimeError("Adjacent ACLGraph FIA task event groups unexpectedly share one ExternalEvent.")
            previous_event = event

    def _record_attention_task_events(
        self,
        entry: NativeACLGraphEntry,
        stream: torch.npu.Stream,
    ) -> None:
        group_size = self._task_event_group_size(entry)
        prefix_size = self._task_event_prefix_group_size(entry, group_size)
        self._validate_task_event_groups(entry.tasks, group_size, prefix_size)
        for group in self._task_event_group_ranges(
            len(entry.tasks),
            group_size,
            prefix_size,
        ):
            entry.tasks[group.start].event.record(stream)

    def _update_attention_tasks(
        self,
        entry: NativeACLGraphEntry,
        *,
        stream: torch.npu.Stream | None = None,
    ) -> None:
        task_lengths = [(entry.actual_seq_lengths_q, entry.sequence_lens)] * len(entry.tasks)
        self._update_attention_task_list(
            entry.tasks,
            task_lengths,
            stream=stream,
            event_group_size=self._task_event_group_size(entry),
            event_prefix_group_size=self._task_event_prefix_group_size(
                entry,
                self._task_event_group_size(entry),
            ),
        )

    def _update_draft_attention_tasks(
        self,
        entry: NativeDraftACLGraphEntry,
        *,
        stream: torch.npu.Stream | None = None,
        record_events: bool = True,
    ) -> None:
        if envs.VLLM_ASCEND_PEARL_DRAFT_STEP_MAJOR_PA_TASK_UPDATE:
            if self._update_draft_paged_attention_tasks_step_major(
                entry,
                stream=stream,
                record_events=record_events,
            ):
                self.draft_step_major_pa_replays += 1
                return
            self.draft_step_major_pa_fallback_replays += 1
        task_lengths = [
            lengths
            for lengths in zip(entry.actual_seq_lengths_q, entry.sequence_lens)
            for _ in range(entry.tasks_per_step)
        ]
        try:
            self._update_attention_task_list(
                entry.tasks,
                task_lengths,
                stream=stream,
                record_events=record_events,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Draft ACLGraph attention-task update failed for "
                f"batch={entry.input_ids.shape[0]}, "
                f"steps={len(entry.positions)}, tasks={len(entry.tasks)}."
            ) from exc

    def _paged_attention_workspace_structure_id(
        self,
        task: NativePagedAttentionGraphTask,
    ) -> int:
        if task.workspace_structure_id is not None:
            return task.workspace_structure_id
        workspace_structure = (
            str(task.query.device),
            tuple(task.query.shape),
            tuple(task.key_cache.shape),
            tuple(task.value_cache.shape),
            task.query.dtype,
            task.key_cache.dtype,
            task.value_cache.dtype,
            task.num_kv_heads,
            task.num_heads,
            task.scale,
            tuple(task.block_table.shape),
            tuple(task.output.shape),
            task.output.dtype,
        )
        structure_id = self._pa_workspace_structure_ids.get(workspace_structure)
        if structure_id is None:
            structure_id = len(self._pa_workspace_structure_ids)
            self._pa_workspace_structure_ids[workspace_structure] = structure_id
        task.workspace_structure_id = structure_id
        return structure_id

    def _update_draft_paged_attention_tasks_step_major(
        self,
        entry: NativeDraftACLGraphEntry,
        *,
        stream: torch.npu.Stream | None = None,
        record_events: bool = True,
    ) -> bool:
        """Refresh a serial PA graph once per step for shared host metadata.

        CANN still requires one begin/op/end update for every captured layer.
        This fast path removes only redundant Python signature construction
        and workspace queries.  Returning ``False`` before any mutation lets
        mixed/packed attention retain the generic compatibility path.
        """

        update_stream = stream or self.update_stream
        assert update_stream is not None
        step_count = len(entry.actual_seq_lengths_q)
        tasks_per_step = int(entry.tasks_per_step)
        if (
            step_count <= 0
            or tasks_per_step <= 0
            or len(entry.sequence_lens) != step_count
            or len(entry.tasks) != step_count * tasks_per_step
            or any(not isinstance(task, NativePagedAttentionGraphTask) for task in entry.tasks)
        ):
            return False

        # Preflight the complete replay before updating its first handle.
        step_groups: list[
            tuple[
                list[NativePagedAttentionGraphTask],
                tuple[int, ...],
                tuple[int, ...],
            ]
        ] = []
        for step, (actual_q, sequence_lens) in enumerate(zip(entry.actual_seq_lengths_q, entry.sequence_lens)):
            start = step * tasks_per_step
            group = entry.tasks[start : start + tasks_per_step]
            typed_group = [task for task in group if isinstance(task, NativePagedAttentionGraphTask)]
            context_rows = int(typed_group[0].context_lens.numel())
            real_rows = len(actual_q)
            if (
                len(typed_group) != tasks_per_step
                or real_rows <= 0
                or actual_q != tuple(range(1, real_rows + 1))
                or len(sequence_lens) != context_rows
                or any(value <= 0 for value in sequence_lens[:real_rows])
                # Graph padding reuses one finite, read-only KV row.  A zero
                # length can make Ascend PA emit NaN at larger buckets, so the
                # padding contract is an exact length-one suffix.
                or any(value != 1 for value in sequence_lens[real_rows:])
                or any(int(task.context_lens.numel()) != context_rows for task in typed_group)
            ):
                return False
            step_groups.append((typed_group, actual_q, sequence_lens))

        with torch.npu.stream(update_stream):
            for tasks, _actual_q, _sequence_lens in step_groups:
                profile_host = self.profile_pa_task_update
                phase_started = time.perf_counter_ns() if profile_host else 0
                representative_by_structure: dict[int, NativePagedAttentionGraphTask] = {}
                for task in tasks:
                    structure_id = self._paged_attention_workspace_structure_id(task)
                    representative_by_structure.setdefault(
                        structure_id,
                        task,
                    )
                self.pa_workspace_host_key_tasks += len(tasks)
                if profile_host:
                    self.pa_task_update_host_ns["key"] += time.perf_counter_ns() - phase_started
                    phase_started = time.perf_counter_ns()

                workspace_by_structure: dict[int, torch.Tensor] = {}
                for structure_id, representative in representative_by_structure.items():
                    self.pa_workspace_get_calls += 1
                    workspace_by_structure[structure_id] = torch_npu._npu_paged_attention_get_workspace(
                        query=representative.query,
                        key_cache=representative.key_cache,
                        value_cache=representative.value_cache,
                        num_kv_heads=representative.num_kv_heads,
                        num_heads=representative.num_heads,
                        scale_value=representative.scale,
                        block_table=representative.block_table,
                        context_lens=representative.context_lens,
                        out=representative.output,
                    )
                self.pa_workspace_cache_hits += len(tasks) - len(workspace_by_structure)
                if profile_host:
                    self.pa_task_update_host_ns["get_workspace"] += time.perf_counter_ns() - phase_started

                for task in tasks:
                    phase_started = time.perf_counter_ns() if profile_host else 0
                    task.workspace = workspace_by_structure[self._paged_attention_workspace_structure_id(task)]
                    torch.npu.graph_task_update_begin(
                        update_stream,
                        task.handle,
                    )
                    torch_npu._npu_paged_attention(
                        query=task.query,
                        key_cache=task.key_cache,
                        value_cache=task.value_cache,
                        num_kv_heads=task.num_kv_heads,
                        num_heads=task.num_heads,
                        scale_value=task.scale,
                        block_table=task.block_table,
                        context_lens=task.context_lens,
                        out=task.output,
                        workspace=task.workspace,
                    )
                    torch.npu.graph_task_update_end(update_stream)
                    if record_events:
                        task.event.record(update_stream)
                    if profile_host:
                        self.pa_task_update_host_ns["task_update"] += time.perf_counter_ns() - phase_started
                        self.pa_task_update_profiled_tasks += 1
        return True

    def _update_target_attention_tasks(self, entry: NativeTargetACLGraphEntry) -> None:
        task_lengths = [
            lengths
            for lengths in zip(entry.actual_seq_lengths_q, entry.sequence_lens)
            for _ in range(entry.tasks_per_step)
        ]
        self._update_attention_task_list(entry.tasks, task_lengths)

    def _update_fia_task_chunk(
        self,
        tasks: list[NativeFusedInferAttentionGraphTask],
        task_lengths: list[tuple[list[int], list[int]]],
        task_indices: Sequence[int],
        stream: torch.npu.Stream,
        event_group_ends: frozenset[int],
    ) -> None:
        """Refresh complete FIA event groups on one NPU update stream."""

        if not task_indices:
            return
        torch.npu.set_device(tasks[0].query.device)
        with torch.npu.stream(stream):
            for task_index in task_indices:
                task = tasks[task_index]
                actual_seq_lengths_q, sequence_lens = task_lengths[task_index]
                record_event = task_index in event_group_ends
                self._update_fused_infer_attention_task(
                    task,
                    actual_seq_lengths_q,
                    sequence_lens,
                    stream=stream,
                    profile_host=False,
                    record_event=record_event,
                )

    def _try_parallel_fia_task_update(
        self,
        tasks: list[NativePagedAttentionGraphTask | NativeFusedInferAttentionGraphTask],
        task_lengths: list[tuple[tuple[int, ...], tuple[int, ...]]],
        *,
        update_stream: torch.npu.Stream,
        record_events: bool,
        event_group_size: int,
        event_prefix_group_size: int,
    ) -> bool:
        """Refresh independent target FIA handles concurrently when qualified.

        CANN captures one refreshable handle per decoder layer.  Rebuilding
        those handles is host dominated and does not execute model compute.
        Each worker owns a disjoint handle set and NPU stream; the primary
        update stream joins every secondary stream before the caller permits
        graph replay.  Any mixed PA/tree task family stays on the serial path.
        """

        executor = self._target_fia_update_executor
        worker_count = self.target_fia_task_update_workers
        if (
            executor is None
            or worker_count <= 1
            or not record_events
            or len(tasks) < worker_count
            or len(tasks) != len(task_lengths)
            or any(not isinstance(task, NativeFusedInferAttentionGraphTask) or task.tree_attention for task in tasks)
        ):
            return False
        typed_tasks = [task for task in tasks if isinstance(task, NativeFusedInferAttentionGraphTask)]
        length_cache: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            tuple[list[int], list[int]],
        ] = {}
        materialized_lengths: list[tuple[list[int], list[int]]] = []
        for lengths in task_lengths:
            pair = length_cache.get(lengths)
            if pair is None:
                pair = (list(lengths[0]), list(lengths[1]))
                length_cache[lengths] = pair
            materialized_lengths.append(pair)

        streams = self._target_fia_parallel_streams
        if len(streams) != worker_count or streams[0] is not update_stream:
            raise RuntimeError("Parallel target FIA update streams are inconsistent.")
        for stream in streams[1:]:
            stream.wait_stream(update_stream)
        groups = self._task_event_group_ranges(
            len(typed_tasks),
            event_group_size,
            event_prefix_group_size,
        )
        event_group_ends = frozenset(group.stop - 1 for group in groups)
        partitions = [
            tuple(index for group in groups[worker::worker_count] for index in group) for worker in range(worker_count)
        ]
        if any(not partition for partition in partitions):
            return False
        # With one event per task, submitting every stream to the persistent
        # pool remains faster on 910B.  Grouped events make the per-worker
        # chunks coarse enough that handling stream zero in the caller avoids
        # a future dispatch and wins measurably.  In both cases a complete
        # event group stays on one stream, so its final record releases only
        # handles that have all been refreshed.
        caller_owns_primary = event_group_size > 1
        first_background_worker = 1 if caller_owns_primary else 0
        futures = [
            executor.submit(
                self._update_fia_task_chunk,
                typed_tasks,
                materialized_lengths,
                partitions[worker],
                streams[worker],
                event_group_ends,
            )
            for worker in range(first_background_worker, worker_count)
        ]
        if caller_owns_primary:
            self._update_fia_task_chunk(
                typed_tasks,
                materialized_lengths,
                partitions[0],
                streams[0],
                event_group_ends,
            )
        for future in futures:
            future.result()
        for stream in streams[1:]:
            update_stream.wait_stream(stream)
        self.target_fia_parallel_update_replays += 1
        self.target_fia_parallel_update_tasks += len(tasks)
        return True

    def _update_attention_task_list(
        self,
        tasks: list[NativePagedAttentionGraphTask | NativeFusedInferAttentionGraphTask],
        task_lengths: list[tuple[tuple[int, ...], tuple[int, ...]]],
        *,
        stream: torch.npu.Stream | None = None,
        record_events: bool = True,
        event_group_size: int = 1,
        event_prefix_group_size: int = 0,
    ) -> None:
        update_stream = stream or self.update_stream
        assert update_stream is not None
        if len(tasks) != len(task_lengths):
            raise RuntimeError("ACLGraph attention tasks and replay metadata differ in length.")
        if event_group_size not in TARGET_FIA_TASK_EVENT_GROUP_SIZES:
            raise RuntimeError("ACLGraph task event group size must be 1, 2, or 4.")
        self._validate_task_event_groups(
            tasks,
            event_group_size,
            event_prefix_group_size,
        )
        if self._try_parallel_fia_task_update(
            tasks,
            task_lengths,
            update_stream=update_stream,
            record_events=record_events,
            event_group_size=event_group_size,
            event_prefix_group_size=event_prefix_group_size,
        ):
            return
        event_group_ends = frozenset(
            group.stop - 1
            for group in self._task_event_group_ranges(
                len(tasks),
                event_group_size,
                event_prefix_group_size,
            )
        )
        fia_length_lists: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            tuple[list[int], list[int]],
        ] = {}
        # PagedAttention workspaces are shared only within this serial task
        # update.  The exact context lengths participate in the key because
        # CANN may select a different workspace/tiling as sequences grow.
        paged_workspaces: dict[tuple[Any, ...], torch.Tensor] = {}
        paged_host_context_keys: dict[tuple[int, int, int], tuple[int, tuple[int, ...], bool]] = {}
        with torch.npu.stream(update_stream):
            for task_index, (
                task,
                (actual_seq_lengths_q, sequence_lens),
            ) in enumerate(zip(tasks, task_lengths)):
                record_task_event = record_events and task_index in event_group_ends
                if isinstance(task, NativeFusedInferAttentionGraphTask):
                    profile_host = self.profile_pa_task_update
                    phase_started = time.perf_counter_ns() if profile_host else 0
                    length_key = (actual_seq_lengths_q, sequence_lens)
                    length_lists = fia_length_lists.get(length_key)
                    if length_lists is None:
                        length_lists = (
                            list(actual_seq_lengths_q),
                            list(sequence_lens),
                        )
                        fia_length_lists[length_key] = length_lists
                    if profile_host:
                        self.fia_task_update_host_ns["key"] += time.perf_counter_ns() - phase_started
                    self._update_fused_infer_attention_task(
                        task,
                        *length_lists,
                        stream=update_stream,
                        profile_host=profile_host,
                        record_event=record_task_event,
                    )
                    continue
                profile_host = self.profile_pa_task_update
                phase_started = time.perf_counter_ns() if profile_host else 0
                # For a one-token-per-request PA call, cumulative query lengths
                # are exactly 1..N.  In that case ``sequence_lens`` was derived
                # from the same Python positions as the CPU ``context_lens``
                # tensor, and graph padding appends finite length-one rows.
                # Reuse this
                # immutable host tuple instead of converting the identical CPU
                # tensor once per layer (112 conversions for gamma=4/Qwen3).
                # Multi-token packed PA cannot use the request-level tuple and
                # remains on the compatibility fallback.
                real_request_count = len(actual_seq_lengths_q)
                context_tensor_rows = int(task.context_lens.numel())
                # ``task_lengths`` deliberately repeats the same immutable
                # tuple object for every layer belonging to one serial step.
                # Object identity is therefore a safe call-local memo key and
                # avoids rehashing up to 64 integer lengths 28 times.
                host_alignment_key = (
                    id(actual_seq_lengths_q),
                    id(sequence_lens),
                    context_tensor_rows,
                )
                if host_alignment_key not in paged_host_context_keys:
                    host_lengths_are_token_aligned = (
                        real_request_count > 0
                        and actual_seq_lengths_q == tuple(range(1, real_request_count + 1))
                        and len(sequence_lens) == context_tensor_rows
                        and all(value > 0 for value in sequence_lens[:real_request_count])
                        and all(value == 1 for value in sequence_lens[real_request_count:])
                    )
                    context_length_key = (
                        sequence_lens
                        if host_lengths_are_token_aligned
                        else tuple(int(value) for value in task.context_lens.tolist())
                    )
                    paged_host_context_keys[host_alignment_key] = (
                        len(paged_host_context_keys),
                        context_length_key,
                        host_lengths_are_token_aligned,
                    )
                (
                    context_group_id,
                    context_length_key,
                    host_lengths_are_token_aligned,
                ) = paged_host_context_keys[host_alignment_key]
                if host_lengths_are_token_aligned:
                    self.pa_workspace_host_key_tasks += 1
                else:
                    # ``context_lens`` is part of the CANN CPU ABI here.  This
                    # fallback is a CPU materialization, not an NPU-to-host
                    # synchronization, and preserves generic packed-PA support.
                    self.pa_workspace_tensor_key_tasks += 1
                # Keep the native runner ABI-aligned with vLLM-Ascend's
                # production PagedAttention graph updater.  Workspace sizing
                # may depend on the current context-length tensor/tiling; a
                # buffer obtained while capturing an earlier sequence length
                # is not a valid lifetime-wide cache key.  Retain the refreshed
                # tensor on the task so it remains alive through graph replay.
                self._paged_attention_workspace_structure_id(task)
                workspace_key = (
                    task.workspace_structure_id,
                    context_group_id,
                )
                if profile_host:
                    self.pa_task_update_host_ns["key"] += time.perf_counter_ns() - phase_started
                    phase_started = time.perf_counter_ns()
                workspace = paged_workspaces.get(workspace_key)
                if workspace is None:
                    self.pa_workspace_get_calls += 1
                    workspace = torch_npu._npu_paged_attention_get_workspace(
                        query=task.query,
                        key_cache=task.key_cache,
                        value_cache=task.value_cache,
                        num_kv_heads=task.num_kv_heads,
                        num_heads=task.num_heads,
                        scale_value=task.scale,
                        block_table=task.block_table,
                        context_lens=task.context_lens,
                        out=task.output,
                    )
                    paged_workspaces[workspace_key] = workspace
                else:
                    self.pa_workspace_cache_hits += 1
                if profile_host:
                    self.pa_task_update_host_ns["get_workspace"] += time.perf_counter_ns() - phase_started
                    phase_started = time.perf_counter_ns()
                task.workspace = workspace
                torch.npu.graph_task_update_begin(update_stream, task.handle)
                torch_npu._npu_paged_attention(
                    query=task.query,
                    key_cache=task.key_cache,
                    value_cache=task.value_cache,
                    num_kv_heads=task.num_kv_heads,
                    num_heads=task.num_heads,
                    scale_value=task.scale,
                    block_table=task.block_table,
                    context_lens=task.context_lens,
                    out=task.output,
                    workspace=task.workspace,
                )
                torch.npu.graph_task_update_end(update_stream)
                if record_task_event:
                    task.event.record(update_stream)
                if profile_host:
                    self.pa_task_update_host_ns["task_update"] += time.perf_counter_ns() - phase_started
                    self.pa_task_update_profiled_tasks += 1

    def _update_fused_infer_attention_task(
        self,
        task: NativeFusedInferAttentionGraphTask,
        actual_seq_lengths_q: list[int],
        sequence_lens: list[int],
        *,
        stream: torch.npu.Stream,
        profile_host: bool = False,
        record_event: bool = True,
    ) -> None:
        args = {
            "query": task.query,
            "key": task.key_cache,
            "value": task.value_cache,
            "atten_mask": task.attention_mask,
            "block_table": task.block_table,
            "input_layout": "TND",
            "block_size": task.block_size,
            "actual_seq_lengths": actual_seq_lengths_q,
            "actual_seq_lengths_kv": sequence_lens,
            "num_key_value_heads": task.num_kv_heads,
            "num_heads": task.num_heads,
            "scale": task.scale,
        }
        if task.tree_attention:
            _validate_tree_fia_mask(
                task.query.shape[0],
                task.attention_mask,
                task.block_table,
                actual_seq_lengths_q,
                sequence_lens,
                task.block_size,
            )
            args.update(sparse_mode=TREE_FIA_SPARSE_MODE, inner_precise=TREE_FIA_INNER_PRECISE)
        else:
            # Keep task refresh ABI-identical to eager/capture construction.
            args.update(sparse_mode=3, next_tokens=0)
        phase_started = time.perf_counter_ns() if profile_host else 0
        torch.npu.graph_task_update_begin(stream, task.handle)
        torch_npu.npu_fused_infer_attention_score.out(
            **args,
            workspace=task.workspace,
            out=[task.output, task.softmax_lse],
        )
        torch.npu.graph_task_update_end(stream)
        if profile_host:
            self.fia_task_update_host_ns["submit"] += time.perf_counter_ns() - phase_started
            phase_started = time.perf_counter_ns()
        if record_event:
            task.event.record(stream)
            if profile_host:
                self.fia_task_update_host_ns["event"] += time.perf_counter_ns() - phase_started
        if profile_host:
            self.fia_task_update_profiled_tasks += 1
