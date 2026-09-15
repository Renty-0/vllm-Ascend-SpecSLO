# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ACLGraph capture and paged-attention graph-task updates for native PEARL."""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch_npu

from vllm_ascend import envs

DEFAULT_MAX_ACLGRAPH_ENTRIES = 16
TREE_FIA_SPARSE_MODE = 1
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
    if (
        not metadata.use_fused_infer_attention
        or not getattr(metadata, "tree_attention", False)
    ):
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
    runtime_validated: bool = False
    validated_real_row_count: int = 0
    # The event is recorded after current-replay input copies.  Since draft
    # replays use the same current stream, waiting for it also proves that the
    # preceding replay has completed and reset every captured ExternalEvent.
    replay_first_copy_done_event: Any | None = None


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

@contextmanager
def _collect_graph_tasks(
    pa_workspaces: dict[tuple[Any, ...], torch.Tensor] | None = None,
    fia_workspaces: dict[tuple[Any, ...], torch.Tensor] | None = None,
):
    global _CAPTURED_FIA_WORKSPACES, _CAPTURED_PA_WORKSPACES, _CAPTURED_TASKS
    if _CAPTURED_TASKS is not None:
        raise RuntimeError("Nested native PEARL ACLGraph capture is not supported.")
    tasks: list[NativePagedAttentionGraphTask | NativeFusedInferAttentionGraphTask] = []
    _CAPTURED_TASKS = tasks
    # Graph entries on one model runner replay serially.  FIA can therefore
    # keep its large max-workspace buffers in a runner-lifetime pool.  PA stays
    # capture-local because its workspace/tiling depends on exact sequence
    # lengths and is refreshed before every task update.
    _CAPTURED_PA_WORKSPACES = {} if pa_workspaces is None else pa_workspaces
    _CAPTURED_FIA_WORKSPACES = {} if fia_workspaces is None else fia_workspaces
    try:
        yield tasks
    finally:
        _CAPTURED_TASKS = None
        _CAPTURED_PA_WORKSPACES = None
        _CAPTURED_FIA_WORKSPACES = None


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
    event = torch.npu.ExternalEvent()
    event.wait(stream)
    event.reset(stream)
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
            query.shape[0], attention_mask, block_table,
            actual_seq_lengths_q, actual_seq_lengths_kv, block_size,
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
        workspace_key = (
            tuple(query.shape),
            tuple(key_cache.shape),
            query.dtype,
            num_kv_heads,
            num_heads,
            block_size,
            False,
            None,
            None,
        )
        workspace = _CAPTURED_FIA_WORKSPACES.get(workspace_key)
    if workspace is None:
        workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(**args)
        _CAPTURED_FIA_WORKSPACES[workspace_key] = workspace
    stream = torch.npu.current_stream()
    event = torch.npu.ExternalEvent()
    event.wait(stream)
    event.reset(stream)
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
        self.update_stream = torch.npu.Stream() if enabled else None
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
        self.task_update_replay_counts = {
            kind: 0 for kind in ("generic", "draft", "target")
        }
        self.task_update_task_counts = {
            kind: 0 for kind in ("generic", "draft", "target")
        }
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
        self.disabled_entry_keys: set[tuple[str, int]] = set()
        self.expected_fia_batch_size: int | None = None
        self.last_fia_shape: tuple[int, ...] = ()
        self.last_fia_expected_batch_size: int | None = None
        self.last_target_execution = NativeGraphExecution()
        self.last_draft_execution = NativeGraphExecution()
        self.last_generic_execution = NativeGraphExecution()
        self.last_target_validation_error: dict[str, Any] | None = None
        # A benchmark may lazily discover and changed-input qualify only the
        # shapes exercised by its real request trace, then seal that resident
        # set before starting the measured window.  Sealing is deliberately a
        # fail-closed audit mode: an unseen or unqualified shape is an error,
        # never an implicit capture, validation replay, or eager fallback that
        # would contaminate timed throughput.
        self.graph_cache_sealed = False
        self.pruned_unvalidated_entry_counts = {
            kind: 0 for kind in ("generic", "draft", "target")
        }

    def graph_qualification_status(self) -> dict[str, int]:
        """Return the resident graph inventory used by strict benchmarks."""

        entry_groups = {
            "generic": self.entries,
            "draft": self.draft_entries,
            "target": self.target_entries,
        }
        status = {
            f"{kind}_entries": len(entries)
            for kind, entries in entry_groups.items()
        }
        status.update(
            {
                f"{kind}_unvalidated_entries": sum(
                    not bool(entry.runtime_validated)
                    for entry in entries.values()
                )
                for kind, entries in entry_groups.items()
            }
        )
        status["unvalidated_entries"] = sum(
            status[f"{kind}_unvalidated_entries"]
            for kind in entry_groups
        )
        status["disabled_entries"] = len(self.disabled_entry_keys)
        status.update(
            {
                f"{kind}_pruned_unvalidated_entries": count
                for kind, count in self.pruned_unvalidated_entry_counts.items()
            }
        )
        status["pruned_unvalidated_entries"] = sum(
            self.pruned_unvalidated_entry_counts.values()
        )
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
            raise RuntimeError(
                "Cannot seal the ACLGraph cache while validate-every-replay "
                "diagnostics are enabled."
            )
        status = self.graph_qualification_status()
        if status["disabled_entries"]:
            raise RuntimeError(
                "Cannot seal an ACLGraph cache containing disabled entries: "
                f"disabled={status['disabled_entries']}."
            )
        if status["unvalidated_entries"]:
            details = ", ".join(
                f"{kind}={status[f'{kind}_unvalidated_entries']}"
                for kind in ("generic", "draft", "target")
                if status[f"{kind}_unvalidated_entries"]
            )
            raise RuntimeError(
                "Cannot seal an ACLGraph cache with unqualified resident "
                f"entries: {details}."
            )
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
            raise RuntimeError(
                "Cannot prune unvalidated entries from a sealed ACLGraph cache."
            )
        entry_groups = {
            "generic": self.entries,
            "draft": self.draft_entries,
            "target": self.target_entries,
        }
        stale_keys = {
            kind: [
                key
                for key, entry in entries.items()
                if not bool(entry.runtime_validated)
            ]
            for kind, entries in entry_groups.items()
        }
        stale_count = sum(len(keys) for keys in stale_keys.values())
        if not stale_count:
            return self.graph_qualification_status()
        if stale_count > self._capture_budget_used:
            raise RuntimeError(
                "ACLGraph resident capture accounting is smaller than the "
                "unvalidated prune set."
            )
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
            "A sealed ACLGraph cache refused a non-replay execution: "
            f"kind={kind}, reason={reason}{suffix}."
        )

    def _record_execution(
        self,
        kind: str,
        execution: NativeGraphExecution,
        output: Any,
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
        counters["logical_row_expansion_validation_calls"] += int(
            logical_row_expansion
        )
        counters["runtime_validation_failures"] += int(failed)
        self.runtime_validation_replay_count += 1

    def graph_execution_metrics(self) -> dict[str, int]:
        """Return stable, flat per-path counters for benchmark auditing."""
        metrics = {
            f"{kind}_{name}": value
            for kind, counters in self.execution_counters.items()
            for name, value in counters.items()
        }
        metrics.update(
            {
                f"{kind}_task_update_replays": self.task_update_replay_counts[
                    kind
                ]
                for kind in self.task_update_replay_counts
            }
        )
        metrics.update(
            {
                f"{kind}_task_update_tasks": self.task_update_task_counts[kind]
                for kind in self.task_update_task_counts
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
        if self.update_stream is None or not callable(
            getattr(self.update_stream, "wait_event", None)
        ):
            raise RuntimeError(
                "Draft ACLGraph replay-first task update requires an auxiliary "
                "stream with wait_event support."
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
                raise RuntimeError(
                    "Draft ACLGraph replay-first task update requires "
                    "torch.npu.Event support."
                )
            try:
                event = event_factory()
            except Exception as exc:
                raise RuntimeError(
                    "Draft ACLGraph replay-first task update could not create "
                    "its input-readiness event."
                ) from exc
            if not callable(getattr(event, "record", None)):
                raise RuntimeError(
                    "Draft ACLGraph replay-first task update created an event "
                    "without record support."
                )
            entry.replay_first_copy_done_event = event
        return event

    def _generic_replay_first_copy_event(
        self,
        entry: NativeACLGraphEntry,
    ) -> Any:
        """Validate the generic target replay-first ExternalEvent contract."""
        if self.update_stream is None or not callable(
            getattr(self.update_stream, "wait_event", None)
        ):
            raise RuntimeError(
                "Generic target ACLGraph replay-first task update requires an "
                "auxiliary stream with wait_event support."
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
                raise RuntimeError(
                    "Generic target ACLGraph replay-first task update requires "
                    "torch.npu.Event support."
                )
            try:
                event = event_factory()
            except Exception as exc:
                raise RuntimeError(
                    "Generic target ACLGraph replay-first task update could not "
                    "create its input-readiness event."
                ) from exc
            if not callable(getattr(event, "record", None)):
                raise RuntimeError(
                    "Generic target ACLGraph replay-first task update created "
                    "an event without record support."
                )
            entry.replay_first_copy_done_event = event
        return event

    @staticmethod
    def _tensor_changed(captured: torch.Tensor, incoming: torch.Tensor) -> bool:
        return (
            captured.shape != incoming.shape
            or captured.dtype != incoming.dtype
            or not torch.equal(captured, incoming)
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
    ) -> torch.Tensor:
        """Replay all draft proposal steps inside one ACLGraph."""
        self.last_draft_execution = NativeGraphExecution()
        if not positions or len(positions) != len(attention_metadatas):
            raise ValueError("A draft ACLGraph requires matching non-empty step metadata.")
        input_row_count = int(input_ids.shape[0])
        if valid_row_count is None:
            valid_row_count = input_row_count
        else:
            valid_row_count = int(valid_row_count)
        if valid_row_count <= 0 or valid_row_count > input_row_count:
            raise ValueError(
                "A draft ACLGraph valid row count must be in "
                f"[1, {input_row_count}], got {valid_row_count}."
            )
        if not self.enabled:
            self._raise_if_graph_cache_sealed("draft", "disabled")
            execution = NativeGraphExecution("eager", "disabled")
            output = self._execute_draft(
                input_ids, positions, attention_metadatas, vocabulary_size
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
                input_ids, positions, attention_metadatas, vocabulary_size
            )
            return self._record_execution("draft", execution, output)
        if all(attention_modes):
            query_shape = ",".join(str(length) for length in attention_metadatas[0].actual_seq_lengths_q)
            attention_key = f"fia:{query_shape}"
        else:
            attention_key = "paged"
        entry_key = (
            f"draft-greedy:{vocabulary_size}|steps:{len(positions)}|{attention_key}",
            input_ids.shape[0],
        )
        if entry_key in self.disabled_entry_keys:
            self._raise_if_graph_cache_sealed(
                "draft", "disabled_entry", entry_key
            )
            execution = NativeGraphExecution("eager", "disabled_entry")
            output = self._execute_draft(
                input_ids, positions, attention_metadatas, vocabulary_size
            )
            return self._record_execution("draft", execution, output)
        entry = self.draft_entries.get(entry_key)
        if entry is None:
            self._raise_if_graph_cache_sealed(
                "draft", "missing_entry", entry_key
            )
            if (
                len(self.entries) + len(self.draft_entries) + len(self.target_entries)
                >= self.max_graph_entries
                or self._capture_budget_used >= self.max_graph_entries
            ):
                self.capacity_fallback_count += 1
                execution = NativeGraphExecution("eager", "entry_capacity")
                output = self._execute_draft(
                    input_ids, positions, attention_metadatas, vocabulary_size
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
            )
            return self._record_execution(
                "draft", self.last_draft_execution, output
            )
        changed_input = (
            not entry.runtime_validated
            and self._draft_inputs_changed(
                entry, input_ids, positions, attention_metadatas
            )
        )
        validated_real_row_count = getattr(
            entry, "validated_real_row_count", input_row_count
        )
        if not isinstance(validated_real_row_count, int):
            validated_real_row_count = input_row_count
        logical_row_expansion = valid_row_count > validated_real_row_count
        if not entry.runtime_validated:
            self._raise_if_graph_cache_sealed(
                "draft", "unvalidated_entry", entry_key
            )
        if logical_row_expansion:
            self._raise_if_graph_cache_sealed(
                "draft", "logical_row_expansion", entry_key
            )
        assert self.update_stream is not None
        current_stream = torch.npu.current_stream()
        inline_update = envs.VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE
        replay_first_update = (
            envs.VLLM_ASCEND_PEARL_DRAFT_REPLAY_FIRST_TASK_UPDATE
        )
        if inline_update and replay_first_update:
            raise RuntimeError(
                "VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE and "
                "VLLM_ASCEND_PEARL_DRAFT_REPLAY_FIRST_TASK_UPDATE are "
                "mutually exclusive."
            )
        copy_done_event = (
            self._draft_replay_first_copy_event(entry)
            if replay_first_update
            else None
        )
        self._copy_draft_inputs(entry, input_ids, positions, attention_metadatas)
        if replay_first_update:
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
                    "Draft ACLGraph replay-first dependency setup failed before "
                    "graph submission."
                ) from exc
            try:
                entry.graph.replay()
            except Exception as exc:
                raise RuntimeError(
                    "Draft ACLGraph replay-first graph submission failed before "
                    "attention-task update."
                ) from exc
            try:
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
            self._update_draft_attention_tasks(entry, stream=current_stream)
        else:
            self.update_stream.wait_stream(current_stream)
            self._update_draft_attention_tasks(entry)
            # Match the target replay contract: graph replay must not race an
            # unfinished task update on the auxiliary stream.  The captured
            # ExternalEvents order each attention task, while this dependency
            # also protects CANN's graph-task metadata between consecutive
            # multi-step draft replays.
            current_stream.wait_stream(self.update_stream)
        if not replay_first_update:
            entry.graph.replay()
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
            )
            torch.npu.synchronize()
            validation_failed = not self._draft_outputs_match(
                graph_output,
                reference_output,
                attention_metadatas,
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
                return self._record_execution(
                    "draft", execution, reference_output
                )
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
                    size, getattr(metadata, "tree_attention_mask", None), metadata.request_block_tables,
                    metadata.actual_seq_lengths_q, metadata.sequence_lens, block_size,
                )
            shape_signature = tuple(
                (
                    tuple(metadata.tree_attention_mask.shape),
                    tuple(metadata.request_block_tables.shape),
                ) for metadata in attention_metadatas
            )
            # Query/KV *values* and both exact length lists are updated through
            # the captured FIA tasks.  Do not round KV lengths to a power-of-two
            # bucket: on Ascend BF16 FULL-mask FIA, appending masked KV columns
            # can change reduction order enough to change a greedy token.  The
            # query partition is a host literal rather than a tensor extent, so
            # it must not split otherwise identical graphs either.  Fixed input
            # and mask shapes are the complete capture contract; changed query
            # partitions or KV lengths take the task-update path below.
            attention_key = (
                "tree-fia-full:"
                f"exact-length-task-update|buffers:{shape_signature}"
            )
        elif all(attention_modes):
            # Packed causal FIA can have the same total query-token count but
            # a different number of request segments.  Those segments own one
            # page-table row each, so reusing an entry keyed only by token
            # count tries to copy e.g. [32, blocks] into a captured
            # [8, blocks] buffer.  Length values remain task-update arguments;
            # only graph-owned buffer shapes belong in this key.
            shape_signature = tuple(
                (
                    tuple(metadata.request_block_tables.shape)
                    if metadata.request_block_tables is not None
                    else None,
                    tuple(metadata.block_tables.shape),
                    tuple(metadata.attention_mask.shape)
                    if metadata.attention_mask is not None
                    else None,
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
            self._raise_if_graph_cache_sealed(
                "target", "disabled_entry", entry_key
            )
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
            self._raise_if_graph_cache_sealed(
                "target", "missing_entry", entry_key
            )
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
                "target", self.last_target_execution, output
            )
        if not entry.runtime_validated:
            self._raise_if_graph_cache_sealed(
                "target", "unvalidated_entry", entry_key
            )
        task_lengths_unchanged = (
            entry.actual_seq_lengths_q
            == tuple(metadata.actual_seq_lengths_q for metadata in attention_metadatas)
            and entry.sequence_lens
            == tuple(metadata.sequence_lens for metadata in attention_metadatas)
        )
        changed_input = (
            not entry.runtime_validated
            and self._target_inputs_changed(
                entry, input_ids, positions, attention_metadatas
            )
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
            validation_failed = not self._target_outputs_match(
                graph_outputs, reference_outputs
            )
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
                execution = NativeGraphExecution(
                    "eager", "runtime_validation", replay_executed=True
                )
                warnings.warn(
                    "Native PEARL disabled a target ACLGraph whose runtime replay did not match eager execution: "
                    f"{entry_key!r}; diagnostics={json.dumps(self.last_target_validation_error)}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self._record_execution(
                    "target", execution, reference_outputs
                )
            entry.runtime_validated = True
        return self._record_execution("target", execution, entry.outputs)

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
        with _collect_graph_tasks(
            fia_workspaces=self._fia_workspace_pool,
        ) as tasks, torch.npu.graph(graph):
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
        if not (
            len(entry.input_ids)
            == len(input_ids)
            == len(positions)
            == len(attention_metadatas)
        ):
            return True
        for step, (step_input, step_positions, metadata) in enumerate(
            zip(input_ids, positions, attention_metadatas)
        ):
            if NativeACLGraphRunner._tensor_changed(
                entry.input_ids[step], step_input
            ) or NativeACLGraphRunner._tensor_changed(
                entry.positions[step], step_positions
            ) or NativeACLGraphRunner._tensor_changed(
                entry.slot_mappings[step], metadata.slot_mapping
            ):
                return True
            captured_attention_mask = entry.attention_masks[step]
            incoming_attention_mask = getattr(metadata, "attention_mask", None)
            if (captured_attention_mask is None) != (
                incoming_attention_mask is None
            ):
                return True
            if (
                captured_attention_mask is not None
                and NativeACLGraphRunner._tensor_changed(
                    captured_attention_mask, incoming_attention_mask
                )
            ):
                return True
            tree_masks = getattr(entry, "tree_attention_masks", ())
            captured_tree_mask = tree_masks[step] if tree_masks else None
            incoming_tree_mask = getattr(metadata, "tree_attention_mask", None)
            if (captured_tree_mask is None) != (incoming_tree_mask is None):
                return True
            if (
                captured_tree_mask is not None
                and NativeACLGraphRunner._tensor_changed(
                    captured_tree_mask, incoming_tree_mask
                )
            ):
                return True
            if entry.request_block_tables[step] is not None:
                if (
                    metadata.request_block_tables is None
                    or NativeACLGraphRunner._tensor_changed(
                        entry.request_block_tables[step],
                        metadata.request_block_tables,
                    )
                ):
                    return True
            elif NativeACLGraphRunner._tensor_changed(
                entry.context_lens[step], metadata.context_lens
            ) or NativeACLGraphRunner._tensor_changed(
                entry.block_tables[step], metadata.block_tables
            ):
                return True
            if (
                entry.actual_seq_lengths_q[step]
                != metadata.actual_seq_lengths_q
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
    ) -> torch.Tensor:
        outputs = []
        step_input = input_ids
        for step_positions, metadata in zip(positions, attention_metadatas):
            hidden_states = self.model(step_input, step_positions, metadata)
            step_input = self.model.compute_greedy_tokens(hidden_states, vocabulary_size)
            outputs.append(step_input)
        return torch.stack(outputs, dim=1)

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
            or graph_output.ndim != 2
            or len(attention_metadatas) != graph_output.shape[1]
        ):
            return False
        row_count = graph_output.shape[0]
        if valid_row_count <= 0 or valid_row_count > row_count:
            return False
        for metadata in attention_metadatas:
            slot_mapping = getattr(metadata, "slot_mapping", None)
            if (
                not torch.is_tensor(slot_mapping)
                or slot_mapping.ndim != 1
                or slot_mapping.numel() != row_count
            ):
                return False
            if not bool(torch.all(slot_mapping[:valid_row_count] >= 0).item()):
                return False
            if valid_row_count < row_count and not bool(
                torch.all(slot_mapping[valid_row_count:] < 0).item()
            ):
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
    ) -> torch.Tensor:
        reference_output = self._execute_draft(
            input_ids,
            positions,
            attention_metadatas,
            vocabulary_size,
        ).clone()
        torch.npu.synchronize()
        captured_input_ids = input_ids.clone()
        captured_positions = tuple(value.clone() for value in positions)
        captured_metadatas = []
        captured_attention_masks: list[torch.Tensor | None] = []
        for metadata in attention_metadatas:
            captured_request_block_tables = (
                metadata.request_block_tables.clone()
                if metadata.use_fused_infer_attention and metadata.request_block_tables is not None
                else None
            )
            captured_attention_mask = (
                metadata.attention_mask.clone()
                if metadata.attention_mask is not None
                else None
            )
            captured_metadatas.append(
                type(metadata)(
                    slot_mapping=metadata.slot_mapping.clone(),
                    context_lens=metadata.context_lens.clone(),
                    block_tables=metadata.block_tables.clone(),
                    actual_seq_lengths_q=metadata.actual_seq_lengths_q,
                    sequence_lens=metadata.sequence_lens,
                    request_block_tables=captured_request_block_tables,
                    attention_mask=captured_attention_mask,
                    use_fused_infer_attention=metadata.use_fused_infer_attention,
                )
            )
            captured_attention_masks.append(captured_attention_mask)
        graph = torch.npu.NPUGraph()
        with _collect_graph_tasks(
            fia_workspaces=self._fia_workspace_pool,
        ) as tasks, torch.npu.graph(graph):
            output = self._execute_draft(
                captured_input_ids,
                list(captured_positions),
                captured_metadatas,
                vocabulary_size,
            )
        if len(tasks) % len(captured_metadatas):
            raise RuntimeError("Draft ACLGraph attention tasks do not divide evenly across proposal steps.")
        tasks_per_step = len(tasks) // len(captured_metadatas)
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
            validated_real_row_count=valid_row_count,
        )
        self.draft_entries[entry_key] = entry
        current_stream = torch.npu.current_stream()
        current_stream.synchronize()
        assert self.update_stream is not None
        if envs.VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE:
            self._update_draft_attention_tasks(entry, stream=current_stream)
        else:
            self.update_stream.wait_stream(current_stream)
            self._update_draft_attention_tasks(entry)
            current_stream.wait_stream(self.update_stream)
        entry.graph.replay()
        torch.npu.synchronize()
        if not self._draft_outputs_match(
            output,
            reference_output,
            captured_metadatas,
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
        for step, (step_positions, metadata) in enumerate(
            zip(positions, attention_metadatas)
        ):
            if NativeACLGraphRunner._tensor_changed(
                entry.positions[step], step_positions
            ) or NativeACLGraphRunner._tensor_changed(
                entry.slot_mappings[step], metadata.slot_mapping
            ):
                return True
            if entry.request_block_tables[step] is not None:
                if metadata.request_block_tables is None or NativeACLGraphRunner._tensor_changed(
                    entry.request_block_tables[step], metadata.request_block_tables
                ):
                    return True
            elif NativeACLGraphRunner._tensor_changed(
                entry.context_lens[step], metadata.context_lens
            ) or NativeACLGraphRunner._tensor_changed(
                entry.block_tables[step], metadata.block_tables
            ):
                return True
            attention_masks = getattr(entry, "attention_masks", ())
            captured_mask = attention_masks[step] if attention_masks else None
            incoming_mask = getattr(metadata, "attention_mask", None)
            if (captured_mask is None) != (incoming_mask is None):
                return True
            if captured_mask is not None and NativeACLGraphRunner._tensor_changed(
                captured_mask, incoming_mask
            ):
                return True
            if (
                entry.actual_seq_lengths_q[step]
                != metadata.actual_seq_lengths_q
                or entry.sequence_lens[step] != metadata.sequence_lens
            ):
                return True
        return False

    @staticmethod
    def _copy_draft_inputs(
        entry: NativeDraftACLGraphEntry,
        input_ids: torch.Tensor,
        positions: list[torch.Tensor],
        attention_metadatas: list[Any],
    ) -> None:
        entry.input_ids.copy_(input_ids)
        for step, (step_positions, metadata) in enumerate(zip(positions, attention_metadatas)):
            entry.positions[step].copy_(step_positions)
            entry.slot_mappings[step].copy_(metadata.slot_mapping)
            attention_masks = getattr(entry, "attention_masks", ())
            if attention_masks and attention_masks[step] is not None:
                if metadata.attention_mask is None:
                    raise RuntimeError(
                        "Draft ACLGraph attention mask disappeared between replays"
                    )
                attention_masks[step].copy_(metadata.attention_mask)
            if entry.request_block_tables[step] is not None:
                assert metadata.request_block_tables is not None
                entry.request_block_tables[step].copy_(metadata.request_block_tables)
            else:
                entry.context_lens[step].copy_(metadata.context_lens)
                entry.block_tables[step].copy_(metadata.block_tables)
        entry.actual_seq_lengths_q = tuple(metadata.actual_seq_lengths_q for metadata in attention_metadatas)
        entry.sequence_lens = tuple(metadata.sequence_lens for metadata in attention_metadatas)

    def _run(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata,
        *,
        output_kind: str,
        output_transform: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> torch.Tensor:
        self.last_generic_execution = NativeGraphExecution()
        num_tokens = input_ids.shape[0]
        if not self.enabled or num_tokens > self.max_graph_tokens:
            reason = "disabled" if not self.enabled else "token_capacity"
            self._raise_if_graph_cache_sealed("generic", reason)
            if reason == "token_capacity":
                self.shape_fallback_count += 1
            execution = NativeGraphExecution("eager", reason)
            output = self._execute(
                input_ids, positions, attention_metadata, output_transform
            )
            return self._record_execution("generic", execution, output)
        if attention_metadata.use_fused_infer_attention:
            self.last_fia_shape = tuple(attention_metadata.actual_seq_lengths_q)
            self.last_fia_expected_batch_size = self.expected_fia_batch_size
            if not self._is_reusable_fia_shape(attention_metadata.actual_seq_lengths_q):
                self._raise_if_graph_cache_sealed("generic", "shape")
                self.shape_fallback_count += 1
                execution = NativeGraphExecution("eager", "shape")
                output = self._execute(
                    input_ids, positions, attention_metadata, output_transform
                )
                return self._record_execution("generic", execution, output)
            query_shape = ",".join(str(length) for length in attention_metadata.actual_seq_lengths_q)
            output_kind = f"{output_kind}|fia:{query_shape}"
            capture_size = num_tokens
        else:
            capture_size = min(
                (size for kind, size in self.entries if kind == output_kind and size >= num_tokens),
                default=num_tokens,
            )
        padded_inputs = self._pad_inputs(input_ids, positions, attention_metadata, capture_size)
        padded_input_ids, padded_positions, padded_metadata = padded_inputs
        entry_key = (output_kind, capture_size)
        if entry_key in self.disabled_entry_keys:
            self._raise_if_graph_cache_sealed(
                "generic", "disabled_entry", entry_key
            )
            execution = NativeGraphExecution("eager", "disabled_entry")
            output = self._execute(
                input_ids, positions, attention_metadata, output_transform
            )
            return self._record_execution("generic", execution, output)
        entry = self.entries.get(entry_key)
        if entry is None:
            self._raise_if_graph_cache_sealed(
                "generic", "missing_entry", entry_key
            )
            if len(self.entries) >= self.max_graph_entries or self._capture_budget_used >= self.max_graph_entries:
                self.capacity_fallback_count += 1
                execution = NativeGraphExecution("eager", "entry_capacity")
                output = self._execute(
                    input_ids, positions, attention_metadata, output_transform
                )
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
            )
        changed_input = (
            not entry.runtime_validated
            and self._generic_inputs_changed(
                entry,
                padded_input_ids,
                padded_positions,
                padded_metadata,
            )
        )
        validated_real_row_count = getattr(
            entry, "validated_real_row_count", capture_size
        )
        if not isinstance(validated_real_row_count, int):
            validated_real_row_count = capture_size
        logical_row_expansion = num_tokens > validated_real_row_count
        if not entry.runtime_validated:
            self._raise_if_graph_cache_sealed(
                "generic", "unvalidated_entry", entry_key
            )
        if logical_row_expansion:
            self._raise_if_graph_cache_sealed(
                "generic", "logical_row_expansion", entry_key
            )
        current_stream = torch.npu.current_stream()
        inline_update = envs.VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE
        replay_first_update = (
            envs.VLLM_ASCEND_PEARL_TARGET_REPLAY_FIRST_TASK_UPDATE
        )
        if inline_update and replay_first_update:
            raise RuntimeError(
                "VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE and "
                "VLLM_ASCEND_PEARL_TARGET_REPLAY_FIRST_TASK_UPDATE are "
                "mutually exclusive."
            )
        copy_done_event = (
            self._generic_replay_first_copy_event(entry)
            if replay_first_update
            else None
        )
        self._copy_inputs(entry, padded_input_ids, padded_positions, padded_metadata)
        if envs.VLLM_ASCEND_PEARL_SYNC_GRAPH_INPUTS:
            current_stream.synchronize()
        # Order task updates after the previous replay and the current input
        # copies. CANN releases a replay early when updates are issued on an
        # auxiliary stream, so an inline mode is available for runtimes where
        # ExternalEvent visibility is not sufficient for multi-token PA.
        assert self.update_stream is not None
        if replay_first_update:
            assert copy_done_event is not None
            try:
                copy_done_event.record(current_stream)
                # The update stream cannot touch graph-owned metadata until
                # this replay's input copies and the preceding current-stream
                # replay have completed.  The graph itself then waits on each
                # captured task's ExternalEvent while its prefix can overlap
                # with the host-side task refresh submissions.
                self.update_stream.wait_event(copy_done_event)
            except Exception as exc:
                raise RuntimeError(
                    "Generic target ACLGraph replay-first dependency setup "
                    "failed before graph submission."
                ) from exc
            try:
                entry.graph.replay()
            except Exception as exc:
                raise RuntimeError(
                    "Generic target ACLGraph replay-first graph submission "
                    "failed before attention-task update."
                ) from exc
            try:
                self._update_attention_tasks(entry)
            except Exception as exc:
                # The submitted graph may already be waiting on one of its
                # captured ExternalEvents.  Eager fallback could observe
                # partial KV writes or stale outputs, so terminate this worker
                # instead of returning a result with unknown provenance.
                raise RuntimeError(
                    "Generic target ACLGraph replay-first attention-task "
                    "update failed after graph submission; the runner cannot "
                    "safely fall back for this replay."
                ) from exc
        elif not inline_update:
            self.update_stream.wait_stream(current_stream)
            self._update_attention_tasks(entry)
        else:
            self._update_attention_tasks(entry, stream=current_stream)
        if not inline_update and not replay_first_update:
            # Attention task updates are issued on the auxiliary stream. The
            # reverse dependency prevents replay from consuming old metadata.
            current_stream.wait_stream(self.update_stream)
        if not inline_update and envs.VLLM_ASCEND_PEARL_SYNC_GRAPH_TASK_UPDATE:
            self.update_stream.synchronize()
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
            validation_failed = not self._outputs_match(
                graph_output, reference_output
            )
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
                execution = NativeGraphExecution(
                    "eager", "runtime_validation", replay_executed=True
                )
                warnings.warn(
                    "Native PEARL disabled an ACLGraph entry whose runtime replay did not match eager execution: "
                    f"{entry_key!r}; {self._mismatch_summary(graph_output, reference_output)}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self._record_execution(
                    "generic", execution, reference_output
                )
            entry.runtime_validated = True
            entry.validated_real_row_count = max(
                validated_real_row_count,
                num_tokens,
            )
        if (
            envs.VLLM_ASCEND_PEARL_VALIDATE_GRAPH_REPLAYS
            and not runtime_validation_performed
        ):
            torch.npu.synchronize()
            replay_output = entry.output[:num_tokens].clone()
            replay_reference = self._execute(
                input_ids,
                positions,
                attention_metadata,
                output_transform,
            )
            torch.npu.synchronize()
            replay_differs = not self._outputs_match(
                replay_output, replay_reference
            )
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
                execution = NativeGraphExecution(
                    "eager", "runtime_validation", replay_executed=True
                )
                warnings.warn(
                    "Native PEARL disabled an ACLGraph entry whose diagnostic "
                    "runtime replay did not match eager execution: "
                    f"{entry_key!r}; "
                    f"{self._mismatch_summary(replay_output, replay_reference)}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self._record_execution(
                    "generic", execution, replay_reference
                )
        return self._record_execution(
            "generic", execution, entry.output[:num_tokens]
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
        padded_context_lens = torch.zeros(capture_size, dtype=attention_metadata.context_lens.dtype)
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
        padded_metadata = type(attention_metadata)(
            slot_mapping=padded_slot_mapping,
            context_lens=padded_context_lens,
            block_tables=padded_block_tables,
        )
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
                "A generic ACLGraph valid row count must be in "
                f"[1, {int(input_ids.shape[0])}], got {valid_row_count}."
            )
        # Warm up allocations and collectives before entering stream capture.
        reference_output = self._execute(
            input_ids,
            positions,
            attention_metadata,
            output_transform,
        ).clone()
        torch.npu.synchronize()
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
            attention_metadata.attention_mask.clone()
            if attention_metadata.attention_mask is not None
            else None
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
        graph = torch.npu.NPUGraph()
        with _collect_graph_tasks(
            fia_workspaces=self._fia_workspace_pool,
        ) as tasks, torch.npu.graph(graph):
            output = self._execute(
                captured_input_ids,
                captured_positions,
                captured_metadata,
                output_transform,
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
        )
        self.entries[entry_key] = entry
        # Capture execution only records the graph; CANN does not guarantee
        # that its output buffers contain a valid inference result. Rebind the
        # host metadata and replay once before serving the triggering request.
        current_stream = torch.npu.current_stream()
        current_stream.synchronize()
        assert self.update_stream is not None
        if envs.VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE:
            self._update_attention_tasks(entry, stream=current_stream)
        else:
            self.update_stream.wait_stream(current_stream)
            self._update_attention_tasks(entry)
            current_stream.wait_stream(self.update_stream)
        entry.graph.replay()
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
        if (
            graph_output.shape != reference_output.shape
            or graph_output.dtype != reference_output.dtype
        ):
            return False
        if graph_output.dtype.is_floating_point:
            return torch.allclose(graph_output, reference_output, rtol=1e-3, atol=1e-3)
        return torch.equal(graph_output, reference_output)

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
            or NativeACLGraphRunner._tensor_changed(
                entry.slot_mapping, attention_metadata.slot_mapping
            )
        ):
            return True
        if entry.request_block_tables is not None:
            if (
                attention_metadata.request_block_tables is None
                or NativeACLGraphRunner._tensor_changed(
                    entry.request_block_tables,
                    attention_metadata.request_block_tables,
                )
            ):
                return True
        elif NativeACLGraphRunner._tensor_changed(
            entry.context_lens, attention_metadata.context_lens
        ) or NativeACLGraphRunner._tensor_changed(
            entry.block_tables, attention_metadata.block_tables
        ):
            return True
        captured_attention_mask = getattr(entry, "attention_mask", None)
        incoming_attention_mask = getattr(attention_metadata, "attention_mask", None)
        if (captured_attention_mask is None) != (incoming_attention_mask is None):
            return True
        if (
            captured_attention_mask is not None
            and NativeACLGraphRunner._tensor_changed(
                captured_attention_mask, incoming_attention_mask
            )
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
                raise RuntimeError(
                    "ACLGraph attention mask disappeared between replays"
                )
            captured_attention_mask.copy_(attention_metadata.attention_mask)
        if entry.request_block_tables is not None:
            assert attention_metadata.request_block_tables is not None
            entry.request_block_tables.copy_(attention_metadata.request_block_tables)
        else:
            entry.context_lens.copy_(attention_metadata.context_lens)
            entry.block_tables.copy_(attention_metadata.block_tables)
        entry.actual_seq_lengths_q = attention_metadata.actual_seq_lengths_q
        entry.sequence_lens = attention_metadata.sequence_lens

    def _update_attention_tasks(
        self,
        entry: NativeACLGraphEntry,
        *,
        stream: torch.npu.Stream | None = None,
    ) -> None:
        task_lengths = [(entry.actual_seq_lengths_q, entry.sequence_lens)] * len(entry.tasks)
        self._update_attention_task_list(entry.tasks, task_lengths, stream=stream)

    def _update_draft_attention_tasks(
        self,
        entry: NativeDraftACLGraphEntry,
        *,
        stream: torch.npu.Stream | None = None,
    ) -> None:
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
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Draft ACLGraph attention-task update failed for "
                f"batch={entry.input_ids.shape[0]}, "
                f"steps={len(entry.positions)}, tasks={len(entry.tasks)}."
            ) from exc

    def _update_target_attention_tasks(self, entry: NativeTargetACLGraphEntry) -> None:
        task_lengths = [
            lengths
            for lengths in zip(entry.actual_seq_lengths_q, entry.sequence_lens)
            for _ in range(entry.tasks_per_step)
        ]
        self._update_attention_task_list(entry.tasks, task_lengths)

    def _update_attention_task_list(
        self,
        tasks: list[NativePagedAttentionGraphTask | NativeFusedInferAttentionGraphTask],
        task_lengths: list[tuple[tuple[int, ...], tuple[int, ...]]],
        *,
        stream: torch.npu.Stream | None = None,
    ) -> None:
        update_stream = stream or self.update_stream
        assert update_stream is not None
        if len(tasks) != len(task_lengths):
            raise RuntimeError("ACLGraph attention tasks and replay metadata differ in length.")
        fia_length_lists: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            tuple[list[int], list[int]],
        ] = {}
        # PagedAttention workspaces are shared only within this serial task
        # update.  The exact context lengths participate in the key because
        # CANN may select a different workspace/tiling as sequences grow.
        paged_workspaces: dict[tuple[Any, ...], torch.Tensor] = {}
        with torch.npu.stream(update_stream):
            for task, (actual_seq_lengths_q, sequence_lens) in zip(tasks, task_lengths):
                if isinstance(task, NativeFusedInferAttentionGraphTask):
                    length_key = (actual_seq_lengths_q, sequence_lens)
                    length_lists = fia_length_lists.get(length_key)
                    if length_lists is None:
                        length_lists = (
                            list(actual_seq_lengths_q),
                            list(sequence_lens),
                        )
                        fia_length_lists[length_key] = length_lists
                    self._update_fused_infer_attention_task(
                        task,
                        *length_lists,
                        stream=update_stream,
                    )
                    continue
                # Keep the native runner ABI-aligned with vLLM-Ascend's
                # production PagedAttention graph updater.  Workspace sizing
                # may depend on the current context-length tensor/tiling; a
                # buffer obtained while capturing an earlier sequence length
                # is not a valid lifetime-wide cache key.  Retain the refreshed
                # tensor on the task so it remains alive through graph replay.
                workspace_key = (
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
                    tuple(int(value) for value in task.context_lens.tolist()),
                    tuple(task.output.shape),
                    task.output.dtype,
                )
                workspace = paged_workspaces.get(workspace_key)
                if workspace is None:
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
                task.event.record(update_stream)

    def _update_fused_infer_attention_task(
        self,
        task: NativeFusedInferAttentionGraphTask,
        actual_seq_lengths_q: list[int],
        sequence_lens: list[int],
        *,
        stream: torch.npu.Stream,
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
                task.query.shape[0], task.attention_mask, task.block_table,
                actual_seq_lengths_q, sequence_lens, task.block_size,
            )
            args.update(sparse_mode=TREE_FIA_SPARSE_MODE, inner_precise=TREE_FIA_INNER_PRECISE)
        else:
            # Keep task refresh ABI-identical to eager/capture construction.
            args.update(sparse_mode=3, next_tokens=0)
        torch.npu.graph_task_update_begin(stream, task.handle)
        torch_npu.npu_fused_infer_attention_score.out(
            **args,
            workspace=task.workspace,
            out=[task.output, task.softmax_lse],
        )
        torch.npu.graph_task_update_end(stream)
        task.event.record(stream)
