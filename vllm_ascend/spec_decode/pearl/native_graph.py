# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ACLGraph capture and paged-attention graph-task updates for native PEARL."""

from __future__ import annotations

import json
import math
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
MAX_GREEDY_REPLAY_EAGER_DIVERGENCE = 0.05
TREE_FIA_SPARSE_MODE = 1
TREE_FIA_INNER_PRECISE = 2


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
    runtime_validated: bool = False


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

# Match vLLM-Ascend's ordinary causal FIA path.  The explicit attention mask
# carries the causal/tree structure; these bounds must stay unbounded so FIA
# does not apply a second relative window to TND queries.
_FIA_INT_MAX = 2147483647


@contextmanager
def _collect_graph_tasks():
    global _CAPTURED_FIA_WORKSPACES, _CAPTURED_PA_WORKSPACES, _CAPTURED_TASKS
    if _CAPTURED_TASKS is not None:
        raise RuntimeError("Nested native PEARL ACLGraph capture is not supported.")
    tasks: list[NativePagedAttentionGraphTask | NativeFusedInferAttentionGraphTask] = []
    _CAPTURED_TASKS = tasks
    _CAPTURED_PA_WORKSPACES = {}
    _CAPTURED_FIA_WORKSPACES = {}
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
        query.dtype,
        num_kv_heads,
        num_heads,
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
        # Preserve the existing linear FIA contract, including its windows.
        args.update(sparse_mode=3, pre_tokens=_FIA_INT_MAX, next_tokens=_FIA_INT_MAX)
    if _CAPTURED_TASKS is None:
        torch_npu.npu_fused_infer_attention_score.out(
            **args,
            out=[output, softmax_lse],
        )
        return

    assert _CAPTURED_FIA_WORKSPACES is not None
    workspace_key = (
        tuple(query.shape),
        tuple(key_cache.shape),
        query.dtype,
        num_kv_heads,
        num_heads,
        block_size,
        tree_attention,
        tuple(attention_mask.shape) if tree_attention else None,
        tuple(block_table.shape) if tree_attention else None,
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
        self.runtime_validation_replay_count = 0
        self.disabled_entry_keys: set[tuple[str, int]] = set()
        self.expected_fia_batch_size: int | None = None
        self.last_fia_shape: tuple[int, ...] = ()
        self.last_fia_expected_batch_size: int | None = None
        self.last_target_execution = NativeGraphExecution()
        self.last_target_validation_error: dict[str, Any] | None = None

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
    ) -> torch.Tensor:
        """Replay all draft proposal steps inside one ACLGraph."""
        if not positions or len(positions) != len(attention_metadatas):
            raise ValueError("A draft ACLGraph requires matching non-empty step metadata.")
        if not self.enabled:
            return self._execute_draft(input_ids, positions, attention_metadatas, vocabulary_size)
        attention_modes = [metadata.use_fused_infer_attention for metadata in attention_metadatas]
        if any(attention_modes) != all(attention_modes):
            raise ValueError("Every step in a draft ACLGraph must use the same attention backend.")
        if (
            all(attention_modes)
            and self.expected_fia_batch_size is not None
            and input_ids.shape[0] != self.expected_fia_batch_size
        ):
            self.shape_fallback_count += 1
            return self._execute_draft(input_ids, positions, attention_metadatas, vocabulary_size)
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
            return self._execute_draft(input_ids, positions, attention_metadatas, vocabulary_size)
        entry = self.draft_entries.get(entry_key)
        if entry is None:
            if (
                len(self.entries) + len(self.draft_entries) + len(self.target_entries)
                >= self.max_graph_entries
                or self._capture_budget_used >= self.max_graph_entries
            ):
                self.capacity_fallback_count += 1
                return self._execute_draft(input_ids, positions, attention_metadatas, vocabulary_size)
            self.capture_attempt_count += 1
            self._capture_budget_used += 1
            return self._capture_draft(
                entry_key,
                input_ids,
                positions,
                attention_metadatas,
                vocabulary_size,
            )
        self._copy_draft_inputs(entry, input_ids, positions, attention_metadatas)
        assert self.update_stream is not None
        self.update_stream.wait_stream(torch.npu.current_stream())
        self._update_draft_attention_tasks(entry)
        entry.graph.replay()
        self.replay_count += 1
        return entry.output

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
            self.last_target_execution = NativeGraphExecution("eager", "disabled")
            return self._execute_target(
                input_ids, positions, attention_metadatas, vocabulary_size, output_kind=output_kind
            )
        attention_modes = [metadata.use_fused_infer_attention for metadata in attention_metadatas]
        if any(attention_modes) != all(attention_modes):
            raise ValueError("Every step in a target ACLGraph must use the same attention backend.")
        step_sizes = tuple(int(value.shape[0]) for value in input_ids)
        if max(step_sizes) > self.max_graph_tokens:
            self.shape_fallback_count += 1
            self.last_target_execution = NativeGraphExecution("eager", "token_capacity")
            return self._execute_target(
                input_ids, positions, attention_metadatas, vocabulary_size, output_kind=output_kind
            )
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
            attention_metadatas = [
                bucket_tree_fia_attention_metadata(
                    metadata, block_size=block_size
                )
                for metadata in attention_metadatas
            ]
            shape_signature = tuple(
                (
                    tuple(metadata.tree_attention_mask.shape),
                    tuple(metadata.request_block_tables.shape),
                    tuple(metadata.block_tables.shape),
                    tuple(metadata.attention_mask.shape) if metadata.attention_mask is not None else None,
                ) for metadata in attention_metadatas
            )
            # Query/KV *values* are updated with the FIA task. Only static
            # buffer shapes and the FULL-mask operator contract form the key.
            kv_buckets = tuple(
                metadata.sequence_lens for metadata in attention_metadatas
            )
            query_partitions = tuple(
                metadata.actual_seq_lengths_q for metadata in attention_metadatas
            )
            attention_key = (
                "tree-fia-full:"
                f"query-partitions:{query_partitions}|"
                f"kv-buckets:{kv_buckets}|buffers:{shape_signature}"
            )
        elif all(attention_modes):
            attention_key = "fia"
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
            self.last_target_execution = NativeGraphExecution("eager", "disabled_entry")
            return self._execute_target(
                input_ids, positions, reference_metadatas, vocabulary_size, output_kind=output_kind
            )
        entry = self.target_entries.get(entry_key)
        if entry is None:
            total_entries = len(self.entries) + len(self.draft_entries) + len(self.target_entries)
            if total_entries >= self.max_graph_entries or self._capture_budget_used >= self.max_graph_entries:
                self.capacity_fallback_count += 1
                self.last_target_execution = NativeGraphExecution("eager", "entry_capacity")
                return self._execute_target(
                    input_ids, positions, reference_metadatas, vocabulary_size, output_kind=output_kind
                )
            self.capture_attempt_count += 1
            self._capture_budget_used += 1
            return self._capture_target(
                entry_key,
                input_ids,
                positions,
                attention_metadatas,
                vocabulary_size,
                reference_metadatas=reference_metadatas,
                output_kind=output_kind,
            )
        task_lengths_unchanged = (
            entry.actual_seq_lengths_q
            == tuple(metadata.actual_seq_lengths_q for metadata in attention_metadatas)
            and entry.sequence_lens
            == tuple(metadata.sequence_lens for metadata in attention_metadatas)
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
            self.task_update_replay_count += 1
        entry.graph.replay()
        self.replay_count += 1
        self.last_target_execution = NativeGraphExecution("replay", replay_executed=True)
        if not entry.runtime_validated:
            self.runtime_validation_replay_count += 1
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
            if not self._target_outputs_match(graph_outputs, reference_outputs):
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
                self.last_target_execution = NativeGraphExecution("eager", "runtime_validation", replay_executed=True)
                warnings.warn(
                    "Native PEARL disabled a target ACLGraph whose runtime replay did not match eager execution: "
                    f"{entry_key!r}; diagnostics={json.dumps(self.last_target_validation_error)}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return reference_outputs
            entry.runtime_validated = True
        return entry.outputs

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
        with _collect_graph_tasks() as tasks, torch.npu.graph(graph):
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
            # Capture already executes a real replay and compares it against
            # the eager reference below.  A second eager 32B forward on the
            # first *later* replay used to turn tail shapes into 180-190 ms
            # decode stalls. Keep that changed-input qualification available
            # as an explicit diagnostic, never on the production hot path.
            runtime_validated=not envs.VLLM_ASCEND_PEARL_VALIDATE_GRAPH_REPLAYS,
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

    def _capture_draft(
        self,
        entry_key: tuple[str, int],
        input_ids: torch.Tensor,
        positions: list[torch.Tensor],
        attention_metadatas: list[Any],
        vocabulary_size: int,
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
        for metadata in attention_metadatas:
            captured_request_block_tables = (
                metadata.request_block_tables.clone()
                if metadata.use_fused_infer_attention and metadata.request_block_tables is not None
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
                    attention_mask=metadata.attention_mask,
                    use_fused_infer_attention=metadata.use_fused_infer_attention,
                )
            )
        graph = torch.npu.NPUGraph()
        with _collect_graph_tasks() as tasks, torch.npu.graph(graph):
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
        )
        self.draft_entries[entry_key] = entry
        current_stream = torch.npu.current_stream()
        current_stream.synchronize()
        assert self.update_stream is not None
        self.update_stream.wait_stream(current_stream)
        self._update_draft_attention_tasks(entry)
        current_stream.wait_stream(self.update_stream)
        entry.graph.replay()
        torch.npu.synchronize()
        if not self._outputs_match(output, reference_output):
            failed_entry = self.draft_entries.pop(entry_key)
            self._reset_graph_entry(failed_entry)
            self.disabled_entry_keys.add(entry_key)
            self.failed_capture_count += 1
            warnings.warn(
                "Native PEARL disabled a gamma-step draft ACLGraph whose replay did not match eager execution: "
                f"{entry_key!r}; {self._mismatch_summary(output, reference_output)}",
                RuntimeWarning,
                stacklevel=2,
            )
            return reference_output
        self.capture_count += 1
        self.replay_count += 1
        return output

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
        num_tokens = input_ids.shape[0]
        if not self.enabled or num_tokens > self.max_graph_tokens:
            return self._execute(input_ids, positions, attention_metadata, output_transform)
        if attention_metadata.use_fused_infer_attention:
            self.last_fia_shape = tuple(attention_metadata.actual_seq_lengths_q)
            self.last_fia_expected_batch_size = self.expected_fia_batch_size
            if not self._is_reusable_fia_shape(attention_metadata.actual_seq_lengths_q):
                self.shape_fallback_count += 1
                return self._execute(input_ids, positions, attention_metadata, output_transform)
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
            return self._execute(input_ids, positions, attention_metadata, output_transform)
        entry = self.entries.get(entry_key)
        if entry is None:
            if len(self.entries) >= self.max_graph_entries or self._capture_budget_used >= self.max_graph_entries:
                self.capacity_fallback_count += 1
                return self._execute(input_ids, positions, attention_metadata, output_transform)
            self.capture_attempt_count += 1
            self._capture_budget_used += 1
            output = self._capture(
                entry_key,
                padded_input_ids,
                padded_positions,
                padded_metadata,
                output_transform,
            )
            return output[:num_tokens]
        self._copy_inputs(entry, padded_input_ids, padded_positions, padded_metadata)
        if envs.VLLM_ASCEND_PEARL_SYNC_GRAPH_INPUTS:
            torch.npu.current_stream().synchronize()
        # Order task updates after the previous replay and the current input
        # copies. CANN releases a replay early when updates are issued on an
        # auxiliary stream, so an inline mode is available for runtimes where
        # ExternalEvent visibility is not sufficient for multi-token PA.
        inline_update = envs.VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE
        assert self.update_stream is not None
        if not inline_update:
            self.update_stream.wait_stream(torch.npu.current_stream())
            self._update_attention_tasks(entry)
        else:
            self._update_attention_tasks(entry, stream=torch.npu.current_stream())
        if not inline_update:
            # Attention task updates are issued on the auxiliary stream. The
            # reverse dependency prevents replay from consuming old metadata.
            torch.npu.current_stream().wait_stream(self.update_stream)
            if envs.VLLM_ASCEND_PEARL_SYNC_GRAPH_TASK_UPDATE:
                self.update_stream.synchronize()
        entry.graph.replay()
        self.replay_count += 1
        if envs.VLLM_ASCEND_PEARL_SYNC_GRAPH_REPLAY:
            torch.npu.synchronize()
        if not entry.runtime_validated:
            torch.npu.synchronize()
            graph_output = entry.output[:num_tokens].clone()
            reference_output = self._execute(
                input_ids,
                positions,
                attention_metadata,
                output_transform,
            )
            torch.npu.synchronize()
            if not self._outputs_match(graph_output, reference_output):
                failed_entry = self.entries.pop(entry_key)
                self._reset_graph_entry(failed_entry)
                self.disabled_entry_keys.add(entry_key)
                self.failed_capture_count += 1
                del entry
                warnings.warn(
                    "Native PEARL disabled an ACLGraph entry whose runtime replay did not match eager execution: "
                    f"{entry_key!r}; {self._mismatch_summary(graph_output, reference_output)}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return reference_output
            entry.runtime_validated = True
        if envs.VLLM_ASCEND_PEARL_VALIDATE_GRAPH_REPLAYS:
            torch.npu.synchronize()
            replay_output = entry.output[:num_tokens].clone()
            replay_reference = self._execute(
                input_ids,
                positions,
                attention_metadata,
                output_transform,
            )
            torch.npu.synchronize()
            replay_differs = (
                not torch.equal(replay_output, replay_reference)
                if not replay_output.dtype.is_floating_point
                else not torch.allclose(replay_output, replay_reference, rtol=1e-3, atol=1e-3)
            )
            if replay_differs:
                print(
                    "[PEARL graph replay mismatch] "
                    f"key={entry_key!r} {self._mismatch_summary(replay_output, replay_reference)}",
                    flush=True,
                )
        return entry.output[:num_tokens]

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
    ) -> torch.Tensor:
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
        captured_metadata = type(attention_metadata)(
            slot_mapping=captured_slot_mapping,
            context_lens=captured_context_lens,
            block_tables=captured_block_tables,
            actual_seq_lengths_q=attention_metadata.actual_seq_lengths_q,
            sequence_lens=attention_metadata.sequence_lens,
            request_block_tables=captured_request_block_tables,
            attention_mask=attention_metadata.attention_mask,
            use_fused_infer_attention=attention_metadata.use_fused_infer_attention,
        )
        graph = torch.npu.NPUGraph()
        with _collect_graph_tasks() as tasks, torch.npu.graph(graph):
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
        )
        self.entries[entry_key] = entry
        # Capture execution only records the graph; CANN does not guarantee
        # that its output buffers contain a valid inference result. Rebind the
        # host metadata and replay once before serving the triggering request.
        torch.npu.current_stream().synchronize()
        self._update_attention_tasks(entry)
        entry.graph.replay()
        torch.npu.synchronize()
        graph_matches_eager = self._outputs_match(output, reference_output)
        if not graph_matches_eager:
            failed_entry = self.entries.pop(entry_key)
            self._reset_graph_entry(failed_entry)
            self.disabled_entry_keys.add(entry_key)
            self.failed_capture_count += 1
            warnings.warn(
                "Native PEARL disabled an ACLGraph entry whose first replay did not match eager execution: "
                f"{entry_key!r}; {self._mismatch_summary(output, reference_output)}",
                RuntimeWarning,
                stacklevel=2,
            )
            return reference_output
        self.capture_count += 1
        self.replay_count += 1
        return output

    @staticmethod
    def _outputs_match(graph_output: torch.Tensor, reference_output: torch.Tensor) -> bool:
        if graph_output.dtype.is_floating_point:
            return torch.allclose(graph_output, reference_output, rtol=1e-3, atol=1e-3)
        mismatch_count = torch.count_nonzero(graph_output != reference_output).item()
        allowed_mismatches = max(
            1,
            math.ceil(reference_output.numel() * MAX_GREEDY_REPLAY_EAGER_DIVERGENCE),
        )
        return mismatch_count <= allowed_mismatches

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
    def _copy_inputs(entry: NativeACLGraphEntry, input_ids, positions, attention_metadata) -> None:
        entry.input_ids.copy_(input_ids)
        entry.positions.copy_(positions)
        entry.slot_mapping.copy_(attention_metadata.slot_mapping)
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

    def _update_draft_attention_tasks(self, entry: NativeDraftACLGraphEntry) -> None:
        task_lengths = [
            lengths
            for lengths in zip(entry.actual_seq_lengths_q, entry.sequence_lens)
            for _ in range(entry.tasks_per_step)
        ]
        self._update_attention_task_list(entry.tasks, task_lengths)

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
            # Do not alter the pre-existing linear replay convention.
            args.update(sparse_mode=3, next_tokens=0)
        torch.npu.graph_task_update_begin(stream, task.handle)
        torch_npu.npu_fused_infer_attention_score.out(
            **args,
            workspace=task.workspace,
            out=[task.output, task.softmax_lse],
        )
        torch.npu.graph_task_update_end(stream)
        task.event.record(stream)
