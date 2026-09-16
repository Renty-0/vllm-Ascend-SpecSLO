# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Ascend port of nano-PEARL's persistent two-model decode loop.

Run this module under ``torchrun``.  Unlike the OpenAI-compatible bridge,
all ranks form one HCCL world and retain their own model KV cache across every
PEARL round.  The round ordering matches nano-PEARL:

``pre-verify -> gamma draft tokens -> packed target verification -> rollback``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer

from vllm_ascend import envs
from vllm_ascend.cpu_binding import bind_cpus
from vllm_ascend.ops.triton.spec_decode.fixed_greedy_verdict import (
    FIXED_GREEDY_FULL_WINDOW_WIDTH,
    fixed_greedy_full_window_bonus_verdict,
    fixed_greedy_full_window_verdict,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.spec_decode.pearl.accounting_transport import (
    ReusableSpecRhythmEnvelope,
)
from vllm_ascend.spec_decode.pearl.admission import decide_prefill_coalesce
from vllm_ascend.spec_decode.pearl.fixed_greedy_transport import (
    CompactFixedGreedyEnvelopeLayout,
    PendingCompactFixedGreedyEnvelope,
    begin_compact_fixed_greedy_broadcast,
)
from vllm_ascend.spec_decode.pearl.linear_fia import (
    LinearDraftFIAFullMaskBuilder,
    LinearDraftFIAPersistentStaging,
    sanitize_and_pad_linear_draft_fia_request_tables,
    validate_linear_draft_fia_final_pages,
)
from vllm_ascend.spec_decode.pearl.mc2 import MC2Profile, normalize_mc2_profile
from vllm_ascend.spec_decode.pearl.native_cache import NativeCacheAllocation, NativePrefixCache
from vllm_ascend.spec_decode.pearl.native_graph import NativeACLGraphRunner, NativeGraphExecution
from vllm_ascend.spec_decode.pearl.native_model import (
    PAGED_ATTENTION_BLOCK_SIZE,
    NativeAttentionMetadata,
    NativeTPContext,
    build_native_model,
    load_native_model_weights,
    make_tree_fia_mask,
)
from vllm_ascend.spec_decode.pearl.pard import (
    NATIVE_DRAFT_MODES,
    PARD_PARALLEL_DRAFT_MODE,
    SERIAL_LINEAR_DRAFT_MODE,
    build_pard_parallel_draft_layout,
    require_experimental_pard_eager,
    validate_native_draft_mode,
    validate_pard_parallel_model_pair,
)
from vllm_ascend.spec_decode.pearl.qwen_pair import validate_model_pair
from vllm_ascend.spec_decode.pearl.roofline import (
    ProfiledRoofline,
    attention_backend_identity,
    normalize_roofline,
    parse_roofline_argument,
    validate_roofline_hardware,
)
from vllm_ascend.spec_decode.pearl.spec_rhythm import (
    SpecRhythmBudgetShaper,
    SpecRhythmPipelineController,
    SpecRhythmProposalTicket,
    SpecRhythmRuntimeState,
)
from vllm_ascend.spec_decode.pearl.topology import PearlProcessGroups, PearlTopology
from vllm_ascend.spec_decode.pearl.tree import (
    TreeSpeculationPlan,
    TreeVerificationOutput,
    cached_cpu_tree_speculation_plan,
    pack_selected_tree_plan,
    tree_primary_path,
    verify_greedy_tree_batch,
)
from vllm_ascend.spec_decode.pearl.tree_budget import (
    DraftWindowBudget,
    DraftWindowEstimator,
    TreeCandidateRequest,
    select_global_tree_candidates,
)
from vllm_ascend.spec_decode.tree_kv import (
    TreeKVCompactionGraphRunner,
    move_kv_cache_slots,
)

AUTO_GAMMA_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
AUTO_GAMMA_WARMUP_STEPS = 5
AUTO_GAMMA_PROFILE_STEPS = 30
AUTO_GAMMA_PROFILE_SEQUENCE_LENGTH = 256
TARGET_VERIFICATION_GRAPH_BUCKETS = 8
SPEC_RHYTHM_ABLATION_MODES = frozenset({"auto", "serial", "nano_pearl", "dual_batch", "dual_batch_rolling"})
PREEMPTIVE_SCHEDULING_EXPLORATION_ROUNDS = 8
PREEMPTIVE_SCHEDULING_PRIOR_ROUNDS = 8
PREEMPTIVE_SCHEDULING_RECENT_ROUNDS = 32

logger = logging.getLogger("vllm_ascend.spec_decode.pearl.native")


MIXED_TARGET_VERIFY_CAPACITIES = (16, 24, 32)
MAX_MIXED_TARGET_VERIFY_CAPACITY = MIXED_TARGET_VERIFY_CAPACITIES[-1]
MIXED_TARGET_PROMPT_CAPACITY = 4
MIXED_TARGET_SCRATCH_ROWS = 5
MIXED_TARGET_PROMPT_TOKEN_BUCKETS = (
    64,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
)
STABLE_TARGET_VERIFY_CAPACITIES = tuple(range(1, 33))
FIXED_FULL_WINDOW_HOST_STAGING_GAMMA = FIXED_GREEDY_FULL_WINDOW_WIDTH
FIXED_FULL_WINDOW_HOST_STAGING_MAX_BATCH = 32
FIXED_FULL_WINDOW_HOST_STAGING_VERDICT_VALUES = 2 * FIXED_FULL_WINDOW_HOST_STAGING_MAX_BATCH
FIXED_FULL_WINDOW_HOST_STAGING_CONTINUATION_VALUES = (
    FIXED_FULL_WINDOW_HOST_STAGING_GAMMA * FIXED_FULL_WINDOW_HOST_STAGING_MAX_BATCH
)
FIXED_FULL_WINDOW_HOST_STAGING_VALUES = (
    FIXED_FULL_WINDOW_HOST_STAGING_VERDICT_VALUES + FIXED_FULL_WINDOW_HOST_STAGING_CONTINUATION_VALUES
)


@dataclass
class _FixedFullWindowHostCorrectionStaging:
    """Persistent device/host storage for one fixed-gamma correction round."""

    verdict_receive: torch.Tensor
    host_values: torch.Tensor
    copy_done_event: Any


def _next_mixed_target_verify_capacity(request_count: int) -> int:
    """Return the smallest resident mixed-target capacity that fits."""

    request_count = int(request_count)
    capacity = next(
        (candidate for candidate in MIXED_TARGET_VERIFY_CAPACITIES if candidate >= request_count),
        None,
    )
    if capacity is None or request_count <= 0:
        raise ValueError("Mixed-target graph verification rows/request count must be in [1, 32].")
    return capacity


def _next_stable_target_verify_capacity(request_count: int) -> int:
    """Return the smallest provisioned verification capacity that fits."""

    request_count = int(request_count)
    capacity = next(
        (candidate for candidate in STABLE_TARGET_VERIFY_CAPACITIES if candidate >= request_count),
        None,
    )
    if capacity is None or request_count <= 0:
        raise ValueError("Stable target-verify request count must be in [1, 32].")
    return capacity


def resolve_mixed_target_graph_buckets(value: str) -> tuple[int, ...]:
    """Validate an optional resident mixed-prefill graph bucket subset."""

    value = str(value).strip()
    if not value:
        return MIXED_TARGET_PROMPT_TOKEN_BUCKETS
    try:
        requested = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise ValueError(
            "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_BUCKETS must be a comma-separated integer list."
        ) from error
    if (
        not requested
        or any(bucket not in MIXED_TARGET_PROMPT_TOKEN_BUCKETS for bucket in requested)
        or len(set(requested)) != len(requested)
    ):
        raise ValueError(
            "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_BUCKETS must contain "
            "unique supported buckets from "
            f"{MIXED_TARGET_PROMPT_TOKEN_BUCKETS}."
        )
    return tuple(sorted(requested))


@dataclass(frozen=True)
class MixedTargetGraphLayout:
    """Device-free shape plan for the bounded mixed-target ACLGraph.

    The bounded graph covers at most four newly arrived prompts and a selected
    aggregate prompt-token bucket up to 2048. Verification occupies the
    smallest qualified 16-, 24-, or 32-row ``gamma`` capacity. Four
    prompt slots plus one mandatory padding segment keep both the number of
    FIA request partitions and the total query-token shape invariant within a
    prompt-token bucket plus five scratch queries.  Real prompt segments are
    never length-padded, so the
    plan does not widen a real request's KV frontier.

    This object contains no tensors and performs no cache mutation.  The NPU
    wiring must reserve disjoint scratch KV slots for every segment marked
    dummy before it may use the layout.
    """

    gamma: int
    verification_rows: int
    verification_capacity: int
    prompt_lengths: tuple[int, ...]
    prompt_token_bucket: int
    query_lengths: tuple[int, ...]
    verification_output_count: int
    prompt_output_indices: tuple[int, ...]

    @property
    def prompt_capacity(self) -> int:
        return MIXED_TARGET_PROMPT_CAPACITY

    @property
    def request_segment_count(self) -> int:
        return len(self.query_lengths)

    @property
    def total_query_tokens(self) -> int:
        return sum(self.query_lengths)

    @property
    def dummy_verification_rows(self) -> int:
        return self.verification_capacity - self.verification_rows

    @property
    def dummy_prompt_segments(self) -> int:
        # Unused real-prompt slots plus the mandatory residual segment.
        return self.prompt_capacity - len(self.prompt_lengths) + 1

    @property
    def graph_key(self) -> str:
        return (
            "mixed-target-hidden"
            f"|gamma:{self.gamma}"
            f"|verify:{self.verification_capacity}"
            f"|prompt:{self.prompt_capacity}+pad1"
            f"|prompt-tokens:{self.prompt_token_bucket}"
        )


@dataclass(frozen=True)
class MixedTargetGraphEnvelope:
    """Host-only routing for one fixed mixed-target FIA invocation."""

    layout: MixedTargetGraphLayout
    token_sequence_ids: tuple[int, ...]
    positions: tuple[int, ...]
    segment_sequence_ids: tuple[int, ...]
    sequence_lens: tuple[int, ...]


@dataclass(frozen=True)
class StableTargetVerifyGraphLayout:
    """Fixed-Q FIA layout for an ordinary full-window verification cycle."""

    query_width: int
    verification_rows: int
    verification_capacity: int
    query_lengths: tuple[int, ...]
    verification_output_count: int

    @property
    def request_segment_count(self) -> int:
        return len(self.query_lengths)

    @property
    def total_query_tokens(self) -> int:
        return sum(self.query_lengths)

    @property
    def dummy_verification_rows(self) -> int:
        return self.verification_capacity - self.verification_rows

    @property
    def graph_key(self) -> str:
        return f"stable-target-verify-hidden|width:{self.query_width}|verify:{self.verification_capacity}"


@dataclass(frozen=True)
class StableTargetVerifyGraphEnvelope:
    """Host-only routing for one fixed decode-only verification graph."""

    layout: StableTargetVerifyGraphLayout
    token_sequence_ids: tuple[int, ...]
    positions: tuple[int, ...]
    segment_sequence_ids: tuple[int, ...]
    sequence_lens: tuple[int, ...]


def plan_stable_target_verify_graph_layout(
    verification_rows: int,
    *,
    query_width: int,
    verification_capacity: int = MAX_MIXED_TARGET_VERIFY_CAPACITY,
) -> StableTargetVerifyGraphLayout:
    """Plan one Q=capacity*width graph independent of active batch size."""

    verification_rows = int(verification_rows)
    query_width = int(query_width)
    verification_capacity = int(verification_capacity)
    if query_width <= 0:
        raise ValueError("Stable target-verify query width must be positive.")
    if verification_capacity <= 0 or not (0 < verification_rows <= verification_capacity):
        raise ValueError("Stable target-verify rows must fit its positive capacity.")
    query_lengths = (query_width,) * verification_capacity
    return StableTargetVerifyGraphLayout(
        query_width=query_width,
        verification_rows=verification_rows,
        verification_capacity=verification_capacity,
        query_lengths=query_lengths,
        verification_output_count=verification_rows * query_width,
    )


@lru_cache(maxsize=128)
def _cached_stable_target_verify_graph_layout(
    verification_rows: int,
    query_width: int,
    verification_capacity: int,
) -> StableTargetVerifyGraphLayout:
    """Reuse the immutable exact verification layout after qualification."""

    return plan_stable_target_verify_graph_layout(
        verification_rows,
        query_width=query_width,
        verification_capacity=verification_capacity,
    )


@lru_cache(maxsize=128)
def _stable_target_verify_cumulative_q(
    query_width: int,
    verification_capacity: int,
) -> tuple[int, ...]:
    """Return the fixed FIA query partition for one exact graph shape."""

    return tuple(
        range(
            query_width,
            query_width * verification_capacity + 1,
            query_width,
        )
    )


def plan_stable_target_verify_graph_envelope(
    layout: StableTargetVerifyGraphLayout,
    verification_sequence_ids: Sequence[int],
    verification_start_positions: Sequence[int],
    scratch_sequence_id: int,
) -> StableTargetVerifyGraphEnvelope:
    """Route inactive verification rows to disjoint slots on one scratch row."""

    verify_ids = tuple(int(value) for value in verification_sequence_ids)
    verify_starts = tuple(int(value) for value in verification_start_positions)
    scratch_id = int(scratch_sequence_id)
    if len(verify_ids) != layout.verification_rows or len(verify_starts) != layout.verification_rows:
        raise ValueError("Stable target-verify envelope inputs must match its real rows.")
    if (
        len(set(verify_ids)) != len(verify_ids)
        or scratch_id in verify_ids
        or scratch_id < 0
        or any(value < 0 for value in (*verify_ids, *verify_starts))
    ):
        raise ValueError(
            "Stable target-verify real/scratch rows and positions must be unique, disjoint, and non-negative."
        )

    token_sequence_ids: list[int] = []
    positions: list[int] = []
    segment_sequence_ids: list[int] = []
    sequence_lens: list[int] = []

    def append_segment(sequence_id: int, start: int) -> None:
        token_sequence_ids.extend([sequence_id] * layout.query_width)
        positions.extend(range(start, start + layout.query_width))
        segment_sequence_ids.append(sequence_id)
        sequence_lens.append(start + layout.query_width)

    for sequence_id, start in zip(verify_ids, verify_starts):
        append_segment(sequence_id, start)
    scratch_position = 0
    for _ in range(layout.dummy_verification_rows):
        append_segment(scratch_id, scratch_position)
        scratch_position += layout.query_width

    envelope = StableTargetVerifyGraphEnvelope(
        layout=layout,
        token_sequence_ids=tuple(token_sequence_ids),
        positions=tuple(positions),
        segment_sequence_ids=tuple(segment_sequence_ids),
        sequence_lens=tuple(sequence_lens),
    )
    if (
        len(envelope.token_sequence_ids) != layout.total_query_tokens
        or len(envelope.positions) != layout.total_query_tokens
        or len(envelope.segment_sequence_ids) != layout.request_segment_count
        or len(envelope.sequence_lens) != layout.request_segment_count
    ):
        raise RuntimeError("Stable target-verify envelope does not match its layout.")
    return envelope


def _record_mixed_target_graph_outcome(
    counters: dict[str, int | float],
    outcome: tuple[str, int, int],
) -> None:
    """Publish mixed-graph use/fallback without adding a device fence."""

    kind, requests, prompt_tokens = outcome
    if requests < 0 or prompt_tokens < 0:
        raise ValueError("Mixed-target graph counters cannot be negative.")
    if kind == "disabled":
        return
    if kind == "graph":
        counters["spec_rhythm_mixed_target_graph_batches"] += 1
        counters["spec_rhythm_mixed_target_graph_requests"] += requests
        counters["spec_rhythm_mixed_target_graph_prompt_tokens"] += prompt_tokens
        return
    fallback_counters = {
        "bounded_shape": ("spec_rhythm_mixed_target_graph_bounded_shape_fallback_batches"),
        "token_capacity": ("spec_rhythm_mixed_target_graph_token_capacity_fallback_batches"),
        "execution_fallback": ("spec_rhythm_mixed_target_graph_execution_fallback_batches"),
    }
    counter = fallback_counters.get(kind)
    if counter is None:
        raise ValueError(f"Unknown mixed-target graph outcome: {kind}.")
    counters[counter] += 1


def _record_stable_target_verify_graph_outcome(
    counters: dict[str, int | float],
    outcome: tuple[str, int],
) -> None:
    """Publish ordinary fixed-window stable-graph use independently.

    These counters deliberately do not share the mixed-prefill namespace:
    decode-only verification reuses a mixed-target resident graph entry, but
    it is not itself a mixed verification/prefill service batch.
    """

    kind, requests = outcome
    if requests < 0:
        raise ValueError("Stable target-verify graph counters cannot be negative.")
    if kind == "disabled":
        return
    if kind == "graph":
        counters["spec_rhythm_stable_target_verify_graph_batches"] += 1
        counters["spec_rhythm_stable_target_verify_graph_requests"] += requests
        return
    fallback_counters = {
        "bounded_shape": ("spec_rhythm_stable_target_verify_graph_bounded_shape_fallback_batches"),
        "token_capacity": ("spec_rhythm_stable_target_verify_graph_token_capacity_fallback_batches"),
        "execution_fallback": ("spec_rhythm_stable_target_verify_graph_execution_fallback_batches"),
    }
    counter = fallback_counters.get(kind)
    if counter is None:
        raise ValueError(f"Unknown stable target-verify graph outcome: {kind}.")
    counters[counter] += 1


def plan_mixed_target_graph_envelope(
    layout: MixedTargetGraphLayout,
    verification_sequence_ids: Sequence[int],
    verification_start_positions: Sequence[int],
    prompt_sequence_ids: Sequence[int],
    prompt_start_positions: Sequence[int],
    scratch_sequence_ids: Sequence[int],
) -> MixedTargetGraphEnvelope:
    """Assign every fixed segment to disjoint real or scratch KV slots.

    Five reserved target-cache rows are sufficient: row zero owns all dummy
    verification segments in consecutive, non-overlapping ranges; rows one
    through four own the four prompt slots.  The mandatory residual segment
    shares only the final scratch row and begins after its optional one-token
    dummy prompt slot.  Repeated request tables are intentional: FIA uses the
    explicit cumulative query partition to keep those segments independent.
    """

    verify_ids = tuple(int(value) for value in verification_sequence_ids)
    verify_starts = tuple(int(value) for value in verification_start_positions)
    prompt_ids = tuple(int(value) for value in prompt_sequence_ids)
    prompt_starts = tuple(int(value) for value in prompt_start_positions)
    scratch_ids = tuple(int(value) for value in scratch_sequence_ids)
    if (
        len(verify_ids) != layout.verification_rows
        or len(verify_starts) != layout.verification_rows
        or len(prompt_ids) != len(layout.prompt_lengths)
        or len(prompt_starts) != len(layout.prompt_lengths)
    ):
        raise ValueError("Mixed-target graph envelope inputs must match the planned real rows.")
    if len(scratch_ids) != MIXED_TARGET_SCRATCH_ROWS or len(set(scratch_ids)) != len(scratch_ids):
        raise ValueError("Mixed-target graph envelope requires five unique scratch rows.")
    real_ids = (*verify_ids, *prompt_ids)
    if (
        len(set(real_ids)) != len(real_ids)
        or set(real_ids).intersection(scratch_ids)
        or any(value < 0 for value in (*real_ids, *scratch_ids))
        or any(value < 0 for value in (*verify_starts, *prompt_starts))
    ):
        raise ValueError(
            "Mixed-target graph real/scratch rows and positions must be unique, disjoint, and non-negative."
        )

    token_sequence_ids: list[int] = []
    positions: list[int] = []
    segment_sequence_ids: list[int] = []
    sequence_lens: list[int] = []

    def append_segment(sequence_id: int, start: int, length: int) -> None:
        if length <= 0:
            raise RuntimeError("Mixed-target graph segment length must be positive.")
        token_sequence_ids.extend([sequence_id] * length)
        positions.extend(range(start, start + length))
        segment_sequence_ids.append(sequence_id)
        sequence_lens.append(start + length)

    for sequence_id, start in zip(verify_ids, verify_starts):
        append_segment(sequence_id, start, layout.gamma)
    scratch_verify_position = 0
    for _ in range(layout.dummy_verification_rows):
        append_segment(scratch_ids[0], scratch_verify_position, layout.gamma)
        scratch_verify_position += layout.gamma

    for sequence_id, start, length in zip(
        prompt_ids,
        prompt_starts,
        layout.prompt_lengths,
    ):
        append_segment(sequence_id, start, length)
    unused_prompt_slots = layout.prompt_capacity - len(layout.prompt_lengths)
    for slot in range(len(layout.prompt_lengths), layout.prompt_capacity):
        append_segment(scratch_ids[slot + 1], 0, 1)

    residual_length = layout.query_lengths[-1]
    residual_start = int(unused_prompt_slots > 0)
    append_segment(scratch_ids[-1], residual_start, residual_length)
    envelope = MixedTargetGraphEnvelope(
        layout=layout,
        token_sequence_ids=tuple(token_sequence_ids),
        positions=tuple(positions),
        segment_sequence_ids=tuple(segment_sequence_ids),
        sequence_lens=tuple(sequence_lens),
    )
    if (
        len(envelope.token_sequence_ids) != layout.total_query_tokens
        or len(envelope.positions) != layout.total_query_tokens
        or len(envelope.segment_sequence_ids) != layout.request_segment_count
        or len(envelope.sequence_lens) != layout.request_segment_count
    ):
        raise RuntimeError("Mixed-target graph envelope does not match its layout.")
    return envelope


def plan_mixed_target_graph_layout(
    verification_rows: int,
    prompt_lengths: Sequence[int],
    *,
    gamma: int,
    verification_capacity: int | None = None,
    prompt_capacity: int = MIXED_TARGET_PROMPT_CAPACITY,
    prompt_token_buckets: Sequence[int] = MIXED_TARGET_PROMPT_TOKEN_BUCKETS,
) -> MixedTargetGraphLayout:
    """Return one fixed-Q/fixed-partition mixed-target graph layout.

    Dummy verification rows retain ``gamma`` positive query tokens.  Real
    prompt rows retain their exact lengths; unused prompt slots consume one
    scratch query each, and a final scratch-only segment absorbs the remainder
    of the selected aggregate prompt bucket.  This makes the cumulative FIA
    partition values dynamic while their count, final value, and tensor shapes
    remain stable for a dedicated graph key.
    """

    verification_rows = int(verification_rows)
    gamma = int(gamma)
    verification_capacity = (
        _next_mixed_target_verify_capacity(verification_rows)
        if verification_capacity is None
        else int(verification_capacity)
    )
    prompt_capacity = int(prompt_capacity)
    lengths = tuple(int(length) for length in prompt_lengths)
    buckets = tuple(int(bucket) for bucket in prompt_token_buckets)
    if gamma <= 0:
        raise ValueError("Mixed-target graph gamma must be positive.")
    if verification_capacity not in MIXED_TARGET_VERIFY_CAPACITIES or not (
        0 < verification_rows <= verification_capacity
    ):
        raise ValueError(
            f"Mixed-target graph verification rows must fit a supported capacity from {MIXED_TARGET_VERIFY_CAPACITIES}."
        )
    if prompt_capacity <= 0 or not 0 < len(lengths) <= prompt_capacity:
        raise ValueError("Mixed-target graph prompt rows must fit its positive capacity.")
    if any(length <= 0 for length in lengths):
        raise ValueError("Mixed-target graph prompt lengths must be positive.")
    if (
        not buckets
        or any(bucket <= 0 for bucket in buckets)
        or any(left >= right for left, right in zip(buckets, buckets[1:]))
    ):
        raise ValueError("Mixed-target graph prompt-token buckets must be positive and strictly increasing.")

    unused_prompt_slots = prompt_capacity - len(lengths)
    real_prompt_tokens = sum(lengths)
    prompt_bucket = next(
        (bucket for bucket in buckets if bucket >= real_prompt_tokens),
        None,
    )
    if prompt_bucket is None:
        raise ValueError(
            "Mixed-target graph prompt tokens exceed its largest stable bucket: "
            f"required={real_prompt_tokens}, maximum={buckets[-1]}."
        )
    # Five scratch queries (one per prompt capacity plus the mandatory
    # residual segment) keep the fixed total independent of the number of real
    # prompt rows, including an exact 512-token real payload.
    prompt_envelope_tokens = prompt_bucket + prompt_capacity + 1
    residual_queries = prompt_envelope_tokens - real_prompt_tokens - unused_prompt_slots
    if residual_queries <= 0:
        raise RuntimeError("Mixed-target graph failed to reserve its mandatory scratch segment.")

    verification_query_lengths = (gamma,) * verification_capacity
    prompt_query_lengths = (
        *lengths,
        *((1,) * unused_prompt_slots),
        residual_queries,
    )
    query_lengths = (*verification_query_lengths, *prompt_query_lengths)
    prompt_base = verification_capacity * gamma
    prompt_output_indices: list[int] = []
    cursor = prompt_base
    for length in lengths:
        cursor += length
        prompt_output_indices.append(cursor - 1)
    layout = MixedTargetGraphLayout(
        gamma=gamma,
        verification_rows=verification_rows,
        verification_capacity=verification_capacity,
        prompt_lengths=lengths,
        prompt_token_bucket=prompt_bucket,
        query_lengths=query_lengths,
        verification_output_count=verification_rows * gamma,
        prompt_output_indices=tuple(prompt_output_indices),
    )
    expected_segments = verification_capacity + prompt_capacity + 1
    expected_tokens = verification_capacity * gamma + prompt_bucket + prompt_capacity + 1
    if (
        layout.request_segment_count != expected_segments
        or layout.total_query_tokens != expected_tokens
        or any(length <= 0 for length in layout.query_lengths)
    ):
        raise RuntimeError("Mixed-target graph planner produced an unstable shape.")
    return layout


@dataclass(frozen=True)
class SpecRhythmPrefillTokenChunk:
    """One request-local span in an aggregate token-capped prefill pass."""

    request_index: int
    start: int
    end: int
    prompt_length: int

    @property
    def token_count(self) -> int:
        return self.end - self.start

    @property
    def completes_prompt(self) -> bool:
        return self.end == self.prompt_length


def plan_spec_rhythm_prefill_token_chunk(
    request_indices: Sequence[int],
    *,
    prompt_lengths: Mapping[int, int],
    cursors: Mapping[int, int],
    token_cap: int,
) -> tuple[SpecRhythmPrefillTokenChunk, ...]:
    """Fairly pack unfinished prompt spans under one fixed aggregate cap.

    The planner is device-free and does not mutate ``cursors``.  A rotating
    start derived from already consumed tokens prevents starvation when the
    cap is smaller than the number of staged requests.
    """

    if token_cap <= 0:
        raise ValueError("SpecRhythm prefill token cap must be positive.")
    ordered = [int(index) for index in request_indices]
    if len(set(ordered)) != len(ordered):
        raise ValueError("SpecRhythm prefill requests must be unique.")
    unfinished: list[int] = []
    for index in ordered:
        if index not in prompt_lengths or index not in cursors:
            raise ValueError("SpecRhythm prefill planner requires a length and cursor for every request.")
        prompt_length = int(prompt_lengths[index])
        cursor = int(cursors[index])
        if prompt_length <= 0:
            raise ValueError("SpecRhythm prefill prompts must be non-empty.")
        if not 0 <= cursor <= prompt_length:
            raise ValueError("SpecRhythm prefill cursor must be within its prompt.")
        if cursor < prompt_length:
            unfinished.append(index)
    if not unfinished:
        return ()

    consumed = sum(int(cursors[index]) for index in unfinished)
    offset = consumed % len(unfinished)
    rotating = unfinished[offset:] + unfinished[:offset]
    allocation = {index: 0 for index in rotating}
    remaining_budget = int(token_cap)
    eligible = list(rotating)
    while remaining_budget > 0 and eligible:
        next_eligible: list[int] = []
        for position, index in enumerate(eligible):
            rows_left = len(eligible) - position
            fair_share = max(1, remaining_budget // rows_left)
            available = int(prompt_lengths[index]) - int(cursors[index]) - allocation[index]
            take = min(available, fair_share, remaining_budget)
            allocation[index] += take
            remaining_budget -= take
            if take < available:
                next_eligible.append(index)
            if remaining_budget == 0:
                next_eligible.extend(eligible[position + 1 :])
                break
        eligible = next_eligible

    chunks: list[SpecRhythmPrefillTokenChunk] = []
    for index in rotating:
        count = allocation[index]
        if count <= 0:
            continue
        start = int(cursors[index])
        prompt_length = int(prompt_lengths[index])
        chunks.append(
            SpecRhythmPrefillTokenChunk(
                request_index=index,
                start=start,
                end=start + count,
                prompt_length=prompt_length,
            )
        )
    if sum(chunk.token_count for chunk in chunks) > token_cap:
        raise RuntimeError("SpecRhythm prefill planner exceeded its token cap.")
    return tuple(chunks)


def _full_window_eager_has_useful_horizon(
    state: PearlPipelineState,
    parent_verification_size: int,
) -> bool:
    """Whether an eager child can contribute before the request must finish.

    Full-window verification commits the complete parent when it is accepted.
    If that already reaches ``max_tokens``, a concurrently generated child can
    never be consumed: rejection invalidates it and acceptance finishes the
    request.  Avoiding that child also keeps the draft KV frontier within the
    single-window tail capacity reserved by :meth:`generate_batch`.
    """

    if parent_verification_size <= 0:
        raise ValueError("A full-window parent must contain at least one token.")
    return len(state.committed_completion_token_ids) + int(parent_verification_size) < state.max_tokens


def _trace_region(owner: object, name: str):
    """Emit nested host ranges only while the heavyweight NPU trace is active."""
    if getattr(owner, "_active_tree_profiler", None) is None:
        return nullcontext()
    return torch.profiler.record_function(name)


def _gate_linear_eager_candidates(
    eligible_indices: Sequence[int],
    *,
    states: Mapping[int, SpecRhythmRuntimeState],
    projected_wait_ms: float,
    normal_request_count: int,
    gamma: int,
    estimator: DraftWindowEstimator,
    max_rows: int | None = None,
    allow_cross_graph_bucket: bool = False,
    residual_indices: frozenset[int] = frozenset(),
) -> tuple[list[int], DraftWindowBudget]:
    """Admit only complete fixed-gamma eager rows that fit measured W.

    This is deliberately a row gate rather than a token-budget override.  The
    scalar shaper remains responsible for assigning exactly ``gamma`` tokens
    to every selected normal/eager request; W only decides how many optional
    rolling-eager rows may accompany mandatory normal work.
    """

    if gamma <= 0 or normal_request_count < 0:
        raise ValueError("Linear SpecRhythm W gating requires positive gamma and row counts.")
    unknown_residual = residual_indices.difference(eligible_indices)
    if unknown_residual:
        raise ValueError("Residual eager rows must be a subset of eligible rows.")

    def ordering_key(index: int) -> tuple[float, float, float, int]:
        if index in residual_indices:
            # Residual work follows every urgent a_need row. Within that tier,
            # rank by the probability that the complete dependency window can
            # be promoted rather than by optimistic per-token acceptance.
            return (
                0.0,
                states[index].expected_continuation_benefit,
                states[index].urgency(projected_wait_ms),
                -index,
            )
        # Section 5.1 ranks urgent rolling-eager work by a_need * expected
        # benefit. Apply that order before the W cutoff so input/controller
        # order cannot displace a more valuable request.
        return (
            1.0,
            states[index].projected_progress_gap(projected_wait_ms) * states[index].expected_acceptance_benefit,
            states[index].urgency(projected_wait_ms),
            -index,
        )

    ordered = sorted(eligible_indices, key=ordering_key, reverse=True)
    normal_tokens = normal_request_count * gamma
    window = estimator.estimate(
        normal_tokens=normal_tokens,
        max_draft_tokens=normal_tokens + len(ordered) * gamma,
        # Linear serial draft has no tree-only parent-frontier/selected-KV
        # fixed overhead.  Charging that proxy would incorrectly suppress W.
        eager_work=False,
    )
    eager_row_cap = min(len(ordered), window.eager_token_budget // gamma)
    if max_rows is not None:
        if normal_request_count > max_rows:
            raise ValueError("Linear SpecRhythm normal rows exceed service capacity.")
        eager_row_cap = min(eager_row_cap, max_rows - normal_request_count)
        if normal_request_count and not allow_cross_graph_bucket:
            normal_bucket = _next_linear_draft_graph_bucket(
                normal_request_count,
                max_rows,
            )
            # Optional rolling-eager work must fit both the measured overlap
            # window and the already selected graph bucket.  Although a padded
            # row has no additional graph-shape cost, it still enlarges the
            # proposal mailbox and may create a later invalidation/correction
            # rendezvous.  Formal p60/t256 regression showed that filling such
            # rows beyond W reduced E2E throughput, so do not treat padding as
            # globally free work.
            eager_row_cap = min(
                eager_row_cap,
                normal_bucket - normal_request_count,
            )
        # With no normal graph there is no paid shape to fill. Likewise, the
        # explicit cross-bucket experiment trusts measured W instead of
        # treating the current graph bucket as a second latency model. The
        # physical service-capacity bound above still applies in both cases.
    return ordered[:eager_row_cap], window


def _fixed_gamma_identity_budgets(
    normal_request_indices: Sequence[int],
    eager_request_indices: Sequence[int],
    *,
    gamma: int,
    verification_roof: int,
) -> tuple[dict[int, int], dict[int, int]]:
    """Return the identity allocation for a fixed-gamma linear envelope.

    The caller has already selected normal work through the two-home
    controller and optional Rolling Eager work through the SLO/W gate.  When
    there is no external roofline or draft-token budget, the default global
    envelope is ``active_rows * gamma``.  Re-running the general two-stage
    shaper cannot change that selection: every admitted row has identical
    minimum and maximum budget ``gamma``.  Keep the invariant checks here so
    this hot-path shortcut cannot silently overfill the target envelope.
    """

    if gamma <= 0 or verification_roof <= 0:
        raise ValueError("A fixed-gamma identity budget requires positive limits.")
    normal = tuple(int(index) for index in normal_request_indices)
    eager = tuple(int(index) for index in eager_request_indices)
    if len(set(normal)) != len(normal) or len(set(eager)) != len(eager):
        raise ValueError("Fixed-gamma work lists must not contain duplicate requests.")
    if set(normal).intersection(eager):
        raise ValueError("A request cannot receive both normal and eager work.")
    if (len(normal) + len(eager)) * gamma > verification_roof:
        raise ValueError("Fixed-gamma work exceeds the target verification envelope.")
    return (
        {index: gamma for index in normal},
        {index: gamma for index in eager},
    )


def _record_linear_draft_full_chain_metrics(
    owner: object,
    *,
    logical_rows: int,
    padded_rows: int,
    gamma: int,
    execution: NativeGraphExecution | None,
) -> None:
    """Record logical work and the actual full-chain runner outcome."""

    if logical_rows <= 0 or padded_rows < logical_rows or gamma <= 0:
        raise ValueError("Linear draft full-chain telemetry received an invalid shape.")

    def increment(name: str, value: int = 1) -> None:
        setattr(owner, name, int(getattr(owner, name, 0)) + int(value))

    bucket_calls = getattr(owner, "_linear_draft_full_chain_bucket_calls", None)
    if bucket_calls is None:
        bucket_calls = {}
        owner._linear_draft_full_chain_bucket_calls = bucket_calls
    bucket_calls[padded_rows] = bucket_calls.get(padded_rows, 0) + 1
    logical_calls = getattr(owner, "_linear_draft_full_chain_logical_row_calls", None)
    if logical_calls is None:
        logical_calls = {}
        owner._linear_draft_full_chain_logical_row_calls = logical_calls
    logical_calls[logical_rows] = logical_calls.get(logical_rows, 0) + 1

    increment("_linear_draft_full_chain_calls")
    increment("_linear_draft_full_chain_logical_rows", logical_rows)
    increment("_linear_draft_full_chain_logical_tokens", logical_rows * gamma)
    increment("_linear_draft_full_chain_padded_rows", padded_rows)
    increment("_linear_draft_full_chain_padded_tokens", padded_rows * gamma)
    increment("_linear_draft_full_chain_padding_rows", padded_rows - logical_rows)
    increment("_linear_draft_full_chain_padding_tokens", (padded_rows - logical_rows) * gamma)

    if execution is None:
        increment("_linear_draft_full_chain_unclassified_calls")
    elif execution.mode == "capture_replay":
        increment("_linear_draft_full_chain_capture_replay_calls")
    elif execution.mode == "replay":
        increment("_linear_draft_full_chain_replay_calls")
    elif execution.mode == "eager":
        increment("_linear_draft_full_chain_eager_fallback_calls")
    else:
        increment("_linear_draft_full_chain_unclassified_calls")


def _set_default_npu_environment(target_tp_size: int | None = None) -> None:
    os.environ.setdefault("TASK_QUEUE_ENABLE", "1")
    os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "AIV")
    if target_tp_size == 3:
        # CANN's deterministic AIV kernel is faster for the 3-7 MiB TP3
        # all-reduces used by packed PEARL verification, and keeps greedy
        # decoding reproducible across independent worker launches.
        os.environ.setdefault("HCCL_DETERMINISTIC", "true")


@dataclass
class PearlPipelineState:
    """Logical sequence state replicated by the draft and target model groups."""

    token_ids: list[int]
    prompt_length: int
    pre_verify: bool = True
    accepted_draft_tokens: int = 0
    verified_draft_tokens: int = 0
    verification_rounds: int = 0
    committed_length: int | None = None
    # Number of leading real tokens whose draft-model KV rows are canonical.
    # PARD mask rows are deliberately excluded even when they occupy later
    # physical slots; accepted/correction tokens repair those slots in the
    # next PARD forward before this frontier advances.
    draft_synced_length: int | None = None
    temperature: float = 0.0
    draft_temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    draft_top_p: float = 1.0
    draft_top_k: int = 0
    max_tokens: int = 64
    ignore_eos: bool = False
    slo_tpot_ms: float | None = None
    slo_class: str | None = None
    finished_decode_elapsed_ms: float | None = None
    num_acc_tokens: list[int] | None = None
    cur_acc_tokens: int = 0
    pending_window_size: int = 0
    continuation_epoch: int = 0
    request_id: str | int | None = None
    arrival_ts: float | None = None
    aborted: bool = False

    def __post_init__(self) -> None:
        if self.committed_length is None:
            self.committed_length = len(self.token_ids)
        if self.draft_synced_length is None:
            # A newly constructed decode state assumes that prompt prefill is
            # the only canonical draft KV prefix.  The target's first sampled
            # token is appended later and must be repaired by the first PARD
            # call.
            self.draft_synced_length = self.prompt_length
        if self.num_acc_tokens is None:
            self.num_acc_tokens = []
        if self.temperature < 0 or self.draft_temperature < 0 or self.max_tokens <= 0:
            raise ValueError("PEARL sampling requires non-negative temperature and positive max_tokens.")
        if not 0 < self.top_p <= 1 or not 0 < self.draft_top_p <= 1 or self.top_k < 0 or self.draft_top_k < 0:
            raise ValueError("PEARL top-p must be in (0, 1] and top-k must be non-negative.")
        if self.slo_tpot_ms is not None and self.slo_tpot_ms <= 0:
            raise ValueError("PEARL TPOT SLO must be positive when supplied.")
        if self.pending_window_size < 0 or self.continuation_epoch < 0:
            raise ValueError("PEARL pending-window state must be non-negative.")
        if not self.prompt_length <= self.draft_synced_length <= self.committed_length:
            raise ValueError("PEARL draft KV frontier must lie between prompt and committed frontiers.")

    def clone(self) -> PearlPipelineState:
        return PearlPipelineState(
            token_ids=list(self.token_ids),
            prompt_length=self.prompt_length,
            pre_verify=self.pre_verify,
            accepted_draft_tokens=self.accepted_draft_tokens,
            verified_draft_tokens=self.verified_draft_tokens,
            verification_rounds=self.verification_rounds,
            committed_length=self.committed_length,
            draft_synced_length=self.draft_synced_length,
            temperature=self.temperature,
            draft_temperature=self.draft_temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            draft_top_p=self.draft_top_p,
            draft_top_k=self.draft_top_k,
            max_tokens=self.max_tokens,
            ignore_eos=self.ignore_eos,
            slo_tpot_ms=self.slo_tpot_ms,
            slo_class=self.slo_class,
            finished_decode_elapsed_ms=self.finished_decode_elapsed_ms,
            num_acc_tokens=list(self.num_acc_tokens or ()),
            cur_acc_tokens=self.cur_acc_tokens,
            pending_window_size=self.pending_window_size,
            continuation_epoch=self.continuation_epoch,
            request_id=self.request_id,
            arrival_ts=self.arrival_ts,
            aborted=self.aborted,
        )

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.prompt_length :]

    @property
    def committed_completion_token_ids(self) -> list[int]:
        assert self.committed_length is not None
        return self.token_ids[self.prompt_length : self.committed_length]

    def pard_repair_suffix(self) -> tuple[list[int], int]:
        """Return real tokens needed to repair PARD KV through the root.

        The current root is replayed when the real KV frontier is already
        synchronized.  That keeps every PARD call rooted in a real token and
        lets the following three mask rows produce the fixed four proposals.
        Uncommitted proposal/eager tails are rejected because their ownership
        cannot be represented by the linear PARD layout.
        """

        assert self.committed_length is not None
        assert self.draft_synced_length is not None
        if len(self.token_ids) != self.committed_length:
            raise RuntimeError("PARD drafting requires an exact committed token frontier.")
        if not self.prompt_length <= self.draft_synced_length <= self.committed_length:
            raise RuntimeError("PARD draft KV frontier is outside the committed sequence.")
        if self.committed_length <= 0:
            raise RuntimeError("PARD drafting requires a real root token.")
        first_position = min(self.draft_synced_length, self.committed_length - 1)
        return list(self.token_ids[first_position : self.committed_length]), first_position

    def mark_pard_draft_repaired(self) -> None:
        """Advance only the canonical real-token KV frontier after one PARD call."""

        assert self.committed_length is not None
        # Re-run all structural checks before mutating the frontier.
        self.pard_repair_suffix()
        self.draft_synced_length = self.committed_length

    def apply_pard_full_window_verification(
        self,
        *,
        proposal_token_ids: Sequence[int] | torch.Tensor,
        accepted: int,
        correction_token_id: int | None,
        bonus_token_id: int | None = None,
    ) -> None:
        """Apply a PARD verdict without promoting disposable mask KV rows."""

        assert self.committed_length is not None
        assert self.draft_synced_length is not None
        if self.draft_synced_length != self.committed_length:
            raise RuntimeError("A PARD verdict requires real-token KV repair through its proposal root.")
        repaired_frontier = self.draft_synced_length
        self.apply_draft_full_window_verification(
            proposal_token_ids=proposal_token_ids,
            accepted=accepted,
            correction_token_id=correction_token_id,
            bonus_token_id=bonus_token_id,
        )
        # Accepted proposals and target corrections were predicted from mask
        # rows but have not themselves been forwarded through the draft model.
        # They remain beyond the canonical KV frontier until the next repair.
        self.draft_synced_length = repaired_frontier

    def apply_target_verification(
        self,
        *,
        gamma: int,
        accepted: int,
        correction_token_id: int | None,
        next_round_token_ids: Sequence[int] | torch.Tensor,
        verification_size: int | None = None,
    ) -> None:
        """Apply nano-PEARL's target-side append/rollback transition."""
        expected = self._verification_size(gamma, verification_size)
        self._validate_verification(
            gamma,
            expected,
            accepted,
            correction_token_id,
            next_round_token_ids,
        )
        was_pre_verify = self.pre_verify
        assert self.committed_length is not None
        self.committed_length += accepted + (accepted < expected)
        self.accepted_draft_tokens += accepted
        self.verified_draft_tokens += expected
        self.verification_rounds += 1
        self.continuation_epoch += 1
        self._record_acceptance(expected=expected, accepted=accepted)
        if accepted == expected:
            self.token_ids.extend(next_round_token_ids)
            self.pre_verify = False
            self.pending_window_size = len(next_round_token_ids)
            return

        assert correction_token_id is not None
        if not was_pre_verify:
            rollout = expected - accepted
            if rollout > 1:
                del self.token_ids[-(rollout - 1) :]
        self.token_ids.append(correction_token_id)
        self.pre_verify = True
        self.pending_window_size = 0

    def apply_draft_verification(
        self,
        *,
        gamma: int,
        accepted: int,
        correction_token_id: int | None,
        next_round_token_ids: Sequence[int] | torch.Tensor,
        verification_size: int | None = None,
    ) -> None:
        """Apply the mirrored draft-side rollback after a target verdict."""
        expected = self._verification_size(gamma, verification_size)
        self._validate_verification(
            gamma,
            expected,
            accepted,
            correction_token_id,
            next_round_token_ids,
        )
        was_pre_verify = self.pre_verify
        assert self.committed_length is not None
        self.committed_length += accepted + (accepted < expected)
        self.accepted_draft_tokens += accepted
        self.verified_draft_tokens += expected
        self.verification_rounds += 1
        self.continuation_epoch += 1
        self._record_acceptance(expected=expected, accepted=accepted)
        if accepted == expected:
            self.pre_verify = False
            self.pending_window_size = len(next_round_token_ids)
            return

        assert correction_token_id is not None
        del self.token_ids[-len(next_round_token_ids) :]
        if not was_pre_verify:
            rollout = expected - accepted
            if rollout > 1:
                del self.token_ids[-(rollout - 1) :]
        self.token_ids.append(correction_token_id)
        self.pre_verify = True
        self.pending_window_size = 0

    def apply_target_full_window_verification(
        self,
        *,
        proposal_token_ids: Sequence[int] | torch.Tensor,
        accepted: int,
        correction_token_id: int | None,
        bonus_token_id: int | None = None,
    ) -> None:
        """Commit one complete serial proposal on the target-side state.

        This transition is intentionally separate from nano-PEARL's
        ``pre_verify``/``post_verify`` state machine.  A full-window target
        verifies ``proposal_token_ids`` directly from the committed frontier,
        so the proposal must not already be present in ``token_ids``.  Rejected
        proposal suffixes have no logical state to retain; their physical KV
        rows can be overwritten by the next forward at the same positions.
        """

        proposal = self._validate_full_window_verification(
            proposal_token_ids,
            accepted,
            correction_token_id,
            bonus_token_id,
        )
        assert self.committed_length is not None
        if len(self.token_ids) != self.committed_length:
            raise RuntimeError("A target full-window transition requires an exact committed token frontier.")
        self.token_ids.extend(proposal[:accepted])
        if accepted < len(proposal):
            assert correction_token_id is not None
            self.token_ids.append(int(correction_token_id))
        elif bonus_token_id is not None:
            self.token_ids.append(int(bonus_token_id))
        self._finish_full_window_verification(
            len(proposal),
            accepted,
            bonus_committed=bonus_token_id is not None,
        )

    def apply_draft_full_window_verification(
        self,
        *,
        proposal_token_ids: Sequence[int] | torch.Tensor,
        accepted: int,
        correction_token_id: int | None,
        staged_eager_token_ids: Sequence[int] | torch.Tensor | None = None,
        bonus_token_id: int | None = None,
    ) -> None:
        """Commit/rollback a complete serial proposal on the draft-side state.

        The draft model has already appended the proposal and may also have
        appended one guarded rolling-eager continuation.  Full acceptance
        advances the committed frontier through the proposal and leaves that
        eager suffix in place for promotion.  Rejection reconstructs the state
        from the old committed prefix, thereby discarding both the rejected
        proposal suffix and every staged eager token.
        """

        proposal = self._validate_full_window_verification(
            proposal_token_ids,
            accepted,
            correction_token_id,
            bonus_token_id,
        )
        eager = (
            []
            if staged_eager_token_ids is None
            else self._materialize_full_window_tokens(
                staged_eager_token_ids,
                label="staged eager continuation",
            )
        )
        assert self.committed_length is not None
        expected_tail = [*proposal, *eager]
        actual_tail = self.token_ids[self.committed_length :]
        if actual_tail != expected_tail:
            raise RuntimeError(
                "A draft full-window transition requires its token tail to equal proposal + staged eager continuation."
            )
        if bonus_token_id is not None and eager:
            raise ValueError("A full-window bonus token cannot retain a staged eager continuation.")
        if accepted < len(proposal):
            del self.token_ids[self.committed_length :]
            self.token_ids.extend(proposal[:accepted])
            assert correction_token_id is not None
            self.token_ids.append(int(correction_token_id))
        elif bonus_token_id is not None:
            self.token_ids.append(int(bonus_token_id))
        self._finish_full_window_verification(
            len(proposal),
            accepted,
            bonus_committed=bonus_token_id is not None,
        )

    @staticmethod
    def _materialize_full_window_tokens(
        token_ids: Sequence[int] | torch.Tensor,
        *,
        label: str,
    ) -> list[int]:
        if torch.is_tensor(token_ids):
            if token_ids.ndim != 1:
                raise ValueError(f"A {label} must be a one-dimensional token sequence.")
            values = [int(value) for value in token_ids.detach().cpu().tolist()]
        else:
            values = [int(value) for value in token_ids]
        if not values:
            raise ValueError(f"A {label} must contain at least one token.")
        if any(value < 0 for value in values):
            raise ValueError(f"A {label} cannot contain negative token IDs.")
        return values

    def _validate_full_window_verification(
        self,
        proposal_token_ids: Sequence[int] | torch.Tensor,
        accepted: int,
        correction_token_id: int | None,
        bonus_token_id: int | None = None,
    ) -> list[int]:
        proposal = self._materialize_full_window_tokens(
            proposal_token_ids,
            label="full-window proposal",
        )
        if not 0 <= accepted <= len(proposal):
            raise ValueError("Full-window accepted length is outside the proposal.")
        if accepted == len(proposal) and correction_token_id is not None:
            raise ValueError("A fully accepted full-window proposal cannot include a correction token.")
        if accepted < len(proposal) and correction_token_id is None:
            raise ValueError("A rejected full-window proposal requires a target correction token.")
        if correction_token_id is not None and int(correction_token_id) < 0:
            raise ValueError("A full-window correction token cannot be negative.")
        if bonus_token_id is not None:
            if accepted != len(proposal) or correction_token_id is not None:
                raise ValueError("A full-window bonus token requires complete acceptance without correction.")
            if int(bonus_token_id) < 0:
                raise ValueError("A full-window bonus token cannot be negative.")
        assert self.committed_length is not None
        if not self.prompt_length <= self.committed_length <= len(self.token_ids):
            raise RuntimeError("A full-window transition has an invalid committed token frontier.")
        return proposal

    def _finish_full_window_verification(
        self,
        expected: int,
        accepted: int,
        *,
        bonus_committed: bool = False,
    ) -> None:
        assert self.committed_length is not None
        self.committed_length += accepted + int(accepted < expected) + int(bonus_committed)
        self.accepted_draft_tokens += accepted
        self.verified_draft_tokens += expected
        self.verification_rounds += 1
        self.continuation_epoch += 1
        self._record_acceptance(expected=expected, accepted=accepted)
        # A full-window proposal has no retained nano-PEARL cross-window
        # verification prefix.  Its service loop must continue to call the
        # explicit full-window API rather than infer width from these legacy
        # flags.
        self.pre_verify = True
        self.pending_window_size = 0

    def apply_tree_verification(
        self,
        *,
        output_token_ids: Sequence[int] | torch.Tensor,
        accepted: int,
        proposed_tokens: int,
    ) -> None:
        """Commit one request-local greedy-tree path.

        ``verify_greedy_tree*`` returns the accepted path followed by a bonus
        token and fills the remaining fixed-depth slots with ``-1``.  Unlike
        the linear PEARL transition there is no speculative continuation row
        to retain: the committed path is the next root for the following
        tree.  The method still records the same acceptance accounting used by
        the SLO shaper.
        """
        values = [int(value) for value in output_token_ids]
        committed = [value for value in values if value >= 0]
        if not committed:
            raise ValueError("a tree verification must commit at least one token")
        if proposed_tokens <= 0 or accepted < 0 or accepted > proposed_tokens:
            raise ValueError("tree acceptance accounting is invalid")
        assert self.committed_length is not None
        self.token_ids.extend(committed)
        self.committed_length += len(committed)
        self.accepted_draft_tokens += int(accepted)
        self.verified_draft_tokens += int(proposed_tokens)
        self.verification_rounds += 1
        self.continuation_epoch += 1
        self._record_acceptance(expected=int(proposed_tokens), accepted=int(accepted))
        self.pre_verify = True
        self.pending_window_size = 0

    def _verification_size(self, gamma: int, verification_size: int | None) -> int:
        if self.pre_verify:
            expected = 1
        elif verification_size is not None:
            expected = int(verification_size)
        elif self.pending_window_size:
            expected = self.pending_window_size
        else:
            # Compatibility for states constructed before variable per-request
            # windows were introduced.
            expected = gamma
        if expected <= 0 or expected > gamma:
            raise ValueError("PEARL verification size must be in [1, gamma].")
        return expected

    def _validate_verification(
        self,
        gamma: int,
        expected: int,
        accepted: int,
        correction_token_id: int | None,
        next_round_token_ids: list[int],
    ) -> None:
        if not 0 < len(next_round_token_ids) <= gamma:
            raise ValueError("Every PEARL next-round window must contain 1..gamma draft tokens.")
        if not 0 <= accepted <= expected:
            raise ValueError("PEARL accepted length is outside the verification window.")
        if accepted == expected and correction_token_id is not None:
            raise ValueError("A fully accepted PEARL window cannot include a correction token.")
        if accepted < expected and correction_token_id is None:
            raise ValueError("A rejected PEARL window requires a target correction token.")

    def _record_acceptance(self, *, expected: int, accepted: int) -> None:
        assert self.num_acc_tokens is not None
        if accepted == expected:
            self.cur_acc_tokens += accepted
        else:
            # Upstream MAT treats the target correction as part of the output
            # segment terminated by this rejection.
            self.num_acc_tokens.append(self.cur_acc_tokens + accepted + 1)
            self.cur_acc_tokens = 0

    @property
    def acceptance_lengths(self) -> list[int]:
        if self.verified_draft_tokens == 0:
            return []
        return [*(self.num_acc_tokens or ()), self.cur_acc_tokens]


@dataclass(frozen=True)
class NativeSamplingParams:
    """Per-request sampling controls supported by upstream nano-PEARL."""

    temperature: float = 1.0
    draft_temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    draft_top_p: float = 1.0
    draft_top_k: int = 0
    max_tokens: int = 64
    ignore_eos: bool = False
    slo_tpot_ms: float | None = None
    slo_class: str | None = None
    spec_rhythm_max_gamma: int | None = None
    request_id: str | int | None = None
    arrival_ts: float | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0 or self.draft_temperature < 0:
            raise ValueError("PEARL and draft temperatures must be non-negative.")
        if not 0 < self.top_p <= 1 or not 0 < self.draft_top_p <= 1:
            raise ValueError("PEARL and draft top-p must be in (0, 1].")
        if self.top_k < 0 or self.draft_top_k < 0:
            raise ValueError("PEARL and draft top-k must be non-negative.")
        if self.max_tokens <= 0:
            raise ValueError("PEARL max_tokens must be positive.")
        if self.slo_tpot_ms is not None and self.slo_tpot_ms <= 0:
            raise ValueError("PEARL TPOT SLO must be positive when supplied.")
        if self.spec_rhythm_max_gamma is not None and self.spec_rhythm_max_gamma <= 0:
            raise ValueError("SpecRhythm per-request max gamma must be positive.")
        if self.arrival_ts is not None and not math.isfinite(self.arrival_ts):
            raise ValueError("SpecRhythm request arrival timestamp must be finite.")


# Match upstream's public name while retaining a native-specific explicit name.
SamplingParams = NativeSamplingParams


@dataclass
class NativeSpecRhythmDevicePayload:
    """Rank-local tensor views for one guarded SpecRhythm proposal."""

    ticket: SpecRhythmProposalTicket
    verification_tokens: torch.Tensor
    next_tokens: torch.Tensor
    verification_size: int
    # Target ranks keep this as a device scalar until the verdict result is
    # materialized.  That lets the proposal and verdict exchanges share one
    # device-to-host synchronization per round.
    draft_confidence: float | torch.Tensor | None

    def validate_for(
        self,
        state: PearlPipelineState,
        *,
        full_window: bool = False,
    ) -> None:
        if self.ticket.required_prefix_epoch != state.continuation_epoch:
            raise RuntimeError(
                "SpecRhythm refused a stale device mailbox proposal: "
                f"request={self.ticket.request_index}, "
                f"required_epoch={self.ticket.required_prefix_epoch}, "
                f"current_epoch={state.continuation_epoch}."
            )
        if self.verification_tokens.shape != (self.verification_size,):
            raise RuntimeError("SpecRhythm mailbox verification tensor has an invalid shape.")
        if self.verification_size <= 0:
            raise RuntimeError("SpecRhythm mailbox verification width must be positive.")
        expected_size = (
            self.ticket.gamma
            if full_window
            else 1
            if state.pre_verify
            else (state.pending_window_size or self.ticket.gamma)
        )
        if self.verification_size != expected_size:
            raise RuntimeError("SpecRhythm mailbox verification width does not match the request prefix.")
        if self.next_tokens.shape != (self.ticket.gamma,):
            raise RuntimeError("SpecRhythm mailbox continuation tensor has an invalid shape.")
        if self.draft_confidence is None:
            return
        if torch.is_tensor(self.draft_confidence):
            if self.draft_confidence.numel() != 1:
                raise RuntimeError("SpecRhythm mailbox confidence must be a scalar.")
            return
        if not math.isfinite(self.draft_confidence) or not 0.0 <= self.draft_confidence <= 1.0:
            raise RuntimeError("SpecRhythm mailbox confidence must be finite and in [0, 1].")


@dataclass(frozen=True)
class NativePearlConfig:
    draft_model: str
    target_model: str
    draft_tp_size: int
    target_tp_size: int
    gamma: int
    max_model_len: int
    max_tokens: int
    draft_dtype: str = "auto"
    target_dtype: str = "auto"
    max_num_seqs: int = 1
    prefill_chunk_size: int | None = None
    auto_gamma_profile_sequence_length: int = AUTO_GAMMA_PROFILE_SEQUENCE_LENGTH
    max_num_queued_seqs: int | None = None
    max_num_batched_tokens: int = 16384
    gpu_memory_utilization: float = 0.9
    kvcache_block_size: int = PAGED_ATTENTION_BLOCK_SIZE
    num_kvcache_blocks: int = -1
    max_aclgraph_entries: int = 32
    target_verification_graph_buckets: int = TARGET_VERIFICATION_GRAPH_BUCKETS
    target_verification_graph_post_counts: tuple[tuple[int, tuple[int, ...]], ...] = ()
    enable_prefix_caching: bool = True
    enable_continuous_batching: bool = False
    enable_preemptive_scheduling: bool = False
    enable_spec_rhythm: bool = False
    # Execution-mechanism ablation on the same online SpecRhythm worker path.
    # ``auto`` preserves the production controller.  The explicit modes are
    # used only for fixed-gamma, linear full-window comparisons.
    spec_rhythm_ablation_mode: str = "auto"
    spec_rhythm_online_prefill: bool = False
    # Opt-in bounded coalescing for arrival-gated fixed-gamma serial prefill.
    # The 1/0 defaults preserve immediate admission exactly.
    spec_rhythm_prefill_coalesce_min_requests: int = 1
    spec_rhythm_prefill_coalesce_max_wait_ms: float = 0.0
    spec_rhythm_prefill_token_chunk_size: int = 0
    # Opt-in only: merging both ready homes removes the alternating
    # dual-batch window used by SpecRhythm's rolling-eager policy.
    spec_rhythm_merge_ready_homes: bool = False
    spec_rhythm_priority_mode: bool = False
    spec_rhythm_priority_burst: int = 2
    spec_rhythm_target_fallback_max_batch: int = 0
    # Optional upper bound for one target verification forward.  Zero keeps
    # the legacy one-home limit; a smaller value trades peak throughput for
    # tighter per-request TPOT under mixed SLO load.
    spec_rhythm_max_target_batch: int = 0
    spec_rhythm_min_gamma: int = 1
    spec_rhythm_max_eager_tokens: int = 0
    # Aggregate upper bound inside global B for complete rolling-eager
    # proposals reserved before normal-row admission. Zero is backward
    # compatible and keeps the normal-first policy.
    spec_rhythm_eager_reserve_tokens: int = 0
    spec_rhythm_urgency_threshold: float = 0.75
    spec_rhythm_acceptance_floor: float = 0.4
    spec_rhythm_acceptance_ema_alpha: float = 0.2
    spec_rhythm_cpu_verdict: bool = False
    # Verify every fixed-gamma serial proposal directly as
    # ``[committed root, proposal[:-1]] -> proposal``.  This is deliberately
    # independent of nano-PEARL's pre/post-verify state machine, whose
    # rejection path verifies one token before rebuilding the wider window.
    spec_rhythm_linear_full_window: bool = False
    # Opt-in standard speculative-decoding bonus token.  A gamma-4 target
    # query executes one additional causal step and may commit its fifth
    # greedy token only when all four draft tokens match.
    spec_rhythm_linear_bonus_token: bool = False
    # Experimental SLO policy: let measured W admit rolling-eager rows even
    # when they require the next serial-draft ACLGraph bucket. False retains
    # the conservative same-bucket gate and is the compatible default.
    spec_rhythm_linear_eager_cross_graph_bucket: bool = False
    # Opt-in residual-Goodput policy for otherwise idle draft windows.  It
    # never replaces mandatory normal work and remains bounded by measured W.
    spec_rhythm_linear_idle_residual_eager: bool = False
    # Legacy bare budgets or an identity-bound measured profile/path.
    spec_rhythm_roofline: Mapping[str, Any] | str | None = None
    spec_rhythm_verification_budget: int | None = None
    spec_rhythm_draft_token_budget: int | None = None
    spec_rhythm_tree_width: int = 1
    spec_rhythm_tree_depth: int = 1
    # Dynamic FIA task updates are not replay-stable for every Ascend model
    # shape.  Keep SpecRhythm on fixed paged-attention graph buckets unless an
    # explicit experiment opts into FIA.
    spec_rhythm_stable_graphs: bool = True
    pad_finished_requests: bool = False
    draft_use_paged_attention: bool = False
    target_use_paged_attention: bool = False
    draft_use_production_rope: bool = True
    target_use_production_rope: bool = True
    precompile_decode_graphs: bool = False
    precompile_serial_draft_graphs: bool = False
    enable_cpu_binding: bool = True
    profile_decode_steps: int = 0
    profile_host_decode_steps: int = 0
    stop_after_profiled_decode_steps: bool = False
    enforce_eager: bool = False
    enable_mc2: bool = False
    mc2_profile: Mapping[str, Any] | str | None = None
    seed: int | None = None
    draft_tp1_greedy_argmax: bool = False
    draft_mode: str = SERIAL_LINEAR_DRAFT_MODE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_verification_graph_post_counts",
            _normalize_target_graph_post_counts(self.target_verification_graph_post_counts),
        )
        if self.draft_tp_size <= 0 or self.target_tp_size <= 0:
            raise ValueError("Draft and target TP sizes must be positive.")
        validate_native_draft_mode(self.draft_mode, self.gamma)
        if self.draft_tp1_greedy_argmax and self.draft_tp_size != 1:
            raise ValueError("The draft greedy argmax fast path requires draft TP size 1.")
        supported_dtypes = {"auto", "bfloat16", "float16"}
        if self.draft_dtype not in supported_dtypes or self.target_dtype not in supported_dtypes:
            raise ValueError("PEARL model dtype must be auto, bfloat16, or float16.")
        if self.gamma == 0 or self.gamma < -1:
            raise ValueError("PEARL gamma must be positive, or -1 for automatic selection.")
        if self.max_model_len <= 0 or self.max_tokens <= 0 or self.max_num_seqs <= 0:
            raise ValueError("PEARL model, generation, and batch limits must be positive.")
        prefill_limit = self.max_num_queued_seqs or self.max_num_seqs
        if self.prefill_chunk_size is not None and not 0 < self.prefill_chunk_size <= prefill_limit:
            raise ValueError("PEARL prefill_chunk_size must fit the queued request capacity.")
        if self.auto_gamma_profile_sequence_length <= 0:
            raise ValueError("PEARL auto-gamma profile sequence length must be positive.")
        if self.max_num_queued_seqs is not None and self.max_num_queued_seqs < self.max_num_seqs:
            raise ValueError("PEARL max_num_queued_seqs must be at least max_num_seqs.")
        if self.enable_preemptive_scheduling and not self.enable_continuous_batching:
            raise ValueError("PEARL preemptive scheduling requires continuous batching.")
        if self.enable_spec_rhythm and not (self.enable_continuous_batching and self.enable_preemptive_scheduling):
            raise ValueError("SpecRhythm requires PEARL continuous batching and preemptive scheduling.")
        if self.enable_spec_rhythm and self.gamma == -1:
            raise ValueError("SpecRhythm requires a fixed maximum PEARL gamma.")
        if self.spec_rhythm_ablation_mode not in SPEC_RHYTHM_ABLATION_MODES:
            choices = ", ".join(sorted(SPEC_RHYTHM_ABLATION_MODES))
            raise ValueError(
                f"Unknown SpecRhythm ablation mode {self.spec_rhythm_ablation_mode!r}; expected one of {choices}."
            )
        if self.spec_rhythm_ablation_mode != "auto" and not (
            self.enable_spec_rhythm
            and self.spec_rhythm_online_prefill
            and self.spec_rhythm_linear_full_window
            and self.gamma > 0
            and self.spec_rhythm_min_gamma == self.gamma
            and self.spec_rhythm_tree_width == 1
            and self.spec_rhythm_tree_depth == 1
        ):
            raise ValueError(
                "Explicit SpecRhythm ablation modes require SpecRhythm fixed-gamma "
                "online linear full-window execution with tree shape 1x1."
            )
        if self.spec_rhythm_ablation_mode != "auto" and self.spec_rhythm_merge_ready_homes:
            raise ValueError("Explicit SpecRhythm ablations cannot merge logical homes.")
        if self.spec_rhythm_linear_full_window and not self.enable_spec_rhythm:
            raise ValueError("Linear full-window verification requires SpecRhythm.")
        if self.spec_rhythm_linear_full_window and (
            self.gamma <= 0
            or self.spec_rhythm_tree_width != 1
            or self.spec_rhythm_tree_depth != 1
            or self.spec_rhythm_min_gamma != self.gamma
        ):
            raise ValueError("Linear full-window verification requires fixed gamma and a 1x1 serial topology.")
        if self.spec_rhythm_linear_bonus_token and not (
            self.enable_spec_rhythm
            and self.spec_rhythm_linear_full_window
            and self.gamma == FIXED_GREEDY_FULL_WINDOW_WIDTH
            and self.spec_rhythm_min_gamma == self.gamma
            and self.spec_rhythm_tree_width == 1
            and self.spec_rhythm_tree_depth == 1
        ):
            raise ValueError(
                "Linear bonus-token mode requires SpecRhythm full-window "
                "verification with fixed gamma 4 and a 1x1 serial topology."
            )
        if self.spec_rhythm_linear_eager_cross_graph_bucket and not (
            self.enable_spec_rhythm and self.spec_rhythm_linear_full_window
        ):
            raise ValueError("Cross-graph-bucket linear eager scheduling requires SpecRhythm linear full-window mode.")
        if self.spec_rhythm_linear_idle_residual_eager and not (
            self.enable_spec_rhythm and self.spec_rhythm_linear_full_window
        ):
            raise ValueError("Idle residual linear eager scheduling requires SpecRhythm linear full-window mode.")
        if self.spec_rhythm_prefill_coalesce_min_requests <= 0:
            raise ValueError("SpecRhythm prefill coalescing minimum must be positive.")
        if (
            not math.isfinite(self.spec_rhythm_prefill_coalesce_max_wait_ms)
            or self.spec_rhythm_prefill_coalesce_max_wait_ms < 0
        ):
            raise ValueError("SpecRhythm prefill coalescing wait must be finite and non-negative.")
        prefill_coalescing = (
            self.spec_rhythm_prefill_coalesce_min_requests != 1 or self.spec_rhythm_prefill_coalesce_max_wait_ms != 0
        )
        if prefill_coalescing and not (
            self.enable_spec_rhythm and self.spec_rhythm_online_prefill and self.spec_rhythm_linear_full_window
        ):
            raise ValueError(
                "SpecRhythm prefill coalescing requires enable_spec_rhythm, "
                "online prefill, and linear full-window mode."
            )
        if prefill_coalescing and (
            self.spec_rhythm_prefill_coalesce_min_requests <= 1 or self.spec_rhythm_prefill_coalesce_max_wait_ms <= 0
        ):
            raise ValueError("SpecRhythm prefill coalescing requires a minimum above one and a positive bounded wait.")
        coalesce_capacity = min(
            self.max_num_seqs,
            self.prefill_chunk_size or self.max_num_seqs,
        )
        if self.spec_rhythm_prefill_coalesce_min_requests > coalesce_capacity:
            raise ValueError("SpecRhythm prefill coalescing minimum must fit the prefill and decode capacities.")
        if self.spec_rhythm_prefill_token_chunk_size < 0:
            raise ValueError("SpecRhythm prefill token chunk size must be non-negative.")
        if self.spec_rhythm_prefill_token_chunk_size > 0:
            if not (
                self.enable_spec_rhythm and self.spec_rhythm_online_prefill and self.spec_rhythm_linear_full_window
            ):
                raise ValueError(
                    "SpecRhythm token-chunk prefill requires enable_spec_rhythm, "
                    "online prefill, and linear full-window mode."
                )
            if self.enable_prefix_caching:
                raise ValueError("SpecRhythm token-chunk prefill requires prefix caching to be disabled.")
            if self.spec_rhythm_prefill_token_chunk_size > self.max_num_batched_tokens:
                raise ValueError("SpecRhythm prefill token chunk size must not exceed max_num_batched_tokens.")
        if self.spec_rhythm_min_gamma <= 0 or (self.gamma != -1 and self.spec_rhythm_min_gamma > self.gamma):
            raise ValueError("SpecRhythm min gamma must be in [1, gamma].")
        if self.spec_rhythm_max_eager_tokens < 0 or (
            self.gamma != -1 and self.spec_rhythm_max_eager_tokens > self.gamma
        ):
            raise ValueError("SpecRhythm eager-token cap must be in [0, gamma].")
        if self.spec_rhythm_linear_full_window and self.spec_rhythm_max_eager_tokens not in (0, self.gamma):
            raise ValueError("Linear full-window eager-token cap must be 0 or gamma.")
        if self.spec_rhythm_eager_reserve_tokens < 0:
            raise ValueError("SpecRhythm eager-token reserve must be non-negative.")
        if self.spec_rhythm_priority_burst <= 0:
            raise ValueError("SpecRhythm priority burst must be positive.")
        if self.spec_rhythm_target_fallback_max_batch < 0:
            raise ValueError("SpecRhythm target fallback batch must be non-negative.")
        if self.spec_rhythm_max_target_batch < 0:
            raise ValueError("SpecRhythm max target batch must be non-negative.")
        if not self.spec_rhythm_urgency_threshold >= 0:
            raise ValueError("SpecRhythm urgency threshold must be non-negative.")
        if not 0 <= self.spec_rhythm_acceptance_floor <= 1:
            raise ValueError("SpecRhythm acceptance floor must be in [0, 1].")
        if not 0 < self.spec_rhythm_acceptance_ema_alpha <= 1:
            raise ValueError("SpecRhythm acceptance EMA alpha must be in (0, 1].")
        if self.spec_rhythm_roofline is not None:
            object.__setattr__(
                self,
                "spec_rhythm_roofline",
                normalize_roofline(
                    self.spec_rhythm_roofline,
                    model=self.target_model,
                    target_tp_size=self.target_tp_size,
                    enforce_eager=self.enforce_eager,
                    max_model_len=self.max_model_len,
                    tree_width=self.spec_rhythm_tree_width,
                    tree_depth=self.spec_rhythm_tree_depth,
                ),
            )
            if (
                isinstance(self.spec_rhythm_roofline, ProfiledRoofline)
                and self.spec_rhythm_verification_budget is not None
            ):
                raise ValueError("A strict measured roofline cannot be overridden by a fixed verification budget")
            if (
                isinstance(self.spec_rhythm_roofline, ProfiledRoofline)
                and self.spec_rhythm_tree_width * self.spec_rhythm_tree_depth <= 1
            ):
                raise ValueError("A measured packed-tree roofline requires the SpecRhythm tree path, not linear PEARL")
            if isinstance(self.spec_rhythm_roofline, ProfiledRoofline):
                if any(value < 0 for value in self.spec_rhythm_roofline.values()):
                    raise ValueError("Measured SpecRhythm roofline values must be non-negative token budgets.")
            elif any(value <= 0 for value in self.spec_rhythm_roofline.values()):
                raise ValueError("Legacy SpecRhythm roofline values must be positive token budgets.")
        if self.spec_rhythm_draft_token_budget is not None and self.spec_rhythm_draft_token_budget <= 0:
            raise ValueError("SpecRhythm draft-token budget must be positive when supplied.")
        if (
            self.spec_rhythm_linear_full_window
            and self.spec_rhythm_draft_token_budget is not None
            and self.spec_rhythm_draft_token_budget < self.gamma
        ):
            raise ValueError("Linear full-window draft-token budget must be at least gamma.")
        if self.spec_rhythm_verification_budget is not None and self.spec_rhythm_verification_budget <= 0:
            raise ValueError("SpecRhythm verification budget must be positive when supplied.")
        if self.spec_rhythm_tree_width <= 0 or self.spec_rhythm_tree_depth <= 0:
            raise ValueError("SpecRhythm tree width and depth must be positive.")
        if (
            self.enable_spec_rhythm
            and self.spec_rhythm_min_gamma > (self.spec_rhythm_tree_width * self.spec_rhythm_tree_depth)
            and (self.spec_rhythm_tree_width > 1 or self.spec_rhythm_tree_depth > 1)
        ):
            raise ValueError("SpecRhythm tree capacity must fit min_gamma.")
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError("PEARL max_num_batched_tokens must be at least max_model_len.")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("PEARL gpu_memory_utilization must be in (0, 1].")
        if self.kvcache_block_size != PAGED_ATTENTION_BLOCK_SIZE:
            raise ValueError(f"Native Ascend PEARL requires kvcache_block_size={PAGED_ATTENTION_BLOCK_SIZE}.")
        if self.num_kvcache_blocks == 0 or self.num_kvcache_blocks < -1:
            raise ValueError("PEARL num_kvcache_blocks must be positive, or -1 for automatic sizing.")
        if self.max_aclgraph_entries <= 0:
            raise ValueError("PEARL max_aclgraph_entries must be positive.")
        if self.target_verification_graph_buckets <= 0:
            raise ValueError("PEARL target_verification_graph_buckets must be positive.")
        if self.precompile_decode_graphs:
            if self.gamma == -1:
                raise ValueError("PEARL decode graph precompilation requires a fixed gamma.")
            if not self.draft_use_paged_attention or not self.target_use_paged_attention:
                raise ValueError("PEARL decode graph precompilation requires paged attention on both models.")
            graph_batch_sizes = [self.max_num_seqs]
            if self.enable_continuous_batching and self.max_num_seqs > 1:
                graph_batch_sizes.append(max(1, self.max_num_seqs // 2))
            graph_shapes = _target_graph_precompile_shapes(
                graph_batch_sizes,
                self.gamma,
                self.target_verification_graph_buckets,
                self.target_verification_graph_post_counts,
            )
            if len(graph_shapes) > self.max_aclgraph_entries:
                raise ValueError("PEARL decode graph precompilation exceeds max_aclgraph_entries.")
        if self.precompile_serial_draft_graphs:
            if (
                not self.enable_spec_rhythm
                or self.gamma <= 0
                or self.spec_rhythm_tree_width != 1
                or self.spec_rhythm_tree_depth != 1
                or not self.spec_rhythm_stable_graphs
            ):
                raise ValueError(
                    "Serial draft graph precompilation requires fixed-gamma "
                    "stable SpecRhythm with a 1x1 serial topology."
                )
            if not self.draft_use_paged_attention:
                raise ValueError("Serial draft graph precompilation requires paged attention on the draft model.")
            if self.enforce_eager:
                raise ValueError("Serial draft graph precompilation is incompatible with enforce_eager.")
            if len(_linear_draft_graph_buckets(self.max_num_seqs)) > self.max_aclgraph_entries:
                raise ValueError("Serial draft graph precompilation exceeds max_aclgraph_entries.")
        if self.profile_decode_steps < 0:
            raise ValueError("PEARL profile_decode_steps must be non-negative.")
        if self.profile_host_decode_steps < 0:
            raise ValueError("PEARL profile_host_decode_steps must be non-negative.")
        if self.stop_after_profiled_decode_steps and self.profile_decode_steps == 0:
            raise ValueError("PEARL profiling-only execution requires profile_decode_steps to be positive.")
        if self.mc2_profile is not None:
            object.__setattr__(self, "mc2_profile", normalize_mc2_profile(self.mc2_profile))
        if self.enable_mc2 and not isinstance(self.mc2_profile, MC2Profile):
            raise ValueError("MC2 execution requires an identity-bound numerical/performance profile")
        if self.seed is not None and self.seed < 0:
            raise ValueError("PEARL sampling seed must be non-negative.")


@dataclass(frozen=True)
class _LinearDraftFIAPersistentEnvelope:
    """Fixed metadata wrappers around one mutable FIA staging allocation."""

    staging: LinearDraftFIAPersistentStaging
    position_tensors: tuple[torch.Tensor, ...]
    attention_metadatas: tuple[NativeAttentionMetadata, ...]


class NativePearlEngine:
    """One rank of nano-PEARL's persistent HCCL runtime."""

    def __init__(self, config: NativePearlConfig) -> None:
        self.config = config
        self.gamma = config.gamma
        self.pard_token: int | None = None
        if config.draft_mode == PARD_PARALLEL_DRAFT_MODE:
            draft_model_config = AutoConfig.from_pretrained(config.draft_model)
            target_model_config = AutoConfig.from_pretrained(config.target_model)
            self.pard_token = validate_pard_parallel_model_pair(
                draft_model_config,
                target_model_config,
                gamma=config.gamma,
            )
            require_experimental_pard_eager(
                config,
                enabled=envs.VLLM_ASCEND_SPECSLO_ENABLE_EXPERIMENTAL_PARD_EAGER,
            )
        # Bind this spawn worker before HCCL creates any communicator. Creating
        # the world group on the inherited default device and switching later
        # leaves MC2 resource allocation associated with the wrong NPU.
        self.local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        torch.npu.set_device(self.local_rank)
        # torch.distributed's env:// rendezvous is established by torchrun.
        if not dist.is_initialized():
            _set_default_npu_environment(config.target_tp_size)
            dist.init_process_group(backend="hccl")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # Match the production vLLM-Ascend model runner so NPU kernels may
        # retain optimized internal layouts (for example FRACTAL_NZ weights).
        torch.npu.config.allow_internal_format = True
        init_device_properties_triton()
        self.device = torch.device("npu")

        self.topology = PearlTopology.from_tensor_parallel_sizes(
            config.draft_tp_size,
            config.target_tp_size,
        )
        self.groups = PearlProcessGroups.create(self.topology, backend="hccl")
        if self.world_size != self.topology.world_size:
            raise ValueError(f"PEARL needs {self.topology.world_size} ranks, received {self.world_size}.")

        self.is_draft = self.groups.is_draft_worker
        self._mixed_target_graph_enabled = bool(not self.is_draft and envs.VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH)
        self._mixed_target_graph_prompt_buckets = resolve_mixed_target_graph_buckets(
            envs.VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_BUCKETS
        )
        mixed_target_graph_max_tokens = int(envs.VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_MAX_TOKENS)
        if mixed_target_graph_max_tokens < 0:
            raise ValueError("VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_MAX_TOKENS must be non-negative.")
        if isinstance(config.spec_rhythm_roofline, ProfiledRoofline):
            hardware_error = None
            if not self.is_draft:
                try:
                    validate_roofline_hardware(config.spec_rhythm_roofline, torch.npu.get_device_name(self.local_rank))
                except (ValueError, RuntimeError) as error:
                    hardware_error = error
            # A target-local failure must not leave draft/peer ranks loading
            # models and eventually waiting forever in a later collective.
            invalid_hardware = torch.tensor(
                [int(hardware_error is not None)],
                dtype=torch.int64,
                device=self.device,
            )
            dist.all_reduce(invalid_hardware, op=dist.ReduceOp.MAX)
            if int(invalid_hardware.cpu().item()):
                if hardware_error is not None:
                    raise ValueError(str(hardware_error)) from hardware_error
                raise ValueError("SpecSLO roofline hardware validation failed on another target rank")
        self.model_group = self.groups.model_group
        self.model_context = NativeTPContext(
            group=self.model_group,
            rank=self.rank if self.is_draft else self.rank - config.draft_tp_size,
            size=config.draft_tp_size if self.is_draft else config.target_tp_size,
            leader_rank=self.topology.draft_leader_rank if self.is_draft else self.topology.target_leader_rank,
        )

        projection = validate_model_pair(config.draft_model, config.target_model)
        self.draft_vocab_size = projection.draft_vocab_size
        self.target_vocab_size = projection.target_vocab_size
        draft_model_config = AutoConfig.from_pretrained(config.draft_model)
        target_model_config = AutoConfig.from_pretrained(config.target_model)
        draft_model_config.pearl_use_production_rope = config.draft_use_production_rope
        target_model_config.pearl_use_production_rope = config.target_use_production_rope
        draft_model_config.pearl_enable_mc2 = config.enable_mc2
        target_model_config.pearl_enable_mc2 = config.enable_mc2
        draft_model_config.pearl_tp1_greedy_argmax = config.draft_tp1_greedy_argmax
        target_model_config.pearl_tp1_greedy_argmax = False
        draft_model_config.pearl_mc2_profile = config.mc2_profile
        target_model_config.pearl_mc2_profile = config.mc2_profile
        for model_config, dtype_name in (
            (draft_model_config, config.draft_dtype),
            (target_model_config, config.target_dtype),
        ):
            model_config.pearl_track_cache_finiteness = config.enable_spec_rhythm and (
                config.spec_rhythm_tree_width > 1 or config.spec_rhythm_tree_depth > 1
            )
            if dtype_name != "auto":
                model_config.torch_dtype = getattr(torch, dtype_name)
        if _normalize_eos_tokens(draft_model_config.eos_token_id) != _normalize_eos_tokens(
            target_model_config.eos_token_id
        ):
            raise ValueError("Native PEARL requires identical draft and target EOS token IDs.")
        model_path = config.draft_model if self.is_draft else config.target_model
        model_config = draft_model_config if self.is_draft else target_model_config
        self.model = build_native_model(
            model_config,
            self.model_context,
            config.max_model_len,
            config.max_num_seqs,
            configure_cache=False,
        )
        load_native_model_weights(self.model, model_path)
        num_cache_blocks = self._resolve_num_cache_blocks()
        self.model.configure_cache(
            config.max_model_len,
            self._cache_storage_sequence_capacity,
            config.kvcache_block_size,
            num_cache_blocks,
        )
        # Layer KV tensors are allocated once for the engine lifetime. Keep a
        # validated view so the decode commit path does not rescan every model
        # layer before every tree move. CPU protocol harnesses construct the
        # engine with ``__new__`` and exercise the uncached validation path.
        self._tree_layer_caches = self._collect_tree_layer_caches()
        self.tree_kv_graph_runner = TreeKVCompactionGraphRunner(
            self._tree_layer_caches,
            enabled=not config.enforce_eager,
            # Exact move counts are normally only 1..tree_depth.  Keep this
            # pool independently bounded so compaction graphs cannot consume
            # the model-forward graph runner's entry budget.
            max_graph_entries=min(16, config.max_aclgraph_entries),
        )
        self._tree_cache_capacity = min(
            int(cache.shape[0]) * int(cache.shape[1]) for pair in self._tree_layer_caches for cache in pair
        )
        attention = self.model.layers[0].self_attn
        assert attention.key_cache is not None
        self.prefix_cache = NativePrefixCache(
            num_blocks=attention.key_cache.shape[0],
            blocks_per_sequence=attention.blocks_per_sequence,
            block_size=attention.block_size,
        )
        self.cache_allocation: NativeCacheAllocation | None = None
        self.cache_block_tables: torch.Tensor | None = None
        # This engine is inference-only.  Disable parameter gradients before
        # any eager qualification or ACLGraph capture so those initialization
        # forwards cannot retain autograd activations for the 32B TP3 target.
        self.model.requires_grad_(False)
        self.model.eval()
        if config.seed is not None:
            torch.manual_seed(config.seed)
            torch.npu.manual_seed_all(config.seed)
        self.graph_runner = NativeACLGraphRunner(
            self.model,
            enabled=(
                not config.enforce_eager and not (self.is_draft and config.draft_mode == PARD_PARALLEL_DRAFT_MODE)
            ),
            max_graph_tokens=max(
                512,
                config.max_num_seqs * max(1, self.gamma),
                (mixed_target_graph_max_tokens if self._mixed_target_graph_enabled else 0),
            ),
            max_graph_entries=config.max_aclgraph_entries,
            update_stream_priority=(
                envs.VLLM_ASCEND_PEARL_DRAFT_GRAPH_UPDATE_STREAM_PRIORITY
                if self.is_draft
                else envs.VLLM_ASCEND_PEARL_TARGET_GRAPH_UPDATE_STREAM_PRIORITY
            ),
        )
        self._fixed_full_window_host_correction_staging: _FixedFullWindowHostCorrectionStaging | None = None
        self._fixed_full_window_host_correction_staging_eligible_calls = 0
        self._fixed_full_window_host_correction_staging_calls = 0
        if config.spec_rhythm_linear_full_window and config.gamma == FIXED_FULL_WINDOW_HOST_STAGING_GAMMA:
            self._fixed_full_window_host_correction_staging = self._allocate_fixed_full_window_host_correction_staging()
        self.last_worker_decode_phase_seconds: dict[str, float] = {}
        self.last_worker_decode_profile_seconds: dict[str, float] = {}
        self.last_worker_decode_profile_detail_seconds: dict[str, float] = {}
        self.last_worker_decode_host_timeline: list[dict[str, int | float]] = []
        self.last_worker_profiled_decode_steps = 0
        self.last_worker_decode_counters: dict[str, int] = {}
        self._linear_draft_full_chain_bucket_calls: dict[int, int] = {}
        self._linear_draft_full_chain_logical_row_calls: dict[int, int] = {}
        self._linear_draft_full_chain_calls = 0
        self._linear_draft_full_chain_capture_replay_calls = 0
        self._linear_draft_full_chain_replay_calls = 0
        self._linear_draft_full_chain_eager_fallback_calls = 0
        self._linear_draft_full_chain_unclassified_calls = 0
        self._linear_draft_full_chain_logical_rows = 0
        self._linear_draft_full_chain_logical_tokens = 0
        self._linear_draft_full_chain_padded_rows = 0
        self._linear_draft_full_chain_padded_tokens = 0
        self._linear_draft_full_chain_padding_rows = 0
        self._linear_draft_full_chain_padding_tokens = 0
        self._linear_draft_stepwise_calls = 0
        self._linear_draft_stepwise_model_calls = 0
        self._linear_draft_fia_full_mask_builder = LinearDraftFIAFullMaskBuilder()
        self._linear_draft_fia_batched_mask_calls = 0
        self._linear_draft_fia_batched_mask_elements = 0
        self._linear_draft_fia_shared_request_table_calls = 0
        self._linear_draft_fia_persistent_staging_pool: dict[
            tuple[Any, ...],
            _LinearDraftFIAPersistentEnvelope,
        ] = {}
        self._linear_draft_fia_persistent_staging_hits = 0
        self._linear_draft_fia_persistent_staging_misses = 0
        self._pard_parallel_eager_forward_calls = 0
        self._pard_parallel_repair_rows = 0
        self._pard_parallel_mask_rows = 0
        # This must remain zero by construction: PARD is never allowed to
        # route through the legacy serial proposal loop.
        self._pard_parallel_serial_fallback_calls = 0
        self._packed_causal_leakage_probe_attempts = 0
        self._packed_causal_leakage_probe_passes = 0
        self._packed_causal_leakage_probe_failures = 0
        self._packed_causal_leakage_probe_hidden_mismatch_steps = 0
        self._packed_causal_leakage_probe_token_mismatch_steps = 0
        self._packed_causal_leakage_probe_restore_hidden_mismatch_steps = 0
        self._packed_causal_leakage_probe_restore_token_mismatch_steps = 0
        self._packed_causal_leakage_probe_max_abs_diff = 0.0
        self._packed_causal_leakage_probe_completed = False
        self._mixed_target_graph_qualified_buckets = 0
        self._stable_target_verify_graph_qualified = 0
        self._stable_target_verify_graph_capacity_map: dict[int, int] = {}
        self._stable_target_verify_graph_qualified_capacities: tuple[int, ...] = ()
        self._stable_target_verify_numerical_validation_attempts = 0
        self._stable_target_verify_numerical_validation_passes = 0
        self._stable_target_verify_numerical_validation_failures = 0
        self._stable_target_verify_numerical_restore_failures = 0
        self._last_mixed_target_graph_outcome: tuple[str, int, int] = (
            "disabled",
            0,
            0,
        )
        self._last_stable_target_verify_graph_outcome: tuple[str, int] = (
            "disabled",
            0,
        )
        self.greedy_verification_layouts: dict[
            tuple[int, tuple[int, ...]],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self.tokenizer = AutoTokenizer.from_pretrained(config.draft_model)
        # validate_model_pair above permits a target-only vocabulary suffix and
        # verifies that every draft token keeps the same target token ID.
        self.eos_token_ids = _normalize_eos_tokens(target_model_config.eos_token_id)
        self.gamma_profiles: dict[int, int] = {}
        if config.gamma == -1:
            self.gamma_profiles = self._profile_auto_gammas()
        dist.barrier()
        if config.enable_cpu_binding:
            try:
                bind_cpus(self.local_rank)
            except Exception as error:
                logger.warning("Bind cpus failed in PEARL rank%s: %s. Skip CPU binding.", self.local_rank, error)
        self.precompiled_decode_batch_sizes: frozenset[int] = frozenset()
        if config.precompile_decode_graphs or config.precompile_serial_draft_graphs:
            stable_task_barrier_uses_workload_capture = bool(
                envs.VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_BUCKET
                and envs.VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_COMMON_KV
                and envs.VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_STABLE_TASK_BARRIER
            )
            self._precompile_decode_graphs(
                include_target_graphs=config.precompile_decode_graphs,
                # A stable-task barrier binds the FIA host KV capacity into
                # the graph ABI.  Synthetic init states cannot know the real
                # workload-wide capacity, so capture/qualify this draft family
                # during untimed workload warmup instead.  Target graph
                # qualification remains independent and still runs here.
                include_draft_graphs=(
                    config.precompile_serial_draft_graphs and not stable_task_barrier_uses_workload_capture
                ),
            )
        dist.barrier()

    def graph_metrics(self) -> dict[str, Any]:
        """Return this worker's cumulative ACLGraph counters."""
        metrics: dict[str, int | float] = {
            "rank": self.rank,
            "is_draft_rank": int(self.is_draft),
            "aclgraph_entries": (
                len(self.graph_runner.entries)
                + len(self.graph_runner.draft_entries)
                + len(self.graph_runner.target_entries)
            ),
            "aclgraph_target_entries": len(self.graph_runner.target_entries),
            "aclgraph_captures": self.graph_runner.capture_count,
            "aclgraph_capture_attempts": self.graph_runner.capture_attempt_count,
            "aclgraph_replays": self.graph_runner.replay_count,
            "aclgraph_failed_captures": self.graph_runner.failed_capture_count,
            "aclgraph_capacity_fallbacks": self.graph_runner.capacity_fallback_count,
            "aclgraph_shape_fallbacks": self.graph_runner.shape_fallback_count,
            "aclgraph_task_update_replays": self.graph_runner.task_update_replay_count,
            "aclgraph_task_update_skipped_replays": self.graph_runner.task_update_skip_replay_count,
            "aclgraph_runtime_validation_replays": self.graph_runner.runtime_validation_replay_count,
            "aclgraph_expected_fia_batch_size": (
                self.graph_runner.expected_fia_batch_size
                if self.graph_runner.expected_fia_batch_size is not None
                else 0
            ),
            "aclgraph_last_fia_request_count": len(self.graph_runner.last_fia_shape),
            "aclgraph_last_fia_query_shape": ",".join(str(value) for value in self.graph_runner.last_fia_shape),
            "specslo_pard_eager_forward_calls": int(getattr(self, "_pard_parallel_eager_forward_calls", 0)),
            "specslo_pard_repair_rows": int(getattr(self, "_pard_parallel_repair_rows", 0)),
            "specslo_pard_mask_rows": int(getattr(self, "_pard_parallel_mask_rows", 0)),
            "specslo_pard_serial_fallback_calls": int(getattr(self, "_pard_parallel_serial_fallback_calls", 0)),
            "spec_rhythm_mixed_target_graph_enabled": int(getattr(self, "_mixed_target_graph_enabled", False)),
            "spec_rhythm_mixed_target_graph_qualified_buckets": int(
                getattr(self, "_mixed_target_graph_qualified_buckets", 0)
            ),
            "spec_rhythm_stable_target_verify_graph_enabled": int(
                getattr(self, "_mixed_target_graph_enabled", False) and not self.is_draft
            ),
            "spec_rhythm_stable_target_verify_graph_qualified": int(
                getattr(self, "_stable_target_verify_graph_qualified", 0)
            ),
            "spec_rhythm_stable_target_verify_graph_qualified_capacities": (
                len(
                    getattr(
                        self,
                        "_stable_target_verify_graph_qualified_capacities",
                        (),
                    )
                )
            ),
            "spec_rhythm_stable_target_verify_graph_qualified_capacity_mask": sum(
                1 << (value - 1)
                for value in getattr(
                    self,
                    "_stable_target_verify_graph_qualified_capacities",
                    (),
                )
            ),
            "spec_rhythm_stable_target_verify_graph_max_qualified_capacity": max(
                getattr(
                    self,
                    "_stable_target_verify_graph_qualified_capacities",
                    (0,),
                )
                or (0,)
            ),
            "spec_rhythm_stable_target_verify_graph_exact_routes": sum(
                request_count == capacity
                for request_count, capacity in getattr(
                    self,
                    "_stable_target_verify_graph_capacity_map",
                    {},
                ).items()
            ),
            "spec_rhythm_stable_target_verify_numerical_validation_attempts": int(
                getattr(
                    self,
                    "_stable_target_verify_numerical_validation_attempts",
                    0,
                )
            ),
            "spec_rhythm_stable_target_verify_numerical_validation_passes": int(
                getattr(
                    self,
                    "_stable_target_verify_numerical_validation_passes",
                    0,
                )
            ),
            "spec_rhythm_stable_target_verify_numerical_validation_failures": int(
                getattr(
                    self,
                    "_stable_target_verify_numerical_validation_failures",
                    0,
                )
            ),
            "spec_rhythm_stable_target_verify_numerical_restore_failures": int(
                getattr(
                    self,
                    "_stable_target_verify_numerical_restore_failures",
                    0,
                )
            ),
            "spec_rhythm_fixed_full_window_host_correction_staging_eligible_calls": int(
                getattr(
                    self,
                    "_fixed_full_window_host_correction_staging_eligible_calls",
                    0,
                )
            ),
            "spec_rhythm_fixed_full_window_host_correction_staging_calls": int(
                getattr(
                    self,
                    "_fixed_full_window_host_correction_staging_calls",
                    0,
                )
            ),
        }
        metrics.update(
            {f"aclgraph_{name}": value for name, value in self.graph_runner.graph_execution_metrics().items()}
        )
        metrics.update(
            {f"aclgraph_{name}": value for name, value in self.graph_runner.graph_qualification_status().items()}
        )
        tree_kv_runner = getattr(self, "tree_kv_graph_runner", None)
        if tree_kv_runner is not None:
            metrics.update(
                {
                    "tree_kv_aclgraph_entries": len(tree_kv_runner.entries),
                    "tree_kv_aclgraph_captures": tree_kv_runner.capture_count,
                    "tree_kv_aclgraph_capture_attempts": tree_kv_runner.capture_attempt_count,
                    "tree_kv_aclgraph_replays": tree_kv_runner.replay_count,
                    "tree_kv_aclgraph_failed_captures": tree_kv_runner.failed_capture_count,
                    "tree_kv_aclgraph_failed_validations": tree_kv_runner.failed_validation_count,
                    "tree_kv_aclgraph_capacity_fallbacks": tree_kv_runner.capacity_fallback_count,
                }
            )
        metrics.update(
            {f"worker_{phase}_seconds": seconds for phase, seconds in self.last_worker_decode_phase_seconds.items()}
        )
        metrics["worker_profiled_decode_steps"] = self.last_worker_profiled_decode_steps
        metrics.update(
            {
                f"worker_profile_{phase}_seconds": seconds
                for phase, seconds in self.last_worker_decode_profile_seconds.items()
            }
        )
        metrics.update(
            {
                f"worker_profile_detail_{phase}_seconds": seconds
                for phase, seconds in self.last_worker_decode_profile_detail_seconds.items()
            }
        )
        metrics.update({f"worker_{name}": value for name, value in self.last_worker_decode_counters.items()})
        metrics.update(
            {
                f"spec_rhythm_linear_draft_full_chain_bucket_{bucket}_calls": calls
                for bucket, calls in sorted(getattr(self, "_linear_draft_full_chain_bucket_calls", {}).items())
            }
        )
        metrics.update(
            {
                f"spec_rhythm_linear_draft_full_chain_logical_rows_{rows}_calls": calls
                for rows, calls in sorted(
                    getattr(
                        self,
                        "_linear_draft_full_chain_logical_row_calls",
                        {},
                    ).items()
                )
            }
        )
        full_chain_metrics = {
            "calls": "_linear_draft_full_chain_calls",
            "capture_replay_calls": "_linear_draft_full_chain_capture_replay_calls",
            "replay_calls": "_linear_draft_full_chain_replay_calls",
            "eager_fallback_calls": "_linear_draft_full_chain_eager_fallback_calls",
            "unclassified_calls": "_linear_draft_full_chain_unclassified_calls",
            "logical_rows": "_linear_draft_full_chain_logical_rows",
            "logical_tokens": "_linear_draft_full_chain_logical_tokens",
            "padded_rows": "_linear_draft_full_chain_padded_rows",
            "padded_tokens": "_linear_draft_full_chain_padded_tokens",
            "padding_rows": "_linear_draft_full_chain_padding_rows",
            "padding_tokens": "_linear_draft_full_chain_padding_tokens",
        }
        metrics.update(
            {
                f"spec_rhythm_linear_draft_full_chain_{name}": getattr(self, attribute, 0)
                for name, attribute in full_chain_metrics.items()
            }
        )
        metrics["spec_rhythm_linear_draft_stepwise_calls"] = getattr(self, "_linear_draft_stepwise_calls", 0)
        metrics["spec_rhythm_linear_draft_stepwise_model_calls"] = getattr(
            self, "_linear_draft_stepwise_model_calls", 0
        )
        metrics["spec_rhythm_linear_draft_fia_batched_masks_enabled"] = int(
            envs.VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_BATCHED_MASKS
        )
        metrics["spec_rhythm_linear_draft_fia_batched_mask_calls"] = getattr(
            self,
            "_linear_draft_fia_batched_mask_calls",
            0,
        )
        metrics["spec_rhythm_linear_draft_fia_batched_mask_elements"] = getattr(
            self,
            "_linear_draft_fia_batched_mask_elements",
            0,
        )
        metrics["spec_rhythm_linear_draft_fia_shared_request_table_calls"] = getattr(
            self,
            "_linear_draft_fia_shared_request_table_calls",
            0,
        )
        metrics["spec_rhythm_linear_draft_fia_persistent_staging_enabled"] = int(
            envs.VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_PERSISTENT_STAGING
        )
        metrics["spec_rhythm_linear_draft_fia_persistent_staging_entries"] = len(
            getattr(
                self,
                "_linear_draft_fia_persistent_staging_pool",
                {},
            )
        )
        metrics["spec_rhythm_linear_draft_fia_persistent_staging_hits"] = getattr(
            self,
            "_linear_draft_fia_persistent_staging_hits",
            0,
        )
        metrics["spec_rhythm_linear_draft_fia_persistent_staging_misses"] = getattr(
            self,
            "_linear_draft_fia_persistent_staging_misses",
            0,
        )
        packed_causal_probe_metrics = {
            "attempts": "_packed_causal_leakage_probe_attempts",
            "passes": "_packed_causal_leakage_probe_passes",
            "failures": "_packed_causal_leakage_probe_failures",
            "hidden_mismatch_steps": ("_packed_causal_leakage_probe_hidden_mismatch_steps"),
            "token_mismatch_steps": ("_packed_causal_leakage_probe_token_mismatch_steps"),
            "restore_hidden_mismatch_steps": ("_packed_causal_leakage_probe_restore_hidden_mismatch_steps"),
            "restore_token_mismatch_steps": ("_packed_causal_leakage_probe_restore_token_mismatch_steps"),
            "max_abs_diff": "_packed_causal_leakage_probe_max_abs_diff",
        }
        metrics.update(
            {
                f"spec_rhythm_packed_causal_leakage_probe_{name}": getattr(
                    self,
                    attribute,
                    0,
                )
                for name, attribute in packed_causal_probe_metrics.items()
            }
        )
        metrics["worker_host_timeline"] = list(self.last_worker_decode_host_timeline)
        return metrics

    def seal_graph_cache(self) -> dict[str, Any]:
        """Seal the changed-input-qualified graph inventory for measurement."""

        self.graph_runner.seal_graph_cache()
        return self.graph_metrics()

    def unseal_graph_cache(self) -> dict[str, Any]:
        """Reopen lazy graph discovery before an untimed qualification run."""

        self.graph_runner.unseal_graph_cache()
        return self.graph_metrics()

    def prune_unvalidated_graph_entries(self) -> dict[str, Any]:
        """Release cold-only, unqualified graphs before another warmup pass."""

        self.graph_runner.prune_unvalidated_graph_entries()
        return self.graph_metrics()

    def configure_decode_profiling(
        self,
        profile_decode_steps: int,
        stop_after_profiled_decode_steps: bool = False,
        profile_host_decode_steps: int = 0,
    ) -> None:
        """Change decode profiling between requests without reloading the models."""
        if profile_decode_steps < 0:
            raise ValueError("PEARL profile_decode_steps must be non-negative.")
        if profile_host_decode_steps < 0:
            raise ValueError("PEARL profile_host_decode_steps must be non-negative.")
        if stop_after_profiled_decode_steps and profile_decode_steps == 0:
            raise ValueError("PEARL profiling-only execution requires profile_decode_steps to be positive.")
        self.config = replace(
            self.config,
            profile_decode_steps=profile_decode_steps,
            profile_host_decode_steps=profile_host_decode_steps,
            stop_after_profiled_decode_steps=stop_after_profiled_decode_steps,
        )

    def _resolve_num_cache_blocks(self) -> int:
        if self.config.num_kvcache_blocks > 0:
            return self.config.num_kvcache_blocks
        attention = self.model.layers[0].self_attn
        bytes_per_element = attention.qkv_proj.weight.element_size()
        bytes_per_block = (
            2
            * self.config.kvcache_block_size
            * attention.num_kv_heads
            * attention.head_dim
            * bytes_per_element
            * len(self.model.layers)
        )
        free_memory, total_memory = torch.npu.mem_get_info(self.local_rank)
        used_memory = total_memory - free_memory
        cache_budget = max(0, int(total_memory * self.config.gpu_memory_utilization) - used_memory)
        max_required_blocks = (
            (self.config.max_model_len + self.config.kvcache_block_size - 1)
            // self.config.kvcache_block_size
            * self._cache_storage_sequence_capacity
        )
        num_blocks = min(max_required_blocks, cache_budget // bytes_per_block)
        if num_blocks < self._cache_storage_sequence_capacity:
            raise MemoryError(
                "PEARL cannot reserve one KV cache page per configured sequence; "
                "lower max_num_seqs or increase available NPU memory."
            )
        return num_blocks

    @property
    def _cache_sequence_capacity(self) -> int:
        return self.config.max_num_queued_seqs or self.config.max_num_seqs

    @property
    def _cache_storage_sequence_capacity(self) -> int:
        """Cache rows, including target-only mixed-graph scratch ownership."""

        return self._cache_sequence_capacity + (
            MIXED_TARGET_SCRATCH_ROWS if getattr(self, "_mixed_target_graph_enabled", False) else 0
        )

    @property
    def _mixed_target_scratch_sequence_ids(self) -> tuple[int, ...]:
        if not getattr(self, "_mixed_target_graph_enabled", False):
            return ()
        start = self._cache_sequence_capacity
        return tuple(range(start, start + MIXED_TARGET_SCRATCH_ROWS))

    def _allocate_cache(
        self,
        prompts: list[list[int]],
        *,
        enable_prefix_caching: bool,
        reserve_sequence_capacity: bool = False,
    ) -> NativeCacheAllocation:
        # Persistent FIA staging contains page-table-shaped buffers.  A new
        # logical cache allocation may retain the same dimensions while
        # assigning completely different physical pages, so make its lifetime
        # explicit instead of relying on an accidental shape-only match.
        getattr(self, "_linear_draft_fia_persistent_staging_pool", {}).clear()
        allocation = self.prefix_cache.allocate(
            prompts,
            enable_prefix_caching=enable_prefix_caching,
            sequence_capacity=(self._cache_storage_sequence_capacity if reserve_sequence_capacity else None),
        )
        self.cache_allocation = allocation
        self.cache_block_tables = torch.tensor(
            allocation.block_tables,
            dtype=torch.int32,
            device=self.device,
        )
        return allocation

    def _activate_cache_sequence(
        self,
        sequence_id: int,
        prompt: list[int],
        *,
        enable_prefix_caching: bool,
    ) -> None:
        """Activate a pre-reserved KV page-table row for online admission."""
        if self.cache_allocation is None or self.cache_block_tables is None:
            raise RuntimeError("Allocate the live PEARL cache before admitting requests.")
        cached_tokens = self.prefix_cache.activate_sequence(
            sequence_id,
            prompt,
            enable_prefix_caching=enable_prefix_caching,
        )
        self.cache_allocation.num_cached_tokens[sequence_id] = cached_tokens
        self.cache_block_tables[sequence_id].copy_(
            torch.tensor(
                self.cache_allocation.block_tables[sequence_id],
                dtype=torch.int32,
                device=self.device,
            )
        )

    def _ensure_cache_capacity(
        self,
        sequence_ids: list[int],
        positions: list[int],
    ) -> None:
        if self.cache_block_tables is None:
            raise RuntimeError("Allocate the native PEARL KV cache before extending it.")
        updates = self.prefix_cache.ensure_capacity(sequence_ids, positions)
        if not updates:
            return
        update_sequences, update_logical_blocks, update_block_ids = zip(*updates)
        self.cache_block_tables[
            torch.tensor(update_sequences, dtype=torch.long, device=self.device),
            torch.tensor(update_logical_blocks, dtype=torch.long, device=self.device),
        ] = torch.tensor(update_block_ids, dtype=torch.int32, device=self.device)

    def _release_cache(self) -> None:
        # Physical pages and ACLGraph entries outlive an allocation. Do not
        # clear sticky numerical faults here: a freed page can still contain
        # a NaN, and graph replay must keep updating the same flag storage.
        self.prefix_cache.release()
        getattr(self, "_linear_draft_fia_persistent_staging_pool", {}).clear()
        self.cache_allocation = None
        self.cache_block_tables = None

    def _release_cache_sequence(self, sequence_id: int) -> int:
        """Release one completed online row on host and device page tables."""

        if self.cache_allocation is None or self.cache_block_tables is None:
            raise RuntimeError("Allocate the native PEARL KV cache before releasing a sequence.")
        released = self.prefix_cache.release_sequence(sequence_id)
        self.cache_block_tables[sequence_id].fill_(-1)
        return released

    def _cache_slot_mapping(
        self,
        sequence_ids: list[int],
        positions: list[int],
    ) -> list[int]:
        if self.cache_allocation is None:
            raise RuntimeError("Allocate the native PEARL KV cache before building slot mappings.")
        block_size = self.config.kvcache_block_size
        return [
            self.cache_allocation.block_tables[sequence_id][position // block_size] * block_size + position % block_size
            for sequence_id, position in zip(sequence_ids, positions)
        ]

    @torch.inference_mode()
    def generate(
        self,
        prompt_token_ids: list[int],
        sampling_params: NativeSamplingParams | None = None,
    ) -> dict[str, Any] | None:
        results = self.generate_batch([prompt_token_ids], sampling_params)
        return results[0] if results is not None else None

    @torch.inference_mode()
    def generate_batch(
        self,
        prompt_token_ids,
        sampling_params=None,
        *,
        max_rounds=None,
        on_token_commit: Callable[[dict[str, Any]], None] | None = None,
        request_admission_callback: Callable[[bool], tuple[bool, Sequence[tuple[list[int], NativeSamplingParams]]]]
        | None = None,
    ):
        self._token_commit_callback = on_token_commit
        self._request_admission_callback = request_admission_callback
        self._token_commit_error = None
        self._stream_delivered_counts = {}
        self._stream_started = time.perf_counter()
        try:
            result = self._generate_batch_impl(prompt_token_ids, sampling_params, max_rounds=max_rounds)
            if self._token_commit_error is not None:
                raise RuntimeError(
                    "Committed-token callback failed after collective drain"
                ) from self._token_commit_error
            return result
        finally:
            self._token_commit_callback = None
            self._request_admission_callback = None
            self._token_commit_error = None
            profiler = getattr(self, "_active_tree_profiler", None)
            if profiler is not None:
                profiler.stop()
                self._active_tree_profiler = None

    @torch.inference_mode()
    def _generate_batch_impl(
        self,
        prompt_token_ids: list[list[int]],
        sampling_params: NativeSamplingParams | Sequence[NativeSamplingParams] | None = None,
        *,
        max_rounds: int | None = None,
    ) -> list[dict[str, Any]] | None:
        """Generate a packed batch, optionally scheduling a prefilled request queue."""
        live_admission = getattr(self, "_request_admission_callback", None) is not None
        if live_admission and not (
            self.config.enable_spec_rhythm
            and self.config.enable_continuous_batching
            and self.config.enable_preemptive_scheduling
            and (self.config.spec_rhythm_tree_width > 1 or self.config.spec_rhythm_tree_depth > 1)
        ):
            raise ValueError("Live PEARL admission requires the complete SpecRhythm tree scheduler.")
        continuous_batching = (
            self.config.enable_continuous_batching
            and max_rounds is None
            and (len(prompt_token_ids) > self.config.max_num_seqs or live_admission)
        )
        if not prompt_token_ids or (len(prompt_token_ids) > self.config.max_num_seqs and not continuous_batching):
            raise ValueError("PEARL batch size must fit max_num_seqs unless continuous batching is enabled.")
        if any(len(prompt) > self.config.max_num_batched_tokens for prompt in prompt_token_ids):
            raise ValueError("A PEARL prompt exceeds max_num_batched_tokens.")
        if continuous_batching and len(prompt_token_ids) > self._cache_sequence_capacity:
            raise ValueError("PEARL continuous requests exceed max_num_queued_seqs.")
        request_params = _normalize_sampling_params(len(prompt_token_ids), sampling_params, self.config.max_tokens)
        tree_mode = self.config.enable_spec_rhythm and (
            self.config.spec_rhythm_tree_width > 1 or self.config.spec_rhythm_tree_depth > 1
        )
        if not tree_mode and any(params.temperature > 0 and params.draft_temperature > 0 for params in request_params):
            raise ValueError(
                "Simultaneous stochastic target and draft sampling requires the "
                "SpecRhythm tree verifier; the linear verifier only has an exact "
                "correction for deterministic draft proposals."
            )
        if max_rounds is not None and max_rounds <= 0:
            raise ValueError("PEARL max_rounds must be positive when supplied.")
        self.gamma = self._auto_select_gamma(prompt_token_ids) if self.config.gamma == -1 else self.config.gamma
        for prompt, params in zip(prompt_token_ids, request_params):
            if not prompt:
                raise ValueError("PEARL generation requires non-empty prompts.")
            completion_capacity = params.max_tokens if max_rounds is None else 1 + (max_rounds + 1) * self.gamma
            tail_capacity = self.gamma
            if self.config.enable_spec_rhythm:
                tail_capacity = max(
                    tail_capacity, self.config.spec_rhythm_tree_width * self.config.spec_rhythm_tree_depth + 1
                )
            if len(prompt) + completion_capacity + tail_capacity > self.config.max_model_len:
                raise ValueError("Prompt plus PEARL completion exceeds max_model_len.")

        all_tokens = [[int(token_id) for token_id in prompt] for prompt in prompt_token_ids]
        initial_batch_size = (
            self.config.max_num_seqs if live_admission else min(len(all_tokens), self.config.max_num_seqs)
        )
        online_prefill = self.config.enable_spec_rhythm and self.config.spec_rhythm_online_prefill
        # Online SpecRhythm admission owns prefill timing.  Do not prefill the
        # initial bucket before its arrival timestamps: doing so consumes the
        # target stream on requests that are not yet eligible and makes the
        # measured first-arrival SLO depend on an untimed future-request pass.
        # The controller will issue one packed prefill as soon as a ready set
        # is admitted.  Static and non-online continuous modes retain their
        # original eager prefill behavior.
        prefill_tokens = (
            [] if online_prefill else (all_tokens if continuous_batching else all_tokens[:initial_batch_size])
        )
        prefill_params = (
            [] if online_prefill else (request_params if continuous_batching else request_params[:initial_batch_size])
        )
        prefill_chunk_size = self.config.prefill_chunk_size or self.config.max_num_seqs
        if any(
            sum(len(prompt) for prompt in prefill_tokens[start : start + prefill_chunk_size])
            > self.config.max_num_batched_tokens
            for start in range(0, len(prefill_tokens), prefill_chunk_size)
        ):
            raise ValueError("A PEARL prefill chunk exceeds max_num_batched_tokens.")
        self.graph_runner.set_expected_fia_batch_size(None if live_admission else initial_batch_size)
        # The linear online scheduler can recycle rows as requests finish.
        # Keep the tree scheduler on its established eager allocation until
        # its abort/eager-proposal cache lifecycle has the same per-row
        # release contract; otherwise initial tree rows would have an empty
        # device page table.
        lazy_online_cache = online_prefill and not tree_mode
        self._allocate_cache(
            [] if lazy_online_cache else all_tokens,
            enable_prefix_caching=self.config.enable_prefix_caching,
            reserve_sequence_capacity=(live_admission or online_prefill),
        )
        draft_states = [
            PearlPipelineState(
                tokens,
                len(tokens),
                temperature=params.temperature,
                draft_temperature=params.draft_temperature,
                top_p=params.top_p,
                top_k=params.top_k,
                draft_top_p=params.draft_top_p,
                draft_top_k=params.draft_top_k,
                max_tokens=params.max_tokens,
                ignore_eos=params.ignore_eos,
                slo_tpot_ms=params.slo_tpot_ms,
                slo_class=params.slo_class,
                request_id=params.request_id,
                arrival_ts=params.arrival_ts,
            )
            for tokens, params in zip(
                all_tokens,
                request_params,
            )
        ]
        target_states = [state.clone() for state in draft_states]
        torch.npu.synchronize()
        prefill_started = time.perf_counter()
        target_tokens: list[int] = []
        for start in range(0, len(prefill_tokens), prefill_chunk_size):
            end = start + prefill_chunk_size
            target_tokens.extend(
                self._prefill_and_sample_target_batch(
                    prefill_tokens[start:end],
                    target_states[start:end],
                    list(range(start, min(end, len(prefill_tokens)))),
                )
            )
        torch.npu.synchronize()
        prefill_elapsed = time.perf_counter() - prefill_started
        for draft_state, target_state, target_token in zip(
            draft_states[: len(prefill_tokens)],
            target_states[: len(prefill_tokens)],
            target_tokens,
        ):
            draft_state.token_ids.append(target_token)
            target_state.token_ids.append(target_token)
            assert draft_state.committed_length is not None and target_state.committed_length is not None
            draft_state.committed_length += 1
            target_state.committed_length += 1

        # A continuous request set may contain more rows than the physical
        # decode bucket.  Pre-capturing only ``initial_batch_size`` rows here
        # can write graph state into a shared KV allocation that also owns the
        # later rows; let the decode runner lazily capture each actual shape
        # instead.  Static batches retain the eager capture fast path.
        if not online_prefill and not continuous_batching and not tree_mode:
            self._capture_decode_graphs(
                prefill_tokens[:initial_batch_size],
                target_tokens[:initial_batch_size],
            )
        torch.npu.synchronize()
        started = time.perf_counter()
        spec_rhythm_has_scheduler_constraints = any(
            params.slo_tpot_ms is not None
            or params.slo_class is not None
            or params.arrival_ts is not None
            or params.spec_rhythm_max_gamma is not None
            for params in prefill_params
        )
        # With no SLO or per-request budget, SpecRhythm has no scheduling
        # decision to make. Reuse PEARL's packed fast path so enabling the
        # policy does not add a control-plane round trip to ordinary serving.
        spec_rhythm_needs_control_plane = self.config.enable_spec_rhythm and (
            spec_rhythm_has_scheduler_constraints
            or self.config.spec_rhythm_linear_full_window
            or self.config.spec_rhythm_min_gamma != self.gamma
            or self.config.spec_rhythm_max_eager_tokens != 0
            or self.config.spec_rhythm_eager_reserve_tokens != 0
            or self.config.spec_rhythm_roofline is not None
            or self.config.spec_rhythm_draft_token_budget is not None
            or self.config.spec_rhythm_verification_budget is not None
            or getattr(self, "_token_commit_callback", None) is not None
            or self.config.spec_rhythm_tree_width > 1
            or self.config.spec_rhythm_tree_depth > 1
            or online_prefill
        )
        if spec_rhythm_needs_control_plane:
            return self._generate_spec_rhythm_decode(
                draft_states=draft_states,
                target_states=target_states,
                request_params=(request_params if online_prefill else prefill_params),
                initial_batch_size=initial_batch_size,
                continuous_batching=continuous_batching,
                prefill_elapsed=prefill_elapsed,
                started=started,
                max_rounds=max_rounds,
                # The non-online path prefills every queued request before the
                # controller starts.  Mark those requests as prefetched too;
                # otherwise a late admission repeats target prefill and turns
                # refill into a second full prompt pass (especially visible
                # when larger gamma values make requests miss their arrival
                # windows).  Online prefill only marks its initial bucket.
                prefilled_indices=set(range(len(prefill_tokens))),
            )
        npu_profiler = None
        npu_profile_dir = envs.VLLM_ASCEND_PEARL_NPU_PROFILE_DIR
        if npu_profile_dir:
            configured_rank = envs.VLLM_ASCEND_PEARL_NPU_PROFILE_RANK
            npu_profile_rank = self.topology.target_leader_rank if configured_rank is None else configured_rank
            if npu_profile_rank != -1 and not 0 <= npu_profile_rank < self.topology.world_size:
                raise ValueError("VLLM_ASCEND_PEARL_NPU_PROFILE_RANK must be -1 or identify a PEARL worker.")
        if npu_profile_dir and (npu_profile_rank == -1 or self.rank == npu_profile_rank):
            import torch_npu

            npu_profiler = torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=3, repeat=1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    npu_profile_dir,
                    worker_name=(f"pearl-{'draft' if self.is_draft else 'target'}-rank-{self.rank}"),
                ),
                record_shapes=True,
            )
            npu_profiler.start()
        round_count = 0
        completed_states: dict[int, PearlPipelineState] = {}
        completed_request_states: dict[int, PearlPipelineState] = {}
        active_request_indices = list(range(initial_batch_size))
        next_request_index = initial_batch_size
        recent_scheduling_token_gains: list[deque[int]] = [
            deque(maxlen=PREEMPTIVE_SCHEDULING_RECENT_ROUNDS) for _ in all_tokens
        ]
        decode_phase_seconds = {
            "draft": 0.0,
            "target": 0.0,
            "exchange": 0.0,
            "verify": 0.0,
            "broadcast": 0.0,
            "state_update": 0.0,
            "refill": 0.0,
        }
        decode_profile_seconds = {
            "draft_compute": 0.0,
            "draft_to_target_communication": 0.0,
            "target_compute": 0.0,
            "target_verdict": 0.0,
            "target_to_draft_communication": 0.0,
            "wait_sync": 0.0,
            "state_update": 0.0,
        }
        profiled_decode_steps = 0
        target_verification_tokens = 0
        target_model_tokens = 0
        scheduled_sequence_slots = 0
        useful_sequence_slots = 0
        verification_shape_rounds: dict[tuple[int, int], int] = {}
        while True:
            local_states = draft_states if self.is_draft else target_states
            if continuous_batching:
                for request_index, state in completed_request_states.items():
                    local_states[request_index] = state.clone()
                candidate_indices = (
                    range(len(local_states)) if self.config.enable_preemptive_scheduling else active_request_indices
                )
                finished_requests = [
                    request_index
                    for request_index in candidate_indices
                    if request_index not in completed_request_states
                    and _finished(local_states[request_index], self.eos_token_ids)
                ]
                for request_index in finished_requests:
                    local_states[request_index].finished_decode_elapsed_ms = (time.perf_counter() - started) * 1000.0
                    completed_request_states[request_index] = local_states[request_index].clone()
                if self.config.enable_preemptive_scheduling:
                    unfinished_request_indices = [
                        request_index
                        for request_index in range(len(local_states))
                        if request_index not in completed_request_states
                    ]
                    active_request_indices = _select_preemptive_continuous_indices(
                        local_states,
                        unfinished_request_indices,
                        initial_batch_size,
                        recent_scheduling_token_gains,
                        decode_elapsed_ms=(time.perf_counter() - started) * 1000.0,
                        slo_aware=self.config.enable_spec_rhythm,
                    )
                elif finished_requests:
                    finished_set = set(finished_requests)
                    active_request_indices = [
                        request_index for request_index in active_request_indices if request_index not in finished_set
                    ]
                    replacement_count = min(
                        len(finished_requests),
                        len(all_tokens) - next_request_index,
                    )
                    active_request_indices.extend(range(next_request_index, next_request_index + replacement_count))
                    next_request_index += replacement_count
                if len(completed_request_states) == len(all_tokens):
                    break

            if max_rounds is None:
                if self.config.pad_finished_requests and not continuous_batching:
                    _restore_completed_states(
                        local_states,
                        completed_states,
                        self.eos_token_ids,
                    )
                if continuous_batching:
                    active_indices = _continuous_bucket_indices(
                        active_request_indices,
                        completed_request_states,
                        initial_batch_size,
                    )
                else:
                    unfinished_indices = [
                        index for index, state in enumerate(local_states) if not _finished(state, self.eos_token_ids)
                    ]
                    active_indices = (
                        list(range(len(local_states)))
                        if unfinished_indices and self.config.pad_finished_requests
                        else unfinished_indices
                    )
            elif round_count < max_rounds:
                active_indices = list(range(len(local_states)))
            else:
                active_indices = []
            if not active_indices:
                break
            if continuous_batching:
                self.graph_runner.set_expected_fia_batch_size(len(active_indices))
                scheduled_sequence_slots += len(active_indices)
                useful_sequence_slots += sum(
                    request_index not in completed_request_states for request_index in active_indices
                )
            # Request order is not semantically observable, so canonicalize it
            # by verification width. This reduces full-batch target FIA shapes
            # from every binary 1/gamma permutation to one shape per count of
            # pre-verify requests, allowing the bounded graph cache to cover
            # the hot path without any token or KV padding.
            active_indices = _canonical_active_indices(local_states, active_indices)
            pre_verify = [local_states[index].pre_verify for index in active_indices]
            verification_sizes = [1 if value else self.gamma for value in pre_verify]
            verification_shape = (len(active_indices), sum(not value for value in pre_verify))
            verification_shape_rounds[verification_shape] = verification_shape_rounds.get(verification_shape, 0) + 1
            profile_this_round = round_count < self.config.profile_decode_steps
            if profile_this_round:
                torch.npu.synchronize()
            phase_started = time.perf_counter()
            verification_tensor, next_window_tensor = self._draft_round_device_batch(
                draft_states,
                active_indices,
            )
            if profile_this_round:
                torch.npu.synchronize()
            phase_elapsed = time.perf_counter() - phase_started
            decode_phase_seconds["draft"] += phase_elapsed
            if profile_this_round and self.is_draft:
                decode_profile_seconds["draft_compute"] += phase_elapsed
            phase_started = time.perf_counter()
            target_token_windows, target_logits = self._target_round_outputs_batch(
                target_states,
                active_indices,
            )
            if profile_this_round:
                torch.npu.synchronize()
            phase_elapsed = time.perf_counter() - phase_started
            decode_phase_seconds["target"] += phase_elapsed
            if profile_this_round and not self.is_draft:
                decode_profile_seconds["target_compute"] += phase_elapsed
            target_verification_tokens += sum(verification_sizes)
            target_model_tokens += sum(
                _bucket_target_verification_widths(
                    pre_verify,
                    self.gamma,
                    self.config.target_verification_graph_buckets,
                    self.config.target_verification_graph_post_counts,
                )[0]
            )
            if profile_this_round and self.groups.is_verification_worker:
                wait_started = time.perf_counter()
                dist.barrier(group=self.groups.verification_group)
                torch.npu.synchronize()
                decode_profile_seconds["wait_sync"] += time.perf_counter() - wait_started
            phase_started = time.perf_counter()
            draft_message = self._exchange_draft_device_windows(
                verification_tensor,
                next_window_tensor,
                verification_sizes,
            )
            if profile_this_round and self.groups.is_verification_worker:
                torch.npu.synchronize()
            phase_elapsed = time.perf_counter() - phase_started
            decode_phase_seconds["exchange"] += phase_elapsed
            if profile_this_round and self.groups.is_verification_worker:
                decode_profile_seconds["draft_to_target_communication"] += phase_elapsed
            temperatures = [target_states[index].temperature for index in active_indices]
            phase_started = time.perf_counter()
            verdict = self._verify_target_tokens_batch(
                target_token_windows,
                target_logits,
                draft_message,
                verification_sizes,
                temperatures,
                top_ps=[target_states[index].top_p for index in active_indices],
                top_ks=[target_states[index].top_k for index in active_indices],
            )
            if profile_this_round:
                torch.npu.synchronize()
            phase_elapsed = time.perf_counter() - phase_started
            decode_phase_seconds["verify"] += phase_elapsed
            if profile_this_round and not self.is_draft:
                decode_profile_seconds["target_verdict"] += phase_elapsed
            replicated_target_verdict = all(temperature == 0 for temperature in temperatures)
            if profile_this_round:
                if replicated_target_verdict:
                    participates_in_correction = self.rank in self.topology.correction_ranks
                    correction_group = self.groups.correction_group
                else:
                    participates_in_correction = True
                    correction_group = None
                if participates_in_correction:
                    wait_started = time.perf_counter()
                    dist.barrier(group=correction_group)
                    torch.npu.synchronize()
                    decode_profile_seconds["wait_sync"] += time.perf_counter() - wait_started
            phase_started = time.perf_counter()
            accepted, corrections, synchronized_next = self._broadcast_device_round_result(
                verdict,
                draft_message,
                sum(verification_sizes),
                len(active_indices),
                next_window_tensor,
                replicated_target_verdict=replicated_target_verdict,
                profile_phase_seconds=decode_profile_seconds if profile_this_round else None,
            )
            bonuses = getattr(
                self,
                "_last_device_round_bonus_tokens",
                [None] * len(active_indices),
            )
            if any(value is not None for value in bonuses):
                raise RuntimeError("The legacy PEARL state machine received an unexpected bonus token.")
            decode_phase_seconds["broadcast"] += time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            committed_lengths_before_update = [
                local_states[sequence_index].committed_length for sequence_index in active_indices
            ]
            for batch_index, sequence_index in enumerate(active_indices):
                if self.is_draft:
                    draft_states[sequence_index].token_ids.extend(synchronized_next[batch_index])
                    draft_states[sequence_index].apply_draft_verification(
                        gamma=self.gamma,
                        accepted=accepted[batch_index],
                        correction_token_id=corrections[batch_index],
                        next_round_token_ids=synchronized_next[batch_index],
                    )
                else:
                    target_states[sequence_index].apply_target_verification(
                        gamma=self.gamma,
                        accepted=accepted[batch_index],
                        correction_token_id=corrections[batch_index],
                        next_round_token_ids=synchronized_next[batch_index],
                    )
                if continuous_batching and sequence_index not in completed_request_states:
                    committed_before = committed_lengths_before_update[batch_index]
                    committed_after = local_states[sequence_index].committed_length
                    assert committed_before is not None and committed_after is not None
                    recent_scheduling_token_gains[sequence_index].append(committed_after - committed_before)
            phase_elapsed = time.perf_counter() - phase_started
            decode_phase_seconds["state_update"] += phase_elapsed
            if profile_this_round:
                decode_profile_seconds["state_update"] += phase_elapsed
                profiled_decode_steps += 1
            round_count += 1
            if npu_profiler is not None:
                npu_profiler.step()
            if (
                self.config.stop_after_profiled_decode_steps
                and profiled_decode_steps >= self.config.profile_decode_steps
            ):
                break

        if npu_profiler is not None:
            npu_profiler.stop()
        torch.npu.synchronize()
        decode_elapsed = time.perf_counter() - started
        self.last_worker_decode_phase_seconds = dict(decode_phase_seconds)
        self.last_worker_decode_profile_seconds = dict(decode_profile_seconds)
        self.last_worker_decode_profile_detail_seconds = {}
        self.last_worker_profiled_decode_steps = profiled_decode_steps
        self.last_worker_decode_counters = {
            "target_verification_tokens": target_verification_tokens,
            "target_model_tokens": target_model_tokens,
            "target_padding_tokens": target_model_tokens - target_verification_tokens,
            "scheduled_sequence_slots": scheduled_sequence_slots,
            "useful_sequence_slots": useful_sequence_slots,
            "padded_sequence_slots": scheduled_sequence_slots - useful_sequence_slots,
            **{
                f"verification_shape_b{batch_size}_post{post_verify_count}_rounds": rounds
                for (batch_size, post_verify_count), rounds in sorted(verification_shape_rounds.items())
            },
        }
        elapsed = prefill_elapsed + decode_elapsed
        results = []
        if continuous_batching:
            result_states = [
                (
                    state,
                    self.cache_allocation.num_cached_tokens[request_index],
                )
                for request_index, state in enumerate(_continuous_result_states(local_states, completed_request_states))
            ]
        else:
            result_states = []
            for sequence_index, state in enumerate(target_states):
                if self.config.pad_finished_requests and sequence_index in completed_states:
                    state = completed_states[sequence_index]
                result_states.append((state, self.cache_allocation.num_cached_tokens[sequence_index]))
        for state, num_cached_tokens in result_states:
            acceptance_lengths = state.acceptance_lengths
            completion_token_ids = _truncate_completion(
                state.committed_completion_token_ids,
                self.eos_token_ids,
                state.max_tokens if max_rounds is None else len(state.committed_completion_token_ids),
                state.ignore_eos if max_rounds is None else True,
            )
            request_decode_elapsed_ms = (
                state.finished_decode_elapsed_ms
                if state.finished_decode_elapsed_ms is not None
                else decode_elapsed * 1000.0
            )
            observed_tpot_ms = request_decode_elapsed_ms / max(1, len(completion_token_ids))
            slo_attained = None if state.slo_tpot_ms is None else observed_tpot_ms <= state.slo_tpot_ms
            finish_reason = (
                "abort" if state.aborted else "length" if len(completion_token_ids) >= state.max_tokens else "stop"
            )
            results.append(
                {
                    "completion_token_ids": completion_token_ids,
                    "request_id": state.request_id,
                    "arrival_ts": state.arrival_ts,
                    "accepted_draft_tokens": state.accepted_draft_tokens,
                    "verified_draft_tokens": state.verified_draft_tokens,
                    "verification_rounds": state.verification_rounds,
                    "acceptance_rate": (
                        state.accepted_draft_tokens / state.verified_draft_tokens
                        if state.verified_draft_tokens
                        else 0.0
                    ),
                    "num_acc_tokens": acceptance_lengths,
                    "mean_accept_tokens": (
                        sum(acceptance_lengths) / len(acceptance_lengths) if acceptance_lengths else 0.0
                    ),
                    "temperature": state.temperature,
                    "draft_temperature": state.draft_temperature,
                    "top_p": state.top_p,
                    "top_k": state.top_k,
                    "draft_top_p": state.draft_top_p,
                    "draft_top_k": state.draft_top_k,
                    "max_tokens": state.max_tokens,
                    "ignore_eos": state.ignore_eos,
                    "slo_tpot_ms": state.slo_tpot_ms,
                    "slo_class": state.slo_class,
                    "finish_reason": finish_reason,
                    "observed_tpot_ms": observed_tpot_ms,
                    "slo_attained": slo_attained,
                    "slo_goodput_tokens": (len(completion_token_ids) if slo_attained is not False else 0),
                    "round_count": round_count,
                    "decode_phase_seconds": dict(decode_phase_seconds),
                    "elapsed_seconds": elapsed,
                    "prefill_elapsed_seconds": prefill_elapsed,
                    "decode_elapsed_seconds": decode_elapsed,
                    "cached_prompt_tokens": num_cached_tokens,
                    "gamma": self.gamma,
                    **self.graph_metrics(),
                }
            )
        self._release_cache()
        return results if self.rank == self.topology.target_leader_rank else None

    def _bound_spec_rhythm_linear_ready(
        self,
        controller: SpecRhythmPipelineController,
        payloads: dict[int, NativeSpecRhythmDevicePayload],
        local_states: Sequence[PearlPipelineState],
        active: Sequence[int],
        roof: int,
        *,
        full_window: bool = False,
    ) -> dict[str, int]:
        """Rebuild oversized linear windows from the committed role-local prefix.

        A linear ticket's gamma describes the *next* draft window, not its
        current target verification size. Shrinking either tensor alone would
        break the paired rollback state. Instead every role discards its own
        unverified suffix at the same service boundary and restarts pre-verify.
        Only ``local_states`` is authoritative: the other model's shadow list
        on this rank is not advanced by the linear execution loop.

        Discarding speculation neither delivers tokens nor changes the prefix
        epoch/home. Old tickets are invalidated and new tickets receive fresh
        monotonic IDs. Existing committed KV rows remain valid; future forwards
        use the shortened sequence length and overwrite the uncommitted tail.
        """
        if isinstance(roof, bool) or not isinstance(roof, int) or roof <= 0:
            raise ValueError("SpecRhythm dynamic linear roof must be a positive integer")
        resets = []
        for index in dict.fromkeys(active):
            state = local_states[index]
            ticket = controller.ready.get(index)
            payload = None
            if ticket is not None:
                controller.validate_verification(index)
                payload = payloads.get(ticket.proposal_id)
                if payload is None or payload.ticket is not ticket:
                    raise RuntimeError("SpecRhythm linear rebuild requires its authoritative ready payload")
                payload.validate_for(state, full_window=full_window)
            current_width = (
                payload.verification_size
                if payload is not None
                else self.gamma
                if full_window
                else state._verification_size(self.gamma, None)
            )
            if current_width <= roof:
                continue
            committed = state.committed_length
            if committed is None or not state.prompt_length <= committed <= len(state.token_ids) or committed <= 0:
                raise RuntimeError("SpecRhythm linear rebuild has an invalid committed prefix boundary")
            runtime = controller.request_states[index]
            if state.continuation_epoch != runtime.prefix_epoch:
                raise RuntimeError("SpecRhythm linear rebuild refused a stale committed prefix epoch")
            staged = controller.staged_eager.get(index)
            if staged is not None:
                staged_payload = payloads.get(staged.proposal_id)
                if staged_payload is None or staged_payload.ticket is not staged:
                    raise RuntimeError("SpecRhythm linear rebuild found a missing staged eager payload")
            discarded_payload_ids = tuple(
                proposal_id for proposal_id, current in payloads.items() if current.ticket.request_index == index
            )
            resets.append((index, state, committed, ticket, staged, discarded_payload_ids))

        counters = {
            "rebuilt_requests": 0,
            "discarded_ready": 0,
            "invalidated_eager": 0,
            "discarded_unverified_tokens": 0,
        }
        for index, state, committed, ticket, staged, discarded_payload_ids in resets:
            counters["rebuilt_requests"] += 1
            counters["discarded_ready"] += int(ticket is not None)
            counters["invalidated_eager"] += int(staged is not None)
            counters["discarded_unverified_tokens"] += len(state.token_ids) - committed
            controller.invalidate_request(index)
            for proposal_id in discarded_payload_ids:
                discarded = payloads.pop(proposal_id)
                discarded.ticket.lifecycle = type(discarded.ticket.lifecycle).INVALIDATED
            del state.token_ids[committed:]
            state.pre_verify = True
            state.pending_window_size = 0
        return counters

    def _bound_spec_rhythm_tree_ready(
        self,
        controller: SpecRhythmPipelineController,
        payloads: dict[int, dict[str, Any]],
        cache_mappings: dict[int, torch.Tensor],
        active: Sequence[int],
        roof: int,
    ) -> dict[str, int]:
        """Safely shrink ready trees when a new profiled roof is smaller.

        A prefix of a topologically ordered, packed tree is ancestor closed.
        Keeping its first ``roof`` candidates preserves their logical and
        physical cache positions; even a strided/non-contiguous mapping can
        therefore be sliced without moving KV. No committed prefix or ticket
        identity changes. A staged continuation survives only when its entire
        original parent dependency still equals the retained primary path.
        All affected payloads are validated before the first mutation.
        """
        if isinstance(roof, bool) or not isinstance(roof, int) or roof <= 0:
            raise ValueError("SpecRhythm dynamic tree roof must be a positive integer")
        updates = []
        for index in dict.fromkeys(active):
            ticket = controller.ready.get(index)
            if ticket is None or ticket.gamma <= roof:
                continue
            controller.validate_verification(index)
            payload = payloads.get(ticket.proposal_id)
            if payload is None or payload.get("ticket") is not ticket:
                raise RuntimeError("SpecRhythm ready-tree pruning requires its authoritative payload ticket")
            plan = payload.get("plan")
            row = payload.get("row")
            if (
                not isinstance(plan, TreeSpeculationPlan)
                or row is None
                or len(row) != ticket.gamma
                or int(plan.candidate_budget) != ticket.gamma
                or plan.parent_indices.numel() != ticket.gamma
            ):
                raise RuntimeError("SpecRhythm ready-tree pruning requires an aligned, packed candidate payload")
            mapping = cache_mappings.get(ticket.proposal_id)
            if mapping is not None and (mapping.ndim != 1 or mapping.numel() < ticket.gamma + 1):
                raise RuntimeError("SpecRhythm ready-tree cache mapping is shorter than its packed payload")
            packed = pack_selected_tree_plan(plan, range(roof))
            kept_row = list(row[:roof])
            retained_dependency = tuple(int(kept_row[node]) for node in tree_primary_path(packed))
            staged = controller.staged_eager.get(index)
            invalidate_staged = False
            if staged is not None:
                eager_payload = payloads.get(staged.proposal_id)
                if eager_payload is None or eager_payload.get("ticket") is not staged:
                    raise RuntimeError("SpecRhythm ready-tree pruning found a missing staged eager payload")
                dependency = tuple(int(token) for token in eager_payload.get("eager_dependency_tokens", ()))
                invalidate_staged = not (
                    staged.request_index == index
                    and staged.home_batch_id == ticket.home_batch_id
                    and staged.lifecycle is type(staged.lifecycle).STAGED_EAGER
                    and staged.required_prefix_epoch == ticket.required_prefix_epoch + 1
                    and eager_payload.get("eager_parent_proposal_id") == ticket.proposal_id
                    and bool(dependency)
                    and eager_payload.get("eager_dependency_length") == len(dependency)
                    and dependency == retained_dependency
                    and eager_payload.get("eager_frontier_token") is not None
                )
            updates.append((ticket, payload, packed, kept_row, mapping, staged, invalidate_staged))

        counters = {"pruned_proposals": 0, "pruned_candidates": 0, "invalidated_eager": 0}
        for ticket, payload, packed, row, mapping, staged, invalidate_staged in updates:
            counters["pruned_proposals"] += 1
            counters["pruned_candidates"] += ticket.gamma - roof
            ticket.gamma = roof
            payload["plan"] = packed
            payload["row"] = row
            if mapping is not None:
                cache_mappings[ticket.proposal_id] = mapping[: roof + 1]
            if invalidate_staged:
                # Retain the ready parent. invalidate_request() would remove
                # both mailboxes and incorrectly discard its verifiable work.
                controller.staged_eager.pop(ticket.request_index)
                staged.lifecycle = type(staged.lifecycle).INVALIDATED
                payloads.pop(staged.proposal_id)
                cache_mappings.pop(staged.proposal_id, None)
                counters["invalidated_eager"] += 1
        return counters

    @staticmethod
    def _spec_rhythm_tree_cache_device(device: torch.device) -> torch.device:
        """Resolve an unindexed NPU device without accepting any other card."""
        if device.type == "npu" and device.index is None:
            return torch.device("npu", torch.npu.current_device())
        return device

    def _preflight_spec_rhythm_tree_cache_commit(
        self,
        controller: SpecRhythmPipelineController,
        payloads: Mapping[int, Mapping[str, Any]],
        cache_mappings: Mapping[int, torch.Tensor],
        target_indices: Sequence[int],
        target_plans: Sequence[TreeSpeculationPlan],
        target_output: Mapping[str, Any] | None,
        *,
        return_host_mappings: bool = False,
    ) -> dict[int, torch.Tensor] | tuple[dict[int, torch.Tensor], dict[int, list[int]]]:
        """Validate every KV move before committing the first request.

        No controller, payload, mapping or KV tensor is mutated here. The
        returned target mappings can be published after the entire preflight
        passes. This catches deterministic malformed metadata, not hardware
        failures/OOM during a subsequent move; those are not rollbackable.

        Slot values are copied to the host in one batch, at the existing
        verdict boundary, rather than synchronizing once per request/layer.
        CPU protocol tests may substitute the model boundary entirely. A real
        model must have allocated paged caches in every layer.
        """
        if len(target_indices) != len(target_plans):
            raise RuntimeError("SpecRhythm tree KV preflight needs one plan per request")
        pending: dict[int, torch.Tensor] = {}
        checked: list[tuple[str, torch.Tensor, int, int | None]] = []
        expected_device = self._spec_rhythm_tree_cache_device(self.device)
        counts = [int(plan.parent_indices.numel()) + 1 for plan in target_plans]
        if any(count != int(plan.candidate_budget) + 1 for count, plan in zip(counts, target_plans)):
            raise RuntimeError("SpecRhythm tree KV preflight requires physically packed candidates")

        def add_mapping(
            label: str,
            mapping: Any,
            count: int,
            proposal_id: int | None = None,
        ) -> None:
            if (
                not isinstance(mapping, torch.Tensor)
                or mapping.ndim != 1
                or mapping.numel() != count
                or mapping.dtype not in (torch.int32, torch.int64)
            ):
                raise RuntimeError(f"SpecRhythm {label} KV mapping must contain exactly {count} integer slots")
            if mapping.device != expected_device:
                raise RuntimeError(f"SpecRhythm {label} KV mapping is on the wrong device")
            checked.append((label, mapping, count, proposal_id))

        if target_output is not None:
            mapping = target_output.get("cache_slot_mapping")
            expected = sum(counts)
            if not isinstance(mapping, torch.Tensor) or mapping.ndim != 1 or mapping.numel() != expected:
                raise RuntimeError(f"SpecRhythm target packed KV mapping must contain exactly {expected} slots")
            cursor = 0
            for index, count in zip(target_indices, counts):
                proposal_id = controller.validate_verification(index).proposal_id
                row_mapping = mapping[cursor : cursor + count]
                add_mapping(
                    f"target proposal {proposal_id}",
                    row_mapping,
                    count,
                    proposal_id,
                )
                pending[proposal_id] = row_mapping
                cursor += count
        elif not self.is_draft:
            raise RuntimeError("SpecRhythm target KV preflight has no forward mapping")

        if self.is_draft:
            for index, count in zip(target_indices, counts):
                proposal_id = controller.validate_verification(index).proposal_id
                proposal_payload = payloads.get(proposal_id)
                if proposal_payload is None:
                    raise RuntimeError(f"SpecRhythm draft proposal {proposal_id} payload disappeared")
                if proposal_payload.get("draft_kv_materialized", True):
                    add_mapping(
                        f"draft proposal {proposal_id}",
                        cache_mappings.get(proposal_id),
                        count,
                        proposal_id,
                    )
                staged = controller.staged_eager.get(index)
                if staged is None:
                    continue
                eager_payload = payloads.get(staged.proposal_id)
                if eager_payload is None:
                    raise RuntimeError("SpecRhythm staged eager tree payload disappeared during KV preflight")
                eager_plan = eager_payload["plan"]
                eager_count = int(eager_plan.parent_indices.numel()) + 1
                add_mapping(
                    f"draft eager proposal {staged.proposal_id}",
                    cache_mappings.get(staged.proposal_id),
                    eager_count,
                    staged.proposal_id,
                )
                destination = torch.tensor(
                    self._cache_slot_mapping(
                        [index] * eager_count,
                        list(range(eager_plan.prefix_len, eager_plan.prefix_len + eager_count)),
                    ),
                    dtype=torch.int32,
                    device=self.device,
                )
                add_mapping(f"draft eager destination {staged.proposal_id}", destination, eager_count)

        if not checked:
            return (pending, {}) if return_host_mappings else pending
        capacity: int | None = None
        model = getattr(self, "model", None)
        if model is not None:
            cached_capacity = getattr(self, "_tree_cache_capacity", None)
            if cached_capacity is not None:
                capacity = int(cached_capacity)
            else:
                layer_caches = self._collect_tree_layer_caches()
                capacity = min(int(cache.shape[0]) * int(cache.shape[1]) for pair in layer_caches for cache in pair)
        slot_values = torch.cat([mapping.to(dtype=torch.long) for _, mapping, _, _ in checked]).detach().cpu().tolist()
        host_mappings: dict[int, list[int]] = {}
        cursor = 0
        for label, _, count, proposal_id in checked:
            row_slots = slot_values[cursor : cursor + count]
            if any(slot < 0 or (capacity is not None and slot >= capacity) for slot in row_slots):
                raise RuntimeError(f"SpecRhythm {label} KV mapping contains an out-of-bounds physical slot")
            if len(set(row_slots)) != count:
                raise RuntimeError(f"SpecRhythm {label} KV mapping aliases distinct tree queries")
            if proposal_id is not None:
                host_mappings[proposal_id] = row_slots
            cursor += count
        return (pending, host_mappings) if return_host_mappings else pending

    def _spec_rhythm_tree_plan(
        self,
        state: PearlPipelineState,
        budget: int,
    ) -> TreeSpeculationPlan:
        """Keep scheduler-only tree topology on the host until model packing."""
        prefix_len = max(0, len(state.token_ids) - 1)
        return cached_cpu_tree_speculation_plan(
            self.config.spec_rhythm_tree_width,
            self.config.spec_rhythm_tree_depth,
            prefix_len,
            self.config.max_model_len,
            candidate_budget=int(budget),
        )

    def _pad_spec_rhythm_target_tree_graph(
        self,
        plans: Sequence[TreeSpeculationPlan],
        root_token_ids: Sequence[int],
        draft_token_ids: Sequence[Sequence[int]],
        request_ids: Sequence[int],
        active_request_ids: Sequence[int],
        states: Sequence[PearlPipelineState],
    ) -> tuple[
        list[TreeSpeculationPlan],
        list[int],
        list[list[int]],
        list[int],
        int,
    ]:
        """Pad a tree verification to a small set of exact ACLGraph shapes.

        Padding never changes a real request's tree or KV length.  Instead it
        uses already-prefilled rows from the other logical home and discards
        their outputs.  A strict next-power-of-two request envelope plus a
        four-token query bucket reduces the continuously changing service
        batch to a bounded set of graph entries.  Each dummy tree contains
        one or two candidates, so it stays inside the configured tree and
        writes only the same current/root slots that its real verification
        will overwrite on a later cycle.
        """
        padded_plans = list(plans)
        padded_roots = [int(value) for value in root_token_ids]
        padded_rows = [list(map(int, row)) for row in draft_token_ids]
        padded_ids = [int(value) for value in request_ids]
        real_count = len(padded_plans)
        if not (real_count == len(padded_roots) == len(padded_rows) == len(padded_ids)):
            raise ValueError("SpecRhythm tree graph padding inputs must be row-aligned")
        if not real_count or not (
            envs.VLLM_ASCEND_SPECRHYTHM_TREE_GRAPH
            and not self.config.enforce_eager
            and self.config.spec_rhythm_stable_graphs
        ):
            return padded_plans, padded_roots, padded_rows, padded_ids, real_count

        # Use a *strictly* larger power-of-two bucket.  Keeping at least one
        # dummy segment lets us absorb candidate-count variation without
        # padding any real request's sequence length.
        real_ids = set(padded_ids)
        available = [int(index) for index in active_request_ids if int(index) not in real_ids]
        real_query_count = sum(int(plan.candidate_budget) + 1 for plan in padded_plans)
        max_dummy_candidates = int(self.config.spec_rhythm_tree_width) * int(self.config.spec_rhythm_tree_depth)
        request_bucket = 1 << real_count.bit_length()
        while True:
            padding_count = request_bucket - real_count
            if padding_count <= 0 or len(available) < padding_count:
                return padded_plans, padded_roots, padded_rows, padded_ids, real_count
            minimum_query_count = real_query_count + 2 * padding_count
            query_bucket = ((minimum_query_count + 3) // 4) * 4
            extra_queries = query_bucket - minimum_query_count
            if extra_queries <= (max_dummy_candidates - 1) * padding_count:
                break
            # A shallow tree cannot absorb three extra queries in one dummy
            # row.  Grow the request envelope when another home has enough
            # resident rows; otherwise use the exact unpadded execution.  In
            # particular this keeps fixed-gamma 2x1 trees within their two
            # candidate topology instead of constructing an invalid budget.
            request_bucket *= 2
        extra_by_row = [0] * padding_count
        for extra in range(extra_queries):
            extra_by_row[extra % padding_count] += 1
        for row, request_id in enumerate(available[:padding_count]):
            candidate_count = 1 + extra_by_row[row]
            padded_plans.append(self._spec_rhythm_tree_plan(states[request_id], candidate_count))
            padded_roots.append(int(states[request_id].token_ids[-1]))
            padded_rows.append([0] * candidate_count)
            padded_ids.append(request_id)
        if sum(int(plan.candidate_budget) + 1 for plan in padded_plans) != query_bucket:
            raise RuntimeError("SpecRhythm tree graph query bucket was not filled exactly")
        return padded_plans, padded_roots, padded_rows, padded_ids, real_count

    def _deliver_committed_tokens(self, index, params, state) -> None:
        callback = getattr(self, "_token_commit_callback", None)
        if callback is None or self.rank != self.topology.target_leader_rank:
            return
        completion = _truncate_completion(
            state.committed_completion_token_ids,
            self.eos_token_ids,
            state.max_tokens,
            state.ignore_eos,
        )
        previous = self._stream_delivered_counts.get(index, 0)
        if len(completion) <= previous:
            return
        try:
            callback(
                {
                    "request_index": index,
                    "request_id": params.request_id,
                    "token_ids": completion[previous:],
                    "finished": _finished(state, self.eos_token_ids),
                    "elapsed_seconds": time.perf_counter() - self._stream_started,
                }
            )
        except Exception as error:
            # Finish all rank collectives before surfacing a client callback
            # failure; only the target leader invokes user callbacks.
            self._token_commit_error = error
            self._token_commit_callback = None
        self._stream_delivered_counts[index] = len(completion)

    @staticmethod
    def _tree_primary_path_length(plan: TreeSpeculationPlan) -> int:
        """Return the dependency-closed spine length of a tree proposal."""
        return len(tree_primary_path(plan))

    def _spec_rhythm_tree_eager_plan(
        self,
        parent_plan: TreeSpeculationPlan,
        state: PearlPipelineState,
        budget: int,
    ) -> TreeSpeculationPlan:
        """Build an ahead-of-turn tree after a speculative frontier token.

        The draft worker predicts one frontier token after the parent's
        dependency-closed spine and uses that token as the root of the eager
        tree.  Its KV row is written at the position immediately following
        the parent spine, so promotion can be guarded by comparing both the
        accepted spine and the target frontier token.
        """
        path_len = self._tree_primary_path_length(parent_plan)
        base_prefix = max(0, len(state.token_ids) - 1)
        expected_prefix = int(parent_plan.prefix_len) + path_len + 1
        if expected_prefix != base_prefix + path_len + 1:
            raise RuntimeError("SpecRhythm eager tree parent prefix does not match request state")
        return cached_cpu_tree_speculation_plan(
            self.config.spec_rhythm_tree_width,
            self.config.spec_rhythm_tree_depth,
            expected_prefix,
            self.config.max_model_len,
            candidate_budget=int(budget),
        )

    def _exchange_spec_rhythm_tree_candidates(
        self,
        candidate_rows: Sequence[Sequence[int]] | None,
        plans: Sequence[TreeSpeculationPlan],
        frontier_tokens: Sequence[int | None] | None = None,
        confidences: Sequence[float] | None = None,
        selected_indices: Sequence[Sequence[int]] | None = None,
        draft_compute_ms: float = 0.0,
    ) -> torch.Tensor | None:
        """Broadcast active tree nodes and the measured draft window.

        Carrying the role-local duration in this existing envelope avoids an
        extra world all-reduce in every production decode cycle.  Float64
        represents both vocabulary IDs and the millisecond observation exactly
        enough for the deterministic host-side EMA.
        """
        if not plans:
            return None
        if not math.isfinite(draft_compute_ms) or draft_compute_ms < 0:
            raise ValueError("draft tree compute time must be finite and non-negative")
        capacities = [int(plan.width * plan.depth) for plan in plans]
        # One extra scalar per row carries the draft-predicted frontier token
        # used to validate a rolling eager continuation.  ``-1`` marks a
        # normal proposal and is never a valid vocabulary id in this control
        # envelope.
        message_size = 2 * sum(capacities) + 3 * len(plans) + 1
        if self.rank == self.topology.draft_leader_rank:
            if candidate_rows is None or len(candidate_rows) != len(plans):
                raise RuntimeError("draft tree worker did not produce all candidate rows")
            counts = [len(row) for row in candidate_rows]
            if any(count > capacity for count, capacity in zip(counts, capacities)):
                raise RuntimeError("refined tree exceeds exploratory capacity")
            values = list(counts)
            for row, capacity in zip(candidate_rows, capacities):
                values.extend([*map(int, row), *([-1] * (capacity - len(row)))])
            for row, (count, capacity) in enumerate(zip(counts, capacities)):
                indices = list(range(count) if selected_indices is None else selected_indices[row])
                if len(indices) != count:
                    raise RuntimeError("selected tree indices must align with refined tokens")
                values.extend([*indices, *([-1] * (capacity - count))])
            frontier_values = [
                -1 if frontier_tokens is None or frontier_tokens[row] is None else int(frontier_tokens[row])
                for row in range(len(plans))
            ]
            values.extend(frontier_values)
            if confidences is None or len(confidences) != len(plans):
                raise RuntimeError("tree proposal confidence must be row-aligned")
            values.extend(float(value) for value in confidences)
            values.append(float(draft_compute_ms))
            if len(values) != message_size:
                raise RuntimeError("tree proposal envelope has an invalid active-node size")
            # Float64 represents vocabulary IDs exactly and carries the same
            # confidence to every rank. A local default would diverge EMA and
            # therefore the next collective's request/shape plan.
            message = torch.tensor(values, dtype=torch.float64, device=self.device)
        else:
            message = torch.empty(message_size, dtype=torch.float64, device=self.device)
        dist.broadcast(
            message,
            src=self.topology.draft_leader_rank,
        )
        return message

    @staticmethod
    def _split_tree_candidates(
        message: torch.Tensor,
        plans: Sequence[TreeSpeculationPlan],
    ) -> tuple[list[list[int]], list[int | None], list[float], list[list[int]], float]:
        capacities = [int(plan.width * plan.depth) for plan in plans]
        if message.ndim != 1 or message.numel() != 2 * sum(capacities) + 3 * len(plans) + 1:
            raise RuntimeError("tree proposal envelope does not match its plan")
        values = message.detach().cpu().tolist()
        rows: list[list[int]] = []
        counts = [int(value) for value in values[: len(plans)]]
        if any(not 0 <= count <= capacity for count, capacity in zip(counts, capacities)):
            raise RuntimeError("invalid refined candidate count")
        cursor = len(plans)
        for count, capacity in zip(counts, capacities):
            rows.append([int(value) for value in values[cursor : cursor + count]])
            cursor += capacity
        selected_rows = []
        for count, capacity in zip(counts, capacities):
            selected_rows.append([int(value) for value in values[cursor : cursor + count]])
            cursor += capacity
        frontier_values = values[cursor : cursor + len(plans)]
        confidence_start = cursor + len(plans)
        confidence_end = confidence_start + len(plans)
        confidences = [float(value) for value in values[confidence_start:confidence_end]]
        draft_compute_ms = float(values[confidence_end])
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in confidences):
            raise RuntimeError("tree proposal contains invalid confidence")
        if not math.isfinite(draft_compute_ms) or draft_compute_ms < 0:
            raise RuntimeError("tree proposal contains invalid draft compute time")
        return (
            rows,
            [int(value) if value >= 0 else None for value in frontier_values],
            confidences,
            selected_rows,
            draft_compute_ms,
        )

    def _broadcast_spec_rhythm_tree_verdict(
        self,
        output: TreeVerificationOutput | None,
        plans: Sequence[TreeSpeculationPlan],
        target_compute_ms: float = 0.0,
    ) -> tuple[list[list[int]], list[list[int]], float]:
        """Replicate tree verdict and target-window observation to all ranks."""
        if not plans:
            return [], [], 0.0
        if not math.isfinite(target_compute_ms) or target_compute_ms < 0:
            raise ValueError("target tree compute time must be finite and non-negative")
        depth = max(int(plan.depth) for plan in plans)
        batch = len(plans)
        width = batch * (depth + 1 + depth) + 1
        is_target_leader = self.rank == self.topology.target_leader_rank
        if is_target_leader:
            if output is None:
                raise RuntimeError("target leader did not produce a tree verdict")
            if tuple(output.token_ids.shape) != (batch, depth + 1):
                raise RuntimeError("tree verifier output has an invalid token shape")
            if tuple(output.accepted_node_indices.shape) != (batch, depth):
                raise RuntimeError("tree verifier output has an invalid acceptance shape")
            message = torch.cat(
                (
                    output.token_ids.to(torch.long).reshape(-1),
                    output.accepted_node_indices.to(torch.long).reshape(-1),
                    torch.tensor(
                        [round(target_compute_ms * 1000.0)],
                        dtype=torch.long,
                        device=self.device,
                    ),
                )
            )
        else:
            message = torch.empty(width, dtype=torch.long, device=self.device)
        # First complete the target-model group broadcast.  The target leader
        # is also a member of correction_group and must then participate in a
        # second broadcast to the draft ranks; branching with ``elif`` here
        # leaves that leader out of correction_group and deadlocks HCCL.
        if self.rank in self.topology.target_ranks:
            dist.broadcast(
                message,
                src=self.topology.target_leader_rank,
                group=self.groups.target_group,
            )
        if self.rank in self.topology.correction_ranks:
            dist.broadcast(
                message,
                src=self.topology.target_leader_rank,
                group=self.groups.correction_group,
            )
        values = message[:-1].reshape(-1)
        token_values = values[: batch * (depth + 1)].reshape(batch, depth + 1)
        accepted_values = values[batch * (depth + 1) :].reshape(batch, depth)
        return (
            [[int(value) for value in row] for row in token_values.cpu().tolist()],
            [[int(value) for value in row] for row in accepted_values.cpu().tolist()],
            float(message[-1].cpu().item()) / 1000.0,
        )

    def _validate_spec_rhythm_target_graph(
        self,
        target_output: Mapping[str, Any] | None,
        *,
        has_target_work: bool,
        target_only: bool = False,
    ) -> None:
        """Validate measured backend/mode at a common point on all ranks."""
        if not has_target_work or not isinstance(self.config.spec_rhythm_roofline, ProfiledRoofline):
            return
        # Draft fallback is irrelevant to a target-only roofline. All target
        # ranks must actually execute the graph; the mode switch is not proof.
        backend_error: ValueError | None = None
        if not self.is_draft:
            try:
                self.config.spec_rhythm_roofline.validate_attention_backend(
                    (target_output or {}).get("attention_backend", "missing"),
                    target_only=target_only,
                )
            except ValueError as error:
                backend_error = error
        graph_invalid = (
            not self.config.enforce_eager
            and not self.is_draft
            and not (target_output or {}).get("used_aclgraph", False)
        )
        invalid = torch.tensor(
            [2 if backend_error is not None else int(graph_invalid)],
            dtype=torch.int64,
            device=self.device,
        )
        dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
        failure = int(invalid.cpu().item())
        if failure == 2:
            raise RuntimeError(
                "Strict SpecSLO roofline attention backend validation failed on a target rank; "
                "no proposal or verdict was published under that unprofiled backend."
            ) from backend_error
        if failure:
            raise RuntimeError(
                "Strict SpecSLO graph roofline encountered a target eager fallback; "
                "no proposal or verdict was published under that unprofiled execution mode."
            )

    @torch.inference_mode()
    def _generate_spec_rhythm_tree_decode(
        self,
        *,
        draft_states: list[PearlPipelineState],
        target_states: list[PearlPipelineState],
        request_params: Sequence[NativeSamplingParams],
        initial_batch_size: int,
        continuous_batching: bool,
        prefill_elapsed: float,
        started: float,
        max_rounds: int | None,
        prefilled_indices: set[int] | None = None,
    ) -> list[dict[str, Any]] | None:
        """Run SpecRhythm with fixed-shape tree proposals end to end.

        The scalar shaper remains the source of truth for every candidate
        budget.  Draft and target workers exchange only active nodes; graph
        padding is local to the tree forward and never counts against B.
        """
        request_params = list(request_params)
        states = {
            index: SpecRhythmRuntimeState(
                request_index=index,
                home_batch_id=index % 2,
                slo_tpot_ms=params.slo_tpot_ms,
                slo_class=params.slo_class,
                max_gamma=params.spec_rhythm_max_gamma,
            )
            for index, params in enumerate(request_params)
        }
        controller = SpecRhythmPipelineController(states)
        shaper = SpecRhythmBudgetShaper(
            min_gamma=self.config.spec_rhythm_min_gamma,
            max_gamma=max(
                self.config.spec_rhythm_tree_width * self.config.spec_rhythm_tree_depth,
                self.config.spec_rhythm_min_gamma,
            ),
            verification_budget=self.config.spec_rhythm_verification_budget,
            acceptance_floor=self.config.spec_rhythm_acceptance_floor,
            acceptance_ema_alpha=self.config.spec_rhythm_acceptance_ema_alpha,
            roofline=self.config.spec_rhythm_roofline,
        )
        has_slo_constraints = any(
            params.slo_tpot_ms is not None or params.slo_class is not None for params in request_params
        )
        effective_eager_cap = self.config.spec_rhythm_max_eager_tokens or (
            shaper.max_gamma if has_slo_constraints else 0
        )
        local_states = draft_states if self.is_draft else target_states
        active: list[int] = []
        pending_admission = list(range(len(local_states)))
        live_admission = getattr(self, "_request_admission_callback", None)
        live_open = live_admission is not None
        prefetched = set(prefilled_indices or ())
        # Keep normal and staged-eager proposals separate.  A request can have
        # one ready payload and one ahead-of-turn payload at the same time;
        # using request id as the key would overwrite the ready tree before
        # its target verdict arrives.
        payloads: dict[int, dict[str, Any]] = {}
        cache_mappings: dict[int, torch.Tensor] = {}
        completed: dict[int, PearlPipelineState] = {}
        finished_decode_elapsed: dict[int, float] = {}
        first_decode_indices: set[int] = set()
        counters = {
            "spec_rhythm_tree_rounds": 0,
            "spec_rhythm_tree_nodes": 0,
            "spec_rhythm_tree_accepted_nodes": 0,
            "spec_rhythm_allocated_draft_tokens": 0,
            "spec_rhythm_verified_tokens": 0,
            "spec_rhythm_tree_full_accepts": 0,
            "spec_rhythm_tree_rejections": 0,
            "spec_rhythm_tree_eager_promoted": 0,
            "spec_rhythm_tree_eager_invalidated": 0,
            "spec_rhythm_tree_eager_dependency_rejections": 0,
            "spec_rhythm_tree_eager_promotion_kv_move_rounds": 0,
            "spec_rhythm_tree_eager_promotion_kv_move_rows": 0,
            "spec_rhythm_tree_eager_promotion_kv_move_slots": 0,
            "spec_rhythm_last_verification_roof": 0,
            "spec_rhythm_unused_verification_tokens": 0,
            "spec_rhythm_prefill_batches": 0,
            "spec_rhythm_prefill_requests": 0,
            "spec_rhythm_peak_active_requests": 0,
            "spec_rhythm_peak_verify_candidates": 0,
            "spec_rhythm_target_query_tokens": 0,
            "spec_rhythm_target_graph_padding_queries": 0,
            "spec_rhythm_draft_model_calls": 0,
            "spec_rhythm_draft_graph_calls": 0,
            "spec_rhythm_draft_graph_padding_queries": 0,
            "spec_rhythm_draft_materialized_nodes": 0,
            "spec_rhythm_draft_publish_rows": 0,
            "spec_rhythm_draft_publish_moved_rows": 0,
            "spec_rhythm_draft_publish_skipped_rows": 0,
            "spec_rhythm_kv_compaction_rows": 0,
            "spec_rhythm_kv_compaction_skipped_rows": 0,
            "spec_rhythm_kv_compaction_rounds": 0,
            "spec_rhythm_live_admitted_requests": 0,
            "spec_rhythm_aborted_requests": 0,
            # The rank-local role calls are deliberately ordered so draft
            # worker compute and target worker compute begin in the same
            # round before the proposal/verdict collectives.
            "spec_rhythm_dual_batch_overlap_protocol": 1,
        }
        round_count = 0
        last_cycle_ms = 0.0
        window_estimator = DraftWindowEstimator(self.config.spec_rhythm_acceptance_ema_alpha)
        decode_timeline = []
        # Rank-local timestamps are deliberately separate from the existing
        # synchronized fine profile.  They add no NPU fence/collective and are
        # therefore suitable for locating queue or graph-lifecycle stalls in
        # the production execution path.
        host_timeline: list[dict[str, int | float]] = []
        npu_profiler = None
        if envs.VLLM_ASCEND_PEARL_NPU_PROFILE_DIR:
            profile_rank = envs.VLLM_ASCEND_PEARL_NPU_PROFILE_RANK
            if profile_rank is None or profile_rank == -1 or profile_rank == self.rank:
                import torch_npu

                npu_profiler = torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        envs.VLLM_ASCEND_PEARL_NPU_PROFILE_DIR,
                        worker_name=f"specslo-tree-rank-{self.rank}",
                    ),
                    record_shapes=True,
                )
                self._active_tree_profiler = npu_profiler
                npu_profiler.start()
        phase_seconds = {
            "draft": 0.0,
            "target": 0.0,
            "exchange": 0.0,
            "verify": 0.0,
            "broadcast": 0.0,
            "state_update": 0.0,
        }
        decode_profile_seconds = {
            "draft_compute": 0.0,
            "draft_to_target_communication": 0.0,
            "target_compute": 0.0,
            "target_verdict": 0.0,
            "target_to_draft_communication": 0.0,
            "wait_sync": 0.0,
            "state_update": 0.0,
        }
        decode_profile_detail_seconds = {
            "scheduler_plan": 0.0,
            "draft_tree_setup": 0.0,
            "draft_tree_level_compute": 0.0,
            "draft_tree_topk": 0.0,
            "draft_tree_materialize_metadata": 0.0,
            "draft_tree_materialize_compute": 0.0,
            "target_tree_setup": 0.0,
            "target_tree_metadata": 0.0,
            "target_tree_model": 0.0,
            "target_tree_output": 0.0,
            "draft_postprocess": 0.0,
            "profile_timeline_collective": 0.0,
            "proposal_publish": 0.0,
            "commit_preflight": 0.0,
            "commit_consensus": 0.0,
            "kv_compaction": 0.0,
            "request_commit": 0.0,
            "cycle_accounting": 0.0,
        }
        profiled_decode_steps = 0
        native_tree_forward = getattr(self.draft_tree_forward, "__func__", None) is NativePearlEngine.draft_tree_forward
        native_target_tree_forward = (
            getattr(self.target_tree_forward, "__func__", None) is NativePearlEngine.target_tree_forward
        )

        def profile_begin(enabled: bool) -> float | None:
            """Start an intrusive, explicitly bounded profile interval.

            Synchronization is rank-local.  In particular, a target rank must
            not wait for the draft rank here: doing so would destroy the very
            cross-device overlap this profile is intended to measure.
            """
            if not enabled:
                return None
            torch.npu.synchronize()
            return time.perf_counter()

        def profile_end(started_at: float | None, phase: str) -> float:
            if started_at is None:
                return 0.0
            torch.npu.synchronize()
            elapsed = time.perf_counter() - started_at
            decode_profile_seconds[phase] += elapsed
            return elapsed

        def profile_detail_end(started_at: float | None, phase: str) -> float:
            """Finish a nested diagnostic interval without broad-phase accounting."""
            if started_at is None:
                return 0.0
            torch.npu.synchronize()
            elapsed = time.perf_counter() - started_at
            decode_profile_detail_seconds[phase] += elapsed
            return elapsed

        def finish_profile_round(started_at: float | None, accounted_before: float) -> None:
            """Charge the unclassified critical-path tail to wait/sync.

            ``profile_end`` measures role-local model and transport intervals.
            Everything else on the rank's cycle critical path -- scheduler
            gaps, host/device synchronizations and control collectives -- is
            intentionally exposed as residual wait/sync instead of vanishing
            from the profile.
            """
            if started_at is None:
                return
            torch.npu.synchronize()
            total = time.perf_counter() - started_at
            accounted = sum(decode_profile_seconds.values()) - accounted_before
            decode_profile_seconds["wait_sync"] += max(0.0, total - accounted)

        def receive_live_admissions(*, block: bool) -> bool:
            """Append controller-synchronized HTTP arrivals at a cycle fence."""
            nonlocal has_slo_constraints, effective_eager_cap
            if live_admission is None:
                return False
            admission_result = live_admission(block)
            if len(admission_result) == 2:
                closed, arrivals = admission_result
                aborted_request_ids: Sequence[str] = ()
            elif len(admission_result) == 3:
                closed, arrivals, aborted_request_ids = admission_result
            else:
                raise RuntimeError("SpecSLO live control callback returned an invalid tuple")
            if closed and arrivals:
                raise RuntimeError("SpecSLO live admission cannot close with pending requests")
            abort_set = {str(value) for value in aborted_request_ids}
            for index, params in enumerate(request_params):
                if str(params.request_id) not in abort_set or index in completed:
                    continue
                for state in (draft_states[index], target_states[index]):
                    state.aborted = True
                if index in pending_admission:
                    pending_admission.remove(index)
                if index in active:
                    active.remove(index)
                for proposal_id, payload in tuple(payloads.items()):
                    if payload["ticket"].request_index == index:
                        payloads.pop(proposal_id, None)
                        cache_mappings.pop(proposal_id, None)
                controller.invalidate_request(index)
                completed[index] = local_states[index].clone()
                finished_decode_elapsed[index] = states[index].decode_elapsed_ms
                counters["spec_rhythm_aborted_requests"] += 1
            if not arrivals:
                return bool(closed)
            if len(request_params) + len(arrivals) > self._cache_sequence_capacity:
                raise RuntimeError("SpecSLO live requests exceed the reserved KV sequence capacity")
            existing_signature = (
                request_params[0].temperature > 0,
                request_params[0].draft_temperature > 0,
            )
            for prompt, params in arrivals:
                if not isinstance(params, NativeSamplingParams):
                    raise TypeError("SpecSLO live admission requires NativeSamplingParams")
                if (params.temperature > 0, params.draft_temperature > 0) != existing_signature:
                    raise ValueError("SpecSLO live admission cannot mix greedy and stochastic sampling modes")
                tokens = [int(token_id) for token_id in prompt]
                tail_capacity = self.config.spec_rhythm_tree_width * self.config.spec_rhythm_tree_depth + 1
                if not tokens or len(tokens) + params.max_tokens + tail_capacity > self.config.max_model_len:
                    raise ValueError("SpecSLO live prompt plus completion exceeds max_model_len")
                index = len(request_params)
                self._activate_cache_sequence(
                    index,
                    tokens,
                    enable_prefix_caching=self.config.enable_prefix_caching,
                )
                draft_state = PearlPipelineState(
                    list(tokens),
                    len(tokens),
                    temperature=params.temperature,
                    draft_temperature=params.draft_temperature,
                    top_p=params.top_p,
                    top_k=params.top_k,
                    draft_top_p=params.draft_top_p,
                    draft_top_k=params.draft_top_k,
                    max_tokens=params.max_tokens,
                    ignore_eos=params.ignore_eos,
                    slo_tpot_ms=params.slo_tpot_ms,
                    slo_class=params.slo_class,
                    request_id=params.request_id,
                    arrival_ts=params.arrival_ts,
                )
                draft_states.append(draft_state)
                target_states.append(draft_state.clone())
                request_params.append(params)
                states[index] = SpecRhythmRuntimeState(
                    request_index=index,
                    home_batch_id=index % 2,
                    slo_tpot_ms=params.slo_tpot_ms,
                    slo_class=params.slo_class,
                    max_gamma=params.spec_rhythm_max_gamma,
                )
                pending_admission.append(index)
                counters["spec_rhythm_live_admitted_requests"] += 1
                if params.slo_tpot_ms is not None or params.slo_class is not None:
                    has_slo_constraints = True
                    effective_eager_cap = self.config.spec_rhythm_max_eager_tokens or shaper.max_gamma
                    self.graph_runner.set_expected_fia_batch_size(None)
            return bool(closed)

        def admit_available(*, poll_live: bool = True, block_live: bool = False) -> bool:
            closed = receive_live_admissions(block=block_live) if poll_live else False
            if not pending_admission or len(active) >= initial_batch_size:
                return closed
            arrival_gated = bool(self.config.spec_rhythm_online_prefill)
            now_tensor = torch.tensor(
                [time.time() if self.rank == self.topology.target_leader_rank else 0.0],
                dtype=torch.float64,
                device=self.device,
            )
            dist.broadcast(now_tensor, src=self.topology.target_leader_rank)
            now = float(now_tensor.cpu().item())
            admitted = [
                index
                for index in pending_admission
                if not arrival_gated
                or request_params[index].arrival_ts is None
                or request_params[index].arrival_ts <= now
            ][: initial_batch_size - len(active)]
            homes = [sum(states[index].home_batch_id == home for index in active) for home in (0, 1)]
            for index in admitted:
                home = 0 if homes[0] <= homes[1] else 1
                states[index].home_batch_id = home
                homes[home] += 1
                arrival = request_params[index].arrival_ts
                if arrival_gated and arrival is not None:
                    states[index].arrival_wait_ms = max(0.0, (now - arrival) * 1000.0)
            needs_prefill = [index for index in admitted if index not in prefetched]
            if needs_prefill:
                refill_started = time.perf_counter()
                tokens = self._prefill_and_sample_target_batch(
                    [local_states[index].token_ids for index in needs_prefill],
                    [local_states[index] for index in needs_prefill],
                    needs_prefill,
                )
                # The batched prefill return contract is already host-visible:
                # it completes the distributed token broadcast and copies the
                # sampled tokens to a Python list.  An additional device-wide
                # synchronize here only serializes unrelated queued work.
                refill_elapsed = torch.tensor(
                    [
                        (time.perf_counter() - refill_started) * 1000.0
                        if self.rank == self.topology.target_leader_rank
                        else 0.0
                    ],
                    dtype=torch.float64,
                    device=self.device,
                )
                dist.broadcast(refill_elapsed, src=self.topology.target_leader_rank)
                refill_ms = float(refill_elapsed.cpu().item())
                # Prefill of newly admitted requests shares these devices.
                # Its pause counts for already-decoding requests, but not
                # as decode time before a new request's first output token.
                for index in active:
                    states[index].add_decode_time(refill_ms)
                counters["spec_rhythm_online_refill_ms"] = counters.get("spec_rhythm_online_refill_ms", 0.0) + refill_ms
                for index, token in zip(needs_prefill, tokens):
                    for state in (draft_states[index], target_states[index]):
                        state.token_ids.append(token)
                        state.committed_length += 1
                prefetched.update(needs_prefill)
                counters["spec_rhythm_prefill_batches"] += 1
                counters["spec_rhythm_prefill_requests"] += len(needs_prefill)
            for index in admitted:
                pending_admission.remove(index)
                states[index].delivered_tokens = len(local_states[index].committed_completion_token_ids)
                states[index].decode_start_elapsed_ms = states[index].decode_elapsed_ms
                self._deliver_committed_tokens(index, request_params[index], local_states[index])
                if _finished(local_states[index], self.eos_token_ids):
                    completed[index] = local_states[index].clone()
                    finished_decode_elapsed[index] = 0.0
                else:
                    active.append(index)
            counters["spec_rhythm_peak_active_requests"] = max(
                counters["spec_rhythm_peak_active_requests"], len(active)
            )
            return closed

        def run_profiled_target_only(indices: Sequence[int], *, profile_this_round: bool) -> list[int]:
            """Advance one logical home when the measured candidate roof is zero.

            Both model partitions consume the current committed token so their
            KV prefixes remain aligned if a later batch/context key permits
            speculation again.  Only the target result is committed.
            """
            rows = [int(index) for index in indices]
            if not rows:
                raise RuntimeError("SpecSLO target-only fallback requires one logical-home request")
            input_ids = torch.tensor(
                [local_states[index].token_ids[-1] for index in rows],
                dtype=torch.long,
                device=self.device,
            )
            positions = [len(local_states[index].token_ids) - 1 for index in rows]
            target_execution: dict[str, Any] | None = None
            compute_phase = "draft_compute" if self.is_draft else "target_compute"
            compute_profile_started = profile_begin(profile_this_round)
            if self.is_draft:
                self._run_device_packed_hidden(
                    input_ids,
                    rows,
                    positions,
                    use_aclgraph=not self.config.enforce_eager,
                    use_fused_infer_attention=False,
                )
                token_ids = torch.zeros(len(rows), dtype=torch.long, device=self.device)
            else:
                packed_positions, metadata = self._prepare_attention_metadata(rows, positions, False)
                temperatures = [target_states[index].temperature for index in rows]
                top_ps = [target_states[index].top_p for index in rows]
                top_ks = [target_states[index].top_k for index in rows]
                # The diagnostic switch promises a completely eager target
                # oracle, including the short target-only tail used when a
                # fixed-gamma proposal no longer fits.  Previously only the
                # speculative verification path honored it, so an otherwise
                # eager oracle still captured one-token target graphs at the
                # end of a batch.
                target_use_aclgraph = (
                    not self.config.enforce_eager and not envs.VLLM_ASCEND_SPECRHYTHM_DISABLE_TARGET_ACLGRAPH
                )
                if not any(temperatures):
                    if not target_use_aclgraph:
                        hidden = self.model(input_ids, packed_positions, metadata)
                        token_ids = self.model.compute_greedy_tokens(hidden, self.draft_vocab_size)
                    else:
                        token_ids = self.graph_runner.run_target_greedy(
                            [input_ids], [packed_positions], [metadata], self.draft_vocab_size
                        )[0]
                else:
                    if not target_use_aclgraph:
                        hidden = self.model(input_ids, packed_positions, metadata)
                        logits = self.model.compute_logits(hidden)[:, : self.draft_vocab_size]
                    else:
                        logits = self.graph_runner.run_target_logits(
                            [input_ids], [packed_positions], [metadata], self.draft_vocab_size
                        )[0]
                    token_ids = _sample_logits(logits, temperatures, top_ps=top_ps, top_ks=top_ks)
                    if dist.is_initialized():
                        dist.broadcast(
                            token_ids,
                            src=self.topology.target_leader_rank,
                            group=self.groups.target_group,
                        )
                target_execution = {
                    "attention_backend": attention_backend_identity(metadata),
                    "used_aclgraph": bool(
                        target_use_aclgraph and self.graph_runner.last_target_execution.used_aclgraph
                    ),
                }
            profile_end(compute_profile_started, compute_phase)
            correction_profile_started = profile_begin(profile_this_round)
            torch.npu.synchronize()
            self._vote_spec_rhythm_prefill_finiteness()
            self._validate_spec_rhythm_target_graph(
                target_execution,
                has_target_work=True,
                target_only=True,
            )
            dist.broadcast(token_ids, src=self.topology.target_leader_rank)
            values = [int(value) for value in token_ids.cpu().tolist()]
            profile_end(correction_profile_started, "target_to_draft_communication")
            return values

        def complete_target_only_cycle(
            target_indices: Sequence[int],
            *,
            cycle_started: float,
            unprofiled: bool,
            target_home: int | None,
            profile_this_round: bool,
            profile_round_started: float | None,
            profile_accounted_before: float,
        ) -> None:
            """Finish one AR fallback cycle and keep both model KV states aligned."""
            nonlocal active, live_open, last_cycle_ms, round_count, profiled_decode_steps
            rows = [int(index) for index in target_indices]
            state_profile_started = profile_begin(profile_this_round)
            invalidated_eager = 0
            for index in active:
                invalidated_eager += int(index in controller.staged_eager)
                for ticket in (controller.ready.get(index), controller.staged_eager.get(index)):
                    if ticket is not None:
                        payloads.pop(ticket.proposal_id, None)
                        cache_mappings.pop(ticket.proposal_id, None)
                controller.invalidate_request(index)
            counters["spec_rhythm_tree_eager_invalidated"] += invalidated_eager
            profile_end(state_profile_started, "state_update")
            tokens = run_profiled_target_only(rows, profile_this_round=profile_this_round)
            state_profile_started = profile_begin(profile_this_round)
            finished: list[int] = []
            for index, token in zip(rows, tokens):
                for state in (draft_states[index], target_states[index]):
                    state.token_ids.append(token)
                    assert state.committed_length is not None
                    state.committed_length += 1
                states[index].record_verification(
                    proposed_tokens=0,
                    accepted_tokens=0,
                    delivered_tokens=1,
                    draft_confidence=None,
                    ema_alpha=self.config.spec_rhythm_acceptance_ema_alpha,
                )
                self._deliver_committed_tokens(index, request_params[index], local_states[index])
                if _finished(local_states[index], self.eos_token_ids):
                    finished.append(index)
            completed_home = states[rows[0]].home_batch_id if target_home is None else int(target_home)
            controller.next_target_home_batch_id = 1 - completed_home
            elapsed_tensor = torch.tensor(
                [time.perf_counter() - cycle_started if self.rank == self.topology.target_leader_rank else 0.0],
                dtype=torch.float64,
                device=self.device,
            )
            dist.broadcast(elapsed_tensor, src=self.topology.target_leader_rank)
            last_cycle_ms = float(elapsed_tensor.cpu().item()) * 1000.0
            for index in active:
                states[index].add_decode_time(last_cycle_ms)
            for index in finished:
                completed[index] = local_states[index].clone()
                finished_decode_elapsed[index] = states[index].decode_elapsed_ms
            finished_set = set(finished)
            active = [index for index in active if index not in finished_set]
            # The outer loop polls live admissions before its next cycle.
            # Keep only the static-pending refill here to avoid a back-to-back
            # callback, clock broadcast and device-to-host clock read.
            if not live_open:
                admit_available(poll_live=False)
            counters["spec_rhythm_last_verification_roof"] = 0
            counters["spec_rhythm_unused_verification_tokens"] = 0
            counters["spec_rhythm_target_query_tokens"] += len(rows)
            counters["spec_rhythm_target_only_fallback_rounds"] = (
                counters.get("spec_rhythm_target_only_fallback_rounds", 0) + 1
            )
            counters["spec_rhythm_target_only_fallback_tokens"] = counters.get(
                "spec_rhythm_target_only_fallback_tokens", 0
            ) + len(tokens)
            if unprofiled:
                counters["spec_rhythm_unprofiled_target_only_fallback_rounds"] = (
                    counters.get("spec_rhythm_unprofiled_target_only_fallback_rounds", 0) + 1
                )
                counters["spec_rhythm_unprofiled_target_only_fallback_tokens"] = counters.get(
                    "spec_rhythm_unprofiled_target_only_fallback_tokens", 0
                ) + len(tokens)
            counters["spec_rhythm_tree_rounds"] += 1
            profile_end(state_profile_started, "state_update")
            finish_profile_round(profile_round_started, profile_accounted_before)
            if profile_this_round:
                profiled_decode_steps += 1
            round_count += 1
            if npu_profiler is not None:
                npu_profiler.step()

        admit_available(poll_live=False)
        while (active or pending_admission or live_open) and (max_rounds is None or round_count < max_rounds):
            if not active:
                if live_open and not pending_admission:
                    live_open = not admit_available(block_live=True)
                    if not live_open and not active and not pending_admission:
                        break
                else:
                    admit_available()
                if active:
                    continue
                time.sleep(0.01)
                continue
            if live_open:
                live_open = not admit_available()
            cycle_started = time.perf_counter()
            host_trace: dict[str, int | float] | None = None
            if round_count < self.config.profile_host_decode_steps:
                host_trace = {
                    "step": round_count,
                    "rank": self.rank,
                    "is_draft_rank": int(self.is_draft),
                    "cycle_start_seconds": cycle_started,
                }
            profile_this_round = round_count < self.config.profile_decode_steps
            profile_round_started = profile_begin(profile_this_round)
            profile_accounted_before = sum(decode_profile_seconds.values())
            scheduler_profile_started = time.perf_counter() if profile_this_round else None
            projected_wait_ms = max(last_cycle_ms * 2.0, 1e-6)
            roof_context_len = max(len(local_states[index].token_ids) for index in active)
            strict_profile = shaper.roofline if isinstance(shaper.roofline, ProfiledRoofline) else None
            unprofiled_target_only = False
            if (
                strict_profile is not None
                and strict_profile.metadata.get("unprofiled_execution_policy") == "target_only"
            ):
                measured_roof = strict_profile.lookup_optional(len(active), roof_context_len)
                if measured_roof is None:
                    verification_roof = 0
                    unprofiled_target_only = True
                else:
                    verification_roof = measured_roof
                    prospective_home = controller.next_target_home_batch_id
                    prospective_rows = [index for index in active if states[index].home_batch_id == prospective_home]
                    if not prospective_rows:
                        prospective_home = 1 - prospective_home
                        prospective_rows = [
                            index for index in active if states[index].home_batch_id == prospective_home
                        ]
                    if not strict_profile.covers_execution(
                        len(active),
                        roof_context_len,
                        len(prospective_rows),
                    ):
                        verification_roof = 0
                        unprofiled_target_only = True
            else:
                verification_roof = shaper.verification_roof(len(active), roof_context_len)
            if host_trace is not None:
                host_trace["scheduler_roof_end_seconds"] = time.perf_counter()
            if verification_roof == 0:
                # A zero is a measured AR-envelope result, never a missing
                # profile key.  Discard speculative state before running one
                # target-only logical home; no candidate is verified or
                # silently reconstructed from gamma.
                home = controller.next_target_home_batch_id
                target_indices = [index for index in active if states[index].home_batch_id == home]
                if not target_indices:
                    home = 1 - home
                    target_indices = [index for index in active if states[index].home_batch_id == home]
                if isinstance(shaper.roofline, ProfiledRoofline) and not unprofiled_target_only:
                    shaper.roofline.validate_execution(
                        len(active),
                        roof_context_len,
                        len(target_indices),
                    )
                complete_target_only_cycle(
                    target_indices,
                    cycle_started=cycle_started,
                    unprofiled=unprofiled_target_only,
                    target_home=home,
                    profile_this_round=profile_this_round,
                    profile_round_started=profile_round_started,
                    profile_accounted_before=profile_accounted_before,
                )
                continue
            shrink = self._bound_spec_rhythm_tree_ready(controller, payloads, cache_mappings, active, verification_roof)
            counters["spec_rhythm_tree_eager_invalidated"] += shrink["invalidated_eager"]
            counters["spec_rhythm_roof_pruned_candidates"] = (
                counters.get("spec_rhythm_roof_pruned_candidates", 0) + shrink["pruned_candidates"]
            )
            plan = controller.build_plan(
                active,
                verification_budget=verification_roof,
                priority=has_slo_constraints,
                projected_wait_ms=projected_wait_ms,
                priority_burst=self.config.spec_rhythm_priority_burst,
                merge_ready_homes=self.config.spec_rhythm_merge_ready_homes,
                max_target_requests=(
                    self.config.spec_rhythm_max_target_batch
                    if has_slo_constraints and self.config.spec_rhythm_max_target_batch > 0
                    else None
                ),
            )
            if host_trace is not None:
                host_trace["scheduler_plan_end_seconds"] = time.perf_counter()
            target_indices = list(plan.target_request_indices)
            if (
                target_indices
                and strict_profile is not None
                and strict_profile.metadata.get("unprofiled_execution_policy") == "target_only"
                and not strict_profile.covers_execution(
                    len(active),
                    roof_context_len,
                    len(target_indices),
                )
            ):
                complete_target_only_cycle(
                    target_indices,
                    cycle_started=cycle_started,
                    unprofiled=True,
                    target_home=plan.target_home_batch_id,
                    profile_this_round=profile_this_round,
                    profile_round_started=profile_round_started,
                    profile_accounted_before=profile_accounted_before,
                )
                continue
            target_payloads = [payloads[controller.ready[index].proposal_id] for index in target_indices]
            actual_candidates = sum(payload["plan"].candidate_budget for payload in target_payloads)
            if target_indices and isinstance(shaper.roofline, ProfiledRoofline):
                shaper.roofline.validate_execution(
                    len(active),
                    max(len(local_states[index].token_ids) for index in active),
                    len(target_indices),
                )
            if actual_candidates > shaper.verification_roof(
                len(active), max(len(local_states[index].token_ids) for index in active)
            ):
                raise RuntimeError("SpecRhythm actual verify input exceeds global B")
            counters["spec_rhythm_peak_verify_candidates"] = max(
                counters["spec_rhythm_peak_verify_candidates"], actual_candidates
            )
            for index, payload in zip(target_indices, target_payloads):
                ticket = controller.validate_verification(index)
                if ticket.proposal_id != payload["ticket"].proposal_id:
                    raise RuntimeError("SpecRhythm tree proposal identity changed")
                if payload["plan"].prefix_len != len(local_states[index].token_ids) - 1:
                    raise RuntimeError("SpecRhythm tree proposal has a stale committed prefix")
            normal = list(plan.normal_draft_request_indices)
            eager = list(plan.eager_candidate_indices)
            if effective_eager_cap <= 0:
                eager = []
            else:
                eager = [
                    index
                    for index in eager
                    if states[index].projected_progress_gap(projected_wait_ms) > 0
                    and states[index].urgency(projected_wait_ms) >= self.config.spec_rhythm_urgency_threshold
                    and states[index].expected_acceptance_benefit >= self.config.spec_rhythm_acceptance_floor
                    and (
                        int(payloads[controller.ready[index].proposal_id]["plan"].cache_positions.max().item())
                        + 2
                        + self.config.spec_rhythm_tree_width * self.config.spec_rhythm_tree_depth
                        <= self.config.max_model_len
                    )
                ]
            exploration_cost = self.config.spec_rhythm_tree_width * self.config.spec_rhythm_tree_depth
            window = window_estimator.estimate(
                normal_tokens=len(normal) * exploration_cost,
                max_draft_tokens=(len(normal) + len(eager)) * exploration_cost,
                eager_work=bool(eager),
            )
            # Optional eager requests must fit the residual measured window.
            # Charge the complete exploratory tree, including later discarded
            # nodes, rather than its cheaper refined verification subset.
            eager = sorted(
                eager,
                # The paper ranks eager admission by a_need * benefit, not
                # the TPOT ratio. This ordering must precede the W cutoff;
                # the shaper cannot recover a higher-gap request cut here.
                key=lambda index: (
                    states[index].projected_progress_gap(projected_wait_ms) * states[index].expected_acceptance_benefit,
                    states[index].urgency(projected_wait_ms),
                    -index,
                ),
                reverse=True,
            )
            # A configured reserve replaces complete normal proposal slots;
            # it does not add draft work beyond the pre-reserve normal home.
            # Count only reservations the scalar shaper can actually grant,
            # then expose that replacement capacity after W is calibrated.
            # Residual W capacity remains independently available above it.
            reserve_remaining = self.config.spec_rhythm_eager_reserve_tokens
            reserved_eager_rows = 0
            if reserve_remaining > 0:
                for index in eager:
                    cap = min(
                        shaper.max_gamma,
                        states[index].max_gamma or shaper.max_gamma,
                        effective_eager_cap,
                    )
                    minimum = min(shaper.min_gamma, cap)
                    if minimum <= 0 or minimum > reserve_remaining:
                        continue
                    reserved_eager_rows += 1
                    reserve_remaining -= minimum
            eager_window_tokens = window.eager_token_budget
            if window.calibrated and window.eager_fixed_overhead_hidden:
                eager_window_tokens += reserved_eager_rows * exploration_cost
            eager = eager[: eager_window_tokens // exploration_cost]
            budget_plan = shaper.shape(
                plan_id=plan.plan_id,
                normal_request_indices=normal,
                eager_request_indices=eager,
                states=states,
                projected_wait_ms=projected_wait_ms,
                context_len=max(len(local_states[index].token_ids) for index in active),
                draft_token_budget=self.config.spec_rhythm_draft_token_budget,
                batch_size=len(active),
                eager_token_cap=effective_eager_cap or None,
                eager_reserve_tokens=self.config.spec_rhythm_eager_reserve_tokens,
                verification_roof=verification_roof,
            )
            if host_trace is not None:
                host_trace["scheduler_budget_end_seconds"] = time.perf_counter()
            counters["spec_rhythm_last_verification_roof"] = budget_plan.verification_roof
            counters["spec_rhythm_unused_verification_tokens"] = int(budget_plan.unused_verification_tokens)
            normal_budgets = dict(budget_plan.normal_budgets)
            eager_budgets = dict(budget_plan.eager_budgets)
            work_indices = [*normal_budgets, *eager_budgets]
            work_budgets = [*normal_budgets.values(), *eager_budgets.values()]
            work_eager = [False] * len(normal_budgets) + [True] * len(eager_budgets)
            counters["spec_rhythm_unused_verification_tokens"] = max(
                0,
                int(budget_plan.verification_roof) - sum(normal_budgets.values()) - sum(eager_budgets.values()),
            )
            eager_parent_sources: dict[int, tuple[TreeSpeculationPlan, Sequence[int]]] = {}
            work_plans: list[TreeSpeculationPlan] = []
            for index, budget, is_eager in zip(work_indices, work_budgets, work_eager):
                if is_eager:
                    parent_ticket = controller.ready.get(index)
                    if parent_ticket is None:
                        raise RuntimeError("SpecRhythm eager tree request has no ready parent proposal")
                    parent_payload = payloads.get(parent_ticket.proposal_id)
                    if parent_payload is None:
                        raise RuntimeError("SpecRhythm eager tree parent payload is missing")
                    parent_plan = parent_payload["plan"]
                    parent_row = parent_payload["row"]
                    work_plans.append(self._spec_rhythm_tree_eager_plan(parent_plan, local_states[index], budget))
                    eager_parent_sources[index] = (parent_plan, parent_row)
                else:
                    work_plans.append(self._spec_rhythm_tree_plan(local_states[index], budget))
            tickets = [
                controller.new_ticket(index, gamma=budget, eager=is_eager)
                for index, budget, is_eager in zip(work_indices, work_budgets, work_eager)
            ]
            profile_detail_end(scheduler_profile_started, "scheduler_plan")
            if host_trace is not None:
                host_trace["scheduler_end_seconds"] = time.perf_counter()
                host_trace["target_requests"] = len(target_indices)
                host_trace["draft_requests"] = len(work_plans)
                host_trace["verify_candidates"] = actual_candidates
            # Keep instrumentation out of the public worker-call signature so
            # protocol harnesses and alternate workers remain drop-in
            # compatible. Native forwards consult this rank-local flag only
            # during an explicitly bounded profile window.
            self._profile_tree_subphases = profile_this_round
            # Protocol harnesses and alternate workers return already
            # materialized rows.  Deferred KV publication is a property of
            # the production native worker, not of the public loop protocol.
            self._defer_tree_materialization = native_tree_forward
            self._defer_normal_tree_materialization = native_tree_forward
            draft_started = time.perf_counter()
            draft_profile_started = profile_begin(profile_this_round and self.is_draft and bool(work_plans))
            draft_tree_kwargs: dict[str, Any] = {
                "eager_parent_sources": eager_parent_sources,
            }
            work_temperatures = [local_states[index].draft_temperature for index in work_indices]
            if any(value > 0 for value in work_temperatures):
                draft_tree_kwargs["draft_temperatures"] = work_temperatures
                work_top_ps = [local_states[index].draft_top_p for index in work_indices]
                work_top_ks = [local_states[index].draft_top_k for index in work_indices]
                if any(value < 1.0 for value in work_top_ps) or any(value > 0 for value in work_top_ks):
                    draft_tree_kwargs["draft_top_ps"] = work_top_ps
                    draft_tree_kwargs["draft_top_ks"] = work_top_ks
            if native_tree_forward:
                draft_tree_kwargs["committed_catchup_token_ids"] = [
                    (
                        local_states[index].token_ids[
                            max(
                                0,
                                len(local_states[index].token_ids) - self.config.spec_rhythm_tree_depth - 1,
                            ) :
                        ]
                        if not is_eager
                        else ()
                    )
                    for index, is_eager in zip(work_indices, work_eager)
                ]
                work_index_set = set(work_indices)
                draft_tree_kwargs["graph_padding_catchup_rows"] = [
                    (
                        index,
                        len(local_states[index].token_ids) - 1,
                        local_states[index].token_ids[
                            max(
                                0,
                                len(local_states[index].token_ids) - self.config.spec_rhythm_tree_depth - 1,
                            ) :
                        ],
                    )
                    for index in active
                    if index not in work_index_set
                ]
            with torch.profiler.record_function("SpecSLO/DraftCompute"):
                draft_output = (
                    self.draft_tree_forward(
                        work_plans,
                        [local_states[index].token_ids[-1] for index in work_indices],
                        work_indices,
                        **draft_tree_kwargs,
                    )
                    if work_plans
                    else None
                )
            if draft_output is not None:
                torch.npu.synchronize()
                counters["spec_rhythm_draft_model_calls"] += draft_output.get("model_calls", 0)
                counters["spec_rhythm_draft_graph_calls"] += draft_output.get("graph_calls", 0)
                counters["spec_rhythm_draft_graph_padding_queries"] += draft_output.get("graph_padding_query_count", 0)
                for phase, seconds in draft_output.get("profile_seconds", {}).items():
                    decode_profile_detail_seconds[phase] += float(seconds)
            # Role-local functions early-return on the other model's ranks.
            # Launch target(A) before receiving draft(B): both device groups
            # run independently until the step-end publication/commit fence.
            target_plans = [payload["plan"] for payload in target_payloads]
            target_rows = [payload["row"] for payload in target_payloads]
            target_roots = [local_states[index].token_ids[-1] for index in target_indices]
            target_request_ids = list(target_indices)
            real_target_count = len(target_plans)
            target_temperatures = [local_states[index].temperature for index in target_indices]
            if native_target_tree_forward and target_plans and not any(target_temperatures):
                (
                    target_execution_plans,
                    target_execution_roots,
                    target_execution_rows,
                    target_execution_ids,
                    real_target_count,
                ) = self._pad_spec_rhythm_target_tree_graph(
                    target_plans,
                    target_roots,
                    target_rows,
                    target_request_ids,
                    active,
                    local_states,
                )
            else:
                target_execution_plans = target_plans
                target_execution_roots = target_roots
                target_execution_rows = target_rows
                target_execution_ids = target_request_ids
            target_started = time.perf_counter()
            target_profile_started = profile_begin(profile_this_round and not self.is_draft and bool(target_indices))
            target_tree_kwargs: dict[str, Any] = {
                "sequence_ids": target_execution_ids,
                "return_logits": False,
            }
            if native_target_tree_forward:
                target_tree_kwargs["real_tree_count"] = real_target_count
            if any(value > 0 for value in target_temperatures):
                target_tree_kwargs["temperatures"] = target_temperatures
                target_top_ps = [local_states[index].top_p for index in target_indices]
                target_top_ks = [local_states[index].top_k for index in target_indices]
                if any(value < 1.0 for value in target_top_ps) or any(value > 0 for value in target_top_ks):
                    target_tree_kwargs["top_ps"] = target_top_ps
                    target_tree_kwargs["top_ks"] = target_top_ks
            with torch.profiler.record_function("SpecSLO/TargetCompute"):
                target_output = (
                    self.target_tree_forward(
                        target_execution_plans,
                        target_execution_roots,
                        target_execution_rows,
                        **target_tree_kwargs,
                    )
                    if target_indices
                    else None
                )
            profile_end(target_profile_started, "target_compute")
            phase_seconds["target"] += time.perf_counter() - target_started
            if target_output is not None:
                torch.npu.synchronize()
                counters["spec_rhythm_target_query_tokens"] += target_output["query_count"]
                counters["spec_rhythm_target_graph_padding_queries"] += target_output.get(
                    "graph_padding_query_count", 0
                )
                for phase, seconds in target_output.get("profile_seconds", {}).items():
                    decode_profile_detail_seconds[phase] += float(seconds)
            target_ended = time.perf_counter()
            draft_postprocess_profile_started = profile_begin(profile_this_round)
            self._validate_spec_rhythm_target_graph(target_output, has_target_work=bool(target_indices))
            local_rows = None if draft_output is None else draft_output["draft_token_ids"]
            selected_indices = None
            confidences = None
            if draft_output is not None:
                selected_indices = []
                selected_rows = []
                confidences = []
                exploratory = {}
                for row_id, (tree_plan, row) in enumerate(zip(work_plans, local_rows)):
                    node_confidence = draft_output["node_confidences"][row_id]
                    index = work_indices[row_id]
                    runtime = states[index]
                    cap = runtime.max_gamma or shaper.max_gamma
                    if work_eager[row_id]:
                        cap = min(cap, effective_eager_cap)
                    exploratory[index] = TreeCandidateRequest(
                        parents=tree_plan.parent_indices.detach().cpu().tolist(),
                        token_ids=row,
                        conditional_confidences=node_confidence.detach().cpu().tolist(),
                        max_candidates=min(cap, tree_plan.width * tree_plan.depth),
                        progress_gap=budget_plan.progress_gaps.get(index, 0),
                        urgency=max(0.0, runtime.urgency(projected_wait_ms)),
                        acceptance_rate=runtime.acceptance_ema,
                        minimum_candidates=0 if work_eager[row_id] else 1,
                    )
                with _trace_region(self, "SpecSLO/GlobalCandidateSelection"):
                    refined = select_global_tree_candidates(exploratory, budget_plan.verification_roof)
                for row_id, index in enumerate(work_indices):
                    indices = refined.selected_indices[index]
                    selected_indices.append(list(indices))
                    selected_rows.append([local_rows[row_id][node] for node in indices])
                    confidences.append(
                        sum(exploratory[index].conditional_confidences[node] for node in indices) / len(indices)
                        if indices
                        else 0.0
                    )
                if draft_output.get("materialization_deferred", False):
                    eager_rows = [row for row, is_eager in enumerate(work_eager) if is_eager]
                    with _trace_region(self, "SpecSLO/SelectedDraftKVMaterialize"):
                        materialized = self.materialize_selected_tree_kv(
                            [work_plans[row] for row in eager_rows],
                            [draft_output["draft_token_ids"][row] for row in eager_rows],
                            [selected_indices[row] for row in eager_rows],
                            [work_indices[row] for row in eager_rows],
                        )
                    if materialized is not None:
                        torch.npu.synchronize()
                        counters["spec_rhythm_draft_model_calls"] += materialized.get("model_calls", 0)
                        counters["spec_rhythm_draft_graph_calls"] += materialized.get("graph_calls", 0)
                        counters["spec_rhythm_draft_materialized_nodes"] += int(
                            materialized.get("materialized_nodes", 0)
                        )
                        for phase, seconds in materialized.get("profile_seconds", {}).items():
                            decode_profile_detail_seconds[phase] += float(seconds)
                local_rows = selected_rows
            profile_detail_end(draft_postprocess_profile_started, "draft_postprocess")
            profile_end(draft_profile_started, "draft_compute")
            draft_ended = time.perf_counter()
            if host_trace is not None:
                host_trace["draft_start_seconds"] = draft_started
                host_trace["draft_end_seconds"] = draft_ended
                host_trace["target_start_seconds"] = target_started
                host_trace["target_end_seconds"] = target_ended
            if self.is_draft:
                phase_seconds["draft"] += draft_ended - draft_started
            exchange_started = time.perf_counter()
            exchange_profile_started = profile_begin(profile_this_round and bool(work_plans))
            frontier_rows = (
                [draft_output["eager_frontier_tokens"].get(index) for index in work_indices]
                if draft_output is not None
                else None
            )
            with torch.profiler.record_function("SpecSLO/DraftToTarget"):
                message = self._exchange_spec_rhythm_tree_candidates(
                    local_rows,
                    work_plans,
                    frontier_rows,
                    confidences,
                    selected_indices=selected_indices,
                    draft_compute_ms=(
                        (draft_ended - draft_started) * 1000.0
                        if self.rank == self.topology.draft_leader_rank and work_plans
                        else 0.0
                    ),
                )
            profile_end(exchange_profile_started, "draft_to_target_communication")
            exchange_ended = time.perf_counter()
            phase_seconds["exchange"] += exchange_ended - exchange_started
            if host_trace is not None:
                host_trace["exchange_start_seconds"] = exchange_started
                host_trace["exchange_end_seconds"] = exchange_ended
            draft_ms = 0.0
            candidate_rows = confidence_rows = selected_rows = None
            if work_plans:
                assert message is not None
                split_started = time.perf_counter()
                if self.rank == self.topology.draft_leader_rank:
                    # The draft leader already owns the exact Python rows used
                    # to construct the device broadcast. Reading that tensor
                    # straight back with ``message.cpu()`` serialized 15--42
                    # ms of otherwise overlapped target work every cycle.
                    # Other ranks still decode the collective envelope; later
                    # collectives retain the same cross-rank ordering fence.
                    assert local_rows is not None
                    assert frontier_rows is not None
                    assert confidences is not None
                    assert selected_indices is not None
                    candidate_rows = local_rows
                    confidence_rows = confidences
                    selected_rows = selected_indices
                    draft_ms = (draft_ended - draft_started) * 1000.0
                else:
                    (
                        candidate_rows,
                        frontier_rows,
                        confidence_rows,
                        selected_rows,
                        draft_ms,
                    ) = self._split_tree_candidates(message, work_plans)
                if host_trace is not None:
                    host_trace["publish_candidate_split_seconds"] = time.perf_counter() - split_started
            if profile_this_round:
                timeline_profile_started = profile_begin(True)
                timing_values = [
                    (draft_ended - draft_started) * 1000.0
                    if self.rank == self.topology.draft_leader_rank and work_plans
                    else 0.0,
                    (target_ended - target_started) * 1000.0
                    if self.rank == self.topology.target_leader_rank and target_indices
                    else 0.0,
                    draft_started if self.rank == self.topology.draft_leader_rank and work_plans else 0.0,
                    draft_ended if self.rank == self.topology.draft_leader_rank and work_plans else 0.0,
                    target_started if self.rank == self.topology.target_leader_rank and target_indices else 0.0,
                    target_ended if self.rank == self.topology.target_leader_rank and target_indices else 0.0,
                ]
                # The global timestamp envelope exists only for explicit
                # profiling. Production W observations ride the existing
                # proposal/verdict messages and add no collective here.
                timing = torch.tensor(
                    [round(value * (1000 if index < 2 else 1_000_000)) for index, value in enumerate(timing_values)],
                    dtype=torch.int64,
                    device=self.device,
                )
                dist.all_reduce(timing)
                _, _, ds, de, ts, te = [
                    value / (1000 if index < 2 else 1_000_000) for index, value in enumerate(timing.cpu().tolist())
                ]
                decode_timeline.append(
                    {
                        "step": round_count,
                        "draft_start": ds,
                        "draft_end": de,
                        "target_start": ts,
                        "target_end": te,
                        "host_compute_window_overlap_ms": max(0.0, min(de, te) - max(ds, ts)) * 1000.0,
                        "verify_candidates": actual_candidates,
                        "verify_requests": len(target_indices),
                        "draft_requests": len(work_plans),
                        "residual_draft_window_ms": window.residual_window_ms,
                        "eager_requests": len(eager_budgets),
                    }
                )
                profile_detail_end(timeline_profile_started, "profile_timeline_collective")
            publish_profile_started = profile_begin(profile_this_round)
            if work_plans:
                publish_pack_seconds = 0.0
                publish_slot_mapping_seconds = 0.0
                publish_payload_seconds = 0.0
                publish_kv_move_seconds = 0.0
                publish_flat_mapping_seconds = 0.0
                publish_controller_seconds = 0.0
                counters["spec_rhythm_allocated_draft_tokens"] += sum(int(plan.candidate_budget) for plan in work_plans)
                assert candidate_rows is not None
                assert confidence_rows is not None
                assert selected_rows is not None
                mapping = None if draft_output is None else draft_output["cache_slot_mapping"]
                cursor = 0
                published_tickets = []
                pending_draft_mappings: list[tuple[int, torch.Tensor]] = []
                normal_publish_sources: list[torch.Tensor] = []
                normal_publish_destinations: list[torch.Tensor] = []
                defer_normal_tree_materialization = bool(getattr(self, "_defer_normal_tree_materialization", False))
                for work_row, (index, ticket, tree_plan, row) in enumerate(
                    zip(work_indices, tickets, work_plans, candidate_rows)
                ):
                    expected = int(tree_plan.width) * int(tree_plan.depth) + 1
                    if not row:
                        cursor += expected
                        continue
                    ticket.gamma = len(row)
                    published_tickets.append(ticket)
                    selection = selected_rows[work_row]
                    publish_part_started = time.perf_counter()
                    packed_plan = pack_selected_tree_plan(tree_plan, selection)
                    publish_pack_seconds += time.perf_counter() - publish_part_started
                    deferred_normal = not ticket.eager and defer_normal_tree_materialization
                    if self.is_draft and deferred_normal:
                        counters["spec_rhythm_draft_publish_rows"] += 1
                        counters["spec_rhythm_draft_publish_skipped_rows"] += 1
                        # Normal draft K/V is deliberately rebuilt from
                        # committed catch-up tokens next turn, so it owns no
                        # physical mapping until that materialization.
                    elif self.is_draft and mapping is not None:
                        identity_selection = list(selection) == list(range(len(selection)))
                        if identity_selection:
                            # Spine-first prefix selection is already stored
                            # in exact packed order. Keep a view instead of
                            # launching index_select.
                            source = mapping[cursor : cursor + len(selection) + 1]
                        else:
                            source = mapping[cursor : cursor + expected].index_select(
                                0,
                                torch.tensor(
                                    [0, *[node + 1 for node in selection]],
                                    dtype=torch.long,
                                    device=self.device,
                                ),
                            )
                        if not ticket.eager and identity_selection:
                            counters["spec_rhythm_draft_publish_rows"] += 1
                            counters["spec_rhythm_draft_publish_skipped_rows"] += 1
                        elif not ticket.eager:
                            counters["spec_rhythm_draft_publish_rows"] += 1
                            publish_part_started = time.perf_counter()
                            destination = torch.tensor(
                                self._cache_slot_mapping(
                                    [index] * (len(selection) + 1),
                                    list(
                                        range(
                                            packed_plan.prefix_len,
                                            packed_plan.prefix_len + len(selection) + 1,
                                        )
                                    ),
                                ),
                                dtype=torch.int32,
                                device=self.device,
                            )
                            publish_slot_mapping_seconds += time.perf_counter() - publish_part_started
                            normal_publish_sources.append(source)
                            normal_publish_destinations.append(destination)
                            source = destination
                            counters["spec_rhythm_draft_publish_moved_rows"] += 1
                        if source is not None:
                            pending_draft_mappings.append((ticket.proposal_id, source))
                    elif self.is_draft:
                        raise RuntimeError("materialized draft tree is missing its KV mapping")
                    eager_metadata: dict[str, Any] = {}
                    publish_part_started = time.perf_counter()
                    if ticket.eager:
                        parent_ticket = controller.ready.get(index)
                        parent_payload = None if parent_ticket is None else payloads.get(parent_ticket.proposal_id)
                        if parent_ticket is None or parent_payload is None:
                            raise RuntimeError("SpecRhythm eager tree parent disappeared before publish")
                        parent_plan = parent_payload["plan"]
                        parent_row = parent_payload["row"]
                        path_len = self._tree_primary_path_length(parent_plan)
                        eager_metadata = {
                            "eager_parent_proposal_id": parent_ticket.proposal_id,
                            "eager_dependency_tokens": tuple(
                                int(parent_row[node]) for node in tree_primary_path(parent_plan)
                            ),
                            "eager_frontier_token": (frontier_rows[work_row] if frontier_rows is not None else None),
                            "eager_dependency_length": path_len,
                        }
                    payloads[ticket.proposal_id] = {
                        "ticket": ticket,
                        "plan": packed_plan,
                        "row": row,
                        "confidence": confidence_rows[work_row],
                        "draft_kv_materialized": not deferred_normal,
                        **eager_metadata,
                    }
                    publish_payload_seconds += time.perf_counter() - publish_part_started
                    cursor += expected
                # The exploratory trees share a model-lifetime layer layout.
                # Move all normal proposals with one K/V layer sweep instead
                # of one complete sweep per request. Eager proposals retain
                # their ahead-of-turn slots until promotion validation.
                if normal_publish_sources:
                    publish_part_started = time.perf_counter()
                    self._move_tree_cache_slots(
                        torch.cat(normal_publish_sources),
                        torch.cat(normal_publish_destinations),
                    )
                    publish_kv_move_seconds += time.perf_counter() - publish_part_started
                publish_part_started = time.perf_counter()
                cache_mappings.update(pending_draft_mappings)
                controller.publish(published_tickets)
                counters["spec_rhythm_tree_nodes"] += sum(ticket.gamma for ticket in published_tickets)
                counters["spec_rhythm_unused_verification_tokens"] = budget_plan.verification_roof - sum(
                    ticket.gamma for ticket in published_tickets
                )
                publish_controller_seconds += time.perf_counter() - publish_part_started
                if host_trace is not None:
                    host_trace.update(
                        {
                            "publish_pack_seconds": publish_pack_seconds,
                            "publish_slot_mapping_seconds": publish_slot_mapping_seconds,
                            "publish_payload_seconds": publish_payload_seconds,
                            "publish_kv_move_seconds": publish_kv_move_seconds,
                            "publish_flat_mapping_seconds": publish_flat_mapping_seconds,
                            "publish_controller_seconds": publish_controller_seconds,
                        }
                    )
            profile_detail_end(publish_profile_started, "proposal_publish")
            if host_trace is not None:
                host_trace["publish_end_seconds"] = time.perf_counter()

            if not target_indices:
                accounting_profile_started = profile_begin(profile_this_round)
                dist.barrier()
                elapsed_tensor = torch.tensor(
                    [time.perf_counter() - cycle_started if self.rank == self.topology.target_leader_rank else 0.0],
                    dtype=torch.float64,
                    device=self.device,
                )
                dist.broadcast(elapsed_tensor, src=self.topology.target_leader_rank)
                last_cycle_ms = float(elapsed_tensor.cpu().item()) * 1000.0
                for index in active:
                    states[index].add_decode_time(last_cycle_ms)
                profile_detail_end(accounting_profile_started, "cycle_accounting")
                if host_trace is not None:
                    host_trace["cycle_end_seconds"] = time.perf_counter()
                    host_timeline.append(host_trace)
                finish_profile_round(profile_round_started, profile_accounted_before)
                if profile_this_round:
                    profiled_decode_steps += 1
                round_count += 1
                if npu_profiler is not None:
                    npu_profiler.step()
                continue
            target_verdict: TreeVerificationOutput | None = None
            verdict_host_started = time.perf_counter() if host_trace is not None else None
            verdict_profile_started = profile_begin(
                profile_this_round and self.rank == self.topology.target_leader_rank
            )
            if self.rank == self.topology.target_leader_rank:
                assert target_output is not None
                if self.config.spec_rhythm_cpu_verdict:
                    # Target rows already contain the root query and every
                    # active-node successor, including the accepted-frontier
                    # bonus.  Copy that one compact tensor to the host, walk
                    # the tiny trees as Python integers, then publish both
                    # result matrices with one H2D transfer.  The former CPU
                    # diagnostic rebuilt several torch tensors, launched the
                    # generic per-row tensor verifier and copied two results
                    # back separately (~4 ms at B=8).
                    target_verdict = self._verify_tree_outputs_host(
                        target_rows,
                        target_plans,
                        target_output["target_query_token_ids"],
                        max(int(plan.depth) for plan in target_plans),
                        self.device,
                    )
                else:
                    target_parents = torch.cat(
                        [plan.parent_indices[: int(plan.candidate_budget)].to(self.device) for plan in target_plans]
                    )
                    draft_tokens = torch.tensor(
                        [
                            token
                            for row, plan in zip(target_rows, target_plans)
                            for token in row[: int(plan.candidate_budget)]
                        ],
                        dtype=torch.long,
                        device=self.device,
                    )
                    target_verdict = self.verify_tree_outputs(
                        draft_tokens,
                        target_parents,
                        target_output["target_query_token_ids"],
                        target_output["bonus_token_ids"],
                        [int(plan.candidate_budget) for plan in target_plans],
                        max(int(plan.depth) for plan in target_plans),
                    )
            profile_end(verdict_profile_started, "target_verdict")
            if host_trace is not None:
                host_trace["verdict_start_seconds"] = verdict_host_started or 0.0
                host_trace["verdict_end_seconds"] = time.perf_counter()
            broadcast_started = time.perf_counter()
            correction_profile_started = profile_begin(profile_this_round)
            with torch.profiler.record_function("SpecSLO/TargetToDraft"):
                output_rows, accepted_rows, target_ms = self._broadcast_spec_rhythm_tree_verdict(
                    target_verdict,
                    target_plans,
                    (
                        (target_ended - target_started) * 1000.0
                        if self.rank == self.topology.target_leader_rank
                        else 0.0
                    ),
                )
            profile_end(correction_profile_started, "target_to_draft_communication")
            broadcast_ended = time.perf_counter()
            phase_seconds["broadcast"] += broadcast_ended - broadcast_started
            if host_trace is not None:
                host_trace["correction_start_seconds"] = broadcast_started
                host_trace["correction_end_seconds"] = broadcast_ended
            window_estimator.observe(
                draft_compute_ms=draft_ms,
                drafted_tokens=len(work_plans) * exploration_cost,
                target_verify_ms=target_ms,
                eager_work=bool(eager_budgets),
            )
            state_started = time.perf_counter()
            state_profile_started = profile_begin(profile_this_round)
            finished: list[int] = []
            preflight_profile_started = time.perf_counter() if profile_this_round else None
            # Validate the whole step before mutating its first request.
            # Capture local errors so every rank still reaches the same vote.
            # A stale row or bad mapping on one rank must not let its peers
            # commit tokens, move accepted KV, or emit a streaming chunk.
            preflight_error: Exception | None = None
            pending_cache_mappings: dict[int, torch.Tensor] = {}
            try:
                for row, index in enumerate(target_indices):
                    ticket = controller.validate_verification(index)
                    if ticket.proposal_id != target_payloads[row]["ticket"].proposal_id:
                        raise RuntimeError("SpecRhythm tree verdict references a different proposal")
                    plan_for_row = target_plans[row]
                    if plan_for_row.prefix_len != len(local_states[index].token_ids) - 1:
                        raise RuntimeError("SpecRhythm tree verdict has a stale prefix")
                    accepted_path = [value for value in accepted_rows[row] if value >= 0]
                    tokens = [value for value in output_rows[row] if value >= 0]
                    if len(tokens) != len(accepted_path) + 1:
                        raise RuntimeError("SpecRhythm tree verdict must include one frontier token")
                    parents = plan_for_row.parent_indices.detach().cpu().tolist()
                    parent = -1
                    for depth, node in enumerate(accepted_path):
                        if node >= len(parents) or parents[node] != parent:
                            raise RuntimeError("SpecRhythm verdict path is not ancestor closed")
                        if tokens[depth] != target_rows[row][node]:
                            raise RuntimeError("SpecRhythm verdict token does not match its proposal")
                        parent = node
                    staged = controller.staged_eager.get(index)
                    if staged is not None and staged.proposal_id not in payloads:
                        raise RuntimeError("SpecRhythm staged eager tree payload disappeared")
                (
                    pending_cache_mappings,
                    preflight_host_cache_mappings,
                ) = self._preflight_spec_rhythm_tree_cache_commit(
                    controller,
                    payloads,
                    cache_mappings,
                    target_indices,
                    target_plans,
                    target_output,
                    return_host_mappings=True,
                )
            except Exception as error:
                preflight_error = error
            if host_trace is not None:
                host_trace["state_preflight_end_seconds"] = time.perf_counter()
            profile_detail_end(preflight_profile_started, "commit_preflight")
            consensus_profile_started = time.perf_counter() if profile_this_round else None
            preflight_failed = torch.tensor(
                [int(preflight_error is not None)],
                dtype=torch.int64,
                device=self.device,
            )
            # Fold the graph-resident numerical health bit into the existing
            # commit vote. The former ``flag.item()`` in local preflight added
            # a second device/host fence in every decode cycle.
            preflight_failed = torch.maximum(
                preflight_failed,
                self._spec_rhythm_nonfinite_flag(include_layer_cache=False).reshape(1).to(dtype=torch.int64),
            )
            dist.all_reduce(preflight_failed, op=dist.ReduceOp.MAX)
            # One scalar synchronization at the collective commit boundary.
            # Earlier forward/transport failures abort the worker, and later
            # hardware/OOM failures during KV moves are not transactional.
            preflight_failed_value = int(preflight_failed.cpu().item())
            if host_trace is not None:
                host_trace["state_consensus_end_seconds"] = time.perf_counter()
            profile_detail_end(consensus_profile_started, "commit_consensus")
            if preflight_failed_value or preflight_error is not None:
                if preflight_error is not None:
                    raise RuntimeError(
                        f"SpecRhythm tree commit preflight failed on rank {self.rank}: {preflight_error}"
                    ) from preflight_error
                raise RuntimeError(
                    "SpecRhythm tree commit preflight failed on another rank; this step was not committed "
                    "(the shared vote also carries numerical-health failures)"
                )
            compaction_profile_started = time.perf_counter() if profile_this_round else None
            cache_mappings.update(pending_cache_mappings)
            # Compact every verified request in one fixed-shape layer sweep.
            # The former per-request call launched K/V gather+scatter for all
            # model layers N times per cycle (512 scatter launches at B8/TP3).
            # Preflight has already validated all mappings, so batching here
            # preserves the same commit boundary and physical destinations.
            compact_proposal_ids = [payload["ticket"].proposal_id for payload in target_payloads]
            rows_to_compact = [
                row
                for row, plan in enumerate(target_plans)
                if not (
                    self.is_draft
                    and getattr(self, "_defer_normal_tree_materialization", False)
                    and not target_payloads[row]["ticket"].eager
                )
                and self._tree_row_requires_kv_compaction(
                    accepted_rows[row],
                    int(plan.candidate_budget),
                )
            ]
            counters["spec_rhythm_kv_compaction_rows"] += len(rows_to_compact)
            counters["spec_rhythm_kv_compaction_skipped_rows"] += len(target_plans) - len(rows_to_compact)
            if rows_to_compact:
                if any(compact_proposal_ids[row] not in preflight_host_cache_mappings for row in rows_to_compact):
                    raise RuntimeError("SpecRhythm compacted tree is missing its preflighted KV mapping")
                self.compact_tree_round(
                    None,
                    [accepted_rows[row] for row in rows_to_compact],
                    [int(target_plans[row].candidate_budget) for row in rows_to_compact],
                    host_slot_rows=[
                        preflight_host_cache_mappings[compact_proposal_ids[row]] for row in rows_to_compact
                    ],
                )
                counters["spec_rhythm_kv_compaction_rounds"] += 1
            if host_trace is not None:
                host_trace["state_compaction_end_seconds"] = time.perf_counter()
            profile_detail_end(compaction_profile_started, "kv_compaction")
            request_commit_profile_started = time.perf_counter() if profile_this_round else None
            promoted_cache_rows: list[tuple[int, int, TreeSpeculationPlan]] = []
            for row, index in enumerate(target_indices):
                tree_plan = target_plans[row]
                was_first_decode_token = states[index].delivered_tokens == 0
                state_for_row = local_states[index]
                committed_tokens = _truncate_completion(
                    [value for value in output_rows[row] if value >= 0],
                    self.eos_token_ids,
                    state_for_row.max_tokens - len(state_for_row.committed_completion_token_ids),
                    state_for_row.ignore_eos,
                )
                will_finish = len(state_for_row.committed_completion_token_ids) + len(
                    committed_tokens
                ) >= state_for_row.max_tokens or (
                    not state_for_row.ignore_eos and any(token in self.eos_token_ids for token in committed_tokens)
                )
                accepted_count = sum(value >= 0 for value in accepted_rows[row])
                proposed_count = int(tree_plan.candidate_budget)
                fully_accepted = accepted_count >= self._tree_primary_path_length(tree_plan)
                if fully_accepted:
                    counters["spec_rhythm_tree_full_accepts"] += 1
                else:
                    counters["spec_rhythm_tree_rejections"] += 1
                counters["spec_rhythm_tree_accepted_nodes"] += accepted_count
                counters["spec_rhythm_verified_tokens"] += proposed_count
                current_ticket = target_payloads[row]["ticket"]
                controller.validate_verification(index)
                eager_ticket = controller.staged_eager.get(index)
                eager_valid = fully_accepted
                if eager_ticket is not None:
                    eager_payload = payloads.get(eager_ticket.proposal_id)
                    if eager_payload is None:
                        raise RuntimeError("SpecRhythm staged eager tree payload disappeared")
                    dependency_length = int(eager_payload.get("eager_dependency_length", 0))
                    dependency_tokens = tuple(int(value) for value in eager_payload.get("eager_dependency_tokens", ()))
                    observed_path = tuple(int(value) for value in output_rows[row][:dependency_length])
                    observed_frontier = (
                        int(output_rows[row][dependency_length]) if dependency_length < len(output_rows[row]) else -1
                    )
                    eager_frontier = eager_payload.get("eager_frontier_token")
                    eager_valid = (
                        eager_valid
                        and eager_payload.get("eager_parent_proposal_id") == current_ticket.proposal_id
                        and dependency_length > 0
                        and observed_path == dependency_tokens
                        and eager_frontier is not None
                        and observed_frontier == int(eager_frontier)
                    )
                    if not eager_valid:
                        counters["spec_rhythm_tree_eager_dependency_rejections"] += 1
                promoted = controller.finish_verification(
                    index,
                    fully_accepted=eager_valid and not will_finish,
                    proposed_tokens=proposed_count,
                    accepted_tokens=accepted_count,
                    delivered_tokens=len(committed_tokens),
                    draft_confidence=target_payloads[row]["confidence"],
                    ema_alpha=self.config.spec_rhythm_acceptance_ema_alpha,
                )
                for state in (draft_states[index], target_states[index]):
                    state.apply_tree_verification(
                        output_token_ids=committed_tokens,
                        accepted=accepted_count,
                        proposed_tokens=proposed_count,
                    )
                delivered = len(committed_tokens)
                if was_first_decode_token and delivered > 0:
                    first_decode_indices.add(index)
                payloads.pop(current_ticket.proposal_id, None)
                if current_ticket.proposal_id in cache_mappings:
                    cache_mappings.pop(current_ticket.proposal_id)
                if eager_ticket is not None and promoted is None:
                    counters["spec_rhythm_tree_eager_invalidated"] += 1
                    payloads.pop(eager_ticket.proposal_id, None)
                    cache_mappings.pop(eager_ticket.proposal_id, None)
                elif eager_ticket is not None:
                    counters["spec_rhythm_tree_eager_promoted"] += 1
                    if self.is_draft:
                        eager_plan = payloads[eager_ticket.proposal_id]["plan"]
                        promoted_cache_rows.append((eager_ticket.proposal_id, index, eager_plan))
                if _finished(local_states[index], self.eos_token_ids):
                    finished.append(index)
                self._deliver_committed_tokens(index, request_params[index], local_states[index])
            if promoted_cache_rows:
                moved_slots = self._move_promoted_tree_cache_slots(
                    promoted_cache_rows,
                    cache_mappings,
                    preflight_host_cache_mappings,
                )
                counters["spec_rhythm_tree_eager_promotion_kv_move_rounds"] += 1
                counters["spec_rhythm_tree_eager_promotion_kv_move_rows"] += len(promoted_cache_rows)
                counters["spec_rhythm_tree_eager_promotion_kv_move_slots"] += moved_slots
            controller.finish_cycle(plan.target_home_batch_id)
            if host_trace is not None:
                host_trace["state_request_commit_end_seconds"] = time.perf_counter()
            profile_detail_end(request_commit_profile_started, "request_commit")
            phase_seconds["state_update"] += time.perf_counter() - state_started
            profile_end(state_profile_started, "state_update")
            if host_trace is not None:
                host_trace["state_end_seconds"] = time.perf_counter()
            accounting_profile_started = time.perf_counter() if profile_this_round else None
            elapsed_tensor = torch.tensor(
                [time.perf_counter() - cycle_started if self.rank == self.topology.target_leader_rank else 0.0],
                dtype=torch.float64,
                device=self.device,
            )
            dist.broadcast(elapsed_tensor, src=self.topology.target_leader_rank)
            last_cycle_ms = float(elapsed_tensor.cpu().item()) * 1000.0
            for index in active:
                states[index].add_decode_time(last_cycle_ms)
            # Decode begins when the first prefill token is delivered at
            # admission, not after the first multi-token verification cycle.
            first_decode_indices.clear()
            for index in finished:
                completed[index] = local_states[index].clone()
                finished_decode_elapsed[index] = states[index].decode_elapsed_ms
                for ticket in (controller.ready.get(index), controller.staged_eager.get(index)):
                    if ticket is not None:
                        payloads.pop(ticket.proposal_id, None)
                        cache_mappings.pop(ticket.proposal_id, None)
                controller.invalidate_request(index)
            active = [index for index in active if index not in set(finished)]
            # Live arrivals are polled at the next loop entrance immediately
            # after this cycle boundary.  Polling here as well repeats the
            # clock broadcast and device-to-host read with no useful gap in
            # which a newly admitted request could execute.  Static pending
            # admissions have no callback, so retain their refill point.
            if not live_open:
                admit_available(poll_live=False)
            profile_detail_end(accounting_profile_started, "cycle_accounting")
            counters["spec_rhythm_tree_rounds"] += 1
            if host_trace is not None:
                host_trace["state_start_seconds"] = state_started
                host_trace["cycle_end_seconds"] = time.perf_counter()
                host_timeline.append(host_trace)
            finish_profile_round(profile_round_started, profile_accounted_before)
            if profile_this_round:
                profiled_decode_steps += 1
            round_count += 1
            if npu_profiler is not None:
                npu_profiler.step()
            if self.config.stop_after_profiled_decode_steps and round_count >= self.config.profile_decode_steps:
                break
        torch.npu.synchronize()
        if npu_profiler is not None:
            npu_profiler.stop()
            self._active_tree_profiler = None
        self._profile_tree_subphases = False
        self._defer_tree_materialization = False
        decode_elapsed = time.perf_counter() - started
        self.last_worker_decode_phase_seconds = phase_seconds
        self.last_worker_decode_profile_seconds = dict(decode_profile_seconds)
        self.last_worker_decode_profile_detail_seconds = dict(decode_profile_detail_seconds)
        self.last_worker_decode_host_timeline = host_timeline
        self.last_worker_profiled_decode_steps = profiled_decode_steps
        self.last_worker_decode_counters = counters
        self._release_cache()
        if self.rank != self.topology.target_leader_rank:
            return None
        results: list[dict[str, Any]] = []
        for index, state in enumerate(target_states):
            state = completed.get(index, state)
            completion = _truncate_completion(
                state.committed_completion_token_ids,
                self.eos_token_ids,
                state.max_tokens,
                state.ignore_eos,
            )
            request_decode_elapsed_ms = finished_decode_elapsed.get(index, states[index].decode_elapsed_ms)
            measured_decode_elapsed_ms = max(
                0.0,
                request_decode_elapsed_ms - (states[index].decode_start_elapsed_ms or 0.0),
            )
            observed_tpot_ms = measured_decode_elapsed_ms / max(1, len(completion) - 1)
            paper_tpot_ms = measured_decode_elapsed_ms / max(1, len(completion))
            slo_attained = None if state.slo_tpot_ms is None else observed_tpot_ms <= state.slo_tpot_ms
            finish_reason = "abort" if state.aborted else "length" if len(completion) >= state.max_tokens else "stop"
            results.append(
                {
                    "completion_token_ids": completion,
                    "request_id": request_params[index].request_id,
                    "accepted_draft_tokens": state.accepted_draft_tokens,
                    "verified_draft_tokens": state.verified_draft_tokens,
                    "verification_rounds": state.verification_rounds,
                    "num_acc_tokens": state.acceptance_lengths,
                    "mean_accept_tokens": (
                        sum(state.acceptance_lengths) / len(state.acceptance_lengths)
                        if state.acceptance_lengths
                        else 0.0
                    ),
                    "acceptance_rate": (
                        state.accepted_draft_tokens / state.verified_draft_tokens
                        if state.verified_draft_tokens
                        else 0.0
                    ),
                    "temperature": state.temperature,
                    "draft_temperature": state.draft_temperature,
                    "top_p": state.top_p,
                    "top_k": state.top_k,
                    "draft_top_p": state.draft_top_p,
                    "draft_top_k": state.draft_top_k,
                    "max_tokens": state.max_tokens,
                    "ignore_eos": state.ignore_eos,
                    "slo_tpot_ms": state.slo_tpot_ms,
                    "slo_class": state.slo_class,
                    "arrival_ts": request_params[index].arrival_ts,
                    "finish_reason": finish_reason,
                    "observed_tpot_ms": observed_tpot_ms,
                    "tpot_definition": "decode_after_first_token_ms / (output_tokens - 1)",
                    "paper_tpot_ms": paper_tpot_ms,
                    "paper_tpot_definition": "same_decode_elapsed_ms / output_tokens",
                    "paper_slo_attained": (None if state.slo_tpot_ms is None else paper_tpot_ms <= state.slo_tpot_ms),
                    "paper_slo_goodput_tokens": (
                        len(completion) if state.slo_tpot_ms is None or paper_tpot_ms <= state.slo_tpot_ms else 0
                    ),
                    "slo_attained": slo_attained,
                    "slo_goodput_tokens": len(completion) if slo_attained is not False else 0,
                    "round_count": round_count,
                    "decode_phase_seconds": dict(phase_seconds),
                    "spec_rhythm": dict(counters),
                    "elapsed_seconds": prefill_elapsed + decode_elapsed,
                    "prefill_elapsed_seconds": prefill_elapsed,
                    "decode_elapsed_seconds": decode_elapsed,
                    "tree_mode": True,
                    "gamma": self.gamma,
                    "decode_timeline": decode_timeline,
                    "tree_width": self.config.spec_rhythm_tree_width,
                    "tree_depth": self.config.spec_rhythm_tree_depth,
                    **self.graph_metrics(),
                }
            )
        return results

    def _generate_spec_rhythm_decode(
        self,
        *,
        draft_states: list[PearlPipelineState],
        target_states: list[PearlPipelineState],
        request_params: Sequence[NativeSamplingParams],
        initial_batch_size: int,
        continuous_batching: bool,
        prefill_elapsed: float,
        started: float,
        max_rounds: int | None,
        prefilled_indices: set[int] | None = None,
    ) -> list[dict[str, Any]] | None:
        """Execute the guarded dual-batch pipeline from the SpecRhythm paper."""

        if self.config.spec_rhythm_tree_width > 1 or self.config.spec_rhythm_tree_depth > 1:
            return self._generate_spec_rhythm_tree_decode(
                draft_states=draft_states,
                target_states=target_states,
                request_params=request_params,
                initial_batch_size=initial_batch_size,
                continuous_batching=continuous_batching,
                prefill_elapsed=prefill_elapsed,
                started=started,
                max_rounds=max_rounds,
                prefilled_indices=prefilled_indices,
            )

        runtime_states = {
            index: SpecRhythmRuntimeState(
                request_index=index,
                home_batch_id=index % 2,
                slo_tpot_ms=params.slo_tpot_ms,
                slo_class=params.slo_class,
                max_gamma=params.spec_rhythm_max_gamma,
            )
            for index, params in enumerate(request_params)
        }
        controller = SpecRhythmPipelineController(runtime_states)
        shaper = SpecRhythmBudgetShaper(
            min_gamma=self.config.spec_rhythm_min_gamma,
            max_gamma=self.gamma,
            verification_budget=self.config.spec_rhythm_verification_budget,
            acceptance_floor=self.config.spec_rhythm_acceptance_floor,
            acceptance_ema_alpha=self.config.spec_rhythm_acceptance_ema_alpha,
            roofline=self.config.spec_rhythm_roofline,
        )
        # A request carrying a TPOT/class constraint is a full SpecRhythm
        # request.  The paper's rolling-eager and urgency stages are part of
        # that contract; leaving them disabled by a zero-valued optional cap
        # silently degrades the request to ordinary PEARL rotation.  Keep the
        # explicit cap as an override, while resolving the default to one
        # bounded gamma window only for SLO-constrained requests.
        has_slo_constraints = any(
            params.slo_tpot_ms is not None or params.slo_class is not None for params in request_params
        )
        ablation_mode = self.config.spec_rhythm_ablation_mode
        serial_ablation = ablation_mode == "serial"
        nano_pearl_ablation = ablation_mode == "nano_pearl"
        single_batch_ablation = serial_ablation or nano_pearl_ablation
        dual_no_rolling_ablation = ablation_mode == "dual_batch"
        dual_rolling_ablation = ablation_mode == "dual_batch_rolling"
        explicit_ablation = ablation_mode != "auto"
        pard_eager_qualification = self.config.draft_mode == PARD_PARALLEL_DRAFT_MODE
        if pard_eager_qualification or serial_ablation or dual_no_rolling_ablation:
            effective_eager_cap = 0
        elif nano_pearl_ablation or dual_rolling_ablation:
            effective_eager_cap = self.gamma
        else:
            effective_eager_cap = self.config.spec_rhythm_max_eager_tokens or (self.gamma if has_slo_constraints else 0)
        effective_priority = (
            has_slo_constraints
            if dual_rolling_ablation
            else False
            if explicit_ablation
            else self.config.spec_rhythm_priority_mode or has_slo_constraints
        )
        linear_full_window = self.config.spec_rhythm_linear_full_window
        linear_bonus_token = self.config.spec_rhythm_linear_bonus_token
        if linear_full_window and any(
            state.temperature != 0 or state.draft_temperature != 0 for state in (*draft_states, *target_states)
        ):
            raise ValueError(
                "Linear full-window verification currently supports greedy target and draft sampling only."
            )
        if linear_full_window and any(
            params.spec_rhythm_max_gamma is not None and params.spec_rhythm_max_gamma < self.gamma
            for params in request_params
        ):
            raise ValueError("Linear full-window verification cannot shorten a request below the fixed gamma.")
        linear_fixed_gamma_window = (
            has_slo_constraints
            and self.config.spec_rhythm_min_gamma == self.gamma
            and (effective_eager_cap == self.gamma or explicit_ablation)
            and all(
                params.spec_rhythm_max_gamma is None or params.spec_rhythm_max_gamma >= self.gamma
                for params in request_params
            )
        )
        fixed_gamma_scheduler_fast_path = bool(
            linear_full_window
            and linear_fixed_gamma_window
            and self.config.spec_rhythm_verification_budget is None
            and self.config.spec_rhythm_draft_token_budget is None
            and self.config.spec_rhythm_eager_reserve_tokens == 0
            and isinstance(shaper.roofline, Mapping)
            and not shaper.roofline
            and not envs.VLLM_ASCEND_SPECRHYTHM_VALIDATE_MAILBOX
        )
        linear_window_estimator = (
            DraftWindowEstimator(self.config.spec_rhythm_acceptance_ema_alpha)
            if linear_fixed_gamma_window and not (serial_ablation or nano_pearl_ablation or dual_no_rolling_ablation)
            else None
        )
        # Timing events are reused across cycles and queried only after an
        # existing result materialization has already fenced the corresponding
        # stream.  They add no explicit per-cycle NPU synchronize.
        draft_window_events = (
            (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
            if linear_fixed_gamma_window and self.device.type == "npu" and self.rank == self.topology.draft_leader_rank
            else None
        )
        target_window_events = (
            (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
            if linear_fixed_gamma_window and self.device.type == "npu" and self.rank == self.topology.target_leader_rank
            else None
        )
        # Query draft event timing one cycle later, after the normal protocol
        # fences have already completed it.  Reading the just-recorded event
        # before posting the proposal broadcast serializes the host on every
        # cycle and destroys the overlap this estimator is meant to protect.
        draft_window_event_pending = False
        draft_window_event_token_count = 0
        if has_slo_constraints or linear_full_window:
            # SpecRhythm verifies one physical home batch at a time, and rows
            # leave the service as soon as they hit their completion limit.
            # The graph key already contains the complete FIA query shape, so
            # retaining an initial-batch equality guard would force every
            # half-batch and short tail (32 -> ... -> 1 rows) through eager
            # execution. Allow shape-specific capture/replay in this
            # control-plane path while leaving ordinary PEARL's fixed-batch
            # guard unchanged.
            self.graph_runner.set_expected_fia_batch_size(None)
        local_states = draft_states if self.is_draft else target_states
        active_indices: list[int] = []
        pending_admission = list(range(len(local_states)))
        completed_states: dict[int, PearlPipelineState] = {}
        payloads: dict[int, NativeSpecRhythmDevicePayload] = {}
        prefetched = set(prefilled_indices or ())
        cache_activated = set(prefilled_indices or ())
        cache_released: set[int] = set()

        def activate_online_cache(request_indices: Sequence[int]) -> None:
            """Lazily bind physical KV pages when an online prompt is served."""

            if not self.config.spec_rhythm_online_prefill:
                return
            for index in request_indices:
                if index in cache_activated:
                    continue
                self._activate_cache_sequence(
                    index,
                    local_states[index].token_ids,
                    enable_prefix_caching=self.config.enable_prefix_caching,
                )
                cache_activated.add(index)
                counters["spec_rhythm_online_cache_activated_sequences"] += 1

        def release_online_cache(request_indices: Sequence[int]) -> None:
            """Return completed online rows to the shared physical page pool."""

            if not self.config.spec_rhythm_online_prefill:
                return
            for index in request_indices:
                if index not in cache_activated or index in cache_released:
                    continue
                released_blocks = self._release_cache_sequence(index)
                cache_released.add(index)
                counters["spec_rhythm_online_cache_released_sequences"] += 1
                counters["spec_rhythm_online_cache_released_blocks"] += released_blocks

        # Online fixed-gamma serial serving can pipeline a newly arrived
        # prompt behind the model work already selected for the current
        # cycle.  Keep this deliberately narrow: with one draft rank the
        # verification ranks are the complete distributed world, so the
        # existing CPU coordination group can safely order the following
        # cross-model token broadcast after both role-local prefills.  Other
        # topologies retain the synchronous admission path.
        staged_prefill_overlap = bool(
            linear_full_window and self.config.spec_rhythm_online_prefill and len(self.topology.draft_ranks) == 1
        )
        gloo_accounting_requested = bool(envs.VLLM_ASCEND_SPECRHYTHM_GLOO_ACCOUNTING)
        # With a TP1 draft, verification_ranks is exactly the complete PEARL
        # world (draft leader + every target rank). Only that topology may
        # replace the two WORLD/HCCL accounting publications with the existing
        # CPU/Gloo coordination group without changing collective membership.
        gloo_accounting = bool(gloo_accounting_requested and len(self.topology.draft_ranks) == 1)
        accounting_group = getattr(self.groups, "verification_coordination_group", None) if gloo_accounting else None
        if gloo_accounting and accounting_group is None:
            raise RuntimeError("SpecRhythm Gloo accounting requires the verification coordination process group.")
        mixed_target_prefill = bool(
            staged_prefill_overlap
            and envs.VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_PREFILL
            and not self.config.target_use_paged_attention
        )
        mixed_draft_prefill = bool(
            staged_prefill_overlap
            and envs.VLLM_ASCEND_SPECRHYTHM_MIXED_DRAFT_PREFILL
            and self.config.draft_use_paged_attention
            and self.config.precompile_serial_draft_graphs
            and self.gamma > 1
        )
        overlap_draft_prefill = bool(
            staged_prefill_overlap
            and envs.VLLM_ASCEND_SPECRHYTHM_OVERLAP_DRAFT_PREFILL
            and self.device.type == "npu"
            and self.config.draft_use_paged_attention
            and self.config.precompile_serial_draft_graphs
        )
        if mixed_draft_prefill and overlap_draft_prefill:
            raise ValueError("Mixed and side-stream draft prefill are mutually exclusive.")
        if mixed_draft_prefill and envs.VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_BUCKET:
            raise ValueError(
                "Bucketed linear-draft FIA and mixed draft prefill are "
                "mutually exclusive: the former owns two home-batch graph "
                "lanes per shape while the latter adds a third tail graph."
            )
        draft_prefill_stream = None
        if overlap_draft_prefill and self.is_draft and self.device.type == "npu":
            draft_prefill_stream = getattr(
                self,
                "_spec_rhythm_draft_prefill_stream",
                None,
            )
            if draft_prefill_stream is None:
                draft_prefill_stream = torch.npu.Stream()
                self._spec_rhythm_draft_prefill_stream = draft_prefill_stream
        prefill_coalescing = bool(
            self.config.spec_rhythm_prefill_coalesce_min_requests > 1
            and self.config.spec_rhythm_prefill_coalesce_max_wait_ms > 0
        )
        prefill_token_chunk_size = int(self.config.spec_rhythm_prefill_token_chunk_size)
        token_chunk_prefill = prefill_token_chunk_size > 0
        staged_prefill_indices: list[int] = []
        staged_prefill_tokens: list[int] | None = None
        staged_prefill_cursors: dict[int, int] = {}
        staged_prefill_target_tokens: dict[int, int] = {}
        # Target-leader monotonic timestamps at the common boundary following
        # first-token publication.  Rows remain here only while a refill tail
        # still has to be charged to the newly activated request.
        activation_tail_starts: dict[int, float] = {}
        last_cycle_ms = 0.0
        round_count = 0
        phase_seconds = {
            "draft": 0.0,
            "target": 0.0,
            "exchange": 0.0,
            "verify": 0.0,
            "broadcast": 0.0,
            "state_update": 0.0,
            "refill": 0.0,
        }
        # Rank-local timestamps deliberately add neither an NPU fence nor a
        # collective.  The benchmark joins these rows by ``step`` to show
        # whether the draft and target ranks submitted their disjoint model
        # work before the proposal exchange rendezvous.  They are a host-side
        # protocol trace; an NPU profiler trace is still required to prove
        # physical kernel overlap.
        host_timeline: list[dict[str, int | float]] = []
        counters = {
            "spec_rhythm_warmup_steps": 0,
            "spec_rhythm_steady_steps": 0,
            "spec_rhythm_drain_steps": 0,
            "spec_rhythm_normal_proposals": 0,
            "spec_rhythm_eager_proposals": 0,
            "spec_rhythm_eager_promoted": 0,
            "spec_rhythm_eager_invalidated": 0,
            "spec_rhythm_allocated_draft_tokens": 0,
            "spec_rhythm_verified_tokens": 0,
            "spec_rhythm_prefill_batches": 0,
            "spec_rhythm_prefill_requests": 0,
            "spec_rhythm_online_cache_lazy_enabled": int(self.config.spec_rhythm_online_prefill),
            "spec_rhythm_online_cache_activated_sequences": 0,
            "spec_rhythm_online_cache_released_sequences": 0,
            "spec_rhythm_online_cache_released_blocks": 0,
            "spec_rhythm_online_refill_ms": 0.0,
            "spec_rhythm_staged_prefill_overlap_enabled": int(staged_prefill_overlap),
            "spec_rhythm_staged_prefill_overlap_batches": 0,
            "spec_rhythm_staged_prefill_overlap_requests": 0,
            "spec_rhythm_staged_prefill_overlap_window_ms": 0.0,
            "spec_rhythm_staged_prefill_reserved_batches": 0,
            "spec_rhythm_staged_prefill_reserved_requests": 0,
            "spec_rhythm_mixed_target_prefill_enabled": int(mixed_target_prefill),
            "spec_rhythm_mixed_target_prefill_batches": 0,
            "spec_rhythm_mixed_target_prefill_requests": 0,
            "spec_rhythm_mixed_target_prefill_verification_tokens": 0,
            "spec_rhythm_mixed_target_graph_enabled": int(getattr(self, "_mixed_target_graph_enabled", False)),
            "spec_rhythm_mixed_target_graph_batches": 0,
            "spec_rhythm_mixed_target_graph_requests": 0,
            "spec_rhythm_mixed_target_graph_prompt_tokens": 0,
            "spec_rhythm_mixed_target_graph_bounded_shape_fallback_batches": 0,
            "spec_rhythm_mixed_target_graph_token_capacity_fallback_batches": 0,
            "spec_rhythm_mixed_target_graph_execution_fallback_batches": 0,
            "spec_rhythm_stable_target_verify_graph_batches": 0,
            "spec_rhythm_stable_target_verify_graph_requests": 0,
            "spec_rhythm_stable_target_verify_graph_bounded_shape_fallback_batches": 0,
            "spec_rhythm_stable_target_verify_graph_token_capacity_fallback_batches": 0,
            "spec_rhythm_stable_target_verify_graph_execution_fallback_batches": 0,
            "spec_rhythm_mixed_draft_prefill_enabled": int(mixed_draft_prefill),
            "spec_rhythm_mixed_draft_prefill_batches": 0,
            "spec_rhythm_mixed_draft_prefill_requests": 0,
            "spec_rhythm_mixed_draft_prefill_proposal_tokens": 0,
            "spec_rhythm_overlap_draft_prefill_enabled": int(overlap_draft_prefill),
            "spec_rhythm_overlap_draft_prefill_batches": 0,
            "spec_rhythm_overlap_draft_prefill_requests": 0,
            "spec_rhythm_overlap_draft_prefill_proposal_tokens": 0,
            "spec_rhythm_prefill_coalesce_enabled": int(prefill_coalescing),
            "spec_rhythm_prefill_coalesce_deferred_polls": 0,
            "spec_rhythm_prefill_coalesce_size_releases": 0,
            "spec_rhythm_prefill_coalesce_timeout_releases": 0,
            "spec_rhythm_prefill_coalesce_forced_releases": 0,
            "spec_rhythm_prefill_coalesce_max_wait_observed_ms": 0.0,
            "spec_rhythm_prefill_token_chunk_enabled": int(token_chunk_prefill),
            "spec_rhythm_prefill_token_chunk_size": prefill_token_chunk_size,
            "spec_rhythm_prefill_token_chunk_submissions": 0,
            "spec_rhythm_prefill_token_chunk_tokens": 0,
            "spec_rhythm_prefill_token_chunk_partial_submissions": 0,
            "spec_rhythm_prefill_token_chunk_completed_rows": 0,
            "spec_rhythm_prefill_token_chunk_max_submission_tokens": 0,
            "spec_rhythm_prefill_singleton_batches": 0,
            "spec_rhythm_prefill_pair_batches": 0,
            "spec_rhythm_prefill_multi_batches": 0,
            "spec_rhythm_prefill_rank_safe_budget_enabled": int(self.config.spec_rhythm_online_prefill),
            "spec_rhythm_first_token_clock_broadcasts": 0,
            "spec_rhythm_new_request_tail_segments": 0,
            "spec_rhythm_new_request_tail_ms": 0.0,
            "spec_rhythm_new_request_tail_max_ms": 0.0,
            "spec_rhythm_initial_activation_carries": 0,
            "spec_rhythm_admission_polls": 0,
            "spec_rhythm_admission_batches": 0,
            "spec_rhythm_admitted_requests": 0,
            "spec_rhythm_first_ready_requests": 0,
            "spec_rhythm_last_admission_batch_size": 0,
            "spec_rhythm_max_admission_batch_size": 0,
            "spec_rhythm_initial_now": 0.0,
            "spec_rhythm_initial_min_arrival_delta_ms": 0.0,
            "spec_rhythm_initial_max_arrival_delta_ms": 0.0,
            "spec_rhythm_last_verification_roof": 0,
            "spec_rhythm_unused_verification_tokens": 0,
            "spec_rhythm_serial_protocol": int(serial_ablation),
            "spec_rhythm_single_batch_overlap_protocol": int(nano_pearl_ablation),
            "spec_rhythm_dual_batch_overlap_protocol": int(not single_batch_ablation),
            "spec_rhythm_serial_draft_only_cycles": 0,
            "spec_rhythm_serial_target_only_cycles": 0,
            "spec_rhythm_concurrent_draft_target_submission_cycles": 0,
            "spec_rhythm_single_batch_overlap_submission_cycles": 0,
            "spec_rhythm_dual_batch_overlap_submission_cycles": 0,
            "spec_rhythm_target_home_0_cycles": 0,
            "spec_rhythm_target_home_1_cycles": 0,
            "spec_rhythm_gloo_accounting_requested": int(gloo_accounting_requested),
            "spec_rhythm_gloo_accounting_active": int(gloo_accounting),
            "spec_rhythm_linear_full_window": int(linear_full_window),
            "spec_rhythm_linear_bonus_token": int(linear_bonus_token),
            "spec_rhythm_fixed_gamma_scheduler_fast_path": int(fixed_gamma_scheduler_fast_path),
            "spec_rhythm_fixed_gamma_scheduler_fast_path_cycles": 0,
            "spec_rhythm_linear_rebuilt_requests": 0,
            "spec_rhythm_linear_discarded_ready": 0,
            "spec_rhythm_linear_invalidated_eager": 0,
            "spec_rhythm_linear_discarded_unverified_tokens": 0,
            "spec_rhythm_linear_bonus_eligible_rows": 0,
            "spec_rhythm_linear_bonus_committed_tokens": 0,
            "spec_rhythm_linear_bonus_draft_kv_prepared_rows": 0,
            "spec_rhythm_linear_bonus_suppressed_eager_rows": 0,
            "spec_rhythm_linear_bonus_suppressed_output_limit_rows": 0,
            "spec_rhythm_compact_proposal_broadcasts": 0,
            "spec_rhythm_compact_proposal_int64_saved": 0,
            "spec_rhythm_linear_w_enabled": int(linear_window_estimator is not None),
            "spec_rhythm_linear_w_lagged_draft_timing": int(draft_window_events is not None),
            "spec_rhythm_linear_w_calibrated_steps": 0,
            "spec_rhythm_linear_w_eligible_eager_rows": 0,
            "spec_rhythm_linear_w_admitted_eager_rows": 0,
            "spec_rhythm_linear_idle_residual_eligible_rows": 0,
            "spec_rhythm_linear_idle_residual_admitted_rows": 0,
            "spec_rhythm_linear_w_deferred_eager_rows": 0,
            "spec_rhythm_linear_w_bucket_deferred_eager_rows": 0,
            "spec_rhythm_linear_w_cross_bucket_candidate_rows": 0,
            "spec_rhythm_linear_w_bucket_blocked_eager_rows": 0,
            "spec_rhythm_linear_w_cross_bucket_admitted_eager_rows": 0,
            "spec_rhythm_linear_w_cross_graph_bucket_enabled": int(
                self.config.spec_rhythm_linear_eager_cross_graph_bucket
            ),
            "spec_rhythm_linear_w_paid_headroom_eager_rows": 0,
            "spec_rhythm_linear_w_last_eager_row_cap": 0,
            "spec_rhythm_linear_w_last_normal_bucket": 0,
            "spec_rhythm_linear_w_last_bucket_headroom": 0,
            "spec_rhythm_linear_w_last_cross_bucket_candidate_rows": 0,
            "spec_rhythm_linear_w_last_window_ms": 0.0,
            "spec_rhythm_linear_w_last_residual_ms": 0.0,
            "spec_rhythm_linear_w_last_predicted_exposed_ms": 0.0,
            "spec_rhythm_linear_w_last_draft_ms_per_token": 0.0,
            "spec_rhythm_linear_w_last_observed_draft_ms": 0.0,
            "spec_rhythm_linear_w_last_observed_target_ms": 0.0,
            "spec_rhythm_cycle_accounted_tail_ms": 0.0,
            "spec_rhythm_cycle_accounted_tail_cycles": 0,
        }
        decode_profile_seconds = {
            "draft_compute": 0.0,
            "draft_to_target_communication": 0.0,
            "target_compute": 0.0,
            "target_verdict": 0.0,
            "target_to_draft_communication": 0.0,
            "wait_sync": 0.0,
            "state_update": 0.0,
        }
        profiled_decode_steps = 0
        npu_profiler = None
        npu_profile_dir = envs.VLLM_ASCEND_PEARL_NPU_PROFILE_DIR
        if npu_profile_dir:
            configured_rank = envs.VLLM_ASCEND_PEARL_NPU_PROFILE_RANK
            npu_profile_rank = self.topology.target_leader_rank if configured_rank is None else configured_rank
            if npu_profile_rank != -1 and not 0 <= npu_profile_rank < self.topology.world_size:
                raise ValueError("VLLM_ASCEND_PEARL_NPU_PROFILE_RANK must be -1 or identify a PEARL worker.")
            if npu_profile_rank == -1 or self.rank == npu_profile_rank:
                import torch_npu

                npu_profiler = torch_npu.profiler.profile(
                    activities=[
                        torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU,
                    ],
                    schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=3, repeat=1),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        npu_profile_dir,
                        worker_name=(f"spec-rhythm-{'draft' if self.is_draft else 'target'}-rank-{self.rank}"),
                    ),
                    record_shapes=True,
                )
                npu_profiler.start()

        def synchronized_wall_time() -> float:
            value = torch.tensor(
                [time.time() if self.rank == self.topology.target_leader_rank else 0.0],
                dtype=torch.float64,
                device=self.device,
            )
            dist.broadcast(value, src=self.topology.target_leader_rank)
            return float(value.cpu().item())

        def select_ready(now: float) -> list[int]:
            """Select one arrival-safe prefill batch without mutating service state."""

            counters["spec_rhythm_admission_polls"] += 1
            if counters["spec_rhythm_admission_polls"] == 1:
                counters["spec_rhythm_initial_now"] = float(now)
                arrival_deltas = [
                    (float(request_params[index].arrival_ts) - now) * 1000.0
                    for index in pending_admission
                    if request_params[index].arrival_ts is not None
                ]
                if arrival_deltas:
                    counters["spec_rhythm_initial_min_arrival_delta_ms"] = min(arrival_deltas)
                    counters["spec_rhythm_initial_max_arrival_delta_ms"] = max(arrival_deltas)
            capacity = initial_batch_size - len(active_indices) - len(staged_prefill_indices)
            if capacity <= 0:
                return []
            # The paper's decode-stage evaluation assumes that prefill has
            # already completed before a request enters this scheduler.  In
            # that mode arrival_ts is trace metadata, not a future admission
            # gate.  Only the explicit online-prefill mode should replay the
            # trace clock and hold requests until their arrival.
            # Online-prefill owns the arrival clock even when an offered load
            # under-fills the physical service batch.  Tying this gate to the
            # queue-overflow definition of ``continuous_batching`` admitted a
            # 40-request/B64 trace all at once and made a 12-second replay
            # appear to finish in only its model execution time.
            arrival_gated = bool(self.config.spec_rhythm_online_prefill)
            request_limit = min(
                capacity,
                self.config.prefill_chunk_size or self.config.max_num_seqs,
            )
            mixed_graph_admission = bool(
                envs.VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH and not self.config.enforce_eager and active_indices
            )
            mixed_graph_prompt_budget = max(self._mixed_target_graph_prompt_buckets) if mixed_graph_admission else None
            if mixed_graph_admission:
                # The fixed envelope owns exactly four prompt partitions.
                # Apply the same replicated cap on every rank before staging;
                # otherwise a coalesced release can be valid service work but
                # force the target-only direct eager fallback.
                request_limit = min(
                    request_limit,
                    MIXED_TARGET_PROMPT_CAPACITY,
                )
            staged_set = set(staged_prefill_indices)
            ready_indices: list[int] = []
            packed_tokens = 0
            token_budget_saturated = False
            for index in pending_admission:
                if index in staged_set:
                    continue
                arrival = request_params[index].arrival_ts
                if arrival_gated and arrival is not None and arrival > now:
                    continue
                prefill_budget_tokens = 0
                if index not in prefetched:
                    # Draft and target ranks own different model KV caches;
                    # prefix-cache hit counts therefore are not a replicated
                    # scheduler input.  Budget the complete prompt instead.
                    # This is conservative (actual uncached work is never
                    # larger) and makes ready selection rank-deterministic.
                    prefill_budget_tokens = 0 if token_chunk_prefill else int(local_states[index].prompt_length)
                if ready_indices and (
                    len(ready_indices) >= request_limit
                    or packed_tokens + prefill_budget_tokens > self.config.max_num_batched_tokens
                    or (
                        mixed_graph_prompt_budget is not None
                        and packed_tokens + prefill_budget_tokens > mixed_graph_prompt_budget
                    )
                ):
                    token_budget_saturated = (
                        packed_tokens + prefill_budget_tokens > self.config.max_num_batched_tokens
                        or (
                            mixed_graph_prompt_budget is not None
                            and packed_tokens + prefill_budget_tokens > mixed_graph_prompt_budget
                        )
                    )
                    break
                if (
                    not ready_indices
                    and mixed_graph_prompt_budget is not None
                    and prefill_budget_tokens > mixed_graph_prompt_budget
                ):
                    raise ValueError("An online SpecSLO prompt exceeds the largest resident mixed-target graph bucket.")
                if prefill_budget_tokens > self.config.max_num_batched_tokens:
                    raise ValueError("An online PEARL prefill request exceeds max_num_batched_tokens.")
                ready_indices.append(index)
                packed_tokens += prefill_budget_tokens
                if len(ready_indices) >= request_limit:
                    break
            if not ready_indices or not prefill_coalescing:
                return ready_indices
            remaining = sum(index not in staged_set for index in pending_admission)
            ready_arrivals = [request_params[index].arrival_ts for index in ready_indices]
            decision = decide_prefill_coalesce(
                ready_count=len(ready_indices),
                remaining_count=remaining,
                active_request_count=len(active_indices),
                request_limit=request_limit,
                minimum_requests=(self.config.spec_rhythm_prefill_coalesce_min_requests),
                maximum_wait_ms=(self.config.spec_rhythm_prefill_coalesce_max_wait_ms),
                now=now,
                ready_arrival_times=ready_arrivals,
                token_budget_saturated=token_budget_saturated,
            )
            if decision.waited_ms is not None:
                counters["spec_rhythm_prefill_coalesce_max_wait_observed_ms"] = max(
                    counters["spec_rhythm_prefill_coalesce_max_wait_observed_ms"],
                    decision.waited_ms,
                )
            if decision.reason == "size":
                counters["spec_rhythm_prefill_coalesce_size_releases"] += 1
            elif decision.reason == "timeout":
                counters["spec_rhythm_prefill_coalesce_timeout_releases"] += 1
            elif decision.release:
                counters["spec_rhythm_prefill_coalesce_forced_releases"] += 1
            else:
                counters["spec_rhythm_prefill_coalesce_deferred_polls"] += 1
            return ready_indices if decision.release else []

        def record_prefill_batch(size: int) -> None:
            if size <= 0:
                return
            if size == 1:
                counters["spec_rhythm_prefill_singleton_batches"] += 1
            elif size == 2:
                counters["spec_rhythm_prefill_pair_batches"] += 1
            else:
                counters["spec_rhythm_prefill_multi_batches"] += 1

        def add_new_request_tail(index: int, elapsed_ms: float) -> None:
            """Charge only time after this request's first token was published."""

            elapsed_ms = max(0.0, float(elapsed_ms))
            if elapsed_ms == 0.0:
                return
            runtime_states[index].add_decode_time(elapsed_ms)
            counters["spec_rhythm_new_request_tail_segments"] += 1
            counters["spec_rhythm_new_request_tail_ms"] += elapsed_ms
            counters["spec_rhythm_new_request_tail_max_ms"] = max(
                counters["spec_rhythm_new_request_tail_max_ms"], elapsed_ms
            )

        def charge_activation_tail(endpoint: float) -> float:
            """Close refill-tail intervals using target-leader monotonic time."""

            total_ms = 0.0
            active_set = set(active_indices)
            for index, started_at in tuple(activation_tail_starts.items()):
                if index not in active_set:
                    continue
                elapsed_ms = max(0.0, (endpoint - started_at) * 1000.0)
                add_new_request_tail(index, elapsed_ms)
                total_ms += elapsed_ms
            activation_tail_starts.clear()
            return total_ms

        def take_activation_cycle_start(default: float) -> float:
            """Carry an idle/bootstrap activation boundary into cycle timing."""

            if not activation_tail_starts:
                return default
            boundary = max(activation_tail_starts.values())
            counters["spec_rhythm_initial_activation_carries"] += len(activation_tail_starts)
            activation_tail_starts.clear()
            # Only the target leader contributes cycle elapsed time.  Other
            # ranks retain a process-local host-trace origin.
            if self.rank == self.topology.target_leader_rank:
                return boundary
            return default

        def decode_finish_envelope(
            cycle_indices: Sequence[int],
            finish_offsets: Sequence[float],
            cycle_elapsed: float,
            *,
            label: str,
        ) -> set[int]:
            """Validate one fixed-capacity target-leader finish envelope."""

            if len(finish_offsets) != initial_batch_size:
                raise RuntimeError(f"{label} finish envelope has the wrong capacity.")
            finished: set[int] = set()
            for slot, finish_offset in enumerate(finish_offsets):
                occupied = slot < len(cycle_indices)
                if not occupied:
                    if finish_offset != -1.0:
                        raise RuntimeError(f"{label} finish envelope populated a padding slot.")
                    continue
                if finish_offset == -1.0:
                    continue
                if not math.isfinite(finish_offset) or finish_offset < 0.0 or finish_offset > cycle_elapsed:
                    raise RuntimeError(f"{label} finish envelope contains an invalid endpoint.")
                request_index = int(cycle_indices[slot])
                if request_index in finished:
                    raise RuntimeError(f"{label} finish envelope contains a duplicate request.")
                finished.add(request_index)
            return finished

        def activate_ready(
            ready_indices: Sequence[int],
            target_tokens: Sequence[int],
        ) -> None:
            """Atomically publish prefilled rows at a cycle boundary."""

            nonlocal prefetched
            ready_indices = list(ready_indices)
            prefill_indices = [index for index in ready_indices if index not in prefetched]
            if len(target_tokens) != len(prefill_indices):
                raise RuntimeError("Online PEARL prefill returned the wrong number of first tokens.")
            if len(set(ready_indices)) != len(ready_indices) or any(
                index not in pending_admission or index in active_indices for index in ready_indices
            ):
                raise RuntimeError("Online PEARL admission attempted to publish a stale reservation.")
            # Validate every row before the first token/state mutation.  A
            # failed local prefill may have written disposable KV slots, but
            # no request becomes visible or partially committed.
            for index in prefill_indices:
                for state in (draft_states[index], target_states[index]):
                    if state.committed_length != state.prompt_length or len(state.token_ids) != state.prompt_length:
                        raise RuntimeError(
                            "Online PEARL prefill cannot publish a request whose prompt frontier already changed."
                        )

            token_by_index = dict(zip(prefill_indices, map(int, target_tokens)))
            for request_index in prefill_indices:
                target_token = token_by_index[request_index]
                for state in (
                    draft_states[request_index],
                    target_states[request_index],
                ):
                    state.token_ids.append(target_token)
                    assert state.committed_length is not None
                    state.committed_length += 1
            prefetched.update(prefill_indices)

            ready_set = set(ready_indices)
            pending_admission[:] = [index for index in pending_admission if index not in ready_set]
            continuing_indices = [
                index for index in ready_indices if not _finished(local_states[index], self.eos_token_ids)
            ]
            finished_ready = set(ready_indices) - set(continuing_indices)
            home_counts = [
                sum(runtime_states[index].home_batch_id == home for index in active_indices) for home in (0, 1)
            ]
            home_by_index: dict[int, int] = {}
            for index in continuing_indices:
                home = 0 if home_counts[0] <= home_counts[1] else 1
                home_by_index[index] = home
                home_counts[home] += 1
            active_indices.extend(continuing_indices)
            publication_values: list[float] = []
            for index in ready_indices:
                home = home_by_index.get(index)
                if home is not None:
                    runtime_states[index].home_batch_id = home
                runtime_states[index].delivered_tokens = len(local_states[index].committed_completion_token_ids)
                if self.rank == self.topology.target_leader_rank:
                    # The callback invocation is the externally visible first
                    # token event (and uses the same pre-callback convention as
                    # _deliver_committed_tokens.elapsed_seconds).  Timestamp
                    # it before calling user code so delivery work is not
                    # silently dropped between TTFT and the next token.
                    publication_values.extend((time.time(), time.perf_counter()))
                self._deliver_committed_tokens(
                    index,
                    request_params[index],
                    local_states[index],
                )

            publication_boundary = 0.0
            if self.config.spec_rhythm_online_prefill:
                if self.rank == self.topology.target_leader_rank:
                    publication_values.append(time.perf_counter())
                else:
                    publication_values = [0.0] * (2 * len(ready_indices) + 1)
                publication_clock = torch.tensor(
                    publication_values,
                    dtype=torch.float64,
                    device=self.device,
                )
                dist.broadcast(
                    publication_clock,
                    src=self.topology.target_leader_rank,
                )
                publication_values = [float(value) for value in publication_clock.cpu().tolist()]
                publication_boundary = publication_values[-1]
                counters["spec_rhythm_first_token_clock_broadcasts"] += 1

            continuing_set = set(continuing_indices)
            for offset, index in enumerate(ready_indices):
                runtime = runtime_states[index]
                arrival = request_params[index].arrival_ts
                if self.config.spec_rhythm_online_prefill:
                    publication_wall = publication_values[2 * offset]
                    publication_mono = publication_values[2 * offset + 1]
                    if arrival is not None:
                        runtime.arrival_wait_ms = max(
                            runtime.arrival_wait_ms,
                            (publication_wall - arrival) * 1000.0,
                        )
                else:
                    publication_mono = 0.0
                # Start TPOT strictly after this row's first-token callback.
                # The per-row publication-to-common-boundary interval is then
                # charged explicitly; subsequent refill work is closed by the
                # existing cycle-tail collective.
                runtime.decode_start_elapsed_ms = runtime.decode_elapsed_ms
                if self.config.spec_rhythm_online_prefill and index in continuing_set:
                    add_new_request_tail(
                        index,
                        (publication_boundary - publication_mono) * 1000.0,
                    )
                    activation_tail_starts[index] = publication_boundary
                if index in finished_ready:
                    local_states[index].finished_decode_elapsed_ms = runtime_states[index].decode_elapsed_ms
                    completed_states[index] = local_states[index].clone()
                    controller.invalidate_request(index)
            release_online_cache(tuple(finished_ready))

            admission_size = len(ready_indices)
            if counters["spec_rhythm_admission_batches"] == 0:
                counters["spec_rhythm_first_ready_requests"] = admission_size
            counters["spec_rhythm_admission_batches"] += 1
            counters["spec_rhythm_admitted_requests"] += admission_size
            counters["spec_rhythm_last_admission_batch_size"] = admission_size
            counters["spec_rhythm_max_admission_batch_size"] = max(
                counters["spec_rhythm_max_admission_batch_size"], admission_size
            )

        def admit_ready(now: float) -> None:
            """Synchronously prefill/admit one batch (bootstrap and fallback)."""

            ready_indices = select_ready(now)
            if not ready_indices:
                return
            prefill_indices = [index for index in ready_indices if index not in prefetched]
            target_tokens: list[int] = []
            if prefill_indices:
                activate_online_cache(prefill_indices)
                refill_started = time.perf_counter()
                if token_chunk_prefill:
                    cursors = {index: 0 for index in prefill_indices}
                    sampled: dict[int, int] = {}
                    while len(sampled) < len(prefill_indices):
                        chunks = plan_spec_rhythm_prefill_token_chunk(
                            prefill_indices,
                            prompt_lengths={index: local_states[index].prompt_length for index in prefill_indices},
                            cursors=cursors,
                            token_cap=prefill_token_chunk_size,
                        )
                        if not chunks:
                            raise RuntimeError(
                                "SpecRhythm token-chunk prefill stalled before all first tokens were sampled."
                            )
                        completed = self._prefill_spec_rhythm_token_chunk_batch(
                            {index: local_states[index].token_ids for index in prefill_indices},
                            {index: local_states[index] for index in prefill_indices},
                            chunks,
                        )
                        for chunk in chunks:
                            cursors[chunk.request_index] = chunk.end
                        sampled.update(completed)
                        submitted_tokens = sum(chunk.token_count for chunk in chunks)
                        counters["spec_rhythm_prefill_token_chunk_submissions"] += 1
                        counters["spec_rhythm_prefill_token_chunk_tokens"] += submitted_tokens
                        counters["spec_rhythm_prefill_token_chunk_completed_rows"] += len(completed)
                        counters["spec_rhythm_prefill_token_chunk_max_submission_tokens"] = max(
                            counters["spec_rhythm_prefill_token_chunk_max_submission_tokens"],
                            submitted_tokens,
                        )
                        if len(sampled) < len(prefill_indices):
                            counters["spec_rhythm_prefill_token_chunk_partial_submissions"] += 1
                    target_tokens = [sampled[index] for index in prefill_indices]
                else:
                    target_tokens = self._prefill_and_sample_target_batch(
                        [local_states[index].token_ids for index in prefill_indices],
                        [local_states[index] for index in prefill_indices],
                        prefill_indices,
                    )
                # This is telemetry only.  TPOT begins at the per-row
                # first-token publication clock captured by activate_ready.
                counters["spec_rhythm_online_refill_ms"] += (time.perf_counter() - refill_started) * 1000.0
                counters["spec_rhythm_prefill_batches"] += 1
                counters["spec_rhythm_prefill_requests"] += len(prefill_indices)
                record_prefill_batch(len(prefill_indices))
            activate_ready(
                ready_indices,
                target_tokens,
            )

        def reserve_ready(now: float) -> None:
            """Reserve an arrived batch for the next cycle's model window."""

            nonlocal staged_prefill_indices
            nonlocal staged_prefill_cursors
            nonlocal staged_prefill_target_tokens
            if staged_prefill_indices:
                return
            ready_indices = select_ready(now)
            if not ready_indices:
                return
            staged_prefill_indices = ready_indices
            activate_online_cache([index for index in ready_indices if index not in prefetched])
            if token_chunk_prefill:
                prefill_indices = [index for index in ready_indices if index not in prefetched]
                staged_prefill_cursors = {index: 0 for index in prefill_indices}
                staged_prefill_target_tokens = {}
            counters["spec_rhythm_staged_prefill_reserved_batches"] += 1
            counters["spec_rhythm_staged_prefill_reserved_requests"] += len(ready_indices)

        def submit_staged_prefill(
            *,
            coordinate_before_broadcast: bool,
            precomputed_target_tokens: torch.Tensor | None = None,
            draft_prefill_completed: bool = False,
        ) -> None:
            """Submit one whole-prompt pass or one token-capped cursor step."""

            nonlocal staged_prefill_tokens
            if not staged_prefill_indices:
                raise RuntimeError("SpecRhythm cannot submit an empty staged prefill.")
            if staged_prefill_tokens is not None:
                raise RuntimeError("A staged online prefill was submitted more than once.")
            if not token_chunk_prefill:
                staged_prefill_tokens = self._prefill_and_sample_target_batch(
                    [local_states[index].token_ids for index in staged_prefill_indices],
                    [local_states[index] for index in staged_prefill_indices],
                    staged_prefill_indices,
                    coordinate_before_broadcast=coordinate_before_broadcast,
                    precomputed_target_tokens=precomputed_target_tokens,
                    draft_prefill_completed=draft_prefill_completed,
                )
                return

            if precomputed_target_tokens is not None or draft_prefill_completed:
                raise RuntimeError("Mixed prefill is incompatible with token-chunk prefill.")

            prefill_indices = [index for index in staged_prefill_indices if index not in prefetched]
            chunks = plan_spec_rhythm_prefill_token_chunk(
                prefill_indices,
                prompt_lengths={index: local_states[index].prompt_length for index in prefill_indices},
                cursors=staged_prefill_cursors,
                token_cap=prefill_token_chunk_size,
            )
            if not chunks:
                raise RuntimeError("SpecRhythm staged token-chunk prefill has no unfinished prompt span.")
            completed = self._prefill_spec_rhythm_token_chunk_batch(
                {index: local_states[index].token_ids for index in prefill_indices},
                {index: local_states[index] for index in prefill_indices},
                chunks,
                coordinate_before_broadcast=coordinate_before_broadcast,
            )
            for chunk in chunks:
                staged_prefill_cursors[chunk.request_index] = chunk.end
            submitted_tokens = sum(chunk.token_count for chunk in chunks)
            staged_prefill_target_tokens.update(completed)
            counters["spec_rhythm_prefill_token_chunk_submissions"] += 1
            counters["spec_rhythm_prefill_token_chunk_tokens"] += submitted_tokens
            counters["spec_rhythm_prefill_token_chunk_completed_rows"] += len(completed)
            counters["spec_rhythm_prefill_token_chunk_max_submission_tokens"] = max(
                counters["spec_rhythm_prefill_token_chunk_max_submission_tokens"],
                submitted_tokens,
            )
            if all(staged_prefill_cursors[index] == local_states[index].prompt_length for index in prefill_indices):
                if set(staged_prefill_target_tokens) != set(prefill_indices):
                    raise RuntimeError(
                        "SpecRhythm token-chunk prefill reached every prompt end without one target sample per row."
                    )
                staged_prefill_tokens = [staged_prefill_target_tokens[index] for index in prefill_indices]
            else:
                counters["spec_rhythm_prefill_token_chunk_partial_submissions"] += 1

        def activate_staged_prefill() -> None:
            """Publish a successfully completed staged prefill exactly once."""

            nonlocal staged_prefill_indices
            nonlocal staged_prefill_tokens
            nonlocal staged_prefill_cursors
            nonlocal staged_prefill_target_tokens
            if not staged_prefill_indices:
                if staged_prefill_tokens is not None:
                    raise RuntimeError("Online PEARL retained tokens without a staged reservation.")
                return
            if staged_prefill_tokens is None:
                if token_chunk_prefill and staged_prefill_cursors:
                    # A partial token chunk has populated private KV slots,
                    # but no first token/state is visible until every staged
                    # request reaches its prompt's final token.
                    return
                raise RuntimeError("Online PEARL attempted to activate an unfinished prefill.")
            activate_ready(
                staged_prefill_indices,
                staged_prefill_tokens,
            )
            counters["spec_rhythm_prefill_batches"] += 1
            counters["spec_rhythm_prefill_requests"] += len(staged_prefill_indices)
            record_prefill_batch(len(staged_prefill_indices))
            staged_prefill_indices = []
            staged_prefill_tokens = None
            staged_prefill_cursors = {}
            staged_prefill_target_tokens = {}

        def admit_available(
            *,
            now: float | None = None,
            stage_for_overlap: bool = False,
        ) -> None:
            """Admit ready requests without a clock collective when the queue is empty.

            Once a fixed SpecSLO batch has been prefetched, repeated wall-clock
            broadcasts do not change scheduler state. Keep the collective for
            online/future arrivals, where every rank must observe the same time.
            """
            if not pending_admission or len(active_indices) + len(staged_prefill_indices) >= initial_batch_size:
                return
            arrival_gated = bool(self.config.spec_rhythm_online_prefill)
            # A static/non-online queue has no time predicate.  During active
            # online decode, callers pass the target-leader timestamp already
            # carried by the cycle accounting broadcast, avoiding a second
            # WORLD broadcast + D2H read on every no-arrival cycle.
            if not arrival_gated:
                admit_ready(0.0)
            else:
                resolved_now = synchronized_wall_time() if now is None else now
                if stage_for_overlap and staged_prefill_overlap:
                    reserve_ready(resolved_now)
                else:
                    admit_ready(resolved_now)

        def invalidate_request(request_index: int) -> None:
            controller.invalidate_request(request_index)
            for proposal_id, payload in tuple(payloads.items()):
                if payload.ticket.request_index == request_index:
                    payloads.pop(proposal_id, None)

        def service_frontier_signature() -> tuple[int, ...]:
            """Return a cheap rank-replicated service-state fingerprint.

            The linear SpecRhythm loop is intentionally driven independently
            on every rank.  A single divergent completion or proposal
            lifecycle decision would otherwise let one model group leave the
            loop while another rank enters the next collective, turning a
            deterministic state bug into a five-minute distributed timeout.
            All values below are small exact integers when represented as
            float64, so they can ride on the existing tail timing broadcast at
            effectively no additional communication cost.
            """

            committed_fingerprint = sum(
                (index + 1) * (int(local_states[index].committed_length or 0) + 1) for index in active_indices
            )
            epoch_fingerprint = sum(
                (index + 1) * (int(runtime_states[index].prefix_epoch) + 1) for index in active_indices
            )
            # Keep the existing ten-scalar signature (the tail envelope also
            # carries elapsed time and a target-leader monotonic endpoint)
            # while making a divergent staged
            # reservation observable.  All packed fields fit exactly in a
            # float64 integer for the supported queue sizes.
            prefill_cursor_fingerprint = sum(
                (index + 1) * (cursor + 1) for index, cursor in staged_prefill_cursors.items()
            ) + sum((index + 1) * (token + 1) for index, token in staged_prefill_target_tokens.items())
            # Preserve the fixed ten-scalar service envelope. Bits 10..19
            # were unused by the established staged-reservation encoding and
            # carry a compact cross-rank cursor/sample checksum.
            prefill_fingerprint = (
                len(prefetched)
                + ((prefill_cursor_fingerprint & ((1 << 10) - 1)) << 10)
                + (len(staged_prefill_indices) << 20)
                + ((sum(index + 1 for index in staged_prefill_indices) & ((1 << 20) - 1)) << 32)
            )
            return (
                int(round_count),
                len(active_indices),
                len(pending_admission),
                prefill_fingerprint,
                len(completed_states),
                len(controller.ready),
                len(controller.staged_eager),
                len(payloads),
                committed_fingerprint,
                epoch_fingerprint,
            )

        def validate_service_frontier(values: Sequence[float]) -> None:
            target_signature = tuple(int(round(value)) for value in values)
            local_signature = service_frontier_signature()
            if local_signature != target_signature:
                ready_detail = {int(index): int(ticket.proposal_id) for index, ticket in controller.ready.items()}
                eager_detail = {
                    int(index): int(ticket.proposal_id) for index, ticket in controller.staged_eager.items()
                }
                payload_detail = {
                    int(proposal_id): int(payload.ticket.request_index) for proposal_id, payload in payloads.items()
                }
                raise RuntimeError(
                    "SpecRhythm rank-local service frontier diverged before "
                    "the next cycle: "
                    f"rank={self.rank}, local={local_signature}, "
                    f"target_leader={target_signature}, "
                    f"local_ready={ready_detail}, "
                    f"local_staged_eager={eager_detail}, "
                    f"local_payloads={payload_detail}."
                )

        def run_target_fallback(indices: Sequence[int]) -> dict[int, float]:
            """Serve tiny resident batches with one exact target token.

            At low arrival rates a two-home speculative pipeline would make a
            request wait for the other home even though no useful draft batch
            exists.  The fallback keeps both KV caches on the same committed
            frontier and returns to SpecRhythm as soon as the resident batch
            reaches the configured threshold.
            """
            indices = list(indices)
            local = draft_states if self.is_draft else target_states
            # Validate the complete batch before mutating any request.  A
            # malformed second row must not leave the first row truncated and
            # its proposal invalidated on only a subset of ranks.
            fallback_rows: list[tuple[int, PearlPipelineState, int, tuple[int, ...]]] = []
            for index in indices:
                state = local[index]
                committed = state.committed_length
                if committed is None or committed <= 0 or not state.prompt_length <= committed <= len(state.token_ids):
                    raise RuntimeError("SpecRhythm target fallback has an invalid committed prefix boundary.")
                runtime = runtime_states[index]
                if state.continuation_epoch != runtime.prefix_epoch:
                    raise RuntimeError("SpecRhythm target fallback refused a stale committed prefix epoch.")
                ready = controller.ready.get(index)
                if ready is not None:
                    controller.validate_verification(index)
                staged = controller.staged_eager.get(index)
                request_payload_ids = tuple(
                    proposal_id for proposal_id, payload in payloads.items() if payload.ticket.request_index == index
                )
                expected_tickets = tuple(ticket for ticket in (ready, staged) if ticket is not None)
                if any(
                    proposal_id not in payloads or payloads[proposal_id].ticket is not ticket
                    for proposal_id, ticket in ((ticket.proposal_id, ticket) for ticket in expected_tickets)
                ):
                    raise RuntimeError("SpecRhythm target fallback found a missing authoritative proposal payload.")
                if len(request_payload_ids) != len(expected_tickets):
                    raise RuntimeError("SpecRhythm target fallback found an orphan proposal payload.")
                fallback_rows.append((index, state, committed, request_payload_ids))

            for index, state, committed, request_payload_ids in fallback_rows:
                # A request can enter the tiny-batch target-only path with a
                # ready proposal and, on the draft rank, an additional staged
                # eager suffix.  Neither belongs to the exact AR frontier.
                # Invalidate the control-plane tickets and truncate only the
                # role-local authoritative state before constructing inputs.
                invalidate_request(index)
                for proposal_id in request_payload_ids:
                    payloads.pop(proposal_id, None)
                del state.token_ids[committed:]
                state.pre_verify = True
                state.pending_window_size = 0
            input_ids = [local[index].token_ids[-1] for index in indices]
            positions = [int(local[index].committed_length) - 1 for index in indices]
            if self.is_draft:
                self._run_packed_hidden(
                    input_ids,
                    indices,
                    positions,
                    use_aclgraph=not self.config.enforce_eager,
                )
                token_ids = torch.zeros(len(indices), dtype=torch.long, device=self.device)
            else:
                target_sampling_kwargs: dict[str, Any] = {}
                top_ps = [target_states[index].top_p for index in indices]
                top_ks = [target_states[index].top_k for index in indices]
                if any(value < 1.0 for value in top_ps) or any(value > 0 for value in top_ks):
                    target_sampling_kwargs.update(top_ps=top_ps, top_ks=top_ks)
                token_ids = self._run_packed_sample(
                    input_ids,
                    indices,
                    positions,
                    [target_states[index].temperature for index in indices],
                    # A target eager oracle must also cover the exact AR tail
                    # used when the remaining completion cannot fill another
                    # fixed-gamma proposal.  Otherwise the diagnostic mixes
                    # eager verification with graph-backed tail tokens.
                    use_aclgraph=(
                        not self.config.enforce_eager and not envs.VLLM_ASCEND_SPECRHYTHM_DISABLE_TARGET_ACLGRAPH
                    ),
                    **target_sampling_kwargs,
                )
            dist.broadcast(token_ids, src=self.topology.target_leader_rank)
            values = [int(value) for value in token_ids.cpu().tolist()]
            finished_publication_endpoints: dict[int, float] = {}
            for index, token_id in zip(indices, values):
                state = local[index]
                state.token_ids.append(token_id)
                assert state.committed_length is not None
                state.committed_length += 1
                state.continuation_epoch += 1
                runtime = runtime_states[index]
                runtime.delivered_tokens += 1
                runtime.verification_rounds += 1
                runtime.prefix_epoch += 1
                self._deliver_committed_tokens(index, request_params[index], local_states[index])
                if _finished(local_states[index], self.eos_token_ids):
                    # A request's TPOT interval ends when its own final-token
                    # callback has returned.  Later rows and the role-local
                    # stream fence below are not request-visible work for this
                    # completed row.
                    finished_publication_endpoints[index] = time.perf_counter()
            return finished_publication_endpoints

        trace_request_env = envs.VLLM_ASCEND_SPECRHYTHM_TRACE_REQUEST
        trace_requests = {
            int(value.strip()) for value in trace_request_env.split(",") if value.strip().lstrip("-").isdigit()
        }
        admit_available()
        accounting_buffer_device: torch.device | str = "cpu" if gloo_accounting else self.device
        cycle_accounting_envelope = ReusableSpecRhythmEnvelope(
            2 + (2 if linear_window_estimator is not None else 0) + 2 * initial_batch_size,
            device=accounting_buffer_device,
        )
        fallback_accounting_envelope = ReusableSpecRhythmEnvelope(
            2 + 2 * initial_batch_size,
            device=accounting_buffer_device,
        )
        tail_accounting_envelope = ReusableSpecRhythmEnvelope(
            2 + len(service_frontier_signature()),
            device=accounting_buffer_device,
        )
        # This is a persistent accounting boundary, rather than merely the
        # host entry time of the current Python iteration.  At the end of each
        # cycle it is moved to the instant immediately before the tail timing
        # collective.  Consequently that collective, its D2H read, and the
        # remaining loop bookkeeping become the beginning of the next
        # request-visible interval instead of disappearing between two calls
        # to ``perf_counter``.
        cycle_accounting_started = take_activation_cycle_start(time.perf_counter())
        while (active_indices or pending_admission) and (max_rounds is None or round_count < max_rounds):
            if not active_indices:
                if staged_prefill_indices:
                    # The incumbent home can drain before a long staged
                    # prompt reaches its final chunk.  With no model cycle to
                    # hide behind, finish one private chunk at a time and
                    # publish only after the final target sample exists.
                    idle_prefill_started = time.perf_counter()
                    submit_staged_prefill(coordinate_before_broadcast=False)
                    counters["spec_rhythm_online_refill_ms"] += (time.perf_counter() - idle_prefill_started) * 1000.0
                    activate_staged_prefill()
                    if active_indices:
                        cycle_accounting_started = take_activation_cycle_start(time.perf_counter())
                    continue
                next_arrival = min(
                    float(request_params[index].arrival_ts)
                    for index in pending_admission
                    if request_params[index].arrival_ts is not None
                )
                delay = max(0.0, next_arrival - synchronized_wall_time())
                if delay:
                    # Poll in short intervals so a malformed far-future trace
                    # cannot make a worker appear hung to the controller.
                    time.sleep(min(delay, 0.05))
                admit_available()
                if active_indices:
                    # No incumbent request may inherit an idle arrival wait or
                    # the prefill that produced its first output token.
                    cycle_accounting_started = take_activation_cycle_start(time.perf_counter())
                continue
            if (
                self.config.spec_rhythm_target_fallback_max_batch
                and len(active_indices) <= self.config.spec_rhythm_target_fallback_max_batch
            ):
                cycle_started = cycle_accounting_started
                fallback_cycle_indices = tuple(active_indices)
                fallback_finish_endpoints = run_target_fallback(fallback_cycle_indices)
                torch.npu.synchronize()
                cycle_accounting_split = time.perf_counter()
                needs_admission_clock = bool(pending_admission and self.config.spec_rhythm_online_prefill)
                fallback_finish_offsets = [-1.0] * initial_batch_size
                for slot, index in enumerate(fallback_cycle_indices):
                    endpoint = fallback_finish_endpoints.get(index)
                    if endpoint is not None:
                        fallback_finish_offsets[slot] = max(
                            0.0,
                            endpoint - cycle_started,
                        )
                fallback_cycle_ids = [
                    *fallback_cycle_indices,
                    *([-1] * (initial_batch_size - len(fallback_cycle_indices))),
                ]
                fallback_clock_values = [
                    cycle_accounting_split - cycle_started if self.rank == self.topology.target_leader_rank else 0.0,
                    time.time() if self.rank == self.topology.target_leader_rank and needs_admission_clock else 0.0,
                    *fallback_cycle_ids,
                    *fallback_finish_offsets,
                ]
                fallback_accounting_envelope.rewrite(fallback_clock_values)
                fallback_clock_values = fallback_accounting_envelope.broadcast_and_materialize(
                    source_rank=self.topology.target_leader_rank,
                    group=accounting_group,
                )
                cycle_elapsed, cycle_wall_time = (float(value) for value in fallback_clock_values[:2])
                received_cycle_ids = [int(round(value)) for value in fallback_clock_values[2 : 2 + initial_batch_size]]
                if received_cycle_ids != fallback_cycle_ids:
                    raise RuntimeError(
                        "SpecRhythm target fallback active-row order diverged across ranks before finish accounting."
                    )
                fallback_finish_offsets = [
                    float(value) for value in fallback_clock_values[2 + initial_batch_size : 2 + 2 * initial_batch_size]
                ]
                if round_count < self.config.profile_decode_steps:
                    profiled_decode_steps += 1
                if npu_profiler is not None:
                    npu_profiler.step()
                last_cycle_ms = cycle_elapsed * 1000.0
                finished_this_cycle = [
                    index for index in fallback_cycle_indices if _finished(local_states[index], self.eos_token_ids)
                ]
                local_finished_set = set(finished_this_cycle)
                leader_finished_set = decode_finish_envelope(
                    fallback_cycle_indices,
                    fallback_finish_offsets,
                    cycle_elapsed,
                    label="SpecRhythm target fallback",
                )
                if leader_finished_set != local_finished_set:
                    raise RuntimeError("SpecRhythm target fallback finish decisions diverged across ranks.")
                finished_set = leader_finished_set
                for slot, index in enumerate(fallback_cycle_indices):
                    elapsed_ms = last_cycle_ms
                    if index in finished_set:
                        finish_offset = fallback_finish_offsets[slot]
                        elapsed_ms = finish_offset * 1000.0
                    runtime_states[index].add_decode_time(elapsed_ms)
                for index in finished_this_cycle:
                    local_states[index].finished_decode_elapsed_ms = runtime_states[index].decode_elapsed_ms
                    completed_states[index] = local_states[index].clone()
                    invalidate_request(index)
                release_online_cache(finished_this_cycle)
                if finished_this_cycle:
                    active_indices = [index for index in active_indices if index not in finished_set]
                # Online arrivals must be considered on every fallback cycle,
                # including while the incumbent request is still running.
                # Otherwise a long request can hold the service in target-only
                # mode until completion and charge newly arrived work an
                # unbounded admission delay.
                admission_now = None
                if pending_admission and len(active_indices) < initial_batch_size:
                    admission_now = cycle_wall_time if self.config.spec_rhythm_online_prefill else 0.0
                tail_incumbent_indices = tuple(active_indices)
                stopping_after_cycle = bool(
                    (max_rounds is not None and round_count + 1 >= max_rounds)
                    or (
                        self.config.stop_after_profiled_decode_steps
                        and profiled_decode_steps >= self.config.profile_decode_steps
                    )
                )
                if not stopping_after_cycle:
                    admit_available(now=admission_now)
                cycle_tail_ended = time.perf_counter()
                local_frontier_signature = service_frontier_signature()
                tail_elapsed_values = [
                    cycle_tail_ended - cycle_accounting_split if self.rank == self.topology.target_leader_rank else 0.0,
                    cycle_tail_ended if self.rank == self.topology.target_leader_rank else 0.0,
                    *(
                        local_frontier_signature
                        if self.rank == self.topology.target_leader_rank
                        else (0,) * len(local_frontier_signature)
                    ),
                ]
                tail_accounting_envelope.rewrite(tail_elapsed_values)
                tail_elapsed_values = tail_accounting_envelope.broadcast_and_materialize(
                    source_rank=self.topology.target_leader_rank,
                    group=accounting_group,
                )
                tail_elapsed_ms = float(tail_elapsed_values[0]) * 1000.0
                if self.device.type != "cpu":
                    validate_service_frontier(tail_elapsed_values[2:])
                for index in tail_incumbent_indices:
                    runtime_states[index].add_decode_time(tail_elapsed_ms)
                charge_activation_tail(float(tail_elapsed_values[1]))
                last_cycle_ms = cycle_elapsed * 1000.0
                if tail_incumbent_indices:
                    last_cycle_ms += tail_elapsed_ms
                    counters["spec_rhythm_cycle_accounted_tail_ms"] += tail_elapsed_ms
                    counters["spec_rhythm_cycle_accounted_tail_cycles"] += 1
                # Start the next interval before this timing broadcast.  Its
                # HCCL/D2H cost is therefore charged by the next cycle rather
                # than being lost; a request completing above is intentionally
                # not charged for work after its final token.
                cycle_accounting_started = cycle_tail_ended
                round_count += 1
                if (
                    self.config.stop_after_profiled_decode_steps
                    and profiled_decode_steps >= self.config.profile_decode_steps
                ):
                    break
                continue
            cycle_started = cycle_accounting_started
            cycle_active_indices = tuple(active_indices)
            host_trace: dict[str, int | float] | None = None
            if round_count < self.config.profile_host_decode_steps:
                host_trace = {
                    "step": round_count,
                    "rank": self.rank,
                    "is_draft_rank": int(self.is_draft),
                    "cycle_start_seconds": cycle_started,
                }
            if fixed_gamma_scheduler_fast_path:
                # With no external B/roofline and min=max=gamma, the default
                # envelope is exactly one gamma window per active row.  A
                # valid ready payload is at most gamma wide, so the general
                # rebuild scan is provably a no-op.  Avoid repeating that
                # active/ready payload walk on every decode cycle.
                roof_context_len = 0
                verification_roof = len(active_indices) * self.gamma
                counters["spec_rhythm_fixed_gamma_scheduler_fast_path_cycles"] += 1
            else:
                # Draft workers may hold longer uncommitted/eager tails than
                # target workers. Only the committed frontier yields the same
                # context bucket and roof lookup on every rank.
                roof_context_len = max(local_states[index].committed_length for index in active_indices)
                verification_roof = shaper.verification_roof(len(active_indices), roof_context_len)
                rebuilt = self._bound_spec_rhythm_linear_ready(
                    controller,
                    payloads,
                    local_states,
                    active_indices,
                    verification_roof,
                    full_window=linear_full_window,
                )
                for name, value in rebuilt.items():
                    key = f"spec_rhythm_linear_{name}"
                    counters[key] += value
            if host_trace is not None:
                host_trace["scheduler_roof_end_seconds"] = time.perf_counter()
            ready_candidate_counts = {
                index: payloads[controller.ready[index].proposal_id].verification_size
                for index in active_indices
                if index in controller.ready
            }
            max_target_requests = (
                self.config.spec_rhythm_max_target_batch
                if has_slo_constraints and self.config.spec_rhythm_max_target_batch > 0
                else None
            )
            if single_batch_ablation:
                plan = controller.build_single_batch_plan(
                    active_indices,
                    overlap=nano_pearl_ablation,
                    verification_budget=verification_roof,
                    ready_candidate_counts=ready_candidate_counts,
                    max_target_requests=max_target_requests,
                )
            else:
                plan = controller.build_plan(
                    active_indices,
                    verification_budget=verification_roof,
                    ready_candidate_counts=ready_candidate_counts,
                    priority=effective_priority,
                    projected_wait_ms=max(last_cycle_ms * 2.0, 1e-6),
                    priority_burst=self.config.spec_rhythm_priority_burst,
                    merge_ready_homes=(
                        has_slo_constraints and self.config.spec_rhythm_merge_ready_homes and not explicit_ablation
                    ),
                    max_target_requests=max_target_requests,
                )
            if trace_requests and self.rank in self.topology.correction_ranks:
                ready_runtime = {
                    int(index): (
                        int(runtime_states[index].home_batch_id),
                        int(controller.ready[index].eager),
                        int(runtime_states[index].delivered_tokens),
                        round(float(runtime_states[index].decode_elapsed_ms), 6),
                        round(float(runtime_states[index].arrival_wait_ms), 6),
                        round(float(runtime_states[index].acceptance_ema), 9),
                        int(runtime_states[index].projected_progress_gap(max(last_cycle_ms * 2.0, 1e-6))),
                    )
                    for index in active_indices
                    if index in controller.ready
                }
                print(
                    "[SpecRhythm plan] "
                    f"rank={self.rank} round={round_count} "
                    f"target={list(plan.target_request_indices)} "
                    f"normal={list(plan.normal_draft_request_indices)} "
                    f"eager={list(plan.eager_candidate_indices)} "
                    f"deferred={list(plan.deferred_target_request_indices)} "
                    f"last_cycle_ms={last_cycle_ms:.6f} "
                    f"ready_runtime={ready_runtime}",
                    flush=True,
                )
            if host_trace is not None:
                host_trace["scheduler_plan_end_seconds"] = time.perf_counter()
            if plan.phase.value == "complete":
                raise RuntimeError("SpecRhythm has active requests but no verifiable or draftable proposal.")
            counters[f"spec_rhythm_{plan.phase.value}_steps"] += 1
            target_indices = list(plan.target_request_indices)
            target_payloads: list[NativeSpecRhythmDevicePayload] = []
            for request_index in target_indices:
                # Fail a stale/routed ticket before any target model or KV
                # mutation. Payload validation then proves role-local prefix
                # parity for that controller-authoritative ticket.
                ticket = controller.validate_verification(request_index)
                payload = payloads.get(ticket.proposal_id)
                if payload is None:
                    raise RuntimeError("SpecRhythm controller referenced a missing device payload.")
                if payload.ticket is not ticket:
                    raise RuntimeError("SpecRhythm controller and device payload disagree on the target ticket.")
                payload.validate_for(
                    local_states[request_index],
                    full_window=linear_full_window,
                )
                target_payloads.append(payload)
            target_payload_by_index = {payload.ticket.request_index: payload for payload in target_payloads}

            projected_wait_ms = max(last_cycle_ms * 2.0, 1e-6)
            normal_indices = list(plan.normal_draft_request_indices)
            eager_candidates: list[int] = []
            residual_eager_indices: frozenset[int] = frozenset()
            if nano_pearl_ablation:
                # PEARL's one-batch pipeline continuously prepares the next
                # full window for every row currently being verified.  This
                # is mechanism overlap, not SpecRhythm's urgency/W policy.
                eager_candidates = [
                    index
                    for index in plan.eager_candidate_indices
                    if (
                        not linear_full_window
                        or _full_window_eager_has_useful_horizon(
                            local_states[index],
                            target_payload_by_index[index].verification_size,
                        )
                    )
                ]
            elif effective_eager_cap:
                urgent_eager_candidates = [
                    index
                    for index in plan.eager_candidate_indices
                    if runtime_states[index].projected_progress_gap(projected_wait_ms) > 0
                    and runtime_states[index].urgency(projected_wait_ms) >= self.config.spec_rhythm_urgency_threshold
                    and runtime_states[index].expected_acceptance_benefit >= self.config.spec_rhythm_acceptance_floor
                    and (
                        not linear_full_window
                        or _full_window_eager_has_useful_horizon(
                            local_states[index],
                            target_payload_by_index[index].verification_size,
                        )
                    )
                ]
                eager_candidates = urgent_eager_candidates
                if (
                    fixed_gamma_scheduler_fast_path
                    and linear_full_window
                    and self.config.spec_rhythm_linear_idle_residual_eager
                    and not normal_indices
                ):
                    urgent_set = set(urgent_eager_candidates)
                    residual_eager_indices = frozenset(
                        index
                        for index in plan.eager_candidate_indices
                        if index not in urgent_set
                        and runtime_states[index].expected_continuation_benefit
                        >= self.config.spec_rhythm_acceptance_floor
                        and _full_window_eager_has_useful_horizon(
                            local_states[index],
                            target_payload_by_index[index].verification_size,
                        )
                    )
                    eager_candidates.extend(residual_eager_indices)
            if linear_window_estimator is not None and not nano_pearl_ablation:
                eligible_eager_rows = len(eager_candidates)
                eager_candidates, draft_window = _gate_linear_eager_candidates(
                    eager_candidates,
                    states=runtime_states,
                    projected_wait_ms=projected_wait_ms,
                    normal_request_count=len(normal_indices),
                    gamma=self.gamma,
                    estimator=linear_window_estimator,
                    max_rows=self.config.max_num_seqs,
                    allow_cross_graph_bucket=(self.config.spec_rhythm_linear_eager_cross_graph_bucket),
                    residual_indices=residual_eager_indices,
                )
                admitted_eager_rows = len(eager_candidates)
                admitted_residual_eager_rows = sum(index in residual_eager_indices for index in eager_candidates)
                normal_bucket = (
                    _next_linear_draft_graph_bucket(
                        len(normal_indices),
                        self.config.max_num_seqs,
                    )
                    if normal_indices
                    else 0
                )
                bucket_headroom = max(0, normal_bucket - len(normal_indices))
                w_eager_row_cap = min(
                    eligible_eager_rows,
                    draft_window.eager_token_budget // self.gamma,
                    self.config.max_num_seqs - len(normal_indices),
                )
                cross_bucket_candidate_rows = max(0, w_eager_row_cap - bucket_headroom) if normal_indices else 0
                cross_bucket_admitted_rows = max(0, admitted_eager_rows - bucket_headroom) if normal_indices else 0
                counters["spec_rhythm_linear_w_eligible_eager_rows"] += eligible_eager_rows
                counters["spec_rhythm_linear_w_admitted_eager_rows"] += admitted_eager_rows
                counters["spec_rhythm_linear_idle_residual_eligible_rows"] += len(residual_eager_indices)
                counters["spec_rhythm_linear_idle_residual_admitted_rows"] += admitted_residual_eager_rows
                counters["spec_rhythm_linear_w_deferred_eager_rows"] += eligible_eager_rows - admitted_eager_rows
                counters["spec_rhythm_linear_w_bucket_deferred_eager_rows"] += max(
                    0,
                    w_eager_row_cap - admitted_eager_rows,
                )
                counters["spec_rhythm_linear_w_cross_bucket_candidate_rows"] += cross_bucket_candidate_rows
                counters["spec_rhythm_linear_w_bucket_blocked_eager_rows"] += (
                    cross_bucket_candidate_rows if not self.config.spec_rhythm_linear_eager_cross_graph_bucket else 0
                )
                counters["spec_rhythm_linear_w_cross_bucket_admitted_eager_rows"] += cross_bucket_admitted_rows
                counters["spec_rhythm_linear_w_paid_headroom_eager_rows"] += max(
                    0,
                    admitted_eager_rows - w_eager_row_cap,
                )
                counters["spec_rhythm_linear_w_last_eager_row_cap"] = admitted_eager_rows
                counters["spec_rhythm_linear_w_last_normal_bucket"] = normal_bucket
                counters["spec_rhythm_linear_w_last_bucket_headroom"] = bucket_headroom
                counters["spec_rhythm_linear_w_last_cross_bucket_candidate_rows"] = cross_bucket_candidate_rows
                counters["spec_rhythm_linear_w_last_window_ms"] = draft_window.draft_window_ms
                counters["spec_rhythm_linear_w_last_residual_ms"] = draft_window.residual_window_ms
                counters["spec_rhythm_linear_w_last_predicted_exposed_ms"] = draft_window.predicted_exposed_draft_ms
                counters["spec_rhythm_linear_w_last_draft_ms_per_token"] = draft_window.draft_ms_per_token or 0.0
                counters["spec_rhythm_linear_w_calibrated_steps"] += int(draft_window.calibrated)
                if host_trace is not None:
                    host_trace.update(
                        {
                            "linear_w_calibrated": int(draft_window.calibrated),
                            "linear_w_eligible_eager_rows": eligible_eager_rows,
                            "linear_w_admitted_eager_rows": admitted_eager_rows,
                            "linear_w_eager_row_cap": counters["spec_rhythm_linear_w_last_eager_row_cap"],
                            "linear_w_normal_bucket": normal_bucket,
                            "linear_w_bucket_headroom": bucket_headroom,
                            "linear_w_cross_bucket_candidate_rows": (cross_bucket_candidate_rows),
                            "linear_w_bucket_blocked_eager_rows": (
                                cross_bucket_candidate_rows
                                if not self.config.spec_rhythm_linear_eager_cross_graph_bucket
                                else 0
                            ),
                            "linear_w_cross_bucket_admitted_eager_rows": (cross_bucket_admitted_rows),
                            "linear_w_window_ms": draft_window.draft_window_ms,
                            "linear_w_residual_ms": draft_window.residual_window_ms,
                        }
                    )
            work_indices: list[int] = []
            work_budgets: list[int] = []
            work_is_eager: list[bool] = []
            if normal_indices or eager_candidates:
                if fixed_gamma_scheduler_fast_path:
                    normal_budgets, eager_budgets = _fixed_gamma_identity_budgets(
                        normal_indices,
                        eager_candidates,
                        gamma=self.gamma,
                        verification_roof=verification_roof,
                    )
                else:
                    budget_plan = shaper.shape(
                        plan_id=plan.plan_id,
                        normal_request_indices=normal_indices,
                        eager_request_indices=eager_candidates,
                        states=runtime_states,
                        projected_wait_ms=projected_wait_ms,
                        context_len=roof_context_len,
                        draft_token_budget=(self.config.spec_rhythm_draft_token_budget),
                        batch_size=len(active_indices),
                        eager_token_cap=effective_eager_cap or None,
                        eager_reserve_tokens=(self.config.spec_rhythm_eager_reserve_tokens),
                        verification_roof=verification_roof,
                    )
                    normal_budgets = dict(budget_plan.normal_budgets)
                    eager_budgets = {
                        index: min(value, effective_eager_cap)
                        for index, value in budget_plan.eager_budgets.items()
                        if min(value, effective_eager_cap) > 0
                    }
                if linear_full_window and any(
                    budget != self.gamma for budget in (*normal_budgets.values(), *eager_budgets.values())
                ):
                    raise RuntimeError("Linear full-window scheduling must not change a selected request's gamma.")
                counters["spec_rhythm_last_verification_roof"] = int(verification_roof)
                counters["spec_rhythm_unused_verification_tokens"] = max(
                    0,
                    int(verification_roof) - sum(normal_budgets.values()) - sum(eager_budgets.values()),
                )
                work_indices = [*normal_budgets, *eager_budgets]
                work_budgets = [
                    *normal_budgets.values(),
                    *eager_budgets.values(),
                ]
                work_is_eager = [False] * len(normal_budgets) + [True] * len(eager_budgets)
                # The scheduler has already selected and budgeted the set.
                # Canonicalize only its execution-row order so alternating
                # priority scores do not churn FIA host-length tuples and
                # force 112 graph-task rebuilds for an unchanged set.
                ordered_work = sorted(
                    zip(work_indices, work_budgets, work_is_eager),
                    key=lambda item: (item[2], item[0]),
                )
                work_indices = [item[0] for item in ordered_work]
                work_budgets = [item[1] for item in ordered_work]
                work_is_eager = [item[2] for item in ordered_work]
                counters["spec_rhythm_allocated_draft_tokens"] += sum(work_budgets)
            if target_indices and work_indices:
                counters["spec_rhythm_concurrent_draft_target_submission_cycles"] += 1
                if nano_pearl_ablation:
                    counters["spec_rhythm_single_batch_overlap_submission_cycles"] += 1
                else:
                    counters["spec_rhythm_dual_batch_overlap_submission_cycles"] += 1
            elif serial_ablation and target_indices:
                counters["spec_rhythm_serial_target_only_cycles"] += 1
            elif serial_ablation and work_indices:
                counters["spec_rhythm_serial_draft_only_cycles"] += 1
            if plan.target_home_batch_id == 0:
                counters["spec_rhythm_target_home_0_cycles"] += 1
            elif plan.target_home_batch_id == 1:
                counters["spec_rhythm_target_home_1_cycles"] += 1
            if host_trace is not None:
                host_trace["scheduler_budget_end_seconds"] = time.perf_counter()

            work_verification_sizes = []
            work_verification_prefixes: list[torch.Tensor | None] = []
            for request_index, eager in zip(work_indices, work_is_eager):
                if linear_full_window:
                    work_verification_sizes.append(self.gamma)
                    work_verification_prefixes.append(None)
                elif eager:
                    parent = target_payload_by_index[request_index]
                    work_verification_sizes.append(parent.ticket.gamma)
                    work_verification_prefixes.append(parent.next_tokens[1:])
                else:
                    work_verification_sizes.append(
                        1
                        if local_states[request_index].pre_verify
                        else (local_states[request_index].pending_window_size or self.gamma)
                    )
                    work_verification_prefixes.append(None)
            tickets = [
                controller.new_ticket(index, gamma=budget, eager=eager)
                for index, budget, eager in zip(work_indices, work_budgets, work_is_eager)
            ]
            compact_layout = (
                CompactFixedGreedyEnvelopeLayout.full_window(
                    len(tickets),
                    self.gamma,
                    include_draft_timing=linear_fixed_gamma_window,
                )
                if tickets
                and linear_full_window
                and self.groups.is_verification_worker
                and not envs.VLLM_ASCEND_SPECRHYTHM_VALIDATE_MAILBOX
                else None
            )
            compact_pending: PendingCompactFixedGreedyEnvelope | None = None
            if host_trace is not None:
                host_trace["scheduler_end_seconds"] = time.perf_counter()
                host_trace["target_requests"] = len(target_indices)
                host_trace["draft_requests"] = len(work_indices)
                host_trace["verify_candidates"] = sum(payload.verification_size for payload in target_payloads)

            mixed_prefill_inputs_eligible = bool(
                staged_prefill_indices
                and not token_chunk_prefill
                and all(index not in prefetched for index in staged_prefill_indices)
                and all(
                    local_states[index].temperature == 0
                    and local_states[index].top_p >= 1.0
                    and local_states[index].top_k <= 0
                    for index in staged_prefill_indices
                )
            )
            mixed_draft_this_cycle = bool(
                not pard_eager_qualification
                and mixed_draft_prefill
                and mixed_prefill_inputs_eligible
                and work_indices
                and all(value == self.gamma for value in work_budgets)
                and all(local_states[index].draft_temperature == 0 for index in work_indices)
                and (
                    len(work_indices) + sum(local_states[index].prompt_length for index in staged_prefill_indices)
                    <= self.config.max_num_batched_tokens
                )
            )
            overlap_draft_this_cycle = bool(
                not pard_eager_qualification
                and overlap_draft_prefill
                and mixed_prefill_inputs_eligible
                and work_indices
                and all(value == self.gamma for value in work_budgets)
                and all(local_states[index].draft_temperature == 0 for index in work_indices)
            )

            profile_this_round = round_count < self.config.profile_decode_steps
            if profile_this_round:
                torch.npu.synchronize()
            reported_draft_token_count = draft_window_event_token_count
            lagged_draft_compute_ms = 0.0
            if draft_window_events is not None and draft_window_event_pending:
                lagged_draft_compute_ms = float(draft_window_events[0].elapsed_time(draft_window_events[1]))
            draft_started = time.perf_counter()
            phase_started = draft_started
            overlapped_draft_prefill_hidden: torch.Tensor | None = None
            if overlap_draft_this_cycle and self.is_draft:
                if draft_prefill_stream is None:
                    raise RuntimeError("Draft prefill overlap has no NPU side stream.")
                # Establish the dependency before queuing this cycle's PA
                # graph.  It covers lazy page-table activation while allowing
                # the later graph and prompt FIA kernels to execute together.
                draft_prefill_stream.wait_stream(torch.npu.current_stream())
            if draft_window_events is not None and work_indices:
                draft_window_events[0].record()
            if mixed_draft_this_cycle:
                (
                    draft_verification,
                    draft_next,
                    draft_confidence,
                ) = self._draft_spec_rhythm_device_batch_with_prefill(
                    draft_states,
                    work_indices,
                    work_budgets,
                    staged_prefill_indices,
                )
                counters["spec_rhythm_mixed_draft_prefill_batches"] += 1
                counters["spec_rhythm_mixed_draft_prefill_requests"] += len(staged_prefill_indices)
                counters["spec_rhythm_mixed_draft_prefill_proposal_tokens"] += len(work_indices) * self.gamma
            elif work_indices:
                draft_batch_kwargs: dict[str, Any] = {
                    "verification_sizes": work_verification_sizes,
                    "verification_prefixes": work_verification_prefixes,
                    "full_window": linear_full_window,
                }
                if envs.VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_BUCKET:
                    draft_batch_kwargs["graph_lane"] = plan.draft_home_batch_id
                (
                    draft_verification,
                    draft_next,
                    draft_confidence,
                ) = self._draft_spec_rhythm_device_batch(
                    draft_states,
                    work_indices,
                    work_budgets,
                    **draft_batch_kwargs,
                )
            else:
                draft_verification, draft_next, draft_confidence = (
                    None,
                    None,
                    None,
                )
            if overlap_draft_this_cycle and self.is_draft:
                assert draft_prefill_stream is not None
                with torch.npu.stream(draft_prefill_stream):
                    overlapped_draft_prefill_hidden = self._draft_prefill_hidden_batch(
                        [local_states[index].token_ids for index in staged_prefill_indices],
                        staged_prefill_indices,
                    )
                counters["spec_rhythm_overlap_draft_prefill_batches"] += 1
                counters["spec_rhythm_overlap_draft_prefill_requests"] += len(staged_prefill_indices)
                counters["spec_rhythm_overlap_draft_prefill_proposal_tokens"] += len(work_indices) * self.gamma
            if draft_window_events is not None and work_indices:
                draft_window_events[1].record()
                draft_window_event_pending = True
            elif draft_window_events is not None:
                draft_window_event_pending = False
            # The scheduler plan is replicated, so every rank retains the
            # denominator paired with the lagged timing carried next cycle.
            draft_window_event_token_count = len(work_indices) * self.gamma
            if linear_bonus_token:
                counters["spec_rhythm_linear_bonus_draft_kv_prepared_rows"] += len(work_indices)
            if profile_this_round:
                torch.npu.synchronize()
            draft_ended = time.perf_counter()
            local_draft_compute_ms = 0.0
            if work_indices and self.rank == self.topology.draft_leader_rank:
                local_draft_compute_ms = (
                    lagged_draft_compute_ms
                    if draft_window_events is not None
                    else (draft_ended - draft_started) * 1000.0
                )
            phase_elapsed = draft_ended - phase_started
            phase_seconds["draft"] += phase_elapsed
            if profile_this_round and self.is_draft:
                decode_profile_seconds["draft_compute"] += phase_elapsed
            if host_trace is not None:
                host_trace["draft_start_seconds"] = draft_started
                host_trace["draft_end_seconds"] = draft_ended

            current_verification_sizes = [payload.verification_size for payload in target_payloads]
            current_next: torch.Tensor | None = (
                torch.stack([payload.next_tokens for payload in target_payloads])
                if linear_full_window and target_indices
                else None
            )
            mixed_prefill_this_cycle = bool(
                mixed_target_prefill
                and mixed_prefill_inputs_eligible
                and target_indices
                and (
                    len(target_indices) * (self.gamma + int(linear_bonus_token))
                    + sum(local_states[index].prompt_length for index in staged_prefill_indices)
                    <= self.config.max_num_batched_tokens
                )
            )
            mixed_prefill_target_tokens: torch.Tensor | None = None
            pending_eager_indices = {index for index, eager in zip(work_indices, work_is_eager) if eager}
            bonus_enabled = [
                bool(
                    linear_bonus_token
                    and controller.staged_eager.get(index) is None
                    and index not in pending_eager_indices
                    and len(local_states[index].committed_completion_token_ids) + self.gamma
                    < local_states[index].max_tokens
                )
                for index in target_indices
            ]
            if linear_bonus_token:
                counters["spec_rhythm_linear_bonus_eligible_rows"] += sum(bonus_enabled)
                counters["spec_rhythm_linear_bonus_suppressed_eager_rows"] += sum(
                    controller.staged_eager.get(index) is not None or index in pending_eager_indices
                    for index in target_indices
                )
                counters["spec_rhythm_linear_bonus_suppressed_output_limit_rows"] += sum(
                    controller.staged_eager.get(index) is None
                    and index not in pending_eager_indices
                    and len(local_states[index].committed_completion_token_ids) + self.gamma
                    >= local_states[index].max_tokens
                    for index in target_indices
                )
            target_started = time.perf_counter()
            phase_started = target_started
            if target_window_events is not None and target_indices:
                target_window_events[0].record()
            if target_indices and linear_full_window:
                if mixed_prefill_this_cycle:
                    (
                        target_tokens,
                        target_logits,
                        mixed_prefill_target_tokens,
                    ) = self._target_full_window_outputs_with_prefill_batch(
                        target_states,
                        target_indices,
                        target_payloads,
                        staged_prefill_indices,
                        proposal_matrix=current_next,
                    )
                    counters["spec_rhythm_mixed_target_prefill_batches"] += 1
                    counters["spec_rhythm_mixed_target_prefill_requests"] += len(staged_prefill_indices)
                    counters["spec_rhythm_mixed_target_prefill_verification_tokens"] += sum(current_verification_sizes)
                    _record_mixed_target_graph_outcome(
                        counters,
                        getattr(
                            self,
                            "_last_mixed_target_graph_outcome",
                            ("disabled", 0, 0),
                        ),
                    )
                else:
                    target_tokens, target_logits = self._target_full_window_outputs_batch(
                        target_states,
                        target_indices,
                        target_payloads,
                        proposal_matrix=current_next,
                    )
                    _record_stable_target_verify_graph_outcome(
                        counters,
                        getattr(
                            self,
                            "_last_stable_target_verify_graph_outcome",
                            ("disabled", 0),
                        ),
                    )
            elif target_indices:
                target_tokens, target_logits = self._target_round_outputs_batch(
                    target_states,
                    target_indices,
                    current_verification_sizes,
                )
            else:
                target_tokens, target_logits = None, None
            if profile_this_round:
                torch.npu.synchronize()
            target_ended = time.perf_counter()
            phase_elapsed = target_ended - phase_started
            phase_seconds["target"] += phase_elapsed
            if profile_this_round and not self.is_draft:
                decode_profile_seconds["target_compute"] += phase_elapsed
            if host_trace is not None:
                host_trace["target_start_seconds"] = target_started
                host_trace["target_end_seconds"] = target_ended

            if (
                trace_requests
                and self.rank == self.topology.target_leader_rank
                and target_indices
                and target_tokens is not None
            ):
                target_offset = 0
                target_row_width = self.gamma + int(linear_bonus_token)
                for row, request_index in enumerate(target_indices):
                    expected = current_verification_sizes[row]
                    if request_index in trace_requests:
                        print(
                            "[SpecRhythm trace-input] "
                            f"round={round_count} request={request_index} "
                            f"expected={expected} target_input_tail="
                            f"{target_states[request_index].token_ids[-8:]} "
                            f"draft_verification="
                            f"{target_payloads[row].verification_tokens.detach().cpu().tolist()} "
                            f"target_tokens="
                            f"{target_tokens[target_offset : target_offset + expected].detach().cpu().tolist()}",
                            flush=True,
                        )
                    target_offset += target_row_width if linear_full_window else expected

            # The verdict depends only on the proposal selected at the start of
            # this cycle, not on the new proposal produced concurrently above.
            # Launch it before the new draft->target mailbox rendezvous.  Target
            # ranks can then finish the small verdict kernel while the draft
            # rank is still completing its longer four-step serial graph,
            # changing the critical path from max(D, T) + V to max(D, T + V).
            if target_indices:
                temperatures = [target_states[index].temperature for index in target_indices]
                if linear_full_window:
                    # verification_tokens and next_tokens are the same fixed
                    # proposal in this protocol.  Pack it once and reuse the
                    # 2-D view for correction instead of issuing both a cat
                    # and a stack launch for identical data every cycle.
                    assert current_next is not None
                    current_message = None if self.is_draft else current_next.reshape(-1)
                else:
                    current_message = (
                        None
                        if self.is_draft
                        else torch.cat([payload.verification_tokens for payload in target_payloads])
                    )
                verdict_started = time.perf_counter()
                phase_started = verdict_started
                verdict = self._verify_target_tokens_batch(
                    target_tokens,
                    target_logits,
                    current_message,
                    current_verification_sizes,
                    temperatures,
                    top_ps=[target_states[index].top_p for index in target_indices],
                    top_ks=[target_states[index].top_k for index in target_indices],
                    bonus_enabled=(bonus_enabled if linear_bonus_token else None),
                )
                if target_window_events is not None:
                    # Rolling-eager may overlap both the target forward and
                    # its device-side verdict.  Measure the whole consumable
                    # window instead of ending W at the model boundary.
                    target_window_events[1].record()
                if profile_this_round:
                    torch.npu.synchronize()
                verdict_ended = time.perf_counter()
                phase_elapsed = verdict_ended - phase_started
                phase_seconds["verify"] += phase_elapsed
                if profile_this_round and not self.is_draft:
                    decode_profile_seconds["target_verdict"] += phase_elapsed
                if host_trace is not None:
                    host_trace["verdict_start_seconds"] = verdict_started
                    host_trace["verdict_end_seconds"] = verdict_ended

            prefill_compute_coordinated = False
            if staged_prefill_indices:
                if staged_prefill_tokens is not None:
                    raise RuntimeError("A staged online prefill was submitted more than once.")
                overlap_prefill_started = time.perf_counter()
                if host_trace is not None:
                    host_trace["overlap_prefill_start_seconds"] = overlap_prefill_started
                    host_trace["overlap_prefill_requests"] = len(staged_prefill_indices)
                # Each rank has already queued this cycle's role-local model
                # phase.  Queue the new prompts behind it on the same stream;
                # draft/current-target work therefore overlaps across the two
                # disjoint model groups instead of paying a separate tail
                # pause.  The helper performs the CPU rendezvous before its
                # cross-model token broadcast.
                if overlap_draft_this_cycle and self.is_draft:
                    if draft_prefill_stream is None or overlapped_draft_prefill_hidden is None:
                        raise RuntimeError("Draft prefill overlap lost its pending output.")
                    # The helper below synchronizes the current stream before
                    # the CPU rendezvous and WORLD token broadcast.  Insert a
                    # device-side join first so publication cannot expose an
                    # incomplete draft KV row.
                    torch.npu.current_stream().wait_stream(draft_prefill_stream)
                submit_staged_prefill(
                    coordinate_before_broadcast=True,
                    precomputed_target_tokens=mixed_prefill_target_tokens,
                    draft_prefill_completed=((mixed_draft_this_cycle or overlap_draft_this_cycle) and self.is_draft),
                )
                # Retain the side-stream result until the join above has been
                # enqueued and completed by the helper's current-stream fence.
                overlapped_draft_prefill_hidden = None
                overlap_prefill_ended = time.perf_counter()
                overlap_prefill_window_ms = (overlap_prefill_ended - overlap_prefill_started) * 1000.0
                counters["spec_rhythm_staged_prefill_overlap_batches"] += 1
                counters["spec_rhythm_staged_prefill_overlap_requests"] += len(staged_prefill_indices)
                counters["spec_rhythm_staged_prefill_overlap_window_ms"] += overlap_prefill_window_ms
                phase_seconds["prefill_overlap"] = phase_seconds.get("prefill_overlap", 0.0) + (
                    overlap_prefill_ended - overlap_prefill_started
                )
                prefill_compute_coordinated = self.device.type == "npu"
                if host_trace is not None:
                    host_trace["overlap_prefill_end_seconds"] = overlap_prefill_ended

            if compact_layout is not None:
                # ProcessGroupHCCL on the deployed CANN release cannot safely
                # execute this cross-model broadcast concurrently with the
                # target TP group's all-reduces: posting the receive before the
                # target forward creates a cross-communicator device-stream
                # wait cycle.  Merely moving the call after Python-level model
                # submission is insufficient when TASK_QUEUE_ENABLE=1: a
                # slower stepwise target can still have queued TP collectives.
                # Synchronize each *local* model stream and use the CPU/Gloo
                # coordination group before touching verification HCCL.  The
                # two model phases remain overlapped (draft(B) on its rank,
                # verify(A) on TP3); only the following communication waits for
                # both sides to become safe.
                coordination_group = getattr(
                    self.groups,
                    "verification_coordination_group",
                    None,
                )
                if coordination_group is not None and not prefill_compute_coordinated:
                    coordination_started = time.perf_counter()
                    torch.npu.current_stream().synchronize()
                    dist.barrier(group=coordination_group)
                    if host_trace is not None:
                        host_trace["compute_coordination_start_seconds"] = coordination_started
                        host_trace["compute_coordination_end_seconds"] = time.perf_counter()
                compact_submit_started = time.perf_counter()
                if self.is_draft:
                    if draft_verification is None or draft_next is None:
                        raise RuntimeError("The compact proposal source lost its full-window draft tensors.")
                    compact_pending = begin_compact_fixed_greedy_broadcast(
                        compact_layout,
                        rank=self.rank,
                        source_rank=self.topology.draft_leader_rank,
                        group=self.groups.verification_group,
                        device=self.device,
                        verification_tokens=draft_verification,
                        continuation_tokens=draft_next,
                        draft_compute_us=(
                            round(local_draft_compute_ms * 1000.0) if linear_fixed_gamma_window else None
                        ),
                    )
                else:
                    compact_pending = begin_compact_fixed_greedy_broadcast(
                        compact_layout,
                        rank=self.rank,
                        source_rank=self.topology.draft_leader_rank,
                        group=self.groups.verification_group,
                        device=self.device,
                    )
                if host_trace is not None:
                    host_trace["compact_submit_start_seconds"] = compact_submit_started
                    host_trace["compact_submit_end_seconds"] = time.perf_counter()
            if self.is_draft and draft_next is not None:
                # Post the device-side HCCL transfer before reading the same
                # proposal for host state bookkeeping. Calling cpu().tolist()
                # once also avoids one stream fence per row.
                draft_next_cpu = draft_next.detach().cpu().tolist()
                for row, (request_index, budget) in enumerate(zip(work_indices, work_budgets)):
                    draft_states[request_index].token_ids.extend(int(value) for value in draft_next_cpu[row][:budget])

            compact_views = None
            compact_wait_seconds = 0.0
            if compact_pending is not None:
                wait_started = time.perf_counter()
                compact_views = compact_pending.wait()
                wait_ended = time.perf_counter()
                compact_wait_seconds = wait_ended - wait_started
                if host_trace is not None:
                    host_trace["compact_wait_start_seconds"] = wait_started
                    host_trace["compact_wait_end_seconds"] = wait_ended
            if profile_this_round and self.groups.is_verification_worker:
                wait_started = time.perf_counter()
                dist.barrier(group=self.groups.verification_group)
                torch.npu.synchronize()
                decode_profile_seconds["wait_sync"] += time.perf_counter() - wait_started
            exchange_started = time.perf_counter()
            phase_started = exchange_started
            if compact_views is not None:
                draft_compute_us = compact_views.draft_compute_us
                new_payloads = [
                    NativeSpecRhythmDevicePayload(
                        ticket=ticket,
                        verification_tokens=verification_tokens,
                        next_tokens=compact_views.continuation_tokens[row],
                        verification_size=self.gamma,
                        draft_confidence=1.0,
                    )
                    for row, (ticket, verification_tokens) in enumerate(zip(tickets, compact_views.verification_rows))
                ]
                counters["spec_rhythm_compact_proposal_broadcasts"] += 1
                counters["spec_rhythm_compact_proposal_int64_saved"] += (
                    compact_layout.legacy_message_numel - compact_layout.message_numel
                )
            else:
                exchanged, confidences, draft_compute_us = self._exchange_spec_rhythm_device_proposals(
                    draft_verification,
                    draft_next,
                    draft_confidence,
                    tickets,
                    work_verification_sizes,
                    draft_compute_ms=(local_draft_compute_ms if linear_fixed_gamma_window else None),
                )
                new_payloads = (
                    self._materialize_spec_rhythm_payloads(
                        tickets=tickets,
                        verification_sizes=work_verification_sizes,
                        local_verification=draft_verification,
                        local_next_windows=draft_next,
                        exchanged_message=exchanged,
                        draft_confidences=confidences,
                    )
                    if tickets
                    else []
                )
            if tickets:
                if linear_fixed_gamma_window:
                    # The fixed greedy full-chain graph assigns confidence
                    # 1.0 to every row by construction.  Preserve that value
                    # as a replicated host scalar so the result path does not
                    # perform an otherwise redundant confidence D2H copy on
                    # every rank and every verification cycle.
                    for payload in new_payloads:
                        payload.draft_confidence = 1.0
                controller.publish(tickets)
                payloads.update((payload.ticket.proposal_id, payload) for payload in new_payloads)
                counters["spec_rhythm_normal_proposals"] += sum(not value for value in work_is_eager)
                counters["spec_rhythm_eager_proposals"] += sum(work_is_eager)
            if profile_this_round and self.groups.is_verification_worker:
                torch.npu.synchronize()
            exchange_ended = time.perf_counter()
            phase_elapsed = exchange_ended - phase_started + compact_wait_seconds
            phase_seconds["exchange"] += phase_elapsed
            if profile_this_round and self.groups.is_verification_worker:
                decode_profile_seconds["draft_to_target_communication"] += (
                    exchange_ended - phase_started + compact_wait_seconds
                )
            if host_trace is not None:
                host_trace["exchange_start_seconds"] = exchange_started
                host_trace["exchange_end_seconds"] = exchange_ended

            finished_this_cycle: list[int] = []
            finished_publication_endpoints: dict[int, float] = {}
            first_decode_indices: set[int] = set()
            if target_indices:
                if current_next is None:
                    if all(payload.ticket.gamma == self.gamma for payload in target_payloads):
                        current_next = torch.stack([payload.next_tokens for payload in target_payloads])
                    else:
                        current_next = torch.full(
                            (len(target_payloads), self.gamma),
                            -1,
                            dtype=torch.long,
                            device=self.device,
                        )
                        for row, payload in enumerate(target_payloads):
                            current_next[row, : payload.ticket.gamma] = payload.next_tokens
                assert current_next is not None
                replicated_target_verdict = all(temperature == 0 for temperature in temperatures)
                if profile_this_round:
                    if replicated_target_verdict:
                        participates_in_correction = self.rank in self.topology.correction_ranks
                        correction_group = self.groups.correction_group
                    else:
                        participates_in_correction = True
                        correction_group = None
                    if participates_in_correction:
                        wait_started = time.perf_counter()
                        dist.barrier(group=correction_group)
                        torch.npu.synchronize()
                        decode_profile_seconds["wait_sync"] += time.perf_counter() - wait_started
                proposal_confidences = None
                proposal_confidence_scale = 1.0
                if target_payloads and not linear_fixed_gamma_window:
                    confidence_values = []
                    for payload in target_payloads:
                        confidence = payload.draft_confidence
                        if confidence is None:
                            confidence = 1.0
                        if not torch.is_tensor(confidence):
                            confidence = torch.tensor(float(confidence), dtype=torch.float32, device=self.device)
                        confidence_values.append(confidence.reshape(()))
                    proposal_confidences = torch.stack(confidence_values)
                    proposal_confidence_scale = 1.0 if self.is_draft else 1e-6
                correction_started = time.perf_counter()
                phase_started = correction_started
                accepted, corrections, synchronized_next = self._broadcast_device_round_result(
                    verdict,
                    current_message,
                    sum(current_verification_sizes),
                    len(target_indices),
                    current_next,
                    replicated_target_verdict=replicated_target_verdict,
                    next_window_sizes=[payload.ticket.gamma for payload in target_payloads],
                    extra_device_values=proposal_confidences,
                    extra_device_scale=proposal_confidence_scale,
                    profile_phase_seconds=(decode_profile_seconds if profile_this_round else None),
                )
                bonuses = getattr(
                    self,
                    "_last_device_round_bonus_tokens",
                    [None] * len(target_indices),
                )
                correction_ended = time.perf_counter()
                phase_seconds["broadcast"] += correction_ended - phase_started
                if host_trace is not None:
                    host_trace["correction_start_seconds"] = correction_started
                    host_trace["correction_end_seconds"] = correction_ended
                state_started = time.perf_counter()
                phase_started = state_started
                bonus_request_indices = [
                    request_index for row, request_index in enumerate(target_indices) if bonuses[row] is not None
                ]
                if bonus_request_indices:
                    if any(controller.staged_eager.get(index) is not None for index in bonus_request_indices):
                        raise RuntimeError("A bonus verdict cannot retain a rolling-eager continuation.")
                    counters["spec_rhythm_linear_bonus_committed_tokens"] += len(bonus_request_indices)
                for row, request_index in enumerate(target_indices):
                    payload = target_payloads[row]
                    expected = current_verification_sizes[row]
                    fully_accepted = accepted[row] == expected
                    was_first_decode_token = runtime_states[request_index].delivered_tokens == 0
                    trace_enabled = request_index in trace_requests and self.rank == self.topology.target_leader_rank
                    eager_ticket = controller.staged_eager.get(request_index)
                    if linear_full_window and self.is_draft:
                        state = draft_states[request_index]
                        assert state.committed_length is not None
                        proposal_tokens = state.token_ids[state.committed_length : state.committed_length + expected]
                        eager_tokens = (
                            state.token_ids[state.committed_length + expected :] if eager_ticket is not None else None
                        )
                        if getattr(self.config, "draft_mode", SERIAL_LINEAR_DRAFT_MODE) == PARD_PARALLEL_DRAFT_MODE:
                            if eager_tokens:
                                raise RuntimeError(
                                    "Native PARD rolling-eager continuation is not implemented; refusing mixed tails."
                                )
                            state.apply_pard_full_window_verification(
                                proposal_token_ids=proposal_tokens,
                                accepted=accepted[row],
                                correction_token_id=corrections[row],
                                bonus_token_id=bonuses[row],
                            )
                        else:
                            state.apply_draft_full_window_verification(
                                proposal_token_ids=proposal_tokens,
                                accepted=accepted[row],
                                correction_token_id=corrections[row],
                                staged_eager_token_ids=eager_tokens,
                                bonus_token_id=bonuses[row],
                            )
                    elif linear_full_window:
                        target_states[request_index].apply_target_full_window_verification(
                            proposal_token_ids=synchronized_next[row],
                            accepted=accepted[row],
                            correction_token_id=corrections[row],
                            bonus_token_id=bonuses[row],
                        )
                    elif self.is_draft:
                        if eager_ticket is not None and not fully_accepted:
                            del draft_states[request_index].token_ids[-eager_ticket.gamma :]
                        draft_states[request_index].apply_draft_verification(
                            gamma=self.gamma,
                            accepted=accepted[row],
                            correction_token_id=corrections[row],
                            next_round_token_ids=synchronized_next[row],
                            verification_size=expected,
                        )
                    else:
                        target_states[request_index].apply_target_verification(
                            gamma=self.gamma,
                            accepted=accepted[row],
                            correction_token_id=corrections[row],
                            next_round_token_ids=synchronized_next[row],
                            verification_size=expected,
                        )
                    if trace_enabled:
                        print(
                            "[SpecRhythm trace] "
                            f"round={round_count} request={request_index} expected={expected} "
                            f"accepted={accepted[row]} correction={corrections[row]} "
                            f"committed={target_states[request_index].committed_length} "
                            f"pre_verify={target_states[request_index].pre_verify} "
                            f"pending={target_states[request_index].pending_window_size} "
                            f"target_tail={target_states[request_index].token_ids[-8:]} "
                            f"draft_tail={draft_states[request_index].token_ids[-8:]}",
                            flush=True,
                        )
                    delivered = accepted[row] + int(not fully_accepted) + int(bonuses[row] is not None)
                    promoted = controller.finish_verification(
                        request_index,
                        fully_accepted=fully_accepted,
                        proposed_tokens=expected,
                        accepted_tokens=accepted[row],
                        delivered_tokens=delivered,
                        draft_confidence=(
                            self._last_device_round_extra_values[row]
                            if proposal_confidences is not None
                            else (None if payload.draft_confidence is None else float(payload.draft_confidence))
                        ),
                        ema_alpha=self.config.spec_rhythm_acceptance_ema_alpha,
                    )
                    if was_first_decode_token and delivered > 0:
                        first_decode_indices.add(request_index)
                    payloads.pop(payload.ticket.proposal_id, None)
                    if eager_ticket is not None:
                        if promoted is None:
                            counters["spec_rhythm_eager_invalidated"] += 1
                            payloads.pop(eager_ticket.proposal_id, None)
                        else:
                            counters["spec_rhythm_eager_promoted"] += 1
                    counters["spec_rhythm_verified_tokens"] += expected
                    self._deliver_committed_tokens(
                        request_index, request_params[request_index], local_states[request_index]
                    )
                    if _finished(local_states[request_index], self.eos_token_ids):
                        finished_this_cycle.append(request_index)
                        # The target leader's value becomes authoritative in
                        # the cycle clock broadcast.  Recording a local value
                        # on followers keeps CPU protocol harnesses capable of
                        # validating the exact same envelope invariant.
                        finished_publication_endpoints[request_index] = time.perf_counter()
                controller.finish_cycle(plan.target_home_batch_id)
                phase_seconds["state_update"] += time.perf_counter() - phase_started
                if profile_this_round:
                    decode_profile_seconds["state_update"] += time.perf_counter() - phase_started
                if host_trace is not None:
                    host_trace["state_start_seconds"] = state_started
                    host_trace["state_end_seconds"] = time.perf_counter()
            else:
                # Warmup has no target verdict/correction collective. Keep every
                # rank at the same service-step boundary before rotating roles.
                dist.barrier()

            if profile_this_round:
                profiled_decode_steps += 1
            if npu_profiler is not None:
                npu_profiler.step()

            local_target_compute_ms = 0.0
            if target_indices and self.rank == self.topology.target_leader_rank:
                local_target_compute_ms = (
                    float(target_window_events[0].elapsed_time(target_window_events[1]))
                    if target_window_events is not None
                    else (target_ended - target_started) * 1000.0
                )
            # Split the request-visible interval at the last token commit.
            # Requests finishing in this cycle stop here; surviving requests
            # must additionally observe the accounting collective, host/D2H
            # bookkeeping, and any online refill below.
            cycle_accounting_split = time.perf_counter()
            cycle_elapsed = cycle_accounting_split - cycle_started
            if host_trace is not None:
                host_trace["accounting_split_seconds"] = cycle_accounting_split
            elapsed_values = [
                cycle_elapsed if self.rank == self.topology.target_leader_rank else 0.0,
                time.time() if self.rank == self.topology.target_leader_rank else 0.0,
            ]
            if linear_window_estimator is not None:
                elapsed_values.extend(
                    [
                        0.0,
                        local_target_compute_ms if self.rank == self.topology.target_leader_rank else 0.0,
                    ]
                )
            cycle_identity_start = len(elapsed_values)
            cycle_identity = [
                *cycle_active_indices,
                *([-1] * (initial_batch_size - len(cycle_active_indices))),
            ]
            elapsed_values.extend(cycle_identity)
            finish_offset_start = len(elapsed_values)
            finish_offsets = [-1.0] * initial_batch_size
            for slot, index in enumerate(cycle_active_indices):
                endpoint = finished_publication_endpoints.get(index)
                if endpoint is not None:
                    finish_offsets[slot] = max(
                        0.0,
                        endpoint - cycle_started,
                    )
            elapsed_values.extend(finish_offsets)
            elapsed_tensor = cycle_accounting_envelope.rewrite(elapsed_values)
            if (
                linear_window_estimator is not None
                and self.rank == self.topology.target_leader_rank
                and draft_compute_us is not None
            ):
                # Keep the timing scalar device-resident through the
                # proposal rendezvous and fold it into the existing cycle
                # broadcast.
                elapsed_tensor[2].copy_(
                    draft_compute_us.to(
                        device=elapsed_tensor.device,
                        dtype=torch.float64,
                    )
                    / 1000.0
                )
            elapsed_results = cycle_accounting_envelope.broadcast_and_materialize(
                source_rank=self.topology.target_leader_rank,
                group=accounting_group,
            )
            cycle_elapsed_value, cycle_wall_time = (float(value) for value in elapsed_results[:2])
            if linear_window_estimator is not None:
                observed_draft_ms, observed_target_ms = (float(value) for value in elapsed_results[2:4])
                linear_window_estimator.observe(
                    draft_compute_ms=observed_draft_ms,
                    # The event is intentionally read one cycle late.  Pair
                    # its duration with the previous cycle's logical token
                    # count rather than the current proposal shape on *every*
                    # rank.  Only the draft leader owns the timing event, but
                    # the lagged duration is broadcast to the whole world;
                    # pairing it with the current work size on target ranks
                    # made their rolling-eager gate diverge whenever the
                    # request count changed.
                    drafted_tokens=reported_draft_token_count,
                    target_verify_ms=observed_target_ms,
                    eager_work=False,
                )
                counters["spec_rhythm_linear_w_last_observed_draft_ms"] = observed_draft_ms
                counters["spec_rhythm_linear_w_last_observed_target_ms"] = observed_target_ms
                if host_trace is not None:
                    host_trace["linear_w_observed_draft_ms"] = observed_draft_ms
                    host_trace["linear_w_observed_target_ms"] = observed_target_ms
            received_cycle_identity = [
                int(round(value))
                for value in elapsed_results[cycle_identity_start : cycle_identity_start + initial_batch_size]
            ]
            if received_cycle_identity != cycle_identity:
                raise RuntimeError("SpecRhythm active-row order diverged across ranks before finish accounting.")
            finish_offsets = [
                float(value)
                for value in elapsed_results[finish_offset_start : finish_offset_start + initial_batch_size]
            ]
            last_cycle_ms = cycle_elapsed_value * 1000.0
            local_finished_set = set(finished_this_cycle)
            leader_finished_set = decode_finish_envelope(
                cycle_active_indices,
                finish_offsets,
                cycle_elapsed_value,
                label="SpecRhythm",
            )
            if leader_finished_set != local_finished_set:
                raise RuntimeError("SpecRhythm finish decisions diverged across ranks.")
            finished_set = leader_finished_set
            for slot, index in enumerate(cycle_active_indices):
                elapsed_ms = last_cycle_ms
                if index in finished_set:
                    finish_offset = finish_offsets[slot]
                    elapsed_ms = finish_offset * 1000.0
                runtime_states[index].add_decode_time(elapsed_ms)
            for index in first_decode_indices:
                # Exclude the cycle that emitted the first measured token from
                # the steady-state TPOT interval, just as vLLM excludes TTFT.
                runtime_states[index].decode_start_elapsed_ms = runtime_states[index].decode_elapsed_ms

            if finished_this_cycle:
                for request_index in finished_this_cycle:
                    local_states[request_index].finished_decode_elapsed_ms = runtime_states[
                        request_index
                    ].decode_elapsed_ms
                    completed_states[request_index] = local_states[request_index].clone()
                    invalidate_request(request_index)
                release_online_cache(finished_this_cycle)
                active_indices = [index for index in active_indices if index not in finished_set]
            tail_incumbent_indices = tuple(active_indices)
            refill_started = time.perf_counter()
            # The role-local work for this reservation completed before the
            # communication phase above.  Publish its first token only at the
            # cycle boundary: incumbents observe the real model cost, while a
            # new request never inherits time preceding its first token.
            activate_staged_prefill()
            stopping_after_cycle = bool(
                (max_rounds is not None and round_count + 1 >= max_rounds)
                or (
                    self.config.stop_after_profiled_decode_steps
                    and profiled_decode_steps >= self.config.profile_decode_steps
                )
            )
            next_cycle_uses_fallback = bool(
                self.config.spec_rhythm_target_fallback_max_batch
                and active_indices
                and len(active_indices) <= self.config.spec_rhythm_target_fallback_max_batch
            )
            if not stopping_after_cycle:
                admit_available(
                    now=cycle_wall_time,
                    stage_for_overlap=bool(active_indices and not next_cycle_uses_fallback),
                )
            refill_ended = time.perf_counter()
            phase_seconds["refill"] += refill_ended - refill_started
            if host_trace is not None:
                host_trace["refill_start_seconds"] = refill_started
                host_trace["refill_end_seconds"] = refill_ended
                host_trace["cycle_end_seconds"] = refill_ended
                host_timeline.append(host_trace)
            local_frontier_signature = service_frontier_signature()
            if trace_requests and self.rank == self.topology.target_leader_rank:
                ready_detail = {int(index): int(ticket.proposal_id) for index, ticket in controller.ready.items()}
                eager_detail = {
                    int(index): int(ticket.proposal_id) for index, ticket in controller.staged_eager.items()
                }
                payload_detail = {
                    int(proposal_id): int(payload.ticket.request_index) for proposal_id, payload in payloads.items()
                }
                print(
                    "[SpecRhythm frontier] "
                    f"round={round_count} active={active_indices} "
                    f"ready={ready_detail} staged_eager={eager_detail} "
                    f"payloads={payload_detail}",
                    flush=True,
                )
            tail_elapsed_values = [
                refill_ended - cycle_accounting_split if self.rank == self.topology.target_leader_rank else 0.0,
                refill_ended if self.rank == self.topology.target_leader_rank else 0.0,
                *(
                    local_frontier_signature
                    if self.rank == self.topology.target_leader_rank
                    else (0,) * len(local_frontier_signature)
                ),
            ]
            tail_accounting_envelope.rewrite(tail_elapsed_values)
            tail_elapsed_values = tail_accounting_envelope.broadcast_and_materialize(
                source_rank=self.topology.target_leader_rank,
                group=accounting_group,
            )
            tail_elapsed_ms = float(tail_elapsed_values[0]) * 1000.0
            if self.device.type != "cpu":
                validate_service_frontier(tail_elapsed_values[2:])
            for index in tail_incumbent_indices:
                runtime_states[index].add_decode_time(tail_elapsed_ms)
            new_activation_tail_ms = charge_activation_tail(float(tail_elapsed_values[1]))
            if tail_incumbent_indices:
                last_cycle_ms += tail_elapsed_ms
                counters["spec_rhythm_cycle_accounted_tail_ms"] += tail_elapsed_ms
                counters["spec_rhythm_cycle_accounted_tail_cycles"] += 1
            if host_trace is not None:
                host_trace["accounted_tail_ms"] = tail_elapsed_ms if tail_incumbent_indices else 0.0
                host_trace["new_activation_tail_request_ms"] = new_activation_tail_ms
                host_trace["accounting_boundary_seconds"] = refill_ended
            # Deliberately retain the endpoint from before the tail timing
            # collective.  On the next cycle its HCCL/D2H cost is part of the
            # primary elapsed interval, so no host gap can disappear between
            # adjacent cycles.  Newly admitted requests begin at this boundary;
            # completed requests were snapshotted before tail accounting.
            cycle_accounting_started = refill_ended
            round_count += 1
            if (
                self.config.stop_after_profiled_decode_steps
                and profiled_decode_steps >= self.config.profile_decode_steps
            ):
                break

        for request_index in active_indices:
            invalidate_request(request_index)
        if npu_profiler is not None:
            npu_profiler.stop()
        torch.npu.synchronize()
        decode_elapsed = time.perf_counter() - started
        self.last_worker_decode_phase_seconds = dict(phase_seconds)
        self.last_worker_decode_profile_seconds = dict(decode_profile_seconds)
        self.last_worker_decode_profile_detail_seconds = {}
        self.last_worker_decode_host_timeline = host_timeline
        self.last_worker_profiled_decode_steps = profiled_decode_steps
        self.last_worker_decode_counters = dict(counters)
        self.last_worker_decode_counters.update(
            {
                "spec_rhythm_slo_adaptive": int(has_slo_constraints),
                "spec_rhythm_eager_enabled": int(bool(effective_eager_cap)),
                "spec_rhythm_priority_enabled": int(effective_priority),
                "spec_rhythm_effective_eager_cap": int(effective_eager_cap),
                "spec_rhythm_merge_ready_homes": int(bool(self.config.spec_rhythm_merge_ready_homes)),
            }
        )
        elapsed = prefill_elapsed + decode_elapsed
        if continuous_batching:
            result_states = [completed_states.get(index, local_states[index]) for index in range(len(local_states))]
        else:
            result_states = target_states
        results: list[dict[str, Any]] = []
        for sequence_index, state in enumerate(result_states):
            completion_token_ids = _truncate_completion(
                state.committed_completion_token_ids,
                self.eos_token_ids,
                state.max_tokens if max_rounds is None else len(state.committed_completion_token_ids),
                state.ignore_eos if max_rounds is None else True,
            )
            acceptance_lengths = state.acceptance_lengths
            request_decode_elapsed_ms = (
                state.finished_decode_elapsed_ms
                if state.finished_decode_elapsed_ms is not None
                else runtime_states[sequence_index].decode_elapsed_ms
            )
            measured_decode_elapsed_ms = max(
                0.0,
                request_decode_elapsed_ms - (runtime_states[sequence_index].decode_start_elapsed_ms or 0.0),
            )
            measured_decode_tokens = max(0, len(completion_token_ids) - 1)
            observed_tpot_ms = measured_decode_elapsed_ms / max(1, measured_decode_tokens)
            paper_tpot_ms = measured_decode_elapsed_ms / max(1, len(completion_token_ids))
            slo_attained = None if state.slo_tpot_ms is None else observed_tpot_ms <= state.slo_tpot_ms
            finish_reason = (
                "abort" if state.aborted else "length" if len(completion_token_ids) >= state.max_tokens else "stop"
            )
            results.append(
                {
                    "completion_token_ids": completion_token_ids,
                    "accepted_draft_tokens": state.accepted_draft_tokens,
                    "verified_draft_tokens": state.verified_draft_tokens,
                    "verification_rounds": state.verification_rounds,
                    "acceptance_rate": (
                        state.accepted_draft_tokens / state.verified_draft_tokens
                        if state.verified_draft_tokens
                        else 0.0
                    ),
                    "num_acc_tokens": acceptance_lengths,
                    "mean_accept_tokens": (
                        sum(acceptance_lengths) / len(acceptance_lengths) if acceptance_lengths else 0.0
                    ),
                    "temperature": state.temperature,
                    "draft_temperature": state.draft_temperature,
                    "top_p": state.top_p,
                    "top_k": state.top_k,
                    "draft_top_p": state.draft_top_p,
                    "draft_top_k": state.draft_top_k,
                    "max_tokens": state.max_tokens,
                    "ignore_eos": state.ignore_eos,
                    "slo_tpot_ms": state.slo_tpot_ms,
                    "slo_class": state.slo_class,
                    "request_id": request_params[sequence_index].request_id,
                    "arrival_ts": request_params[sequence_index].arrival_ts,
                    "finish_reason": finish_reason,
                    "observed_tpot_ms": observed_tpot_ms,
                    "tpot_definition": "decode_after_first_token_ms / (output_tokens - 1)",
                    "paper_tpot_ms": paper_tpot_ms,
                    "paper_tpot_definition": "same_decode_elapsed_ms / output_tokens",
                    "paper_slo_attained": (None if state.slo_tpot_ms is None else paper_tpot_ms <= state.slo_tpot_ms),
                    "paper_slo_goodput_tokens": (
                        len(completion_token_ids)
                        if state.slo_tpot_ms is None or paper_tpot_ms <= state.slo_tpot_ms
                        else 0
                    ),
                    "slo_attained": slo_attained,
                    "slo_goodput_tokens": (len(completion_token_ids) if slo_attained is not False else 0),
                    "round_count": round_count,
                    "decode_phase_seconds": dict(phase_seconds),
                    "spec_rhythm": dict(counters),
                    "elapsed_seconds": elapsed,
                    "prefill_elapsed_seconds": prefill_elapsed,
                    "decode_elapsed_seconds": decode_elapsed,
                    "cached_prompt_tokens": self.cache_allocation.num_cached_tokens[sequence_index],
                    "tree_mode": False,
                    "linear_full_window": linear_full_window,
                    "gamma": self.gamma,
                    "tree_width": 1,
                    "tree_depth": 1,
                    **self.graph_metrics(),
                }
            )
        self._release_cache()
        return results if self.rank == self.topology.target_leader_rank else None

    @torch.inference_mode()
    def generate_target_ar_batch(
        self,
        prompt_token_ids: list[list[int]],
        sampling_params: NativeSamplingParams | Sequence[NativeSamplingParams] | None = None,
    ) -> list[dict[str, Any]] | None:
        """Generate a static batch autoregressively on the target model only."""
        if not prompt_token_ids or len(prompt_token_ids) > self.config.max_num_seqs:
            raise ValueError("PEARL batch size must be between one and max_num_seqs.")
        if sum(len(prompt) for prompt in prompt_token_ids) > self.config.max_num_batched_tokens:
            raise ValueError("PEARL prompts exceed max_num_batched_tokens.")
        request_params = _normalize_sampling_params(len(prompt_token_ids), sampling_params, self.config.max_tokens)
        for prompt, params in zip(prompt_token_ids, request_params):
            if not prompt:
                raise ValueError("Target AR generation requires non-empty prompts.")
            if len(prompt) + params.max_tokens > self.config.max_model_len:
                raise ValueError("Prompt plus target completion exceeds max_model_len.")

        if self.is_draft:
            return None

        initial_tokens = [[int(token_id) for token_id in prompt] for prompt in prompt_token_ids]
        self._allocate_cache(
            initial_tokens,
            enable_prefix_caching=self.config.enable_prefix_caching,
        )
        states = [
            PearlPipelineState(
                tokens,
                len(tokens),
                temperature=params.temperature,
                draft_temperature=params.draft_temperature,
                top_p=params.top_p,
                top_k=params.top_k,
                draft_top_p=params.draft_top_p,
                draft_top_k=params.draft_top_k,
                max_tokens=params.max_tokens,
                ignore_eos=params.ignore_eos,
                slo_tpot_ms=params.slo_tpot_ms,
                slo_class=params.slo_class,
                request_id=params.request_id,
                arrival_ts=params.arrival_ts,
            )
            for tokens, params in zip(initial_tokens, request_params)
        ]
        torch.npu.synchronize()
        prefill_started = time.perf_counter()
        prefill_chunk_size = self.config.prefill_chunk_size or self.config.max_num_seqs
        target_tokens: list[int] = []
        for start in range(0, len(initial_tokens), prefill_chunk_size):
            end = start + prefill_chunk_size
            target_tokens.extend(
                self._target_ar_prefill(
                    initial_tokens[start:end],
                    states[start:end],
                    list(range(start, min(end, len(initial_tokens)))),
                )
            )
        torch.npu.synchronize()
        prefill_elapsed = time.perf_counter() - prefill_started
        for state, target_token in zip(states, target_tokens):
            state.token_ids.append(target_token)
            assert state.committed_length is not None
            state.committed_length += 1
        self._capture_target_ar_graph(states)

        torch.npu.synchronize()
        started = time.perf_counter()
        while True:
            active_indices = [index for index, state in enumerate(states) if not _finished(state, self.eos_token_ids)]
            if not active_indices:
                break
            input_token_ids = [states[index].token_ids[-1] for index in active_indices]
            positions = [len(states[index].token_ids) - 1 for index in active_indices]
            target_sampling_kwargs: dict[str, Any] = {}
            top_ps = [states[index].top_p for index in active_indices]
            top_ks = [states[index].top_k for index in active_indices]
            if any(value < 1.0 for value in top_ps) or any(value > 0 for value in top_ks):
                target_sampling_kwargs.update(top_ps=top_ps, top_ks=top_ks)
            next_tokens = self._run_packed_sample(
                input_token_ids,
                active_indices,
                positions,
                [states[index].temperature for index in active_indices],
                **target_sampling_kwargs,
            )
            if any(states[index].temperature > 0 for index in active_indices):
                dist.broadcast(
                    next_tokens,
                    src=self.topology.target_leader_rank,
                    group=self.groups.target_group,
                )
            for sequence_index, token_id in zip(active_indices, next_tokens.cpu().tolist()):
                states[sequence_index].token_ids.append(int(token_id))
                assert states[sequence_index].committed_length is not None
                states[sequence_index].committed_length += 1

        torch.npu.synchronize()
        decode_elapsed = time.perf_counter() - started
        elapsed = prefill_elapsed + decode_elapsed
        results = []
        for sequence_index, state in enumerate(states):
            results.append(
                {
                    "completion_token_ids": _truncate_completion(
                        state.committed_completion_token_ids,
                        self.eos_token_ids,
                        state.max_tokens,
                        state.ignore_eos,
                    ),
                    "request_id": state.request_id,
                    "arrival_ts": state.arrival_ts,
                    "accepted_draft_tokens": 0,
                    "verified_draft_tokens": 0,
                    "acceptance_rate": 0.0,
                    "num_acc_tokens": [],
                    "mean_accept_tokens": 0.0,
                    "temperature": state.temperature,
                    "draft_temperature": state.draft_temperature,
                    "top_p": state.top_p,
                    "top_k": state.top_k,
                    "draft_top_p": state.draft_top_p,
                    "draft_top_k": state.draft_top_k,
                    "max_tokens": state.max_tokens,
                    "ignore_eos": state.ignore_eos,
                    "slo_tpot_ms": state.slo_tpot_ms,
                    "slo_class": state.slo_class,
                    "observed_tpot_ms": (decode_elapsed * 1000.0 / max(1, len(state.committed_completion_token_ids))),
                    "slo_attained": (
                        None
                        if state.slo_tpot_ms is None
                        else (
                            decode_elapsed * 1000.0 / max(1, len(state.committed_completion_token_ids))
                            <= state.slo_tpot_ms
                        )
                    ),
                    "slo_goodput_tokens": (
                        len(state.committed_completion_token_ids)
                        if state.slo_tpot_ms is None
                        or (
                            decode_elapsed * 1000.0 / max(1, len(state.committed_completion_token_ids))
                            <= state.slo_tpot_ms
                        )
                        else 0
                    ),
                    "elapsed_seconds": elapsed,
                    "prefill_elapsed_seconds": prefill_elapsed,
                    "decode_elapsed_seconds": decode_elapsed,
                    "cached_prompt_tokens": self.cache_allocation.num_cached_tokens[sequence_index],
                    "gamma": 0,
                    **self.graph_metrics(),
                }
            )
        self._release_cache()
        return results if self.rank == self.topology.target_leader_rank else None

    @torch.inference_mode()
    def target_tree_forward(
        self,
        plans: Sequence[TreeSpeculationPlan],
        root_token_ids: Sequence[int],
        draft_token_ids: Sequence[Sequence[int]],
        sequence_ids: Sequence[int] | None = None,
        return_logits: bool = True,
        profile_subphases: bool = False,
        real_tree_count: int | None = None,
        *,
        temperatures: Sequence[float] | None = None,
        top_ps: Sequence[float] | None = None,
        top_ks: Sequence[int] | None = None,
    ) -> dict[str, Any] | None:
        """Run the native target forward for request-local tree candidates.

        The explicit ancestor mask is a dense-attention correctness path.  A
        fixed tree shape can be captured by the ordinary target ACLGraph; the
        graph runner validates replay and falls back to eager when the CANN
        task shape is not stable.  KV writes use the same native cache in both
        modes, and target argmax tensors stay on device until verification.
        """
        if self.is_draft:
            return None
        profile_subphases = profile_subphases or bool(getattr(self, "_profile_tree_subphases", False))
        profile_seconds = {
            "target_tree_setup": 0.0,
            "target_tree_metadata": 0.0,
            "target_tree_model": 0.0,
            "target_tree_output": 0.0,
        }

        def profile_start() -> float | None:
            if not profile_subphases:
                return None
            torch.npu.synchronize()
            return time.perf_counter()

        def profile_stop(started_at: float | None, phase: str) -> None:
            if started_at is None:
                return
            torch.npu.synchronize()
            profile_seconds[phase] += time.perf_counter() - started_at

        setup_started = profile_start()
        plan_list = list(plans)
        roots = [int(value) for value in root_token_ids]
        candidates = [list(map(int, row)) for row in draft_token_ids]
        request_ids = list(range(len(plan_list))) if sequence_ids is None else [int(value) for value in sequence_ids]
        real_count = len(plan_list) if real_tree_count is None else int(real_tree_count)
        if self.cache_allocation is None or self.cache_block_tables is None:
            raise RuntimeError("Allocate a target PEARL cache before tree forward")
        if (
            not plan_list
            or len(plan_list) != len(roots)
            or len(plan_list) != len(candidates)
            or len(plan_list) != len(request_ids)
        ):
            raise ValueError("tree plans, roots and candidate rows must have equal non-zero length")
        if not 1 <= real_count <= len(plan_list):
            raise ValueError("real_tree_count must identify a non-empty prefix of tree rows")
        request_temperatures = (
            [0.0] * len(plan_list) if temperatures is None else [float(value) for value in temperatures]
        )
        if len(request_temperatures) != len(plan_list):
            raise ValueError("tree temperatures must contain one value per request")
        if not (all(value == 0 for value in request_temperatures) or all(value > 0 for value in request_temperatures)):
            raise ValueError("tree temperatures must be either all zero or all positive")
        request_top_ps = [1.0] * len(plan_list) if top_ps is None else [float(value) for value in top_ps]
        request_top_ks = [0] * len(plan_list) if top_ks is None else [int(value) for value in top_ks]
        if len(request_top_ps) != len(plan_list) or len(request_top_ks) != len(plan_list):
            raise ValueError("tree top-p/top-k values must contain one value per request")
        sequence_ids: list[int] = []
        positions: list[int] = []
        padded_candidates: list[list[int]] = []
        for row_id, (sequence_id, (plan, row, root)) in enumerate(zip(request_ids, zip(plan_list, candidates, roots))):
            expected = plan.parent_indices.numel()
            active = int(getattr(plan, "candidate_budget", expected))
            if not 1 <= active <= expected or len(row) not in (active, expected):
                raise ValueError("tree candidate row does not match its plan")
            # Pack exactly the selected ancestor-closed nodes. Refined plans
            # already carry remapped parents; no discarded node is restored.
            plan = pack_selected_tree_plan(plan, range(active))
            plan_list[row_id] = plan
            padded_candidates.append(row[:active])
            cache_positions = plan.cache_positions
            assert cache_positions is not None
            local_positions = [int(value) for value in cache_positions.detach().cpu().tolist()]
            sequence_ids.extend([sequence_id] * len(local_positions))
            positions.extend(local_positions)
        profile_stop(setup_started, "target_tree_setup")
        metadata_started = profile_start()
        self._ensure_cache_capacity(sequence_ids, positions)
        input_ids, packed_positions, metadata = self.model.make_tree_attention_metadata(
            plan_list,
            roots,
            padded_candidates,
            self.cache_allocation.block_tables,
            sequence_ids=request_ids,
            # A dynamically shaped tree may happen to select only its primary
            # chain in one cycle.  Switching that cycle to causal FIA changes
            # the graph/operator contract and reintroduces per-layer task
            # updates for every new context length.  Keep verification on one
            # FULL-mask FIA path; graph-local KV bucketing masks its tail and
            # preserves the exact selected tree semantics.
            allow_causal_fast_path=False,
        )
        # The request and total-query buckets above are stable, but a cycle
        # may contain no four-candidate real row.  Keep the FULL-mask maxQ
        # extent fixed at the configured tree envelope without changing any
        # request's exact FIA query/KV lengths.
        if (
            metadata.use_fused_infer_attention
            and getattr(metadata, "tree_attention", False)
            and getattr(metadata, "tree_attention_mask", None) is not None
        ):
            graph_query_width = (
                int(
                    getattr(
                        self.config,
                        "spec_rhythm_tree_width",
                        max(plan.width for plan in plan_list),
                    )
                )
                * int(
                    getattr(
                        self.config,
                        "spec_rhythm_tree_depth",
                        max(plan.depth for plan in plan_list),
                    )
                )
                + 1
            )
            current_width = int(metadata.tree_attention_mask.shape[2])
            if current_width < graph_query_width:
                padded_tree_mask = torch.ones(
                    (
                        metadata.tree_attention_mask.shape[0],
                        metadata.tree_attention_mask.shape[1],
                        graph_query_width,
                        metadata.tree_attention_mask.shape[3],
                    ),
                    dtype=metadata.tree_attention_mask.dtype,
                    device=metadata.tree_attention_mask.device,
                )
                padded_tree_mask[:, :, :current_width].copy_(metadata.tree_attention_mask)
                if isinstance(metadata, NativeAttentionMetadata):
                    metadata = replace(
                        metadata,
                        tree_attention_mask=padded_tree_mask,
                    )
                else:
                    # Lightweight protocol/unit-test doubles are mutable and
                    # intentionally do not implement the metadata dataclass.
                    metadata.tree_attention_mask = padded_tree_mask
        profile_stop(metadata_started, "target_tree_metadata")
        # Tree masks are dense request-local masks, so they cannot use the
        # linear FIA contract.  They can nevertheless be captured by the
        # ordinary target ACLGraph when the complete packed tree shape is
        # stable.  The graph runner owns runtime validation and falls back to
        # eager when capture/replay is unavailable or mismatches eager.
        tree_graph_enabled = not self.config.enforce_eager and envs.VLLM_ASCEND_SPECRHYTHM_TREE_GRAPH
        stochastic = any(value > 0 for value in request_temperatures)
        packed_temperatures = [
            temperature
            for temperature, plan in zip(request_temperatures, plan_list)
            for _ in range(int(plan.parent_indices.numel()) + 1)
        ]
        packed_top_ps = [
            value for value, plan in zip(request_top_ps, plan_list) for _ in range(plan.candidate_budget + 1)
        ]
        packed_top_ks = [
            value for value, plan in zip(request_top_ks, plan_list) for _ in range(plan.candidate_budget + 1)
        ]
        # The draft model must be able to consume every committed target
        # token on the following cycle.  For heterogeneous-vocabulary pairs,
        # sample/argmax only from the common prefix validated at startup.
        verification_vocabulary_size = getattr(self, "draft_vocab_size", self.target_vocab_size)
        model_started = profile_start()
        if tree_graph_enabled and stochastic:
            target_logits = self.graph_runner.run_tree_logits(
                input_ids,
                packed_positions,
                metadata,
                verification_vocabulary_size,
            )
            target_tokens = _sample_logits(
                target_logits,
                packed_temperatures,
                top_ps=packed_top_ps,
                top_ks=packed_top_ks,
            )
        elif tree_graph_enabled:
            target_outputs = self.graph_runner.run_target_greedy(
                [input_ids],
                [packed_positions],
                [metadata],
                verification_vocabulary_size,
            )
            target_tokens = target_outputs[0]
            target_logits = None
        else:
            hidden_states = self.model(input_ids, packed_positions, metadata)
            if stochastic:
                target_logits = self.model.compute_logits(hidden_states)[:, :verification_vocabulary_size]
                target_tokens = _sample_logits(
                    target_logits,
                    packed_temperatures,
                    top_ps=packed_top_ps,
                    top_ks=packed_top_ks,
                )
            else:
                target_tokens = self.model.compute_greedy_tokens(
                    hidden_states,
                    verification_vocabulary_size,
                )
                # The greedy service loop needs only token IDs. Preserve the
                # diagnostic helper's legacy logits result when explicitly
                # used, but do not run a second vocabulary projection in
                # every cycle.
                target_logits = self.model.compute_logits(hidden_states) if return_logits else None
        profile_stop(model_started, "target_tree_model")
        if stochastic and dist.is_initialized():
            # All target TP ranks must advance from the same sampled token.
            # Sampling independently on identical logits is still divergent
            # because each process owns a different RNG stream.
            dist.broadcast(
                target_tokens,
                src=self.topology.target_leader_rank,
                group=self.groups.target_group,
            )
        output_started = profile_start()
        target_token_rows: list[torch.Tensor] = []
        target_query_rows: list[torch.Tensor] = []
        bonus_rows: list[torch.Tensor] = []
        cursor = 0
        active_node_counts: list[int] = []
        logical_query_count = 0
        for row_id, (plan, candidate_row, root_token) in enumerate(zip(plan_list, candidates, roots)):
            node_count = plan.parent_indices.numel()
            active_count = int(getattr(plan, "candidate_budget", node_count))
            if not 1 <= active_count <= node_count:
                raise ValueError("tree candidate budget does not match its plan")
            is_real = row_id < real_count
            if is_real:
                active_node_counts.append(active_count)
                logical_query_count += active_count + 1
            # The first target query predicts the root successor; each later
            # query predicts a tree node successor. The final query is exposed
            # separately as the bonus token expected by the verifier.
            row = target_tokens[cursor : cursor + node_count + 1]
            # Only active nodes are physically evaluated. Root + active node
            # outputs retain all possible accepted-branch frontier predictions.
            if is_real:
                target_token_rows.append(row[:active_count])
                # Keep the root-query output plus every active node output.  The
                # device verifier uses this row to select the bonus token at the
                # actual accepted branch frontier (which may be a sibling).
                target_query_rows.append(row[: active_count + 1])
                bonus_rows.append(row[active_count])
            cursor += node_count + 1
        profile_stop(output_started, "target_tree_output")
        return {
            "target_token_ids": torch.cat(target_token_rows),
            "target_query_token_ids": torch.cat(target_query_rows),
            "bonus_token_ids": torch.stack(bonus_rows),
            "num_draft_tokens": active_node_counts,
            "target_logits": target_logits,
            "used_aclgraph": bool(tree_graph_enabled and self.graph_runner.last_target_execution.used_aclgraph),
            "attention_backend": attention_backend_identity(metadata),
            "cache_slot_mapping": metadata.slot_mapping[:logical_query_count],
            "query_count": int(input_ids.numel()),
            "logical_query_count": logical_query_count,
            "graph_padding_query_count": int(input_ids.numel()) - logical_query_count,
            "tree_count": real_count,
            "profile_seconds": profile_seconds,
        }

    @staticmethod
    def _tree_eager_scratch_plan(plan: TreeSpeculationPlan, parent: TreeSpeculationPlan) -> TreeSpeculationPlan:
        """Keep an eager continuation off the uncommitted parent KV slots.

        Logical RoPE positions follow the accepted dependency path. Physical
        positions live after the entire parent tree; its rejected siblings are
        never exposed as a contiguous committed prefix to this continuation.
        """
        assert parent.cache_positions is not None
        parent_slots = parent.cache_positions.detach().cpu().tolist()
        path = tree_primary_path(parent)
        scratch_start = max(int(max(parent_slots)) + 1, int(plan.prefix_len))
        count = int(plan.parent_indices.numel())
        if scratch_start + count + 1 > plan.max_model_len:
            raise RuntimeError("SpecRhythm eager scratch tree exceeds max_model_len")
        slots = torch.arange(
            scratch_start,
            scratch_start + count + 1,
            dtype=plan.cache_positions.dtype,
            device=plan.positions.device,
        )
        mask = torch.ones_like(plan.attention_mask, dtype=torch.bool)
        mask[:, : parent.prefix_len] = False
        dependencies = [int(parent_slots[0]), *[int(parent_slots[node + 1]) for node in path]]
        mask[:, dependencies] = False
        mask[:, scratch_start] = False
        parents = plan.parent_indices.detach().cpu().tolist()
        for node in range(count):
            ancestor = node
            while ancestor >= 0:
                mask[node + 1, scratch_start + ancestor + 1] = False
                ancestor = parents[ancestor]
        return replace(plan, cache_positions=slots, attention_mask=mask)

    def _pack_tree_draft_level(self, requests):
        """Pack one or more query nodes per request into a common model call."""
        # The generic helper below is intentionally retained as the CPU
        # oracle.  On NPU it performs several tiny tensor constructions, a
        # device-side ``physical_blocks < 0`` check and one mask transfer per
        # request before concatenating everything again.  Profiling the B=8
        # tree loop showed that control path taking 6--8 ms for a model call
        # whose graph replay is only a few milliseconds.  Build the same
        # request-major FIA envelope from the authoritative host allocation
        # and transfer each packed field once instead.
        if self.device.type == "npu":
            if not requests:
                raise ValueError("tree draft level requires at least one request")
            if self.cache_allocation is None or self.cache_block_tables is None:
                raise RuntimeError("Allocate a draft PEARL cache before packing a tree level")
            attention = self.model.layers[0].self_attn
            block_size = int(attention.block_size)
            host_tables = self.cache_allocation.block_tables
            if isinstance(host_tables, torch.Tensor):
                host_tables = host_tables.detach().cpu().tolist()

            input_values: list[int] = []
            logical_positions: list[int] = []
            physical_positions: list[int] = []
            slot_values: list[int] = []
            query_sequence_ids: list[int] = []
            request_sequence_ids: list[int] = []
            query_lengths: list[int] = []
            sequence_lens: list[int] = []
            mask_rows: list[torch.Tensor] = []
            cumulative_query_lengths: list[int] = []
            causal_level = callable(getattr(self.model, "make_attention_metadata", None))

            for plan, request_id, node_indices, token_ids in requests:
                indices = (
                    [int(node_indices)] if isinstance(node_indices, int) else [int(value) for value in node_indices]
                )
                tokens = [int(token_ids)] if isinstance(token_ids, int) else [int(value) for value in token_ids]
                if not indices or len(indices) != len(tokens):
                    raise ValueError("tree level must contain aligned query nodes and tokens")
                expected_nodes = int(plan.parent_indices.numel())
                if any(index < -1 or index >= expected_nodes for index in indices):
                    raise ValueError("tree level node index is outside the plan")
                request_id = int(request_id)
                if not 0 <= request_id < len(host_tables):
                    raise ValueError("tree level block tables do not cover the request")
                cache_positions = getattr(plan, "cache_positions", None)
                if cache_positions is None:
                    cache_positions = plan.positions
                row_logical = [int(plan.positions[index + 1 if index >= 0 else 0]) for index in indices]
                row_physical = [int(cache_positions[index + 1 if index >= 0 else 0]) for index in indices]
                # A normal draft level contains only the primary spine.  Its
                # logical and physical positions are one contiguous causal
                # suffix, so the production TND FIA contract is exactly
                # equivalent to the FULL tree mask.  Eager scratch rows and
                # any future non-spine query keep the general FULL-mask path.
                causal_level = causal_level and (
                    row_logical == row_physical
                    and row_logical
                    == list(
                        range(
                            row_logical[0],
                            row_logical[0] + len(row_logical),
                        )
                    )
                )
                table = host_tables[request_id]
                row_slots = []
                for position in row_physical:
                    logical_block = position // block_size
                    if logical_block >= len(table) or int(table[logical_block]) < 0:
                        raise RuntimeError("tree level referenced an unallocated KV cache page")
                    row_slots.append(int(table[logical_block]) * block_size + position % block_size)
                row_mask = torch.stack([plan.attention_mask[index + 1 if index >= 0 else 0] for index in indices]).to(
                    device="cpu", dtype=torch.bool
                )

                input_values.extend(tokens)
                logical_positions.extend(row_logical)
                physical_positions.extend(row_physical)
                slot_values.extend(row_slots)
                query_sequence_ids.extend([request_id] * len(indices))
                request_sequence_ids.append(request_id)
                query_lengths.append(len(indices))
                sequence_lens.append(max(row_physical) + 1)
                mask_rows.append(row_mask)
                cumulative_query_lengths.append(
                    (cumulative_query_lengths[-1] if cumulative_query_lengths else 0) + len(indices)
                )

            if causal_level:
                packed_positions, metadata = self.model.make_attention_metadata(
                    query_sequence_ids,
                    logical_positions,
                    self.cache_block_tables,
                    slot_mapping=slot_values,
                    use_fused_infer_attention=bool(getattr(attention, "uses_paged_attention", False)),
                )
                return (
                    torch.tensor(input_values, dtype=torch.long, device=self.device),
                    packed_positions,
                    metadata,
                )

            request_sequence_tensor = torch.tensor(request_sequence_ids, dtype=torch.long, device=self.device)
            physical_tables = self.cache_block_tables
            request_tables = physical_tables.index_select(0, request_sequence_tensor)
            metadata = NativeAttentionMetadata(
                slot_mapping=torch.tensor(slot_values, dtype=torch.int32, device=self.device),
                context_lens=torch.tensor([position + 1 for position in physical_positions], dtype=torch.int32),
                # FULL-tree FIA consumes request-major tables only. Reusing
                # the same tensor avoids an unused token-major index_select.
                block_tables=request_tables,
                actual_seq_lengths_q=tuple(cumulative_query_lengths),
                sequence_lens=tuple(sequence_lens),
                request_block_tables=request_tables,
                attention_mask=None,
                use_fused_infer_attention=bool(getattr(attention, "uses_paged_attention", False)),
                tree_attention=True,
                tree_attention_mask=make_tree_fia_mask(mask_rows).to(device=self.device),
            )
            return (
                torch.tensor(input_values, dtype=torch.long, device=self.device),
                torch.tensor(logical_positions, dtype=torch.long, device=self.device),
                metadata,
            )

        parts = [
            self.model.make_tree_level_attention_metadata(
                plan,
                request_id,
                ([int(node_indices)] if isinstance(node_indices, int) else [int(value) for value in node_indices]),
                ([int(token_ids)] if isinstance(token_ids, int) else [int(value) for value in token_ids]),
                self.cache_block_tables,
            )
            for plan, request_id, node_indices, token_ids in requests
        ]
        if len(parts) == 1:
            return parts[0]
        metadatas = [part[2] for part in parts]
        cumulative_query_lengths: list[int] = []
        for input_ids, _, _ in parts:
            cumulative_query_lengths.append(
                (cumulative_query_lengths[-1] if cumulative_query_lengths else 0) + int(input_ids.numel())
            )
        metadata = type(metadatas[0])(
            slot_mapping=torch.cat([value.slot_mapping for value in metadatas]),
            context_lens=torch.cat([value.context_lens for value in metadatas]),
            block_tables=torch.cat([value.block_tables for value in metadatas]),
            actual_seq_lengths_q=tuple(cumulative_query_lengths),
            sequence_lens=tuple(int(value.sequence_lens[0]) for value in metadatas),
            request_block_tables=(
                torch.cat([value.request_block_tables for value in metadatas])
                if all(value.request_block_tables is not None for value in metadatas)
                else None
            ),
            attention_mask=torch.cat([value.attention_mask for value in metadatas]),
            use_fused_infer_attention=all(value.use_fused_infer_attention for value in metadatas),
            tree_attention=True,
            tree_attention_mask=make_tree_fia_mask([value.attention_mask for value in metadatas]),
        )
        return torch.cat([part[0] for part in parts]), torch.cat([part[1] for part in parts]), metadata

    @torch.inference_mode()
    def materialize_selected_tree_kv(
        self,
        plans: Sequence[TreeSpeculationPlan],
        draft_token_ids: Sequence[Sequence[int]],
        selected_indices: Sequence[Sequence[int]],
        sequence_ids: Sequence[int],
    ) -> dict[str, Any] | None:
        """Write only selected tree nodes whose K/V was not produced by expansion.

        Spine expansion already evaluates the root and every primary node
        except the final depth. Siblings are leaves in the SpecRhythm tree.
        Consequently the remaining selected nodes are independent queries
        whose ancestors are resident and can be materialized in one batch.
        The old path re-evaluated root plus *all* exploratory nodes even though
        the fixed global B discarded most of them immediately afterwards.
        """
        if not self.is_draft:
            return None
        if not (len(plans) == len(draft_token_ids) == len(selected_indices) == len(sequence_ids)):
            raise ValueError("selected tree KV materialization inputs must be row-aligned")
        requests = []
        for plan, row, selection, sequence_id in zip(plans, draft_token_ids, selected_indices, sequence_ids):
            node_count = int(plan.parent_indices.numel())
            if len(row) != node_count:
                raise ValueError("selected tree KV row does not match its exploratory plan")
            indices = [int(value) for value in selection]
            if indices != sorted(set(indices)) or any(value < 0 or value >= node_count for value in indices):
                raise ValueError("selected tree KV indices must be unique and topologically ordered")
            first_unwritten_primary = max(0, int(plan.depth) - 1)
            missing = [value for value in indices if value >= first_unwritten_primary]
            if missing:
                requests.append(
                    (
                        plan,
                        int(sequence_id),
                        missing,
                        [int(row[value]) for value in missing],
                    )
                )
        if not requests:
            return {
                "model_calls": 0,
                "graph_calls": 0,
                "materialized_nodes": 0,
                "profile_seconds": {},
            }
        profile_subphases = bool(getattr(self, "_profile_tree_subphases", False))
        metadata_started = None
        if profile_subphases:
            torch.npu.synchronize()
            metadata_started = time.perf_counter()
        with _trace_region(self, "SpecSLO/DraftMaterializeMetadata"):
            input_ids, positions, metadata = self._pack_tree_draft_level(requests)
        profile_seconds: dict[str, float] = {}
        if metadata_started is not None:
            torch.npu.synchronize()
            profile_seconds["draft_tree_materialize_metadata"] = time.perf_counter() - metadata_started
        graph_runner = getattr(self, "graph_runner", None)
        use_graph = isinstance(graph_runner, NativeACLGraphRunner) and not getattr(
            getattr(self, "config", None), "enforce_eager", True
        )
        compute_started = None
        if profile_subphases:
            torch.npu.synchronize()
            compute_started = time.perf_counter()
        with _trace_region(self, "SpecSLO/DraftMaterializeModel"):
            if use_graph:
                graph_runner.run_tree_hidden(input_ids, positions, metadata)
                graph_calls = int(graph_runner.last_target_execution.used_aclgraph)
            else:
                self.model(input_ids, positions, metadata)
                graph_calls = 0
        if compute_started is not None:
            torch.npu.synchronize()
            profile_seconds["draft_tree_materialize_compute"] = time.perf_counter() - compute_started
        return {
            "model_calls": 1,
            "graph_calls": graph_calls,
            "materialized_nodes": int(input_ids.numel()),
            "profile_seconds": profile_seconds,
        }

    @torch.inference_mode()
    def draft_tree_forward(
        self,
        plans: Sequence[TreeSpeculationPlan],
        root_token_ids: Sequence[int],
        sequence_ids: Sequence[int],
        eager_parent_sources: Mapping[int, tuple[TreeSpeculationPlan, Sequence[int]]] | None = None,
        profile_subphases: bool = False,
        *,
        committed_catchup_token_ids: Sequence[Sequence[int]] | None = None,
        graph_padding_catchup_rows: Sequence[tuple[int, int, Sequence[int]]] | None = None,
        draft_temperatures: Sequence[float] | None = None,
        draft_top_ps: Sequence[float] | None = None,
        draft_top_ks: Sequence[int] | None = None,
    ) -> dict[str, Any] | None:
        """Expand normal and eager trees in unified per-depth draft batches."""
        if not self.is_draft:
            return None
        profile_subphases = profile_subphases or bool(getattr(self, "_profile_tree_subphases", False))
        profile_seconds = {
            "draft_tree_setup": 0.0,
            "draft_tree_level_compute": 0.0,
            "draft_tree_topk": 0.0,
            "draft_tree_materialize_metadata": 0.0,
            "draft_tree_materialize_compute": 0.0,
        }

        def profile_start() -> float | None:
            if not profile_subphases:
                return None
            torch.npu.synchronize()
            return time.perf_counter()

        def profile_stop(started_at: float | None, phase: str) -> None:
            if started_at is None:
                return
            torch.npu.synchronize()
            profile_seconds[phase] += time.perf_counter() - started_at

        setup_started = profile_start()
        plan_list = list(plans)
        roots = [int(value) for value in root_token_ids]
        request_ids = [int(value) for value in sequence_ids]
        if not plan_list or len(plan_list) != len(roots) or len(plan_list) != len(request_ids):
            raise ValueError("tree plans, roots and sequence IDs must have equal non-zero length")
        catchup_rows = (
            [()] * len(plan_list)
            if committed_catchup_token_ids is None
            else [tuple(int(token) for token in row) for row in committed_catchup_token_ids]
        )
        if len(catchup_rows) != len(plan_list):
            raise ValueError("committed draft catch-up rows must be request-aligned")
        padding_catchup_rows = [
            (int(request_id), int(prefix_len), tuple(int(token) for token in tokens))
            for request_id, prefix_len, tokens in (graph_padding_catchup_rows or ())
        ]
        if any(not tokens for _, _, tokens in padding_catchup_rows):
            raise ValueError("draft graph padding catch-up rows cannot be empty")
        request_temperatures = (
            [0.0] * len(plan_list) if draft_temperatures is None else [float(value) for value in draft_temperatures]
        )
        if len(request_temperatures) != len(plan_list) or any(value < 0 for value in request_temperatures):
            raise ValueError("draft tree temperatures must be non-negative and request-aligned")
        request_top_ps = [1.0] * len(plan_list) if draft_top_ps is None else [float(value) for value in draft_top_ps]
        request_top_ks = [0] * len(plan_list) if draft_top_ks is None else [int(value) for value in draft_top_ks]
        if len(request_top_ps) != len(plan_list) or len(request_top_ks) != len(plan_list):
            raise ValueError("draft tree top-p/top-k values must be request-aligned")
        if self.cache_allocation is None or self.cache_block_tables is None:
            raise RuntimeError("Allocate a draft PEARL cache before tree drafting")
        effective_roots = list(roots)
        eager_frontier_tokens: dict[int, int] = {}
        work_plans = list(plan_list)
        frontier_requests = []
        frontier_rows = []
        rows: list[list[int]] = []
        node_confidences: list[list[float]] = []
        for row_index, (plan, request_id) in enumerate(zip(plan_list, request_ids)):
            node_count = int(plan.width) * int(plan.depth)
            active_count = int(getattr(plan, "candidate_budget", node_count))
            if not 1 <= active_count <= node_count:
                raise ValueError("tree candidate budget does not match its plan")
            rows.append([-1] * node_count)
            node_confidences.append([0.0] * node_count)
            source = (eager_parent_sources or {}).get(request_id)
            if source is not None:
                parent_plan, parent_row = source
                path = tree_primary_path(parent_plan)
                if not path or len(parent_row) <= path[-1]:
                    raise RuntimeError("SpecRhythm eager tree parent row is incomplete")
                parent_index = path[-1]
                parent_token = int(parent_row[parent_index])
                parent_positions = parent_plan.cache_positions
                assert parent_positions is not None
                parent_position_values = [int(value) for value in parent_positions.detach().cpu().tolist()]
                self._ensure_cache_capacity([request_id] * len(parent_position_values), parent_position_values)
                frontier_requests.append((parent_plan, request_id, parent_index, parent_token))
                frontier_rows.append(row_index)
                work_plans[row_index] = self._tree_eager_scratch_plan(plan, parent_plan)
            work_slots = work_plans[row_index].cache_positions.detach().cpu().tolist()
            self._ensure_cache_capacity([request_id] * len(work_slots), work_slots)
        profile_stop(setup_started, "draft_tree_setup")

        graph_runner = getattr(self, "graph_runner", None)
        use_graph = isinstance(graph_runner, NativeACLGraphRunner) and not getattr(
            getattr(self, "config", None), "enforce_eager", True
        )
        defer_materialization = bool(getattr(self, "_defer_tree_materialization", False))
        graph_calls = 0
        model_calls = 0
        query_count = 0
        graph_padding_query_count = 0

        def run_level(requests):
            nonlocal graph_calls, model_calls, query_count
            with _trace_region(self, "SpecSLO/DraftLevelMetadata"):
                input_ids, positions, metadata = self._pack_tree_draft_level(requests)
            model_calls += 1
            query_count += int(input_ids.numel())
            compute_started = profile_start()
            with _trace_region(self, "SpecSLO/DraftLevelModel"):
                if use_graph:
                    logits = graph_runner.run_tree_logits(input_ids, positions, metadata, self.draft_vocab_size)
                    graph_calls += int(graph_runner.last_target_execution.used_aclgraph)
                    profile_stop(compute_started, "draft_tree_level_compute")
                    return logits
                hidden = self.model(input_ids, positions, metadata)
                logits = self.model.compute_logits(hidden)[:, : self.draft_vocab_size]
            profile_stop(compute_started, "draft_tree_level_compute")
            return logits

        if frontier_requests:
            frontier_logits = run_level(frontier_requests)
            frontier_temperatures = [request_temperatures[index] for index in frontier_rows]
            frontier_token_tensor = _sample_logits(
                frontier_logits,
                frontier_temperatures,
                top_ps=[request_top_ps[index] for index in frontier_rows],
                top_ks=[request_top_ks[index] for index in frontier_rows],
            )
            if any(value > 0 for value in frontier_temperatures) and dist.is_initialized():
                dist.broadcast(
                    frontier_token_tensor,
                    src=self.topology.draft_leader_rank,
                    group=self.groups.draft_group,
                )
            frontier_tokens = frontier_token_tensor.detach().cpu().tolist()
            for row_index, token in zip(frontier_rows, frontier_tokens):
                effective_roots[row_index] = int(token)
                eager_frontier_tokens[request_ids[row_index]] = int(token)

        for depth_index in range(max(int(plan.depth) for plan in work_plans)):
            level_rows = [index for index, plan in enumerate(work_plans) if depth_index < plan.depth]
            requests = []
            level_logit_indices: list[int] = []
            packed_query_count = 0
            for index in level_rows:
                catchup = catchup_rows[index] if depth_index == 0 else ()
                if catchup:
                    if eager_parent_sources and request_ids[index] in eager_parent_sources:
                        raise ValueError("ahead-of-turn draft rows cannot also carry committed catch-up")
                    if catchup[-1] != effective_roots[index]:
                        raise ValueError("committed draft catch-up must end at the current root token")
                if len(catchup) > 1:
                    catchup_depth = len(catchup) - 1
                    catchup_prefix = int(work_plans[index].prefix_len) - catchup_depth
                    if catchup_prefix < 0:
                        raise ValueError("committed draft catch-up starts before the sequence")
                    catchup_plan = cached_cpu_tree_speculation_plan(
                        1,
                        catchup_depth,
                        catchup_prefix,
                        self.config.max_model_len,
                        candidate_budget=catchup_depth,
                    )
                    requests.append(
                        (
                            catchup_plan,
                            request_ids[index],
                            list(range(-1, catchup_depth)),
                            list(catchup),
                        )
                    )
                    packed_query_count += len(catchup)
                else:
                    requests.append(
                        (
                            work_plans[index],
                            request_ids[index],
                            depth_index - 1,
                            (effective_roots[index] if depth_index == 0 else rows[index][depth_index - 1]),
                        )
                    )
                    packed_query_count += 1
                # Only the final catch-up/root query predicts this tree's
                # next candidate.  Earlier queries exist solely to restore
                # the accepted committed KV prefix in the same model call.
                level_logit_indices.append(packed_query_count - 1)
            if use_graph and getattr(self.config, "spec_rhythm_stable_graphs", True):
                # A fixed number of catch-up queries per request plus a
                # power-of-two request envelope leaves only O(log batch)
                # draft graph shapes.  Padding comes from distinct requests
                # in the other logical home, so NPU kernels never race while
                # writing one KV slot.  Their logits are discarded; no
                # logical proposal or scheduler budget changes.
                request_bucket = 1 << (len(requests) - 1).bit_length()
                padding_count = request_bucket - len(requests)
                if padding_count <= len(padding_catchup_rows):
                    for request_id, prefix_len, padding_tokens in padding_catchup_rows[:padding_count]:
                        if depth_index == 0 and len(padding_tokens) > 1:
                            padding_depth = len(padding_tokens) - 1
                            padding_plan = cached_cpu_tree_speculation_plan(
                                1,
                                padding_depth,
                                prefix_len - padding_depth,
                                self.config.max_model_len,
                                candidate_budget=padding_depth,
                            )
                            padding_indices: int | list[int] = list(range(-1, padding_depth))
                            padding_input: int | list[int] = list(padding_tokens)
                        else:
                            padding_plan = cached_cpu_tree_speculation_plan(
                                1,
                                1,
                                prefix_len,
                                self.config.max_model_len,
                                candidate_budget=1,
                            )
                            padding_indices = -1
                            padding_input = padding_tokens[-1]
                        requests.append(
                            (
                                padding_plan,
                                request_id,
                                padding_indices,
                                padding_input,
                            )
                        )
                        padding_queries = 1 if isinstance(padding_input, int) else len(padding_input)
                        packed_query_count += padding_queries
                        graph_padding_query_count += padding_queries
            logits = run_level(requests)
            if level_logit_indices != list(range(len(level_rows))):
                logits = logits.index_select(
                    0,
                    torch.tensor(
                        level_logit_indices,
                        dtype=torch.long,
                        device=logits.device,
                    ),
                )
            topk_started = profile_start()
            max_width = min(max(work_plans[index].width for index in level_rows), int(logits.shape[-1]))
            sampled_rows: list[tuple[list[int], list[float]]] = []
            if all(request_temperatures[index] == 0 for index in level_rows):
                # Greedy tree expansion previously launched softmax+topk and
                # forced a D2H synchronization once *per request*.  At the
                # B64 endpoint that serialized dozens of tiny vocabulary ops
                # after every draft depth and erased most draft/target
                # overlap.  The rows share one vocabulary and width, so issue
                # one batched pair and transfer the small [rows, width]
                # result once.  Probabilities are retained for the scheduler;
                # token selection is exactly the same as the row-wise path.
                probabilities = torch.softmax(logits[: len(level_rows)].float(), dim=-1)
                values, tokens = probabilities.topk(max_width, dim=-1)
                token_rows = tokens.detach().cpu().tolist()
                probability_rows = values.detach().cpu().tolist()
                sampled_rows = [
                    (
                        [int(value) for value in token_row],
                        [float(value) for value in probability_row],
                    )
                    for token_row, probability_row in zip(token_rows, probability_rows)
                ]
            else:
                for local_row, row_index in enumerate(level_rows):
                    temperature = request_temperatures[row_index]
                    row_logits = logits[local_row].float()
                    if temperature == 0:
                        probabilities = torch.softmax(row_logits, dim=-1)
                        values, tokens = probabilities.topk(max_width)
                    else:
                        probabilities = _sampling_probabilities(
                            row_logits.unsqueeze(0),
                            [temperature],
                            top_ps=[request_top_ps[row_index]],
                            top_ks=[request_top_ks[row_index]],
                        ).squeeze(0)
                        noise = torch.empty_like(probabilities).exponential_(1)
                        tokens = probabilities.div(noise.clamp_min(1e-10)).topk(max_width).indices
                        if dist.is_initialized():
                            dist.broadcast(
                                tokens,
                                src=self.topology.draft_leader_rank,
                                group=self.groups.draft_group,
                            )
                        values = probabilities.index_select(0, tokens)
                    sampled_rows.append(
                        (
                            [int(value) for value in tokens.detach().cpu().tolist()],
                            [float(value) for value in values.detach().cpu().tolist()],
                        )
                    )
            for row_index, (token_ids, probabilities) in zip(level_rows, sampled_rows):
                plan = work_plans[row_index]
                sibling_start = int(plan.depth) + depth_index * (int(plan.width) - 1)
                level_indices = [depth_index] + list(range(sibling_start, sibling_start + int(plan.width) - 1))
                for node_index, token_id, score in zip(level_indices, token_ids, probabilities):
                    rows[row_index][node_index] = int(token_id)
                    node_confidences[row_index][node_index] = float(score)
            profile_stop(topk_started, "draft_tree_topk")
        rows = [[row[0] if token < 0 else token for token in row] for row in rows]

        materialize_metadata_started = profile_start()
        if defer_materialization:
            if not eager_parent_sources:
                # Normal exploratory K/V is not published; the next turn
                # rebuilds its selected path from committed catch-up tokens.
                # Avoid materializing a device slot map that no consumer reads.
                cache_slot_mapping = None
            else:
                mapping_values: list[int] = []
                for request_id, plan in zip(request_ids, work_plans):
                    cache_positions = plan.cache_positions
                    assert cache_positions is not None
                    local_positions = [int(value) for value in cache_positions.detach().cpu().tolist()]
                    mapping_values.extend(
                        self._cache_slot_mapping(
                            [request_id] * len(local_positions),
                            local_positions,
                        )
                    )
                # Mixed normal/eager rows retain one aligned mapping envelope;
                # eager continuations need their scratch K/V at promotion.
                cache_slot_mapping = torch.tensor(mapping_values, dtype=torch.int32, device=self.device)
        else:
            packed_sequences: list[int] = []
            packed_positions: list[int] = []
            for request_id, plan in zip(request_ids, work_plans):
                cache_positions = plan.cache_positions
                assert cache_positions is not None
                local_positions = [int(value) for value in cache_positions.detach().cpu().tolist()]
                packed_sequences.extend([request_id] * len(local_positions))
                packed_positions.extend(local_positions)
            self._ensure_cache_capacity(packed_sequences, packed_positions)
            input_ids, positions, metadata = self.model.make_tree_attention_metadata(
                work_plans,
                effective_roots,
                rows,
                self.cache_allocation.block_tables,
                sequence_ids=request_ids,
            )
            cache_slot_mapping = metadata.slot_mapping
        profile_stop(materialize_metadata_started, "draft_tree_materialize_metadata")
        if not defer_materialization:
            model_calls += 1
            query_count += int(input_ids.numel())
            materialize_compute_started = profile_start()
            if use_graph:
                graph_runner.run_tree_hidden(input_ids, positions, metadata)
                graph_calls += int(graph_runner.last_target_execution.used_aclgraph)
            else:
                self.model(input_ids, positions, metadata)
            profile_stop(materialize_compute_started, "draft_tree_materialize_compute")
        active_counts = [
            int(getattr(plan, "candidate_budget", int(plan.width) * int(plan.depth))) for plan in plan_list
        ]
        output_device = torch.device("cpu") if defer_materialization else self.device
        parent_rows = [
            plan.parent_indices[:count].to(device=output_device) for plan, count in zip(plan_list, active_counts)
        ]
        return {
            "draft_token_ids": rows,
            "root_token_ids": effective_roots,
            "eager_frontier_tokens": eager_frontier_tokens,
            "parent_indices": torch.cat(parent_rows),
            "num_draft_tokens": active_counts,
            "draft_confidence": torch.tensor(
                [sum(row) / max(1, len(row)) for row in node_confidences],
                dtype=torch.float32,
                device=output_device,
            ),
            "node_confidences": [
                torch.tensor(row, dtype=torch.float32, device=output_device) for row in node_confidences
            ],
            "cache_slot_mapping": cache_slot_mapping,
            "query_count": query_count,
            "graph_padding_query_count": graph_padding_query_count,
            "model_calls": model_calls,
            "graph_calls": graph_calls,
            "profile_seconds": profile_seconds,
            "materialization_deferred": defer_materialization,
        }

    @torch.inference_mode()
    def execute_tree_round(
        self,
        plans: Sequence[TreeSpeculationPlan],
        root_token_ids: Sequence[int],
        draft_token_ids: Sequence[Sequence[int]],
        parent_indices: torch.Tensor,
        num_draft_tokens: Sequence[int] | None = None,
        *,
        placeholder_token_id: int = -1,
    ) -> TreeVerificationOutput | None:
        """Run target tree verification and return device-resident commit data."""
        target = self.target_tree_forward(plans, root_token_ids, draft_token_ids)
        if target is None:
            return None
        active_rows = [
            list(row[: int(getattr(plan, "candidate_budget", len(row)))]) for plan, row in zip(plans, draft_token_ids)
        ]
        flat_drafts = torch.tensor(
            [token for row in active_rows for token in row],
            dtype=torch.long,
            device=self.device,
        )
        active_counts = [len(row) for row in active_rows]
        if parent_indices.numel() != flat_drafts.numel():
            raise ValueError("parent_indices must contain one entry per active tree node")
        return self.verify_tree_outputs(
            flat_drafts,
            parent_indices.to(device=self.device),
            target["target_query_token_ids"],
            target["bonus_token_ids"],
            num_draft_tokens or active_counts,
            max(plan.depth for plan in plans),
            placeholder_token_id,
        )

    @torch.inference_mode()
    def compact_tree_round(
        self,
        cache_slot_mapping: torch.Tensor | None,
        accepted_node_indices: torch.Tensor | Sequence[Sequence[int]],
        num_draft_tokens: Sequence[int],
        *,
        host_slot_rows: Sequence[Sequence[int]] | None = None,
    ) -> None:
        """Compact accepted paths with one all-layer move for the whole batch.

        Invalid tail depths become fixed-shape self copies.  This mirrors the
        production vLLM tree compactor and avoids both dynamic host reads and a
        full K/V layer sweep for every request.
        """
        if host_slot_rows is not None:
            accepted_rows = (
                accepted_node_indices.detach().cpu().tolist()
                if isinstance(accepted_node_indices, torch.Tensor)
                else [list(map(int, row)) for row in accepted_node_indices]
            )
            if not (len(host_slot_rows) == len(accepted_rows) == len(num_draft_tokens)):
                raise ValueError("host tree KV compaction rows must be aligned")
            slot_pairs: list[tuple[int, int]] = []
            for slots, accepted, count_value in zip(
                host_slot_rows,
                accepted_rows,
                num_draft_tokens,
            ):
                count = int(count_value)
                row_slots = [int(value) for value in slots]
                if count <= 0 or len(row_slots) != count + 1:
                    raise ValueError("host tree KV compaction mapping has the wrong size")
                node_slots = row_slots[1:]
                for depth, node_value in enumerate(accepted):
                    node = int(node_value)
                    if node < -1 or node >= count:
                        raise ValueError("accepted tree node index exceeds its request-local mapping")
                    destination_index = min(depth, count - 1)
                    if node >= 0 and node != destination_index:
                        slot_pairs.append((node_slots[node], node_slots[destination_index]))
            if slot_pairs:
                # Preflight has already synchronized and validated these host
                # slots.  Transfer one compact pair table and move only real
                # branch corrections; rejected tails and primary-spine rows
                # were deterministic self-copies in the old fixed-depth plan.
                pairs = torch.tensor(
                    slot_pairs,
                    dtype=torch.int32,
                    device=self.device,
                )
                self._move_tree_cache_slots(pairs[:, 0], pairs[:, 1])
            return
        if cache_slot_mapping is None:
            raise ValueError("cache_slot_mapping is required without host slot rows")
        if not isinstance(accepted_node_indices, torch.Tensor):
            accepted_node_indices = torch.tensor(
                accepted_node_indices,
                dtype=torch.int32,
                device=cache_slot_mapping.device,
            )
        if cache_slot_mapping.ndim != 1:
            raise ValueError("cache_slot_mapping must be a flat tensor")
        if accepted_node_indices.ndim != 2 or accepted_node_indices.shape[0] != len(num_draft_tokens):
            raise ValueError("accepted_node_indices must contain one row per tree")
        cursor = 0
        source_batches = []
        destination_batches = []
        for row, count in enumerate(num_draft_tokens):
            count = int(count)
            if count <= 0 or cursor + count + 1 > cache_slot_mapping.numel():
                raise ValueError("tree KV compaction count exceeds its slot mapping")
            query_slots = cache_slot_mapping[cursor : cursor + count + 1]
            node_slots = query_slots[1:]
            accepted = accepted_node_indices[row].to(torch.long)
            if accepted.device.type != "npu" and (torch.any(accepted >= count) or torch.any(accepted < -1)):
                raise ValueError("accepted tree node index exceeds its request-local mapping")
            destination_indices = torch.arange(accepted.numel(), dtype=torch.long, device=accepted.device).clamp_max(
                count - 1
            )
            source_indices = torch.where(accepted >= 0, accepted, destination_indices)
            # Logical consecutive positions may cross non-consecutive KV
            # pages. Never derive physical destinations by root_slot + n.
            source_batches.append(node_slots.index_select(0, source_indices))
            destination_batches.append(node_slots.index_select(0, destination_indices))
            cursor += count + 1
        if cursor != cache_slot_mapping.numel():
            raise ValueError("tree KV compaction slot mapping contains an unused suffix")
        if source_batches:
            self._move_tree_cache_slots(torch.cat(source_batches), torch.cat(destination_batches))

    @staticmethod
    def _tree_row_requires_kv_compaction(
        accepted_node_indices: Sequence[int],
        num_draft_tokens: int,
    ) -> bool:
        """Return whether an accepted spine leaves its packed logical slots.

        Tree plans are spine-first, so the common top-1 accepted path already
        occupies destination nodes ``0..depth-1``. Rejected tail entries are
        also self-copies. Skipping those rows avoids a full all-layer KV sweep.
        """
        count = int(num_draft_tokens)
        if count <= 0:
            raise ValueError("tree KV compaction requires a positive candidate count")
        return any(
            int(node) >= 0 and int(node) != min(depth, count - 1) for depth, node in enumerate(accepted_node_indices)
        )

    def _move_promoted_tree_cache_slots(
        self,
        promotions: Sequence[tuple[int, int, TreeSpeculationPlan]],
        cache_mappings: dict[int, torch.Tensor],
        host_cache_mappings: Mapping[int, Sequence[int]],
    ) -> int:
        """Materialize every promoted eager tree with one all-layer move.

        ``promotions`` contains ``(proposal_id, request_index, plan)`` rows.
        The commit preflight has already copied and validated each source
        mapping on the host, so build one flat source/destination envelope
        instead of launching a complete K/V layer sweep per request.  Stored
        per-proposal mappings are read-only slices of the flat destination;
        their values, device and lifetime remain tied to the proposal entry.
        """

        if not promotions:
            return 0
        proposal_ids = [int(proposal_id) for proposal_id, _, _ in promotions]
        if len(set(proposal_ids)) != len(proposal_ids):
            raise ValueError("promoted eager tree proposal ids must be unique")
        source_slots: list[int] = []
        destination_slots: list[int] = []
        destination_ranges: list[tuple[int, int, int]] = []
        for proposal_id, request_index, plan in promotions:
            count = int(plan.parent_indices.numel()) + 1
            current_mapping = cache_mappings.get(int(proposal_id))
            if (
                not isinstance(current_mapping, torch.Tensor)
                or current_mapping.ndim != 1
                or current_mapping.numel() != count
            ):
                raise RuntimeError("promoted eager tree has no authoritative KV mapping")
            source = host_cache_mappings.get(int(proposal_id))
            if source is None or len(source) != count:
                raise RuntimeError("promoted eager tree is missing its preflighted KV mapping")
            destination = self._cache_slot_mapping(
                [int(request_index)] * count,
                list(range(int(plan.prefix_len), int(plan.prefix_len) + count)),
            )
            if len(destination) != count:
                raise RuntimeError("promoted eager tree destination mapping is incomplete")
            start = len(destination_slots)
            source_slots.extend(int(value) for value in source)
            destination_slots.extend(int(value) for value in destination)
            destination_ranges.append((int(proposal_id), start, count))

        source_tensor = torch.tensor(
            source_slots,
            dtype=torch.int32,
            device=self.device,
        )
        destination_tensor = torch.tensor(
            destination_slots,
            dtype=torch.int32,
            device=self.device,
        )
        self._move_tree_cache_slots(source_tensor, destination_tensor)
        for proposal_id, start, count in destination_ranges:
            cache_mappings[proposal_id] = destination_tensor.narrow(0, start, count)
        return len(destination_slots)

    def _move_tree_cache_slots(self, source: torch.Tensor, destination: torch.Tensor) -> None:
        layer_caches = getattr(self, "_tree_layer_caches", None)
        if layer_caches is None:
            layer_caches = self._collect_tree_layer_caches()
        graph_runner = getattr(self, "tree_kv_graph_runner", None)
        if graph_runner is not None:
            graph_runner.move(source, destination)
            return
        move_kv_cache_slots(layer_caches, source, destination)

    def _collect_tree_layer_caches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Validate and collect the model-lifetime paged KV tensors."""
        configured_device = getattr(self, "device", None)
        expected_device = (
            self._spec_rhythm_tree_cache_device(configured_device)
            if isinstance(configured_device, torch.device)
            else None
        )
        layer_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.model.layers:
            key_cache, value_cache = layer.self_attn.key_cache, layer.self_attn.value_cache
            for cache in (key_cache, value_cache):
                if not isinstance(cache, torch.Tensor) or cache.ndim != 4:
                    raise RuntimeError("SpecRhythm tree KV preflight requires allocated paged layer caches")
                if expected_device is not None and cache.device != expected_device:
                    raise RuntimeError("SpecRhythm tree KV cache is on the wrong device")
            layer_caches.append((key_cache, value_cache))
        if not layer_caches:
            raise RuntimeError("SpecRhythm tree KV preflight requires at least one layer cache")
        return layer_caches

    @staticmethod
    def _verify_tree_outputs_host(
        draft_token_rows: Sequence[Sequence[int]],
        plans: Sequence[TreeSpeculationPlan],
        target_query_token_ids: torch.Tensor,
        max_depth: int,
        output_device: torch.device,
        placeholder_token_id: int = -1,
    ) -> TreeVerificationOutput:
        """Traverse small greedy trees on the host with one D2H and one H2D.

        This is the optimized host-verdict path used while the captured NPU
        verifier is still slower than its control-plane cost.  It implements
        exactly ``verify_greedy_tree_batch``'s root-query convention: each
        request contributes ``candidate_count + 1`` target successors and
        the successor after the last accepted node is the bonus token.
        """
        if max_depth < 1 or not plans or len(draft_token_rows) != len(plans):
            raise ValueError("host tree verifier requires aligned non-empty plans")
        counts = [int(plan.candidate_budget) for plan in plans]
        expected = sum(count + 1 for count in counts)
        if target_query_token_ids.ndim != 1 or target_query_token_ids.numel() != expected:
            raise ValueError("host tree verifier target rows do not match candidate counts")
        target_values = [int(value) for value in target_query_token_ids.detach().cpu().tolist()]
        packed_output: list[list[int]] = []
        cursor = 0
        for row, plan, count in zip(draft_token_rows, plans, counts):
            candidates = [int(value) for value in row[:count]]
            if len(candidates) != count:
                raise ValueError("host tree verifier draft row is shorter than its plan")
            parents = [int(value) for value in plan.parent_indices[:count].detach().cpu().tolist()]
            targets = target_values[cursor : cursor + count + 1]
            cursor += count + 1
            emitted = [placeholder_token_id] * (max_depth + 1)
            accepted = [-1] * max_depth
            parent = -1
            prediction_index = 0
            active = True
            for depth in range(max_depth):
                if not active or prediction_index >= len(targets):
                    break
                prediction = targets[prediction_index]
                emitted[depth] = prediction
                selected = next(
                    (
                        index
                        for index, (candidate_parent, candidate) in enumerate(zip(parents, candidates))
                        if candidate_parent == parent and candidate == prediction
                    ),
                    -1,
                )
                if selected < 0:
                    active = False
                    break
                accepted[depth] = selected
                parent = selected
                prediction_index = selected + 1
            if active:
                emitted[max_depth] = targets[min(prediction_index, len(targets) - 1)]
            packed_output.append([*emitted, *accepted])
        packed = torch.tensor(packed_output, dtype=torch.long, device=output_device)
        return TreeVerificationOutput(
            packed[:, : max_depth + 1],
            packed[:, max_depth + 1 :],
        )

    @staticmethod
    @torch.inference_mode()
    def verify_tree_outputs(
        draft_token_ids: torch.Tensor,
        parent_indices: torch.Tensor,
        target_token_ids: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        num_draft_tokens: Sequence[int] | None,
        max_depth: int,
        placeholder_token_id: int = -1,
    ) -> TreeVerificationOutput:
        """Apply the same device-side tree verifier used by the vLLM path."""
        return verify_greedy_tree_batch(
            draft_token_ids,
            parent_indices,
            num_draft_tokens,
            target_token_ids,
            bonus_token_ids,
            max_depth,
            placeholder_token_id,
        )

    def _run_packed_hidden(
        self,
        input_token_ids: list[int],
        sequence_ids: list[int],
        positions: list[int],
        use_aclgraph: bool = True,
        logit_indices: list[int] | None = None,
        use_fused_infer_attention: bool = False,
    ) -> torch.Tensor:
        if self.cache_allocation is None:
            raise RuntimeError("Allocate the native PEARL KV cache before running the model.")
        input_ids = torch.tensor(input_token_ids, dtype=torch.long, device=self.device)
        return self._run_device_packed_hidden(
            input_ids,
            sequence_ids,
            positions,
            use_aclgraph,
            logit_indices,
            use_fused_infer_attention,
        )

    def _run_device_packed_hidden(
        self,
        input_ids: torch.Tensor,
        sequence_ids: list[int],
        positions: list[int],
        use_aclgraph: bool = True,
        logit_indices: list[int] | None = None,
        use_fused_infer_attention: bool = False,
    ) -> torch.Tensor:
        if self.cache_allocation is None or self.cache_block_tables is None:
            raise RuntimeError("Allocate the native PEARL KV cache before running the model.")
        position_tensor, attention_metadata = self._prepare_attention_metadata(
            sequence_ids,
            positions,
            use_fused_infer_attention,
        )
        if use_aclgraph:
            hidden_states = self.graph_runner(input_ids, position_tensor, attention_metadata)
        else:
            hidden_states = self.model(input_ids, position_tensor, attention_metadata)
        if logit_indices is not None:
            hidden_states = hidden_states[logit_indices]
        return hidden_states

    def _prepare_attention_metadata(
        self,
        sequence_ids: list[int],
        positions: list[int],
        use_fused_infer_attention: bool,
    ) -> tuple[torch.Tensor, Any]:
        if self.cache_allocation is None or self.cache_block_tables is None:
            raise RuntimeError("Allocate the native PEARL KV cache before running the model.")
        self._ensure_cache_capacity(sequence_ids, positions)
        slot_mapping = self._cache_slot_mapping(sequence_ids, positions)
        return self.model.make_attention_metadata(
            sequence_ids,
            positions,
            self.cache_block_tables,
            slot_mapping,
            use_fused_infer_attention,
        )

    def _run_packed_model(
        self,
        input_token_ids: list[int],
        sequence_ids: list[int],
        positions: list[int],
        use_aclgraph: bool = True,
        logit_indices: list[int] | None = None,
        use_fused_infer_attention: bool = False,
    ) -> torch.Tensor:
        hidden_states = self._run_packed_hidden(
            input_token_ids,
            sequence_ids,
            positions,
            use_aclgraph,
            logit_indices,
            use_fused_infer_attention,
        )
        return self.model.compute_logits(hidden_states)

    def _run_packed_greedy(
        self,
        input_token_ids: list[int],
        sequence_ids: list[int],
        positions: list[int],
        use_aclgraph: bool = True,
        logit_indices: list[int] | None = None,
        use_fused_infer_attention: bool = False,
    ) -> torch.Tensor:
        if logit_indices is not None or not use_aclgraph:
            hidden_states = self._run_packed_hidden(
                input_token_ids,
                sequence_ids,
                positions,
                use_aclgraph,
                logit_indices,
                use_fused_infer_attention,
            )
            return self.model.compute_greedy_tokens(hidden_states, self.draft_vocab_size)
        input_ids = torch.tensor(input_token_ids, dtype=torch.long, device=self.device)
        return self._run_device_packed_greedy(
            input_ids,
            sequence_ids,
            positions,
            use_aclgraph=True,
            use_fused_infer_attention=use_fused_infer_attention,
        )

    def _run_packed_sample(
        self,
        input_token_ids: list[int],
        sequence_ids: list[int],
        positions: list[int],
        temperatures: list[float],
        top_ps: Sequence[float] | None = None,
        top_ks: Sequence[int] | None = None,
        use_aclgraph: bool = True,
        logit_indices: list[int] | None = None,
        use_fused_infer_attention: bool = False,
    ) -> torch.Tensor:
        if all(temperature == 0 for temperature in temperatures):
            return self._run_packed_greedy(
                input_token_ids,
                sequence_ids,
                positions,
                use_aclgraph,
                logit_indices,
                use_fused_infer_attention,
            )
        logits = self._run_packed_model(
            input_token_ids,
            sequence_ids,
            positions,
            use_aclgraph,
            logit_indices,
            use_fused_infer_attention,
        )[:, : self.draft_vocab_size]
        return _sample_logits(logits, temperatures, top_ps=top_ps, top_ks=top_ks)

    def _run_device_packed_greedy(
        self,
        input_ids: torch.Tensor,
        sequence_ids: list[int],
        positions: list[int],
        use_aclgraph: bool = True,
        use_fused_infer_attention: bool = False,
    ) -> torch.Tensor:
        if self.cache_allocation is None or self.cache_block_tables is None:
            raise RuntimeError("Allocate the native PEARL KV cache before running the model.")
        position_tensor, attention_metadata = self._prepare_attention_metadata(
            sequence_ids,
            positions,
            use_fused_infer_attention,
        )
        if use_aclgraph:
            return self.graph_runner.run_greedy(
                input_ids,
                position_tensor,
                attention_metadata,
                self.draft_vocab_size,
            )
        hidden_states = self.model(input_ids, position_tensor, attention_metadata)
        return self.model.compute_greedy_tokens(hidden_states, self.draft_vocab_size)

    def _run_device_packed_greedy_with_confidence(
        self,
        input_ids: torch.Tensor,
        sequence_ids: list[int],
        positions: list[int],
        use_aclgraph: bool = True,
        use_fused_infer_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run greedy drafting and retain the confidence used by SpecRhythm."""

        hidden_states = self._run_device_packed_hidden(
            input_ids,
            sequence_ids,
            positions,
            use_aclgraph=use_aclgraph,
            use_fused_infer_attention=use_fused_infer_attention,
        )
        return self.model.compute_greedy_tokens_with_confidence(hidden_states, self.draft_vocab_size)

    def _run_device_packed_sample_with_confidence(
        self,
        input_ids: torch.Tensor,
        sequence_ids: list[int],
        positions: list[int],
        temperatures: Sequence[float],
        top_ps: Sequence[float] | None = None,
        top_ks: Sequence[int] | None = None,
        *,
        use_aclgraph: bool = True,
        use_fused_infer_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample draft proposals and retain confidence for SpecRhythm.

        Draft temperature is an optional proposal policy.  The target still
        owns verification and correction, so a greedy target remains exact
        even when this proposal is stochastic.
        """

        if len(temperatures) != input_ids.shape[0] or any(value <= 0 for value in temperatures):
            raise ValueError("stochastic draft temperatures must be positive and row-aligned")
        hidden_states = self._run_device_packed_hidden(
            input_ids,
            sequence_ids,
            positions,
            use_aclgraph=use_aclgraph,
            use_fused_infer_attention=use_fused_infer_attention,
        )
        logits = self.model.compute_logits(hidden_states)[:, : self.draft_vocab_size]
        probabilities = _sampling_probabilities(logits, temperatures, top_ps=top_ps, top_ks=top_ks)
        tokens = _sample_logits(logits, temperatures, top_ps=top_ps, top_ks=top_ks)
        if dist.is_initialized():
            dist.broadcast(
                tokens,
                src=self.topology.draft_leader_rank,
                group=self.groups.draft_group,
            )
        return tokens, probabilities.gather(1, tokens.unsqueeze(1)).squeeze(1)

    def _target_ar_prefill(
        self,
        prompts: list[list[int]],
        states: list[PearlPipelineState],
        sequence_ids: list[int] | None = None,
    ) -> list[int]:
        if sequence_ids is None:
            sequence_ids = list(range(len(prompts)))
        if len(sequence_ids) != len(prompts):
            raise ValueError("Every target AR prefill prompt requires one cache sequence ID.")
        input_token_ids: list[int] = []
        packed_sequence_ids: list[int] = []
        positions: list[int] = []
        last_token_indices: list[int] = []
        for sequence_id, prompt in zip(sequence_ids, prompts):
            assert self.cache_allocation is not None
            cached_tokens = self.cache_allocation.num_cached_tokens[sequence_id]
            input_token_ids.extend(prompt[cached_tokens:])
            packed_sequence_ids.extend([sequence_id] * (len(prompt) - cached_tokens))
            positions.extend(range(cached_tokens, len(prompt)))
            last_token_indices.append(len(input_token_ids) - 1)
        prefill_kwargs = {
            "use_aclgraph": False,
            "logit_indices": last_token_indices,
        }
        # Packed prompt tokens use the TND FIA prefill contract on NPU. Keep
        # the keyword absent for lightweight CPU mocks used by unit tests.
        if getattr(self, "device", torch.device("cpu")).type == "npu":
            prefill_kwargs["use_fused_infer_attention"] = True
        top_ps = [state.top_p for state in states]
        top_ks = [state.top_k for state in states]
        if any(value < 1.0 for value in top_ps) or any(value > 0 for value in top_ks):
            prefill_kwargs.update(top_ps=top_ps, top_ks=top_ks)
        token_ids = self._run_packed_sample(
            input_token_ids,
            packed_sequence_ids,
            positions,
            [state.temperature for state in states],
            **prefill_kwargs,
        )
        if any(state.temperature > 0 for state in states):
            dist.broadcast(
                token_ids,
                src=self.topology.target_leader_rank,
                group=self.groups.target_group,
            )
        return [int(token_id) for token_id in token_ids.cpu().tolist()]

    def _capture_target_ar_graph(self, states: list[PearlPipelineState]) -> None:
        if not self.config.enforce_eager:
            sequence_ids = list(range(len(states)))
            input_ids = torch.tensor(
                [state.token_ids[-1] for state in states],
                dtype=torch.long,
                device=self.device,
            )
            self._run_device_packed_greedy(
                input_ids,
                sequence_ids,
                [len(state.token_ids) - 1 for state in states],
            )
        dist.barrier(group=self.groups.target_group)

    def _spec_rhythm_nonfinite_flag(self, *, include_layer_cache: bool = True) -> torch.Tensor:
        """Aggregate graph-resident health flags without a host synchronization.

        Layer-cache flags are needed by prefill admission. Tree decode does
        not update them: its full-model hidden/logit boundary owns the dynamic
        guard, while KV compaction only copies storage that already passed a
        prior vote. The commit path can therefore avoid stacking one stale
        scalar per decoder layer.
        """
        model = getattr(self, "model", None)
        flags: list[torch.Tensor] = []
        if model is not None:
            owners_and_names = [(model, "output_nonfinite"), (getattr(model, "lm_head", None), "logits_nonfinite")]
            if include_layer_cache:
                owners_and_names.extend(
                    (layer.self_attn, "tree_cache_nonfinite") for layer in getattr(model, "layers", ())
                )
            for owner, name in owners_and_names:
                flag = getattr(owner, name, None)
                if isinstance(flag, torch.Tensor):
                    flags.append(flag)
        if flags:
            return torch.stack(flags).any()
        # CPU protocol harnesses may replace the model boundary entirely.
        return torch.zeros((), dtype=torch.bool, device=self.device)

    def _vote_spec_rhythm_prefill_finiteness(self) -> None:
        """Reject numerical faults on every rank before any first-token commit.

        This is a device failure-bit vote, not rollback after a hardware/TP
        collective failure. Both roles arrive here after their local prefill.
        """
        config = getattr(self, "config", None)
        if not (
            getattr(config, "enable_spec_rhythm", False)
            and (getattr(config, "spec_rhythm_tree_width", 1) > 1 or getattr(config, "spec_rhythm_tree_depth", 1) > 1)
        ):
            return
        failed = self._spec_rhythm_nonfinite_flag().reshape(1).to(dtype=torch.int64)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if int(failed.cpu().item()):
            raise RuntimeError(
                "SpecRhythm prefill produced nonfinite Q/K/V or output on a rank; "
                "no first token was committed; discard this cache"
            )

    def _prefill_spec_rhythm_token_chunk_batch(
        self,
        prompts: Mapping[int, Sequence[int]],
        states: Mapping[int, PearlPipelineState],
        chunks: Sequence[SpecRhythmPrefillTokenChunk],
        *,
        coordinate_before_broadcast: bool = False,
    ) -> dict[int, int]:
        """Populate one private prompt-KV chunk and sample only final rows.

        The caller owns the cursors and does not publish a request until every
        prompt in its staged reservation is complete.  Prefix-cache hits are
        intentionally rejected by configuration and rechecked here so a
        request-local cursor always denotes an absolute prompt position on
        both model roles.
        """

        if not chunks:
            raise ValueError("SpecRhythm token-chunk prefill requires at least one span.")
        if self.config.enable_prefix_caching:
            raise RuntimeError("SpecRhythm token-chunk prefill cannot run with prefix caching.")
        if self.cache_allocation is None:
            raise RuntimeError("Allocate the native PEARL KV cache before token-chunk prefill.")

        input_token_ids: list[int] = []
        packed_sequence_ids: list[int] = []
        positions: list[int] = []
        final_hidden_indices: list[int] = []
        final_request_indices: list[int] = []
        seen: set[int] = set()
        for chunk in chunks:
            request_index = int(chunk.request_index)
            if request_index in seen:
                raise ValueError("One token-chunk prefill submission may contain only one span per request.")
            seen.add(request_index)
            if request_index not in prompts or request_index not in states:
                raise ValueError("SpecRhythm token-chunk prefill lost request state.")
            prompt = prompts[request_index]
            if len(prompt) != chunk.prompt_length:
                raise ValueError("SpecRhythm token-chunk prompt length changed after reservation.")
            if not 0 <= chunk.start < chunk.end <= len(prompt):
                raise ValueError("SpecRhythm token-chunk prefill received an invalid span.")
            if self.cache_allocation.num_cached_tokens[request_index] != 0:
                raise RuntimeError("SpecRhythm token-chunk prefill requires an uncached prompt.")
            input_token_ids.extend(int(token_id) for token_id in prompt[chunk.start : chunk.end])
            packed_sequence_ids.extend([request_index] * chunk.token_count)
            positions.extend(range(chunk.start, chunk.end))
            if chunk.completes_prompt:
                final_hidden_indices.append(len(input_token_ids) - 1)
                final_request_indices.append(request_index)

        if len(input_token_ids) > self.config.spec_rhythm_prefill_token_chunk_size:
            raise RuntimeError("SpecRhythm token-chunk prefill exceeded its configured cap.")
        prefill_kwargs = {"use_aclgraph": False}
        if getattr(self, "device", torch.device("cpu")).type == "npu":
            prefill_kwargs["use_fused_infer_attention"] = True
        if self.is_draft or not final_request_indices:
            self._run_packed_hidden(
                input_token_ids,
                packed_sequence_ids,
                positions,
                **prefill_kwargs,
            )
            token_ids = torch.zeros(
                len(final_request_indices),
                dtype=torch.long,
                device=self.device,
            )
        else:
            final_states = [states[index] for index in final_request_indices]
            top_ps = [state.top_p for state in final_states]
            top_ks = [state.top_k for state in final_states]
            sample_kwargs = {
                **prefill_kwargs,
                "logit_indices": final_hidden_indices,
            }
            if any(value < 1.0 for value in top_ps) or any(value > 0 for value in top_ks):
                sample_kwargs.update(top_ps=top_ps, top_ks=top_ks)
            token_ids = self._run_packed_sample(
                input_token_ids,
                packed_sequence_ids,
                positions,
                [state.temperature for state in final_states],
                **sample_kwargs,
            )
        self._vote_spec_rhythm_prefill_finiteness()
        if coordinate_before_broadcast:
            if len(self.topology.draft_ranks) != 1:
                raise RuntimeError("Overlapped token-chunk prefill currently requires one draft rank.")
            coordination_group = getattr(
                self.groups,
                "verification_coordination_group",
                None,
            )
            if self.device.type == "npu":
                if coordination_group is None:
                    raise RuntimeError(
                        "Overlapped token-chunk prefill requires the CPU verification coordination group."
                    )
                torch.npu.current_stream().synchronize()
            if coordination_group is not None:
                dist.barrier(group=coordination_group)
        if final_request_indices:
            dist.broadcast(
                token_ids,
                src=self.topology.target_leader_rank,
            )
        return dict(
            zip(
                final_request_indices,
                (int(token_id) for token_id in token_ids.cpu().tolist()),
            )
        )

    def _draft_prefill_hidden_batch(
        self,
        prompts: Sequence[Sequence[int]],
        sequence_ids: Sequence[int],
    ) -> torch.Tensor:
        """Populate disjoint draft prompt KV rows without sampling a token.

        The returned last-token hidden rows are intentionally retained by the
        caller until a side-stream join completes.  First-token publication
        remains target-owned, exactly as in the ordinary draft prefill path.
        """

        if not self.is_draft:
            raise RuntimeError("Only the draft rank may run draft prefill.")
        if len(prompts) != len(sequence_ids) or not prompts:
            raise ValueError("Draft prefill requires aligned non-empty prompts and rows.")
        input_token_ids: list[int] = []
        packed_sequence_ids: list[int] = []
        positions: list[int] = []
        last_token_indices: list[int] = []
        for sequence_id, prompt in zip(sequence_ids, prompts):
            if not prompt:
                raise ValueError("Draft prefill prompts cannot be empty.")
            assert self.cache_allocation is not None
            cached_tokens = int(self.cache_allocation.num_cached_tokens[sequence_id])
            if not 0 <= cached_tokens < len(prompt):
                raise RuntimeError("Draft prefill needs at least one uncached prompt token.")
            uncached = prompt[cached_tokens:]
            input_token_ids.extend(int(token_id) for token_id in uncached)
            packed_sequence_ids.extend([int(sequence_id)] * len(uncached))
            positions.extend(range(cached_tokens, len(prompt)))
            last_token_indices.append(len(input_token_ids) - 1)
        return self._run_packed_hidden(
            input_token_ids,
            packed_sequence_ids,
            positions,
            use_aclgraph=False,
            logit_indices=last_token_indices,
            use_fused_infer_attention=(self.device.type == "npu"),
        )

    def _prefill_and_sample_target_batch(
        self,
        prompts: list[list[int]],
        states: list[PearlPipelineState],
        sequence_ids: list[int] | None = None,
        *,
        coordinate_before_broadcast: bool = False,
        precomputed_target_tokens: torch.Tensor | None = None,
        draft_prefill_completed: bool = False,
    ) -> list[int]:
        if sequence_ids is None:
            sequence_ids = list(range(len(prompts)))
        if len(sequence_ids) != len(prompts):
            raise ValueError("Every PEARL prefill prompt requires one cache sequence ID.")
        input_token_ids: list[int] = []
        packed_sequence_ids: list[int] = []
        positions: list[int] = []
        last_token_indices: list[int] = []
        for sequence_id, prompt in zip(sequence_ids, prompts):
            assert self.cache_allocation is not None
            cached_tokens = self.cache_allocation.num_cached_tokens[sequence_id]
            input_token_ids.extend(prompt[cached_tokens:])
            packed_sequence_ids.extend([sequence_id] * (len(prompt) - cached_tokens))
            positions.extend(range(cached_tokens, len(prompt)))
            last_token_indices.append(len(input_token_ids) - 1)
        if self.is_draft:
            if precomputed_target_tokens is not None:
                raise RuntimeError("Only target ranks may supply mixed-prefill first tokens.")
            if not draft_prefill_completed:
                # The target sample is the common committed frontier, but the
                # draft prefill still runs so its persistent KV cache is
                # populated. A mixed first step has already populated it.
                prefill_kwargs = {
                    "use_aclgraph": False,
                    "logit_indices": last_token_indices,
                }
                if getattr(self, "device", torch.device("cpu")).type == "npu":
                    prefill_kwargs["use_fused_infer_attention"] = True
                self._run_packed_hidden(
                    input_token_ids,
                    packed_sequence_ids,
                    positions,
                    **prefill_kwargs,
                )
            token_ids = torch.zeros(len(prompts), dtype=torch.long, device=self.device)
        elif precomputed_target_tokens is not None:
            if draft_prefill_completed:
                raise RuntimeError("Target ranks cannot mark draft prefill as completed.")
            if (
                precomputed_target_tokens.dtype != torch.long
                or precomputed_target_tokens.device.type != self.device.type
                or precomputed_target_tokens.shape != (len(prompts),)
            ):
                raise ValueError("Mixed-prefill target tokens have an invalid device, dtype, or shape.")
            token_ids = precomputed_target_tokens
        else:
            if draft_prefill_completed:
                raise RuntimeError("Target ranks cannot mark draft prefill as completed.")
            prefill_kwargs = {
                "use_aclgraph": False,
                "logit_indices": last_token_indices,
            }
            if getattr(self, "device", torch.device("cpu")).type == "npu":
                prefill_kwargs["use_fused_infer_attention"] = True
            top_ps = [state.top_p for state in states]
            top_ks = [state.top_k for state in states]
            if any(value < 1.0 for value in top_ps) or any(value > 0 for value in top_ks):
                prefill_kwargs.update(top_ps=top_ps, top_ks=top_ks)
            token_ids = self._run_packed_sample(
                input_token_ids,
                packed_sequence_ids,
                positions,
                [state.temperature for state in states],
                **prefill_kwargs,
            )
        self._vote_spec_rhythm_prefill_finiteness()
        if coordinate_before_broadcast:
            if len(self.topology.draft_ranks) != 1:
                raise RuntimeError("Overlapped online prefill currently requires one draft rank.")
            coordination_group = getattr(
                self.groups,
                "verification_coordination_group",
                None,
            )
            if self.device.type == "npu":
                if coordination_group is None:
                    raise RuntimeError("Overlapped online prefill requires the CPU verification coordination group.")
                # Both model roles have queued their current-cycle work and
                # this role-local prefill on their own streams.  Complete
                # those streams, then rendezvous on CPU before any rank posts
                # the all-world HCCL token broadcast.  This avoids a cycle
                # between target TP all-reduces and the cross-model group.
                torch.npu.current_stream().synchronize()
            if coordination_group is not None:
                dist.barrier(group=coordination_group)
        dist.broadcast(token_ids, src=self.topology.target_leader_rank)
        return [int(token_id) for token_id in token_ids.cpu().tolist()]

    def _auto_select_gamma(self, prompts: list[list[int]]) -> int:
        if not self.gamma_profiles:
            raise RuntimeError("PEARL auto-gamma profiles were not initialized.")
        batch_size = len(prompts)
        bucket = next(
            (size for size in self.gamma_profiles if size >= batch_size),
            max(self.gamma_profiles),
        )
        return self.gamma_profiles[bucket]

    def _profile_auto_gammas(self) -> dict[int, int]:
        profile_length = min(
            self.config.auto_gamma_profile_sequence_length,
            self.config.max_model_len - 1,
        )
        if profile_length <= 0:
            raise ValueError("PEARL max_model_len is too small for automatic gamma profiling.")
        batch_sizes = [
            batch_size
            for batch_size in AUTO_GAMMA_BATCH_SIZES
            if batch_size <= self.config.max_num_seqs
            and batch_size * profile_length <= self.config.max_num_batched_tokens
        ]
        if not batch_sizes:
            raise ValueError("PEARL batch limits cannot fit the automatic gamma profile.")

        profiles: dict[int, int] = {}
        for batch_size in batch_sizes:
            prompts = [[0] * profile_length for _ in range(batch_size)]
            self._allocate_cache(prompts, enable_prefix_caching=False)
            input_token_ids = [token for prompt in prompts for token in prompt]
            sequence_ids = [index for index in range(batch_size) for _ in range(profile_length)]
            positions = list(range(profile_length)) * batch_size
            last_token_indices = [(index + 1) * profile_length - 1 for index in range(batch_size)]
            decode_tokens = self._run_packed_greedy(
                input_token_ids,
                sequence_ids,
                positions,
                use_aclgraph=False,
                logit_indices=last_token_indices,
            )
            decode_positions = [profile_length] * batch_size
            decode_sequence_ids = list(range(batch_size))
            for _ in range(AUTO_GAMMA_WARMUP_STEPS):
                self._run_device_packed_greedy(
                    decode_tokens,
                    decode_sequence_ids,
                    decode_positions,
                )
            torch.npu.synchronize()
            dist.barrier()
            started = time.perf_counter()
            for _ in range(AUTO_GAMMA_PROFILE_STEPS):
                self._run_device_packed_greedy(
                    decode_tokens,
                    decode_sequence_ids,
                    decode_positions,
                )
            torch.npu.synchronize()
            elapsed = time.perf_counter() - started
            local_speed = AUTO_GAMMA_PROFILE_STEPS / elapsed
            speeds = torch.zeros(2, dtype=torch.float32, device=self.device)
            if self.rank == self.topology.draft_leader_rank:
                speeds[0] = local_speed
            if self.rank == self.topology.target_leader_rank:
                speeds[1] = local_speed
            dist.all_reduce(speeds)
            draft_speed, target_speed = (float(value) for value in speeds.cpu().tolist())
            profiles[batch_size] = _gamma_from_decode_speeds(draft_speed, target_speed)
            self._release_cache()
        return profiles

    def _capture_decode_graphs(self, prompts: list[list[int]], target_tokens: list[int]) -> None:
        if self.config.enforce_eager or self.gamma > 16:
            return
        if len(prompts) in self.precompiled_decode_batch_sizes:
            dist.barrier()
            return
        sequence_ids = list(range(len(prompts)))
        first_decode_positions = [len(prompt) for prompt in prompts]
        input_ids = torch.tensor(target_tokens, dtype=torch.long, device=self.device)
        target_use_fia = not self.config.target_use_paged_attention
        if envs.VLLM_ASCEND_SPECRHYTHM_USE_FIA:
            target_use_fia = True
        target_aclgraph_enabled = not self.is_draft and not envs.VLLM_ASCEND_SPECRHYTHM_DISABLE_TARGET_ACLGRAPH
        if self.is_draft and self.gamma > 1:
            # SpecRhythm normally drafts a fixed-width window.  Capture the
            # whole greedy window before timing starts so the measured loop can
            # replay one ACLGraph instead of launching one model call per token.
            draft_use_fia = not self.config.draft_use_paged_attention
            stable_linear_spec_rhythm = bool(
                self.config.enable_spec_rhythm
                and self.config.spec_rhythm_tree_width == 1
                and self.config.spec_rhythm_tree_depth == 1
                and self.config.spec_rhythm_stable_graphs
            )
            if stable_linear_spec_rhythm:
                # Match the production linear SpecRhythm path exactly.  It
                # keeps paged attention for stable task updates and selects a
                # home/full service-capacity graph bucket for normal/eager work.
                draft_use_fia = False
            home_capture_size = max(1, self.config.max_num_seqs // 2)
            real_capture_sizes = [min(len(sequence_ids), home_capture_size)]
            if stable_linear_spec_rhythm and len(sequence_ids) > home_capture_size:
                real_capture_sizes.append(min(len(sequence_ids), self.config.max_num_seqs))
            for real_capture_size in real_capture_sizes:
                draft_sequence_ids = sequence_ids[:real_capture_size]
                draft_input_ids = input_ids[:real_capture_size]
                draft_first_positions = first_decode_positions[:real_capture_size]
                graph_capture_size = real_capture_size
                if stable_linear_spec_rhythm:
                    graph_capture_size = (
                        home_capture_size if real_capture_size <= home_capture_size else self.config.max_num_seqs
                    )
                position_tensors = []
                attention_metadatas = []
                graph_input_ids = None
                for step in range(self.gamma):
                    step_positions = [position + step for position in draft_first_positions]
                    position_tensor, attention_metadata = self._prepare_attention_metadata(
                        draft_sequence_ids,
                        step_positions,
                        use_fused_infer_attention=draft_use_fia,
                    )
                    if stable_linear_spec_rhythm:
                        graph_input_ids, position_tensor, attention_metadata = NativeACLGraphRunner._pad_inputs(
                            draft_input_ids,
                            position_tensor,
                            attention_metadata,
                            graph_capture_size,
                        )
                    else:
                        graph_input_ids = draft_input_ids
                    position_tensors.append(position_tensor)
                    attention_metadatas.append(attention_metadata)
                assert graph_input_ids is not None
                self.graph_runner.run_draft_greedy(
                    graph_input_ids,
                    position_tensors,
                    attention_metadatas,
                    self.draft_vocab_size,
                    valid_row_count=real_capture_size,
                )
        if target_aclgraph_enabled:
            self._run_device_packed_greedy(
                input_ids,
                sequence_ids,
                first_decode_positions,
                use_fused_infer_attention=target_use_fia,
            )
        if target_aclgraph_enabled and self.gamma > 1:
            packed_tokens = [token_id for token_id in target_tokens for _ in range(self.gamma)]
            packed_sequence_ids = [sequence_id for sequence_id in sequence_ids for _ in range(self.gamma)]
            packed_positions = [
                position for prompt in prompts for position in range(len(prompt) + 1, len(prompt) + self.gamma + 1)
            ]
            packed_input_ids = torch.tensor(packed_tokens, dtype=torch.long, device=self.device)
            self._run_device_packed_greedy(
                packed_input_ids,
                packed_sequence_ids,
                packed_positions,
                use_fused_infer_attention=target_use_fia,
            )
        dist.barrier()

    def _qualify_linear_draft_graph_bucket(
        self,
        states: list[PearlPipelineState],
        batch_size: int,
        *,
        graph_lane: int | None = None,
    ) -> None:
        """Capture and changed-input qualify one full serial-draft bucket.

        Capture validation only proves that a graph can replay the values used
        while it was captured.  A production replay also refreshes every paged
        attention task with new token, position, and slot metadata.  Advance a
        complete bucket by one real draft token before the second replay so that
        this stronger validation happens during precompilation, never in a
        measured request.  Restore the synthetic host state afterwards; its KV
        cache is private to precompilation and is released by the caller.
        """

        batch_states = states[:batch_size]
        active_indices = list(range(batch_size))
        counters = self.graph_runner.execution_counters["draft"]
        validation_calls_before = counters["runtime_validation_calls"]
        changed_input_calls_before = counters["changed_input_validation_calls"]
        validation_failures_before = counters["runtime_validation_failures"]
        task_update_skips_before = self.graph_runner.task_update_skip_replay_count
        stable_length_qualification = bool(envs.VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_BUCKET)
        stable_task_barrier_qualification = bool(
            stable_length_qualification
            and envs.VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_COMMON_KV
            and envs.VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_STABLE_TASK_BARRIER
        )

        _, captured_next, _ = self._draft_spec_rhythm_device_batch(
            batch_states,
            active_indices,
            [self.gamma] * batch_size,
            verification_sizes=[self.gamma] * batch_size,
            full_window=True,
            graph_lane=graph_lane,
        )
        if captured_next is None or tuple(captured_next.shape) != (
            batch_size,
            self.gamma,
        ):
            raise RuntimeError(
                "Serial draft graph precompilation did not return one full proposal window per bucket row."
            )

        qualification_tokens = captured_next[:, 0].detach().cpu().tolist()
        if any(state.committed_length is None for state in batch_states):
            raise RuntimeError("Serial draft graph qualification requires committed state.")
        original_max_tokens = [state.max_tokens for state in batch_states]
        for state, token_id in zip(batch_states, qualification_tokens):
            state.token_ids.append(int(token_id))
            assert state.committed_length is not None
            state.committed_length += 1
            # Legacy FIA qualification also exercises the dynamic task-update
            # path by changing the lifetime K envelope.  A stable-task barrier
            # deliberately has no such path: keep its common-KV signature fixed
            # and validate only changing token/slot/mask/table tensor contents.
            if stable_length_qualification and not stable_task_barrier_qualification:
                state.max_tokens += 1
        appended_tokens = 1
        try:
            _, changed_length_next, _ = self._draft_spec_rhythm_device_batch(
                batch_states,
                active_indices,
                [self.gamma] * batch_size,
                verification_sizes=[self.gamma] * batch_size,
                full_window=True,
                graph_lane=graph_lane,
            )
            if changed_length_next is None:
                raise RuntimeError("Serial draft qualification changed-length replay did not return a proposal window.")
            if stable_length_qualification:
                # Keep the lifetime K envelope unchanged, but advance every
                # real row once more. This changes tokens, positions, slots,
                # masks and visible page-table prefixes while exercising the
                # event-only (zero task rebuild) path. Temporarily mark the
                # entry unvalidated so run_draft_greedy compares this replay
                # to the same FULL-envelope eager execution.
                for state, token_id in zip(
                    batch_states,
                    changed_length_next[:, 0].detach().cpu().tolist(),
                ):
                    state.token_ids.append(int(token_id))
                    assert state.committed_length is not None
                    state.committed_length += 1
                appended_tokens += 1
                entry_key = self.graph_runner.last_draft_entry_key
                entry = self.graph_runner.draft_entries.get(entry_key)
                if entry is None or not entry.runtime_validated:
                    raise RuntimeError(
                        "Serial draft qualification could not locate its changed-length validated graph entry."
                    )
                entry.runtime_validated = False
                self._draft_spec_rhythm_device_batch(
                    batch_states,
                    active_indices,
                    [self.gamma] * batch_size,
                    verification_sizes=[self.gamma] * batch_size,
                    full_window=True,
                    graph_lane=graph_lane,
                )
        finally:
            for state, max_tokens in zip(batch_states, original_max_tokens):
                for _ in range(appended_tokens):
                    state.token_ids.pop()
                assert state.committed_length is not None
                state.committed_length -= appended_tokens
                state.max_tokens = max_tokens

        entry_key = self.graph_runner.last_draft_entry_key
        entry = self.graph_runner.draft_entries.get(entry_key)
        if (
            entry is None
            or not entry.runtime_validated
            or entry.validated_real_row_count != batch_size
            or counters["runtime_validation_calls"]
            != validation_calls_before + (2 if stable_length_qualification else 1)
            or counters["changed_input_validation_calls"]
            != changed_input_calls_before + (2 if stable_length_qualification else 1)
            or counters["runtime_validation_failures"] != validation_failures_before
            or (
                stable_length_qualification
                and self.graph_runner.task_update_skip_replay_count
                != task_update_skips_before + (2 if stable_task_barrier_qualification else 1)
            )
        ):
            raise RuntimeError(
                "Serial draft ACLGraph bucket did not complete changed-input "
                f"full-row qualification: batch_size={batch_size}."
            )

    def _qualify_mixed_draft_tail_graph_bucket(
        self,
        states: list[PearlPipelineState],
        batch_size: int,
    ) -> None:
        """Qualify the gamma-1 graph used after a mixed prefill first step."""

        proposal_tail_steps = self.gamma - 1
        tail_steps = proposal_tail_steps + int(
            getattr(
                self.config,
                "spec_rhythm_linear_bonus_token",
                False,
            )
        )
        if tail_steps <= 0:
            raise ValueError("Mixed draft prefill requires gamma greater than one.")
        batch_states = states[:batch_size]
        active_indices = list(range(batch_size))
        counters = self.graph_runner.execution_counters["draft"]
        validation_calls_before = counters["runtime_validation_calls"]
        changed_input_calls_before = counters["changed_input_validation_calls"]
        validation_failures_before = counters["runtime_validation_failures"]

        def run_tail(input_ids: torch.Tensor) -> torch.Tensor:
            first_positions = [len(state.token_ids) for state in batch_states]
            position_tensors: list[torch.Tensor] = []
            attention_metadatas: list[Any] = []
            graph_input_ids: torch.Tensor | None = None
            for step in range(tail_steps):
                step_positions = [position + step for position in first_positions]
                position_tensor, attention_metadata = self._prepare_attention_metadata(
                    active_indices,
                    step_positions,
                    use_fused_infer_attention=False,
                )
                (
                    graph_input_ids,
                    padded_positions,
                    padded_metadata,
                ) = NativeACLGraphRunner._pad_inputs(
                    input_ids,
                    position_tensor,
                    attention_metadata,
                    batch_size,
                )
                position_tensors.append(padded_positions)
                attention_metadatas.append(padded_metadata)
            assert graph_input_ids is not None
            run_kwargs: dict[str, Any] = {
                "valid_row_count": batch_size,
            }
            if getattr(
                self.config,
                "spec_rhythm_linear_bonus_token",
                False,
            ):
                run_kwargs["final_kv_only"] = True
            return self.graph_runner.run_draft_greedy(
                graph_input_ids,
                position_tensors,
                attention_metadatas,
                self.draft_vocab_size,
                **run_kwargs,
            )

        initial_inputs = torch.tensor(
            [state.token_ids[-1] for state in batch_states],
            dtype=torch.long,
            device=self.device,
        )
        captured = run_tail(initial_inputs)
        if tuple(captured.shape) != (batch_size, proposal_tail_steps):
            raise RuntimeError("Mixed draft-tail graph precompilation returned an invalid shape.")
        qualification_tokens = captured[:, 0].detach().cpu().tolist()
        for state, token_id in zip(batch_states, qualification_tokens):
            state.token_ids.append(int(token_id))
            assert state.committed_length is not None
            state.committed_length += 1
        try:
            run_tail(
                torch.tensor(
                    qualification_tokens,
                    dtype=torch.long,
                    device=self.device,
                )
            )
        finally:
            for state in batch_states:
                state.token_ids.pop()
                assert state.committed_length is not None
                state.committed_length -= 1

        entry_key = (
            f"draft-greedy:{self.draft_vocab_size}|steps:{tail_steps}|paged"
            + (
                "|final-kv"
                if getattr(
                    self.config,
                    "spec_rhythm_linear_bonus_token",
                    False,
                )
                else ""
            ),
            batch_size,
        )
        entry = self.graph_runner.draft_entries.get(entry_key)
        if (
            entry is None
            or not entry.runtime_validated
            or entry.validated_real_row_count != batch_size
            or counters["runtime_validation_calls"] != validation_calls_before + 1
            or counters["changed_input_validation_calls"] != changed_input_calls_before + 1
            or counters["runtime_validation_failures"] != validation_failures_before
        ):
            raise RuntimeError(
                "Mixed draft-tail ACLGraph bucket did not complete changed-input "
                f"qualification: batch_size={batch_size}."
            )

    @torch.inference_mode()
    def _precompile_decode_graphs(
        self,
        *,
        include_target_graphs: bool = True,
        include_draft_graphs: bool = True,
    ) -> None:
        """Build stable paged-attention graphs before serving requests."""
        batch_sizes = [self.config.max_num_seqs]
        if self.config.enable_continuous_batching and self.config.max_num_seqs > 1:
            batch_sizes.append(max(1, self.config.max_num_seqs // 2))
        batch_sizes = sorted(set(batch_sizes))
        prompts = [[0] for _ in range(max(batch_sizes))]
        allocation_kwargs: dict[str, bool] = {}
        if getattr(self, "_mixed_target_graph_enabled", False):
            allocation_kwargs["reserve_sequence_capacity"] = True
        self._allocate_cache(
            prompts,
            enable_prefix_caching=False,
            **allocation_kwargs,
        )
        try:
            states = [
                PearlPipelineState(
                    [0],
                    prompt_length=1,
                    temperature=0.0,
                    # PEARLConfig uses max_model_len as its global native
                    # fallback max_tokens; feeding that synthetic value into
                    # a fixed-lifetime FIA envelope would make a one-token
                    # qualification row span the entire cache.  Sixty-four
                    # tokens are sufficient to qualify both changed-length
                    # and stable-length graph-task paths; real request limits
                    # replace this host literal before serving.
                    max_tokens=min(self.config.max_tokens, 64),
                    ignore_eos=True,
                )
                for _ in prompts
            ]
            # Capturing against an uninitialized KV cache is not replay-stable
            # on CANN, so initialize a real one-token prefix on every worker.
            self._prefill_and_sample_target_batch(prompts, states)
            for state in states:
                state.token_ids.append(0)
                assert state.committed_length is not None
                state.committed_length += 1

            if self.is_draft and include_draft_graphs:
                stable_linear_spec_rhythm = bool(
                    self.config.enable_spec_rhythm
                    and self.config.spec_rhythm_tree_width == 1
                    and self.config.spec_rhythm_tree_depth == 1
                    and self.config.spec_rhythm_stable_graphs
                )
                draft_batch_sizes = (
                    _linear_draft_graph_buckets(self.config.max_num_seqs) if stable_linear_spec_rhythm else batch_sizes
                )
                if (
                    stable_linear_spec_rhythm
                    and envs.VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_BUCKET
                    and 2 * len(draft_batch_sizes) > self.config.max_aclgraph_entries
                ):
                    raise RuntimeError("Lane-stable linear draft FIA needs two ACLGraphs per serial draft bucket.")
                if (
                    stable_linear_spec_rhythm
                    and envs.VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_BUCKET
                    and envs.VLLM_ASCEND_SPECRHYTHM_MIXED_DRAFT_PREFILL
                ):
                    raise RuntimeError(
                        "Bucketed linear-draft FIA cannot share the graph cache with mixed draft prefill."
                    )
                if (
                    stable_linear_spec_rhythm
                    and envs.VLLM_ASCEND_SPECRHYTHM_MIXED_DRAFT_PREFILL
                    and 2 * len(draft_batch_sizes) > self.config.max_aclgraph_entries
                ):
                    raise RuntimeError(
                        "Mixed draft prefill needs one full-chain and one gamma-1 ACLGraph per serial draft bucket."
                    )
                for batch_size in draft_batch_sizes:
                    self.graph_runner.set_expected_fia_batch_size(batch_size)
                    batch_states = states[:batch_size]
                    active_indices = list(range(batch_size))
                    if stable_linear_spec_rhythm:
                        graph_lanes = (0, 1) if envs.VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_BUCKET else (None,)
                        for graph_lane in graph_lanes:
                            self._qualify_linear_draft_graph_bucket(
                                states,
                                batch_size,
                                graph_lane=graph_lane,
                            )
                        if envs.VLLM_ASCEND_SPECRHYTHM_MIXED_DRAFT_PREFILL:
                            self._qualify_mixed_draft_tail_graph_bucket(
                                states,
                                batch_size,
                            )
                    else:
                        # Capture plus one init-time replay keeps graph setup out
                        # of the first measured request.
                        self._draft_round_device_batch(batch_states, active_indices)
                        self._draft_round_device_batch(batch_states, active_indices)
            elif not self.is_draft and (include_target_graphs or getattr(self, "_mixed_target_graph_enabled", False)):
                if getattr(self, "_mixed_target_graph_enabled", False):
                    self._qualify_mixed_target_graph_buckets()
                if include_target_graphs:
                    # Normal paged replay may pad into an existing larger
                    # graph. Ascending compilation guarantees an exact entry
                    # per bucket. This remains independent of the packed-FIA
                    # mixed-target qualification above.
                    for _, batch_size, post_verify_count in _target_graph_precompile_shapes(
                        batch_sizes,
                        self.gamma,
                        self.config.target_verification_graph_buckets,
                        self.config.target_verification_graph_post_counts,
                    ):
                        pre_verify = [False] * post_verify_count + [True] * (batch_size - post_verify_count)
                        model_widths, _ = _bucket_target_verification_widths(
                            pre_verify,
                            self.gamma,
                            self.config.target_verification_graph_buckets,
                            self.config.target_verification_graph_post_counts,
                        )
                        input_token_ids: list[int] = []
                        sequence_ids: list[int] = []
                        positions: list[int] = []
                        for sequence_id, model_width in enumerate(model_widths):
                            input_token_ids.extend([0] * model_width)
                            sequence_ids.extend([sequence_id] * model_width)
                            positions.extend(range(1, model_width + 1))
                        input_ids = torch.tensor(
                            input_token_ids,
                            dtype=torch.long,
                            device=self.device,
                        )
                        self.graph_runner.set_expected_fia_batch_size(batch_size)
                        self._run_device_packed_greedy(
                            input_ids,
                            sequence_ids,
                            positions,
                            use_fused_infer_attention=False,
                        )
                        self._run_device_packed_greedy(
                            input_ids,
                            sequence_ids,
                            positions,
                            use_fused_infer_attention=False,
                        )
            torch.npu.synchronize()
            if include_target_graphs and (not self.is_draft or include_draft_graphs):
                # This marker is consumed by a later cross-rank static decode
                # capture. It must only be set when both model roles were
                # precompiled; draft-only qualification intentionally leaves
                # target graphs lazy.
                self.precompiled_decode_batch_sizes = frozenset(batch_sizes)
        finally:
            self._release_cache()

    def _qualify_mixed_target_graph_buckets(self) -> None:
        """Qualify every prompt-token and verification-capacity graph pair."""

        if self.is_draft or not self._mixed_target_graph_enabled:
            return
        if self.cache_allocation is None or self.cache_block_tables is None:
            raise RuntimeError("Mixed-target graph qualification requires an allocated cache.")
        minimum_service_rows = MIXED_TARGET_PROMPT_CAPACITY + 1
        if self._cache_sequence_capacity < minimum_service_rows:
            raise RuntimeError("Mixed-target graph qualification needs at least five service rows.")
        scratch_ids = self._mixed_target_scratch_sequence_ids
        query_width = self.gamma + int(getattr(self.config, "spec_rhythm_linear_bonus_token", False))
        if query_width <= 0:
            raise RuntimeError("Mixed-target graph qualification requires a fixed positive gamma.")
        self.graph_runner.set_expected_fia_batch_size(None)
        # Capture the largest (and therefore most memory-demanding) shape
        # first.  On 910B a late q2181 capture can OOM after several small
        # graph pools have become resident even though q2181 alone fits.
        selected_prompt_buckets = getattr(
            self,
            "_mixed_target_graph_prompt_buckets",
            MIXED_TARGET_PROMPT_TOKEN_BUCKETS,
        )
        for prompt_bucket in reversed(selected_prompt_buckets):
            for verification_capacity in reversed(MIXED_TARGET_VERIFY_CAPACITIES):
                verification_rows = min(
                    verification_capacity,
                    self._cache_sequence_capacity - MIXED_TARGET_PROMPT_CAPACITY,
                )
                verification_ids = tuple(range(verification_rows))
                verification_starts = tuple(1 + index % 4 for index in range(verification_rows))
                changed_verification_rows = min(17, verification_rows)
                if changed_verification_rows == verification_rows and verification_rows > 1:
                    changed_verification_rows -= 1
                prompt_ids = tuple(
                    range(
                        verification_rows,
                        verification_rows + MIXED_TARGET_PROMPT_CAPACITY,
                    )
                )
                first_layout = plan_mixed_target_graph_layout(
                    verification_rows,
                    [prompt_bucket],
                    gamma=query_width,
                    verification_capacity=verification_capacity,
                )
                changed_layout = plan_mixed_target_graph_layout(
                    changed_verification_rows,
                    [prompt_bucket - 3, 1, 1, 1],
                    gamma=query_width,
                    verification_capacity=verification_capacity,
                )
                if (
                    prompt_bucket > self.config.max_model_len
                    or first_layout.total_query_tokens > self.graph_runner.max_graph_tokens
                    or max(
                        first_layout.dummy_verification_rows * query_width,
                        changed_layout.dummy_verification_rows * query_width,
                        first_layout.query_lengths[-1] + 1,
                        changed_layout.query_lengths[-1] + 1,
                    )
                    > self.config.max_model_len
                ):
                    continue
                layouts_and_rows = (
                    (
                        first_layout,
                        verification_ids,
                        verification_starts,
                        prompt_ids[:1],
                        (0,),
                    ),
                    (
                        changed_layout,
                        verification_ids[:changed_verification_rows],
                        tuple(9 + index % 5 for index in range(changed_verification_rows)),
                        prompt_ids,
                        (7, 11, 13, 17),
                    ),
                )
                validation_calls_before = self.graph_runner.execution_counters["generic"][
                    "changed_input_validation_calls"
                ]
                validation_failures_before = self.graph_runner.execution_counters["generic"][
                    "runtime_validation_failures"
                ]
                entry_key = (
                    f"hidden|fia-stable:{first_layout.graph_key}",
                    first_layout.total_query_tokens,
                )
                for qualification_pass, (
                    layout,
                    real_verification_ids,
                    real_verification_starts,
                    real_prompt_ids,
                    prompt_starts,
                ) in enumerate(layouts_and_rows):
                    envelope = plan_mixed_target_graph_envelope(
                        layout,
                        real_verification_ids,
                        real_verification_starts,
                        real_prompt_ids,
                        prompt_starts,
                        scratch_ids,
                    )
                    inputs = torch.arange(
                        1 + qualification_pass,
                        1 + qualification_pass + layout.total_query_tokens,
                        dtype=torch.long,
                        device=self.device,
                    )
                    positions, metadata = self._prepare_mixed_target_graph_call(
                        layout,
                        envelope,
                        inputs,
                    )
                    self.graph_runner.run_stable_fia_hidden(
                        inputs,
                        positions,
                        metadata,
                        graph_key=layout.graph_key,
                        expected_tokens=layout.total_query_tokens,
                        expected_request_segments=layout.request_segment_count,
                    )
                entry = self.graph_runner.entries.get(entry_key)
                counters = self.graph_runner.execution_counters["generic"]
                if (
                    entry is None
                    or not entry.runtime_validated
                    or counters["changed_input_validation_calls"] != validation_calls_before + 1
                    or counters["runtime_validation_failures"] != validation_failures_before
                ):
                    raise RuntimeError(
                        "Mixed-target graph bucket failed changed-partition "
                        "qualification: "
                        f"prompt_tokens={prompt_bucket}, "
                        f"verification_capacity={verification_capacity}."
                    )
                self._mixed_target_graph_qualified_buckets = (
                    getattr(
                        self,
                        "_mixed_target_graph_qualified_buckets",
                        0,
                    )
                    + 1
                )
        self._qualify_stable_target_verify_graph()

    def _validate_stable_target_verify_numerics(
        self,
        layout: StableTargetVerifyGraphLayout,
        envelope: StableTargetVerifyGraphEnvelope,
        inputs: torch.Tensor,
    ) -> None:
        """Prove stable-envelope eager output against exact eager execution.

        Run exact eager first, run the stable FIA envelope eagerly, then run
        exact eager again.  The final exact pass restores the authoritative
        real-row KV contents and detects any scratch-row/cross-request
        contamination before graph capture is allowed.
        """

        self._stable_target_verify_numerical_validation_attempts = (
            getattr(
                self,
                "_stable_target_verify_numerical_validation_attempts",
                0,
            )
            + 1
        )
        real_tokens = layout.verification_output_count
        exact_inputs = inputs[:real_tokens]
        exact_sequence_ids = list(envelope.token_sequence_ids[:real_tokens])
        exact_positions = list(envelope.positions[:real_tokens])

        def run_exact() -> torch.Tensor:
            return (
                self._run_device_packed_greedy(
                    exact_inputs,
                    exact_sequence_ids,
                    exact_positions,
                    use_aclgraph=False,
                    use_fused_infer_attention=True,
                )
                .detach()
                .clone()
            )

        exact_before = run_exact()
        stable_positions, stable_metadata = self._prepare_stable_target_verify_graph_call(
            layout,
            envelope,
            inputs,
        )
        stable_hidden = self.model(
            inputs,
            stable_positions,
            stable_metadata,
        )
        stable_tokens = (
            self.model.compute_greedy_tokens(
                stable_hidden[:real_tokens],
                self.draft_vocab_size,
            )
            .detach()
            .clone()
        )
        exact_after = run_exact()

        if not torch.equal(exact_before, exact_after):
            self._stable_target_verify_numerical_validation_failures = (
                getattr(
                    self,
                    "_stable_target_verify_numerical_validation_failures",
                    0,
                )
                + 1
            )
            self._stable_target_verify_numerical_restore_failures = (
                getattr(
                    self,
                    "_stable_target_verify_numerical_restore_failures",
                    0,
                )
                + 1
            )
            raise RuntimeError(
                "Stable target-verify exact eager output changed after the "
                "envelope replay; KV restoration is not trustworthy: "
                f"rows={layout.verification_rows}, "
                f"capacity={layout.verification_capacity}."
            )
        if not torch.equal(exact_before, stable_tokens):
            self._stable_target_verify_numerical_validation_failures = (
                getattr(
                    self,
                    "_stable_target_verify_numerical_validation_failures",
                    0,
                )
                + 1
            )
            raise RuntimeError(
                "Stable target-verify envelope changed greedy tokens relative "
                "to exact unpadded eager execution: "
                f"rows={layout.verification_rows}, "
                f"capacity={layout.verification_capacity}."
            )
        self._stable_target_verify_numerical_validation_passes = (
            getattr(
                self,
                "_stable_target_verify_numerical_validation_passes",
                0,
            )
            + 1
        )

    def _qualify_stable_target_verify_graph(self) -> None:
        """Qualify exact Q=N*width graphs for every runtime N in [1, 32]."""

        if self.is_draft or not self._mixed_target_graph_enabled:
            return
        if self.cache_allocation is None or self.cache_block_tables is None:
            raise RuntimeError("Stable target-verify graph qualification requires an allocated cache.")
        if self._cache_sequence_capacity < MAX_MIXED_TARGET_VERIFY_CAPACITY:
            raise RuntimeError("Stable target-verify graph qualification requires 32 service rows.")
        query_width = self.gamma + int(getattr(self.config, "spec_rhythm_linear_bonus_token", False))
        if max(STABLE_TARGET_VERIFY_CAPACITIES) * query_width > (self.graph_runner.max_graph_tokens):
            raise RuntimeError("Stable target-verify exact graphs exceed the configured graph-token capacity.")
        scratch_id = self._mixed_target_scratch_sequence_ids[0]
        self.graph_runner.set_expected_fia_batch_size(None)
        self._stable_target_verify_graph_qualified = 0
        self._stable_target_verify_graph_capacity_map = {}
        self._stable_target_verify_graph_qualified_capacities = ()
        self._stable_target_verify_numerical_validation_attempts = 0
        self._stable_target_verify_numerical_validation_passes = 0
        self._stable_target_verify_numerical_validation_failures = 0
        self._stable_target_verify_numerical_restore_failures = 0
        qualified_capacities: list[int] = []

        # Largest-first keeps the most demanding causal-FIA workspace as the
        # pool owner.  Every graph is exact-Q: no measured request ever pays
        # dummy-row compute or makes a runtime padding decision.
        for qualification_index, capacity in enumerate(reversed(STABLE_TARGET_VERIFY_CAPACITIES)):
            layout = _cached_stable_target_verify_graph_layout(
                capacity,
                query_width,
                capacity,
            )
            sequence_ids = tuple(range(capacity))
            first_starts = (1,) * capacity
            changed_starts = (2,) * capacity
            first_envelope = plan_stable_target_verify_graph_envelope(
                layout,
                sequence_ids,
                first_starts,
                scratch_id,
            )
            changed_envelope = plan_stable_target_verify_graph_envelope(
                layout,
                sequence_ids,
                changed_starts,
                scratch_id,
            )
            token_start = 1 + qualification_index * 2
            first_inputs = torch.arange(
                token_start,
                token_start + layout.total_query_tokens,
                dtype=torch.long,
                device=self.device,
            )
            changed_inputs = torch.arange(
                token_start + 1,
                token_start + 1 + layout.total_query_tokens,
                dtype=torch.long,
                device=self.device,
            )
            counters = self.graph_runner.execution_counters["generic"]
            validation_calls_before = counters["changed_input_validation_calls"]
            validation_failures_before = counters["runtime_validation_failures"]

            self._validate_stable_target_verify_numerics(
                layout,
                first_envelope,
                first_inputs,
            )
            first_positions, first_metadata = self._prepare_stable_target_verify_graph_call(
                layout,
                first_envelope,
                first_inputs,
            )
            self.graph_runner.run_stable_fia_greedy(
                first_inputs,
                first_positions,
                first_metadata,
                self.draft_vocab_size,
                graph_key=layout.graph_key,
                expected_tokens=layout.total_query_tokens,
                expected_request_segments=layout.request_segment_count,
            )
            self._validate_stable_target_verify_numerics(
                layout,
                changed_envelope,
                changed_inputs,
            )
            changed_positions, changed_metadata = self._prepare_stable_target_verify_graph_call(
                layout,
                changed_envelope,
                changed_inputs,
            )
            self.graph_runner.run_stable_fia_greedy(
                changed_inputs,
                changed_positions,
                changed_metadata,
                self.draft_vocab_size,
                graph_key=layout.graph_key,
                expected_tokens=layout.total_query_tokens,
                expected_request_segments=layout.request_segment_count,
            )
            entry_key = (
                f"greedy:{self.draft_vocab_size}|fia-stable:{layout.graph_key}",
                layout.total_query_tokens,
            )
            entry = self.graph_runner.entries.get(entry_key)
            if (
                entry is None
                or not entry.runtime_validated
                or counters["changed_input_validation_calls"] != validation_calls_before + 1
                or counters["runtime_validation_failures"] != validation_failures_before
            ):
                raise RuntimeError(
                    f"Stable target-verify exact graph failed changed-input qualification: capacity={capacity}."
                )
            qualified_capacities.append(capacity)

        qualified_capacities.sort()
        if tuple(qualified_capacities) != STABLE_TARGET_VERIFY_CAPACITIES:
            raise RuntimeError("Stable target-verify qualification did not cover every exact request count.")
        expected_numerical_validations = 2 * len(STABLE_TARGET_VERIFY_CAPACITIES)
        if (
            self._stable_target_verify_numerical_validation_attempts != expected_numerical_validations
            or self._stable_target_verify_numerical_validation_passes != expected_numerical_validations
            or self._stable_target_verify_numerical_validation_failures
            or self._stable_target_verify_numerical_restore_failures
        ):
            raise RuntimeError(
                "Stable target-verify qualification did not complete every exact/envelope/restore numerical comparison."
            )
        self._stable_target_verify_graph_capacity_map = {
            capacity: capacity for capacity in STABLE_TARGET_VERIFY_CAPACITIES
        }
        self._stable_target_verify_graph_qualified_capacities = tuple(qualified_capacities)
        self._stable_target_verify_graph_qualified = 1

    def _draft_round_batch(
        self,
        states: list[PearlPipelineState],
        active_indices: list[int],
    ) -> tuple[list[list[int]], list[list[int]]]:
        if not self.is_draft:
            return [], []
        was_pre_verify = [states[index].pre_verify for index in active_indices]
        input_ids = torch.tensor(
            [states[index].token_ids[-1] for index in active_indices],
            dtype=torch.long,
            device=self.device,
        )
        first_positions = [len(states[index].token_ids) - 1 for index in active_indices]
        draft_use_fia = not getattr(self.config, "draft_use_paged_attention", False)
        if getattr(self.config, "enable_spec_rhythm", False) and getattr(
            self.config, "spec_rhythm_stable_graphs", True
        ):
            draft_use_fia = False
        if (
            not self.config.enforce_eager
            and self.gamma <= 16
            and hasattr(self, "graph_runner")
            and hasattr(self.graph_runner, "run_draft_greedy")
        ):
            position_tensors = []
            attention_metadatas = []
            for step in range(self.gamma):
                step_positions = [position + step for position in first_positions]
                position_tensor, attention_metadata = self._prepare_attention_metadata(
                    active_indices,
                    step_positions,
                    use_fused_infer_attention=draft_use_fia,
                )
                position_tensors.append(position_tensor)
                attention_metadatas.append(attention_metadata)
            draft_tensor = self.graph_runner.run_draft_greedy(
                input_ids,
                position_tensors,
                attention_metadatas,
                self.draft_vocab_size,
            )
            draft_windows = draft_tensor.cpu().tolist()
        else:
            draft_steps: list[torch.Tensor] = []
            for step in range(self.gamma):
                positions = [position + step for position in first_positions]
                input_ids = self._run_device_packed_greedy(
                    input_ids,
                    active_indices,
                    positions,
                    use_aclgraph=(not self.config.enforce_eager and self.gamma <= 16),
                    use_fused_infer_attention=draft_use_fia and self.gamma <= 16,
                )
                # ACLGraph replays reuse one persistent output buffer. Preserve
                # each proposal before the next replay overwrites that buffer.
                draft_steps.append(input_ids.clone())
            draft_windows = torch.stack(draft_steps, dim=1).cpu().tolist()
        for sequence_index, token_ids in zip(active_indices, draft_windows):
            states[sequence_index].token_ids.extend(int(token_id) for token_id in token_ids)
        next_windows = [states[index].token_ids[-self.gamma :] for index in active_indices]
        verification_windows = []
        for sequence_index, is_pre_verify, next_window in zip(active_indices, was_pre_verify, next_windows):
            state = states[sequence_index]
            verification_windows.append(
                [next_window[0]] if is_pre_verify else state.token_ids[-2 * self.gamma + 1 : -self.gamma + 1]
            )
        return verification_windows, next_windows

    def _draft_round_device_batch(
        self,
        states: list[PearlPipelineState],
        active_indices: list[int],
        draft_budgets: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Produce packed proposals without an intermediate NPU-to-CPU copy."""
        if not self.is_draft:
            return None, None
        if getattr(self.config, "draft_mode", SERIAL_LINEAR_DRAFT_MODE) == PARD_PARALLEL_DRAFT_MODE:
            raise RuntimeError(
                "pard_parallel cannot use the legacy cross-window draft loop; "
                "it requires the explicit full-window PARD path."
            )
        budgets = (
            [self.gamma] * len(active_indices) if draft_budgets is None else [int(value) for value in draft_budgets]
        )
        if len(budgets) != len(active_indices) or any(value <= 0 or value > self.gamma for value in budgets):
            raise ValueError("PEARL draft budgets must contain one value in [1, gamma] per request.")
        was_pre_verify = [states[index].pre_verify for index in active_indices]
        input_ids = torch.tensor(
            [states[index].token_ids[-1] for index in active_indices],
            dtype=torch.long,
            device=self.device,
        )
        first_positions = [len(states[index].token_ids) - 1 for index in active_indices]
        draft_temperatures = [states[index].draft_temperature for index in active_indices]
        draft_top_ps = [states[index].draft_top_p for index in active_indices]
        draft_top_ks = [states[index].draft_top_k for index in active_indices]
        draft_use_fia = not getattr(self.config, "draft_use_paged_attention", False)
        if (
            not self.config.enforce_eager
            and self.gamma <= 16
            and all(value == 0 for value in draft_temperatures)
            and all(value == self.gamma for value in budgets)
            and hasattr(self, "graph_runner")
            and hasattr(self.graph_runner, "run_draft_greedy")
        ):
            position_tensors = []
            attention_metadatas = []
            for step in range(self.gamma):
                step_positions = [position + step for position in first_positions]
                position_tensor, attention_metadata = self._prepare_attention_metadata(
                    active_indices,
                    step_positions,
                    use_fused_infer_attention=draft_use_fia,
                )
                position_tensors.append(position_tensor)
                attention_metadatas.append(attention_metadata)
            next_windows = self.graph_runner.run_draft_greedy(
                input_ids,
                position_tensors,
                attention_metadatas,
                self.draft_vocab_size,
            )
        else:
            next_windows = torch.full(
                (len(active_indices), self.gamma),
                -1,
                dtype=torch.long,
                device=self.device,
            )
            for step in range(max(budgets)):
                active_rows = [row for row, budget in enumerate(budgets) if step < budget]
                row_tensor = torch.tensor(active_rows, dtype=torch.long, device=self.device)
                step_sequence_ids = [active_indices[row] for row in active_rows]
                positions = [first_positions[row] + step for row in active_rows]
                step_input = input_ids.index_select(0, row_tensor)
                step_temperatures = [draft_temperatures[row] for row in active_rows]
                if all(value == 0 for value in step_temperatures):
                    step_output = self._run_device_packed_greedy(
                        step_input,
                        step_sequence_ids,
                        positions,
                        use_aclgraph=not self.config.enforce_eager and self.gamma <= 16,
                        use_fused_infer_attention=draft_use_fia and self.gamma <= 16,
                    )
                else:
                    step_output, _ = self._run_device_packed_sample_with_confidence(
                        step_input,
                        step_sequence_ids,
                        positions,
                        step_temperatures,
                        top_ps=[draft_top_ps[row] for row in active_rows],
                        top_ks=[draft_top_ks[row] for row in active_rows],
                        use_aclgraph=False,
                        use_fused_infer_attention=draft_use_fia and self.gamma <= 16,
                    )
                # Graph replays may return a persistent output buffer, so copy
                # both the proposal column and the next-step input explicitly.
                step_output = step_output.clone()
                next_windows[row_tensor, step] = step_output
                input_ids = input_ids.clone()
                input_ids.index_copy_(0, row_tensor, step_output)

        verification_sizes = [
            1 if pre_verify else (states[index].pending_window_size or self.gamma)
            for index, pre_verify in zip(active_indices, was_pre_verify)
        ]
        verification_size = sum(verification_sizes)
        current_indices: list[int] = []
        previous_indices: list[int] = []
        previous_tokens: list[int] = []
        offset = 0
        for sequence_index, pre_verify, size in zip(
            active_indices,
            was_pre_verify,
            verification_sizes,
        ):
            if not pre_verify and size > 1:
                previous_indices.extend(range(offset, offset + size - 1))
                previous_tokens.extend(states[sequence_index].token_ids[-(size - 1) :])
            current_indices.append(offset + size - 1)
            offset += size
        verification = torch.empty(
            verification_size,
            dtype=torch.long,
            device=self.device,
        )
        if previous_indices:
            verification[torch.tensor(previous_indices, dtype=torch.long, device=self.device)] = torch.tensor(
                previous_tokens, dtype=torch.long, device=self.device
            )
        verification[torch.tensor(current_indices, dtype=torch.long, device=self.device)] = next_windows[:, 0]
        return verification, next_windows

    def _draft_pard_parallel_eager_batch(
        self,
        states: list[PearlPipelineState],
        active_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Produce one fixed PARD window per request in one eager forward.

        Every request contributes its unsynchronized real-token suffix (or a
        replay of an already synchronized root) followed by three PARD mask
        tokens.  Only the real suffix advances ``draft_synced_length``; mask KV
        rows remain disposable and are overwritten by the next repair pass.

        This method is intentionally eager-only.  Public PARD execution stays
        fail-closed until this exact layout and KV overwrite contract pass NPU
        numerical qualification; callers must never silently use the serial
        four-forward implementation for ``pard_parallel``.
        """

        if not self.is_draft:
            raise RuntimeError("Only a draft worker can execute PARD proposals.")
        if self.gamma != 4:
            raise RuntimeError("Native PARD eager drafting requires fixed gamma=4.")
        if self.pard_token is None:
            raise RuntimeError("Native PARD eager drafting has no validated mask token.")
        if not active_indices or len(set(active_indices)) != len(active_indices):
            raise ValueError("PARD eager drafting requires unique active requests.")
        if any(not 0 <= index < len(states) for index in active_indices):
            raise ValueError("PARD eager drafting received an invalid request index.")
        if any(states[index].draft_temperature != 0 for index in active_indices):
            raise RuntimeError("Native PARD eager drafting is qualified for greedy proposals only.")

        repair_suffixes: list[list[int]] = []
        first_positions: list[int] = []
        for index in active_indices:
            suffix, first_position = states[index].pard_repair_suffix()
            repair_suffixes.append(suffix)
            first_positions.append(first_position)
        layout = build_pard_parallel_draft_layout(
            repair_suffixes,
            active_indices,
            first_positions,
            pard_token=self.pard_token,
            vocab_size=self.draft_vocab_size,
            gamma=self.gamma,
        )
        hidden_states = self._run_packed_hidden(
            list(layout.input_token_ids),
            list(layout.sequence_ids),
            list(layout.positions),
            use_aclgraph=False,
            logit_indices=list(layout.sample_row_indices),
            use_fused_infer_attention=False,
        )
        self._pard_parallel_eager_forward_calls = (
            getattr(
                self,
                "_pard_parallel_eager_forward_calls",
                0,
            )
            + 1
        )
        self._pard_parallel_repair_rows = getattr(
            self,
            "_pard_parallel_repair_rows",
            0,
        ) + sum(len(suffix) for suffix in repair_suffixes)
        self._pard_parallel_mask_rows = getattr(
            self,
            "_pard_parallel_mask_rows",
            0,
        ) + len(active_indices) * (self.gamma - 1)
        self._pard_parallel_serial_fallback_calls = getattr(
            self,
            "_pard_parallel_serial_fallback_calls",
            0,
        )
        flat_proposals = self.model.compute_greedy_tokens(
            hidden_states,
            self.draft_vocab_size,
        )
        expected = len(active_indices) * self.gamma
        if flat_proposals.ndim != 1 or flat_proposals.numel() != expected:
            raise RuntimeError("PARD eager LM head did not return the fixed [B, 4] proposal envelope.")
        proposals = flat_proposals.reshape(len(active_indices), self.gamma).clone()
        if torch.any(proposals < 0) or torch.any(proposals >= self.draft_vocab_size):
            raise RuntimeError("PARD eager LM head returned a token outside the shared vocabulary.")

        # Mutate host KV frontiers only after the complete batched output has
        # passed shape/range checks, so one malformed request cannot partially
        # advance another request's state.
        for index in active_indices:
            states[index].mark_pard_draft_repaired()
        confidence = torch.ones(
            len(active_indices),
            dtype=torch.float32,
            device=proposals.device,
        )
        return proposals.reshape(-1), proposals, confidence

    def _get_linear_draft_fia_persistent_envelope(
        self,
        *,
        graph_lane: int | None,
        capture_size: int,
        step_count: int,
        block_size: int,
        table_width: int,
        sequence_lens: Sequence[int],
        input_dtype: torch.dtype,
        table_dtype: torch.dtype,
    ) -> _LinearDraftFIAPersistentEnvelope:
        """Return one lane-local common-KV staging allocation.

        Request identity, exact position and page-table contents are refreshed
        by the caller.  The common KV signature is part of the key because it
        remains embedded in both the cached metadata objects and the resident
        draft ACLGraph task contract.
        """

        stable_sequence_lens = tuple(int(value) for value in sequence_lens)
        if (
            capture_size <= 0
            or step_count <= 0
            or block_size <= 0
            or table_width <= 0
            or len(stable_sequence_lens) != capture_size
            or not stable_sequence_lens
            or any(value != stable_sequence_lens[0] for value in stable_sequence_lens)
        ):
            raise ValueError(
                "Persistent linear draft FIA staging requires one positive common-KV length per captured row."
            )
        capacity = table_width * block_size
        if stable_sequence_lens[0] <= 0 or stable_sequence_lens[0] > capacity:
            raise ValueError("Persistent linear draft FIA common-KV length must fit the request page-table capacity.")
        key = (
            graph_lane,
            int(capture_size),
            int(step_count),
            int(block_size),
            int(table_width),
            int(capacity),
            stable_sequence_lens[0],
            torch.device(self.device),
            input_dtype,
            table_dtype,
        )
        pool = getattr(
            self,
            "_linear_draft_fia_persistent_staging_pool",
            None,
        )
        if pool is None:
            # CPU protocol harnesses may instantiate the engine with
            # ``__new__`` and skip the production initializer.
            pool = {}
            self._linear_draft_fia_persistent_staging_pool = pool
        envelope = pool.get(key)
        if envelope is not None:
            self._linear_draft_fia_persistent_staging_hits = (
                getattr(
                    self,
                    "_linear_draft_fia_persistent_staging_hits",
                    0,
                )
                + 1
            )
            return envelope

        staging = LinearDraftFIAPersistentStaging(
            step_count=step_count,
            capture_size=capture_size,
            capacity=capacity,
            table_width=table_width,
            device=self.device,
            input_dtype=input_dtype,
            position_dtype=torch.long,
            slot_dtype=torch.int32,
            table_dtype=table_dtype,
        )
        actual_seq_lengths_q = tuple(range(1, capture_size + 1))
        context_lens = torch.tensor(
            stable_sequence_lens,
            dtype=torch.int32,
        )
        attention_metadatas = tuple(
            NativeAttentionMetadata(
                slot_mapping=staging.slot_mapping_views[step],
                context_lens=context_lens,
                block_tables=staging.request_block_tables,
                actual_seq_lengths_q=actual_seq_lengths_q,
                sequence_lens=stable_sequence_lens,
                request_block_tables=staging.request_block_tables,
                attention_mask=None,
                use_fused_infer_attention=True,
                tree_attention=True,
                tree_attention_mask=staging.full_mask_views[step],
            )
            for step in range(step_count)
        )
        envelope = _LinearDraftFIAPersistentEnvelope(
            staging=staging,
            position_tensors=staging.position_views,
            attention_metadatas=attention_metadatas,
        )
        pool[key] = envelope
        self._linear_draft_fia_persistent_staging_misses = (
            getattr(
                self,
                "_linear_draft_fia_persistent_staging_misses",
                0,
            )
            + 1
        )
        return envelope

    def _draft_spec_rhythm_device_batch(
        self,
        states: list[PearlPipelineState],
        active_indices: list[int],
        draft_budgets: Sequence[int],
        *,
        verification_sizes: Sequence[int] | None = None,
        verification_prefixes: Sequence[torch.Tensor | None] | None = None,
        full_window: bool = False,
        graph_lane: int | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Draft variable windows and report mean softmax confidence per row."""

        if not self.is_draft:
            return None, None, None
        budgets = [int(value) for value in draft_budgets]
        if len(budgets) != len(active_indices) or any(value <= 0 or value > self.gamma for value in budgets):
            raise ValueError("SpecRhythm draft budgets must be in [1, gamma].")
        if getattr(self.config, "draft_mode", SERIAL_LINEAR_DRAFT_MODE) == PARD_PARALLEL_DRAFT_MODE:
            prefixes = [None] * len(active_indices) if verification_prefixes is None else list(verification_prefixes)
            sizes = [self.gamma] * len(active_indices) if verification_sizes is None else list(verification_sizes)
            if not full_window:
                raise RuntimeError("Native PARD supports only explicit full-window verification.")
            if any(value != self.gamma for value in budgets) or any(value != self.gamma for value in sizes):
                raise RuntimeError("Native PARD requires one complete fixed-gamma window per request.")
            if any(prefix is not None for prefix in prefixes):
                raise RuntimeError("Native PARD full-window proposals cannot consume cross-window prefixes.")
            return self._draft_pard_parallel_eager_batch(states, active_indices)
        input_ids = torch.tensor(
            [states[index].token_ids[-1] for index in active_indices],
            dtype=torch.long,
            device=self.device,
        )
        first_positions = [len(states[index].token_ids) - 1 for index in active_indices]
        draft_temperatures = [states[index].draft_temperature for index in active_indices]
        draft_top_ps = [states[index].draft_top_p for index in active_indices]
        draft_top_ks = [states[index].draft_top_k for index in active_indices]
        draft_use_fia = not getattr(self.config, "draft_use_paged_attention", False)
        stable_spec_rhythm_graphs = bool(
            getattr(self.config, "enable_spec_rhythm", False)
            and getattr(self.config, "spec_rhythm_stable_graphs", True)
        )
        if stable_spec_rhythm_graphs:
            draft_use_fia = False
        bucketed_linear_draft_fia = bool(
            envs.VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_BUCKET and stable_spec_rhythm_graphs and full_window
        )
        batched_linear_draft_fia_masks = bool(
            bucketed_linear_draft_fia and envs.VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_BATCHED_MASKS
        )
        common_linear_draft_fia_kv = bool(
            bucketed_linear_draft_fia and envs.VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_COMMON_KV
        )
        ranked_linear_draft_fia_kv = bool(
            bucketed_linear_draft_fia and envs.VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_RANKED_KV
        )
        if common_linear_draft_fia_kv and ranked_linear_draft_fia_kv:
            raise ValueError("Ranked and common linear-draft FIA KV modes are mutually exclusive.")
        persistent_linear_draft_fia_staging = bool(
            envs.VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_PERSISTENT_STAGING
            and batched_linear_draft_fia_masks
            and common_linear_draft_fia_kv
        )
        prefixes = [None] * len(active_indices) if verification_prefixes is None else list(verification_prefixes)
        max_graph_batch_size = max(
            1,
            getattr(self.config, "max_num_seqs", len(active_indices)),
        )
        # Online traces often contain only a handful of resident requests.
        # Padding every replay to half of the service capacity made the TP1
        # draft the critical path (for B64, one real row executed as 32).
        # A bounded graph family keeps shapes stable while adding intermediate
        # service buckets where the formal online trace spends most cycles.
        # This avoids executing a 64-row serial graph for, for example, 33
        # useful rows without turning every online batch size into a graph.
        graph_batch_size = _next_linear_draft_graph_bucket(
            len(active_indices),
            max_graph_batch_size,
        )
        fixed_greedy_graph_eligible = (
            not self.config.enforce_eager
            and self.gamma <= 16
            and not draft_use_fia
            # Normal work replays the home-capacity entry; rolling-eager work
            # replays the full-capacity entry.  Online admission changes the
            # resident rows, not either graph shape; paged-attention tasks are
            # refreshed before every replay.
            and len(active_indices) <= max_graph_batch_size
            and all(value == 0 for value in draft_temperatures)
            and all(value == self.gamma for value in budgets)
            and hasattr(self, "graph_runner")
            and hasattr(self.graph_runner, "run_draft_greedy")
        )
        # The proposal remains one autoregressive chain: token t+1 consumes
        # token t.  Capture all fixed-gamma steps into one ACLGraph so a small
        # online batch does not pay four Python dispatch/replay boundaries.
        # PagedAttention task updates refresh their sequence-dependent
        # workspaces before replay, including for the tiny graph buckets.
        use_full_draft_graph = fixed_greedy_graph_eligible
        if use_full_draft_graph:
            ranked_kv_plan = (
                _ranked_linear_draft_fia_plan(
                    states,
                    active_indices,
                    self.gamma,
                    graph_batch_size,
                )
                if ranked_linear_draft_fia_kv
                else None
            )
            graph_active_indices = active_indices
            graph_first_positions = first_positions
            graph_input_root_ids = input_ids
            if ranked_kv_plan is not None:
                graph_active_indices = [active_indices[row] for row in ranked_kv_plan.graph_to_caller_rows]
                graph_first_positions = [first_positions[row] for row in ranked_kv_plan.graph_to_caller_rows]
                graph_input_root_ids = input_ids.index_select(
                    0,
                    torch.tensor(
                        ranked_kv_plan.graph_to_caller_rows,
                        dtype=torch.long,
                        device=input_ids.device,
                    ),
                )
            if ranked_kv_plan is not None:
                linear_draft_fia_kv_limits = list(ranked_kv_plan.capacity_limits[: len(active_indices)])
                dummy_linear_draft_fia_kv_limits = list(ranked_kv_plan.capacity_limits[len(active_indices) :])
            elif bucketed_linear_draft_fia:
                linear_draft_fia_kv_limits = _linear_draft_fia_lifetime_limits(
                    states,
                    active_indices,
                    self.gamma,
                    common_kv=common_linear_draft_fia_kv,
                )
                dummy_linear_draft_fia_kv_limits = (
                    [linear_draft_fia_kv_limits[0]] * (graph_batch_size - len(active_indices))
                    if common_linear_draft_fia_kv
                    else None
                )
            else:
                linear_draft_fia_kv_limits = None
                dummy_linear_draft_fia_kv_limits = None
            position_tensors = []
            attention_metadatas = []
            graph_input_ids = None
            draft_graph_steps = self.gamma + int(
                getattr(
                    self.config,
                    "spec_rhythm_linear_bonus_token",
                    False,
                )
            )
            precomputed_full_masks: tuple[torch.Tensor, ...] | None = None
            shared_host_request_tables: list[list[int]] | None = None
            shared_request_block_tables: torch.Tensor | None = None
            persistent_fia_envelope: _LinearDraftFIAPersistentEnvelope | None = None
            if batched_linear_draft_fia_masks:
                final_step_positions = [position + draft_graph_steps - 1 for position in graph_first_positions]
                # Allocate through the final serial step before taking the
                # request-table snapshot shared by all steps. Earlier queries
                # cannot observe these pages because their exact FULL masks
                # continue to hide every future key.
                self._ensure_cache_capacity(
                    graph_active_indices,
                    final_step_positions,
                )
                assert self.cache_allocation is not None
                shared_host_request_tables = [
                    self.cache_allocation.block_tables[index] for index in graph_active_indices
                ]
                block_size = self.model.layers[0].self_attn.block_size
                table_capacity = len(shared_host_request_tables[0]) * block_size
                validate_linear_draft_fia_final_pages(
                    graph_first_positions,
                    shared_host_request_tables,
                    step_count=draft_graph_steps,
                    block_size=block_size,
                    capacity=table_capacity,
                )
                if persistent_linear_draft_fia_staging:
                    assert self.cache_block_tables is not None
                    assert linear_draft_fia_kv_limits is not None
                    assert dummy_linear_draft_fia_kv_limits is not None
                    stable_sequence_lens = (
                        *linear_draft_fia_kv_limits,
                        *dummy_linear_draft_fia_kv_limits,
                    )
                    persistent_fia_envelope = self._get_linear_draft_fia_persistent_envelope(
                        graph_lane=graph_lane,
                        capture_size=graph_batch_size,
                        step_count=draft_graph_steps,
                        block_size=block_size,
                        table_width=len(shared_host_request_tables[0]),
                        sequence_lens=stable_sequence_lens,
                        input_dtype=graph_input_root_ids.dtype,
                        table_dtype=self.cache_block_tables.dtype,
                    )
                    request_sequence_tensor = torch.tensor(
                        graph_active_indices,
                        dtype=torch.long,
                        device=self.device,
                    )
                    # One request-table gather replaces the four gathers made
                    # by per-step ``_prepare_attention_metadata`` calls. Keep
                    # an allocating gather until index_select(out=...) has an
                    # explicit Ascend compatibility gate.
                    selected_request_tables = self.cache_block_tables.index_select(
                        0,
                        request_sequence_tensor,
                    )
                    step_slot_mappings = tuple(
                        self._cache_slot_mapping(
                            graph_active_indices,
                            [position + step for position in graph_first_positions],
                        )
                        for step in range(draft_graph_steps)
                    )
                    persistent_fia_envelope.staging.refresh(
                        graph_first_positions,
                        step_slot_mappings,
                        selected_request_tables,
                        graph_input_root_ids,
                    )
                    graph_input_ids = persistent_fia_envelope.staging.input_ids
                    position_tensors.extend(persistent_fia_envelope.position_tensors)
                    attention_metadatas.extend(persistent_fia_envelope.attention_metadatas)
                    precomputed_full_masks = persistent_fia_envelope.staging.full_mask_views
                    shared_request_block_tables = persistent_fia_envelope.staging.request_block_tables
                    self._linear_draft_fia_shared_request_table_calls = (
                        getattr(
                            self,
                            "_linear_draft_fia_shared_request_table_calls",
                            0,
                        )
                        + 1
                    )
                else:
                    full_mask_builder = getattr(
                        self,
                        "_linear_draft_fia_full_mask_builder",
                        None,
                    )
                    if full_mask_builder is None:
                        # CPU protocol harnesses may construct an engine through
                        # ``__new__`` without running the production initializer.
                        full_mask_builder = LinearDraftFIAFullMaskBuilder()
                        self._linear_draft_fia_full_mask_builder = full_mask_builder
                    precomputed_full_masks = full_mask_builder.build(
                        graph_first_positions,
                        step_count=draft_graph_steps,
                        capture_size=graph_batch_size,
                        capacity=table_capacity,
                        device=self.device,
                    )
                self._linear_draft_fia_batched_mask_calls = getattr(self, "_linear_draft_fia_batched_mask_calls", 0) + 1
                self._linear_draft_fia_batched_mask_elements = getattr(
                    self,
                    "_linear_draft_fia_batched_mask_elements",
                    0,
                ) + sum(mask.numel() for mask in precomputed_full_masks)
            for step in range(draft_graph_steps) if persistent_fia_envelope is None else ():
                step_positions = [position + step for position in graph_first_positions]
                position_tensor, attention_metadata = self._prepare_attention_metadata(
                    graph_active_indices,
                    step_positions,
                    use_fused_infer_attention=(draft_use_fia or bucketed_linear_draft_fia),
                )
                if bucketed_linear_draft_fia:
                    if precomputed_full_masks is not None and shared_request_block_tables is None:
                        assert attention_metadata.request_block_tables is not None
                        shared_request_block_tables = sanitize_and_pad_linear_draft_fia_request_tables(
                            attention_metadata.request_block_tables,
                            capture_size=graph_batch_size,
                        )
                        self._linear_draft_fia_shared_request_table_calls = (
                            getattr(
                                self,
                                "_linear_draft_fia_shared_request_table_calls",
                                0,
                            )
                            + 1
                        )
                    padded_positions, padded_metadata = _pad_bucketed_linear_draft_fia_graph_metadata(
                        position_tensor,
                        attention_metadata,
                        graph_batch_size,
                        block_size=self.model.layers[0].self_attn.block_size,
                        exact_positions=step_positions,
                        host_request_block_tables=(
                            shared_host_request_tables
                            if shared_host_request_tables is not None
                            else [self.cache_allocation.block_tables[index] for index in graph_active_indices]
                        ),
                        kv_length_limits=linear_draft_fia_kv_limits,
                        dummy_kv_length_limits=(dummy_linear_draft_fia_kv_limits),
                        precomputed_full_mask=(
                            precomputed_full_masks[step] if precomputed_full_masks is not None else None
                        ),
                        shared_request_block_tables=(shared_request_block_tables),
                    )
                    if graph_input_ids is None:
                        graph_input_ids = torch.zeros(
                            graph_batch_size,
                            dtype=graph_input_root_ids.dtype,
                            device=graph_input_root_ids.device,
                        )
                        graph_input_ids[: graph_input_root_ids.shape[0]].copy_(graph_input_root_ids)
                elif graph_input_ids is None:
                    (
                        graph_input_ids,
                        padded_positions,
                        padded_metadata,
                    ) = NativeACLGraphRunner._pad_inputs(
                        graph_input_root_ids,
                        position_tensor,
                        attention_metadata,
                        graph_batch_size,
                    )
                else:
                    padded_positions, padded_metadata = _pad_linear_draft_graph_metadata(
                        position_tensor,
                        attention_metadata,
                        graph_batch_size,
                    )
                position_tensors.append(padded_positions)
                attention_metadatas.append(padded_metadata)
            assert graph_input_ids is not None
            run_kwargs: dict[str, Any] = {
                "valid_row_count": len(active_indices),
            }
            if bucketed_linear_draft_fia:
                run_kwargs["graph_lane"] = graph_lane
            if common_linear_draft_fia_kv and envs.VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_STABLE_TASK_BARRIER:
                run_kwargs["stable_task_barrier"] = True
            if getattr(
                self.config,
                "spec_rhythm_linear_bonus_token",
                False,
            ):
                run_kwargs["final_kv_only"] = True
            graph_windows = self.graph_runner.run_draft_greedy(
                graph_input_ids,
                position_tensors,
                attention_metadatas,
                self.draft_vocab_size,
                **run_kwargs,
            )[: len(active_indices)].clone()
            if ranked_kv_plan is not None:
                graph_windows = _restore_ranked_linear_draft_rows(
                    graph_windows,
                    ranked_kv_plan.caller_to_graph_rows,
                )
            # The final bonus-preparation output is not a draft proposal. Its
            # forward exists only to write d_gamma into the draft KV cache.
            next_windows = graph_windows[:, : self.gamma].contiguous()
            draft_execution = getattr(self.graph_runner, "last_draft_execution", None)
            _record_linear_draft_full_chain_metrics(
                self,
                logical_rows=len(active_indices),
                padded_rows=graph_batch_size,
                gamma=self.gamma,
                execution=(draft_execution if isinstance(draft_execution, NativeGraphExecution) else None),
            )
            # Confidence only affects optional eager prioritization.  Greedy
            # default SpecRhythm does not branch on it, and a constant device
            # value avoids materializing logits from the graph output.
            graph_confidence = torch.ones(
                len(active_indices),
                dtype=torch.float32,
                device=self.device,
            )
            confidence = (
                _restore_ranked_linear_draft_rows(
                    graph_confidence,
                    ranked_kv_plan.caller_to_graph_rows,
                )
                if ranked_kv_plan is not None
                else graph_confidence
            )
        else:
            next_windows = torch.full(
                (len(active_indices), self.gamma),
                -1,
                dtype=torch.long,
                device=self.device,
            )
            confidence_sums = torch.zeros(
                len(active_indices),
                dtype=torch.float32,
                device=self.device,
            )
            self._linear_draft_stepwise_calls = getattr(self, "_linear_draft_stepwise_calls", 0) + 1
            self._linear_draft_stepwise_model_calls = getattr(self, "_linear_draft_stepwise_model_calls", 0) + max(
                budgets
            )
            for step in range(max(budgets)):
                active_rows = [row for row, budget in enumerate(budgets) if step < budget]
                row_tensor = torch.tensor(active_rows, dtype=torch.long, device=self.device)
                sequence_ids = [active_indices[row] for row in active_rows]
                positions = [first_positions[row] + step for row in active_rows]
                step_input = input_ids.index_select(0, row_tensor)
                step_temperatures = [draft_temperatures[row] for row in active_rows]
                if all(value == 0 for value in step_temperatures):
                    step_tokens, step_confidence = self._run_device_packed_greedy_with_confidence(
                        step_input,
                        sequence_ids,
                        positions,
                        use_aclgraph=(not self.config.enforce_eager and self.gamma <= 16),
                        use_fused_infer_attention=draft_use_fia and self.gamma <= 16,
                    )
                else:
                    step_tokens, step_confidence = self._run_device_packed_sample_with_confidence(
                        step_input,
                        sequence_ids,
                        positions,
                        step_temperatures,
                        top_ps=[draft_top_ps[row] for row in active_rows],
                        top_ks=[draft_top_ks[row] for row in active_rows],
                        use_aclgraph=False,
                        use_fused_infer_attention=draft_use_fia and self.gamma <= 16,
                    )
                step_tokens = step_tokens.clone()
                next_windows[row_tensor, step] = step_tokens
                confidence_sums.index_add_(0, row_tensor, step_confidence.float())
                input_ids = input_ids.clone()
                input_ids.index_copy_(0, row_tensor, step_tokens)

            if getattr(
                self.config,
                "spec_rhythm_linear_bonus_token",
                False,
            ):
                final_rows = [row for row, budget in enumerate(budgets) if budget == self.gamma]
                if final_rows:
                    final_row_tensor = torch.tensor(
                        final_rows,
                        dtype=torch.long,
                        device=self.device,
                    )
                    self._run_device_packed_hidden(
                        next_windows[:, self.gamma - 1].index_select(
                            0,
                            final_row_tensor,
                        ),
                        [active_indices[row] for row in final_rows],
                        [first_positions[row] + self.gamma for row in final_rows],
                        use_aclgraph=False,
                        use_fused_infer_attention=(draft_use_fia and self.gamma <= 16),
                    )

            confidence = confidence_sums / torch.tensor(
                budgets,
                dtype=torch.float32,
                device=self.device,
            )
        actual_verification_sizes = (
            [
                1 if states[index].pre_verify else (states[index].pending_window_size or self.gamma)
                for index in active_indices
            ]
            if verification_sizes is None
            else [int(value) for value in verification_sizes]
        )
        if len(actual_verification_sizes) != len(active_indices) or len(prefixes) != len(active_indices):
            raise ValueError("SpecRhythm verification metadata must contain one row per request.")
        if full_window:
            if any(prefix is not None for prefix in prefixes):
                raise ValueError("Full-window serial proposals do not consume a cross-window prefix.")
            if any(
                size != self.gamma or budget != self.gamma for size, budget in zip(actual_verification_sizes, budgets)
            ):
                raise ValueError("Full-window serial proposals require one complete fixed-gamma chain per row.")
            return next_windows.flatten(), next_windows, confidence
        verification_parts: list[torch.Tensor] = []
        for row, (sequence_index, size, prefix) in enumerate(zip(active_indices, actual_verification_sizes, prefixes)):
            if not 0 < size <= self.gamma:
                raise ValueError("SpecRhythm verification sizes must be in [1, gamma].")
            if prefix is None:
                prefix = torch.tensor(
                    states[sequence_index].token_ids[-(size - 1) :] if size > 1 else [],
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                prefix = prefix.reshape(-1).to(device=self.device, dtype=torch.long)
            if prefix.numel() != size - 1:
                raise ValueError("SpecRhythm eager verification prefix does not match its parent window.")
            verification_parts.append(torch.cat((prefix, next_windows[row, :1])))
        verification = torch.cat(verification_parts)
        return verification, next_windows, confidence

    def _draft_spec_rhythm_device_batch_with_prefill(
        self,
        states: list[PearlPipelineState],
        active_indices: list[int],
        draft_budgets: Sequence[int],
        prefill_indices: Sequence[int],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Fuse staged prompt rows into step zero of a fixed draft chain."""

        if not self.is_draft:
            return None, None, None
        budgets = [int(value) for value in draft_budgets]
        if (
            not active_indices
            or not prefill_indices
            or len(budgets) != len(active_indices)
            or any(value != self.gamma for value in budgets)
        ):
            raise ValueError("Mixed draft prefill requires non-empty complete fixed-gamma rows.")
        if self.gamma <= 1 or not self.config.draft_use_paged_attention:
            raise ValueError("Mixed draft prefill requires gamma>1 and draft paged attention.")
        if (
            self.config.enforce_eager
            or not hasattr(self, "graph_runner")
            or not hasattr(self.graph_runner, "run_draft_greedy")
        ):
            raise ValueError("Mixed draft prefill requires the qualified gamma-1 ACLGraph.")
        if len(set(prefill_indices)) != len(prefill_indices) or set(active_indices).intersection(prefill_indices):
            raise ValueError("Mixed draft prefill requires unique, disjoint sequence rows.")
        if any(states[index].draft_temperature != 0 for index in active_indices):
            raise ValueError("Mixed draft prefill currently supports greedy drafting only.")
        for index in prefill_indices:
            state = states[index]
            if state.committed_length != state.prompt_length or len(state.token_ids) != state.prompt_length:
                raise RuntimeError("Mixed draft prefill requires an unpublished prompt frontier.")

        root_inputs = torch.tensor(
            [states[index].token_ids[-1] for index in active_indices],
            dtype=torch.long,
            device=self.device,
        )
        root_positions = [len(states[index].token_ids) - 1 for index in active_indices]
        prefill_token_ids: list[int] = []
        prefill_sequence_ids: list[int] = []
        prefill_positions: list[int] = []
        assert self.cache_allocation is not None
        for index in prefill_indices:
            state = states[index]
            cached_tokens = int(self.cache_allocation.num_cached_tokens[index])
            if not 0 <= cached_tokens < state.prompt_length:
                raise RuntimeError("Mixed draft prefill needs at least one uncached prompt token.")
            uncached = state.token_ids[cached_tokens : state.prompt_length]
            prefill_token_ids.extend(int(token_id) for token_id in uncached)
            prefill_sequence_ids.extend([index] * len(uncached))
            prefill_positions.extend(range(cached_tokens, state.prompt_length))
        if len(active_indices) + len(prefill_token_ids) > self.config.max_num_batched_tokens:
            raise ValueError("Mixed draft first step exceeds max_num_batched_tokens.")

        mixed_inputs = torch.cat(
            (
                root_inputs,
                torch.tensor(
                    prefill_token_ids,
                    dtype=torch.long,
                    device=self.device,
                ),
            )
        )
        mixed_hidden = self._run_device_packed_hidden(
            mixed_inputs,
            [*active_indices, *prefill_sequence_ids],
            [*root_positions, *prefill_positions],
            use_aclgraph=False,
            use_fused_infer_attention=True,
        )
        first_tokens = self.model.compute_greedy_tokens(
            mixed_hidden[: len(active_indices)],
            self.draft_vocab_size,
        )

        graph_batch_size = _next_linear_draft_graph_bucket(
            len(active_indices),
            self.config.max_num_seqs,
        )
        proposal_tail_steps = self.gamma - 1
        tail_steps = proposal_tail_steps + int(
            getattr(
                self.config,
                "spec_rhythm_linear_bonus_token",
                False,
            )
        )
        position_tensors: list[torch.Tensor] = []
        attention_metadatas: list[Any] = []
        graph_input_ids: torch.Tensor | None = None
        for step in range(tail_steps):
            step_positions = [position + step + 1 for position in root_positions]
            position_tensor, attention_metadata = self._prepare_attention_metadata(
                active_indices,
                step_positions,
                use_fused_infer_attention=False,
            )
            (
                graph_input_ids,
                padded_positions,
                padded_metadata,
            ) = NativeACLGraphRunner._pad_inputs(
                first_tokens,
                position_tensor,
                attention_metadata,
                graph_batch_size,
            )
            position_tensors.append(padded_positions)
            attention_metadatas.append(padded_metadata)
        assert graph_input_ids is not None
        run_kwargs: dict[str, Any] = {
            "valid_row_count": len(active_indices),
        }
        if getattr(
            self.config,
            "spec_rhythm_linear_bonus_token",
            False,
        ):
            run_kwargs["final_kv_only"] = True
        tail_windows = self.graph_runner.run_draft_greedy(
            graph_input_ids,
            position_tensors,
            attention_metadatas,
            self.draft_vocab_size,
            **run_kwargs,
        )[: len(active_indices)].clone()
        if tail_windows.shape != (len(active_indices), proposal_tail_steps):
            raise RuntimeError("Mixed draft-tail ACLGraph returned an invalid proposal shape.")
        next_windows = torch.cat(
            (
                first_tokens.unsqueeze(1),
                tail_windows[:, :proposal_tail_steps],
            ),
            dim=1,
        )
        confidence = torch.ones(
            len(active_indices),
            dtype=torch.float32,
            device=self.device,
        )
        return next_windows.flatten(), next_windows, confidence

    def _validate_packed_causal_fia_leakage_once(
        self,
        query_tokens: torch.Tensor,
        sequence_ids: list[int],
        positions: list[int],
        *,
        query_width: int | None = None,
    ) -> None:
        """Probe packed FIA causality once without changing production output.

        Full-window verification queries ``[root, d1, ..., d_(gamma-1)]``;
        ``d_gamma`` is a comparison label and is not an input to the target
        forward.  The probe therefore changes only ``d_(gamma-1)`` (the last
        query row) in every request while preserving the exact tensor shape,
        query partition, positions, and block tables.  Earlier rows must be
        bitwise invariant because they cannot attend to that future token.

        The diagnostic deliberately bypasses ACLGraph.  A third eager forward
        with the original inputs restores every overwritten KV slot before the
        caller executes its normal production forward.  It is opt-in and the
        attempted flag is sticky so a caught failure cannot repeatedly add
        three target forwards to service traffic.
        """

        if not envs.VLLM_ASCEND_SPECRHYTHM_VALIDATE_PACKED_CAUSAL_LEAKAGE or getattr(
            self,
            "_packed_causal_leakage_probe_completed",
            False,
        ):
            return

        # Mark the attempt before touching the model: even when the diagnostic
        # raises and an outer harness catches it, this engine must not rerun it.
        self._packed_causal_leakage_probe_completed = True
        for attribute, default in (
            ("_packed_causal_leakage_probe_attempts", 0),
            ("_packed_causal_leakage_probe_passes", 0),
            ("_packed_causal_leakage_probe_failures", 0),
            ("_packed_causal_leakage_probe_hidden_mismatch_steps", 0),
            ("_packed_causal_leakage_probe_token_mismatch_steps", 0),
            (
                "_packed_causal_leakage_probe_restore_hidden_mismatch_steps",
                0,
            ),
            (
                "_packed_causal_leakage_probe_restore_token_mismatch_steps",
                0,
            ),
            ("_packed_causal_leakage_probe_max_abs_diff", 0.0),
        ):
            if not hasattr(self, attribute):
                setattr(self, attribute, default)
        self._packed_causal_leakage_probe_attempts = (
            getattr(
                self,
                "_packed_causal_leakage_probe_attempts",
                0,
            )
            + 1
        )
        width = self.gamma if query_width is None else int(query_width)
        if width < 2:
            self._packed_causal_leakage_probe_failures = (
                getattr(
                    self,
                    "_packed_causal_leakage_probe_failures",
                    0,
                )
                + 1
            )
            raise RuntimeError("Packed causal FIA leakage validation requires gamma >= 2.")
        if (
            query_tokens.ndim != 1
            or query_tokens.numel() == 0
            or query_tokens.numel() % width
            or len(sequence_ids) != query_tokens.numel()
            or len(positions) != query_tokens.numel()
        ):
            self._packed_causal_leakage_probe_failures = (
                getattr(
                    self,
                    "_packed_causal_leakage_probe_failures",
                    0,
                )
                + 1
            )
            raise RuntimeError("Packed causal FIA leakage validation received a malformed fixed-gamma query.")
        request_count = query_tokens.numel() // width
        request_ids = [sequence_ids[row * width] for row in range(request_count)]
        if len(set(request_ids)) != request_count or any(
            sequence_ids[row * width : (row + 1) * width] != [request_ids[row]] * width
            or positions[row * width : (row + 1) * width]
            != list(
                range(
                    positions[row * width],
                    positions[row * width] + width,
                )
            )
            for row in range(request_count)
        ):
            self._packed_causal_leakage_probe_failures = (
                getattr(
                    self,
                    "_packed_causal_leakage_probe_failures",
                    0,
                )
                + 1
            )
            raise RuntimeError(
                "Packed causal FIA leakage validation requires contiguous request-major query rows and positions."
            )
        vocabulary_size = int(self.draft_vocab_size)
        if vocabulary_size <= 1:
            self._packed_causal_leakage_probe_failures = (
                getattr(
                    self,
                    "_packed_causal_leakage_probe_failures",
                    0,
                )
                + 1
            )
            raise RuntimeError("Packed causal FIA leakage validation needs at least two valid token IDs.")

        original_tokens = query_tokens.clone()
        perturbed_tokens = original_tokens.clone()
        last_query_indices = torch.arange(
            width - 1,
            query_tokens.numel(),
            width,
            dtype=torch.long,
            device=query_tokens.device,
        )
        replacements = torch.remainder(
            perturbed_tokens.index_select(0, last_query_indices) + 1,
            vocabulary_size,
        )
        perturbed_tokens.index_copy_(0, last_query_indices, replacements)

        def run_eager_fia(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            hidden = self._run_device_packed_hidden(
                inputs,
                sequence_ids,
                positions,
                use_aclgraph=False,
                use_fused_infer_attention=True,
            ).clone()
            greedy = self.model.compute_greedy_tokens(
                hidden,
                vocabulary_size,
            ).clone()
            return hidden, greedy

        try:
            reference_hidden, reference_tokens = run_eager_fia(original_tokens)
            try:
                perturbed_hidden, perturbed_outputs = run_eager_fia(perturbed_tokens)
            finally:
                # This must run before any mismatch is raised: the perturbed
                # forward overwrote the last query's K/V in every request.
                restored_hidden, restored_tokens = run_eager_fia(original_tokens)
        except Exception:
            self._packed_causal_leakage_probe_failures = (
                getattr(
                    self,
                    "_packed_causal_leakage_probe_failures",
                    0,
                )
                + 1
            )
            raise

        if (
            reference_hidden.shape != perturbed_hidden.shape
            or reference_hidden.shape != restored_hidden.shape
            or reference_hidden.shape[0] != query_tokens.numel()
            or reference_tokens.shape != (query_tokens.numel(),)
            or perturbed_outputs.shape != reference_tokens.shape
            or restored_tokens.shape != reference_tokens.shape
        ):
            self._packed_causal_leakage_probe_failures = (
                getattr(
                    self,
                    "_packed_causal_leakage_probe_failures",
                    0,
                )
                + 1
            )
            raise RuntimeError(
                "Packed causal FIA leakage validation received unexpected target output shapes after restoring KV."
            )

        reference_rows = reference_hidden.reshape(
            request_count,
            width,
            -1,
        )
        perturbed_rows = perturbed_hidden.reshape(
            request_count,
            width,
            -1,
        )
        restored_rows = restored_hidden.reshape(
            request_count,
            width,
            -1,
        )
        reference_token_rows = reference_tokens.reshape(request_count, width)
        perturbed_token_rows = perturbed_outputs.reshape(request_count, width)
        restored_token_rows = restored_tokens.reshape(request_count, width)

        hidden_mismatch = (reference_rows[:, :-1] != perturbed_rows[:, :-1]).any(dim=-1)
        token_mismatch = reference_token_rows[:, :-1] != perturbed_token_rows[:, :-1]
        restore_hidden_mismatch = (reference_rows != restored_rows).any(dim=-1)
        restore_token_mismatch = reference_token_rows != restored_token_rows
        hidden_mismatch_steps = int(hidden_mismatch.sum().item())
        token_mismatch_steps = int(token_mismatch.sum().item())
        restore_hidden_mismatch_steps = int(restore_hidden_mismatch.sum().item())
        restore_token_mismatch_steps = int(restore_token_mismatch.sum().item())
        checked_difference = (reference_rows[:, :-1].float() - perturbed_rows[:, :-1].float()).abs()
        max_abs_diff = float(
            torch.nan_to_num(
                checked_difference,
                nan=float("inf"),
                posinf=float("inf"),
                neginf=float("inf"),
            )
            .max()
            .item()
        )
        self._packed_causal_leakage_probe_hidden_mismatch_steps = hidden_mismatch_steps
        self._packed_causal_leakage_probe_token_mismatch_steps = token_mismatch_steps
        self._packed_causal_leakage_probe_restore_hidden_mismatch_steps = restore_hidden_mismatch_steps
        self._packed_causal_leakage_probe_restore_token_mismatch_steps = restore_token_mismatch_steps
        self._packed_causal_leakage_probe_max_abs_diff = max_abs_diff

        if (
            hidden_mismatch_steps
            or token_mismatch_steps
            or restore_hidden_mismatch_steps
            or restore_token_mismatch_steps
        ):
            self._packed_causal_leakage_probe_failures = (
                getattr(
                    self,
                    "_packed_causal_leakage_probe_failures",
                    0,
                )
                + 1
            )
            raise RuntimeError(
                "Packed causal FIA leakage validation failed after restoring "
                "the original KV rows: "
                f"hidden_mismatch_steps={hidden_mismatch_steps}, "
                f"token_mismatch_steps={token_mismatch_steps}, "
                "restore_hidden_mismatch_steps="
                f"{restore_hidden_mismatch_steps}, "
                "restore_token_mismatch_steps="
                f"{restore_token_mismatch_steps}, "
                f"max_abs_diff={max_abs_diff}."
            )
        self._packed_causal_leakage_probe_passes = (
            getattr(
                self,
                "_packed_causal_leakage_probe_passes",
                0,
            )
            + 1
        )
        logger.info(
            "Packed causal FIA leakage validation passed on rank %d for %d requests and gamma=%d.",
            self.rank,
            request_count,
            self.gamma,
        )

    def _target_full_window_stable_fia_graph(
        self,
        query_tokens: torch.Tensor,
        active_indices: Sequence[int],
        first_positions: Sequence[int],
        *,
        query_width: int,
    ) -> torch.Tensor | None:
        """Run an exact resident stable graph for decode-only verification.

        Qualification provisions one exact graph for every request count from
        one through 32.  Runtime therefore keeps Q at
        ``request_count * query_width`` with neither dummy-row compute nor
        graph discovery.  The entries share the runner's causal-FIA workspace
        pool with the mixed graph family.

        ``None`` means the bounded envelope cannot represent this call and the
        caller must retain the existing exact-shape FIA path.  A graph-runner
        eager fallback still returns padded hidden states directly; telemetry
        records that fallback without executing the target model twice.
        """

        self._last_stable_target_verify_graph_outcome = ("disabled", 0)
        request_count = len(active_indices)
        if not (
            getattr(self, "_mixed_target_graph_enabled", False)
            and not self.config.enforce_eager
            and envs.VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH
        ):
            return None
        try:
            _next_stable_target_verify_capacity(request_count)
        except ValueError:
            self._last_stable_target_verify_graph_outcome = (
                "bounded_shape",
                request_count,
            )
            return None
        capacity = getattr(
            self,
            "_stable_target_verify_graph_capacity_map",
            {},
        ).get(request_count)
        if capacity != request_count:
            raise RuntimeError(
                "Stable target-verify exact graph was not qualified before "
                f"service: requests={request_count}, capacity={capacity}."
            )
        layout = _cached_stable_target_verify_graph_layout(
            request_count,
            query_width,
            capacity,
        )
        if (
            layout.total_query_tokens > self.graph_runner.max_graph_tokens
            or layout.dummy_verification_rows * query_width > self.config.max_model_len
        ):
            self._last_stable_target_verify_graph_outcome = (
                (
                    "token_capacity"
                    if layout.total_query_tokens > self.graph_runner.max_graph_tokens
                    else "bounded_shape"
                ),
                request_count,
            )
            return None
        expected_real_tokens = request_count * query_width
        if tuple(query_tokens.shape) != (expected_real_tokens,) or len(first_positions) != request_count:
            raise ValueError("Stable target-verify inputs do not match their fixed window.")
        envelope = plan_stable_target_verify_graph_envelope(
            layout,
            active_indices,
            first_positions,
            self._mixed_target_scratch_sequence_ids[0],
        )
        # Every qualified service entry is exact-Q (capacity == request count),
        # so query_tokens already has the captured shape.  Do not allocate an
        # empty padding tensor and copy the complete input through a second cat.
        if layout.total_query_tokens != expected_real_tokens:
            raise RuntimeError("Stable target-verify exact graph unexpectedly requires padding.")
        graph_inputs = query_tokens
        if getattr(self.graph_runner, "graph_cache_sealed", False) is True:
            if self.cache_block_tables is None:
                raise RuntimeError("Sealed stable target verification requires a live cache block table.")
            # Qualification intentionally retains the ordinary materialized
            # path.  In sealed service, update the resident graph buffers
            # directly so positions/slots/request tables are not first built
            # as temporary NPU tensors and then copied by the graph runner.
            token_sequence_ids = list(envelope.token_sequence_ids)
            graph_position_values = list(envelope.positions)
            self._ensure_cache_capacity(
                token_sequence_ids,
                graph_position_values,
            )
            slot_mapping = self._cache_slot_mapping(
                token_sequence_ids,
                graph_position_values,
            )
            target_tokens = self.graph_runner.run_stable_fia_greedy_staged(
                graph_inputs,
                self.draft_vocab_size,
                positions=envelope.positions,
                slot_mapping=slot_mapping,
                actual_seq_lengths_q=(
                    _stable_target_verify_cumulative_q(
                        layout.query_width,
                        layout.verification_capacity,
                    )
                ),
                sequence_lens=envelope.sequence_lens,
                segment_sequence_ids=envelope.segment_sequence_ids,
                cache_block_tables=self.cache_block_tables,
                attention_mask=self.model.attention_mask,
                graph_key=layout.graph_key,
                expected_tokens=layout.total_query_tokens,
                expected_request_segments=layout.request_segment_count,
            )
        else:
            graph_positions, graph_metadata = self._prepare_stable_target_verify_graph_call(
                layout,
                envelope,
                graph_inputs,
            )
            target_tokens = self.graph_runner.run_stable_fia_greedy(
                graph_inputs,
                graph_positions,
                graph_metadata,
                self.draft_vocab_size,
                graph_key=layout.graph_key,
                expected_tokens=layout.total_query_tokens,
                expected_request_segments=layout.request_segment_count,
            )
        execution = getattr(self.graph_runner, "last_generic_execution", None)
        used_aclgraph = bool(execution is None or getattr(execution, "used_aclgraph", True))
        self._last_stable_target_verify_graph_outcome = (
            "graph" if used_aclgraph else "execution_fallback",
            request_count,
        )
        return target_tokens[: layout.verification_output_count]

    def _target_full_window_outputs_batch(
        self,
        states: list[PearlPipelineState],
        active_indices: list[int],
        payloads: Sequence[NativeSpecRhythmDevicePayload],
        *,
        proposal_matrix: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, None]:
        """Verify complete serial proposals from the committed target frontier.

        For proposal ``[d1, ..., d_gamma]`` the target normally queries
        ``[root, d1, ..., d_(gamma-1)]`` and compares the resulting greedy
        tokens with the complete proposal.  Bonus-token mode also queries
        ``d_gamma``; the extra target result can be committed only after a
        complete match. Proposal tensors stay resident on the target NPU;
        only the committed roots originate from host state.
        """

        self._last_stable_target_verify_graph_outcome = ("disabled", 0)
        if self.is_draft:
            return None, None
        if len(active_indices) != len(payloads):
            raise ValueError("Full-window target payloads must be request-aligned.")
        if not active_indices:
            return torch.empty(0, dtype=torch.long, device=self.device), None
        if any(states[index].temperature != 0 for index in active_indices):
            raise ValueError("Full-window target verification currently supports greedy sampling only.")
        if any(
            payload.ticket.request_index != request_index
            or payload.verification_size != self.gamma
            or payload.ticket.gamma != self.gamma
            or payload.verification_tokens.shape != (self.gamma,)
            for request_index, payload in zip(active_indices, payloads)
        ):
            raise ValueError("Full-window target verification requires aligned complete fixed-gamma proposals.")
        for index in active_indices:
            state = states[index]
            if state.committed_length != len(state.token_ids):
                raise RuntimeError("Full-window target state contains an uncommitted token suffix.")

        proposals = (
            torch.stack([payload.verification_tokens for payload in payloads])
            if proposal_matrix is None
            else proposal_matrix
        )
        if (
            proposals.dtype != torch.long
            or proposals.device.type != self.device.type
            or (self.device.index is not None and proposals.device.index != self.device.index)
            or proposals.shape != (len(active_indices), self.gamma)
        ):
            raise ValueError("Full-window target proposal matrix has an invalid device, dtype, or shape.")
        roots = torch.tensor(
            [states[index].token_ids[-1] for index in active_indices],
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)
        bonus_mode = bool(getattr(self.config, "spec_rhythm_linear_bonus_token", False))
        query_width = self.gamma + int(bonus_mode)
        first_positions = [int(states[index].committed_length) - 1 for index in active_indices]
        sequence_ids = [index for index in active_indices for _ in range(query_width)]
        positions = [first_position + step for first_position in first_positions for step in range(query_width)]
        use_aclgraph = (
            not self.config.enforce_eager
            and self.gamma <= 16
            and not envs.VLLM_ASCEND_SPECRHYTHM_DISABLE_TARGET_ACLGRAPH
        )
        target_use_fia = not self.config.target_use_paged_attention

        force_stepwise_target = envs.VLLM_ASCEND_SPECRHYTHM_FORCE_STEPWISE_TARGET
        # CANN's decode-oriented paged-attention ABI does not define the
        # causal dependency between several query rows belonging to the same
        # request.  A fixed-gamma proposal therefore has to execute as gamma
        # ordered one-token model steps.  This is not merely a large-shape
        # guard: a same-source static Qwen3 TP3 differential changes greedy
        # output rows even at B=40/gamma=4 when the queries are packed.
        #
        # In graph mode, capture the complete ordered chain as *one* target
        # ACLGraph.  The four model calls and their KV writes remain causal on
        # the captured stream, while service pays a single Python replay.  Use
        # exact row shapes here: padding can change TP GEMM/reduction numerics
        # and requires an independent hardware qualification.  The explicit
        # PACKED_TARGET switch remains a diagnostic-only backend probe.
        paged_attention_requires_causal_chain = not target_use_fia and not envs.VLLM_ASCEND_SPECRHYTHM_PACKED_TARGET
        if force_stepwise_target or paged_attention_requires_causal_chain:
            step_inputs = [roots[:, 0] if step == 0 else proposals[:, step - 1] for step in range(query_width)]
            step_positions = [
                [first_position + step for first_position in first_positions] for step in range(query_width)
            ]
            if use_aclgraph:
                position_tensors: list[torch.Tensor] = []
                attention_metadatas: list[Any] = []
                for positions in step_positions:
                    position_tensor, attention_metadata = self._prepare_attention_metadata(
                        active_indices,
                        positions,
                        use_fused_infer_attention=(target_use_fia and self.gamma <= 16),
                    )
                    position_tensors.append(position_tensor)
                    attention_metadatas.append(attention_metadata)
                graph_outputs = self.graph_runner.run_target_greedy(
                    step_inputs,
                    position_tensors,
                    attention_metadatas,
                    self.draft_vocab_size,
                )
                return torch.stack(graph_outputs, dim=1).flatten(), None

            step_outputs = [
                self._run_device_packed_greedy(
                    step_input,
                    active_indices,
                    positions,
                    use_aclgraph=False,
                    use_fused_infer_attention=(target_use_fia and self.gamma <= 16),
                ).clone()
                for step_input, positions in zip(step_inputs, step_positions)
            ]
            return torch.stack(step_outputs, dim=1).flatten(), None

        proposal_inputs = proposals if bonus_mode else proposals[:, :-1]
        query_tokens = torch.cat((roots, proposal_inputs), dim=1).flatten()
        if target_use_fia:
            self._validate_packed_causal_fia_leakage_once(
                query_tokens,
                sequence_ids,
                positions,
                query_width=query_width,
            )
            stable_output = self._target_full_window_stable_fia_graph(
                query_tokens,
                active_indices,
                first_positions,
                query_width=query_width,
            )
            if stable_output is not None:
                return stable_output, None
        return (
            self._run_device_packed_greedy(
                query_tokens,
                sequence_ids,
                positions,
                use_aclgraph=use_aclgraph,
                use_fused_infer_attention=target_use_fia and self.gamma <= 16,
            ),
            None,
        )

    def _prepare_stable_target_verify_graph_call(
        self,
        layout: StableTargetVerifyGraphLayout,
        envelope: StableTargetVerifyGraphEnvelope,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, NativeAttentionMetadata]:
        """Materialize a fixed verification-only FIA envelope."""

        if not self._mixed_target_graph_enabled or self.is_draft:
            raise RuntimeError("Stable target-verify graph preparation is target-only and opt-in.")
        if (
            self.cache_allocation is None
            or self.cache_block_tables is None
            or tuple(envelope.layout.query_lengths) != tuple(layout.query_lengths)
            or tuple(input_ids.shape) != (layout.total_query_tokens,)
            or input_ids.dtype != torch.long
            or input_ids.device.type != self.device.type
        ):
            raise ValueError("Stable target-verify graph input/cache state does not match its layout.")
        if any(position >= self.config.max_model_len for position in envelope.positions):
            raise ValueError("Stable target-verify scratch or real position exceeds max_model_len.")
        self._ensure_cache_capacity(
            list(envelope.token_sequence_ids),
            list(envelope.positions),
        )
        slot_mapping = self._cache_slot_mapping(
            list(envelope.token_sequence_ids),
            list(envelope.positions),
        )
        segment_ids = torch.tensor(
            envelope.segment_sequence_ids,
            dtype=torch.long,
            device=self.device,
        )
        request_tables = self.cache_block_tables.index_select(0, segment_ids)
        cumulative_q = _stable_target_verify_cumulative_q(
            layout.query_width,
            layout.verification_capacity,
        )
        metadata = NativeAttentionMetadata(
            slot_mapping=torch.tensor(
                slot_mapping,
                dtype=torch.int32,
                device=self.device,
            ),
            context_lens=torch.tensor(
                [position + 1 for position in envelope.positions],
                dtype=torch.int32,
            ),
            block_tables=request_tables,
            actual_seq_lengths_q=cumulative_q,
            sequence_lens=envelope.sequence_lens,
            request_block_tables=request_tables,
            attention_mask=self.model.attention_mask,
            use_fused_infer_attention=True,
        )
        return (
            torch.tensor(
                envelope.positions,
                dtype=torch.long,
                device=self.device,
            ),
            metadata,
        )

    def _prepare_mixed_target_graph_call(
        self,
        layout: MixedTargetGraphLayout,
        envelope: MixedTargetGraphEnvelope,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, NativeAttentionMetadata]:
        """Materialize a validated fixed envelope on dedicated target KV rows."""

        if not self._mixed_target_graph_enabled or self.is_draft:
            raise RuntimeError("Mixed-target graph preparation is target-only and opt-in.")
        if (
            self.cache_allocation is None
            or self.cache_block_tables is None
            or tuple(envelope.layout.query_lengths) != tuple(layout.query_lengths)
            or tuple(input_ids.shape) != (layout.total_query_tokens,)
            or input_ids.dtype != torch.long
            or input_ids.device.type != self.device.type
        ):
            raise ValueError("Mixed-target graph input/cache state does not match its layout.")
        if any(position >= self.config.max_model_len for position in envelope.positions):
            raise ValueError("Mixed-target graph scratch or real position exceeds max_model_len.")
        self._ensure_cache_capacity(
            list(envelope.token_sequence_ids),
            list(envelope.positions),
        )
        slot_mapping = self._cache_slot_mapping(
            list(envelope.token_sequence_ids),
            list(envelope.positions),
        )
        segment_ids = torch.tensor(
            envelope.segment_sequence_ids,
            dtype=torch.long,
            device=self.device,
        )
        request_tables = self.cache_block_tables.index_select(0, segment_ids)
        cumulative_q: list[int] = []
        for length in layout.query_lengths:
            cumulative_q.append(length + (cumulative_q[-1] if cumulative_q else 0))
        metadata = NativeAttentionMetadata(
            slot_mapping=torch.tensor(
                slot_mapping,
                dtype=torch.int32,
                device=self.device,
            ),
            context_lens=torch.tensor(
                [position + 1 for position in envelope.positions],
                dtype=torch.int32,
            ),
            # FIA consumes request-major page tables. Keeping the compatibility
            # field aliased avoids a second device allocation/index_select.
            block_tables=request_tables,
            actual_seq_lengths_q=tuple(cumulative_q),
            sequence_lens=envelope.sequence_lens,
            request_block_tables=request_tables,
            attention_mask=self.model.attention_mask,
            use_fused_infer_attention=True,
        )
        return (
            torch.tensor(
                envelope.positions,
                dtype=torch.long,
                device=self.device,
            ),
            metadata,
        )

    def _target_full_window_outputs_with_prefill_batch(
        self,
        states: list[PearlPipelineState],
        active_indices: list[int],
        payloads: Sequence[NativeSpecRhythmDevicePayload],
        prefill_indices: Sequence[int],
        *,
        proposal_matrix: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, None, torch.Tensor | None]:
        """Fuse fixed-window verification and staged target prefill.

        Both row families use disjoint sequence IDs and one packed causal FIA
        call on the target model.  This removes a second weight sweep and TP
        collective sequence on refill cycles without introducing another NPU
        stream or changing the existing cross-model communication ordering.
        The dynamic mixed shape intentionally runs eager; ordinary decode-only
        cycles continue to use the qualified target ACLGraph.
        """

        self._last_mixed_target_graph_outcome = ("disabled", 0, 0)
        if self.is_draft:
            return None, None, None
        if self.config.target_use_paged_attention:
            raise ValueError("Mixed target prefill requires packed causal FIA.")
        if not active_indices or not prefill_indices:
            raise ValueError("Mixed target prefill requires both verification and prefill rows.")
        if len(active_indices) != len(payloads):
            raise ValueError("Full-window target payloads must be request-aligned.")
        if len(set(prefill_indices)) != len(prefill_indices) or set(active_indices).intersection(prefill_indices):
            raise ValueError("Mixed target prefill requires unique, disjoint sequence rows.")
        if any(states[index].temperature != 0 for index in active_indices):
            raise ValueError("Full-window target verification currently supports greedy sampling only.")
        if any(
            states[index].temperature != 0 or states[index].top_p < 1.0 or states[index].top_k > 0
            for index in prefill_indices
        ):
            raise ValueError("Mixed target prefill currently supports greedy first-token sampling only.")
        if any(
            payload.ticket.request_index != request_index
            or payload.verification_size != self.gamma
            or payload.ticket.gamma != self.gamma
            or payload.verification_tokens.shape != (self.gamma,)
            for request_index, payload in zip(active_indices, payloads)
        ):
            raise ValueError("Full-window target verification requires aligned complete fixed-gamma proposals.")
        for index in active_indices:
            state = states[index]
            if state.committed_length != len(state.token_ids):
                raise RuntimeError("Full-window target state contains an uncommitted token suffix.")
        for index in prefill_indices:
            state = states[index]
            if state.committed_length != state.prompt_length or len(state.token_ids) != state.prompt_length:
                raise RuntimeError("Mixed target prefill requires an unpublished prompt frontier.")

        proposals = (
            torch.stack([payload.verification_tokens for payload in payloads])
            if proposal_matrix is None
            else proposal_matrix
        )
        if (
            proposals.dtype != torch.long
            or proposals.device.type != self.device.type
            or proposals.shape != (len(active_indices), self.gamma)
        ):
            raise ValueError("Full-window target proposal matrix has an invalid device, dtype, or shape.")
        roots = torch.tensor(
            [states[index].token_ids[-1] for index in active_indices],
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)
        proposal_inputs = (
            proposals if getattr(self.config, "spec_rhythm_linear_bonus_token", False) else proposals[:, :-1]
        )
        verification_inputs = torch.cat(
            (roots, proposal_inputs),
            dim=1,
        ).flatten()
        query_width = self.gamma + int(getattr(self.config, "spec_rhythm_linear_bonus_token", False))
        verification_positions = [
            int(states[index].committed_length) - 1 + step for index in active_indices for step in range(query_width)
        ]
        verification_sequence_ids = [index for index in active_indices for _ in range(query_width)]

        prefill_token_ids: list[int] = []
        prefill_sequence_ids: list[int] = []
        prefill_positions: list[int] = []
        prefill_last_hidden_indices: list[int] = []
        verification_token_count = int(verification_inputs.numel())
        assert self.cache_allocation is not None
        for index in prefill_indices:
            state = states[index]
            cached_tokens = int(self.cache_allocation.num_cached_tokens[index])
            if not 0 <= cached_tokens < state.prompt_length:
                raise RuntimeError("Mixed target prefill needs at least one uncached prompt token.")
            uncached = state.token_ids[cached_tokens : state.prompt_length]
            prefill_token_ids.extend(int(token_id) for token_id in uncached)
            prefill_sequence_ids.extend([index] * len(uncached))
            prefill_positions.extend(range(cached_tokens, state.prompt_length))
            prefill_last_hidden_indices.append(verification_token_count + len(prefill_token_ids) - 1)

        mixed_token_count = verification_token_count + len(prefill_token_ids)
        if mixed_token_count > self.config.max_num_batched_tokens:
            raise ValueError("Mixed target verification and prefill exceed max_num_batched_tokens.")
        prefill_inputs = torch.tensor(
            prefill_token_ids,
            dtype=torch.long,
            device=self.device,
        )
        layout: MixedTargetGraphLayout | None = None
        graph_fallback_reason: str | None = None
        if (
            getattr(self, "_mixed_target_graph_enabled", False)
            and not self.config.enforce_eager
            and envs.VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH
        ):
            try:
                layout = plan_mixed_target_graph_layout(
                    len(active_indices),
                    [
                        state.prompt_length - int(self.cache_allocation.num_cached_tokens[index])
                        for index, state in ((index, states[index]) for index in prefill_indices)
                    ],
                    gamma=query_width,
                    verification_capacity=(_next_mixed_target_verify_capacity(len(active_indices))),
                    prompt_token_buckets=getattr(
                        self,
                        "_mixed_target_graph_prompt_buckets",
                        MIXED_TARGET_PROMPT_TOKEN_BUCKETS,
                    ),
                )
            except ValueError:
                # Shapes outside the bounded stable-envelope contract retain
                # the already-qualified eager mixed-target pass.
                layout = None
                graph_fallback_reason = "bounded_shape"
            if layout is not None and (
                layout.total_query_tokens > self.graph_runner.max_graph_tokens
                or max(
                    layout.dummy_verification_rows * query_width,
                    layout.query_lengths[-1] + 1,
                )
                > self.config.max_model_len
            ):
                # Do not execute a padded envelope through the graph runner's
                # eager token-capacity fallback; it would add dummy work with
                # no chance of replay. Capacity is an explicit runner knob;
                # an unusually short max_model_len likewise cannot safely own
                # the scratch positions required by this bucket.
                graph_fallback_reason = (
                    "token_capacity"
                    if layout.total_query_tokens > self.graph_runner.max_graph_tokens
                    else "bounded_shape"
                )
                layout = None

        if layout is None:
            # Only the eager fallback consumes this compact, unpadded tensor.
            # The stable graph branch below assembles its fixed envelope
            # directly and must not pay for this otherwise-dead device cat.
            mixed_inputs = torch.cat((verification_inputs, prefill_inputs))
            mixed_hidden = self._run_device_packed_hidden(
                mixed_inputs,
                [*verification_sequence_ids, *prefill_sequence_ids],
                [*verification_positions, *prefill_positions],
                use_aclgraph=False,
                use_fused_infer_attention=True,
            )
            output_indices = torch.tensor(
                [
                    *range(verification_token_count),
                    *prefill_last_hidden_indices,
                ],
                dtype=torch.long,
                device=self.device,
            )
            if graph_fallback_reason is not None:
                self._last_mixed_target_graph_outcome = (
                    graph_fallback_reason,
                    len(active_indices) + len(prefill_indices),
                    len(prefill_token_ids),
                )
        else:
            envelope = plan_mixed_target_graph_envelope(
                layout,
                active_indices,
                [int(states[index].committed_length) - 1 for index in active_indices],
                prefill_indices,
                [int(self.cache_allocation.num_cached_tokens[index]) for index in prefill_indices],
                self._mixed_target_scratch_sequence_ids,
            )
            dummy_verification_tokens = layout.dummy_verification_rows * query_width
            prompt_padding_tokens = layout.prompt_token_bucket + layout.prompt_capacity + 1 - len(prefill_token_ids)
            graph_inputs = torch.cat(
                (
                    verification_inputs,
                    torch.zeros(
                        dummy_verification_tokens,
                        dtype=torch.long,
                        device=self.device,
                    ),
                    prefill_inputs,
                    torch.zeros(
                        prompt_padding_tokens,
                        dtype=torch.long,
                        device=self.device,
                    ),
                )
            )
            graph_positions, graph_metadata = self._prepare_mixed_target_graph_call(
                layout,
                envelope,
                graph_inputs,
            )
            mixed_hidden = self.graph_runner.run_stable_fia_hidden(
                graph_inputs,
                graph_positions,
                graph_metadata,
                graph_key=layout.graph_key,
                expected_tokens=layout.total_query_tokens,
                expected_request_segments=layout.request_segment_count,
            )
            execution = getattr(
                self.graph_runner,
                "last_generic_execution",
                None,
            )
            used_aclgraph = bool(execution is None or getattr(execution, "used_aclgraph", True))
            self._last_mixed_target_graph_outcome = (
                "graph" if used_aclgraph else "execution_fallback",
                len(active_indices) + len(prefill_indices),
                len(prefill_token_ids),
            )
            output_indices = torch.tensor(
                [
                    *range(layout.verification_output_count),
                    *layout.prompt_output_indices,
                ],
                dtype=torch.long,
                device=self.device,
            )
        output_hidden = mixed_hidden.index_select(0, output_indices)
        output_tokens = self.model.compute_greedy_tokens(
            output_hidden,
            self.draft_vocab_size,
        )
        return (
            output_tokens[:verification_token_count],
            None,
            output_tokens[verification_token_count:],
        )

    def _exchange_spec_rhythm_device_proposals(
        self,
        verification_window: torch.Tensor | None,
        next_windows: torch.Tensor | None,
        confidences: torch.Tensor | None,
        tickets: Sequence[SpecRhythmProposalTicket],
        verification_sizes: Sequence[int],
        *,
        draft_compute_ms: float | None = None,
    ) -> tuple[
        torch.Tensor | None,
        Sequence[float | torch.Tensor],
        torch.Tensor | None,
    ]:
        """Broadcast a self-describing variable-offset proposal envelope."""

        count = len(tickets)
        if count == 0:
            return None, [], None
        if len(verification_sizes) != count:
            raise RuntimeError("SpecRhythm proposal metadata has inconsistent row counts.")
        include_draft_timing = draft_compute_ms is not None
        if include_draft_timing and (not math.isfinite(draft_compute_ms) or draft_compute_ms < 0.0):
            raise ValueError("SpecRhythm draft timing must be finite and non-negative.")
        metadata_width = 7
        verification_size = sum(verification_sizes)
        message_size = metadata_width * count + verification_size + count * self.gamma + int(include_draft_timing)
        if self.is_draft:
            if confidences is None or confidences.shape != (count,):
                raise RuntimeError("SpecRhythm draft confidence tensor has an invalid shape.")
            quantized_confidences = torch.round(confidences * 1_000_000).to(torch.long)
        else:
            quantized_confidences = None

        if self.groups.is_verification_worker:
            if self.rank == self.topology.draft_leader_rank:
                if verification_window is None or verification_window.shape != (verification_size,):
                    raise RuntimeError("SpecRhythm verification payload has an invalid shape.")
                if next_windows is None or next_windows.shape != (count, self.gamma):
                    raise RuntimeError("SpecRhythm continuation payload has an invalid shape.")
                metadata = torch.tensor(
                    [
                        value
                        for ticket, size in zip(tickets, verification_sizes)
                        for value in (
                            ticket.proposal_id,
                            ticket.request_index,
                            ticket.home_batch_id,
                            size,
                            ticket.gamma,
                            ticket.required_prefix_epoch,
                            0,
                        )
                    ],
                    dtype=torch.long,
                    device=self.device,
                )
                metadata = metadata.reshape(count, metadata_width)
                metadata[:, 6] = quantized_confidences
                message_parts = [metadata.reshape(-1), verification_window, next_windows.flatten()]
                if include_draft_timing:
                    message_parts.append(
                        torch.tensor(
                            [round(draft_compute_ms * 1000.0)],
                            dtype=torch.long,
                            device=self.device,
                        )
                    )
                message = torch.cat(message_parts)
            else:
                message = torch.empty(message_size, dtype=torch.long, device=self.device)
            dist.broadcast(
                message,
                src=self.topology.draft_leader_rank,
                group=self.groups.verification_group,
            )
            metadata = message[: metadata_width * count].reshape(count, metadata_width)
            if envs.VLLM_ASCEND_SPECRHYTHM_VALIDATE_MAILBOX:
                metadata_values = metadata.cpu().tolist()
                expected = [
                    [
                        ticket.proposal_id,
                        ticket.request_index,
                        ticket.home_batch_id,
                        int(size),
                        ticket.gamma,
                        ticket.required_prefix_epoch,
                    ]
                    for ticket, size in zip(tickets, verification_sizes)
                ]
                if [row[:6] for row in metadata_values] != expected:
                    raise RuntimeError("SpecRhythm received a misrouted or stale HCCL mailbox envelope.")
            if self.is_draft:
                confidence_values: Sequence[float | torch.Tensor] = confidences
            else:
                confidence_values = metadata[:, 6]
        else:
            message = None
            # Draft tensor-parallel followers are not members of the
            # verification group, but they still build the same rank-local
            # proposal tensors as the draft leader.  Preserve their local
            # confidences so materialization creates one guarded payload per
            # ticket for the following service cycle.
            confidence_values = confidences if self.is_draft else []
        timing_value = message[-1] if include_draft_timing and message is not None else None
        return message, confidence_values, timing_value

    def _materialize_spec_rhythm_payloads(
        self,
        *,
        tickets: Sequence[SpecRhythmProposalTicket],
        verification_sizes: Sequence[int],
        local_verification: torch.Tensor | None,
        local_next_windows: torch.Tensor | None,
        exchanged_message: torch.Tensor | None,
        draft_confidences: Sequence[float | torch.Tensor],
    ) -> list[NativeSpecRhythmDevicePayload]:
        count = len(tickets)
        if len(verification_sizes) != count or len(draft_confidences) != count:
            raise RuntimeError("SpecRhythm payload rows do not match proposal tickets.")
        metadata_size = count * 7
        verification_size = sum(verification_sizes)
        if self.is_draft:
            if local_verification is None or local_next_windows is None:
                raise RuntimeError("SpecRhythm draft worker lost a local proposal payload.")
            verification = local_verification
            continuations = local_next_windows
        else:
            if exchanged_message is None:
                raise RuntimeError("SpecRhythm target worker did not receive a proposal payload.")
            verification = exchanged_message[metadata_size : metadata_size + verification_size]
            continuation_start = metadata_size + verification_size
            continuations = exchanged_message[continuation_start : continuation_start + count * self.gamma].reshape(
                count, self.gamma
            )
        payloads: list[NativeSpecRhythmDevicePayload] = []
        offset = 0
        for row, (ticket, size, confidence) in enumerate(zip(tickets, verification_sizes, draft_confidences)):
            payloads.append(
                NativeSpecRhythmDevicePayload(
                    ticket=ticket,
                    verification_tokens=verification[offset : offset + size],
                    next_tokens=continuations[row, : ticket.gamma],
                    verification_size=int(size),
                    draft_confidence=(confidence.reshape(()) if torch.is_tensor(confidence) else confidence),
                )
            )
            offset += size
        return payloads

    def _target_round_outputs_batch(
        self,
        states: list[PearlPipelineState],
        active_indices: list[int],
        verification_sizes: Sequence[int] | None = None,
        pad_to_batch_size: int | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.is_draft:
            return None, None
        input_token_ids: list[int] = []
        sequence_ids: list[int] = []
        positions: list[int] = []
        actual_verification_sizes = (
            [
                1 if states[index].pre_verify else (states[index].pending_window_size or self.gamma)
                for index in active_indices
            ]
            if verification_sizes is None
            else [int(value) for value in verification_sizes]
        )
        if len(actual_verification_sizes) != len(active_indices) or any(
            value <= 0 or value > self.gamma for value in actual_verification_sizes
        ):
            raise ValueError("PEARL target verification sizes must be in [1, gamma].")
        # CANN's paged-attention ABI is decode-oriented: a packed query with
        # several tokens for one sequence can read a stale KV row for one of
        # the later tokens. Expand greedy verification into one-token model
        # calls so each K/V write is visible before the next query. The graph
        # runner still applies to these fixed one-token shapes; only the
        # unsupported multi-token PA contract is avoided. FIA and stochastic
        # verification retain their existing packed implementations.
        force_stepwise_target = envs.VLLM_ASCEND_SPECRHYTHM_FORCE_STEPWISE_TARGET
        force_stepwise_fia = envs.VLLM_ASCEND_SPECRHYTHM_STEPWISE_TARGET_FIA
        # The installed CANN PA kernel is replay-stable for the bounded B<=64
        # decode buckets used by the paper's normal service regime.  Larger
        # packed queries have shown stale-KV rows on 910B2, so retain the
        # causal one-token fallback there.  An explicit force switch is kept
        # for correctness probes and for new runtime versions.
        stepwise_safe_guard = (
            len(active_indices) > 64 or max(actual_verification_sizes, default=1) > 8 or force_stepwise_target
        )
        if (
            (self.config.target_use_paged_attention or force_stepwise_fia)
            and all(states[index].temperature == 0 for index in active_indices)
            and any(value > 1 for value in actual_verification_sizes)
            and not envs.VLLM_ASCEND_SPECRHYTHM_PACKED_TARGET
            and stepwise_safe_guard
        ):
            return self._target_round_outputs_stepwise(
                states,
                active_indices,
                actual_verification_sizes,
            )
        real_request_count = len(active_indices)
        padded_indices = list(active_indices)
        padded_verification_sizes = list(actual_verification_sizes)
        if pad_to_batch_size is not None:
            if pad_to_batch_size < real_request_count:
                raise ValueError("PEARL target graph padding cannot shrink a request batch.")
            active_set = set(active_indices)
            padding_indices = [index for index in range(len(states)) if index not in active_set][
                : pad_to_batch_size - real_request_count
            ]
            if len(padding_indices) != pad_to_batch_size - real_request_count:
                raise RuntimeError("PEARL target graph padding ran out of resident cache rows.")
            padded_indices.extend(padding_indices)
            # A padding row is a single-token query whose output is discarded.
            # It rewrites only its current KV slot and never advances the
            # corresponding logical request state.
            padded_verification_sizes.extend([1] * len(padding_indices))
        model_widths, valid_token_indices = _bucket_variable_target_verification_widths(
            padded_verification_sizes,
            self.gamma,
            self.config.target_verification_graph_buckets,
            getattr(self.config, "target_verification_graph_post_counts", ()),
        )
        for sequence_index, valid_width, model_width in zip(padded_indices, padded_verification_sizes, model_widths):
            state = states[sequence_index]
            start_position = len(state.token_ids) - valid_width
            tokens = state.token_ids[-valid_width:]
            if model_width > valid_width:
                tokens = [*tokens, *([tokens[-1]] * (model_width - valid_width))]
            input_token_ids.extend(tokens)
            sequence_ids.extend([sequence_index] * model_width)
            positions.extend(range(start_position, start_position + model_width))
        temperatures = [
            states[index].temperature
            for index, model_width in zip(padded_indices, model_widths)
            for _ in range(model_width)
        ]
        # Speculative execution uses FIA graphs; the decode-only paged-attention
        # graph can be selected for target verification on model/topology pairs
        # whose dynamic FIA graph-task updates are not replay-stable.
        use_aclgraph = (
            not self.config.enforce_eager
            and self.gamma <= 16
            and not envs.VLLM_ASCEND_SPECRHYTHM_DISABLE_TARGET_ACLGRAPH
        )
        target_use_fia = not self.config.target_use_paged_attention
        if getattr(self.config, "enable_spec_rhythm", False) and getattr(
            self.config, "spec_rhythm_stable_graphs", True
        ):
            target_use_fia = False
        if all(temperature == 0 for temperature in temperatures):
            target_tokens = self._run_packed_greedy(
                input_token_ids,
                sequence_ids,
                positions,
                use_aclgraph=use_aclgraph,
                use_fused_infer_attention=target_use_fia and self.gamma <= 16,
            )
            if len(padded_indices) != real_request_count or len(valid_token_indices) != len(input_token_ids):
                target_tokens = target_tokens.index_select(
                    0,
                    torch.tensor(
                        valid_token_indices[: sum(actual_verification_sizes)],
                        dtype=torch.long,
                        device=self.device,
                    ),
                )
            return target_tokens, None
        logits = self._run_packed_model(
            input_token_ids,
            sequence_ids,
            positions,
            use_aclgraph=use_aclgraph,
            use_fused_infer_attention=target_use_fia and self.gamma <= 16,
        )
        if len(padded_indices) != real_request_count or len(valid_token_indices) != len(input_token_ids):
            logits = logits.index_select(
                0,
                torch.tensor(
                    valid_token_indices[: sum(actual_verification_sizes)],
                    dtype=torch.long,
                    device=self.device,
                ),
            )
        return None, logits[:, : self.draft_vocab_size]

    def _target_round_outputs_stepwise(
        self,
        states: list[PearlPipelineState],
        active_indices: list[int],
        verification_sizes: Sequence[int],
    ) -> tuple[torch.Tensor, None]:
        """Run greedy target verification as causal one-token queries.

        A single multi-token paged-attention call is not semantically reliable
        on the installed Ascend runtime. Each step writes its K/V before the
        next step, matching autoregressive target-only execution while still
        batching the same step across all active requests.
        """
        outputs: list[list[torch.Tensor]] = [[] for _ in active_indices]
        # Build and execute each step together.  Preparing all metadata before
        # the first model call leaves later context_lens/block_tables pointing
        # at the pre-step KV frontier; that is incorrect for causal paged
        # attention because the previous token's KV is written by the model
        # call immediately before it.
        use_aclgraph = (
            not self.config.enforce_eager
            and self.gamma <= 16
            and not envs.VLLM_ASCEND_SPECRHYTHM_DISABLE_TARGET_ACLGRAPH
        )
        for step in range(max(int(value) for value in verification_sizes)):
            rows = [row for row, size in enumerate(verification_sizes) if step < int(size)]
            if not rows:
                continue
            input_token_ids = [
                states[active_indices[row]].token_ids[
                    len(states[active_indices[row]].token_ids) - int(verification_sizes[row]) + step
                ]
                for row in rows
            ]
            sequence_ids = [active_indices[row] for row in rows]
            positions = [
                len(states[active_indices[row]].token_ids) - int(verification_sizes[row]) + step for row in rows
            ]
            input_ids = torch.tensor(input_token_ids, dtype=torch.long, device=self.device)
            position_tensor, attention_metadata = self._prepare_attention_metadata(
                sequence_ids,
                positions,
                (not self.config.target_use_paged_attention or envs.VLLM_ASCEND_SPECRHYTHM_STEPWISE_TARGET_FIA),
            )
            if use_aclgraph:
                step_tokens = self.graph_runner.run_target_greedy(
                    [input_ids],
                    [position_tensor],
                    [attention_metadata],
                    self.draft_vocab_size,
                )[0]
            else:
                hidden_states = self.model(input_ids, position_tensor, attention_metadata)
                step_tokens = self.model.compute_greedy_tokens(hidden_states, self.draft_vocab_size)
            for row, token in zip(rows, step_tokens):
                # ACLGraph outputs are backed by a persistent replay buffer;
                # retain this step before the next replay overwrites it.
                outputs[row].append(token.reshape(1).clone())
        return torch.cat([torch.cat(row) for row in outputs]), None

    def _exchange_draft_windows(
        self,
        verification_windows: list[list[int]],
        next_windows: list[list[int]],
        verification_sizes: list[int],
    ) -> torch.Tensor | None:
        verification_size = sum(verification_sizes)
        continuation_size = self.gamma * len(verification_sizes)
        if self.groups.is_verification_worker:
            if self.rank == self.topology.draft_leader_rank:
                if [len(window) for window in verification_windows] != verification_sizes:
                    raise RuntimeError("PEARL draft window has an unexpected shape.")
                if any(len(window) != self.gamma for window in next_windows):
                    raise RuntimeError("PEARL continuation window has an unexpected shape.")
                message = torch.tensor(
                    [token for window in verification_windows for token in window]
                    + [token for window in next_windows for token in window],
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                message = torch.empty(verification_size + continuation_size, dtype=torch.long, device=self.device)
            dist.broadcast(message, src=self.topology.draft_leader_rank, group=self.groups.verification_group)
            return message
        return None

    def _exchange_draft_device_windows(
        self,
        verification_window: torch.Tensor | None,
        next_windows: torch.Tensor | None,
        verification_sizes: list[int],
        next_window_sizes: Sequence[int] | None = None,
    ) -> torch.Tensor | None:
        verification_size = sum(verification_sizes)
        continuation_size = self.gamma * len(verification_sizes)
        if not self.groups.is_verification_worker:
            return None
        if self.rank == self.topology.draft_leader_rank:
            if verification_window is None or verification_window.shape != (verification_size,):
                raise RuntimeError("PEARL draft verification window has an unexpected shape.")
            if next_windows is None or next_windows.shape != (len(verification_sizes), self.gamma):
                raise RuntimeError("PEARL draft continuation window has an unexpected shape.")
            if next_window_sizes is not None and (
                len(next_window_sizes) != len(verification_sizes)
                or any(value <= 0 or value > self.gamma for value in next_window_sizes)
            ):
                raise RuntimeError("PEARL draft continuation lengths are invalid.")
            message = torch.cat((verification_window, next_windows.flatten()))
        else:
            message = torch.empty(
                verification_size + continuation_size,
                dtype=torch.long,
                device=self.device,
            )
        dist.broadcast(
            message,
            src=self.topology.draft_leader_rank,
            group=self.groups.verification_group,
        )
        return message

    def _verify_target_tokens_batch(
        self,
        target_tokens: torch.Tensor | None,
        target_logits: torch.Tensor | None,
        draft_message: torch.Tensor | None,
        verification_sizes: list[int],
        temperatures: list[float],
        top_ps: Sequence[float] | None = None,
        top_ks: Sequence[int] | None = None,
        bonus_enabled: Sequence[bool] | None = None,
    ) -> torch.Tensor | None:
        verification_size = sum(verification_sizes)
        packed_temperatures = [
            temperature for temperature, size in zip(temperatures, verification_sizes) for _ in range(size)
        ]
        request_top_ps = [1.0] * len(temperatures) if top_ps is None else list(top_ps)
        request_top_ks = [0] * len(temperatures) if top_ks is None else list(top_ks)
        if len(request_top_ps) != len(temperatures) or len(request_top_ks) != len(temperatures):
            raise ValueError("PEARL verification top-p/top-k values must be request-aligned")
        packed_top_ps = [value for value, size in zip(request_top_ps, verification_sizes) for _ in range(size)]
        packed_top_ks = [value for value, size in zip(request_top_ks, verification_sizes) for _ in range(size)]
        greedy = all(temperature == 0 for temperature in packed_temperatures)
        if self.is_draft or (not greedy and self.rank != self.topology.target_leader_rank):
            return None
        if draft_message is None:
            raise RuntimeError("A PEARL target rank did not receive draft verification tokens.")
        if greedy:
            uniform_width = (
                self.gamma
                if all(size == self.gamma for size in verification_sizes)
                else (1 if all(size == 1 for size in verification_sizes) else None)
            )
            bonus_mode = bool(getattr(self.config, "spec_rhythm_linear_bonus_token", False))
            target_verification_tokens = target_tokens
            if bonus_mode:
                if target_tokens is None or target_tokens.shape != (verification_size + len(verification_sizes),):
                    raise RuntimeError(
                        "PEARL bonus target output must contain gamma verification "
                        "tokens and one bonus token per request."
                    )
                if bonus_enabled is None or len(bonus_enabled) != len(verification_sizes):
                    raise RuntimeError("PEARL bonus eligibility must be request-aligned.")
                if not self.config.spec_rhythm_linear_full_window or uniform_width != FIXED_GREEDY_FULL_WINDOW_WIDTH:
                    raise RuntimeError(
                        "PEARL linear bonus tokens require uniform fixed-width full-window verification."
                    )
                target_rows = target_tokens.view(
                    len(verification_sizes),
                    self.gamma + 1,
                )
                mask_key = tuple(bool(value) for value in bonus_enabled)
                mask_cache = getattr(
                    self,
                    "_spec_rhythm_bonus_mask_cache",
                    None,
                )
                if mask_cache is None:
                    mask_cache = {}
                    self._spec_rhythm_bonus_mask_cache = mask_cache
                bonus_enabled_mask = mask_cache.get(mask_key)
                if bonus_enabled_mask is None:
                    bonus_enabled_mask = torch.tensor(
                        mask_key,
                        dtype=torch.bool,
                        device=target_tokens.device,
                    )
                    mask_cache[mask_key] = bonus_enabled_mask
                return fixed_greedy_full_window_bonus_verdict(
                    target_rows,
                    draft_message[:verification_size],
                    bonus_enabled_mask,
                )
            elif target_tokens is None or target_tokens.shape != (verification_size,):
                raise RuntimeError("PEARL target and draft verification windows differ in length.")
            assert target_verification_tokens is not None
            if self.config.spec_rhythm_cpu_verdict:
                return _build_greedy_verdict_cpu(
                    target_verification_tokens,
                    draft_message[:verification_size],
                    verification_sizes,
                )
            if self.config.spec_rhythm_linear_full_window and uniform_width == FIXED_GREEDY_FULL_WINDOW_WIDTH:
                return fixed_greedy_full_window_verdict(
                    target_verification_tokens,
                    draft_message[:verification_size],
                )
            layout_key = (self.gamma, tuple(verification_sizes))
            layout = self.greedy_verification_layouts.get(layout_key)
            if layout is None:
                layout = _build_verification_layout(
                    verification_sizes,
                    self.gamma,
                    target_tokens.device,
                )
                self.greedy_verification_layouts[layout_key] = layout
            return _build_greedy_verdict_with_layout(
                target_verification_tokens,
                draft_message[:verification_size],
                *layout,
                self.gamma,
                uniform_width=uniform_width,
            )
        if target_logits is None or target_logits.shape[0] != verification_size:
            raise RuntimeError("PEARL target logits do not match the packed verification window.")
        return _build_stochastic_verdict(
            target_logits,
            draft_message[:verification_size],
            verification_sizes,
            self.gamma,
            packed_temperatures,
            top_ps=packed_top_ps,
            top_ks=packed_top_ks,
        )

    def _broadcast_round_result(
        self,
        verdict: torch.Tensor | None,
        draft_message: torch.Tensor | None,
        verification_size: int,
        batch_size: int,
        *,
        replicated_target_verdict: bool,
    ) -> tuple[list[int], list[int | None], list[list[int]]]:
        continuation_size = batch_size * self.gamma
        is_target_rank = self.rank in self.topology.target_ranks
        if replicated_target_verdict and is_target_rank:
            if verdict is None or draft_message is None:
                raise RuntimeError("A PEARL target rank did not produce a replicated round result.")
            result = torch.cat((verdict.flatten(), draft_message[verification_size:]))
        else:
            result = torch.empty(batch_size * 2 + continuation_size, dtype=torch.long, device=self.device)
        if replicated_target_verdict:
            if self.rank in self.topology.correction_ranks:
                dist.broadcast(
                    result,
                    src=self.topology.target_leader_rank,
                    group=self.groups.correction_group,
                )
        else:
            if self.rank == self.topology.target_leader_rank:
                if verdict is None or draft_message is None:
                    raise RuntimeError("The PEARL target leader did not produce a round result.")
                result = torch.cat((verdict.flatten(), draft_message[verification_size:]))
            dist.broadcast(result, src=self.topology.target_leader_rank)
        values = [int(value) for value in result.cpu().tolist()]
        verdict_values = values[: batch_size * 2]
        continuation_values = values[batch_size * 2 :]
        accepted = verdict_values[::2]
        corrections = [None if value == -1 else value for value in verdict_values[1::2]]
        next_windows = [
            continuation_values[index : index + self.gamma] for index in range(0, continuation_size, self.gamma)
        ]
        return accepted, corrections, next_windows

    def _allocate_fixed_full_window_host_correction_staging(
        self,
    ) -> _FixedFullWindowHostCorrectionStaging:
        """Allocate the fixed-shape correction buffers once per engine."""

        return _FixedFullWindowHostCorrectionStaging(
            verdict_receive=torch.empty(
                FIXED_FULL_WINDOW_HOST_STAGING_VERDICT_VALUES,
                dtype=torch.long,
                device=self.device,
            ),
            host_values=torch.empty(
                FIXED_FULL_WINDOW_HOST_STAGING_VALUES,
                dtype=torch.long,
                device="cpu",
                pin_memory=True,
            ),
            copy_done_event=torch.npu.Event(),
        )

    def _use_fixed_full_window_host_correction_staging(
        self,
        *,
        batch_size: int,
        replicated_target_verdict: bool,
        next_window_sizes: Sequence[int] | None,
        extra_device_values: torch.Tensor | None,
        profile_phase_seconds: dict[str, float] | None,
        use_gloo_correction: bool,
    ) -> bool:
        """Whether this round satisfies the bounded gamma-4 fast-path contract."""

        if not (
            getattr(getattr(self, "config", None), "spec_rhythm_linear_full_window", False)
            and getattr(self.device, "type", None) == "npu"
            and self.gamma == FIXED_FULL_WINDOW_HOST_STAGING_GAMMA
            and 0 < batch_size <= FIXED_FULL_WINDOW_HOST_STAGING_MAX_BATCH
            and replicated_target_verdict
            and extra_device_values is None
            and profile_phase_seconds is None
            and not use_gloo_correction
        ):
            return False
        return next_window_sizes is None or (
            len(next_window_sizes) == batch_size
            and all(int(value) == FIXED_FULL_WINDOW_HOST_STAGING_GAMMA for value in next_window_sizes)
        )

    def _fixed_full_window_host_correction_buffers(
        self,
    ) -> _FixedFullWindowHostCorrectionStaging:
        staging = getattr(self, "_fixed_full_window_host_correction_staging", None)
        if staging is None:
            staging = self._allocate_fixed_full_window_host_correction_staging()
            self._fixed_full_window_host_correction_staging = staging
        return staging

    def _broadcast_device_round_result(
        self,
        verdict: torch.Tensor | None,
        draft_message: torch.Tensor | None,
        verification_size: int,
        batch_size: int,
        local_next_windows: torch.Tensor | None,
        *,
        replicated_target_verdict: bool,
        next_window_sizes: Sequence[int] | None = None,
        extra_device_values: torch.Tensor | None = None,
        extra_device_scale: float = 1.0,
        profile_phase_seconds: dict[str, float] | None = None,
    ) -> tuple[list[int], list[int | None], list[list[int]]]:
        """Synchronize only the verdict; proposal continuations stay local."""
        continuation_size = batch_size * self.gamma
        self._last_device_round_extra_values: list[float] = []
        # Preserve the long-standing three-value return contract used by the
        # ordinary PEARL loop and downstream harnesses. Bonus-token mode is an
        # opt-in serial service path, so publish its fourth logical channel as
        # per-engine round state just like `_last_device_round_extra_values`.
        self._last_device_round_bonus_tokens: list[int | None] = []
        if extra_device_values is not None:
            if extra_device_values.ndim != 1:
                raise ValueError("Extra PEARL round values must be a one-dimensional tensor.")
            if extra_device_values.numel() != batch_size:
                raise ValueError("Extra PEARL round values must match the target batch size.")
            extra_device_values = extra_device_values.to(device=self.device)
        use_gloo_correction = (
            replicated_target_verdict
            and envs.VLLM_ASCEND_SPECRHYTHM_GLOO_CORRECTION
            # The coordination group contains the single draft leader and
            # every target rank.  A wider draft TP topology has extra ranks
            # that cannot receive this CPU envelope, so retain the original
            # correction collective for that configuration.
            and len(self.topology.draft_ranks) == 1
        )
        use_fixed_host_staging = self._use_fixed_full_window_host_correction_staging(
            batch_size=batch_size,
            replicated_target_verdict=replicated_target_verdict,
            next_window_sizes=next_window_sizes,
            extra_device_values=extra_device_values,
            profile_phase_seconds=profile_phase_seconds,
            use_gloo_correction=use_gloo_correction,
        )
        if use_fixed_host_staging:
            self._fixed_full_window_host_correction_staging_eligible_calls = (
                getattr(
                    self,
                    "_fixed_full_window_host_correction_staging_eligible_calls",
                    0,
                )
                + 1
            )
            fixed_host_staging = self._fixed_full_window_host_correction_buffers()
            self._fixed_full_window_host_correction_staging_calls = (
                getattr(
                    self,
                    "_fixed_full_window_host_correction_staging_calls",
                    0,
                )
                + 1
            )
        else:
            fixed_host_staging = None
        is_target_rank = self.rank in self.topology.target_ranks
        if is_target_rank:
            if verdict is None:
                raise RuntimeError("A PEARL target rank did not produce a round verdict.")
            verdict_result = verdict.flatten()
        elif fixed_host_staging is not None:
            verdict_result = fixed_host_staging.verdict_receive[: batch_size * 2]
        else:
            verdict_result = torch.empty(batch_size * 2, dtype=torch.long, device=self.device)

        communication_started = time.perf_counter()
        participated_in_broadcast = False
        broadcast_work = None
        gloo_verdict_result: torch.Tensor | None = None
        if use_gloo_correction:
            coordination_group = getattr(
                self.groups,
                "verification_coordination_group",
                None,
            )
            if coordination_group is None:
                raise RuntimeError(
                    "The PEARL Gloo correction path requires the verification coordination process group."
                )
            # All greedy target ranks compute the same compact verdict.  Only
            # the target leader materializes it from the NPU; the existing CPU
            # coordination group then publishes that authoritative value to
            # the draft and target followers.  This avoids a latency-bound
            # HCCL launch for two int64 values per active request on every
            # decode cycle.
            if self.rank == self.topology.target_leader_rank:
                gloo_verdict_result = verdict_result.detach().cpu()
            else:
                gloo_verdict_result = torch.empty(
                    batch_size * 2,
                    dtype=torch.long,
                )
            participated_in_broadcast = True
            dist.broadcast(
                gloo_verdict_result,
                src=self.topology.target_leader_rank,
                group=coordination_group,
            )
        elif replicated_target_verdict:
            if self.rank in self.topology.correction_ranks:
                participated_in_broadcast = True
                broadcast_work = dist.broadcast(
                    verdict_result,
                    src=self.topology.target_leader_rank,
                    group=self.groups.correction_group,
                    async_op=True,
                )
        else:
            participated_in_broadcast = True
            broadcast_work = dist.broadcast(
                verdict_result,
                src=self.topology.target_leader_rank,
                async_op=True,
            )

        if local_next_windows is not None:
            if local_next_windows.shape != (batch_size, self.gamma):
                raise RuntimeError("A PEARL rank retained a malformed local continuation window.")
            continuation = local_next_windows.flatten()
        else:
            # Compatibility for callers that still carry continuation tokens
            # after the verification prefix in the exchanged draft message.
            if self.is_draft:
                raise RuntimeError("A PEARL draft rank did not retain its local continuation window.")
            if draft_message is None:
                raise RuntimeError("A PEARL target rank did not receive the draft continuation window.")
            continuation = draft_message[verification_size:]
        # Continuations are already resident on the local rank.  Convert them
        # while HCCL transfers the small verdict tensor, then wait only before
        # reading the broadcast result.  The previous implementation first
        # concatenated all device values and synchronously copied the combined
        # tensor, serializing communication and host materialization every
        # SpecRhythm round.
        # Draft states already appended their local proposal window before
        # entering this collective.  They only need its length for rollback,
        # so avoid a stream-synchronizing NPU->CPU copy on the draft worker.
        # Target states must materialize the continuation values when a window
        # is accepted, because those values are not present in target KV state.
        if fixed_host_staging is not None:
            if self.is_draft:
                continuation_values = continuation
            else:
                # Submit this D2H before waiting on the independent verdict
                # collective.  Both this copy and the verdict copy below land
                # in one persistent pinned envelope and share a single event
                # fence, instead of forcing two ``cpu().tolist()`` stream
                # synchronizations per target round.
                fixed_host_staging.host_values[:continuation_size].copy_(
                    continuation,
                    non_blocking=True,
                )
                continuation_values = None
        else:
            continuation_values = continuation if self.is_draft else continuation.cpu().tolist()
        if broadcast_work is not None:
            broadcast_work.wait()
        if profile_phase_seconds is not None and participated_in_broadcast:
            torch.npu.synchronize()
            profile_phase_seconds["target_to_draft_communication"] += time.perf_counter() - communication_started
        materialize_started = time.perf_counter()
        if fixed_host_staging is not None:
            verdict_offset = 0 if self.is_draft else continuation_size
            verdict_size = batch_size * 2
            packed_size = verdict_offset + verdict_size
            fixed_host_staging.host_values[verdict_offset:packed_size].copy_(
                verdict_result,
                non_blocking=True,
            )
            fixed_host_staging.copy_done_event.record()
            fixed_host_staging.copy_done_event.synchronize()
            packed_values = fixed_host_staging.host_values[:packed_size].tolist()
            if self.is_draft:
                verdict_values = packed_values
            else:
                continuation_values = packed_values[:continuation_size]
                verdict_values = packed_values[verdict_offset:]
        else:
            verdict_values = (
                gloo_verdict_result.tolist() if gloo_verdict_result is not None else verdict_result.cpu().tolist()
            )
        extra_size = 0 if extra_device_values is None else extra_device_values.numel()
        extra_values = extra_device_values.cpu().tolist() if extra_device_values is not None else []
        if extra_size:
            self._last_device_round_extra_values = [float(value) * extra_device_scale for value in extra_values]
        if profile_phase_seconds is not None:
            profile_phase_seconds["wait_sync"] += time.perf_counter() - materialize_started
        window_sizes = (
            [self.gamma] * batch_size if next_window_sizes is None else [int(value) for value in next_window_sizes]
        )
        if len(window_sizes) != batch_size or any(value <= 0 or value > self.gamma for value in window_sizes):
            raise RuntimeError("PEARL synchronized continuation lengths are invalid.")
        next_windows = []
        for row in range(batch_size):
            values = continuation_values[row * self.gamma : row * self.gamma + window_sizes[row]]
            next_windows.append(values if self.is_draft else [int(value) for value in values])
        accepted = [int(value) for value in verdict_values[::2]]
        result_tokens = [int(value) for value in verdict_values[1::2]]
        bonus_mode = bool(
            getattr(
                getattr(self, "config", None),
                "spec_rhythm_linear_bonus_token",
                False,
            )
        )
        corrections: list[int | None] = []
        bonuses: list[int | None] = []
        for row, (accepted_count, expected, token_id) in enumerate(zip(accepted, window_sizes, result_tokens)):
            if not bonus_mode:
                corrections.append(None if token_id == -1 else token_id)
                bonuses.append(None)
                continue
            if not 0 <= accepted_count <= expected:
                raise RuntimeError(f"PEARL verdict row {row} has an invalid accepted length.")
            if accepted_count < expected:
                if token_id < 0:
                    raise RuntimeError(f"PEARL rejected verdict row {row} has no correction token.")
                corrections.append(token_id)
                bonuses.append(None)
            else:
                corrections.append(None)
                bonuses.append(None if token_id == -1 else token_id)
        self._last_device_round_bonus_tokens = bonuses
        return accepted, corrections, next_windows


def _normalize_eos_tokens(eos_token_id: int | list[int] | None) -> frozenset[int]:
    if eos_token_id is None:
        return frozenset()
    if isinstance(eos_token_id, int):
        return frozenset((eos_token_id,))
    return frozenset(int(token_id) for token_id in eos_token_id)


def _canonical_active_indices(
    states: Sequence[PearlPipelineState],
    active_indices: Sequence[int],
) -> list[int]:
    """Group post-verify and pre-verify requests into stable FIA shapes."""
    return sorted(active_indices, key=lambda index: states[index].pre_verify)


def _bucket_target_verification_widths(
    pre_verify: Sequence[bool],
    gamma: int,
    num_buckets: int = TARGET_VERIFICATION_GRAPH_BUCKETS,
    configured_post_counts: Sequence[tuple[int, Sequence[int]]] = (),
) -> tuple[list[int], list[int]]:
    """Quantize target FIA shapes while retaining only real verification rows."""
    if gamma <= 0 or num_buckets <= 0:
        raise ValueError("PEARL verification gamma and graph bucket count must be positive.")
    if not pre_verify:
        return [], []
    post_verify_count = sum(not value for value in pre_verify)
    post_verify_counts = _target_graph_post_counts_for_batch(
        len(pre_verify),
        num_buckets,
        configured_post_counts,
    )
    padded_post_verify_count = next(value for value in post_verify_counts if value >= post_verify_count)
    padded_pre_verify_count = padded_post_verify_count - post_verify_count
    model_widths: list[int] = []
    valid_token_indices: list[int] = []
    offset = 0
    for is_pre_verify in pre_verify:
        if not is_pre_verify:
            model_width = gamma
            valid_token_indices.extend(range(offset, offset + gamma))
        elif padded_pre_verify_count:
            model_width = gamma
            valid_token_indices.append(offset)
            padded_pre_verify_count -= 1
        else:
            model_width = 1
            valid_token_indices.append(offset)
        model_widths.append(model_width)
        offset += model_width
    return model_widths, valid_token_indices


def _bucket_variable_target_verification_widths(
    verification_sizes: Sequence[int],
    gamma: int,
    num_buckets: int = TARGET_VERIFICATION_GRAPH_BUCKETS,
    configured_post_counts: Sequence[tuple[int, Sequence[int]]] = (),
) -> tuple[list[int], list[int]]:
    """Quantize mixed per-request widths without exposing padded rows."""

    if any(size <= 0 or size > gamma for size in verification_sizes):
        raise ValueError("PEARL verification sizes must be in [1, gamma].")
    post_verify = [size > 1 for size in verification_sizes]
    model_widths, _ = _bucket_target_verification_widths(
        [not value for value in post_verify],
        gamma,
        num_buckets,
        configured_post_counts,
    )
    valid_token_indices: list[int] = []
    offset = 0
    for size, model_width in zip(verification_sizes, model_widths):
        valid_token_indices.extend(range(offset, offset + size))
        offset += model_width
    return model_widths, valid_token_indices


def _target_graph_precompile_shapes(
    batch_sizes: Sequence[int],
    gamma: int,
    num_buckets: int,
    configured_post_counts: Sequence[tuple[int, Sequence[int]]] = (),
) -> list[tuple[int, int, int]]:
    """Return unique target graph sizes with representative request layouts."""
    shape_cases: dict[int, tuple[int, int]] = {}
    for batch_size in batch_sizes:
        if batch_size <= 0:
            raise ValueError("PEARL graph precompile batch sizes must be positive.")
        post_verify_counts = _target_graph_post_counts_for_batch(
            batch_size,
            num_buckets,
            configured_post_counts,
        )
        for post_verify_count in post_verify_counts:
            pre_verify = [False] * post_verify_count + [True] * (batch_size - post_verify_count)
            model_widths, _ = _bucket_target_verification_widths(
                pre_verify,
                gamma,
                num_buckets,
                configured_post_counts,
            )
            shape_cases.setdefault(
                sum(model_widths),
                (batch_size, post_verify_count),
            )
    return [
        (num_tokens, batch_size, post_verify_count)
        for num_tokens, (batch_size, post_verify_count) in sorted(shape_cases.items())
    ]


def _normalize_target_graph_post_counts(
    configured_post_counts: Sequence[tuple[int, Sequence[int]]],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Validate and freeze workload-specific target verification graph buckets."""
    normalized: list[tuple[int, tuple[int, ...]]] = []
    configured_batches: set[int] = set()
    for batch_size, raw_post_counts in configured_post_counts:
        if batch_size <= 0:
            raise ValueError("PEARL target graph post-count batch sizes must be positive.")
        if batch_size in configured_batches:
            raise ValueError("PEARL target graph post-count batch sizes must be unique.")
        post_counts = tuple(raw_post_counts)
        if (
            not post_counts
            or post_counts[0] != 0
            or post_counts[-1] != batch_size
            or any(left >= right for left, right in zip(post_counts, post_counts[1:]))
        ):
            raise ValueError(
                "PEARL target graph post counts must be strictly increasing from zero through their batch size."
            )
        configured_batches.add(batch_size)
        normalized.append((batch_size, post_counts))
    return tuple(sorted(normalized))


def _target_graph_post_counts_for_batch(
    batch_size: int,
    num_buckets: int,
    configured_post_counts: Sequence[tuple[int, Sequence[int]]] = (),
) -> tuple[int, ...]:
    for configured_batch_size, post_counts in configured_post_counts:
        if configured_batch_size == batch_size:
            return tuple(post_counts)
    bucket_width = max(1, (batch_size + num_buckets - 1) // num_buckets)
    values = set(range(0, batch_size + 1, bucket_width))
    values.add(batch_size)
    return tuple(sorted(values))


def _normalize_sampling_params(
    batch_size: int,
    sampling_params: NativeSamplingParams | Sequence[NativeSamplingParams] | None,
    default_max_tokens: int,
) -> list[NativeSamplingParams]:
    if sampling_params is None:
        params = [NativeSamplingParams(temperature=0.0, max_tokens=default_max_tokens)] * batch_size
    elif isinstance(sampling_params, NativeSamplingParams):
        params = [sampling_params] * batch_size
    else:
        params = list(sampling_params)
        if len(params) != batch_size:
            raise ValueError("PEARL requires one SamplingParams value per prompt.")
        if not all(isinstance(value, NativeSamplingParams) for value in params):
            raise TypeError("PEARL sampling_params must contain SamplingParams values.")
    temperatures = [value.temperature for value in params]
    if not (all(value == 0 for value in temperatures) or all(value > 0 for value in temperatures)):
        raise ValueError("A PEARL batch requires temperatures that are either all zero or all non-zero.")
    draft_temperatures = [value.draft_temperature for value in params]
    if not (all(value == 0 for value in draft_temperatures) or all(value > 0 for value in draft_temperatures)):
        raise ValueError("A PEARL batch requires draft temperatures that are either all zero or all non-zero.")
    return params


def _sampling_probabilities(
    logits: torch.Tensor,
    temperatures: Sequence[float],
    *,
    top_ps: Sequence[float] | None = None,
    top_ks: Sequence[int] | None = None,
) -> torch.Tensor:
    """Build row-wise temperature/top-k/top-p distributions on device."""
    if logits.ndim != 2 or logits.shape[0] != len(temperatures):
        raise ValueError("PEARL logits and temperatures must have the same batch dimension.")
    if not all(temperature > 0 for temperature in temperatures):
        raise ValueError("PEARL probability construction requires positive temperatures.")
    row_top_ps = [1.0] * len(temperatures) if top_ps is None else [float(value) for value in top_ps]
    row_top_ks = [0] * len(temperatures) if top_ks is None else [int(value) for value in top_ks]
    if len(row_top_ps) != len(temperatures) or len(row_top_ks) != len(temperatures):
        raise ValueError("PEARL top-p/top-k values must match the logits batch dimension.")
    if any(not 0 < value <= 1 for value in row_top_ps) or any(value < 0 for value in row_top_ks):
        raise ValueError("PEARL top-p must be in (0,1] and top-k must be non-negative.")
    temperature_tensor = torch.tensor(temperatures, dtype=torch.float32, device=logits.device).unsqueeze(1)
    scaled_logits = logits.float() / temperature_tensor
    vocabulary_size = int(scaled_logits.shape[-1])
    for row, top_k in enumerate(row_top_ks):
        if 0 < top_k < vocabulary_size:
            threshold = scaled_logits[row].topk(top_k).values[-1]
            scaled_logits[row].masked_fill_(scaled_logits[row] < threshold, -float("inf"))
    probabilities = torch.softmax(scaled_logits, dim=-1)
    if any(value < 1.0 for value in row_top_ps):
        sorted_probabilities, sorted_indices = probabilities.sort(dim=-1, descending=True)
        cumulative = sorted_probabilities.cumsum(dim=-1)
        remove = cumulative - sorted_probabilities >= torch.tensor(
            row_top_ps,
            dtype=torch.float32,
            device=logits.device,
        ).unsqueeze(1)
        sorted_probabilities.masked_fill_(remove, 0.0)
        probabilities.zero_().scatter_(1, sorted_indices, sorted_probabilities)
        probabilities.div_(probabilities.sum(dim=-1, keepdim=True))
    return probabilities


def _sample_logits(
    logits: torch.Tensor,
    temperatures: Sequence[float],
    *,
    top_ps: Sequence[float] | None = None,
    top_ks: Sequence[int] | None = None,
    exponential_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply nano-PEARL's greedy or exponential-race target sampler."""
    if logits.ndim != 2 or logits.shape[0] != len(temperatures):
        raise ValueError("PEARL logits and temperatures must have the same batch dimension.")
    if all(temperature == 0 for temperature in temperatures):
        return logits.argmax(dim=-1)
    probabilities = _sampling_probabilities(logits, temperatures, top_ps=top_ps, top_ks=top_ks)
    return _sample_probabilities(probabilities, exponential_noise=exponential_noise)


def _sample_probabilities(
    probabilities: torch.Tensor,
    *,
    exponential_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    if probabilities.ndim != 2:
        raise ValueError("PEARL probabilities must have rank two.")
    if exponential_noise is None:
        exponential_noise = torch.empty_like(probabilities).exponential_(1)
    elif exponential_noise.shape != probabilities.shape:
        raise ValueError("PEARL exponential sampling noise must match the logits shape.")
    return probabilities.div(exponential_noise.clamp_min(1e-10)).argmax(dim=-1)


def _build_verification_layout(
    verification_sizes: Sequence[int],
    gamma: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expected_sizes = torch.tensor(verification_sizes, dtype=torch.long, device=device)
    dense_positions = torch.tensor(
        [row * gamma + column for row, size in enumerate(verification_sizes) for column in range(size)],
        dtype=torch.long,
        device=device,
    )
    packed_positions: list[list[int]] = []
    offset = 0
    for size in verification_sizes:
        row = list(range(offset, offset + size))
        packed_positions.append(row + [row[-1]] * (gamma - size))
        offset += size
    correction_positions = torch.tensor(packed_positions, dtype=torch.long, device=device)
    return expected_sizes, dense_positions, correction_positions


def _build_greedy_verdict_with_layout(
    target_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    expected_sizes: torch.Tensor,
    dense_positions: torch.Tensor,
    correction_positions: torch.Tensor,
    gamma: int,
    *,
    uniform_width: int | None = None,
) -> torch.Tensor:
    """Build a greedy verdict without per-request device slice writes."""
    # Equal-width windows are the steady-state case.  Keep the packed layout
    # contiguous so verdict construction needs no scatter into a dense mask.
    # This removes several tiny NPU launches from every verification round;
    # mixed pre-verify/post-verify rows use the indexed path below.
    if uniform_width is not None and expected_sizes.numel():
        batch_size = expected_sizes.shape[0]
        target_matrix = target_tokens.reshape(batch_size, uniform_width)
        draft_matrix = draft_tokens.reshape(batch_size, uniform_width)
        matches = target_matrix == draft_matrix
        all_match = matches.all(dim=1)
        first_mismatch = (~matches).to(dtype=torch.int32).argmax(dim=1)
        accepted = torch.where(all_match, expected_sizes, first_mismatch)
        correction = target_matrix.gather(
            1,
            accepted.clamp(max=uniform_width - 1).unsqueeze(1),
        ).squeeze(1)
        correction = torch.where(all_match, torch.full_like(correction, -1), correction)
        return torch.stack((accepted, correction), dim=1)
    matches = target_tokens == draft_tokens
    dense_size = expected_sizes.shape[0] * gamma
    match_matrix = torch.ones(dense_size, dtype=torch.bool, device=matches.device).scatter(
        0,
        dense_positions,
        matches,
    )
    match_matrix = match_matrix.reshape(-1, gamma)
    all_match = match_matrix.all(dim=1)
    first_mismatch = (~match_matrix).to(dtype=torch.int32).argmax(dim=1)
    accepted = torch.where(all_match, expected_sizes, first_mismatch)
    correction_indices = correction_positions.gather(
        1,
        accepted.clamp(max=gamma - 1).unsqueeze(1),
    ).squeeze(1)
    correction = target_tokens.gather(0, correction_indices)
    correction = torch.where(all_match, torch.full_like(correction, -1), correction)
    return torch.stack((accepted, correction), dim=1)


def _build_greedy_verdict_cpu(
    target_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    verification_sizes: Sequence[int],
) -> torch.Tensor:
    """Build a compact greedy verdict on the host for tiny control tensors.

    The verdict contains only two integers per request.  On Ascend, several
    device-side reductions/scatters for this control tensor can cost more than
    copying a few hundred token IDs to the host, especially for small or mixed
    verification windows.  Model activations remain device-resident; this
    path is strictly an optional control-plane optimization.
    """
    if target_tokens.ndim != 1 or draft_tokens.shape != target_tokens.shape:
        raise ValueError("CPU PEARL verdict expects aligned one-dimensional token tensors.")
    sizes = [int(value) for value in verification_sizes]
    if any(value <= 0 for value in sizes) or sum(sizes) != target_tokens.numel():
        raise ValueError("CPU PEARL verdict sizes must partition the token tensors.")
    target_values = target_tokens.detach().cpu().tolist()
    draft_values = draft_tokens.detach().cpu().tolist()
    result: list[int] = []
    offset = 0
    for size in sizes:
        accepted = 0
        while accepted < size and target_values[offset + accepted] == draft_values[offset + accepted]:
            accepted += 1
        correction = -1 if accepted == size else int(target_values[offset + accepted])
        result.extend((accepted, correction))
        offset += size
    return torch.tensor(result, dtype=torch.long, device=target_tokens.device).reshape(-1, 2)


def _build_greedy_verdict(
    target_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    verification_sizes: list[int],
    gamma: int,
) -> torch.Tensor:
    """Find each sequence's accepted greedy prefix without a host sync."""
    if gamma <= 0 or any(size <= 0 or size > gamma for size in verification_sizes):
        raise ValueError("PEARL verification windows must have length in [1, gamma].")
    verification_size = sum(verification_sizes)
    if target_tokens.shape != (verification_size,) or draft_tokens.shape != (verification_size,):
        raise ValueError("PEARL target and draft token tensors must match the packed verification size.")

    matches = target_tokens == draft_tokens
    return _build_verdict_from_acceptance(matches, target_tokens, verification_sizes, gamma)


def _build_stochastic_verdict(
    target_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    verification_sizes: list[int],
    gamma: int,
    temperatures: Sequence[float],
    *,
    top_ps: Sequence[float] | None = None,
    top_ks: Sequence[int] | None = None,
    random_values: torch.Tensor | None = None,
    exponential_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Verify greedy draft tokens against a sampled target distribution."""
    verification_size = sum(verification_sizes)
    if target_logits.ndim != 2 or target_logits.shape[0] != verification_size:
        raise ValueError("PEARL target logits must match the packed verification size.")
    if draft_tokens.shape != (verification_size,) or len(temperatures) != verification_size:
        raise ValueError("PEARL draft tokens and temperatures must match the packed verification size.")
    if not all(temperature > 0 for temperature in temperatures):
        raise ValueError("Stochastic PEARL verification requires positive target temperatures.")
    if draft_tokens.device.type == "cpu" and draft_tokens.numel() and int(draft_tokens.max()) >= target_logits.shape[1]:
        raise ValueError("A PEARL draft token lies outside the target verification vocabulary.")

    probabilities = _sampling_probabilities(
        target_logits,
        temperatures,
        top_ps=top_ps,
        top_ks=top_ks,
    )
    candidate_probabilities = probabilities.gather(1, draft_tokens.unsqueeze(1)).squeeze(1)
    if random_values is None:
        random_values = torch.rand_like(candidate_probabilities)
    elif random_values.shape != candidate_probabilities.shape:
        raise ValueError("PEARL verification random values must match the packed window.")
    accepted = random_values <= candidate_probabilities

    correction_probabilities = probabilities.clone()
    correction_probabilities.scatter_(1, draft_tokens.unsqueeze(1), 0.0)
    correction_mass = correction_probabilities.sum(dim=-1, keepdim=True)
    correction_probabilities = torch.where(
        correction_mass > 0,
        correction_probabilities / correction_mass.clamp_min(1e-20),
        probabilities,
    )
    correction_tokens = _sample_probabilities(correction_probabilities, exponential_noise=exponential_noise)
    return _build_verdict_from_acceptance(
        accepted,
        correction_tokens,
        verification_sizes,
        gamma,
    )


def _build_verdict_from_acceptance(
    accepted_tokens: torch.Tensor,
    correction_tokens: torch.Tensor,
    verification_sizes: list[int],
    gamma: int,
) -> torch.Tensor:
    if gamma <= 0 or any(size <= 0 or size > gamma for size in verification_sizes):
        raise ValueError("PEARL verification windows must have length in [1, gamma].")
    verification_size = sum(verification_sizes)
    if accepted_tokens.shape != (verification_size,) or correction_tokens.shape != (verification_size,):
        raise ValueError("PEARL verification tensors must match the packed verification size.")
    match_matrix = torch.ones(
        (len(verification_sizes), gamma),
        dtype=torch.bool,
        device=accepted_tokens.device,
    )
    token_matrix = torch.zeros(
        (len(verification_sizes), gamma),
        dtype=torch.long,
        device=accepted_tokens.device,
    )
    offset = 0
    for row, size in enumerate(verification_sizes):
        match_matrix[row, :size] = accepted_tokens[offset : offset + size]
        token_matrix[row, :size] = correction_tokens[offset : offset + size]
        offset += size
    expected = torch.tensor(verification_sizes, dtype=torch.long, device=accepted_tokens.device)
    all_match = match_matrix.all(dim=1)
    first_mismatch = (~match_matrix).to(dtype=torch.int32).argmax(dim=1)
    accepted = torch.where(all_match, expected, first_mismatch)
    correction = token_matrix.gather(1, accepted.clamp(max=gamma - 1).unsqueeze(1)).squeeze(1)
    correction = torch.where(all_match, torch.full_like(correction, -1), correction)
    return torch.stack((accepted, correction), dim=1)


def _gamma_from_decode_speeds(draft_tokens_per_second: float, target_tokens_per_second: float) -> int:
    if draft_tokens_per_second <= 0 or target_tokens_per_second <= 0:
        raise ValueError("PEARL auto-gamma profiling produced a non-positive decode speed.")
    return max(1, round(draft_tokens_per_second / target_tokens_per_second))


def _finished(state: PearlPipelineState, eos_token_ids: frozenset[int]) -> bool:
    if state.aborted:
        return True
    completion_token_ids = state.committed_completion_token_ids
    return len(completion_token_ids) >= state.max_tokens or (
        not state.ignore_eos and any(token_id in eos_token_ids for token_id in completion_token_ids)
    )


def _restore_completed_states(
    states: list[PearlPipelineState],
    completed_states: dict[int, PearlPipelineState],
    eos_token_ids: frozenset[int],
) -> None:
    """Bound static padding by replaying completed rows from their snapshot."""
    for index, state in enumerate(states):
        if index in completed_states:
            states[index] = completed_states[index].clone()
        elif _finished(state, eos_token_ids):
            completed_states[index] = state.clone()


def _continuous_bucket_indices(
    unfinished_indices: list[int],
    completed_states: dict[int, PearlPipelineState],
    initial_batch_size: int,
) -> list[int]:
    """Pad a draining continuous batch to at most two stable graph shapes."""
    if not unfinished_indices:
        return []
    tail_bucket_size = max(1, initial_batch_size // 2)
    bucket_size = initial_batch_size if len(unfinished_indices) > tail_bucket_size else tail_bucket_size
    padding_count = bucket_size - len(unfinished_indices)
    if padding_count <= 0:
        return unfinished_indices
    unfinished = set(unfinished_indices)
    padding_indices = [index for index in sorted(completed_states) if index not in unfinished][:padding_count]
    if len(padding_indices) != padding_count:
        raise RuntimeError("PEARL continuous batching cannot fill its ACLGraph tail bucket.")
    return [*unfinished_indices, *padding_indices]


def _next_power_of_two_graph_bucket(work_size: int, max_size: int) -> int:
    """Return the smallest stable graph bucket that can hold ``work_size``.

    The final bucket is allowed to be non-power-of-two so deployments with a
    non-power-of-two ``max_num_seqs`` still remain within their configured
    service capacity.
    """
    if work_size <= 0:
        raise ValueError("Graph work size must be positive.")
    if max_size <= 0 or work_size > max_size:
        raise ValueError("Graph work size must not exceed the positive maximum.")
    return min(1 << (work_size - 1).bit_length(), max_size)


def _power_of_two_graph_buckets(max_size: int) -> list[int]:
    """Enumerate every stable bucket up to the configured service capacity."""
    if max_size <= 0:
        raise ValueError("Graph maximum size must be positive.")
    buckets: list[int] = []
    bucket = 1
    while bucket < max_size:
        buckets.append(bucket)
        bucket *= 2
    buckets.append(max_size)
    return buckets


def _linear_draft_graph_buckets(max_size: int) -> list[int]:
    """Return the bounded stable bucket family for fixed-gamma serial draft.

    The dense upper-half buckets target the common online SpecSLO occupancy
    range.  Keeping this family bounded is important because each bucket owns
    a complete multi-step ACLGraph and its paged-attention task metadata.
    """
    if max_size <= 0:
        raise ValueError("Linear draft graph maximum size must be positive.")
    preferred = (1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 48, 64)
    buckets = [value for value in preferred if value < max_size]
    buckets.append(max_size)
    return buckets


def _next_linear_draft_graph_bucket(work_size: int, max_size: int) -> int:
    """Return the smallest fixed-gamma serial-draft bucket holding the work."""
    if work_size <= 0:
        raise ValueError("Linear draft graph work size must be positive.")
    if max_size <= 0 or work_size > max_size:
        raise ValueError("Linear draft graph work size must not exceed the positive maximum.")
    return next(bucket for bucket in _linear_draft_graph_buckets(max_size) if bucket >= work_size)


def _pad_linear_draft_graph_metadata(
    positions: torch.Tensor,
    attention_metadata: Any,
    capture_size: int,
) -> tuple[torch.Tensor, Any]:
    """Pad one serial-draft graph step without recopying its shared token IDs.

    A full-chain draft graph consumes the root token IDs only once, while each
    autoregressive step has distinct positions and paged-attention metadata.
    ``NativeACLGraphRunner._pad_inputs`` remains the source of the first
    padded token buffer; subsequent steps use this metadata-only equivalent.
    """

    num_tokens = int(positions.shape[0])
    pad_size = capture_size - num_tokens
    if pad_size < 0:
        raise ValueError("Draft graph metadata exceeds its capture size.")
    if pad_size == 0:
        return positions, attention_metadata

    padded_positions = torch.zeros(
        capture_size,
        dtype=positions.dtype,
        device=positions.device,
    )
    padded_slot_mapping = torch.full(
        (capture_size,),
        -1,
        dtype=attention_metadata.slot_mapping.dtype,
        device=attention_metadata.slot_mapping.device,
    )
    padded_context_lens = torch.zeros(
        capture_size,
        dtype=attention_metadata.context_lens.dtype,
    )
    padded_block_tables = torch.zeros(
        (capture_size, attention_metadata.block_tables.shape[1]),
        dtype=attention_metadata.block_tables.dtype,
        device=attention_metadata.block_tables.device,
    )
    padded_positions[:num_tokens].copy_(positions)
    padded_slot_mapping[:num_tokens].copy_(attention_metadata.slot_mapping)
    padded_context_lens[:num_tokens].copy_(attention_metadata.context_lens)
    padded_block_tables[:num_tokens].copy_(attention_metadata.block_tables)
    metadata_updates = {
        "slot_mapping": padded_slot_mapping,
        "context_lens": padded_context_lens,
        "block_tables": padded_block_tables,
    }
    sequence_lens = tuple(getattr(attention_metadata, "sequence_lens", ()))
    if not getattr(attention_metadata, "use_fused_infer_attention", False) and len(sequence_lens) == num_tokens:
        metadata_updates["sequence_lens"] = (
            *sequence_lens,
            *((0,) * pad_size),
        )
    if hasattr(attention_metadata, "__dataclass_fields__"):
        padded_metadata = replace(attention_metadata, **metadata_updates)
    else:
        padded_metadata = type(attention_metadata)(**metadata_updates)
    return padded_positions, padded_metadata


def _pad_bucketed_linear_draft_fia_graph_metadata(
    positions: torch.Tensor,
    attention_metadata: NativeAttentionMetadata,
    capture_size: int,
    *,
    block_size: int,
    exact_positions: Sequence[int] | None = None,
    host_request_block_tables: Sequence[Sequence[int]] | None = None,
    kv_length_limits: Sequence[int] | None = None,
    dummy_kv_length_limits: Sequence[int] | None = None,
    precomputed_full_mask: torch.Tensor | None = None,
    shared_request_block_tables: torch.Tensor | None = None,
) -> tuple[torch.Tensor, NativeAttentionMetadata]:
    """Build a fixed-row FULL-mask FIA envelope for one serial draft step.

    Every real row still sees exactly ``[0, position]``.  Only the host FIA
    KV length is rounded up, and every newly exposed physical position is
    blocked by the request-local FULL mask.  Dummy graph rows write no cache
    slot and read only logical position zero from a valid page-table row.  An
    Optional dummy KV limits let common/ranked lifetime modes keep the
    complete capture signature fixed without widening dummy-row visibility.
    Consequently the query partition and KV-length literals stay unchanged
    within a context bucket and their per-layer graph tasks need not be
    rebuilt on every replay.
    """

    num_tokens = int(positions.shape[0])
    pad_size = capture_size - num_tokens
    if num_tokens <= 0 or pad_size < 0:
        raise ValueError("Bucketed linear draft FIA needs a non-empty batch no larger than its capture size.")
    if block_size <= 0:
        raise ValueError("Bucketed linear draft FIA block size must be positive.")
    if (
        not attention_metadata.use_fused_infer_attention
        or getattr(attention_metadata, "tree_attention", False)
        or attention_metadata.request_block_tables is None
        or attention_metadata.request_block_tables.ndim != 2
    ):
        raise ValueError("Bucketed linear draft FIA requires causal FIA metadata with request-major page tables.")
    sequence_lens = tuple(int(value) for value in attention_metadata.sequence_lens)
    if exact_positions is not None and (
        len(exact_positions) != num_tokens or tuple(int(position) + 1 for position in exact_positions) != sequence_lens
    ):
        raise ValueError("Bucketed linear draft FIA positions and exact KV lengths must align.")
    expected_queries = tuple(range(1, num_tokens + 1))
    if (
        len(sequence_lens) != num_tokens
        or attention_metadata.actual_seq_lengths_q != expected_queries
        or attention_metadata.request_block_tables.shape[0] != num_tokens
        or any(value <= 0 for value in sequence_lens)
    ):
        raise ValueError("Bucketed linear draft FIA requires one positive query segment per request.")
    table_capacity = int(attention_metadata.request_block_tables.shape[1]) * block_size
    capacity = table_capacity
    maximum_length = max(sequence_lens)
    if maximum_length > capacity:
        raise ValueError("Bucketed linear draft FIA context exceeds its page-table capacity.")
    if host_request_block_tables is None:
        raise ValueError(
            "Bucketed linear draft FIA requires host page tables so visible "
            "KV holes cannot be hidden by device-side tail sanitization."
        )
    if len(host_request_block_tables) != num_tokens:
        raise ValueError("Bucketed linear draft FIA needs one host page table per request.")
    for length, table in zip(sequence_lens, host_request_block_tables):
        visible_pages = (length + block_size - 1) // block_size
        if (
            len(table) * block_size != capacity
            or visible_pages <= 0
            or any(int(page) < 0 for page in table[:visible_pages])
        ):
            raise RuntimeError("Bucketed linear draft FIA found an unallocated page in a request's visible KV prefix.")
    context_buckets = (
        tuple(min(1 << (length - 1).bit_length(), capacity) for length in sequence_lens)
        if kv_length_limits is None
        else tuple(int(value) for value in kv_length_limits)
    )
    if len(context_buckets) != num_tokens or any(
        bucket < length or bucket > capacity for bucket, length in zip(context_buckets, sequence_lens)
    ):
        raise ValueError(
            "Bucketed linear draft FIA KV limits must cover every exact "
            "request length without exceeding cache capacity: "
            f"exact={sequence_lens}, limits={context_buckets}, "
            f"capacity={capacity}."
        )
    dummy_context_buckets = (
        (1,) * pad_size if dummy_kv_length_limits is None else tuple(int(value) for value in dummy_kv_length_limits)
    )
    if len(dummy_context_buckets) != pad_size or any(value < 1 or value > capacity for value in dummy_context_buckets):
        raise ValueError(
            "Bucketed linear draft FIA dummy KV limits must contain one "
            "positive value per padded row without exceeding cache "
            f"capacity: dummy={dummy_context_buckets}, "
            f"capacity={capacity}."
        )

    padded_positions = torch.zeros(
        capture_size,
        dtype=positions.dtype,
        device=positions.device,
    )
    padded_positions[:num_tokens].copy_(positions)
    padded_slot_mapping = torch.full(
        (capture_size,),
        -1,
        dtype=attention_metadata.slot_mapping.dtype,
        device=attention_metadata.slot_mapping.device,
    )
    padded_slot_mapping[:num_tokens].copy_(attention_metadata.slot_mapping)
    # Tail logical pages may be unallocated. They are masked, but the FIA ABI
    # still requires valid physical page indices. Reuse an allocated page from
    # the same request for every negative tail entry; dummy rows reuse the
    # first real request's sanitized table and never write a cache slot.
    if shared_request_block_tables is None:
        padded_request_tables = torch.empty(
            (
                capture_size,
                attention_metadata.request_block_tables.shape[1],
            ),
            dtype=attention_metadata.request_block_tables.dtype,
            device=attention_metadata.request_block_tables.device,
        )
        real_request_tables = attention_metadata.request_block_tables
        fallback_pages = real_request_tables[:, :1]
        sanitized_request_tables = torch.where(
            real_request_tables >= 0,
            real_request_tables,
            fallback_pages,
        )
        padded_request_tables[:num_tokens].copy_(sanitized_request_tables)
        if pad_size:
            padded_request_tables[num_tokens:].copy_(sanitized_request_tables[:1].expand(pad_size, -1))
    else:
        padded_request_tables = shared_request_block_tables
        if (
            padded_request_tables.shape
            != (
                capture_size,
                attention_metadata.request_block_tables.shape[1],
            )
            or padded_request_tables.dtype != attention_metadata.request_block_tables.dtype
            or padded_request_tables.device != attention_metadata.request_block_tables.device
        ):
            raise ValueError(
                "A shared linear draft FIA request table must preserve the padded shape, dtype and device."
            )
    if precomputed_full_mask is None:
        visible_lengths = torch.ones(
            capture_size,
            dtype=torch.long,
            device=positions.device,
        )
        visible_lengths[:num_tokens].copy_(torch.tensor(sequence_lens, dtype=torch.long, device=positions.device))
        full_mask = (
            torch.arange(capacity, dtype=torch.long, device=positions.device)
            .view(1, capacity)
            .ge(visible_lengths.view(capture_size, 1))
            .view(
                capture_size,
                1,
                1,
                capacity,
            )
        )
    else:
        full_mask = precomputed_full_mask
        if (
            full_mask.shape != (capture_size, 1, 1, capacity)
            or full_mask.dtype != torch.bool
            or full_mask.device != positions.device
        ):
            raise ValueError(
                "A precomputed linear draft FIA FULL mask must preserve the padded shape, bool dtype and device."
            )
    padded_context_buckets = (
        *context_buckets,
        *dummy_context_buckets,
    )
    padded_metadata = replace(
        attention_metadata,
        slot_mapping=padded_slot_mapping,
        context_lens=torch.tensor(
            padded_context_buckets,
            dtype=attention_metadata.context_lens.dtype,
        ),
        block_tables=padded_request_tables,
        actual_seq_lengths_q=tuple(range(1, capture_size + 1)),
        sequence_lens=padded_context_buckets,
        request_block_tables=padded_request_tables,
        attention_mask=None,
        tree_attention=True,
        tree_attention_mask=full_mask,
    )
    return padded_positions, padded_metadata


def _linear_draft_fia_lifetime_limits(
    states: Sequence[PearlPipelineState],
    active_indices: Sequence[int],
    gamma: int,
    *,
    common_kv: bool,
) -> list[int]:
    """Return exact-safe lifetime KV envelopes for serial-draft FIA rows.

    The common mode intentionally takes its maximum over the complete service
    state list, rather than only the currently active rows.  Request admission,
    completion and rolling-eager selection can therefore change row identity
    without changing the FIA ``sequence_lens`` tuple for a graph bucket/lane.
    The request-local FULL mask remains responsible for exact visibility.
    """

    if gamma <= 0:
        raise ValueError("Linear draft FIA gamma must be positive.")
    if not active_indices:
        return []
    if not states:
        raise ValueError("Linear draft FIA requires non-empty service state.")

    def lifetime_limit(state: PearlPipelineState) -> int:
        return int(state.prompt_length) + int(state.max_tokens) + gamma

    if common_kv:
        common_limit = max(lifetime_limit(state) for state in states)
        return [common_limit] * len(active_indices)
    return [lifetime_limit(states[index]) for index in active_indices]


@dataclass(frozen=True)
class _RankedLinearDraftFIAPlan:
    """Membership-invariant capacity slots plus active-row permutations."""

    capacity_limits: tuple[int, ...]
    graph_to_caller_rows: tuple[int, ...]
    caller_to_graph_rows: tuple[int, ...]


def _ranked_linear_draft_fia_plan(
    states: Sequence[PearlPipelineState],
    active_indices: Sequence[int],
    gamma: int,
    capture_size: int,
) -> _RankedLinearDraftFIAPlan:
    """Assign an arbitrary active subset to fixed descending capacity slots.

    The slot vector is the service-wide top-``capture_size`` lifetime limits,
    independent of current membership. If the service contains fewer rows
    than the graph bucket, unused tail slots use the minimum safe KV length
    one. Sorting the active subset by the same key guarantees order-statistic
    domination: active row ``i`` never exceeds capacity slot ``i``.
    """

    if gamma <= 0:
        raise ValueError("Ranked linear draft FIA gamma must be positive.")
    if capture_size <= 0:
        raise ValueError("Ranked linear draft FIA capture size must be positive.")
    if not active_indices or len(active_indices) > capture_size:
        raise ValueError("Ranked linear draft FIA needs a non-empty active subset no larger than its capture size.")
    if len(set(active_indices)) != len(active_indices) or any(
        index < 0 or index >= len(states) for index in active_indices
    ):
        raise ValueError("Ranked linear draft FIA active indices must be a unique service subset.")

    def lifetime_limit(state: PearlPipelineState) -> int:
        return int(state.prompt_length) + int(state.max_tokens) + gamma

    service_limits = sorted(
        (lifetime_limit(state) for state in states),
        reverse=True,
    )
    capacity_limits = tuple([*service_limits[:capture_size]] + [1] * max(0, capture_size - len(service_limits)))
    graph_to_caller = tuple(
        sorted(
            range(len(active_indices)),
            key=lambda row: (
                -lifetime_limit(states[active_indices[row]]),
                row,
            ),
        )
    )
    caller_to_graph = [0] * len(active_indices)
    for graph_row, caller_row in enumerate(graph_to_caller):
        caller_to_graph[caller_row] = graph_row
    active_limits = [lifetime_limit(states[active_indices[caller_row]]) for caller_row in graph_to_caller]
    if any(
        active_limit > capacity_limit
        for active_limit, capacity_limit in zip(
            active_limits,
            capacity_limits,
        )
    ):
        raise RuntimeError("Ranked linear draft FIA active lifetime exceeds its assigned service-wide capacity slot.")
    return _RankedLinearDraftFIAPlan(
        capacity_limits=capacity_limits,
        graph_to_caller_rows=graph_to_caller,
        caller_to_graph_rows=tuple(caller_to_graph),
    )


def _restore_ranked_linear_draft_rows(
    value: torch.Tensor,
    caller_to_graph_rows: Sequence[int],
) -> torch.Tensor:
    """Restore a ranked graph-row tensor to the caller's original row order."""

    row_count = len(caller_to_graph_rows)
    if value.ndim < 1 or value.shape[0] != row_count:
        raise ValueError("Ranked linear draft FIA output must contain exactly one graph row per caller row.")
    if sorted(int(row) for row in caller_to_graph_rows) != list(range(row_count)):
        raise ValueError("Ranked linear draft FIA inverse row mapping must be a permutation.")
    return value.index_select(
        0,
        torch.tensor(
            caller_to_graph_rows,
            dtype=torch.long,
            device=value.device,
        ),
    )


def _select_preemptive_continuous_indices(
    states: list[PearlPipelineState],
    unfinished_indices: list[int],
    max_active: int,
    recent_token_gains: Sequence[Sequence[int]] | None = None,
    *,
    decode_elapsed_ms: float = 0.0,
    slo_aware: bool = False,
) -> list[int]:
    """Prioritize SLO debt, then the largest online remaining-work estimate."""
    observed_tokens = sum(max(0, len(states[index].committed_completion_token_ids) - 1) for index in unfinished_indices)
    observed_rounds = sum(states[index].verification_rounds for index in unfinished_indices)
    global_tokens_per_round = observed_tokens / observed_rounds if observed_rounds else 1.0

    def priority(index: int) -> tuple[float, ...]:
        state = states[index]
        rounds = state.verification_rounds
        if rounds < PREEMPTIVE_SCHEDULING_EXPLORATION_ROUNDS:
            base_priority = (0.0, float(rounds), float(index))
        else:
            generated = min(state.max_tokens, len(state.committed_completion_token_ids))
            observed = max(0, generated - 1)
            smoothed_rate = (observed + PREEMPTIVE_SCHEDULING_PRIOR_ROUNDS * global_tokens_per_round) / (
                rounds + PREEMPTIVE_SCHEDULING_PRIOR_ROUNDS
            )
            if recent_token_gains is not None and recent_token_gains[index]:
                recent = recent_token_gains[index]
                recent_rate = (sum(recent) + PREEMPTIVE_SCHEDULING_PRIOR_ROUNDS * global_tokens_per_round) / (
                    len(recent) + PREEMPTIVE_SCHEDULING_PRIOR_ROUNDS
                )
                smoothed_rate = min(smoothed_rate, recent_rate)
            estimated_remaining_rounds = (state.max_tokens - generated) / max(smoothed_rate, 1e-6)
            base_priority = (1.0, -estimated_remaining_rounds, float(rounds), float(index))

        if not slo_aware:
            return base_priority
        if state.slo_tpot_ms is None:
            return (1.0, 0.0, *base_priority)
        generated = max(1, len(state.committed_completion_token_ids))
        expected_elapsed_ms = state.slo_tpot_ms * generated
        urgency = max(0.0, decode_elapsed_ms) / max(expected_elapsed_ms, 1e-6)
        return (0.0, -urgency, *base_priority)

    return sorted(
        unfinished_indices,
        key=priority,
    )[:max_active]


def _continuous_result_states(
    states: list[PearlPipelineState],
    completed_states: dict[int, PearlPipelineState],
) -> list[PearlPipelineState]:
    """Return completed snapshots or the current state after an early profile stop."""
    return [completed_states.get(index, state) for index, state in enumerate(states)]


def _truncate_completion(
    completion_token_ids: list[int],
    eos_token_ids: frozenset[int],
    max_tokens: int,
    ignore_eos: bool = False,
) -> list[int]:
    truncated = completion_token_ids[:max_tokens]
    if ignore_eos:
        return truncated
    return truncated[: next((index + 1 for index, token_id in enumerate(truncated) if token_id in eos_token_ids), None)]


def _load_gsm8k_questions(dataset_path: str, max_samples: int) -> list[str]:
    import pyarrow.parquet as pq

    rows = pq.read_table(dataset_path, columns=["question"]).to_pylist()
    return [str(row["question"]) for row in rows[:max_samples]]


def _prompt_token_ids(tokenizer, prompt: str) -> list[int]:
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return list(tokenizer.encode(formatted_prompt, add_special_tokens=False))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-tp-size", type=int, default=1)
    parser.add_argument("--target-tp-size", type=int, default=2)
    parser.add_argument("--draft-dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    parser.add_argument("--target-dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    parser.add_argument(
        "--draft-mode",
        choices=tuple(sorted(NATIVE_DRAFT_MODES)),
        default=SERIAL_LINEAR_DRAFT_MODE,
        help=(
            "Draft execution algorithm. pard_parallel uses one experimental eager "
            "parallel-draft forward and may be paired with target ACLGraph replay."
        ),
    )
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--mode", choices=("pearl", "target-ar"), default="pearl")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument(
        "--draft-temperature",
        type=float,
        default=0.0,
        help="Optional proposal temperature; target verification remains authoritative.",
    )
    parser.add_argument("--draft-top-p", type=float, default=1.0)
    parser.add_argument("--draft-top-k", type=int, default=0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--slo-tpot-ms", type=float)
    parser.add_argument("--slo-class")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--num-kvcache-blocks", type=int, default=-1)
    parser.add_argument("--max-aclgraph-entries", type=int, default=32)
    parser.add_argument(
        "--target-verification-graph-buckets",
        type=int,
        default=TARGET_VERIFICATION_GRAPH_BUCKETS,
    )
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--enable-continuous-batching", action="store_true")
    parser.add_argument("--enable-preemptive-scheduling", action="store_true")
    parser.add_argument("--enable-spec-rhythm", action="store_true")
    parser.add_argument(
        "--spec-rhythm-linear-full-window",
        action="store_true",
        help="Verify every fixed-gamma serial proposal directly from the committed target frontier.",
    )
    parser.add_argument(
        "--spec-rhythm-linear-eager-cross-graph-bucket",
        action="store_true",
        help=(
            "Let measured W admit fixed-gamma rolling-eager rows into the next "
            "serial-draft graph bucket (experimental; default stays same-bucket)."
        ),
    )
    parser.add_argument(
        "--spec-rhythm-linear-idle-residual-eager",
        action="store_true",
        help=("Fill otherwise idle fixed-gamma draft windows with measured-W bounded dependency-exact continuations."),
    )
    parser.add_argument(
        "--spec-rhythm-online-prefill",
        action="store_true",
        help="Gate continuous-batching admission on each request's arrival_ts.",
    )
    parser.add_argument(
        "--spec-rhythm-prefill-coalesce-min-requests",
        type=int,
        default=1,
        help="Minimum arrived requests for bounded fixed-serial online prefill coalescing.",
    )
    parser.add_argument(
        "--spec-rhythm-prefill-coalesce-max-wait-ms",
        type=float,
        default=0.0,
        help="Maximum fixed-serial online prefill coalescing wait in milliseconds.",
    )
    parser.add_argument(
        "--spec-rhythm-prefill-token-chunk-size",
        type=int,
        default=0,
        help=(
            "Aggregate token cap for one online SpecRhythm prefill submission (0 disables cross-cycle token chunking)."
        ),
    )
    parser.add_argument(
        "--spec-rhythm-merge-ready-homes",
        action="store_true",
        help="Allow both ready logical homes into one target verification window.",
    )
    parser.add_argument(
        "--spec-rhythm-priority-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use SLO urgency when rotating target homes.",
    )
    parser.add_argument("--spec-rhythm-priority-burst", type=int, default=2)
    parser.add_argument("--spec-rhythm-target-fallback-max-batch", type=int, default=0)
    parser.add_argument(
        "--spec-rhythm-stable-graphs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--spec-rhythm-min-gamma", type=int, default=1)
    parser.add_argument("--spec-rhythm-max-eager-tokens", type=int, default=0)
    parser.add_argument("--spec-rhythm-eager-reserve-tokens", type=int, default=0)
    parser.add_argument("--spec-rhythm-max-target-batch", type=int, default=0)
    parser.add_argument("--spec-rhythm-urgency-threshold", type=float, default=0.75)
    parser.add_argument("--spec-rhythm-acceptance-floor", type=float, default=0.4)
    parser.add_argument("--spec-rhythm-acceptance-ema-alpha", type=float, default=0.2)
    parser.add_argument(
        "--spec-rhythm-cpu-verdict",
        action="store_true",
        help="Run the exact verifier on CPU for correctness diagnostics.",
    )
    parser.add_argument(
        "--spec-rhythm-roofline",
        type=parse_roofline_argument,
        help="Legacy JSON budgets or a strict measured profile JSON file path.",
    )
    parser.add_argument(
        "--spec-rhythm-verification-budget",
        type=int,
        help="Fixed global SpecRhythm candidate-token budget B.",
    )
    parser.add_argument("--spec-rhythm-draft-token-budget", type=int)
    parser.add_argument("--spec-rhythm-tree-width", type=int, default=1)
    parser.add_argument("--spec-rhythm-tree-depth", type=int, default=1)
    parser.add_argument("--spec-rhythm-request-max-gamma", type=int)
    parser.add_argument("--disable-cpu-binding", action="store_true")
    parser.add_argument("--draft-use-paged-attention", action="store_true")
    parser.add_argument("--target-use-paged-attention", action="store_true")
    parser.add_argument(
        "--precompile-decode-graphs",
        action="store_true",
        help="Precompile both draft and target paged-attention graph families.",
    )
    parser.add_argument(
        "--precompile-serial-draft-graphs",
        action="store_true",
        help=(
            "Precompile and changed-input qualify only fixed-gamma serial "
            "draft graphs; leave target verification graphs lazy."
        ),
    )
    parser.add_argument(
        "--draft-use-production-rope",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--target-use-production-rope",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--enable-mc2",
        action="store_true",
        help="Enable the optional TP projection/all-reduce/RMSNorm MC2 path.",
    )
    parser.add_argument(
        "--mc2-profile",
        help="Identity-bound MC2 numerical/performance qualification JSON.",
    )
    parser.add_argument("--prompt")
    parser.add_argument("--repeat-prompt", type=int, default=1)
    parser.add_argument("--gsm8k", help="Path to a GSM8K parquet file.")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--summary-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if bool(args.prompt) == bool(args.gsm8k):
        raise ValueError("Pass exactly one of --prompt or --gsm8k.")
    if args.repeat_prompt <= 0 or (args.gsm8k and args.repeat_prompt != 1):
        raise ValueError("repeat-prompt must be positive and is only valid with --prompt.")
    config = NativePearlConfig(
        draft_model=args.draft_model,
        target_model=args.target_model,
        draft_tp_size=args.draft_tp_size,
        target_tp_size=args.target_tp_size,
        gamma=args.gamma,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        draft_dtype=args.draft_dtype,
        target_dtype=args.target_dtype,
        draft_mode=args.draft_mode,
        max_num_seqs=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_kvcache_blocks=args.num_kvcache_blocks,
        max_aclgraph_entries=args.max_aclgraph_entries,
        target_verification_graph_buckets=args.target_verification_graph_buckets,
        enable_continuous_batching=(args.enable_continuous_batching or args.enable_spec_rhythm),
        enable_preemptive_scheduling=(args.enable_preemptive_scheduling or args.enable_spec_rhythm),
        enable_spec_rhythm=args.enable_spec_rhythm,
        spec_rhythm_linear_full_window=args.spec_rhythm_linear_full_window,
        spec_rhythm_linear_eager_cross_graph_bucket=(args.spec_rhythm_linear_eager_cross_graph_bucket),
        spec_rhythm_linear_idle_residual_eager=(args.spec_rhythm_linear_idle_residual_eager),
        spec_rhythm_online_prefill=args.spec_rhythm_online_prefill,
        spec_rhythm_prefill_coalesce_min_requests=(args.spec_rhythm_prefill_coalesce_min_requests),
        spec_rhythm_prefill_coalesce_max_wait_ms=(args.spec_rhythm_prefill_coalesce_max_wait_ms),
        spec_rhythm_prefill_token_chunk_size=(args.spec_rhythm_prefill_token_chunk_size),
        spec_rhythm_merge_ready_homes=args.spec_rhythm_merge_ready_homes,
        spec_rhythm_priority_mode=args.spec_rhythm_priority_mode,
        spec_rhythm_priority_burst=args.spec_rhythm_priority_burst,
        spec_rhythm_target_fallback_max_batch=args.spec_rhythm_target_fallback_max_batch,
        spec_rhythm_stable_graphs=args.spec_rhythm_stable_graphs,
        spec_rhythm_min_gamma=args.spec_rhythm_min_gamma,
        spec_rhythm_max_eager_tokens=args.spec_rhythm_max_eager_tokens,
        spec_rhythm_eager_reserve_tokens=args.spec_rhythm_eager_reserve_tokens,
        spec_rhythm_max_target_batch=args.spec_rhythm_max_target_batch,
        spec_rhythm_urgency_threshold=args.spec_rhythm_urgency_threshold,
        spec_rhythm_acceptance_floor=args.spec_rhythm_acceptance_floor,
        spec_rhythm_acceptance_ema_alpha=args.spec_rhythm_acceptance_ema_alpha,
        spec_rhythm_cpu_verdict=args.spec_rhythm_cpu_verdict,
        spec_rhythm_roofline=args.spec_rhythm_roofline,
        spec_rhythm_verification_budget=args.spec_rhythm_verification_budget,
        spec_rhythm_draft_token_budget=args.spec_rhythm_draft_token_budget,
        spec_rhythm_tree_width=args.spec_rhythm_tree_width,
        spec_rhythm_tree_depth=args.spec_rhythm_tree_depth,
        draft_use_paged_attention=args.draft_use_paged_attention,
        target_use_paged_attention=args.target_use_paged_attention,
        draft_use_production_rope=args.draft_use_production_rope,
        target_use_production_rope=args.target_use_production_rope,
        precompile_decode_graphs=args.precompile_decode_graphs,
        precompile_serial_draft_graphs=args.precompile_serial_draft_graphs,
        enable_prefix_caching=not args.disable_prefix_caching,
        enable_cpu_binding=not args.disable_cpu_binding,
        enforce_eager=args.enforce_eager,
        enable_mc2=args.enable_mc2,
        mc2_profile=args.mc2_profile,
        seed=args.seed,
    )
    engine = NativePearlEngine(config)
    sampling_params = NativeSamplingParams(
        temperature=args.temperature,
        draft_temperature=args.draft_temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        draft_top_p=args.draft_top_p,
        draft_top_k=args.draft_top_k,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
        slo_tpot_ms=args.slo_tpot_ms,
        slo_class=args.slo_class,
        spec_rhythm_max_gamma=args.spec_rhythm_request_max_gamma,
    )
    prompts = [args.prompt] * args.repeat_prompt if args.prompt else _load_gsm8k_questions(args.gsm8k, args.max_samples)
    results: list[dict[str, Any]] = []
    total_elapsed = 0.0
    for start in range(0, len(prompts), args.batch_size):
        prompt_batch = prompts[start : start + args.batch_size]
        prompt_token_ids = [_prompt_token_ids(engine.tokenizer, prompt) for prompt in prompt_batch]
        if args.mode == "pearl":
            batch_results = engine.generate_batch(prompt_token_ids, sampling_params)
        else:
            batch_results = engine.generate_target_ar_batch(prompt_token_ids, sampling_params)
        if batch_results is not None:
            total_elapsed += batch_results[0]["elapsed_seconds"]
            for prompt, result in zip(prompt_batch, batch_results):
                result["prompt"] = prompt
                result["text"] = engine.tokenizer.decode(result["completion_token_ids"], skip_special_tokens=True)
                results.append(result)
    if engine.rank == engine.topology.target_leader_rank:
        total_verified = sum(result["verified_draft_tokens"] for result in results)
        total_accepted = sum(result["accepted_draft_tokens"] for result in results)
        generated_token_count = sum(len(result["completion_token_ids"]) for result in results)
        aggregate_mat = sum(result["mean_accept_tokens"] for result in results) / len(results) if results else 0.0
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "samples": [] if args.summary_only else results,
                    "num_samples": len(results),
                    "generated_token_count": generated_token_count,
                    "aggregate_acceptance_rate": total_accepted / total_verified if total_verified else 0.0,
                    "aggregate_mat": aggregate_mat,
                    "decode_throughput_tokens_per_second": (
                        generated_token_count / total_elapsed if total_elapsed else 0.0
                    ),
                },
                ensure_ascii=True,
            )
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
