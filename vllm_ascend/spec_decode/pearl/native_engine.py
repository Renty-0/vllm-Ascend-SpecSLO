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
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer

from vllm_ascend import envs
from vllm_ascend.cpu_binding import bind_cpus
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.spec_decode.pearl.mc2 import MC2Profile, normalize_mc2_profile
from vllm_ascend.spec_decode.pearl.native_cache import NativeCacheAllocation, NativePrefixCache
from vllm_ascend.spec_decode.pearl.native_graph import NativeACLGraphRunner
from vllm_ascend.spec_decode.pearl.native_model import (
    PAGED_ATTENTION_BLOCK_SIZE,
    NativeTPContext,
    build_native_model,
    load_native_model_weights,
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
    build_tree_speculation_plan,
    pack_selected_tree_plan,
    tree_primary_path,
    verify_greedy_tree_batch,
)
from vllm_ascend.spec_decode.pearl.tree_budget import (
    DraftWindowEstimator,
    TreeCandidateRequest,
    select_global_tree_candidates,
)
from vllm_ascend.spec_decode.tree_kv import move_kv_cache_slots

AUTO_GAMMA_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
AUTO_GAMMA_WARMUP_STEPS = 5
AUTO_GAMMA_PROFILE_STEPS = 30
AUTO_GAMMA_PROFILE_SEQUENCE_LENGTH = 256
TARGET_VERIFICATION_GRAPH_BUCKETS = 8
PREEMPTIVE_SCHEDULING_EXPLORATION_ROUNDS = 8
PREEMPTIVE_SCHEDULING_PRIOR_ROUNDS = 8
PREEMPTIVE_SCHEDULING_RECENT_ROUNDS = 32

logger = logging.getLogger("vllm_ascend.spec_decode.pearl.native")


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

    def clone(self) -> PearlPipelineState:
        return PearlPipelineState(
            token_ids=list(self.token_ids),
            prompt_length=self.prompt_length,
            pre_verify=self.pre_verify,
            accepted_draft_tokens=self.accepted_draft_tokens,
            verified_draft_tokens=self.verified_draft_tokens,
            verification_rounds=self.verification_rounds,
            committed_length=self.committed_length,
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

    def validate_for(self, state: PearlPipelineState) -> None:
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
        expected_size = 1 if state.pre_verify else (state.pending_window_size or self.ticket.gamma)
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
    spec_rhythm_online_prefill: bool = False
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
    spec_rhythm_urgency_threshold: float = 0.75
    spec_rhythm_acceptance_floor: float = 0.4
    spec_rhythm_acceptance_ema_alpha: float = 0.2
    spec_rhythm_cpu_verdict: bool = False
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
    enable_cpu_binding: bool = True
    profile_decode_steps: int = 0
    stop_after_profiled_decode_steps: bool = False
    enforce_eager: bool = False
    enable_mc2: bool = False
    mc2_profile: Mapping[str, Any] | str | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_verification_graph_post_counts",
            _normalize_target_graph_post_counts(self.target_verification_graph_post_counts),
        )
        if self.draft_tp_size <= 0 or self.target_tp_size <= 0:
            raise ValueError("Draft and target TP sizes must be positive.")
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
        if self.spec_rhythm_min_gamma <= 0 or (self.gamma != -1 and self.spec_rhythm_min_gamma > self.gamma):
            raise ValueError("SpecRhythm min gamma must be in [1, gamma].")
        if self.spec_rhythm_max_eager_tokens < 0 or (
            self.gamma != -1 and self.spec_rhythm_max_eager_tokens > self.gamma
        ):
            raise ValueError("SpecRhythm eager-token cap must be in [0, gamma].")
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
        if self.profile_decode_steps < 0:
            raise ValueError("PEARL profile_decode_steps must be non-negative.")
        if self.stop_after_profiled_decode_steps and self.profile_decode_steps == 0:
            raise ValueError("PEARL profiling-only execution requires profile_decode_steps to be positive.")
        if self.mc2_profile is not None:
            object.__setattr__(self, "mc2_profile", normalize_mc2_profile(self.mc2_profile))
        if self.enable_mc2 and not isinstance(self.mc2_profile, MC2Profile):
            raise ValueError("MC2 execution requires an identity-bound numerical/performance profile")
        if self.seed is not None and self.seed < 0:
            raise ValueError("PEARL sampling seed must be non-negative.")


class NativePearlEngine:
    """One rank of nano-PEARL's persistent HCCL runtime."""

    def __init__(self, config: NativePearlConfig) -> None:
        self.config = config
        self.gamma = config.gamma
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
            self._cache_sequence_capacity,
            config.kvcache_block_size,
            num_cache_blocks,
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
        self.model.eval()
        if config.seed is not None:
            torch.manual_seed(config.seed)
            torch.npu.manual_seed_all(config.seed)
        self.graph_runner = NativeACLGraphRunner(
            self.model,
            enabled=not config.enforce_eager,
            max_graph_tokens=max(512, config.max_num_seqs * max(1, self.gamma)),
            max_graph_entries=config.max_aclgraph_entries,
        )
        self.last_worker_decode_phase_seconds: dict[str, float] = {}
        self.last_worker_decode_profile_seconds: dict[str, float] = {}
        self.last_worker_profiled_decode_steps = 0
        self.last_worker_decode_counters: dict[str, int] = {}
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
        if config.precompile_decode_graphs:
            self._precompile_decode_graphs()
        dist.barrier()

    def graph_metrics(self) -> dict[str, int | float]:
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
            "aclgraph_expected_fia_batch_size": (
                self.graph_runner.expected_fia_batch_size
                if self.graph_runner.expected_fia_batch_size is not None
                else 0
            ),
            "aclgraph_last_fia_request_count": len(self.graph_runner.last_fia_shape),
            "aclgraph_last_fia_query_shape": ",".join(str(value) for value in self.graph_runner.last_fia_shape),
        }
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
        metrics.update({f"worker_{name}": value for name, value in self.last_worker_decode_counters.items()})
        return metrics

    def configure_decode_profiling(
        self,
        profile_decode_steps: int,
        stop_after_profiled_decode_steps: bool = False,
    ) -> None:
        """Change decode profiling between requests without reloading the models."""
        if profile_decode_steps < 0:
            raise ValueError("PEARL profile_decode_steps must be non-negative.")
        if stop_after_profiled_decode_steps and profile_decode_steps == 0:
            raise ValueError("PEARL profiling-only execution requires profile_decode_steps to be positive.")
        self.config = replace(
            self.config,
            profile_decode_steps=profile_decode_steps,
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
            * self._cache_sequence_capacity
        )
        num_blocks = min(max_required_blocks, cache_budget // bytes_per_block)
        if num_blocks < self._cache_sequence_capacity:
            raise MemoryError(
                "PEARL cannot reserve one KV cache page per configured sequence; "
                "lower max_num_seqs or increase available NPU memory."
            )
        return num_blocks

    @property
    def _cache_sequence_capacity(self) -> int:
        return self.config.max_num_queued_seqs or self.config.max_num_seqs

    def _allocate_cache(
        self,
        prompts: list[list[int]],
        *,
        enable_prefix_caching: bool,
        reserve_sequence_capacity: bool = False,
    ) -> NativeCacheAllocation:
        allocation = self.prefix_cache.allocate(
            prompts,
            enable_prefix_caching=enable_prefix_caching,
            sequence_capacity=(self._cache_sequence_capacity if reserve_sequence_capacity else None),
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
        self.cache_allocation = None
        self.cache_block_tables = None

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
        if not tree_mode and any(
            params.temperature > 0 and params.draft_temperature > 0 for params in request_params
        ):
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
        self._allocate_cache(
            all_tokens,
            enable_prefix_caching=self.config.enable_prefix_caching,
            reserve_sequence_capacity=live_admission,
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
            or self.config.spec_rhythm_min_gamma != self.gamma
            or self.config.spec_rhythm_max_eager_tokens != 0
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
                "abort"
                if state.aborted
                else "length"
                if len(completion_token_ids) >= state.max_tokens
                else "stop"
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
                payload.validate_for(state)
            current_width = (
                payload.verification_size if payload is not None else state._verification_size(self.gamma, None)
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
    ) -> dict[int, torch.Tensor]:
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
        checked: list[tuple[str, torch.Tensor, int]] = []
        expected_device = self._spec_rhythm_tree_cache_device(self.device)
        counts = [int(plan.parent_indices.numel()) + 1 for plan in target_plans]
        if any(count != int(plan.candidate_budget) + 1 for count, plan in zip(counts, target_plans)):
            raise RuntimeError("SpecRhythm tree KV preflight requires physically packed candidates")

        def add_mapping(label: str, mapping: Any, count: int) -> None:
            if (
                not isinstance(mapping, torch.Tensor)
                or mapping.ndim != 1
                or mapping.numel() != count
                or mapping.dtype not in (torch.int32, torch.int64)
            ):
                raise RuntimeError(f"SpecRhythm {label} KV mapping must contain exactly {count} integer slots")
            if mapping.device != expected_device:
                raise RuntimeError(f"SpecRhythm {label} KV mapping is on the wrong device")
            checked.append((label, mapping, count))

        if target_output is not None:
            mapping = target_output.get("cache_slot_mapping")
            expected = sum(counts)
            if not isinstance(mapping, torch.Tensor) or mapping.ndim != 1 or mapping.numel() != expected:
                raise RuntimeError(f"SpecRhythm target packed KV mapping must contain exactly {expected} slots")
            cursor = 0
            for index, count in zip(target_indices, counts):
                proposal_id = controller.validate_verification(index).proposal_id
                row_mapping = mapping[cursor : cursor + count]
                add_mapping(f"target proposal {proposal_id}", row_mapping, count)
                pending[proposal_id] = row_mapping
                cursor += count
        elif not self.is_draft:
            raise RuntimeError("SpecRhythm target KV preflight has no forward mapping")

        if self.is_draft:
            for index, count in zip(target_indices, counts):
                proposal_id = controller.validate_verification(index).proposal_id
                add_mapping(f"draft proposal {proposal_id}", cache_mappings.get(proposal_id), count)
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
            return pending
        capacity: int | None = None
        model = getattr(self, "model", None)
        if model is not None:
            if bool(self._spec_rhythm_nonfinite_flag().item()):
                raise RuntimeError("SpecRhythm model produced nonfinite Q/K/V or output; discard this cache")
            for layer in model.layers:
                attention = layer.self_attn
                for cache in (attention.key_cache, attention.value_cache):
                    if not isinstance(cache, torch.Tensor) or cache.ndim != 4:
                        raise RuntimeError("SpecRhythm tree KV preflight requires allocated paged layer caches")
                    if cache.device != expected_device:
                        raise RuntimeError("SpecRhythm tree KV cache is on the wrong device")
                    slots = int(cache.shape[0]) * int(cache.shape[1])
                    capacity = slots if capacity is None else min(capacity, slots)
            if capacity is None:
                raise RuntimeError("SpecRhythm tree KV preflight requires at least one layer cache")
        slot_values = torch.cat([mapping.to(dtype=torch.long) for _, mapping, _ in checked]).detach().cpu().tolist()
        cursor = 0
        for label, _, count in checked:
            row_slots = slot_values[cursor : cursor + count]
            if any(slot < 0 or (capacity is not None and slot >= capacity) for slot in row_slots):
                raise RuntimeError(f"SpecRhythm {label} KV mapping contains an out-of-bounds physical slot")
            if len(set(row_slots)) != count:
                raise RuntimeError(f"SpecRhythm {label} KV mapping aliases distinct tree queries")
            cursor += count
        return pending

    def _spec_rhythm_tree_plan(
        self,
        state: PearlPipelineState,
        budget: int,
    ) -> TreeSpeculationPlan:
        """Keep scheduler-only tree topology on the host until model packing."""
        prefix_len = max(0, len(state.token_ids) - 1)
        return build_tree_speculation_plan(
            self.config.spec_rhythm_tree_width,
            self.config.spec_rhythm_tree_depth,
            prefix_len,
            self.config.max_model_len,
            candidate_budget=int(budget),
            device="cpu",
        )

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
        return build_tree_speculation_plan(
            self.config.spec_rhythm_tree_width,
            self.config.spec_rhythm_tree_depth,
            expected_prefix,
            self.config.max_model_len,
            candidate_budget=int(budget),
            device="cpu",
        )

    def _exchange_spec_rhythm_tree_candidates(
        self,
        candidate_rows: Sequence[Sequence[int]] | None,
        plans: Sequence[TreeSpeculationPlan],
        frontier_tokens: Sequence[int | None] | None = None,
        confidences: Sequence[float] | None = None,
        selected_indices: Sequence[Sequence[int]] | None = None,
    ) -> torch.Tensor | None:
        """Broadcast active tree nodes through the proposal verification group."""
        if not plans:
            return None
        capacities = [int(plan.width * plan.depth) for plan in plans]
        # One extra scalar per row carries the draft-predicted frontier token
        # used to validate a rolling eager continuation.  ``-1`` marks a
        # normal proposal and is never a valid vocabulary id in this control
        # envelope.
        message_size = 2 * sum(capacities) + 3 * len(plans)
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
    ) -> tuple[list[list[int]], list[int | None], list[float], list[list[int]]]:
        capacities = [int(plan.width * plan.depth) for plan in plans]
        if message.ndim != 1 or message.numel() != 2 * sum(capacities) + 3 * len(plans):
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
        confidences = [float(value) for value in values[cursor + len(plans) :]]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in confidences):
            raise RuntimeError("tree proposal contains invalid confidence")
        return rows, [int(value) if value >= 0 else None for value in frontier_values], confidences, selected_rows

    def _broadcast_spec_rhythm_tree_verdict(
        self,
        output: TreeVerificationOutput | None,
        plans: Sequence[TreeSpeculationPlan],
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Replicate tree path tokens to target ranks and draft ranks."""
        if not plans:
            return [], []
        depth = max(int(plan.depth) for plan in plans)
        batch = len(plans)
        width = batch * (depth + 1 + depth)
        is_target_leader = self.rank == self.topology.target_leader_rank
        if is_target_leader:
            if output is None:
                raise RuntimeError("target leader did not produce a tree verdict")
            if tuple(output.token_ids.shape) != (batch, depth + 1):
                raise RuntimeError("tree verifier output has an invalid token shape")
            if tuple(output.accepted_node_indices.shape) != (batch, depth):
                raise RuntimeError("tree verifier output has an invalid acceptance shape")
            message = torch.cat(
                (output.token_ids.to(torch.long).reshape(-1), output.accepted_node_indices.to(torch.long).reshape(-1))
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
        values = message.reshape(-1)
        token_values = values[: batch * (depth + 1)].reshape(batch, depth + 1)
        accepted_values = values[batch * (depth + 1) :].reshape(batch, depth)
        return (
            [[int(value) for value in row] for row in token_values.cpu().tolist()],
            [[int(value) for value in row] for row in accepted_values.cpu().tolist()],
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
            "spec_rhythm_last_verification_roof": 0,
            "spec_rhythm_unused_verification_tokens": 0,
            "spec_rhythm_prefill_batches": 0,
            "spec_rhythm_prefill_requests": 0,
            "spec_rhythm_peak_active_requests": 0,
            "spec_rhythm_peak_verify_candidates": 0,
            "spec_rhythm_target_query_tokens": 0,
            "spec_rhythm_draft_model_calls": 0,
            "spec_rhythm_draft_graph_calls": 0,
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
                torch.npu.synchronize()
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

        def run_profiled_target_only(indices: Sequence[int]) -> list[int]:
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
                if not any(temperatures):
                    if self.config.enforce_eager:
                        hidden = self.model(input_ids, packed_positions, metadata)
                        token_ids = self.model.compute_greedy_tokens(hidden, self.draft_vocab_size)
                    else:
                        token_ids = self.graph_runner.run_target_greedy(
                            [input_ids], [packed_positions], [metadata], self.draft_vocab_size
                        )[0]
                else:
                    if self.config.enforce_eager:
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
                        not self.config.enforce_eager
                        and self.graph_runner.last_target_execution.used_aclgraph
                    ),
                }
            torch.npu.synchronize()
            self._vote_spec_rhythm_prefill_finiteness()
            self._validate_spec_rhythm_target_graph(
                target_execution,
                has_target_work=True,
                target_only=True,
            )
            dist.broadcast(token_ids, src=self.topology.target_leader_rank)
            return [int(value) for value in token_ids.cpu().tolist()]

        def complete_target_only_cycle(
            target_indices: Sequence[int],
            *,
            cycle_started: float,
            unprofiled: bool,
            target_home: int | None,
        ) -> None:
            """Finish one AR fallback cycle and keep both model KV states aligned."""
            nonlocal active, live_open, last_cycle_ms, round_count
            rows = [int(index) for index in target_indices]
            invalidated_eager = 0
            for index in active:
                invalidated_eager += int(index in controller.staged_eager)
                for ticket in (controller.ready.get(index), controller.staged_eager.get(index)):
                    if ticket is not None:
                        payloads.pop(ticket.proposal_id, None)
                        cache_mappings.pop(ticket.proposal_id, None)
                controller.invalidate_request(index)
            counters["spec_rhythm_tree_eager_invalidated"] += invalidated_eager
            tokens = run_profiled_target_only(rows)
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
            if live_open:
                live_open = not admit_available()
            else:
                admit_available(poll_live=False)
            counters["spec_rhythm_last_verification_roof"] = 0
            counters["spec_rhythm_unused_verification_tokens"] = 0
            counters["spec_rhythm_target_query_tokens"] += len(rows)
            counters["spec_rhythm_target_only_fallback_rounds"] = (
                counters.get("spec_rhythm_target_only_fallback_rounds", 0) + 1
            )
            counters["spec_rhythm_target_only_fallback_tokens"] = (
                counters.get("spec_rhythm_target_only_fallback_tokens", 0) + len(tokens)
            )
            if unprofiled:
                counters["spec_rhythm_unprofiled_target_only_fallback_rounds"] = (
                    counters.get("spec_rhythm_unprofiled_target_only_fallback_rounds", 0) + 1
                )
                counters["spec_rhythm_unprofiled_target_only_fallback_tokens"] = (
                    counters.get("spec_rhythm_unprofiled_target_only_fallback_tokens", 0) + len(tokens)
                )
            counters["spec_rhythm_tree_rounds"] += 1
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
                    prospective_rows = [
                        index for index in active if states[index].home_batch_id == prospective_home
                    ]
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
            )
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
            eager = eager[: window.eager_token_budget // exploration_cost]
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
                verification_roof=verification_roof,
            )
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
            draft_started = time.perf_counter()
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
            phase_seconds["draft"] += time.perf_counter() - draft_started
            if draft_output is not None:
                torch.npu.synchronize()
                counters["spec_rhythm_draft_model_calls"] += draft_output.get("model_calls", 0)
                counters["spec_rhythm_draft_graph_calls"] += draft_output.get("graph_calls", 0)
            draft_ended = time.perf_counter()
            # Role-local functions early-return on the other model's ranks.
            # Launch target(A) before receiving draft(B): both device groups
            # run independently until the step-end publication/commit fence.
            target_plans = [payload["plan"] for payload in target_payloads]
            target_rows = [payload["row"] for payload in target_payloads]
            target_started = time.perf_counter()
            target_tree_kwargs: dict[str, Any] = {
                "sequence_ids": target_indices,
                "return_logits": False,
            }
            target_temperatures = [local_states[index].temperature for index in target_indices]
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
                        target_plans,
                        [local_states[index].token_ids[-1] for index in target_indices],
                        target_rows,
                        **target_tree_kwargs,
                    )
                    if target_indices
                    else None
                )
            phase_seconds["target"] += time.perf_counter() - target_started
            if target_output is not None:
                torch.npu.synchronize()
                counters["spec_rhythm_target_query_tokens"] += target_output["query_count"]
            target_ended = time.perf_counter()
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
                local_rows = selected_rows
            exchange_started = time.perf_counter()
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
                    selected_indices,
                )
            phase_seconds["exchange"] += time.perf_counter() - exchange_started
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
            # HCCL reduction does not implement float64. Preserve clock
            # precision with integer microseconds rather than float32
            # absolute timestamps (which lose subsecond precision).
            timing = torch.tensor(
                [round(value * (1000 if index < 2 else 1_000_000)) for index, value in enumerate(timing_values)],
                dtype=torch.int64,
                device=self.device,
            )
            dist.all_reduce(timing)
            draft_ms, target_ms, ds, de, ts, te = [
                value / (1000 if index < 2 else 1_000_000) for index, value in enumerate(timing.cpu().tolist())
            ]
            if len(decode_timeline) < self.config.profile_decode_steps:
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
            window_estimator.observe(
                draft_compute_ms=draft_ms,
                drafted_tokens=len(work_plans) * exploration_cost,
                target_verify_ms=target_ms,
            )
            if work_plans:
                counters["spec_rhythm_allocated_draft_tokens"] += sum(int(plan.candidate_budget) for plan in work_plans)
                assert message is not None
                candidate_rows, frontier_rows, confidence_rows, selected_rows = self._split_tree_candidates(
                    message, work_plans
                )
                mapping = None if draft_output is None else draft_output["cache_slot_mapping"]
                cursor = 0
                published_tickets = []
                for index, ticket, tree_plan, row in zip(work_indices, tickets, work_plans, candidate_rows):
                    expected = int(tree_plan.width) * int(tree_plan.depth) + 1
                    if not row:
                        cursor += expected
                        continue
                    ticket.gamma = len(row)
                    published_tickets.append(ticket)
                    selection = selected_rows[work_indices.index(index)]
                    packed_plan = pack_selected_tree_plan(tree_plan, selection)
                    if self.is_draft and mapping is not None:
                        source = mapping[cursor : cursor + expected].index_select(
                            0,
                            torch.tensor([0, *[node + 1 for node in selection]], dtype=torch.long, device=self.device),
                        )
                        if not ticket.eager:
                            destination = torch.tensor(
                                self._cache_slot_mapping(
                                    [index] * (len(selection) + 1),
                                    list(range(packed_plan.prefix_len, packed_plan.prefix_len + len(selection) + 1)),
                                ),
                                dtype=torch.int32,
                                device=self.device,
                            )
                            self._move_tree_cache_slots(source, destination)
                            source = destination
                        cache_mappings[ticket.proposal_id] = source
                    eager_metadata: dict[str, Any] = {}
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
                            "eager_frontier_token": (
                                frontier_rows[work_indices.index(index)] if frontier_rows is not None else None
                            ),
                            "eager_dependency_length": path_len,
                        }
                    payloads[ticket.proposal_id] = {
                        "ticket": ticket,
                        "plan": packed_plan,
                        "row": row,
                        "confidence": confidence_rows[work_indices.index(index)],
                        **eager_metadata,
                    }
                    cursor += expected
                controller.publish(published_tickets)
                counters["spec_rhythm_tree_nodes"] += sum(ticket.gamma for ticket in published_tickets)
                counters["spec_rhythm_unused_verification_tokens"] = budget_plan.verification_roof - sum(
                    ticket.gamma for ticket in published_tickets
                )

            if not target_indices:
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
                round_count += 1
                if npu_profiler is not None:
                    npu_profiler.step()
                continue
            target_verdict: TreeVerificationOutput | None = None
            if self.rank == self.topology.target_leader_rank:
                assert target_output is not None
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
            broadcast_started = time.perf_counter()
            with torch.profiler.record_function("SpecSLO/TargetToDraft"):
                output_rows, accepted_rows = self._broadcast_spec_rhythm_tree_verdict(target_verdict, target_plans)
            phase_seconds["broadcast"] += time.perf_counter() - broadcast_started
            state_started = time.perf_counter()
            finished: list[int] = []
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
                pending_cache_mappings = self._preflight_spec_rhythm_tree_cache_commit(
                    controller,
                    payloads,
                    cache_mappings,
                    target_indices,
                    target_plans,
                    target_output,
                )
            except Exception as error:
                preflight_error = error
            preflight_failed = torch.tensor(
                [int(preflight_error is not None)],
                dtype=torch.int64,
                device=self.device,
            )
            dist.all_reduce(preflight_failed, op=dist.ReduceOp.MAX)
            # One scalar synchronization at the collective commit boundary.
            # Earlier forward/transport failures abort the worker, and later
            # hardware/OOM failures during KV moves are not transactional.
            if preflight_failed.cpu().tolist()[0] or preflight_error is not None:
                if preflight_error is not None:
                    raise RuntimeError(
                        f"SpecRhythm tree commit preflight failed on rank {self.rank}: {preflight_error}"
                    ) from preflight_error
                raise RuntimeError(
                    "SpecRhythm tree commit preflight failed on another rank; this step was not committed"
                )
            cache_mappings.update(pending_cache_mappings)
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
                    self.compact_tree_round(
                        cache_mappings.pop(current_ticket.proposal_id).to(device=self.device),
                        torch.tensor(accepted_rows[row], dtype=torch.int32, device=self.device).reshape(1, -1),
                        [proposed_count],
                    )
                if eager_ticket is not None and promoted is None:
                    counters["spec_rhythm_tree_eager_invalidated"] += 1
                    payloads.pop(eager_ticket.proposal_id, None)
                    cache_mappings.pop(eager_ticket.proposal_id, None)
                elif eager_ticket is not None:
                    counters["spec_rhythm_tree_eager_promoted"] += 1
                    if self.is_draft:
                        eager_plan = payloads[eager_ticket.proposal_id]["plan"]
                        count = eager_plan.parent_indices.numel() + 1
                        destination = torch.tensor(
                            self._cache_slot_mapping(
                                [index] * count,
                                list(range(eager_plan.prefix_len, eager_plan.prefix_len + count)),
                            ),
                            dtype=torch.int32,
                            device=self.device,
                        )
                        self._move_tree_cache_slots(cache_mappings[eager_ticket.proposal_id], destination)
                        cache_mappings[eager_ticket.proposal_id] = destination
                if _finished(local_states[index], self.eos_token_ids):
                    finished.append(index)
                self._deliver_committed_tokens(index, request_params[index], local_states[index])
            phase_seconds["state_update"] += time.perf_counter() - state_started
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
            if live_open:
                live_open = not admit_available()
            else:
                admit_available(poll_live=False)
            counters["spec_rhythm_tree_rounds"] += 1
            round_count += 1
            if npu_profiler is not None:
                npu_profiler.step()
            if self.config.stop_after_profiled_decode_steps and round_count >= self.config.profile_decode_steps:
                break
        torch.npu.synchronize()
        if npu_profiler is not None:
            npu_profiler.stop()
            self._active_tree_profiler = None
        decode_elapsed = time.perf_counter() - started
        self.last_worker_decode_phase_seconds = phase_seconds
        self.last_worker_decode_profile_seconds = {}
        self.last_worker_profiled_decode_steps = 0
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
            finish_reason = (
                "abort"
                if state.aborted
                else "length"
                if len(completion) >= state.max_tokens
                else "stop"
            )
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
        effective_eager_cap = self.config.spec_rhythm_max_eager_tokens or (self.gamma if has_slo_constraints else 0)
        effective_priority = self.config.spec_rhythm_priority_mode or has_slo_constraints
        if has_slo_constraints:
            # SLO mode verifies one physical home batch at a time, and rows
            # leave the service as soon as they hit their completion limit.
            # The graph key already contains the complete FIA query shape, so
            # retaining an initial-batch equality guard would force every
            # short tail (32 -> ... -> 1 rows) through eager execution. Allow
            # shape-specific capture/replay in this control-plane path while
            # leaving ordinary PEARL's fixed-batch guard unchanged.
            self.graph_runner.set_expected_fia_batch_size(None)
        local_states = draft_states if self.is_draft else target_states
        active_indices: list[int] = []
        pending_admission = list(range(len(local_states)))
        completed_states: dict[int, PearlPipelineState] = {}
        payloads: dict[int, NativeSpecRhythmDevicePayload] = {}
        prefetched = set(prefilled_indices or ())
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
            "spec_rhythm_last_verification_roof": 0,
            "spec_rhythm_unused_verification_tokens": 0,
            "spec_rhythm_dual_batch_overlap_protocol": 1,
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

        def admit_ready(now: float) -> None:
            nonlocal prefetched
            capacity = initial_batch_size - len(active_indices)
            if capacity <= 0:
                return
            # The paper's decode-stage evaluation assumes that prefill has
            # already completed before a request enters this scheduler.  In
            # that mode arrival_ts is trace metadata, not a future admission
            # gate.  Only the explicit online-prefill mode should replay the
            # trace clock and hold requests until their arrival.
            arrival_gated = bool(self.config.spec_rhythm_online_prefill and continuous_batching)
            ready_indices = [
                index
                for index in pending_admission
                if not arrival_gated
                or request_params[index].arrival_ts is None
                or request_params[index].arrival_ts <= now
            ][:capacity]
            if not ready_indices:
                return
            ready_set = set(ready_indices)
            pending_admission[:] = [index for index in pending_admission if index not in ready_set]
            home_counts = [
                sum(runtime_states[index].home_batch_id == home for index in active_indices) for home in (0, 1)
            ]
            for request_index in ready_indices:
                runtime_states[request_index].home_batch_id = 0 if home_counts[0] <= home_counts[1] else 1
                home_counts[runtime_states[request_index].home_batch_id] += 1
                arrival = request_params[request_index].arrival_ts
                if arrival_gated and arrival is not None:
                    runtime_states[request_index].arrival_wait_ms = max(
                        runtime_states[request_index].arrival_wait_ms,
                        (now - arrival) * 1000.0,
                    )

            # New arrivals share one packed prefill.  Apart from reducing
            # kernel launches, this keeps the draft and target frontiers in
            # lockstep for all requests admitted by the same scheduler tick.
            prefill_indices = [index for index in ready_indices if index not in prefetched]
            if prefill_indices:
                target_tokens = self._prefill_and_sample_target_batch(
                    [local_states[index].token_ids for index in prefill_indices],
                    [local_states[index] for index in prefill_indices],
                    prefill_indices,
                )
                counters["spec_rhythm_prefill_batches"] += 1
                counters["spec_rhythm_prefill_requests"] += len(prefill_indices)
                for request_index, target_token in zip(prefill_indices, target_tokens):
                    for state in (
                        draft_states[request_index],
                        target_states[request_index],
                    ):
                        state.token_ids.append(target_token)
                        assert state.committed_length is not None
                        state.committed_length += 1
                prefetched.update(prefill_indices)
            active_indices.extend(ready_indices)
            for index in ready_indices:
                self._deliver_committed_tokens(index, request_params[index], local_states[index])

        def admit_available() -> None:
            """Admit ready requests without a clock collective when the queue is empty.

            Once a fixed SpecSLO batch has been prefetched, repeated wall-clock
            broadcasts do not change scheduler state. Keep the collective for
            online/future arrivals, where every rank must observe the same time.
            """
            if pending_admission:
                admit_ready(synchronized_wall_time())

        def invalidate_request(request_index: int) -> None:
            controller.invalidate_request(request_index)
            for proposal_id, payload in tuple(payloads.items()):
                if payload.ticket.request_index == request_index:
                    payloads.pop(proposal_id, None)

        def run_target_fallback(indices: Sequence[int]) -> None:
            """Serve tiny resident batches with one exact target token.

            At low arrival rates a two-home speculative pipeline would make a
            request wait for the other home even though no useful draft batch
            exists.  The fallback keeps both KV caches on the same committed
            frontier and returns to SpecRhythm as soon as the resident batch
            reaches the configured threshold.
            """
            indices = list(indices)
            local = draft_states if self.is_draft else target_states
            input_ids = [local[index].token_ids[-1] for index in indices]
            positions = [len(local[index].token_ids) - 1 for index in indices]
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
                    use_aclgraph=not self.config.enforce_eager,
                    **target_sampling_kwargs,
                )
            dist.broadcast(token_ids, src=self.topology.target_leader_rank)
            values = [int(value) for value in token_ids.cpu().tolist()]
            for index, token_id in zip(indices, values):
                for state in (draft_states[index], target_states[index]):
                    state.token_ids.append(token_id)
                    assert state.committed_length is not None
                    state.committed_length += 1
                    state.pre_verify = True
                    state.pending_window_size = 0
                runtime_states[index].delivered_tokens += 1
                runtime_states[index].verification_rounds += 1
                self._deliver_committed_tokens(index, request_params[index], local_states[index])

        admit_available()
        while (active_indices or pending_admission) and (max_rounds is None or round_count < max_rounds):
            if not active_indices:
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
                continue
            if (
                self.config.spec_rhythm_target_fallback_max_batch
                and len(active_indices) <= self.config.spec_rhythm_target_fallback_max_batch
            ):
                cycle_started = time.perf_counter()
                run_target_fallback(active_indices)
                torch.npu.synchronize()
                cycle_elapsed = time.perf_counter() - cycle_started
                last_cycle_ms = cycle_elapsed * 1000.0
                for index in active_indices:
                    runtime_states[index].add_decode_time(last_cycle_ms)
                finished_this_cycle = [
                    index for index in active_indices if _finished(local_states[index], self.eos_token_ids)
                ]
                for index in finished_this_cycle:
                    local_states[index].finished_decode_elapsed_ms = runtime_states[index].decode_elapsed_ms
                    completed_states[index] = local_states[index].clone()
                    invalidate_request(index)
                if finished_this_cycle:
                    finished_set = set(finished_this_cycle)
                    active_indices = [index for index in active_indices if index not in finished_set]
                    admit_available()
                round_count += 1
                continue
            cycle_started = time.perf_counter()
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
            )
            for name, value in rebuilt.items():
                key = f"spec_rhythm_linear_{name}"
                counters[key] = counters.get(key, 0) + value
            plan = controller.build_plan(
                active_indices,
                verification_budget=verification_roof,
                ready_candidate_counts={
                    index: payloads[controller.ready[index].proposal_id].verification_size
                    for index in active_indices
                    if index in controller.ready
                },
                priority=effective_priority,
                projected_wait_ms=max(last_cycle_ms * 2.0, 1e-6),
                priority_burst=self.config.spec_rhythm_priority_burst,
                merge_ready_homes=(has_slo_constraints and self.config.spec_rhythm_merge_ready_homes),
                max_target_requests=(
                    self.config.spec_rhythm_max_target_batch
                    if has_slo_constraints and self.config.spec_rhythm_max_target_batch > 0
                    else None
                ),
            )
            if plan.phase.value == "complete":
                raise RuntimeError("SpecRhythm has active requests but no verifiable or draftable proposal.")
            counters[f"spec_rhythm_{plan.phase.value}_steps"] += 1
            target_indices = list(plan.target_request_indices)
            target_payloads: list[NativeSpecRhythmDevicePayload] = []
            for request_index in target_indices:
                ticket = controller.ready[request_index]
                payload = payloads.get(ticket.proposal_id)
                if payload is None:
                    raise RuntimeError("SpecRhythm controller referenced a missing device payload.")
                payload.validate_for(local_states[request_index])
                target_payloads.append(payload)

            eager_candidates: list[int] = []
            if effective_eager_cap:
                projected_wait_ms = max(last_cycle_ms * 2.0, 1e-6)
                eager_candidates = [
                    index
                    for index in plan.eager_candidate_indices
                    if runtime_states[index].projected_progress_gap(projected_wait_ms) > 0
                    and runtime_states[index].urgency(projected_wait_ms) >= self.config.spec_rhythm_urgency_threshold
                    and runtime_states[index].expected_acceptance_benefit >= self.config.spec_rhythm_acceptance_floor
                ]
            normal_indices = list(plan.normal_draft_request_indices)
            work_indices: list[int] = []
            work_budgets: list[int] = []
            work_is_eager: list[bool] = []
            if normal_indices or eager_candidates:
                projected_wait_ms = max(last_cycle_ms * 2.0, 1e-6)
                budget_plan = shaper.shape(
                    plan_id=plan.plan_id,
                    normal_request_indices=normal_indices,
                    eager_request_indices=eager_candidates,
                    states=runtime_states,
                    projected_wait_ms=projected_wait_ms,
                    context_len=roof_context_len,
                    draft_token_budget=self.config.spec_rhythm_draft_token_budget,
                    batch_size=len(active_indices),
                    verification_roof=verification_roof,
                )
                counters["spec_rhythm_last_verification_roof"] = int(budget_plan.verification_roof)
                counters["spec_rhythm_unused_verification_tokens"] = int(budget_plan.unused_verification_tokens)
                normal_budgets = dict(budget_plan.normal_budgets)
                eager_budgets = {
                    index: min(value, effective_eager_cap)
                    for index, value in budget_plan.eager_budgets.items()
                    if min(value, effective_eager_cap) > 0
                }
                counters["spec_rhythm_unused_verification_tokens"] = max(
                    0,
                    int(budget_plan.verification_roof) - sum(normal_budgets.values()) - sum(eager_budgets.values()),
                )
                work_indices = [*normal_budgets, *eager_budgets]
                work_budgets = [
                    *normal_budgets.values(),
                    *eager_budgets.values(),
                ]
                work_is_eager = [False] * len(normal_budgets) + [True] * len(eager_budgets)
                counters["spec_rhythm_allocated_draft_tokens"] += sum(work_budgets)

            target_payload_by_index = {payload.ticket.request_index: payload for payload in target_payloads}
            work_verification_sizes = []
            work_verification_prefixes: list[torch.Tensor | None] = []
            for request_index, eager in zip(work_indices, work_is_eager):
                if eager:
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

            profile_this_round = round_count < self.config.profile_decode_steps
            if profile_this_round:
                torch.npu.synchronize()
            phase_started = time.perf_counter()
            draft_verification, draft_next, draft_confidence = (
                self._draft_spec_rhythm_device_batch(
                    draft_states,
                    work_indices,
                    work_budgets,
                    verification_sizes=work_verification_sizes,
                    verification_prefixes=work_verification_prefixes,
                )
                if work_indices
                else (None, None, None)
            )
            if self.is_draft and draft_next is not None:
                # Materialize the proposal matrix once per round.  Calling
                # ``cpu().tolist()`` separately for every request introduces
                # one stream-synchronizing NPU->CPU transfer per row.
                draft_next_cpu = draft_next.detach().cpu().tolist()
                for row, (request_index, budget) in enumerate(zip(work_indices, work_budgets)):
                    draft_states[request_index].token_ids.extend(int(value) for value in draft_next_cpu[row][:budget])
            if profile_this_round:
                torch.npu.synchronize()
            phase_elapsed = time.perf_counter() - phase_started
            phase_seconds["draft"] += phase_elapsed
            if profile_this_round and self.is_draft:
                decode_profile_seconds["draft_compute"] += phase_elapsed

            current_verification_sizes = [payload.verification_size for payload in target_payloads]
            phase_started = time.perf_counter()
            target_tokens, target_logits = (
                self._target_round_outputs_batch(
                    target_states,
                    target_indices,
                    current_verification_sizes,
                )
                if target_indices
                else (None, None)
            )
            if profile_this_round:
                torch.npu.synchronize()
            phase_elapsed = time.perf_counter() - phase_started
            phase_seconds["target"] += phase_elapsed
            if profile_this_round and not self.is_draft:
                decode_profile_seconds["target_compute"] += phase_elapsed

            trace_request_env = envs.VLLM_ASCEND_SPECRHYTHM_TRACE_REQUEST
            trace_requests = {
                int(value.strip()) for value in trace_request_env.split(",") if value.strip().lstrip("-").isdigit()
            }
            if (
                trace_requests
                and self.rank == self.topology.target_leader_rank
                and target_indices
                and target_tokens is not None
            ):
                target_offset = 0
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
                    target_offset += expected

            if profile_this_round and self.groups.is_verification_worker:
                wait_started = time.perf_counter()
                dist.barrier(group=self.groups.verification_group)
                torch.npu.synchronize()
                decode_profile_seconds["wait_sync"] += time.perf_counter() - wait_started
            phase_started = time.perf_counter()
            exchanged, confidences = self._exchange_spec_rhythm_device_proposals(
                draft_verification,
                draft_next,
                draft_confidence,
                tickets,
                work_verification_sizes,
            )
            if tickets:
                new_payloads = self._materialize_spec_rhythm_payloads(
                    tickets=tickets,
                    verification_sizes=work_verification_sizes,
                    local_verification=draft_verification,
                    local_next_windows=draft_next,
                    exchanged_message=exchanged,
                    draft_confidences=confidences,
                )
                controller.publish(tickets)
                payloads.update((payload.ticket.proposal_id, payload) for payload in new_payloads)
                counters["spec_rhythm_normal_proposals"] += sum(not value for value in work_is_eager)
                counters["spec_rhythm_eager_proposals"] += sum(work_is_eager)
            if profile_this_round and self.groups.is_verification_worker:
                torch.npu.synchronize()
            phase_elapsed = time.perf_counter() - phase_started
            phase_seconds["exchange"] += phase_elapsed
            if profile_this_round and self.groups.is_verification_worker:
                decode_profile_seconds["draft_to_target_communication"] += phase_elapsed

            finished_this_cycle: list[int] = []
            first_decode_indices: set[int] = set()
            if target_indices:
                temperatures = [target_states[index].temperature for index in target_indices]
                if self.is_draft:
                    current_message = None
                else:
                    verification_tensor = torch.cat([payload.verification_tokens for payload in target_payloads])
                    continuation_tensor = torch.full(
                        (len(target_payloads), self.gamma),
                        -1,
                        dtype=torch.long,
                        device=self.device,
                    )
                    for row, payload in enumerate(target_payloads):
                        continuation_tensor[row, : payload.ticket.gamma] = payload.next_tokens
                    current_message = torch.cat((verification_tensor, continuation_tensor.flatten()))
                phase_started = time.perf_counter()
                verdict = self._verify_target_tokens_batch(
                    target_tokens,
                    target_logits,
                    current_message,
                    current_verification_sizes,
                    temperatures,
                    top_ps=[target_states[index].top_p for index in target_indices],
                    top_ks=[target_states[index].top_k for index in target_indices],
                )
                if profile_this_round:
                    torch.npu.synchronize()
                phase_elapsed = time.perf_counter() - phase_started
                phase_seconds["verify"] += phase_elapsed
                if profile_this_round and not self.is_draft:
                    decode_profile_seconds["target_verdict"] += phase_elapsed
                current_next = torch.full(
                    (len(target_payloads), self.gamma),
                    -1,
                    dtype=torch.long,
                    device=self.device,
                )
                for row, payload in enumerate(target_payloads):
                    current_next[row, : payload.ticket.gamma] = payload.next_tokens
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
                if target_payloads:
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
                phase_started = time.perf_counter()
                accepted, corrections, synchronized_next = self._broadcast_device_round_result(
                    verdict,
                    current_message,
                    sum(current_verification_sizes),
                    len(target_indices),
                    current_next if self.is_draft else None,
                    replicated_target_verdict=replicated_target_verdict,
                    next_window_sizes=[payload.ticket.gamma for payload in target_payloads],
                    extra_device_values=proposal_confidences,
                    extra_device_scale=proposal_confidence_scale,
                    profile_phase_seconds=(decode_profile_seconds if profile_this_round else None),
                )
                phase_seconds["broadcast"] += time.perf_counter() - phase_started
                phase_started = time.perf_counter()
                for row, request_index in enumerate(target_indices):
                    payload = target_payloads[row]
                    expected = current_verification_sizes[row]
                    fully_accepted = accepted[row] == expected
                    was_first_decode_token = runtime_states[request_index].delivered_tokens == 0
                    trace_enabled = request_index in trace_requests and self.rank == self.topology.target_leader_rank
                    eager_ticket = controller.staged_eager.get(request_index)
                    if self.is_draft:
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
                    delivered = accepted[row] + int(not fully_accepted)
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
                phase_seconds["state_update"] += time.perf_counter() - phase_started
                if profile_this_round:
                    decode_profile_seconds["state_update"] += time.perf_counter() - phase_started
            else:
                # Warmup has no target verdict/correction collective. Keep every
                # rank at the same service-step boundary before rotating roles.
                dist.barrier()

            if profile_this_round:
                profiled_decode_steps += 1
            if npu_profiler is not None:
                npu_profiler.step()

            cycle_elapsed = time.perf_counter() - cycle_started
            elapsed_tensor = torch.tensor(
                [cycle_elapsed if self.rank == self.topology.target_leader_rank else 0.0],
                dtype=torch.float64,
                device=self.device,
            )
            dist.broadcast(elapsed_tensor, src=self.topology.target_leader_rank)
            last_cycle_ms = float(elapsed_tensor.cpu().item()) * 1000.0
            for index in active_indices:
                runtime_states[index].add_decode_time(last_cycle_ms)
            for index in first_decode_indices:
                # Exclude the cycle that emitted the first measured token from
                # the steady-state TPOT interval, just as vLLM excludes TTFT.
                runtime_states[index].decode_start_elapsed_ms = runtime_states[index].decode_elapsed_ms

            if finished_this_cycle:
                finished_set = set(finished_this_cycle)
                for request_index in finished_this_cycle:
                    local_states[request_index].finished_decode_elapsed_ms = runtime_states[
                        request_index
                    ].decode_elapsed_ms
                    completed_states[request_index] = local_states[request_index].clone()
                    invalidate_request(request_index)
                active_indices = [index for index in active_indices if index not in finished_set]
            refill_started = time.perf_counter()
            admit_available()
            phase_seconds["refill"] += time.perf_counter() - refill_started
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
            slo_attained = None if state.slo_tpot_ms is None else observed_tpot_ms <= state.slo_tpot_ms
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
                    "observed_tpot_ms": observed_tpot_ms,
                    "slo_attained": slo_attained,
                    "slo_goodput_tokens": (len(completion_token_ids) if slo_attained is not False else 0),
                    "round_count": round_count,
                    "decode_phase_seconds": dict(phase_seconds),
                    "spec_rhythm": dict(counters),
                    "elapsed_seconds": elapsed,
                    "prefill_elapsed_seconds": prefill_elapsed,
                    "decode_elapsed_seconds": decode_elapsed,
                    "cached_prompt_tokens": self.cache_allocation.num_cached_tokens[sequence_index],
                    "gamma": self.gamma,
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
        plan_list = list(plans)
        roots = [int(value) for value in root_token_ids]
        candidates = [list(map(int, row)) for row in draft_token_ids]
        request_ids = list(range(len(plan_list))) if sequence_ids is None else [int(value) for value in sequence_ids]
        if self.cache_allocation is None or self.cache_block_tables is None:
            raise RuntimeError("Allocate a target PEARL cache before tree forward")
        if (
            not plan_list
            or len(plan_list) != len(roots)
            or len(plan_list) != len(candidates)
            or len(plan_list) != len(request_ids)
        ):
            raise ValueError("tree plans, roots and candidate rows must have equal non-zero length")
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
        self._ensure_cache_capacity(sequence_ids, positions)
        input_ids, packed_positions, metadata = self.model.make_tree_attention_metadata(
            plan_list,
            roots,
            padded_candidates,
            self.cache_allocation.block_tables,
            sequence_ids=request_ids,
        )
        # Tree masks are dense request-local masks, so they cannot use the
        # linear FIA contract.  They can nevertheless be captured by the
        # ordinary target ACLGraph when the complete packed tree shape is
        # stable.  The graph runner owns runtime validation and falls back to
        # eager when capture/replay is unavailable or mismatches eager.
        tree_graph_enabled = (
            not self.config.enforce_eager and envs.VLLM_ASCEND_SPECRHYTHM_TREE_GRAPH
        )
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
        if stochastic and dist.is_initialized():
            # All target TP ranks must advance from the same sampled token.
            # Sampling independently on identical logits is still divergent
            # because each process owns a different RNG stream.
            dist.broadcast(
                target_tokens,
                src=self.topology.target_leader_rank,
                group=self.groups.target_group,
            )
        target_token_rows: list[torch.Tensor] = []
        target_query_rows: list[torch.Tensor] = []
        bonus_rows: list[torch.Tensor] = []
        cursor = 0
        active_node_counts: list[int] = []
        for plan, candidate_row, root_token in zip(plan_list, candidates, roots):
            node_count = plan.parent_indices.numel()
            active_count = int(getattr(plan, "candidate_budget", node_count))
            if not 1 <= active_count <= node_count:
                raise ValueError("tree candidate budget does not match its plan")
            active_node_counts.append(active_count)
            # The first target query predicts the root successor; each later
            # query predicts a tree node successor. The final query is exposed
            # separately as the bonus token expected by the verifier.
            row = target_tokens[cursor : cursor + node_count + 1]
            # Only active nodes are physically evaluated. Root + active node
            # outputs retain all possible accepted-branch frontier predictions.
            target_token_rows.append(row[:active_count])
            # Keep the root-query output plus every active node output.  The
            # device verifier uses this row to select the bonus token at the
            # actual accepted branch frontier (which may be a sibling).
            target_query_rows.append(row[: active_count + 1])
            bonus_rows.append(row[active_count])
            cursor += node_count + 1
        return {
            "target_token_ids": torch.cat(target_token_rows),
            "target_query_token_ids": torch.cat(target_query_rows),
            "bonus_token_ids": torch.stack(bonus_rows),
            "num_draft_tokens": active_node_counts,
            "target_logits": target_logits,
            "used_aclgraph": bool(tree_graph_enabled and self.graph_runner.last_target_execution.used_aclgraph),
            "attention_backend": attention_backend_identity(metadata),
            "cache_slot_mapping": metadata.slot_mapping,
            "query_count": int(input_ids.numel()),
            "tree_count": len(plan_list),
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
        """Pack one query per normal/eager request into a common model call."""
        parts = [
            self.model.make_tree_level_attention_metadata(
                plan, request_id, [node_index], [token_id], self.cache_block_tables
            )
            for plan, request_id, node_index, token_id in requests
        ]
        if len(parts) == 1:
            return parts[0]
        metadatas = [part[2] for part in parts]
        metadata = type(metadatas[0])(
            slot_mapping=torch.cat([value.slot_mapping for value in metadatas]),
            context_lens=torch.cat([value.context_lens for value in metadatas]),
            block_tables=torch.cat([value.block_tables for value in metadatas]),
            actual_seq_lengths_q=tuple(range(1, len(parts) + 1)),
            sequence_lens=tuple(int(value.context_lens[0]) for value in metadatas),
            request_block_tables=torch.cat([value.block_tables[:1] for value in metadatas]),
            attention_mask=torch.cat([value.attention_mask for value in metadatas]),
            use_fused_infer_attention=all(value.use_fused_infer_attention for value in metadatas),
            tree_attention=True,
            tree_attention_mask=torch.cat([value.attention_mask[:, None, None, :] for value in metadatas]),
        )
        return torch.cat([part[0] for part in parts]), torch.cat([part[1] for part in parts]), metadata

    @torch.inference_mode()
    def draft_tree_forward(
        self,
        plans: Sequence[TreeSpeculationPlan],
        root_token_ids: Sequence[int],
        sequence_ids: Sequence[int],
        eager_parent_sources: Mapping[int, tuple[TreeSpeculationPlan, Sequence[int]]] | None = None,
        *,
        draft_temperatures: Sequence[float] | None = None,
        draft_top_ps: Sequence[float] | None = None,
        draft_top_ks: Sequence[int] | None = None,
    ) -> dict[str, Any] | None:
        """Expand normal and eager trees in unified per-depth draft batches."""
        if not self.is_draft:
            return None
        plan_list = list(plans)
        roots = [int(value) for value in root_token_ids]
        request_ids = [int(value) for value in sequence_ids]
        if not plan_list or len(plan_list) != len(roots) or len(plan_list) != len(request_ids):
            raise ValueError("tree plans, roots and sequence IDs must have equal non-zero length")
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

        graph_runner = getattr(self, "graph_runner", None)
        use_graph = isinstance(graph_runner, NativeACLGraphRunner) and not getattr(
            getattr(self, "config", None), "enforce_eager", True
        )
        graph_calls = 0
        model_calls = 0

        def run_level(requests):
            nonlocal graph_calls, model_calls
            input_ids, positions, metadata = self._pack_tree_draft_level(requests)
            model_calls += 1
            if use_graph:
                logits = graph_runner.run_tree_logits(input_ids, positions, metadata, self.draft_vocab_size)
                graph_calls += int(graph_runner.last_target_execution.used_aclgraph)
                return logits
            hidden = self.model(input_ids, positions, metadata)
            return self.model.compute_logits(hidden)[:, : self.draft_vocab_size]

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
            requests = [
                (
                    work_plans[index],
                    request_ids[index],
                    depth_index - 1,
                    effective_roots[index] if depth_index == 0 else rows[index][depth_index - 1],
                )
                for index in level_rows
            ]
            logits = run_level(requests)
            max_width = min(max(work_plans[index].width for index in level_rows), int(logits.shape[-1]))
            sampled_rows: list[tuple[list[int], list[float]]] = []
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
        rows = [[row[0] if token < 0 else token for token in row] for row in rows]

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
        model_calls += 1
        if use_graph:
            graph_runner.run_tree_hidden(input_ids, positions, metadata)
            graph_calls += int(graph_runner.last_target_execution.used_aclgraph)
        else:
            self.model(input_ids, positions, metadata)
        active_counts = [
            int(getattr(plan, "candidate_budget", int(plan.width) * int(plan.depth))) for plan in plan_list
        ]
        parent_rows = [
            plan.parent_indices[:count].to(device=self.device) for plan, count in zip(plan_list, active_counts)
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
                device=self.device,
            ),
            "node_confidences": [
                torch.tensor(row, dtype=torch.float32, device=self.device) for row in node_confidences
            ],
            "cache_slot_mapping": metadata.slot_mapping,
            "query_count": int(input_ids.numel()),
            "model_calls": model_calls,
            "graph_calls": graph_calls,
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
        cache_slot_mapping: torch.Tensor,
        accepted_node_indices: torch.Tensor,
        num_draft_tokens: Sequence[int],
    ) -> None:
        """Compact accepted tree nodes into each request's linear KV suffix."""
        if cache_slot_mapping.ndim != 1:
            raise ValueError("cache_slot_mapping must be a flat tensor")
        cursor = 0
        for row, count in enumerate(num_draft_tokens):
            count = int(count)
            query_slots = cache_slot_mapping[cursor : cursor + count + 1]
            accepted = accepted_node_indices[row]
            source_slots = query_slots[1:]
            selected = accepted[accepted >= 0].to(torch.long)
            # Logical consecutive positions may cross non-consecutive KV
            # pages. Never derive physical destinations by root_slot + n.
            self._move_tree_cache_slots(source_slots.index_select(0, selected), source_slots[: selected.numel()])
            cursor += count + 1

    def _move_tree_cache_slots(self, source: torch.Tensor, destination: torch.Tensor) -> None:
        layer_caches = []
        for layer in self.model.layers:
            key_cache, value_cache = layer.self_attn.key_cache, layer.self_attn.value_cache
            if key_cache is None or value_cache is None:
                raise RuntimeError("Tree KV movement requires allocated layer caches")
            layer_caches.append((key_cache, value_cache))
        move_kv_cache_slots(layer_caches, source, destination)

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

    def _spec_rhythm_nonfinite_flag(self) -> torch.Tensor:
        """Aggregate graph-resident health flags without a host synchronization."""
        model = getattr(self, "model", None)
        flags: list[torch.Tensor] = []
        if model is not None:
            owners_and_names = [(model, "output_nonfinite"), (getattr(model, "lm_head", None), "logits_nonfinite")]
            owners_and_names.extend((layer.self_attn, "tree_cache_nonfinite") for layer in getattr(model, "layers", ()))
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

    def _prefill_and_sample_target_batch(
        self,
        prompts: list[list[int]],
        states: list[PearlPipelineState],
        sequence_ids: list[int] | None = None,
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
            # The target sample is the common committed frontier, but the draft
            # prefill still runs so its persistent KV cache is populated.
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
        else:
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
        if self.is_draft and self.gamma > 1:
            # SpecRhythm normally drafts a fixed-width window.  Capture the
            # whole greedy window before timing starts so the measured loop can
            # replay one ACLGraph instead of launching one model call per token.
            draft_use_fia = not self.config.draft_use_paged_attention
            draft_capture_size = min(
                len(sequence_ids),
                max(1, self.config.max_num_seqs // 2),
            )
            draft_sequence_ids = sequence_ids[:draft_capture_size]
            draft_input_ids = input_ids[:draft_capture_size]
            draft_first_positions = first_decode_positions[:draft_capture_size]
            position_tensors = []
            attention_metadatas = []
            for step in range(self.gamma):
                step_positions = [position + step for position in draft_first_positions]
                position_tensor, attention_metadata = self._prepare_attention_metadata(
                    draft_sequence_ids,
                    step_positions,
                    use_fused_infer_attention=draft_use_fia,
                )
                position_tensors.append(position_tensor)
                attention_metadatas.append(attention_metadata)
            self.graph_runner.run_draft_greedy(
                draft_input_ids,
                position_tensors,
                attention_metadatas,
                self.draft_vocab_size,
            )
        if not self.is_draft:
            self._run_device_packed_greedy(
                input_ids,
                sequence_ids,
                first_decode_positions,
                use_fused_infer_attention=target_use_fia,
            )
        if not self.is_draft and self.gamma > 1:
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

    def _precompile_decode_graphs(self) -> None:
        """Build stable paged-attention graphs before serving requests."""
        batch_sizes = [self.config.max_num_seqs]
        if self.config.enable_continuous_batching and self.config.max_num_seqs > 1:
            batch_sizes.append(max(1, self.config.max_num_seqs // 2))
        batch_sizes = sorted(set(batch_sizes))
        prompts = [[0] for _ in range(max(batch_sizes))]
        self._allocate_cache(prompts, enable_prefix_caching=False)
        try:
            states = [
                PearlPipelineState(
                    [0],
                    prompt_length=1,
                    temperature=0.0,
                    max_tokens=self.config.max_tokens,
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

            if self.is_draft:
                for batch_size in batch_sizes:
                    self.graph_runner.set_expected_fia_batch_size(batch_size)
                    batch_states = states[:batch_size]
                    active_indices = list(range(batch_size))
                    # Capture plus one init-time replay keeps graph setup out of
                    # the first measured request.
                    self._draft_round_device_batch(batch_states, active_indices)
                    self._draft_round_device_batch(batch_states, active_indices)
            else:
                # Normal paged replay may pad into an existing larger graph.
                # Ascending compilation guarantees an exact entry per bucket.
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
            self.precompiled_decode_batch_sizes = frozenset(batch_sizes)
        finally:
            self._release_cache()

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

    def _draft_spec_rhythm_device_batch(
        self,
        states: list[PearlPipelineState],
        active_indices: list[int],
        draft_budgets: Sequence[int],
        *,
        verification_sizes: Sequence[int] | None = None,
        verification_prefixes: Sequence[torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Draft variable windows and report mean softmax confidence per row."""

        if not self.is_draft:
            return None, None, None
        budgets = [int(value) for value in draft_budgets]
        if len(budgets) != len(active_indices) or any(value <= 0 or value > self.gamma for value in budgets):
            raise ValueError("SpecRhythm draft budgets must be in [1, gamma].")
        input_ids = torch.tensor(
            [states[index].token_ids[-1] for index in active_indices],
            dtype=torch.long,
            device=self.device,
        )
        first_positions = [len(states[index].token_ids) - 1 for index in active_indices]
        draft_temperatures = [states[index].draft_temperature for index in active_indices]
        draft_top_ps = [states[index].draft_top_p for index in active_indices]
        draft_top_ks = [states[index].draft_top_k for index in active_indices]
        next_windows = torch.full(
            (len(active_indices), self.gamma),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        confidence_sums = torch.zeros(len(active_indices), dtype=torch.float32, device=self.device)
        draft_use_fia = not getattr(self.config, "draft_use_paged_attention", False)
        if getattr(self.config, "enable_spec_rhythm", False) and getattr(
            self.config, "spec_rhythm_stable_graphs", True
        ):
            draft_use_fia = False
        prefixes = [None] * len(active_indices) if verification_prefixes is None else list(verification_prefixes)
        graph_batch_size = max(
            1,
            getattr(self.config, "max_num_seqs", len(active_indices)) // 2,
        )
        use_full_draft_graph = (
            not self.config.enforce_eager
            and self.gamma <= 16
            and not draft_use_fia
            # A SpecRhythm step drafts one of the two home batches.  Pad its
            # active rows to the fixed home-batch capacity so every round can
            # replay the same gamma-step graph.  Online admission changes the
            # resident rows, not this graph shape; the paged-attention task is
            # refreshed before every replay.
            and len(active_indices) <= graph_batch_size
            and all(value == 0 for value in draft_temperatures)
            and hasattr(self, "graph_runner")
            and hasattr(self.graph_runner, "run_draft_greedy")
        )
        if use_full_draft_graph:
            position_tensors = []
            attention_metadatas = []
            graph_input_ids = None
            for step in range(self.gamma):
                step_positions = [position + step for position in first_positions]
                position_tensor, attention_metadata = self._prepare_attention_metadata(
                    active_indices,
                    step_positions,
                    use_fused_infer_attention=draft_use_fia,
                )
                padded = NativeACLGraphRunner._pad_inputs(
                    input_ids,
                    position_tensor,
                    attention_metadata,
                    graph_batch_size,
                )
                padded_input_ids, padded_positions, padded_metadata = padded
                graph_input_ids = padded_input_ids
                position_tensors.append(padded_positions)
                attention_metadatas.append(padded_metadata)
            assert graph_input_ids is not None
            next_windows = self.graph_runner.run_draft_greedy(
                graph_input_ids,
                position_tensors,
                attention_metadatas,
                self.draft_vocab_size,
            )[: len(active_indices)].clone()
            # Confidence only affects optional eager prioritization.  Greedy
            # default SpecRhythm does not branch on it, and a constant device
            # value avoids materializing logits from the graph output.
            confidence_sums.fill_(1.0 * self.gamma)
        else:
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

        confidence = confidence_sums / torch.tensor(budgets, dtype=torch.float32, device=self.device)
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

    def _exchange_spec_rhythm_device_proposals(
        self,
        verification_window: torch.Tensor | None,
        next_windows: torch.Tensor | None,
        confidences: torch.Tensor | None,
        tickets: Sequence[SpecRhythmProposalTicket],
        verification_sizes: Sequence[int],
    ) -> tuple[torch.Tensor | None, Sequence[float | torch.Tensor]]:
        """Broadcast a self-describing variable-offset proposal envelope."""

        count = len(tickets)
        if count == 0:
            return None, []
        if len(verification_sizes) != count:
            raise RuntimeError("SpecRhythm proposal metadata has inconsistent row counts.")
        metadata_width = 7
        verification_size = sum(verification_sizes)
        message_size = metadata_width * count + verification_size + count * self.gamma
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
                message = torch.cat((metadata.reshape(-1), verification_window, next_windows.flatten()))
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
            confidence_values = []
        return message, confidence_values

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
            continuations = exchanged_message[metadata_size + verification_size :].reshape(count, self.gamma)
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
                (
                    not self.config.target_use_paged_attention
                    or envs.VLLM_ASCEND_SPECRHYTHM_STEPWISE_TARGET_FIA
                ),
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
            if target_tokens is None or target_tokens.shape != (verification_size,):
                raise RuntimeError("PEARL target and draft verification windows differ in length.")
            if self.config.spec_rhythm_cpu_verdict:
                return _build_greedy_verdict_cpu(
                    target_tokens,
                    draft_message[:verification_size],
                    verification_sizes,
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
                target_tokens,
                draft_message[:verification_size],
                *layout,
                self.gamma,
                uniform_width=(
                    self.gamma
                    if all(size == self.gamma for size in verification_sizes)
                    else (1 if all(size == 1 for size in verification_sizes) else None)
                ),
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
        self._last_device_round_extra_values: list[float] = []
        if extra_device_values is not None:
            if extra_device_values.ndim != 1:
                raise ValueError("Extra PEARL round values must be a one-dimensional tensor.")
            if extra_device_values.numel() != batch_size:
                raise ValueError("Extra PEARL round values must match the target batch size.")
            extra_device_values = extra_device_values.to(device=self.device)
        is_target_rank = self.rank in self.topology.target_ranks
        if is_target_rank:
            if verdict is None:
                raise RuntimeError("A PEARL target rank did not produce a round verdict.")
            verdict_result = verdict.flatten()
        else:
            verdict_result = torch.empty(batch_size * 2, dtype=torch.long, device=self.device)

        communication_started = time.perf_counter()
        participated_in_broadcast = False
        broadcast_work = None
        if replicated_target_verdict:
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

        if self.is_draft:
            if local_next_windows is None or local_next_windows.shape != (batch_size, self.gamma):
                raise RuntimeError("A PEARL draft rank did not retain its local continuation window.")
            continuation = local_next_windows.flatten()
        else:
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
        continuation_values = continuation if self.is_draft else continuation.cpu().tolist()
        if broadcast_work is not None:
            broadcast_work.wait()
        if profile_phase_seconds is not None and participated_in_broadcast:
            torch.npu.synchronize()
            profile_phase_seconds["target_to_draft_communication"] += time.perf_counter() - communication_started
        materialize_started = time.perf_counter()
        verdict_values = verdict_result.cpu().tolist()
        extra_size = 0 if extra_device_values is None else extra_device_values.numel()
        extra_values = extra_device_values.cpu().tolist() if extra_device_values is not None else []
        if extra_size:
            self._last_device_round_extra_values = [float(value) * extra_device_scale for value in extra_values]
        if profile_phase_seconds is not None:
            profile_phase_seconds["wait_sync"] += time.perf_counter() - materialize_started
        accepted = [int(value) for value in verdict_values[::2]]
        corrections = [None if value == -1 else int(value) for value in verdict_values[1::2]]
        window_sizes = (
            [self.gamma] * batch_size if next_window_sizes is None else [int(value) for value in next_window_sizes]
        )
        if len(window_sizes) != batch_size or any(value <= 0 or value > self.gamma for value in window_sizes):
            raise RuntimeError("PEARL synchronized continuation lengths are invalid.")
        next_windows = []
        for row in range(batch_size):
            values = continuation_values[row * self.gamma : row * self.gamma + window_sizes[row]]
            next_windows.append(values if self.is_draft else [int(value) for value in values])
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
        "--spec-rhythm-online-prefill",
        action="store_true",
        help="Gate continuous-batching admission on each request's arrival_ts.",
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
    parser.add_argument("--target-use-paged-attention", action="store_true")
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
        max_num_seqs=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_kvcache_blocks=args.num_kvcache_blocks,
        max_aclgraph_entries=args.max_aclgraph_entries,
        target_verification_graph_buckets=args.target_verification_graph_buckets,
        enable_continuous_batching=(args.enable_continuous_batching or args.enable_spec_rhythm),
        enable_preemptive_scheduling=(args.enable_preemptive_scheduling or args.enable_spec_rhythm),
        enable_spec_rhythm=args.enable_spec_rhythm,
        spec_rhythm_online_prefill=args.spec_rhythm_online_prefill,
        spec_rhythm_merge_ready_homes=args.spec_rhythm_merge_ready_homes,
        spec_rhythm_priority_mode=args.spec_rhythm_priority_mode,
        spec_rhythm_priority_burst=args.spec_rhythm_priority_burst,
        spec_rhythm_target_fallback_max_batch=args.spec_rhythm_target_fallback_max_batch,
        spec_rhythm_stable_graphs=args.spec_rhythm_stable_graphs,
        spec_rhythm_min_gamma=args.spec_rhythm_min_gamma,
        spec_rhythm_max_eager_tokens=args.spec_rhythm_max_eager_tokens,
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
        target_use_paged_attention=args.target_use_paged_attention,
        draft_use_production_rope=args.draft_use_production_rope,
        target_use_production_rope=args.target_use_production_rope,
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
