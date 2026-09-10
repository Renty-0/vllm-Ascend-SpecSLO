# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Upstream-compatible nano-PEARL controller backed by Ascend HCCL workers."""

from __future__ import annotations

import atexit
import logging
import os
import socket
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from multiprocessing.connection import Connection
from multiprocessing.connection import wait as wait_for_connections
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoConfig, AutoTokenizer

from vllm_ascend.spec_decode.pearl.native_engine import (
    TARGET_VERIFICATION_GRAPH_BUCKETS,
    NativePearlConfig,
    NativePearlEngine,
    SamplingParams,
    _normalize_target_graph_post_counts,
    _set_default_npu_environment,
)
from vllm_ascend.spec_decode.pearl.native_model import (
    PAGED_ATTENTION_BLOCK_SIZE,
    SUPPORTED_NATIVE_ARCHITECTURES,
)
from vllm_ascend.spec_decode.pearl.qwen_pair import validate_model_pair
from vllm_ascend.spec_decode.pearl.roofline import ProfiledRoofline, normalize_roofline

logger = logging.getLogger("vllm_ascend.spec_decode.pearl")


class _TokenCommitDeliveryError(RuntimeError):
    """Callback failed after generation was drained; consumed work is not retried."""


@dataclass(frozen=True)
class PEARLModelGroupConfig:
    """Read-only compatibility view of an upstream nano-PEARL model group."""

    model: str
    tensor_parallel_size: int
    devices: list[int]
    group_name: str
    hf_config: Any
    eos: int | list[int] | None
    master_rank: int


@dataclass(frozen=True)
class PEARLConfig:
    """Configuration surface compatible with upstream nano-PEARL."""

    draft_model_path: str
    target_model_path: str
    draft_tensor_parallel_size: int = 2
    target_tensor_parallel_size: int = 2
    draft_group_name: str = "draft_group"
    target_group_name: str = "target_group"
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    prefill_chunk_size: int | None = None
    max_num_queued_seqs: int | None = None
    max_model_len: int = 4096
    draft_dtype: str = "auto"
    target_dtype: str = "auto"
    gpu_memory_utilization: float = 0.9
    kvcache_block_size: int = PAGED_ATTENTION_BLOCK_SIZE
    num_kvcache_blocks: int = -1
    max_aclgraph_entries: int = 32
    target_verification_graph_buckets: int = TARGET_VERIFICATION_GRAPH_BUCKETS
    target_verification_graph_post_counts: tuple[tuple[int, tuple[int, ...]], ...] = ()
    auto_gamma_profile_sequence_length: int = 256
    enforce_eager: bool = False
    gamma: int = -1
    enable_prefix_caching: bool = True
    enable_continuous_batching: bool = False
    enable_preemptive_scheduling: bool = False
    enable_spec_rhythm: bool = False
    # With an arrival-aware manifest, prefill only the initial decode bucket;
    # later requests are prefetched when they enter a free slot.
    spec_rhythm_online_prefill: bool = False
    # Keep the paper's alternating dual-batch schedule by default.  Merging
    # both logical homes into one target forward is an opt-in throughput probe
    # because it removes the rolling-eager scheduling window.
    spec_rhythm_merge_ready_homes: bool = False
    spec_rhythm_priority_mode: bool = False
    spec_rhythm_priority_burst: int = 2
    spec_rhythm_target_fallback_max_batch: int = 0
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
    # Opt-in fixed-shape tree path.  Width/depth one keeps the linear PEARL
    # path; larger values enable paper-style individual tree budgets.
    spec_rhythm_tree_width: int = 1
    spec_rhythm_tree_depth: int = 1
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
    enable_mc2: bool = False
    mc2_profile: Mapping[str, Any] | str | None = None
    seed: int | None = None
    worker_timeout_seconds: float = 300.0
    draft_config: PEARLModelGroupConfig = field(init=False, repr=False)
    target_config: PEARLModelGroupConfig = field(init=False, repr=False)
    eos: int | list[int] | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_verification_graph_post_counts",
            _normalize_target_graph_post_counts(self.target_verification_graph_post_counts),
        )
        if self.draft_tensor_parallel_size <= 0 or self.target_tensor_parallel_size <= 0:
            raise ValueError("PEARL tensor-parallel sizes must be positive.")
        if self.world_size > 8:
            raise ValueError("Upstream nano-PEARL supports at most eight model workers.")
        if self.max_model_len <= 0 or self.max_num_seqs <= 0:
            raise ValueError("PEARL max_model_len and max_num_seqs must be positive.")
        supported_dtypes = {"auto", "bfloat16", "float16"}
        if self.draft_dtype not in supported_dtypes or self.target_dtype not in supported_dtypes:
            raise ValueError("PEARL model dtype must be auto, bfloat16, or float16.")
        prefill_limit = self.max_num_queued_seqs or self.max_num_seqs
        if self.prefill_chunk_size is not None and not 0 < self.prefill_chunk_size <= prefill_limit:
            raise ValueError("PEARL prefill_chunk_size must fit the queued request capacity.")
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
                    model=self.target_model_path,
                    target_tp_size=self.target_tensor_parallel_size,
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
            raise ValueError(
                f"The Ascend PEARL paged-attention backend requires kvcache_block_size={PAGED_ATTENTION_BLOCK_SIZE}."
            )
        if self.num_kvcache_blocks == 0 or self.num_kvcache_blocks < -1:
            raise ValueError("PEARL num_kvcache_blocks must be positive, or -1 for automatic sizing.")
        if self.max_aclgraph_entries <= 0:
            raise ValueError("PEARL max_aclgraph_entries must be positive.")
        if self.target_verification_graph_buckets <= 0:
            raise ValueError("PEARL target_verification_graph_buckets must be positive.")
        if self.auto_gamma_profile_sequence_length <= 0:
            raise ValueError("PEARL auto-gamma profile sequence length must be positive.")
        if self.profile_decode_steps < 0:
            raise ValueError("PEARL profile_decode_steps must be non-negative.")
        if self.stop_after_profiled_decode_steps and self.profile_decode_steps == 0:
            raise ValueError("PEARL profiling-only execution requires profile_decode_steps to be positive.")
        if self.gamma == 0 or self.gamma < -1:
            raise ValueError("PEARL gamma must be positive, or -1 for automatic selection.")
        if self.seed is not None and self.seed < 0:
            raise ValueError("PEARL seed must be non-negative.")
        if self.worker_timeout_seconds <= 0:
            raise ValueError("PEARL worker_timeout_seconds must be positive.")

        draft_config = AutoConfig.from_pretrained(self.draft_model_path)
        target_config = AutoConfig.from_pretrained(self.target_model_path)
        for name, model_config in (("draft", draft_config), ("target", target_config)):
            architecture = model_config.architectures[0]
            if architecture not in SUPPORTED_NATIVE_ARCHITECTURES:
                raise ValueError(
                    f"Unsupported {name} architecture {architecture!r}; expected one of "
                    f"{sorted(SUPPORTED_NATIVE_ARCHITECTURES)}."
                )
        if _eos_set(draft_config.eos_token_id) != _eos_set(target_config.eos_token_id):
            raise ValueError("PEARL draft and target models must use identical EOS token IDs.")
        draft_devices = list(range(self.draft_tensor_parallel_size))
        target_devices = list(
            range(
                self.draft_tensor_parallel_size,
                self.draft_tensor_parallel_size + self.target_tensor_parallel_size,
            )
        )
        object.__setattr__(
            self,
            "draft_config",
            PEARLModelGroupConfig(
                model=self.draft_model_path,
                tensor_parallel_size=self.draft_tensor_parallel_size,
                devices=draft_devices,
                group_name=self.draft_group_name,
                hf_config=draft_config,
                eos=draft_config.eos_token_id,
                master_rank=draft_devices[0],
            ),
        )
        object.__setattr__(
            self,
            "target_config",
            PEARLModelGroupConfig(
                model=self.target_model_path,
                tensor_parallel_size=self.target_tensor_parallel_size,
                devices=target_devices,
                group_name=self.target_group_name,
                hf_config=target_config,
                eos=target_config.eos_token_id,
                master_rank=target_devices[0],
            ),
        )
        object.__setattr__(self, "eos", draft_config.eos_token_id)

    @property
    def world_size(self) -> int:
        return self.draft_tensor_parallel_size + self.target_tensor_parallel_size

    def to_native(self) -> NativePearlConfig:
        return NativePearlConfig(
            draft_model=self.draft_model_path,
            target_model=self.target_model_path,
            draft_tp_size=self.draft_tensor_parallel_size,
            target_tp_size=self.target_tensor_parallel_size,
            gamma=self.gamma,
            max_model_len=self.max_model_len,
            max_tokens=self.max_model_len,
            draft_dtype=self.draft_dtype,
            target_dtype=self.target_dtype,
            max_num_seqs=self.max_num_seqs,
            prefill_chunk_size=self.prefill_chunk_size,
            max_num_queued_seqs=self.max_num_queued_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            gpu_memory_utilization=self.gpu_memory_utilization,
            kvcache_block_size=self.kvcache_block_size,
            num_kvcache_blocks=self.num_kvcache_blocks,
            max_aclgraph_entries=self.max_aclgraph_entries,
            target_verification_graph_buckets=self.target_verification_graph_buckets,
            target_verification_graph_post_counts=self.target_verification_graph_post_counts,
            auto_gamma_profile_sequence_length=self.auto_gamma_profile_sequence_length,
            enable_prefix_caching=self.enable_prefix_caching,
            enable_continuous_batching=self.enable_continuous_batching,
            enable_preemptive_scheduling=self.enable_preemptive_scheduling,
            enable_spec_rhythm=self.enable_spec_rhythm,
            spec_rhythm_online_prefill=self.spec_rhythm_online_prefill,
            spec_rhythm_merge_ready_homes=self.spec_rhythm_merge_ready_homes,
            spec_rhythm_priority_mode=self.spec_rhythm_priority_mode,
            spec_rhythm_priority_burst=self.spec_rhythm_priority_burst,
            spec_rhythm_target_fallback_max_batch=self.spec_rhythm_target_fallback_max_batch,
            spec_rhythm_max_target_batch=self.spec_rhythm_max_target_batch,
            spec_rhythm_min_gamma=self.spec_rhythm_min_gamma,
            spec_rhythm_max_eager_tokens=self.spec_rhythm_max_eager_tokens,
            spec_rhythm_urgency_threshold=self.spec_rhythm_urgency_threshold,
            spec_rhythm_acceptance_floor=self.spec_rhythm_acceptance_floor,
            spec_rhythm_acceptance_ema_alpha=self.spec_rhythm_acceptance_ema_alpha,
            spec_rhythm_cpu_verdict=self.spec_rhythm_cpu_verdict,
            spec_rhythm_roofline=self.spec_rhythm_roofline,
            spec_rhythm_verification_budget=self.spec_rhythm_verification_budget,
            spec_rhythm_draft_token_budget=self.spec_rhythm_draft_token_budget,
            spec_rhythm_tree_width=self.spec_rhythm_tree_width,
            spec_rhythm_tree_depth=self.spec_rhythm_tree_depth,
            spec_rhythm_stable_graphs=self.spec_rhythm_stable_graphs,
            pad_finished_requests=self.pad_finished_requests,
            draft_use_paged_attention=self.draft_use_paged_attention,
            target_use_paged_attention=self.target_use_paged_attention,
            draft_use_production_rope=self.draft_use_production_rope,
            target_use_production_rope=self.target_use_production_rope,
            precompile_decode_graphs=self.precompile_decode_graphs,
            enable_cpu_binding=self.enable_cpu_binding,
            profile_decode_steps=self.profile_decode_steps,
            stop_after_profiled_decode_steps=self.stop_after_profiled_decode_steps,
            enforce_eager=self.enforce_eager,
            enable_mc2=self.enable_mc2,
            mc2_profile=self.mc2_profile,
            seed=self.seed,
        )


class PEARLEngine:
    """Spawn and control the draft and target HCCL workers like nano-PEARL."""

    def __init__(self, config: PEARLConfig) -> None:
        self.config = config
        validate_model_pair(config.draft_model_path, config.target_model_path)
        # Worker processes must inherit queue mode before importing torch-npu;
        # setting it inside NativePearlEngine is too late for this runtime knob.
        _set_default_npu_environment(config.target_tensor_parallel_size)
        self.tokenizer = AutoTokenizer.from_pretrained(config.draft_model_path, use_fast=True)
        self._requests: list[tuple[int, list[int], SamplingParams]] = []
        self._next_request_id = 0
        self.last_metrics: list[dict[str, Any]] = []
        self.last_worker_metrics: list[dict[str, int | float]] = []
        self.last_worker_metrics_by_chunk: list[dict[str, Any]] = []
        self._closed = False
        self._processes: list[mp.Process] = []
        self._connections: list[Connection] = []
        self._admission_connections: list[Connection] = []
        self._live_lock = threading.Lock()
        self._live_epoch_counter = 0
        self._live_epoch: int | None = None
        self._live_accepting = False
        self._live_admission_seq = 0
        self._live_total_requests = 0
        self._live_sampling_signature: tuple[bool, bool] | None = None
        try:
            self._start_workers()
        except Exception:
            self.exit()
            raise
        atexit.register(self.exit)

    def _start_workers(self) -> None:
        context = mp.get_context("spawn")
        master_port = _reserve_local_port()
        native_config = self.config.to_native()
        for rank in range(self.config.world_size):
            parent_connection, child_connection = context.Pipe(duplex=True)
            child_admission_connection, parent_admission_connection = context.Pipe(duplex=False)
            process = context.Process(
                target=_pearl_worker,
                args=(native_config, rank, master_port, child_connection, child_admission_connection),
                daemon=True,
            )
            process.start()
            child_connection.close()
            child_admission_connection.close()
            self._processes.append(process)
            self._connections.append(parent_connection)
            self._admission_connections.append(parent_admission_connection)
        replies = self._receive_all("worker initialization")
        if any(reply[0] != "ready" for reply in replies):
            raise RuntimeError(f"Unexpected PEARL worker initialization replies: {replies!r}")
        self.last_worker_metrics = [reply[2] for reply in replies if len(reply) > 2]

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
        request_id: str | int | None = None,
        arrival_ts: float | None = None,
        slo_tpot_ms: float | None = None,
        slo_class: str | None = None,
        per_request_gamma: int | None = None,
    ) -> None:
        if sampling_params is None:
            sampling_params = SamplingParams()
        if not isinstance(sampling_params, SamplingParams):
            raise TypeError("PEARL sampling_params must be a SamplingParams value.")
        sampling_params = replace(
            sampling_params,
            request_id=(request_id if request_id is not None else sampling_params.request_id),
            arrival_ts=(arrival_ts if arrival_ts is not None else sampling_params.arrival_ts),
            slo_tpot_ms=(slo_tpot_ms if slo_tpot_ms is not None else sampling_params.slo_tpot_ms),
            slo_class=(slo_class if slo_class is not None else sampling_params.slo_class),
            spec_rhythm_max_gamma=(
                per_request_gamma if per_request_gamma is not None else sampling_params.spec_rhythm_max_gamma
            ),
        )
        if isinstance(prompt, str):
            formatted_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            token_ids = list(self.tokenizer.encode(formatted_prompt))
        else:
            token_ids = [int(token_id) for token_id in prompt]
        if not token_ids:
            raise ValueError("PEARL requests require a non-empty prompt.")
        if len(token_ids) + sampling_params.max_tokens > self.config.max_model_len:
            raise ValueError("Prompt plus PEARL completion exceeds max_model_len.")
        self._requests.append((self._next_request_id, token_ids, sampling_params))
        self._next_request_id += 1

    def log(self, content: str) -> None:
        """Emit one controller message from every PEARL worker."""
        self._send_all(("log", str(content), None, None))
        replies = self._receive_all("worker logging")
        if any(reply[0] != "logged" for reply in replies):
            raise RuntimeError(f"Unexpected PEARL worker log replies: {replies!r}")

    def configure_decode_profiling(
        self,
        profile_decode_steps: int,
        stop_after_profiled_decode_steps: bool = False,
    ) -> None:
        """Configure synchronized decode profiling on every persistent worker."""
        if profile_decode_steps < 0:
            raise ValueError("PEARL profile_decode_steps must be non-negative.")
        if stop_after_profiled_decode_steps and profile_decode_steps == 0:
            raise ValueError("PEARL profiling-only execution requires profile_decode_steps to be positive.")
        self._send_all(
            (
                "configure_decode_profiling",
                profile_decode_steps,
                stop_after_profiled_decode_steps,
                None,
            )
        )
        replies = self._receive_all("worker profiling configuration")
        if any(reply[0] != "configured" for reply in replies):
            raise RuntimeError(f"Unexpected PEARL worker profiling replies: {replies!r}")

    def generate(self, *, on_token_commit: Callable[[dict[str, Any]], None] | None = None):
        """Generate queued requests, optionally delivering each guarded commit.

        The callback runs in the controller process while workers are still
        decoding. It receives ``request_index``, ``request_id``, newly committed
        ``token_ids``, ``finished`` and ``elapsed_seconds``. Callbacks should
        return promptly; slow callbacks apply backpressure to the worker pipe.
        The ordinary final batch return value remains unchanged.
        """
        if on_token_commit is not None and not callable(on_token_commit):
            raise TypeError("on_token_commit must be callable when supplied")
        if on_token_commit is not None and not getattr(self.config, "enable_spec_rhythm", False):
            raise ValueError("Guarded token streaming requires enable_spec_rhythm=True")
        if self.config.gamma != -1 and any(
            len(prompt) + params.max_tokens + self.config.gamma > self.config.max_model_len
            for _, prompt, params in self._requests
        ):
            raise ValueError("Prompt plus the PEARL verification window exceeds max_model_len.")
        return self._generate("pearl", on_token_commit=on_token_commit)

    def generate_live(self, *, on_token_commit: Callable[[dict[str, Any]], None] | None = None):
        """Run one bounded online-admission epoch on persistent workers.

        New requests may be sent with :meth:`admit_live_requests` while this
        call is decoding.  The epoch closes atomically when every resident
        request finishes and no controller admission is pending.
        """
        if not (
            self.config.enable_spec_rhythm
            and self.config.enable_continuous_batching
            and self.config.enable_preemptive_scheduling
            and (self.config.spec_rhythm_tree_width > 1 or self.config.spec_rhythm_tree_depth > 1)
        ):
            raise ValueError("Live admission requires the complete SpecRhythm tree scheduler")
        if on_token_commit is not None and not callable(on_token_commit):
            raise TypeError("on_token_commit must be callable when supplied")
        return self._generate("pearl", on_token_commit=on_token_commit, live=True)

    def admit_live_requests(
        self,
        requests: list[tuple[list[int], SamplingParams]],
    ) -> None:
        """Atomically publish new HTTP requests to an active worker epoch."""
        if not requests:
            return
        normalized: list[tuple[list[int], SamplingParams]] = []
        for prompt, params in requests:
            if not isinstance(params, SamplingParams):
                raise TypeError("Live PEARL admission requires SamplingParams values")
            tokens = [int(token_id) for token_id in prompt]
            tail = self.config.spec_rhythm_tree_width * self.config.spec_rhythm_tree_depth + 1
            if not tokens or len(tokens) + params.max_tokens + tail > self.config.max_model_len:
                raise ValueError("Live PEARL prompt plus completion exceeds max_model_len")
            normalized.append((tokens, params))
        signature = (
            normalized[0][1].temperature > 0,
            normalized[0][1].draft_temperature > 0,
        )
        if any((params.temperature > 0, params.draft_temperature > 0) != signature for _, params in normalized):
            raise ValueError("One live PEARL admission must use a homogeneous sampling mode")
        with self._live_lock:
            if not self._live_accepting or self._live_epoch is None:
                raise RuntimeError("The PEARL live epoch is no longer accepting requests")
            if signature != self._live_sampling_signature:
                raise ValueError("A live PEARL epoch cannot mix greedy and stochastic sampling modes")
            capacity = self.config.max_num_queued_seqs or self.config.max_num_seqs
            if self._live_total_requests + len(normalized) > capacity:
                raise RuntimeError("The PEARL live epoch has reached its reserved request capacity")
            self._live_admission_seq += 1
            message = (
                "admit",
                self._live_epoch,
                self._live_admission_seq,
                normalized,
            )
            for connection in self._admission_connections:
                connection.send(message)
            self._live_total_requests += len(normalized)

    def abort_live_requests(self, request_ids: Sequence[str | int]) -> bool:
        """Abort live requests at the next collective-safe cycle fence.

        Returns ``False`` when no live epoch is open. The controller uses one
        ordered admission/control sequence, so an abort cannot overtake an
        earlier admission or race the epoch close marker.
        """

        normalized = tuple(str(value) for value in request_ids)
        if not normalized:
            return True
        with self._live_lock:
            if not self._live_accepting or self._live_epoch is None:
                return False
            self._live_admission_seq += 1
            message = (
                "abort",
                self._live_epoch,
                self._live_admission_seq,
                normalized,
            )
            for connection in self._admission_connections:
                connection.send(message)
        return True

    def AR_generate(self):
        output_text, num_tokens, _, elapsed = self._generate("target_ar")
        return output_text, num_tokens, None, elapsed

    def bench_generate(self, num_pearl_steps: int = 100):
        if num_pearl_steps <= 0:
            raise ValueError("num_pearl_steps must be positive.")
        if self.config.gamma != -1 and any(
            len(prompt) + 1 + (num_pearl_steps + 2) * self.config.gamma > self.config.max_model_len
            for _, prompt, _ in self._requests
        ):
            raise ValueError("The fixed PEARL benchmark steps exceed max_model_len.")
        return self._generate("bench", num_pearl_steps=num_pearl_steps)

    def _generate(
        self,
        mode: str,
        *,
        num_pearl_steps: int | None = None,
        on_token_commit: Callable[[dict[str, Any]], None] | None = None,
        live: bool = False,
    ):
        if not self._requests:
            self.last_metrics = []
            self.last_worker_metrics = []
            self.last_worker_metrics_by_chunk = []
            return [], [], (() if mode != "target_ar" else None), 0.0
        requests = self._requests
        self._requests = []
        live_epoch: int | None = None
        if live:
            signatures = {(params.temperature > 0, params.draft_temperature > 0) for _, _, params in requests}
            if len(signatures) != 1:
                self._requests = requests + self._requests
                raise ValueError("A live PEARL epoch requires one homogeneous sampling mode")
            with self._live_lock:
                if self._live_epoch is not None:
                    self._requests = requests + self._requests
                    raise RuntimeError("A PEARL live epoch is already running")
                self._live_epoch_counter += 1
                live_epoch = self._live_epoch_counter
                self._live_epoch = live_epoch
                self._live_accepting = True
                self._live_admission_seq = 0
                self._live_total_requests = len(requests)
                self._live_sampling_signature = next(iter(signatures))
        outputs: list[tuple[int, dict[str, Any]]] = []
        total_elapsed = 0.0
        self.last_worker_metrics_by_chunk = []
        try:
            request_chunks = (
                [requests]
                if live or (mode == "pearl" and getattr(self.config, "enable_continuous_batching", False))
                else self._request_chunks(requests)
            )
            for chunk in request_chunks:
                request_ids = [request_id for request_id, _, _ in chunk]
                prompts = [prompt for _, prompt, _ in chunk]
                params = [params for _, _, params in chunk]
                if mode == "bench":
                    params = [
                        replace(params, max_tokens=self.config.max_model_len, ignore_eos=True) for params in params
                    ]
                command = "pearl_stream_live" if live else "pearl_stream" if on_token_commit is not None else mode
                self._send_all((command, prompts, params, live_epoch if live else num_pearl_steps))
                if on_token_commit is None:
                    if live:
                        replies = self._receive_all(f"{mode} generation", live_epoch=live_epoch)
                    else:
                        replies = self._receive_all(f"{mode} generation")
                else:

                    def deliver_commit(
                        event: dict[str, Any],
                        *,
                        chunk_size: int = len(chunk),
                        chunk_request_ids: tuple[int, ...] = tuple(request_ids),
                    ) -> None:
                        if live:
                            on_token_commit(dict(event))
                            return
                        local_index = int(event["request_index"])
                        if not 0 <= local_index < chunk_size:
                            raise ValueError("worker commit has an invalid request index")
                        delivered_event = dict(event)
                        delivered_event["request_index"] = chunk_request_ids[local_index]
                        if delivered_event.get("request_id") is None:
                            delivered_event["request_id"] = chunk_request_ids[local_index]
                        on_token_commit(delivered_event)

                    receive_kwargs: dict[str, Any] = {"on_token_commit": deliver_commit}
                    if live:
                        receive_kwargs["live_epoch"] = live_epoch
                    replies = self._receive_all(f"{mode} generation", **receive_kwargs)
                leader_payloads = [reply[1] for reply in replies if reply[0] == "result" and reply[1] is not None]
                if len(leader_payloads) != 1:
                    raise RuntimeError("PEARL target leader did not return exactly one result batch.")
                batch_results = leader_payloads[0]
                worker_graph_metrics = [reply[2] for reply in replies if len(reply) > 2]
                if worker_graph_metrics:
                    self.last_worker_metrics = worker_graph_metrics
                    self.last_worker_metrics_by_chunk.append(
                        {
                            "batch_size": min(len(chunk), self.config.max_num_seqs),
                            "num_requests": len(chunk),
                            "worker_metrics": worker_graph_metrics,
                        }
                    )
                    aggregate_graph_metrics = {
                        name: max(metrics[name] for metrics in worker_graph_metrics) for name in worker_graph_metrics[0]
                    }
                    for result in batch_results:
                        result.update(aggregate_graph_metrics)
                total_elapsed += batch_results[0]["elapsed_seconds"] if batch_results else 0.0
                if live:
                    outputs.extend(enumerate(batch_results))
                else:
                    outputs.extend(zip(request_ids, batch_results))
        except _TokenCommitDeliveryError:
            # The workers have completed and their pipes are drained. Requeue
            # only chunks not yet submitted, never replay delivered tokens.
            submitted = {request_id for request_id, _, _ in chunk}
            completed_ids = {request_id for request_id, _ in outputs}
            self._requests = [
                request for request in requests if request[0] not in submitted | completed_ids
            ] + self._requests
            raise
        except Exception:
            self._requests = requests + self._requests
            raise
        finally:
            if live:
                with self._live_lock:
                    self._live_accepting = False
                    self._live_epoch = None
                    self._live_sampling_signature = None

        outputs.sort(key=lambda item: item[0])
        results = [result for _, result in outputs]
        self.last_metrics = results
        output_text = [
            self.tokenizer.decode(result["completion_token_ids"], skip_special_tokens=False) for result in results
        ]
        num_tokens = [len(result["completion_token_ids"]) for result in results]
        num_acc_tokens = tuple(result["num_acc_tokens"] for result in results)
        return output_text, num_tokens, num_acc_tokens, total_elapsed

    def _request_chunks(
        self,
        requests: list[tuple[int, list[int], SamplingParams]],
    ) -> list[list[tuple[int, list[int], SamplingParams]]]:
        chunks: list[list[tuple[int, list[int], SamplingParams]]] = []
        current: list[tuple[int, list[int], SamplingParams]] = []
        current_tokens = 0
        for request in requests:
            prompt_tokens = len(request[1])
            if prompt_tokens > self.config.max_num_batched_tokens:
                raise ValueError("A PEARL prompt exceeds max_num_batched_tokens.")
            if current and (
                len(current) == self.config.max_num_seqs
                or current_tokens + prompt_tokens > self.config.max_num_batched_tokens
            ):
                chunks.append(current)
                current = []
                current_tokens = 0
            current.append(request)
            current_tokens += prompt_tokens
        if current:
            chunks.append(current)
        return chunks

    def _send_all(self, message: tuple[Any, ...]) -> None:
        if self._closed:
            raise RuntimeError("The PEARL engine is closed.")
        for connection in self._connections:
            connection.send(message)

    def _receive_all(
        self,
        operation: str,
        *,
        on_token_commit: Callable[[dict[str, Any]], None] | None = None,
        live_epoch: int | None = None,
    ) -> list[tuple[str, Any]]:
        replies: list[tuple[str, Any] | None] = [None] * len(self._connections)
        pending = {
            connection: (rank, process)
            for rank, (process, connection) in enumerate(zip(self._processes, self._connections))
        }
        deadline = time.monotonic() + self.config.worker_timeout_seconds
        callback_error: Exception | None = None
        while pending:
            for connection, (rank, process) in tuple(pending.items()):
                if not process.is_alive():
                    raise RuntimeError(f"PEARL rank {rank} exited during {operation} with code {process.exitcode}.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                pending_ranks = sorted(rank for rank, _ in pending.values())
                raise TimeoutError(
                    f"PEARL timed out after {self.config.worker_timeout_seconds:g}s during {operation}; "
                    f"pending ranks: {pending_ranks}."
                )
            for connection in wait_for_connections(pending, timeout=min(1.0, remaining)):
                rank, process = pending[connection]
                try:
                    reply = connection.recv()
                except (EOFError, OSError) as error:
                    raise RuntimeError(
                        f"PEARL rank {rank} exited during {operation} with code {process.exitcode}."
                    ) from error
                if reply[0] == "error":
                    raise RuntimeError(f"PEARL rank {rank} failed during {operation}:\n{reply[1]}")
                if reply[0] == "commit":
                    if on_token_commit is not None and callback_error is None:
                        try:
                            on_token_commit(reply[1])
                        except Exception as error:
                            # Continue draining commit/result messages so a
                            # failed client callback cannot strand other ranks
                            # in collectives or poison the next command.
                            callback_error = error
                    continue
                if reply[0] == "live_idle":
                    if live_epoch is None or int(reply[1]) != live_epoch:
                        raise RuntimeError("PEARL worker reported an unexpected live-admission epoch")
                    self._resolve_live_idle(live_epoch, int(reply[2]))
                    continue
                pending.pop(connection)
                replies[rank] = reply
        if callback_error is not None:
            raise _TokenCommitDeliveryError(
                f"Token commit callback failed during {operation}; workers were drained."
            ) from callback_error
        return [reply for reply in replies if reply is not None]

    def _resolve_live_idle(self, epoch: int, worker_admission_seq: int) -> None:
        """Close an idle epoch unless a controller admission won the lock."""
        with self._live_lock:
            if self._live_epoch != epoch:
                return
            if self._live_admission_seq > worker_admission_seq:
                return
            self._live_accepting = False
            message = ("close_live", epoch, self._live_admission_seq, ())
            for connection in self._admission_connections:
                connection.send(message)

    def exit(self) -> None:
        if self._closed:
            return
        self._closed = True
        for process, connection in zip(self._processes, self._connections):
            if process.is_alive():
                with suppress(BrokenPipeError, EOFError):
                    connection.send(("exit", None, None, None))
        graceful_deadline = time.monotonic() + 30
        for process in self._processes:
            process.join(timeout=max(0.0, graceful_deadline - time.monotonic()))
        live_processes = [process for process in self._processes if process.is_alive()]
        for process in live_processes:
            process.terminate()
        terminate_deadline = time.monotonic() + 5
        for process in live_processes:
            process.join(timeout=max(0.0, terminate_deadline - time.monotonic()))
        for connection in self._connections:
            connection.close()
        for connection in self._admission_connections:
            connection.close()

    def __enter__(self) -> PEARLEngine:
        return self

    def __exit__(self, *_args) -> None:
        self.exit()


def _pearl_worker(
    config: NativePearlConfig,
    rank: int,
    master_port: int,
    connection: Connection,
    admission_connection: Connection | None = None,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(config.draft_tp_size + config.target_tp_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    engine: NativePearlEngine | None = None
    try:
        engine = NativePearlEngine(config)
        connection.send(("ready", rank, engine.graph_metrics()))
        while True:
            mode, prompts, sampling_params, num_pearl_steps = connection.recv()
            if mode == "exit":
                break
            if mode == "log":
                logger.info("[PEARL rank %d] %s", rank, prompts)
                connection.send(("logged", rank))
                continue
            if mode == "configure_decode_profiling":
                engine.configure_decode_profiling(int(prompts), bool(sampling_params))
                connection.send(("configured", rank))
                continue
            if mode == "pearl_stream_live":
                if admission_connection is None:
                    raise RuntimeError("Live PEARL generation requires an admission channel")
                live_epoch = int(num_pearl_steps)
                last_admission_seq = 0

                def receive_live_admissions(block: bool, *, epoch_id: int = live_epoch):
                    nonlocal last_admission_seq
                    leader = engine.topology.target_leader_rank
                    message = None
                    header = torch.zeros(2, dtype=torch.int64, device=engine.device)
                    if rank == leader:
                        if block and not admission_connection.poll():
                            connection.send(("live_idle", epoch_id, last_admission_seq))
                        if block or admission_connection.poll():
                            message = admission_connection.recv()
                            kind, epoch, sequence, _ = message
                            if int(epoch) != epoch_id:
                                raise RuntimeError("SpecSLO worker received a stale live-admission epoch")
                            header[0] = 2 if kind == "close_live" else 3 if kind == "abort" else 1
                            header[1] = int(sequence)
                    dist.broadcast(header, src=leader)
                    kind_code, sequence = [int(value) for value in header.cpu().tolist()]
                    if kind_code == 0:
                        return False, ()
                    if rank != leader:
                        message = admission_connection.recv()
                    assert message is not None
                    kind, epoch, local_sequence, payload = message
                    if int(epoch) != epoch_id or int(local_sequence) != sequence:
                        raise RuntimeError("SpecSLO workers observed divergent live-admission ordering")
                    if kind_code == 2:
                        if kind != "close_live":
                            raise RuntimeError("SpecSLO live close header does not match its payload")
                        return True, ()
                    if kind_code == 3:
                        if kind != "abort" or sequence <= last_admission_seq:
                            raise RuntimeError("SpecSLO live abort sequence is invalid")
                        last_admission_seq = sequence
                        return False, (), tuple(str(value) for value in payload)
                    if kind != "admit" or sequence <= last_admission_seq:
                        raise RuntimeError("SpecSLO live admission sequence is invalid")
                    last_admission_seq = sequence
                    return False, payload, ()

                result = engine.generate_batch(
                    prompts,
                    sampling_params,
                    on_token_commit=lambda event: connection.send(("commit", event)),
                    request_admission_callback=receive_live_admissions,
                )
            elif mode == "pearl_stream":
                result = engine.generate_batch(
                    prompts,
                    sampling_params,
                    on_token_commit=lambda event: connection.send(("commit", event)),
                )
            elif mode == "pearl":
                result = engine.generate_batch(prompts, sampling_params)
            elif mode == "target_ar":
                result = engine.generate_target_ar_batch(prompts, sampling_params)
            elif mode == "bench":
                result = engine.generate_batch(
                    prompts,
                    sampling_params,
                    max_rounds=num_pearl_steps,
                )
            else:
                raise ValueError(f"Unknown PEARL worker command {mode!r}.")
            connection.send(("result", result, engine.graph_metrics()))
    except BaseException:
        with suppress(BrokenPipeError, EOFError):
            connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()
        if admission_connection is not None:
            admission_connection.close()
        if engine is not None:
            if dist.is_initialized():
                dist.destroy_process_group()


def _reserve_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _eos_set(eos_token_id: int | list[int] | None) -> frozenset[int]:
    if eos_token_id is None:
        return frozenset()
    if isinstance(eos_token_id, int):
        return frozenset((eos_token_id,))
    return frozenset(int(token_id) for token_id in eos_token_id)


__all__ = ["PEARLConfig", "PEARLEngine", "SamplingParams", "logger"]
