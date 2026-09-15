# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor-parallel Qwen2, Qwen3, and Llama models for native PEARL.

The upstream nano-PEARL model uses CUDA-only FlashAttention and Triton cache
write kernels. This port keeps its weight layout and tensor-parallel scheme.
On Ascend, packed prefill, verification, and decode use the same CANN paged KV
cache and ``_npu_paged_attention`` operator as the vLLM Ascend attention
backend. Every packed token carries its own page table and context length,
matching nano-PEARL's static-batch attention contract. Dense SDPA is retained
only as the CPU test fallback.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from safetensors import safe_open
from torch import nn
from vllm.model_executor.layers.rotary_embedding import get_rope

from vllm_ascend import envs as ascend_envs
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.spec_decode.pearl.mc2 import (
    MC2Profile,
    matmul_allreduce_add_rmsnorm_or_fallback,
    resolve_hccl_comm_name,
)
from vllm_ascend.spec_decode.pearl.native_graph import (
    run_native_fused_infer_attention,
    run_native_paged_attention,
)

PAGED_ATTENTION_BLOCK_SIZE = 128
"""CANN's recommended page size for paged attention."""

MIN_PAGED_ATTENTION_BLOCKS = 16
"""Smallest CANN paged cache allocation accepted by the runtime."""

SUPPORTED_NATIVE_ARCHITECTURES = frozenset(("LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM"))
TENSOR_CORE_TILE_SIZE = 128
ACL_FORMAT_FRACTAL_NZ = 29

# TP3 is intentionally opt-in: older Ascend CANN releases reject the fused
# communicator during allocation.  Once a worker sees one failure, keep the
# ordinary HCCL path for all later layers instead of retrying every matmul.
_TP3_MM_ALL_REDUCE_DISABLED = False


def _is_contiguous_linear_tree_plan(plan: object) -> bool:
    """Return whether a selected tree is one ordinary contiguous causal chain."""
    parents = [int(value) for value in plan.parent_indices.detach().cpu().tolist()]
    positions = [int(value) for value in plan.positions.detach().cpu().tolist()]
    cache_positions = getattr(plan, "cache_positions", None)
    if cache_positions is None:
        cache_positions = plan.positions
    cache_positions = [int(value) for value in cache_positions.detach().cpu().tolist()]
    if not positions:
        return False
    prefix_len = int(getattr(plan, "prefix_len", positions[0]))
    expected_positions = list(range(prefix_len, prefix_len + len(parents) + 1))
    return (
        parents == [-1, *range(max(0, len(parents) - 1))]
        and positions == expected_positions
        and cache_positions == expected_positions
    )


def _use_native_fused_mm_all_reduce(tp_size: int, device_type: str) -> bool:
    """Match vLLM-Ascend's opt-in contract for the standalone model.

    Merely having the torch-npu symbol does not establish that the active
    communicator/shape can allocate MC2 resources.  In particular, unit and
    functional graph runs must not silently enter a fused collective which
    was never requested or qualified.
    """

    if device_type != "npu":
        return False
    if tp_size == 3:
        return bool(
            ascend_envs.VLLM_ASCEND_PEARL_ENABLE_TP3_MM_ALL_REDUCE
            and not _TP3_MM_ALL_REDUCE_DISABLED
        )
    return bool(tp_size in (2, 4, 8) and ascend_envs.VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE)


def _divide(numerator: int, denominator: int) -> int:
    if numerator % denominator:
        raise ValueError(f"Cannot partition {numerator} values across TP={denominator}.")
    return numerator // denominator


def prepare_native_model_config(config, tensor_parallel_size: int):
    """Copy and pad a Hugging Face config using nano-PEARL's dynamic-TP rules."""
    if tensor_parallel_size <= 0:
        raise ValueError("PEARL tensor-parallel size must be positive.")
    architecture = config.architectures[0]
    if architecture not in SUPPORTED_NATIVE_ARCHITECTURES:
        raise ValueError(
            f"Native PEARL does not support architecture {architecture!r}; "
            f"expected one of {sorted(SUPPORTED_NATIVE_ARCHITECTURES)}."
        )

    prepared = deepcopy(config)
    prepared.valid_vocab_size = config.vocab_size
    prepared.valid_num_attention_heads = config.num_attention_heads
    prepared.valid_num_key_value_heads = config.num_key_value_heads
    prepared.valid_intermediate_size = config.intermediate_size
    prepared.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    if tensor_parallel_size in (1, 2, 4, 8):
        for name in ("num_attention_heads", "num_key_value_heads", "intermediate_size", "vocab_size"):
            if getattr(prepared, name) % tensor_parallel_size:
                raise ValueError(
                    f"Cannot partition {name}={getattr(prepared, name)} across TP={tensor_parallel_size}. "
                    "Upstream nano-PEARL only pads non-power-of-two TP sizes."
                )
        return prepared

    if config.num_attention_heads % config.num_key_value_heads:
        raise ValueError("Dynamic PEARL TP requires an integral grouped-query attention ratio.")
    gqa_ratio = config.num_attention_heads // config.num_key_value_heads
    prepared.num_key_value_heads = ceil(config.num_key_value_heads / tensor_parallel_size) * tensor_parallel_size
    prepared.num_attention_heads = prepared.num_key_value_heads * gqa_ratio
    prepared.intermediate_size = (
        ceil(config.intermediate_size / (tensor_parallel_size * TENSOR_CORE_TILE_SIZE))
        * tensor_parallel_size
        * TENSOR_CORE_TILE_SIZE
    )
    prepared.vocab_size = ceil(config.vocab_size / tensor_parallel_size) * tensor_parallel_size
    return prepared


def _copy_padded_shard(
    destination: torch.Tensor,
    loaded: torch.Tensor,
    *,
    dim: int,
    start: int,
) -> None:
    """Copy one shard from a logically zero-padded checkpoint tensor."""
    destination.zero_()
    available = max(0, min(destination.shape[dim], loaded.shape[dim] - start))
    if available:
        destination.narrow(dim, 0, available).copy_(loaded.narrow(dim, start, available))


@dataclass(frozen=True)
class NativeTPContext:
    """The model-parallel coordinates for one PEARL model group."""

    group: dist.ProcessGroup
    rank: int
    size: int
    leader_rank: int


@dataclass(frozen=True)
class NativeAttentionMetadata:
    """Paged-cache coordinates for every token in a packed model invocation."""

    slot_mapping: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    actual_seq_lengths_q: tuple[int, ...] = ()
    sequence_lens: tuple[int, ...] = ()
    request_block_tables: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    use_fused_infer_attention: bool = False
    # Keep the packed 2-D mask for an independent dense oracle. FIA takes a
    # request-major 4-D envelope, never additional query/token rows.
    tree_attention: bool = False
    tree_attention_mask: torch.Tensor | None = None


def make_tree_fia_mask(mask_rows: Sequence[torch.Tensor]) -> torch.Tensor:
    """Pack heterogeneous request masks without padding physical queries."""
    if not mask_rows or any(mask.ndim != 2 or mask.shape[0] == 0 for mask in mask_rows):
        raise ValueError("Tree FIA needs non-empty two-dimensional request masks")
    columns = mask_rows[0].shape[1]
    if any(mask.shape[1] != columns for mask in mask_rows):
        raise ValueError("Tree FIA request masks must share the cache capacity")
    result = torch.ones(
        (len(mask_rows), 1, max(mask.shape[0] for mask in mask_rows), columns),
        dtype=torch.bool,
        device=mask_rows[0].device,
    )
    for index, mask in enumerate(mask_rows):
        result[index, 0, : mask.shape[0]].copy_(mask)
    return result


class NativeRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.device.type == "npu":
            if residual is None:
                normalized, _ = torch_npu.npu_rms_norm(
                    hidden_states,
                    self.weight,
                    self.eps,
                )
                return normalized
            normalized, _, residual = torch_npu.npu_add_rms_norm(
                hidden_states,
                residual,
                self.weight,
                self.eps,
            )
            return normalized, residual

        if residual is not None:
            residual = hidden_states.float().add(residual.float()).to(hidden_states.dtype)
            hidden_states = residual
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(variance + self.eps)
        normalized = normalized.to(hidden_states.dtype) * self.weight
        if residual is None:
            return normalized
        return normalized, residual

    def forward_mc2(
        self,
        local_hidden_states: torch.Tensor,
        residual: torch.Tensor,
        projection_weight: torch.Tensor,
        context: NativeTPContext,
        *,
        projection_bias: torch.Tensor | None = None,
        profile: MC2Profile | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse TP projection, all-reduce, residual add and RMSNorm when available."""
        if projection_bias is not None:
            # The production MC2 ABI has no bias input. Preserve exact
            # semantics through the regular path when a model enables bias.
            projected = F.linear(local_hidden_states, projection_weight, projection_bias)
            if context.size > 1 and dist.is_available() and dist.is_initialized():
                dist.all_reduce(projected, group=context.group)
            return self.forward(projected, residual)  # type: ignore[return-value]
        comm_name = resolve_hccl_comm_name(
            context.group,
            device=local_hidden_states.device,
            rank=context.rank,
        )
        return matmul_allreduce_add_rmsnorm_or_fallback(
            local_hidden_states,
            projection_weight,
            residual,
            self.weight,
            group_tp=comm_name,
            tp_rank_size=context.size,
            tp_rank_id=context.rank,
            epsilon=self.eps,
            process_group=context.group,
            use_fused=True,
            profile=profile,
        )


class NativeColumnLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, context: NativeTPContext, bias: bool = False) -> None:
        super().__init__()
        self.context = context
        self.output_size = output_size
        self.output_size_per_rank = _divide(output_size, context.size)
        self.weight = nn.Parameter(torch.empty(self.output_size_per_rank, input_size))
        self.bias = nn.Parameter(torch.empty(self.output_size_per_rank)) if bias else None

    def load_weight(self, loaded_weight: torch.Tensor) -> None:
        start = self.context.rank * self.output_size_per_rank
        _copy_padded_shard(self.weight.data, loaded_weight, dim=0, start=start)

    def load_bias(self, loaded_bias: torch.Tensor) -> None:
        assert self.bias is not None
        start = self.context.rank * self.output_size_per_rank
        _copy_padded_shard(self.bias.data, loaded_bias, dim=0, start=start)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight, self.bias)


class NativeMergedColumnLinear(NativeColumnLinear):
    def __init__(
        self,
        input_size: int,
        output_sizes: tuple[int, int],
        context: NativeTPContext,
        bias: bool = False,
    ) -> None:
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), context, bias=bias)

    def load_shard(self, loaded_weight: torch.Tensor, shard_id: int, is_bias: bool = False) -> None:
        shard_output_size = self.output_sizes[shard_id]
        shard_per_rank = _divide(shard_output_size, self.context.size)
        destination_start = sum(self.output_sizes[:shard_id]) // self.context.size
        source_start = self.context.rank * shard_per_rank
        destination = self.bias if is_bias else self.weight
        assert destination is not None
        _copy_padded_shard(
            destination.data.narrow(0, destination_start, shard_per_rank),
            loaded_weight,
            dim=0,
            start=source_start,
        )


class NativeQKVLinear(NativeColumnLinear):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        context: NativeTPContext,
        bias: bool,
    ) -> None:
        self.head_dim = head_dim
        self.num_heads_per_rank = _divide(num_heads, context.size)
        self.num_kv_heads_per_rank = _divide(num_kv_heads, context.size)
        self.q_size = self.num_heads_per_rank * head_dim
        self.kv_size = self.num_kv_heads_per_rank * head_dim
        super().__init__(
            hidden_size,
            (num_heads + 2 * num_kv_heads) * head_dim,
            context,
            bias=bias,
        )

    def load_shard(self, loaded_weight: torch.Tensor, shard_id: str, is_bias: bool = False) -> None:
        if shard_id == "q":
            destination_start, shard_size = 0, self.q_size
        elif shard_id == "k":
            destination_start, shard_size = self.q_size, self.kv_size
        elif shard_id == "v":
            destination_start, shard_size = self.q_size + self.kv_size, self.kv_size
        else:
            raise ValueError(f"Unknown QKV shard {shard_id!r}.")
        source_start = self.context.rank * shard_size
        destination = self.bias if is_bias else self.weight
        assert destination is not None
        _copy_padded_shard(
            destination.data.narrow(0, destination_start, shard_size),
            loaded_weight,
            dim=0,
            start=source_start,
        )


class NativeRowLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, context: NativeTPContext, bias: bool = False) -> None:
        super().__init__()
        self.context = context
        self.input_size_per_rank = _divide(input_size, context.size)
        self.weight = nn.Parameter(torch.empty(output_size, self.input_size_per_rank))
        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None

    def load_weight(self, loaded_weight: torch.Tensor) -> None:
        start = self.context.rank * self.input_size_per_rank
        _copy_padded_shard(self.weight.data, loaded_weight, dim=1, start=start)

    def load_bias(self, loaded_bias: torch.Tensor) -> None:
        assert self.bias is not None
        if self.context.rank == 0:
            self.bias.data.copy_(loaded_bias)
        else:
            self.bias.data.zero_()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # vLLM-Ascend's production row-parallel path fuses the local matrix
        # multiply and TP all-reduce.  Native PEARL used separate F.linear and
        # dist.all_reduce calls, which leaves a large synchronization bubble
        # on TP3 target workers.  Keep the ordinary path for CPU tests,
        # unsupported CANN builds, and biased layers whose ABI is unavailable.
        global _TP3_MM_ALL_REDUCE_DISABLED
        fused_mm_reduce = getattr(torch_npu, "npu_mm_all_reduce_base", None)
        allow_tp3_fused = self.context.size == 3 and _use_native_fused_mm_all_reduce(
            self.context.size, hidden_states.device.type
        )
        if (
            _use_native_fused_mm_all_reduce(self.context.size, hidden_states.device.type)
            and fused_mm_reduce is not None
        ):
            hcomm_info = resolve_hccl_comm_name(
                self.context.group,
                device=hidden_states.device,
                rank=self.context.rank,
            )
            if hcomm_info:
                try:
                    return fused_mm_reduce(
                        hidden_states,
                        self.weight.t(),
                        hcomm_info,
                        bias=self.bias,
                    )
                except (RuntimeError, ValueError) as error:
                    if allow_tp3_fused:
                        _TP3_MM_ALL_REDUCE_DISABLED = True
                        if ascend_envs.VLLM_ASCEND_PEARL_VERBOSE:
                            print(
                                f"[PEARL] disabling TP3 fused mm+all-reduce after CANN rejection: {error}",
                                flush=True,
                            )
                    else:
                        raise
        output = F.linear(hidden_states, self.weight, self.bias)
        if self.context.size > 1:
            dist.all_reduce(output, group=self.context.group)
        return output


class NativeVocabEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, context: NativeTPContext) -> None:
        super().__init__()
        self.context = context
        self.vocab_size = vocab_size
        self.vocab_size_per_rank = _divide(vocab_size, context.size)
        self.vocab_start = context.rank * self.vocab_size_per_rank
        self.vocab_end = self.vocab_start + self.vocab_size_per_rank
        self.weight = nn.Parameter(torch.empty(self.vocab_size_per_rank, hidden_size))

    def load_weight(self, loaded_weight: torch.Tensor) -> None:
        _copy_padded_shard(self.weight.data, loaded_weight, dim=0, start=self.vocab_start)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.context.size == 1:
            return F.embedding(input_ids, self.weight)
        in_partition = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)
        local_ids = torch.where(in_partition, input_ids - self.vocab_start, torch.zeros_like(input_ids))
        output = F.embedding(local_ids, self.weight) * in_partition.unsqueeze(-1)
        dist.all_reduce(output, group=self.context.group)
        return output


class NativeLMHead(NativeVocabEmbedding):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        context: NativeTPContext,
        *,
        track_cache_finiteness: bool = False,
    ) -> None:
        super().__init__(vocab_size, hidden_size, context)
        self.track_cache_finiteness = track_cache_finiteness
        self.register_buffer("logits_nonfinite", torch.zeros((), dtype=torch.bool), persistent=False)

    def _track_logits(self, logits: torch.Tensor) -> None:
        if self.track_cache_finiteness:
            # Inspect only real local logits, before any synthetic -inf used
            # to exclude empty vocabulary shards from the TP argmax.
            self.logits_nonfinite.logical_or_(~torch.isfinite(logits).all())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        local_logits = F.linear(hidden_states, self.weight)
        self._track_logits(local_logits)
        if self.context.size == 1:
            return local_logits
        gathered_logits = [torch.empty_like(local_logits) for _ in range(self.context.size)]
        dist.all_gather(gathered_logits, local_logits, group=self.context.group)
        return torch.cat(gathered_logits, dim=-1)

    def greedy(self, hidden_states: torch.Tensor, vocabulary_size: int) -> torch.Tensor:
        local_vocabulary_size = min(self.vocab_end, vocabulary_size) - self.vocab_start
        if local_vocabulary_size <= 0:
            local_values = torch.full(
                (hidden_states.shape[0],),
                -torch.inf,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            local_token_ids = torch.zeros(hidden_states.shape[0], dtype=torch.long, device=hidden_states.device)
        else:
            # Slicing a FRACTAL_NZ weight materializes an ND view that cannot be
            # consumed by the NZ matmul kernel. Project the complete local TP
            # shard first, then remove a target-only vocabulary suffix from the
            # much smaller logits tensor.
            local_logits = F.linear(hidden_states, self.weight)
            local_logits = local_logits[:, :local_vocabulary_size]
            self._track_logits(local_logits)
            local_values, local_token_ids = local_logits.max(dim=-1)
            local_token_ids += self.vocab_start
        if self.context.size == 1:
            return local_token_ids

        local_candidates = torch.stack(
            (local_values.float(), local_token_ids.float()),
            dim=-1,
        )
        gathered_candidates = [torch.empty_like(local_candidates) for _ in range(self.context.size)]
        dist.all_gather(gathered_candidates, local_candidates, group=self.context.group)
        candidates = torch.stack(gathered_candidates, dim=1)
        values = candidates[..., 0]
        token_ids = candidates[..., 1].to(dtype=torch.long)
        winning_rank = values.argmax(dim=-1, keepdim=True)
        return token_ids.gather(dim=-1, index=winning_rank).squeeze(-1)

    def greedy_with_confidence(
        self, hidden_states: torch.Tensor, vocabulary_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the global greedy token and its softmax probability."""

        local_vocabulary_size = min(self.vocab_end, vocabulary_size) - self.vocab_start
        if local_vocabulary_size <= 0:
            local_values = torch.full(
                (hidden_states.shape[0],),
                -torch.inf,
                dtype=torch.float32,
                device=hidden_states.device,
            )
            local_token_ids = torch.zeros(hidden_states.shape[0], dtype=torch.long, device=hidden_states.device)
            local_logsumexp = local_values
        else:
            local_logits = F.linear(hidden_states, self.weight)[:, :local_vocabulary_size]
            self._track_logits(local_logits)
            local_values, local_token_ids = local_logits.float().max(dim=-1)
            local_token_ids += self.vocab_start
            local_logsumexp = torch.logsumexp(local_logits.float(), dim=-1)
        if self.context.size == 1:
            return local_token_ids, (local_values - local_logsumexp).exp()

        local_summary = torch.stack((local_values, local_token_ids.float(), local_logsumexp), dim=-1)
        gathered = [torch.empty_like(local_summary) for _ in range(self.context.size)]
        dist.all_gather(gathered, local_summary, group=self.context.group)
        summaries = torch.stack(gathered, dim=1)
        values = summaries[..., 0]
        token_ids = summaries[..., 1].to(dtype=torch.long)
        winning_rank = values.argmax(dim=-1, keepdim=True)
        global_values = values.gather(dim=-1, index=winning_rank).squeeze(-1)
        global_token_ids = token_ids.gather(dim=-1, index=winning_rank).squeeze(-1)
        global_logsumexp = torch.logsumexp(summaries[..., 2], dim=-1)
        return global_token_ids, (global_values - global_logsumexp).exp()


class NativeRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        use_production_rope: bool = False,
    ) -> None:
        super().__init__()
        inverse_frequencies = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_position_embeddings, dtype=torch.float32)
        frequencies = torch.outer(positions, inverse_frequencies)
        cache_dtype = torch.get_default_dtype()
        self.register_buffer(
            "cos_sin_cache",
            torch.cat((frequencies.cos(), frequencies.sin()), dim=-1).to(cache_dtype),
            persistent=False,
        )
        self.use_production_rope = (
            use_production_rope
            and hasattr(torch.ops.vllm, "npu_rotary_embedding")
            and not ascend_envs.VLLM_ASCEND_USE_NATIVE_QWEN2_ROPE
        )

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.device.type == "npu" and self.use_production_rope:
            return torch.ops.vllm.npu_rotary_embedding(
                positions,
                query,
                key,
                self.cos_sin_cache,
                query.shape[-1],
                query.shape[-1],
                True,
            )
        cos, sin = self.cos_sin_cache[positions].chunk(2, dim=-1)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        if query.device.type == "npu":
            cos = torch.cat((cos, cos), dim=-1)
            sin = torch.cat((sin, sin), dim=-1)
            return (
                torch_npu.npu_rotary_mul(query, cos, sin),
                torch_npu.npu_rotary_mul(key, cos, sin),
            )
        return _apply_rope(query, cos, sin), _apply_rope(key, cos, sin)


def _apply_rope(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    first, second = value.float().chunk(2, dim=-1)
    rotated = torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
    return rotated.to(value.dtype)


class NativeAttention(nn.Module):
    def __init__(self, config, context: NativeTPContext) -> None:
        super().__init__()
        self.context = context
        self.track_cache_finiteness = bool(getattr(config, "pearl_track_cache_finiteness", False))
        architecture = getattr(config, "architectures", ("Qwen2ForCausalLM",))[0]
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = _divide(config.num_attention_heads, context.size)
        self.num_kv_heads = _divide(config.num_key_value_heads, context.size)
        self.scale = self.head_dim**-0.5
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        attention_bias = bool(getattr(config, "attention_bias", False) or getattr(config, "bias", False))
        qkv_bias = bool(getattr(config, "qkv_bias", attention_bias))
        if architecture == "Qwen2ForCausalLM":
            qkv_bias = True
        self.qkv_proj = NativeQKVLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            context,
            bias=qkv_bias,
        )
        self.o_proj = NativeRowLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            context,
            bias=attention_bias if architecture == "LlamaForCausalLM" else False,
        )
        rope_parameters = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None)
        rope_type = (rope_parameters or {}).get("rope_type", "default")
        if rope_type == "default":
            self.rotary_emb = NativeRotaryEmbedding(
                self.head_dim,
                config.max_position_embeddings,
                getattr(config, "rope_theta", (rope_parameters or {}).get("rope_theta", 10_000.0)),
                use_production_rope=bool(getattr(config, "pearl_use_production_rope", False)),
            )
        else:
            self.rotary_emb = get_rope(
                self.head_dim,
                config.max_position_embeddings,
                is_neox_style=True,
                rope_parameters=rope_parameters,
            )
        if architecture == "Qwen3ForCausalLM":
            self.q_norm = NativeRMSNorm(self.head_dim, config.rms_norm_eps)
            self.k_norm = NativeRMSNorm(self.head_dim, config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        self.use_qknorm_rope_fusion = (
            architecture == "Qwen3ForCausalLM"
            and self.head_dim == 128
            and isinstance(self.rotary_emb, NativeRotaryEmbedding)
            and hasattr(torch.ops.vllm, "qkv_rmsnorm_rope")
        )
        self.key_cache: torch.Tensor | None = None
        self.value_cache: torch.Tensor | None = None
        self.register_buffer("block_table", None, persistent=False)
        self.register_buffer("context_lens", None, persistent=False)
        self.uses_paged_attention = False
        self.max_model_len = 0
        self.blocks_per_sequence = 0
        self.block_size = PAGED_ATTENTION_BLOCK_SIZE

    def configure_cache(
        self,
        max_model_len: int,
        max_num_seqs: int = 1,
        use_paged_attention: bool | None = None,
        block_size: int = PAGED_ATTENTION_BLOCK_SIZE,
        num_cache_blocks: int | None = None,
    ) -> None:
        """Allocate a vLLM-compatible paged cache when running on an NPU.

        Each static-batch sequence owns a disjoint physical-page range. Rollback
        only changes its logical length; later rounds overwrite stale slots.
        """
        if max_model_len <= 0 or max_num_seqs <= 0 or block_size <= 0:
            raise ValueError("PEARL cache dimensions must be positive.")
        device = self.qkv_proj.weight.device
        if use_paged_attention is None:
            use_paged_attention = device.type == "npu"
        self.uses_paged_attention = use_paged_attention
        self.max_model_len = max_model_len
        self.block_size = block_size
        self.blocks_per_sequence = ceil(max_model_len / block_size)
        required_blocks = self.blocks_per_sequence * max_num_seqs
        if num_cache_blocks is not None and num_cache_blocks < max_num_seqs:
            raise ValueError("PEARL needs at least one KV cache block per configured sequence.")
        num_blocks = max(MIN_PAGED_ATTENTION_BLOCKS, num_cache_blocks or required_blocks)
        paged_cache_shape = (num_blocks, block_size, self.num_kv_heads, self.head_dim)
        cache_shape = (
            paged_cache_shape if self.uses_paged_attention else (num_blocks * block_size,) + paged_cache_shape[2:]
        )
        # FULL-mask FIA still multiplies blocked value columns by zero. A NaN
        # from uninitialized scratch/hole storage is not masked by 0 * NaN.
        # Initialize once; subsequent valid writes and KV moves stay finite.
        # Never sanitize valid model NaNs: the tree commit guard rejects them.
        self.key_cache = torch.zeros(cache_shape, dtype=self.qkv_proj.weight.dtype, device=device)
        self.value_cache = torch.zeros_like(self.key_cache)
        previous_flag = getattr(self, "tree_cache_nonfinite", None)
        if isinstance(previous_flag, torch.Tensor) and previous_flag.device == device:
            previous_flag.zero_()
        else:
            self.tree_cache_nonfinite = torch.zeros((), dtype=torch.bool, device=device)
        if num_blocks >= required_blocks:
            self.block_table = torch.arange(required_blocks, dtype=torch.int32, device=device).view(
                max_num_seqs, self.blocks_per_sequence
            )
        else:
            self.block_table = torch.zeros((max_num_seqs, self.blocks_per_sequence), dtype=torch.int32, device=device)
        # CANN's eager paged-attention ABI takes sequence lengths from CPU.
        self.context_lens = torch.empty(max_num_seqs, dtype=torch.int32)

    def _write_to_cache(self, slot_mapping: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
        assert self.key_cache is not None and self.value_cache is not None
        if self.uses_paged_attention:
            DeviceOperator.reshape_and_cache(
                key=key,
                value=value,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                slot_mapping=slot_mapping.to(dtype=torch.int32),
            )
            return
        self.key_cache[slot_mapping] = key
        self.value_cache[slot_mapping] = value

    def _paged_attention(self, query: torch.Tensor, metadata: NativeAttentionMetadata) -> torch.Tensor:
        assert self.key_cache is not None and self.value_cache is not None
        attended = torch.empty_like(query)
        run_native_paged_attention(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            block_table=metadata.block_tables,
            context_lens=metadata.context_lens,
            output=attended,
        )
        return attended

    def _fused_infer_attention(self, query: torch.Tensor, metadata: NativeAttentionMetadata) -> torch.Tensor:
        assert self.key_cache is not None and self.value_cache is not None
        assert metadata.request_block_tables is not None
        attention_mask = (
            metadata.tree_attention_mask
            if metadata.tree_attention
            else metadata.attention_mask
        )
        if attention_mask is None:
            raise RuntimeError("Fused infer attention requires its selected mask")
        attended = torch.empty_like(query)
        run_native_fused_infer_attention(
            query=query,
            key_cache=self.key_cache.view(self.key_cache.shape[0], self.block_size, -1),
            value_cache=self.value_cache.view(self.value_cache.shape[0], self.block_size, -1),
            attention_mask=attention_mask,
            block_table=metadata.request_block_tables,
            block_size=self.block_size,
            actual_seq_lengths_q=list(metadata.actual_seq_lengths_q),
            actual_seq_lengths_kv=list(metadata.sequence_lens),
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            output=attended,
            tree_attention=metadata.tree_attention,
        )
        return attended

    def _dense_attention(self, query: torch.Tensor, metadata: NativeAttentionMetadata) -> torch.Tensor:
        assert self.key_cache is not None and self.value_cache is not None
        attended = torch.empty_like(query)
        repeat_factor = self.num_heads // self.num_kv_heads
        for offset in range(query.shape[0]):
            context_length = int(metadata.context_lens[offset].item())
            if metadata.block_tables.ndim != 2:
                raise ValueError("Dense attention requires a 2-D physical block table")
            logical_positions = torch.arange(context_length, dtype=torch.long, device=query.device)
            logical_blocks = torch.div(logical_positions, self.block_size, rounding_mode="floor")
            physical_blocks = metadata.block_tables[offset].index_select(0, logical_blocks)
            slots = physical_blocks.to(torch.long) * self.block_size + logical_positions.remainder(self.block_size)
            if self.uses_paged_attention:
                key_storage = self.key_cache.flatten(0, 1)
                value_storage = self.value_cache.flatten(0, 1)
            else:
                key_storage = self.key_cache
                value_storage = self.value_cache
            keys = key_storage.index_select(0, slots)
            values = value_storage.index_select(0, slots)
            keys = keys.transpose(0, 1).repeat_interleave(repeat_factor, dim=0)
            values = values.transpose(0, 1).repeat_interleave(repeat_factor, dim=0)
            attention_mask = None
            attention_query = query[offset].unsqueeze(0).unsqueeze(2)
            if metadata.attention_mask is not None:
                if metadata.attention_mask.ndim != 2 or offset >= metadata.attention_mask.shape[0]:
                    raise ValueError("Tree attention mask rows must match packed query rows")
                # PEARL tree masks use True=blocked, while SDPA uses True=keep.
                visible = (~metadata.attention_mask[offset, :context_length].to(torch.bool)).view(1, 1, 1, -1)
                # Tree graph context buckets can read allocated but unwritten
                # KV slots, including NaNs. An additive/boolean SDPA mask alone
                # does not make NaN * 0 safe. Sanitize blocked K/V before the
                # score/value matrix products; this also protects eager sibling
                # masking from stale cache data after a rejected proposal.
                key_visible = visible.view(1, context_length, 1)
                keys = torch.where(key_visible, keys, 0)
                values = torch.where(key_visible, values, 0)
                attention_mask = visible
                # With BF16 SDPA on Ascend, merely adding fully masked tail
                # columns (e.g. 260 -> 512 for ACLGraph) can change reduction
                # rounding enough to alter intermediate KV/hidden states. The
                # graph itself matches same-shape eager; the padding does not.
                # FP32 inputs on BOTH eager and graph paths are a diagnostic
                # implementation, not a guarantee of padding invariance: the
                # 2026-09-10 NPU boundary regression still found drift. Keep
                # strict graph validation/fallback; do not infer accumulation
                # precision solely from the public tensor dtype.
                # Ordinary paged/FIA attention is intentionally unaffected.
                attention_query = attention_query.float()
                keys = keys.float()
                values = values.float()
            attended[offset] = (
                F.scaled_dot_product_attention(
                    attention_query,
                    keys.unsqueeze(0),
                    values.unsqueeze(0),
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=self.scale,
                )
                .squeeze(0)
                .squeeze(1)
            )
        return attended

    def _default_metadata(self, positions: torch.Tensor) -> NativeAttentionMetadata:
        assert self.block_table is not None
        return NativeAttentionMetadata(
            slot_mapping=positions.to(dtype=torch.long),
            context_lens=(positions + 1).to(device="cpu", dtype=torch.int32),
            block_tables=self.block_table[:1].expand(positions.shape[0], -1),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_metadata: NativeAttentionMetadata | None = None,
        *,
        return_pre_projection: bool = False,
    ) -> torch.Tensor:
        if self.key_cache is None or self.value_cache is None:
            raise RuntimeError("Configure the PEARL KV cache before running the model.")
        qkv = self.qkv_proj(hidden_states)
        metadata = attention_metadata or self._default_metadata(positions)
        # Ordinary prefill checks each cache write because no later tree
        # output boundary is guaranteed before its first token is committed.
        # Tree decode, however, executes the complete decoder and LM head
        # before the shared commit vote.  Its model-level hidden/residual and
        # real-logit sticky flags cover every value that can affect a
        # committed token.  Repeating two full-tensor isfinite reductions in
        # every decoder layer added 128 reductions to a 64-layer graph and
        # serialized the TP3 critical path without strengthening that commit
        # boundary.
        track_layer_finiteness = self.track_cache_finiteness and not metadata.tree_attention
        if track_layer_finiteness:
            # Q, K and V are slices (plus finite normalization/RoPE
            # transforms) of this fused projection. Checking the projection
            # once and the attention result once preserves the per-layer
            # numerical fault boundary while avoiding three independent
            # isfinite+reduce chains on every graph replay.
            self.tree_cache_nonfinite.logical_or_(~torch.isfinite(qkv).all())
        if self.use_qknorm_rope_fusion and qkv.device.type == "npu" and qkv.dtype == torch.bfloat16:
            assert isinstance(self.q_norm, NativeRMSNorm)
            assert isinstance(self.k_norm, NativeRMSNorm)
            assert isinstance(self.rotary_emb, NativeRotaryEmbedding)
            query, key, value = DeviceOperator.split_qkv_rmsnorm_rope(
                input=qkv,
                q_weight=self.q_norm.weight,
                k_weight=self.k_norm.weight,
                q_hidden_size=self.q_size,
                kv_hidden_size=self.kv_size,
                head_dim=self.head_dim,
                eps=self.q_norm.eps,
                q_bias=None,
                k_bias=None,
                cos_sin_cache=self.rotary_emb.cos_sin_cache,
                positions=positions,
            )
            query = query.view(-1, self.num_heads, self.head_dim)
            key = key.view(-1, self.num_kv_heads, self.head_dim)
            value = value.view(-1, self.num_kv_heads, self.head_dim)
        else:
            query, key, value = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
            query = self.q_norm(query.view(-1, self.num_heads, self.head_dim))
            key = self.k_norm(key.view(-1, self.num_kv_heads, self.head_dim))
            value = value.view(-1, self.num_kv_heads, self.head_dim)
            query, key = self.rotary_emb(positions, query, key)
        self._write_to_cache(metadata.slot_mapping, key, value)

        if metadata.use_fused_infer_attention:
            attended = self._fused_infer_attention(query, metadata)
        elif self.uses_paged_attention and metadata.attention_mask is None:
            attended = self._paged_attention(query, metadata)
        else:
            attended = self._dense_attention(query, metadata)
        if track_layer_finiteness:
            self.tree_cache_nonfinite.logical_or_(~torch.isfinite(attended).all())
        attended = attended.flatten(1)
        if return_pre_projection:
            return attended
        return self.o_proj(attended)


class NativeQwen2MLP(nn.Module):
    def __init__(self, config, context: NativeTPContext) -> None:
        super().__init__()
        self.gate_up_proj = NativeMergedColumnLinear(
            config.hidden_size,
            (config.intermediate_size, config.intermediate_size),
            context,
            bias=bool(getattr(config, "mlp_bias", False)),
        )
        self.down_proj = NativeRowLinear(
            config.intermediate_size,
            config.hidden_size,
            context,
            bias=bool(getattr(config, "mlp_bias", False)),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(hidden_states)
        if gate_up.device.type == "npu":
            return self.down_proj(torch_npu.npu_swiglu(gate_up))
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class NativeQwen2DecoderLayer(nn.Module):
    def __init__(self, config, context: NativeTPContext) -> None:
        super().__init__()
        self.self_attn = NativeAttention(config, context)
        self.input_layernorm = NativeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = NativeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = NativeQwen2MLP(config, context)
        self.context = context
        self.enable_mc2 = bool(getattr(config, "pearl_enable_mc2", False))
        self.mc2_profile = getattr(config, "pearl_mc2_profile", None)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        attention_metadata: NativeAttentionMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if self.enable_mc2:
            local_attended = self.self_attn(
                positions,
                hidden_states,
                attention_metadata,
                return_pre_projection=True,
            )
            hidden_states, residual = self.post_attention_layernorm.forward_mc2(
                local_attended,
                residual,
                self.self_attn.o_proj.weight,
                self.context,
                projection_bias=self.self_attn.o_proj.bias,
                profile=self.mc2_profile,
            )
        else:
            hidden_states = self.self_attn(positions, hidden_states, attention_metadata)
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class NativeQwen2ForCausalLM(nn.Module):
    """Upstream-supported decoder model with a persistent paged KV cache."""

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config, context: NativeTPContext) -> None:
        super().__init__()
        self.config = config
        self.context = context
        self.track_cache_finiteness = bool(getattr(config, "pearl_track_cache_finiteness", False))
        self.register_buffer("output_nonfinite", torch.zeros((), dtype=torch.bool), persistent=False)
        self.embed_tokens = NativeVocabEmbedding(config.vocab_size, config.hidden_size, context)
        self.layers = nn.ModuleList(NativeQwen2DecoderLayer(config, context) for _ in range(config.num_hidden_layers))
        self.norm = NativeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = NativeLMHead(
            config.vocab_size,
            config.hidden_size,
            context,
            track_cache_finiteness=self.track_cache_finiteness,
        )
        self.register_buffer("attention_mask", None, persistent=False)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.embed_tokens.weight

    def configure_cache(
        self,
        max_model_len: int,
        max_num_seqs: int = 1,
        block_size: int = PAGED_ATTENTION_BLOCK_SIZE,
        num_cache_blocks: int | None = None,
    ) -> None:
        if max_model_len <= 0 or max_num_seqs <= 0:
            raise ValueError("Native PEARL cache dimensions must be positive.")
        # Tree metadata is assembled at the model level, while the physical
        # KV pages live in each attention layer. Keep the dimensions on the
        # owner as well so native tree masks do not depend on a test-only
        # attribute injected by callers.
        self.max_model_len = int(max_model_len)
        self.max_num_seqs = int(max_num_seqs)
        self.block_size = int(block_size)
        for layer in self.layers:
            layer.self_attn.configure_cache(
                max_model_len,
                max_num_seqs,
                block_size=block_size,
                num_cache_blocks=num_cache_blocks,
            )
        # Only a complete physical-cache reinitialization permits clearing a
        # sticky fault. Keep flag storage stable for captured graph operators.
        # Any graph referring to the reallocated KV itself must be discarded.
        self.output_nonfinite.zero_()
        self.lm_head.logits_nonfinite.zero_()
        if self.embed_tokens.weight.device.type == "npu" and self.attention_mask is None:
            self.attention_mask = torch.triu(
                torch.ones(2048, 2048, dtype=torch.int8, device=self.embed_tokens.weight.device),
                diagonal=1,
            )

    def make_attention_metadata(
        self,
        sequence_ids: list[int],
        positions: list[int],
        block_tables: list[list[int]] | torch.Tensor | None = None,
        slot_mapping: list[int] | None = None,
        use_fused_infer_attention: bool = False,
    ) -> tuple[torch.Tensor, NativeAttentionMetadata]:
        if len(sequence_ids) != len(positions):
            raise ValueError("Every packed token needs a sequence ID and position.")
        attention = self.layers[0].self_attn
        assert attention.block_table is not None
        device = self.embed_tokens.weight.device
        sequence_tensor = torch.tensor(sequence_ids, dtype=torch.long, device=device)
        position_tensor = torch.tensor(positions, dtype=torch.long, device=device)
        if block_tables is None:
            physical_block_tables = attention.block_table
        elif isinstance(block_tables, torch.Tensor):
            physical_block_tables = block_tables.to(device=device, dtype=torch.int32)
        else:
            if not block_tables or any(len(table) != attention.blocks_per_sequence for table in block_tables):
                raise ValueError(
                    f"Every PEARL block table must contain {attention.blocks_per_sequence} physical pages."
                )
            for sequence_id, position in zip(sequence_ids, positions):
                logical_block = position // attention.block_size
                if sequence_id >= len(block_tables) or block_tables[sequence_id][logical_block] < 0:
                    raise RuntimeError("PEARL attention referenced an unallocated KV cache page.")
            physical_block_tables = torch.tensor(block_tables, dtype=torch.int32, device=device)
        if physical_block_tables.ndim != 2 or physical_block_tables.shape[1] != attention.blocks_per_sequence:
            raise ValueError(f"Every PEARL block table must contain {attention.blocks_per_sequence} physical pages.")
        if physical_block_tables.shape[0] <= max(sequence_ids):
            raise ValueError("PEARL block tables do not cover every packed sequence ID.")
        if slot_mapping is not None and len(slot_mapping) != len(positions):
            raise ValueError("Every packed token needs one PEARL KV slot.")
        if slot_mapping is None:
            logical_blocks = torch.div(position_tensor, attention.block_size, rounding_mode="floor")
            physical_blocks = physical_block_tables[sequence_tensor, logical_blocks]
            slot_mapping_tensor = physical_blocks * attention.block_size + position_tensor.remainder(
                attention.block_size
            )
        else:
            slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int32, device=device)
        query_lengths: list[int] = []
        request_sequence_ids: list[int] = []
        sequence_lens: list[int] = []
        for sequence_id, position in zip(sequence_ids, positions):
            if not request_sequence_ids or request_sequence_ids[-1] != sequence_id:
                if sequence_id in request_sequence_ids:
                    raise ValueError("Packed PEARL tokens for a request must be contiguous.")
                request_sequence_ids.append(sequence_id)
                query_lengths.append(0)
                sequence_lens.append(0)
            query_lengths[-1] += 1
            sequence_lens[-1] = position + 1
        cumulative_query_lengths: list[int] = []
        for query_length in query_lengths:
            previous_length = cumulative_query_lengths[-1] if cumulative_query_lengths else 0
            cumulative_query_lengths.append(query_length + previous_length)
        if use_fused_infer_attention:
            request_sequence_tensor = torch.tensor(request_sequence_ids, dtype=torch.long, device=device)
            request_block_tables = physical_block_tables.index_select(0, request_sequence_tensor)
            # FIA consumes one table per request; the per-token PA table is not
            # read on this path, so retain the required metadata field without
            # issuing a second index_select.
            token_block_tables = request_block_tables
        else:
            request_block_tables = None
            token_block_tables = physical_block_tables.index_select(0, sequence_tensor)
        metadata = NativeAttentionMetadata(
            slot_mapping=slot_mapping_tensor,
            context_lens=torch.tensor([position + 1 for position in positions], dtype=torch.int32),
            block_tables=token_block_tables,
            actual_seq_lengths_q=tuple(cumulative_query_lengths),
            sequence_lens=tuple(sequence_lens),
            request_block_tables=request_block_tables,
            # Packed PA uses sequence metadata for causality, while FIA needs
            # the cached causal mask for its TND query segments. Tree metadata
            # supplies its own request-local mask separately.
            attention_mask=self.attention_mask if use_fused_infer_attention else None,
            use_fused_infer_attention=use_fused_infer_attention,
        )
        return position_tensor, metadata

    def make_tree_attention_metadata(
        self,
        plans: list[object],
        root_token_ids: list[int],
        draft_token_ids: list[list[int]],
        block_tables: list[list[int]] | torch.Tensor,
        sequence_ids: Sequence[int] | None = None,
        *,
        allow_causal_fast_path: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, NativeAttentionMetadata]:
        """Pack request-local tree queries for a target forward.

        Each request contributes one root query followed by its draft nodes.
        The target model writes all K/V entries into unique physical cache
        positions while RoPE receives the logical depth positions.  The
        request-local mask is indexed by physical slots, so sibling branches
        cannot read one another. NPU uses only the production FULL-mask FIA
        contract; the CPU path retains the 2-D mask for its independent dense
        reference. Linear PEARL keeps its causal FIA contract unchanged.
        """
        if not plans or len(plans) != len(root_token_ids) or len(plans) != len(draft_token_ids):
            raise ValueError("tree plans, root tokens and draft rows must have equal non-zero length")
        device = self.embed_tokens.weight.device
        if isinstance(block_tables, torch.Tensor):
            physical_tables = block_tables.to(device=device, dtype=torch.int32)
        else:
            physical_tables = torch.tensor(block_tables, dtype=torch.int32, device=device)
        request_ids = list(range(len(plans))) if sequence_ids is None else [int(value) for value in sequence_ids]
        if len(request_ids) != len(plans) or len(set(request_ids)) != len(request_ids):
            raise ValueError("tree sequence IDs must be unique and row-aligned")
        attention = self.layers[0].self_attn
        if physical_tables.ndim != 2 or not request_ids or max(request_ids, default=-1) >= physical_tables.shape[0]:
            raise ValueError("tree block tables must contain one row per request")
        input_ids: list[int] = []
        logical_positions: list[int] = []
        physical_positions: list[int] = []
        packed_sequence_ids: list[int] = []
        mask_rows: list[torch.Tensor] = []
        query_lengths: list[int] = []
        sequence_lens: list[int] = []
        all_linear = bool(allow_causal_fast_path)
        for sequence_id, (plan, root_token, candidates) in zip(
            request_ids, zip(plans, root_token_ids, draft_token_ids)
        ):
            # A selected ancestor-closed tree can have fewer physical nodes
            # than its exploration envelope (width * depth). Verification must
            # pack only those selected nodes, not restore discarded candidates.
            expected_nodes = int(plan.parent_indices.numel())
            if len(candidates) != expected_nodes:
                raise ValueError("draft row length does not match its tree plan")
            cache_positions = getattr(plan, "cache_positions", None)
            if cache_positions is None:
                cache_positions = plan.positions
            cache_positions = [int(value) for value in cache_positions.detach().cpu().tolist()]
            if len(cache_positions) != expected_nodes + 1 or len(set(cache_positions)) != len(cache_positions):
                raise ValueError("tree cache positions must be unique and include the root")
            input_ids.extend([int(root_token), *map(int, candidates)])
            plan_positions = [int(value) for value in plan.positions.detach().cpu().tolist()]
            if len(plan_positions) != len(cache_positions):
                raise ValueError("tree logical/cache positions must have equal length")
            logical_positions.extend(plan_positions)
            physical_positions.extend(cache_positions)
            packed_sequence_ids.extend([sequence_id] * (expected_nodes + 1))
            mask = plan.attention_mask
            if tuple(mask.shape) != (expected_nodes + 1, self.max_model_len):
                raise ValueError("tree attention mask shape does not match its plan")
            # Production FULL-tree FIA consumes only the request-major 4-D
            # mask.  Keep CPU plans on CPU while packing so a batch performs
            # one H2D mask transfer instead of one transfer per request plus
            # a duplicate 2-D NPU concatenation.  The dense CPU oracle keeps
            # its original 2-D mask below.
            mask_rows.append(
                mask
                if device.type == "npu" and attention.uses_paged_attention
                else mask.to(device=device, dtype=torch.bool)
            )
            query_lengths.append(expected_nodes + 1)
            sequence_lens.append(max(cache_positions) + 1)
            all_linear = all_linear and _is_contiguous_linear_tree_plan(plan)
        if any(position >= self.max_model_len for position in physical_positions):
            raise ValueError("tree cache position exceeds max_model_len")
        if device.type == "npu" and attention.uses_paged_attention and all_linear:
            # Before the first sibling is selected, a tree is exactly one
            # contiguous causal chain per request. Reuse the ordinary TND FIA
            # contract instead of constructing a 4-D FULL tree mask. The
            # guard deliberately excludes eager scratch layouts, whose
            # physical slots are non-contiguous until promotion/compaction.
            causal_positions, causal_metadata = self.make_attention_metadata(
                packed_sequence_ids,
                logical_positions,
                physical_tables,
                use_fused_infer_attention=True,
            )
            return (
                torch.tensor(input_ids, dtype=torch.long, device=device),
                causal_positions,
                causal_metadata,
            )
        sequence_tensor = torch.tensor(packed_sequence_ids, dtype=torch.long, device=device)
        position_tensor = torch.tensor(logical_positions, dtype=torch.long, device=device)
        physical_position_tensor = torch.tensor(physical_positions, dtype=torch.long, device=device)
        logical_blocks = torch.div(physical_position_tensor, attention.block_size, rounding_mode="floor")
        physical_blocks = physical_tables[sequence_tensor, logical_blocks]
        if (physical_blocks < 0).any():
            raise RuntimeError("tree target forward referenced an unallocated KV cache page")
        slot_mapping = physical_blocks * attention.block_size + physical_position_tensor.remainder(attention.block_size)
        cumulative: list[int] = []
        for length in query_lengths:
            cumulative.append((cumulative[-1] if cumulative else 0) + length)
        request_tensor = torch.tensor(request_ids, dtype=torch.long, device=device)
        request_tables = physical_tables.index_select(0, request_tensor)
        use_fused_tree_attention = device.type == "npu" and attention.uses_paged_attention
        if use_fused_tree_attention:
            tree_mask = None
            tree_fia_mask = make_tree_fia_mask(mask_rows).to(
                device=device,
                dtype=torch.bool,
            )
            # FIA indexes one block-table row per request.  As in the linear
            # FIA path, token-major tables are not read by the operator.
            token_tables = request_tables
        else:
            tree_mask = torch.cat(mask_rows, dim=0)
            tree_fia_mask = make_tree_fia_mask(mask_rows)
            token_tables = physical_tables.index_select(0, sequence_tensor)
        metadata = NativeAttentionMetadata(
            slot_mapping=slot_mapping.to(torch.int32),
            # Dense tree attention reads this vector on the host to choose the
            # valid physical KV prefix. Keep it CPU-resident so ACLGraph
            # capture never performs ``NPU tensor.item()`` on a captured
            # stream; slot/block tensors remain device-resident.
            context_lens=torch.tensor(
                [int(value) + 1 for value in physical_positions],
                dtype=torch.int32,
            ),
            block_tables=token_tables,
            actual_seq_lengths_q=tuple(cumulative),
            sequence_lens=tuple(sequence_lens),
            request_block_tables=request_tables,
            attention_mask=tree_mask,
            use_fused_infer_attention=use_fused_tree_attention,
            tree_attention=True,
            tree_attention_mask=tree_fia_mask,
        )
        return torch.tensor(input_ids, dtype=torch.long, device=device), position_tensor, metadata

    def make_tree_level_attention_metadata(
        self,
        plan: object,
        sequence_id: int,
        node_indices: Sequence[int],
        input_token_ids: Sequence[int],
        block_tables: list[list[int]] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, NativeAttentionMetadata]:
        """Pack one tree level for draft-side autoregressive expansion.

        A tree draft cannot use the linear causal mask: sibling branches must
        not see one another.  This helper keeps the query rows at their fixed
        tree cache positions and selects the corresponding blocked-mask rows,
        allowing the draft worker to expand the spine one level at a time.
        """
        indices = [int(value) for value in node_indices]
        token_ids = [int(value) for value in input_token_ids]
        if not indices or len(indices) != len(token_ids):
            raise ValueError("tree level must contain at least one node")
        expected_nodes = int(plan.parent_indices.numel())
        if any(index < -1 or index >= expected_nodes for index in indices):
            raise ValueError("tree level node index is outside the plan")
        cache_positions = getattr(plan, "cache_positions", None)
        if cache_positions is None:
            cache_positions = plan.positions
        logical_positions = [int(plan.positions[index + 1 if index >= 0 else 0].item()) for index in indices]
        physical_positions = [int(cache_positions[index + 1 if index >= 0 else 0].item()) for index in indices]
        device = self.embed_tokens.weight.device
        physical_tables = (
            block_tables.to(device=device, dtype=torch.int32)
            if isinstance(block_tables, torch.Tensor)
            else torch.tensor(block_tables, dtype=torch.int32, device=device)
        )
        if physical_tables.ndim != 2 or not 0 <= int(sequence_id) < physical_tables.shape[0]:
            raise ValueError("tree level block tables do not cover the request")
        attention = self.layers[0].self_attn
        logical_blocks = torch.div(
            torch.tensor(physical_positions, dtype=torch.long, device=device),
            attention.block_size,
            rounding_mode="floor",
        )
        physical_blocks = physical_tables[int(sequence_id), logical_blocks]
        if (physical_blocks < 0).any():
            raise RuntimeError("tree level referenced an unallocated KV cache page")
        position_tensor = torch.tensor(logical_positions, dtype=torch.long, device=device)
        physical_position_tensor = torch.tensor(physical_positions, dtype=torch.long, device=device)
        slot_mapping = physical_blocks * attention.block_size + physical_position_tensor.remainder(attention.block_size)
        use_fused_tree_attention = device.type == "npu" and attention.uses_paged_attention
        mask_rows = []
        for index in indices:
            row = plan.attention_mask[index + 1 if index >= 0 else 0]
            mask_rows.append(
                row
                if use_fused_tree_attention
                else row.to(device=device, dtype=torch.bool)
            )
        level_mask = torch.stack(mask_rows)
        if use_fused_tree_attention:
            attention_mask = None
            tree_attention_mask = level_mask.unsqueeze(0).unsqueeze(0).to(
                device=device,
                dtype=torch.bool,
            )
            token_tables = physical_tables[int(sequence_id) : int(sequence_id) + 1]
        else:
            attention_mask = level_mask
            tree_attention_mask = make_tree_fia_mask([level_mask])
            token_tables = physical_tables[int(sequence_id)].expand(len(indices), -1)
        metadata = NativeAttentionMetadata(
            slot_mapping=slot_mapping.to(torch.int32),
            context_lens=torch.tensor(
                [int(value) + 1 for value in physical_positions],
                dtype=torch.int32,
            ),
            block_tables=token_tables,
            actual_seq_lengths_q=(len(indices),),
            sequence_lens=(max(physical_positions) + 1,),
            request_block_tables=physical_tables[int(sequence_id) : int(sequence_id) + 1],
            attention_mask=attention_mask,
            use_fused_infer_attention=use_fused_tree_attention,
            tree_attention=True,
            tree_attention_mask=tree_attention_mask,
        )
        return torch.tensor(token_ids, dtype=torch.long, device=device), position_tensor, metadata

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_metadata: NativeAttentionMetadata | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual, attention_metadata)
        if self.track_cache_finiteness:
            self.output_nonfinite.logical_or_(~(torch.isfinite(hidden_states).all() & torch.isfinite(residual).all()))
        hidden_states, _ = self.norm(hidden_states, residual)
        if self.track_cache_finiteness:
            self.output_nonfinite.logical_or_(~torch.isfinite(hidden_states).all())
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def compute_greedy_tokens(self, hidden_states: torch.Tensor, vocabulary_size: int) -> torch.Tensor:
        return self.lm_head.greedy(hidden_states, vocabulary_size)

    def compute_greedy_tokens_with_confidence(
        self, hidden_states: torch.Tensor, vocabulary_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.lm_head.greedy_with_confidence(hidden_states, vocabulary_size)


def build_native_model(
    config,
    context: NativeTPContext,
    max_model_len: int,
    max_num_seqs: int = 1,
    *,
    configure_cache: bool = True,
    block_size: int = PAGED_ATTENTION_BLOCK_SIZE,
    num_cache_blocks: int | None = None,
) -> NativeQwen2ForCausalLM:
    """Construct an upstream-supported model directly on NPU."""
    config = prepare_native_model_config(config, context.size)
    original_dtype = torch.get_default_dtype()
    original_device = torch.get_default_device()
    model_dtype = getattr(config, "torch_dtype", torch.bfloat16)
    if isinstance(model_dtype, str):
        model_dtype = getattr(torch, model_dtype)
    try:
        torch.set_default_dtype(model_dtype)
        torch.set_default_device("npu")
        model = NativeQwen2ForCausalLM(config, context)
    finally:
        torch.set_default_dtype(original_dtype)
        torch.set_default_device(original_device)
    if configure_cache:
        model.configure_cache(max_model_len, max_num_seqs, block_size, num_cache_blocks)
    return model


def build_native_qwen2_model(
    config,
    context: NativeTPContext,
    max_model_len: int,
    max_num_seqs: int = 1,
) -> NativeQwen2ForCausalLM:
    """Backward-compatible alias for the original Qwen2-only builder."""
    return build_native_model(config, context, max_model_len, max_num_seqs)


def load_native_model_weights(model: NativeQwen2ForCausalLM, model_path: str) -> None:
    """Load supported Hugging Face safetensors with padded TP sharding."""
    weight_files = sorted(Path(model_path).glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors checkpoints found under {model_path!r}.")
    packed_mapping = model.packed_modules_mapping
    for weight_file in weight_files:
        with safe_open(weight_file, framework="pt", device="cpu") as checkpoint:
            weight_names = checkpoint.keys()
            for weight_name in weight_names:
                loaded_weight = checkpoint.get_tensor(weight_name)
                parameter_base_name = weight_name.removeprefix("model.")
                packed_match = next(
                    (
                        (source, replacement, shard_id)
                        for source, (replacement, shard_id) in packed_mapping.items()
                        if source in weight_name
                    ),
                    None,
                )
                try:
                    if packed_match is not None:
                        source, replacement, shard_id = packed_match
                        parameter_name = parameter_base_name.replace(source, replacement)
                        if replacement == "qkv_proj":
                            model.get_submodule(parameter_name.rsplit(".", 1)[0]).load_shard(
                                loaded_weight,
                                shard_id,
                                parameter_name.endswith(".bias"),
                            )
                        else:
                            model.get_submodule(parameter_name.rsplit(".", 1)[0]).load_shard(
                                loaded_weight,
                                shard_id,
                                parameter_name.endswith(".bias"),
                            )
                    else:
                        parameter = model.get_parameter(parameter_base_name)
                        module = model.get_submodule(parameter_base_name.rsplit(".", 1)[0])
                        if parameter_base_name.endswith(".weight") and hasattr(module, "load_weight"):
                            module.load_weight(loaded_weight)
                        elif parameter_base_name.endswith(".bias") and hasattr(module, "load_bias"):
                            module.load_bias(loaded_weight)
                        else:
                            parameter.data.copy_(loaded_weight)
                except AttributeError:
                    # HF checkpoints may retain rotary buffers that are derived
                    # from config in this implementation.
                    if "rotary_emb" not in weight_name:
                        raise
    _maybe_convert_linear_weights_to_nz(model)


def _maybe_convert_linear_weights_to_nz(model: NativeQwen2ForCausalLM) -> None:
    """Match vLLM-Ascend's opt-in BF16/FP16 FRACTAL_NZ weight path."""
    if ascend_envs.VLLM_ASCEND_ENABLE_NZ != 2:
        return
    tied_embeddings = getattr(model.config, "tie_word_embeddings", False)
    linear_types = (NativeColumnLinear, NativeRowLinear, NativeLMHead)
    for module in model.modules():
        if tied_embeddings and isinstance(module, NativeLMHead):
            # GatherV2 requires the tied embedding table to stay in ND format.
            continue
        if (
            isinstance(module, linear_types)
            and getattr(module, "bias", None) is None
            and module.weight.device.type == "npu"
        ):
            module.weight.data = torch_npu.npu_format_cast(
                module.weight.data,
                ACL_FORMAT_FRACTAL_NZ,
            )


def load_native_qwen2_weights(model: NativeQwen2ForCausalLM, model_path: str) -> None:
    """Backward-compatible alias for the original Qwen2-only loader."""
    load_native_model_weights(model, model_path)
