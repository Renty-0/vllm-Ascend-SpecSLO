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

import os
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from math import ceil, prod
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
    MC2_TP3_NATIVE_EPILOGUE_OPERATOR,
    MC2Profile,
    MC2Qualification,
    MC2StaticRoute,
    MC2StaticRouteManifest,
    _bind_mc2_dispatch_ticket,
    _MC2DispatchTicket,
    bind_mc2_static_route,
    build_mc2_static_route,
    build_mc2_static_route_manifest,
    detect_mc2_capability,
    matmul_allreduce_add_rmsnorm_or_fallback,
    resolve_hccl_comm_name,
    validate_mc2_static_environment,
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
TP3_DOWN_MC2_LOCAL_K = 8576
"""Qualified Qwen3-32B TP3 down-projection width per target rank."""

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
        return bool(ascend_envs.VLLM_ASCEND_PEARL_ENABLE_TP3_MM_ALL_REDUCE and not _TP3_MM_ALL_REDUCE_DISABLED)
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
    ffn_shift = int(getattr(prepared, "pearl_tp3_balanced_ffn_shift", 0))
    light_rank = int(getattr(prepared, "pearl_tp3_light_rank", -1))
    if light_rank not in (-1, 0, 1, 2):
        raise ValueError("PEARL TP3 light rank must be -1, 0, 1, or 2.")
    if ffn_shift and light_rank >= 0:
        raise ValueError("Balanced FFN shift and TP3 light-rank sharding are mutually exclusive.")
    if ffn_shift or light_rank >= 0:
        if tensor_parallel_size != 3:
            raise ValueError("Exact PEARL attention/FFN shards currently require TP3.")
        if ffn_shift % TENSOR_CORE_TILE_SIZE:
            raise ValueError(
                f"Balanced PEARL FFN shift must be a multiple of {TENSOR_CORE_TILE_SIZE}."
            )
        valid_kv_heads = int(prepared.valid_num_key_value_heads)
        valid_q_heads = int(prepared.valid_num_attention_heads)
        if valid_q_heads % valid_kv_heads:
            raise ValueError("Balanced PEARL shards require integral GQA groups.")
        kv_base, kv_remainder = divmod(valid_kv_heads, tensor_parallel_size)
        if light_rank >= 0:
            # Qwen3-32B has eight KV groups.  TP3 therefore has two heavy
            # shards and one naturally light shard.  Put the light shard on
            # the observed target leader/critical rank without increasing
            # either peer above the padded default (3 KV groups each).
            if kv_remainder != tensor_parallel_size - 1:
                raise ValueError(
                    "TP3 light-rank sharding requires exactly one short KV partition."
                )
            kv_partitions = tuple(
                kv_base + int(rank != light_rank) for rank in range(tensor_parallel_size)
            )
        else:
            kv_partitions = tuple(
                kv_base + int(rank < kv_remainder) for rank in range(tensor_parallel_size)
            )
        gqa_ratio = valid_q_heads // valid_kv_heads
        q_partitions = tuple(kv_heads * gqa_ratio for kv_heads in kv_partitions)
        intermediate_units = int(prepared.valid_intermediate_size) // TENSOR_CORE_TILE_SIZE
        if intermediate_units * TENSOR_CORE_TILE_SIZE != int(prepared.valid_intermediate_size):
            raise ValueError("Balanced PEARL FFN size must align to the tensor-core tile.")
        unit_base, unit_remainder = divmod(intermediate_units, tensor_parallel_size)
        intermediate_partitions = [
            (unit_base + int(rank < unit_remainder)) * TENSOR_CORE_TILE_SIZE
            for rank in range(tensor_parallel_size)
        ]
        if light_rank >= 0:
            # Rotate the one naturally short exact FFN shard to the same
            # leader rank.  The two peers retain the padded-default width;
            # no rank receives extra matrix work.
            short_size = min(intermediate_partitions)
            long_size = max(intermediate_partitions)
            if intermediate_partitions.count(short_size) != 1:
                raise ValueError(
                    "TP3 light-rank sharding requires exactly one short FFN partition."
                )
            intermediate_partitions = [
                short_size if rank == light_rank else long_size
                for rank in range(tensor_parallel_size)
            ]
        else:
            intermediate_partitions[0] -= ffn_shift
            intermediate_partitions[1] -= ffn_shift
            intermediate_partitions[2] += 2 * ffn_shift
        if min(intermediate_partitions) <= 0:
            raise ValueError("Balanced PEARL FFN shift leaves an empty TP shard.")
        prepared.pearl_q_head_partitions = q_partitions
        prepared.pearl_kv_head_partitions = kv_partitions
        prepared.pearl_intermediate_partitions = tuple(intermediate_partitions)
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


def _pad_token_rows(
    hidden_states: torch.Tensor,
    multiple: int,
) -> tuple[torch.Tensor, int]:
    """Pad a packed 2-D token matrix for shape-stable target GEMMs.

    The padded rows are local to one projection and are removed immediately
    after it.  They therefore never enter attention metadata or the KV cache.
    """
    real_rows = int(hidden_states.shape[0])
    if multiple <= 1 or hidden_states.ndim != 2:
        return hidden_states, real_rows
    padded_rows = ((real_rows + multiple - 1) // multiple) * multiple
    if padded_rows == real_rows:
        return hidden_states, real_rows
    return F.pad(hidden_states, (0, 0, 0, padded_rows - real_rows)), real_rows


@dataclass(frozen=True)
class NativeTPContext:
    """The model-parallel coordinates for one PEARL model group."""

    group: dist.ProcessGroup
    rank: int
    size: int
    leader_rank: int


@dataclass
class _MC2IntraLayerChain:
    """Carry one explicit MC2 mailbox dependency across the whole model.

    Attention publishes its READ_DONE records without waiting for both peers.
    The following down projection performs its independent local MatMul first,
    then consumes this state before reusing the symmetric payload window.  The
    chain continues through later layers and only the final down projection
    flushes, so every ACLGraph replay still starts and ends with a drained
    mailbox while intermediate communication tails can overlap useful work.
    """

    state: torch.Tensor
    layer_count: int
    defer_read_done: bool = True
    next_step: int = 0
    _pending_key: tuple[int, str] | None = None

    def begin(self, route: MC2StaticRoute) -> tuple[torch.Tensor, bool]:
        if self._pending_key is not None:
            raise RuntimeError("MC2 chain dispatch was begun twice without committing its state")
        expected_layer = self.next_step // 2
        expected_kind = "attention" if self.next_step % 2 == 0 else "down"
        key = (int(route.layer_index), str(route.projection_kind))
        if key != (expected_layer, expected_kind):
            raise RuntimeError(
                "MC2 intra-layer chain route order changed after static admission: "
                f"expected {(expected_layer, expected_kind)!r}, got {key!r}"
            )
        if expected_layer >= self.layer_count:
            raise RuntimeError("MC2 intra-layer chain consumed more routes than the model owns")
        self._pending_key = key
        flush = not self.defer_read_done or (
            expected_kind == "down" and expected_layer + 1 == self.layer_count
        )
        return self.state, flush

    def commit(self, route: MC2StaticRoute, next_state: torch.Tensor) -> None:
        key = (int(route.layer_index), str(route.projection_kind))
        if self._pending_key != key:
            raise RuntimeError("MC2 chain state was committed by a different static route")
        if next_state.dtype != torch.int64 or tuple(next_state.shape) != (64, 4):
            raise RuntimeError("MC2 chained operator returned an invalid dependency state")
        self.state = next_state
        self.next_step += 1
        self._pending_key = None

    def finish(self) -> None:
        if self._pending_key is not None or self.next_step != 2 * self.layer_count:
            raise RuntimeError(
                "MC2 intra-layer chain did not consume exactly one attention/down pair per layer"
            )


def _qualified_mc2_chain_row_counts(
    profile: MC2Profile,
    routes: Sequence[MC2StaticRoute],
    *,
    layer_count: int,
    graph_execution: bool = True,
) -> frozenset[int]:
    """Return rows admitted by every route and explicit chain evidence."""

    chain_evidence = profile.metadata.get("deferred_read_done_chain")
    if (
        not graph_execution
        or profile.metadata.get("operator") != MC2_TP3_NATIVE_EPILOGUE_OPERATOR
        or not isinstance(chain_evidence, Mapping)
        or chain_evidence.get("qualified") is not True
        or chain_evidence.get("execution_mode") != "graph"
        or chain_evidence.get("chain_scope") != "whole_model"
        or int(chain_evidence.get("layer_count", 0)) != layer_count
        or int(chain_evidence.get("operations_per_chain", 0)) != 2 * layer_count
        or len(routes) != 2 * layer_count
        or not all(route.enabled for route in routes)
    ):
        return frozenset()
    qualified_sets = [
        {
            int(rows)
            for rows, qualification in route.decisions.items()
            if qualification.qualified
        }
        for route in routes
    ]
    return (
        frozenset(set.intersection(*qualified_sets))
        if qualified_sets
        else frozenset()
    )


def _qualified_mc2_chained_flush_row_counts(
    profile: MC2Profile,
    routes: Sequence[MC2StaticRoute],
    *,
    layer_count: int,
    graph_execution: bool = True,
) -> frozenset[int]:
    """Return rows measured with the allocation-free chained flush ABI."""

    if (
        not graph_execution
        or profile.metadata.get("operator") != MC2_TP3_NATIVE_EPILOGUE_OPERATOR
        or profile.metadata.get("standalone_chained_flush") is not True
        or len(routes) != 2 * layer_count
        or not all(route.enabled for route in routes)
    ):
        return frozenset()
    qualified_sets = [
        {
            int(rows)
            for rows, qualification in route.decisions.items()
            if qualification.qualified
        }
        for route in routes
    ]
    return (
        frozenset(set.intersection(*qualified_sets))
        if qualified_sets
        else frozenset()
    )


class _MC2RealInputCapture:
    """One-shot real-layer inputs for offline MC2 qualification.

    This probe is intentionally incompatible with ACLGraph capture and must
    not be enabled during performance measurements. Its output is consumed by
    ``examples/measure_specslo_mc2.py --input-dir
    <capture-dir>/<kind>/layer-<L>/m<M>``.  Projection and layer are part of
    the path so an attention o_proj observation can never consume or replace
    the down_proj observation for the same flattened row count.
    """

    def __init__(
        self,
        directory: Path,
        rows: frozenset[int],
        context: NativeTPContext,
        *,
        enforce_eager: bool | None,
        projection_kind: str = "attention",
        layer_filter: frozenset[int] | None = None,
    ) -> None:
        self.directory = directory
        self.rows = rows
        self.rank = int(context.rank)
        self.tp_size = int(context.size)
        self.enforce_eager = enforce_eager
        self.projection_kind = projection_kind
        self.layer_filter = layer_filter
        self._implicit_layer: int | None = None
        self._completed: set[tuple[int, int]] = set()

    @classmethod
    def from_env(
        cls,
        config: object,
        context: NativeTPContext,
    ) -> _MC2RealInputCapture | None:
        raw_directory = ascend_envs.VLLM_ASCEND_PEARL_MC2_CAPTURE_DIR
        raw_rows = ascend_envs.VLLM_ASCEND_PEARL_MC2_CAPTURE_ROWS
        raw_kind = ascend_envs.VLLM_ASCEND_PEARL_MC2_CAPTURE_KIND
        raw_layers = ascend_envs.VLLM_ASCEND_PEARL_MC2_CAPTURE_LAYERS
        if not raw_directory and not raw_rows:
            if raw_layers or raw_kind.strip().lower() != "attention":
                raise ValueError(
                    "MC2 capture kind/layers require "
                    "VLLM_ASCEND_PEARL_MC2_CAPTURE_DIR and "
                    "VLLM_ASCEND_PEARL_MC2_CAPTURE_ROWS."
                )
            return None
        if not raw_directory or not raw_rows:
            raise ValueError(
                "VLLM_ASCEND_PEARL_MC2_CAPTURE_DIR and "
                "VLLM_ASCEND_PEARL_MC2_CAPTURE_ROWS must be set together; "
                "the MC2 input probe is enforce-eager diagnostics only."
            )

        directory = Path(raw_directory).expanduser()
        if not directory.is_absolute() or directory == Path(directory.anchor):
            raise ValueError("MC2 capture directory must be an absolute, non-root path.")
        if directory.exists() and not directory.is_dir():
            raise ValueError(f"MC2 capture path is not a directory: {directory}")

        tokens = raw_rows.split(",")
        if any(not token.strip() or not token.strip().isdigit() for token in tokens):
            raise ValueError("MC2 capture rows must be comma-separated positive integers.")
        parsed_rows = tuple(int(token.strip()) for token in tokens)
        if any(row <= 0 for row in parsed_rows) or len(set(parsed_rows)) != len(parsed_rows):
            raise ValueError("MC2 capture rows must be unique positive integers.")

        projection_kind = raw_kind.strip().lower()
        if projection_kind not in {"attention", "down"}:
            raise ValueError("MC2 capture kind must be 'attention' or 'down'.")
        layer_filter: frozenset[int] | None = None
        if raw_layers:
            layer_tokens = raw_layers.split(",")
            if any(not token.strip() or not token.strip().isdigit() for token in layer_tokens):
                raise ValueError("MC2 capture layers must be comma-separated non-negative integers.")
            parsed_layers = tuple(int(token.strip()) for token in layer_tokens)
            if len(set(parsed_layers)) != len(parsed_layers):
                raise ValueError("MC2 capture layers must be unique non-negative integers.")
            num_hidden_layers = getattr(config, "num_hidden_layers", None)
            if num_hidden_layers is not None and any(
                layer >= int(num_hidden_layers) for layer in parsed_layers
            ):
                raise ValueError("MC2 capture layer is outside the configured decoder.")
            layer_filter = frozenset(parsed_layers)

        configured_eager = getattr(
            config,
            "pearl_enforce_eager",
            getattr(config, "enforce_eager", None),
        )
        enforce_eager = None if configured_eager is None else bool(configured_eager)
        if enforce_eager is False:
            raise ValueError(
                "MC2 real-input capture is allowed only for enforce-eager diagnostics; "
                "disable ACLGraph and do not use the probe for performance measurements."
            )
        if context.size <= 1 or not 0 <= context.rank < context.size:
            raise ValueError("MC2 capture requires valid tensor-parallel rank metadata.")
        return cls(
            directory.resolve(strict=False),
            frozenset(parsed_rows),
            context,
            enforce_eager=enforce_eager,
            projection_kind=projection_kind,
            layer_filter=layer_filter,
        )

    def _reject_graph_capture(self, activation: torch.Tensor) -> None:
        if self.enforce_eager is True or activation.device.type != "npu":
            return
        is_capturing = getattr(getattr(torch, "npu", None), "is_current_stream_capturing", None)
        if is_capturing is None:
            raise RuntimeError(
                "Cannot verify enforce-eager mode for MC2 input capture; pass an "
                "enforce_eager=True model config and do not run performance measurements."
            )
        if bool(is_capturing()):
            raise RuntimeError(
                "MC2 real-input capture cannot run during ACLGraph capture; "
                "use enforce-eager diagnostics and remove the capture environment variables before measurements."
            )

    def maybe_capture(
        self,
        activation: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        *,
        projection_kind: str = "attention",
        layer_index: int = 0,
    ) -> None:
        if projection_kind not in {"attention", "down"}:
            raise ValueError("MC2 capture projection kind must be 'attention' or 'down'.")
        if projection_kind != self.projection_kind:
            return
        layer_index = int(layer_index)
        if layer_index < 0:
            raise ValueError("MC2 capture layer index must be non-negative.")
        if self.layer_filter is not None:
            if layer_index not in self.layer_filter:
                return
        elif self._implicit_layer is not None and layer_index != self._implicit_layer:
            return
        if activation.ndim < 2:
            raise ValueError("MC2 capture activation must have at least two dimensions.")
        rows = int(prod(activation.shape[:-1]))
        if rows not in self.rows or (layer_index, rows) in self._completed:
            return
        if self.layer_filter is None and self._implicit_layer is None:
            self._implicit_layer = layer_index
        self._reject_graph_capture(activation)

        local_width = int(activation.shape[-1])
        if weight.ndim != 2 or tuple(weight.shape) != (int(residual.shape[-1]), local_width):
            raise ValueError("MC2 capture projection weight is incompatible with activation/residual shapes.")
        if residual.shape[:-1] != activation.shape[:-1] or gamma.numel() != residual.shape[-1]:
            raise ValueError("MC2 capture residual/gamma shapes are incompatible with the activation.")

        row_directory = (
            self.directory
            / projection_kind
            / f"layer-{layer_index}"
            / f"m{rows}"
        )
        destination = row_directory / f"rank-{self.rank}.pt"
        if destination.exists():
            self._completed.add((layer_index, rows))
            return
        row_directory.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            self._completed.add((layer_index, rows))
            return

        payload = {
            "activation": activation.detach().reshape(rows, local_width).to(device="cpu").contiguous().clone(),
            "weight": weight.detach().to(device="cpu").contiguous().clone(),
            "residual": residual.detach().reshape(rows, int(residual.shape[-1])).to(device="cpu").contiguous().clone(),
            "gamma": gamma.detach().reshape(-1).to(device="cpu").contiguous().clone(),
        }
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".rank-{self.rank}.",
            suffix=".tmp",
            dir=row_directory,
        )
        os.close(file_descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        self._completed.add((layer_index, rows))


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

    def qualify_forward_mc2(
        self,
        local_hidden_states: torch.Tensor,
        residual: torch.Tensor,
        projection_weight: torch.Tensor,
        context: NativeTPContext,
        *,
        projection_bias: torch.Tensor | None = None,
        profile: MC2Profile | None = None,
    ) -> _MC2DispatchTicket:
        """Qualify this exact projection once before entering the MC2 adapter.

        Native PEARL's ordinary attention output path owns several production
        optimizations, including row padding, alternate weight formats and the
        native AddRMSNorm operator.  An unavailable or unprofiled MC2 shape
        must therefore stay on that path instead of entering the adapter's
        generic correctness fallback.
        """

        if projection_bias is not None:
            qualification = MC2Qualification(False, "MC2 production ABI does not support projection bias")
        elif profile is None:
            qualification = MC2Qualification(False, "no identity-bound MC2 qualification profile")
        else:
            operator_kind = getattr(profile, "metadata", {}).get("operator")
            capability = (
                detect_mc2_capability(
                    local_hidden_states.device,
                    context.size,
                    operator=operator_kind,
                )
                if operator_kind == MC2_TP3_NATIVE_EPILOGUE_OPERATOR
                else detect_mc2_capability(local_hidden_states.device, context.size)
            )
            qualification = (
                profile.qualify(
                    local_hidden_states,
                    projection_weight,
                    residual,
                    tp_size=context.size,
                    is_trans_b=True,
                    tp_rank_id=context.rank,
                    epsilon=self.eps,
                )
                if capability.available
                else MC2Qualification(False, capability.reason)
            )
        return _bind_mc2_dispatch_ticket(
            qualification,
            local_hidden_states,
            projection_weight,
            residual,
            self.weight,
            tp_rank_size=context.size,
            tp_rank_id=context.rank,
            epsilon=self.eps,
            is_trans_b=True,
            profile=profile,
        )

    def can_forward_mc2(
        self,
        local_hidden_states: torch.Tensor,
        residual: torch.Tensor,
        projection_weight: torch.Tensor,
        context: NativeTPContext,
        *,
        projection_bias: torch.Tensor | None = None,
        profile: MC2Profile | None = None,
    ) -> bool:
        """Compatibility boolean wrapper around the single qualification path."""

        return self.qualify_forward_mc2(
            local_hidden_states,
            residual,
            projection_weight,
            context,
            projection_bias=projection_bias,
            profile=profile,
        ).qualification.qualified

    def forward_mc2(
        self,
        local_hidden_states: torch.Tensor,
        residual: torch.Tensor,
        projection_weight: torch.Tensor,
        context: NativeTPContext,
        *,
        projection_bias: torch.Tensor | None = None,
        profile: MC2Profile | None = None,
        qualification: _MC2DispatchTicket | None = None,
        static_route: MC2StaticRoute | None = None,
        mc2_chain: _MC2IntraLayerChain | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse TP projection, all-reduce, residual add and RMSNorm when available."""
        if projection_bias is not None:
            raise RuntimeError("Qualified MC2 dispatch cannot consume projection bias")
        if static_route is not None:
            if (
                static_route.tp_rank_size != context.size
                or static_route.tp_rank_id != context.rank
                or static_route.weight_identity != id(projection_weight)
                or static_route.gamma_identity != id(self.weight)
            ):
                raise RuntimeError("MC2 static route does not match this decoder projection")
            comm_name = static_route.group_tp
        else:
            comm_name = resolve_hccl_comm_name(
                context.group,
                device=local_hidden_states.device,
                rank=context.rank,
            )
        if context.size > 1 and not comm_name:
            raise RuntimeError("Qualified MC2 dispatch could not resolve the target HCCL communicator.")
        chain_state = None
        flush_chain = True
        if mc2_chain is not None:
            if static_route is None:
                raise RuntimeError("MC2 chaining requires a frozen static route")
            chain_state, flush_chain = mc2_chain.begin(static_route)
        outputs = matmul_allreduce_add_rmsnorm_or_fallback(
            local_hidden_states,
            projection_weight,
            residual,
            self.weight,
            group_tp=comm_name,
            tp_rank_size=context.size,
            tp_rank_id=context.rank,
            epsilon=self.eps,
            # The following decoder layer consumes the pre-normalization
            # residual.  The custom MC2 kernel only materializes that second
            # result for the GatherAddOut specialization, so production must
            # request it explicitly instead of relying on the wrapper default.
            is_gather_add_out=True,
            process_group=context.group,
            use_fused=True,
            # ``can_forward_mc2`` has already established an exact qualified
            # shape.  An exception here must abort graph capture instead of
            # silently baking the split fallback into an allegedly MC2 graph.
            strict_fused=True,
            profile=profile,
            prequalification=qualification,
            chain_state=chain_state,
            flush_chain=flush_chain,
        )
        if mc2_chain is None:
            # Preserve the legacy two-output object's identity.  Besides
            # avoiding an unnecessary tuple rebuild, a few model-runner
            # adapters use identity to distinguish the native non-chained
            # ABI from the explicit three-output chained ABI below.
            return outputs  # type: ignore[return-value]
        if len(outputs) != 3:
            raise RuntimeError("MC2 chained dispatch did not return its next dependency state")
        normalized, next_residual, next_chain_state = outputs
        assert static_route is not None
        mc2_chain.commit(static_route, next_chain_state)
        return normalized, next_residual


class NativeColumnLinear(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        context: NativeTPContext,
        bias: bool = False,
        output_partition_sizes: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.context = context
        self.output_size = output_size
        if output_partition_sizes is None:
            output_partition_sizes = (_divide(output_size, context.size),) * context.size
        if len(output_partition_sizes) != context.size or sum(output_partition_sizes) != output_size:
            raise ValueError("Column-parallel partition sizes must cover the complete output.")
        self.output_partition_sizes = tuple(int(size) for size in output_partition_sizes)
        self.output_size_per_rank = self.output_partition_sizes[context.rank]
        self.output_start = sum(self.output_partition_sizes[: context.rank])
        self.weight = nn.Parameter(torch.empty(self.output_size_per_rank, input_size))
        self.bias = nn.Parameter(torch.empty(self.output_size_per_rank)) if bias else None
        self.token_pad_multiple = 1

    def load_weight(self, loaded_weight: torch.Tensor) -> None:
        _copy_padded_shard(self.weight.data, loaded_weight, dim=0, start=self.output_start)

    def load_bias(self, loaded_bias: torch.Tensor) -> None:
        assert self.bias is not None
        _copy_padded_shard(self.bias.data, loaded_bias, dim=0, start=self.output_start)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        padded_states, real_rows = _pad_token_rows(hidden_states, self.token_pad_multiple)
        output = F.linear(padded_states, self.weight, self.bias)
        return output[:real_rows]


class NativeMergedColumnLinear(NativeColumnLinear):
    def __init__(
        self,
        input_size: int,
        output_sizes: tuple[int, int],
        context: NativeTPContext,
        bias: bool = False,
        shard_partition_sizes: Sequence[int] | None = None,
    ) -> None:
        self.output_sizes = output_sizes
        self.shard_partition_sizes = (
            tuple(int(size) for size in shard_partition_sizes)
            if shard_partition_sizes is not None
            else None
        )
        if self.shard_partition_sizes is not None:
            if len(self.shard_partition_sizes) != context.size:
                raise ValueError("Merged-column shard partitions must match TP size.")
            if any(sum(self.shard_partition_sizes) != size for size in output_sizes):
                raise ValueError("Merged-column shard partitions must cover every logical output.")
            output_partition_sizes = tuple(
                len(output_sizes) * size for size in self.shard_partition_sizes
            )
        else:
            output_partition_sizes = None
        super().__init__(
            input_size,
            sum(output_sizes),
            context,
            bias=bias,
            output_partition_sizes=output_partition_sizes,
        )

    def load_shard(self, loaded_weight: torch.Tensor, shard_id: int, is_bias: bool = False) -> None:
        shard_output_size = self.output_sizes[shard_id]
        if self.shard_partition_sizes is None:
            shard_per_rank = _divide(shard_output_size, self.context.size)
            source_start = self.context.rank * shard_per_rank
        else:
            shard_per_rank = self.shard_partition_sizes[self.context.rank]
            source_start = sum(self.shard_partition_sizes[: self.context.rank])
        destination_start = shard_id * shard_per_rank
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
        q_head_partitions: Sequence[int] | None = None,
        kv_head_partitions: Sequence[int] | None = None,
    ) -> None:
        self.head_dim = head_dim
        if (q_head_partitions is None) != (kv_head_partitions is None):
            raise ValueError("Q and KV head partitions must be configured together.")
        if q_head_partitions is None:
            q_head_partitions = (_divide(num_heads, context.size),) * context.size
            kv_head_partitions = (_divide(num_kv_heads, context.size),) * context.size
        assert kv_head_partitions is not None
        if (
            len(q_head_partitions) != context.size
            or len(kv_head_partitions) != context.size
            or sum(q_head_partitions) != num_heads
            or sum(kv_head_partitions) != num_kv_heads
        ):
            raise ValueError("QKV head partitions must cover all configured heads.")
        self.q_head_partitions = tuple(int(size) for size in q_head_partitions)
        self.kv_head_partitions = tuple(int(size) for size in kv_head_partitions)
        self.num_heads_per_rank = self.q_head_partitions[context.rank]
        self.num_kv_heads_per_rank = self.kv_head_partitions[context.rank]
        self.q_size = self.num_heads_per_rank * head_dim
        self.kv_size = self.num_kv_heads_per_rank * head_dim
        output_partition_sizes = tuple(
            (q_heads + 2 * kv_heads) * head_dim
            for q_heads, kv_heads in zip(self.q_head_partitions, self.kv_head_partitions)
        )
        super().__init__(
            hidden_size,
            (num_heads + 2 * num_kv_heads) * head_dim,
            context,
            bias=bias,
            output_partition_sizes=output_partition_sizes,
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
        partitions = self.q_head_partitions if shard_id == "q" else self.kv_head_partitions
        source_start = sum(partitions[: self.context.rank]) * self.head_dim
        destination = self.bias if is_bias else self.weight
        assert destination is not None
        _copy_padded_shard(
            destination.data.narrow(0, destination_start, shard_size),
            loaded_weight,
            dim=0,
            start=source_start,
        )


class NativeRowLinear(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        context: NativeTPContext,
        bias: bool = False,
        input_partition_sizes: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.context = context
        if input_partition_sizes is None:
            input_partition_sizes = (_divide(input_size, context.size),) * context.size
        if len(input_partition_sizes) != context.size or sum(input_partition_sizes) != input_size:
            raise ValueError("Row-parallel partition sizes must cover the complete input.")
        self.input_partition_sizes = tuple(int(size) for size in input_partition_sizes)
        self.input_size_per_rank = self.input_partition_sizes[context.rank]
        self.input_start = sum(self.input_partition_sizes[: context.rank])
        self.weight = nn.Parameter(torch.empty(output_size, self.input_size_per_rank))
        self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
        self.token_pad_multiple = 1
        self.register_buffer("large_m_nz_weight", None, persistent=False)
        self.large_m_nz_min_rows = 0
        self.pre_resolved_comm_name: str | None = None

    def load_weight(self, loaded_weight: torch.Tensor) -> None:
        _copy_padded_shard(self.weight.data, loaded_weight, dim=1, start=self.input_start)

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
        hidden_states, real_rows = _pad_token_rows(hidden_states, self.token_pad_multiple)
        projection_weight = self.weight
        if (
            self.large_m_nz_weight is not None
            and self.large_m_nz_min_rows > 0
            and real_rows >= self.large_m_nz_min_rows
        ):
            projection_weight = self.large_m_nz_weight
        fused_mm_reduce = getattr(torch_npu, "npu_mm_all_reduce_base", None)
        allow_tp3_fused = self.context.size == 3 and _use_native_fused_mm_all_reduce(
            self.context.size, hidden_states.device.type
        )
        if (
            _use_native_fused_mm_all_reduce(self.context.size, hidden_states.device.type)
            and fused_mm_reduce is not None
        ):
            hcomm_info = self.pre_resolved_comm_name
            if hcomm_info is None:
                hcomm_info = resolve_hccl_comm_name(
                    self.context.group,
                    device=hidden_states.device,
                    rank=self.context.rank,
                )
            if hcomm_info:
                try:
                    output = fused_mm_reduce(
                        hidden_states,
                        projection_weight.t(),
                        hcomm_info,
                        bias=self.bias,
                    )
                    return output[:real_rows]
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
        output = F.linear(hidden_states, projection_weight, self.bias)
        if self.context.size > 1:
            dist.all_reduce(output, group=self.context.group)
        return output[:real_rows]


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
        tp1_greedy_argmax: bool = False,
    ) -> None:
        super().__init__(vocab_size, hidden_size, context)
        if tp1_greedy_argmax and context.size != 1:
            raise ValueError("The greedy argmax fast path is supported only for TP1.")
        self.track_cache_finiteness = track_cache_finiteness
        self.tp1_greedy_argmax = tp1_greedy_argmax
        self.token_pad_multiple = 1
        self.register_buffer("logits_nonfinite", torch.zeros((), dtype=torch.bool), persistent=False)

    def _project_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        padded_states, real_rows = _pad_token_rows(hidden_states, self.token_pad_multiple)
        return F.linear(padded_states, self.weight)[:real_rows]

    def _track_logits(self, logits: torch.Tensor) -> None:
        if self.track_cache_finiteness:
            # Inspect only real local logits, before any synthetic -inf used
            # to exclude empty vocabulary shards from the TP argmax.
            self.logits_nonfinite.logical_or_(~torch.isfinite(logits).all())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        local_logits = self._project_local(hidden_states)
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
            local_logits = self._project_local(hidden_states)
            local_logits = local_logits[:, :local_vocabulary_size]
            self._track_logits(local_logits)
            if self.tp1_greedy_argmax:
                # TP1 does not need the winning value for a cross-rank
                # reduction. Request only the token ID so Ascend can select
                # its ArgMax path instead of ArgMaxWithValue.
                return torch.argmax(local_logits, dim=-1)
            local_values, local_token_ids = local_logits.max(dim=-1)
            # TP1 owns the vocabulary from offset zero.  Avoid recording an
            # otherwise redundant in-place add in every captured draft step.
            if self.vocab_start:
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
            local_logits = self._project_local(hidden_states)[:, :local_vocabulary_size]
            self._track_logits(local_logits)
            if self.context.size == 1:
                # TP1 owns the complete vocabulary.  Computing max(logits)
                # and logsumexp(logits) separately materializes the 151K-wide
                # FP32 view twice on Ascend.  A single FP32 softmax followed by
                # max returns the exact greedy token probability and uses one
                # cast/reduction path, which is also stable under ACLGraph.
                probabilities = torch.softmax(
                    local_logits,
                    dim=-1,
                    dtype=torch.float32,
                )
                confidence, local_token_ids = probabilities.max(dim=-1)
                return local_token_ids, confidence
            local_logits_float = local_logits.float()
            local_values, local_token_ids = local_logits_float.max(dim=-1)
            if self.vocab_start:
                local_token_ids += self.vocab_start
            local_logsumexp = torch.logsumexp(local_logits_float, dim=-1)

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
        self.use_device_paged_attention = bool(
            getattr(config, "pearl_device_paged_attention", False)
        )
        architecture = getattr(config, "architectures", ("Qwen2ForCausalLM",))[0]
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        q_head_partitions = getattr(config, "pearl_q_head_partitions", None)
        kv_head_partitions = getattr(config, "pearl_kv_head_partitions", None)
        if q_head_partitions is None:
            total_num_heads = config.num_attention_heads
            total_num_kv_heads = config.num_key_value_heads
            self.num_heads = _divide(total_num_heads, context.size)
            self.num_kv_heads = _divide(total_num_kv_heads, context.size)
            attention_input_partitions = None
        else:
            if kv_head_partitions is None:
                raise ValueError("Balanced Q partitions require KV partitions.")
            total_num_heads = sum(q_head_partitions)
            total_num_kv_heads = sum(kv_head_partitions)
            self.num_heads = int(q_head_partitions[context.rank])
            self.num_kv_heads = int(kv_head_partitions[context.rank])
            attention_input_partitions = tuple(
                int(heads) * self.head_dim for heads in q_head_partitions
            )
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
            total_num_heads,
            total_num_kv_heads,
            context,
            bias=qkv_bias,
            q_head_partitions=q_head_partitions,
            kv_head_partitions=kv_head_partitions,
        )
        self.o_proj = NativeRowLinear(
            total_num_heads * self.head_dim,
            config.hidden_size,
            context,
            bias=attention_bias if architecture == "LlamaForCausalLM" else False,
            input_partition_sizes=attention_input_partitions,
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

    def _paged_attention(
        self,
        query: torch.Tensor,
        positions: torch.Tensor,
        metadata: NativeAttentionMetadata,
    ) -> torch.Tensor:
        assert self.key_cache is not None and self.value_cache is not None
        attended = torch.empty_like(query)
        if self.use_device_paged_attention:
            from vllm_ascend.ops.triton.spec_decode.device_paged_attention import (
                device_paged_attention,
            )

            return device_paged_attention(
                query,
                self.key_cache,
                self.value_cache,
                metadata.block_tables,
                positions,
                scale=self.scale,
                output=attended,
                # A 64-token tile is the fastest numerically-qualified shape
                # on 910B2 for Qwen3's GQA=2 draft attention.  The 128-token
                # physical page still divides exactly, while halving the
                # online-softmax loop count compared with the prototype's
                # original 32-token tile.
                tokens_per_iteration=64,
                lengths_are_positions=True,
            )
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
        attention_mask = metadata.tree_attention_mask if metadata.tree_attention else metadata.attention_mask
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
            attended = self._paged_attention(query, positions, metadata)
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
        intermediate_partitions = getattr(config, "pearl_intermediate_partitions", None)
        intermediate_size = (
            sum(intermediate_partitions)
            if intermediate_partitions is not None
            else config.intermediate_size
        )
        self.gate_up_proj = NativeMergedColumnLinear(
            config.hidden_size,
            (intermediate_size, intermediate_size),
            context,
            bias=bool(getattr(config, "mlp_bias", False)),
            shard_partition_sizes=intermediate_partitions,
        )
        self.down_proj = NativeRowLinear(
            intermediate_size,
            config.hidden_size,
            context,
            bias=bool(getattr(config, "mlp_bias", False)),
            input_partition_sizes=intermediate_partitions,
        )

    def _activate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(hidden_states)
        if gate_up.device.type == "npu":
            return torch_npu.npu_swiglu(gate_up)
        gate, up = gate_up.chunk(2, dim=-1)
        return F.silu(gate) * up

    def _maybe_capture_down(
        self,
        activation: torch.Tensor,
        *,
        mc2_input_capture: _MC2RealInputCapture | None,
        residual: torch.Tensor | None,
        next_layernorm_gamma: torch.Tensor | None,
        layer_index: int,
    ) -> None:
        if mc2_input_capture is None:
            return
        if residual is None or next_layernorm_gamma is None:
            if mc2_input_capture.projection_kind == "down":
                raise ValueError(
                    "down_proj MC2 capture requires the residual and next normalization gamma."
                )
            return
        mc2_input_capture.maybe_capture(
            activation,
            self.down_proj.weight,
            residual,
            next_layernorm_gamma,
            projection_kind="down",
            layer_index=layer_index,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        mc2_input_capture: _MC2RealInputCapture | None = None,
        residual: torch.Tensor | None = None,
        next_layernorm_gamma: torch.Tensor | None = None,
        layer_index: int = 0,
    ) -> torch.Tensor:
        activation = self._activate(hidden_states)
        self._maybe_capture_down(
            activation,
            mc2_input_capture=mc2_input_capture,
            residual=residual,
            next_layernorm_gamma=next_layernorm_gamma,
            layer_index=layer_index,
        )
        return self.down_proj(activation)

    def forward_down_mc2(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        next_layernorm: NativeRMSNorm,
        context: NativeTPContext,
        *,
        profile: MC2Profile,
        static_route: MC2StaticRoute | None = None,
        mc2_input_capture: _MC2RealInputCapture | None = None,
        layer_index: int = 0,
        mc2_chain: _MC2IntraLayerChain | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Try down_proj + all-reduce + the following RMSNorm as one op.

        The third result is a host/static execution-state bit.  When true,
        the returned hidden state is already normalized for the next decoder
        layer (or by the model's final norm) and the returned residual is the
        pre-normalization add output.  A false result keeps the ordinary
        down-projection semantics and leaves ``residual`` unchanged.
        """

        activation = self._activate(hidden_states)
        self._maybe_capture_down(
            activation,
            mc2_input_capture=mc2_input_capture,
            residual=residual,
            next_layernorm_gamma=next_layernorm.weight,
            layer_index=layer_index,
        )
        if static_route is None:
            qualification = next_layernorm.qualify_forward_mc2(
                activation,
                residual,
                self.down_proj.weight,
                context,
                projection_bias=self.down_proj.bias,
                profile=profile,
            )
        else:
            qualification = bind_mc2_static_route(
                static_route,
                profile,
                activation,
                self.down_proj.weight,
                residual,
                next_layernorm.weight,
            )
        if not qualification.qualification.qualified:
            if mc2_chain is not None:
                raise RuntimeError(
                    "MC2 intra-layer chain lost its qualified down route after static admission"
                )
            return self.down_proj(activation), residual, False
        normalized, next_residual = next_layernorm.forward_mc2(
            activation,
            residual,
            self.down_proj.weight,
            context,
            projection_bias=self.down_proj.bias,
            profile=profile,
            qualification=qualification,
            static_route=static_route,
            mc2_chain=mc2_chain,
        )
        return normalized, next_residual, True


def _enable_tp3_down_mc2(
    mlp: NativeQwen2MLP,
    context: NativeTPContext,
    *,
    enable_mc2: bool,
    profile: MC2Profile | None,
) -> bool:
    """Fail closed unless the qualified Qwen3-32B TP3 FFN is exact."""

    down_partitions = mlp.down_proj.input_partition_sizes
    return bool(
        ascend_envs.VLLM_ASCEND_PEARL_MC2_DOWN_PROJ
        and enable_mc2
        and profile is not None
        and context.size == 3
        and down_partitions == (TP3_DOWN_MC2_LOCAL_K,) * 3
        and mlp.down_proj.weight.shape[1] == TP3_DOWN_MC2_LOCAL_K
        and mlp.down_proj.bias is None
        and mlp.gate_up_proj.bias is None
    )


class NativeQwen2DecoderLayer(nn.Module):
    mc2_input_capture: _MC2RealInputCapture | None = None

    def __init__(
        self,
        config,
        context: NativeTPContext,
        mc2_input_capture: _MC2RealInputCapture | None = None,
        *,
        layer_index: int = 0,
    ) -> None:
        super().__init__()
        self.self_attn = NativeAttention(config, context)
        self.input_layernorm = NativeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = NativeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = NativeQwen2MLP(config, context)
        self.context = context
        self.enable_mc2 = bool(getattr(config, "pearl_enable_mc2", False))
        self.mc2_profile = getattr(config, "pearl_mc2_profile", None)
        self.mc2_input_capture = mc2_input_capture
        self.layer_index = int(layer_index)
        self.mc2_attention_route: MC2StaticRoute | None = None
        self.mc2_down_route: MC2StaticRoute | None = None
        self.mc2_routes_frozen = False
        self.enable_down_mc2 = _enable_tp3_down_mc2(
            self.mlp,
            context,
            enable_mc2=self.enable_mc2,
            profile=self.mc2_profile,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        attention_metadata: NativeAttentionMetadata | None = None,
        *,
        next_layernorm_gamma: torch.Tensor | None = None,
        next_layernorm: NativeRMSNorm | None = None,
        input_is_normalized: bool = False,
        return_normalization_state: bool = False,
        mc2_chain: _MC2IntraLayerChain | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, bool]:
        if input_is_normalized:
            if residual is None:
                raise ValueError("A pre-normalized decoder input requires its residual state.")
        elif residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if self.enable_mc2:
            routes_frozen = bool(getattr(self, "mc2_routes_frozen", False))
            if routes_frozen and (
                getattr(self, "mc2_attention_route", None) is None
                or getattr(self, "mc2_down_route", None) is None
            ):
                raise RuntimeError("Frozen MC2 decoder layer is missing a static route")
            local_attended = self.self_attn(
                positions,
                hidden_states,
                attention_metadata,
                return_pre_projection=True,
            )
            if self.mc2_input_capture is not None:
                self.mc2_input_capture.maybe_capture(
                    local_attended,
                    self.self_attn.o_proj.weight,
                    residual,
                    self.post_attention_layernorm.weight,
                    projection_kind="attention",
                    layer_index=self.layer_index,
                )
            static_route = getattr(self, "mc2_attention_route", None)
            if static_route is None:
                if routes_frozen:
                    raise RuntimeError("Frozen MC2 attention route cannot use dynamic qualification")
                qualification = self.post_attention_layernorm.qualify_forward_mc2(
                    local_attended,
                    residual,
                    self.self_attn.o_proj.weight,
                    self.context,
                    projection_bias=self.self_attn.o_proj.bias,
                    profile=self.mc2_profile,
                )
            else:
                qualification = bind_mc2_static_route(
                    static_route,
                    self.mc2_profile,
                    local_attended,
                    self.self_attn.o_proj.weight,
                    residual,
                    self.post_attention_layernorm.weight,
                )
            if qualification.qualification.qualified:
                hidden_states, residual = self.post_attention_layernorm.forward_mc2(
                    local_attended,
                    residual,
                    self.self_attn.o_proj.weight,
                    self.context,
                    projection_bias=self.self_attn.o_proj.bias,
                    profile=self.mc2_profile,
                    qualification=qualification,
                    static_route=static_route,
                    mc2_chain=mc2_chain,
                )
            else:
                if mc2_chain is not None:
                    raise RuntimeError(
                        "MC2 intra-layer chain lost its qualified attention route after static admission"
                    )
                hidden_states = self.self_attn.o_proj(local_attended)
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states,
                    residual,
                )
        else:
            hidden_states = self.self_attn(positions, hidden_states, attention_metadata)
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        down_mc2_fused = False
        if (
            return_normalization_state
            and self.enable_down_mc2
            and next_layernorm is not None
        ):
            hidden_states, residual, down_mc2_fused = self.mlp.forward_down_mc2(
                hidden_states,
                residual,
                next_layernorm,
                self.context,
                profile=self.mc2_profile,
                static_route=getattr(self, "mc2_down_route", None),
                mc2_input_capture=self.mc2_input_capture,
                layer_index=self.layer_index,
                mc2_chain=mc2_chain,
            )
        else:
            if self.mc2_input_capture is None:
                hidden_states = self.mlp(hidden_states)
            else:
                hidden_states = self.mlp(
                    hidden_states,
                    mc2_input_capture=self.mc2_input_capture,
                    residual=residual,
                    next_layernorm_gamma=next_layernorm_gamma,
                    layer_index=self.layer_index,
                )
        if return_normalization_state:
            return hidden_states, residual, down_mc2_fused
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
        # PEARL target-only diagnostics still launch the TP1 draft rank. MC2
        # qualification is meaningful only for a sharded projection, so do
        # not let the process-wide capture environment opt that TP1 model in.
        mc2_input_capture = (
            _MC2RealInputCapture.from_env(config, context)
            if context.size > 1
            else None
        )
        self.layers = nn.ModuleList(
            NativeQwen2DecoderLayer(
                config,
                context,
                mc2_input_capture,
                layer_index=layer_index,
            )
            for layer_index in range(config.num_hidden_layers)
        )
        self.norm = NativeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = NativeLMHead(
            config.vocab_size,
            config.hidden_size,
            context,
            track_cache_finiteness=self.track_cache_finiteness,
            tp1_greedy_argmax=bool(getattr(config, "pearl_tp1_greedy_argmax", False)),
        )
        self.register_buffer("attention_mask", None, persistent=False)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.embed_tokens.weight
        token_pad_multiple = int(getattr(config, "pearl_token_pad_multiple", 1))
        if token_pad_multiple < 1:
            raise ValueError("PEARL token-row padding multiple must be positive.")
        for module in self.modules():
            if isinstance(module, (NativeColumnLinear, NativeRowLinear, NativeLMHead)):
                module.token_pad_multiple = token_pad_multiple
            if isinstance(module, NativeRowLinear):
                module.large_m_nz_min_rows = int(
                    getattr(config, "pearl_large_m_nz_min_rows", 0)
                )
        self._mc2_static_route_manifest: MC2StaticRouteManifest | None = None
        self._mc2_static_routes_frozen = False
        self._mc2_chained_row_counts: frozenset[int] = frozenset()
        self._mc2_deferred_row_counts: frozenset[int] = frozenset()
        self.register_buffer(
            "_mc2_chain_zero",
            torch.zeros((64, 4), dtype=torch.int64),
            persistent=False,
        )

    def freeze_mc2_static_routes(self) -> MC2StaticRouteManifest:
        """Resolve and freeze every target-layer MC2 route before capture.

        Weight layout conversion is complete when this method runs.  The
        resulting routes therefore bind the final parameter objects/formats
        and one communicator resolved from the target TP process group.  A
        later forward may vary only its profiled token-row count.
        """

        existing = self._mc2_static_route_manifest
        if existing is not None:
            return existing
        profile = getattr(self.config, "pearl_mc2_profile", None)
        if not bool(getattr(self.config, "pearl_enable_mc2", False)) or not isinstance(
            profile, MC2Profile
        ):
            raise RuntimeError("MC2 static routes require an enabled, normalized target profile")
        if not self.layers:
            raise RuntimeError("MC2 static routes require at least one decoder layer")
        device = self.embed_tokens.weight.device
        operator_kind = getattr(profile, "metadata", {}).get("operator")
        capability = (
            detect_mc2_capability(
                device,
                self.context.size,
                operator=operator_kind,
            )
            if operator_kind == MC2_TP3_NATIVE_EPILOGUE_OPERATOR
            else detect_mc2_capability(device, self.context.size)
        )
        if not capability.available:
            raise RuntimeError(f"MC2 static route capability failed: {capability.reason}")
        environment_error = validate_mc2_static_environment(
            profile,
            device,
            tp_size=self.context.size,
            epsilon=float(self.config.rms_norm_eps),
        )
        if environment_error is not None:
            raise RuntimeError(f"MC2 static route environment failed: {environment_error}")
        comm_name = resolve_hccl_comm_name(
            self.context.group,
            device=device,
            rank=self.context.rank,
        )
        if not comm_name:
            raise RuntimeError("MC2 static routes could not resolve the target HCCL communicator")

        pending: list[tuple[NativeQwen2DecoderLayer, MC2StaticRoute, MC2StaticRoute]] = []
        routes: list[MC2StaticRoute] = []
        for layer_index, layer in enumerate(self.layers):
            attention_enabled = layer.self_attn.o_proj.bias is None
            attention_route = build_mc2_static_route(
                profile,
                projection_kind="attention",
                layer_index=layer_index,
                group_tp=comm_name,
                tp_rank_size=self.context.size,
                tp_rank_id=self.context.rank,
                epsilon=layer.post_attention_layernorm.eps,
                weight=layer.self_attn.o_proj.weight,
                gamma=layer.post_attention_layernorm.weight,
                enabled=attention_enabled,
                disabled_reason=(
                    "MC2 production ABI does not support projection bias"
                    if not attention_enabled
                    else ""
                ),
            )
            next_layernorm = (
                self.layers[layer_index + 1].input_layernorm
                if layer_index + 1 < len(self.layers)
                else self.norm
            )
            down_enabled = layer.enable_down_mc2
            down_disabled_reason = (
                "TP3 down-projection MC2 is disabled by model policy"
                if not down_enabled
                else ""
            )
            if (
                down_enabled
                and operator_kind == MC2_TP3_NATIVE_EPILOGUE_OPERATOR
                and layer.mlp.down_proj.large_m_nz_weight is not None
                and 0 < layer.mlp.down_proj.large_m_nz_min_rows <= 160
            ):
                # The v126 route binds the base weight object once before graph
                # capture.  Mode 11 would switch to another NZ tensor for part
                # of the admitted M envelope; disable this route until routes
                # can bind one exact weight per row bucket.
                down_enabled = False
                down_disabled_reason = (
                    "native-MatMul MC2 epilogue cannot bind the mode-11 alternate "
                    "down-projection weight within its M<=160 envelope"
                )
            down_route = build_mc2_static_route(
                profile,
                projection_kind="down",
                layer_index=layer_index,
                group_tp=comm_name,
                tp_rank_size=self.context.size,
                tp_rank_id=self.context.rank,
                epsilon=next_layernorm.eps,
                weight=layer.mlp.down_proj.weight,
                gamma=next_layernorm.weight,
                enabled=down_enabled,
                disabled_reason=down_disabled_reason,
            )
            pending.append((layer, attention_route, down_route))
            routes.extend((attention_route, down_route))
        manifest = build_mc2_static_route_manifest(profile, routes)
        if not any(
            qualification.qualified
            for route in manifest.routes
            if route.enabled
            for qualification in route.decisions.values()
        ):
            raise RuntimeError(
                "Explicit MC2 configuration has zero qualified production row routes"
            )
        chained_row_counts = _qualified_mc2_chained_flush_row_counts(
            profile,
            manifest.routes,
            layer_count=len(self.layers),
            graph_execution=not bool(
                getattr(self.config, "pearl_enforce_eager", False)
            ),
        )
        deferred_row_counts = _qualified_mc2_chain_row_counts(
            profile,
            manifest.routes,
            layer_count=len(self.layers),
            graph_execution=not bool(
                getattr(self.config, "pearl_enforce_eager", False)
            ),
        )
        for layer, attention_route, down_route in pending:
            layer.mc2_attention_route = attention_route
            layer.mc2_down_route = down_route
            layer.self_attn.o_proj.pre_resolved_comm_name = comm_name
            layer.mlp.down_proj.pre_resolved_comm_name = comm_name
            layer.mc2_routes_frozen = True
        self._mc2_static_route_manifest = manifest
        self._mc2_static_routes_frozen = True
        self._mc2_chained_row_counts = chained_row_counts
        self._mc2_deferred_row_counts = deferred_row_counts
        return manifest

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
            mask_rows.append(row if use_fused_tree_attention else row.to(device=device, dtype=torch.bool))
        level_mask = torch.stack(mask_rows)
        if use_fused_tree_attention:
            attention_mask = None
            tree_attention_mask = (
                level_mask.unsqueeze(0)
                .unsqueeze(0)
                .to(
                    device=device,
                    dtype=torch.bool,
                )
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
        input_is_normalized = False
        row_count = prod(hidden_states.shape[:-1])
        chained_row_counts = getattr(self, "_mc2_chained_row_counts", frozenset())
        deferred_row_counts = getattr(self, "_mc2_deferred_row_counts", frozenset())
        chain_zero = getattr(self, "_mc2_chain_zero", None)
        mc2_chain = (
            _MC2IntraLayerChain(
                chain_zero,
                len(self.layers),
                defer_read_done=row_count in deferred_row_counts,
            )
            if row_count in chained_row_counts and isinstance(chain_zero, torch.Tensor)
            else None
        )
        for layer_index, layer in enumerate(self.layers):
            next_layernorm = (
                self.layers[layer_index + 1].input_layernorm
                if layer_index + 1 < len(self.layers)
                else self.norm
            )
            layer_kwargs = {
                "next_layernorm_gamma": next_layernorm.weight,
                "next_layernorm": next_layernorm,
                "input_is_normalized": input_is_normalized,
                "return_normalization_state": True,
            }
            if mc2_chain is not None:
                layer_kwargs["mc2_chain"] = mc2_chain
            hidden_states, residual, input_is_normalized = layer(
                positions,
                hidden_states,
                residual,
                attention_metadata,
                **layer_kwargs,
            )
        if mc2_chain is not None:
            mc2_chain.finish()
        if self.track_cache_finiteness:
            self.output_nonfinite.logical_or_(~(torch.isfinite(hidden_states).all() & torch.isfinite(residual).all()))
        if not input_is_normalized:
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
    _maybe_untie_lm_head_for_nz(model)
    _maybe_convert_linear_weights_to_nz(model)


def _maybe_untie_lm_head_for_nz(model: NativeQwen2ForCausalLM) -> None:
    """Give a tied draft LM head its own inference-only NZ-capable weight.

    Qwen3-0.6B ties the output projection to the token embedding table.  The
    embedding must remain ND for GatherV2, whereas the much hotter full-vocab
    projection benefits from FRACTAL_NZ.  Mode 9 therefore clones the already
    loaded table exactly once before layout conversion.  This preserves model
    values and leaves all non-mode-9 configurations genuinely tied.
    """
    if getattr(model.config, "pearl_weight_nz_mode", None) != 9:
        return
    if model.lm_head.weight is not model.embed_tokens.weight:
        return
    model.lm_head.weight = nn.Parameter(
        model.embed_tokens.weight.detach().clone(),
        requires_grad=False,
    )


def _maybe_convert_linear_weights_to_nz(model: NativeQwen2ForCausalLM) -> None:
    """Apply the model-local BF16/FP16 FRACTAL_NZ weight policy.

    Mode 2 converts every eligible linear. TP3 target modes 3/4/5 convert
    respectively both FFN projections, only ``down_proj``, or only
    ``gate_up_proj``. Modes 6/7 split ``down_proj`` across even/odd decoder
    layers. Mode 8 converts QKV, attention output, down projection and LM
    head while retaining gate-up in ND. Mode 9 converts only the LM head and,
    for tied models, uses a private inference copy so embeddings remain ND.
    Mode 10 is the TP3 small-M candidate: QKV plus ``down_proj`` in NZ while
    leaving the less reliable attention-output and vocabulary projections ND.
    Mode 11 retains ``down_proj`` in ND and stores a second NZ copy for matrix
    row counts at or above ``pearl_large_m_nz_min_rows``.
    """
    configured_mode = getattr(model.config, "pearl_weight_nz_mode", None)
    if not isinstance(configured_mode, int):
        configured_mode = int(ascend_envs.VLLM_ASCEND_ENABLE_NZ)
    if configured_mode not in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11):
        return
    tied_embeddings = model.lm_head.weight is model.embed_tokens.weight
    linear_types = (NativeColumnLinear, NativeRowLinear, NativeLMHead)
    named_modules = (
        model.named_modules()
        if configured_mode in (3, 4, 5, 6, 7, 8, 9, 10, 11)
        else (("", module) for module in model.modules())
    )
    for module_name, module in named_modules:
        if configured_mode in (3, 4, 5, 6, 7, 8, 9, 10, 11):
            suffixes = {
                3: (".mlp.gate_up_proj", ".mlp.down_proj"),
                4: (".mlp.down_proj",),
                5: (".mlp.gate_up_proj",),
                6: (".mlp.down_proj",),
                7: (".mlp.down_proj",),
                8: (
                    ".self_attn.qkv_proj",
                    ".self_attn.o_proj",
                    ".mlp.down_proj",
                    "lm_head",
                ),
                9: ("lm_head",),
                10: (".self_attn.qkv_proj", ".mlp.down_proj"),
                11: (".mlp.down_proj",),
            }[configured_mode]
            if not module_name.endswith(suffixes):
                continue
            if configured_mode in (6, 7):
                components = module_name.split(".")
                try:
                    layer_index = int(components[components.index("layers") + 1])
                except (ValueError, IndexError):
                    continue
                if layer_index % 2 != configured_mode - 6:
                    continue
        if tied_embeddings and isinstance(module, NativeLMHead):
            # GatherV2 requires the tied embedding table to stay in ND format.
            continue
        if (
            isinstance(module, linear_types)
            and getattr(module, "bias", None) is None
            and module.weight.device.type == "npu"
        ):
            converted = torch_npu.npu_format_cast(
                module.weight.data,
                ACL_FORMAT_FRACTAL_NZ,
            )
            if configured_mode == 11:
                module.large_m_nz_weight = converted
            else:
                module.weight.data = converted


def load_native_qwen2_weights(model: NativeQwen2ForCausalLM, model_path: str) -> None:
    """Backward-compatible alias for the original Qwen2-only loader."""
    load_native_model_weights(model, model_path)
