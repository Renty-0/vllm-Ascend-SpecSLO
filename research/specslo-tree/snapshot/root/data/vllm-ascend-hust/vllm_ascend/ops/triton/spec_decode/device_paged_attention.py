# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Device-length PagedAttention prototype for serial draft decoding.

The production CANN PagedAttention graph ABI consumes sequence lengths on the
host.  A changed length therefore requires one graph-task refresh per model
layer and per serial draft step.  This narrow Triton kernel reads both page
tables and exact sequence lengths from device tensors, so an ACLGraph replay
can keep the attention task resident when those values change.

The first version intentionally targets the one-query-per-request GQA decode
shape used by the SpecSLO TP1 draft worker.  It is an opt-in prototype until
NPU numerical and throughput qualification have both passed.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _device_paged_attention_kernel(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
    context_lens_ptr,
    output_ptr,
    query_stride_row: tl.constexpr,
    query_stride_head: tl.constexpr,
    query_stride_dim: tl.constexpr,
    block_table_stride: tl.constexpr,
    output_stride_row: tl.constexpr,
    output_stride_head: tl.constexpr,
    output_stride_dim: tl.constexpr,
    scale,
    HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_GROUP: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TOKENS_PER_ITERATION: tl.constexpr,
    LENGTHS_ARE_POSITIONS: tl.constexpr,
):
    row = tl.program_id(axis=0)
    kv_head = tl.program_id(axis=1)

    dim_offsets = tl.arange(0, HEAD_DIM)
    group_offsets = tl.arange(0, HEAD_GROUP)
    query_heads = kv_head * HEAD_GROUP + group_offsets
    query = tl.load(
        query_ptr
        + row * query_stride_row
        + query_heads[:, None] * query_stride_head
        + dim_offsets[None, :] * query_stride_dim
    )
    context_len = tl.load(context_lens_ptr + row).to(tl.int32)
    if LENGTHS_ARE_POSITIONS:
        context_len += 1

    running_max = tl.full((HEAD_GROUP,), -float("inf"), dtype=tl.float32)
    running_sum = tl.zeros((HEAD_GROUP,), dtype=tl.float32)
    accumulator = tl.zeros((HEAD_GROUP, HEAD_DIM), dtype=tl.float32)

    for token_start in tl.range(0, context_len, TOKENS_PER_ITERATION):
        token_offsets = token_start + tl.arange(0, TOKENS_PER_ITERATION)
        token_mask = token_offsets < context_len
        # Every supported token tile divides the 128-token physical page and
        # every loop begins at a tile boundary.  The complete tile therefore
        # belongs to one page; load its ID once instead of once per token.
        logical_page = token_start // PAGE_SIZE
        page_offsets = token_offsets % PAGE_SIZE
        physical_page = tl.load(block_table_ptr + row * block_table_stride + logical_page).to(tl.int64)
        cache_offsets = (
            (
                (physical_page * PAGE_SIZE + page_offsets[:, None])
                * NUM_KV_HEADS
                + kv_head
            )
            * HEAD_DIM
            + dim_offsets[None, :]
        )
        keys = tl.load(
            key_cache_ptr + cache_offsets,
            mask=token_mask[:, None],
            other=0.0,
        )
        scores = tl.sum(keys[None, :, :] * query[:, None, :], axis=2).to(tl.float32) * scale
        scores = tl.where(token_mask[None, :], scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        next_max = tl.maximum(running_max, block_max)
        old_scale = tl.exp(running_max - next_max)
        probabilities = tl.exp(scores - next_max[:, None])
        probabilities = tl.where(token_mask[None, :], probabilities, 0.0)
        next_sum = running_sum * old_scale + tl.sum(probabilities, axis=1)

        values = tl.load(
            value_cache_ptr + cache_offsets,
            mask=token_mask[:, None],
            other=0.0,
        )
        # Keep the online-softmax recurrence in FP32, but use the native BF16
        # vector datapath for P @ V.  The reduction still accumulates into the
        # FP32 ``accumulator`` below.  On 910B the previous explicit FP32
        # expansion doubled this kernel's dominant tile traffic and made the
        # device-length path slower than the host-refreshed CANN operator.
        value_probabilities = probabilities.to(tl.bfloat16)
        accumulator = accumulator * old_scale[:, None] + tl.sum(
            value_probabilities[:, :, None] * values[None, :, :],
            axis=1,
        )
        running_max = next_max
        running_sum = next_sum

    result = accumulator / running_sum[:, None]
    tl.store(
        output_ptr
        + row * output_stride_row
        + query_heads[:, None] * output_stride_head
        + dim_offsets[None, :] * output_stride_dim,
        result,
    )


def device_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    scale: float,
    output: torch.Tensor | None = None,
    tokens_per_iteration: int = 32,
    lengths_are_positions: bool = False,
) -> torch.Tensor:
    """Run one-query-per-request GQA PagedAttention from device metadata.

    ``lengths_are_positions`` lets a captured serial-draft graph consume its
    already-resident position tensor directly.  This avoids materializing a
    second device length tensor (and, critically, avoids the host length ABI
    that forces CANN PagedAttention task refreshes on every replay).
    """

    if query.ndim != 3:
        raise ValueError("Device PagedAttention query must have shape [rows, heads, head_dim].")
    if key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
        raise ValueError("Device PagedAttention K/V caches must have one aligned four-dimensional shape.")
    if block_table.ndim != 2 or block_table.shape[0] != query.shape[0]:
        raise ValueError("Device PagedAttention block table must contain one row per query.")
    if context_lens.shape != (query.shape[0],):
        raise ValueError("Device PagedAttention context lengths must contain one scalar per query.")
    if query.shape[-1] != key_cache.shape[-1]:
        raise ValueError("Device PagedAttention query/cache head dimensions differ.")
    if query.shape[1] % key_cache.shape[2]:
        raise ValueError("Device PagedAttention requires an integral GQA head group.")
    if query.shape[1] // key_cache.shape[2] not in (1, 2, 4):
        raise ValueError("Device PagedAttention supports GQA head groups of 1, 2, or 4.")
    if any(tensor.device != query.device for tensor in (key_cache, value_cache, block_table, context_lens)):
        raise ValueError("Device PagedAttention inputs must share one device.")
    if block_table.dtype != torch.int32:
        raise ValueError("Device PagedAttention page tables must use int32.")
    valid_length_dtypes = (torch.int32, torch.int64) if lengths_are_positions else (torch.int32,)
    if context_lens.dtype not in valid_length_dtypes:
        kind = "positions" if lengths_are_positions else "context lengths"
        raise ValueError(f"Device PagedAttention {kind} use an unsupported integer dtype.")
    if query.device.type != "npu":
        raise ValueError("The device-length PagedAttention prototype is restricted to Ascend NPU.")
    if tokens_per_iteration not in (16, 32, 64):
        raise ValueError("Device PagedAttention token tile must be 16, 32, or 64.")
    if key_cache.shape[1] % tokens_per_iteration:
        raise ValueError("Device PagedAttention token tile must divide the physical page size.")
    if output is None:
        output = torch.empty_like(query)
    elif output.shape != query.shape or output.dtype != query.dtype or output.device != query.device:
        raise ValueError("Device PagedAttention output must match the query tensor.")

    rows, num_heads, head_dim = query.shape
    if not rows:
        return output
    _device_paged_attention_kernel[(rows, key_cache.shape[2])](
        query,
        key_cache,
        value_cache,
        block_table,
        context_lens,
        output,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        block_table.stride(0),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        float(scale),
        HEAD_DIM=head_dim,
        NUM_HEADS=num_heads,
        NUM_KV_HEADS=key_cache.shape[2],
        HEAD_GROUP=num_heads // key_cache.shape[2],
        PAGE_SIZE=key_cache.shape[1],
        TOKENS_PER_ITERATION=tokens_per_iteration,
        LENGTHS_ARE_POSITIONS=bool(lengths_are_positions),
    )
    return output


__all__ = ["device_paged_attention"]
