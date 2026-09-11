# SPDX-License-Identifier: Apache-2.0
"""Measure the two Ascend tree-KV compaction implementations in isolation."""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch_npu


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--layers", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--kv-heads", type=int, default=3)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--moves", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    return parser


def _separate(key, value, source, destination) -> None:
    indices = destination.to(torch.int32).view(-1, 1)
    for key_cache, value_cache in zip(key, value):
        key_flat = key_cache.flatten(0, 1)
        value_flat = value_cache.flatten(0, 1)
        torch_npu.npu_scatter_nd_update_(key_flat, indices, torch.index_select(key_flat, 0, source))
        torch_npu.npu_scatter_nd_update_(value_flat, indices, torch.index_select(value_flat, 0, source))


def _fused(key, value, source, destination) -> None:
    slots = destination.to(torch.int32)
    for key_cache, value_cache in zip(key, value):
        key_values = torch.index_select(key_cache.flatten(0, 1), 0, source)
        value_values = torch.index_select(value_cache.flatten(0, 1), 0, source)
        torch_npu._npu_reshape_and_cache(
            key=key_values,
            value=value_values,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_indices=slots,
        )


def _measure(fn, key, value, source, destination, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn(key, value, source, destination)
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        fn(key, value, source, destination)
    torch.npu.synchronize()
    return (time.perf_counter() - started) * 1000.0 / iterations


def main() -> None:
    args = _parser().parse_args()
    torch.npu.set_device(args.device)
    shape = (args.blocks, args.block_size, args.kv_heads, args.head_dim)
    separate_key = [torch.randn(shape, dtype=torch.bfloat16, device="npu") for _ in range(args.layers)]
    separate_value = [torch.randn(shape, dtype=torch.bfloat16, device="npu") for _ in range(args.layers)]
    fused_key = [value.clone() for value in separate_key]
    fused_value = [value.clone() for value in separate_value]
    source = torch.arange(args.moves, dtype=torch.long, device="npu") + args.moves * 2
    destination = torch.arange(args.moves, dtype=torch.long, device="npu")

    # One untimed call checks exact cache semantics before repeated timing.
    _separate(separate_key, separate_value, source, destination)
    _fused(fused_key, fused_value, source, destination)
    torch.npu.synchronize()
    exact = all(torch.equal(lhs, rhs) for lhs, rhs in zip(separate_key, fused_key)) and all(
        torch.equal(lhs, rhs) for lhs, rhs in zip(separate_value, fused_value)
    )
    if not exact:
        raise RuntimeError("fused reshape-and-cache does not match separate tree KV movement")

    separate_ms = _measure(
        _separate, separate_key, separate_value, source, destination, args.warmup, args.iterations
    )
    fused_ms = _measure(_fused, fused_key, fused_value, source, destination, args.warmup, args.iterations)
    # Minimum useful traffic: read and write both K and V for each moved slot.
    bytes_per_round = (
        args.layers * args.moves * args.kv_heads * args.head_dim * 2 * 2 * torch.bfloat16.itemsize
    )
    payload = {
        "shape": vars(args),
        "exact_match": exact,
        "separate_milliseconds": separate_ms,
        "fused_milliseconds": fused_ms,
        "speedup": separate_ms / fused_ms,
        "minimum_payload_bytes": bytes_per_round,
        "separate_effective_gigabytes_per_second": bytes_per_round / separate_ms / 1e6,
        "fused_effective_gigabytes_per_second": bytes_per_round / fused_ms / 1e6,
        "roofline_classification": "launch-bound" if bytes_per_round / fused_ms / 1e6 < 10 else "bandwidth-sensitive",
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
