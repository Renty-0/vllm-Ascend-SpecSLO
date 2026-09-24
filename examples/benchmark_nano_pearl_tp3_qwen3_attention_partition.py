# SPDX-License-Identifier: Apache-2.0
"""Measure uniform-padded versus uneven TP3 Qwen3 attention projections.

Qwen3-32B has 64 query heads and 8 KV heads.  The current native PEARL
implementation pads these to 72/9 so every TP3 rank owns 24/3 heads.  An
uneven partition can instead use Q=[22, 21, 21] and KV=[3, 3, 2], avoiding
synthetic heads while preserving the same collective sequence.  This probe
measures the QKV projection plus O projection/all-reduce envelope; it does not
claim numerical qualification for the production attention kernel.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu


HIDDEN_SIZE = 5120
HEAD_DIM = 128
UNIFORM_Q_HEADS = (24, 24, 24)
UNIFORM_KV_HEADS = (3, 3, 3)
UNEVEN_Q_HEADS = (22, 21, 21)
UNEVEN_KV_HEADS = (3, 3, 2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-counts", type=int, nargs="+", default=[40, 80, 120, 160])
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--profile-steps", type=int, default=100)
    parser.add_argument("--output-json")
    return parser.parse_args()


def _capture(operation):
    operation()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = operation()
    torch.npu.synchronize()
    return graph, outputs


def _measure(graph: torch.npu.NPUGraph, warmup_steps: int, profile_steps: int) -> float:
    for _ in range(warmup_steps):
        graph.replay()
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(profile_steps):
        graph.replay()
    torch.npu.synchronize()
    return (time.perf_counter() - started) * 1_000_000 / profile_steps


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "AIV")
    os.environ.setdefault("HCCL_DETERMINISTIC", "true")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    if dist.get_world_size() != 3:
        raise ValueError("This benchmark requires exactly three ranks")
    torch.npu.config.allow_internal_format = True

    results: list[dict[str, object]] = []
    for token_count in args.token_counts:
        for layout, q_heads, kv_heads in (
            ("uniform_padded", UNIFORM_Q_HEADS, UNIFORM_KV_HEADS),
            ("uneven_exact", UNEVEN_Q_HEADS, UNEVEN_KV_HEADS),
        ):
            local_q_heads = q_heads[rank]
            local_kv_heads = kv_heads[rank]
            qkv_width = (local_q_heads + 2 * local_kv_heads) * HEAD_DIM
            q_width = local_q_heads * HEAD_DIM
            torch.manual_seed(20260920 + rank)
            hidden = torch.randn(
                token_count, HIDDEN_SIZE, dtype=torch.bfloat16, device="npu"
            )
            attended = torch.randn(
                token_count, q_width, dtype=torch.bfloat16, device="npu"
            )
            qkv_weight = torch.randn(
                qkv_width, HIDDEN_SIZE, dtype=torch.bfloat16, device="npu"
            )
            o_weight = torch.randn(
                HIDDEN_SIZE, q_width, dtype=torch.bfloat16, device="npu"
            )

            def operation():
                qkv = F.linear(hidden, qkv_weight)
                output = F.linear(attended, o_weight)
                dist.all_reduce(output)
                return qkv, output

            dist.barrier()
            graph, outputs = _capture(operation)
            latency_us = _measure(graph, args.warmup_steps, args.profile_steps)
            # Materialize both outputs so dead-output elimination cannot make
            # the QKV projection disappear from a future compiler version.
            tuple(value[0, 0].cpu() for value in outputs)
            row = {
                "rank": rank,
                "token_count": token_count,
                "layout": layout,
                "q_heads": local_q_heads,
                "kv_heads": local_kv_heads,
                "qkv_width": qkv_width,
                "o_input_width": q_width,
                "latency_microseconds": latency_us,
            }
            gathered: list[dict[str, object] | None] = [None, None, None]
            dist.all_gather_object(gathered, row)
            if rank == 0:
                results.extend(value for value in gathered if value is not None)

    if rank == 0:
        payload = {
            "world_size": 3,
            "model_shape": "Qwen3-32B attention projection envelope",
            "results": results,
        }
        rendered = json.dumps(payload, indent=2)
        print(rendered)
        if args.output_json:
            output = Path(args.output_json)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered + "\n", encoding="utf-8")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
