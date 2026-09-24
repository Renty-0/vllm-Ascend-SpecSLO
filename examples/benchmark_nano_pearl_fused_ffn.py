# SPDX-License-Identifier: Apache-2.0
"""Benchmark CANN fused FFN against native PEARL's two matmuls.

Both ND and FRACTAL_NZ BF16 paths are measured.  Production SpecSLO keeps
target weights in ND by default, so comparing only an NZ baseline can hide a
useful (or harmful) layout interaction on the exact target-model shape.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-counts", type=int, nargs="+", default=[320, 464, 512])
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--local-intermediate-size", type=int, default=4608)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--profile-steps", type=int, default=200)
    parser.add_argument("--output-json")
    return parser.parse_args()


def _capture(operation: Callable[[], torch.Tensor]) -> tuple[torch.npu.NPUGraph, torch.Tensor]:
    operation()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = operation()
    torch.npu.synchronize()
    return graph, output


def _measure(graph: torch.npu.NPUGraph, warmup_steps: int, profile_steps: int) -> float:
    for _ in range(warmup_steps):
        graph.replay()
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(profile_steps):
        graph.replay()
    torch.npu.synchronize()
    return (time.perf_counter() - started) * 1_000_000 / profile_steps


def _max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).abs().max().cpu())


def main() -> None:
    args = _parse_args()
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.manual_seed(20260810)

    gate_up_fp16 = (torch.randn(
        2 * args.local_intermediate_size,
        args.hidden_size,
        device="npu",
    ) * 0.02).to(torch.float16)
    down_fp16 = (torch.randn(
        args.hidden_size,
        args.local_intermediate_size,
        device="npu",
    ) * 0.02).to(torch.float16)
    gate_up_bf16_nz = torch_npu.npu_format_cast(gate_up_fp16.to(torch.bfloat16), 29)
    down_bf16_nz = torch_npu.npu_format_cast(down_fp16.to(torch.bfloat16), 29)
    gate_up_bf16 = gate_up_fp16.to(torch.bfloat16)
    down_bf16 = down_fp16.to(torch.bfloat16)
    fused_gate_up_fp16 = gate_up_fp16.t().contiguous()
    fused_down_fp16 = down_fp16.t().contiguous()
    fused_gate_up_bf16 = gate_up_bf16.t().contiguous()
    fused_down_bf16 = down_bf16.t().contiguous()

    results: list[dict[str, float | int | str]] = []
    for token_count in args.token_counts:
        source = (torch.randn(token_count, args.hidden_size, device="npu") * 0.02).to(torch.bfloat16)
        replay_source = (torch.randn_like(source, dtype=torch.float32) * 0.02).to(torch.bfloat16)

        def native_bf16() -> torch.Tensor:
            gate_up = F.linear(source, gate_up_bf16_nz)
            return F.linear(torch_npu.npu_swiglu(gate_up), down_bf16_nz)

        def sequential_bf16() -> torch.Tensor:
            gate_up = F.linear(source, gate_up_bf16)
            return F.linear(torch_npu.npu_swiglu(gate_up), down_bf16)

        def fused_bf16() -> torch.Tensor:
            return torch_npu.npu_ffn(
                source,
                fused_gate_up_bf16,
                fused_down_bf16,
                "swiglu",
                inner_precise=1,
            )

        def sequential_fp16() -> torch.Tensor:
            fp16_source = source.to(torch.float16)
            gate_up = F.linear(fp16_source, gate_up_fp16)
            return F.linear(torch_npu.npu_swiglu(gate_up), down_fp16).to(torch.bfloat16)

        def fused_fp16() -> torch.Tensor:
            return torch_npu.npu_ffn(
                source.to(torch.float16),
                fused_gate_up_fp16,
                fused_down_fp16,
                "swiglu",
                inner_precise=1,
            ).to(torch.bfloat16)

        graphs: dict[str, tuple[torch.npu.NPUGraph, torch.Tensor]] = {}
        for strategy, operation in (
            ("native_bf16_nz", native_bf16),
            ("sequential_bf16_nd", sequential_bf16),
            ("fused_npu_ffn_bf16_nd", fused_bf16),
            ("sequential_fp16_nd", sequential_fp16),
            ("fused_npu_ffn_fp16_nd", fused_fp16),
        ):
            try:
                graph, output = _capture(operation)
                latency_us = _measure(graph, args.warmup_steps, args.profile_steps)
                graphs[strategy] = (graph, output)
                status = "ok"
            except Exception as exception:
                latency_us = float("nan")
                status = f"{type(exception).__name__}: {exception}"
            results.append({
                "token_count": token_count,
                "strategy": strategy,
                "latency_microseconds": latency_us,
                "status": status,
            })

        source.copy_(replay_source)
        for graph, _ in graphs.values():
            graph.replay()
        torch.npu.synchronize()
        if "native_bf16_nz" in graphs:
            native_output = graphs["native_bf16_nz"][1].cpu()
            for result in results[-5:]:
                strategy = str(result["strategy"])
                if strategy in graphs:
                    result["changed_input_replay_max_abs_error_vs_native_bf16"] = _max_abs_error(
                        graphs[strategy][1].cpu(),
                        native_output,
                    )
        if "sequential_fp16_nd" in graphs and "fused_npu_ffn_fp16_nd" in graphs:
            fused_error = _max_abs_error(
                graphs["fused_npu_ffn_fp16_nd"][1].cpu(),
                graphs["sequential_fp16_nd"][1].cpu(),
            )
            next(
                result
                for result in results[-5:]
                if result["strategy"] == "fused_npu_ffn_fp16_nd"
            )["changed_input_replay_max_abs_error_vs_sequential_fp16"] = fused_error
        if "sequential_bf16_nd" in graphs and "fused_npu_ffn_bf16_nd" in graphs:
            fused_error = _max_abs_error(
                graphs["fused_npu_ffn_bf16_nd"][1].cpu(),
                graphs["sequential_bf16_nd"][1].cpu(),
            )
            next(
                result
                for result in results[-5:]
                if result["strategy"] == "fused_npu_ffn_bf16_nd"
            )["changed_input_replay_max_abs_error_vs_sequential_bf16"] = fused_error

    rendered = json.dumps({"results": results}, indent=2)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
