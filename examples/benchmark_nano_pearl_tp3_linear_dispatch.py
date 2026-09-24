# SPDX-License-Identifier: Apache-2.0
"""Compare graph-safe linear layouts for native PEARL TP3 matrix shapes."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu


TP3_PROJECTIONS = (
    ("qkv_nd_bias", 5120, 2688, False, True),
    ("gate_up_nz", 5120, 9216, True, False),
    ("attention_out_nz", 1920, 5120, True, False),
    ("down_nz", 4608, 5120, True, False),
)

# Qwen3-32B uses 64 Q / 8 KV heads and an intermediate width of 25,600.
# Native dynamic TP pads those to 72 / 9 heads and 25,632 elements for TP3,
# yielding the exact local shapes below.  Measure both ND and FRACTAL_NZ with
# the same F.linear dispatch used by the production target graph.
QWEN3_32B_TP3_PROJECTIONS = (
    ("qkv_nd", 5120, 3840, False, False),
    ("qkv_nz", 5120, 3840, True, False),
    ("gate_up_nd", 5120, 17088, False, False),
    ("gate_up_nz", 5120, 17088, True, False),
    ("attention_out_nd", 3072, 5120, False, False),
    ("attention_out_nz", 3072, 5120, True, False),
    ("down_nd", 8544, 5120, False, False),
    ("down_nz", 8544, 5120, True, False),
    ("lm_head_nd", 5120, 50646, False, False),
    ("lm_head_nz", 5120, 50646, True, False),
)

# Qwen3-0.6B is the TP1 draft paired with the Qwen3-32B TP3 target.  The
# full-chain draft graph executes four consecutive decode forwards, so its
# exposed latency can become the pipeline bottleneck once the target's
# down-projection uses the faster TP3 NZ layout.  Keep both layouts here to
# select a production policy using the exact TP1 matrix shapes rather than a
# target-only proxy.
QWEN3_06B_TP1_PROJECTIONS = (
    ("qkv_nd", 1024, 2048, False, False),
    ("qkv_nz", 1024, 2048, True, False),
    ("gate_up_nd", 1024, 6144, False, False),
    ("gate_up_nz", 1024, 6144, True, False),
    ("attention_out_nd", 1024, 1024, False, False),
    ("attention_out_nz", 1024, 1024, True, False),
    ("down_nd", 3072, 1024, False, False),
    ("down_nz", 3072, 1024, True, False),
    ("lm_head_nd", 1024, 151936, False, False),
    ("lm_head_nz", 1024, 151936, True, False),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--token-counts", type=int, nargs="+", default=[320, 464, 512])
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--profile-steps", type=int, default=200)
    parser.add_argument(
        "--shape-set",
        choices=("qwen25-14b", "qwen3-32b", "qwen3-0.6b"),
        default="qwen25-14b",
    )
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
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    torch.manual_seed(20260810)
    results: list[dict[str, float | int | str | bool]] = []
    projections = {
        "qwen25-14b": TP3_PROJECTIONS,
        "qwen3-32b": QWEN3_32B_TP3_PROJECTIONS,
        "qwen3-0.6b": QWEN3_06B_TP1_PROJECTIONS,
    }[args.shape_set]
    for projection, input_size, output_size, use_nz, use_bias in projections:
        weight = (
            torch.randn(output_size, input_size, dtype=torch.float32, device="npu") * 0.02
        ).to(dtype)
        if use_nz:
            weight = torch_npu.npu_format_cast(weight, 29)
        bias = None
        if use_bias:
            bias = (
                torch.randn(output_size, dtype=torch.float32, device="npu") * 0.02
            ).to(dtype)

        for token_count in args.token_counts:
            source = (
                torch.randn(token_count, input_size, dtype=torch.float32, device="npu")
                * 0.02
            ).to(dtype)
            replay_source = (
                torch.randn_like(source, dtype=torch.float32) * 0.02
            ).to(dtype)

            def functional_linear() -> torch.Tensor:
                return F.linear(source, weight, bias)

            def npu_linear() -> torch.Tensor:
                return torch_npu.npu_linear(source, weight, bias)

            graphs: dict[str, tuple[torch.npu.NPUGraph, torch.Tensor]] = {}
            for strategy, operation in (
                ("functional_linear", functional_linear),
                ("npu_linear", npu_linear),
            ):
                try:
                    graph, output = _capture(operation)
                    latency_us = _measure(graph, args.warmup_steps, args.profile_steps)
                    graphs[strategy] = (graph, output)
                    status = "ok"
                except Exception as error:
                    latency_us = float("nan")
                    status = f"{type(error).__name__}: {error}"
                results.append({
                    "projection": projection,
                    "token_count": token_count,
                    "input_size": input_size,
                    "output_size": output_size,
                    "weight_nz": use_nz,
                    "bias": use_bias,
                    "strategy": strategy,
                    "latency_microseconds": latency_us,
                    "status": status,
                })

            source.copy_(replay_source)
            for graph, _ in graphs.values():
                graph.replay()
            torch.npu.synchronize()
            if "functional_linear" in graphs:
                expected = graphs["functional_linear"][1]
                for result in results[-2:]:
                    strategy = str(result["strategy"])
                    if strategy in graphs:
                        result["changed_input_replay_max_abs_error"] = _max_abs_error(
                            graphs[strategy][1],
                            expected,
                        )

    payload = {"dtype": args.dtype, "shape_set": args.shape_set, "results": results}
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
