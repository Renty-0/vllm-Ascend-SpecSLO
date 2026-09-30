# SPDX-License-Identifier: Apache-2.0
"""Screen TP3 target GEMMs for profitable token-row padding on Ascend.

This is deliberately a one-device kernel screen: TP3 Qwen3-32B uses the same
rank-local QKV, gate/up and vocabulary widths on every target rank.  It keeps
the pad operation and output slice inside ACLGraph so the measured candidate
matches ``NativeColumnLinear``/``NativeLMHead`` rather than timing MatMul in
isolation.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECTIONS = {
    "qkv": 2688,
    "gate_up": 17152,
    "lm_head": 50646,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", required=True)
    parser.add_argument("--multiples", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--warmup-replays", type=int, default=50)
    parser.add_argument("--replays-per-sample", type=int, default=100)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def _capture(operation):
    operation()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = operation()
    torch.npu.synchronize()
    return graph, output


def _measure(graph, *, warmup: int, replays: int, samples: int) -> list[float]:
    for _ in range(warmup):
        graph.replay()
    torch.npu.synchronize()
    values = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        for _ in range(replays):
            graph.replay()
        torch.npu.synchronize()
        values.append((time.perf_counter_ns() - started) / 1_000_000 / replays)
    return values


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def main() -> None:
    args = _parser().parse_args()
    if min(args.rows) <= 0 or len(set(args.rows)) != len(args.rows):
        raise ValueError("--rows must contain unique positive values")
    if min(args.multiples) <= 1 or len(set(args.multiples)) != len(args.multiples):
        raise ValueError("--multiples must contain unique integers greater than one")
    if min(args.warmup_replays, args.replays_per_sample, args.samples) <= 0:
        raise ValueError("replay/sample counts must be positive")

    torch.npu.set_device(0)
    torch.manual_seed(20260929)
    hidden_size = 5120
    records = []
    for projection, output_width in PROJECTIONS.items():
        weight = (
            torch.randn(output_width, hidden_size, device="npu", dtype=torch.float32)
            * 0.02
        ).to(torch.bfloat16)
        for rows in args.rows:
            source = (
                torch.randn(rows, hidden_size, device="npu", dtype=torch.float32)
                * 0.02
            ).to(torch.bfloat16)
            changed = (
                torch.randn_like(source, dtype=torch.float32) * 0.02
            ).to(torch.bfloat16)

            def baseline():
                return F.linear(source, weight)

            baseline_graph, baseline_output = _capture(baseline)
            baseline_samples = _measure(
                baseline_graph,
                warmup=args.warmup_replays,
                replays=args.replays_per_sample,
                samples=args.samples,
            )
            source.copy_(changed)
            baseline_graph.replay()
            torch.npu.synchronize()
            baseline_changed = baseline_output.detach().clone()

            candidates = []
            for multiple in args.multiples:
                padded_rows = (rows + multiple - 1) // multiple * multiple
                if padded_rows == rows:
                    continue

                def padded(padded_rows: int = padded_rows):
                    padded_source = F.pad(source, (0, 0, 0, padded_rows - rows))
                    return F.linear(padded_source, weight)[:rows]

                source.copy_(changed)
                padded_graph, padded_output = _capture(padded)
                padded_samples = _measure(
                    padded_graph,
                    warmup=args.warmup_replays,
                    replays=args.replays_per_sample,
                    samples=args.samples,
                )
                padded_graph.replay()
                torch.npu.synchronize()
                max_abs_error = float(
                    (baseline_changed.float() - padded_output.float()).abs().max().cpu()
                )
                baseline_p50 = statistics.median(baseline_samples)
                baseline_p95 = _percentile(baseline_samples, 95)
                padded_p50 = statistics.median(padded_samples)
                padded_p95 = _percentile(padded_samples, 95)
                candidates.append(
                    {
                        "multiple": multiple,
                        "padded_rows": padded_rows,
                        "p50_ms": padded_p50,
                        "p95_ms": padded_p95,
                        "p50_speedup": baseline_p50 / padded_p50,
                        "p95_speedup": baseline_p95 / padded_p95,
                        "max_abs_error": max_abs_error,
                        "samples_ms": padded_samples,
                    }
                )
            records.append(
                {
                    "projection": projection,
                    "rows": rows,
                    "output_width": output_width,
                    "baseline_p50_ms": statistics.median(baseline_samples),
                    "baseline_p95_ms": _percentile(baseline_samples, 95),
                    "baseline_samples_ms": baseline_samples,
                    "candidates": candidates,
                }
            )
            del source, changed, baseline_graph, baseline_output, baseline_changed
        del weight
        torch.npu.empty_cache()

    output = {
        "schema_version": 1,
        "device": "Ascend 910B2",
        "dtype": "bfloat16",
        "weight_format": "ND",
        "hidden_size": hidden_size,
        "warmup_replays": args.warmup_replays,
        "replays_per_sample": args.replays_per_sample,
        "samples": args.samples,
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output_json)


if __name__ == "__main__":
    main()
