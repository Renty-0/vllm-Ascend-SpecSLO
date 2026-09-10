# SPDX-License-Identifier: Apache-2.0
"""Qualify exact TP3 MC2 shapes on real Ascend workers.

Run with three ranks.  The output is accepted directly by ``--mc2-profile``;
no row is inferred from another M/K/N or storage format::

    torchrun --standalone --nproc-per-node 3 examples/measure_specslo_mc2.py \
      --m 64 128 256 --k 1920 4608 --n 5120 --output mc2-profile.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch_npu

from vllm_ascend.spec_decode.pearl.mc2 import mc2_source_sha256, resolve_hccl_comm_name


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, nargs="+", required=True, help="Exact flattened query-token counts.")
    parser.add_argument("--k", type=int, nargs="+", required=True, help="Exact rank-local projection input widths.")
    parser.add_argument("--n", type=int, required=True, help="Replicated residual/output hidden width.")
    parser.add_argument("--weight-formats", choices=("ND", "FRACTAL_NZ"), nargs="+", default=["ND"])
    parser.add_argument("--warmup-replays", type=int, default=20)
    parser.add_argument("--replays-per-sample", type=int, default=20)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--latency-percentile", type=float, default=95.0)
    parser.add_argument("--minimum-speedup", type=float, default=1.02)
    parser.add_argument("--norm-atol", type=float, default=0.05)
    parser.add_argument("--added-atol", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _capture(operation: Callable[[], tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.npu.NPUGraph, Any]:
    operation()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = operation()
    torch.npu.synchronize()
    return graph, output


def _measure_pair(
    fallback_graph: torch.npu.NPUGraph,
    fused_graph: torch.npu.NPUGraph,
    *,
    warmup_replays: int,
    replays_per_sample: int,
    samples: int,
) -> tuple[list[float], list[float]]:
    for _ in range(warmup_replays):
        fallback_graph.replay()
        fused_graph.replay()
    torch.npu.synchronize()
    values = {"fallback": [], "fused": []}
    graphs = {"fallback": fallback_graph, "fused": fused_graph}
    for sample in range(samples):
        # Reverse A/B order every sample so thermal and queue drift cannot
        # systematically favor the fused candidate.
        for name in (("fallback", "fused") if sample % 2 == 0 else ("fused", "fallback")):
            dist.barrier()
            torch.npu.synchronize()
            started = time.perf_counter()
            for _ in range(replays_per_sample):
                graphs[name].replay()
            torch.npu.synchronize()
            values[name].append((time.perf_counter() - started) * 1000.0 / replays_per_sample)
    return values["fallback"], values["fused"]


def _validate_args(args: argparse.Namespace) -> None:
    if any(value <= 0 for value in (*args.m, *args.k, args.n)):
        raise ValueError("M/K/N must be positive")
    if args.samples < 3 or args.replays_per_sample <= 0 or args.warmup_replays < 0:
        raise ValueError("MC2 qualification requires >=3 samples and positive replay counts")
    if not 0 <= args.latency_percentile <= 100 or args.minimum_speedup <= 1:
        raise ValueError("latency percentile must be [0,100] and minimum speedup >1")
    if args.norm_atol < 0 or args.added_atol < 0:
        raise ValueError("MC2 error tolerances must be non-negative")


def _max_rank_samples(local: Sequence[float]) -> list[float]:
    gathered: list[list[float] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, list(local))
    return [max(float(row[index]) for row in gathered if row is not None) for index in range(len(local))]


def _max_rank_value(local: float) -> float:
    # HCCL on the target Ascend stack does not implement float64 all-reduce.
    # These are BF16 absolute-error diagnostics, so float32 is both sufficient
    # and the portable collective dtype.
    value = torch.tensor([local], dtype=torch.float32, device="npu")
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.cpu().item())


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    if dist.get_world_size() != 3:
        raise ValueError("SpecSLO TP3 MC2 qualification requires exactly three ranks")
    rank = dist.get_rank()
    torch.npu.config.allow_internal_format = True

    from vllm_ascend.utils import enable_custom_op

    if not enable_custom_op():
        raise RuntimeError("vLLM-Ascend custom operators could not be loaded")
    operator = torch.ops._C_ascend.matmul_allreduce_add_rmsnorm
    comm_name = resolve_hccl_comm_name(dist.group.WORLD, device="npu", rank=rank)
    if not comm_name:
        raise RuntimeError("Could not resolve the TP3 HCCL communicator")

    measurements = []
    for m in args.m:
        for k in args.k:
            for weight_format in args.weight_formats:
                torch.manual_seed(1701 + rank * 97 + m + k)
                x = (torch.randn((m, k), dtype=torch.float32, device="npu") * 0.02).to(torch.bfloat16)
                weight = (torch.randn((args.n, k), dtype=torch.float32, device="npu") * 0.02).to(torch.bfloat16)
                if weight_format == "FRACTAL_NZ":
                    weight = torch_npu.npu_format_cast(weight, 29)
                residual = (torch.randn((m, args.n), dtype=torch.float32, device="npu") * 0.02).to(torch.bfloat16)
                dist.broadcast(residual, src=0)
                gamma = torch.ones(args.n, dtype=torch.bfloat16, device="npu")

                def fallback(
                    source: torch.Tensor = x,
                    projection_weight: torch.Tensor = weight,
                    residual_input: torch.Tensor = residual,
                    norm_weight: torch.Tensor = gamma,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    projected = torch.nn.functional.linear(source, projection_weight)
                    dist.all_reduce(projected)
                    normalized, _, added = torch_npu.npu_add_rms_norm(
                        projected, residual_input, norm_weight, 1e-6
                    )
                    return normalized, added

                def fused(
                    source: torch.Tensor = x,
                    projection_weight: torch.Tensor = weight,
                    residual_input: torch.Tensor = residual,
                    norm_weight: torch.Tensor = gamma,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    return operator(
                        source,
                        projection_weight,
                        residual_input,
                        norm_weight,
                        comm_name,
                        3,
                        rank,
                        1e-6,
                        True,
                        True,
                    )

                dist.barrier()
                fallback_graph, expected = _capture(fallback)
                dist.barrier()
                fused_graph, actual = _capture(fused)
                norm_error = _max_rank_value(float((actual[0].float() - expected[0].float()).abs().max().cpu()))
                added_error = _max_rank_value(float((actual[1].float() - expected[1].float()).abs().max().cpu()))
                local_fallback, local_fused = _measure_pair(
                    fallback_graph,
                    fused_graph,
                    warmup_replays=args.warmup_replays,
                    replays_per_sample=args.replays_per_sample,
                    samples=args.samples,
                )
                fallback_samples = _max_rank_samples(local_fallback)
                fused_samples = _max_rank_samples(local_fused)
                if rank == 0:
                    measurements.append(
                        {
                            "m": m,
                            "k": k,
                            "n": args.n,
                            "dtype": "bfloat16",
                            "weight_format": str(torch_npu.get_npu_format(weight)),
                            "requested_weight_format": weight_format,
                            "is_trans_b": True,
                            "baseline_latency_ms": fallback_samples,
                            "fused_latency_ms": fused_samples,
                            "max_abs_norm": norm_error,
                            "norm_atol": args.norm_atol,
                            "max_abs_added": added_error,
                            "added_atol": args.added_atol,
                        }
                    )

    if rank == 0:
        finite = all(
            math.isfinite(float(row[key]))
            for row in measurements
            for key in ("max_abs_norm", "max_abs_added")
        )
        document = {
            "schema_version": 1,
            "status": "measured" if finite else "failed_nonfinite",
            "metadata": {
                "operator": "matmul_allreduce_add_rmsnorm",
                "hardware": str(torch.npu.get_device_name(local_rank)),
                "tensor_parallel_size": 3,
                "source_sha256": mc2_source_sha256(),
                "measurement_scope": "captured_operator_replay",
            },
            "latency_percentile": args.latency_percentile,
            "minimum_samples": args.samples,
            "minimum_speedup": args.minimum_speedup,
            "measurements": measurements,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not finite:
            raise RuntimeError(
                "MC2 qualification produced a non-finite numerical error; "
                f"failure evidence was written to {args.output}"
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
