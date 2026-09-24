# SPDX-License-Identifier: Apache-2.0
"""Measure eager TP3 MC2 against the unfused projection path.

This diagnostic deliberately avoids ACLGraph capture.  It is used to separate
operator cost from graph admission/replay cost while keeping the same real
activation, weight, residual, communicator, and epilogue as
``measure_specslo_mc2.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch_npu

from vllm_ascend.spec_decode.pearl.mc2 import (
    current_mc2_runtime_binding,
    resolve_hccl_comm_name,
)


Output = tuple[torch.Tensor, torch.Tensor]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--epilogue", choices=("native", "custom"), default="custom")
    parser.add_argument("--warmup-invocations", type=int, default=20)
    parser.add_argument("--invocations-per-sample", type=int, default=20)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if min(args.m, args.k, args.n) <= 0:
        raise ValueError("M/K/N must be positive")
    if args.warmup_invocations < 0 or args.invocations_per_sample <= 0:
        raise ValueError("warmup must be non-negative and sample size positive")
    if args.samples < 3:
        raise ValueError("at least three samples are required")


def _physical_device_id(local_rank: int, world_size: int) -> str:
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.environ.get(
        "ASCEND_VISIBLE_DEVICES"
    )
    if not visible:
        return str(local_rank)
    devices = tuple(value.strip() for value in visible.split(",") if value.strip())
    if len(devices) == 1:
        return devices[0]
    if len(devices) < world_size:
        raise ValueError("visible NPU list is shorter than the distributed world")
    return devices[local_rank]


def _rank_device_mapping(local_rank: int) -> tuple[str, ...]:
    world_size = dist.get_world_size()
    gathered: list[str | None] = [None] * world_size
    dist.all_gather_object(
        gathered,
        _physical_device_id(local_rank, world_size),
    )
    if any(value is None for value in gathered):
        raise RuntimeError("could not collect the TP rank-to-NPU mapping")
    mapping = tuple(str(value) for value in gathered)
    if len(set(mapping)) != world_size:
        raise ValueError("TP ranks must use distinct physical NPUs")
    return mapping


def _max_rank_samples(local: list[float]) -> list[float]:
    gathered: list[list[float] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    return [
        max(float(row[index]) for row in gathered if row is not None)
        for index in range(len(local))
    ]


def _max_rank_value(local: float) -> float:
    value = torch.tensor([local], dtype=torch.float32, device="npu")
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.cpu().item())


def _measure(
    operations: dict[str, Callable[[], Output]],
    *,
    warmup_invocations: int,
    invocations_per_sample: int,
    samples: int,
) -> dict[str, list[float]]:
    # Alternate during warm-up as well so neither path receives a systematic
    # clock/thermal advantage.
    for index in range(warmup_invocations):
        operations["baseline" if index % 2 == 0 else "fused"]()
    torch.npu.synchronize()

    values = {"baseline": [], "fused": []}
    for sample in range(samples):
        order = ("baseline", "fused") if sample % 2 == 0 else ("fused", "baseline")
        for name in order:
            dist.barrier()
            torch.npu.synchronize()
            started = time.perf_counter()
            for _ in range(invocations_per_sample):
                operations[name]()
            torch.npu.synchronize()
            values[name].append(
                (time.perf_counter() - started) * 1000.0 / invocations_per_sample
            )
    return values


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot take percentile of an empty sample")
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    if dist.get_world_size() != 3:
        raise ValueError("this benchmark requires exactly three TP ranks")
    rank = dist.get_rank()
    torch.npu.config.allow_internal_format = True

    from vllm_ascend.utils import enable_custom_op

    if not enable_custom_op():
        raise RuntimeError("vLLM-Ascend custom operators could not be loaded")
    operator = torch.ops._C_ascend.matmul_allreduce_add_rmsnorm
    comm_name = resolve_hccl_comm_name(dist.group.WORLD, device="npu", rank=rank)
    if not comm_name:
        raise RuntimeError("could not resolve the TP3 HCCL communicator")

    payload = torch.load(
        args.input_dir / f"rank-{rank}.pt",
        map_location="cpu",
        weights_only=True,
    )
    x = payload["activation"].to(device="npu")
    weight = payload["weight"].to(device="npu")
    residual = payload["residual"].to(device="npu")
    gamma = payload["gamma"].to(device="npu")
    actual_shapes = {
        "activation": tuple(x.shape),
        "weight": tuple(weight.shape),
        "residual": tuple(residual.shape),
        "gamma": tuple(gamma.shape),
    }
    expected_shapes = {
        "activation": (args.m, args.k),
        "weight": (args.n, args.k),
        "residual": (args.m, args.n),
        "gamma": (args.n,),
    }
    if actual_shapes != expected_shapes:
        raise ValueError(f"real payload shapes {actual_shapes} != {expected_shapes}")

    def baseline() -> Output:
        projected = torch.nn.functional.linear(x, weight)
        dist.all_reduce(projected)
        normalized, _, added = torch_npu.npu_add_rms_norm(
            projected,
            residual,
            gamma,
            1e-6,
        )
        return normalized, added

    def fused() -> Output:
        output, added = operator(
            x,
            weight,
            residual,
            gamma,
            comm_name,
            3,
            rank,
            1e-6,
            True,
            args.epilogue == "custom",
            args.epilogue == "native",
        )
        if args.epilogue == "custom":
            return output, added
        normalized, _, native_added = torch_npu.npu_add_rms_norm(
            output,
            residual,
            gamma,
            1e-6,
        )
        return normalized, native_added

    expected = baseline()
    torch.npu.synchronize()
    expected = (expected[0].clone(), expected[1].clone())
    actual = fused()
    torch.npu.synchronize()
    max_abs_norm = _max_rank_value(
        float((actual[0].float() - expected[0].float()).abs().max().cpu())
    )
    max_abs_added = _max_rank_value(
        float((actual[1].float() - expected[1].float()).abs().max().cpu())
    )

    local_samples = _measure(
        {"baseline": baseline, "fused": fused},
        warmup_invocations=args.warmup_invocations,
        invocations_per_sample=args.invocations_per_sample,
        samples=args.samples,
    )
    baseline_samples = _max_rank_samples(local_samples["baseline"])
    fused_samples = _max_rank_samples(local_samples["fused"])
    mapping = _rank_device_mapping(local_rank)
    binding = current_mc2_runtime_binding(tp_rank_device_mapping=mapping)
    bindings: list[dict[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(bindings, binding)
    if any(value != binding for value in bindings):
        raise RuntimeError("TP ranks resolved different MC2 runtime bindings")

    if rank == 0:
        baseline_p50 = _percentile(baseline_samples, 50.0)
        baseline_p95 = _percentile(baseline_samples, 95.0)
        fused_p50 = _percentile(fused_samples, 50.0)
        fused_p95 = _percentile(fused_samples, 95.0)
        document = {
            "schema_version": 1,
            "status": "measured",
            "metadata": {
                "measurement_scope": "eager_operator_invocation",
                "operator": (
                    "tp3_matmul_allreduce_add_rmsnorm"
                    if args.epilogue == "custom"
                    else "tp3_matmul_allreduce+native_add_rmsnorm"
                ),
                "tensor_parallel_size": 3,
                "input_source": str(args.input_dir),
                "runtime_binding": binding,
                "runtime_binding_by_rank": bindings,
            },
            "shape": {"m": args.m, "k": args.k, "n": args.n},
            "samples": {
                "baseline_latency_ms": baseline_samples,
                "fused_latency_ms": fused_samples,
            },
            "summary": {
                "baseline_p50_ms": baseline_p50,
                "baseline_p95_ms": baseline_p95,
                "fused_p50_ms": fused_p50,
                "fused_p95_ms": fused_p95,
                "p50_speedup": baseline_p50 / fused_p50,
                "p95_speedup": baseline_p95 / fused_p95,
                "max_abs_norm": max_abs_norm,
                "max_abs_added": max_abs_added,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2) + "\n")
        print(json.dumps(document["summary"], indent=2), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
