# SPDX-License-Identifier: Apache-2.0
"""Compare TP3 collective implementations used by native nano-PEARL."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch_npu


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-counts", type=int, nargs="+", default=[128, 320, 512])
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--projection-input-sizes", type=int, nargs="+", default=[1920, 4608])
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--profile-steps", type=int, default=200)
    parser.add_argument("--collectives-only", action="store_true")
    parser.add_argument(
        "--hccl-expansion-mode",
        choices=("AIV", "runtime-default"),
        default="AIV",
        help="Select AIV explicitly or leave HCCL_OP_EXPANSION_MODE unset.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=(
            "all_reduce",
            "reduce_broadcast",
            "all_gather_sum",
            "quant_all_gather_sum",
            "quant_all_reduce",
        ),
        default=None,
        help=(
            "Only benchmark the selected collective strategies. This is useful "
            "for isolating unsupported experimental collectives from the HCCL "
            "communicator used by the production all-reduce path."
        ),
    )
    parser.add_argument(
        "--custom-opp-path",
        help="Development-only custom-operator vendor path to prioritize.",
    )
    parser.add_argument("--output-json")
    return parser.parse_args()


def _capture(operation: Callable[[], Any]) -> tuple[torch.npu.NPUGraph, Any]:
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
    if dist.is_initialized():
        raise RuntimeError("The TP3 collective benchmark must initialize its own process group.")
    if args.hccl_expansion_mode == "AIV":
        os.environ["HCCL_OP_EXPANSION_MODE"] = "AIV"
    else:
        os.environ.pop("HCCL_OP_EXPANSION_MODE", None)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 3:
        raise ValueError(f"This benchmark requires exactly three ranks, received {world_size}.")
    torch.npu.config.allow_internal_format = True

    from vllm_ascend.utils import enable_custom_op

    if not enable_custom_op():
        raise RuntimeError("vLLM-Ascend custom operators could not be loaded.")
    if args.custom_opp_path:
        custom_opp_path = str(Path(args.custom_opp_path).resolve())
        current_paths = [
            path
            for path in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":")
            if path and path != custom_opp_path
        ]
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join(
            [custom_opp_path, *current_paths]
        )

    backend = dist.group.WORLD._get_backend(torch.device("npu"))
    hccl_group_name = backend.get_hccl_comm_name(rank)

    results: list[dict[str, float | int | str]] = []
    for token_count in args.token_counts:
        def make_operation(name: str) -> Callable[[], torch.Tensor]:
            source = torch.zeros((token_count, args.hidden_size), dtype=torch.bfloat16, device="npu")
            if name == "all_reduce":

                def operation() -> torch.Tensor:
                    dist.all_reduce(source)
                    return source

                return operation
            if name == "reduce_broadcast":

                def operation() -> torch.Tensor:
                    dist.reduce(source, dst=0)
                    dist.broadcast(source, src=0)
                    return source

                return operation
            if name == "all_gather_sum":
                gathered = torch.empty(
                    (world_size * token_count, args.hidden_size),
                    dtype=source.dtype,
                    device=source.device,
                )

                def operation() -> torch.Tensor:
                    dist.all_gather_into_tensor(gathered, source)
                    return gathered.view(world_size, token_count, args.hidden_size).sum(dim=0)

                return operation
            if name == "quant_all_gather_sum":
                gathered = torch.empty(
                    (world_size * token_count, args.hidden_size),
                    dtype=torch.int8,
                    device=source.device,
                )
                gathered_scales = torch.empty(
                    world_size * token_count,
                    dtype=torch.float32,
                    device=source.device,
                )

                def operation() -> torch.Tensor:
                    quantized, scales = torch_npu.npu_dynamic_quant(source, dst_type=torch.int8)
                    dist.all_gather_into_tensor(gathered, quantized)
                    dist.all_gather_into_tensor(gathered_scales, scales)
                    dequantized = gathered.to(source.dtype) * gathered_scales.unsqueeze(-1).to(source.dtype)
                    return dequantized.view(world_size, token_count, args.hidden_size).sum(dim=0)

                return operation
            if name == "quant_all_reduce":

                def operation() -> torch.Tensor:
                    grouped = source.view(token_count, args.hidden_size // 128, 128)
                    quantized, scales = torch_npu.npu_dynamic_quant(
                        grouped,
                        dst_type=torch.int8,
                    )
                    return torch_npu.npu_quant_all_reduce(
                        quantized.view_as(source),
                        scales,
                        hccl_group_name,
                        # Probe whether this argument is only a wrapper-level
                        # shape hint; the communicator itself still has TP3.
                        4,
                    )

                return operation
            raise ValueError(f"Unknown TP3 collective strategy: {name}")

        strategy_names = args.strategies or (
            "all_reduce",
            "reduce_broadcast",
            "all_gather_sum",
            "quant_all_gather_sum",
            "quant_all_reduce",
        )
        for name in strategy_names:
            dist.barrier()
            try:
                operation = make_operation(name)
                graph, output = _capture(operation)
                latency_us = _measure(graph, args.warmup_steps, args.profile_steps)
                output.cpu()
                status = "ok"
            except Exception as error:
                latency_us = float("nan")
                status = f"{type(error).__name__}: {error}"
            result = {
                "rank": rank,
                "token_count": token_count,
                "hidden_size": args.hidden_size,
                "strategy": name,
                "latency_microseconds": latency_us,
                "status": status,
            }
            gathered_results: list[dict[str, float | int | str] | None] = [None] * world_size
            dist.all_gather_object(gathered_results, result)
            if rank == 0:
                results.extend(value for value in gathered_results if value is not None)

        if args.collectives_only:
            continue

        for projection_input_size in args.projection_input_sizes:
            torch.manual_seed(41 + rank)
            source = (torch.randn(
                (token_count, projection_input_size),
                dtype=torch.float32,
                device="npu",
            ) * 0.02).to(torch.bfloat16)
            weight_nd = (torch.randn(
                (args.hidden_size, projection_input_size),
                dtype=torch.float32,
                device="npu",
            ) * 0.02).to(torch.bfloat16)
            weight_nz = torch_npu.npu_format_cast(weight_nd, 29)
            residual = (torch.randn(
                (token_count, args.hidden_size),
                dtype=torch.float32,
                device="npu",
            ) * 0.02).to(torch.bfloat16)
            # Tensor-parallel decoder residuals are replicated across ranks.
            dist.broadcast(residual, src=0)
            gamma = torch.ones(args.hidden_size, dtype=torch.bfloat16, device="npu")
            epsilon = 1e-6

            def baseline_projection() -> tuple[torch.Tensor, torch.Tensor]:
                projected = torch.nn.functional.linear(source, weight_nz)
                dist.all_reduce(projected)
                normalized, _, add_out = torch_npu.npu_add_rms_norm(
                    projected,
                    residual,
                    gamma,
                    epsilon,
                )
                return normalized, add_out

            def quantized_projection() -> tuple[torch.Tensor, torch.Tensor]:
                projected = torch.nn.functional.linear(source, weight_nz)
                grouped = projected.view(token_count, args.hidden_size // 128, 128)
                quantized, scales = torch_npu.npu_dynamic_quant(
                    grouped,
                    dst_type=torch.int8,
                )
                projected = torch_npu.npu_quant_all_reduce(
                    quantized.view_as(projected),
                    scales,
                    hccl_group_name,
                    # The real HCCL communicator remains TP3.
                    4,
                )
                normalized, _, add_out = torch_npu.npu_add_rms_norm(
                    projected,
                    residual,
                    gamma,
                    epsilon,
                )
                return normalized, add_out

            def make_fused_projection(
                projection_weight: torch.Tensor,
                *,
                gather_add_out: bool,
            ) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
                def fused_projection() -> tuple[torch.Tensor, torch.Tensor]:
                    return torch.ops._C_ascend.matmul_allreduce_add_rmsnorm(
                        source,
                        projection_weight,
                        residual,
                        gamma,
                        hccl_group_name,
                        world_size,
                        rank,
                        epsilon,
                        True,
                        gather_add_out,
                    )

                return fused_projection

            def make_builtin_mc2_projection(
                projection_weight: torch.Tensor,
            ) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
                def builtin_mc2_projection() -> tuple[torch.Tensor, torch.Tensor]:
                    projected = torch_npu.npu_mm_all_reduce_base(
                        source,
                        projection_weight.t(),
                        hccl_group_name,
                    )
                    normalized, _, add_out = torch_npu.npu_add_rms_norm(
                        projected,
                        residual,
                        gamma,
                        epsilon,
                    )
                    return normalized, add_out

                return builtin_mc2_projection

            projection_results: list[dict[str, float | int | str]] = []
            dist.barrier()
            try:
                baseline_graph, baseline_output = _capture(baseline_projection)
                baseline_latency_us = _measure(
                    baseline_graph,
                    args.warmup_steps,
                    args.profile_steps,
                )
                torch.npu.synchronize()
                expected_normalized = baseline_output[0].cpu()
                expected_add_out = baseline_output[1].cpu()
                baseline_status = "ok"
            except Exception as error:
                baseline_latency_us = float("nan")
                expected_normalized = None
                expected_add_out = None
                baseline_status = f"{type(error).__name__}: {error}"
            projection_results.append({
                "rank": rank,
                "token_count": token_count,
                "projection_input_size": projection_input_size,
                "hidden_size": args.hidden_size,
                "strategy": "nz_matmul_allreduce_add_rmsnorm",
                "latency_microseconds": baseline_latency_us,
                "normalized_max_abs_error": 0.0,
                "add_out_max_abs_error": 0.0,
                "status": baseline_status,
            })

            dist.barrier()
            try:
                if token_count % world_size:
                    raise ValueError("Sequence parallelism requires a token count divisible by TP3.")
                local_token_count = token_count // world_size
                local_residual = residual.narrow(
                    0,
                    rank * local_token_count,
                    local_token_count,
                ).contiguous()

                def sequence_parallel_projection() -> tuple[torch.Tensor, torch.Tensor]:
                    projected = torch.nn.functional.linear(source, weight_nz)
                    local_projected = torch.empty(
                        (local_token_count, args.hidden_size),
                        dtype=projected.dtype,
                        device=projected.device,
                    )
                    dist.reduce_scatter_tensor(local_projected, projected)
                    local_normalized, _, local_add_out = torch_npu.npu_add_rms_norm(
                        local_projected,
                        local_residual,
                        gamma,
                        epsilon,
                    )
                    normalized = torch.empty_like(projected)
                    dist.all_gather_into_tensor(normalized, local_normalized)
                    return normalized, local_add_out

                sequence_graph, sequence_output = _capture(sequence_parallel_projection)
                sequence_latency_us = _measure(
                    sequence_graph,
                    args.warmup_steps,
                    args.profile_steps,
                )
                torch.npu.synchronize()
                if expected_normalized is None or expected_add_out is None:
                    raise RuntimeError("The baseline projection did not produce reference outputs.")
                sequence_normalized_error = _max_abs_error(
                    sequence_output[0].cpu(),
                    expected_normalized,
                )
                sequence_add_out_error = _max_abs_error(
                    sequence_output[1].cpu(),
                    expected_add_out.narrow(
                        0,
                        rank * local_token_count,
                        local_token_count,
                    ),
                )
                sequence_status = "ok"
            except Exception as error:
                sequence_latency_us = float("nan")
                sequence_normalized_error = float("nan")
                sequence_add_out_error = float("nan")
                sequence_status = f"{type(error).__name__}: {error}"
            projection_results.append({
                "rank": rank,
                "token_count": token_count,
                "projection_input_size": projection_input_size,
                "hidden_size": args.hidden_size,
                "strategy": "nz_matmul_reduce_scatter_local_rmsnorm_allgather",
                "latency_microseconds": sequence_latency_us,
                "normalized_max_abs_error": sequence_normalized_error,
                "add_out_max_abs_error": sequence_add_out_error,
                "status": sequence_status,
            })

            dist.barrier()
            try:
                quantized_graph, quantized_output = _capture(quantized_projection)
                quantized_latency_us = _measure(
                    quantized_graph,
                    args.warmup_steps,
                    args.profile_steps,
                )
                torch.npu.synchronize()
                if expected_normalized is None or expected_add_out is None:
                    raise RuntimeError("The baseline projection did not produce reference outputs.")
                quantized_normalized_error = _max_abs_error(
                    quantized_output[0].cpu(),
                    expected_normalized,
                )
                quantized_add_out_error = _max_abs_error(
                    quantized_output[1].cpu(),
                    expected_add_out,
                )
                quantized_status = "ok"
            except Exception as error:
                quantized_latency_us = float("nan")
                quantized_normalized_error = float("nan")
                quantized_add_out_error = float("nan")
                quantized_status = f"{type(error).__name__}: {error}"
            projection_results.append({
                "rank": rank,
                "token_count": token_count,
                "projection_input_size": projection_input_size,
                "hidden_size": args.hidden_size,
                "strategy": "nz_matmul_quant_allreduce_add_rmsnorm",
                "latency_microseconds": quantized_latency_us,
                "normalized_max_abs_error": quantized_normalized_error,
                "add_out_max_abs_error": quantized_add_out_error,
                "status": quantized_status,
            })

            for strategy, projection_weight in (
                ("builtin_mm_allreduce_add_rmsnorm_nd", weight_nd),
                ("builtin_mm_allreduce_add_rmsnorm_nz", weight_nz),
            ):
                dist.barrier()
                try:
                    builtin_graph, builtin_output = _capture(
                        make_builtin_mc2_projection(projection_weight)
                    )
                    builtin_latency_us = _measure(
                        builtin_graph,
                        args.warmup_steps,
                        args.profile_steps,
                    )
                    torch.npu.synchronize()
                    if expected_normalized is None or expected_add_out is None:
                        raise RuntimeError("The baseline projection did not produce reference outputs.")
                    builtin_normalized_error = _max_abs_error(
                        builtin_output[0].cpu(),
                        expected_normalized,
                    )
                    builtin_add_out_error = _max_abs_error(
                        builtin_output[1].cpu(),
                        expected_add_out,
                    )
                    builtin_status = "ok"
                except Exception as error:
                    builtin_latency_us = float("nan")
                    builtin_normalized_error = float("nan")
                    builtin_add_out_error = float("nan")
                    builtin_status = f"{type(error).__name__}: {error}"
                projection_results.append({
                    "rank": rank,
                    "token_count": token_count,
                    "projection_input_size": projection_input_size,
                    "hidden_size": args.hidden_size,
                    "strategy": strategy,
                    "latency_microseconds": builtin_latency_us,
                    "normalized_max_abs_error": builtin_normalized_error,
                    "add_out_max_abs_error": builtin_add_out_error,
                    "status": builtin_status,
                })

            for strategy, projection_weight, gather_add_out in (
                ("fused_matmul_allreduce_add_rmsnorm_nd", weight_nd, True),
                ("fused_matmul_allreduce_add_rmsnorm_nz", weight_nz, True),
                ("fused_matmul_allreduce_add_rmsnorm_nd_output_only", weight_nd, False),
                ("fused_matmul_allreduce_add_rmsnorm_nz_output_only", weight_nz, False),
            ):
                dist.barrier()
                try:
                    fused_graph, fused_output = _capture(
                        make_fused_projection(
                            projection_weight,
                            gather_add_out=gather_add_out,
                        )
                    )
                    fused_latency_us = _measure(
                        fused_graph,
                        args.warmup_steps,
                        args.profile_steps,
                    )
                    torch.npu.synchronize()
                    actual_normalized = fused_output[0].cpu()
                    actual_add_out = fused_output[1].cpu()
                    if expected_normalized is None or expected_add_out is None:
                        raise RuntimeError("The baseline projection did not produce reference outputs.")
                    normalized_error = _max_abs_error(actual_normalized, expected_normalized)
                    add_out_error = (
                        _max_abs_error(actual_add_out, expected_add_out)
                        if gather_add_out
                        else float("nan")
                    )
                    fused_status = "ok"
                except Exception as error:
                    fused_latency_us = float("nan")
                    normalized_error = float("nan")
                    add_out_error = float("nan")
                    fused_status = f"{type(error).__name__}: {error}"
                projection_results.append({
                    "rank": rank,
                    "token_count": token_count,
                    "projection_input_size": projection_input_size,
                    "hidden_size": args.hidden_size,
                    "strategy": strategy,
                    "latency_microseconds": fused_latency_us,
                    "normalized_max_abs_error": normalized_error,
                    "add_out_max_abs_error": add_out_error,
                    "status": fused_status,
                })

            gathered_projection_results: list[dict[str, float | int | str] | None] = [None] * world_size
            dist.all_gather_object(gathered_projection_results, projection_results)
            if rank == 0:
                for rank_results in gathered_projection_results:
                    if rank_results is not None:
                        results.extend(rank_results)

    if rank == 0:
        rendered = json.dumps({"world_size": world_size, "results": results}, indent=2)
        print(rendered)
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered + "\n", encoding="utf-8")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
