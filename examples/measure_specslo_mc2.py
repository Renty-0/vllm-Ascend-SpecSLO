# SPDX-License-Identifier: Apache-2.0
"""Qualify exact TP3 MC2 shapes on real Ascend workers.

Run with three ranks.  The output is accepted directly by ``--mc2-profile``;
no row is inferred from another M/K/N or storage format::

    torchrun --standalone --nproc-per-node 3 examples/measure_specslo_mc2.py \
      --m 64 128 256 --k 1920 4608 --n 5120 --output mc2-profile.json
"""

from __future__ import annotations

import argparse
import hashlib
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

from vllm_ascend.spec_decode.pearl.mc2 import (
    current_mc2_runtime_binding,
    mc2_source_sha256,
    resolve_hccl_comm_name,
)


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
    parser.add_argument(
        "--epilogue",
        choices=("native", "custom"),
        default="native",
        help="Run native AddRMSNorm after projection-only MC2, or the custom fused epilogue.",
    )
    parser.add_argument(
        "--diagnose-rounded-epilogue",
        action="store_true",
        help=(
            "Record whether the custom result matches an AddRMSNorm rerun "
            "from the already-rounded BF16 add_out. This is diagnostic only "
            "and does not affect qualification or latency samples."
        ),
    )
    parser.add_argument(
        "--diagnose-projection-reduction",
        action="store_true",
        help=(
            "Compare projection-only MC2 with deterministic HCCL all-reduce "
            "and record mismatch ranges at 64-element granularity. This is "
            "diagnostic only and does not affect qualification or latency."
        ),
    )
    parser.add_argument("--norm-atol", type=float, default=0.05)
    parser.add_argument("--norm-rtol", type=float, default=0.0)
    parser.add_argument("--added-atol", type=float, default=0.01)
    parser.add_argument("--added-rtol", type=float, default=0.0)
    parser.add_argument(
        "--changed-input-replays",
        type=int,
        default=8,
        help=(
            "Replay each captured graph with this many in-place activation "
            "updates and fold the worst error into the qualification row."
        ),
    )
    parser.add_argument(
        "--changed-input-delta",
        type=float,
        default=0.015625,
        help="Per-rank BF16 activation increment used by changed-input qualification.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Load rank-{0,1,2}.pt real-layer activation/weight/residual/gamma payloads.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _capture(operation: Callable[[], tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.npu.NPUGraph, Any]:
    operation()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = operation()
    # Capturing records the launch but does not guarantee that the returned
    # static output buffers contain a completed invocation.  Replay once so
    # numerical checks never inspect uninitialized graph-pool storage.
    graph.replay()
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
        for name in ("fallback", "fused") if sample % 2 == 0 else ("fused", "fallback"):
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
    if args.norm_atol < 0 or args.norm_rtol < 0 or args.added_atol < 0 or args.added_rtol < 0:
        raise ValueError("MC2 error tolerances must be non-negative")
    if (args.norm_atol == 0 and args.norm_rtol == 0) or (args.added_atol == 0 and args.added_rtol == 0):
        raise ValueError("MC2 numerical gates require a non-zero atol or rtol")
    if args.changed_input_replays < 0 or not math.isfinite(args.changed_input_delta):
        raise ValueError("changed-input replay count must be non-negative and delta finite")
    if args.changed_input_replays and (args.changed_input_replays < 2 or args.changed_input_delta == 0):
        raise ValueError("changed-input qualification needs at least two replays and a non-zero delta")
    if args.input_dir is not None and (len(args.m) != 1 or len(args.k) != 1 or args.weight_formats != ["ND"]):
        raise ValueError("--input-dir requires one M/K pair and ND weight format")


def _physical_device_id_for_local_rank(local_rank: int, world_size: int) -> str:
    """Resolve the physical NPU selected by one torchrun local rank."""

    visible_value = os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.environ.get("ASCEND_VISIBLE_DEVICES")
    if not visible_value:
        return str(local_rank)
    visible = tuple(item.strip() for item in visible_value.split(",") if item.strip())
    if len(visible) == 1:
        # Also support launchers that isolate every worker to one physical NPU
        # and expose it as logical npu:0 inside that worker.
        return visible[0]
    if len(visible) < world_size or not 0 <= local_rank < len(visible):
        raise ValueError(
            "ASCEND_RT_VISIBLE_DEVICES does not provide a physical NPU "
            f"for local rank {local_rank} in world size {world_size}"
        )
    return visible[local_rank]


def _tp_rank_device_mapping(local_rank: int) -> tuple[str, ...]:
    world_size = dist.get_world_size()
    local_device = _physical_device_id_for_local_rank(local_rank, world_size)
    gathered: list[str | None] = [None] * world_size
    dist.all_gather_object(gathered, local_device)
    if any(item is None for item in gathered):
        raise RuntimeError("could not collect the TP rank-to-physical-NPU mapping")
    mapping = tuple(str(item) for item in gathered)
    if len(set(mapping)) != world_size:
        raise ValueError("TP ranks must resolve to distinct physical NPUs")
    return mapping


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


def _max_scaled_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> float:
    error = (actual.float() - expected.float()).abs()
    allowance = float(atol) + float(rtol) * expected.float().abs()
    scaled = torch.where(
        allowance > 0,
        error / allowance,
        torch.where(error == 0, torch.zeros_like(error), torch.full_like(error, float("inf"))),
    )
    return float(scaled.max().cpu())


def _tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _contiguous_true_ranges(values: list[bool]) -> list[dict[str, int]]:
    ranges: list[dict[str, int]] = []
    start: int | None = None
    for index, value in enumerate((*values, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            ranges.append({"start": start, "end_exclusive": index})
            start = None
    return ranges


def _projection_reduction_diagnostics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, Any]:
    """Classify projection-only MC2 differences from deterministic HCCL."""

    actual_cpu = actual.detach().cpu().contiguous()
    expected_cpu = expected.detach().cpu().contiguous()
    mismatch = actual_cpu.ne(expected_cpu)
    flat_mismatch = mismatch.reshape(-1)
    flat_error = (actual_cpu.float() - expected_cpu.float()).abs().reshape(-1)
    mismatch_count = int(flat_mismatch.sum().item())
    first_mismatch = None
    worst_mismatch = None
    if mismatch_count:
        width = int(actual_cpu.shape[-1])
        first_flat = int(flat_mismatch.nonzero()[0].item())
        worst_flat = int(flat_error.argmax().item())
        first_mismatch = {
            "flat_index": first_flat,
            "coordinate": list(divmod(first_flat, width)),
            "actual": float(actual_cpu.reshape(-1)[first_flat].float().item()),
            "expected": float(expected_cpu.reshape(-1)[first_flat].float().item()),
        }
        worst_mismatch = {
            "flat_index": worst_flat,
            "coordinate": list(divmod(worst_flat, width)),
            "actual": float(actual_cpu.reshape(-1)[worst_flat].float().item()),
            "expected": float(expected_cpu.reshape(-1)[worst_flat].float().item()),
        }

    block_elements = 64
    flat_count = int(flat_mismatch.numel())
    if flat_count % block_elements:
        raise ValueError("projection reduction diagnostic requires 64-element alignment")
    block_mismatch_counts = (
        flat_mismatch.reshape(-1, block_elements).sum(dim=1).to(torch.int64).tolist()
    )
    block_ranges = _contiguous_true_ranges([count != 0 for count in block_mismatch_counts])
    for item in block_ranges:
        start = item["start"]
        end = item["end_exclusive"]
        item["flat_start"] = start * block_elements
        item["flat_end_exclusive"] = end * block_elements
        item["mismatch_count"] = int(sum(block_mismatch_counts[start:end]))

    # Deterministic TP3 HCCL divides a 64-element-block stream into two
    # rounds of three logical lanes. Keep these formula-derived ranges in the
    # result so a bad MC2 tree can be distinguished from a graph replay bug.
    total_blocks = flat_count // block_elements
    lane_blocks = (
        (total_blocks + 2) // 3,
        total_blocks // 3,
    )
    lane_blocks += (total_blocks - sum(lane_blocks),)
    first_half = tuple((value + 1) // 2 for value in lane_blocks)
    second_half = tuple(value // 2 for value in lane_blocks)
    boundaries = [0]
    for block_count in (*first_half, *second_half):
        boundaries.append(boundaries[-1] + block_count * block_elements)
    chunk_mismatch_counts = [
        int(flat_mismatch[start:end].sum().item())
        for start, end in zip(boundaries, boundaries[1:])
    ]
    return {
        "actual_sha256": _tensor_sha256(actual_cpu),
        "expected_sha256": _tensor_sha256(expected_cpu),
        "exact_match": mismatch_count == 0,
        "mismatch_count": mismatch_count,
        "element_count": flat_count,
        "mismatch_fraction": mismatch_count / flat_count,
        "max_abs_error": float(flat_error.max().item()),
        "first_mismatch": first_mismatch,
        "worst_mismatch": worst_mismatch,
        "hccl_chunk_boundaries": boundaries,
        "hccl_chunk_mismatch_counts": chunk_mismatch_counts,
        "mismatching_64_block_ranges": block_ranges,
    }


def _qualify_changed_inputs(
    fallback_graph: torch.npu.NPUGraph,
    fused_graph: torch.npu.NPUGraph,
    fallback_output: tuple[torch.Tensor, torch.Tensor],
    fused_output: tuple[torch.Tensor, torch.Tensor],
    activation: torch.Tensor,
    *,
    replays: int,
    delta: float,
    rank: int,
    norm_atol: float,
    norm_rtol: float,
    added_atol: float,
    added_rtol: float,
) -> tuple[float, float, float, float]:
    """Compare two captured paths while their static activation changes.

    References are materialized before any fused replay.  This avoids using a
    normal process-group all-reduce between two MC2 launches, which could
    overwrite the custom operator's communicator window and obscure a stale
    graph-input defect.
    """

    if replays == 0:
        return 0.0, 0.0, 0.0, 0.0
    initial = activation.clone()
    references: list[tuple[torch.Tensor, torch.Tensor]] = []
    increment = float(delta) * (rank + 1)
    for replay in range(replays):
        activation.copy_(initial)
        activation.add_(increment * _changed_input_offset_units(replay))
        fallback_graph.replay()
        torch.npu.synchronize()
        references.append((fallback_output[0].clone(), fallback_output[1].clone()))

    activation.copy_(initial)
    torch.npu.synchronize()
    dist.barrier()
    local_norm_error = 0.0
    local_added_error = 0.0
    local_norm_scaled = 0.0
    local_added_scaled = 0.0
    for replay, expected in enumerate(references):
        activation.copy_(initial)
        activation.add_(increment * _changed_input_offset_units(replay))
        fused_graph.replay()
        torch.npu.synchronize()
        local_norm_error = max(
            local_norm_error,
            float((fused_output[0].float() - expected[0].float()).abs().max().cpu()),
        )
        local_added_error = max(
            local_added_error,
            float((fused_output[1].float() - expected[1].float()).abs().max().cpu()),
        )
        local_norm_scaled = max(
            local_norm_scaled,
            _max_scaled_error(
                fused_output[0],
                expected[0],
                atol=norm_atol,
                rtol=norm_rtol,
            ),
        )
        local_added_scaled = max(
            local_added_scaled,
            _max_scaled_error(
                fused_output[1],
                expected[1],
                atol=added_atol,
                rtol=added_rtol,
            ),
        )

    activation.copy_(initial)
    torch.npu.synchronize()
    dist.barrier()
    return (
        _max_rank_value(local_norm_error),
        _max_rank_value(local_added_error),
        _max_rank_value(local_norm_scaled),
        _max_rank_value(local_added_scaled),
    )


def _changed_input_offset_units(replay: int) -> int:
    """Return a bounded, always-changing graph-liveness input offset.

    A cumulative ``activation.add_`` makes a long liveness run exercise an
    activation distribution that the model never sees: with 1024 replays the
    old rank-2 input was shifted by almost 48.  Cycling around the captured
    real activation still proves that replay consumes fresh static-buffer
    contents, while keeping every sample within two configured increments of
    the production payload.
    """

    if replay < 0:
        raise ValueError("replay index must be non-negative")
    return (0, 1, -1, 2, -2)[replay % 5]


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
    # Bind provenance only after bootstrap/import has frozen the actual custom
    # operator route.  Recording the pre-bootstrap search path can otherwise
    # hash a different vendor from the DSO used by EXEC_NPU_CMD.
    rank_device_mapping = _tp_rank_device_mapping(local_rank)
    runtime_binding = current_mc2_runtime_binding(
        tp_rank_device_mapping=rank_device_mapping,
    )
    runtime_bindings_by_rank: list[dict[str, Any] | None] = [
        None
    ] * dist.get_world_size()
    dist.all_gather_object(runtime_bindings_by_rank, runtime_binding)
    if any(binding != runtime_binding for binding in runtime_bindings_by_rank):
        raise RuntimeError(
            "TP3 MC2 qualification resolved different runtime bindings across ranks"
        )
    operator = torch.ops._C_ascend.matmul_allreduce_add_rmsnorm
    comm_name = resolve_hccl_comm_name(dist.group.WORLD, device="npu", rank=rank)
    if not comm_name:
        raise RuntimeError("Could not resolve the TP3 HCCL communicator")

    measurements = []
    # Production keeps every qualified ACLGraph and its static tensors alive
    # in the graph cache.  Retain the same ownership here: destroying an
    # earlier graph while the following shape reuses the communicator can
    # release shared MC2/HCCL resources and turn a lifecycle bug in this
    # benchmark into an apparent numerical kernel failure.
    retained_captures: list[tuple[Any, ...]] = []
    for m in args.m:
        for k in args.k:
            for weight_format in args.weight_formats:
                if args.input_dir is not None:
                    payload = torch.load(
                        args.input_dir / f"rank-{rank}.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    x = payload["activation"].to(device="npu")
                    weight = payload["weight"].to(device="npu")
                    residual = payload["residual"].to(device="npu")
                    gamma = payload["gamma"].to(device="npu")
                    expected_shapes = {
                        "activation": (m, k),
                        "weight": (args.n, k),
                        "residual": (m, args.n),
                        "gamma": (args.n,),
                    }
                    actual_shapes = {
                        "activation": tuple(x.shape),
                        "weight": tuple(weight.shape),
                        "residual": tuple(residual.shape),
                        "gamma": tuple(gamma.shape),
                    }
                    if actual_shapes != expected_shapes:
                        raise ValueError(f"real MC2 payload shapes {actual_shapes} != {expected_shapes}")
                else:
                    torch.manual_seed(1701 + rank * 97 + m + k)
                    x = (torch.randn((m, k), dtype=torch.float32, device="npu") * 0.02).to(torch.bfloat16)
                    weight = (torch.randn((args.n, k), dtype=torch.float32, device="npu") * 0.02).to(torch.bfloat16)
                    if weight_format == "FRACTAL_NZ":
                        weight = torch_npu.npu_format_cast(weight, 29)
                    residual = (torch.randn((m, args.n), dtype=torch.float32, device="npu") * 0.02).to(torch.bfloat16)
                    dist.broadcast(residual, src=0)
                    gamma = torch.ones(args.n, dtype=torch.bfloat16, device="npu")
                if not (
                    x.is_contiguous()
                    and weight.is_contiguous()
                    and residual.is_contiguous()
                    and gamma.is_contiguous()
                ):
                    raise ValueError(
                        "MC2 qualification requires contiguous activation, weight, residual, and gamma inputs"
                    )

                def fallback(
                    source: torch.Tensor = x,
                    projection_weight: torch.Tensor = weight,
                    residual_input: torch.Tensor = residual,
                    norm_weight: torch.Tensor = gamma,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    projected = torch.nn.functional.linear(source, projection_weight)
                    dist.all_reduce(projected)
                    normalized, _, added = torch_npu.npu_add_rms_norm(projected, residual_input, norm_weight, 1e-6)
                    return normalized, added

                def fused(
                    source: torch.Tensor = x,
                    projection_weight: torch.Tensor = weight,
                    residual_input: torch.Tensor = residual,
                    norm_weight: torch.Tensor = gamma,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    fused_output, fused_added = operator(
                        source,
                        projection_weight,
                        residual_input,
                        norm_weight,
                        comm_name,
                        3,
                        rank,
                        1e-6,
                        True,
                        args.epilogue == "custom",
                        args.epilogue == "native",
                    )
                    if args.epilogue == "custom":
                        return fused_output, fused_added
                    normalized, _, added = torch_npu.npu_add_rms_norm(fused_output, residual_input, norm_weight, 1e-6)
                    return normalized, added

                projection_reduction_diagnostics: list[dict[str, Any] | None] = []
                if args.diagnose_projection_reduction:
                    dist.barrier()
                    reference_projection = torch.nn.functional.linear(x, weight)
                    dist.all_reduce(reference_projection)
                    torch.npu.synchronize()
                    reference_projection = reference_projection.clone()

                    fused_projection, _ = operator(
                        x,
                        weight,
                        residual,
                        gamma,
                        comm_name,
                        3,
                        rank,
                        1e-6,
                        True,
                        False,
                        True,
                    )
                    torch.npu.synchronize()
                    local_projection_diagnostics = _projection_reduction_diagnostics(
                        fused_projection,
                        reference_projection,
                    )
                    local_projection_diagnostics["rank"] = rank
                    projection_reduction_diagnostics = [None] * dist.get_world_size()
                    dist.all_gather_object(
                        projection_reduction_diagnostics,
                        local_projection_diagnostics,
                    )
                    dist.barrier()

                dist.barrier()
                fallback_graph, expected = _capture(fallback)
                # NPUGraph outputs live in a graph-managed pool.  Capturing a
                # second MC2 graph on the same communicator may legally reuse
                # internal graph/workspace views, so preserve an ordinary
                # eager tensor copy for numerical qualification.
                expected_reference = (expected[0].clone(), expected[1].clone())
                torch.npu.synchronize()
                dist.barrier()
                fused_graph, actual = _capture(fused)
                retained_captures.append(
                    (
                        fallback_graph,
                        fused_graph,
                        expected,
                        expected_reference,
                        actual,
                        x,
                        weight,
                        residual,
                        gamma,
                    )
                )
                # Materialize both graph outputs before launching another HCCL
                # collective.  The custom MC2 operator and process group share
                # the communicator; interleaving an all-reduce between the two
                # reads can overwrite a graph-era symmetric-window view and
                # report a false add_out error while the operator is correct.
                local_norm_error = float((actual[0].float() - expected_reference[0].float()).abs().max().cpu())
                local_added_error = float((actual[1].float() - expected_reference[1].float()).abs().max().cpu())
                local_norm_scaled = _max_scaled_error(
                    actual[0],
                    expected_reference[0],
                    atol=args.norm_atol,
                    rtol=args.norm_rtol,
                )
                local_added_scaled = _max_scaled_error(
                    actual[1],
                    expected_reference[1],
                    atol=args.added_atol,
                    rtol=args.added_rtol,
                )
                rounded_epilogue_diagnostics: dict[str, float] = {}
                if args.diagnose_rounded_epilogue:
                    # Native AddRMSNorm keeps the FP32 x1+x2 value for its RMS
                    # calculation while materializing add_out as BF16. Feeding
                    # that BF16 add_out back with a zero residual deliberately
                    # discards the hidden FP32 sum. Comparing against this
                    # result identifies whether a fused epilogue accidentally
                    # normalizes the rounded output rather than the native
                    # pre-round value.
                    rounded_norm, _, rounded_added = torch_npu.npu_add_rms_norm(
                        expected_reference[1],
                        torch.zeros_like(expected_reference[1]),
                        gamma,
                        1e-6,
                    )
                    torch.npu.synchronize()
                    rounded_epilogue_diagnostics = {
                        "max_abs_fused_norm_vs_rounded_add_norm": _max_rank_value(
                            float((actual[0].float() - rounded_norm.float()).abs().max().cpu())
                        ),
                        "max_abs_native_norm_vs_rounded_add_norm": _max_rank_value(
                            float(
                                (expected_reference[0].float() - rounded_norm.float())
                                .abs()
                                .max()
                                .cpu()
                            )
                        ),
                        "max_abs_rounded_added_vs_reference_added": _max_rank_value(
                            float(
                                (rounded_added.float() - expected_reference[1].float())
                                .abs()
                                .max()
                                .cpu()
                            )
                        ),
                    }
                norm_error = _max_rank_value(local_norm_error)
                added_error = _max_rank_value(local_added_error)
                norm_scaled = _max_rank_value(local_norm_scaled)
                added_scaled = _max_rank_value(local_added_scaled)
                (
                    changed_norm_error,
                    changed_added_error,
                    changed_norm_scaled,
                    changed_added_scaled,
                ) = _qualify_changed_inputs(
                    fallback_graph,
                    fused_graph,
                    expected,
                    actual,
                    x,
                    replays=args.changed_input_replays,
                    delta=args.changed_input_delta,
                    rank=rank,
                    norm_atol=args.norm_atol,
                    norm_rtol=args.norm_rtol,
                    added_atol=args.added_atol,
                    added_rtol=args.added_rtol,
                )
                norm_error = max(norm_error, changed_norm_error)
                added_error = max(added_error, changed_added_error)
                norm_scaled = max(norm_scaled, changed_norm_scaled)
                added_scaled = max(added_scaled, changed_added_scaled)
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
                            "activation_layout": "contiguous",
                            "weight_layout": "contiguous",
                            "residual_layout": "contiguous",
                            "gamma_layout": "contiguous",
                            "baseline_latency_ms": fallback_samples,
                            "fused_latency_ms": fused_samples,
                            "max_abs_norm": norm_error,
                            "norm_atol": args.norm_atol,
                            "norm_rtol": args.norm_rtol,
                            "max_scaled_norm": norm_scaled,
                            "max_abs_added": added_error,
                            "added_atol": args.added_atol,
                            "added_rtol": args.added_rtol,
                            "max_scaled_added": added_scaled,
                            "changed_input_replays": args.changed_input_replays,
                            "changed_input_delta": args.changed_input_delta,
                            "changed_input_max_abs_norm": changed_norm_error,
                            "changed_input_max_abs_added": changed_added_error,
                            "changed_input_max_scaled_norm": changed_norm_scaled,
                            "changed_input_max_scaled_added": changed_added_scaled,
                            "projection_reduction_diagnostics_by_rank": (
                                projection_reduction_diagnostics
                                if args.diagnose_projection_reduction
                                else []
                            ),
                            **rounded_epilogue_diagnostics,
                        }
                    )

    if rank == 0:
        finite = all(
            math.isfinite(float(row[key]))
            for row in measurements
            for key in ("max_abs_norm", "max_abs_added", "max_scaled_norm", "max_scaled_added")
        )
        document = {
            "schema_version": 1,
            "status": "measured" if finite else "failed_nonfinite",
            "metadata": {
                "operator": (
                    "tp3_matmul_allreduce+native_add_rmsnorm"
                    if args.epilogue == "native"
                    else "tp3_matmul_allreduce_add_rmsnorm"
                ),
                "hardware": str(torch.npu.get_device_name(local_rank)),
                "tensor_parallel_size": 3,
                "source_sha256": mc2_source_sha256(),
                "rms_norm_epsilon": 1e-6,
                "runtime_binding": runtime_binding,
                "runtime_binding_by_rank": runtime_bindings_by_rank,
                "runtime_binding_rank_consensus": True,
                "measurement_scope": "captured_operator_replay",
                "input_source": str(args.input_dir) if args.input_dir is not None else "synthetic",
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
