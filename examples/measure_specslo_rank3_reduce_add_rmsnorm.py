# SPDX-License-Identifier: Apache-2.0
"""Qualify native MatMul + TP3 AllReduce/AddRMSNorm on real layer inputs.

The candidate deliberately keeps the model's native ``F.linear`` path and
replaces only ``HCCL all_reduce + AddRMSNorm``.  This lets the experiment
separate local-MatMul tiling/numerics from the rank-3 communication epilogue.
Both paths are captured independently and measured in alternating order.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch_npu

from vllm_ascend.spec_decode.pearl.mc2 import (
    MC2_TP3_NATIVE_EPILOGUE_OPERATOR,
    current_mc2_runtime_binding,
    mc2_source_sha256,
    normalize_mc2_profile,
    resolve_hccl_comm_name,
)

Output = tuple[torch.Tensor, torch.Tensor]


@dataclass
class SharedProjectionParameters:
    """Stable parameter buffers shared by every resident graph of one kind."""

    weight: torch.Tensor
    gamma: torch.Tensor


@dataclass
class ResidentGraphResources:
    """All device objects whose addresses are captured by one graph pair."""

    x: torch.Tensor
    residual: torch.Tensor
    chain_zero: torch.Tensor
    parameters: SharedProjectionParameters
    baseline_graph: torch.npu.NPUGraph
    baseline_output: Output
    fused_graph: torch.npu.NPUGraph
    fused_output: Output

    def baseline_runner(self) -> Output:
        self.baseline_graph.replay()
        return self.baseline_output

    def fused_runner(self) -> Output:
        self.fused_graph.replay()
        return self.fused_output


@dataclass
class MeasurementResourcePool:
    """Campaign-lifetime graph cache; never reset between qualification jobs."""

    parameters: dict[tuple[Any, ...], SharedProjectionParameters] = field(
        default_factory=dict
    )
    graphs: dict[tuple[Any, ...], ResidentGraphResources] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class MeasurementRuntime:
    """Process-group resources shared by one or more measurement jobs."""

    local_rank: int
    rank: int
    operator_name: str
    operator: Callable[..., Any]
    comm_name: str
    device_mapping: tuple[str, ...]
    runtime_binding: dict[str, Any]
    chained_flush: bool
    resources: MeasurementResourcePool


@dataclass(frozen=True)
class RankZeroFinalizationOutcome:
    """Collective distinction between a rejected row and a fatal error."""

    admission_failure: str | None = None
    fatal_failure: str | None = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--warmup-replays", type=int, default=40)
    parser.add_argument("--replays-per-sample", type=int, default=50)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--changed-input-replays", type=int, default=8)
    parser.add_argument("--changed-input-delta", type=float, default=0.015625)
    parser.add_argument(
        "--execution-mode",
        choices=("eager", "graph"),
        default="graph",
    )
    parser.add_argument(
        "--operations-per-graph",
        type=int,
        default=1,
        help="sequential native-MatMul/epilogue pairs captured in each graph",
    )
    parser.add_argument(
        "--chained-flush",
        action="store_true",
        help=(
            "Use the three-output chained ABI with one preallocated zero state "
            "and flush every invocation."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--production-profile-output",
        type=Path,
        help="Also write a strict single-shape profile consumable by Native PEARL.",
    )
    parser.add_argument("--latency-percentile", type=float, default=95.0)
    parser.add_argument("--minimum-speedup", type=float, default=1.02)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if min(args.m, args.k, args.n) <= 0:
        raise ValueError("M/K/N must be positive")
    if args.samples < 3 or args.replays_per_sample <= 0 or args.warmup_replays < 0:
        raise ValueError("need >=3 samples, positive replay count, and non-negative warmup")
    if args.changed_input_replays < 2:
        raise ValueError("changed-input qualification needs at least two replays")
    if args.operations_per_graph <= 0:
        raise ValueError("operations-per-graph must be positive")
    if args.production_profile_output is not None and (
        args.execution_mode != "graph" or args.operations_per_graph != 1
    ):
        raise ValueError("a production profile requires graph mode and exactly one operation per graph")
    if not 0.0 <= args.latency_percentile <= 100.0 or args.minimum_speedup <= 1.0:
        raise ValueError("production profile gates require percentile [0,100] and speedup >1")
    if not math.isfinite(args.changed_input_delta) or args.changed_input_delta == 0:
        raise ValueError("changed-input delta must be finite and non-zero")


def _resolve_operator(*, chained: bool = False) -> tuple[str, Callable[..., Any]]:
    operator_name = (
        "allreduce_add_rmsnorm_chained" if chained else "allreduce_add_rmsnorm"
    )
    for namespace in ("vllm_ascend", "_C_ascend"):
        try:
            operator = getattr(getattr(torch.ops, namespace), operator_name)
        except AttributeError:
            continue
        return f"torch.ops.{namespace}.{operator_name}", operator
    raise RuntimeError(f"{operator_name} custom operator is not registered")


def _capture(operation: Callable[[], Output]) -> tuple[torch.npu.NPUGraph, Output]:
    operation()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = operation()
    graph.replay()
    torch.npu.synchronize()
    return graph, output


def _validate_payload_shapes(
    args: argparse.Namespace,
    payload: dict[str, torch.Tensor],
) -> None:
    actual_shapes = {
        "activation": tuple(payload["activation"].shape),
        "weight": tuple(payload["weight"].shape),
        "residual": tuple(payload["residual"].shape),
        "gamma": tuple(payload["gamma"].shape),
    }
    expected_shapes = {
        "activation": (args.m, args.k),
        "weight": (args.n, args.k),
        "residual": (args.m, args.n),
        "gamma": (args.n,),
    }
    if actual_shapes != expected_shapes:
        raise ValueError(f"real payload shapes {actual_shapes} != {expected_shapes}")


def _parameter_resource_key(
    args: argparse.Namespace,
    payload: dict[str, torch.Tensor],
) -> tuple[Any, ...]:
    """Key stable weight/gamma buffers independently of M within one layer."""

    return (
        str(args.input_dir.parent.resolve()),
        args.k,
        args.n,
        tuple(payload["weight"].shape),
        str(payload["weight"].dtype),
        tuple(payload["gamma"].shape),
        str(payload["gamma"].dtype),
    )


def _graph_resource_key(
    args: argparse.Namespace,
    payload: dict[str, torch.Tensor],
) -> tuple[Any, ...]:
    """Key one layer-bound graph pair; only the same layer/shape reuses it."""

    return (
        str(args.input_dir.parent.resolve()),
        args.m,
        args.k,
        args.n,
        str(payload["activation"].dtype),
        str(payload["residual"].dtype),
        args.operations_per_graph,
        args.chained_flush,
    )


def _copy_stable_buffer(
    destination: torch.Tensor,
    source: torch.Tensor,
    *,
    name: str,
) -> None:
    if destination.shape != source.shape or destination.dtype != source.dtype:
        raise ValueError(
            f"resident {name} buffer {tuple(destination.shape)}/{destination.dtype} "
            f"cannot load {tuple(source.shape)}/{source.dtype}"
        )
    destination.copy_(source)


def _load_graph_resources(
    args: argparse.Namespace,
    runtime: MeasurementRuntime,
    payload: dict[str, torch.Tensor],
) -> ResidentGraphResources:
    """Load one job into stable buffers, capturing its shape only once."""

    parameter_key = _parameter_resource_key(args, payload)
    parameters = runtime.resources.parameters.get(parameter_key)
    if parameters is None:
        parameters = SharedProjectionParameters(
            weight=payload["weight"].to(device="npu"),
            gamma=payload["gamma"].to(device="npu"),
        )
        runtime.resources.parameters[parameter_key] = parameters
    else:
        _copy_stable_buffer(parameters.weight, payload["weight"], name="weight")
        _copy_stable_buffer(parameters.gamma, payload["gamma"], name="gamma")

    graph_key = _graph_resource_key(args, payload)
    resources = runtime.resources.graphs.get(graph_key)
    if resources is not None:
        if resources.parameters is not parameters:
            raise RuntimeError("resident graph resolved a different shared parameter buffer")
        _copy_stable_buffer(resources.x, payload["activation"], name="activation")
        _copy_stable_buffer(resources.residual, payload["residual"], name="residual")
        torch.npu.synchronize()
        return resources

    x = payload["activation"].to(device="npu")
    residual = payload["residual"].to(device="npu")
    chain_zero = torch.zeros((64, 4), dtype=torch.int64, device="npu")

    def baseline() -> Output:
        for _ in range(args.operations_per_graph):
            projected = torch.nn.functional.linear(x, parameters.weight)
            dist.all_reduce(projected)
            normalized, _, added = torch_npu.npu_add_rms_norm(
                projected,
                residual,
                parameters.gamma,
                1e-6,
            )
        return normalized, added

    def fused() -> Output:
        for _ in range(args.operations_per_graph):
            local_projection = torch.nn.functional.linear(x, parameters.weight)
            if args.chained_flush:
                chained_output = runtime.operator(
                    local_projection,
                    residual,
                    parameters.gamma,
                    chain_zero,
                    runtime.comm_name,
                    3,
                    runtime.rank,
                    1e-6,
                    True,
                    True,
                )
                output = (chained_output[0], chained_output[1])
            else:
                output = runtime.operator(
                    local_projection,
                    residual,
                    parameters.gamma,
                    runtime.comm_name,
                    3,
                    runtime.rank,
                    1e-6,
                    True,
                )
        return output

    dist.barrier()
    baseline_graph, baseline_output = _capture(baseline)
    dist.barrier()
    fused_graph, fused_output = _capture(fused)
    resources = ResidentGraphResources(
        x=x,
        residual=residual,
        chain_zero=chain_zero,
        parameters=parameters,
        baseline_graph=baseline_graph,
        baseline_output=baseline_output,
        fused_graph=fused_graph,
        fused_output=fused_output,
    )
    runtime.resources.graphs[graph_key] = resources
    return resources


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


def _measure_pair(
    baseline_runner: Callable[[], Output],
    fused_runner: Callable[[], Output],
    *,
    warmup_replays: int,
    replays_per_sample: int,
    samples: int,
) -> tuple[list[float], list[float]]:
    for _ in range(warmup_replays):
        baseline_runner()
        fused_runner()
    torch.npu.synchronize()
    values = {"baseline": [], "fused": []}
    runners = {"baseline": baseline_runner, "fused": fused_runner}
    for sample in range(samples):
        order = ("baseline", "fused") if sample % 2 == 0 else ("fused", "baseline")
        for name in order:
            dist.barrier()
            torch.npu.synchronize()
            started = time.perf_counter()
            for _ in range(replays_per_sample):
                runners[name]()
            torch.npu.synchronize()
            values[name].append(
                (time.perf_counter() - started) * 1000.0 / replays_per_sample
            )
    return values["baseline"], values["fused"]


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _sha256(value: torch.Tensor) -> str:
    data = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _capture_rank_zero_finalization(
    finalizer: Callable[[], str | None],
) -> RankZeroFinalizationOutcome:
    """Convert every rank-0-only finalization error into consensus data."""

    try:
        admission_failure = finalizer()
    except Exception as error:
        return RankZeroFinalizationOutcome(
            fatal_failure=(
                "rank-0 MC2 qualification finalization failed: "
                f"{type(error).__name__}: {error}"
            )
        )
    if admission_failure is not None and not isinstance(admission_failure, str):
        return RankZeroFinalizationOutcome(
            fatal_failure=(
                "rank-0 MC2 qualification finalization returned an invalid "
                f"admission payload: {type(admission_failure).__name__}"
            )
        )
    return RankZeroFinalizationOutcome(admission_failure=admission_failure)


def _broadcast_rank_zero_finalization(
    outcome: RankZeroFinalizationOutcome | None,
) -> RankZeroFinalizationOutcome:
    """Publish the rank-0 finalization result before any rank can exit."""

    outcome_box = [outcome]
    dist.broadcast_object_list(outcome_box, src=0)
    received = outcome_box[0]
    if not isinstance(received, RankZeroFinalizationOutcome):
        return RankZeroFinalizationOutcome(
            fatal_failure=(
                "rank-0 MC2 qualification finalization returned an invalid "
                f"outcome payload: {type(received).__name__}"
            )
        )
    return received


def _destroy_and_raise_collective_failure(profile_failure: str) -> None:
    """Make teardown best-effort while preserving the shared gate failure."""

    try:
        dist.destroy_process_group()
    finally:
        raise RuntimeError(profile_failure)


def _mapping(local_rank: int) -> tuple[str, ...]:
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    devices = tuple(part.strip() for part in visible.split(",") if part.strip())
    local_device = devices[local_rank] if len(devices) > 1 else (devices[0] if devices else str(local_rank))
    gathered: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_device)
    if any(value is None for value in gathered):
        raise RuntimeError("could not collect rank-to-device mapping")
    result = tuple(str(value) for value in gathered)
    if len(set(result)) != dist.get_world_size():
        raise ValueError("TP ranks must map to distinct devices")
    return result


def _materialize_and_resolve_comm(rank: int) -> str:
    """Materialize HCCL before resolving and collectively validate its name."""

    dist.barrier()
    probe = torch.tensor([rank + 1], dtype=torch.int32, device="npu")
    dist.all_reduce(probe)
    expected = dist.get_world_size() * (dist.get_world_size() + 1) // 2
    if int(probe.cpu().item()) != expected:
        raise RuntimeError("TP3 HCCL materialization probe produced an invalid sum")

    local_comm_name = resolve_hccl_comm_name(
        dist.group.WORLD,
        device="npu",
        rank=rank,
    )
    # HCCL handles may be process-local.  Reach consensus on successful
    # resolution, but each rank must retain its own handle rather than copying
    # rank 0's opaque value.
    local_status = (rank, bool(local_comm_name))
    gathered_status: list[tuple[int, bool] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_status, local_status)
    expected_status = [(index, True) for index in range(dist.get_world_size())]
    if gathered_status != expected_status:
        raise RuntimeError(
            "could not resolve a local TP3 HCCL communicator on every rank: "
            f"{gathered_status}"
        )
    return local_comm_name


def initialize_measurement_runtime(
    *,
    chained_flush: bool,
    require_production_operator: bool,
) -> MeasurementRuntime:
    """Initialize one TP3 HCCL group that may serve multiple shape jobs."""

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    if dist.get_world_size() != 3:
        raise ValueError("this qualification requires exactly three ranks")
    rank = dist.get_rank()
    torch.npu.config.allow_internal_format = True

    from vllm_ascend.utils import enable_custom_op

    if not enable_custom_op():
        raise RuntimeError("vLLM-Ascend custom operators could not be loaded")
    operator_name, operator = _resolve_operator(chained=chained_flush)
    required_operator = (
        "torch.ops._C_ascend.allreduce_add_rmsnorm_chained"
        if chained_flush
        else "torch.ops._C_ascend.allreduce_add_rmsnorm"
    )
    if require_production_operator and operator_name != required_operator:
        raise RuntimeError(
            "a production profile requires the in-tree _C_ascend registration"
        )
    comm_name = _materialize_and_resolve_comm(rank)
    device_mapping = _mapping(local_rank)
    runtime_binding = current_mc2_runtime_binding(
        tp_rank_device_mapping=device_mapping,
        operator=MC2_TP3_NATIVE_EPILOGUE_OPERATOR,
    )
    return MeasurementRuntime(
        local_rank=local_rank,
        rank=rank,
        operator_name=operator_name,
        operator=operator,
        comm_name=comm_name,
        device_mapping=device_mapping,
        runtime_binding=runtime_binding,
        chained_flush=chained_flush,
        resources=MeasurementResourcePool(),
    )


def release_measurement_job_resources() -> None:
    """Collect per-job temporaries while resident captured resources survive."""

    gc.collect()


def finalize_measurement_runtime(runtime: MeasurementRuntime) -> None:
    """End HCCL only after all resident graphs have completed their campaign."""

    dist.barrier()
    torch.npu.synchronize()
    dist.destroy_process_group()
    # Do not reset or empty the resident graph pool here.  CANN graph-task
    # handles may still own TP3 stream state; process teardown releases them
    # after the communicator lifecycle, with no subsequent capture to poison.


def run_measurement_job(
    args: argparse.Namespace,
    runtime: MeasurementRuntime,
) -> str | None:
    """Run one shape/layer qualification without recreating the HCCL group.

    A normal numerical or latency admission failure is returned as a string on
    every rank, allowing a persistent caller to record the rejected row and
    continue.  Unexpected runtime/device/HCCL exceptions are deliberately not
    swallowed so ``torchrun`` can terminate all ranks immediately.
    """

    _validate_args(args)
    if args.chained_flush != runtime.chained_flush:
        raise ValueError(
            "measurement job chained-flush mode differs from its runtime"
        )
    local_rank = runtime.local_rank
    rank = runtime.rank
    operator_name = runtime.operator_name

    payload = torch.load(
        args.input_dir / f"rank-{rank}.pt",
        map_location="cpu",
        weights_only=True,
    )
    _validate_payload_shapes(args, payload)
    if args.execution_mode == "graph":
        resources = _load_graph_resources(args, runtime, payload)
        x = resources.x
        residual = resources.residual
        weight = resources.parameters.weight
        gamma = resources.parameters.gamma
        baseline_runner = resources.baseline_runner
        fused_runner = resources.fused_runner

    else:
        x = payload["activation"].to(device="npu")
        weight = payload["weight"].to(device="npu")
        residual = payload["residual"].to(device="npu")
        gamma = payload["gamma"].to(device="npu")
        chain_zero = torch.zeros((64, 4), dtype=torch.int64, device="npu")

        def baseline() -> Output:
            for _ in range(args.operations_per_graph):
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
            for _ in range(args.operations_per_graph):
                local_projection = torch.nn.functional.linear(x, weight)
                if args.chained_flush:
                    chained_output = runtime.operator(
                        local_projection,
                        residual,
                        gamma,
                        chain_zero,
                        runtime.comm_name,
                        3,
                        rank,
                        1e-6,
                        True,
                        True,
                    )
                    output = (chained_output[0], chained_output[1])
                else:
                    output = runtime.operator(
                        local_projection,
                        residual,
                        gamma,
                        runtime.comm_name,
                        3,
                        rank,
                        1e-6,
                        True,
                    )
            return output

        baseline_output = baseline()
        torch.npu.synchronize()
        baseline_runner = baseline
        fused_output = fused()
        torch.npu.synchronize()
        fused_runner = fused

    # Replay both paths after loading this job into the stable buffers.  The
    # graph path may be reusing a capture created for another real layer.
    dist.barrier()
    baseline_output = baseline_runner()
    torch.npu.synchronize()
    baseline_reference = (baseline_output[0].clone(), baseline_output[1].clone())
    torch.npu.synchronize()
    dist.barrier()
    fused_output = fused_runner()
    torch.npu.synchronize()
    max_norm_error = _max_rank_value(
        float((fused_output[0].float() - baseline_reference[0].float()).abs().max().cpu())
    )
    max_added_error = _max_rank_value(
        float((fused_output[1].float() - baseline_reference[1].float()).abs().max().cpu())
    )
    exact = bool(
        torch.equal(fused_output[0], baseline_reference[0])
        and torch.equal(fused_output[1], baseline_reference[1])
    )
    exact_tensor = torch.tensor([int(exact)], dtype=torch.int32, device="npu")
    dist.all_reduce(exact_tensor, op=dist.ReduceOp.MIN)
    exact_all_ranks = bool(exact_tensor.cpu().item())

    initial = x.clone()
    changed_max_norm = 0.0
    changed_max_added = 0.0
    changed_exact = True
    for replay in range(args.changed_input_replays):
        units = (0, 1, -1, 2, -2)[replay % 5]
        x.copy_(initial)
        x.add_(args.changed_input_delta * (rank + 1) * units)
        expected_output = baseline_runner()
        torch.npu.synchronize()
        expected = (expected_output[0].clone(), expected_output[1].clone())
        actual_output = fused_runner()
        torch.npu.synchronize()
        changed_max_norm = max(
            changed_max_norm,
            float((actual_output[0].float() - expected[0].float()).abs().max().cpu()),
        )
        changed_max_added = max(
            changed_max_added,
            float((actual_output[1].float() - expected[1].float()).abs().max().cpu()),
        )
        changed_exact = changed_exact and torch.equal(actual_output[0], expected[0])
        changed_exact = changed_exact and torch.equal(actual_output[1], expected[1])
    x.copy_(initial)
    # A graph output tensor is mutable replay storage.  Restore one replay on
    # the original input before hashing/reporting it; otherwise graph mode
    # would report the final changed-input sample while eager mode reports the
    # original input, making provenance depend on the execution mode.
    fused_output = fused_runner()
    torch.npu.synchronize()
    changed_max_norm = _max_rank_value(changed_max_norm)
    changed_max_added = _max_rank_value(changed_max_added)
    changed_exact_tensor = torch.tensor([int(changed_exact)], dtype=torch.int32, device="npu")
    dist.all_reduce(changed_exact_tensor, op=dist.ReduceOp.MIN)
    changed_exact_all_ranks = bool(changed_exact_tensor.cpu().item())

    local_baseline, local_fused = _measure_pair(
        baseline_runner,
        fused_runner,
        warmup_replays=args.warmup_replays,
        replays_per_sample=args.replays_per_sample,
        samples=args.samples,
    )
    baseline_samples = _max_rank_samples(local_baseline)
    fused_samples = _max_rank_samples(local_fused)
    device_mapping = runtime.device_mapping
    runtime_binding = runtime.runtime_binding

    def _finalize_rank_zero() -> str | None:
        profile_failure: str | None = None
        baseline_p50 = _percentile(baseline_samples, 50.0)
        baseline_p95 = _percentile(baseline_samples, 95.0)
        fused_p50 = _percentile(fused_samples, 50.0)
        fused_p95 = _percentile(fused_samples, 95.0)
        document: dict[str, Any] = {
            "schema_version": 1,
            "status": "measured",
            "metadata": {
                "measurement_scope": (
                    f"{args.execution_mode}_native_matmul_plus_rank3_epilogue"
                ),
                "execution_mode": args.execution_mode,
                "operator": operator_name,
                "chained_flush": args.chained_flush,
                "operations_per_graph": args.operations_per_graph,
                "tensor_parallel_size": 3,
                "tp_rank_device_mapping": list(device_mapping),
                # The combined two-shape profile builder consumes this raw
                # document, so bind its evidence to the measured source tree.
                "source_sha256": mc2_source_sha256(),
                "input_source": str(args.input_dir.resolve()),
                "custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH", ""),
            },
            "shape": {"m": args.m, "k": args.k, "n": args.n},
            "correctness": {
                "exact_all_ranks": exact_all_ranks,
                "max_abs_norm": max_norm_error,
                "max_abs_added": max_added_error,
                "changed_input_replays": args.changed_input_replays,
                "changed_input_delta": args.changed_input_delta,
                "changed_input_exact_all_ranks": changed_exact_all_ranks,
                "changed_input_max_abs_norm": changed_max_norm,
                "changed_input_max_abs_added": changed_max_added,
                "fused_norm_sha256": _sha256(fused_output[0]),
                "fused_added_sha256": _sha256(fused_output[1]),
            },
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
                "baseline_per_operation_p50_ms": (
                    baseline_p50 / args.operations_per_graph
                ),
                "baseline_per_operation_p95_ms": (
                    baseline_p95 / args.operations_per_graph
                ),
                "fused_per_operation_p50_ms": (
                    fused_p50 / args.operations_per_graph
                ),
                "fused_per_operation_p95_ms": (
                    fused_p95 / args.operations_per_graph
                ),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2) + "\n")
        if args.production_profile_output is not None:
            if not exact_all_ranks or not changed_exact_all_ranks:
                profile_failure = (
                    "production MC2 profile requires bit-exact original and "
                    "changed-input outputs"
                )
            else:
                gate_baseline = _percentile(baseline_samples, args.latency_percentile)
                gate_fused = _percentile(fused_samples, args.latency_percentile)
                if gate_fused * args.minimum_speedup > gate_baseline:
                    profile_failure = (
                        "native-MatMul MC2 epilogue did not pass the "
                        "configured repeated latency gate"
                    )
            if profile_failure is None:
                production_profile = {
                    "schema_version": 1,
                    "status": "measured",
                    "metadata": {
                        "operator": MC2_TP3_NATIVE_EPILOGUE_OPERATOR,
                        # Production Native PEARL selects the allocation-free
                        # three-output ABI only when the qualification artifact
                        # proves that the chained state was flushed on every
                        # measured invocation.  Preserve that distinction when
                        # single-shape profiles are later aggregated.
                        "standalone_chained_flush": bool(args.chained_flush),
                        "hardware": str(torch.npu.get_device_name(local_rank)),
                        "tensor_parallel_size": 3,
                        "source_sha256": mc2_source_sha256(),
                        "rms_norm_epsilon": 1e-6,
                        "runtime_binding": runtime_binding,
                        "measurement_scope": "captured_operator_replay",
                        "input_source": str(args.input_dir.resolve()),
                    },
                    "latency_percentile": args.latency_percentile,
                    "minimum_samples": args.samples,
                    "minimum_speedup": args.minimum_speedup,
                    "measurements": [
                        {
                            "m": args.m,
                            "k": args.k,
                            "n": args.n,
                            "dtype": "bfloat16",
                            "weight_format": str(torch_npu.get_npu_format(weight)),
                            "is_trans_b": True,
                            "activation_layout": "contiguous",
                            "weight_layout": "contiguous",
                            "residual_layout": "contiguous",
                            "gamma_layout": "contiguous",
                            "baseline_latency_ms": baseline_samples,
                            "fused_latency_ms": fused_samples,
                            "max_abs_norm": max(max_norm_error, changed_max_norm),
                            "norm_atol": 0.0,
                            "norm_rtol": 1e-3,
                            "max_scaled_norm": 0.0,
                            "max_abs_added": max(max_added_error, changed_max_added),
                            "added_atol": 0.0,
                            "added_rtol": 1e-3,
                            "max_scaled_added": 0.0,
                            "changed_input_replays": args.changed_input_replays,
                            "changed_input_delta": args.changed_input_delta,
                            "changed_input_max_abs_norm": changed_max_norm,
                            "changed_input_max_abs_added": changed_max_added,
                            "changed_input_max_scaled_norm": 0.0,
                            "changed_input_max_scaled_added": 0.0,
                        }
                    ],
                }
                try:
                    # Re-run the exact production loader before persisting a
                    # profile; malformed or slower rows must remain diagnostics
                    # only.  This is the sole expected validation rejection;
                    # I/O and device-query exceptions remain campaign-fatal.
                    normalize_mc2_profile(production_profile)
                except ValueError as error:
                    profile_failure = (
                        "production MC2 profile validation failed: "
                        f"{type(error).__name__}: {error}"
                    )
                if profile_failure is None:
                    args.production_profile_output.parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    args.production_profile_output.write_text(
                        json.dumps(production_profile, indent=2) + "\n",
                        encoding="utf-8",
                    )
        print(json.dumps({**document["correctness"], **document["summary"]}, indent=2))
        return profile_failure

    finalization_outcome = None
    if rank == 0:
        finalization_outcome = _capture_rank_zero_finalization(_finalize_rank_zero)

    # A rank-0-only exception leaves the other ranks inside the final HCCL
    # barrier.  Besides hanging the qualifier, peer teardown can surface as a
    # misleading SDMA/MTE device error.  Broadcast the structured result so
    # every rank either returns one admission rejection or raises one fatal.
    finalization_outcome = _broadcast_rank_zero_finalization(
        finalization_outcome
    )
    if finalization_outcome.fatal_failure is not None:
        raise RuntimeError(finalization_outcome.fatal_failure)
    dist.barrier()
    return finalization_outcome.admission_failure


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    runtime = initialize_measurement_runtime(
        chained_flush=args.chained_flush,
        require_production_operator=args.production_profile_output is not None,
    )
    profile_failure = run_measurement_job(args, runtime)
    if profile_failure is not None:
        _destroy_and_raise_collective_failure(profile_failure)
    finalize_measurement_runtime(runtime)


if __name__ == "__main__":
    main()
