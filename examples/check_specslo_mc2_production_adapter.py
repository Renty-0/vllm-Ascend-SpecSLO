# SPDX-License-Identifier: Apache-2.0
"""Smoke-test the production SpecSLO TP3 MC2 adapter on real inputs.

The active custom-op vendor package must be selected by the caller before
launching this script.  A typical invocation uses exactly three workers::

    torchrun --standalone --nproc-per-node=3 -- \
      examples/check_specslo_mc2_production_adapter.py \
      --profile mc2-profile.json --input-dir mc2-real-inputs

The check is deliberately fail-closed: the eager launch and ACLGraph capture
must both use the qualified fused adapter.  A fallback, dispatch exception,
numerical mismatch, or rank disagreement makes the process exit non-zero.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu

from vllm_ascend.spec_decode.pearl.mc2 import (
    matmul_allreduce_add_rmsnorm_or_fallback,
    normalize_mc2_profile,
    reset_mc2_dispatch_counters,
    resolve_hccl_comm_name,
    snapshot_mc2_dispatch_counters,
)

M = 64
K = 3072
N = 5120
EPSILON = 1e-6
TP_SIZE = 3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True, help="Identity-bound M64/K3072/N5120 profile.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing rank-{0,1,2}.pt real-layer payloads.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of ACLGraph replays after capture (host dispatch counters must remain unchanged).",
    )
    parser.add_argument(
        "--changed-input-replays",
        type=int,
        default=8,
        help="Number of in-place activation updates validated against precomputed split references.",
    )
    parser.add_argument(
        "--changed-input-delta",
        type=float,
        default=0.015625,
        help="Per-rank BF16 activation increment used by changed-input validation.",
    )
    return parser


def _load_inputs(input_dir: Path, rank: int) -> tuple[torch.Tensor, ...]:
    payload = torch.load(
        input_dir / f"rank-{rank}.pt",
        map_location="cpu",
        weights_only=True,
    )
    required = {"activation", "weight", "residual", "gamma"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"rank-{rank}.pt is missing tensors: {sorted(missing)}")
    tensors = tuple(payload[name].to(device="npu") for name in ("activation", "weight", "residual", "gamma"))
    expected_shapes = ((M, K), (N, K), (M, N), (N,))
    actual_shapes = tuple(tuple(tensor.shape) for tensor in tensors)
    if actual_shapes != expected_shapes:
        raise ValueError(f"rank {rank} input shapes {actual_shapes!r} != {expected_shapes!r}")
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        raise ValueError(f"rank {rank} production smoke inputs must all be bfloat16")
    return tensors


def _profile_tolerances(
    profile: Any,
    x: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[float, float, float, float]:
    key = (
        M,
        K,
        N,
        str(x.dtype).removeprefix("torch."),
        str(torch_npu.get_npu_format(weight)),
        True,
    )
    row = profile.entries.get(key)
    if row is None:
        raise ValueError(f"profile does not contain the exact production shape {key!r}")
    return (
        float(row["norm_atol"]),
        float(row.get("norm_rtol", 0.0)),
        float(row["added_atol"]),
        float(row.get("added_rtol", 0.0)),
    )


def _split_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    projected = F.linear(x, weight)
    dist.all_reduce(projected, group=dist.group.WORLD)
    normalized, _, added = torch_npu.npu_add_rms_norm(projected, residual, gamma, EPSILON)
    torch.npu.synchronize()
    return normalized, added


def _max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).abs().max().cpu().item())


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


def _assert_numerical(
    label: str,
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
    *,
    norm_atol: float,
    norm_rtol: float,
    added_atol: float,
    added_rtol: float,
) -> dict[str, float]:
    torch.npu.synchronize()
    norm_error = _max_abs_error(actual[0], expected[0])
    added_error = _max_abs_error(actual[1], expected[1])
    norm_scaled = _max_scaled_error(actual[0], expected[0], atol=norm_atol, rtol=norm_rtol)
    added_scaled = _max_scaled_error(actual[1], expected[1], atol=added_atol, rtol=added_rtol)
    if norm_scaled > 1.0 or added_scaled > 1.0:
        raise AssertionError(
            f"{label} differs from split reference: norm_abs={norm_error:.8f}, "
            f"norm_scaled={norm_scaled:.8f}; added_abs={added_error:.8f}, "
            f"added_scaled={added_scaled:.8f}"
        )
    return {
        "norm_max_abs": norm_error,
        "norm_max_scaled": norm_scaled,
        "added_max_abs": added_error,
        "added_max_scaled": added_scaled,
    }


def _assert_rank_consistency(
    label: str,
    outputs: tuple[torch.Tensor, torch.Tensor],
    *,
    norm_atol: float,
    norm_rtol: float,
    added_atol: float,
    added_rtol: float,
) -> dict[str, float]:
    errors: dict[str, float] = {}
    for name, tensor, atol, rtol in (
        ("norm", outputs[0], norm_atol, norm_rtol),
        ("added", outputs[1], added_atol, added_rtol),
    ):
        gathered = [torch.empty_like(tensor) for _ in range(TP_SIZE)]
        dist.all_gather(gathered, tensor, group=dist.group.WORLD)
        maximum = max(_max_abs_error(candidate, gathered[0]) for candidate in gathered[1:])
        maximum_scaled = max(
            _max_scaled_error(candidate, gathered[0], atol=atol, rtol=rtol) for candidate in gathered[1:]
        )
        if maximum_scaled > 1.0:
            raise AssertionError(
                f"{label} differs across TP ranks for {name}: abs={maximum:.8f}, scaled={maximum_scaled:.8f}"
            )
        errors[f"{name}_rank_max_abs"] = maximum
        errors[f"{name}_rank_max_scaled"] = maximum_scaled
    return errors


def _assert_fused_counters(counters: dict[str, int]) -> None:
    if counters["fused_attempt"] <= 0 or counters["fused_attempt"] != counters["fused_success"]:
        raise AssertionError(f"not every fused attempt succeeded: {counters}")
    if counters["fallback"] != 0 or counters["exception"] != 0:
        raise AssertionError(f"production adapter fell back or raised internally: {counters}")


def main() -> None:
    args = _parser().parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.changed_input_replays <= 0 or not math.isfinite(args.changed_input_delta):
        raise ValueError("changed-input replay count must be positive and delta finite")

    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.npu.set_device(local_rank)
        dist.init_process_group("hccl")
        if dist.get_world_size() != TP_SIZE:
            raise ValueError("SpecSLO MC2 production smoke requires exactly three HCCL ranks")
        rank = dist.get_rank()
        torch.npu.config.allow_internal_format = True

        # The caller owns vendor selection (for example by sourcing the
        # package's set_env.bash).  This only loads the already-active package.
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            raise RuntimeError("the externally selected vLLM-Ascend custom-op package could not be loaded")

        profile = normalize_mc2_profile(args.profile)
        x, weight, residual, gamma = _load_inputs(args.input_dir, rank)
        norm_atol, norm_rtol, added_atol, added_rtol = _profile_tolerances(profile, x, weight)
        comm_name = resolve_hccl_comm_name(dist.group.WORLD, device="npu", rank=rank)
        if not comm_name:
            raise RuntimeError("could not resolve the TP3 HCCL communicator")

        initial_x = x.clone()
        changed_references: list[tuple[torch.Tensor, torch.Tensor]] = []
        increment = float(args.changed_input_delta) * (rank + 1)
        for replay in range(args.changed_input_replays):
            if replay:
                x.add_(increment)
            reference = _split_reference(x, weight, residual, gamma)
            changed_references.append((reference[0].clone(), reference[1].clone()))
        x.copy_(initial_x)
        torch.npu.synchronize()
        dist.barrier()
        reference = changed_references[0]

        def invoke() -> tuple[torch.Tensor, torch.Tensor]:
            return matmul_allreduce_add_rmsnorm_or_fallback(
                x,
                weight,
                residual,
                gamma,
                group_tp=comm_name,
                tp_rank_size=TP_SIZE,
                tp_rank_id=rank,
                epsilon=EPSILON,
                is_trans_b=True,
                is_gather_add_out=True,
                process_group=dist.group.WORLD,
                use_fused=True,
                strict_fused=True,
                profile=profile,
            )

        reset_mc2_dispatch_counters()
        eager_raw = invoke()
        eager = (eager_raw[0].clone(), eager_raw[1].clone())
        torch.npu.synchronize()
        eager_errors = _assert_numerical(
            "eager fused adapter",
            eager,
            reference,
            norm_atol=norm_atol,
            norm_rtol=norm_rtol,
            added_atol=added_atol,
            added_rtol=added_rtol,
        )

        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            graph_output = invoke()
        torch.npu.synchronize()
        counters_after_capture = snapshot_mc2_dispatch_counters()
        _assert_fused_counters(counters_after_capture)

        for _ in range(args.repeats):
            graph.replay()
        torch.npu.synchronize()
        counters_after_replay = snapshot_mc2_dispatch_counters()
        if counters_after_replay != counters_after_capture:
            raise AssertionError(
                "ACLGraph replay unexpectedly re-entered the Python dispatch adapter: "
                f"capture={counters_after_capture}, replay={counters_after_replay}"
            )
        _assert_fused_counters(counters_after_replay)
        graph_errors = _assert_numerical(
            "ACLGraph fused adapter",
            graph_output,
            reference,
            norm_atol=norm_atol,
            norm_rtol=norm_rtol,
            added_atol=added_atol,
            added_rtol=added_rtol,
        )
        initial_graph_output = (graph_output[0].clone(), graph_output[1].clone())

        changed_errors = {
            "norm_max_abs": 0.0,
            "norm_max_scaled": 0.0,
            "added_max_abs": 0.0,
            "added_max_scaled": 0.0,
        }
        final_changed_output: tuple[torch.Tensor, torch.Tensor] | None = None
        x.copy_(initial_x)
        for replay, changed_reference in enumerate(changed_references):
            if replay:
                x.add_(increment)
            graph.replay()
            torch.npu.synchronize()
            replay_errors = _assert_numerical(
                f"ACLGraph changed-input replay {replay}",
                graph_output,
                changed_reference,
                norm_atol=norm_atol,
                norm_rtol=norm_rtol,
                added_atol=added_atol,
                added_rtol=added_rtol,
            )
            for name, value in replay_errors.items():
                changed_errors[name] = max(changed_errors[name], value)
            final_changed_output = (graph_output[0].clone(), graph_output[1].clone())
        x.copy_(initial_x)
        torch.npu.synchronize()
        assert final_changed_output is not None
        changed_rank_errors = _assert_rank_consistency(
            "final ACLGraph changed-input replay",
            final_changed_output,
            norm_atol=norm_atol,
            norm_rtol=norm_rtol,
            added_atol=added_atol,
            added_rtol=added_rtol,
        )
        counters_after_changed_input = snapshot_mc2_dispatch_counters()
        if counters_after_changed_input != counters_after_capture:
            raise AssertionError(
                "changed-input ACLGraph replay unexpectedly re-entered Python dispatch: "
                f"capture={counters_after_capture}, replay={counters_after_changed_input}"
            )
        _assert_fused_counters(counters_after_changed_input)
        eager_rank_errors = _assert_rank_consistency(
            "eager fused adapter",
            eager,
            norm_atol=norm_atol,
            norm_rtol=norm_rtol,
            added_atol=added_atol,
            added_rtol=added_rtol,
        )
        graph_rank_errors = _assert_rank_consistency(
            "ACLGraph fused adapter",
            initial_graph_output,
            norm_atol=norm_atol,
            norm_rtol=norm_rtol,
            added_atol=added_atol,
            added_rtol=added_rtol,
        )

        counters_by_rank: list[dict[str, int] | None] = [None] * TP_SIZE
        dist.all_gather_object(counters_by_rank, counters_after_changed_input)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "shape": {"m": M, "k": K, "n": N},
                        "graph_replays": args.repeats,
                        "changed_input_replays": args.changed_input_replays,
                        "changed_input_delta": args.changed_input_delta,
                        "tolerances": {
                            "norm_atol": norm_atol,
                            "norm_rtol": norm_rtol,
                            "added_atol": added_atol,
                            "added_rtol": added_rtol,
                        },
                        "eager": {**eager_errors, **eager_rank_errors},
                        "aclgraph": {**graph_errors, **graph_rank_errors},
                        "changed_input_aclgraph": {**changed_errors, **changed_rank_errors},
                        "dispatch_counters_by_rank": counters_by_rank,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
