# SPDX-License-Identifier: Apache-2.0
"""Numerically inspect TP3 fused MC2 invocations, optionally with ACLGraph.

This is intentionally a small eager-only diagnostic.  It reports the rows and
columns with the largest errors so TP3 window/synchronization failures are not
hidden behind a single global tolerance value.
"""

from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu

from vllm_ascend.spec_decode.pearl.mc2 import resolve_hccl_comm_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=80)
    parser.add_argument("--k", type=int, default=3072)
    parser.add_argument("--n", type=int, default=5120)
    parser.add_argument(
        "--layers",
        type=int,
        default=1,
        help="Number of sequential MC2 launches recorded in one graph.",
    )
    parser.add_argument(
        "--interleave-all-reduce",
        action="store_true",
        help="Record a regular HCCL all-reduce after every fused MC2 launch.",
    )
    parser.add_argument(
        "--dependent-all-reduce",
        action="store_true",
        help="Feed each fused output through all-reduce into the next MC2 input.",
    )
    parser.add_argument(
        "--target-subgroup",
        action="store_true",
        help="Use global ranks 1,2,3 as a TP3 subgroup inside a four-rank world.",
    )
    parser.add_argument(
        "--pearl-groups",
        action="store_true",
        help=(
            "Create PEARL's overlapping draft/target/verification/correction "
            "HCCL groups before selecting the TP3 target subgroup."
        ),
    )
    parser.add_argument(
        "--init-triton-properties",
        action="store_true",
        help="Initialize vLLM-Ascend Triton device properties before process groups.",
    )
    parser.add_argument(
        "--rank0-busy",
        action="store_true",
        help="Run independent rank-0 NPU matmuls while the TP3 subgroup executes MC2.",
    )
    parser.add_argument(
        "--prefix-all-reduces",
        type=int,
        default=0,
        help="Run ordinary target-group all-reduces before first MC2 use.",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--mutate-input-between-repeats",
        action="store_true",
        help=(
            "Change the rank-local activation after every invocation and "
            "rebuild the reference, exposing one-invocation-stale windows."
        ),
    )
    parser.add_argument(
        "--precompute-references",
        action="store_true",
        help=(
            "Build all dynamic references before the first MC2 invocation "
            "and defer diagnostic collectives until the end. This prevents "
            "ordinary HCCL reference work from reusing the MC2 window between "
            "two fused invocations."
        ),
    )
    parser.add_argument(
        "--host-reference-collectives",
        action="store_true",
        help=(
            "Build references and coordinate diagnostics through Gloo/CPU, "
            "so ordinary HCCL collectives cannot populate the MC2 windows."
        ),
    )
    parser.add_argument(
        "--fresh-input-between-repeats",
        action="store_true",
        help=(
            "Allocate a fresh activation tensor for every dynamic invocation "
            "instead of updating the original address in place. This isolates "
            "stable-address cache visibility from TP3 reduction correctness."
        ),
    )
    parser.add_argument(
        "--sync-after-mutation",
        action="store_true",
        help="Synchronize the NPU immediately after changing the activation.",
    )
    parser.add_argument("--graph", action="store_true")
    parser.add_argument(
        "--projection-only",
        action="store_true",
        help="Inspect only fused MatMul+AllReduce, excluding AddRMSNorm.",
    )
    parser.add_argument(
        "--expect-local-projection",
        action="store_true",
        help=(
            "Compare projection-only output with this rank's local matmul. "
            "Used only by the Cube-publication diagnostic kernel build."
        ),
    )
    parser.add_argument(
        "--official-mm-all-reduce",
        action="store_true",
        help="Use torch-npu's public fused MatMul+AllReduce followed by AddRMSNorm.",
    )
    parser.add_argument(
        "--skip-baseline-before",
        action="store_true",
        help="Skip the ordinary matmul/all-reduce reference before the first MC2 launch.",
    )
    parser.add_argument(
        "--reserve-gib",
        type=float,
        default=0.0,
        help="Reserve device memory before creating MC2 inputs to reproduce model-memory pressure.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help=(
            "Load activation/weight/residual/gamma from rank-{tp_rank}.pt "
            "instead of generating synthetic inputs."
        ),
    )
    args = parser.parse_args()
    if args.fresh_input_between_repeats and args.graph:
        raise ValueError("--fresh-input-between-repeats is an eager-only diagnostic")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    if args.init_triton_properties:
        from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

        init_device_properties_triton()
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    if args.target_subgroup:
        if dist.get_world_size() != 4:
            raise ValueError("--target-subgroup requires exactly four world ranks")
        if args.pearl_groups:
            dist.new_group([0])
            process_group = dist.new_group([1, 2, 3])
            dist.new_group([0, 1, 2, 3])
            dist.new_group([0, 1])
            dist.new_group([0, 1, 2, 3], backend="gloo")
            dist.new_group([0, 1], backend="gloo")
        else:
            process_group = dist.new_group([1, 2, 3])
        if rank == 0:
            if args.rank0_busy:
                busy_x = torch.randn((64, 1024), dtype=torch.bfloat16, device="npu")
                busy_weight = torch.randn((1024, 1024), dtype=torch.bfloat16, device="npu")
                for _ in range(64):
                    busy_x = torch.nn.functional.linear(busy_x, busy_weight)
                torch.npu.synchronize()
            dist.barrier()
            dist.destroy_process_group()
            return
        tp_rank = rank - 1
        source_rank = 1
    else:
        if args.pearl_groups:
            raise ValueError("--pearl-groups requires --target-subgroup")
        if dist.get_world_size() != 3:
            raise ValueError("This diagnostic requires exactly three ranks")
        process_group = dist.group.WORLD
        tp_rank = rank
        source_rank = 0
    torch.npu.config.allow_internal_format = True

    from vllm_ascend.utils import enable_custom_op

    if not enable_custom_op():
        raise RuntimeError("vLLM-Ascend custom operators could not be loaded")
    operator = torch.ops._C_ascend.matmul_allreduce_add_rmsnorm
    comm_name = resolve_hccl_comm_name(process_group, device="npu", rank=tp_rank)
    if not comm_name:
        raise RuntimeError("Could not resolve the TP3 HCCL communicator")
    print(
        f"[rank={rank} tp_rank={tp_rank}] comm_name={comm_name}",
        flush=True,
    )
    reference_group = None
    if args.host_reference_collectives:
        if args.target_subgroup:
            raise ValueError(
                "--host-reference-collectives currently supports the three-rank diagnostic"
            )
        reference_group = dist.new_group([0, 1, 2], backend="gloo")
    reserved = None
    if args.reserve_gib:
        if args.reserve_gib < 0:
            raise ValueError("--reserve-gib must be non-negative")
        reserved = torch.empty(
            int(args.reserve_gib * (1024**3) // 2),
            dtype=torch.bfloat16,
            device="npu",
        )

    if args.input_dir is not None:
        payload = torch.load(
            args.input_dir / f"rank-{tp_rank}.pt",
            map_location="cpu",
            weights_only=True,
        )
        x = payload["activation"].to(device="npu")
        weight = payload["weight"].to(device="npu")
        residual = payload["residual"].to(device="npu")
        gamma = payload["gamma"].to(device="npu")
        if x.shape != (args.m, args.k):
            raise ValueError(f"loaded activation shape {tuple(x.shape)} != {(args.m, args.k)}")
        if weight.shape != (args.n, args.k):
            raise ValueError(f"loaded weight shape {tuple(weight.shape)} != {(args.n, args.k)}")
        if residual.shape != (args.m, args.n):
            raise ValueError(f"loaded residual shape {tuple(residual.shape)} != {(args.m, args.n)}")
        if gamma.shape != (args.n,):
            raise ValueError(f"loaded gamma shape {tuple(gamma.shape)} != {(args.n,)}")
    else:
        torch.manual_seed(1701 + rank * 97 + args.m + args.k)
        x = (torch.randn((args.m, args.k), dtype=torch.float32, device="npu") * 0.02).to(torch.bfloat16)
        weight = (torch.randn((args.n, args.k), dtype=torch.float32, device="npu") * 0.02).to(torch.bfloat16)
        residual = (torch.randn((args.m, args.n), dtype=torch.float32, device="npu") * 0.02).to(torch.bfloat16)
        if args.host_reference_collectives:
            residual_object = [residual.cpu() if tp_rank == 0 else None]
            dist.broadcast_object_list(
                residual_object,
                src=source_rank,
                group=reference_group,
            )
            residual = residual_object[0].to(device="npu")
        else:
            dist.broadcast(residual, src=source_rank, group=process_group)
        gamma = torch.ones(args.n, dtype=torch.bfloat16, device="npu")
    interleave = torch.zeros((args.m, args.n), dtype=torch.bfloat16, device="npu")
    for _ in range(args.prefix_all_reduces):
        dist.all_reduce(interleave, group=process_group)
    torch.npu.synchronize()

    def build_reference(input_x: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        dict[str, float],
        torch.Tensor,
    ]:
        projected = torch.nn.functional.linear(input_x, weight)
        if args.host_reference_collectives:
            projected_objects: list[torch.Tensor | None] = [None] * 3
            dist.all_gather_object(
                projected_objects,
                projected.cpu(),
                group=reference_group,
            )
            projected_by_rank = [
                value.to(device="npu")
                for value in projected_objects
                if value is not None
            ]
            expected_projection = projected_by_rank[0].clone()
            for value in projected_by_rank[1:]:
                expected_projection.add_(value)
        else:
            projected_by_rank = [torch.empty_like(projected) for _ in range(3)]
            dist.all_gather(projected_by_rank, projected, group=process_group)
            dist.all_reduce(projected, group=process_group)
            expected_projection = projected.clone()
        expected_norm, _, expected_added = torch_npu.npu_add_rms_norm(
            expected_projection, residual, gamma, 1e-6
        )
        expected_rms_only, _ = torch_npu.npu_rms_norm(expected_added, gamma, 1e-6)
        ordered_candidates: dict[str, torch.Tensor] = {}
        for order in itertools.permutations(range(3)):
            candidate = projected_by_rank[order[0]].clone()
            candidate.add_(projected_by_rank[order[1]])
            candidate.add_(projected_by_rank[order[2]])
            _, _, candidate_added = torch_npu.npu_add_rms_norm(
                candidate,
                residual,
                gamma,
                1e-6,
            )
            ordered_candidates["".join(str(value) for value in order)] = candidate_added
        torch.npu.synchronize()
        ordered_reduction_errors = {
            order: float((candidate.float() - expected_added.float()).abs().max().cpu())
            for order, candidate in ordered_candidates.items()
        }
        return (
            expected_norm,
            expected_rms_only,
            expected_added,
            projected_by_rank,
            ordered_reduction_errors,
            expected_projection,
        )

    precomputed_references: list[tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        dict[str, float],
        torch.Tensor,
    ]] = []
    if args.precompute_references:
        if not args.mutate_input_between_repeats:
            raise ValueError("--precompute-references requires --mutate-input-between-repeats")
        initial_x = x.clone()
        for invocation in range(args.repeats):
            if invocation > 0:
                x.add_(0.015625 * (tp_rank + 1))
            precomputed_references.append(build_reference(x))
        x.copy_(initial_x)
        torch.npu.synchronize()

    if args.skip_baseline_before:
        expected_norm = torch.zeros_like(residual)
        expected_rms_only = torch.zeros_like(residual)
        expected_added = torch.zeros_like(residual)
        projected_by_rank: list[torch.Tensor] = []
        ordered_reduction_errors: dict[str, float] = {}
        expected_projection = torch.zeros_like(residual)
    elif args.precompute_references:
        (
            expected_norm,
            expected_rms_only,
            expected_added,
            projected_by_rank,
            ordered_reduction_errors,
            expected_projection,
        ) = precomputed_references[0]
    else:
        (
            expected_norm,
            expected_rms_only,
            expected_added,
            projected_by_rank,
            ordered_reduction_errors,
            expected_projection,
        ) = build_reference(x)

    graph = None
    actual: tuple[torch.Tensor, torch.Tensor] | None = None
    previous_projected_by_rank: list[torch.Tensor] | None = None
    deferred_locals: list[dict[str, object]] = []

    def invoke_chain() -> tuple[torch.Tensor, torch.Tensor]:
        output: tuple[torch.Tensor, torch.Tensor] | None = None
        chain_x = x
        for _ in range(args.layers):
            if args.official_mm_all_reduce:
                projected = torch_npu.npu_mm_all_reduce_base(
                    chain_x,
                    weight.t(),
                    comm_name,
                    bias=None,
                )
                normalized, _, added = torch_npu.npu_add_rms_norm(
                    projected,
                    residual,
                    gamma,
                    1e-6,
                )
                output = (normalized, added)
            else:
                output = operator(
                    chain_x,
                    weight,
                    residual,
                    gamma,
                    comm_name,
                    3,
                    tp_rank,
                    1e-6,
                    True,
                    True,
                    args.projection_only,
                )
            if args.dependent_all_reduce:
                interleave.copy_(output[0])
                dist.all_reduce(interleave, group=process_group)
                chain_x = interleave[:, : args.k].contiguous()
            elif args.interleave_all_reduce:
                dist.all_reduce(interleave, group=process_group)
        assert output is not None
        return output

    if args.graph:
        # Run the same eager warmup used by production before creating a
        # static-address graph instance.
        invoke_chain()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = invoke_chain()
        torch.npu.synchronize()

    for invocation in range(args.repeats):
        if args.mutate_input_between_repeats and invocation > 0:
            previous_projected_by_rank = [value.clone() for value in projected_by_rank]
            delta = 0.015625 * (tp_rank + 1)
            if args.fresh_input_between_repeats:
                x = (x.float() + delta).to(torch.bfloat16).contiguous()
            else:
                x.add_(delta)
            if args.sync_after_mutation:
                torch.npu.synchronize()
            if args.precompute_references:
                (
                    expected_norm,
                    expected_rms_only,
                    expected_added,
                    projected_by_rank,
                    ordered_reduction_errors,
                    expected_projection,
                ) = precomputed_references[invocation]
            else:
                (
                    expected_norm,
                    expected_rms_only,
                    expected_added,
                    projected_by_rank,
                    ordered_reduction_errors,
                    expected_projection,
                ) = build_reference(x)
        dist.barrier(
            group=reference_group if args.host_reference_collectives else process_group
        )
        if graph is None:
            actual_norm, actual_added = invoke_chain()
        else:
            graph.replay()
            assert actual is not None
            actual_norm, actual_added = actual
        torch.npu.synchronize()
        if args.projection_only:
            actual_added = actual_norm
            projection_reference = (
                projected_by_rank[tp_rank]
                if args.expect_local_projection
                else expected_projection
            )
            expected_norm = projection_reference
            expected_rms_only = projection_reference
            expected_added = projection_reference
        norm_error = (actual_norm.float() - expected_norm.float()).abs().cpu()
        rms_only_error = (actual_norm.float() - expected_rms_only.float()).abs().cpu()
        native_norm_delta = (expected_norm.float() - expected_rms_only.float()).abs().cpu()
        added_error = (actual_added.float() - expected_added.float()).abs().cpu()
        stale_rank_combination_errors: dict[str, float] = {}
        if args.projection_only and previous_projected_by_rank is not None:
            for current_mask in range(8):
                candidate = torch.zeros_like(actual_added)
                label = []
                for source_rank in range(3):
                    use_current = bool(current_mask & (1 << source_rank))
                    candidate.add_(
                        projected_by_rank[source_rank]
                        if use_current
                        else previous_projected_by_rank[source_rank]
                    )
                    label.append("C" if use_current else "P")
                stale_rank_combination_errors["".join(label)] = float(
                    (actual_added.float() - candidate.float()).abs().max().cpu()
                )
        row_error = added_error.amax(dim=1)
        column_error = added_error.amax(dim=0)
        worst_rows = torch.topk(row_error, k=min(10, args.m))
        worst_columns = torch.topk(column_error, k=min(10, args.n))
        flat_worst = int(added_error.reshape(-1).argmax())
        worst_row, worst_column = divmod(flat_worst, args.n)
        local = {
            "rank": rank,
            "tp_rank": tp_rank,
            "invocation": invocation,
            "norm_max": float(norm_error.max()),
            "norm_vs_rms_only_max": float(rms_only_error.max()),
            "native_add_vs_rms_only_max": float(native_norm_delta.max()),
            "added_max": float(added_error.max()),
            "added_worst_coordinate": (worst_row, worst_column),
            "added_worst_actual": float(actual_added[worst_row, worst_column].float().cpu()),
            "added_worst_expected": float(expected_added[worst_row, worst_column].float().cpu()),
            "added_worst_residual": float(residual[worst_row, worst_column].float().cpu()),
            "added_worst_rank_projections": [
                float(value[worst_row, worst_column].float().cpu())
                for value in projected_by_rank
            ] if not args.skip_baseline_before else [],
            "all_bad_rows": [
                (int(row), float(error))
                for row, error in enumerate(row_error)
                if error > 0
            ],
            "bad_rows": [
                (int(row), float(error))
                for row, error in zip(worst_rows.indices, worst_rows.values, strict=True)
                if error > 0
            ],
            "column_tile_max": [
                float(column_error[start:min(start + 256, args.n)].max())
                for start in range(0, args.n, 256)
            ],
            "bad_columns": [
                (int(column), float(error))
                for column, error in zip(
                    worst_columns.indices, worst_columns.values, strict=True
                )
                if error > 0
            ],
            "ordered_reduction_errors": ordered_reduction_errors,
            "stale_rank_combination_errors": stale_rank_combination_errors,
        }
        if args.precompute_references:
            deferred_locals.append(local)
        else:
            gathered: list[dict[str, object] | None] = [None] * 3
            dist.all_gather_object(
                gathered,
                local,
                group=reference_group if args.host_reference_collectives else process_group,
            )
            if tp_rank == 0:
                print(gathered, flush=True)

    if args.precompute_references:
        for local in deferred_locals:
            gathered = [None] * 3
            dist.all_gather_object(
                gathered,
                local,
                group=reference_group if args.host_reference_collectives else process_group,
            )
            if tp_rank == 0:
                print(gathered, flush=True)

    if args.target_subgroup:
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
