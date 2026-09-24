# SPDX-License-Identifier: Apache-2.0
"""Probe the TP3 BF16 HCCL tree in PEARL's four-process topology.

Global rank 0 represents the TP1 draft worker and performs coordination only.
Global target ranks 1, 2, and 3 form a TP3 HCCL subgroup and load captured
``rank-0.pt``, ``rank-1.pt``, and ``rank-2.pt`` inputs, respectively.  Every
world rank creates every process group in the same order to avoid subgroup
bootstrap divergence.

Example::

    HCCL_DETERMINISTIC=true HCCL_OP_EXPANSION_MODE=AIV \
    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
    torchrun --standalone --nproc-per-node=4 \
      examples/probe_specslo_mc2_subgroup_hccl_tree.py \
      --physical-devices 0,1,2,3 \
      --input-dir /path/to/real-inputs \
      --output-json /path/to/subgroup-hccl-tree.json

The JSON and ``signature-u8.bin.gz`` artifacts use the same candidate bit
legend and per-element representation as the original standalone TP3 probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from examples.specslo_hccl_bf16_tree_report import (
    CANDIDATE_NAMES,
    build_signature_map,
    make_candidates,
    summarize_output,
    tensor_sha256,
    write_element_signature_artifact,
)

WORLD_SIZE = 4
COORDINATOR_GLOBAL_RANK = 0
TARGET_GLOBAL_RANKS = (1, 2, 3)
TARGET_WORLD_SIZE = len(TARGET_GLOBAL_RANKS)
GROUP_CREATION_PLAN = (
    ("target_hccl", TARGET_GLOBAL_RANKS, "hccl"),
    ("target_gloo", TARGET_GLOBAL_RANKS, "gloo"),
    ("world_gloo", tuple(range(WORLD_SIZE)), "gloo"),
)


def _parse_physical_devices(raw: str, world_size: int = WORLD_SIZE) -> tuple[str, ...]:
    devices = tuple(item.strip() for item in raw.split(",") if item.strip())
    if len(devices) != world_size:
        raise ValueError(f"physical-device mapping must contain {world_size} entries, got {len(devices)} from {raw!r}")
    if len(set(devices)) != len(devices):
        raise ValueError(f"physical-device mapping contains duplicates: {raw!r}")
    return devices


def _resolve_physical_devices(
    cli_value: str | None,
    environment_value: str | None,
    world_size: int = WORLD_SIZE,
) -> tuple[str, ...]:
    if cli_value is None and environment_value is None:
        raise ValueError(
            "pass --physical-devices or set ASCEND_RT_VISIBLE_DEVICES so the result records the physical rank order"
        )
    cli_devices = _parse_physical_devices(cli_value, world_size) if cli_value is not None else None
    environment_devices = (
        _parse_physical_devices(environment_value, world_size) if environment_value is not None else None
    )
    if cli_devices is not None and environment_devices is not None and cli_devices != environment_devices:
        raise ValueError(
            f"--physical-devices does not match ASCEND_RT_VISIBLE_DEVICES: {cli_devices!r} != {environment_devices!r}"
        )
    return cli_devices if cli_devices is not None else environment_devices  # type: ignore[return-value]


def _target_rank_for_global_rank(global_rank: int) -> int | None:
    if global_rank == COORDINATOR_GLOBAL_RANK:
        return None
    try:
        return TARGET_GLOBAL_RANKS.index(global_rank)
    except ValueError as error:
        raise ValueError(f"unexpected global rank {global_rank}") from error


def _physical_device_records(
    physical_devices: Sequence[str],
) -> list[dict[str, Any]]:
    if len(physical_devices) != WORLD_SIZE:
        raise ValueError(f"expected {WORLD_SIZE} physical devices")
    records: list[dict[str, Any]] = []
    for global_rank, physical_device in enumerate(physical_devices):
        target_rank = _target_rank_for_global_rank(global_rank)
        records.append(
            {
                "global_rank": global_rank,
                "local_rank": global_rank,
                "physical_device": str(physical_device),
                "role": "draft_coordinator" if target_rank is None else "target",
                "target_rank": target_rank,
                "input_rank": target_rank,
            }
        )
    return records


def _validate_layout(global_rank: int, local_rank: int, world_size: int) -> None:
    if world_size != WORLD_SIZE:
        raise ValueError(f"expected full world size {WORLD_SIZE}, got {world_size}")
    if global_rank < 0 or global_rank >= WORLD_SIZE:
        raise ValueError(f"global rank {global_rank} is outside the four-rank world")
    if local_rank != global_rank:
        raise ValueError(
            "this single-node probe requires LOCAL_RANK == global rank so its "
            f"physical-device record is unambiguous; got {local_rank} != {global_rank}"
        )


def _create_process_groups() -> dict[str, Any]:
    """Create all groups in one world-wide, deterministic call order."""
    groups: dict[str, Any] = {}
    for name, ranks, backend in GROUP_CREATION_PLAN:
        groups[name] = dist.new_group(ranks=list(ranks), backend=backend)
    return groups


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--physical-devices",
        help=(
            "comma-separated physical NPU IDs in LOCAL_RANK order; when "
            "ASCEND_RT_VISIBLE_DEVICES is set the two mappings must agree"
        ),
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=None,
        help="use only the first N captured activation rows",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--row-block-size", type=int, default=16)
    parser.add_argument("--tile-widths", type=int, nargs="+", default=(256, 512))
    parser.add_argument("--max-flat-ranges", type=int, default=20000)
    return parser


def _validate_args(args: argparse.Namespace) -> tuple[str, ...]:
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.row_limit is not None and args.row_limit < 1:
        raise ValueError("--row-limit must be positive")
    if args.row_block_size < 1:
        raise ValueError("--row-block-size must be positive")
    if any(width < 1 for width in args.tile_widths):
        raise ValueError("all --tile-widths must be positive")
    if args.max_flat_ranges < 0:
        raise ValueError("--max-flat-ranges must be non-negative")
    if os.environ.get("HCCL_DETERMINISTIC", "").strip().lower() != "true":
        raise RuntimeError("this probe requires HCCL_DETERMINISTIC=true")
    if os.environ.get("HCCL_OP_EXPANSION_MODE", "").strip() != "AIV":
        raise RuntimeError("this probe requires HCCL_OP_EXPANSION_MODE=AIV")
    return _resolve_physical_devices(
        args.physical_devices,
        os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _target_probe(
    args: argparse.Namespace,
    global_rank: int,
    target_hccl_group: Any,
    target_gloo_group: Any,
    physical_devices: Sequence[str],
) -> dict[str, Any] | None:
    target_rank = _target_rank_for_global_rank(global_rank)
    if target_rank is None:
        raise AssertionError("the draft coordinator cannot execute the target probe")
    actual_target_rank = dist.get_rank(target_hccl_group)
    if actual_target_rank != target_rank:
        raise RuntimeError(f"target subgroup rank mismatch: expected {target_rank}, got {actual_target_rank}")

    input_path = args.input_dir / f"rank-{target_rank}.pt"
    payload = torch.load(input_path, map_location="cpu", weights_only=True)
    source_activation = payload["activation"]
    source_activation_shape = list(source_activation.shape)
    if args.row_limit is not None:
        if args.row_limit > source_activation.shape[0]:
            raise ValueError(
                f"--row-limit={args.row_limit} exceeds captured rows {source_activation.shape[0]} in {input_path}"
            )
        source_activation = source_activation[: args.row_limit]
    activation = source_activation.to(device="npu", dtype=torch.bfloat16)
    weight = payload["weight"].to(device="npu", dtype=torch.bfloat16)
    projection = torch.nn.functional.linear(activation, weight)
    torch.npu.synchronize()
    if projection.dtype != torch.bfloat16 or projection.ndim != 2:
        raise TypeError(f"expected a 2-D BF16 projection, got shape={tuple(projection.shape)} dtype={projection.dtype}")

    gathered_cpu: list[torch.Tensor | None] = [None] * TARGET_WORLD_SIZE
    dist.all_gather_object(gathered_cpu, projection.cpu(), group=target_gloo_group)
    if any(value is None for value in gathered_cpu):
        raise RuntimeError("failed to gather all target-rank projections")
    projected_by_target_rank = [
        value.to(device="npu", dtype=torch.bfloat16).contiguous() for value in gathered_cpu if value is not None
    ]
    candidates_npu = make_candidates(projected_by_target_rank)
    torch.npu.synchronize()
    candidates_cpu = {name: value.cpu().contiguous() for name, value in candidates_npu.items()}

    local_input_metadata = {
        "rank": target_rank,
        "target_rank": target_rank,
        "global_rank": global_rank,
        "local_rank": global_rank,
        "physical_device": str(physical_devices[global_rank]),
        "input_rank": target_rank,
        "input_path": str(input_path.resolve()),
        "input_file_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "source_activation_shape": source_activation_shape,
        "row_limit": args.row_limit,
        "activation_shape": list(activation.shape),
        "weight_shape": list(weight.shape),
        "projection_shape": list(projection.shape),
        "activation_sha256": tensor_sha256(activation),
        "weight_sha256": tensor_sha256(weight),
        "projection_sha256": tensor_sha256(projection),
    }
    input_metadata: list[dict[str, Any] | None] = [None] * TARGET_WORLD_SIZE
    dist.all_gather_object(input_metadata, local_input_metadata, group=target_gloo_group)

    trial_records: list[dict[str, Any]] = []
    unique_target_rank0_outputs: dict[str, torch.Tensor] = {}
    for trial in range(args.repeats):
        dist.barrier(group=target_gloo_group)
        hccl_output = projection.clone()
        dist.all_reduce(
            hccl_output,
            op=dist.ReduceOp.SUM,
            group=target_hccl_group,
        )
        torch.npu.synchronize()

        output_cpu = hccl_output.cpu().contiguous()
        local_sha = tensor_sha256(output_cpu)
        target_rank_hashes: list[str | None] = [None] * TARGET_WORLD_SIZE
        dist.all_gather_object(target_rank_hashes, local_sha, group=target_gloo_group)
        if target_rank == 0:
            unique_target_rank0_outputs.setdefault(local_sha, output_cpu)
            trial_records.append(
                {
                    "trial": trial,
                    "rank_sha256": target_rank_hashes,
                    "target_rank_sha256": target_rank_hashes,
                    "rank_outputs_bitwise_identical": (len(set(target_rank_hashes)) == 1),
                }
            )

    if target_rank != 0:
        return None

    candidate_metadata = {
        name: {"sha256": tensor_sha256(value), "shape": list(value.shape)} for name, value in candidates_cpu.items()
    }
    analyses_by_output_sha256: dict[str, dict[str, Any]] = {}
    for sha, value in unique_target_rank0_outputs.items():
        _, signature = build_signature_map(value, candidates_cpu)
        analysis = summarize_output(
            actual=value,
            candidates=candidates_cpu,
            tile_widths=args.tile_widths,
            row_block_size=args.row_block_size,
            max_flat_ranges=args.max_flat_ranges,
        )
        analysis["element_signature_artifact"] = write_element_signature_artifact(
            signature=signature,
            output_json=args.output_json,
            output_sha256=sha,
        )
        analyses_by_output_sha256[sha] = analysis

    first_sha = trial_records[0]["rank_sha256"][0]
    rank0_repeats_identical = all(record["rank_sha256"][0] == first_sha for record in trial_records)
    return {
        "schema_version": 1,
        "probe": "deterministic_tp3_hccl_bf16_chunk_map",
        "topology": "full_world4_draft_rank0_target_subgroup_ranks1_2_3",
        "input_dir": str(args.input_dir.resolve()),
        "row_limit": args.row_limit,
        "physical_devices": ",".join(physical_devices),
        "physical_device_mapping": _physical_device_records(physical_devices),
        "hccl_deterministic": os.environ.get("HCCL_DETERMINISTIC"),
        "hccl_op_expansion_mode": os.environ.get("HCCL_OP_EXPANSION_MODE"),
        "world_size": WORLD_SIZE,
        "coordinator_global_rank": COORDINATOR_GLOBAL_RANK,
        "target_global_ranks": list(TARGET_GLOBAL_RANKS),
        "target_world_size": TARGET_WORLD_SIZE,
        "target_rank0_global_rank": TARGET_GLOBAL_RANKS[0],
        "group_creation_plan": [
            {"name": name, "ranks": list(ranks), "backend": backend} for name, ranks, backend in GROUP_CREATION_PLAN
        ],
        "repeats": args.repeats,
        "row_block_size": args.row_block_size,
        "tile_widths": list(args.tile_widths),
        "candidate_bit_legend": {str(1 << index): name for index, name in enumerate(CANDIDATE_NAMES)},
        "candidate_metadata": candidate_metadata,
        "input_metadata": input_metadata,
        "trials": trial_records,
        # Preserve the old standalone field while naming its subgroup meaning.
        "rank0_repeat_outputs_bitwise_identical": rank0_repeats_identical,
        "target_rank0_repeat_outputs_bitwise_identical": rank0_repeats_identical,
        "unique_rank0_output_count": len(unique_target_rank0_outputs),
        "unique_target_rank0_output_count": len(unique_target_rank0_outputs),
        "analyses_by_output_sha256": analyses_by_output_sha256,
    }


def main() -> None:
    args = _parser().parse_args()
    physical_devices = _validate_args(args)

    # Delayed import keeps mapping/report unit tests CPU-only.
    import torch_npu  # noqa: F401

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    _validate_layout(global_rank, local_rank, world_size)

    groups = _create_process_groups()
    report = None
    if global_rank in TARGET_GLOBAL_RANKS:
        report = _target_probe(
            args=args,
            global_rank=global_rank,
            target_hccl_group=groups["target_hccl"],
            target_gloo_group=groups["target_gloo"],
            physical_devices=physical_devices,
        )

    gathered_reports: list[dict[str, Any] | None] | None = None
    if global_rank == COORDINATOR_GLOBAL_RANK:
        gathered_reports = [None] * WORLD_SIZE
    dist.gather_object(
        report,
        object_gather_list=gathered_reports,
        dst=COORDINATOR_GLOBAL_RANK,
        group=groups["world_gloo"],
    )

    if global_rank == COORDINATOR_GLOBAL_RANK:
        assert gathered_reports is not None
        final_report = gathered_reports[TARGET_GLOBAL_RANKS[0]]
        if final_report is None:
            raise RuntimeError("target subgroup leader did not produce a report")
        _atomic_write_json(args.output_json, final_report)
        first_sha = final_report["trials"][0]["rank_sha256"][0]
        whole = final_report["analyses_by_output_sha256"][first_sha]["whole_tensor"]
        print(
            "HCCL_CHUNK_MAP_SUMMARY="
            + json.dumps(
                {
                    "output_json": str(args.output_json.resolve()),
                    "topology": final_report["topology"],
                    "physical_device_mapping": final_report["physical_device_mapping"],
                    "rank_outputs_identical_every_trial": all(
                        record["rank_outputs_bitwise_identical"] for record in final_report["trials"]
                    ),
                    "repeat_outputs_identical": final_report["target_rank0_repeat_outputs_bitwise_identical"],
                    "unique_output_count": final_report["unique_target_rank0_output_count"],
                    "whole_tensor": whole,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    dist.barrier(group=groups["world_gloo"])
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
