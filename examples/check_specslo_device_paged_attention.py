# SPDX-License-Identifier: Apache-2.0
"""Qualify device-position PagedAttention eager and ACLGraph numerics.

This is a single-NPU correctness gate, not a throughput benchmark.  The
default shape exercises GQA-2, crosses a physical KV page, and changes both
the query and the resident position tensor between ACLGraph replays::

    ASCEND_RT_VISIBLE_DEVICES=<one-free-device> \
      python examples/check_specslo_device_paged_attention.py \
      --output /path/device-paged-attention.json

Every NPU result is checked against an independent CPU float64
matmul-softmax-matmul reference.  Capture failure, non-finite output, or a
comparison outside tolerance exits non-zero and leaves a failure JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any

import torch


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--qheads", type=int, default=8)
    parser.add_argument("--kvheads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page", type=int, default=128)
    parser.add_argument("--tile", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--output", required=True)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    dimensions = (args.rows, args.qheads, args.kvheads, args.head_dim, args.page)
    if min(dimensions) <= 0:
        raise ValueError("rows, heads, head_dim, and page must be positive")
    if args.qheads % args.kvheads:
        raise ValueError("qheads must be divisible by kvheads")
    if args.qheads // args.kvheads not in (1, 2, 4):
        raise ValueError("the device PagedAttention kernel supports GQA groups 1, 2, or 4")
    if args.tile not in (16, 32, 64):
        raise ValueError("tile must be 16, 32, or 64")
    if args.page % args.tile:
        raise ValueError("tile must divide page")
    if args.head_dim & (args.head_dim - 1):
        raise ValueError("head_dim must be a power of two")
    if args.atol < 0 or args.rtol < 0 or not math.isfinite(args.atol + args.rtol):
        raise ValueError("comparison tolerances must be finite and non-negative")
    if not args.device.startswith("npu:"):
        raise ValueError("this qualification requires one explicit npu:<index> device")


def _position_pair(rows: int, page: int, tile: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return two different, cross-page, in-capacity position vectors."""

    offsets = torch.arange(rows, dtype=torch.int64) % max(1, min(tile, page - 1))
    initial = page + offsets
    changed = torch.minimum(initial + max(1, tile // 2), torch.full_like(initial, 2 * page - 1))
    if not bool((changed > initial).all()):
        raise ValueError("page/tile combination leaves no room for a changed cross-page position")
    return initial, changed


def _build_inputs(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    initial_positions, changed_positions = _position_pair(args.rows, args.page, args.tile)
    logical_pages = math.ceil((int(changed_positions.max()) + 1) / args.page)
    physical_pages = args.rows * logical_pages

    # Quantize on CPU first so eager, graph, and the CPU oracle consume the
    # exact same BF16 payload rather than independently rounded random values.
    query = (torch.randn(args.rows, args.qheads, args.head_dim, generator=generator) * 0.25).to(torch.bfloat16)
    changed_query = (query.float() + torch.randn(query.shape, generator=generator) * 0.0625).to(torch.bfloat16)
    key_cache = (
        torch.randn(
            physical_pages,
            args.page,
            args.kvheads,
            args.head_dim,
            generator=generator,
        )
        * 0.25
    ).to(torch.bfloat16)
    value_cache = (
        torch.randn(
            physical_pages,
            args.page,
            args.kvheads,
            args.head_dim,
            generator=generator,
        )
        * 0.25
    ).to(torch.bfloat16)
    block_table = torch.arange(physical_pages, dtype=torch.int32).reshape(args.rows, logical_pages)
    # A non-identity per-row logical-to-physical mapping catches references
    # that accidentally treat logical and physical page IDs as identical.
    block_table = torch.flip(block_table, dims=(1,)).contiguous()
    return {
        "query": query,
        "changed_query": changed_query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_table": block_table,
        "positions": initial_positions,
        "changed_positions": changed_positions,
    }


def _cpu_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    positions: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Independent CPU float64 paged GQA attention reference."""

    rows, query_heads, head_dim = query.shape
    kv_heads = key_cache.shape[2]
    if query_heads % kv_heads:
        raise ValueError("reference requires an integral GQA head group")
    group = query_heads // kv_heads
    outputs: list[torch.Tensor] = []
    for row in range(rows):
        length = int(positions[row]) + 1
        pages_needed = math.ceil(length / key_cache.shape[1])
        if pages_needed > block_table.shape[1]:
            raise ValueError("position exceeds the supplied page table capacity")
        page_ids = block_table[row, :pages_needed].to(torch.int64)
        keys = key_cache.index_select(0, page_ids).reshape(-1, kv_heads, head_dim)[:length]
        values = value_cache.index_select(0, page_ids).reshape(-1, kv_heads, head_dim)[:length]
        keys = keys.double().repeat_interleave(group, dim=1).permute(1, 0, 2)
        values = values.double().repeat_interleave(group, dim=1).permute(1, 0, 2)
        scores = (query[row].double().unsqueeze(1) * keys).sum(dim=2) * scale
        probabilities = scores.softmax(dim=1)
        outputs.append((probabilities.unsqueeze(2) * values).sum(dim=1))
    return torch.stack(outputs)


def _comparison(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    actual = actual.detach().double().cpu()
    reference = reference.detach().double().cpu()
    if actual.shape != reference.shape:
        return {
            "finite": False,
            "within_tolerance": False,
            "actual_shape": list(actual.shape),
            "reference_shape": list(reference.shape),
            "reason": "shape_mismatch",
        }
    finite_actual = torch.isfinite(actual)
    finite_reference = torch.isfinite(reference)
    finite = bool(finite_actual.all() and finite_reference.all())
    if not finite:
        return {
            "finite": False,
            "within_tolerance": False,
            "nonfinite_actual": int((~finite_actual).sum()),
            "nonfinite_reference": int((~finite_reference).sum()),
        }
    error = (actual - reference).abs()
    allowance = atol + rtol * reference.abs()
    return {
        "finite": True,
        "exact_equal": bool(torch.equal(actual, reference)),
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
        "rmse": float(error.square().mean().sqrt()),
        "outside_tolerance_count": int((error > allowance).sum()),
        "within_tolerance": bool((error <= allowance).all()),
    }


def _run_npu(args: argparse.Namespace, inputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    import torch_npu

    from vllm_ascend.ops.triton.spec_decode.device_paged_attention import device_paged_attention

    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    scale = 1.0 / math.sqrt(args.head_dim)
    tensors = {name: value.to(args.device) for name, value in inputs.items()}
    output = torch.empty_like(tensors["query"])

    def invoke() -> torch.Tensor:
        return device_paged_attention(
            tensors["query"],
            tensors["key_cache"],
            tensors["value_cache"],
            tensors["block_table"],
            tensors["positions"],
            scale=scale,
            output=output,
            tokens_per_iteration=args.tile,
            lengths_are_positions=True,
        )

    # Compile the Triton kernel and obtain the independent eager result before
    # graph capture.  Reuse the same output pointer to match production replay.
    with torch.inference_mode():
        invoke()
        torch.npu.synchronize()
        eager = output.cpu().clone()

        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            invoke()
        torch.npu.synchronize()

        graph.replay()
        torch.npu.synchronize()
        initial_replay = output.cpu().clone()

        # Change only the resident position first.  A simultaneous query
        # change can make the output differ even when a captured graph has
        # accidentally frozen its loop bound, so this isolated replay is the
        # explicit proof that the graph reads dynamic positions.
        tensors["positions"].copy_(tensors["changed_positions"])
        graph.replay()
        torch.npu.synchronize()
        position_only_replay = output.cpu().clone()

        tensors["query"].copy_(tensors["changed_query"])
        graph.replay()
        torch.npu.synchronize()
        changed_replay = output.cpu().clone()

    initial_reference = _cpu_reference(
        inputs["query"],
        inputs["key_cache"],
        inputs["value_cache"],
        inputs["block_table"],
        inputs["positions"],
        scale=scale,
    )
    changed_reference = _cpu_reference(
        inputs["changed_query"],
        inputs["key_cache"],
        inputs["value_cache"],
        inputs["block_table"],
        inputs["changed_positions"],
        scale=scale,
    )
    compare = {"atol": args.atol, "rtol": args.rtol}
    comparisons = {
        "eager_vs_cpu_reference": _comparison(eager, initial_reference, **compare),
        "initial_aclgraph_replay_vs_cpu_reference": _comparison(initial_replay, initial_reference, **compare),
        "initial_aclgraph_replay_vs_eager": _comparison(initial_replay, eager, **compare),
        "position_only_aclgraph_replay_vs_cpu_reference": _comparison(
            position_only_replay,
            _cpu_reference(
                inputs["query"],
                inputs["key_cache"],
                inputs["value_cache"],
                inputs["block_table"],
                inputs["changed_positions"],
                scale=scale,
            ),
            **compare,
        ),
        "changed_aclgraph_replay_vs_cpu_reference": _comparison(changed_replay, changed_reference, **compare),
    }
    position_effect = _comparison(position_only_replay, initial_replay, atol=0.0, rtol=0.0)
    position_effect["observed"] = bool(position_effect.get("finite")) and not bool(
        position_effect.get("exact_equal", True)
    )
    changed_effect = _comparison(changed_replay, initial_replay, atol=0.0, rtol=0.0)
    changed_effect["observed"] = bool(changed_effect.get("finite")) and not bool(
        changed_effect.get("exact_equal", True)
    )
    return {
        "torch_npu_version": torch_npu.__version__,
        "device_name": torch.npu.get_device_name(args.device),
        "scale": scale,
        "graph_capture_count": 1,
        "graph_replay_count": 3,
        "comparisons": comparisons,
        "position_change_effect": position_effect,
        "changed_input_effect": changed_effect,
    }


def _config(args: argparse.Namespace) -> dict[str, Any]:
    values = vars(args).copy()
    values["output"] = str(values["output"])
    return values


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "purpose": "Device-position PagedAttention eager/ACLGraph numerical qualification; no throughput claim",
        "config": _config(args),
        "torch_version": torch.__version__,
        "reference": "Identical BF16 inputs with independent CPU float64 paged GQA matmul-softmax-matmul",
        "status": "running",
    }

    def save() -> None:
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    try:
        _validate_args(args)
        inputs = _build_inputs(args)
        report["input_contract"] = {
            "gqa_group": args.qheads // args.kvheads,
            "logical_pages": int(inputs["block_table"].shape[1]),
            "physical_pages": int(inputs["key_cache"].shape[0]),
            "initial_positions": inputs["positions"].tolist(),
            "changed_positions": inputs["changed_positions"].tolist(),
            "initial_crosses_page": bool((inputs["positions"] >= args.page).all()),
            "positions_change_in_place": True,
            "query_changes_in_place": True,
            "shapes": {name: list(tensor.shape) for name, tensor in inputs.items()},
            "dtypes": {name: str(tensor.dtype).removeprefix("torch.") for name, tensor in inputs.items()},
        }
        report.update(_run_npu(args, inputs))
        failed = [
            name
            for name, comparison in report["comparisons"].items()
            if not comparison["finite"] or not comparison["within_tolerance"]
        ]
        if failed:
            raise AssertionError(f"device PagedAttention numerical comparisons failed: {failed}")
        if not report["position_change_effect"]["observed"]:
            raise AssertionError("changed position replay did not change the graph output")
        if not report["changed_input_effect"]["observed"]:
            raise AssertionError("changed query/position replay did not change the graph output")
        report["status"] = "passed"
        save()
        return 0
    except Exception as error:
        report.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
