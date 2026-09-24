# SPDX-License-Identifier: Apache-2.0
"""Reusable reporting helpers for deterministic TP3 BF16 HCCL probes.

The helpers in this module are device agnostic.  In particular, they can be
unit-tested with CPU tensors even though the probe which produces ``actual``
uses HCCL on Ascend NPUs.  The candidate bit layout and the lossless signature
artifact intentionally match the original standalone HCCL chunk-map probe.
"""

from __future__ import annotations

import gzip
import hashlib
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch

CANDIDATE_NAMES = (
    "tree_01_then_2",
    "tree_02_then_1",
    "tree_12_then_0",
    "fp32_sum_then_bf16",
)
TREE_CANDIDATE_COUNT = 3


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def signature_name(signature: int) -> str:
    if signature == 0:
        return "unmatched"
    return "|".join(name for index, name in enumerate(CANDIDATE_NAMES) if signature & (1 << index))


def make_candidates(
    projected_by_target_rank: Sequence[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Build the three BF16 trees and the FP32-accumulation reference."""
    if len(projected_by_target_rank) != 3:
        raise ValueError(f"expected three target-rank projections, got {len(projected_by_target_rank)}")
    p0, p1, p2 = projected_by_target_rank
    if any(value.dtype != torch.bfloat16 for value in (p0, p1, p2)):
        raise TypeError("all target-rank projections must use torch.bfloat16")
    if p0.shape != p1.shape or p0.shape != p2.shape:
        raise ValueError("all target-rank projections must have the same shape")

    def bf16_tree(
        first: torch.Tensor,
        second: torch.Tensor,
        third: torch.Tensor,
    ) -> torch.Tensor:
        # Materializing the intermediate tensor preserves the first BF16
        # rounding point instead of allowing a compiler to reassociate it.
        intermediate = torch.add(first, second)
        return torch.add(intermediate, third)

    return {
        "tree_01_then_2": bf16_tree(p0, p1, p2),
        "tree_02_then_1": bf16_tree(p0, p2, p1),
        "tree_12_then_0": bf16_tree(p1, p2, p0),
        "fp32_sum_then_bf16": (p0.float().add(p1.float()).add(p2.float()).to(torch.bfloat16)),
    }


def build_signature_map(
    actual: torch.Tensor,
    candidates: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    missing = [name for name in CANDIDATE_NAMES if name not in candidates]
    if missing:
        raise ValueError(f"candidate map is missing {missing}")
    membership = torch.stack([actual.eq(candidates[name]) for name in CANDIDATE_NAMES], dim=0)
    signature = torch.zeros_like(actual, dtype=torch.uint8)
    for index in range(len(CANDIDATE_NAMES)):
        signature.bitwise_or_(membership[index].to(torch.uint8) << index)
    return membership, signature


def region_stats(membership: torch.Tensor, signature: torch.Tensor) -> dict[str, Any]:
    element_count = int(signature.numel())
    candidate_match_counts = {name: int(membership[index].sum().item()) for index, name in enumerate(CANDIDATE_NAMES)}
    candidate_mismatch_counts = {name: element_count - candidate_match_counts[name] for name in CANDIDATE_NAMES}
    signature_counts_raw = Counter(int(value) for value in signature.reshape(-1).tolist())
    signature_counts = {signature_name(key): value for key, value in sorted(signature_counts_raw.items())}

    tree_mismatches = [candidate_mismatch_counts[name] for name in CANDIDATE_NAMES[:TREE_CANDIDATE_COUNT]]
    best_tree_mismatch = min(tree_mismatches)
    best_trees = [
        CANDIDATE_NAMES[index] for index, mismatch in enumerate(tree_mismatches) if mismatch == best_tree_mismatch
    ]
    exact_trees = [CANDIDATE_NAMES[index] for index, mismatch in enumerate(tree_mismatches) if mismatch == 0]
    if exact_trees:
        tree_label = "exact:" + "|".join(exact_trees)
    else:
        tree_label = "mixed(best=" + "|".join(best_trees) + ")"

    return {
        "element_count": element_count,
        "candidate_match_counts": candidate_match_counts,
        "candidate_mismatch_counts": candidate_mismatch_counts,
        "signature_counts": signature_counts,
        "unmatched_count": signature_counts_raw.get(0, 0),
        "exact_candidates": [name for name, mismatch in candidate_mismatch_counts.items() if mismatch == 0],
        "exact_trees": exact_trees,
        "best_trees": best_trees,
        "best_tree_mismatch_count": best_tree_mismatch,
        "tree_label": tree_label,
    }


def contiguous_ranges(values: Sequence[Any]) -> list[dict[str, Any]]:
    if not values:
        return []
    ranges: list[dict[str, Any]] = []
    start = 0
    previous = values[0]
    for index in range(1, len(values)):
        if values[index] != previous:
            ranges.append({"start": start, "end_exclusive": index, "value": previous})
            start = index
            previous = values[index]
    ranges.append({"start": start, "end_exclusive": len(values), "value": previous})
    return ranges


def limited_flat_signature_ranges(
    signature: torch.Tensor,
    max_ranges: int,
) -> dict[str, Any]:
    flat = signature.reshape(-1).tolist()
    all_ranges = contiguous_ranges(flat)
    emitted = all_ranges[:max_ranges]
    width = signature.shape[1]
    for item in emitted:
        start = int(item["start"])
        end = int(item["end_exclusive"])
        item["signature"] = int(item.pop("value"))
        item["signature_name"] = signature_name(item["signature"])
        item["start_coordinate"] = list(divmod(start, width))
        item["end_exclusive_coordinate"] = list(divmod(end, width))
    return {
        "total_range_count": len(all_ranges),
        "emitted_range_count": len(emitted),
        "truncated": len(emitted) != len(all_ranges),
        "ranges": emitted,
    }


def summarize_output(
    actual: torch.Tensor,
    candidates: dict[str, torch.Tensor],
    tile_widths: Iterable[int],
    row_block_size: int,
    max_flat_ranges: int,
) -> dict[str, Any]:
    """Produce the region summaries used by the original chunk-map report."""
    membership, signature = build_signature_map(actual, candidates)
    rows, columns = actual.shape
    tile_widths = tuple(tile_widths)

    row_stats: list[dict[str, Any]] = []
    for row in range(rows):
        stats = region_stats(membership[:, row : row + 1, :], signature[row : row + 1, :])
        stats["row"] = row
        row_stats.append(stats)

    row_block_stats: list[dict[str, Any]] = []
    for row_start in range(0, rows, row_block_size):
        row_end = min(row_start + row_block_size, rows)
        stats = region_stats(
            membership[:, row_start:row_end, :],
            signature[row_start:row_end, :],
        )
        stats.update({"row_start": row_start, "row_end_exclusive": row_end})
        row_block_stats.append(stats)

    column_tiles: dict[str, list[dict[str, Any]]] = {}
    row_block_by_column_tiles: dict[str, list[dict[str, Any]]] = {}
    row_block_tile_label_ranges: dict[str, list[dict[str, Any]]] = {}
    for tile_width in tile_widths:
        width_key = str(tile_width)
        global_tiles: list[dict[str, Any]] = []
        for column_start in range(0, columns, tile_width):
            column_end = min(column_start + tile_width, columns)
            stats = region_stats(
                membership[:, :, column_start:column_end],
                signature[:, column_start:column_end],
            )
            stats.update(
                {
                    "column_start": column_start,
                    "column_end_exclusive": column_end,
                }
            )
            global_tiles.append(stats)
        column_tiles[width_key] = global_tiles

        two_dimensional_tiles: list[dict[str, Any]] = []
        block_ranges: list[dict[str, Any]] = []
        for row_start in range(0, rows, row_block_size):
            row_end = min(row_start + row_block_size, rows)
            labels: list[str] = []
            for column_start in range(0, columns, tile_width):
                column_end = min(column_start + tile_width, columns)
                stats = region_stats(
                    membership[:, row_start:row_end, column_start:column_end],
                    signature[row_start:row_end, column_start:column_end],
                )
                stats.update(
                    {
                        "row_start": row_start,
                        "row_end_exclusive": row_end,
                        "column_start": column_start,
                        "column_end_exclusive": column_end,
                    }
                )
                two_dimensional_tiles.append(stats)
                labels.append(stats["tree_label"])

            ranges = contiguous_ranges(labels)
            for item in ranges:
                tile_start = int(item.pop("start"))
                tile_end = int(item.pop("end_exclusive"))
                item["tree_label"] = item.pop("value")
                item.update(
                    {
                        "row_start": row_start,
                        "row_end_exclusive": row_end,
                        "tile_start": tile_start,
                        "tile_end_exclusive": tile_end,
                        "column_start": tile_start * tile_width,
                        "column_end_exclusive": min(tile_end * tile_width, columns),
                    }
                )
            block_ranges.append(
                {
                    "row_start": row_start,
                    "row_end_exclusive": row_end,
                    "ranges": ranges,
                }
            )
        row_block_by_column_tiles[width_key] = two_dimensional_tiles
        row_block_tile_label_ranges[width_key] = block_ranges

    row_label_ranges = contiguous_ranges([stats["tree_label"] for stats in row_stats])
    for item in row_label_ranges:
        item["row_start"] = item.pop("start")
        item["row_end_exclusive"] = item.pop("end_exclusive")
        item["tree_label"] = item.pop("value")

    return {
        "shape": [rows, columns],
        "whole_tensor": region_stats(membership, signature),
        "rows": row_stats,
        "row_label_ranges": row_label_ranges,
        # Keep the original key for consumers of the standalone probe and add
        # an explicit size-independent alias for new callers.
        "row_blocks_16": row_block_stats,
        "row_blocks": row_block_stats,
        "column_tiles": column_tiles,
        "row_blocks_16_by_column_tiles": row_block_by_column_tiles,
        "row_blocks_by_column_tiles": row_block_by_column_tiles,
        "row_block_tile_label_ranges": row_block_tile_label_ranges,
        "flat_element_signature_ranges": limited_flat_signature_ranges(signature, max_flat_ranges),
    }


def write_element_signature_artifact(
    signature: torch.Tensor,
    output_json: Path,
    output_sha256: str,
) -> dict[str, Any]:
    """Persist the complete row-major per-element classification losslessly."""
    raw = signature.detach().cpu().contiguous().numpy().tobytes()
    artifact = output_json.parent / (f"{output_json.stem}.{output_sha256[:16]}.signature-u8.bin.gz")
    temporary = artifact.with_suffix(artifact.suffix + ".tmp")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(temporary, "wb", compresslevel=9) as stream:
        stream.write(raw)
    temporary.replace(artifact)
    return {
        "path": str(artifact.resolve()),
        "encoding": "gzip-compressed row-major uint8 bitmask",
        "shape": list(signature.shape),
        "raw_byte_count": len(raw),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
