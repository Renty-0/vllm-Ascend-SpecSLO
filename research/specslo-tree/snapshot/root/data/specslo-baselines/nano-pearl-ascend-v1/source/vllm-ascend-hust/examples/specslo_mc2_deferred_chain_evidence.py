# SPDX-License-Identifier: Apache-2.0
"""Validate and combine full-model deferred MC2 chain reports.

The custom TP3 epilogue may leave its READ_DONE tail outstanding only when
that exact flattened row count has survived a 128-op (64 attention/down
pairs) ACLGraph qualification.  This module is deliberately CPU-only so both
the legacy two-shape profile builder and the conservative multi-shape
aggregator use the same fail-closed evidence contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

_CHAIN_MODES = ("fully_flushed", "deferred_chain")


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _canonical_source(value: object, *, field: str, report: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{report}: deferred chain {field} is missing")
    return str(Path(value).expanduser().resolve())


def _positive_samples(
    samples: Mapping[str, Any],
    field: str,
    *,
    minimum_samples: int,
    report: Path,
) -> list[float]:
    raw = samples.get(field)
    if not isinstance(raw, list) or len(raw) < minimum_samples:
        raise ValueError(
            f"{report}: deferred chain {field} has fewer than "
            f"{minimum_samples} samples"
        )
    values = [float(value) for value in raw]
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(f"{report}: deferred chain {field} contains invalid samples")
    return values


def _load_one_report(
    path: Path,
    *,
    expected_source_sha256: str,
    expected_mapping: tuple[str, ...],
    minimum_samples: int,
    minimum_speedup: float,
    allowed_input_sources: frozenset[str] | None,
    allowed_row_counts: frozenset[int] | None,
) -> dict[str, Any]:
    report = path.expanduser().resolve(strict=True)
    payload_bytes = report.read_bytes()
    payload = json.loads(payload_bytes)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{report}: deferred chain report must be an object")
    if payload.get("schema_version") != 1 or payload.get("status") != "measured":
        raise ValueError(
            f"{report}: expected a measured schema_version=1 chain document"
        )
    if payload.get("execution_mode") != "graph" or payload.get(
        "tensor_parallel_size"
    ) != 3:
        raise ValueError(
            f"{report}: deferred chain production admission requires graph TP3"
        )
    if payload.get("chain_scope") != "whole_model":
        raise ValueError(f"{report}: deferred chain does not cover a whole model")
    source_sha256 = payload.get("source_sha256")
    if source_sha256 != expected_source_sha256:
        raise ValueError(
            f"{report}: deferred chain source SHA does not match the profile source"
        )
    row_count = payload.get("row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        raise ValueError(
            f"{report}: deferred chain report lacks a positive explicit row_count"
        )
    if allowed_row_counts is not None and row_count not in allowed_row_counts:
        raise ValueError(
            f"{report}: deferred chain row_count {row_count} is absent from the profile"
        )
    layer_count = payload.get("layer_count")
    operations_per_chain = payload.get("operations_per_chain")
    if (
        isinstance(layer_count, bool)
        or not isinstance(layer_count, int)
        or layer_count <= 0
        or operations_per_chain != 2 * layer_count
    ):
        raise ValueError(f"{report}: deferred chain has an invalid model-length operation count")
    mapping = tuple(str(value) for value in payload.get("tp_rank_device_mapping", ()))
    if mapping != expected_mapping:
        raise ValueError(f"{report}: deferred chain TP rank mapping differs from qualification")

    attention_source = _canonical_source(
        payload.get("attention_input_source"),
        field="attention_input_source",
        report=report,
    )
    down_source = _canonical_source(
        payload.get("down_input_source"),
        field="down_input_source",
        report=report,
    )
    if allowed_input_sources is not None:
        missing = [
            source
            for source in (attention_source, down_source)
            if source not in allowed_input_sources
        ]
        if missing:
            raise ValueError(
                f"{report}: deferred chain input source is absent from profile provenance: "
                f"{missing[0]}"
            )

    correctness = payload.get("correctness")
    samples = payload.get("samples")
    summary = payload.get("summary")
    if not all(isinstance(value, Mapping) for value in (correctness, samples, summary)):
        raise ValueError(f"{report}: incomplete deferred chain qualification")
    initial = correctness.get("initial")
    changed_exact = correctness.get("changed_input_exact_all_ranks")
    changed_max = correctness.get("changed_input_max_abs")
    if not all(
        isinstance(value, Mapping) for value in (initial, changed_exact, changed_max)
    ):
        raise ValueError(f"{report}: incomplete deferred chain correctness evidence")
    for mode in _CHAIN_MODES:
        mode_initial = initial.get(mode)
        mode_changed_max = changed_max.get(mode)
        if (
            not isinstance(mode_initial, Mapping)
            or mode_initial.get("exact_all_ranks") is not True
        ):
            raise ValueError(
                f"{report}: {mode} initial result is not exact on every rank"
            )
        initial_max = mode_initial.get("max_abs")
        if not isinstance(initial_max, Mapping) or not initial_max or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) != 0.0
            for value in initial_max.values()
        ):
            raise ValueError(f"{report}: {mode} initial result is not bit-exact")
        if changed_exact.get(mode) is not True:
            raise ValueError(
                f"{report}: {mode} changed-input result is not exact on every rank"
            )
        if not isinstance(mode_changed_max, Mapping) or not mode_changed_max or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) != 0.0
            for value in mode_changed_max.values()
        ):
            raise ValueError(
                f"{report}: {mode} changed-input result is not bit-exact"
            )
    if correctness.get("final_chain_state_zero_all_ranks") is not True:
        raise ValueError(f"{report}: deferred chain did not drain its mailbox state")
    changed_replays = correctness.get("changed_input_replays")
    if (
        isinstance(changed_replays, bool)
        or not isinstance(changed_replays, int)
        or changed_replays < 2
    ):
        raise ValueError(
            f"{report}: deferred chain has insufficient changed-input replays"
        )
    changed_output_delta = correctness.get("changed_input_output_delta")
    if (
        isinstance(changed_output_delta, bool)
        or not isinstance(changed_output_delta, (int, float))
        or not math.isfinite(float(changed_output_delta))
        or float(changed_output_delta) <= 0.0
    ):
        raise ValueError(
            f"{report}: deferred chain lacks a nonzero changed-input output delta"
        )

    fully_flushed = _positive_samples(
        samples,
        "fully_flushed_latency_ms",
        minimum_samples=minimum_samples,
        report=report,
    )
    deferred = _positive_samples(
        samples,
        "deferred_chain_latency_ms",
        minimum_samples=minimum_samples,
        report=report,
    )
    fully_flushed_p50 = _percentile(fully_flushed, 50.0)
    fully_flushed_p95 = _percentile(fully_flushed, 95.0)
    deferred_p50 = _percentile(deferred, 50.0)
    deferred_p95 = _percentile(deferred, 95.0)
    p50_speedup = fully_flushed_p50 / deferred_p50
    p95_speedup = fully_flushed_p95 / deferred_p95
    if p95_speedup < minimum_speedup:
        raise ValueError(
            f"{report}: deferred chain p95 speedup {p95_speedup:.6f} is below "
            f"the required {minimum_speedup:.6f}"
        )
    return {
        "row_count": row_count,
        "report_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "report_path": str(report),
        "attention_input_source": attention_source,
        "down_input_source": down_source,
        "fully_flushed_latency_ms": fully_flushed,
        "deferred_chain_latency_ms": deferred,
        "p50_speedup": p50_speedup,
        "p95_speedup": p95_speedup,
        "layer_count": layer_count,
        "operations_per_chain": operations_per_chain,
    }


def load_deferred_chain_evidence(
    paths: Iterable[Path],
    *,
    expected_source_sha256: str,
    expected_mapping: Sequence[str],
    minimum_samples: int,
    minimum_speedup: float,
    allowed_input_sources: Iterable[str | Path] | None = None,
    allowed_row_counts: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Return canonical row-scoped evidence from one or more graph reports."""

    if minimum_samples < 3:
        raise ValueError("deferred chain minimum_samples must be at least three")
    if not math.isfinite(minimum_speedup) or minimum_speedup <= 0.0:
        raise ValueError("deferred chain minimum_speedup must be finite and positive")
    report_paths = tuple(Path(path) for path in paths)
    if not report_paths:
        raise ValueError("at least one deferred chain report is required")
    resolved_paths = tuple(path.expanduser().resolve(strict=True) for path in report_paths)
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("deferred chain report paths must be unique")
    normalized_mapping = tuple(str(value) for value in expected_mapping)
    normalized_sources = (
        frozenset(str(Path(value).expanduser().resolve()) for value in allowed_input_sources)
        if allowed_input_sources is not None
        else None
    )
    normalized_rows = (
        frozenset(int(value) for value in allowed_row_counts)
        if allowed_row_counts is not None
        else None
    )
    reports = [
        _load_one_report(
            path,
            expected_source_sha256=expected_source_sha256,
            expected_mapping=normalized_mapping,
            minimum_samples=minimum_samples,
            minimum_speedup=minimum_speedup,
            allowed_input_sources=normalized_sources,
            allowed_row_counts=normalized_rows,
        )
        for path in resolved_paths
    ]
    row_counts = [int(report["row_count"]) for report in reports]
    if len(set(row_counts)) != len(row_counts):
        raise ValueError("deferred chain reports duplicate an explicit row_count")
    layer_counts = {int(report["layer_count"]) for report in reports}
    operation_counts = {int(report["operations_per_chain"]) for report in reports}
    if len(layer_counts) != 1 or len(operation_counts) != 1:
        raise ValueError("deferred chain reports disagree on the full-model topology")
    reports.sort(key=lambda report: int(report["row_count"]))
    layer_count = next(iter(layer_counts))
    operations_per_chain = next(iter(operation_counts))
    return {
        "qualified": True,
        "execution_mode": "graph",
        "chain_scope": "whole_model",
        "layer_count": layer_count,
        "operations_per_chain": operations_per_chain,
        "source_sha256": expected_source_sha256,
        "qualified_row_counts": sorted(row_counts),
        "minimum_samples": minimum_samples,
        "minimum_speedup": minimum_speedup,
        "reports": reports,
    }
