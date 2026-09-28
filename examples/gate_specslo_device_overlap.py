#!/usr/bin/env python3
"""Fail-closed, CPU-only gate for SpecSLO cross-NPU device overlap.

The input traces must have been exported by the official CANN/torch_npu
profiler tooling.  This program never opens an NPU context.  It proves overlap
only from ``Ascend Hardware`` kernel intervals inside decode-only
``ProfilerStep`` windows; host submission spans are deliberately ignored.

Example::

    python examples/gate_specslo_device_overlap.py \
      --selection-json parsed/selected-traces.json \
      --result-json run/result.json \
      --output device-overlap-gate.json

Exit status is zero only when every fail-closed gate passes.  A JSON report is
also written on validation failure so automated experiments can retain the
reason instead of silently losing the evidence.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from collections.abc import Iterable
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
EXPECTED_RANKS = (0, 1, 2, 3)
TARGET_RANKS = (1, 2, 3)
COMPUTE_TASK_TYPES = frozenset({"KERNEL_AICORE", "KERNEL_MIX_AIC", "KERNEL_AIVEC", "KERNEL_MIX_AIV"})
CUBE_TASK_TYPES = frozenset({"KERNEL_AICORE", "KERNEL_MIX_AIC"})
HCCL_RUNTIME_NAME_FRAGMENTS = ("hccl", "hcom")
COLLECTIVE_NAME_FRAGMENTS = (
    "allreduce",
    "allgather",
    "reducescatter",
    "broadcast",
)


class GateError(RuntimeError):
    """Raised when required evidence is absent or internally inconsistent."""


def _ns(value: Any) -> int:
    try:
        decimal = Decimal(str(value))
        if not decimal.is_finite():
            raise GateError(f"non-finite microsecond timestamp/duration: {value!r}")
        return int(decimal * 1000)
    except (InvalidOperation, OverflowError, TypeError, ValueError) as exc:
        raise GateError(f"invalid microsecond timestamp/duration: {value!r}") from exc


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[list[int]]:
    """Return a sorted union; overlapping streams are never double-counted."""
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def intersect_intervals(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    """Intersect two sorted, internally disjoint interval unions."""
    output: list[list[int]] = []
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        start = max(left[left_index][0], right[right_index][0])
        end = min(left[left_index][1], right[right_index][1])
        if start < end:
            output.append([start, end])
        if left[left_index][1] < right[right_index][1]:
            left_index += 1
        else:
            right_index += 1
    return output


def interval_duration(intervals: Iterable[list[int]]) -> int:
    return sum(end - start for start, end in intervals)


def _normalized_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _is_hccl_alias(name: str) -> bool:
    normalized = _normalized_name(name)
    # HCCL AIV expansion emits a second hardware row named exactly AivKernel.
    # Counting it would reintroduce communication after hcom_* was excluded.
    if normalized == "aivkernel" or any(fragment in normalized for fragment in HCCL_RUNTIME_NAME_FRAGMENTS):
        return True
    # A production MC2 kernel legitimately contains both ``MatMul`` and
    # ``AllReduce`` in its name. It performs real cube work and must not be
    # mistaken for the standalone HCCL communication alias.
    return "matmul" not in normalized and any(fragment in normalized for fragment in COLLECTIVE_NAME_FRAGMENTS)


def _is_matmul_name(name: str) -> bool:
    return "matmul" in _normalized_name(name)


def _is_matmul_event(event: dict[str, Any]) -> bool:
    """Recognize a real cube/MIX_AIC MatMul hardware kernel, not a host label."""
    return event["task_type"] in CUBE_TASK_TYPES and _is_matmul_name(event["name"])


def _resolve_path(base: Path, raw_path: Any, field: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise GateError(f"trace manifest has invalid {field!r}: {raw_path!r}")
    path = Path(raw_path)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _load_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GateError(f"missing {description}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GateError(f"invalid JSON in {description}: {path}: {exc}") from exc


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise GateError(f"{field} must be a positive integer")
    if isinstance(value, str) and not value.isdecimal():
        raise GateError(f"{field} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise GateError(f"{field} must be a positive integer")
    return result


def _extract_visible_cards(document: dict[str, Any]) -> list[int]:
    environment = document.get("runtime_environment")
    if not isinstance(environment, dict):
        raise GateError("result.json is missing runtime_environment")
    raw = environment.get("ASCEND_RT_VISIBLE_DEVICES")
    if not isinstance(raw, str):
        raise GateError("result.json is missing ASCEND_RT_VISIBLE_DEVICES")
    try:
        cards = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise GateError(f"invalid ASCEND_RT_VISIBLE_DEVICES: {raw!r}") from exc
    if len(cards) != 4 or len(set(cards)) != 4:
        raise GateError("ASCEND_RT_VISIBLE_DEVICES must identify exactly four unique cards")
    return cards


def load_result_contract(path: Path) -> dict[str, Any]:
    document = _load_json(path, "benchmark result")
    if not isinstance(document, dict):
        raise GateError("benchmark result must be a JSON object")
    draft_tp = document.get("draft_tensor_parallel_size")
    target_tp = document.get("target_tensor_parallel_size")
    if type(draft_tp) is not int or draft_tp != 1:
        raise GateError("overlap gate requires draft_tensor_parallel_size=1")
    if type(target_tp) is not int or target_tp != 3:
        raise GateError("overlap gate requires target_tensor_parallel_size=3")
    results = document.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise GateError("benchmark result must contain exactly one measured result")
    if not isinstance(results[0], dict):
        raise GateError("the measured benchmark result must be a JSON object")
    profile = results[0].get("decode_host_profile")
    cycles = profile.get("cycles") if isinstance(profile, dict) else None
    if not isinstance(cycles, list) or not cycles:
        raise GateError("benchmark result has no decode_host_profile cycles")
    by_step: dict[int, dict[str, Any]] = {}
    for row in cycles:
        if not isinstance(row, dict) or "step" not in row:
            raise GateError("decode_host_profile contains a malformed cycle")
        try:
            step = int(row["step"])
        except (TypeError, ValueError) as exc:
            raise GateError(f"invalid decode cycle step: {row.get('step')!r}") from exc
        if step in by_step:
            raise GateError(f"duplicate decode cycle step {step}")
        by_step[step] = row
    return {
        "path": str(path.resolve()),
        "visible_cards": _extract_visible_cards(document),
        "cycles": by_step,
    }


def _rank_and_role(row: dict[str, Any], trace_path: Path) -> tuple[int, str]:
    rank = row.get("rank")
    if type(rank) is not int:
        raise GateError(f"trace entry has invalid rank: {row!r}")
    expected_role = "draft" if rank == 0 else "target"
    explicit_role = row.get("role")
    if explicit_role is not None and explicit_role != expected_role:
        raise GateError(f"rank {rank} role mismatch: expected {expected_role}, got {explicit_role}")
    name_match = re.search(r"(?:^|[-_])(draft|target)(?:[-_])", trace_path.name)
    if name_match and name_match.group(1) != expected_role:
        raise GateError(f"rank {rank} trace name identifies the wrong role: {trace_path.name}")
    rank_match = re.search(r"rank[-_](\d+)", trace_path.name)
    if rank_match and int(rank_match.group(1)) != rank:
        raise GateError(f"rank {rank} trace name embeds rank {rank_match.group(1)}: {trace_path.name}")
    return rank, expected_role


def load_selection(path: Path) -> dict[int, dict[str, Any]]:
    document = _load_json(path, "selected trace manifest")
    if not isinstance(document, dict) or document.get("status") != "complete":
        raise GateError("selected trace manifest status must be 'complete'")
    rows = document.get("traces")
    if not isinstance(rows, list) or len(rows) != 4:
        raise GateError("selected trace manifest must contain exactly four traces")
    selected: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise GateError("selected trace manifest contains a non-object trace")
        raw_path = row.get("parsed", row.get("path"))
        trace_path = _resolve_path(path.parent, raw_path, "parsed/path")
        rank, role = _rank_and_role(row, trace_path)
        if rank in selected:
            raise GateError(f"selected trace manifest duplicates rank {rank}")
        selected[rank] = {"rank": rank, "role": role, "path": trace_path}
    if set(selected) != set(EXPECTED_RANKS):
        raise GateError(f"selected trace ranks must be {EXPECTED_RANKS}, got {sorted(selected)}")
    if {rank for rank, row in selected.items() if row["role"] == "target"} != set(TARGET_RANKS):
        raise GateError("selected traces must contain exactly three target ranks")
    return selected


def _metadata_argument(event: dict[str, Any], key: str) -> Any:
    arguments = event.get("args")
    if not isinstance(arguments, dict):
        raise GateError(f"malformed {event.get('name')} metadata event")
    return arguments.get(key)


def _trace_process_metadata(
    events: list[dict[str, Any]],
) -> tuple[int, set[int], set[int]]:
    process_names = {
        event.get("pid"): _metadata_argument(event, "name")
        for event in events
        if event.get("ph") == "M" and event.get("name") == "process_name"
    }
    process_labels = {
        event.get("pid"): _metadata_argument(event, "labels")
        for event in events
        if event.get("ph") == "M" and event.get("name") == "process_labels"
    }
    hardware_pids = {pid for pid, name in process_names.items() if name == "Ascend Hardware"}
    if not hardware_pids:
        raise GateError("trace contains no Ascend Hardware process")
    cards_by_pid: dict[int, int] = {}
    for pid in hardware_pids:
        match = re.fullmatch(r"NPU\s+(\d+)", str(process_labels.get(pid, "")))
        if not match:
            raise GateError(f"Ascend Hardware pid {pid} has no valid NPU label")
        cards_by_pid[pid] = int(match.group(1))
    cards = set(cards_by_pid.values())
    if len(cards) != 1:
        raise GateError(f"trace must identify exactly one Ascend Hardware NPU label; found {sorted(cards)}")
    python_pids = {pid for pid, name in process_names.items() if name == "Python"}
    if len(python_pids) != 1:
        raise GateError(
            "trace must identify exactly one Python process for ProfilerStep "
            f"provenance; found {sorted(python_pids, key=str)}"
        )
    return next(iter(cards)), hardware_pids, python_pids


def _load_calibration(trace_path: Path) -> tuple[int, str]:
    paths = list(trace_path.glob("PROF_*/host/start_info"))
    if len(paths) != 1:
        raise GateError(f"trace must contain exactly one CANN host/start_info: {trace_path}")
    try:
        info = ast.literal_eval(paths[0].read_text(encoding="utf-8"))
        if not isinstance(info, dict):
            raise GateError(f"invalid CANN clock calibration: {paths[0]}")
        wall_ns = _ns(info["collectionTimeBegin"])
        raw_ns = _positive_integer(info["clockMonotonicRaw"], "clockMonotonicRaw")
    except (KeyError, SyntaxError, TypeError, ValueError) as exc:
        raise GateError(f"invalid CANN clock calibration: {paths[0]}") from exc
    return wall_ns - raw_ns, str(paths[0].resolve())


def load_trace(row: dict[str, Any], expected_card: int) -> dict[str, Any]:
    rank = row["rank"]
    trace_path: Path = row["path"]
    if not trace_path.is_dir():
        raise GateError(f"missing parsed trace directory for rank {rank}: {trace_path}")

    info_paths = list(trace_path.glob("profiler_info*.json"))
    if len(info_paths) != 1:
        raise GateError(f"rank {rank} must have exactly one profiler_info JSON, found {len(info_paths)}")
    info = _load_json(info_paths[0], f"rank {rank} profiler_info")
    if not isinstance(info, dict) or str(info.get("rank_id")) != str(rank):
        raise GateError(f"rank identity mismatch in {info_paths[0]}")
    end_info = info.get("end_info")
    if not isinstance(end_info, dict):
        raise GateError(f"rank {rank} profiler_info has no end_info")
    _positive_integer(end_info.get("collectionTimeEnd"), "collectionTimeEnd")
    _positive_integer(end_info.get("MonotonicTimeEnd"), "MonotonicTimeEnd")

    device_dirs = list(trace_path.glob("PROF_*/device_*"))
    if len(device_dirs) != 1:
        raise GateError(f"rank {rank} must have exactly one PROF_*/device_* directory")
    device_match = re.fullmatch(r"device_(\d+)", device_dirs[0].name)
    if not device_match:
        raise GateError(f"invalid device directory: {device_dirs[0]}")
    device_card = int(device_match.group(1))
    end_markers = list(device_dirs[0].glob("end_info.*.done"))
    if len(end_markers) != 1:
        raise GateError(f"rank {rank} must have exactly one completed device end marker")
    expected_marker_name = f"end_info.{device_card}.done"
    if end_markers[0].name != expected_marker_name or end_markers[0].stat().st_size <= 0:
        raise GateError(f"rank {rank} has an invalid device completion marker: {end_markers[0]}")

    view_path = trace_path / "ASCEND_PROFILER_OUTPUT/trace_view.json"
    events = _load_json(view_path, f"rank {rank} trace_view")
    if not isinstance(events, list):
        raise GateError(f"rank {rank} trace_view must be a JSON event list")
    if any(not isinstance(event, dict) for event in events):
        raise GateError(f"rank {rank} trace_view contains a non-object event")
    metadata_card, hardware_pids, python_pids = _trace_process_metadata(events)
    if device_card != metadata_card or metadata_card != expected_card:
        raise GateError(
            f"rank {rank} card mismatch: device_dir={device_card}, "
            f"trace_label={metadata_card}, expected={expected_card}"
        )

    steps: dict[int, tuple[int, int]] = {}
    for event in events:
        match = re.fullmatch(r"ProfilerStep#(\d+)", str(event.get("name", "")))
        if event.get("ph") != "X" or not match:
            continue
        if event.get("pid") not in python_pids:
            raise GateError(f"rank {rank} ProfilerStep#{match.group(1)} is not owned by the trace's Python process")
        step = int(match.group(1))
        if step in steps:
            raise GateError(f"rank {rank} has duplicate ProfilerStep#{step}")
        start = _ns(event.get("ts"))
        end = start + _ns(event.get("dur"))
        if end <= start:
            raise GateError(f"rank {rank} has empty ProfilerStep#{step}")
        steps[step] = (start, end)
    if not steps:
        raise GateError(f"rank {rank} trace contains no ProfilerStep intervals")

    compute_events = []
    excluded_hccl = []
    for event in events:
        arguments = event.get("args")
        if event.get("ph") == "X" and event.get("pid") in hardware_pids:
            if not isinstance(arguments, dict):
                raise GateError(f"rank {rank} has a malformed hardware event")
        else:
            arguments = arguments if isinstance(arguments, dict) else {}
        task_type = arguments.get("Task Type")
        if event.get("ph") != "X" or event.get("pid") not in hardware_pids or task_type not in COMPUTE_TASK_TYPES:
            continue
        start = _ns(event.get("ts"))
        end = start + _ns(event.get("dur"))
        if end <= start:
            continue
        compact = {
            "name": str(event.get("name", "")),
            "task_type": task_type,
            "stream": event.get("tid"),
            "start_ns": start,
            "end_ns": end,
        }
        if _is_hccl_alias(compact["name"]):
            excluded_hccl.append(compact)
        else:
            compute_events.append(compact)
    if not compute_events:
        raise GateError(f"rank {rank} trace has no non-HCCL device compute kernels")

    calibration_offset, calibration_path = _load_calibration(trace_path)
    return {
        "rank": rank,
        "role": row["role"],
        "card": metadata_card,
        "path": str(trace_path),
        "trace_view": str(view_path.resolve()),
        "profiler_info": str(info_paths[0].resolve()),
        "end_marker": str(end_markers[0].resolve()),
        "calibration_path": calibration_path,
        "wall_minus_monotonic_raw_ns": calibration_offset,
        "steps": steps,
        "compute_events": compute_events,
        "excluded_hccl_events": excluded_hccl,
    }


def _decode_only_cycle(row: dict[str, Any]) -> bool:
    numeric_fields = ("draft_requests", "target_requests", "verify_candidates")
    for field in numeric_fields:
        value = row.get(field)
        if type(value) is not int or value <= 0:
            raise GateError(f"decode cycle {field} must be a finite positive integer")
    staged_prefill_ms = row.get("staged_prefill_critical_ms")
    if isinstance(staged_prefill_ms, bool) or not isinstance(staged_prefill_ms, (int, float)):
        raise GateError("decode cycle lacks numeric staged-prefill evidence")
    if not math.isfinite(staged_prefill_ms):
        raise GateError("decode cycle staged-prefill evidence is non-finite")
    has_both_sides = True
    return has_both_sides and abs(staged_prefill_ms) <= 1e-12


def _clip_events(events: list[dict[str, Any]], window: tuple[int, int]) -> list[dict[str, Any]]:
    clipped = []
    for event in events:
        start = max(event["start_ns"], window[0])
        end = min(event["end_ns"], window[1])
        if start < end:
            clipped.append({**event, "start_ns": start, "end_ns": end})
    return clipped


def _matmul_overlap_evidence(
    draft_events: list[dict[str, Any]],
    target_events: list[dict[str, Any]],
    limit: int = 8,
) -> tuple[int, int, list[dict[str, Any]]]:
    drafts = sorted(
        (event for event in draft_events if _is_matmul_event(event)),
        key=lambda row: row["start_ns"],
    )
    targets = sorted(
        (event for event in target_events if _is_matmul_event(event)),
        key=lambda row: row["start_ns"],
    )
    count = 0
    overlap_intervals: list[tuple[int, int]] = []
    examples: list[dict[str, Any]] = []
    target_cursor = 0
    active: list[dict[str, Any]] = []
    for draft in drafts:
        active = [row for row in active if row["end_ns"] > draft["start_ns"]]
        while target_cursor < len(targets) and targets[target_cursor]["start_ns"] < draft["end_ns"]:
            candidate = targets[target_cursor]
            target_cursor += 1
            if candidate["end_ns"] > draft["start_ns"]:
                active.append(candidate)
        for target in active:
            start = max(draft["start_ns"], target["start_ns"])
            end = min(draft["end_ns"], target["end_ns"])
            if start >= end:
                continue
            count += 1
            overlap_intervals.append((start, end))
            if len(examples) < limit:
                examples.append(
                    {
                        "draft_name": draft["name"],
                        "draft_stream": draft["stream"],
                        "target_name": target["name"],
                        "target_stream": target["stream"],
                        "overlap_ns": end - start,
                        "start_ns": start,
                        "end_ns": end,
                    }
                )
    overlap_ns = interval_duration(merge_intervals(overlap_intervals))
    return count, overlap_ns, examples


def _step_pair_measurement(
    step: int,
    window: tuple[int, int],
    draft: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, Any]:
    draft_events = _clip_events(draft["compute_events"], window)
    target_events = _clip_events(target["compute_events"], window)
    draft_union = merge_intervals((event["start_ns"], event["end_ns"]) for event in draft_events)
    target_union = merge_intervals((event["start_ns"], event["end_ns"]) for event in target_events)
    overlap = intersect_intervals(draft_union, target_union)
    draft_busy_ns = interval_duration(draft_union)
    target_busy_ns = interval_duration(target_union)
    overlap_ns = interval_duration(overlap)
    shorter_busy_ns = min(draft_busy_ns, target_busy_ns)
    matmul_count, matmul_overlap_ns, matmul_examples = _matmul_overlap_evidence(draft_events, target_events)
    return {
        "step": step,
        "window_ns": list(window),
        "window_us": (window[1] - window[0]) / 1000,
        "draft_compute_kernel_count": len(draft_events),
        "target_compute_kernel_count": len(target_events),
        "draft_compute_interval_union_count": len(draft_union),
        "target_compute_interval_union_count": len(target_union),
        "draft_busy_us": draft_busy_ns / 1000,
        "target_busy_us": target_busy_ns / 1000,
        "overlap_interval_count": len(overlap),
        "overlap_us": overlap_ns / 1000,
        "shorter_side_coverage": (overlap_ns / shorter_busy_ns if shorter_busy_ns else 0.0),
        "matmul_overlap_pair_count": matmul_count,
        "matmul_overlap_us": matmul_overlap_ns / 1000,
        "matmul_overlap_examples": matmul_examples,
    }


def analyze(
    selection_json: Path,
    result_json: Path,
    *,
    requested_steps: set[int] | None = None,
    max_clock_skew_us: float = 10.0,
    min_overlap_us: float = 1.0,
    min_matmul_overlap_us: float = 1.0,
    min_shorter_side_fraction: float = 0.0,
) -> dict[str, Any]:
    thresholds = (
        max_clock_skew_us,
        min_overlap_us,
        min_matmul_overlap_us,
        min_shorter_side_fraction,
    )
    if any(isinstance(value, bool) or not math.isfinite(value) for value in thresholds):
        raise GateError("all gate thresholds must be finite numbers")
    if max_clock_skew_us < 0 or min_overlap_us <= 0 or min_matmul_overlap_us <= 0:
        raise GateError("clock skew must be nonnegative and overlap thresholds must be positive")
    if not 0 <= min_shorter_side_fraction <= 1:
        raise GateError("minimum shorter-side fraction must be within [0, 1]")

    contract = load_result_contract(result_json)
    selected = load_selection(selection_json)
    traces = {rank: load_trace(selected[rank], contract["visible_cards"][rank]) for rank in EXPECTED_RANKS}
    if len({trace["card"] for trace in traces.values()}) != 4:
        raise GateError("the four ranks do not map to four unique cards")

    offsets = [trace["wall_minus_monotonic_raw_ns"] for trace in traces.values()]
    clock_skew_ns = max(offsets) - min(offsets)
    clock_skew_passed = clock_skew_ns <= max_clock_skew_us * 1000
    if not clock_skew_passed:
        raise GateError(
            f"cross-rank CANN clock calibration spread {clock_skew_ns / 1000:.3f} us exceeds {max_clock_skew_us:.3f} us"
        )

    common_steps = set.intersection(*(set(trace["steps"]) for trace in traces.values()))
    decode_steps = {
        step for step in common_steps if step in contract["cycles"] and _decode_only_cycle(contract["cycles"][step])
    }
    if requested_steps is not None:
        missing = requested_steps - decode_steps
        if missing:
            raise GateError(f"requested steps are not common, dual-work, decode-only steps: {sorted(missing)}")
        decode_steps = set(requested_steps)
    if not decode_steps:
        raise GateError("no common ProfilerStep has draft+target work and zero staged prefill")

    windows: dict[int, tuple[int, int]] = {}
    step_timing = {}
    for step in sorted(decode_steps):
        starts = [trace["steps"][step][0] for trace in traces.values()]
        ends = [trace["steps"][step][1] for trace in traces.values()]
        window = (max(starts), min(ends))
        if window[1] <= window[0]:
            raise GateError(f"ProfilerStep#{step} has no common cross-rank window")
        windows[step] = window
        step_timing[str(step)] = {
            "common_window_ns": list(window),
            "common_window_us": (window[1] - window[0]) / 1000,
            "rank_step_start_spread_us": (max(starts) - min(starts)) / 1000,
            "rank_step_end_spread_us": (max(ends) - min(ends)) / 1000,
            "cycle": {
                key: contract["cycles"][step].get(key)
                for key in (
                    "draft_requests",
                    "target_requests",
                    "verify_candidates",
                    "staged_prefill_critical_ms",
                )
            },
        }

    ordered_windows = [(step, windows[step]) for step in sorted(windows)]
    for (left_step, left), (right_step, right) in zip(ordered_windows, ordered_windows[1:]):
        if right[0] < left[1]:
            raise GateError(
                f"decode ProfilerStep windows overlap ({left_step}, {right_step}); kernel time would be double-counted"
            )

    pair_reports = []
    for target_rank in TARGET_RANKS:
        measurements = [
            _step_pair_measurement(step, windows[step], traces[0], traces[target_rank]) for step in sorted(decode_steps)
        ]
        draft_busy_us = sum(row["draft_busy_us"] for row in measurements)
        target_busy_us = sum(row["target_busy_us"] for row in measurements)
        overlap_us = sum(row["overlap_us"] for row in measurements)
        shorter_busy_us = min(draft_busy_us, target_busy_us)
        shorter_fraction = overlap_us / shorter_busy_us if shorter_busy_us else 0.0
        matmul_pair_count = sum(row["matmul_overlap_pair_count"] for row in measurements)
        matmul_overlap_us = sum(row["matmul_overlap_us"] for row in measurements)
        every_step_has_overlap = all(row["overlap_us"] >= min_overlap_us for row in measurements)
        passed = (
            every_step_has_overlap
            and overlap_us >= min_overlap_us
            and shorter_fraction >= min_shorter_side_fraction
            and matmul_pair_count > 0
            and matmul_overlap_us >= min_matmul_overlap_us
        )
        pair_reports.append(
            {
                "draft_rank": 0,
                "draft_card": traces[0]["card"],
                "target_rank": target_rank,
                "target_card": traces[target_rank]["card"],
                "decode_step_count": len(measurements),
                "draft_busy_us": draft_busy_us,
                "target_busy_us": target_busy_us,
                "overlap_us": overlap_us,
                "shorter_side_coverage": shorter_fraction,
                "matmul_overlap_pair_count": matmul_pair_count,
                "matmul_overlap_us": matmul_overlap_us,
                "every_decode_step_has_overlap": every_step_has_overlap,
                "passed": passed,
                "steps": measurements,
            }
        )

    all_pairs_passed = all(row["passed"] for row in pair_reports)
    true_matmul_pairs = sum(row["matmul_overlap_pair_count"] for row in pair_reports)
    true_matmul_overlap_us = sum(row["matmul_overlap_us"] for row in pair_reports)
    hccl_excluded = sum(len(trace["excluded_hccl_events"]) for trace in traces.values())
    gates = {
        "exactly_four_completed_traces": True,
        "rank_role_identity": True,
        "four_unique_card_identities": True,
        "three_target_ranks": True,
        "clock_calibration_within_threshold": clock_skew_passed,
        "decode_only_steps_present": bool(decode_steps),
        "hccl_excluded_from_compute_intervals": True,
        "all_three_draft_target_pairs_pass": all_pairs_passed,
        "true_matmul_overlap_present_on_every_pair": all(
            row["matmul_overlap_pair_count"] > 0 and row["matmul_overlap_us"] >= min_matmul_overlap_us
            for row in pair_reports
        ),
    }
    passed = all(gates.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "method": (
            "Official CANN trace_view Ascend Hardware ph=X intervals; per-rank "
            "compute intervals are clipped to common decode-only ProfilerStep "
            "windows, unioned, then intersected across draft/target cards"
        ),
        "inputs": {
            "selection_json": str(selection_json.resolve()),
            "result_json": str(result_json.resolve()),
        },
        "thresholds": {
            "max_clock_skew_us": max_clock_skew_us,
            "min_overlap_us_per_pair_per_decode_step": min_overlap_us,
            "min_matmul_overlap_us_per_pair": min_matmul_overlap_us,
            "min_shorter_side_fraction_per_pair": min_shorter_side_fraction,
        },
        "gates": gates,
        "clock": {
            "wall_minus_monotonic_raw_ns_by_rank": {
                str(rank): traces[rank]["wall_minus_monotonic_raw_ns"] for rank in EXPECTED_RANKS
            },
            "calibration_spread_ns": clock_skew_ns,
            "calibration_spread_us": clock_skew_ns / 1000,
        },
        "decode_only_steps": sorted(decode_steps),
        "step_timing": step_timing,
        "traces": [
            {
                key: trace[key]
                for key in (
                    "rank",
                    "role",
                    "card",
                    "path",
                    "trace_view",
                    "profiler_info",
                    "end_marker",
                    "calibration_path",
                )
            }
            | {
                "compute_kernel_count": len(trace["compute_events"]),
                "excluded_hccl_like_compute_kernel_count": len(trace["excluded_hccl_events"]),
            }
            for trace in traces.values()
        ],
        "excluded_hccl_like_compute_kernel_count": hccl_excluded,
        "true_matmul_overlap_pair_count": true_matmul_pairs,
        "true_matmul_overlap_us": true_matmul_overlap_us,
        "pairs": pair_reports,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--step",
        type=int,
        action="append",
        help="Restrict the gate to a step (repeatable); it must still be decode-only",
    )
    parser.add_argument("--max-clock-skew-us", type=float, default=10.0)
    parser.add_argument("--min-overlap-us", type=float, default=1.0)
    parser.add_argument("--min-matmul-overlap-us", type=float, default=1.0)
    parser.add_argument("--min-shorter-side-fraction", type=float, default=0.0)
    args = parser.parse_args()

    try:
        report = analyze(
            args.selection_json,
            args.result_json,
            requested_steps=set(args.step) if args.step else None,
            max_clock_skew_us=args.max_clock_skew_us,
            min_overlap_us=args.min_overlap_us,
            min_matmul_overlap_us=args.min_matmul_overlap_us,
            min_shorter_side_fraction=args.min_shorter_side_fraction,
        )
    except Exception as exc:
        # This is a gate: an unexpected parser/data-shape failure must produce
        # a machine-readable FAIL artifact, never an implicit success or a
        # missing report. KeyboardInterrupt/SystemExit remain unsuppressed.
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "passed": False,
            "inputs": {
                "selection_json": str(args.selection_json.resolve()),
                "result_json": str(args.result_json.resolve()),
            },
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        _write_report(args.output, report)
        print(json.dumps(report, ensure_ascii=False), file=sys.stderr)
        return 2

    _write_report(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
