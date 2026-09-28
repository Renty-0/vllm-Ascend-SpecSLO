from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "examples/gate_specslo_device_overlap.py"
SPEC = importlib.util.spec_from_file_location("specslo_device_overlap_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def _metadata(pid: int, card: int) -> list[dict[str, Any]]:
    return [
        {
            "name": "process_name",
            "ph": "M",
            "pid": pid - 100,
            "args": {"name": "Python"},
        },
        {
            "name": "process_labels",
            "ph": "M",
            "pid": pid - 100,
            "args": {"labels": "CPU"},
        },
        {
            "name": "process_name",
            "ph": "M",
            "pid": pid,
            "args": {"name": "Ascend Hardware"},
        },
        {
            "name": "process_labels",
            "ph": "M",
            "pid": pid,
            "args": {"labels": f"NPU {card}"},
        },
    ]


def _event(
    name: str,
    pid: int,
    start_us: float,
    duration_us: float,
    task_type: str = "KERNEL_AICORE",
) -> dict[str, Any]:
    return {
        "name": name,
        "pid": pid,
        "tid": 7,
        "ts": str(start_us),
        "dur": duration_us,
        "ph": "X",
        "args": {"Task Type": task_type},
    }


def _make_case(tmp_path: Path) -> tuple[Path, Path, dict[int, Path]]:
    cards = [2, 3, 4, 5]
    traces = []
    trace_paths = {}
    base_offset_ns = 100_000
    for rank, card in enumerate(cards):
        role = "draft" if rank == 0 else "target"
        trace = tmp_path / f"spec-rhythm-{role}-rank-{rank}_123_ascend_pt"
        trace_paths[rank] = trace
        output = trace / "ASCEND_PROFILER_OUTPUT"
        output.mkdir(parents=True)
        profiler = trace / "PROF_000001"
        host = profiler / "host"
        device = profiler / f"device_{card}"
        host.mkdir(parents=True)
        device.mkdir()
        (device / f"end_info.{card}.done").write_text("done", encoding="utf-8")
        wall_begin_ns = 2_000_000_000
        raw_begin_ns = wall_begin_ns - base_offset_ns - rank * 500
        (host / "start_info").write_text(
            str(
                {
                    "collectionTimeBegin": str(wall_begin_ns // 1000),
                    "clockMonotonicRaw": str(raw_begin_ns),
                }
            ),
            encoding="utf-8",
        )
        (trace / f"profiler_info_{rank}.json").write_text(
            json.dumps(
                {
                    "rank_id": str(rank),
                    "end_info": {
                        "collectionTimeEnd": 3_000_000_000 + rank,
                        "MonotonicTimeEnd": 2_000_000_000 + rank,
                    },
                }
            ),
            encoding="utf-8",
        )
        pid = 100 + rank
        events = _metadata(pid, card)
        events.extend(
            [
                {
                    "name": "ProfilerStep#17",
                    "pid": rank,
                    "tid": rank,
                    "ts": str(1000.0 + rank * 0.1),
                    "dur": 100.0 - rank * 0.1,
                    "ph": "X",
                    "args": {},
                },
                {
                    "name": "ProfilerStep#18",
                    "pid": rank,
                    "tid": rank,
                    "ts": str(1200.0 + rank * 0.1),
                    "dur": 100.0 - rank * 0.1,
                    "ph": "X",
                    "args": {},
                },
                _event(
                    "aclnnMatmul_MatMulV2" if rank == 0 else "MatMulV2_kernel",
                    pid,
                    1010.0 + rank,
                    20.0,
                ),
                _event(
                    "vector_compute",
                    pid,
                    1040.0 + rank,
                    10.0,
                    "KERNEL_AIVEC",
                ),
                _event(
                    "hcom_allReduce_",
                    pid,
                    1005.0,
                    80.0,
                    "KERNEL_AIVEC",
                ),
                _event(
                    "AivKernel",
                    pid,
                    1006.0,
                    80.0,
                    "KERNEL_AIVEC",
                ),
                _event("prefill_MatMul", pid, 1210.0, 20.0),
            ]
        )
        (output / "trace_view.json").write_text(json.dumps(events), encoding="utf-8")
        traces.append({"rank": rank, "role": role, "parsed": str(trace)})

    selection = tmp_path / "selected-traces.json"
    selection.write_text(json.dumps({"status": "complete", "traces": traces}), encoding="utf-8")
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "draft_tensor_parallel_size": 1,
                "target_tensor_parallel_size": 3,
                "runtime_environment": {"ASCEND_RT_VISIBLE_DEVICES": ",".join(map(str, cards))},
                "results": [
                    {
                        "decode_host_profile": {
                            "cycles": [
                                {
                                    "step": 17,
                                    "draft_requests": 8,
                                    "target_requests": 8,
                                    "verify_candidates": 32,
                                    "staged_prefill_critical_ms": 0.0,
                                },
                                {
                                    "step": 18,
                                    "draft_requests": 8,
                                    "target_requests": 8,
                                    "verify_candidates": 32,
                                    "staged_prefill_critical_ms": 7.5,
                                },
                            ]
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return selection, result, trace_paths


def _read_events(trace: Path) -> list[dict[str, Any]]:
    path = trace / "ASCEND_PROFILER_OUTPUT/trace_view.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _write_events(trace: Path, events: list[dict[str, Any]]) -> None:
    path = trace / "ASCEND_PROFILER_OUTPUT/trace_view.json"
    path.write_text(json.dumps(events), encoding="utf-8")


def test_interval_union_and_intersection_do_not_double_count() -> None:
    left = GATE.merge_intervals([(1, 4), (3, 6), (8, 9)])
    right = GATE.merge_intervals([(2, 5), (6, 8)])
    assert left == [[1, 6], [8, 9]]
    assert right == [[2, 5], [6, 8]]
    assert GATE.intersect_intervals(left, right) == [[2, 5]]


def test_hccl_alias_filter_preserves_fused_mc2_matmul() -> None:
    assert GATE._is_hccl_alias("hcom_allReduce_") is True
    assert GATE._is_hccl_alias("AivKernel") is True
    assert GATE._is_hccl_alias("standalone_all_reduce") is True
    assert GATE._is_hccl_alias("MatMulAllReduceAddRmsNorm") is False
    assert GATE._is_matmul_name("MatMulAllReduceAddRmsNorm") is True


def test_gate_passes_all_three_target_pairs_and_excludes_hccl(
    tmp_path: Path,
) -> None:
    selection, result, _ = _make_case(tmp_path)
    report = GATE.analyze(selection, result)

    assert report["passed"] is True
    assert report["decode_only_steps"] == [17]
    assert report["clock"]["calibration_spread_us"] == pytest.approx(1.5)
    assert report["excluded_hccl_like_compute_kernel_count"] == 8
    assert report["true_matmul_overlap_pair_count"] == 3
    assert len(report["pairs"]) == 3
    assert all(pair["passed"] for pair in report["pairs"])
    assert [pair["overlap_us"] for pair in report["pairs"]] == pytest.approx([28.0, 26.0, 24.0])
    assert report["gates"]["all_three_draft_target_pairs_pass"] is True


def test_selection_requires_exactly_four_completed_traces(tmp_path: Path) -> None:
    selection, result, _ = _make_case(tmp_path)
    payload = json.loads(selection.read_text(encoding="utf-8"))
    payload["traces"].pop()
    selection.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(GATE.GateError, match="exactly four traces"):
        GATE.analyze(selection, result)


def test_trace_requires_completion_marker(tmp_path: Path) -> None:
    selection, result, traces = _make_case(tmp_path)
    next(traces[2].glob("PROF_*/device_*/end_info.*.done")).unlink()

    with pytest.raises(GATE.GateError, match="completed device end marker"):
        GATE.analyze(selection, result)


def test_rank_card_identity_is_checked_against_runtime_contract(
    tmp_path: Path,
) -> None:
    selection, result, _ = _make_case(tmp_path)
    document = json.loads(result.read_text(encoding="utf-8"))
    document["runtime_environment"]["ASCEND_RT_VISIBLE_DEVICES"] = "2,3,5,4"
    result.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(GATE.GateError, match="card mismatch"):
        GATE.analyze(selection, result)


def test_unlabelled_hardware_pid_cannot_pollute_rank_intervals(
    tmp_path: Path,
) -> None:
    selection, result, traces = _make_case(tmp_path)
    events = _read_events(traces[1])
    events.append(
        {
            "name": "process_name",
            "ph": "M",
            "pid": 999,
            "args": {"name": "Ascend Hardware"},
        }
    )
    events.append(_event("MatMulV2_unlabelled", 999, 1012.0, 10.0))
    _write_events(traces[1], events)

    with pytest.raises(GATE.GateError, match="has no valid NPU label"):
        GATE.analyze(selection, result)


def test_clock_calibration_threshold_is_fail_closed(tmp_path: Path) -> None:
    selection, result, traces = _make_case(tmp_path)
    start_info = next(traces[3].glob("PROF_*/host/start_info"))
    info = ast_literal(start_info.read_text(encoding="utf-8"))
    info["clockMonotonicRaw"] = str(int(info["clockMonotonicRaw"]) - 50_000)
    start_info.write_text(str(info), encoding="utf-8")

    with pytest.raises(GATE.GateError, match="clock calibration spread"):
        GATE.analyze(selection, result, max_clock_skew_us=10.0)


def ast_literal(value: str) -> dict[str, str]:
    # Kept local so importing this CPU-only test cannot accidentally import
    # torch_npu through unrelated profiling helpers.
    import ast

    return ast.literal_eval(value)


def test_prefill_steps_are_not_accepted_as_decode_overlap(tmp_path: Path) -> None:
    selection, result, _ = _make_case(tmp_path)
    document = json.loads(result.read_text(encoding="utf-8"))
    cycles = document["results"][0]["decode_host_profile"]["cycles"]
    cycles[0]["staged_prefill_critical_ms"] = 0.01
    result.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(GATE.GateError, match="no common ProfilerStep"):
        GATE.analyze(selection, result)


def test_real_compute_overlap_without_matmul_fails_pair_gate(tmp_path: Path) -> None:
    selection, result, traces = _make_case(tmp_path)
    events = _read_events(traces[1])
    for event in events:
        if "matmul" in str(event.get("name", "")).lower() and float(event.get("ts", 0)) < 1200:
            event["name"] = "cube_without_gemm_identity"
    _write_events(traces[1], events)

    report = GATE.analyze(selection, result)
    pair = next(row for row in report["pairs"] if row["target_rank"] == 1)
    assert pair["overlap_us"] > 0
    assert pair["matmul_overlap_pair_count"] == 0
    assert pair["passed"] is False
    assert report["passed"] is False


def test_subthreshold_matmul_overlap_is_not_accepted(tmp_path: Path) -> None:
    selection, result, traces = _make_case(tmp_path)
    events = _read_events(traces[1])
    for event in events:
        if event.get("name") == "MatMulV2_kernel" and float(event["ts"]) < 1200:
            event["ts"] = "1029.999"
            event["dur"] = 1.0
    _write_events(traces[1], events)

    report = GATE.analyze(selection, result, min_matmul_overlap_us=1.0)
    pair = next(row for row in report["pairs"] if row["target_rank"] == 1)
    assert pair["overlap_us"] > 1.0
    assert pair["matmul_overlap_pair_count"] == 1
    assert pair["matmul_overlap_us"] == pytest.approx(0.001)
    assert pair["passed"] is False


def test_hccl_only_time_is_not_accepted_as_compute_overlap(tmp_path: Path) -> None:
    selection, result, traces = _make_case(tmp_path)
    events = _read_events(traces[1])
    for event in events:
        name = str(event.get("name", ""))
        if event.get("ph") == "X" and ("matmul" in name.lower() or name == "vector_compute"):
            event["ts"] = "1210.0"
    _write_events(traces[1], events)

    report = GATE.analyze(selection, result)
    pair = next(row for row in report["pairs"] if row["target_rank"] == 1)
    assert pair["steps"][0]["target_compute_kernel_count"] == 0
    assert pair["overlap_us"] == 0
    assert pair["passed"] is False
    assert report["passed"] is False


def test_requested_step_must_still_be_decode_only(tmp_path: Path) -> None:
    selection, result, _ = _make_case(tmp_path)
    with pytest.raises(GATE.GateError, match="not common, dual-work, decode-only"):
        GATE.analyze(selection, result, requested_steps={18})


def test_overlapping_decode_step_windows_fail_before_summing(tmp_path: Path) -> None:
    selection, result, traces = _make_case(tmp_path)
    document = json.loads(result.read_text(encoding="utf-8"))
    document["results"][0]["decode_host_profile"]["cycles"][1]["staged_prefill_critical_ms"] = 0.0
    result.write_text(json.dumps(document), encoding="utf-8")
    for trace in traces.values():
        events = _read_events(trace)
        for event in events:
            if event.get("name") == "ProfilerStep#18":
                event["ts"] = str(1050.0 + int(event["pid"]) * 0.1)
        _write_events(trace, events)

    with pytest.raises(GATE.GateError, match="kernel time would be double-counted"):
        GATE.analyze(selection, result)


def test_cli_writes_machine_readable_failure_and_returns_two(
    tmp_path: Path,
) -> None:
    selection, result, _ = _make_case(tmp_path)
    payload = json.loads(selection.read_text(encoding="utf-8"))
    payload["status"] = "running"
    selection.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "failure.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--selection-json",
            str(selection),
            "--result-json",
            str(result),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["passed"] is False
    assert report["error"]["type"] == "GateError"


def test_cli_converts_malformed_trace_event_to_failure_report(
    tmp_path: Path,
) -> None:
    selection, result, traces = _make_case(tmp_path)
    events = _read_events(traces[0])
    events.append(None)
    _write_events(traces[0], events)
    output = tmp_path / "malformed-failure.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--selection-json",
            str(selection),
            "--result-json",
            str(result),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert "non-object event" in report["error"]["message"]
