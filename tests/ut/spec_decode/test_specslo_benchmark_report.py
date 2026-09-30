# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage for the fixed-gamma benchmark audit artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.benchmark_nano_pearl_speculative import (
    _aggregate_decode_profile,
    _resolve_profile_batch_size,
)
from examples.specslo_benchmark_report import (
    ONLINE_E2E_TIMING_SCOPE,
    PAPER_GOODPUT_DEFINITION,
    PAPER_TPOT_DEFINITION,
    capture_runtime_environment,
    collect_provenance,
    draft_pa_parallel_update_evidence,
    evaluate_gate,
    finalize_provenance,
    measure_qualified_mc2_replay_coverage,
    parse_visible_devices,
    qualified_mc2_profile_replay_contract,
    sha256_file,
    target_fia_parallel_update_evidence,
    target_replay_ahead_evidence,
    write_json_exclusive,
)
from vllm_ascend.spec_decode.pearl.mc2 import (
    MC2_TP3_NATIVE_EPILOGUE_OPERATOR,
    MC2Profile,
)


def _runtime_environment() -> dict[str, str | None]:
    return capture_runtime_environment(
        {
            "ASCEND_RT_VISIBLE_DEVICES": "1,2,3,4",
            "TASK_QUEUE_ENABLE": "1",
            "HCCL_OP_EXPANSION_MODE": "AIV",
            "HCCL_DETERMINISTIC": "true",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )


def test_capture_runtime_environment_records_mc2_input_probe_contract():
    environment = capture_runtime_environment(
        {
            "VLLM_ASCEND_PEARL_MC2_CAPTURE_DIR": "/tmp/mc2-inputs",
            "VLLM_ASCEND_PEARL_MC2_CAPTURE_ROWS": "64,100",
        }
    )

    assert environment["VLLM_ASCEND_PEARL_MC2_CAPTURE_DIR"] == "/tmp/mc2-inputs"
    assert environment["VLLM_ASCEND_PEARL_MC2_CAPTURE_ROWS"] == "64,100"


def test_capture_runtime_environment_records_draft_device_pa_ab_switch_and_profiler():
    environment = capture_runtime_environment(
        {
            "VLLM_ASCEND_SPECSLO_DRAFT_DEVICE_PAGED_ATTENTION": "1",
            "VLLM_ASCEND_PEARL_PROFILE_PA_TASK_UPDATE": "0",
        }
    )

    assert environment["VLLM_ASCEND_SPECSLO_DRAFT_DEVICE_PAGED_ATTENTION"] == "1"
    assert environment["VLLM_ASCEND_PEARL_PROFILE_PA_TASK_UPDATE"] == "0"


def test_capture_runtime_environment_records_host_continuation_ab_switch():
    environment = capture_runtime_environment(
        {
            "VLLM_ASCEND_SPECRHYTHM_GLOO_PROPOSAL_REUSE_HOST_CONTINUATION": "0",
        }
    )

    assert environment["VLLM_ASCEND_SPECRHYTHM_GLOO_PROPOSAL_REUSE_HOST_CONTINUATION"] == "0"


def test_capture_runtime_environment_records_target_fia_update_workers():
    environment = capture_runtime_environment({"VLLM_ASCEND_PEARL_TARGET_FIA_TASK_UPDATE_WORKERS": "2"})

    assert environment["VLLM_ASCEND_PEARL_TARGET_FIA_TASK_UPDATE_WORKERS"] == "2"


def test_capture_runtime_environment_records_draft_pa_update_workers():
    environment = capture_runtime_environment({"VLLM_ASCEND_PEARL_DRAFT_PA_TASK_UPDATE_WORKERS": "4"})

    assert environment["VLLM_ASCEND_PEARL_DRAFT_PA_TASK_UPDATE_WORKERS"] == "4"


def test_capture_runtime_environment_records_new_drain_and_mixed_submit_switches():
    environment = capture_runtime_environment(
        {
            "VLLM_ASCEND_SPECRHYTHM_DRAIN_TARGET_LRTF": "1",
            "VLLM_ASCEND_SPECRHYTHM_TARGET_RESIDUAL_LRTF": "1",
            "VLLM_ASCEND_SPECRHYTHM_EARLY_CURRENT_MIXED_TARGET_SUBMIT": "0",
        }
    )

    assert environment["VLLM_ASCEND_SPECRHYTHM_DRAIN_TARGET_LRTF"] == "1"
    assert environment["VLLM_ASCEND_SPECRHYTHM_TARGET_RESIDUAL_LRTF"] == "1"
    assert environment["VLLM_ASCEND_SPECRHYTHM_EARLY_CURRENT_MIXED_TARGET_SUBMIT"] == "0"


def _target_replay_ahead_rows(*, enabled: bool) -> list[dict]:
    fields = (
        "attempts",
        "eligible",
        "submitted",
        "consumed",
        "signature_mismatches",
    )
    counters = (6, 2, 2, 2, 0) if enabled else (0, 0, 0, 0, 0)
    return [
        {
            "rank": rank,
            "worker_spec_rhythm_target_replay_ahead_enabled": int(enabled),
            **{f"worker_spec_rhythm_target_replay_ahead_{name}": value for name, value in zip(fields, counters)},
            **{
                f"worker_spec_rhythm_target_replay_ahead_reject_{reason}": (
                    4 if enabled and reason == "mutable_frontier" else 0
                )
                for reason in (
                    "protocol",
                    "mutable_frontier",
                    "pending_eager",
                    "schedule",
                    "ticket_epoch",
                    "payload_shape",
                    "graph_contract",
                    "resource_contract",
                    "staged_correction",
                )
            },
        }
        for rank in range(4)
    ]


@pytest.mark.parametrize("enabled", [False, True])
def test_target_replay_ahead_evidence_accepts_consistent_ranks(enabled):
    result = target_replay_ahead_evidence(_target_replay_ahead_rows(enabled=enabled), enabled=enabled)
    assert result["passed"] is True
    assert result["errors"] == []


def test_target_replay_ahead_evidence_rejects_unconsumed_target_graph():
    rows = _target_replay_ahead_rows(enabled=True)
    rows[1]["worker_spec_rhythm_target_replay_ahead_consumed"] = 1
    result = target_replay_ahead_evidence(rows, enabled=True)
    assert result["passed"] is False
    assert any("submitted/consumed" in error for error in result["errors"])
    assert any("differs across ranks" in error for error in result["errors"])


def test_target_replay_ahead_evidence_requires_measured_submission():
    rows = _target_replay_ahead_rows(enabled=True)
    for row in rows:
        row["worker_spec_rhythm_target_replay_ahead_attempts"] = 4
        for field in ("eligible", "submitted", "consumed"):
            row[f"worker_spec_rhythm_target_replay_ahead_{field}"] = 0
    result = target_replay_ahead_evidence(rows, enabled=True)
    assert result["passed"] is False
    assert any("submitted no replay-ahead" in error for error in result["errors"])


def _target_fia_evidence_rows(
    workers: int,
) -> tuple[list[dict], list[dict]]:
    graph_zero_fields = (
        "aclgraph_failed_captures",
        "aclgraph_capacity_fallbacks",
        "aclgraph_shape_fallbacks",
        "aclgraph_generic_eager_fallback_calls",
        "aclgraph_draft_eager_fallback_calls",
        "aclgraph_target_eager_fallback_calls",
        "aclgraph_generic_runtime_validation_failures",
        "aclgraph_draft_runtime_validation_failures",
        "aclgraph_target_runtime_validation_failures",
    )
    target_service_zero_fields = (
        "worker_spec_rhythm_mixed_target_graph_bounded_shape_fallback_batches",
        "worker_spec_rhythm_mixed_target_graph_token_capacity_fallback_batches",
        "worker_spec_rhythm_mixed_target_graph_execution_fallback_batches",
        "worker_spec_rhythm_stable_target_verify_graph_bounded_shape_fallback_batches",
        "worker_spec_rhythm_stable_target_verify_graph_token_capacity_fallback_batches",
        "worker_spec_rhythm_stable_target_verify_graph_execution_fallback_batches",
    )
    absolute = []
    measured = []
    for rank in range(4):
        is_draft = int(rank == 0)
        absolute.append(
            {
                "rank": rank,
                "is_draft_rank": is_draft,
                "aclgraph_target_fia_task_update_workers": (1 if is_draft else workers),
                **{name: 0 for name in graph_zero_fields},
            }
        )
        measured.append(
            {
                "rank": rank,
                "is_draft_rank": is_draft,
                "aclgraph_target_fia_parallel_update_replays_delta": (0 if is_draft or workers == 1 else 7),
                "aclgraph_target_fia_parallel_update_tasks_delta": (0 if is_draft or workers == 1 else 448),
                **{f"{name}_delta": 0 for name in graph_zero_fields},
                **({} if is_draft else {name: 0 for name in target_service_zero_fields}),
            }
        )
    return absolute, measured


@pytest.mark.parametrize("workers", (1, 2, 3, 4))
def test_target_fia_parallel_update_evidence_accepts_strict_tp1_tp3_contract(
    workers: int,
):
    absolute, measured = _target_fia_evidence_rows(workers)

    evidence = target_fia_parallel_update_evidence(
        absolute,
        measured,
        expected_workers=workers,
    )

    assert evidence["passed"] is True
    assert evidence["errors"] == []
    assert evidence["ranks"]["0"]["configured_workers"] == 1
    assert evidence["ranks"]["1"]["configured_workers"] == workers


def test_target_fia_parallel_update_evidence_fails_closed_on_missing_parallel_work():
    absolute, measured = _target_fia_evidence_rows(2)
    measured[2]["aclgraph_target_fia_parallel_update_replays_delta"] = 0
    measured[3]["aclgraph_target_eager_fallback_calls_delta"] = 1

    evidence = target_fia_parallel_update_evidence(
        absolute,
        measured,
        expected_workers=2,
    )

    assert evidence["passed"] is False
    assert any("target rank 2 has no parallel FIA update replay" in error for error in evidence["errors"])
    assert any("rank 3 aclgraph_target_eager_fallback_calls_delta" in error for error in evidence["errors"])


def test_target_fia_parallel_update_evidence_rejects_wrong_rank_topology_and_worker_count():
    absolute, measured = _target_fia_evidence_rows(4)
    absolute[0]["aclgraph_target_fia_task_update_workers"] = 4
    absolute.pop(3)
    measured.pop(3)

    evidence = target_fia_parallel_update_evidence(
        absolute,
        measured,
        expected_workers=4,
    )

    assert evidence["passed"] is False
    assert any("TP1+TP3 set" in error for error in evidence["errors"])
    assert any("rank 0 target-FIA update workers 4 != 1" in error for error in evidence["errors"])


def _draft_pa_evidence_rows(
    workers: int,
) -> tuple[list[dict], list[dict]]:
    graph_zero_fields = (
        "aclgraph_failed_captures",
        "aclgraph_capacity_fallbacks",
        "aclgraph_shape_fallbacks",
        "aclgraph_generic_eager_fallback_calls",
        "aclgraph_draft_eager_fallback_calls",
        "aclgraph_target_eager_fallback_calls",
        "aclgraph_generic_runtime_validation_failures",
        "aclgraph_draft_runtime_validation_failures",
        "aclgraph_target_runtime_validation_failures",
    )
    absolute = []
    measured = []
    for rank in range(4):
        is_draft = int(rank == 0)
        absolute.append(
            {
                "rank": rank,
                "is_draft_rank": is_draft,
                "aclgraph_draft_pa_task_update_workers": (workers if is_draft else 1),
                **{name: 0 for name in graph_zero_fields},
            }
        )
        measured.append(
            {
                "rank": rank,
                "is_draft_rank": is_draft,
                "aclgraph_draft_pa_parallel_update_replays_delta": (7 if is_draft and workers > 1 else 0),
                "aclgraph_draft_pa_parallel_update_tasks_delta": (1792 if is_draft and workers > 1 else 0),
                "aclgraph_draft_pa_parallel_update_fallback_replays_delta": 0,
                "aclgraph_draft_pa_parallel_update_rejections_delta": {},
                **{f"{name}_delta": 0 for name in graph_zero_fields},
            }
        )
    return absolute, measured


@pytest.mark.parametrize("workers", (1, 2, 4))
def test_draft_pa_parallel_update_evidence_accepts_strict_tp1_tp3_contract(
    workers: int,
):
    absolute, measured = _draft_pa_evidence_rows(workers)

    evidence = draft_pa_parallel_update_evidence(
        absolute,
        measured,
        expected_workers=workers,
    )

    assert evidence["passed"] is True
    assert evidence["errors"] == []
    assert evidence["ranks"]["0"]["configured_workers"] == workers
    assert evidence["ranks"]["1"]["configured_workers"] == 1


def test_draft_pa_parallel_update_evidence_fails_closed_on_missing_or_misrouted_work():
    absolute, measured = _draft_pa_evidence_rows(2)
    measured[0]["aclgraph_draft_pa_parallel_update_replays_delta"] = 0
    measured[2]["aclgraph_draft_pa_parallel_update_tasks_delta"] = 1
    measured[0]["aclgraph_draft_pa_parallel_update_fallback_replays_delta"] = 1
    measured[0]["aclgraph_draft_pa_parallel_update_rejections_delta"] = {
        "non_full_chain_output": 1,
    }

    evidence = draft_pa_parallel_update_evidence(
        absolute,
        measured,
        expected_workers=2,
    )

    assert evidence["passed"] is False
    assert any("draft rank 0 has no parallel PA update replay" in error for error in evidence["errors"])
    assert any("target rank 2 unexpectedly reports draft-PA" in error for error in evidence["errors"])
    assert any("parallel fallback replays=1" in error for error in evidence["errors"])
    assert any("non_full_chain_output" in error for error in evidence["errors"])


def test_draft_pa_parallel_update_evidence_rejects_parallel_work_with_one_worker():
    absolute, measured = _draft_pa_evidence_rows(1)
    measured[0]["aclgraph_draft_pa_parallel_update_replays_delta"] = 1
    measured[0]["aclgraph_draft_pa_parallel_update_tasks_delta"] = 256

    evidence = draft_pa_parallel_update_evidence(
        absolute,
        measured,
        expected_workers=1,
    )

    assert evidence["passed"] is False
    assert any("with one configured worker" in error for error in evidence["errors"])


def test_draft_pa_parallel_update_evidence_rejects_wrong_topology_and_workers():
    absolute, measured = _draft_pa_evidence_rows(4)
    absolute[1]["aclgraph_draft_pa_task_update_workers"] = 4
    absolute.pop(3)
    measured.pop(3)

    evidence = draft_pa_parallel_update_evidence(
        absolute,
        measured,
        expected_workers=4,
    )

    assert evidence["passed"] is False
    assert any("TP1+TP3 set" in error for error in evidence["errors"])
    assert any("rank 1 draft-PA update workers 4 != 1" in error for error in evidence["errors"])


def _profile_worker(rank: int, *, is_draft: bool) -> dict:
    return {
        "rank": rank,
        "is_draft_rank": is_draft,
        "worker_profiled_decode_steps": 2,
        "worker_profile_draft_compute_seconds": 0.02 if is_draft else 0.0,
        "worker_profile_draft_to_target_communication_seconds": 0.003,
        "worker_profile_target_compute_seconds": 0.0 if is_draft else 0.04,
        "worker_profile_target_verdict_seconds": 0.0 if is_draft else 0.002,
        "worker_profile_target_to_draft_communication_seconds": 0.004,
        "worker_profile_wait_sync_seconds": 0.005,
        "worker_profile_state_update_seconds": 0.001,
    }


def _baseline_payload(*, goodput: float = 100.0) -> dict:
    return {
        "backend": "vllm-ascend-target-only",
        "model": "/models/Qwen3-32B",
        "tensor_parallel_size": 4,
        "request_manifest": "/workload.jsonl",
        "request_manifest_sha256": "manifest",
        "max_tokens": 256,
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.8,
        "warmup_prompts": 60,
        "warmup_max_tokens": 256,
        "warmup_runs": 1,
        "warmup_excluded_from_measurement": True,
        "enforce_eager": False,
        "enable_prefix_caching": False,
        "online_arrivals": True,
        "runtime_environment": _runtime_environment(),
        "results": [
            {
                "batch_size": 64,
                "num_prompts": 60,
                "output_tokens": 15360,
                "prompt_token_ids_sha256": "prompts",
                "elapsed_seconds": 30.0,
                "e2e_elapsed_seconds": 30.0,
                "e2e_timing_scope": ONLINE_E2E_TIMING_SCOPE,
                "request_output_token_limits": [256] * 60,
                "warmup_output_token_limits": [256] * 60,
                "slo": {
                    "paper": {
                        "tpot_definition": PAPER_TPOT_DEFINITION,
                        "goodput_definition": PAPER_GOODPUT_DEFINITION,
                        "goodput_tokens": int(goodput * 30.0),
                        "goodput_tokens_per_e2e_second": goodput,
                        "attainment": 0.6,
                        "attained_requests": 36,
                        "constrained_requests": 60,
                        "missing_timing_requests": 0,
                    }
                },
            }
        ],
    }


def _candidate_payload(*, goodput: float = 130.0, attainment: float = 0.8) -> dict:
    return {
        "backend": "specslo-native-specrhythm",
        "target_model": "/models/Qwen3-32B",
        "draft_tensor_parallel_size": 1,
        "target_tensor_parallel_size": 3,
        "gamma": 4,
        "spec_rhythm_linear_full_window": True,
        "spec_rhythm_draft_mode": "serial_linear",
        "request_manifest": "/workload.jsonl",
        "request_manifest_sha256": "manifest",
        "max_tokens": 256,
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.8,
        "warmup_prompts": 60,
        # Strict graph qualification cannot set the scalar CLI cap, but its
        # per-request effective limits below are identical to the baseline.
        "warmup_max_tokens": None,
        "warmup_runs": 1,
        "warmup_excluded_from_measurement": True,
        "enforce_eager": False,
        "enable_prefix_caching": False,
        "online_arrivals": True,
        "runtime_environment": _runtime_environment(),
        "results": [
            {
                "batch_size": 64,
                "num_prompts": 60,
                "output_tokens": 15360,
                "prompt_token_ids_sha256": "prompts",
                "e2e_elapsed_seconds": 25.0,
                "e2e_timing_scope": ONLINE_E2E_TIMING_SCOPE,
                "request_output_token_limits": [256] * 60,
                "warmup_output_token_limits": [256] * 60,
                "slo": {
                    "paper": {
                        "tpot_definition": PAPER_TPOT_DEFINITION,
                        "goodput_definition": PAPER_GOODPUT_DEFINITION,
                        "goodput_tokens": int(goodput * 25.0),
                        "goodput_tokens_per_e2e_second": goodput,
                        "attainment": attainment,
                        "attained_requests": 48,
                        "constrained_requests": 60,
                        "missing_timing_requests": 0,
                    }
                },
            }
        ],
    }


def _mc2_profile() -> MC2Profile:
    def row(m: int, k: int, *, qualified: bool = True) -> dict:
        return {
            "m": m,
            "k": k,
            "n": 5120,
            "dtype": "bfloat16",
            "weight_format": "ND",
            "is_trans_b": True,
            "activation_layout": "contiguous",
            "weight_layout": "contiguous",
            "residual_layout": "contiguous",
            "gamma_layout": "contiguous",
            "changed_input_replays": 8,
            "changed_input_delta": 0.015625,
            "baseline_latency_ms": [1.0, 1.0, 1.0],
            "fused_latency_ms": ([0.5, 0.5, 0.5] if qualified else [1.0, 1.0, 1.0]),
            "max_abs_norm": 0.0,
            "norm_atol": 0.0,
            "norm_rtol": 0.001,
            "max_scaled_norm": 0.0,
            "max_abs_added": 0.0,
            "added_atol": 0.0,
            "added_rtol": 0.001,
            "max_scaled_added": 0.0,
        }

    rows = [
        row(52, 3072),
        row(56, 3072),
        row(52, 8576),
        row(56, 8576, qualified=False),
    ]
    return MC2Profile(
        metadata={
            "operator": MC2_TP3_NATIVE_EPILOGUE_OPERATOR,
            "tensor_parallel_size": 3,
            "rms_norm_epsilon": 1e-6,
            "source_sha256": "a" * 64,
            "standalone_chained_flush": True,
        },
        entries={
            (
                item["m"],
                item["k"],
                item["n"],
                item["dtype"],
                item["weight_format"],
                item["is_trans_b"],
            ): item
            for item in rows
        },
        latency_percentile=95.0,
        minimum_samples=3,
        minimum_speedup=1.02,
    )


def _mc2_target_delta(rank: int, *, histogram: dict[str, int]) -> dict:
    attention_shape = "(3072, 5120, 'bfloat16', 'ND', True)"
    down_shape = "(8576, 5120, 'bfloat16', 'ND', True)"
    return {
        "rank": rank,
        "is_draft_rank": 0,
        "mc2_static_route_digest": "route-digest",
        "mc2_static_route_count": 128,
        "mc2_static_operator_kinds": [MC2_TP3_NATIVE_EPILOGUE_OPERATOR],
        "mc2_static_qualified_row_routes": 192,
        "mc2_static_qualified_rows_by_shape": {
            attention_shape: [52, 56],
            down_shape: [52],
        },
        "mc2_static_route_counts_by_shape": {
            attention_shape: 64,
            down_shape: 64,
        },
        "mc2_dispatch_fallback_delta": 0,
        "mc2_dispatch_exception_delta": 0,
        "aclgraph_generic_replay_token_rows_histogram_delta": histogram,
        "aclgraph_target_replay_token_rows_histogram_delta": {},
    }


def test_mc2_replay_contract_intersects_every_profile_shape_family():
    contract = qualified_mc2_profile_replay_contract(_mc2_profile())

    assert contract["standalone_chained_flush"] is True
    assert contract["profile_shape_families"] == 2
    assert contract["qualified_any_projection_rows"] == [52, 56]
    assert contract["qualified_full_chain_rows"] == [52]


def test_mc2_measured_replay_coverage_is_per_target_rank_and_auditable():
    contract = qualified_mc2_profile_replay_contract(_mc2_profile())
    workers = [
        {"rank": 0, "is_draft_rank": 1},
        *[_mc2_target_delta(rank, histogram={"52": 3, "56": 7}) for rank in (1, 2, 3)],
    ]

    evidence = measure_qualified_mc2_replay_coverage(workers, contract)

    assert evidence["passed"] is True
    assert evidence["static_routes_consistent"] is True
    assert evidence["measured_replay_histograms_consistent"] is True
    assert len(evidence["target_ranks"]) == 3
    for target in evidence["target_ranks"]:
        assert target["qualified_mc2_replay_count"] == 10
        assert target["qualified_mc2_replay_coverage"] == pytest.approx(1.0)
        assert target["qualified_full_chain_replay_count"] == 3
        assert target["qualified_projection_replay_counts_by_shape"] == {
            "(3072, 5120, 'bfloat16', 'ND', True)": 10,
            "(8576, 5120, 'bfloat16', 'ND', True)": 3,
        }
        assert target["static_projection_routes_match_profile"] is True
        assert target["unqualified_replay_token_row_observations"] == {}


def test_mc2_measured_replay_coverage_fails_closed_on_rank_divergence():
    contract = qualified_mc2_profile_replay_contract(_mc2_profile())
    workers = [
        _mc2_target_delta(1, histogram={"52": 3}),
        _mc2_target_delta(2, histogram={"60": 3}),
        _mc2_target_delta(3, histogram={"52": 3}),
    ]
    workers[2]["mc2_static_route_digest"] = "different"
    workers[2]["mc2_dispatch_exception_delta"] = 1

    evidence = measure_qualified_mc2_replay_coverage(workers, contract)

    assert evidence["passed"] is False
    assert evidence["static_routes_consistent"] is False
    assert evidence["measured_replay_histograms_consistent"] is False
    assert evidence["target_ranks"][1]["qualified_mc2_replay_count"] == 0
    assert any("target rank 2" in error for error in evidence["errors"])


def test_mc2_measured_replay_coverage_requires_exact_projection_route_map():
    contract = qualified_mc2_profile_replay_contract(_mc2_profile())
    workers = [_mc2_target_delta(rank, histogram={"56": 3}) for rank in (1, 2, 3)]
    workers[1]["mc2_static_qualified_rows_by_shape"]["(3072, 5120, 'bfloat16', 'ND', True)"] = [52]

    evidence = measure_qualified_mc2_replay_coverage(workers, contract)

    assert evidence["passed"] is False
    assert evidence["target_ranks"][1]["static_projection_routes_match_profile"] is False
    assert any("projection-shape routes do not match" in error for error in evidence["errors"])


def test_under_capacity_continuous_profile_selects_actual_worker_chunk():
    profile_batch_size = _resolve_profile_batch_size(
        64,
        40,
        continuous_batching=True,
    )
    profile = _aggregate_decode_profile(
        [
            {
                "batch_size": 40,
                "worker_metrics": [
                    _profile_worker(0, is_draft=True),
                    _profile_worker(1, is_draft=False),
                    _profile_worker(2, is_draft=False),
                    _profile_worker(3, is_draft=False),
                ],
            }
        ],
        profile_batch_size,
    )

    assert profile_batch_size == 40
    assert profile is not None
    assert profile["profiled_full_batch_chunks"] == 1
    assert profile["profiled_decode_steps"] == 2
    assert profile["phase_milliseconds_per_decode_step"]["draft_compute"] == pytest.approx(10.0)


def test_static_profile_keeps_historical_full_chunk_selection():
    assert _resolve_profile_batch_size(64, 40, continuous_batching=False) == 64
    assert _resolve_profile_batch_size(64, 80, continuous_batching=False) == 64


@pytest.mark.parametrize("configured,measured", [(0, 1), (1, 0), (-1, 1), (1, -1)])
def test_profile_batch_size_rejects_non_positive_values(configured: int, measured: int):
    with pytest.raises(ValueError, match="positive"):
        _resolve_profile_batch_size(configured, measured, continuous_batching=True)


def test_gate_passes_only_matching_fixed_gamma_topology_and_paper_metrics():
    report = evaluate_gate(
        _candidate_payload(),
        _baseline_payload(),
        candidate_path="candidate.json",
        baseline_path="baseline.json",
    )

    assert report["comparability"]["passed"] is True
    assert report["observed"]["goodput_speedup"] == pytest.approx(1.3)
    assert report["observed"]["goodput_pass"] is True
    assert report["observed"]["attainment_pass"] is True
    assert report["passed"] is True


def test_gate_requires_measured_qualified_replay_for_mc2_candidate():
    candidate = _candidate_payload()
    candidate["enable_mc2"] = True
    report = evaluate_gate(
        candidate,
        _baseline_payload(),
        candidate_path="candidate.json",
        baseline_path="baseline.json",
    )

    assert report["comparability"]["passed"] is False
    contract = report["comparability"]["contract"]
    assert contract["candidate_mc2_measured_replay_coverage"]["match"] is False
    assert contract["candidate_mc2_target_rank_coverage"]["match"] is False

    evidence = measure_qualified_mc2_replay_coverage(
        [_mc2_target_delta(rank, histogram={"52": 3}) for rank in (1, 2, 3)],
        qualified_mc2_profile_replay_contract(_mc2_profile()),
    )
    candidate["results"][0]["mc2_measured_replay_coverage"] = evidence
    report = evaluate_gate(
        candidate,
        _baseline_payload(),
        candidate_path="candidate.json",
        baseline_path="baseline.json",
    )
    assert report["comparability"]["passed"] is True
    assert report["passed"] is True


def test_gate_reports_threshold_and_comparability_failures():
    candidate = _candidate_payload(goodput=119.0, attainment=0.79)
    candidate["results"][0]["prompt_token_ids_sha256"] = "different"

    report = evaluate_gate(
        candidate,
        _baseline_payload(),
        candidate_path="candidate.json",
        baseline_path="baseline.json",
        goodput_multiplier=1.2,
        min_attainment=0.8,
    )

    assert report["comparability"]["passed"] is False
    assert report["observed"]["goodput_pass"] is False
    assert report["observed"]["attainment_pass"] is False
    assert report["passed"] is False
    assert len(report["reasons"]) == 3


def test_gate_fails_closed_on_missing_runtime_environment_provenance():
    baseline = _baseline_payload()
    baseline.pop("runtime_environment")

    report = evaluate_gate(
        _candidate_payload(),
        baseline,
        candidate_path="candidate.json",
        baseline_path="legacy-baseline.json",
    )

    assert report["comparability"]["passed"] is False
    assert report["comparability"]["contract"]["baseline_missing_runtime_environment"]["match"] is False


def test_gate_rejects_mislabeled_or_non_e2e_paper_goodput():
    candidate = _candidate_payload()
    candidate["results"][0]["slo"]["paper"]["tpot_definition"] = (
        "decode_after_first_token_ms / max(1, output_tokens - 1)"
    )
    candidate["results"][0]["slo"]["paper"]["goodput_tokens_per_e2e_second"] = 131.0

    report = evaluate_gate(
        candidate,
        _baseline_payload(),
        candidate_path="candidate.json",
        baseline_path="baseline.json",
    )

    assert report["comparability"]["passed"] is False
    assert report["comparability"]["contract"]["candidate_paper_tpot_definition"]["match"] is False
    assert report["comparability"]["contract"]["candidate_paper_goodput_e2e_consistent"]["match"] is False


@pytest.mark.parametrize(
    ("multiplier", "attainment", "message"),
    [
        (0.0, 0.8, "goodput_multiplier"),
        (float("inf"), 0.8, "goodput_multiplier"),
        (1.3, -0.1, "min_attainment"),
        (1.3, 1.1, "min_attainment"),
    ],
)
def test_gate_rejects_invalid_thresholds(multiplier: float, attainment: float, message: str):
    with pytest.raises(ValueError, match=message):
        evaluate_gate(
            _candidate_payload(),
            _baseline_payload(),
            candidate_path="candidate.json",
            baseline_path="baseline.json",
            goodput_multiplier=multiplier,
            min_attainment=attainment,
        )


def test_provenance_records_cards_environment_command_and_artifact_hashes(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "examples.specslo_benchmark_report._run_git",
        lambda _repo, *args: "abc123" if args == ("rev-parse", "HEAD") else "",
    )
    result_path = tmp_path / "result.json"
    log_path = tmp_path / "run.log"
    provenance_path = tmp_path / "provenance.json"
    device_pa_path = tmp_path / "vllm_ascend/ops/triton/spec_decode/device_paged_attention.py"
    device_pa_path.parent.mkdir(parents=True)
    device_pa_path.write_text("# device PA prototype\n", encoding="utf-8")
    environment = {
        "ASCEND_RT_VISIBLE_DEVICES": "1, 2,3,4",
        "TASK_QUEUE_ENABLE": "1",
        "HCCL_DETERMINISTIC": "true",
    }
    payload = collect_provenance(
        repo=tmp_path,
        mode="formal",
        run_id="formal-1",
        result_json=result_path,
        log_file=log_path,
        command=["--", "python", "benchmark.py"],
        environment=environment,
    )
    write_json_exclusive(provenance_path, payload)
    result_path.write_text("{}\n", encoding="utf-8")
    log_path.write_text("complete\n", encoding="utf-8")

    finalized = finalize_provenance(provenance_path, exit_code=0)

    assert parse_visible_devices("1, 2,3,4") == ["1", "2", "3", "4"]
    assert finalized["cards"]["visible_devices"] == ["1", "2", "3", "4"]
    assert finalized["environment"]["TASK_QUEUE_ENABLE"] == "1"
    assert "VLLM_ASCEND_SPECSLO_ENABLE_EXPERIMENTAL_PARD_EAGER" in finalized["environment"]
    assert "VLLM_ASCEND_PEARL_DRAFT_REPLAY_FIRST_TASK_UPDATE" in finalized["environment"]
    assert "VLLM_ASCEND_PEARL_TARGET_REPLAY_FIRST_TASK_UPDATE" in finalized["environment"]
    assert "VLLM_ASCEND_SPECRHYTHM_VALIDATE_PACKED_CAUSAL_LEAKAGE" in finalized["environment"]
    assert "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_COMMON_KV" in finalized["environment"]
    assert "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_RANKED_KV" in finalized["environment"]
    assert finalized["command"] == ["python", "benchmark.py"]
    assert finalized["git"]["revision"] == "abc123"
    assert finalized["critical_source_artifacts"] == {
        "vllm_ascend/ops/triton/spec_decode/device_paged_attention.py": sha256_file(device_pa_path)
    }
    assert finalized["artifacts"]["result_json_sha256"] == sha256_file(result_path)
    assert finalized["artifacts"]["log_file_sha256"] == sha256_file(log_path)
    assert finalized["exit_code"] == 0
    json.loads(provenance_path.read_text(encoding="utf-8"))


def test_exclusive_json_writer_refuses_to_replace_existing_artifact(tmp_path: Path):
    path = tmp_path / "result.json"
    write_json_exclusive(path, {"run": 1})

    with pytest.raises(FileExistsError):
        write_json_exclusive(path, {"run": 2})

    assert json.loads(path.read_text(encoding="utf-8")) == {"run": 1}
