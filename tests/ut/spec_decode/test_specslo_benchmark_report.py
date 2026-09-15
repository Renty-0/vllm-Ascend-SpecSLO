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
    evaluate_gate,
    finalize_provenance,
    parse_visible_devices,
    sha256_file,
    write_json_exclusive,
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
    assert report["comparability"]["contract"][
        "baseline_missing_runtime_environment"
    ]["match"] is False


def test_gate_rejects_mislabeled_or_non_e2e_paper_goodput():
    candidate = _candidate_payload()
    candidate["results"][0]["slo"]["paper"]["tpot_definition"] = (
        "decode_after_first_token_ms / max(1, output_tokens - 1)"
    )
    candidate["results"][0]["slo"]["paper"][
        "goodput_tokens_per_e2e_second"
    ] = 131.0

    report = evaluate_gate(
        candidate,
        _baseline_payload(),
        candidate_path="candidate.json",
        baseline_path="baseline.json",
    )

    assert report["comparability"]["passed"] is False
    assert report["comparability"]["contract"][
        "candidate_paper_tpot_definition"
    ]["match"] is False
    assert report["comparability"]["contract"][
        "candidate_paper_goodput_e2e_consistent"
    ]["match"] is False


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
    assert (
        "VLLM_ASCEND_PEARL_DRAFT_REPLAY_FIRST_TASK_UPDATE"
        in finalized["environment"]
    )
    assert (
        "VLLM_ASCEND_PEARL_TARGET_REPLAY_FIRST_TASK_UPDATE"
        in finalized["environment"]
    )
    assert (
        "VLLM_ASCEND_SPECRHYTHM_VALIDATE_PACKED_CAUSAL_LEAKAGE"
        in finalized["environment"]
    )
    assert finalized["command"] == ["python", "benchmark.py"]
    assert finalized["git"]["revision"] == "abc123"
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
