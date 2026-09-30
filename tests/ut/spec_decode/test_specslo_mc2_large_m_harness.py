# SPDX-License-Identifier: Apache-2.0
"""Static contracts for the v134 large-M qualification/formal harness."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS = REPO_ROOT / "tools/specslo_experiments"
QUALIFY = TOOLS / "qualify_mc2_multishape.sh"
BASELINE = TOOLS / "run_online_coalesce1_baseline.sh"
CANDIDATE = TOOLS / "run_online_coalesce1_candidate.sh"
FORMAL = TOOLS / "run_online_coalesce1_formal_ab.sh"
NATIVE_ENGINE = REPO_ROOT / "vllm_ascend/spec_decode/pearl/native_engine.py"

DEFAULT_QUALIFICATION_ROOT = "/root/data/nano-pearl-benchmark-results/20260927-specslo-mc2-v134-large-m-qualification"
DEFAULT_ONLINE_ROOT = "/root/data/nano-pearl-benchmark-results/20260927-specslo-v134-large-m-online-coalesce1"
DEFAULT_PROFILE = (
    f"{DEFAULT_QUALIFICATION_ROOT}/runs/"
    "production-current-target345-v134-large-m-s31/"
    "qwen3-32b-tp3-v134-large-m-production-profile.json"
)
DEFAULT_VENDOR = (
    "/root/data/specslo-mc2-v134-large-m/package-extracted/packages/vendors/specslo_mc2_v134_large_m_transformer"
)


def _formal_final_gate_python() -> str:
    source = FORMAL.read_text(encoding="utf-8")
    marker = 'write_final_gate() {\n  "$python" - "$campaign_dir" <<\'PY\'\n'
    tail = "\nPY\n}\n\nif [[ $mode == validate-only ]]"
    assert marker in source
    body = source.split(marker, 1)[1]
    assert tail in body
    return body.split(tail, 1)[0]


def _write_round_gates(
    campaign_dir: Path,
    *,
    raw_speedups: list[float],
    goodput_speedups: list[float],
) -> None:
    for round_number, (raw_speedup, goodput_speedup) in enumerate(
        zip(raw_speedups, goodput_speedups, strict=True),
        start=1,
    ):
        payload = {
            "status": "pass",
            "round": round_number,
            "source_identity_sha256": "source",
            "manifest_sha256": "manifest",
            "profile_sha256": "profile",
            "vendor_sha256": "vendor",
            "physical_npus": [2, 3, 4, 5],
            "draft_pa_task_update_workers": 1,
            "target_replay_ahead_enabled": 0,
            "raw_speedup": raw_speedup,
            "paper_goodput_speedup": goodput_speedup,
        }
        (campaign_dir / f"round-{round_number}-gate.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )


@pytest.mark.parametrize("launcher", (QUALIFY, BASELINE, CANDIDATE, FORMAL))
def test_v134_harness_shell_is_syntactically_valid(launcher: Path):
    result = subprocess.run(
        ["bash", "-n", str(launcher)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_v134_defaults_are_isolated_from_v133_artifacts():
    qualify = QUALIFY.read_text(encoding="utf-8")
    baseline = BASELINE.read_text(encoding="utf-8")
    candidate = CANDIDATE.read_text(encoding="utf-8")
    formal = FORMAL.read_text(encoding="utf-8")

    for source in (qualify, baseline, candidate, formal):
        assert "v133" not in source
    assert DEFAULT_QUALIFICATION_ROOT in qualify
    assert DEFAULT_ONLINE_ROOT in baseline
    assert DEFAULT_ONLINE_ROOT in candidate
    assert DEFAULT_ONLINE_ROOT in formal
    assert DEFAULT_PROFILE in candidate
    assert DEFAULT_PROFILE in formal
    assert DEFAULT_VENDOR in qualify
    assert DEFAULT_VENDOR in candidate
    assert DEFAULT_VENDOR in formal


def test_qualification_records_profile_vendor_and_result_root_contracts():
    source = QUALIFY.read_text(encoding="utf-8")

    assert "profile_filename=${SPECSLO_MC2_PROFILE_FILENAME:-" in source
    assert "vendor=${SPECSLO_MC2_VENDOR:-" in source
    assert "result_root=${SPECSLO_QUALIFY_ROOT:-" in source
    assert "result_root=%s\\nrun_dir=%s\\nprofile_filename=%s\\n" in source
    assert "attention_root=%s\\ndown_root=%s\\nvendor=%s\\n" in source
    assert "profile_output=%s\\n" in source
    assert "from examples.aggregate_specslo_mc2_profiles import aggregate_mc2_profiles" in source
    assert 'run_dir / "cross-layer-screen.json"' in source
    assert 'run_dir / "qualified-rows.txt"' in source
    assert (
        "no projection/M passed the conservative cross-layer production envelope"
        in source
    )
    assert "production-projection-rows.json" in source


def test_candidate_cann_profile_is_explicit_and_measurement_only():
    source = CANDIDATE.read_text(encoding="utf-8")

    assert "profile_host_decode_steps=${SPECSLO_PROFILE_HOST_DECODE_STEPS:-0}" in source
    assert "enable_npu_profile=${SPECSLO_ENABLE_NPU_PROFILE:-1}" in source
    assert "profile_args=(--profile-host-decode-steps" in source
    assert "export VLLM_ASCEND_PEARL_NPU_PROFILE_MEASUREMENT_ONLY=1" in source
    assert "unset VLLM_ASCEND_PEARL_NPU_PROFILE_DIR" in source
    assert "unset VLLM_ASCEND_PEARL_NPU_PROFILE_MEASUREMENT_ONLY" in source
    assert "profile_host_decode_steps=%s\\nnpu_profile_wait_steps=%s\\nenable_npu_profile=%s\\n" in source
    assert "npu_profile_measurement_only=%s\\nnpu_profile_dir=%s\\n" in source


def test_candidate_target_fia_worker_experiment_is_explicit_and_fail_closed():
    source = CANDIDATE.read_text(encoding="utf-8")

    assert "target_fia_task_update_workers=${SPECSLO_TARGET_FIA_TASK_UPDATE_WORKERS:-1}" in source
    assert "SPECSLO_TARGET_FIA_TASK_UPDATE_WORKERS must be 1, 2, 3, or 4" in source
    assert "export VLLM_ASCEND_PEARL_TARGET_FIA_TASK_UPDATE_WORKERS=$target_fia_task_update_workers" in source
    assert "target_fia_task_update_workers=%s\\n" in source
    assert "target_fia_parallel_update_evidence(" in source


def test_candidate_linear_bonus_token_experiment_is_explicit_and_fail_closed():
    source = CANDIDATE.read_text(encoding="utf-8")

    assert "linear_bonus_token=${SPECSLO_LINEAR_BONUS_TOKEN:-0}" in source
    assert "bonus_args=(--spec-rhythm-linear-bonus-token)" in source
    assert "linear_bonus_token=%s\\n" in source
    assert "645 + 32 * linear_bonus_token" in source
    assert "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_MAX_TOKENS=$mixed_target_graph_max_tokens" in source
    assert 'errors.append("linear bonus-token contract mismatch")' in source
    assert 'errors.append("mixed-target graph token-capacity contract mismatch")' in source
    assert '"linear_bonus_token": bonus_evidence' in source


def test_candidate_target_cohort_experiment_is_explicit_and_fail_closed():
    source = CANDIDATE.read_text(encoding="utf-8")

    assert "target_max_batch=${SPECSLO_MAX_TARGET_BATCH:-32}" in source
    assert "SPECSLO_MAX_TARGET_BATCH must be 32, 48, or 64" in source
    assert "mixed_target_verify_capacities=8,16,24,32" in source
    assert "VLLM_ASCEND_SPECRHYTHM_STABLE_TARGET_VERIFY_MAX_CAPACITY=$target_max_batch" in source
    assert "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_VERIFY_CAPACITIES=$mixed_target_verify_capacities" in source
    assert '--spec-rhythm-max-target-batch "$target_max_batch"' in source
    assert 'errors.append("target max-batch contract mismatch")' in source
    assert 'errors.append("stable target max-capacity contract mismatch")' in source
    assert 'errors.append("mixed-target verify-capacity contract mismatch")' in source


def test_mixed_target_cohort_is_a_world_wide_scheduler_contract():
    source = NATIVE_ENGINE.read_text(encoding="utf-8")

    assert "self._mixed_target_graph_collective_enabled = bool(" in source
    assert "not self.is_draft and self._mixed_target_graph_collective_enabled" in source
    assert 'getattr(self, "_mixed_target_graph_collective_enabled", False)' in source
    assert "spec_rhythm_mixed_target_scheduler_cap_cycles" in source
    assert "spec_rhythm_mixed_target_scheduler_deferred_slots" in source


def test_candidate_rejects_invalid_target_cohort_before_npu_use():
    environment = os.environ.copy()
    environment["SPECSLO_MAX_TARGET_BATCH"] = "40"
    result = subprocess.run(
        ["bash", str(CANDIDATE), "on", "invalid-target-cohort"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "SPECSLO_MAX_TARGET_BATCH must be 32, 48, or 64" in result.stderr


def test_candidate_acceptance_floor_experiment_is_explicit_and_provenanced():
    source = CANDIDATE.read_text(encoding="utf-8")

    assert "acceptance_floor=${SPECSLO_ACCEPTANCE_FLOOR:-0.4}" in source
    assert '--spec-rhythm-acceptance-floor "$acceptance_floor"' in source
    assert "acceptance_floor=%s\\n" in source
    assert 'errors.append("acceptance-floor contract mismatch")' in source


def test_candidate_rejects_invalid_acceptance_floor_before_npu_use():
    environment = os.environ.copy()
    environment["SPECSLO_ACCEPTANCE_FLOOR"] = "1.1"
    result = subprocess.run(
        ["bash", str(CANDIDATE), "on", "invalid-acceptance-floor"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "SPECSLO_ACCEPTANCE_FLOOR must be a decimal in [0, 1]" in result.stderr


def test_candidate_prefetch_ahead_experiment_is_explicit_and_provenanced():
    source = CANDIDATE.read_text(encoding="utf-8")

    assert "prefetch_ahead=${SPECSLO_PREFETCH_AHEAD:-0}" in source
    assert "export VLLM_ASCEND_SPECRHYTHM_PREFETCH_AHEAD=$prefetch_ahead" in source
    assert "prefetch_ahead=%s\\n" in source
    assert 'errors.append("prefetch-ahead contract mismatch")' in source


def test_candidate_rejects_invalid_target_fia_worker_count_before_npu_use():
    environment = os.environ.copy()
    environment["SPECSLO_TARGET_FIA_TASK_UPDATE_WORKERS"] = "5"
    result = subprocess.run(
        ["bash", str(CANDIDATE), "on", "invalid-fia-workers"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "SPECSLO_TARGET_FIA_TASK_UPDATE_WORKERS must be 1, 2, 3, or 4" in result.stderr


def test_candidate_draft_pa_worker_experiment_is_explicit_and_fail_closed():
    source = CANDIDATE.read_text(encoding="utf-8")

    assert "draft_pa_task_update_workers=${SPECSLO_DRAFT_PA_TASK_UPDATE_WORKERS:-1}" in source
    assert "SPECSLO_DRAFT_PA_TASK_UPDATE_WORKERS must be 1, 2, or 4" in source
    assert "export VLLM_ASCEND_PEARL_DRAFT_PA_TASK_UPDATE_WORKERS=$draft_pa_task_update_workers" in source
    assert "draft_pa_task_update_workers=%s\\n" in source
    assert "draft_pa_parallel_update_evidence(" in source
    assert '"draft_pa_task_update": draft_pa_evidence' in source


def test_candidate_rejects_invalid_draft_pa_worker_count_before_npu_use():
    environment = os.environ.copy()
    environment["SPECSLO_DRAFT_PA_TASK_UPDATE_WORKERS"] = "3"
    result = subprocess.run(
        ["bash", str(CANDIDATE), "on", "invalid-draft-pa-workers"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "SPECSLO_DRAFT_PA_TASK_UPDATE_WORKERS must be 1, 2, or 4" in result.stderr


def test_candidate_rejects_invalid_profile_step_count_before_npu_use():
    environment = os.environ.copy()
    environment["SPECSLO_PROFILE_HOST_DECODE_STEPS"] = "not-an-integer"
    result = subprocess.run(
        ["bash", str(CANDIDATE), "on", "invalid-profile-steps"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "profiling step counts must be non-negative integers" in result.stderr


def test_candidate_unified_drain_lrtf_is_explicit_provenanced_and_gated():
    candidate = CANDIDATE.read_text(encoding="utf-8")
    formal = FORMAL.read_text(encoding="utf-8")

    assert "drain_lrtf=${SPECSLO_DRAIN_LRTF:-0}" in candidate
    assert "drain_lrtf_max_active=${SPECSLO_DRAIN_LRTF_MAX_ACTIVE:-0}" in candidate
    assert "export VLLM_ASCEND_SPECRHYTHM_DRAIN_LRTF=$drain_lrtf" in candidate
    assert "export VLLM_ASCEND_SPECRHYTHM_DRAIN_LRTF_MAX_ACTIVE=$drain_lrtf_max_active" in candidate
    assert "drain_fifo=%s\\ndrain_lrtf=%s\\n" in candidate
    for metric in (
        "target_ranked_cycles",
        "normal_ranked_cycles",
        "unified_cycles",
        "target_candidates",
        "normal_candidates",
        "repair_penalty_rows",
        "normal_selected",
        "normal_cross_home_selected",
    ):
        assert f"worker_spec_rhythm_drain_lrtf_{metric}" in candidate
    for field in (
        "target_ranked_cycles_by_rank",
        "normal_ranked_cycles_by_rank",
        "unified_cycles_by_rank",
        "target_candidates_by_rank",
        "normal_candidates_by_rank",
        "repair_penalty_rows_by_rank",
        "normal_selected_by_rank",
        "normal_cross_home_selected_by_rank",
    ):
        assert field in candidate
        assert field in formal


def test_candidate_rejects_drain_lrtf_without_immutable_drain_fifo():
    environment = os.environ.copy()
    environment["SPECSLO_DRAIN_LRTF"] = "1"
    environment["SPECSLO_DRAIN_FIFO"] = "0"
    result = subprocess.run(
        ["bash", str(CANDIDATE), "on", "invalid-drain-lrtf"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "SPECSLO_DRAIN_LRTF=1 requires SPECSLO_DRAIN_FIFO=1" in result.stderr


def test_candidate_rejects_drain_lrtf_active_guard_without_lrtf():
    environment = os.environ.copy()
    environment["SPECSLO_DRAIN_LRTF"] = "0"
    environment["SPECSLO_DRAIN_LRTF_MAX_ACTIVE"] = "16"
    result = subprocess.run(
        ["bash", str(CANDIDATE), "on", "invalid-drain-lrtf-active-guard"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "SPECSLO_DRAIN_LRTF_MAX_ACTIVE requires SPECSLO_DRAIN_LRTF=1" in result.stderr


@pytest.mark.parametrize("profile_filename", ("../profile.json", "nested/p.json"))
def test_qualification_rejects_non_basename_profile_overrides(profile_filename: str):
    environment = os.environ.copy()
    environment["SPECSLO_MC2_PROFILE_FILENAME"] = profile_filename
    result = subprocess.run(
        ["bash", str(QUALIFY), "/missing-a", "/missing-b", "static-contract"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "SPECSLO_MC2_PROFILE_FILENAME must be a JSON basename" in result.stderr


def test_formal_dry_run_preserves_and_records_all_path_overrides(tmp_path: Path):
    result_root = tmp_path / "results"
    profile = tmp_path / "qualified-profile.json"
    vendor = tmp_path / "vendor"
    environment = os.environ.copy()
    environment.update(
        {
            "SPECSLO_RESULT_ROOT": str(result_root),
            "SPECSLO_MC2_PROFILE": str(profile),
            "SPECSLO_MC2_VENDOR": str(vendor),
            "SPECSLO_VISIBLE_DEVICES": "2,3,4,5",
            "SPECSLO_DRAFT_PA_TASK_UPDATE_WORKERS": "2",
        }
    )

    result = subprocess.run(
        ["bash", str(FORMAL), "--dry-run", "override-contract"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert f"result_root={result_root}" in result.stdout
    assert f"profile={profile}" in result.stdout
    assert f"vendor={vendor}" in result.stdout
    assert "physical_npus=2,3,4,5" in result.stdout
    assert f"campaign_dir={result_root}/formal-ab-override-contract" in result.stdout
    assert "draft_pa_task_update_workers=2" in result.stdout


def test_formal_rejects_invalid_draft_pa_worker_count_before_npu_use():
    environment = os.environ.copy()
    environment["SPECSLO_DRAFT_PA_TASK_UPDATE_WORKERS"] = "3"
    result = subprocess.run(
        ["bash", str(FORMAL), "--dry-run", "invalid-draft-pa-workers"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "SPECSLO_DRAFT_PA_TASK_UPDATE_WORKERS must be 1, 2, or 4" in result.stderr


def test_formal_harness_snapshots_validates_and_exports_vendor():
    source = FORMAL.read_text(encoding="utf-8")

    assert "[[ ! -d $vendor || ! -f $vendor/op_api/lib/libcust_opapi.so ]]" in source
    assert '"vendor": str(vendor.resolve())' in source
    assert '"vendor_sha256": vendor_sha' in source
    assert 'vendor_path = Path(identity["vendor"])' in source
    assert 'candidate_contract.get("vendor", "")' in source
    assert "export SPECSLO_MC2_VENDOR=$vendor" in source
    assert "SPECSLO_PROFILE_HOST_DECODE_STEPS=0" in source
    assert "draft_pa_task_update_workers=${SPECSLO_DRAFT_PA_TASK_UPDATE_WORKERS:-1}" in source
    assert "export SPECSLO_DRAFT_PA_TASK_UPDATE_WORKERS=$draft_pa_task_update_workers" in source
    assert '"draft_pa_task_update_workers": draft_pa_task_update_workers' in source
    assert 'candidate_contract.get("draft_pa_task_update_workers")' in source
    assert 'candidate_gate.get("draft_pa_task_update", {})' in source


def test_formal_round_gate_recomputes_each_result_token_digest():
    source = FORMAL.read_text(encoding="utf-8")

    assert 'token_rows = row.get("output_token_ids")' in source
    assert 'json.dumps(token_rows, separators=(",", ":"))' in source
    assert 'row.get("output_token_ids_sha256") == recomputed_output_sha' in source
    assert "stored output-token SHA256 differs from the serialized rows" in source


@pytest.mark.parametrize(
    ("raw_speedups", "goodput_speedups", "expected_error"),
    (
        (
            [1.01, 1.0, 1.02],
            [1.3, 1.31, 1.32],
            "not every round has Raw(candidate) > Raw(paired baseline)",
        ),
        (
            [1.01, 1.02, 1.03],
            [1.3, 1.2999, 1.31],
            "not every round has paper Goodput(candidate) >= 1.3x paired baseline",
        ),
    ),
)
def test_formal_final_gate_rejects_a_single_round_below_threshold(
    tmp_path: Path,
    raw_speedups: list[float],
    goodput_speedups: list[float],
    expected_error: str,
):
    _write_round_gates(
        tmp_path,
        raw_speedups=raw_speedups,
        goodput_speedups=goodput_speedups,
    )

    result = subprocess.run(
        [sys.executable, "-", str(tmp_path)],
        input=_formal_final_gate_python(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    gate = json.loads((tmp_path / "formal-ab-gate.json").read_text())
    assert gate["status"] == "fail"
    assert expected_error in gate["errors"]


def test_formal_final_gate_accepts_only_when_all_rounds_meet_thresholds(
    tmp_path: Path,
):
    _write_round_gates(
        tmp_path,
        raw_speedups=[1.001, 1.01, 1.02],
        goodput_speedups=[1.3, 1.31, 1.32],
    )

    result = subprocess.run(
        [sys.executable, "-", str(tmp_path)],
        input=_formal_final_gate_python(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    gate = json.loads((tmp_path / "formal-ab-gate.json").read_text())
    assert gate["status"] == "pass"
    assert gate["all_rounds_raw_above_baseline"] is True
    assert gate["all_rounds_paper_goodput_at_least_1_3x"] is True
