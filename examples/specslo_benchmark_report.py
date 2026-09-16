# SPDX-License-Identifier: Apache-2.0
"""Create auditable SpecSLO benchmark provenance and acceptance reports.

This module is deliberately NPU-independent.  The benchmark wrapper uses it
before and after a run, while unit tests can exercise the complete reporting
contract with small JSON fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PAPER_TPOT_DEFINITION = "same_decode_elapsed_ms / output_tokens"
PAPER_GOODPUT_DEFINITION = "sum(output_tokens where paper_tpot_ms <= slo_tpot_ms) / measured_e2e_seconds"
ONLINE_E2E_TIMING_SCOPE = "arrival origin, enqueue, worker IPC, prefill and decode; inputs pretokenized"


CRITICAL_ENVIRONMENT = (
    "ASCEND_RT_VISIBLE_DEVICES",
    "TASK_QUEUE_ENABLE",
    "HCCL_OP_EXPANSION_MODE",
    "HCCL_DETERMINISTIC",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VLLM_ASCEND_PEARL_ENABLE_TP3_MM_ALL_REDUCE",
    "VLLM_ASCEND_SPECSLO_ENABLE_EXPERIMENTAL_PARD_EAGER",
    "VLLM_ASCEND_PEARL_VERBOSE",
    "VLLM_ASCEND_PEARL_SYNC_GRAPH_INPUTS",
    "VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE",
    "VLLM_ASCEND_PEARL_DRAFT_REPLAY_FIRST_TASK_UPDATE",
    "VLLM_ASCEND_PEARL_DRAFT_STEP_MAJOR_PA_TASK_UPDATE",
    "VLLM_ASCEND_PEARL_TARGET_REPLAY_FIRST_TASK_UPDATE",
    "VLLM_ASCEND_PEARL_SYNC_GRAPH_TASK_UPDATE",
    "VLLM_ASCEND_PEARL_SYNC_GRAPH_REPLAY",
    "VLLM_ASCEND_PEARL_VALIDATE_GRAPH_REPLAYS",
    "VLLM_ASCEND_PEARL_SHARED_GRAPH_POOL",
    "VLLM_ASCEND_PEARL_NPU_PROFILE_DIR",
    "VLLM_ASCEND_PEARL_NPU_PROFILE_RANK",
    "VLLM_ASCEND_PEARL_NPU_PROFILE_WAIT_STEPS",
    "VLLM_ASCEND_SPECRHYTHM_TRACE_REQUEST",
    "VLLM_ASCEND_SPECRHYTHM_TREE_GRAPH",
    "VLLM_ASCEND_SPECRHYTHM_USE_FIA",
    "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_BUCKET",
    "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_COMMON_KV",
    "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_RANKED_KV",
    "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_BATCHED_MASKS",
    "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_STABLE_TASK_BARRIER",
    "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_PREFILL",
    "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH",
    "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_MAX_TOKENS",
    "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_BUCKETS",
    "VLLM_ASCEND_SPECRHYTHM_MIXED_DRAFT_PREFILL",
    "VLLM_ASCEND_SPECRHYTHM_OVERLAP_DRAFT_PREFILL",
    "VLLM_ASCEND_SPECRHYTHM_GLOO_CORRECTION",
    "VLLM_ASCEND_SPECRHYTHM_GLOO_ACCOUNTING",
    "VLLM_ASCEND_SPECRHYTHM_VALIDATE_MAILBOX",
    "VLLM_ASCEND_SPECRHYTHM_FORCE_STEPWISE_TARGET",
    "VLLM_ASCEND_SPECRHYTHM_PACKED_TARGET",
    "VLLM_ASCEND_SPECRHYTHM_DISABLE_TARGET_ACLGRAPH",
    "VLLM_ASCEND_SPECRHYTHM_STEPWISE_TARGET_FIA",
    "VLLM_ASCEND_SPECRHYTHM_VALIDATE_PACKED_CAUSAL_LEAKAGE",
    "VLLM_ASCEND_USE_NATIVE_QWEN2_ROPE",
)

SHARED_PERFORMANCE_ENVIRONMENT = (
    "ASCEND_RT_VISIBLE_DEVICES",
    "TASK_QUEUE_ENABLE",
    "HCCL_OP_EXPANSION_MODE",
    "HCCL_DETERMINISTIC",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


def capture_runtime_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    """Return a stable, exhaustive SpecSLO performance environment record."""

    source = os.environ if environment is None else environment
    return {name: source.get(name) for name in CRITICAL_ENVIRONMENT}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_visible_devices(value: str | None) -> list[str]:
    """Normalize the user-visible physical-card selection without renumbering it."""
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _run_git(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", "-C", str(repo), *args),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.rstrip("\n")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str | None:
    artifact = Path(path)
    if not artifact.is_file():
        return None
    digest = hashlib.sha256()
    with artifact.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_provenance(
    *,
    repo: str | os.PathLike[str],
    mode: str,
    run_id: str,
    result_json: str | os.PathLike[str],
    log_file: str | os.PathLike[str],
    command: Sequence[str],
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Capture the immutable pre-run context needed to audit a result."""
    repo_path = Path(repo).resolve()
    source_environment = os.environ if environment is None else environment
    selected_environment = capture_runtime_environment(source_environment)
    visible = selected_environment["ASCEND_RT_VISIBLE_DEVICES"]
    status = _run_git(repo_path, "status", "--short", "--untracked-files=all")
    diff = _run_git(repo_path, "diff", "--binary", "HEAD")
    staged_diff = _run_git(repo_path, "diff", "--binary", "--cached", "HEAD")
    recorded_command = list(command)
    if recorded_command[:1] == ["--"]:
        recorded_command = recorded_command[1:]
    return {
        "schema": "specslo-benchmark-provenance-v1",
        "run_id": run_id,
        "mode": mode,
        "started_utc": utc_now(),
        "repo": str(repo_path),
        "git": {
            "revision": _run_git(repo_path, "rev-parse", "HEAD"),
            "branch": _run_git(repo_path, "branch", "--show-current"),
            "dirty": bool(status),
            "status": [] if status is None or not status else status.splitlines(),
            "tracked_diff_sha256": (None if diff is None else _sha256_bytes(diff.encode("utf-8"))),
            "staged_diff_sha256": (None if staged_diff is None else _sha256_bytes(staged_diff.encode("utf-8"))),
        },
        "cards": {
            "visible_devices_raw": visible,
            "visible_devices": parse_visible_devices(visible),
        },
        "environment": selected_environment,
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "cwd": str(Path.cwd()),
        },
        "artifacts": {
            "result_json": str(Path(result_json).resolve()),
            "log_file": str(Path(log_file).resolve()),
        },
        "command": recorded_command,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json_exclusive(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    """Write a new JSON artifact and refuse to replace an existing run."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError:
        raise
    except BaseException:
        # ``x`` created a new file before the write failed; remove only that
        # incomplete artifact.  A pre-existing path is handled above.
        output.unlink(missing_ok=True)
        raise


def finalize_provenance(
    path: str | os.PathLike[str],
    *,
    exit_code: int,
) -> dict[str, Any]:
    """Bind the completed result and log hashes to their pre-run provenance."""
    provenance_path = Path(path)
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    artifacts = payload["artifacts"]
    payload["finished_utc"] = utc_now()
    payload["exit_code"] = int(exit_code)
    payload["artifacts"] = {
        **artifacts,
        "result_json_sha256": sha256_file(artifacts["result_json"]),
        "log_file_sha256": sha256_file(artifacts["log_file"]),
    }
    _atomic_write_json(provenance_path, payload)
    return payload


def _load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Benchmark artifact must be a JSON object: {path}")
    return value


def _result_row(payload: Mapping[str, Any], artifact_name: str) -> Mapping[str, Any]:
    rows = payload.get("results")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError(f"{artifact_name} must contain exactly one benchmark result row")
    return rows[0]


def _finite_metric(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Missing or invalid {name}") from error
    if not math.isfinite(result):
        raise ValueError(f"Missing or invalid {name}")
    return result


def _paper_slo(row: Mapping[str, Any], artifact_name: str) -> Mapping[str, Any]:
    slo = row.get("slo")
    paper = slo.get("paper") if isinstance(slo, Mapping) else None
    if not isinstance(paper, Mapping):
        raise ValueError(f"{artifact_name} does not contain results[0].slo.paper")
    return paper


def _goodput_uses_e2e_denominator(
    paper: Mapping[str, Any],
    row: Mapping[str, Any],
) -> bool:
    """Check that the published paper Goodput is bound to measured E2E time."""

    try:
        elapsed = float(row["e2e_elapsed_seconds"])
        tokens = int(paper["goodput_tokens"])
        reported = float(paper["goodput_tokens_per_e2e_second"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        math.isfinite(elapsed)
        and elapsed > 0
        and tokens >= 0
        and math.isfinite(reported)
        and math.isclose(reported, tokens / elapsed, rel_tol=1e-9, abs_tol=1e-12)
    )


def _runtime_environment_comparison(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    candidate_environment = candidate.get("runtime_environment")
    baseline_environment = baseline.get("runtime_environment")
    if not isinstance(candidate_environment, Mapping):
        candidate_environment = {}
    if not isinstance(baseline_environment, Mapping):
        baseline_environment = {}
    comparisons = {
        name: {
            "candidate": candidate_environment.get(name),
            "baseline": baseline_environment.get(name),
            "match": (
                name in candidate_environment
                and name in baseline_environment
                and candidate_environment.get(name) is not None
                and candidate_environment.get(name) == baseline_environment.get(name)
            ),
        }
        for name in SHARED_PERFORMANCE_ENVIRONMENT
    }
    candidate_missing = [name for name in CRITICAL_ENVIRONMENT if name not in candidate_environment]
    baseline_missing = [name for name in CRITICAL_ENVIRONMENT if name not in baseline_environment]
    return comparisons, candidate_missing, baseline_missing


def evaluate_gate(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    candidate_path: str,
    baseline_path: str,
    goodput_multiplier: float = 1.3,
    min_attainment: float = 0.8,
    expected_draft_tp: int | None = 1,
    expected_target_tp: int | None = 3,
    expected_baseline_tp: int | None = 4,
    expected_gamma: int | None = 4,
) -> dict[str, Any]:
    """Evaluate the paper Goodput and attainment gates with fairness checks."""
    if not math.isfinite(goodput_multiplier) or goodput_multiplier <= 0:
        raise ValueError("goodput_multiplier must be finite and positive")
    if not math.isfinite(min_attainment) or not 0 <= min_attainment <= 1:
        raise ValueError("min_attainment must be finite and in [0, 1]")

    candidate_row = _result_row(candidate, "candidate")
    baseline_row = _result_row(baseline, "baseline")
    candidate_paper = _paper_slo(candidate_row, "candidate")
    baseline_paper = _paper_slo(baseline_row, "baseline")
    candidate_goodput = _finite_metric(
        candidate_paper.get("goodput_tokens_per_e2e_second"),
        "candidate paper Goodput",
    )
    baseline_goodput = _finite_metric(
        baseline_paper.get("goodput_tokens_per_e2e_second"),
        "baseline paper Goodput",
    )
    attainment = _finite_metric(
        candidate_paper.get("attainment"),
        "candidate paper attainment",
    )
    baseline_attainment = _finite_metric(
        baseline_paper.get("attainment"),
        "baseline paper attainment",
    )
    if baseline_goodput <= 0:
        raise ValueError("baseline paper Goodput must be positive")
    if not 0 <= attainment <= 1:
        raise ValueError("candidate paper attainment must be in [0, 1]")
    if not 0 <= baseline_attainment <= 1:
        raise ValueError("baseline paper attainment must be in [0, 1]")

    environment_report, candidate_missing_environment, baseline_missing_environment = _runtime_environment_comparison(
        candidate, baseline
    )

    comparisons = {
        "prompt_token_ids_sha256": (
            candidate_row.get("prompt_token_ids_sha256"),
            baseline_row.get("prompt_token_ids_sha256"),
        ),
        "batch_size": (
            candidate_row.get("batch_size"),
            baseline_row.get("batch_size"),
        ),
        "num_prompts": (
            candidate_row.get("num_prompts"),
            baseline_row.get("num_prompts"),
        ),
        "output_tokens": (
            candidate_row.get("output_tokens"),
            baseline_row.get("output_tokens"),
        ),
        "request_manifest": (
            candidate.get("request_manifest"),
            baseline.get("request_manifest"),
        ),
        "request_manifest_sha256": (
            candidate.get("request_manifest_sha256"),
            baseline.get("request_manifest_sha256"),
        ),
        "max_tokens": (
            candidate.get("max_tokens"),
            baseline.get("max_tokens"),
        ),
        "max_model_len": (
            candidate.get("max_model_len"),
            baseline.get("max_model_len"),
        ),
        "gpu_memory_utilization": (
            candidate.get("gpu_memory_utilization"),
            baseline.get("gpu_memory_utilization"),
        ),
        "warmup_prompts": (
            candidate.get("warmup_prompts"),
            baseline.get("warmup_prompts"),
        ),
        "request_output_token_limits": (
            candidate_row.get("request_output_token_limits"),
            baseline_row.get("request_output_token_limits"),
        ),
        "warmup_output_token_limits": (
            candidate_row.get("warmup_output_token_limits"),
            baseline_row.get("warmup_output_token_limits"),
        ),
        "warmup_runs": (
            candidate.get("warmup_runs"),
            baseline.get("warmup_runs"),
        ),
        "enforce_eager": (
            candidate.get("enforce_eager"),
            baseline.get("enforce_eager"),
        ),
        "enable_prefix_caching": (
            candidate.get("enable_prefix_caching"),
            baseline.get("enable_prefix_caching"),
        ),
        "online_arrivals": (
            candidate.get("online_arrivals"),
            baseline.get("online_arrivals"),
        ),
        "warmup_excluded_from_measurement": (
            candidate.get("warmup_excluded_from_measurement"),
            baseline.get("warmup_excluded_from_measurement"),
        ),
        "e2e_timing_scope": (
            candidate_row.get("e2e_timing_scope"),
            baseline_row.get("e2e_timing_scope"),
        ),
        "target_model": (
            candidate.get("target_model"),
            baseline.get("model"),
        ),
    }
    comparison_report = {
        name: {
            "candidate": values[0],
            "baseline": values[1],
            "match": values[0] is not None and values[0] == values[1],
        }
        for name, values in comparisons.items()
    }
    topology_report = {
        "draft_tp": {
            "actual": candidate.get("draft_tensor_parallel_size"),
            "expected": expected_draft_tp,
        },
        "target_tp": {
            "actual": candidate.get("target_tensor_parallel_size"),
            "expected": expected_target_tp,
        },
        "baseline_tp": {
            "actual": baseline.get("tensor_parallel_size"),
            "expected": expected_baseline_tp,
        },
    }
    for value in topology_report.values():
        value["match"] = value["expected"] is None or value["actual"] == value["expected"]

    contract_report = {
        "candidate_backend": {
            "actual": candidate.get("backend"),
            "expected": "specslo-native-specrhythm",
        },
        "baseline_backend": {
            "actual": baseline.get("backend"),
            "expected": "vllm-ascend-target-only",
        },
        "candidate_online_arrivals": {
            "actual": candidate.get("online_arrivals"),
            "expected": True,
        },
        "baseline_online_arrivals": {
            "actual": baseline.get("online_arrivals"),
            "expected": True,
        },
        "candidate_e2e_timing_scope": {
            "actual": candidate_row.get("e2e_timing_scope"),
            "expected": ONLINE_E2E_TIMING_SCOPE,
        },
        "baseline_e2e_timing_scope": {
            "actual": baseline_row.get("e2e_timing_scope"),
            "expected": ONLINE_E2E_TIMING_SCOPE,
        },
        "candidate_warmup_excluded": {
            "actual": candidate.get("warmup_excluded_from_measurement"),
            "expected": True,
        },
        "baseline_warmup_excluded": {
            "actual": baseline.get("warmup_excluded_from_measurement"),
            "expected": True,
        },
        "gamma": {
            "actual": candidate.get("gamma"),
            "expected": expected_gamma,
        },
        "linear_full_window": {
            "actual": candidate.get("spec_rhythm_linear_full_window"),
            "expected": True,
        },
        "serial_draft": {
            "actual": candidate.get("spec_rhythm_draft_mode"),
            "expected": "serial_linear",
        },
        "candidate_slo_rows": {
            "actual": candidate_paper.get("constrained_requests"),
            "expected": candidate_row.get("num_prompts"),
        },
        "baseline_slo_rows": {
            "actual": baseline_paper.get("constrained_requests"),
            "expected": baseline_row.get("num_prompts"),
        },
        "candidate_paper_tpot_definition": {
            "actual": candidate_paper.get("tpot_definition"),
            "expected": PAPER_TPOT_DEFINITION,
        },
        "baseline_paper_tpot_definition": {
            "actual": baseline_paper.get("tpot_definition"),
            "expected": PAPER_TPOT_DEFINITION,
        },
        "candidate_paper_goodput_definition": {
            "actual": candidate_paper.get("goodput_definition"),
            "expected": PAPER_GOODPUT_DEFINITION,
        },
        "baseline_paper_goodput_definition": {
            "actual": baseline_paper.get("goodput_definition"),
            "expected": PAPER_GOODPUT_DEFINITION,
        },
        "candidate_paper_goodput_e2e_consistent": {
            "actual": _goodput_uses_e2e_denominator(
                candidate_paper,
                candidate_row,
            ),
            "expected": True,
        },
        "baseline_paper_goodput_e2e_consistent": {
            "actual": _goodput_uses_e2e_denominator(
                baseline_paper,
                baseline_row,
            ),
            "expected": True,
        },
        "candidate_missing_paper_timings": {
            "actual": candidate_paper.get("missing_timing_requests"),
            "expected": 0,
        },
        "baseline_missing_paper_timings": {
            "actual": baseline_paper.get("missing_timing_requests"),
            "expected": 0,
        },
        "candidate_missing_runtime_environment": {
            "actual": candidate_missing_environment,
            "expected": [],
        },
        "baseline_missing_runtime_environment": {
            "actual": baseline_missing_environment,
            "expected": [],
        },
        "candidate_request_limit_rows": {
            "actual": len(candidate_row.get("request_output_token_limits", ())),
            "expected": candidate_row.get("num_prompts"),
        },
        "baseline_request_limit_rows": {
            "actual": len(baseline_row.get("request_output_token_limits", ())),
            "expected": baseline_row.get("num_prompts"),
        },
        "candidate_warmup_limit_rows": {
            "actual": len(candidate_row.get("warmup_output_token_limits", ())),
            "expected": candidate.get("warmup_prompts"),
        },
        "baseline_warmup_limit_rows": {
            "actual": len(baseline_row.get("warmup_output_token_limits", ())),
            "expected": baseline.get("warmup_prompts"),
        },
    }
    for value in contract_report.values():
        value["match"] = value["expected"] is None or value["actual"] == value["expected"]

    comparable = (
        all(item["match"] for item in comparison_report.values())
        and all(item["match"] for item in environment_report.values())
        and all(item["match"] for item in topology_report.values())
        and all(item["match"] for item in contract_report.values())
    )
    required_goodput = baseline_goodput * goodput_multiplier
    speedup = candidate_goodput / baseline_goodput
    goodput_pass = speedup >= goodput_multiplier
    attainment_pass = attainment >= min_attainment
    reasons = []
    if not comparable:
        reasons.append("candidate and baseline workload/topology provenance do not match")
    if not goodput_pass:
        reasons.append(f"paper Goodput speedup {speedup:.6f}x is below {goodput_multiplier:.6f}x")
    if not attainment_pass:
        reasons.append(f"paper attainment {attainment:.6%} is below {min_attainment:.6%}")

    return {
        "schema": "specslo-benchmark-gate-v1",
        "generated_utc": utc_now(),
        "candidate": {
            "path": str(Path(candidate_path).resolve()),
            "sha256": sha256_file(candidate_path),
            "paper_goodput_tokens_per_e2e_second": candidate_goodput,
            "paper_attainment": attainment,
            "attained_requests": candidate_paper.get("attained_requests"),
            "constrained_requests": candidate_paper.get("constrained_requests"),
            "e2e_elapsed_seconds": candidate_row.get("e2e_elapsed_seconds"),
        },
        "baseline": {
            "path": str(Path(baseline_path).resolve()),
            "sha256": sha256_file(baseline_path),
            "paper_goodput_tokens_per_e2e_second": baseline_goodput,
            "paper_attainment": baseline_attainment,
            "attained_requests": baseline_paper.get("attained_requests"),
            "constrained_requests": baseline_paper.get("constrained_requests"),
            "e2e_elapsed_seconds": baseline_row.get(
                "e2e_elapsed_seconds",
                baseline_row.get("elapsed_seconds"),
            ),
        },
        "thresholds": {
            "goodput_multiplier": goodput_multiplier,
            "required_goodput_tokens_per_e2e_second": required_goodput,
            "min_attainment": min_attainment,
        },
        "observed": {
            "goodput_speedup": speedup,
            "goodput_pass": goodput_pass,
            "attainment_pass": attainment_pass,
        },
        "comparability": {
            "passed": comparable,
            "workload": comparison_report,
            "environment": environment_report,
            "topology": topology_report,
            "contract": contract_report,
        },
        "passed": comparable and goodput_pass and attainment_pass,
        "reasons": reasons,
    }


def _provenance_command(args: argparse.Namespace) -> int:
    payload = collect_provenance(
        repo=args.repo,
        mode=args.mode,
        run_id=args.run_id,
        result_json=args.result_json,
        log_file=args.log_file,
        command=args.command,
    )
    write_json_exclusive(args.output, payload)
    return 0


def _finalize_command(args: argparse.Namespace) -> int:
    finalize_provenance(args.provenance, exit_code=args.exit_code)
    return 0


def _gate_command(args: argparse.Namespace) -> int:
    candidate = _load_json(args.candidate)
    baseline = _load_json(args.baseline)
    report = evaluate_gate(
        candidate,
        baseline,
        candidate_path=args.candidate,
        baseline_path=args.baseline,
        goodput_multiplier=args.goodput_multiplier,
        min_attainment=args.min_attainment,
        expected_draft_tp=args.expected_draft_tp,
        expected_target_tp=args.expected_target_tp,
        expected_baseline_tp=args.expected_baseline_tp,
        expected_gamma=args.expected_gamma,
    )
    write_json_exclusive(args.output, report)
    print(
        "SpecSLO gate: "
        f"speedup={report['observed']['goodput_speedup']:.6f}x "
        f"(required={report['thresholds']['goodput_multiplier']:.6f}x), "
        f"attainment={report['candidate']['paper_attainment']:.6%} "
        f"(required={report['thresholds']['min_attainment']:.6%}), "
        f"comparable={report['comparability']['passed']}, "
        f"passed={report['passed']}"
    )
    for reason in report["reasons"]:
        print(f"- {reason}")
    return int(args.require_pass and not report["passed"])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)

    provenance = commands.add_parser("provenance", help="write immutable pre-run provenance")
    provenance.add_argument("--repo", required=True)
    provenance.add_argument("--mode", required=True)
    provenance.add_argument("--run-id", required=True)
    provenance.add_argument("--result-json", required=True)
    provenance.add_argument("--log-file", required=True)
    provenance.add_argument("--output", required=True)
    provenance.add_argument("command", nargs=argparse.REMAINDER)
    provenance.set_defaults(callback=_provenance_command)

    finalize = commands.add_parser("finalize", help="record exit status and artifact hashes")
    finalize.add_argument("--provenance", required=True)
    finalize.add_argument("--exit-code", required=True, type=int)
    finalize.set_defaults(callback=_finalize_command)

    gate = commands.add_parser("gate", help="compare candidate paper metrics with a baseline")
    gate.add_argument("--candidate", required=True)
    gate.add_argument("--baseline", required=True)
    gate.add_argument("--output", required=True)
    gate.add_argument("--goodput-multiplier", type=float, default=1.3)
    gate.add_argument("--min-attainment", type=float, default=0.8)
    gate.add_argument("--expected-draft-tp", type=int, default=1)
    gate.add_argument("--expected-target-tp", type=int, default=3)
    gate.add_argument("--expected-baseline-tp", type=int, default=4)
    gate.add_argument("--expected-gamma", type=int, default=4)
    gate.add_argument("--require-pass", action="store_true")
    gate.set_defaults(callback=_gate_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return int(args.callback(args))


if __name__ == "__main__":
    raise SystemExit(main())
