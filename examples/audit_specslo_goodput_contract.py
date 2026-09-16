# SPDX-License-Identifier: Apache-2.0
"""Audit the finite-trace ceiling of a SpecSLO Goodput comparison.

This helper is CPU-only.  It does not run either inference backend and does
not reinterpret the benchmark's Goodput definition.  The online benchmark
starts its E2E clock before admitting requests whose manifest timestamps are
offset from that clock.  Consequently, even a zero-service-time system cannot
finish the trace before the last arrival offset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object in {path}")
    return value


def _paper_metrics(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise ValueError("Benchmark result must contain exactly one result row")
    row = results[0]
    slo = row.get("slo")
    paper = slo.get("paper") if isinstance(slo, dict) else None
    if not isinstance(paper, dict):
        raise ValueError("Benchmark result is missing slo.paper metrics")
    return row, paper


def _candidate_contract_checks(
    candidate: dict[str, Any],
    candidate_row: dict[str, Any],
    candidate_paper: dict[str, Any],
    *,
    baseline: dict[str, Any],
    baseline_row: dict[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    expected_coalesce_min_requests: int,
    expected_coalesce_max_wait_ms: float,
) -> dict[str, bool]:
    candidate_goodput = float(candidate_paper["goodput_tokens_per_e2e_second"])
    return {
        "manifest_sha256_matches": candidate.get("request_manifest_sha256") == manifest_sha256,
        "manifest_path_matches": str(candidate.get("request_manifest")) == str(manifest_path),
        "num_prompts_matches_baseline": (candidate_row.get("num_prompts") == baseline_row.get("num_prompts")),
        "batch_size_matches_baseline": (candidate_row.get("batch_size") == baseline_row.get("batch_size")),
        "output_tokens_match_baseline": (candidate_row.get("output_tokens") == baseline_row.get("output_tokens")),
        "prompt_tokens_match_baseline": (
            candidate_row.get("prompt_token_ids_sha256") == baseline_row.get("prompt_token_ids_sha256")
        ),
        "max_tokens_matches_baseline": candidate.get("max_tokens") == baseline.get("max_tokens"),
        "candidate_coalesce_min_matches": (
            candidate.get("spec_rhythm_prefill_coalesce_min_requests") == expected_coalesce_min_requests
        ),
        "candidate_coalesce_wait_matches": math.isclose(
            float(candidate.get("spec_rhythm_prefill_coalesce_max_wait_ms", math.nan)),
            expected_coalesce_max_wait_ms,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "paper_goodput_uses_e2e": math.isclose(
            candidate_goodput,
            float(candidate_paper["goodput_tokens"]) / float(candidate_row["e2e_elapsed_seconds"]),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ),
    }


def audit(
    manifest_path: Path,
    baseline_path: Path,
    multiplier: float,
    *,
    candidate_path: Path | None = None,
    expected_coalesce_min_requests: int = 4,
    expected_coalesce_max_wait_ms: float = 1100.0,
) -> dict[str, Any]:
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("--multiplier must be finite and positive")
    manifest_bytes = manifest_path.read_bytes()
    rows = [json.loads(line) for line in manifest_bytes.splitlines() if line.strip()]
    if not rows:
        raise ValueError("Manifest must contain at least one request")
    arrivals = [float(row["arrival_offset_sec"]) for row in rows]
    token_limits = [int(row["max_tokens"]) for row in rows]
    if any(not math.isfinite(value) or value < 0 for value in arrivals):
        raise ValueError("Arrival offsets must be finite and non-negative")
    if any(value <= 0 for value in token_limits):
        raise ValueError("Every request must have a positive max_tokens")

    baseline = _load_json(baseline_path)
    baseline_row, baseline_paper = _paper_metrics(baseline)
    baseline_goodput = float(baseline_paper["goodput_tokens_per_e2e_second"])
    required_goodput = baseline_goodput * multiplier
    total_tokens = sum(token_limits)
    last_arrival = max(arrivals)
    first_arrival = min(arrivals)
    arrival_span = last_arrival - first_arrival
    timer_origin_ceiling = math.inf if last_arrival == 0 else total_tokens / last_arrival
    relaxed_span_ceiling = math.inf if arrival_span == 0 else total_tokens / arrival_span
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    checks = {
        "manifest_sha256_matches": baseline.get("request_manifest_sha256") == manifest_sha256,
        "manifest_path_matches": str(baseline.get("request_manifest")) == str(manifest_path),
        "num_prompts_matches": baseline_row.get("num_prompts") == len(rows),
        "output_tokens_match_limits": baseline_row.get("output_tokens") == total_tokens,
        "max_tokens_matches": (len(set(token_limits)) == 1 and baseline.get("max_tokens") == token_limits[0]),
        "paper_goodput_uses_e2e": math.isclose(
            baseline_goodput,
            float(baseline_paper["goodput_tokens"]) / float(baseline_row["e2e_elapsed_seconds"]),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ),
        "baseline_coalesce_min_matches": (
            baseline.get("prefill_coalesce_min_requests") == expected_coalesce_min_requests
        ),
        "baseline_coalesce_wait_matches": math.isclose(
            float(baseline.get("prefill_coalesce_max_wait_ms", math.nan)),
            expected_coalesce_max_wait_ms,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
    }
    report = {
        "schema": "specslo-finite-goodput-contract-audit-v1",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "num_requests": len(rows),
        "total_output_token_limit": total_tokens,
        "first_arrival_offset_seconds": first_arrival,
        "last_arrival_offset_seconds": last_arrival,
        "arrival_span_seconds": arrival_span,
        "baseline_result": str(baseline_path),
        "baseline_goodput_tokens_per_e2e_second": baseline_goodput,
        "multiplier": multiplier,
        "required_candidate_goodput_tokens_per_e2e_second": required_goodput,
        "zero_service_ceilings": {
            "timer_origin_tokens_per_second": timer_origin_ceiling,
            "timer_origin_max_speedup_over_baseline": timer_origin_ceiling / baseline_goodput,
            "first_to_last_arrival_tokens_per_second_relaxed": relaxed_span_ceiling,
            "first_to_last_arrival_max_speedup_over_baseline_relaxed": (relaxed_span_ceiling / baseline_goodput),
        },
        "mathematically_feasible_under_timer_origin_ceiling": (timer_origin_ceiling >= required_goodput),
        "baseline_contract_checks": checks,
        "baseline_contract_checks_pass": all(checks.values()),
        "expected_admission_coalescing": {
            "minimum_requests": expected_coalesce_min_requests,
            "maximum_wait_ms": expected_coalesce_max_wait_ms,
        },
    }
    if candidate_path is not None:
        candidate = _load_json(candidate_path)
        candidate_row, candidate_paper = _paper_metrics(candidate)
        candidate_goodput = float(candidate_paper["goodput_tokens_per_e2e_second"])
        candidate_checks = _candidate_contract_checks(
            candidate,
            candidate_row,
            candidate_paper,
            baseline=baseline,
            baseline_row=baseline_row,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            expected_coalesce_min_requests=expected_coalesce_min_requests,
            expected_coalesce_max_wait_ms=expected_coalesce_max_wait_ms,
        )
        report["candidate_result"] = str(candidate_path)
        report["candidate_goodput_tokens_per_e2e_second"] = candidate_goodput
        report["observed_goodput_speedup"] = candidate_goodput / baseline_goodput
        report["candidate_contract_checks"] = candidate_checks
        report["candidate_contract_checks_pass"] = all(candidate_checks.values())
        report["goodput_gate_pass"] = (
            all(checks.values()) and all(candidate_checks.values()) and candidate_goodput >= required_goodput
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--multiplier", type=float, default=1.3)
    parser.add_argument("--expected-coalesce-min-requests", type=int, default=4)
    parser.add_argument("--expected-coalesce-max-wait-ms", type=float, default=1100.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(
        args.manifest.resolve(),
        args.baseline.resolve(),
        args.multiplier,
        candidate_path=args.candidate.resolve() if args.candidate is not None else None,
        expected_coalesce_min_requests=args.expected_coalesce_min_requests,
        expected_coalesce_max_wait_ms=args.expected_coalesce_max_wait_ms,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
