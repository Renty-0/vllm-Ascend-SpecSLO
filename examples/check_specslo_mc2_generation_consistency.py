# SPDX-License-Identifier: Apache-2.0
"""Compare complete generation outputs from MC2-off and MC2-on runs.

The native SpecSLO benchmark stores every completion row as well as a compact
SHA256.  This CPU-only diagnostic recomputes both hashes, identifies the first
divergent token, and emits a small auditable report instead of relying on the
hash alone.

Optional target-token diagnostic JSON can add the target argmax and top-2
margin at the divergent coordinate.  The trace may contain either compact
``top1``/``top2`` records or a full ``logits`` row.  This script never changes
model or sampling semantics and does not claim that a missing trace contains
logit evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

# Fields which must remain fixed for MC2 to be the intended independent
# variable.  Missing fields are reported too; older/minimal payloads therefore
# cannot silently pass the comparability gate.
RUN_CONTRACT_FIELDS = (
    "backend",
    "policy",
    "draft_model",
    "target_model",
    "draft_tensor_parallel_size",
    "target_tensor_parallel_size",
    "draft_dtype",
    "target_dtype",
    "draft_mode",
    "draft_tp1_greedy_argmax",
    "gamma",
    "seed",
    "target_verification_graph_buckets",
    "target_verification_graph_post_counts",
    "max_model_len",
    "prefill_chunk_size",
    "enforce_eager",
    "require_no_graph_fallback",
    "enable_prefix_caching",
    "enable_continuous_batching",
    "enable_preemptive_scheduling",
    "enable_spec_rhythm",
    "spec_rhythm_ablation_mode",
    "spec_rhythm_linear_full_window",
    "spec_rhythm_online_prefill",
    "spec_rhythm_kv_ready_arrivals",
    "spec_rhythm_stable_graphs",
    "spec_rhythm_priority_mode_resolved",
    "spec_rhythm_min_gamma",
    "spec_rhythm_tree_width",
    "spec_rhythm_tree_depth",
    "draft_use_paged_attention",
    "target_use_paged_attention",
    "max_tokens",
    "respect_eos",
    "ignore_eos",
    "saturated_arrivals",
    "arrival_mode",
    "request_manifest_sha256",
    "requested_output_tokens",
    "num_pearl_steps",
)

RESULT_CONTRACT_FIELDS = (
    "batch_size",
    "num_prompts",
    "prompt_token_ids_sha256",
    "request_output_token_limits",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc2-off", type=Path, required=True, help="MC2-off benchmark JSON.")
    parser.add_argument("--mc2-on", type=Path, required=True, help="MC2-on benchmark JSON.")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Select this result when a benchmark payload contains multiple batch sizes.",
    )
    parser.add_argument(
        "--mc2-off-target-trace",
        type=Path,
        help="Optional MC2-off target-token diagnostic JSON.",
    )
    parser.add_argument(
        "--mc2-on-target-trace",
        type=Path,
        help="Optional MC2-on target-token diagnostic JSON.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_token_rows(rows: list[list[int]]) -> str:
    encoded = json.dumps(rows, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _select_result(payload: dict[str, Any], batch_size: int | None) -> dict[str, Any]:
    if "output_token_ids" in payload:
        result = payload
        if batch_size is not None and result.get("batch_size") != batch_size:
            raise ValueError(f"Requested batch size {batch_size}, but the result records {result.get('batch_size')!r}.")
        return result
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("Benchmark JSON must contain a non-empty results list with output_token_ids.")
    if batch_size is None:
        if len(results) != 1:
            available = [row.get("batch_size") for row in results if isinstance(row, dict)]
            raise ValueError(
                f"Benchmark JSON contains multiple results; select one with --batch-size (available: {available!r})."
            )
        result = results[0]
    else:
        matches = [row for row in results if isinstance(row, dict) and row.get("batch_size") == batch_size]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one result for batch size {batch_size}; found {len(matches)}.")
        result = matches[0]
    if not isinstance(result, dict) or "output_token_ids" not in result:
        raise ValueError("Selected benchmark result does not contain output_token_ids.")
    return result


def _token_rows(result: dict[str, Any]) -> list[list[int]]:
    rows = result.get("output_token_ids")
    if not isinstance(rows, list):
        raise ValueError("output_token_ids must be a list of token rows.")
    normalized: list[list[int]] = []
    for request_index, row in enumerate(rows):
        if not isinstance(row, list):
            raise ValueError(f"output_token_ids[{request_index}] must be a list.")
        if any(not isinstance(token, int) or isinstance(token, bool) for token in row):
            raise ValueError(f"output_token_ids[{request_index}] contains a non-integer token ID.")
        normalized.append(list(row))
    return normalized


def _contract_differences(
    off_payload: dict[str, Any],
    on_payload: dict[str, Any],
    off_result: dict[str, Any],
    on_result: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    differences: dict[str, dict[str, Any]] = {}
    for field in RUN_CONTRACT_FIELDS:
        off_value = off_payload.get(field)
        on_value = on_payload.get(field)
        if off_value != on_value or field not in off_payload or field not in on_payload:
            differences[field] = {"mc2_off": off_value, "mc2_on": on_value}
    for field in RESULT_CONTRACT_FIELDS:
        off_value = off_result.get(field)
        on_value = on_result.get(field)
        if off_value != on_value or field not in off_result or field not in on_result:
            differences[f"result.{field}"] = {"mc2_off": off_value, "mc2_on": on_value}
    return differences


def _request_token_mismatches(
    off_rows: list[list[int]],
    on_rows: list[list[int]],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for request_index in range(max(len(off_rows), len(on_rows))):
        off_row = off_rows[request_index] if request_index < len(off_rows) else []
        on_row = on_rows[request_index] if request_index < len(on_rows) else []
        offset = 0
        while offset < min(len(off_row), len(on_row)) and off_row[offset] == on_row[offset]:
            offset += 1
        if offset == len(off_row) == len(on_row):
            continue
        mismatches.append(
            {
                "request_index": request_index,
                "first_difference": offset,
                "mc2_off_token": off_row[offset] if offset < len(off_row) else None,
                "mc2_on_token": on_row[offset] if offset < len(on_row) else None,
                "mc2_off_length": len(off_row),
                "mc2_on_length": len(on_row),
            }
        )
    return mismatches


def _positional_difference_count(off_rows: list[list[int]], on_rows: list[list[int]]) -> int:
    count = 0
    for request_index in range(max(len(off_rows), len(on_rows))):
        off_row = off_rows[request_index] if request_index < len(off_rows) else []
        on_row = on_rows[request_index] if request_index < len(on_rows) else []
        count += sum(off_token != on_token for off_token, on_token in zip(off_row, on_row))
        count += abs(len(off_row) - len(on_row))
    return count


def _compare_outputs(off_rows: list[list[int]], on_rows: list[list[int]]) -> dict[str, Any]:
    mismatches = _request_token_mismatches(off_rows, on_rows)
    first_by_request = mismatches[0] if mismatches else None
    first_by_offset = (
        min(mismatches, key=lambda row: (row["first_difference"], row["request_index"])) if mismatches else None
    )
    return {
        "equal": not mismatches,
        "mc2_off_request_count": len(off_rows),
        "mc2_on_request_count": len(on_rows),
        "mc2_off_token_count": sum(map(len, off_rows)),
        "mc2_on_token_count": sum(map(len, on_rows)),
        "requests_with_differences": len(mismatches),
        "positional_token_differences": _positional_difference_count(off_rows, on_rows),
        "first_difference": first_by_offset,
        "first_difference_order": (
            "minimum completion-token offset, then request index; benchmark JSON has no token-emission timeline"
        ),
        "first_difference_by_request_order": first_by_request,
        "request_mismatches": mismatches,
    }


def _trace_rows(trace: Any) -> list[dict[str, Any]]:
    if trace is None:
        return []
    if isinstance(trace, list):
        rows = trace
    elif isinstance(trace, dict):
        rows = trace.get("target_token_diagnostics", [])
    else:
        raise ValueError("Target trace must be an object or a list.")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("target_token_diagnostics must be a list of objects.")
    return rows


def _top_two_from_logits(logits: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(logits, list) or len(logits) < 2:
        raise ValueError("A diagnostic logits row must contain at least two values.")
    values: list[float] = []
    for value in logits:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("Diagnostic logits must be numeric.")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError("Diagnostic logits must be finite.")
        values.append(converted)
    best = heapq.nsmallest(2, enumerate(values), key=lambda pair: (-pair[1], pair[0]))
    return (
        {"token_id": best[0][0], "logit": best[0][1]},
        {"token_id": best[1][0], "logit": best[1][1]},
    )


def _ranked_token(value: Any, name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        token_id, logit = value.get("token_id"), value.get("logit")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        token_id, logit = value
    else:
        raise ValueError(f"{name} must contain token_id and logit.")
    if not isinstance(token_id, int) or isinstance(token_id, bool):
        raise ValueError(f"{name}.token_id must be an integer.")
    if not isinstance(logit, (int, float)) or isinstance(logit, bool) or not math.isfinite(float(logit)):
        raise ValueError(f"{name}.logit must be finite and numeric.")
    return {"token_id": token_id, "logit": float(logit)}


def _normalize_target_diagnostic(row: dict[str, Any]) -> dict[str, Any]:
    if "logits" in row:
        top1, top2 = _top_two_from_logits(row["logits"])
    else:
        top1 = _ranked_token(row.get("top1"), "top1")
        top2 = _ranked_token(row.get("top2"), "top2")
    argmax = row.get("target_argmax_token_id", row.get("argmax_token_id"))
    if argmax is None and top1 is not None:
        argmax = top1["token_id"]
    if argmax is not None and (not isinstance(argmax, int) or isinstance(argmax, bool)):
        raise ValueError("target_argmax_token_id must be an integer.")
    margin = row.get("top1_top2_margin")
    if margin is None and top1 is not None and top2 is not None:
        margin = top1["logit"] - top2["logit"]
    if margin is not None and (
        not isinstance(margin, (int, float)) or isinstance(margin, bool) or not math.isfinite(float(margin))
    ):
        raise ValueError("top1_top2_margin must be finite and numeric.")
    return {
        "available": True,
        "target_argmax_token_id": argmax,
        "top1": top1,
        "top2": top2,
        "top1_top2_margin": None if margin is None else float(margin),
    }


def _target_diagnostic_at(
    trace: Any,
    request_index: int,
    output_offset: int,
    generated_token: int | None,
) -> dict[str, Any]:
    matches = [
        row
        for row in _trace_rows(trace)
        if row.get("request_index") == request_index and row.get("output_offset") == output_offset
    ]
    if not matches:
        return {
            "available": False,
            "reason": "target diagnostic was not recorded at this request/offset",
        }
    if len(matches) != 1:
        raise ValueError(
            f"Target trace contains duplicate rows for request {request_index}, output offset {output_offset}."
        )
    normalized = _normalize_target_diagnostic(matches[0])
    normalized["generated_token_id"] = generated_token
    argmax = normalized["target_argmax_token_id"]
    normalized["target_argmax_matches_generated_token"] = (
        None if argmax is None or generated_token is None else argmax == generated_token
    )
    return normalized


def _target_divergence_report(
    first_difference: dict[str, Any] | None,
    off_trace: Any,
    on_trace: Any,
) -> dict[str, Any]:
    if first_difference is None:
        return {
            "status": "not_applicable_outputs_are_equal",
            "coordinate": None,
            "mc2_off": None,
            "mc2_on": None,
        }
    request_index = int(first_difference["request_index"])
    output_offset = int(first_difference["first_difference"])
    off = _target_diagnostic_at(
        off_trace,
        request_index,
        output_offset,
        first_difference["mc2_off_token"],
    )
    on = _target_diagnostic_at(
        on_trace,
        request_index,
        output_offset,
        first_difference["mc2_on_token"],
    )
    if off["available"] and on["available"]:
        status = "available_for_both_runs"
    elif off["available"] or on["available"]:
        status = "available_for_one_run_only"
    else:
        status = "not_recorded_by_source_artifacts"
    return {
        "status": status,
        "coordinate": {
            "request_index": request_index,
            "output_offset": output_offset,
            "watch": f"{request_index}:{output_offset}",
        },
        "mc2_off": off,
        "mc2_on": on,
        "argmax_equal": (
            off["target_argmax_token_id"] == on["target_argmax_token_id"]
            if off["available"]
            and on["available"]
            and off["target_argmax_token_id"] is not None
            and on["target_argmax_token_id"] is not None
            else None
        ),
    }


def compare_payloads(
    off_payload: dict[str, Any],
    on_payload: dict[str, Any],
    *,
    batch_size: int | None = None,
    off_trace: Any = None,
    on_trace: Any = None,
) -> dict[str, Any]:
    off_result = _select_result(off_payload, batch_size)
    on_result = _select_result(on_payload, batch_size)
    off_rows = _token_rows(off_result)
    on_rows = _token_rows(on_result)
    off_digest = _digest_token_rows(off_rows)
    on_digest = _digest_token_rows(on_rows)
    output_comparison = _compare_outputs(off_rows, on_rows)
    differences = _contract_differences(off_payload, on_payload, off_result, on_result)
    if off_trace is None:
        off_trace = off_result if "target_token_diagnostics" in off_result else off_payload
    if on_trace is None:
        on_trace = on_result if "target_token_diagnostics" in on_result else on_payload
    switch_contract = {
        "mc2_off_enable_mc2": off_payload.get("enable_mc2"),
        "mc2_on_enable_mc2": on_payload.get("enable_mc2"),
        "passed": off_payload.get("enable_mc2") is False and on_payload.get("enable_mc2") is True,
    }
    comparable = not differences and switch_contract["passed"]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "specslo_mc2_generation_consistency",
        "passed": comparable and output_comparison["equal"],
        "comparable": comparable,
        "contract_differences": differences,
        "mc2_switch_contract": switch_contract,
        "complete_output_hashes": {
            "mc2_off_recomputed_sha256": off_digest,
            "mc2_on_recomputed_sha256": on_digest,
            "equal": off_digest == on_digest,
            "mc2_off_stored_sha256": off_result.get("output_token_ids_sha256"),
            "mc2_on_stored_sha256": on_result.get("output_token_ids_sha256"),
            "mc2_off_stored_matches_recomputed": (
                None
                if off_result.get("output_token_ids_sha256") is None
                else off_result["output_token_ids_sha256"] == off_digest
            ),
            "mc2_on_stored_matches_recomputed": (
                None
                if on_result.get("output_token_ids_sha256") is None
                else on_result["output_token_ids_sha256"] == on_digest
            ),
        },
        "output_comparison": output_comparison,
        "target_at_first_difference": _target_divergence_report(
            output_comparison["first_difference"],
            off_trace,
            on_trace,
        ),
        "production_semantics_changed": False,
        "scope": (
            "Generation-output comparison only. Target argmax/top-2 claims are present only when supplied "
            "by an explicit target-token diagnostic trace."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    off_payload = _load_json(args.mc2_off)
    on_payload = _load_json(args.mc2_on)
    if not isinstance(off_payload, dict) or not isinstance(on_payload, dict):
        raise ValueError("MC2 benchmark inputs must be JSON objects.")
    off_trace = _load_json(args.mc2_off_target_trace) if args.mc2_off_target_trace else None
    on_trace = _load_json(args.mc2_on_target_trace) if args.mc2_on_target_trace else None
    report = compare_payloads(
        off_payload,
        on_payload,
        batch_size=args.batch_size,
        off_trace=off_trace,
        on_trace=on_trace,
    )
    report["source_artifacts"] = {
        "mc2_off": {"path": str(args.mc2_off), "sha256": _sha256_file(args.mc2_off)},
        "mc2_on": {"path": str(args.mc2_on), "sha256": _sha256_file(args.mc2_on)},
        "mc2_off_target_trace": (
            None
            if args.mc2_off_target_trace is None
            else {
                "path": str(args.mc2_off_target_trace),
                "sha256": _sha256_file(args.mc2_off_target_trace),
            }
        ),
        "mc2_on_target_trace": (
            None
            if args.mc2_on_target_trace is None
            else {
                "path": str(args.mc2_on_target_trace),
                "sha256": _sha256_file(args.mc2_on_target_trace),
            }
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
