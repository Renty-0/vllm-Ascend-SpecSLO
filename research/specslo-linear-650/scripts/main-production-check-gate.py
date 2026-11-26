"""Audit an original explicit mechanism run without production-only predicates."""

import hashlib
import json
import math
import os
import sys
from pathlib import Path


def main():
    result_path, mode, output_path = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    if output_path.exists():
        raise FileExistsError(output_path)
    document = json.loads(result_path.read_text())
    manifest = Path(document["request_manifest"])
    requests = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    errors = []

    def require(condition, explanation):
        if not condition:
            errors.append(explanation)

    expected_hash = "dc58dc5af1c5b28c6944f8f3faadeaae2dae34298c07b3ac84745d367c67d662"
    require(hashlib.sha256(manifest.read_bytes()).hexdigest() == expected_hash, "Manifest bytes changed")
    require(document.get("request_manifest_sha256") == expected_hash, "Recorded manifest hash mismatch")
    require(document.get("spec_rhythm_ablation_mode") == mode, "Wrong explicit mechanism mode")
    require(document.get("draft_tensor_parallel_size") == 1 and document.get("target_tensor_parallel_size") == 3,
            "Wrong TP1+TP3 allocation")
    require(document.get("gamma") == 4 and document.get("spec_rhythm_min_gamma") == 4, "Wrong fixed gamma")
    require(document.get("spec_rhythm_linear_full_window") is True, "Missing original full-window execution")
    require(document.get("online_arrivals") is True and document.get("arrival_mode") == "manifest_trace",
            "Not original online arrival trace replay")
    require(document.get("spec_rhythm_online_prefill") is True and document.get("spec_rhythm_kv_ready_arrivals") is False,
            "Online prefill contract changed")
    require(document.get("respect_eos") is False and document.get("saturated_arrivals") is False,
            "EOS/saturated arrivals changed")
    require(document.get("spec_rhythm_prefill_coalesce_min_requests") ==
            int(os.environ.get("SPECSLO_PREFILL_COALESCE_MIN_REQUESTS", "1"))
            and document.get("spec_rhythm_prefill_coalesce_max_wait_ms") ==
            int(os.environ.get("SPECSLO_PREFILL_COALESCE_MAX_WAIT_MS", "0")),
            "Arrival coalescing contract changed")
    require(document.get("enable_prefix_caching") is
            (os.environ.get("SPECSLO_ENABLE_PREFIX_CACHING", "0") == "1"),
            "Prefix caching contract changed")
    for key in ("spec_rhythm_merge_ready_homes", "spec_rhythm_linear_eager_cross_graph_bucket",
                "spec_rhythm_linear_idle_residual_eager", "spec_rhythm_linear_bonus_token"):
        expected = (mode == "auto") if key == "spec_rhythm_merge_ready_homes" else (key != "spec_rhythm_linear_bonus_token")
        require(document.get(key) is expected,
                f"Historical main production policy mismatch: {key}")
    environment = document.get("runtime_environment", {})
    require(environment.get("ASCEND_RT_VISIBLE_DEVICES") == "0,1,2,3", "Wrong container device allocation")
    require(environment.get("PYTORCH_NPU_ALLOC_CONF") == "expandable_segments:True", "Wrong allocator setting")
    require(document.get("enable_mc2") is True, "Common MC2 setting changed")
    for key, value in (("VLLM_ASCEND_SPECRHYTHM_GLOO_CORRECTION", "1"), ("VLLM_ASCEND_SPECRHYTHM_GLOO_PROPOSAL", "1"), ("VLLM_ASCEND_SPECRHYTHM_GLOO_ACCOUNTING", "1")):
        require(environment.get(key) == value, f"Common transport differs: {key}")
    rows = document.get("results", [])
    require(len(rows) == 1, "Expected one complete measured result")
    result = rows[0]
    require(result.get("num_prompts") == len(requests) == 463, "Request count mismatch")
    require(result.get("output_tokens") == document.get("requested_output_tokens") == 118528,
            "Output work mismatch")
    require(result.get("request_output_token_limits") == [256] * 463, "Request limits changed")
    token_rows = result.get("output_token_ids", [])
    require(len(token_rows) == 463 and all(len(row) == 256 for row in token_rows), "Incomplete per-request output")
    encoded = json.dumps(token_rows, separators=(",", ":")).encode()
    require(hashlib.sha256(encoded).hexdigest() == result.get("output_token_ids_sha256"), "Token hash mismatch")
    elapsed = float(result["e2e_elapsed_seconds"])
    require(math.isfinite(elapsed) and elapsed > 0, "Invalid E2E timing")
    require(math.isclose(result["e2e_throughput_tokens_per_second"], 118528 / elapsed, rel_tol=1e-10),
            "Raw metric calculation mismatch")
    require(result.get("graph_qualification_fixed_point") is True, "Graph qualification did not converge")
    for key in ("aclgraph_shape_fallbacks", "aclgraph_capacity_fallbacks"):
        require(result.get(key) == 0, f"Measured {key}")
    request_metrics = result["slo"]["request_metrics"]
    require(len(request_metrics) == 463, "Missing request metrics")
    max_arrival_error = 0.0
    attained = 0
    for expected, measured in zip(requests, request_metrics):
        require(measured.get("request_id") == expected["request_id"], "Request identity mismatch")
        require(measured.get("output_tokens") == 256, "Request output count mismatch")
        require(measured.get("slo_tpot_ms") == expected["slo_tpot_ms"], "Request SLO mismatch")
        # The original report normalizes retained offsets to the first
        # request's arrival, while the manifest starts before that arrival.
        expected_offset = float(expected["arrival_offset_sec"]) - float(requests[0]["arrival_offset_sec"])
        error = abs(float(measured["arrival_offset_sec"]) - expected_offset)
        max_arrival_error = max(max_arrival_error, error)
        value = float(measured["paper_tpot_ms"])
        require(math.isfinite(value), "Missing or nonfinite TPOT")
        attained += int(value <= float(expected["slo_tpot_ms"]))
    require(max_arrival_error <= 1e-6, "Requested arrival offsets differ from the shared trace")
    paper = result["slo"]["paper"]
    require(paper["attained_requests"] == attained and paper["goodput_tokens"] == attained * 256,
            "Goodput qualification count mismatch")
    require(math.isclose(paper["goodput_tokens_per_e2e_second"], attained * 256 / elapsed, rel_tol=1e-10),
            "Goodput calculation mismatch")
    require(paper.get("missing_timing_requests") == 0, "Missing request timing")
    workers = result["worker_metrics_by_chunk"][0]["worker_metrics"]
    deltas = result["measured_worker_aclgraph_deltas"]
    require(sorted(w["rank"] for w in workers) == [0, 1, 2, 3], "Wrong measured rank set")
    require(sorted(w["rank"] for w in deltas) == [0, 1, 2, 3], "Wrong graph delta rank set")
    protocol = []
    for worker in workers:
        rank = worker["rank"]
        nano = mode == "nano_pearl"
        rolling = mode in ("auto", "dual_batch_rolling")
        eager = mode != "dual_batch"
        wanted = {
            "worker_spec_rhythm_serial_protocol": 0,
            "worker_spec_rhythm_single_batch_overlap_protocol": int(nano),
            "worker_spec_rhythm_dual_batch_overlap_protocol": int(not nano),
            "worker_spec_rhythm_eager_enabled": int(eager),
            "worker_spec_rhythm_effective_eager_cap": 4 if eager else 0,
            "worker_spec_rhythm_priority_enabled": int(rolling),
        }
        for key, value in wanted.items():
            require(worker.get(key) == value, f"Rank{rank} mechanism mismatch: {key}")
        submission = "worker_spec_rhythm_single_batch_overlap_submission_cycles" if nano else "worker_spec_rhythm_dual_batch_overlap_submission_cycles"
        require(worker.get(submission, 0) > 0, f"Rank{rank} missing mechanism execution")
        proposals = worker.get("worker_spec_rhythm_eager_proposals", 0)
        promoted = worker.get("worker_spec_rhythm_eager_promoted", 0)
        invalidated = worker.get("worker_spec_rhythm_eager_invalidated", 0)
        require(proposals > 0 if eager else proposals == 0, f"Rank{rank} wrong eager activity")
        require(promoted + invalidated == proposals, f"Rank{rank} eager proposal lifecycle count mismatch")
        for key in ("mc2_dispatch_fallback", "mc2_dispatch_exception"):
            require(worker.get(key) == 0, f"Rank{rank} {key}")
        if rank:
            require(worker.get("mc2_dispatch_native_epilogue_chained_success", 0) > 0,
                    f"Rank{rank} no common MC2 graph execution")
        protocol.append({"rank": rank, **{k: worker.get(k) for k in wanted},
                         "submission_cycles": worker.get(submission), "eager_proposals": proposals,
                         "eager_promoted": promoted, "eager_invalidated": invalidated})
    for worker in deltas:
        rank = worker["rank"]
        for key, value in worker.items():
            if key.startswith("aclgraph_") and any(word in key for word in (
                    "captures", "capture_attempts", "failed_captures", "fallback", "validation_failures")):
                require(value == 0, f"Rank{rank} measured graph failure/capture: {key}={value}")
            if key in ("spec_rhythm_stable_target_verify_numerical_validation_failures",
                       "spec_rhythm_stable_target_verify_numerical_restore_failures"):
                require(value == 0, f"Rank{rank} target graph numerical validation failure")
    report = {
        "status": "pass" if not errors else "fail", "mode": mode, "errors": errors,
        "scope": "Workload, arrival offsets, graph/MC2 predicates and original named mechanism counters. Does not establish whole-workload cross-mode token equality or rolling-only causal attribution.",
        "result": str(result_path), "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "raw": result["e2e_throughput_tokens_per_second"],
        "paper_goodput": paper["goodput_tokens_per_e2e_second"], "attained_requests": attained,
        "max_prescribed_arrival_offset_error_seconds": max_arrival_error,
        "arrival_comparison_origin": "First prescribed request arrival; original native result offsets use that origin",
        "output_token_hash": result.get("output_token_ids_sha256"), "protocol": protocol,
        "confound": "Historical main production auto mode retains merged homes, cross-bucket rolling and idle residual eager; allocator expandable_segments:True is recorded separately from frozen v1.",
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"mode": mode, "status": report["status"], "errors": errors, "output": str(output_path)}))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
