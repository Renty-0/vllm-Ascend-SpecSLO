"""Validate a dynamic-gamma b64 run without applying the fixed-gamma gate."""

import hashlib
import json
import math
import sys
from pathlib import Path


def main() -> int:
    result_path, mode, output_path = map(Path, (sys.argv[1], sys.argv[2], sys.argv[3]))
    document = json.loads(result_path.read_text())
    rows = document.get("results", [])
    errors: list[str] = []

    def require(ok: bool, message: str) -> None:
        if not ok:
            errors.append(message)

    manifest = Path(document["request_manifest"])
    manifest_hash = "dc58dc5af1c5b28c6944f8f3faadeaae2dae34298c07b3ac84745d367c67d662"
    require(hashlib.sha256(manifest.read_bytes()).hexdigest() == manifest_hash, "Manifest changed")
    require(document.get("request_manifest_sha256") == manifest_hash, "Recorded manifest hash changed")
    require(document.get("gamma") == 4, "Maximum gamma changed")
    require(document.get("spec_rhythm_min_gamma") in (1, 2, 3), "Gamma is not dynamic")
    require(document.get("spec_rhythm_linear_full_window") is False, "Fixed full-window path enabled")
    require(document.get("spec_rhythm_verification_budget", 0) > 0, "Missing verification budget")
    require(document.get("spec_rhythm_ablation_mode") == str(mode), "Ablation mode changed")
    require(document.get("draft_tensor_parallel_size") == 1, "Draft TP changed")
    require(document.get("target_tensor_parallel_size") == 3, "Target TP changed")
    require(document.get("arrival_mode") == "manifest_trace", "Arrival trace changed")
    require(document.get("online_arrivals") is True, "Online arrivals disabled")
    require(document.get("enable_mc2") is True, "MC2 disabled")
    require(len(rows) == 1, "Expected one result row")
    if rows:
        row = rows[0]
        require(row.get("batch_size") == 64, "Global batch limit changed")
        require(row.get("num_prompts") == 463, "Request count changed")
        require(row.get("output_tokens") == 118528, "Output token count changed")
        require(len(row.get("output_token_ids", [])) == 463, "Missing output rows")
        require(all(len(tokens) == 256 for tokens in row.get("output_token_ids", [])),
                "Incomplete request output")
        require(row.get("graph_qualification_fixed_point") is True, "Graph qualification not stable")
        for key in ("aclgraph_shape_fallbacks", "aclgraph_capacity_fallbacks"):
            require(row.get(key) == 0, f"{key} is nonzero")
        seconds = row.get("e2e_elapsed_seconds", 0)
        require(isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds > 0,
                "Invalid E2E timing")
        paper = row.get("slo", {}).get("paper", {})
        require(paper.get("constrained_requests") == 463, "Wrong SLO population")
        require(paper.get("missing_timing_requests") == 0, "Missing TPOT records")
        workers = row.get("worker_aclgraph_metrics", [])
        require(len(workers) == 4, "Missing worker metrics")
        budget_rows = {}
        if workers:
            leader = workers[0]
            budget_rows = {
                label: {
                    str(depth): leader.get(f"worker_spec_rhythm_allocated_gamma_{label}_{depth}_rows", 0)
                    for depth in range(1, 5)
                }
                for label in ("tight", "normal", "loose", "unclassified")
            }
            require(sum(budget_rows[label][str(depth)]
                        for label in budget_rows for depth in range(1, 4)) > 0,
                    "No shortened proposal was executed")
        report = {
            "status": "pass" if not errors else "fail",
            "errors": errors,
            "result": str(result_path),
            "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            "gamma_range": [document.get("spec_rhythm_min_gamma"), document.get("gamma")],
            "verification_budget": document.get("spec_rhythm_verification_budget"),
            "raw": row.get("e2e_throughput_tokens_per_second"),
            "paper_goodput": paper.get("goodput_tokens_per_e2e_second"),
            "attained_requests": paper.get("attained_requests"),
            "by_class": paper.get("by_class"),
            "allocated_gamma_rows": budget_rows,
        }
    else:
        report = {"status": "fail", "errors": errors}
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report.get(key) for key in ("status", "errors", "paper_goodput", "raw")}))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
