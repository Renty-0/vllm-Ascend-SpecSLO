"""Check a Llama 1B TP1 + 70B TP3 b64 run against the frozen TP4 baseline."""

import hashlib
import json
import math
import sys
from pathlib import Path


FROZEN = Path(__file__).with_name("baseline-v021-llama70b-b64-three-run-mean.json")
DRAFT = "/data/shared_upload/models/Llama-3.2-1B-Instruct"


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: gate.py RESULT_JSON MODE REPORT_JSON")
    result_path = Path(sys.argv[1])
    mode = sys.argv[2]
    report_path = Path(sys.argv[3])
    document = json.loads(result_path.read_text())
    frozen = json.loads(FROZEN.read_text())
    contract = frozen["contract"]
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(mode == "auto", "Expected the dual-batch rolling mechanism")
    require(document.get("draft_model") == DRAFT, "Wrong draft model")
    require(document.get("target_model") == contract["model"], "Wrong target model")
    require(document.get("prompt_tokenizer_model") == "target", "Prompt tokenizer is not target")
    require(document.get("draft_tensor_parallel_size") == 1, "Wrong draft TP")
    require(document.get("target_tensor_parallel_size") == 3, "Wrong target TP")
    require(document.get("enable_mc2") is False, "MC2 differs from the baseline")
    require(document.get("online_arrivals") is True, "Online arrivals disabled")
    require(document.get("respect_eos") is False, "Output contract differs")
    require(
        document.get("request_manifest_sha256") == contract["request_manifest_sha256"],
        "Request manifest changed",
    )
    rows = document.get("results", [])
    require(len(rows) == 1, "Expected one complete measurement")
    result = rows[0] if len(rows) == 1 else {}
    paper = result.get("slo", {}).get("paper", {})
    require(result.get("batch_size") == contract["max_num_seqs"] == 64, "Batch differs")
    require(result.get("num_prompts") == contract["num_prompts"] == 463, "Request count differs")
    require(result.get("output_tokens") == contract["output_tokens"] == 118528, "Output work differs")
    require(result.get("request_output_token_limits") == [256] * 463, "Output limits differ")
    require(
        result.get("prompt_token_ids_sha256") == contract["prompt_token_ids_sha256"],
        "Actual prompt tokens differ",
    )
    output_rows = result.get("output_token_ids", [])
    require(len(output_rows) == 463 and all(len(row) == 256 for row in output_rows), "Incomplete outputs")
    require(paper.get("constrained_requests") == 463, "Incomplete SLO accounting")
    require(paper.get("missing_timing_requests") == 0, "Missing TPOT measurements")
    elapsed = float(result.get("e2e_elapsed_seconds") or 0)
    require(math.isfinite(elapsed) and elapsed > 0, "Invalid elapsed time")
    if elapsed > 0:
        require(
            math.isclose(result.get("e2e_throughput_tokens_per_second", 0), 118528 / elapsed, rel_tol=1e-8),
            "Raw denominator differs",
        )
        require(
            math.isclose(
                paper.get("goodput_tokens_per_e2e_second", 0), paper.get("goodput_tokens", 0) / elapsed,
                rel_tol=1e-8,
            ),
            "Goodput denominator differs",
        )
    require(result.get("graph_qualification_fixed_point") is True, "Graph qualification did not converge")
    require(result.get("aclgraph_shape_fallbacks") == 0, "Graph shape fallback")
    require(result.get("aclgraph_capacity_fallbacks") == 0, "Graph capacity fallback")
    graph_rows = result.get("measured_worker_aclgraph_deltas", [])
    require(sorted(row.get("rank") for row in graph_rows) == [0, 1, 2, 3], "Missing worker graph counters")
    for row in graph_rows:
        for key, value in row.items():
            if key.startswith("aclgraph_") and any(
                term in key for term in ("failed_captures", "capacity_fallbacks", "shape_fallbacks", "validation_failures")
            ):
                require(value == 0, f"Rank{row.get('rank')} graph failure: {key}={value}")

    baseline_goodput = float(frozen["baseline_goodput_mean_tokens_per_second"])
    goodput = paper.get("goodput_tokens_per_e2e_second")
    report = {
        "status": "pass" if not errors else "fail",
        "mode": mode,
        "errors": errors,
        "result": str(result_path),
        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "frozen_baseline": str(FROZEN),
        "frozen_baseline_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        "frozen_baseline_goodput": baseline_goodput,
        "goodput_ratio_to_frozen_baseline": goodput / baseline_goodput if goodput is not None else None,
        "required_goodput_1_5x": frozen["specslo_1_5x_goodput_threshold_tokens_per_second"],
        "raw": result.get("e2e_throughput_tokens_per_second"),
        "paper_goodput": goodput,
        "attained_requests": paper.get("attained_requests"),
        "acceptance_rate": result.get("acceptance_rate"),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
