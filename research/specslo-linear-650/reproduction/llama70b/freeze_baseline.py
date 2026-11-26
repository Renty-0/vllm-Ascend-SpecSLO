#!/usr/bin/env python3
"""Freeze the mean of three independent, identically configured 70B runs."""

import hashlib
import json
import argparse
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "baseline-results" / "runs"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=(8, 16, 32, 64), default=64)
    parser.add_argument("--tags", nargs=3, default=("r1", "r2", "r3"))
    args = parser.parse_args()
    output = ROOT / f"baseline-v021-llama70b-b{args.batch}-three-run-mean.json"
    rows = []
    reference = None
    for tag in args.tags:
        run = RUNS / f"official-v021-llama70b-b{args.batch}-{tag}"
        result_path = run / "result.json"
        result = json.loads(result_path.read_text())
        launch = json.loads((run / "launch-environment.json").read_text())
        measured = result["results"]
        assert len(measured) == 1
        measured = measured[0]
        paper = measured["slo"]["paper"]
        contract = {
            "model": result["model"],
            "tensor_parallel_size": result["tensor_parallel_size"],
            "max_num_seqs": result["max_num_seqs"],
            "request_manifest_sha256": result["request_manifest_sha256"],
            "prompt_token_ids_sha256": measured["prompt_token_ids_sha256"],
            "output_tokens": measured["output_tokens"],
            "num_prompts": measured["num_prompts"],
            "max_tokens": result["max_tokens"],
            "online_arrivals": result["online_arrivals"],
            "ascend_commit": launch["ascend_commit"],
            "vllm_commit": launch["vllm_commit"],
            "physical_devices": launch["physical_devices"],
            "mc2_matmul_allreduce": launch["environment"]["VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE"],
            "mc2_fused": launch["environment"]["VLLM_ASCEND_ENABLE_FUSED_MC2"],
        }
        if reference is None:
            reference = contract
        assert contract == reference, f"configuration mismatch in {tag}: {contract!r}"
        assert contract["tensor_parallel_size"] == 4
        assert contract["max_num_seqs"] == args.batch
        assert contract["num_prompts"] == 463
        assert contract["output_tokens"] == 463 * 256
        assert contract["mc2_matmul_allreduce"] == "0"
        assert contract["mc2_fused"] == "0"
        assert paper["missing_timing_requests"] == 0
        rows.append({
            "tag": tag,
            "result_path": str(result_path),
            "result_sha256": digest(result_path),
            "goodput_tokens_per_second": paper["goodput_tokens_per_e2e_second"],
            "raw_tokens_per_second": measured["output_throughput_tokens_per_second"],
            "attained_requests": paper["attained_requests"],
            "attained_by_class": {
                label: paper["by_class"][label]["attained_requests"]
                for label in ("tight", "normal", "loose")
            },
        })

    goodputs = [r["goodput_tokens_per_second"] for r in rows]
    raws = [r["raw_tokens_per_second"] for r in rows]
    frozen = {
        "contract": reference,
        "runs": rows,
        "aggregation": "arithmetic mean of the three prespecified independent runs",
        "baseline_goodput_mean_tokens_per_second": mean(goodputs),
        "baseline_goodput_population_std_tokens_per_second": pstdev(goodputs),
        "baseline_goodput_min_tokens_per_second": min(goodputs),
        "baseline_goodput_max_tokens_per_second": max(goodputs),
        "baseline_raw_mean_tokens_per_second": mean(raws),
        "specslo_1_5x_goodput_threshold_tokens_per_second": 1.5 * mean(goodputs),
    }
    payload = json.dumps(frozen, ensure_ascii=False, indent=2) + "\n"
    if output.exists() and output.read_text() != payload:
        raise FileExistsError(f"Frozen baseline already exists with different content: {output}")
    output.write_text(payload)
    print(json.dumps(frozen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
