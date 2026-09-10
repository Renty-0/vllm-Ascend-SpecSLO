# SPDX-License-Identifier: Apache-2.0
"""Prepare the paper-faithful SpecRhythm three-class workload manifest."""

from __future__ import annotations

import argparse
import json

from vllm_ascend.spec_decode.pearl.specslo_workload import (
    DEFAULT_GAMMAS,
    DEFAULT_SLOS_MS,
    build_workload,
    write_workload,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--humaneval", required=True, help="HumanEval JSONL")
    parser.add_argument("--alpaca", required=True, help="Stanford Alpaca JSON/JSONL")
    parser.add_argument("--cnndm", required=True, help="CNN/DailyMail JSONL")
    parser.add_argument("--rps", type=float, required=True)
    parser.add_argument("--duration-sec", type=float, default=120.0)
    parser.add_argument("--num-requests", type=int)
    parser.add_argument("--mix", default="0.6,0.2,0.2")
    parser.add_argument("--tight-slo-tpot-ms", type=float, default=DEFAULT_SLOS_MS["tight"])
    parser.add_argument("--normal-slo-tpot-ms", type=float, default=DEFAULT_SLOS_MS["normal"])
    parser.add_argument("--loose-slo-tpot-ms", type=float, default=DEFAULT_SLOS_MS["loose"])
    parser.add_argument("--tight-gamma", type=int, default=DEFAULT_GAMMAS["tight"])
    parser.add_argument("--normal-gamma", type=int, default=DEFAULT_GAMMAS["normal"])
    parser.add_argument("--loose-gamma", type=int, default=DEFAULT_GAMMAS["loose"])
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arrival-trace")
    parser.add_argument("--request-id-prefix")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    requests, metadata = build_workload(
        coding_path=args.humaneval,
        chat_path=args.alpaca,
        summarization_path=args.cnndm,
        rps=args.rps,
        num_requests=args.num_requests,
        duration_sec=args.duration_sec,
        mix=args.mix,
        tight_slo_tpot_ms=args.tight_slo_tpot_ms,
        normal_slo_tpot_ms=args.normal_slo_tpot_ms,
        loose_slo_tpot_ms=args.loose_slo_tpot_ms,
        tight_gamma=args.tight_gamma,
        normal_gamma=args.normal_gamma,
        loose_gamma=args.loose_gamma,
        max_tokens=args.max_tokens,
        seed=args.seed,
        arrival_trace=args.arrival_trace,
        request_id_prefix=args.request_id_prefix,
    )
    write_workload(requests, metadata, args.out)
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

