# SPDX-License-Identifier: Apache-2.0
"""Run the native Ascend nano-PEARL engine with its upstream-compatible API."""

from __future__ import annotations

import argparse
import json

from vllm_ascend.spec_decode.pearl import PEARLConfig, PEARLEngine, SamplingParams
from vllm_ascend.spec_decode.pearl.roofline import parse_roofline_argument


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-tp-size", type=int, default=1)
    parser.add_argument("--target-tp-size", type=int, default=1)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--num-kvcache-blocks", type=int, default=-1)
    parser.add_argument("--max-aclgraph-entries", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--slo-tpot-ms", type=float)
    parser.add_argument("--slo-class")
    parser.add_argument("--enable-spec-rhythm", action="store_true")
    parser.add_argument(
        "--spec-rhythm-linear-full-window",
        action="store_true",
        help=(
            "Use the independent full-window protocol for fixed-gamma serial "
            "linear SpecRhythm (requires min-gamma == gamma and tree 1x1)."
        ),
    )
    parser.add_argument(
        "--spec-rhythm-linear-bonus-token",
        action="store_true",
        help=(
            "Commit the target bonus token after a fully accepted fixed gamma-4 "
            "serial full window (experimental; disabled by default)."
        ),
    )
    parser.add_argument("--spec-rhythm-min-gamma", type=int, default=1)
    parser.add_argument("--spec-rhythm-max-eager-tokens", type=int, default=0)
    parser.add_argument("--spec-rhythm-eager-reserve-tokens", type=int, default=0)
    parser.add_argument("--spec-rhythm-urgency-threshold", type=float, default=0.75)
    parser.add_argument("--spec-rhythm-acceptance-floor", type=float, default=0.4)
    parser.add_argument("--spec-rhythm-acceptance-ema-alpha", type=float, default=0.2)
    parser.add_argument("--spec-rhythm-request-max-gamma", type=int)
    parser.add_argument("--spec-rhythm-draft-token-budget", type=int)
    parser.add_argument("--spec-rhythm-tree-width", type=int, default=1)
    parser.add_argument("--spec-rhythm-tree-depth", type=int, default=1)
    parser.add_argument("--spec-rhythm-roofline", type=parse_roofline_argument)
    parser.add_argument(
        "--spec-rhythm-verification-budget",
        type=int,
        help="Fixed global SpecRhythm candidate-token budget B.",
    )
    parser.add_argument("--draft-use-paged-attention", action="store_true")
    parser.add_argument("--target-use-paged-attention", action="store_true")
    parser.add_argument("--precompile-decode-graphs", action="store_true")
    parser.add_argument(
        "--precompile-serial-draft-graphs",
        action="store_true",
        help=(
            "Precompile and changed-input qualify only fixed-gamma serial "
            "draft graphs; keep target verification graphs lazy."
        ),
    )
    parser.add_argument("--disable-cpu-binding", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--mode", choices=("pearl", "target-ar", "bench"), default="pearl")
    parser.add_argument("--num-pearl-steps", type=int, default=100)
    parser.add_argument("--worker-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--profile-decode-steps", type=int, default=0)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument(
        "--print-metrics",
        action="store_true",
        help="Print per-request decode, SLO, and SpecRhythm metrics as JSON.",
    )
    parser.add_argument("prompt", nargs="+", help="One or more prompts to generate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PEARLConfig(
        draft_model_path=args.draft_model,
        target_model_path=args.target_model,
        draft_tensor_parallel_size=args.draft_tp_size,
        target_tensor_parallel_size=args.target_tp_size,
        max_num_batched_tokens=max(args.max_model_len, args.max_model_len * args.max_num_seqs),
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_kvcache_blocks=args.num_kvcache_blocks,
        max_aclgraph_entries=args.max_aclgraph_entries,
        enable_cpu_binding=not args.disable_cpu_binding,
        enforce_eager=args.enforce_eager,
        gamma=args.gamma,
        enable_continuous_batching=args.enable_spec_rhythm,
        enable_preemptive_scheduling=args.enable_spec_rhythm,
        enable_spec_rhythm=args.enable_spec_rhythm,
        spec_rhythm_linear_full_window=args.spec_rhythm_linear_full_window,
        spec_rhythm_linear_bonus_token=args.spec_rhythm_linear_bonus_token,
        spec_rhythm_min_gamma=args.spec_rhythm_min_gamma,
        spec_rhythm_max_eager_tokens=args.spec_rhythm_max_eager_tokens,
        spec_rhythm_eager_reserve_tokens=args.spec_rhythm_eager_reserve_tokens,
        spec_rhythm_urgency_threshold=args.spec_rhythm_urgency_threshold,
        spec_rhythm_acceptance_floor=args.spec_rhythm_acceptance_floor,
        spec_rhythm_acceptance_ema_alpha=args.spec_rhythm_acceptance_ema_alpha,
        spec_rhythm_roofline=args.spec_rhythm_roofline,
        spec_rhythm_verification_budget=args.spec_rhythm_verification_budget,
        spec_rhythm_draft_token_budget=args.spec_rhythm_draft_token_budget,
        spec_rhythm_tree_width=args.spec_rhythm_tree_width,
        spec_rhythm_tree_depth=args.spec_rhythm_tree_depth,
        draft_use_paged_attention=args.draft_use_paged_attention,
        target_use_paged_attention=args.target_use_paged_attention,
        precompile_decode_graphs=args.precompile_decode_graphs,
        precompile_serial_draft_graphs=args.precompile_serial_draft_graphs,
        profile_decode_steps=args.profile_decode_steps,
        stop_after_profiled_decode_steps=args.profile_only,
        worker_timeout_seconds=args.worker_timeout_seconds,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
        slo_tpot_ms=args.slo_tpot_ms,
        slo_class=args.slo_class,
        spec_rhythm_max_gamma=args.spec_rhythm_request_max_gamma,
    )
    with PEARLEngine(config) as engine:
        for prompt in args.prompt:
            engine.add_request(prompt, sampling_params)
        if args.mode == "pearl":
            outputs = engine.generate()
        elif args.mode == "target-ar":
            outputs = engine.AR_generate()
        else:
            outputs = engine.bench_generate(args.num_pearl_steps)
    print(outputs)
    if args.print_metrics:
        print(json.dumps(engine.last_metrics, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
