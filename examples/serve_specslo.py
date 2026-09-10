# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serve native SpecSLO through persistent OpenAI-compatible HTTP endpoints."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--spec-rhythm-roofline", required=True)
    parser.add_argument("--draft-tp-size", type=int, default=1)
    parser.add_argument("--target-tp-size", type=int, default=3)
    parser.add_argument("--draft-dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    parser.add_argument("--target-dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--tree-width", type=int, default=2)
    parser.add_argument("--tree-depth", type=int, default=2)
    parser.add_argument("--min-gamma", type=int, default=1)
    parser.add_argument("--max-eager-tokens", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-queued-seqs", type=int, default=256)
    parser.add_argument("--prefill-chunk-size", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--num-kvcache-blocks", type=int, default=-1)
    parser.add_argument("--max-aclgraph-entries", type=int, default=32)
    parser.add_argument("--target-verification-graph-buckets", type=int, default=8)
    parser.add_argument("--target-fallback-max-batch", type=int, default=0)
    parser.add_argument("--max-target-batch", type=int, default=0)
    parser.add_argument("--priority-burst", type=int, default=2)
    parser.add_argument("--urgency-threshold", type=float, default=0.75)
    parser.add_argument("--acceptance-floor", type=float, default=0.4)
    parser.add_argument("--acceptance-ema-alpha", type=float, default=0.2)
    parser.add_argument("--batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--http-max-batch-size", type=int)
    parser.add_argument("--worker-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-mc2", action="store_true")
    parser.add_argument(
        "--mc2-profile",
        help="Identity-bound TP MC2 numerical/performance qualification JSON.",
    )
    parser.add_argument("--draft-use-paged-attention", action="store_true")
    parser.add_argument("--target-use-paged-attention", action="store_true")
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--disable-cpu-binding", action="store_true")
    parser.add_argument("--precompile-decode-graphs", action="store_true")
    parser.add_argument(
        "--stable-graphs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--online-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--slo-priority",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.max_num_queued_seqs < args.max_num_seqs:
        raise ValueError("--max-num-queued-seqs must be at least --max-num-seqs")
    if args.prefill_chunk_size > args.max_num_queued_seqs:
        raise ValueError("--prefill-chunk-size must fit --max-num-queued-seqs")
    if args.http_max_batch_size is not None and args.http_max_batch_size > args.max_num_queued_seqs:
        raise ValueError("--http-max-batch-size must fit --max-num-queued-seqs")

    import uvicorn

    from vllm_ascend.spec_decode.pearl.api import PEARLConfig
    from vllm_ascend.spec_decode.pearl.http_server import create_specslo_app
    from vllm_ascend.spec_decode.pearl.roofline import ProfiledRoofline

    config = PEARLConfig(
        draft_model_path=args.draft_model,
        target_model_path=args.target_model,
        draft_tensor_parallel_size=args.draft_tp_size,
        target_tensor_parallel_size=args.target_tp_size,
        draft_dtype=args.draft_dtype,
        target_dtype=args.target_dtype,
        gamma=args.gamma,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_queued_seqs=args.max_num_queued_seqs,
        prefill_chunk_size=args.prefill_chunk_size,
        max_num_batched_tokens=args.max_model_len * args.prefill_chunk_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_kvcache_blocks=args.num_kvcache_blocks,
        max_aclgraph_entries=args.max_aclgraph_entries,
        target_verification_graph_buckets=args.target_verification_graph_buckets,
        enable_prefix_caching=not args.disable_prefix_caching,
        enable_continuous_batching=True,
        enable_preemptive_scheduling=True,
        enable_spec_rhythm=True,
        spec_rhythm_online_prefill=args.online_prefill,
        spec_rhythm_priority_mode=args.slo_priority,
        spec_rhythm_priority_burst=args.priority_burst,
        spec_rhythm_target_fallback_max_batch=args.target_fallback_max_batch,
        spec_rhythm_max_target_batch=args.max_target_batch,
        spec_rhythm_min_gamma=args.min_gamma,
        spec_rhythm_max_eager_tokens=args.max_eager_tokens,
        spec_rhythm_urgency_threshold=args.urgency_threshold,
        spec_rhythm_acceptance_floor=args.acceptance_floor,
        spec_rhythm_acceptance_ema_alpha=args.acceptance_ema_alpha,
        spec_rhythm_roofline=args.spec_rhythm_roofline,
        spec_rhythm_tree_width=args.tree_width,
        spec_rhythm_tree_depth=args.tree_depth,
        spec_rhythm_stable_graphs=args.stable_graphs,
        draft_use_paged_attention=args.draft_use_paged_attention,
        target_use_paged_attention=args.target_use_paged_attention,
        precompile_decode_graphs=args.precompile_decode_graphs,
        enable_cpu_binding=not args.disable_cpu_binding,
        enforce_eager=args.enforce_eager,
        enable_mc2=args.enable_mc2,
        mc2_profile=args.mc2_profile,
        seed=args.seed,
        worker_timeout_seconds=args.worker_timeout_seconds,
    )
    if not isinstance(config.spec_rhythm_roofline, ProfiledRoofline):
        raise ValueError(
            "SpecSLO production serving requires a strict measured roofline profile; "
            "a handwritten B mapping is not accepted"
        )
    app = create_specslo_app(
        config,
        batch_wait_ms=args.batch_wait_ms,
        max_batch_size=args.http_max_batch_size,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
