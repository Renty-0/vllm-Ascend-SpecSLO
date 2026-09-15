# SPDX-License-Identifier: Apache-2.0
"""Benchmark the native nano-PEARL speculative path with one engine load."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from examples.specslo_benchmark_report import (
    ONLINE_E2E_TIMING_SCOPE,
    capture_runtime_environment,
    sha256_file,
)


def _parse_roofline_argument(value: str):
    # Keep CLI parsing/help independent of the heavy model/worker imports
    # unless this optional profile argument is actually used.
    from vllm_ascend.spec_decode.pearl.roofline import parse_roofline_argument

    return parse_roofline_argument(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-tp-size", type=int, default=1)
    parser.add_argument("--target-tp-size", type=int, default=1)
    parser.add_argument("--draft-dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    parser.add_argument("--target-dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        help=(
            "Limit packed prefill requests without changing the decode batch size. "
            "In continuous mode this may be larger than the decode batch."
        ),
    )
    parser.add_argument(
        "--num-pearl-steps",
        type=int,
        help="Run upstream-compatible fixed-step PEARL instead of fixed-token generation.",
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument(
        "--num-prompts",
        type=int,
        help="Total requests per measurement. Defaults to the batch size.",
    )
    parser.add_argument(
        "--warmup-prompts",
        type=int,
        help="Warm up on only this many prompts. Defaults to one full batch.",
    )
    parser.add_argument(
        "--warmup-prompt-offset",
        type=int,
        default=0,
        help="Start warmup prompts at this dataset offset.",
    )
    parser.add_argument(
        "--warmup-max-tokens",
        type=int,
        help="Limit warmup generation tokens without changing the measured request limit.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of untimed warmup generations to run before measurement.",
    )
    parser.add_argument(
        "--graph-qualification-max-runs",
        type=int,
        default=6,
        help=(
            "Maximum untimed workload replays used to reach a sealed graph "
            "fixed point with --require-no-graph-fallback."
        ),
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--num-kvcache-blocks", type=int, default=-1)
    parser.add_argument("--max-aclgraph-entries", type=int, default=32)
    parser.add_argument("--target-verification-graph-buckets", type=int, default=8)
    parser.add_argument(
        "--target-verification-graph-post-counts",
        action="append",
        default=[],
        metavar="BATCH:COUNT,COUNT,...",
        help=(
            "Use explicit target verification graph post-verify counts for a "
            "batch size. May be repeated; counts must start at 0 and end at BATCH."
        ),
    )
    parser.add_argument("--auto-gamma-profile-sequence-length", type=int, default=256)
    parser.add_argument(
        "--profile-decode-steps",
        type=int,
        default=0,
        help="Synchronously profile this many decode steps per static chunk.",
    )
    parser.add_argument(
        "--profile-host-decode-steps",
        type=int,
        default=0,
        help=(
            "Record rank-local host timestamps for this many decode steps "
            "without inserting NPU synchronization or profile collectives."
        ),
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Stop after the synchronously profiled decode steps.",
    )
    parser.add_argument("--worker-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--require-no-graph-fallback",
        action="store_true",
        help="Fail unless every worker executes Graph with zero measured fallback.",
    )
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enable-continuous-batching", action="store_true")
    parser.add_argument(
        "--enable-preemptive-scheduling",
        action="store_true",
        help="Balance decode rounds across all resident continuous requests.",
    )
    parser.add_argument("--enable-spec-rhythm", action="store_true")
    parser.add_argument(
        "--spec-rhythm-linear-full-window",
        action="store_true",
        help=(
            "Use the independent full-window protocol for fixed-gamma serial "
            "linear SpecRhythm (requires min-gamma == gamma and tree 1x1). "
            "The legacy nano-PEARL protocol remains the default."
        ),
    )
    parser.add_argument(
        "--spec-rhythm-linear-eager-cross-graph-bucket",
        action="store_true",
        help=(
            "Let measured W admit fixed-gamma rolling-eager rows into the next "
            "serial-draft graph bucket (experimental; disabled by default)."
        ),
    )
    parser.add_argument(
        "--spec-rhythm-online-prefill",
        action="store_true",
        help="Prefill only the initial decode bucket and prefill later arrivals on admission.",
    )
    parser.add_argument(
        "--spec-rhythm-prefill-coalesce-min-requests",
        type=int,
        default=1,
        help=(
            "Wait for this many arrived requests before an online fixed-serial "
            "prefill (1 disables coalescing)."
        ),
    )
    parser.add_argument(
        "--spec-rhythm-prefill-coalesce-max-wait-ms",
        type=float,
        default=0.0,
        help=(
            "Bound online fixed-serial prefill coalescing wait in milliseconds "
            "(0 disables coalescing)."
        ),
    )
    parser.add_argument(
        "--spec-rhythm-prefill-token-chunk-size",
        type=int,
        default=0,
        help=(
            "Cap aggregate prompt tokens in one online SpecRhythm prefill "
            "submission (0 disables cross-cycle token chunking)."
        ),
    )
    parser.add_argument(
        "--spec-rhythm-merge-ready-homes",
        action="store_true",
        help=(
            "Merge both ready logical homes into one target forward. This is "
            "an opt-in throughput probe; the default follows the paper's "
            "alternating dual-batch schedule."
        ),
    )
    parser.add_argument(
        "--spec-rhythm-stable-graphs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use fixed paged-attention graph buckets for SpecRhythm. Disable only for explicit dynamic-FIA experiments."
        ),
    )
    parser.add_argument(
        "--spec-rhythm-slo-priority",
        action="store_true",
        help="Allow a bounded priority burst for ready requests that have exceeded their TPOT SLO.",
    )
    parser.add_argument(
        "--spec-rhythm-priority-burst",
        type=int,
        default=2,
        help="Maximum consecutive target rounds granted to an urgent ready home.",
    )
    parser.add_argument(
        "--spec-rhythm-target-fallback-max-batch",
        type=int,
        default=0,
        help=(
            "Use exact target-only token steps while the resident SpecRhythm batch "
            "is at or below this size (0 disables it)."
        ),
    )
    parser.add_argument(
        "--spec-rhythm-max-target-batch",
        type=int,
        default=0,
        help=(
            "Cap rows in one SpecRhythm target verification forward. Zero keeps "
            "the default one-home cap; smaller values reduce TPOT at the cost "
            "of more verification rounds."
        ),
    )
    parser.add_argument("--spec-rhythm-min-gamma", type=int, default=1)
    parser.add_argument("--spec-rhythm-max-eager-tokens", type=int, default=0)
    parser.add_argument(
        "--spec-rhythm-eager-reserve-tokens",
        type=int,
        default=0,
        help=(
            "Maximum global-B slots reserved for complete eligible rolling-eager "
            "proposals before normal admission (0 preserves normal-first scheduling)."
        ),
    )
    parser.add_argument(
        "--spec-rhythm-auto-eager-tokens",
        action="store_true",
        help=(
            "Enable rolling eager proposals up to the fixed gamma. Per-request "
            "manifest gamma values still provide the tighter cap."
        ),
    )
    parser.add_argument("--spec-rhythm-urgency-threshold", type=float, default=0.75)
    parser.add_argument("--spec-rhythm-acceptance-floor", type=float, default=0.4)
    parser.add_argument("--spec-rhythm-acceptance-ema-alpha", type=float, default=0.2)
    parser.add_argument(
        "--spec-rhythm-cpu-verdict",
        action="store_true",
        help=(
            "Build the tiny greedy verification verdict on CPU and copy back "
            "only two integers per request. Useful for measuring control-plane "
            "overhead on Ascend; disabled by default."
        ),
    )
    parser.add_argument(
        "--spec-rhythm-roofline",
        type=_parse_roofline_argument,
        help="Legacy JSON budgets or a strict measured profile JSON file path.",
    )
    parser.add_argument(
        "--spec-rhythm-verification-budget",
        type=int,
        help="Fixed global candidate-token budget B; overrides gamma-derived roofline.",
    )
    parser.add_argument("--spec-rhythm-draft-token-budget", type=int)
    parser.add_argument("--spec-rhythm-tree-width", type=int, default=1)
    parser.add_argument("--spec-rhythm-tree-depth", type=int, default=1)
    parser.add_argument("--spec-rhythm-request-max-gamma", type=int)
    parser.add_argument("--slo-tpot-ms", type=float)
    parser.add_argument("--slo-class")
    parser.add_argument("--pad-finished-requests", action="store_true")
    parser.add_argument("--draft-use-paged-attention", action="store_true")
    parser.add_argument("--target-use-paged-attention", action="store_true")
    parser.add_argument(
        "--enable-mc2",
        action="store_true",
        help=(
            "Fuse TP projection, HCCL all-reduce, residual add and RMSNorm when the Ascend MC2 extension is available."
        ),
    )
    parser.add_argument(
        "--mc2-profile",
        help="Identity-bound MC2 numerical/performance qualification JSON.",
    )
    parser.add_argument(
        "--draft-use-production-rope",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use vLLM-Ascend's fused rotary operator in the draft model.",
    )
    parser.add_argument(
        "--target-use-production-rope",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use vLLM-Ascend's fused rotary operator in the target model.",
    )
    parser.add_argument(
        "--precompile-decode-graphs",
        action="store_true",
        help="Compile fixed paged-attention decode graph buckets during engine initialization.",
    )
    parser.add_argument(
        "--precompile-serial-draft-graphs",
        action="store_true",
        help=(
            "Precompile and changed-input qualify only fixed-gamma serial "
            "draft graphs, leaving target verification graphs lazy."
        ),
    )
    parser.add_argument(
        "--disable-cpu-binding",
        action="store_true",
        help="Disable the production vLLM-Ascend CPU and IRQ affinity policy.",
    )
    parser.add_argument(
        "--output-json",
        help="Also write the final benchmark payload to this path.",
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--gsm8k", help="Path to a GSM8K parquet or JSONL file.")
    prompt_group.add_argument(
        "--request-manifest",
        help="JSONL rows with prompt and max_tokens, shared with the baseline.",
    )
    return parser


def _load_prompts(
    prompt: str | None,
    gsm8k: str | None,
    request_manifest: str | None,
    max_samples: int,
) -> tuple[list[str], list[int] | None, list[dict] | None]:
    if prompt is not None:
        return [prompt] * max_samples, None, None
    if request_manifest is not None:
        rows = [json.loads(line) for line in Path(request_manifest).read_text(encoding="utf-8").splitlines() if line]
        if len(rows) < max_samples:
            raise ValueError(f"Request manifest contains {len(rows)} rows, but {max_samples} were requested.")
        selected = rows[:max_samples]
        return (
            [str(row["prompt"]) for row in selected],
            [int(row["max_tokens"]) for row in selected],
            selected,
        )
    dataset_path = Path(gsm8k)
    if dataset_path.suffix == ".jsonl":
        rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
        if len(rows) < max_samples:
            raise ValueError(f"GSM8K contains {len(rows)} rows, but {max_samples} were requested.")
        return [str(row["turns"][0]) for row in rows[:max_samples]], None, None
    import pyarrow.parquet as pq

    rows = pq.read_table(dataset_path, columns=["question"]).to_pylist()
    if len(rows) < max_samples:
        raise ValueError(f"GSM8K contains {len(rows)} rows, but batch size {max_samples} was requested.")
    return [str(row["question"]) for row in rows[:max_samples]], None, None


def _add_requests(engine, prompts, sampling_params) -> None:
    if isinstance(sampling_params, Sequence):
        if len(prompts) != len(sampling_params):
            raise ValueError("PEARL requires one SamplingParams value per prompt.")
        for prompt, params in zip(prompts, sampling_params):
            engine.add_request(prompt, params)
        return
    for prompt in prompts:
        engine.add_request(prompt, sampling_params)


def _sampling_max_token_limits(sampling_params, count: int) -> list[int]:
    """Materialize the exact per-request output limits used by one run."""

    if isinstance(sampling_params, Sequence):
        if len(sampling_params) != count:
            raise ValueError("Expected one SamplingParams value per request.")
        return [int(params.max_tokens) for params in sampling_params]
    return [int(sampling_params.max_tokens)] * count


def _parse_target_graph_post_counts(
    values: Sequence[str],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    parsed: list[tuple[int, tuple[int, ...]]] = []
    for value in values:
        try:
            batch_size_text, post_counts_text = value.split(":", 1)
            post_counts = tuple(int(item) for item in post_counts_text.split(",") if item)
            parsed.append((int(batch_size_text), post_counts))
        except ValueError as error:
            raise ValueError("--target-verification-graph-post-counts must use BATCH:COUNT,COUNT,...") from error
    return tuple(parsed)


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise ValueError("Every speculative batch size must be positive.")
    if args.num_prompts is not None and args.num_prompts <= 0:
        raise ValueError("The speculative prompt count must be positive.")
    if args.warmup_prompts is not None and args.warmup_prompts <= 0:
        raise ValueError("--warmup-prompts must be positive.")
    if args.warmup_prompt_offset < 0:
        raise ValueError("--warmup-prompt-offset must be non-negative.")
    if args.warmup_max_tokens is not None and args.warmup_max_tokens <= 0:
        raise ValueError("--warmup-max-tokens must be positive.")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative.")
    if args.graph_qualification_max_runs <= 0:
        raise ValueError("--graph-qualification-max-runs must be positive.")
    if (
        args.require_no_graph_fallback
        and max(2, args.warmup_runs) > args.graph_qualification_max_runs
    ):
        raise ValueError(
            "--graph-qualification-max-runs must allow at least two runs and "
            "cover every requested warmup run."
        )
    if args.spec_rhythm_priority_burst <= 0:
        raise ValueError("--spec-rhythm-priority-burst must be positive.")
    if args.spec_rhythm_target_fallback_max_batch < 0:
        raise ValueError("--spec-rhythm-target-fallback-max-batch must be non-negative.")
    if args.spec_rhythm_max_target_batch < 0:
        raise ValueError("--spec-rhythm-max-target-batch must be non-negative.")
    if args.spec_rhythm_prefill_coalesce_min_requests <= 0:
        raise ValueError("--spec-rhythm-prefill-coalesce-min-requests must be positive.")
    if (
        not math.isfinite(args.spec_rhythm_prefill_coalesce_max_wait_ms)
        or args.spec_rhythm_prefill_coalesce_max_wait_ms < 0
    ):
        raise ValueError(
            "--spec-rhythm-prefill-coalesce-max-wait-ms must be finite and non-negative."
        )
    if args.spec_rhythm_prefill_token_chunk_size < 0:
        raise ValueError(
            "--spec-rhythm-prefill-token-chunk-size must be non-negative."
        )
    if args.profile_decode_steps < 0:
        raise ValueError("--profile-decode-steps must be non-negative.")
    if args.profile_host_decode_steps < 0:
        raise ValueError("--profile-host-decode-steps must be non-negative.")
    if args.profile_only and args.profile_decode_steps == 0:
        raise ValueError("--profile-only requires --profile-decode-steps to be positive.")
    if args.require_no_graph_fallback and args.enforce_eager:
        raise ValueError("--require-no-graph-fallback cannot be combined with --enforce-eager.")
    if args.num_pearl_steps is not None and args.num_pearl_steps <= 0:
        raise ValueError("--num-pearl-steps must be positive.")
    if args.num_pearl_steps is not None and args.enable_continuous_batching:
        raise ValueError("Fixed-step PEARL does not support continuous batching.")
    max_batch_size = max(args.batch_sizes)
    target_graph_post_counts = _parse_target_graph_post_counts(args.target_verification_graph_post_counts)
    if args.spec_rhythm_auto_eager_tokens and args.gamma <= 0:
        raise ValueError("--spec-rhythm-auto-eager-tokens requires a positive fixed --gamma.")
    from vllm_ascend.spec_decode.pearl import PEARLConfig, PEARLEngine, SamplingParams

    prompt_count = args.num_prompts or max_batch_size
    max_warmup_count = args.warmup_prompts or max_batch_size
    load_count = max(prompt_count, args.warmup_prompt_offset + max_warmup_count) if args.warmup_runs else prompt_count
    prompts, request_max_tokens, request_metadata = _load_prompts(
        args.prompt,
        args.gsm8k,
        args.request_manifest,
        load_count,
    )
    request_has_slo = bool(
        args.slo_tpot_ms is not None
        or args.slo_class is not None
        or (
            request_metadata
            and any(row.get("slo_tpot_ms") is not None or row.get("slo_class") is not None for row in request_metadata)
        )
    )
    # SpecSLO's rolling-eager stage is mandatory when the workload carries a
    # TPOT/class constraint.  An explicit CLI cap still wins; zero means the
    # bounded default of one gamma window for a constrained workload.
    effective_eager_cap = (
        args.gamma
        if args.spec_rhythm_auto_eager_tokens
        else (args.spec_rhythm_max_eager_tokens or (args.gamma if args.enable_spec_rhythm and request_has_slo else 0))
    )
    config = PEARLConfig(
        draft_model_path=args.draft_model,
        target_model_path=args.target_model,
        draft_tensor_parallel_size=args.draft_tp_size,
        target_tensor_parallel_size=args.target_tp_size,
        draft_dtype=args.draft_dtype,
        target_dtype=args.target_dtype,
        max_num_batched_tokens=args.max_model_len * max_batch_size,
        max_num_seqs=max_batch_size,
        prefill_chunk_size=args.prefill_chunk_size,
        # A scaling point may intentionally request a service batch larger
        # than the finite replay trace (for example the exact 6:2:2 60-request
        # mix at the B64 endpoint).  Queue capacity must still cover the engine
        # batch; otherwise PEARLConfig rejects a valid under-filled endpoint
        # before any request is submitted.
        max_num_queued_seqs=(
            max(prompt_count, max_batch_size)
            if args.enable_continuous_batching or args.enable_spec_rhythm
            else None
        ),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_kvcache_blocks=args.num_kvcache_blocks,
        max_aclgraph_entries=args.max_aclgraph_entries,
        target_verification_graph_buckets=args.target_verification_graph_buckets,
        target_verification_graph_post_counts=target_graph_post_counts,
        auto_gamma_profile_sequence_length=args.auto_gamma_profile_sequence_length,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_continuous_batching=(args.enable_continuous_batching or args.enable_spec_rhythm),
        enable_preemptive_scheduling=(args.enable_preemptive_scheduling or args.enable_spec_rhythm),
        enable_spec_rhythm=args.enable_spec_rhythm,
        spec_rhythm_linear_full_window=args.spec_rhythm_linear_full_window,
        spec_rhythm_linear_eager_cross_graph_bucket=(
            args.spec_rhythm_linear_eager_cross_graph_bucket
        ),
        spec_rhythm_online_prefill=args.spec_rhythm_online_prefill,
        spec_rhythm_prefill_coalesce_min_requests=(
            args.spec_rhythm_prefill_coalesce_min_requests
        ),
        spec_rhythm_prefill_coalesce_max_wait_ms=(
            args.spec_rhythm_prefill_coalesce_max_wait_ms
        ),
        spec_rhythm_prefill_token_chunk_size=(
            args.spec_rhythm_prefill_token_chunk_size
        ),
        spec_rhythm_merge_ready_homes=args.spec_rhythm_merge_ready_homes,
        spec_rhythm_stable_graphs=args.spec_rhythm_stable_graphs,
        spec_rhythm_priority_mode=args.spec_rhythm_slo_priority,
        spec_rhythm_priority_burst=args.spec_rhythm_priority_burst,
        spec_rhythm_target_fallback_max_batch=args.spec_rhythm_target_fallback_max_batch,
        spec_rhythm_max_target_batch=args.spec_rhythm_max_target_batch,
        spec_rhythm_min_gamma=args.spec_rhythm_min_gamma,
        spec_rhythm_max_eager_tokens=effective_eager_cap,
        spec_rhythm_eager_reserve_tokens=args.spec_rhythm_eager_reserve_tokens,
        spec_rhythm_urgency_threshold=args.spec_rhythm_urgency_threshold,
        spec_rhythm_acceptance_floor=args.spec_rhythm_acceptance_floor,
        spec_rhythm_acceptance_ema_alpha=args.spec_rhythm_acceptance_ema_alpha,
        spec_rhythm_cpu_verdict=args.spec_rhythm_cpu_verdict,
        spec_rhythm_roofline=args.spec_rhythm_roofline,
        spec_rhythm_verification_budget=args.spec_rhythm_verification_budget,
        spec_rhythm_draft_token_budget=args.spec_rhythm_draft_token_budget,
        spec_rhythm_tree_width=args.spec_rhythm_tree_width,
        spec_rhythm_tree_depth=args.spec_rhythm_tree_depth,
        pad_finished_requests=args.pad_finished_requests,
        draft_use_paged_attention=args.draft_use_paged_attention,
        target_use_paged_attention=args.target_use_paged_attention,
        enable_mc2=args.enable_mc2,
        mc2_profile=args.mc2_profile,
        draft_use_production_rope=args.draft_use_production_rope,
        target_use_production_rope=args.target_use_production_rope,
        precompile_decode_graphs=args.precompile_decode_graphs,
        precompile_serial_draft_graphs=args.precompile_serial_draft_graphs,
        enable_cpu_binding=not args.disable_cpu_binding,
        profile_decode_steps=0,
        profile_host_decode_steps=0,
        stop_after_profiled_decode_steps=False,
        enforce_eager=args.enforce_eager,
        gamma=args.gamma,
        seed=args.seed,
        worker_timeout_seconds=args.worker_timeout_seconds,
    )
    sampling_params = (
        [
            SamplingParams(
                temperature=0.0,
                max_tokens=int(row["max_tokens"]),
                ignore_eos=True,
                request_id=row.get("request_id"),
                arrival_ts=(float(row["arrival_ts"]) if row.get("arrival_ts") is not None else None),
                slo_tpot_ms=(float(row["slo_tpot_ms"]) if row.get("slo_tpot_ms") is not None else None),
                slo_class=row.get("slo_class"),
                spec_rhythm_max_gamma=(
                    min(
                        int(row["per_request_gamma"]),
                        args.spec_rhythm_request_max_gamma,
                    )
                    if row.get("per_request_gamma") is not None
                    and args.spec_rhythm_request_max_gamma is not None
                    else (
                        int(row["per_request_gamma"])
                        if row.get("per_request_gamma") is not None
                        else args.spec_rhythm_request_max_gamma
                    )
                ),
            )
            for row in request_metadata
        ]
        if request_metadata is not None
        else SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            ignore_eos=True,
            slo_tpot_ms=args.slo_tpot_ms,
            slo_class=args.slo_class,
            spec_rhythm_max_gamma=args.spec_rhythm_request_max_gamma,
        )
    )
    warmup_sampling_params = (
        [
            replace(
                value,
                max_tokens=min(
                    value.max_tokens,
                    args.warmup_max_tokens or value.max_tokens,
                ),
                arrival_ts=None,
            )
            for value in sampling_params
        ]
        if isinstance(sampling_params, list)
        else replace(
            sampling_params,
            max_tokens=args.warmup_max_tokens or args.max_tokens,
            arrival_ts=None,
        )
    )

    results = []
    with PEARLEngine(config) as engine:
        # Match the production baseline: tokenize before starting the arrival
        # trace, then include enqueue/IPC/generation in the measured window.
        tokenized_prompts = [
            list(
                engine.tokenizer.encode(
                    engine.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            )
            for prompt in prompts
        ]
        first_prompt_token_ids = tokenized_prompts[0]
        for batch_size in args.batch_sizes:
            engine.configure_decode_profiling(0)
            measured_prompts = tokenized_prompts[: args.num_prompts or batch_size]
            measured_sampling_params = (
                sampling_params[: len(measured_prompts)] if isinstance(sampling_params, list) else sampling_params
            )

            def materialize_arrivals(
                params,
                *,
                metadata_offset=0,
            ):
                if request_metadata is None:
                    return params
                arrival_origin = time.time()
                rows = request_metadata[
                    metadata_offset : metadata_offset + len(params)
                ]
                return [
                    replace(
                        value,
                        arrival_ts=(
                            float(row["arrival_ts"])
                            if row.get("arrival_ts") is not None
                            else arrival_origin + float(row.get("arrival_offset_sec", 0.0))
                        ),
                    )
                    for value, row in zip(
                        params,
                        rows,
                    )
                ]

            warmup_count = args.warmup_prompts or batch_size
            warmup_start = args.warmup_prompt_offset
            if args.require_no_graph_fallback:
                if warmup_start != 0:
                    raise ValueError(
                        "Strict graph qualification requires --warmup-prompt-offset=0."
                    )
                if args.warmup_prompts not in (None, len(measured_prompts)):
                    raise ValueError(
                        "Strict graph qualification must replay every measured prompt; "
                        "omit --warmup-prompts or set it to --num-prompts."
                    )
                if args.warmup_max_tokens is not None:
                    raise ValueError(
                        "Strict graph qualification must use the measured output lengths; "
                        "omit --warmup-max-tokens."
                    )
                warmup_count = len(measured_prompts)
            warmup_prompts = tokenized_prompts[warmup_start : warmup_start + warmup_count]
            current_warmup_params = (
                warmup_sampling_params[warmup_start : warmup_start + warmup_count]
                if isinstance(warmup_sampling_params, list)
                else warmup_sampling_params
            )
            if args.require_no_graph_fallback:
                # A previous batch-size point may have sealed the same
                # persistent workers.  Qualification is always untimed, so
                # reopen discovery before replaying this point's exact trace.
                engine.unseal_graph_cache()
            minimum_warmup_runs = (
                max(2, args.warmup_runs)
                if args.require_no_graph_fallback
                else args.warmup_runs
            )
            warmup_run_limit = (
                args.graph_qualification_max_runs
                if args.require_no_graph_fallback
                else args.warmup_runs
            )
            qualification_complete = not args.require_no_graph_fallback
            qualification_issues: list[str] = []
            warmup_runs_executed = 0
            for warmup_run in range(warmup_run_limit):
                # A graph entry first sees only the values used for capture.
                # Replaying an identical deterministic trace cannot qualify
                # its changed-input/task-update path, especially for tail
                # batch shapes visited only once per generation.  Rotate the
                # already-selected prompt rows on later warmups so every
                # recurring physical shape sees different token/KV contents
                # before the measured window.  Keep SamplingParams in their
                # original rows: this is graph qualification, not a second
                # measured workload, and preserves the requested SLO mix and
                # output limits used to exercise the scheduler.
                # Strict mode alternates the original row order with a
                # one-row rotation.  Reusing these two exact workload traces
                # reaches a bounded union of real shapes instead of growing
                # the graph inventory with a new synthetic layout each pass.
                rotation_seed = (
                    warmup_run % 2
                    if args.require_no_graph_fallback
                    else warmup_run
                )
                rotation = (
                    rotation_seed % len(warmup_prompts)
                    if warmup_prompts
                    else 0
                )
                run_warmup_prompts = (
                    warmup_prompts[rotation:]
                    + warmup_prompts[:rotation]
                )
                graph_metrics_before_warmup = engine.last_worker_metrics
                _add_requests(
                    engine,
                    run_warmup_prompts,
                    materialize_arrivals(
                        current_warmup_params,
                        metadata_offset=warmup_start,
                    ),
                )
                if args.num_pearl_steps is None:
                    engine.generate()
                else:
                    engine.bench_generate(args.num_pearl_steps)
                warmup_runs_executed += 1
                if (
                    args.require_no_graph_fallback
                    and warmup_runs_executed >= minimum_warmup_runs
                ):
                    qualification_complete, qualification_issues = (
                        _graph_qualification_fixed_point(
                            graph_metrics_before_warmup,
                            engine.last_worker_metrics,
                        )
                    )
                    if qualification_complete:
                        break
                    if _graph_qualification_can_prune(
                        graph_metrics_before_warmup,
                        engine.last_worker_metrics,
                    ):
                        # Entries left unvalidated after a capture-free full
                        # trace belong only to an earlier cold schedule.  Drop
                        # exactly those entries and retry.  Any genuinely hot
                        # key must then recapture, preventing a false fixed
                        # point before sealing.
                        engine.prune_unvalidated_graph_entries()

            if args.require_no_graph_fallback:
                if not qualification_complete:
                    raise RuntimeError(
                        "ACLGraph qualification did not reach a fixed point "
                        f"within {warmup_run_limit} workload replays: "
                        f"{qualification_issues or ['unknown qualification state']}"
                    )
                warmup_worker_metrics = engine.seal_graph_cache()
            else:
                warmup_worker_metrics = engine.last_worker_metrics

            if args.profile_only:
                engine.configure_decode_profiling(
                    args.profile_decode_steps,
                    stop_after_profiled_decode_steps=True,
                    profile_host_decode_steps=args.profile_host_decode_steps,
                )
                _add_requests(
                    engine,
                    measured_prompts,
                    materialize_arrivals(measured_sampling_params),
                )
                engine.generate()
                # The intrusive profile is outside the reported measurement,
                # but it runs against the same sealed graph set.  Snapshot its
                # counters so the following timed deltas remain measurement-only.
                warmup_worker_metrics = engine.last_worker_metrics

            engine.configure_decode_profiling(
                args.profile_decode_steps,
                args.profile_only,
                args.profile_host_decode_steps,
            )
            started = time.perf_counter()
            _add_requests(
                engine,
                measured_prompts,
                materialize_arrivals(measured_sampling_params),
            )
            if args.num_pearl_steps is None:
                _, num_tokens, _, inference_elapsed = engine.generate()
            else:
                _, num_tokens, _, inference_elapsed = engine.bench_generate(args.num_pearl_steps)
            e2e_elapsed = time.perf_counter() - started
            output_tokens = sum(num_tokens)
            metrics = engine.last_metrics
            output_token_rows = [metric["completion_token_ids"] for metric in metrics]
            verified_tokens = sum(metric["verified_draft_tokens"] for metric in metrics)
            accepted_tokens = sum(metric["accepted_draft_tokens"] for metric in metrics)
            chunk_metrics = metrics[:1] if args.enable_continuous_batching else metrics[::batch_size]
            decode_phase_seconds = {
                phase: sum(metric["decode_phase_seconds"][phase] for metric in chunk_metrics)
                for phase in chunk_metrics[0]["decode_phase_seconds"]
            }
            measured_graph_deltas = _worker_aclgraph_deltas(
                warmup_worker_metrics,
                engine.last_worker_metrics,
            )
            # ``batch_size`` is the configured capacity.  A continuous/live
            # run may contain fewer requests (for example the p40 smoke test
            # with a B64 capacity), and the engine records that actual chunk
            # size in ``last_worker_metrics_by_chunk``.  Aggregate the chunk
            # that was really executed instead of silently returning no host
            # or device profile for every under-capacity workload.
            profile_batch_size = _resolve_profile_batch_size(
                batch_size,
                len(measured_prompts),
                continuous_batching=args.enable_continuous_batching,
            )
            if args.require_no_graph_fallback:
                _require_no_graph_fallback(
                    measured_graph_deltas,
                    require_full_window=args.spec_rhythm_linear_full_window,
                )
            results.append(
                {
                    "batch_size": batch_size,
                    "num_prompts": len(measured_prompts),
                    "warmup_runs_executed": warmup_runs_executed,
                    "graph_qualification_fixed_point": qualification_complete,
                    "num_static_chunks": (
                        1 if args.enable_continuous_batching else math.ceil(len(measured_prompts) / batch_size)
                    ),
                    "output_tokens": output_tokens,
                    "prompt_token_ids_sha256": hashlib.sha256(
                        json.dumps(measured_prompts, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "e2e_timing_scope": ONLINE_E2E_TIMING_SCOPE,
                    "request_output_token_limits": _sampling_max_token_limits(
                        measured_sampling_params,
                        len(measured_prompts),
                    ),
                    "warmup_output_token_limits": _sampling_max_token_limits(
                        current_warmup_params,
                        len(warmup_prompts),
                    ),
                    "inference_elapsed_seconds": inference_elapsed,
                    "e2e_elapsed_seconds": e2e_elapsed,
                    "inference_throughput_tokens_per_second": output_tokens / inference_elapsed,
                    "e2e_throughput_tokens_per_second": output_tokens / e2e_elapsed,
                    "acceptance_rate": accepted_tokens / verified_tokens if verified_tokens else 0.0,
                    "mean_accept_tokens": sum(metric["mean_accept_tokens"] for metric in metrics) / len(metrics),
                    "selected_gamma": metrics[0]["gamma"],
                    "decode_rounds": sum(metric["round_count"] for metric in chunk_metrics),
                    "prefill_elapsed_seconds": sum(metric["prefill_elapsed_seconds"] for metric in chunk_metrics),
                    "decode_elapsed_seconds": sum(metric["decode_elapsed_seconds"] for metric in chunk_metrics),
                    "slo": _summarize_slo_metrics(metrics, inference_elapsed, e2e_elapsed),
                    "request_verification_rounds": [metric["verification_rounds"] for metric in metrics],
                    "decode_phase_seconds": decode_phase_seconds,
                    "aclgraph_captures": max(metric["aclgraph_captures"] for metric in metrics),
                    "aclgraph_capture_attempts": max(metric["aclgraph_capture_attempts"] for metric in metrics),
                    "aclgraph_replays": max(metric["aclgraph_replays"] for metric in metrics),
                    "aclgraph_failed_captures": max(metric["aclgraph_failed_captures"] for metric in metrics),
                    "aclgraph_capacity_fallbacks": max(metric["aclgraph_capacity_fallbacks"] for metric in metrics),
                    "aclgraph_shape_fallbacks": max(metric["aclgraph_shape_fallbacks"] for metric in metrics),
                    "worker_aclgraph_metrics": engine.last_worker_metrics,
                    "measured_worker_aclgraph_deltas": measured_graph_deltas,
                    "worker_metrics_by_chunk": engine.last_worker_metrics_by_chunk,
                    "decode_profile": _aggregate_decode_profile(
                        engine.last_worker_metrics_by_chunk,
                        profile_batch_size,
                    ),
                    "decode_host_profile": _aggregate_decode_host_profile(
                        engine.last_worker_metrics_by_chunk,
                        profile_batch_size,
                    ),
                    # Per-cycle timestamps are populated only for the
                    # explicitly bounded intrusive profile window. Retaining
                    # them makes claimed Draft/Target overlap auditable rather
                    # than inferring it from two accumulated phase totals.
                    "decode_timeline": (
                        list(metrics[0].get("decode_timeline", ()))
                        if args.profile_decode_steps > 0
                        else []
                    ),
                    "output_token_ids_sha256": hashlib.sha256(
                        json.dumps(output_token_rows, separators=(",", ":")).encode()
                    ).hexdigest(),
                    # Retain rows for semantic regression against target-only;
                    # the hash remains the compact report-level check.
                    "output_token_ids": output_token_rows,
                    "first_output_token_ids": metrics[0]["completion_token_ids"],
                }
            )

    payload = {
        "backend": ("specslo-native-specrhythm" if args.enable_spec_rhythm else "nano-pearl-native-speculative"),
        "policy": "SpecSLO/SpecRhythm" if args.enable_spec_rhythm else "nano-PEARL",
        "draft_model": args.draft_model,
        "target_model": args.target_model,
        "draft_tensor_parallel_size": args.draft_tp_size,
        "target_tensor_parallel_size": args.target_tp_size,
        "draft_dtype": args.draft_dtype,
        "target_dtype": args.target_dtype,
        "gamma": args.gamma,
        "seed": args.seed,
        "runtime_environment": capture_runtime_environment(),
        "target_verification_graph_buckets": args.target_verification_graph_buckets,
        "target_verification_graph_post_counts": target_graph_post_counts,
        "auto_gamma_profile_sequence_length": args.auto_gamma_profile_sequence_length,
        "profile_decode_steps": args.profile_decode_steps,
        "profile_host_decode_steps": args.profile_host_decode_steps,
        "profile_only": args.profile_only,
        "profile_shape_warmup": args.profile_only,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "num_kvcache_blocks": args.num_kvcache_blocks,
        "max_aclgraph_entries": args.max_aclgraph_entries,
        "prefill_chunk_size": args.prefill_chunk_size,
        "enforce_eager": args.enforce_eager,
        "require_no_graph_fallback": args.require_no_graph_fallback,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_continuous_batching": (args.enable_continuous_batching or args.enable_spec_rhythm),
        "enable_preemptive_scheduling": (args.enable_preemptive_scheduling or args.enable_spec_rhythm),
        "enable_spec_rhythm": args.enable_spec_rhythm,
        "spec_rhythm_linear_full_window": args.spec_rhythm_linear_full_window,
        "spec_rhythm_linear_eager_cross_graph_bucket": (
            args.spec_rhythm_linear_eager_cross_graph_bucket
        ),
        "spec_rhythm_online_prefill": args.spec_rhythm_online_prefill,
        "spec_rhythm_prefill_coalesce_min_requests": (
            args.spec_rhythm_prefill_coalesce_min_requests
        ),
        "spec_rhythm_prefill_coalesce_max_wait_ms": (
            args.spec_rhythm_prefill_coalesce_max_wait_ms
        ),
        "spec_rhythm_prefill_token_chunk_size": (
            args.spec_rhythm_prefill_token_chunk_size
        ),
        "spec_rhythm_merge_ready_homes": args.spec_rhythm_merge_ready_homes,
        "spec_rhythm_stable_graphs": args.spec_rhythm_stable_graphs,
        "spec_rhythm_priority_mode": args.spec_rhythm_slo_priority,
        "spec_rhythm_priority_mode_resolved": (args.spec_rhythm_slo_priority or request_has_slo),
        "spec_rhythm_priority_burst": args.spec_rhythm_priority_burst,
        "spec_rhythm_target_fallback_max_batch": args.spec_rhythm_target_fallback_max_batch,
        "spec_rhythm_max_target_batch": args.spec_rhythm_max_target_batch,
        "spec_rhythm_min_gamma": args.spec_rhythm_min_gamma,
        "spec_rhythm_request_max_gamma": args.spec_rhythm_request_max_gamma,
        "spec_rhythm_draft_mode": (
            "serial_linear"
            if args.spec_rhythm_tree_width == 1 and args.spec_rhythm_tree_depth == 1
            else "tree"
        ),
        "spec_rhythm_max_eager_tokens": effective_eager_cap,
        "spec_rhythm_eager_reserve_tokens": args.spec_rhythm_eager_reserve_tokens,
        "spec_rhythm_slo_adaptive": request_has_slo,
        "spec_rhythm_auto_eager_tokens": args.spec_rhythm_auto_eager_tokens,
        "spec_rhythm_cpu_verdict": args.spec_rhythm_cpu_verdict,
        "spec_rhythm_roofline": args.spec_rhythm_roofline,
        "spec_rhythm_verification_budget": args.spec_rhythm_verification_budget,
        "spec_rhythm_draft_token_budget": args.spec_rhythm_draft_token_budget,
        "spec_rhythm_tree_width": args.spec_rhythm_tree_width,
        "spec_rhythm_tree_depth": args.spec_rhythm_tree_depth,
        "pad_finished_requests": args.pad_finished_requests,
        "draft_use_paged_attention": args.draft_use_paged_attention,
        "target_use_paged_attention": args.target_use_paged_attention,
        "enable_mc2": args.enable_mc2,
        "mc2_profile": args.mc2_profile,
        "draft_use_production_rope": args.draft_use_production_rope,
        "target_use_production_rope": args.target_use_production_rope,
        "precompile_decode_graphs": args.precompile_decode_graphs,
        "precompile_serial_draft_graphs": args.precompile_serial_draft_graphs,
        "enable_cpu_binding": not args.disable_cpu_binding,
        "max_tokens": args.max_tokens,
        "request_manifest": args.request_manifest,
        "request_manifest_sha256": (
            sha256_file(args.request_manifest)
            if args.request_manifest is not None
            else None
        ),
        "requested_output_tokens": (sum(request_max_tokens[:prompt_count]) if request_max_tokens is not None else None),
        "num_pearl_steps": args.num_pearl_steps,
        "warmup_prompts": args.warmup_prompts,
        "warmup_prompt_offset": args.warmup_prompt_offset,
        "warmup_max_tokens": args.warmup_max_tokens,
        "warmup_runs": args.warmup_runs,
        "warmup_excluded_from_measurement": True,
        "online_arrivals": bool(request_metadata),
        "graph_qualification_max_runs": args.graph_qualification_max_runs,
        "warmup_rotates_prompt_rows": (
            args.warmup_runs > 1 or args.require_no_graph_fallback
        ),
        "warmup_replays_request_arrivals_and_policy": bool(request_metadata),
        "first_prompt_token_ids": first_prompt_token_ids,
        "results": results,
    }
    output = json.dumps(payload, ensure_ascii=True)
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"{output}\n", encoding="utf-8")
    print(output)


def _summarize_slo_metrics(metrics, inference_elapsed: float, e2e_elapsed: float):
    """Aggregate per-request TPOT attainment and Goodput for a workload run."""
    constrained = [metric for metric in metrics if metric.get("slo_tpot_ms") is not None]
    if not constrained:
        return {
            "constrained_requests": 0,
            "attained_requests": None,
            "attainment": None,
            "goodput_tokens": None,
            "goodput_tokens_per_inference_second": None,
            "goodput_tokens_per_e2e_second": None,
            "by_class": {},
        }
    attained = sum(bool(metric.get("slo_attained")) for metric in constrained)
    goodput_tokens = sum(int(metric.get("slo_goodput_tokens", 0)) for metric in constrained)
    by_class = {}
    for slo_class in sorted({metric.get("slo_class") or "unspecified" for metric in constrained}):
        class_metrics = [metric for metric in constrained if (metric.get("slo_class") or "unspecified") == slo_class]
        class_attained = sum(bool(metric.get("slo_attained")) for metric in class_metrics)
        class_goodput = sum(int(metric.get("slo_goodput_tokens", 0)) for metric in class_metrics)
        by_class[slo_class] = {
            "requests": len(class_metrics),
            "attained_requests": class_attained,
            "attainment": class_attained / len(class_metrics),
            "goodput_tokens": class_goodput,
            "mean_tpot_ms": sum(float(metric.get("observed_tpot_ms", 0.0)) for metric in class_metrics)
            / len(class_metrics),
        }
    summary = {
        "constrained_requests": len(constrained),
        "attained_requests": attained,
        "attainment": attained / len(constrained),
        "goodput_tokens": goodput_tokens,
        "goodput_tokens_per_inference_second": (goodput_tokens / inference_elapsed if inference_elapsed > 0 else 0.0),
        "goodput_tokens_per_e2e_second": (goodput_tokens / e2e_elapsed if e2e_elapsed > 0 else 0.0),
        "by_class": by_class,
    }
    # Keep historical N-1 accounting separate from the paper's N denominator.
    # Linear/legacy results lacking paper fields are explicitly incomplete,
    # never converted by guessing their timing convention.
    from examples.specslo_slo_metrics import summarize_slo_rows

    arrival_values = [
        float(metric["arrival_ts"])
        for metric in constrained
        if metric.get("arrival_ts") is not None
    ]
    arrival_origin = min(arrival_values) if arrival_values else None
    request_rows = [
        {
            "request_id": metric.get("request_id"),
            "output_tokens": len(metric["completion_token_ids"]),
            "slo_class": metric.get("slo_class"),
            "slo_tpot_ms": metric["slo_tpot_ms"],
            "observed_tpot_ms": metric.get("observed_tpot_ms"),
            "paper_tpot_ms": metric.get("paper_tpot_ms"),
            # Preserve the worker-visible replay clock so a malformed online
            # run (for example all arrivals admitted in the first prefill) can
            # be diagnosed from the result artifact rather than only its log.
            "arrival_ts": metric.get("arrival_ts"),
            "arrival_offset_sec": (
                None
                if arrival_origin is None or metric.get("arrival_ts") is None
                else float(metric["arrival_ts"]) - arrival_origin
            ),
        }
        for metric in constrained
    ]
    summary["paper"] = summarize_slo_rows(
        request_rows,
        tpot_field="paper_tpot_ms",
        definition="same_decode_elapsed_ms / output_tokens",
        elapsed_seconds=e2e_elapsed,
    )
    summary["request_metrics"] = request_rows
    return summary


def _aggregate_decode_profile(worker_metrics_by_chunk, batch_size: int):
    full_chunks = [chunk for chunk in worker_metrics_by_chunk if chunk["batch_size"] == batch_size]
    totals = {
        "draft_compute": 0.0,
        "draft_to_target_communication": 0.0,
        "target_verify": 0.0,
        "target_to_draft_communication": 0.0,
        "wait_sync_state_update": 0.0,
    }
    component_totals = {
        "target_compute": 0.0,
        "target_verdict": 0.0,
        "wait_sync": 0.0,
        "state_update": 0.0,
    }
    detail_totals: dict[str, float] = {}
    profiled_steps = 0
    profiled_chunks = 0
    for chunk in full_chunks:
        workers = chunk["worker_metrics"]
        chunk_steps = max(int(worker.get("worker_profiled_decode_steps", 0)) for worker in workers)
        if chunk_steps == 0:
            continue
        draft_workers = [worker for worker in workers if worker["is_draft_rank"]]
        target_workers = [worker for worker in workers if not worker["is_draft_rank"]]
        draft_compute = max(worker["worker_profile_draft_compute_seconds"] for worker in draft_workers)
        draft_to_target = max(worker["worker_profile_draft_to_target_communication_seconds"] for worker in workers)
        target_compute = max(worker["worker_profile_target_compute_seconds"] for worker in target_workers)
        target_verdict = max(worker["worker_profile_target_verdict_seconds"] for worker in target_workers)
        target_to_draft = max(worker["worker_profile_target_to_draft_communication_seconds"] for worker in workers)
        wait_sync = max(worker["worker_profile_wait_sync_seconds"] for worker in workers)
        state_update = max(worker["worker_profile_state_update_seconds"] for worker in workers)
        totals["draft_compute"] += draft_compute
        totals["draft_to_target_communication"] += draft_to_target
        totals["target_verify"] += target_compute + target_verdict
        totals["target_to_draft_communication"] += target_to_draft
        totals["wait_sync_state_update"] += wait_sync + state_update
        component_totals["target_compute"] += target_compute
        component_totals["target_verdict"] += target_verdict
        component_totals["wait_sync"] += wait_sync
        component_totals["state_update"] += state_update
        detail_prefix = "worker_profile_detail_"
        detail_suffix = "_seconds"
        detail_names = {
            key[len(detail_prefix) : -len(detail_suffix)]
            for worker in workers
            for key in worker
            if key.startswith(detail_prefix) and key.endswith(detail_suffix)
        }
        for name in detail_names:
            detail_totals[name] = detail_totals.get(name, 0.0) + max(
                float(worker.get(f"{detail_prefix}{name}{detail_suffix}", 0.0))
                for worker in workers
            )
        profiled_steps += chunk_steps
        profiled_chunks += 1
    if profiled_steps == 0:
        return None
    return {
        "profiled_full_batch_chunks": profiled_chunks,
        "profiled_decode_steps": profiled_steps,
        "phase_seconds": totals,
        "phase_milliseconds_per_decode_step": {
            phase: seconds * 1000 / profiled_steps for phase, seconds in totals.items()
        },
        "component_seconds": component_totals,
        "component_milliseconds_per_decode_step": {
            phase: seconds * 1000 / profiled_steps for phase, seconds in component_totals.items()
        },
        "detail_seconds": detail_totals,
        "detail_milliseconds_per_decode_step": {
            phase: seconds * 1000 / profiled_steps for phase, seconds in detail_totals.items()
        },
    }


def _resolve_profile_batch_size(
    configured_batch_size: int,
    measured_prompt_count: int,
    *,
    continuous_batching: bool,
) -> int:
    """Return the worker chunk size used to select profiling records.

    Continuous SpecRhythm runs are sent to the workers as one logical chunk,
    whose recorded size is the number of requests actually submitted.  That
    can be smaller than the configured service capacity.  Static runs retain
    the historical full-chunk-only aggregation behavior so their final partial
    chunk is not mixed into a full-batch profile.
    """
    if configured_batch_size <= 0 or measured_prompt_count <= 0:
        raise ValueError("profile batch sizes must be positive")
    if continuous_batching:
        return min(measured_prompt_count, configured_batch_size)
    return configured_batch_size


def _aggregate_decode_host_profile(worker_metrics_by_chunk, batch_size: int):
    """Merge non-synchronizing rank-local timestamps into cycle diagnostics.

    ``time.perf_counter`` uses one host monotonic clock, including across the
    spawned workers, so absolute rank timestamps can be compared directly.
    No timing collective is inserted into the measured decode path.
    """
    cycles = []
    for chunk_index, chunk in enumerate(worker_metrics_by_chunk):
        if int(chunk.get("batch_size", 0)) != int(batch_size):
            continue
        workers = chunk.get("worker_metrics", ())
        rows_by_step: dict[int, list[dict]] = {}
        for worker in workers:
            for row in worker.get("worker_host_timeline", ()):
                rows_by_step.setdefault(int(row["step"]), []).append(row)
        for step, rows in sorted(rows_by_step.items()):
            draft_rows = [row for row in rows if int(row.get("is_draft_rank", 0))]
            target_rows = [row for row in rows if not int(row.get("is_draft_rank", 0))]
            if not draft_rows or not target_rows:
                continue
            draft = min(draft_rows, key=lambda row: int(row["rank"]))
            target = min(target_rows, key=lambda row: int(row["rank"]))

            def duration_ms(row, start: str, end: str) -> float:
                if start not in row or end not in row:
                    return 0.0
                return max(0.0, float(row[end]) - float(row[start])) * 1000.0

            cycle_start = min(float(row["cycle_start_seconds"]) for row in rows)
            cycle_end = max(float(row.get("cycle_end_seconds", cycle_start)) for row in rows)
            draft_start = float(draft.get("draft_start_seconds", cycle_start))
            draft_end = float(draft.get("draft_end_seconds", draft_start))
            target_start = float(target.get("target_start_seconds", cycle_start))
            target_end = float(target.get("target_end_seconds", target_start))
            correction_end = max(
                float(row.get("correction_end_seconds", row.get("publish_end_seconds", cycle_start)))
                for row in rows
            )
            cycles.append(
                {
                    "chunk": chunk_index,
                    "step": step,
                    "target_requests": int(target.get("target_requests", 0)),
                    "draft_requests": int(draft.get("draft_requests", 0)),
                    "verify_candidates": int(target.get("verify_candidates", 0)),
                    "cycle_wall_ms": max(0.0, cycle_end - cycle_start) * 1000.0,
                    "scheduler_critical_ms": max(
                        duration_ms(row, "cycle_start_seconds", "scheduler_end_seconds") for row in rows
                    ),
                    "scheduler_roof_critical_ms": max(
                        duration_ms(row, "cycle_start_seconds", "scheduler_roof_end_seconds") for row in rows
                    ),
                    "scheduler_plan_critical_ms": max(
                        duration_ms(row, "scheduler_roof_end_seconds", "scheduler_plan_end_seconds")
                        for row in rows
                    ),
                    "scheduler_budget_critical_ms": max(
                        duration_ms(row, "scheduler_plan_end_seconds", "scheduler_budget_end_seconds")
                        for row in rows
                    ),
                    "scheduler_tree_plan_critical_ms": max(
                        duration_ms(row, "scheduler_budget_end_seconds", "scheduler_end_seconds") for row in rows
                    ),
                    "draft_compute_host_ms": duration_ms(draft, "draft_start_seconds", "draft_end_seconds"),
                    "target_compute_host_ms": duration_ms(target, "target_start_seconds", "target_end_seconds"),
                    "host_compute_overlap_ms": max(
                        0.0,
                        min(draft_end, target_end) - max(draft_start, target_start),
                    )
                    * 1000.0,
                    "draft_to_target_critical_ms": max(
                        duration_ms(row, "exchange_start_seconds", "exchange_end_seconds") for row in rows
                    ),
                    "target_verdict_host_ms": duration_ms(
                        target, "verdict_start_seconds", "verdict_end_seconds"
                    ),
                    "target_to_draft_critical_ms": max(
                        duration_ms(row, "correction_start_seconds", "correction_end_seconds") for row in rows
                    ),
                    "state_update_critical_ms": max(
                        duration_ms(row, "state_start_seconds", "state_end_seconds") for row in rows
                    ),
                    "state_preflight_critical_ms": max(
                        duration_ms(row, "state_start_seconds", "state_preflight_end_seconds") for row in rows
                    ),
                    "state_consensus_critical_ms": max(
                        duration_ms(row, "state_preflight_end_seconds", "state_consensus_end_seconds")
                        for row in rows
                    ),
                    "state_compaction_critical_ms": max(
                        duration_ms(row, "state_consensus_end_seconds", "state_compaction_end_seconds")
                        for row in rows
                    ),
                    "state_request_commit_critical_ms": max(
                        duration_ms(row, "state_compaction_end_seconds", "state_request_commit_end_seconds")
                        for row in rows
                    ),
                    "post_correction_tail_ms": max(0.0, cycle_end - correction_end) * 1000.0,
                    "rank_cycle_ms": {
                        str(int(row["rank"])): duration_ms(
                            row, "cycle_start_seconds", "cycle_end_seconds"
                        )
                        for row in rows
                    },
                }
            )
    if not cycles:
        return None
    return {
        "mode": "rank-local host timestamps; no added NPU synchronize or collective",
        "cycles": cycles,
    }


def _worker_aclgraph_deltas(before_workers, after_workers):
    execution_counter_names = tuple(
        f"aclgraph_{kind}_{name}"
        for kind in ("generic", "draft", "target")
        for name in (
            "total_calls",
            "capture_replay_calls",
            "replay_calls",
            "eager_fallback_calls",
            "disabled_entry_calls",
            "unclassified_calls",
            "runtime_validation_calls",
            "runtime_validation_failures",
            "changed_input_validation_calls",
            "logical_row_expansion_validation_calls",
        )
    )
    counter_names = (
        "aclgraph_captures",
        "aclgraph_capture_attempts",
        "aclgraph_replays",
        "aclgraph_failed_captures",
        "aclgraph_capacity_fallbacks",
        "aclgraph_shape_fallbacks",
        "aclgraph_runtime_validation_replays",
        "spec_rhythm_linear_draft_full_chain_calls",
        "spec_rhythm_linear_draft_full_chain_capture_replay_calls",
        "spec_rhythm_linear_draft_full_chain_replay_calls",
        "spec_rhythm_linear_draft_full_chain_eager_fallback_calls",
        "spec_rhythm_linear_draft_full_chain_unclassified_calls",
        "spec_rhythm_linear_draft_stepwise_calls",
        "spec_rhythm_linear_draft_stepwise_model_calls",
        *execution_counter_names,
    )
    before_by_rank = {int(worker["rank"]): worker for worker in before_workers}
    return [
        {
            "rank": int(worker["rank"]),
            "is_draft_rank": int(worker.get("is_draft_rank", 0)),
            **{
                f"{name}_delta": int(worker.get(name, 0))
                - int(before_by_rank.get(int(worker["rank"]), {}).get(name, 0))
                for name in counter_names
            },
        }
        for worker in after_workers
    ]


def _graph_qualification_fixed_point(before_workers, after_workers):
    """Return whether one untimed trace closed and qualified its graph set.

    The final qualifying pass must already be a production-hot trace: capture,
    runtime validation, fallback, disabled entries, and remaining unvalidated
    entries are all disallowed.  Validation may qualify entries on an earlier
    untimed pass, but never on the pass used to prove the fixed point.
    """

    issues: list[str] = []
    before_ranks = {int(worker["rank"]) for worker in before_workers}
    after_ranks = {int(worker["rank"]) for worker in after_workers}
    if before_ranks != after_ranks:
        issues.append(
            "worker ranks changed during qualification: "
            f"before={sorted(before_ranks)}, after={sorted(after_ranks)}"
        )
        return False, issues

    deltas = _worker_aclgraph_deltas(before_workers, after_workers)
    delta_by_rank = {int(worker["rank"]): worker for worker in deltas}
    for worker in after_workers:
        rank = int(worker["rank"])
        entries = int(worker.get("aclgraph_entries", 0))
        unvalidated = int(
            worker.get("aclgraph_unvalidated_entries", -1)
        )
        if entries <= 0:
            issues.append(f"rank {rank} has no resident graph entries")
        disabled = int(worker.get("aclgraph_disabled_entries", 0))
        if disabled:
            issues.append(
                f"rank {rank} retains {disabled} disabled graph entries"
            )
        if unvalidated < 0:
            issues.append(
                f"rank {rank} did not report graph qualification status"
            )
        elif unvalidated:
            by_kind = {
                kind: int(
                    worker.get(
                        f"aclgraph_{kind}_unvalidated_entries",
                        0,
                    )
                )
                for kind in ("generic", "draft", "target")
            }
            issues.append(
                f"rank {rank} retains {unvalidated} unvalidated entries "
                f"({by_kind})"
            )

        delta = delta_by_rank[rank]
        captures = {
            name: int(delta.get(name, 0))
            for name in (
                "aclgraph_capture_attempts_delta",
                "aclgraph_captures_delta",
            )
        }
        if any(captures.values()):
            issues.append(f"rank {rank} discovered graphs {captures}")
        runtime_validations = int(
            delta.get("aclgraph_runtime_validation_replays_delta", 0)
        )
        if runtime_validations:
            issues.append(
                f"rank {rank} performed {runtime_validations} runtime validations"
            )
        fallbacks = {
            name: int(delta.get(name, 0))
            for name in (
                "aclgraph_failed_captures_delta",
                "aclgraph_capacity_fallbacks_delta",
                "aclgraph_shape_fallbacks_delta",
            )
        }
        if any(fallbacks.values()):
            issues.append(f"rank {rank} used graph fallback {fallbacks}")
        for kind in ("generic", "draft", "target"):
            path_failures = {
                name: int(
                    delta.get(f"aclgraph_{kind}_{name}_delta", 0)
                )
                for name in (
                    "eager_fallback_calls",
                    "disabled_entry_calls",
                    "unclassified_calls",
                    "runtime_validation_failures",
                )
            }
            if any(path_failures.values()):
                issues.append(
                    f"rank {rank} {kind} qualification failed "
                    f"{path_failures}"
                )
    return not issues, issues


def _graph_qualification_can_prune(before_workers, after_workers):
    """Return whether only cold, unvalidated resident entries block sealing."""

    before_ranks = {int(worker["rank"]) for worker in before_workers}
    after_ranks = {int(worker["rank"]) for worker in after_workers}
    if before_ranks != after_ranks or not after_workers:
        return False
    deltas = _worker_aclgraph_deltas(before_workers, after_workers)
    if not any(
        int(worker.get("aclgraph_unvalidated_entries", 0)) > 0
        for worker in after_workers
    ):
        return False
    if any(
        int(worker.get("aclgraph_disabled_entries", 0)) != 0
        for worker in after_workers
    ):
        return False
    for delta in deltas:
        if any(
            int(delta.get(name, 0)) != 0
            for name in (
                "aclgraph_capture_attempts_delta",
                "aclgraph_captures_delta",
                "aclgraph_failed_captures_delta",
                "aclgraph_capacity_fallbacks_delta",
                "aclgraph_shape_fallbacks_delta",
            )
        ):
            return False
        for kind in ("generic", "draft", "target"):
            if any(
                int(
                    delta.get(
                        f"aclgraph_{kind}_{name}_delta",
                        0,
                    )
                )
                != 0
                for name in (
                    "eager_fallback_calls",
                    "disabled_entry_calls",
                    "unclassified_calls",
                    "runtime_validation_failures",
                )
            ):
                return False
    return True


def _require_no_graph_fallback(
    worker_deltas,
    *,
    require_full_window: bool = False,
    full_window_target_graph_kind: str = "auto",
) -> None:
    """Require measured calls—not only graph configuration—to be graph-only.

    ``full_window_target_graph_kind`` names the public graph-runner path used
    by target verification.  The exact-row causal chain is recorded as
    ``target`` while the diagnostic packed target path is recorded as
    ``generic``.  ``auto`` is strict rather than permissive: every target rank
    must have exactly one active target-side path, which is then audited.
    """
    supported_target_graph_kinds = ("target", "generic")
    if full_window_target_graph_kind not in (
        "auto",
        *supported_target_graph_kinds,
    ):
        raise ValueError(
            "full_window_target_graph_kind must be auto, target, or generic."
        )
    fallback_names = (
        "aclgraph_failed_captures_delta",
        "aclgraph_capacity_fallbacks_delta",
        "aclgraph_shape_fallbacks_delta",
    )
    failures = [
        (
            int(worker["rank"]),
            {name: int(worker.get(name, 0)) for name in fallback_names},
        )
        for worker in worker_deltas
        if any(int(worker.get(name, 0)) != 0 for name in fallback_names)
    ]
    measured_captures = [
        (
            int(worker["rank"]),
            {
                name: int(worker.get(name, 0))
                for name in (
                    "aclgraph_captures_delta",
                    "aclgraph_capture_attempts_delta",
                )
            },
        )
        for worker in worker_deltas
        if int(worker.get("aclgraph_captures_delta", 0)) != 0
        or int(worker.get("aclgraph_capture_attempts_delta", 0)) != 0
    ]
    measured_validations = [
        (
            int(worker["rank"]),
            int(worker.get("aclgraph_runtime_validation_replays_delta", 0)),
        )
        for worker in worker_deltas
        if int(worker.get("aclgraph_runtime_validation_replays_delta", 0)) != 0
    ]
    path_failures = []
    for worker in worker_deltas:
        rank = int(worker["rank"])
        for kind in ("generic", "draft", "target"):
            values = {
                name: int(
                    worker.get(f"aclgraph_{kind}_{name}_delta", 0)
                )
                for name in (
                    "total_calls",
                    "capture_replay_calls",
                    "replay_calls",
                    "eager_fallback_calls",
                    "disabled_entry_calls",
                    "unclassified_calls",
                    "runtime_validation_calls",
                    "runtime_validation_failures",
                    "changed_input_validation_calls",
                    "logical_row_expansion_validation_calls",
                )
            }
            classified = sum(
                values[name]
                for name in (
                    "capture_replay_calls",
                    "replay_calls",
                    "eager_fallback_calls",
                    "unclassified_calls",
                )
            )
            if (
                values["total_calls"] != classified
                or values["capture_replay_calls"] != 0
                or values["eager_fallback_calls"] != 0
                or values["disabled_entry_calls"] != 0
                or values["unclassified_calls"] != 0
                or values["runtime_validation_calls"] != 0
                or values["runtime_validation_failures"] != 0
                or values["changed_input_validation_calls"] != 0
                or values["logical_row_expansion_validation_calls"] != 0
            ):
                path_failures.append((rank, kind, values))
    inactive = [
        int(worker["rank"])
        for worker in worker_deltas
        if int(worker.get("aclgraph_replays_delta", 0)) <= 0
    ]
    full_window_failures = []
    if require_full_window:
        for worker in worker_deltas:
            rank = int(worker["rank"])
            is_draft = bool(int(worker.get("is_draft_rank", 0)))
            if is_draft:
                calls = int(
                    worker.get(
                        "spec_rhythm_linear_draft_full_chain_calls_delta", 0
                    )
                )
                draft_total = int(
                    worker.get("aclgraph_draft_total_calls_delta", 0)
                )
                draft_replays = int(
                    worker.get("aclgraph_draft_replay_calls_delta", 0)
                )
                details = {
                    "full_chain_calls": calls,
                    "draft_total_calls": draft_total,
                    "draft_replay_calls": draft_replays,
                    "full_chain_capture_replay_calls": int(
                        worker.get(
                            "spec_rhythm_linear_draft_full_chain_capture_replay_calls_delta",
                            0,
                        )
                    ),
                    "full_chain_replay_calls": int(
                        worker.get(
                            "spec_rhythm_linear_draft_full_chain_replay_calls_delta",
                            0,
                        )
                    ),
                    "full_chain_eager_fallback_calls": int(
                        worker.get(
                            "spec_rhythm_linear_draft_full_chain_eager_fallback_calls_delta",
                            0,
                        )
                    ),
                    "full_chain_unclassified_calls": int(
                        worker.get(
                            "spec_rhythm_linear_draft_full_chain_unclassified_calls_delta",
                            0,
                        )
                    ),
                    "stepwise_calls": int(
                        worker.get(
                            "spec_rhythm_linear_draft_stepwise_calls_delta", 0
                        )
                    ),
                }
                if (
                    calls <= 0
                    # A fixed-window cycle can also issue graph-only draft
                    # catch-up/rolling-eager calls.  The path-level audit
                    # above already requires every such call to be a replay;
                    # only require that all declared full-chain calls are a
                    # replay-backed subset instead of falsely equating the
                    # two counters.
                    or draft_total < calls
                    or draft_replays < calls
                    or details["full_chain_replay_calls"] != calls
                    or any(
                        details[name] != 0
                        for name in (
                            "full_chain_capture_replay_calls",
                            "full_chain_eager_fallback_calls",
                            "full_chain_unclassified_calls",
                            "stepwise_calls",
                        )
                    )
                ):
                    full_window_failures.append((rank, "draft", details))
            else:
                active_kinds = tuple(
                    kind
                    for kind in supported_target_graph_kinds
                    if int(
                        worker.get(
                            f"aclgraph_{kind}_total_calls_delta",
                            0,
                        )
                    )
                    != 0
                )
                selected_kind = (
                    active_kinds[0]
                    if full_window_target_graph_kind == "auto"
                    and len(active_kinds) == 1
                    else full_window_target_graph_kind
                )
                selected_values = {
                    name: int(
                        worker.get(
                            f"aclgraph_{selected_kind}_{name}_delta",
                            0,
                        )
                    )
                    for name in (
                        "total_calls",
                        "capture_replay_calls",
                        "replay_calls",
                        "eager_fallback_calls",
                        "disabled_entry_calls",
                        "unclassified_calls",
                        "runtime_validation_calls",
                        "runtime_validation_failures",
                        "changed_input_validation_calls",
                        "logical_row_expansion_validation_calls",
                    )
                }
                details = {
                    "selected_graph_kind": selected_kind,
                    "active_graph_kinds": active_kinds,
                    **selected_values,
                }
                if (
                    len(active_kinds) != 1
                    or selected_kind not in active_kinds
                    or selected_values["total_calls"] <= 0
                    or selected_values["replay_calls"]
                    != selected_values["total_calls"]
                    or any(
                        selected_values[name] != 0
                        for name in (
                            "capture_replay_calls",
                            "eager_fallback_calls",
                            "disabled_entry_calls",
                            "unclassified_calls",
                            "runtime_validation_calls",
                            "runtime_validation_failures",
                            "changed_input_validation_calls",
                            "logical_row_expansion_validation_calls",
                        )
                    )
                ):
                    full_window_failures.append(
                        (rank, "target", details)
                    )
    if (
        failures
        or measured_captures
        or measured_validations
        or path_failures
        or inactive
        or full_window_failures
    ):
        raise RuntimeError(
            "Graph-only benchmark invariant failed: "
            f"fallbacks={failures or 'none'}, "
            f"measured_captures={measured_captures or 'none'}, "
            f"measured_validations={measured_validations or 'none'}, "
            f"path_failures={path_failures or 'none'}, "
            f"inactive_ranks={inactive or 'none'}, "
            f"full_window_failures={full_window_failures or 'none'}"
        )


if __name__ == "__main__":
    main()
