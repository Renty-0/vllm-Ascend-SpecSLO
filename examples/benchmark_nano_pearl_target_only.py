# SPDX-License-Identifier: Apache-2.0
"""Benchmark the production vLLM-Ascend target-only baseline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=32)
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
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--max-cudagraph-capture-size",
        type=int,
        help="Override the inferred graph capture ceiling for decode shapes.",
    )
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument(
        "--enable-log-stats",
        action="store_true",
        help="Enable native vLLM runtime and speculative-decoding statistics.",
    )
    parser.add_argument(
        "--speculative-model",
        help="Optional local EAGLE/EAGLE3/draft-model checkpoint.",
    )
    parser.add_argument(
        "--speculative-method",
        choices=("eagle", "eagle3", "draft_model"),
        help="Speculative method used with --speculative-model.",
    )
    parser.add_argument("--num-speculative-tokens", type=int, default=3)
    parser.add_argument(
        "--eagle-post-eos-token-id",
        type=int,
        help="Use this fixed EAGLE candidate token after the target samples EOS.",
    )
    parser.add_argument(
        "--eagle-skip-drafter-after-steps",
        type=int,
        help="Stop EAGLE forwards after this many verification rounds.",
    )
    parser.add_argument(
        "--tree-width",
        type=int,
        help="Fixed EAGLE tree width. Must be used with --tree-depth.",
    )
    parser.add_argument(
        "--tree-depth",
        type=int,
        help="Fixed EAGLE tree depth. Must be used with --tree-width.",
    )
    parser.add_argument("--draft-tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--use-heterogeneous-vocab",
        action="store_true",
        help="Map unequal draft/target vocabularies for draft-model speculation.",
    )
    parser.add_argument(
        "--disable-padded-drafter-batch",
        action="store_true",
        help="Disable EAGLE drafter padding; off matches the Ascend E2E coverage.",
    )
    parser.add_argument(
        "--disable-async-scheduling",
        action="store_true",
        help="Disable vLLM asynchronous scheduling for correctness/performance isolation.",
    )
    parser.add_argument("--raw-prompts", action="store_true")
    parser.add_argument(
        "--respect-eos",
        action="store_true",
        help="Stop each request at EOS instead of forcing it to max_tokens.",
    )
    parser.add_argument(
        "--static-chunks",
        action="store_true",
        help="Submit the measured prompts in explicit fixed-size chunks.",
    )
    parser.add_argument(
        "--online-arrivals",
        action="store_true",
        help=(
            "Submit request-manifest rows at their arrival offsets and drive the "
            "native vLLM engine step-by-step. Emits TPOT/SLO attainment metrics."
        ),
    )
    parser.add_argument(
        "--output-json",
        help="Also write the final benchmark payload to this path.",
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--gsm8k", help="Path to a GSM8K parquet file.")
    prompt_group.add_argument(
        "--request-manifest",
        help="JSONL rows with prompt and max_tokens, shared with nano-PEARL.",
    )
    return parser


def _run_online_arrivals(
    llm,
    prompt_inputs,
    sampling_params,
    request_rows: list[dict],
    count: int,
):
    """Drive vLLM's engine with the manifest's wall-clock arrival trace."""
    if not request_rows:
        raise ValueError("--online-arrivals requires a non-empty request manifest.")
    if len(request_rows) < count:
        raise ValueError(f"Request manifest contains {len(request_rows)} rows, expected {count}.")
    from vllm.sampling_params import RequestOutputKind

    selected_rows = request_rows[:count]
    selected_prompts = prompt_inputs[:count]
    selected_params = sampling_params[:count] if isinstance(sampling_params, list) else [sampling_params] * count
    wall_start = time.time()
    mono_start = time.monotonic()
    # vLLM request arrival timestamps use ``time.time`` while engine metrics
    # use ``time.monotonic``. Keep the conversion explicit for TPOT/E2E SLO
    # accounting below.
    mono_minus_wall = mono_start - wall_start
    pending = list(enumerate(selected_rows))
    outputs = []
    service_step_seconds = 0.0
    arrival_sleep_seconds = 0.0
    while pending or llm.llm_engine.has_unfinished_requests():
        elapsed = time.monotonic() - mono_start
        while pending and float(pending[0][1].get("arrival_offset_sec", 0.0)) <= elapsed:
            index, row = pending.pop(0)
            params = copy.copy(selected_params[index])
            params.output_kind = RequestOutputKind.FINAL_ONLY
            request_id = str(row.get("request_id", f"online-{index:06d}"))
            arrival_wall = wall_start + float(row.get("arrival_offset_sec", 0.0))
            llm.llm_engine.add_request(
                request_id,
                selected_prompts[index],
                params,
                arrival_time=arrival_wall,
            )
        if llm.llm_engine.has_unfinished_requests():
            step_started = time.perf_counter()
            for output in llm.llm_engine.step():
                if output.finished:
                    outputs.append(output)
            service_step_seconds += time.perf_counter() - step_started
        elif pending:
            delay = float(pending[0][1].get("arrival_offset_sec", 0.0)) - elapsed
            sleep_seconds = min(max(delay, 0.0), 0.01)
            time.sleep(sleep_seconds)
            arrival_sleep_seconds += sleep_seconds
    input_order = {str(row.get("request_id", f"online-{index:06d}")): index for index, row in enumerate(selected_rows)}
    outputs.sort(key=lambda output: input_order[str(output.request_id)])
    return (
        outputs,
        time.monotonic() - mono_start,
        mono_minus_wall,
        {
            "wall_seconds": time.monotonic() - mono_start,
            "service_step_seconds": service_step_seconds,
            "arrival_sleep_seconds": arrival_sleep_seconds,
        },
    )


def _online_slo_summary(outputs, request_rows: list[dict], mono_minus_wall: float, elapsed_seconds=None) -> dict:
    from examples.specslo_slo_metrics import summarize_slo_rows

    rows_by_id = {str(row.get("request_id", index)): row for index, row in enumerate(request_rows)}
    request_metrics = []
    for output in outputs:
        row = rows_by_id.get(str(output.request_id), {})
        metrics = output.metrics
        token_count = len(output.outputs[0].token_ids)
        first = getattr(metrics, "first_token_ts", None)
        last = getattr(metrics, "last_token_ts", None)
        valid = (
            token_count > 0
            and first is not None
            and last is not None
            and math.isfinite(first)
            and math.isfinite(last)
            and 0 < first <= last
        )
        tpot_ms = paper_tpot_ms = e2e_ms = decode_seconds = None
        if valid:
            decode_seconds = last - first
            tpot_ms = decode_seconds * 1000.0 / max(1, token_count - 1)
            paper_tpot_ms = decode_seconds * 1000.0 / token_count
            arrival_mono = float(metrics.arrival_time) + mono_minus_wall
            e2e_ms = max(0.0, last - arrival_mono) * 1000.0
        request_metrics.append(
            {
                "request_id": str(output.request_id),
                "output_tokens": token_count,
                "slo_class": row.get("slo_class"),
                "slo_tpot_ms": row.get("slo_tpot_ms"),
                "first_token_ts": first if first is not None and math.isfinite(first) else None,
                "last_token_ts": last if last is not None and math.isfinite(last) else None,
                "valid_timing": valid,
                "decode_elapsed_seconds": decode_seconds,
                "observed_tpot_ms": tpot_ms,
                "paper_tpot_ms": paper_tpot_ms,
                "request_e2e_ms": e2e_ms,
            }
        )
    summary = summarize_slo_rows(
        request_metrics,
        tpot_field="observed_tpot_ms",
        definition="decode_after_first_token_ms / max(1, output_tokens - 1)",
        elapsed_seconds=elapsed_seconds,
    )
    summary["paper"] = summarize_slo_rows(
        request_metrics,
        tpot_field="paper_tpot_ms",
        definition="same_decode_elapsed_ms / output_tokens",
        elapsed_seconds=elapsed_seconds,
    )
    summary["request_metrics"] = request_metrics
    return summary


def _load_prompts(
    prompt: str | None,
    gsm8k: str | None,
    request_manifest: str | None,
    max_samples: int,
) -> tuple[list[str], list[int] | None]:
    if prompt is not None:
        return [prompt] * max_samples, None
    if request_manifest is not None:
        rows = [json.loads(line) for line in Path(request_manifest).read_text(encoding="utf-8").splitlines() if line]
        if len(rows) < max_samples:
            raise ValueError(f"Request manifest contains {len(rows)} rows, but {max_samples} were requested.")
        selected = rows[:max_samples]
        return (
            [str(row["prompt"]) for row in selected],
            [int(row["max_tokens"]) for row in selected],
        )
    dataset_path = Path(gsm8k)
    if dataset_path.suffix == ".jsonl":
        rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
        if len(rows) < max_samples:
            raise ValueError(f"GSM8K contains {len(rows)} rows, but {max_samples} were requested.")
        return [str(row["turns"][0]) for row in rows[:max_samples]], None
    import pyarrow.parquet as pq

    rows = pq.read_table(dataset_path, columns=["question"]).to_pylist()
    if len(rows) < max_samples:
        raise ValueError(f"GSM8K contains {len(rows)} rows, but batch size {max_samples} was requested.")
    return [str(row["question"]) for row in rows[:max_samples]], None


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise ValueError("Every target-only batch size must be positive.")
    if args.num_prompts is not None and args.num_prompts <= 0:
        raise ValueError("The target-only prompt count must be positive.")
    if args.warmup_prompts is not None and args.warmup_prompts <= 0:
        raise ValueError("--warmup-prompts must be positive.")
    if args.warmup_prompt_offset < 0:
        raise ValueError("--warmup-prompt-offset must be non-negative.")
    if args.warmup_max_tokens is not None and args.warmup_max_tokens <= 0:
        raise ValueError("--warmup-max-tokens must be positive.")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative.")
    if (args.speculative_model is None) != (args.speculative_method is None):
        raise ValueError("--speculative-model and --speculative-method must be provided together.")
    if args.num_speculative_tokens <= 0:
        raise ValueError("--num-speculative-tokens must be positive.")
    if args.eagle_skip_drafter_after_steps is not None and args.eagle_post_eos_token_id is None:
        raise ValueError("--eagle-skip-drafter-after-steps requires --eagle-post-eos-token-id.")
    if (args.tree_width is None) != (args.tree_depth is None):
        raise ValueError("--tree-width and --tree-depth must be provided together.")
    if args.tree_width is not None:
        if args.tree_width <= 0 or args.tree_depth <= 0:
            raise ValueError("--tree-width and --tree-depth must be positive.")
        if args.tree_width * args.tree_depth != args.num_speculative_tokens:
            raise ValueError("--num-speculative-tokens must equal --tree-width * --tree-depth.")
    if args.max_cudagraph_capture_size is not None and args.max_cudagraph_capture_size <= 0:
        raise ValueError("--max-cudagraph-capture-size must be positive.")
    if args.num_prompts is not None and len(args.batch_sizes) != 1 and not args.static_chunks:
        raise ValueError("--num-prompts requires exactly one --batch-sizes value.")
    if max(args.batch_sizes) > args.max_num_seqs:
        raise ValueError("The largest batch size exceeds max_num_seqs.")

    from vllm import LLM, SamplingParams, TokensPrompt

    prompt_count = args.num_prompts or max(args.batch_sizes)
    max_warmup_count = args.warmup_prompts or max(args.batch_sizes)
    load_count = max(prompt_count, args.warmup_prompt_offset + max_warmup_count) if args.warmup_runs else prompt_count
    prompts, request_max_tokens = _load_prompts(
        args.prompt,
        args.gsm8k,
        args.request_manifest,
        load_count,
    )
    request_rows = None
    if args.request_manifest is not None:
        request_rows = [
            json.loads(line)
            for line in Path(args.request_manifest).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if args.online_arrivals and request_rows is None:
        raise ValueError("--online-arrivals requires --request-manifest.")
    speculative_config = None
    if args.speculative_model is not None:
        speculative_config = {
            "model": args.speculative_model,
            "method": args.speculative_method,
            "num_speculative_tokens": args.num_speculative_tokens,
            "draft_tensor_parallel_size": args.draft_tensor_parallel_size,
            "disable_padded_drafter_batch": args.disable_padded_drafter_batch,
            "eagle_post_eos_token_id": args.eagle_post_eos_token_id,
            "eagle_skip_drafter_after_steps": (args.eagle_skip_drafter_after_steps),
        }
        if args.tree_width is not None:
            speculative_config.update(
                tree_width=args.tree_width,
                tree_depth=args.tree_depth,
            )
        if args.speculative_method == "draft_model":
            speculative_config.update(
                use_heterogeneous_vocab=args.use_heterogeneous_vocab,
                draft_sample_method="greedy",
            )
    llm_kwargs = {}
    if args.disable_async_scheduling:
        llm_kwargs["async_scheduling"] = False
    if args.max_cudagraph_capture_size is not None:
        from vllm.config import CompilationConfig

        llm_kwargs["compilation_config"] = CompilationConfig(
            max_cudagraph_capture_size=args.max_cudagraph_capture_size,
        )
    # Online SLO accounting consumes RequestMetrics (arrival/first/last token
    # timestamps).  Runtime stats are therefore mandatory for an online run,
    # even when the user did not request verbose logging.
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=args.enable_prefix_caching,
        speculative_config=speculative_config,
        disable_log_stats=not (args.enable_log_stats or args.online_arrivals),
        **llm_kwargs,
    )
    sampling_params = (
        [
            SamplingParams(
                temperature=0.0,
                max_tokens=value,
                ignore_eos=not args.respect_eos,
            )
            for value in request_max_tokens
        ]
        if request_max_tokens is not None
        else SamplingParams(
            temperature=0.0,
            max_tokens=args.max_tokens,
            ignore_eos=not args.respect_eos,
        )
    )
    warmup_sampling_params = (
        [
            SamplingParams(
                temperature=0.0,
                max_tokens=min(value, args.warmup_max_tokens or value),
                ignore_eos=not args.respect_eos,
            )
            for value in request_max_tokens
        ]
        if request_max_tokens is not None
        else SamplingParams(
            temperature=0.0,
            max_tokens=args.warmup_max_tokens or args.max_tokens,
            ignore_eos=not args.respect_eos,
        )
    )
    if args.raw_prompts:
        prompt_inputs = prompts
        prompt_token_ids = [list(llm.get_tokenizer().encode(prompt)) for prompt in prompts]
        first_prompt_token_ids = prompt_token_ids[0]
    else:
        tokenizer = llm.get_tokenizer()
        prompt_token_ids = [
            list(
                tokenizer.encode(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            )
            for prompt in prompts
        ]
        prompt_inputs = [TokensPrompt(prompt_token_ids=token_ids) for token_ids in prompt_token_ids]
        first_prompt_token_ids = prompt_token_ids[0]
    eos_token_id = llm.get_tokenizer().eos_token_id
    results = []
    for batch_size in args.batch_sizes:
        measured_inputs = prompt_inputs[: args.num_prompts or batch_size]
        measured_sampling_params = (
            sampling_params[: len(measured_inputs)] if isinstance(sampling_params, list) else sampling_params
        )
        warmup_count = args.warmup_prompts or batch_size
        warmup_start = args.warmup_prompt_offset
        warmup_inputs = prompt_inputs[warmup_start : warmup_start + warmup_count]
        current_warmup_params = (
            warmup_sampling_params[warmup_start : warmup_start + warmup_count]
            if isinstance(warmup_sampling_params, list)
            else warmup_sampling_params
        )
        for _ in range(args.warmup_runs):
            llm.generate(warmup_inputs, current_warmup_params, use_tqdm=False)
        if args.online_arrivals:
            assert request_rows is not None
            outputs, elapsed, mono_minus_wall, online_timing = _run_online_arrivals(
                llm,
                prompt_inputs,
                measured_sampling_params,
                request_rows,
                len(measured_inputs),
            )
            online_slo = _online_slo_summary(outputs, request_rows, mono_minus_wall, elapsed)
        else:
            started = time.perf_counter()
            if args.static_chunks:
                outputs = []
                for start in range(0, len(measured_inputs), batch_size):
                    outputs.extend(
                        llm.generate(
                            measured_inputs[start : start + batch_size],
                            (
                                measured_sampling_params[start : start + batch_size]
                                if isinstance(measured_sampling_params, list)
                                else measured_sampling_params
                            ),
                            use_tqdm=False,
                        )
                    )
            else:
                outputs = llm.generate(measured_inputs, measured_sampling_params, use_tqdm=False)
            elapsed = time.perf_counter() - started
            online_slo = None
        output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        output_token_rows = [list(output.outputs[0].token_ids) for output in outputs]
        first_eos_positions = [row.index(eos_token_id) if eos_token_id in row else None for row in output_token_rows]
        results.append(
            {
                "batch_size": batch_size,
                "num_prompts": len(measured_inputs),
                "num_static_chunks": (math.ceil(len(measured_inputs) / batch_size) if args.static_chunks else 1),
                "output_tokens": output_tokens,
                "prompt_token_ids_sha256": hashlib.sha256(
                    json.dumps(prompt_token_ids[: len(measured_inputs)], separators=(",", ":")).encode()
                ).hexdigest(),
                "elapsed_seconds": elapsed,
                "output_throughput_tokens_per_second": output_tokens / elapsed,
                "output_token_ids_sha256": hashlib.sha256(
                    json.dumps(output_token_rows, separators=(",", ":")).encode()
                ).hexdigest(),
                # Keep the complete rows for deterministic/speculative
                # correctness comparisons.  The hash remains the compact
                # regression key used by existing reports.
                "output_token_ids": output_token_rows,
                "first_eos_positions": first_eos_positions,
                "first_output_token_ids": list(outputs[0].outputs[0].token_ids),
                "online_arrivals": args.online_arrivals,
                "online_timing": online_timing if args.online_arrivals else None,
                "slo": online_slo,
            }
        )
    payload = {
        "backend": ("vllm-ascend-target-only" if speculative_config is None else "vllm-ascend-single-npu-speculative"),
        "model": args.model,
        "speculative_model": args.speculative_model,
        "speculative_method": args.speculative_method,
        "num_speculative_tokens": (args.num_speculative_tokens if speculative_config is not None else None),
        "tree_width": args.tree_width if speculative_config is not None else None,
        "tree_depth": args.tree_depth if speculative_config is not None else None,
        "draft_tensor_parallel_size": (args.draft_tensor_parallel_size if speculative_config is not None else None),
        "disable_padded_drafter_batch": (args.disable_padded_drafter_batch if speculative_config is not None else None),
        "eagle_post_eos_token_id": (args.eagle_post_eos_token_id if speculative_config is not None else None),
        "eagle_skip_drafter_after_steps": (
            args.eagle_skip_drafter_after_steps if speculative_config is not None else None
        ),
        "use_heterogeneous_vocab": (args.use_heterogeneous_vocab if speculative_config is not None else None),
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "enforce_eager": args.enforce_eager,
        "max_cudagraph_capture_size": args.max_cudagraph_capture_size,
        "max_tokens": args.max_tokens,
        "respect_eos": args.respect_eos,
        "request_manifest": args.request_manifest,
        "requested_output_tokens": (sum(request_max_tokens[:prompt_count]) if request_max_tokens is not None else None),
        "warmup_prompts": args.warmup_prompts,
        "warmup_prompt_offset": args.warmup_prompt_offset,
        "warmup_max_tokens": args.warmup_max_tokens,
        "warmup_runs": args.warmup_runs,
        "static_chunks": args.static_chunks,
        "online_arrivals": args.online_arrivals,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_log_stats": args.enable_log_stats,
        "first_prompt_token_ids": first_prompt_token_ids,
        "results": results,
    }
    output = json.dumps(payload, ensure_ascii=True)
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"{output}\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
