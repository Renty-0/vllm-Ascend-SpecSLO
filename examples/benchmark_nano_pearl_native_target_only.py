# SPDX-License-Identifier: Apache-2.0
"""Benchmark the native nano-PEARL target-only path with one engine load."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path

from examples.specslo_benchmark_report import capture_runtime_environment


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-tp-size", type=int, default=1)
    parser.add_argument("--target-tp-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=1024)
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
        help="Warm up on only this many prompts. Defaults to the measured workload.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--num-kvcache-blocks", type=int, default=-1)
    parser.add_argument("--prefill-chunk-size", type=int)
    parser.add_argument("--worker-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--target-use-paged-attention",
        action="store_true",
        help="Use the native paged-attention cache for the target model.",
    )
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument(
        "--seal-graph-cache-after-warmup",
        action="store_true",
        help=(
            "Seal the changed-input-qualified ACLGraph inventory after warmup. "
            "This requires one batch-size point and makes any measured-time "
            "graph discovery fail closed."
        ),
    )
    parser.add_argument("--enable-mc2", action="store_true")
    parser.add_argument("--mc2-profile")
    parser.add_argument("--output-json")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--gsm8k", help="Path to a GSM8K parquet file.")
    return parser


def _load_prompts(prompt: str | None, gsm8k: str | None, max_samples: int) -> list[str]:
    if prompt is not None:
        return [prompt] * max_samples
    dataset_path = Path(gsm8k)
    if dataset_path.suffix == ".jsonl":
        rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
        if len(rows) < max_samples:
            raise ValueError(f"GSM8K contains {len(rows)} rows, but {max_samples} were requested.")
        return [str(row["turns"][0]) for row in rows[:max_samples]]
    import pyarrow.parquet as pq

    rows = pq.read_table(dataset_path, columns=["question"]).to_pylist()
    if len(rows) < max_samples:
        raise ValueError(f"GSM8K contains {len(rows)} rows, but batch size {max_samples} was requested.")
    return [str(row["question"]) for row in rows[:max_samples]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mc2_profile_provenance(path: str | None) -> dict[str, object | None]:
    if path is None:
        return {
            "mc2_profile_resolved": None,
            "mc2_profile_sha256": None,
            "mc2_profile_rms_norm_epsilon": None,
        }
    resolved = Path(path).expanduser().resolve(strict=True)
    document = json.loads(resolved.read_text(encoding="utf-8"))
    metadata = document.get("metadata", {})
    epsilon = metadata.get("rms_norm_epsilon") if isinstance(metadata, dict) else None
    return {
        "mc2_profile_resolved": str(resolved),
        "mc2_profile_sha256": _sha256_file(resolved),
        "mc2_profile_rms_norm_epsilon": epsilon,
    }


def _target_rms_norm_epsilon(model_path: str) -> float | None:
    config_path = Path(model_path).expanduser() / "config.json"
    if not config_path.is_file():
        return None
    document = json.loads(config_path.read_text(encoding="utf-8"))
    value = document.get("rms_norm_eps")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise ValueError("Every target-only batch size must be positive.")
    if args.num_prompts is not None and args.num_prompts <= 0:
        raise ValueError("The target-only prompt count must be positive.")
    if args.warmup_prompts is not None and args.warmup_prompts <= 0:
        raise ValueError("--warmup-prompts must be positive.")
    if args.num_prompts is not None and len(args.batch_sizes) != 1:
        raise ValueError("--num-prompts requires exactly one --batch-sizes value.")
    if args.seal_graph_cache_after_warmup and len(args.batch_sizes) != 1:
        raise ValueError("--seal-graph-cache-after-warmup requires exactly one batch size.")
    max_batch_size = max(args.batch_sizes)
    profile_provenance = _mc2_profile_provenance(args.mc2_profile)
    target_rms_norm_epsilon = _target_rms_norm_epsilon(args.target_model)

    from vllm_ascend.spec_decode.pearl import PEARLConfig, PEARLEngine, SamplingParams

    prompt_count = args.num_prompts or max_batch_size
    prompts = _load_prompts(args.prompt, args.gsm8k, prompt_count)
    config = PEARLConfig(
        draft_model_path=args.draft_model,
        target_model_path=args.target_model,
        draft_tensor_parallel_size=args.draft_tp_size,
        target_tensor_parallel_size=args.target_tp_size,
        max_num_batched_tokens=args.max_model_len * max_batch_size,
        max_num_seqs=max_batch_size,
        prefill_chunk_size=args.prefill_chunk_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_kvcache_blocks=args.num_kvcache_blocks,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_mc2=args.enable_mc2,
        mc2_profile=args.mc2_profile,
        enforce_eager=args.enforce_eager,
        target_use_paged_attention=args.target_use_paged_attention,
        seed=args.seed,
        gamma=4,
        worker_timeout_seconds=args.worker_timeout_seconds,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)

    results = []
    with PEARLEngine(config) as engine:
        first_formatted_prompt = engine.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompts[0]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        first_prompt_token_ids = list(engine.tokenizer.encode(first_formatted_prompt))
        for batch_size in args.batch_sizes:
            measured_prompts = prompts[: args.num_prompts or batch_size]
            warmup_prompts = measured_prompts[: args.warmup_prompts]
            for prompt in warmup_prompts:
                engine.add_request(prompt, sampling_params)
            engine.AR_generate()
            worker_metrics_before_measurement = (
                engine.seal_graph_cache()
                if args.seal_graph_cache_after_warmup
                else engine.last_worker_metrics
            )

            for prompt in measured_prompts:
                engine.add_request(prompt, sampling_params)
            started = time.perf_counter()
            _, num_tokens, _, inference_elapsed = engine.AR_generate()
            e2e_elapsed = time.perf_counter() - started
            output_tokens = sum(num_tokens)
            chunk_metrics = engine.last_metrics[::batch_size]
            output_token_rows = [metric["completion_token_ids"] for metric in engine.last_metrics]
            results.append(
                {
                    "batch_size": batch_size,
                    "num_prompts": len(measured_prompts),
                    "num_static_chunks": math.ceil(len(measured_prompts) / batch_size),
                    "output_tokens": output_tokens,
                    "inference_elapsed_seconds": inference_elapsed,
                    "e2e_elapsed_seconds": e2e_elapsed,
                    "inference_throughput_tokens_per_second": output_tokens / inference_elapsed,
                    "e2e_throughput_tokens_per_second": output_tokens / e2e_elapsed,
                    "prefill_elapsed_seconds": sum(
                        metric["prefill_elapsed_seconds"] for metric in chunk_metrics
                    ),
                    "decode_elapsed_seconds": sum(
                        metric["decode_elapsed_seconds"] for metric in chunk_metrics
                    ),
                    "output_token_ids_sha256": hashlib.sha256(
                        json.dumps(output_token_rows, separators=(",", ":")).encode()
                    ).hexdigest(),
                    # Keep the complete rows in the native regression payload
                    # so speculative runs can be checked at the first
                    # divergent token, not only by an aggregate hash.
                    "output_token_ids": output_token_rows,
                    "first_output_token_ids": engine.last_metrics[0]["completion_token_ids"],
                    # Preserve per-rank production counters around the timed
                    # window.  MC2 is dispatched while a graph is captured;
                    # later graph replays intentionally do not re-enter the
                    # Python adapter, so both the absolute warm state and the
                    # measured replay delta are needed to prove that the
                    # resident model-runner graph contains the fused path.
                    "worker_metrics_before_measurement": worker_metrics_before_measurement,
                    "worker_metrics_after_measurement": engine.last_worker_metrics,
                }
            )

    payload = {
        "backend": "nano-pearl-native-target-only",
        "draft_model": args.draft_model,
        "target_model": args.target_model,
        "draft_tensor_parallel_size": args.draft_tp_size,
        "target_tensor_parallel_size": args.target_tp_size,
        "prefill_chunk_size": args.prefill_chunk_size,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_mc2": args.enable_mc2,
        "mc2_profile": args.mc2_profile,
        **profile_provenance,
        "target_rms_norm_epsilon": target_rms_norm_epsilon,
        "enforce_eager": args.enforce_eager,
        "execution_mode": "eager" if args.enforce_eager else "aclgraph",
        "seal_graph_cache_after_warmup": args.seal_graph_cache_after_warmup,
        "max_tokens": args.max_tokens,
        "warmup_prompts": args.warmup_prompts,
        "runtime_environment": capture_runtime_environment(),
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
