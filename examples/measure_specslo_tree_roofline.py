# SPDX-License-Identifier: Apache-2.0
"""Collect real TP3 target-forward samples for SpecSLO's offline roofline.

Run under torchrun with four assigned devices (draft TP1 + target TP3)::

    torchrun --standalone --nproc_per_node=4 examples/measure_specslo_tree_roofline.py \
      --mode graph --batch-size 8 --verification-requests 4 --contexts 128 512 \
      --candidate-budgets 4 8 16 --output /path/target_samples.json

Then aggregate with examples/profile_specslo_tree_roofline.py. This is a
synthetic, fixed-context kernel-capacity experiment, NOT dataset accuracy,
end-to-end serving throughput, or a Goodput comparison against baseline TP4.
The standard AR control uses all active-batch requests with one query each.
The packed tree uses the physical request rows in the currently verified
logical slot. This is the paper section 5.3 comparison against standard
batched decoding at the same active batch size, not a half-sized AR control.
Only explicitly sampled shapes/contexts are evidence.  The default sweep
therefore measures every canonical candidate-count distribution at each total
budget, rather than treating one balanced allocation as universal evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-model", default="/data/shared-models/Qwen3-0.6B")
    parser.add_argument("--target-model", default="/data/shared-models/Qwen3-32B")
    parser.add_argument("--mode", required=True, choices=("eager", "graph"))
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Total active dual-batch requests, not physical target rows."
    )
    parser.add_argument("--verification-requests", type=int, default=4)
    parser.add_argument(
        "--batch-shape",
        action="append",
        default=[],
        metavar="ACTIVE:VERIFY",
        help=(
            "Measure another active-batch/physical-verification-row pair in the same model lifecycle. "
            "When supplied, the single --batch-size/--verification-requests pair is ignored."
        ),
    )
    parser.add_argument("--contexts", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--candidate-budgets", type=int, nargs="+")
    parser.add_argument(
        "--candidate-counts",
        action="append",
        default=[],
        help="Explicit comma-separated positive node counts per physical target request; repeatable.",
    )
    parser.add_argument("--tree-width", type=int, default=2)
    parser.add_argument("--tree-depth", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument(
        "--prefill-token-chunk-size",
        type=int,
        default=256,
        help="Maximum prefix tokens per request in each untimed KV prefill forward.",
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--epsilon-relative", type=float, default=0.10)
    parser.add_argument("--latency-percentile", type=float, default=95.0)
    parser.add_argument(
        "--stop-after-first-violation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Stop an exact increasing budget sweep after its first measured AR-envelope violation. "
            "This matches the conservative aggregator and avoids capturing irrelevant larger graphs."
        ),
    )
    parser.add_argument("--max-aclgraph-entries", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="Write an unmeasured shape manifest without loading models."
    )
    return parser


def _batch_shapes(args) -> list[tuple[int, int]]:
    if not args.batch_shape:
        pairs = [(args.batch_size, args.verification_requests)]
    else:
        pairs = []
        for raw in args.batch_shape:
            try:
                active, verify = (int(value.strip()) for value in raw.split(":", 1))
            except (TypeError, ValueError) as error:
                raise ValueError("Every --batch-shape must use ACTIVE:VERIFY positive integers") from error
            pairs.append((active, verify))
    if any(active <= 0 or verify <= 0 or verify > active for active, verify in pairs):
        raise ValueError("Batch shapes require 0 < verification requests <= active batch")
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate --batch-shape entries are not allowed")
    return pairs


def _shape_counts(args, requests: int | None = None) -> list[list[int]]:
    single_shape = requests is None
    requests = args.verification_requests if single_shape else requests
    if single_shape and requests > args.batch_size:
        raise ValueError("verification_requests must not exceed active batch_size")
    maximum = args.tree_width * args.tree_depth
    positive = [
        requests,
        args.tree_width,
        args.tree_depth,
        args.max_model_len,
        args.prefill_token_chunk_size,
        args.iterations,
        args.max_aclgraph_entries,
        *args.contexts,
    ]
    if any(value <= 0 for value in positive):
        raise ValueError("Counts must be positive")
    if args.warmup_iterations < 0 or (args.mode == "graph" and args.warmup_iterations < 2):
        raise ValueError("Graph profiling requires at least two untimed warmup iterations; eager permits zero")
    if not math.isfinite(args.epsilon_relative) or args.epsilon_relative < 0:
        raise ValueError("epsilon_relative must be finite and non-negative")
    if not math.isfinite(args.latency_percentile) or not 0 <= args.latency_percentile <= 100:
        raise ValueError("latency_percentile must be in [0, 100]")
    if max(args.contexts) + maximum > args.max_model_len:
        raise ValueError("Context and tree physical KV slots exceed max_model_len")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    budgets = args.candidate_budgets
    if budgets is None and not args.candidate_counts:
        # Section 5.3 asks for the largest total candidate-token count inside
        # the AR latency envelope. Scan every reachable total instead of only
        # powers of two so the reported roof is measured, not interpolated.
        budgets = list(range(requests, requests * maximum + 1))
    shapes = []
    for budget in budgets or ():
        if not requests <= budget <= requests * maximum:
            raise ValueError("Every candidate budget must give each physical target request 1..width*depth nodes")
        shapes.extend(_canonical_count_distributions(requests, maximum, budget))
    for raw in args.candidate_counts:
        counts = [int(value.strip()) for value in raw.split(",")]
        if len(counts) != requests or any(not 1 <= value <= maximum for value in counts):
            raise ValueError("Explicit candidate counts must match physical target requests and fit the tree")
        shapes.append(counts)
    unique = sorted(set(map(tuple, shapes)), key=lambda counts: (sum(counts), counts))
    return [[0] * requests, *[list(counts) for counts in unique]]


def _canonical_count_distributions(requests: int, maximum: int, budget: int) -> list[list[int]]:
    """Enumerate permutation-equivalent per-request counts exactly once.

    Requests have identical fixed contexts in the section 5.3 kernel probe,
    so permutations of one count histogram have the same operator shape.
    Non-decreasing canonical rows cover every histogram without an exponential
    permutation sweep.  In particular, budget ``requests + 2`` includes both
    the linear ``[..., 2, 2]`` shape and the branching ``[..., 1, 3]`` shape.
    """
    if requests <= 0 or maximum <= 0 or not requests <= budget <= requests * maximum:
        raise ValueError("Canonical candidate distributions require a reachable positive budget")
    result: list[list[int]] = []

    def visit(prefix: list[int], minimum: int, slots: int, remaining: int) -> None:
        if slots == 0:
            if remaining == 0:
                result.append(prefix.copy())
            return
        lower = max(minimum, remaining - maximum * (slots - 1))
        upper = min(maximum, remaining // slots)
        for value in range(lower, upper + 1):
            prefix.append(value)
            visit(prefix, value, slots - 1, remaining - value)
            prefix.pop()

    visit([], 1, requests, budget)
    return result


def _measurement_shapes(active_batch: int, tree_shapes: Sequence[Sequence[int]]) -> list[list[int]]:
    """Pair full-active standard AR with positive one-slot tree shapes."""
    if active_batch <= 0:
        raise ValueError("Roofline active batch must be positive")
    return [[0] * active_batch, *[list(counts) for counts in tree_shapes if any(counts)]]


def _metadata(args, hardware: str, *, source_fingerprints: dict[str, str] | None = None) -> dict:
    from vllm_ascend.spec_decode.pearl.native_graph import TREE_FIA_INNER_PRECISE, TREE_FIA_SPARSE_MODE

    if not hardware:
        raise ValueError("Measured metadata requires a detected hardware name")
    result = {
        "model": args.target_model,
        "target_tensor_parallel_size": 3,
        "execution_mode": args.mode,
        "hardware": hardware,
        "measurement_scope": "target_forward",
        "verification_layout": "packed_tree",
        # FULL-mask capacity and tree topology change the physical operator
        # contract even when active batch/context lookup keys are identical.
        "max_model_len": args.max_model_len,
        "tree_width": args.tree_width,
        "tree_depth": args.tree_depth,
        "collector": "measure_specslo_tree_roofline.py",
        "output_head": "greedy",
        "finiteness_guard": "spec_rhythm_tree_full_model",
        "draft_tensor_parallel_size": 1,
        "latency_rank_aggregation": "per_iteration_max_over_target_tp_ranks",
        "timing_excludes": ["input_preparation", "prefix_prefill", "warmup", "graph_capture", "timing_all_reduce"],
        "timing_includes": ["model_forward", "greedy_output_head", "graph_replay_management", "device_completion"],
        "workload": "synthetic_fixed_prefix",
        "ar_comparator": "standard_decode_full_active_batch",
        "candidate_distribution_protocol": "all_canonical_histograms_until_first_violation",
        # CANN graph-task handles are production-lifetime objects. Repeatedly
        # resetting one custom-attention graph and immediately capturing a
        # different graph can strand the TP3 stream. Each collector process
        # therefore owns exactly one active-batch shape, one maximum prefix
        # cache, and a small resident graph set reused across all contexts.
        "collector_graph_lifecycle": "resident_per_active_batch_max_context_cache",
        # Dynamic admission/tails can reach an unmeasured key or physical
        # home size. Serving must use ordinary AR for that cycle, never infer
        # a speculative budget from gamma.
        "unprofiled_execution_policy": "target_only",
        "tree_fia_sparse_mode": TREE_FIA_SPARSE_MODE,
        "tree_fia_inner_precise": TREE_FIA_INNER_PRECISE,
    }
    if source_fingerprints is not None:
        result.update(source_fingerprints)
    # Backend fields are filled from actual prepared metadata, not guessed
    # from the requested mode or from which source checkout was imported.
    return result


def _engine_config(args):
    from vllm_ascend.spec_decode.pearl.native_engine import NativePearlConfig

    return NativePearlConfig(
        args.draft_model,
        args.target_model,
        1,
        3,
        args.tree_width * args.tree_depth,
        args.max_model_len,
        1,
        max_num_seqs=max(active for active, _ in _batch_shapes(args)),
        enforce_eager=args.mode == "eager",
        enable_prefix_caching=False,
        enable_spec_rhythm=True,
        enable_continuous_batching=True,
        enable_preemptive_scheduling=True,
        spec_rhythm_tree_width=args.tree_width,
        spec_rhythm_tree_depth=args.tree_depth,
        max_aclgraph_entries=args.max_aclgraph_entries,
        max_num_batched_tokens=max(
            16384,
            max(requests for _, requests in _batch_shapes(args)) * args.max_model_len,
        ),
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=0,
    )


def _record_backend(metadata: dict, row: dict) -> None:
    backend = row["attention_backend"]
    if row["kind"] == "ar":
        if "ar_attention_backend" in metadata and metadata["ar_attention_backend"] != backend:
            raise ValueError("One roofline profile cannot mix different AR attention backends")
        metadata["ar_attention_backend"] = backend
        return
    values = set(metadata.get("verification_attention_backends", ()))
    values.add(backend)
    metadata["verification_attention_backends"] = sorted(values)


def _reduce_sample_latencies(samples_ms, *, device, group):
    """Elementwise slowest target rank, outside the timed forward sections."""
    import torch
    import torch.distributed as dist

    # HCCL rejects float64 all-reduce. Integer microseconds retain precision
    # and one vector reduction still computes the maximum of each iteration.
    samples_us = torch.tensor([round(value * 1000.0) for value in samples_ms], dtype=torch.int64, device=device)
    dist.all_reduce(samples_us, op=dist.ReduceOp.MAX, group=group)
    return [value / 1000.0 for value in samples_us.cpu().tolist()]


def _prepare_shape(engine, args, context: int, counts: Sequence[int], root_token: int):
    """Preallocate all user inputs/metadata outside the timing callback."""
    import torch

    from vllm_ascend.spec_decode.pearl.roofline import attention_backend_identity
    from vllm_ascend.spec_decode.pearl.tree import build_tree_speculation_plan, pack_selected_tree_plan

    sequence_ids = list(range(len(counts)))
    root_position = context - 1
    if not any(counts):
        positions, metadata = engine._prepare_attention_metadata(sequence_ids, [root_position] * len(counts), False)
        input_ids = torch.full((len(counts),), root_token, dtype=torch.long, device=engine.device)
    else:
        base = build_tree_speculation_plan(
            args.tree_width, args.tree_depth, root_position, args.max_model_len, device=engine.device
        )
        plans = [pack_selected_tree_plan(base, range(count)) for count in counts]
        slots = [int(slot) for plan in plans for slot in plan.cache_positions.cpu().tolist()]
        slot_sequences = [index for index, plan in enumerate(plans) for _ in range(plan.cache_positions.numel())]
        engine._ensure_cache_capacity(slot_sequences, slots)
        candidates = [[(root_token + node + 1) % engine.target_vocab_size for node in range(count)] for count in counts]
        input_ids, positions, metadata = engine.model.make_tree_attention_metadata(
            plans,
            [root_token] * len(counts),
            candidates,
            engine.cache_block_tables,
            sequence_ids=sequence_ids,
        )
    physical_queries = int(input_ids.numel())
    if physical_queries != len(counts) + sum(counts):
        raise RuntimeError("Prepared physical query shape does not match root + active candidate accounting")
    inputs, position_rows, metadata_rows = [input_ids], [positions], [metadata]

    def forward():
        if args.mode == "graph":
            return engine.graph_runner.run_target_greedy(inputs, position_rows, metadata_rows, engine.target_vocab_size)
        hidden = engine.model(input_ids, positions, metadata)
        return engine.model.compute_greedy_tokens(hidden, engine.target_vocab_size)

    return forward, physical_queries, attention_backend_identity(metadata)


def _measure_shape(engine, args, active_batch, context, counts, root_token):
    import torch
    import torch.distributed as dist

    from examples.profile_specslo_tree_roofline import measure_target_forward

    forward, physical_queries, attention_backend = _prepare_shape(engine, args, context, counts, root_token)
    graph = engine.graph_runner
    graph_counters = (
        {"graph_capture_count": lambda: graph.capture_count, "graph_replay_count": lambda: graph.replay_count}
        if args.mode == "graph"
        else {}
    )
    local_error = None
    try:
        measured = measure_target_forward(
            forward,
            torch.npu.synchronize,
            iterations=args.iterations,
            warmup_iterations=args.warmup_iterations,
            **graph_counters,
        )
    except ValueError as error:
        measured, local_error = None, error
    invalid = torch.tensor([int(local_error is not None)], dtype=torch.int64, device=engine.device)
    dist.all_reduce(invalid, op=dist.ReduceOp.MAX, group=engine.groups.target_group)
    if int(invalid.cpu().item()):
        raise ValueError(
            str(local_error) if local_error else "A target rank failed steady-state graph/timing validation"
        )
    measured["latency_ms"] = _reduce_sample_latencies(
        measured["latency_ms"], device=engine.device, group=engine.groups.target_group
    )
    if args.mode == "graph":
        replays = torch.tensor([measured["timed_graph_replays"]], dtype=torch.int64, device=engine.device)
        dist.all_reduce(replays, op=dist.ReduceOp.MIN, group=engine.groups.target_group)
        measured["timed_graph_replays"] = int(replays.cpu().item())
    return {
        "kind": "packed_tree" if any(counts) else "ar",
        "batch_size": active_batch,
        "verification_requests": len(counts),
        "context_len": context,
        "candidate_counts": list(counts),
        "physical_query_tokens": physical_queries,
        "attention_backend": attention_backend,
        **measured,
    }


def _prefill_prefix(engine, prompts: Sequence[Sequence[int]], token_chunk_size: int) -> None:
    """Populate committed prefix KV in bounded, untimed incremental forwards."""
    if not prompts or token_chunk_size <= 0:
        raise ValueError("Roofline prefix prefill requires prompts and a positive token chunk size")
    lengths = {len(prompt) for prompt in prompts}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
        raise ValueError("Roofline prefix prompts must have one equal positive context length")
    prefix_length = next(iter(lengths)) - 1
    for position_start in range(0, prefix_length, token_chunk_size):
        position_end = min(position_start + token_chunk_size, prefix_length)
        engine._run_packed_hidden(
            [token for row in prompts for token in row[position_start:position_end]],
            [index for index in range(len(prompts)) for _ in range(position_start, position_end)],
            list(range(position_start, position_end)) * len(prompts),
            use_aclgraph=False,
            use_fused_infer_attention=True,
        )


def _write(path, document):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def main(argv=None):
    args = _build_parser().parse_args(argv)
    batch_shapes = _batch_shapes(args)
    shapes_by_requests = {requests: _shape_counts(args, requests) for _, requests in batch_shapes}
    if args.dry_run:
        _write(
            args.output,
            {
                "status": "unmeasured_dry_run",
                "config": vars(args),
                "batch_shapes": [
                    {
                        "batch_size": active,
                        "verification_requests": requests,
                        "candidate_shapes": shapes_by_requests[requests],
                        "physical_query_tokens": [
                            len(counts) + sum(counts) for counts in shapes_by_requests[requests]
                        ],
                    }
                    for active, requests in batch_shapes
                ],
            },
        )
        return 0
    if args.mode == "graph" and len(batch_shapes) != 1:
        raise ValueError(
            "Graph roofline collection requires exactly one --batch-shape per torchrun; "
            "run active batches in fresh processes and merge their complete JSON documents"
        )

    import torch
    import torch.distributed as dist

    from vllm_ascend.spec_decode.pearl.native_engine import NativePearlEngine
    from vllm_ascend.spec_decode.pearl.roofline import runtime_source_fingerprints

    engine = NativePearlEngine(_engine_config(args))
    if not engine.model.track_cache_finiteness:
        raise RuntimeError("Roofline must include the same full-model finite-value guard as tree serving")
    leader = engine.rank == engine.topology.target_leader_rank
    detected = torch.npu.get_device_name(engine.local_rank)
    # Compare target hardware identities collectively without HCCL FP64 or
    # object collectives; the draft card may be a different device variant.
    identity = int.from_bytes(hashlib.sha256(detected.encode()).digest()[:8], "big") & ((1 << 63) - 1)
    device_ids = torch.zeros(4, dtype=torch.int64, device=engine.device)
    device_ids[engine.rank] = identity
    dist.all_reduce(device_ids, op=dist.ReduceOp.MAX)
    if len(set(device_ids.cpu().tolist()[1:])) != 1:
        raise ValueError("Target TP hardware is heterogeneous; a single hardware-bound roofline would be invalid")
    report = {
        "metadata": _metadata(args, detected, source_fingerprints=runtime_source_fingerprints()),
        "config": vars(args),
        "measurements": [],
        "status": "running",
    }
    try:
        filler = engine.tokenizer.encode(
            " A deterministic context for offline target capacity measurement.", add_special_tokens=False
        )
        if not filler:
            raise ValueError("Tokenizer returned an empty synthetic prefix")
        for active_batch, verification_requests in batch_shapes:
            maximum_context = max(args.contexts)
            maximum_base = (
                filler * ((maximum_context + len(filler) - 1) // len(filler))
            )[:maximum_context]
            prompts = [list(maximum_base) for _ in range(active_batch)]
            engine._allocate_cache(prompts, enable_prefix_caching=False)
            if not engine.is_draft and maximum_context > 1:
                # Allocate and fill one maximum prefix. Contexts are measured
                # from largest to smallest, so query/candidate writes at a
                # prior larger context can never alter the visible prefix of
                # a later smaller context. Graphs and their KV tensors remain
                # resident for the entire process, matching serving lifetime.
                with torch.inference_mode():
                    _prefill_prefix(engine, prompts, args.prefill_token_chunk_size)
            for context in sorted(args.contexts, reverse=True):
                base = maximum_base[:context]
                ar_latency_ms = None
                measured_shapes = _measurement_shapes(active_batch, shapes_by_requests[verification_requests])
                for counts in measured_shapes:
                    error = None
                    row = None
                    if not engine.is_draft:
                        try:
                            with torch.inference_mode():
                                row = _measure_shape(
                                    engine,
                                    args,
                                    active_batch,
                                    context,
                                    counts,
                                    base[-1],
                                )
                            if leader:
                                _record_backend(report["metadata"], row)
                                report["measurements"].append(row)
                                _write(args.output, report)
                                print(
                                    json.dumps(
                                        {
                                            "event": "shape_measured",
                                            "batch_size": active_batch,
                                            "verification_requests": verification_requests,
                                            "context": context,
                                            "counts": counts,
                                            "latency_ms": row["latency_ms"],
                                        }
                                    ),
                                    flush=True,
                                )
                        except (ValueError, RuntimeError) as failure:
                            error = failure
                    failed = torch.tensor([int(error is not None)], dtype=torch.int64, device=engine.device)
                    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
                    failed_anywhere = bool(int(failed.cpu().item()))
                    # Keep every graph resident. Reset/re-capture churn is not
                    # a production execution pattern and is unsafe for CANN
                    # custom FIA graph-task handles on TP3.
                    dist.barrier()
                    if failed_anywhere:
                        raise RuntimeError(str(error) if error else "Another rank failed target profiling")
                    breached = False
                    if row is not None:
                        latency = _percentile(row["latency_ms"], args.latency_percentile)
                        if not any(counts):
                            ar_latency_ms = latency
                        else:
                            if ar_latency_ms is None:
                                raise RuntimeError("Packed-tree profiling ran before its matched AR control")
                            breached = latency > ar_latency_ms * (1.0 + args.epsilon_relative)
                    stop = torch.tensor(
                        [int(args.stop_after_first_violation and breached)],
                        dtype=torch.int64,
                        device=engine.device,
                    )
                    dist.all_reduce(stop, op=dist.ReduceOp.MAX)
                    if int(stop.cpu().item()):
                        break
            engine._release_cache()
        if leader:
            report["status"] = "complete"
            _write(args.output, report)
    except Exception as error:
        if leader:
            report.update(status="failed", error=str(error))
            _write(args.output, report)
        raise
    finally:
        if engine.cache_allocation is not None:
            engine._release_cache()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
