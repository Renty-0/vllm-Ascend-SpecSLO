# SPDX-License-Identifier: Apache-2.0
"""Strict measured-roofline online acceptance on four NPUs, not a benchmark.

torchrun --standalone --nproc_per_node=4 examples/check_specslo_profile_online.py \
  --profile /path/qualified-multiple-batch-keys.json --output /path/online.json

One engine serves four synthetic requests at capacity two, with staggered
arrivals and different output limits. The profile, not gamma or a CLI scalar,
supplies every verification roof. Missing/unqualified tail keys are rejected
before model loading; the script never fills, interpolates or rewrites them.
Context lookup retains the runtime's declared 512-token bucket convention;
the report distinguishes exact sampled contexts from bucket-only coverage.
Every actual target forward is tied to its real validate_execution call and
records candidate shape, roots, backend, graph execution and evidence rows.
Observers do not alter scheduling, tokens, cache, budgets or graph decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

from vllm_ascend.spec_decode.pearl.roofline import ProfiledRoofline, normalize_roofline


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--draft-model", default="/data/shared-models/Qwen3-0.6B")
    parser.add_argument("--target-model", default="/data/shared-models/Qwen3-32B")
    parser.add_argument("--mode", choices=("eager", "graph"), help="Default: exact mode recorded in the profile")
    parser.add_argument("--capacity", type=int, default=2)
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[64, 65, 96, 97])
    parser.add_argument("--max-tokens", type=int, nargs="+", default=[8, 10, 12, 14])
    parser.add_argument("--arrival-offsets", type=float, nargs="+", default=[0.0, 0.0, 0.05, 0.1])
    parser.add_argument("--arrival-lead", type=float, default=0.1)
    parser.add_argument(
        "--allow-simultaneous-arrivals",
        action="store_true",
        help="Permit one full-capacity initial batch instead of requiring online refill",
    )
    parser.add_argument("--tree-width", type=int, default=2)
    parser.add_argument("--tree-depth", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-aclgraph-entries", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fixed-budget-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After observed generation, compare tokens with a separate same-backend fixed-B1 generation",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate real profile coverage without loading models")
    return parser


def _validate_args(args):
    if not args.prompt_lengths or not (
        len(args.prompt_lengths) == len(args.max_tokens) == len(args.arrival_offsets)
    ):
        raise ValueError("Prompt lengths, output limits and arrival offsets must have equal non-zero length")
    if not 0 < args.capacity <= len(args.prompt_lengths):
        raise ValueError("Capacity must be positive and cannot exceed the request count")
    if min(*args.prompt_lengths, *args.max_tokens, args.tree_width, args.tree_depth, args.max_aclgraph_entries) <= 0:
        raise ValueError("Prompt/output/tree/graph limits must be positive")
    if args.tree_width * args.tree_depth <= 1 or min(args.max_tokens) <= 1:
        raise ValueError("Online tree acceptance needs real decode work, not only prefill")
    reserve = 2 * args.tree_width * args.tree_depth + 2
    if (
        max(length + limit for length, limit in zip(args.prompt_lengths, args.max_tokens)) + reserve
        > args.max_model_len
    ):
        raise ValueError("max-model-len cannot fit prompt, completion and eager scratch")
    if any(not math.isfinite(value) or value < 0 for value in [args.arrival_lead, *args.arrival_offsets]):
        raise ValueError("Arrival times must be finite and nonnegative")
    if args.arrival_offsets != sorted(args.arrival_offsets):
        raise ValueError("Arrival offsets must be sorted")
    if not getattr(args, "allow_simultaneous_arrivals", False) and len(set(args.arrival_offsets)) < 2:
        raise ValueError("Acceptance requires staggered arrivals unless explicitly disabled")
    if not math.isfinite(args.gpu_memory_utilization) or not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("gpu-memory-utilization must be strictly between zero and one")


def _load_profile(args):
    document = json.loads(args.profile.read_text())
    mode = args.mode or document.get("metadata", {}).get("execution_mode")
    if mode not in ("eager", "graph"):
        raise ValueError("A real strict profile must declare eager/graph execution_mode")
    profile = normalize_roofline(
        document,
        model=args.target_model,
        target_tp_size=3,
        enforce_eager=mode == "eager",
        max_model_len=args.max_model_len,
        tree_width=args.tree_width,
        tree_depth=args.tree_depth,
    )
    if not isinstance(profile, ProfiledRoofline):
        raise ValueError("Online acceptance requires a full measured profile, not a legacy bare budget map")
    return profile, mode


def _required_coverage(profile, prompt_lengths, output_limits, capacity=2):
    """Describe measured and target-only fallback coverage, including tail."""
    smallest = min(prompt_lengths) + 1
    largest = max(length + limit for length, limit in zip(prompt_lengths, output_limits))
    buckets = range(
        math.ceil(smallest / profile.context_bucket_size), math.ceil(largest / profile.context_bucket_size) + 1
    )
    rows = []
    fallback_allowed = profile.metadata.get("unprofiled_execution_policy") == "target_only"
    for batch in range(1, capacity + 1):
        for bucket in buckets:
            context = max(smallest, (bucket - 1) * profile.context_bucket_size + 1)
            budget = profile.lookup_optional(batch, context)
            if budget is None:
                if not fallback_allowed:
                    raise ValueError(
                        f"Unprofiled SpecSLO roofline key {batch}:{bucket}; strict measured tables "
                        "cannot fill or interpolate missing coverage"
                    )
                rows.append(
                    {
                        "lookup_key": f"{batch}:{bucket}",
                        "candidate_roof": None,
                        "execution": "target_only_unprofiled_fallback",
                    }
                )
                continue
            covered = sorted(
                {
                    row["verification_requests"]
                    for row in profile.evidence
                    if row["lookup_key"] == f"{batch}:{bucket}"
                }
            )
            rows.append(
                {
                    "lookup_key": f"{batch}:{bucket}",
                    "candidate_roof": budget,
                    "verification_requests": covered[0] if len(covered) == 1 else covered,
                }
            )
    return rows


class _ObservedProfile(ProfiledRoofline):
    """Preserve strict runtime semantics while recording real consumption."""

    def __init__(self, profile):
        super().__init__(profile.entries, profile.metadata, profile.evidence, profile.context_bucket_size)
        object.__setattr__(self, "lookups", [])
        object.__setattr__(self, "executions", [])

    def lookup(self, batch_size, context_len):
        budget = super().lookup(batch_size, context_len)
        self._record_lookup(batch_size, context_len, budget)
        return budget

    def lookup_optional(self, batch_size, context_len):
        budget = super().lookup_optional(batch_size, context_len)
        self._record_lookup(batch_size, context_len, budget)
        return budget

    def _record_lookup(self, batch_size, context_len, budget):
        self.lookups.append(
            {
                "batch_size": int(batch_size),
                "context_len": int(context_len),
                "lookup_key": f"{batch_size}:{math.ceil(context_len / self.context_bucket_size)}",
                "candidate_roof": budget,
            }
        )

    def validate_execution(self, batch_size, context_len, verification_requests):
        super().validate_execution(batch_size, context_len, verification_requests)
        self.executions.append(
            {
                **self.lookups[-1],
                "execution_index": len(self.executions),
                "verification_requests": verification_requests,
            }
        )


def _forward_evidence(profile, execution, counts, output):
    errors = []
    if execution is None:
        return {"passed": False, "errors": ["target forward lacks an actual validate_execution record"]}
    key = execution["lookup_key"]
    budget = execution["candidate_roof"]
    expected_key = f"{execution['batch_size']}:{math.ceil(execution['context_len'] / profile.context_bucket_size)}"
    if key != expected_key or profile.entries.get(expected_key) != budget:
        errors.append("observed budget/key does not match the real measured lookup")
    if any(count <= 0 for count in counts) or sum(counts) > budget:
        errors.append("actual candidate sum exceeds the measured roof or contains an empty tree")
    if len(counts) != execution["verification_requests"]:
        errors.append("actual target partition differs from the consumed profile partition")
    if output.get("query_count") != sum(counts) + len(counts):
        errors.append("physical query count is not roots plus exact selected candidates")
    actual_backend = output.get("attention_backend")
    try:
        profile.validate_attention_backend(actual_backend)
    except ValueError:
        errors.append("actual target backend differs from the measured profile")
    if profile.metadata["execution_mode"] == "graph" and not output.get("used_aclgraph", False):
        errors.append("actual target fell back to eager under a graph-only measured roof")
    evidence = []
    matching_contexts = set()
    for index, row in enumerate(profile.evidence):
        if row["lookup_key"] != key or row["verification_requests"] != len(counts):
            continue
        matching_contexts.add(row["context_len"])
        shapes = [
            shape
            for sweep in row.get("candidate_sweep", [])
            for shape in sweep.get("shapes", [])
            if sorted(shape["candidate_counts"]) == sorted(counts)
            and shape.get("attention_backend") == actual_backend
        ]
        evidence.append(
            {
                "evidence_index": index,
                "sampled_context_len": row["context_len"],
                "measured_roof": row["measured_roof"],
                "matching_candidate_shapes": shapes,
            }
        )
        if not shapes:
            errors.append(f"actual candidate shape {counts} was not measured at evidence row {index}")
    if not evidence:
        errors.append("actual target execution has no measurement evidence")
    return {
        **execution,
        "candidate_counts": counts,
        "actual_candidates": sum(counts),
        "physical_query_tokens": output.get("query_count"),
        "attention_backend": output.get("attention_backend"),
        "used_aclgraph": bool(output.get("used_aclgraph", False)),
        "actual_context_was_sampled": execution["context_len"] in matching_contexts,
        "coverage_convention": "512-token context bucket; only listed evidence contexts were directly measured",
        "profile_evidence": evidence,
        "passed": not errors,
        "errors": errors,
    }


def _install_observers(engine, profile, forwards, prefills):
    original_target = engine.target_tree_forward
    original_prefill = engine._prefill_and_sample_target_batch

    def target(plans, roots, rows, *args, **kwargs):
        execution = dict(profile.executions[-1]) if profile.executions else None
        output = original_target(plans, roots, rows, *args, **kwargs)
        if not engine.is_draft:
            forwards.append(
                _forward_evidence(profile, execution, [int(plan.candidate_budget) for plan in plans], output)
            )
        return output

    def prefill(prompts, states, sequence_ids=None):
        started = time.time()
        result = original_prefill(prompts, states, sequence_ids)
        prefills.append(
            {
                "request_indices": list(sequence_ids or range(len(prompts))),
                "started_wall_time": started,
                "finished_wall_time": time.time(),
            }
        )
        return result

    engine.target_tree_forward = target
    engine._prefill_and_sample_target_batch = prefill

    def restore():
        engine.target_tree_forward = original_target
        engine._prefill_and_sample_target_batch = original_prefill

    return restore


def _summarize(profile, records, output_limits, request_ids, arrivals, capacity=2):
    from examples.regression_specslo_tree import _stream_check

    errors = []
    if sorted(row["rank"] for row in records) != [0, 1, 2, 3]:
        return {"passed": False, "errors": ["missing or duplicate worker evidence"]}
    leader = next(row for row in records if row["rank"] == 1)
    if any(row["error"] is not None for row in records):
        errors.append("at least one worker failed its real generation")
    outputs = leader.get("outputs") or []
    prompt_count = len(output_limits)
    if len(outputs) != prompt_count or [len(row) for row in outputs] != output_limits:
        errors.append("not all requests reached their exact output limits")
    stream = _stream_check(leader["stream_events"], request_ids, outputs, arrivals)
    if not stream["passed"]:
        errors.append("streamed tokens/finish events/arrival ordering failed")
    reference = None
    for worker in records:
        if worker["rank"] != 1 and worker["stream_events"]:
            errors.append("a nonleader emitted a stream callback")
        observed = [index for event in worker["prefills"] for index in event["request_indices"]]
        minimum_prefills = 2 if prompt_count > capacity else 1
        if sorted(observed) != list(range(prompt_count)) or len(worker["prefills"]) < minimum_prefills:
            errors.append(f"rank {worker['rank']} did not prefill every request exactly once")
        for event in worker["prefills"]:
            if any(
                event["started_wall_time"] + 0.005 < arrivals[request_ids[index]] for index in event["request_indices"]
            ):
                errors.append("prefill began before a request's declared arrival")
        if any(not 1 <= lookup["batch_size"] <= capacity for lookup in worker["lookups"]):
            errors.append("runtime active batch exceeded physical capacity")
        if worker["rank"] == 0:
            if worker["target_forwards"]:
                errors.append("draft rank unexpectedly reported target execution")
            continue
        forwards = worker["target_forwards"]
        if len(forwards) != len(worker["validated_executions"]):
            errors.append("target forwards and validated execution records are not one-to-one")
        if not forwards or any(not event["passed"] for event in forwards):
            errors.append(f"rank {worker['rank']} has invalid or missing actual profile consumption")
        executed_batches = {event.get("batch_size") for event in forwards}
        if profile.metadata.get("unprofiled_execution_policy") == "target_only":
            if capacity not in executed_batches:
                errors.append(f"rank {worker['rank']} did not exercise the measured full-capacity profile key")
            metrics = worker.get("graph_metrics", {})
            fallback_rounds = metrics.get(
                "worker_spec_rhythm_unprofiled_target_only_fallback_rounds",
                metrics.get("spec_rhythm_unprofiled_target_only_fallback_rounds", 0),
            )
            if fallback_rounds <= 0:
                errors.append(f"rank {worker['rank']} did not exercise safe unprofiled tail fallback")
        elif executed_batches != {1, capacity}:
            errors.append(f"rank {worker['rank']} did not exercise full capacity and tail batch one")
        signature = [
            (
                event.get("lookup_key"),
                event.get("context_len"),
                event.get("candidate_counts"),
                event.get("physical_query_tokens"),
                event.get("attention_backend"),
                event.get("used_aclgraph"),
            )
            for event in forwards
        ]
        if reference is None:
            reference = signature
        elif signature != reference:
            errors.append("target TP ranks consumed different measured execution sequences")
    return {
        "passed": not errors,
        "errors": errors,
        "stream_check": stream,
        "consumed_lookup_keys": sorted({event[0] for event in (reference or [])}),
        "actual_target_forwards": len(reference or []),
    }


def _write(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _fixed_budget_reference(engine, prompts, params):
    """Separate post-test oracle, never change the observed online invocation.

    Keep the actual models, attention backend, precision and tree configuration.
    Fresh request/cache state is created by generate_batch; existing graph
    storage may be reused. Remove arrivals only for this numerical reference:
    its timing and SLO counters are not a performance comparison or warmup.
    """
    saved_config = engine.config
    try:
        engine.config = replace(saved_config, spec_rhythm_roofline=None, spec_rhythm_verification_budget=1)
        reference_params = [replace(param, arrival_ts=None) for param in params]
        result = engine.generate_batch(prompts, reference_params)
        return {
            "error": None,
            "outputs": None if result is None else [row["completion_token_ids"] for row in result],
            "candidate_budget": 1,
            "attention_backend_unchanged": True,
            "scope": "Separate fixed-B1 token reference after online test; no arrival/SLO/performance claim",
        }
    except Exception as error:
        return {"error": repr(error), "outputs": None, "candidate_budget": 1}
    finally:
        engine.config = saved_config


def main(argv=None):
    args = _build_parser().parse_args(argv)
    _validate_args(args)
    profile, mode = _load_profile(args)
    required = _required_coverage(profile, args.prompt_lengths, args.max_tokens, args.capacity)
    report = {
        "status": "dry_run" if args.dry_run else "running",
        "passed": None,
        "npu_executed": False,
        "purpose": "Strict online profile/stream acceptance and optional fixed-B1 token reference, not goodput",
        "profile_path": str(args.profile),
        "profile_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest(),
        "profile_metadata": dict(profile.metadata),
        "measured_roofline": dict(profile),
        "required_coverage": required,
        "capacity": args.capacity,
        "prompt_count": len(args.prompt_lengths),
        "prompt_lengths": args.prompt_lengths,
        "output_limits": args.max_tokens,
        "max_model_len": args.max_model_len,
        "tree_width": args.tree_width,
        "tree_depth": args.tree_depth,
        "arrival_offsets": args.arrival_offsets,
        "excluded_generation_warmups": 0,
        "worker_records": [],
        "collective_close_completed": False,
        "fixed_budget_reference_requested": args.fixed_budget_reference,
    }
    if args.dry_run:
        _write(args.output, report)
        return 0
    import torch.distributed as dist

    from vllm_ascend.spec_decode.pearl.native_engine import NativePearlConfig, NativePearlEngine, NativeSamplingParams

    if int(os.environ.get("WORLD_SIZE", 0)) != 4:
        raise ValueError("Use torchrun --nproc_per_node=4 for TP1 draft + TP3 target")
    observed_profile = _ObservedProfile(profile)
    config = NativePearlConfig(
        draft_model=args.draft_model,
        target_model=args.target_model,
        draft_tp_size=1,
        target_tp_size=3,
        gamma=args.tree_width * args.tree_depth,
        max_model_len=args.max_model_len,
        max_tokens=max(args.max_tokens),
        max_num_seqs=args.capacity,
        max_num_queued_seqs=len(args.prompt_lengths),
        max_num_batched_tokens=16384,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_aclgraph_entries=args.max_aclgraph_entries,
        enforce_eager=mode == "eager",
        enable_prefix_caching=False,
        enable_continuous_batching=True,
        enable_preemptive_scheduling=True,
        enable_spec_rhythm=True,
        spec_rhythm_online_prefill=True,
        spec_rhythm_tree_width=args.tree_width,
        spec_rhythm_tree_depth=args.tree_depth,
        spec_rhythm_roofline=observed_profile,
        spec_rhythm_max_eager_tokens=args.tree_width * args.tree_depth,
        spec_rhythm_urgency_threshold=0.0,
        spec_rhythm_acceptance_floor=0.0,
        seed=413,
    )
    engine = NativePearlEngine(config)
    forwards, prefills, events = [], [], []
    restore = _install_observers(engine, observed_profile, forwards, prefills)
    leader = engine.rank == 1
    error = None
    result = None
    report["npu_executed"] = True
    try:
        filler = engine.tokenizer.encode(" This is a synthetic online acceptance request.", add_special_tokens=False)
        if not filler:
            raise ValueError("Tokenizer produced an empty synthetic prefix")
        prompts = [(filler * math.ceil(length / len(filler)))[:length] for length in args.prompt_lengths]
        epoch = [time.time() + args.arrival_lead if leader else None]
        dist.broadcast_object_list(epoch, src=1)
        request_ids = [f"strict-online:{index}" for index in range(len(args.prompt_lengths))]
        arrivals = {identity: epoch[0] + offset for identity, offset in zip(request_ids, args.arrival_offsets)}
        params = [
            NativeSamplingParams(
                temperature=0,
                draft_temperature=0,
                max_tokens=limit,
                ignore_eos=True,
                request_id=identity,
                arrival_ts=arrivals[identity],
                slo_tpot_ms=(40, 50, 150)[index % 3],
            )
            for index, (identity, limit) in enumerate(zip(request_ids, args.max_tokens))
        ]
        report.update(prompt_token_ids=prompts, request_ids=request_ids, arrivals=arrivals)
        if leader:
            _write(args.output, report)
        try:
            result = engine.generate_batch(
                prompts,
                params,
                on_token_commit=lambda event: events.append({**event, "host_received_wall_time": time.time()}),
            )
        except Exception as caught:
            error = repr(caught)
        local = {
            "rank": engine.rank,
            "error": error,
            "lookups": observed_profile.lookups,
            "validated_executions": observed_profile.executions,
            "target_forwards": forwards,
            "prefills": prefills,
            "stream_events": events,
            "outputs": None if result is None else [row["completion_token_ids"] for row in result],
            "graph_metrics": engine.graph_metrics(),
        }
        records = [None] * 4
        dist.all_gather_object(records, local)
        report["worker_records"] = records
        report["acceptance"] = _summarize(
            profile, records, args.max_tokens, request_ids, arrivals, args.capacity
        )
        report["passed"] = report["acceptance"]["passed"]
        # Stop observers before running the distinct numerical reference.
        # The measured-profile evidence must contain only its own invocation.
        restore()
        if args.fixed_budget_reference and report["passed"]:
            from examples.regression_specslo_tree import _compare_tokens

            reference_local = {"rank": engine.rank, **_fixed_budget_reference(engine, prompts, params)}
            reference_records = [None] * 4
            dist.all_gather_object(reference_records, reference_local)
            reference_leader = next(row for row in reference_records if row["rank"] == 1)
            observed_leader = next(row for row in records if row["rank"] == 1)
            comparison = _compare_tokens(observed_leader["outputs"], reference_leader["outputs"] or [])
            reference_passed = comparison["exact_match"] and all(row["error"] is None for row in reference_records)
            report["fixed_budget_reference"] = {
                "passed": reference_passed,
                "token_comparison": comparison,
                "worker_records": reference_records,
            }
            report["passed"] = reference_passed
        elif args.fixed_budget_reference:
            report["fixed_budget_reference"] = {"skipped": "Primary online acceptance failed"}
    finally:
        restore()
        if engine.cache_allocation is not None:
            engine._release_cache()
        dist.barrier()
        dist.destroy_process_group()
        report["collective_close_completed"] = True
        report["status"] = "complete" if report["passed"] else "failed"
        if leader:
            _write(args.output, report)
            print(json.dumps({"passed": report["passed"], "acceptance": report.get("acceptance")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
