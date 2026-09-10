# SPDX-License-Identifier: Apache-2.0
"""Real-NPU SpecSLO tree correctness/graph/admission regression, not a benchmark.

Example (four already assigned NPUs, TP1 draft + TP3 target)::

    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python examples/regression_specslo_tree.py \
      --gsm8k /data/datasets/gsm8k/test.parquet --verification-budgets 1 4 \
      --max-tokens 32 64 --output /root/data/specslo-tree-regression.json

Every (mode, B) owns a fresh engine; repeated cases inside that engine reuse
weights, workers and graph entries. No untimed generation warmup is performed.
Boundary probes deliberately change tokenized inputs and must not be compared
against an unmodified GSM8K baseline. Synthetic boundary probes are explicitly
labelled and are not presented as GSM8K accuracy/performance measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--gsm8k", help="Local parquet or JSONL (question/turns[0]).")
    source.add_argument("--manifest", help="JSONL with prompt or prompt_token_ids; optional arrival_offset_sec/SLO.")
    parser.add_argument("--draft-model", default="/data/shared-models/Qwen3-0.6B")
    parser.add_argument("--target-model", default="/data/shared-models/Qwen3-32B")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--verification-budgets", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--max-tokens", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--modes", choices=("eager", "graph"), nargs="+", default=["eager", "graph"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--tree-width", type=int, default=2)
    parser.add_argument("--tree-depth", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-aclgraph-entries", type=int, default=64)
    parser.add_argument("--arrival-interval", type=float, default=0.05)
    parser.add_argument("--arrival-lead", type=float, default=0.1)
    parser.add_argument("--boundary-lengths", type=int, nargs="+", default=[127, 128, 129, 255, 256, 257])
    parser.add_argument("--boundary-probes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--respect-eos", action="store_true")
    parser.add_argument("--require-graph-reuse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--baseline-json", help="Optional existing target-only result; no new baseline engine is launched."
    )
    parser.add_argument(
        "--baseline-inputs", help="JSON list of token rows, or JSONL prompts proving every baseline input."
    )
    parser.add_argument("--worker-timeout", type=float, default=1800)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="Write exact input/config manifest without loading NPU engines."
    )
    return parser


def _validate_args(args) -> None:
    positive = [
        args.num_prompts,
        args.batch_size,
        args.repeats,
        args.tree_width,
        args.tree_depth,
        args.max_model_len,
        args.max_aclgraph_entries,
        *args.verification_budgets,
        *args.max_tokens,
        *args.boundary_lengths,
    ]
    if any(value <= 0 for value in positive):
        raise ValueError("Counts, token limits, tree sizes and budgets must be positive")
    if args.num_prompts <= args.batch_size:
        raise ValueError("Online-admission regression requires num_prompts > batch_size")
    if args.tree_width * args.tree_depth <= 1:
        raise ValueError("A tree regression requires width * depth > 1")
    if args.arrival_interval < 0 or args.arrival_lead < 0:
        raise ValueError("Arrival intervals and lead must be non-negative")
    if args.worker_timeout <= 0 or not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("Invalid worker timeout or memory utilization")
    if args.baseline_inputs and not args.baseline_json:
        raise ValueError("baseline_inputs requires baseline_json")


def _load_rows(path: str, count: int) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".parquet":
        import pyarrow.parquet as pq

        rows = pq.read_table(source, columns=["question"]).to_pylist()
    else:
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if len(rows) < count:
        raise ValueError(f"Input has {len(rows)} requests, expected {count}")
    return rows[:count]


def _encode_rows(rows, tokenizer) -> list[list[int]]:
    result = []
    for row in rows:
        if "prompt_token_ids" in row:
            tokens = [int(value) for value in row["prompt_token_ids"]]
        else:
            prompt = row.get("prompt", row.get("question"))
            if prompt is None and row.get("turns"):
                prompt = row["turns"][0]
            if not isinstance(prompt, str):
                raise ValueError("Each input needs prompt_token_ids, prompt, question or turns[0]")
            # Exactly matches PEARLEngine.add_request(str), including the
            # tokenizer's default Qwen3 thinking/chat-template behavior.
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            tokens = list(tokenizer.encode(formatted))
        if not tokens or any(value < 0 for value in tokens):
            raise ValueError("Prompt token rows must be non-empty and non-negative")
        result.append(tokens)
    return result


def _boundary_rows(prompts, lengths, filler_token_id: int) -> list[list[int]]:
    """Produce exact lengths while preserving the final assistant template suffix."""
    result = []
    for index, row in enumerate(prompts):
        length = lengths[index % len(lengths)]
        if len(row) >= length:
            result.append(list(row[-length:]))
        else:
            result.append([filler_token_id] * (length - len(row)) + list(row))
    return result


def _digest(rows) -> str:
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _compare_tokens(actual, expected, *, prefix_only=False) -> dict[str, Any]:
    mismatches = []
    if len(actual) != len(expected):
        return {"exact_match": False, "row_count_mismatch": [len(actual), len(expected)]}
    for index, (row, reference) in enumerate(zip(actual, expected)):
        compare_reference = reference[: len(row)] if prefix_only else reference
        if row == compare_reference:
            continue
        common = 0
        while common < min(len(row), len(compare_reference)) and row[common] == compare_reference[common]:
            common += 1
        mismatches.append(
            {
                "request_index": index,
                "first_difference": common,
                "actual_length": len(row),
                "expected_length": len(compare_reference),
                "actual_token": row[common] if common < len(row) else None,
                "expected_token": compare_reference[common] if common < len(compare_reference) else None,
            }
        )
    return {"exact_match": not mismatches, "mismatches": mismatches}


def _stream_check(events, request_ids, outputs, arrivals) -> dict[str, Any]:
    chunks = defaultdict(list)
    finish_counts = defaultdict(int)
    errors = []
    previous_elapsed = defaultdict(float)
    for event in events:
        identity = str(event.get("request_id"))
        if identity not in request_ids:
            errors.append(f"unknown request_id {identity}")
            continue
        if finish_counts[identity]:
            errors.append(f"token event after finished: {identity}")
        chunks[identity].extend(event["token_ids"])
        finish_counts[identity] += int(bool(event.get("finished")))
        elapsed = float(event.get("elapsed_seconds", 0))
        if elapsed < previous_elapsed[identity]:
            errors.append(f"nonmonotonic commit timestamp: {identity}")
        previous_elapsed[identity] = elapsed
        if event["host_received_wall_time"] + 0.005 < arrivals[identity]:
            errors.append(f"token emitted before request arrival: {identity}")
    concatenated = [chunks[identity] for identity in request_ids]
    equality = _compare_tokens(concatenated, outputs)
    missing_finished = [identity for identity in request_ids if finish_counts[identity] != 1]
    return {
        "passed": equality["exact_match"] and not errors and not missing_finished,
        "concatenation": equality,
        "errors": errors,
        "missing_or_duplicate_finished": missing_finished,
    }


def _graph_deltas(before, after) -> list[dict[str, Any]]:
    previous = {int(row["rank"]): row for row in before}
    deltas = []
    for row in after:
        rank = int(row["rank"])
        result = {"rank": rank, "is_draft_rank": int(row.get("is_draft_rank", 0))}
        for key in (
            "aclgraph_captures",
            "aclgraph_replays",
            "aclgraph_failed_captures",
            "aclgraph_capacity_fallbacks",
            "aclgraph_shape_fallbacks",
        ):
            result[key] = int(row.get(key, 0)) - int(previous.get(rank, {}).get(key, 0))
        result["reused_graph_replays"] = max(0, result["aclgraph_replays"] - result["aclgraph_captures"])
        deltas.append(result)
    return deltas


def _baseline_compare(baseline, baseline_inputs, prompts, outputs, respect_eos) -> dict[str, Any]:
    if baseline is None:
        return {"status": "not_requested"}
    if baseline_inputs is not None and baseline_inputs != prompts:
        return {"status": "different_explicit_inputs", "all_inputs_verified": False}
    complete_identity = baseline_inputs is not None and baseline_inputs == prompts
    stored_prompts = baseline.get("prompt_token_ids", baseline.get("all_prompt_token_ids"))
    if stored_prompts is not None:
        complete_identity = stored_prompts == prompts
    first_match = baseline.get("first_prompt_token_ids") == prompts[0]
    if not complete_identity and not first_match:
        return {"status": "different_or_unproven_inputs", "all_inputs_verified": False}
    if bool(baseline.get("respect_eos", False)) != respect_eos:
        return {"status": "different_eos_policy", "all_inputs_verified": complete_identity}
    entries = [row for row in baseline.get("results", []) if len(row.get("output_token_ids", [])) == len(prompts)]
    if not entries:
        return {"status": "missing_matching_output_rows", "all_inputs_verified": complete_identity}
    comparison = _compare_tokens(outputs, entries[0]["output_token_ids"], prefix_only=True)
    return {
        "status": "compared" if complete_identity else "legacy_first_input_only_unverified_comparison",
        "all_inputs_verified": complete_identity,
        "token_comparison": comparison,
        "note": "TP4/TP3 floating-point reductions can differ; mismatches require diagnosis, not tolerance hiding.",
    }


def _write_report(path: Path, report) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_args(args)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, local_files_only=True)
    rows = _load_rows(args.manifest or args.gsm8k, args.num_prompts)
    prompts = _encode_rows(rows, tokenizer)
    suites = {"source": prompts}
    if args.boundary_probes:
        filler = tokenizer.encode(" context", add_special_tokens=False)[0]
        suites["synthetic_kv_boundary"] = _boundary_rows(prompts, args.boundary_lengths, filler)
    reserve = 2 * args.tree_width * args.tree_depth + 2
    if (
        max(len(row) for suite in suites.values() for row in suite) + max(args.max_tokens) + reserve
        > args.max_model_len
    ):
        raise ValueError("max_model_len cannot fit prompt + completion + eager scratch reserve")
    baseline = json.loads(Path(args.baseline_json).read_text()) if args.baseline_json else None
    baseline_inputs = None
    if args.baseline_inputs:
        source = Path(args.baseline_inputs)
        if source.suffix == ".json":
            baseline_inputs = json.loads(source.read_text())
        else:
            baseline_inputs = _encode_rows(_load_rows(str(source), args.num_prompts), tokenizer)
    report = {
        "purpose": "SpecSLO functional/numerical regression; no speedup or goodput claim",
        "config": vars(args),
        "topology": {"draft_tp": 1, "target_tp": 3},
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "arrival_policy": "Rebase arrival_offset_sec for each case; historical absolute arrival_ts is not reused.",
        "source_sha256": hashlib.sha256(Path(args.manifest or args.gsm8k).read_bytes()).hexdigest(),
        "prompt_suites": {
            name: {"prompt_token_ids": values, "lengths": list(map(len, values)), "sha256": _digest(values)}
            for name, values in suites.items()
        },
        "warmup_generations": 0,
        "runs": [],
        "engines": [],
        "status": "dry_run" if args.dry_run else "running",
    }
    output_path = Path(args.output)
    _write_report(output_path, report)
    if args.dry_run:
        return 0
    from vllm_ascend.spec_decode.pearl.api import PEARLConfig, PEARLEngine, SamplingParams

    # Always execute eager before graph when both were requested, so identical
    # input IDs can be compared immediately without loading engines twice.
    modes = sorted(set(args.modes), key=lambda mode: mode == "graph")
    references = {}
    failed = False
    for mode in modes:
        for budget in args.verification_budgets:
            config_values = dict(
                draft_model_path=args.draft_model,
                target_model_path=args.target_model,
                draft_tensor_parallel_size=1,
                target_tensor_parallel_size=3,
                max_num_seqs=args.batch_size,
                max_num_queued_seqs=args.num_prompts,
                max_num_batched_tokens=max(16384, args.max_model_len),
                max_model_len=args.max_model_len,
                prefill_chunk_size=args.batch_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_aclgraph_entries=args.max_aclgraph_entries,
                enforce_eager=mode == "eager",
                gamma=args.tree_width * args.tree_depth,
                enable_prefix_caching=False,
                enable_continuous_batching=True,
                enable_preemptive_scheduling=True,
                enable_spec_rhythm=True,
                spec_rhythm_online_prefill=True,
                spec_rhythm_priority_mode=True,
                spec_rhythm_verification_budget=budget,
                spec_rhythm_max_eager_tokens=args.tree_width * args.tree_depth,
                spec_rhythm_tree_width=args.tree_width,
                spec_rhythm_tree_depth=args.tree_depth,
                spec_rhythm_urgency_threshold=0.0,
                spec_rhythm_acceptance_floor=0.0,
                worker_timeout_seconds=args.worker_timeout,
                seed=0,
            )
            identity = f"{mode}-B{budget}"
            engine_record = {"id": identity, "config": config_values, "status": "loading"}
            report["engines"].append(engine_record)
            _write_report(output_path, report)
            print(json.dumps({"event": "engine_loading", "id": identity}), flush=True)
            engine = None
            try:
                load_started = time.perf_counter()
                engine = PEARLEngine(PEARLConfig(**config_values))
                engine_record["load_seconds"] = time.perf_counter() - load_started
                engine_record["status"] = "running"
                for suite_name, token_rows in suites.items():
                    for max_tokens in args.max_tokens:
                        for repeat in range(args.repeats):
                            case_id = f"{identity}-{suite_name}-T{max_tokens}-R{repeat}"
                            print(json.dumps({"event": "case_start", "id": case_id}), flush=True)
                            events = []
                            request_ids = [f"{case_id}:{index}" for index in range(args.num_prompts)]
                            epoch = time.time() + args.arrival_lead
                            arrivals = {}
                            for index, (tokens, row, request_id) in enumerate(zip(token_rows, rows, request_ids)):
                                # Rebase every case; historical absolute timestamps
                                # in a manifest must not silently bypass admission.
                                offset_value = row.get("arrival_offset_sec")
                                offset = float(index * args.arrival_interval if offset_value is None else offset_value)
                                arrivals[request_id] = epoch + offset
                                slo_value = row.get("slo_tpot_ms")
                                slo = float((40.0, 50.0, 150.0)[index % 3] if slo_value is None else slo_value)
                                engine.add_request(
                                    tokens,
                                    SamplingParams(
                                        temperature=0,
                                        draft_temperature=0,
                                        max_tokens=max_tokens,
                                        ignore_eos=not args.respect_eos,
                                        request_id=request_id,
                                        arrival_ts=arrivals[request_id],
                                        slo_tpot_ms=slo,
                                    ),
                                )
                            before = [dict(metric) for metric in engine.last_worker_metrics]

                            def on_commit(event, event_buffer=events):
                                event_buffer.append({**event, "host_received_wall_time": time.time()})

                            started = time.perf_counter()
                            engine.generate(on_token_commit=on_commit)
                            elapsed = time.perf_counter() - started
                            metrics = engine.last_metrics
                            outputs = [list(metric["completion_token_ids"]) for metric in metrics]
                            deltas = _graph_deltas(before, engine.last_worker_metrics)
                            stream = _stream_check(events, request_ids, outputs, arrivals)
                            reference_key = (suite_name, max_tokens)
                            if mode == "eager" and reference_key not in references:
                                references[reference_key] = (case_id, outputs)
                            reference = references.get(reference_key)
                            comparison = None if reference is None else _compare_tokens(outputs, reference[1])
                            counters = metrics[0].get("spec_rhythm", {}) if metrics else {}
                            coverage = {
                                "online_peak_active": counters.get("spec_rhythm_peak_active_requests"),
                                "admitted_prefills": counters.get("spec_rhythm_prefill_requests"),
                                "peak_verify_candidates": counters.get("spec_rhythm_peak_verify_candidates"),
                                "rejections": counters.get("spec_rhythm_tree_rejections", 0),
                                "eager_invalidations": counters.get("spec_rhythm_tree_eager_invalidated", 0),
                                "eager_promotions": counters.get("spec_rhythm_tree_eager_promoted", 0),
                                "draft_reused_graph": any(
                                    row["is_draft_rank"] and row["reused_graph_replays"] > 0 for row in deltas
                                ),
                                "target_reused_graph": any(
                                    not row["is_draft_rank"] and row["reused_graph_replays"] > 0 for row in deltas
                                ),
                            }
                            checks = {
                                "stream": stream["passed"],
                                "request_count": len(outputs) == args.num_prompts,
                                "output_lengths": all(0 < len(row) <= max_tokens for row in outputs)
                                and (args.respect_eos or all(len(row) == max_tokens for row in outputs)),
                                "active_capacity": coverage["online_peak_active"] is not None
                                and coverage["online_peak_active"] <= args.batch_size,
                                "verify_budget": coverage["peak_verify_candidates"] is not None
                                and coverage["peak_verify_candidates"] <= budget,
                                "all_requests_prefilled": coverage["admitted_prefills"] == args.num_prompts,
                                "reference_tokens": None if comparison is None else comparison["exact_match"],
                                "target_graph_reuse": mode != "graph"
                                or not args.require_graph_reuse
                                or coverage["target_reused_graph"],
                                "graph_validation": mode != "graph"
                                or all(row["aclgraph_failed_captures"] == 0 for row in deltas),
                            }
                            passed = all(value is not False for value in checks.values())
                            failed |= not passed
                            baseline_comparison = (
                                _baseline_compare(baseline, baseline_inputs, token_rows, outputs, args.respect_eos)
                                if suite_name == "source"
                                else {"status": "different_synthetic_inputs"}
                            )
                            if (
                                baseline_comparison.get("all_inputs_verified")
                                and "token_comparison" in baseline_comparison
                            ):
                                checks["verified_baseline_tokens"] = baseline_comparison["token_comparison"][
                                    "exact_match"
                                ]
                                passed = all(value is not False for value in checks.values())
                                failed |= not passed
                            run = {
                                "id": case_id,
                                "mode": mode,
                                "verification_budget": budget,
                                "suite": suite_name,
                                "max_tokens": max_tokens,
                                "repeat": repeat,
                                "elapsed_seconds_including_arrivals": elapsed,
                                "request_ids": request_ids,
                                "arrival_timestamps": arrivals,
                                "output_token_ids": outputs,
                                "output_token_ids_sha256": _digest(outputs),
                                "checks": checks,
                                "passed": passed,
                                "skipped_checks": [name for name, value in checks.items() if value is None],
                                "stream_events": events,
                                "stream_check": stream,
                                "reference_case": None if reference is None else reference[0],
                                "reference_comparison": comparison,
                                "coverage": coverage,
                                "worker_metrics": engine.last_worker_metrics,
                                "worker_graph_deltas": deltas,
                                "request_metrics": metrics,
                                "baseline_comparison": baseline_comparison,
                            }
                            report["runs"].append(run)
                            _write_report(output_path, report)
                            print(
                                json.dumps(
                                    {
                                        "event": "case_end",
                                        "id": case_id,
                                        "passed": passed,
                                        "checks": checks,
                                        "coverage": coverage,
                                    }
                                ),
                                flush=True,
                            )
                engine_record["status"] = "completed"
            except BaseException as error:
                engine_record["status"] = "error"
                engine_record["error"] = f"{type(error).__name__}: {error}"
                report["status"] = "error"
                _write_report(output_path, report)
                raise
            finally:
                if engine is not None:
                    engine.exit()
                _write_report(output_path, report)
    report["status"] = "failed" if failed else "passed"
    report["interpretation"] = (
        "Passing covers the checked runs only. "
        "Unobserved eager promotion/rejection is not claimed as hardware coverage. "
        "Graph replay counters do not prove cross-device overlap; no throughput speedup is inferred."
    )
    _write_report(output_path, report)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
