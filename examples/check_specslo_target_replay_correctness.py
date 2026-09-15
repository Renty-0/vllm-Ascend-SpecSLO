# SPDX-License-Identifier: Apache-2.0
"""Static same-engine correctness oracle for SpecSLO target ACLGraph replay.

This is a correctness diagnostic, not a benchmark.  One persistent TP1+TP3
engine runs the same eight pre-tokenized prompts in this order:

1. packed-FIA eager target oracle;
2. conservative update-first target graph control;
3. replay-first target graph candidate, at least three times;
4. optionally, replay-first with an eager reference on every graph replay.

Every case uses batch 8, gamma 4, 32 output tokens, greedy decoding and
``ignore_eos=True``.  The tool requires exact token, round and acceptance
equality.  Its JSON intentionally contains no throughput or speedup claim.

Example::

    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
      python examples/check_specslo_target_replay_correctness.py \
      --validate-every-replay --output /root/data/target-replay-correctness.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import traceback
from pathlib import Path
from typing import Any

from vllm_ascend.spec_decode.pearl import PEARLConfig, PEARLEngine, SamplingParams

FIXED_BATCH_SIZE = 8
FIXED_MAX_TOKENS = 32
FIXED_GAMMA = 4
FIXED_PROMPTS = (
    "A shop sold 18 notebooks on Monday and 27 on Tuesday. How many notebooks were sold in total?",
    "Mina has 45 beads and gives 17 away. How many beads remain?",
    "Six boxes each contain 8 pencils. How many pencils are there altogether?",
    "A 72-page book is read equally over 9 days. How many pages are read per day?",
    "A train travels 120 kilometers in 2 hours at a constant speed. What is its speed per hour?",
    "There are 35 red balls and 28 blue balls in a bin. How many balls are in the bin?",
    "Kai saves 12 dollars each week for 5 weeks. How many dollars does Kai save?",
    "A baker made 96 rolls and packs 12 rolls in each tray. How many trays are needed?",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-model", default="/data/shared-models/Qwen3-0.6B")
    parser.add_argument("--target-model", default="/data/shared-models/Qwen3-32B")
    parser.add_argument("--batch-size", type=int, default=FIXED_BATCH_SIZE)
    parser.add_argument("--num-prompts", type=int, default=len(FIXED_PROMPTS))
    parser.add_argument("--max-tokens", type=int, default=FIXED_MAX_TOKENS)
    parser.add_argument("--gamma", type=int, default=FIXED_GAMMA)
    parser.add_argument("--candidate-repeats", type=int, default=3)
    parser.add_argument(
        "--validate-every-replay",
        action="store_true",
        help="Add one synchronized replay-first diagnostic with an eager reference on every replay.",
    )
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-aclgraph-entries", type=int, default=64)
    parser.add_argument("--target-verification-graph-buckets", type=int, default=8)
    parser.add_argument("--num-kvcache-blocks", type=int, default=-1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--worker-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--disable-cpu-binding", action="store_true")
    parser.add_argument("--output", required=True)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    fixed = {
        "batch_size": FIXED_BATCH_SIZE,
        "num_prompts": len(FIXED_PROMPTS),
        "max_tokens": FIXED_MAX_TOKENS,
        "gamma": FIXED_GAMMA,
    }
    mismatches = {
        name: (getattr(args, name), expected) for name, expected in fixed.items() if getattr(args, name) != expected
    }
    if mismatches:
        raise ValueError(f"This diagnostic has a frozen static shape; invalid overrides: {mismatches}.")
    if args.candidate_repeats < 3:
        raise ValueError("Replay-first correctness requires at least three unsynchronized repeats.")
    if args.max_model_len <= args.max_tokens:
        raise ValueError("max_model_len must leave room for the fixed prompts and completion.")
    if args.max_aclgraph_entries <= 0 or args.target_verification_graph_buckets <= 0:
        raise ValueError("ACLGraph limits must be positive.")
    if args.num_kvcache_blocks == 0 or args.num_kvcache_blocks < -1:
        raise ValueError("num_kvcache_blocks must be positive, or -1 for automatic sizing.")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1].")
    if args.worker_timeout_seconds <= 0:
        raise ValueError("worker_timeout_seconds must be positive.")


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _encode_fixed_prompts(tokenizer: Any, max_model_len: int) -> list[list[int]]:
    rows: list[list[int]] = []
    for prompt in FIXED_PROMPTS:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        tokens = [int(value) for value in tokenizer.encode(formatted)]
        if not tokens:
            raise ValueError("A fixed correctness prompt tokenized to an empty row.")
        if len(tokens) + FIXED_MAX_TOKENS + FIXED_GAMMA > max_model_len:
            raise ValueError("A fixed prompt plus completion and verification window exceeds max_model_len.")
        rows.append(tokens)
    return rows


def _metric_deltas(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    previous = {int(row["rank"]): row for row in before}
    keys = (
        "aclgraph_captures",
        "aclgraph_capture_attempts",
        "aclgraph_replays",
        "aclgraph_failed_captures",
        "aclgraph_capacity_fallbacks",
        "aclgraph_shape_fallbacks",
        "aclgraph_runtime_validation_replays",
        "aclgraph_disabled_entries",
        "aclgraph_generic_total_calls",
        "aclgraph_generic_capture_replay_calls",
        "aclgraph_generic_replay_calls",
        "aclgraph_generic_eager_fallback_calls",
        "aclgraph_generic_runtime_validation_calls",
        "aclgraph_generic_runtime_validation_failures",
        "aclgraph_generic_task_update_replays",
        "aclgraph_generic_task_update_tasks",
    )
    deltas: list[dict[str, Any]] = []
    for row in after:
        rank = int(row["rank"])
        prior = previous.get(rank, {})
        delta: dict[str, Any] = {
            "rank": rank,
            "is_draft_rank": int(row.get("is_draft_rank", 0)),
        }
        for key in keys:
            delta[key] = int(row.get(key, 0)) - int(prior.get(key, 0))
        deltas.append(delta)
    return deltas


def _snapshot_case(
    name: str,
    mode: str,
    decoded_outputs: list[str],
    token_counts: list[int],
    metrics: list[dict[str, Any]],
    before_worker_metrics: list[dict[str, Any]],
    after_worker_metrics: list[dict[str, Any]],
    worker_environments: list[dict[str, str]],
    *,
    validate_every_replay: bool,
) -> dict[str, Any]:
    token_rows = [list(map(int, row["completion_token_ids"])) for row in metrics]
    if len(token_rows) != FIXED_BATCH_SIZE:
        raise RuntimeError(f"{name} returned {len(token_rows)} rows, expected {FIXED_BATCH_SIZE}.")
    if any(len(row) != FIXED_MAX_TOKENS for row in token_rows):
        raise RuntimeError(f"{name} did not return exactly {FIXED_MAX_TOKENS} tokens per request.")
    round_counts = [int(row["round_count"]) for row in metrics]
    if len(set(round_counts)) != 1:
        raise RuntimeError(f"{name} workers exposed request-divergent global round counts: {round_counts!r}.")
    verification_rounds = [int(row["verification_rounds"]) for row in metrics]
    accepted = [int(row["accepted_draft_tokens"]) for row in metrics]
    verified = [int(row["verified_draft_tokens"]) for row in metrics]
    acceptance_rates = [float(row["acceptance_rate"]) for row in metrics]
    acceptance_lengths = [list(map(int, row["num_acc_tokens"])) for row in metrics]
    return {
        "name": name,
        "mode": mode,
        "validate_every_replay": validate_every_replay,
        "worker_environments": worker_environments,
        "output_token_ids": token_rows,
        "output_token_ids_sha256": _digest(token_rows),
        "decoded_outputs": list(decoded_outputs),
        "decoded_outputs_sha256": _digest(decoded_outputs),
        "token_counts": list(map(int, token_counts)),
        "global_round_count": round_counts[0],
        "request_verification_rounds": verification_rounds,
        "accepted_draft_tokens": accepted,
        "verified_draft_tokens": verified,
        "acceptance_rates": acceptance_rates,
        "num_acc_tokens": acceptance_lengths,
        "graph_metric_deltas": _metric_deltas(
            before_worker_metrics,
            after_worker_metrics,
        ),
    }


def _run_case(
    engine: PEARLEngine,
    name: str,
    mode: str,
    prompt_token_ids: list[list[int]],
    *,
    validate_every_replay: bool = False,
) -> dict[str, Any]:
    before_worker_metrics = copy.deepcopy(engine.last_worker_metrics)
    worker_environments = engine.configure_target_graph_correctness_mode(
        mode,
        validate_every_replay=validate_every_replay,
    )
    params = SamplingParams(
        temperature=0.0,
        draft_temperature=0.0,
        max_tokens=FIXED_MAX_TOKENS,
        ignore_eos=True,
    )
    for index, tokens in enumerate(prompt_token_ids):
        engine.add_request(tokens, params, request_id=f"static-{index}")
    decoded_outputs, token_counts, _acceptance, _elapsed = engine.generate()
    return _snapshot_case(
        name,
        mode,
        decoded_outputs,
        token_counts,
        copy.deepcopy(engine.last_metrics),
        before_worker_metrics,
        copy.deepcopy(engine.last_worker_metrics),
        worker_environments,
        validate_every_replay=validate_every_replay,
    )


def _run_case_matrix(
    engine: PEARLEngine,
    prompt_token_ids: list[list[int]],
    *,
    candidate_repeats: int,
    validate_every_replay: bool,
) -> list[dict[str, Any]]:
    cases = [
        _run_case(
            engine,
            "target_packed_fia_eager_oracle",
            "packed_fia_eager",
            prompt_token_ids,
        ),
        _run_case(
            engine,
            "target_graph_update_first_control",
            "graph_update_first",
            prompt_token_ids,
        ),
    ]
    for repeat in range(candidate_repeats):
        cases.append(
            _run_case(
                engine,
                f"target_graph_replay_first_candidate_{repeat + 1}",
                "graph_replay_first",
                prompt_token_ids,
            )
        )
    if validate_every_replay:
        cases.append(
            _run_case(
                engine,
                "target_graph_replay_first_validate_every_replay",
                "graph_replay_first",
                prompt_token_ids,
                validate_every_replay=True,
            )
        )
    return cases


def _first_token_mismatches(
    actual: list[list[int]],
    expected: list[list[int]],
) -> list[dict[str, Any]]:
    if len(actual) != len(expected):
        return [{"row_count": [len(actual), len(expected)]}]
    mismatches: list[dict[str, Any]] = []
    for request_index, (row, reference) in enumerate(zip(actual, expected)):
        common = 0
        while common < min(len(row), len(reference)) and row[common] == reference[common]:
            common += 1
        if common == len(row) == len(reference):
            continue
        mismatches.append(
            {
                "request_index": request_index,
                "first_difference": common,
                "actual_length": len(row),
                "expected_length": len(reference),
                "actual_token": row[common] if common < len(row) else None,
                "expected_token": reference[common] if common < len(reference) else None,
            }
        )
    return mismatches


def _compare_case(actual: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "output_token_ids_sha256",
        "decoded_outputs_sha256",
        "token_counts",
        "global_round_count",
        "request_verification_rounds",
        "accepted_draft_tokens",
        "verified_draft_tokens",
        "acceptance_rates",
        "num_acc_tokens",
    )
    unequal_fields = [field for field in fields if actual[field] != oracle[field]]
    token_mismatches = _first_token_mismatches(
        actual["output_token_ids"],
        oracle["output_token_ids"],
    )
    return {
        "case": actual["name"],
        "oracle": oracle["name"],
        "passed": not unequal_fields and not token_mismatches,
        "unequal_fields": unequal_fields,
        "token_mismatches": token_mismatches,
    }


def _route_check(case: dict[str, Any]) -> dict[str, Any]:
    target_rows = sorted(
        (row for row in case["graph_metric_deltas"] if not row["is_draft_rank"]),
        key=lambda row: row["rank"],
    )
    expected_target_ranks = [1, 2, 3]
    rank_layout_ok = [row["rank"] for row in target_rows] == expected_target_ranks
    per_rank: list[dict[str, Any]] = []
    for row in target_rows:
        generic_calls = int(row["aclgraph_generic_total_calls"])
        captures = int(row["aclgraph_generic_capture_replay_calls"])
        replays = int(row["aclgraph_generic_replay_calls"])
        capture_attempts = int(row["aclgraph_capture_attempts"])
        validations = int(row["aclgraph_generic_runtime_validation_calls"])
        failures = {
            key: int(row[key])
            for key in (
                "aclgraph_failed_captures",
                "aclgraph_capacity_fallbacks",
                "aclgraph_shape_fallbacks",
                "aclgraph_disabled_entries",
                "aclgraph_generic_eager_fallback_calls",
                "aclgraph_generic_runtime_validation_failures",
            )
            if int(row[key])
        }
        if case["mode"] == "packed_fia_eager":
            execution_ok = generic_calls == captures == replays == capture_attempts == 0
            expectation = "packed-FIA eager: no target generic graph activity"
        elif case["mode"] == "graph_update_first":
            execution_ok = captures > 0 and replays > 0
            expectation = "update-first control: each target rank captures and then replays"
        else:
            execution_ok = generic_calls == replays and replays > 0 and captures == 0 and capture_attempts == 0
            expectation = "replay-first candidate: resident replays only, with no recapture"
        validation_ok = not case["validate_every_replay"] or validations == replays > 0
        per_rank.append(
            {
                "rank": row["rank"],
                "passed": execution_ok and validation_ok and not failures,
                "expectation": expectation,
                "generic_calls": generic_calls,
                "capture_replay_calls": captures,
                "replay_calls": replays,
                "capture_attempts": capture_attempts,
                "runtime_validation_calls": validations,
                "validate_every_replay_covered_every_resident_replay": validation_ok,
                "failures_or_fallbacks": failures,
            }
        )
    return {
        "case": case["name"],
        "passed": rank_layout_ok and all(row["passed"] for row in per_rank),
        "target_rank_layout": [row["rank"] for row in target_rows],
        "expected_target_rank_layout": expected_target_ranks,
        "per_rank": per_rank,
    }


def _build_config(args: argparse.Namespace) -> PEARLConfig:
    return PEARLConfig(
        draft_model_path=args.draft_model,
        target_model_path=args.target_model,
        draft_tensor_parallel_size=1,
        target_tensor_parallel_size=3,
        max_num_batched_tokens=args.max_model_len * FIXED_BATCH_SIZE,
        max_num_seqs=FIXED_BATCH_SIZE,
        max_num_queued_seqs=FIXED_BATCH_SIZE,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_kvcache_blocks=args.num_kvcache_blocks,
        max_aclgraph_entries=args.max_aclgraph_entries,
        target_verification_graph_buckets=args.target_verification_graph_buckets,
        enforce_eager=False,
        gamma=FIXED_GAMMA,
        enable_prefix_caching=False,
        enable_continuous_batching=True,
        enable_preemptive_scheduling=True,
        enable_spec_rhythm=True,
        spec_rhythm_linear_full_window=True,
        spec_rhythm_online_prefill=False,
        spec_rhythm_priority_mode=False,
        spec_rhythm_min_gamma=FIXED_GAMMA,
        spec_rhythm_max_eager_tokens=0,
        spec_rhythm_tree_width=1,
        spec_rhythm_tree_depth=1,
        spec_rhythm_stable_graphs=True,
        draft_use_paged_attention=True,
        target_use_paged_attention=False,
        precompile_decode_graphs=False,
        precompile_serial_draft_graphs=True,
        enable_cpu_binding=not args.disable_cpu_binding,
        seed=0,
        worker_timeout_seconds=args.worker_timeout_seconds,
    )


def _write_document(path: str, document: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)
    document: dict[str, Any] = {
        "schema_version": 1,
        "claim_scope": "correctness_only_no_performance_claim",
        "passed": False,
        "configuration": {
            "draft_model": args.draft_model,
            "target_model": args.target_model,
            "draft_tensor_parallel_size": 1,
            "target_tensor_parallel_size": 3,
            "batch_size": FIXED_BATCH_SIZE,
            "num_prompts": len(FIXED_PROMPTS),
            "max_tokens": FIXED_MAX_TOKENS,
            "gamma": FIXED_GAMMA,
            "candidate_repeats": args.candidate_repeats,
            "validate_every_replay": args.validate_every_replay,
            "max_model_len": args.max_model_len,
            "greedy": True,
            "ignore_eos": True,
            "single_engine_load": True,
        },
    }
    try:
        config = _build_config(args)
        with PEARLEngine(config) as engine:
            prompt_token_ids = _encode_fixed_prompts(
                engine.tokenizer,
                args.max_model_len,
            )
            document["inputs"] = {
                "prompts": list(FIXED_PROMPTS),
                "prompt_token_ids": prompt_token_ids,
                "prompt_token_ids_sha256": _digest(prompt_token_ids),
            }
            cases = _run_case_matrix(
                engine,
                prompt_token_ids,
                candidate_repeats=args.candidate_repeats,
                validate_every_replay=args.validate_every_replay,
            )
        oracle = cases[0]
        comparisons = [_compare_case(case, oracle) for case in cases[1:]]
        route_checks = [_route_check(case) for case in cases]
        document.update(
            {
                "cases": cases,
                "comparisons": comparisons,
                "route_checks": route_checks,
                "passed": all(row["passed"] for row in comparisons) and all(row["passed"] for row in route_checks),
            }
        )
    except Exception as error:
        document["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    _write_document(args.output, document)
    print(json.dumps(document, ensure_ascii=False, indent=2))
    return 0 if document["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
