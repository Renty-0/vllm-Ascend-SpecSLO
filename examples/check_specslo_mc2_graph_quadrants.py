# SPDX-License-Identifier: Apache-2.0
"""Validate eager/ACLGraph x MC2-off/on target-only evidence.

This CPU-only checker consumes four completed
``benchmark_nano_pearl_native_target_only.py`` JSON files.  It proves that MC2
is the only switch inside each execution mode, that eager and graph runs use
the expected routes, and that all four runs emit identical token rows.  It
does not launch a model or access an NPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
QUADRANTS = {
    "eager_off": (True, False),
    "eager_on": (True, True),
    "graph_off": (False, False),
    "graph_on": (False, True),
}
RUN_CONTRACT_FIELDS = (
    "backend",
    "draft_model",
    "target_model",
    "draft_tensor_parallel_size",
    "target_tensor_parallel_size",
    "prefill_chunk_size",
    "enable_prefix_caching",
    "max_tokens",
    "warmup_prompts",
    "first_prompt_token_ids",
    "target_rms_norm_epsilon",
)
RESULT_CONTRACT_FIELDS = (
    "batch_size",
    "num_prompts",
    "num_static_chunks",
    "output_tokens",
)
MC2_COUNTERS = (
    "mc2_dispatch_fused_attempt",
    "mc2_dispatch_fused_success",
    "mc2_dispatch_fallback",
    "mc2_dispatch_exception",
)
MC2_CHAIN_COUNTERS = (
    "mc2_dispatch_native_epilogue_chained_attempt",
    "mc2_dispatch_native_epilogue_chained_success",
    "mc2_dispatch_native_epilogue_chain_flush",
)
GRAPH_ZERO_FIELDS = (
    "aclgraph_failed_captures",
    "aclgraph_capacity_fallbacks",
    "aclgraph_shape_fallbacks",
    "aclgraph_generic_eager_fallback_calls",
    "aclgraph_generic_disabled_entry_calls",
    "aclgraph_generic_unclassified_calls",
    "aclgraph_generic_runtime_validation_failures",
    "aclgraph_unvalidated_entries",
    "aclgraph_generic_unvalidated_entries",
    "aclgraph_disabled_entries",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in QUADRANTS:
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument(
        "--expected-graph-on-resident-dispatches-per-rank",
        type=int,
        help=(
            "Optional exact capture-time MC2 attempt/success count on every target rank. "
            "For the Qwen3-32B one-entry attention+down graph this is 3*2*64=384."
        ),
    )
    parser.add_argument(
        "--expected-chain-operations-per-flush",
        type=int,
        help=(
            "Require graph-on to use the deferred chain with this exact "
            "attempt/flush ratio. Qwen3-32B's 64-layer whole-model chain is 128."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read quadrant artifact {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"quadrant artifact {path} must contain a JSON object")
    return document


def _one_result(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    rows = payload.get("results")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise ValueError(f"{name} must contain exactly one result row")
    return rows[0]


def _token_rows(result: Mapping[str, Any], name: str) -> list[list[int]]:
    rows = result.get("output_token_ids")
    if not isinstance(rows, list):
        raise ValueError(f"{name} output_token_ids must be a list")
    normalized = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, list) or any(not isinstance(token, int) or isinstance(token, bool) for token in row):
            raise ValueError(f"{name} output_token_ids[{row_index}] must contain integer tokens")
        normalized.append(list(row))
    return normalized


def _token_digest(rows: Sequence[Sequence[int]]) -> str:
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _integer(metric: Mapping[str, Any], field: str, location: str, errors: list[str]) -> int:
    value = metric.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{location} lacks integer {field}")
        return 0
    return value


def _target_workers(
    metrics: Any,
    *,
    expected_ranks: set[int],
    location: str,
    errors: list[str],
) -> dict[int, Mapping[str, Any]]:
    if not isinstance(metrics, list):
        errors.append(f"{location} worker metrics are not a list")
        return {}
    workers: dict[int, Mapping[str, Any]] = {}
    for item in metrics:
        if not isinstance(item, Mapping) or item.get("is_draft_rank") in (1, True):
            continue
        rank = item.get("rank")
        if isinstance(rank, int) and not isinstance(rank, bool):
            if rank in workers:
                errors.append(f"{location} duplicates target rank {rank}")
            workers[rank] = item
    if set(workers) != expected_ranks:
        errors.append(f"{location} target ranks {sorted(workers)} != {sorted(expected_ranks)}")
    return workers


def _validate_route(
    name: str,
    payload: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    eager: bool,
    mc2_on: bool,
    expected_graph_on_resident_dispatches_per_rank: int | None,
    expected_chain_operations_per_flush: int | None,
    errors: list[str],
) -> dict[str, Any]:
    draft_tp = int(payload["draft_tensor_parallel_size"])
    target_tp = int(payload["target_tensor_parallel_size"])
    target_ranks = set(range(draft_tp, draft_tp + target_tp))
    before = _target_workers(
        result.get("worker_metrics_before_measurement"),
        expected_ranks=target_ranks,
        location=f"{name}.before",
        errors=errors,
    )
    after = _target_workers(
        result.get("worker_metrics_after_measurement"),
        expected_ranks=target_ranks,
        location=f"{name}.after",
        errors=errors,
    )
    evidence: list[dict[str, Any]] = []
    mc2_states: list[tuple[int, int, int, int]] = []
    for rank in sorted(target_ranks & set(before) & set(after)):
        old, new = before[rank], after[rank]
        old_mc2 = tuple(_integer(old, field, f"{name}.before.rank{rank}", errors) for field in MC2_COUNTERS)
        new_mc2 = tuple(_integer(new, field, f"{name}.after.rank{rank}", errors) for field in MC2_COUNTERS)
        delta_mc2 = tuple(new_value - old_value for old_value, new_value in zip(old_mc2, new_mc2))
        mc2_states.append(new_mc2)
        old_chain: tuple[int, ...] = ()
        delta_chain: tuple[int, ...] = ()
        if expected_chain_operations_per_flush is not None:
            old_chain = tuple(
                _integer(old, field, f"{name}.before.rank{rank}", errors)
                for field in MC2_CHAIN_COUNTERS
            )
            new_chain = tuple(
                _integer(new, field, f"{name}.after.rank{rank}", errors)
                for field in MC2_CHAIN_COUNTERS
            )
            delta_chain = tuple(
                new_value - old_value
                for old_value, new_value in zip(old_chain, new_chain)
            )

        old_replays = _integer(old, "aclgraph_replays", f"{name}.before.rank{rank}", errors)
        new_replays = _integer(new, "aclgraph_replays", f"{name}.after.rank{rank}", errors)
        old_generic = _integer(old, "aclgraph_generic_replay_calls", f"{name}.before.rank{rank}", errors)
        new_generic = _integer(new, "aclgraph_generic_replay_calls", f"{name}.after.rank{rank}", errors)
        replay_delta = new_replays - old_replays
        generic_delta = new_generic - old_generic
        if eager:
            for field in ("aclgraph_captures", "aclgraph_capture_attempts", "aclgraph_replays"):
                if _integer(old, field, f"{name}.before.rank{rank}", errors) != 0 or _integer(
                    new, field, f"{name}.after.rank{rank}", errors
                ) != 0:
                    errors.append(f"{name} rank {rank} used ACLGraph in eager mode ({field})")
            if _integer(old, "aclgraph_sealed", f"{name}.before.rank{rank}", errors) != 0 or _integer(
                new, "aclgraph_sealed", f"{name}.after.rank{rank}", errors
            ) != 0:
                errors.append(f"{name} rank {rank} sealed a graph cache in eager mode")
        else:
            expected_replays = int(payload["max_tokens"])
            if replay_delta != expected_replays or generic_delta != expected_replays:
                errors.append(
                    f"{name} rank {rank} replay delta total/generic={replay_delta}/{generic_delta}, "
                    f"expected {expected_replays}/{expected_replays}"
                )
            if _integer(old, "aclgraph_sealed", f"{name}.before.rank{rank}", errors) != 1 or _integer(
                new, "aclgraph_sealed", f"{name}.after.rank{rank}", errors
            ) != 1:
                errors.append(f"{name} rank {rank} graph cache is not sealed before and after measurement")
            for field in ("aclgraph_entries", "aclgraph_captures", "aclgraph_capture_attempts"):
                if _integer(new, field, f"{name}.after.rank{rank}", errors) != _integer(
                    old, field, f"{name}.before.rank{rank}", errors
                ):
                    errors.append(f"{name} rank {rank} changed {field} in the measured window")
            if _integer(old, "aclgraph_entries", f"{name}.before.rank{rank}", errors) <= 0:
                errors.append(f"{name} rank {rank} has no resident graph entry")
            for field in GRAPH_ZERO_FIELDS:
                if _integer(old, field, f"{name}.before.rank{rank}", errors) != 0 or _integer(
                    new, field, f"{name}.after.rank{rank}", errors
                ) != 0:
                    errors.append(f"{name} rank {rank} has nonzero graph failure/fallback field {field}")

        if not mc2_on:
            if any(old_mc2) or any(new_mc2):
                errors.append(f"{name} rank {rank} dispatched MC2 while disabled")
            if any(old_chain) or any(delta_chain):
                errors.append(f"{name} rank {rank} dispatched chained MC2 while disabled")
        elif eager:
            if old_mc2[0] <= 0 or old_mc2[0] != old_mc2[1] or old_mc2[2:] != (0, 0):
                errors.append(f"{name} rank {rank} warmup MC2 counters are not clean: {old_mc2}")
            if delta_mc2[0] <= 0 or delta_mc2[0] != delta_mc2[1] or delta_mc2[2:] != (0, 0):
                errors.append(f"{name} rank {rank} measured eager MC2 delta is not clean: {delta_mc2}")
            if any(old_chain) or any(delta_chain):
                errors.append(f"{name} rank {rank} used graph-only MC2 chaining in eager mode")
        else:
            if old_mc2[0] <= 0 or old_mc2[0] != old_mc2[1] or old_mc2[2:] != (0, 0):
                errors.append(f"{name} rank {rank} resident graph MC2 counters are not clean: {old_mc2}")
            if (
                expected_graph_on_resident_dispatches_per_rank is not None
                and old_mc2[:2]
                != (
                    expected_graph_on_resident_dispatches_per_rank,
                    expected_graph_on_resident_dispatches_per_rank,
                )
            ):
                errors.append(
                    f"{name} rank {rank} resident graph MC2 attempt/success={old_mc2[:2]}, "
                    f"expected {(expected_graph_on_resident_dispatches_per_rank,) * 2}"
                )
            if delta_mc2 != (0, 0, 0, 0):
                errors.append(f"{name} rank {rank} graph replay re-entered Python MC2 dispatch: {delta_mc2}")
            if expected_chain_operations_per_flush is not None:
                attempts, successes, flushes = old_chain
                if (
                    attempts <= 0
                    or attempts != successes
                    or flushes <= 0
                    or attempts != flushes * expected_chain_operations_per_flush
                ):
                    errors.append(
                        f"{name} rank {rank} resident chain attempt/success/flush="
                        f"{old_chain}, expected attempt=success=flush*"
                        f"{expected_chain_operations_per_flush}"
                    )
                if any(delta_chain):
                    errors.append(
                        f"{name} rank {rank} graph replay re-entered Python chain dispatch: "
                        f"{delta_chain}"
                    )

        evidence.append(
            {
                "rank": rank,
                "graph_replay_delta": replay_delta,
                "generic_replay_delta": generic_delta,
                "mc2_before": dict(zip(MC2_COUNTERS, old_mc2)),
                "mc2_delta": dict(zip(MC2_COUNTERS, delta_mc2)),
                "mc2_chain_before": dict(zip(MC2_CHAIN_COUNTERS, old_chain)),
                "mc2_chain_delta": dict(zip(MC2_CHAIN_COUNTERS, delta_chain)),
            }
        )
    if mc2_on and len(set(mc2_states)) > 1:
        errors.append(f"{name} target ranks have divergent MC2 dispatch state: {mc2_states}")
    return {"target_workers": evidence}


def validate_quadrants(
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    expected_graph_on_resident_dispatches_per_rank: int | None = None,
    expected_chain_operations_per_flush: int | None = None,
) -> dict[str, Any]:
    if set(payloads) != set(QUADRANTS):
        raise ValueError(f"quadrants must be exactly {sorted(QUADRANTS)}")
    if (
        expected_graph_on_resident_dispatches_per_rank is not None
        and expected_graph_on_resident_dispatches_per_rank <= 0
    ):
        raise ValueError("expected graph-on resident MC2 dispatches must be positive")
    if expected_chain_operations_per_flush is not None and expected_chain_operations_per_flush <= 1:
        raise ValueError("expected chain operations per flush must be greater than one")
    results = {name: _one_result(payload, name) for name, payload in payloads.items()}
    errors: list[str] = []

    for field in RUN_CONTRACT_FIELDS:
        values = {name: payload.get(field) for name, payload in payloads.items()}
        serialized_values = {json.dumps(value, sort_keys=True) for value in values.values()}
        if any(field not in payload for payload in payloads.values()) or len(serialized_values) != 1:
            errors.append(f"run contract differs at {field}: {values}")
    for field in RESULT_CONTRACT_FIELDS:
        values = {name: result.get(field) for name, result in results.items()}
        if any(field not in result for result in results.values()) or len(set(values.values())) != 1:
            errors.append(f"result contract differs at {field}: {values}")

    route_evidence = {}
    token_hashes = {}
    for name, (expected_eager, expected_mc2) in QUADRANTS.items():
        payload, result = payloads[name], results[name]
        if payload.get("enforce_eager") is not expected_eager:
            errors.append(f"{name} enforce_eager does not match its quadrant")
        expected_mode = "eager" if expected_eager else "aclgraph"
        if payload.get("execution_mode") != expected_mode:
            errors.append(f"{name} execution_mode is not {expected_mode}")
        if payload.get("enable_mc2") is not expected_mc2:
            errors.append(f"{name} enable_mc2 does not match its quadrant")
        expected_seal = not expected_eager
        if payload.get("seal_graph_cache_after_warmup") is not expected_seal:
            errors.append(f"{name} graph-cache seal flag does not match execution mode")
        rows = _token_rows(result, name)
        digest = _token_digest(rows)
        token_hashes[name] = digest
        if result.get("output_token_ids_sha256") != digest:
            errors.append(f"{name} stored output hash does not match complete token rows")
        route_evidence[name] = _validate_route(
            name,
            payload,
            result,
            eager=expected_eager,
            mc2_on=expected_mc2,
            expected_graph_on_resident_dispatches_per_rank=(
                expected_graph_on_resident_dispatches_per_rank
                if name == "graph_on"
                else None
            ),
            expected_chain_operations_per_flush=expected_chain_operations_per_flush,
            errors=errors,
        )

    if len(set(token_hashes.values())) != 1:
        errors.append(f"four-quadrant output tokens differ: {token_hashes}")
    on_profiles = [payloads[name] for name in ("eager_on", "graph_on")]
    profile_hashes = {payload.get("mc2_profile_sha256") for payload in on_profiles}
    profile_paths = {payload.get("mc2_profile_resolved") for payload in on_profiles}
    if len(profile_hashes) != 1 or None in profile_hashes or len(profile_paths) != 1 or None in profile_paths:
        errors.append("eager-on and graph-on do not bind the same resolved MC2 profile and SHA256")
    for name in ("eager_off", "graph_off"):
        payload = payloads[name]
        disabled_profile_fields = (
            "mc2_profile",
            "mc2_profile_sha256",
            "mc2_profile_rms_norm_epsilon",
        )
        if any(payload.get(field) is not None for field in disabled_profile_fields):
            errors.append(f"{name} carries MC2 profile state while MC2 is disabled")
    for name in ("eager_on", "graph_on"):
        payload = payloads[name]
        epsilon = payload.get("mc2_profile_rms_norm_epsilon")
        target_epsilon = payload.get("target_rms_norm_epsilon")
        if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or not math.isfinite(float(epsilon)):
            errors.append(f"{name} lacks a finite MC2 profile epsilon")
        elif epsilon != target_epsilon:
            errors.append(f"{name} profile/model RMSNorm epsilon mismatch: {epsilon} != {target_epsilon}")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if not errors else "fail",
        "passed": not errors,
        "complete_output_sha256": token_hashes,
        "profile_sha256": next(iter(profile_hashes)) if len(profile_hashes) == 1 else None,
        "expected_graph_on_resident_dispatches_per_rank": (
            expected_graph_on_resident_dispatches_per_rank
        ),
        "expected_chain_operations_per_flush": expected_chain_operations_per_flush,
        "route_evidence": route_evidence,
        "errors": errors,
    }


def main() -> None:
    args = _parser().parse_args()
    payloads = {
        name: _load(getattr(args, name))
        for name in QUADRANTS
    }
    report = validate_quadrants(
        payloads,
        expected_graph_on_resident_dispatches_per_rank=(
            args.expected_graph_on_resident_dispatches_per_rank
        ),
        expected_chain_operations_per_flush=args.expected_chain_operations_per_flush,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("; ".join(report["errors"]))


if __name__ == "__main__":
    main()
