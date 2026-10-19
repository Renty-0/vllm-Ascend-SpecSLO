# SPDX-License-Identifier: Apache-2.0
"""Create auditable SpecSLO benchmark provenance and acceptance reports.

This module is deliberately NPU-independent.  The benchmark wrapper uses it
before and after a run, while unit tests can exercise the complete reporting
contract with small JSON fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PAPER_TPOT_DEFINITION = "same_decode_elapsed_ms / output_tokens"
PAPER_GOODPUT_DEFINITION = "sum(output_tokens where paper_tpot_ms <= slo_tpot_ms) / measured_e2e_seconds"
ONLINE_E2E_TIMING_SCOPE = "arrival origin, enqueue, worker IPC, prefill and decode; inputs pretokenized"

CRITICAL_SOURCE_ARTIFACTS = ("vllm_ascend/ops/triton/spec_decode/device_paged_attention.py",)


CRITICAL_ENVIRONMENT = (
    "ASCEND_RT_VISIBLE_DEVICES",
    "TASK_QUEUE_ENABLE",
    "HCCL_OP_EXPANSION_MODE",
    "HCCL_DETERMINISTIC",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PYTORCH_NPU_ALLOC_CONF",
    "VLLM_ASCEND_PEARL_ENABLE_TP3_MM_ALL_REDUCE",
    "VLLM_ASCEND_PEARL_DRAFT_NZ_MODE",
    "VLLM_ASCEND_PEARL_TARGET_NZ_MODE",
    "VLLM_ASCEND_PEARL_TARGET_LARGE_M_NZ_MIN_ROWS",
    "VLLM_ASCEND_PEARL_TARGET_REPLICATE_INPUT_EMBEDDING",
    "VLLM_ASCEND_SPECSLO_ENABLE_EXPERIMENTAL_PARD_EAGER",
    "VLLM_ASCEND_PEARL_VERBOSE",
    "VLLM_ASCEND_PEARL_SYNC_GRAPH_INPUTS",
    "VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE",
    "VLLM_ASCEND_PEARL_DRAFT_REPLAY_FIRST_TASK_UPDATE",
    "VLLM_ASCEND_PEARL_DRAFT_STEP_MAJOR_PA_TASK_UPDATE",
    "VLLM_ASCEND_PEARL_DRAFT_PA_TASK_UPDATE_WORKERS",
    "VLLM_ASCEND_PEARL_DRAFT_PA_CONTIGUOUS_PARTITION",
    "VLLM_ASCEND_PEARL_DRAFT_GRAPH_UPDATE_STREAM_PRIORITY",
    "VLLM_ASCEND_PEARL_TARGET_GRAPH_UPDATE_STREAM_PRIORITY",
    "VLLM_ASCEND_SPECSLO_DRAFT_DEVICE_PAGED_ATTENTION",
    "VLLM_ASCEND_PEARL_TARGET_REPLAY_FIRST_TASK_UPDATE",
    "VLLM_ASCEND_PEARL_TARGET_FIA_TASK_EVENT_GROUP_SIZE",
    "VLLM_ASCEND_PEARL_TARGET_FIA_TASK_PREFIX_EVENT_GROUP_SIZE",
    "VLLM_ASCEND_PEARL_TARGET_FIA_TASK_UPDATE_WORKERS",
    "VLLM_ASCEND_PEARL_SYNC_GRAPH_TASK_UPDATE",
    "VLLM_ASCEND_PEARL_SYNC_GRAPH_REPLAY",
    "VLLM_ASCEND_PEARL_VALIDATE_GRAPH_REPLAYS",
    "VLLM_ASCEND_PEARL_SHARED_GRAPH_POOL",
    "VLLM_ASCEND_PEARL_PROFILE_PA_TASK_UPDATE",
    "VLLM_ASCEND_PEARL_NPU_PROFILE_DIR",
    "VLLM_ASCEND_PEARL_NPU_PROFILE_RANK",
    "VLLM_ASCEND_PEARL_NPU_PROFILE_WAIT_STEPS",
    "VLLM_ASCEND_PEARL_NPU_PROFILE_MEASUREMENT_ONLY",
    "VLLM_ASCEND_PEARL_MC2_CAPTURE_DIR",
    "VLLM_ASCEND_PEARL_MC2_CAPTURE_ROWS",
    "VLLM_ASCEND_PEARL_MC2_CAPTURE_KIND",
    "VLLM_ASCEND_PEARL_MC2_CAPTURE_LAYERS",
    "VLLM_ASCEND_PEARL_MC2_DOWN_PROJ",
    "VLLM_ASCEND_SPECRHYTHM_TRACE_REQUEST",
    "VLLM_ASCEND_SPECRHYTHM_GOODPUT_EDF",
    "VLLM_ASCEND_SPECRHYTHM_TARGET_SLO_FIRST",
    "VLLM_ASCEND_SPECRHYTHM_TARGET_REPLAY_AHEAD",
    "VLLM_ASCEND_SPECRHYTHM_EARLY_CURRENT_TARGET_SUBMIT",
    "VLLM_ASCEND_SPECRHYTHM_EARLY_CURRENT_MIXED_TARGET_SUBMIT",
    "VLLM_ASCEND_SPECRHYTHM_DRAIN_FIFO",
    "VLLM_ASCEND_SPECRHYTHM_ONLINE_TARGET_MAX_BATCH",
    "VLLM_ASCEND_SPECRHYTHM_DRAIN_LRTF",
    "VLLM_ASCEND_SPECRHYTHM_DRAIN_LRTF_MAX_ACTIVE",
    "VLLM_ASCEND_SPECRHYTHM_DRAIN_TARGET_LRTF",
    "VLLM_ASCEND_SPECRHYTHM_DRAIN_PADDING_REPAIR",
    "VLLM_ASCEND_SPECRHYTHM_TARGET_RESIDUAL_LRTF",
    "VLLM_ASCEND_SPECRHYTHM_RESIDUAL_EAGER_TIGHT_ONLY",
    "VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION",
    "VLLM_ASCEND_SPECRHYTHM_ARRIVAL_DEBT_CAP_TOKENS",
    "VLLM_ASCEND_SPECRHYTHM_NORMAL_ACTIVE_CAP",
    "VLLM_ASCEND_SPECRHYTHM_LOOSE_ACTIVE_CAP",
    "VLLM_ASCEND_SPECRHYTHM_PREEMPT_LOOSE_FOR_TIGHT",
    "VLLM_ASCEND_SPECRHYTHM_TIGHT_TARGET_BURST",
    "VLLM_ASCEND_SPECRHYTHM_TIGHT_HOME_FRACTION",
    "VLLM_ASCEND_SPECRHYTHM_TREE_GRAPH",
    "VLLM_ASCEND_SPECRHYTHM_USE_FIA",
    "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_BUCKET",
    "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_COMMON_KV",
    "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_RANKED_KV",
    "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_BATCHED_MASKS",
    "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_STABLE_TASK_BARRIER",
    "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_PREFILL",
    "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH",
    "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_PREFILL_ONLY",
    "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_MAX_TOKENS",
    "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_BUCKETS",
    "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_VERIFY_CAPACITIES",
    "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_PROMPT_CAPACITY",
    "VLLM_ASCEND_SPECRHYTHM_PREFETCH_AHEAD",
    "VLLM_ASCEND_SPECRHYTHM_STABLE_TARGET_VERIFY_MAX_CAPACITY",
    "VLLM_ASCEND_SPECRHYTHM_STABLE_TARGET_VERIFY_DENSE_MAX_CAPACITY",
    "VLLM_ASCEND_SPECRHYTHM_STABLE_TARGET_VERIFY_REQUEST_BUCKET",
    "VLLM_ASCEND_PEARL_TARGET_TOKEN_PAD_MULTIPLE",
    "VLLM_ASCEND_PEARL_TARGET_TP3_BALANCED_FFN_SHIFT",
    "VLLM_ASCEND_PEARL_TARGET_TP3_LIGHT_RANK",
    "VLLM_ASCEND_SPECRHYTHM_MIXED_DRAFT_PREFILL",
    "VLLM_ASCEND_SPECRHYTHM_OVERLAP_DRAFT_PREFILL",
    "VLLM_ASCEND_SPECRHYTHM_GLOO_CORRECTION",
    "VLLM_ASCEND_SPECRHYTHM_GLOO_ACCOUNTING",
    "VLLM_ASCEND_SPECRHYTHM_GLOO_PROPOSAL",
    "VLLM_ASCEND_SPECRHYTHM_GLOO_PROPOSAL_REUSE_SOURCE_BUFFERS",
    "VLLM_ASCEND_SPECRHYTHM_GLOO_PROPOSAL_REUSE_HOST_CONTINUATION",
    "VLLM_ASCEND_SPECRHYTHM_LINEAR_ACCEPTANCE_ONLY",
    "VLLM_ASCEND_SPECRHYTHM_VALIDATE_MAILBOX",
    "VLLM_ASCEND_SPECRHYTHM_FORCE_STEPWISE_TARGET",
    "VLLM_ASCEND_SPECRHYTHM_PACKED_TARGET",
    "VLLM_ASCEND_SPECRHYTHM_DISABLE_TARGET_ACLGRAPH",
    "VLLM_ASCEND_SPECRHYTHM_STEPWISE_TARGET_FIA",
    "VLLM_ASCEND_SPECRHYTHM_VALIDATE_PACKED_CAUSAL_LEAKAGE",
    "VLLM_ASCEND_USE_NATIVE_QWEN2_ROPE",
)

TARGET_REPLAY_AHEAD_REJECT_REASONS = (
    "protocol",
    "mutable_frontier",
    "pending_eager",
    "schedule",
    "ticket_epoch",
    "payload_shape",
    "graph_contract",
    "resource_contract",
    "staged_correction",
)

SHARED_PERFORMANCE_ENVIRONMENT = (
    "ASCEND_RT_VISIBLE_DEVICES",
    "TASK_QUEUE_ENABLE",
    "HCCL_OP_EXPANSION_MODE",
    "HCCL_DETERMINISTIC",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


def qualified_mc2_profile_replay_contract(
    profile_value: Any,
    *,
    profile_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Describe the exact token-row set that may contain production MC2.

    MC2 dispatch happens while an ACLGraph entry is captured.  A later graph
    replay deliberately does not re-enter the Python dispatch adapter, so its
    host-side ``fused_success`` counter cannot prove measured MC2 execution.
    This contract derives row counts independently for every profiled
    projection shape.  The full-chain intersection remains explicit, but is
    not used to erase a valid attention-only or down-only route: measured
    replay is later bound to the matching frozen static-route shape map.

    ``profile_value`` may be an already normalized ``MC2Profile`` (the normal
    benchmark path) or any value accepted by ``normalize_mc2_profile``.  The
    import is intentionally lazy so provenance/report commands that do not
    use MC2 stay NPU-independent.
    """

    from vllm_ascend.spec_decode.pearl.mc2 import (
        MC2Profile,
        normalize_mc2_profile,
    )

    profile = profile_value if isinstance(profile_value, MC2Profile) else normalize_mc2_profile(profile_value)
    metadata = profile.metadata
    tp_size = int(metadata["tensor_parallel_size"])
    epsilon = float(metadata["rms_norm_epsilon"])
    qualified_by_shape: dict[tuple[Any, ...], set[int]] = {}
    for key in profile.entries:
        rows = int(key[0])
        shape_signature = tuple(key[1:])
        qualification = profile.qualify_static_shape(
            key,
            tp_size=tp_size,
            epsilon=epsilon,
        )
        qualified_by_shape.setdefault(shape_signature, set())
        if qualification.qualified:
            qualified_by_shape[shape_signature].add(rows)

    full_chain_rows = set.intersection(*qualified_by_shape.values()) if qualified_by_shape else set()
    any_projection_rows = set().union(*qualified_by_shape.values()) if qualified_by_shape else set()
    resolved_path = (
        Path(profile_path).resolve()
        if profile_path is not None
        else (Path(profile_value).resolve() if isinstance(profile_value, (str, os.PathLike)) else None)
    )
    return {
        "schema": "specslo-mc2-replay-profile-contract-v2",
        "profile_path": str(resolved_path) if resolved_path is not None else None,
        "profile_sha256": (sha256_file(resolved_path) if resolved_path is not None else None),
        "profile_source_sha256": metadata.get("source_sha256"),
        "operator": metadata.get("operator"),
        "tensor_parallel_size": tp_size,
        "rms_norm_epsilon": epsilon,
        "standalone_chained_flush": (metadata.get("standalone_chained_flush") is True),
        "profile_shape_families": len(qualified_by_shape),
        "qualified_rows_by_shape": {
            repr(signature): sorted(rows)
            for signature, rows in sorted(
                qualified_by_shape.items(),
                key=lambda item: repr(item[0]),
            )
        },
        "qualified_any_projection_rows": sorted(any_projection_rows),
        "qualified_full_chain_rows": sorted(full_chain_rows),
    }


def measure_qualified_mc2_replay_coverage(
    worker_deltas: Sequence[Mapping[str, Any]],
    profile_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind measured target ACLGraph replay rows to qualified MC2 rows.

    Counts are token-row histogram observations.  A target graph replay that
    records more than one physical row contributes once per recorded row;
    the denominator therefore remains auditable and is not mislabeled as the
    number of model forwards.  Formal admission requires a positive qualified
    observation on every target rank, zero measured MC2 fallback/exception,
    and one identical frozen static-route manifest across target ranks.
    """

    errors: list[str] = []
    raw_rows_by_shape = profile_contract.get("qualified_rows_by_shape")
    qualified_rows_by_shape: dict[str, set[int]] = {}
    if not isinstance(raw_rows_by_shape, Mapping) or not raw_rows_by_shape:
        errors.append("MC2 profile contract lacks qualified_rows_by_shape")
    else:
        for raw_signature, raw_rows in raw_rows_by_shape.items():
            signature = str(raw_signature)
            if not signature or isinstance(raw_rows, (str, bytes)) or not isinstance(raw_rows, Sequence):
                errors.append(f"MC2 profile shape {signature!r} has invalid qualified rows")
                continue
            rows = {
                int(value) for value in raw_rows if not isinstance(value, bool) and isinstance(value, int) and value > 0
            }
            if len(rows) != len(raw_rows):
                errors.append(f"MC2 profile shape {signature!r} rows are not unique positive integers")
            qualified_rows_by_shape[signature] = rows
    qualified_rows = set().union(*qualified_rows_by_shape.values()) if qualified_rows_by_shape else set()
    full_chain_rows = set.intersection(*qualified_rows_by_shape.values()) if qualified_rows_by_shape else set()
    if sorted(qualified_rows) != profile_contract.get("qualified_any_projection_rows"):
        errors.append("MC2 profile any-projection row union is inconsistent")
    if sorted(full_chain_rows) != profile_contract.get("qualified_full_chain_rows"):
        errors.append("MC2 profile full-chain row intersection is inconsistent")
    if not qualified_rows:
        errors.append("MC2 profile has no independently qualified projection row")
    if profile_contract.get("standalone_chained_flush") is not True:
        errors.append("MC2 profile lacks standalone chained-flush qualification")

    target_workers = [worker for worker in worker_deltas if int(worker.get("is_draft_rank", 0)) == 0]
    if not target_workers:
        errors.append("measured graph evidence has no target workers")

    target_evidence: list[dict[str, Any]] = []
    static_route_signatures: list[tuple[Any, ...]] = []
    replay_signatures: list[tuple[Any, ...]] = []
    profile_operator = profile_contract.get("operator")
    for worker in target_workers:
        rank = int(worker.get("rank", -1))
        histograms: dict[str, dict[str, int]] = {}
        invalid_histogram_bins: list[dict[str, Any]] = []
        combined_counts: dict[int, int] = {}
        for kind in ("generic", "target"):
            raw_histogram = worker.get(
                f"aclgraph_{kind}_replay_token_rows_histogram_delta",
                {},
            )
            if not isinstance(raw_histogram, Mapping):
                invalid_histogram_bins.append({"kind": kind, "row": None, "count": raw_histogram})
                raw_histogram = {}
            normalized: dict[str, int] = {}
            for raw_row, raw_count in raw_histogram.items():
                try:
                    row = int(raw_row)
                    count = int(raw_count)
                except (TypeError, ValueError):
                    invalid_histogram_bins.append({"kind": kind, "row": raw_row, "count": raw_count})
                    continue
                if row <= 0 or count < 0:
                    invalid_histogram_bins.append({"kind": kind, "row": raw_row, "count": raw_count})
                    continue
                if count == 0:
                    continue
                normalized[str(row)] = normalized.get(str(row), 0) + count
                combined_counts[row] = combined_counts.get(row, 0) + count
            histograms[kind] = dict(sorted(normalized.items(), key=lambda item: int(item[0])))

        total_count = sum(combined_counts.values())
        qualified_count = sum(count for rows, count in combined_counts.items() if rows in qualified_rows)
        full_chain_count = sum(count for rows, count in combined_counts.items() if rows in full_chain_rows)
        qualified_counts_by_shape = {
            signature: sum(count for rows, count in combined_counts.items() if rows in shape_rows)
            for signature, shape_rows in sorted(qualified_rows_by_shape.items())
        }
        unqualified_counts = {
            str(rows): count for rows, count in sorted(combined_counts.items()) if rows not in qualified_rows
        }
        route_digest = str(worker.get("mc2_static_route_digest", ""))
        route_count = int(worker.get("mc2_static_route_count", 0))
        qualified_route_count = int(worker.get("mc2_static_qualified_row_routes", 0))
        raw_static_rows_by_shape = worker.get(
            "mc2_static_qualified_rows_by_shape",
            {},
        )
        static_shape_evidence_valid = isinstance(raw_static_rows_by_shape, Mapping)
        static_rows_by_shape: dict[str, tuple[int, ...]] = {}
        if static_shape_evidence_valid:
            for raw_signature, values in raw_static_rows_by_shape.items():
                signature = str(raw_signature)
                if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                    static_shape_evidence_valid = False
                    continue
                rows = tuple(
                    int(value)
                    for value in values
                    if not isinstance(value, bool) and isinstance(value, int) and value > 0
                )
                if len(rows) != len(values) or tuple(sorted(set(rows))) != rows:
                    static_shape_evidence_valid = False
                static_rows_by_shape[signature] = rows
        raw_route_counts_by_shape = worker.get(
            "mc2_static_route_counts_by_shape",
            {},
        )
        if not isinstance(raw_route_counts_by_shape, Mapping):
            static_shape_evidence_valid = False
            route_counts_by_shape: dict[str, int] = {}
        else:
            route_counts_by_shape = {
                str(signature): int(count)
                for signature, count in raw_route_counts_by_shape.items()
                if not isinstance(count, bool) and isinstance(count, int)
            }
            if len(route_counts_by_shape) != len(raw_route_counts_by_shape):
                static_shape_evidence_valid = False
        expected_static_rows_by_shape = {
            signature: tuple(sorted(rows)) for signature, rows in sorted(qualified_rows_by_shape.items())
        }
        expected_qualified_route_count = sum(
            route_counts_by_shape.get(signature, 0) * len(rows)
            for signature, rows in expected_static_rows_by_shape.items()
        )
        static_projection_routes_match_profile = (
            static_shape_evidence_valid
            and bool(route_counts_by_shape)
            and (
                static_rows_by_shape == expected_static_rows_by_shape
                and set(route_counts_by_shape) == set(expected_static_rows_by_shape)
                and all(count > 0 for count in route_counts_by_shape.values())
                and sum(route_counts_by_shape.values()) == route_count
                and expected_qualified_route_count == qualified_route_count
            )
        )
        operator_kinds_value = worker.get("mc2_static_operator_kinds", ())
        operator_kinds = (
            tuple(str(value) for value in operator_kinds_value)
            if isinstance(operator_kinds_value, Sequence) and not isinstance(operator_kinds_value, (str, bytes))
            else ()
        )
        fallback = int(worker.get("mc2_dispatch_fallback_delta", 0))
        exception = int(worker.get("mc2_dispatch_exception_delta", 0))
        active_graph_kinds = tuple(kind for kind, histogram in histograms.items() if histogram)
        static_signature = (
            route_digest,
            route_count,
            operator_kinds,
            qualified_route_count,
            tuple(sorted(static_rows_by_shape.items())),
            tuple(sorted(route_counts_by_shape.items())),
        )
        replay_signature = tuple((kind, tuple(histograms[kind].items())) for kind in ("generic", "target"))
        static_route_signatures.append(static_signature)
        replay_signatures.append(replay_signature)

        rank_errors = []
        if invalid_histogram_bins:
            rank_errors.append("invalid measured replay histogram bins")
        if total_count <= 0:
            rank_errors.append("no measured target replay token-row observations")
        if qualified_count <= 0:
            rank_errors.append("no measured replay at an independently qualified MC2 projection row")
        if fallback != 0:
            rank_errors.append("measured MC2 fallback is nonzero")
        if exception != 0:
            rank_errors.append("measured MC2 exception is nonzero")
        if not route_digest or route_count <= 0 or qualified_route_count <= 0:
            rank_errors.append("frozen MC2 static-route evidence is incomplete")
        if not static_projection_routes_match_profile:
            rank_errors.append("frozen MC2 projection-shape routes do not match the profile contract")
        if not operator_kinds or (profile_operator is not None and operator_kinds != (str(profile_operator),)):
            rank_errors.append("MC2 static operator does not match the profile")
        if len(active_graph_kinds) != 1:
            rank_errors.append("measured target replay does not use exactly one graph kind")
        errors.extend(f"target rank {rank}: {message}" for message in rank_errors)
        target_evidence.append(
            {
                "rank": rank,
                "active_graph_kinds": list(active_graph_kinds),
                "measured_replay_token_row_histograms": histograms,
                "measured_replay_token_row_observations": total_count,
                "qualified_mc2_replay_count": qualified_count,
                "qualified_mc2_replay_coverage": (qualified_count / total_count if total_count > 0 else 0.0),
                "qualified_full_chain_replay_count": full_chain_count,
                "qualified_projection_replay_counts_by_shape": qualified_counts_by_shape,
                "unqualified_replay_token_row_observations": unqualified_counts,
                "invalid_replay_histogram_bins": invalid_histogram_bins,
                "measured_mc2_fallback": fallback,
                "measured_mc2_exception": exception,
                "static_route_digest": route_digest,
                "static_route_count": route_count,
                "static_operator_kinds": list(operator_kinds),
                "static_qualified_row_routes": qualified_route_count,
                "static_qualified_rows_by_shape": {
                    signature: list(rows) for signature, rows in sorted(static_rows_by_shape.items())
                },
                "static_route_counts_by_shape": dict(sorted(route_counts_by_shape.items())),
                "static_projection_routes_match_profile": (static_projection_routes_match_profile),
                "passed": not rank_errors,
                "errors": rank_errors,
            }
        )

    static_routes_consistent = bool(static_route_signatures) and len(set(static_route_signatures)) == 1
    measured_replay_histograms_consistent = bool(replay_signatures) and len(set(replay_signatures)) == 1
    if not static_routes_consistent:
        errors.append("target ranks do not share one identical frozen MC2 route manifest")
    if not measured_replay_histograms_consistent:
        errors.append("target ranks do not report identical measured replay histograms")

    return {
        "schema": "specslo-mc2-measured-replay-coverage-v2",
        "count_definition": (
            "positive target ACLGraph replay token-row histogram observations "
            "whose physical M is independently qualified for at least one "
            "exact projection shape in the frozen static-route manifest"
        ),
        "profile": dict(profile_contract),
        "qualified_any_projection_rows": sorted(qualified_rows),
        "qualified_full_chain_rows": sorted(full_chain_rows),
        "target_ranks": target_evidence,
        "static_routes_consistent": static_routes_consistent,
        "measured_replay_histograms_consistent": (measured_replay_histograms_consistent),
        "passed": not errors,
        "errors": errors,
    }


def target_fia_parallel_update_evidence(
    worker_metrics: Sequence[Mapping[str, Any]],
    measured_worker_deltas: Sequence[Mapping[str, Any]],
    *,
    expected_workers: int,
) -> dict[str, Any]:
    """Validate the measured TP1+TP3 target-FIA update-worker contract.

    The configured worker count is an absolute graph-runner property, while
    parallel-update replay/task counters are cumulative.  The latter must
    therefore come from the post-warmup measurement deltas.  Keeping those
    two evidence sources separate prevents graph qualification from being
    mistaken for work in the timed online window.
    """

    if expected_workers not in (1, 2, 3, 4):
        raise ValueError("target FIA task-update workers must be 1, 2, 3, or 4")

    errors: list[str] = []

    def by_rank(
        values: Sequence[Mapping[str, Any]],
        label: str,
    ) -> dict[int, Mapping[str, Any]]:
        ranks: dict[int, Mapping[str, Any]] = {}
        for value in values:
            if not isinstance(value, Mapping):
                errors.append(f"{label} contains a non-object worker row")
                continue
            try:
                rank = int(value["rank"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{label} contains a worker without an integer rank")
                continue
            if rank in ranks:
                errors.append(f"{label} contains duplicate rank {rank}")
                continue
            ranks[rank] = value
        return ranks

    workers = by_rank(worker_metrics, "worker metrics")
    measured = by_rank(measured_worker_deltas, "measured worker deltas")
    expected_ranks = {0, 1, 2, 3}
    if set(workers) != expected_ranks:
        errors.append("worker metrics rank set is not the TP1+TP3 set [0, 1, 2, 3]")
    if set(measured) != expected_ranks:
        errors.append("measured worker-delta rank set is not the TP1+TP3 set [0, 1, 2, 3]")

    role_rows: dict[int, dict[str, Any]] = {}
    graph_zero_fields = (
        "aclgraph_failed_captures",
        "aclgraph_capacity_fallbacks",
        "aclgraph_shape_fallbacks",
        "aclgraph_generic_eager_fallback_calls",
        "aclgraph_draft_eager_fallback_calls",
        "aclgraph_target_eager_fallback_calls",
        "aclgraph_generic_runtime_validation_failures",
        "aclgraph_draft_runtime_validation_failures",
        "aclgraph_target_runtime_validation_failures",
    )
    measured_graph_zero_fields = tuple(f"{name}_delta" for name in graph_zero_fields)
    target_service_graph_zero_fields = (
        "worker_spec_rhythm_mixed_target_graph_bounded_shape_fallback_batches",
        "worker_spec_rhythm_mixed_target_graph_token_capacity_fallback_batches",
        "worker_spec_rhythm_mixed_target_graph_execution_fallback_batches",
        "worker_spec_rhythm_stable_target_verify_graph_bounded_shape_fallback_batches",
        "worker_spec_rhythm_stable_target_verify_graph_token_capacity_fallback_batches",
        "worker_spec_rhythm_stable_target_verify_graph_execution_fallback_batches",
    )

    for rank in sorted(expected_ranks & set(workers)):
        worker = workers[rank]
        measured_worker = measured.get(rank, {})
        is_draft = int(worker.get("is_draft_rank", -1))
        measured_is_draft = int(measured_worker.get("is_draft_rank", -1))
        expected_is_draft = int(rank == 0)
        if is_draft != expected_is_draft:
            errors.append(f"worker rank {rank} has the wrong draft/target role")
        if measured_is_draft != expected_is_draft:
            errors.append(f"measured worker rank {rank} has the wrong draft/target role")

        configured_workers = int(worker.get("aclgraph_target_fia_task_update_workers", -1))
        expected_rank_workers = 1 if expected_is_draft else expected_workers
        if configured_workers != expected_rank_workers:
            errors.append(
                f"worker rank {rank} target-FIA update workers {configured_workers} != {expected_rank_workers}"
            )

        for field in graph_zero_fields:
            if field not in worker or int(worker.get(field, -1)) != 0:
                errors.append(f"worker rank {rank} {field} is nonzero or missing")
        for field in measured_graph_zero_fields:
            if field not in measured_worker or int(measured_worker.get(field, -1)) != 0:
                errors.append(f"measured worker rank {rank} {field} is nonzero or missing")

        parallel_replays = int(
            measured_worker.get(
                "aclgraph_target_fia_parallel_update_replays_delta",
                -1,
            )
        )
        parallel_tasks = int(
            measured_worker.get(
                "aclgraph_target_fia_parallel_update_tasks_delta",
                -1,
            )
        )
        if expected_is_draft:
            if parallel_replays != 0 or parallel_tasks != 0:
                errors.append("draft rank 0 unexpectedly reports target-FIA parallel updates")
        else:
            for field in target_service_graph_zero_fields:
                if field not in measured_worker or int(measured_worker.get(field, -1)) != 0:
                    errors.append(f"measured target rank {rank} {field} is nonzero or missing")
            if expected_workers > 1 and parallel_replays <= 0:
                errors.append(f"measured target rank {rank} has no parallel FIA update replay")
            if expected_workers > 1 and parallel_tasks <= 0:
                errors.append(f"measured target rank {rank} has no parallel FIA update task")
            if expected_workers == 1 and (parallel_replays != 0 or parallel_tasks != 0):
                errors.append(f"measured target rank {rank} used the parallel FIA path with one configured worker")

        role_rows[rank] = {
            "role": "draft" if expected_is_draft else "target",
            "configured_workers": configured_workers,
            "measured_parallel_update_replays": parallel_replays,
            "measured_parallel_update_tasks": parallel_tasks,
        }

    return {
        "schema": "specslo-target-fia-update-workers-evidence-v1",
        "expected_target_workers": expected_workers,
        "expected_draft_workers": 1,
        "ranks": {str(rank): value for rank, value in role_rows.items()},
        "passed": not errors,
        "errors": errors,
    }


def draft_pa_parallel_update_evidence(
    worker_metrics: Sequence[Mapping[str, Any]],
    measured_worker_deltas: Sequence[Mapping[str, Any]],
    *,
    expected_workers: int,
) -> dict[str, Any]:
    """Validate measured TP1 draft-PA task-update parallelism.

    The configured worker count is an absolute graph-runner property.  Work
    and fallback counters are cumulative, so the evidence for the timed
    window must come from post-warmup deltas.  Exactly rank zero may use this
    path; all three target ranks must remain on their single-worker route.
    """

    if expected_workers not in (1, 2, 4):
        raise ValueError("draft PA task-update workers must be 1, 2, or 4")

    errors: list[str] = []

    def by_rank(
        values: Sequence[Mapping[str, Any]],
        label: str,
    ) -> dict[int, Mapping[str, Any]]:
        ranks: dict[int, Mapping[str, Any]] = {}
        for value in values:
            if not isinstance(value, Mapping):
                errors.append(f"{label} contains a non-object worker row")
                continue
            try:
                rank = int(value["rank"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{label} contains a worker without an integer rank")
                continue
            if rank in ranks:
                errors.append(f"{label} contains duplicate rank {rank}")
                continue
            ranks[rank] = value
        return ranks

    workers = by_rank(worker_metrics, "worker metrics")
    measured = by_rank(measured_worker_deltas, "measured worker deltas")
    expected_ranks = {0, 1, 2, 3}
    if set(workers) != expected_ranks:
        errors.append("worker metrics rank set is not the TP1+TP3 set [0, 1, 2, 3]")
    if set(measured) != expected_ranks:
        errors.append("measured worker-delta rank set is not the TP1+TP3 set [0, 1, 2, 3]")

    graph_zero_fields = (
        "aclgraph_failed_captures",
        "aclgraph_capacity_fallbacks",
        "aclgraph_shape_fallbacks",
        "aclgraph_generic_eager_fallback_calls",
        "aclgraph_draft_eager_fallback_calls",
        "aclgraph_target_eager_fallback_calls",
        "aclgraph_generic_runtime_validation_failures",
        "aclgraph_draft_runtime_validation_failures",
        "aclgraph_target_runtime_validation_failures",
    )
    measured_graph_zero_fields = tuple(f"{name}_delta" for name in graph_zero_fields)
    role_rows: dict[int, dict[str, Any]] = {}
    for rank in sorted(expected_ranks & set(workers)):
        worker = workers[rank]
        measured_worker = measured.get(rank, {})
        expected_is_draft = int(rank == 0)
        is_draft = int(worker.get("is_draft_rank", -1))
        measured_is_draft = int(measured_worker.get("is_draft_rank", -1))
        if is_draft != expected_is_draft:
            errors.append(f"worker rank {rank} has the wrong draft/target role")
        if measured_is_draft != expected_is_draft:
            errors.append(f"measured worker rank {rank} has the wrong draft/target role")

        configured_workers = int(worker.get("aclgraph_draft_pa_task_update_workers", -1))
        expected_rank_workers = expected_workers if expected_is_draft else 1
        if configured_workers != expected_rank_workers:
            errors.append(f"worker rank {rank} draft-PA update workers {configured_workers} != {expected_rank_workers}")

        for field in graph_zero_fields:
            if field not in worker or int(worker.get(field, -1)) != 0:
                errors.append(f"worker rank {rank} {field} is nonzero or missing")
        for field in measured_graph_zero_fields:
            if field not in measured_worker or int(measured_worker.get(field, -1)) != 0:
                errors.append(f"measured worker rank {rank} {field} is nonzero or missing")

        parallel_replays = int(
            measured_worker.get(
                "aclgraph_draft_pa_parallel_update_replays_delta",
                -1,
            )
        )
        parallel_tasks = int(
            measured_worker.get(
                "aclgraph_draft_pa_parallel_update_tasks_delta",
                -1,
            )
        )
        parallel_fallbacks = int(
            measured_worker.get(
                "aclgraph_draft_pa_parallel_update_fallback_replays_delta",
                -1,
            )
        )
        raw_rejections = measured_worker.get("aclgraph_draft_pa_parallel_update_rejections_delta")
        if not isinstance(raw_rejections, Mapping):
            parallel_rejections: dict[str, int] = {}
            errors.append(f"measured worker rank {rank} lacks draft-PA parallel rejection deltas")
        else:
            parallel_rejections = {}
            for reason, count in raw_rejections.items():
                try:
                    normalized_count = int(count)
                except (TypeError, ValueError):
                    errors.append(
                        f"measured worker rank {rank} draft-PA rejection count for {reason!r} is not an integer"
                    )
                    continue
                if normalized_count != 0:
                    parallel_rejections[str(reason)] = normalized_count
            if parallel_rejections:
                errors.append(
                    f"measured worker rank {rank} draft-PA parallel rejections={parallel_rejections}, expected empty"
                )
        if parallel_fallbacks != 0:
            errors.append(
                f"measured worker rank {rank} draft-PA parallel fallback replays={parallel_fallbacks}, expected 0"
            )
        if expected_is_draft and expected_workers > 1:
            if parallel_replays <= 0:
                errors.append("measured draft rank 0 has no parallel PA update replay")
            if parallel_tasks <= 0:
                errors.append("measured draft rank 0 has no parallel PA update task")
        elif parallel_replays != 0 or parallel_tasks != 0:
            if expected_is_draft:
                errors.append("measured draft rank 0 used the parallel PA path with one configured worker")
            else:
                errors.append(f"target rank {rank} unexpectedly reports draft-PA parallel updates")

        role_rows[rank] = {
            "role": "draft" if expected_is_draft else "target",
            "configured_workers": configured_workers,
            "measured_parallel_update_replays": parallel_replays,
            "measured_parallel_update_tasks": parallel_tasks,
            "measured_parallel_fallback_replays": parallel_fallbacks,
            "measured_parallel_rejections": parallel_rejections,
        }

    return {
        "schema": "specslo-draft-pa-update-workers-evidence-v1",
        "expected_draft_workers": expected_workers,
        "expected_target_workers": 1,
        "ranks": {str(rank): value for rank, value in role_rows.items()},
        "passed": not errors,
        "errors": errors,
    }


def target_replay_ahead_evidence(
    worker_metrics: Sequence[Mapping[str, Any]],
    *,
    enabled: bool,
    require_submissions: bool = True,
) -> dict[str, Any]:
    """Check that every rank consumed exactly the target graph it submitted.

    The worker service counters are reset for the measured generate call.
    They are in worker metrics, not the ACLGraph-only delta rows. A
    replay-ahead result mutates target KV before the next
    scheduler cycle, so a missing or mismatched rank is a hard failure.
    """

    errors: list[str] = []
    rows: dict[int, dict[str, Any]] = {}
    counter_names = (
        "attempts",
        "eligible",
        "submitted",
        "consumed",
        "signature_mismatches",
    )
    prefix = "worker_spec_rhythm_target_replay_ahead_"
    for worker in worker_metrics:
        try:
            rank = int(worker["rank"])
        except (KeyError, TypeError, ValueError):
            errors.append("replay-ahead evidence has a worker without an integer rank")
            continue
        if rank in rows:
            errors.append(f"replay-ahead evidence has duplicate rank {rank}")
            continue
        counters: dict[str, int] = {}
        for name in counter_names:
            field = prefix + name
            try:
                value = int(worker[field])
            except (KeyError, TypeError, ValueError):
                errors.append(f"rank {rank} lacks integer {field}")
                value = -1
            if value < 0:
                errors.append(f"rank {rank} has negative {field}")
            counters[name] = value
        active = int(worker.get(prefix + "enabled", -1))
        if active != int(enabled):
            errors.append(f"rank {rank} replay-ahead enablement differs from the campaign")
        rejects: dict[str, int] = {}
        for reason in TARGET_REPLAY_AHEAD_REJECT_REASONS:
            field = prefix + "reject_" + reason
            try:
                count = int(worker[field])
            except (KeyError, TypeError, ValueError):
                errors.append(f"rank {rank} lacks integer {field}")
                count = -1
            if count < 0:
                errors.append(f"rank {rank} has negative {field}")
            rejects[reason] = count
        if enabled:
            if counters["attempts"] <= 0:
                errors.append(f"rank {rank} made no replay-ahead attempt")
            if require_submissions and counters["submitted"] <= 0:
                errors.append(f"rank {rank} submitted no replay-ahead target graph")
            if counters["eligible"] != counters["submitted"]:
                errors.append(f"rank {rank} eligible/submitted replay-ahead counts differ")
            if counters["submitted"] != counters["consumed"]:
                errors.append(f"rank {rank} submitted/consumed replay-ahead counts differ")
            if counters["signature_mismatches"] != 0:
                errors.append(f"rank {rank} observed a replay-ahead signature mismatch")
            if counters["attempts"] != counters["eligible"] + sum(rejects.values()):
                errors.append(f"rank {rank} replay-ahead attempts are not accounted for")
        elif any(counters.values()) or any(rejects.values()):
            errors.append(f"rank {rank} used replay-ahead while it was disabled")
        rows[rank] = {"enabled": active, **counters, "rejects": rejects}

    if set(rows) != {0, 1, 2, 3}:
        errors.append("replay-ahead evidence does not cover ranks 0,1,2,3")
    if enabled and len(rows) == 4:
        for name in counter_names:
            if len({row[name] for row in rows.values()}) != 1:
                errors.append(f"replay-ahead {name} differs across ranks")
        if len({tuple(row["rejects"].items()) for row in rows.values()}) != 1:
            errors.append("replay-ahead rejection reasons differ across ranks")
    return {
        "schema": "specslo-target-replay-ahead-evidence-v1",
        "enabled": enabled,
        "ranks": {str(rank): row for rank, row in sorted(rows.items())},
        "passed": not errors,
        "errors": errors,
    }


def capture_runtime_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    """Return a stable, exhaustive SpecSLO performance environment record."""

    source = os.environ if environment is None else environment
    return {name: source.get(name) for name in CRITICAL_ENVIRONMENT}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_visible_devices(value: str | None) -> list[str]:
    """Normalize the user-visible physical-card selection without renumbering it."""
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _run_git(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", "-C", str(repo), *args),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.rstrip("\n")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str | None:
    artifact = Path(path)
    if not artifact.is_file():
        return None
    digest = hashlib.sha256()
    with artifact.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_provenance(
    *,
    repo: str | os.PathLike[str],
    mode: str,
    run_id: str,
    result_json: str | os.PathLike[str],
    log_file: str | os.PathLike[str],
    command: Sequence[str],
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Capture the immutable pre-run context needed to audit a result."""
    repo_path = Path(repo).resolve()
    source_environment = os.environ if environment is None else environment
    selected_environment = capture_runtime_environment(source_environment)
    visible = selected_environment["ASCEND_RT_VISIBLE_DEVICES"]
    status = _run_git(repo_path, "status", "--short", "--untracked-files=all")
    diff = _run_git(repo_path, "diff", "--binary", "HEAD")
    staged_diff = _run_git(repo_path, "diff", "--binary", "--cached", "HEAD")
    recorded_command = list(command)
    if recorded_command[:1] == ["--"]:
        recorded_command = recorded_command[1:]
    return {
        "schema": "specslo-benchmark-provenance-v1",
        "run_id": run_id,
        "mode": mode,
        "started_utc": utc_now(),
        "repo": str(repo_path),
        "git": {
            "revision": _run_git(repo_path, "rev-parse", "HEAD"),
            "branch": _run_git(repo_path, "branch", "--show-current"),
            "dirty": bool(status),
            "status": [] if status is None or not status else status.splitlines(),
            "tracked_diff_sha256": (None if diff is None else _sha256_bytes(diff.encode("utf-8"))),
            "staged_diff_sha256": (None if staged_diff is None else _sha256_bytes(staged_diff.encode("utf-8"))),
        },
        # The device-position PA prototype can be exercised directly from a
        # dirty worktree.  Git status records an untracked path but not its
        # contents, so bind this performance-sensitive source to every run.
        "critical_source_artifacts": {
            relative_path: sha256_file(repo_path / relative_path) for relative_path in CRITICAL_SOURCE_ARTIFACTS
        },
        "cards": {
            "visible_devices_raw": visible,
            "visible_devices": parse_visible_devices(visible),
        },
        "environment": selected_environment,
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "cwd": str(Path.cwd()),
        },
        "artifacts": {
            "result_json": str(Path(result_json).resolve()),
            "log_file": str(Path(log_file).resolve()),
        },
        "command": recorded_command,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json_exclusive(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    """Write a new JSON artifact and refuse to replace an existing run."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError:
        raise
    except BaseException:
        # ``x`` created a new file before the write failed; remove only that
        # incomplete artifact.  A pre-existing path is handled above.
        output.unlink(missing_ok=True)
        raise


def finalize_provenance(
    path: str | os.PathLike[str],
    *,
    exit_code: int,
) -> dict[str, Any]:
    """Bind the completed result and log hashes to their pre-run provenance."""
    provenance_path = Path(path)
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    artifacts = payload["artifacts"]
    payload["finished_utc"] = utc_now()
    payload["exit_code"] = int(exit_code)
    payload["artifacts"] = {
        **artifacts,
        "result_json_sha256": sha256_file(artifacts["result_json"]),
        "log_file_sha256": sha256_file(artifacts["log_file"]),
    }
    _atomic_write_json(provenance_path, payload)
    return payload


def _load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Benchmark artifact must be a JSON object: {path}")
    return value


def _result_row(payload: Mapping[str, Any], artifact_name: str) -> Mapping[str, Any]:
    rows = payload.get("results")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError(f"{artifact_name} must contain exactly one benchmark result row")
    return rows[0]


def _finite_metric(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Missing or invalid {name}") from error
    if not math.isfinite(result):
        raise ValueError(f"Missing or invalid {name}")
    return result


def _paper_slo(row: Mapping[str, Any], artifact_name: str) -> Mapping[str, Any]:
    slo = row.get("slo")
    paper = slo.get("paper") if isinstance(slo, Mapping) else None
    if not isinstance(paper, Mapping):
        raise ValueError(f"{artifact_name} does not contain results[0].slo.paper")
    return paper


def _goodput_uses_e2e_denominator(
    paper: Mapping[str, Any],
    row: Mapping[str, Any],
) -> bool:
    """Check that the published paper Goodput is bound to measured E2E time."""

    try:
        elapsed = float(row["e2e_elapsed_seconds"])
        tokens = int(paper["goodput_tokens"])
        reported = float(paper["goodput_tokens_per_e2e_second"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        math.isfinite(elapsed)
        and elapsed > 0
        and tokens >= 0
        and math.isfinite(reported)
        and math.isclose(reported, tokens / elapsed, rel_tol=1e-9, abs_tol=1e-12)
    )


def _runtime_environment_comparison(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    candidate_environment = candidate.get("runtime_environment")
    baseline_environment = baseline.get("runtime_environment")
    if not isinstance(candidate_environment, Mapping):
        candidate_environment = {}
    if not isinstance(baseline_environment, Mapping):
        baseline_environment = {}
    comparisons = {
        name: {
            "candidate": candidate_environment.get(name),
            "baseline": baseline_environment.get(name),
            "match": (
                name in candidate_environment
                and name in baseline_environment
                and candidate_environment.get(name) is not None
                and candidate_environment.get(name) == baseline_environment.get(name)
            ),
        }
        for name in SHARED_PERFORMANCE_ENVIRONMENT
    }
    candidate_missing = [name for name in CRITICAL_ENVIRONMENT if name not in candidate_environment]
    baseline_missing = [name for name in CRITICAL_ENVIRONMENT if name not in baseline_environment]
    return comparisons, candidate_missing, baseline_missing


def evaluate_gate(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    candidate_path: str,
    baseline_path: str,
    goodput_multiplier: float = 1.3,
    min_attainment: float = 0.8,
    expected_draft_tp: int | None = 1,
    expected_target_tp: int | None = 3,
    expected_baseline_tp: int | None = 4,
    expected_gamma: int | None = 4,
) -> dict[str, Any]:
    """Evaluate the paper Goodput and attainment gates with fairness checks."""
    if not math.isfinite(goodput_multiplier) or goodput_multiplier <= 0:
        raise ValueError("goodput_multiplier must be finite and positive")
    if not math.isfinite(min_attainment) or not 0 <= min_attainment <= 1:
        raise ValueError("min_attainment must be finite and in [0, 1]")

    candidate_row = _result_row(candidate, "candidate")
    baseline_row = _result_row(baseline, "baseline")
    candidate_paper = _paper_slo(candidate_row, "candidate")
    baseline_paper = _paper_slo(baseline_row, "baseline")
    candidate_goodput = _finite_metric(
        candidate_paper.get("goodput_tokens_per_e2e_second"),
        "candidate paper Goodput",
    )
    baseline_goodput = _finite_metric(
        baseline_paper.get("goodput_tokens_per_e2e_second"),
        "baseline paper Goodput",
    )
    attainment = _finite_metric(
        candidate_paper.get("attainment"),
        "candidate paper attainment",
    )
    baseline_attainment = _finite_metric(
        baseline_paper.get("attainment"),
        "baseline paper attainment",
    )
    if baseline_goodput <= 0:
        raise ValueError("baseline paper Goodput must be positive")
    if not 0 <= attainment <= 1:
        raise ValueError("candidate paper attainment must be in [0, 1]")
    if not 0 <= baseline_attainment <= 1:
        raise ValueError("baseline paper attainment must be in [0, 1]")

    environment_report, candidate_missing_environment, baseline_missing_environment = _runtime_environment_comparison(
        candidate, baseline
    )

    comparisons = {
        "prompt_token_ids_sha256": (
            candidate_row.get("prompt_token_ids_sha256"),
            baseline_row.get("prompt_token_ids_sha256"),
        ),
        "batch_size": (
            candidate_row.get("batch_size"),
            baseline_row.get("batch_size"),
        ),
        "num_prompts": (
            candidate_row.get("num_prompts"),
            baseline_row.get("num_prompts"),
        ),
        "output_tokens": (
            candidate_row.get("output_tokens"),
            baseline_row.get("output_tokens"),
        ),
        "request_manifest": (
            candidate.get("request_manifest"),
            baseline.get("request_manifest"),
        ),
        "request_manifest_sha256": (
            candidate.get("request_manifest_sha256"),
            baseline.get("request_manifest_sha256"),
        ),
        "max_tokens": (
            candidate.get("max_tokens"),
            baseline.get("max_tokens"),
        ),
        "max_model_len": (
            candidate.get("max_model_len"),
            baseline.get("max_model_len"),
        ),
        "gpu_memory_utilization": (
            candidate.get("gpu_memory_utilization"),
            baseline.get("gpu_memory_utilization"),
        ),
        "warmup_prompts": (
            candidate.get("warmup_prompts"),
            baseline.get("warmup_prompts"),
        ),
        "request_output_token_limits": (
            candidate_row.get("request_output_token_limits"),
            baseline_row.get("request_output_token_limits"),
        ),
        "warmup_output_token_limits": (
            candidate_row.get("warmup_output_token_limits"),
            baseline_row.get("warmup_output_token_limits"),
        ),
        "warmup_runs": (
            candidate.get("warmup_runs"),
            baseline.get("warmup_runs"),
        ),
        "enforce_eager": (
            candidate.get("enforce_eager"),
            baseline.get("enforce_eager"),
        ),
        "enable_prefix_caching": (
            candidate.get("enable_prefix_caching"),
            baseline.get("enable_prefix_caching"),
        ),
        "online_arrivals": (
            candidate.get("online_arrivals"),
            baseline.get("online_arrivals"),
        ),
        "warmup_excluded_from_measurement": (
            candidate.get("warmup_excluded_from_measurement"),
            baseline.get("warmup_excluded_from_measurement"),
        ),
        "e2e_timing_scope": (
            candidate_row.get("e2e_timing_scope"),
            baseline_row.get("e2e_timing_scope"),
        ),
        "target_model": (
            candidate.get("target_model"),
            baseline.get("model"),
        ),
    }
    comparison_report = {
        name: {
            "candidate": values[0],
            "baseline": values[1],
            "match": values[0] is not None and values[0] == values[1],
        }
        for name, values in comparisons.items()
    }
    topology_report = {
        "draft_tp": {
            "actual": candidate.get("draft_tensor_parallel_size"),
            "expected": expected_draft_tp,
        },
        "target_tp": {
            "actual": candidate.get("target_tensor_parallel_size"),
            "expected": expected_target_tp,
        },
        "baseline_tp": {
            "actual": baseline.get("tensor_parallel_size"),
            "expected": expected_baseline_tp,
        },
    }
    for value in topology_report.values():
        value["match"] = value["expected"] is None or value["actual"] == value["expected"]

    contract_report = {
        "candidate_backend": {
            "actual": candidate.get("backend"),
            "expected": "specslo-native-specrhythm",
        },
        "baseline_backend": {
            "actual": baseline.get("backend"),
            "expected": "vllm-ascend-target-only",
        },
        "candidate_online_arrivals": {
            "actual": candidate.get("online_arrivals"),
            "expected": True,
        },
        "baseline_online_arrivals": {
            "actual": baseline.get("online_arrivals"),
            "expected": True,
        },
        "candidate_e2e_timing_scope": {
            "actual": candidate_row.get("e2e_timing_scope"),
            "expected": ONLINE_E2E_TIMING_SCOPE,
        },
        "baseline_e2e_timing_scope": {
            "actual": baseline_row.get("e2e_timing_scope"),
            "expected": ONLINE_E2E_TIMING_SCOPE,
        },
        "candidate_warmup_excluded": {
            "actual": candidate.get("warmup_excluded_from_measurement"),
            "expected": True,
        },
        "baseline_warmup_excluded": {
            "actual": baseline.get("warmup_excluded_from_measurement"),
            "expected": True,
        },
        "gamma": {
            "actual": candidate.get("gamma"),
            "expected": expected_gamma,
        },
        "linear_full_window": {
            "actual": candidate.get("spec_rhythm_linear_full_window"),
            "expected": True,
        },
        "serial_draft": {
            "actual": candidate.get("spec_rhythm_draft_mode"),
            "expected": "serial_linear",
        },
        "candidate_slo_rows": {
            "actual": candidate_paper.get("constrained_requests"),
            "expected": candidate_row.get("num_prompts"),
        },
        "baseline_slo_rows": {
            "actual": baseline_paper.get("constrained_requests"),
            "expected": baseline_row.get("num_prompts"),
        },
        "candidate_paper_tpot_definition": {
            "actual": candidate_paper.get("tpot_definition"),
            "expected": PAPER_TPOT_DEFINITION,
        },
        "baseline_paper_tpot_definition": {
            "actual": baseline_paper.get("tpot_definition"),
            "expected": PAPER_TPOT_DEFINITION,
        },
        "candidate_paper_goodput_definition": {
            "actual": candidate_paper.get("goodput_definition"),
            "expected": PAPER_GOODPUT_DEFINITION,
        },
        "baseline_paper_goodput_definition": {
            "actual": baseline_paper.get("goodput_definition"),
            "expected": PAPER_GOODPUT_DEFINITION,
        },
        "candidate_paper_goodput_e2e_consistent": {
            "actual": _goodput_uses_e2e_denominator(
                candidate_paper,
                candidate_row,
            ),
            "expected": True,
        },
        "baseline_paper_goodput_e2e_consistent": {
            "actual": _goodput_uses_e2e_denominator(
                baseline_paper,
                baseline_row,
            ),
            "expected": True,
        },
        "candidate_missing_paper_timings": {
            "actual": candidate_paper.get("missing_timing_requests"),
            "expected": 0,
        },
        "baseline_missing_paper_timings": {
            "actual": baseline_paper.get("missing_timing_requests"),
            "expected": 0,
        },
        "candidate_missing_runtime_environment": {
            "actual": candidate_missing_environment,
            "expected": [],
        },
        "baseline_missing_runtime_environment": {
            "actual": baseline_missing_environment,
            "expected": [],
        },
        "candidate_request_limit_rows": {
            "actual": len(candidate_row.get("request_output_token_limits", ())),
            "expected": candidate_row.get("num_prompts"),
        },
        "baseline_request_limit_rows": {
            "actual": len(baseline_row.get("request_output_token_limits", ())),
            "expected": baseline_row.get("num_prompts"),
        },
        "candidate_warmup_limit_rows": {
            "actual": len(candidate_row.get("warmup_output_token_limits", ())),
            "expected": candidate.get("warmup_prompts"),
        },
        "baseline_warmup_limit_rows": {
            "actual": len(baseline_row.get("warmup_output_token_limits", ())),
            "expected": baseline.get("warmup_prompts"),
        },
    }
    if candidate.get("enable_mc2") is True:
        replay_coverage = candidate_row.get("mc2_measured_replay_coverage")
        raw_replay_targets = replay_coverage.get("target_ranks", ()) if isinstance(replay_coverage, Mapping) else ()
        replay_targets = (
            raw_replay_targets
            if isinstance(raw_replay_targets, Sequence) and not isinstance(raw_replay_targets, (str, bytes))
            else ()
        )
        contract_report.update(
            {
                "candidate_mc2_measured_replay_coverage": {
                    "actual": (replay_coverage.get("passed") if isinstance(replay_coverage, Mapping) else None),
                    "expected": True,
                },
                "candidate_mc2_static_routes_consistent": {
                    "actual": (
                        replay_coverage.get("static_routes_consistent")
                        if isinstance(replay_coverage, Mapping)
                        else None
                    ),
                    "expected": True,
                },
                "candidate_mc2_target_rank_coverage": {
                    "actual": (
                        len(replay_targets) == expected_target_tp
                        if expected_target_tp is not None
                        else bool(replay_targets)
                    )
                    and all(
                        int(item.get("qualified_mc2_replay_count", 0)) > 0
                        and int(item.get("measured_mc2_fallback", -1)) == 0
                        and int(item.get("measured_mc2_exception", -1)) == 0
                        for item in replay_targets
                        if isinstance(item, Mapping)
                    )
                    and all(isinstance(item, Mapping) for item in replay_targets),
                    "expected": True,
                },
            }
        )
    for value in contract_report.values():
        value["match"] = value["expected"] is None or value["actual"] == value["expected"]

    comparable = (
        all(item["match"] for item in comparison_report.values())
        and all(item["match"] for item in environment_report.values())
        and all(item["match"] for item in topology_report.values())
        and all(item["match"] for item in contract_report.values())
    )
    required_goodput = baseline_goodput * goodput_multiplier
    speedup = candidate_goodput / baseline_goodput
    goodput_pass = speedup >= goodput_multiplier
    attainment_pass = attainment >= min_attainment
    reasons = []
    if not comparable:
        reasons.append("candidate and baseline workload/topology provenance do not match")
    if not goodput_pass:
        reasons.append(f"paper Goodput speedup {speedup:.6f}x is below {goodput_multiplier:.6f}x")
    if not attainment_pass:
        reasons.append(f"paper attainment {attainment:.6%} is below {min_attainment:.6%}")

    return {
        "schema": "specslo-benchmark-gate-v1",
        "generated_utc": utc_now(),
        "candidate": {
            "path": str(Path(candidate_path).resolve()),
            "sha256": sha256_file(candidate_path),
            "paper_goodput_tokens_per_e2e_second": candidate_goodput,
            "paper_attainment": attainment,
            "attained_requests": candidate_paper.get("attained_requests"),
            "constrained_requests": candidate_paper.get("constrained_requests"),
            "e2e_elapsed_seconds": candidate_row.get("e2e_elapsed_seconds"),
        },
        "baseline": {
            "path": str(Path(baseline_path).resolve()),
            "sha256": sha256_file(baseline_path),
            "paper_goodput_tokens_per_e2e_second": baseline_goodput,
            "paper_attainment": baseline_attainment,
            "attained_requests": baseline_paper.get("attained_requests"),
            "constrained_requests": baseline_paper.get("constrained_requests"),
            "e2e_elapsed_seconds": baseline_row.get(
                "e2e_elapsed_seconds",
                baseline_row.get("elapsed_seconds"),
            ),
        },
        "thresholds": {
            "goodput_multiplier": goodput_multiplier,
            "required_goodput_tokens_per_e2e_second": required_goodput,
            "min_attainment": min_attainment,
        },
        "observed": {
            "goodput_speedup": speedup,
            "goodput_pass": goodput_pass,
            "attainment_pass": attainment_pass,
        },
        "comparability": {
            "passed": comparable,
            "workload": comparison_report,
            "environment": environment_report,
            "topology": topology_report,
            "contract": contract_report,
        },
        "passed": comparable and goodput_pass and attainment_pass,
        "reasons": reasons,
    }


def _provenance_command(args: argparse.Namespace) -> int:
    payload = collect_provenance(
        repo=args.repo,
        mode=args.mode,
        run_id=args.run_id,
        result_json=args.result_json,
        log_file=args.log_file,
        command=args.command,
    )
    write_json_exclusive(args.output, payload)
    return 0


def _finalize_command(args: argparse.Namespace) -> int:
    finalize_provenance(args.provenance, exit_code=args.exit_code)
    return 0


def _gate_command(args: argparse.Namespace) -> int:
    candidate = _load_json(args.candidate)
    baseline = _load_json(args.baseline)
    report = evaluate_gate(
        candidate,
        baseline,
        candidate_path=args.candidate,
        baseline_path=args.baseline,
        goodput_multiplier=args.goodput_multiplier,
        min_attainment=args.min_attainment,
        expected_draft_tp=args.expected_draft_tp,
        expected_target_tp=args.expected_target_tp,
        expected_baseline_tp=args.expected_baseline_tp,
        expected_gamma=args.expected_gamma,
    )
    write_json_exclusive(args.output, report)
    print(
        "SpecSLO gate: "
        f"speedup={report['observed']['goodput_speedup']:.6f}x "
        f"(required={report['thresholds']['goodput_multiplier']:.6f}x), "
        f"attainment={report['candidate']['paper_attainment']:.6%} "
        f"(required={report['thresholds']['min_attainment']:.6%}), "
        f"comparable={report['comparability']['passed']}, "
        f"passed={report['passed']}"
    )
    for reason in report["reasons"]:
        print(f"- {reason}")
    return int(args.require_pass and not report["passed"])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)

    provenance = commands.add_parser("provenance", help="write immutable pre-run provenance")
    provenance.add_argument("--repo", required=True)
    provenance.add_argument("--mode", required=True)
    provenance.add_argument("--run-id", required=True)
    provenance.add_argument("--result-json", required=True)
    provenance.add_argument("--log-file", required=True)
    provenance.add_argument("--output", required=True)
    provenance.add_argument("command", nargs=argparse.REMAINDER)
    provenance.set_defaults(callback=_provenance_command)

    finalize = commands.add_parser("finalize", help="record exit status and artifact hashes")
    finalize.add_argument("--provenance", required=True)
    finalize.add_argument("--exit-code", required=True, type=int)
    finalize.set_defaults(callback=_finalize_command)

    gate = commands.add_parser("gate", help="compare candidate paper metrics with a baseline")
    gate.add_argument("--candidate", required=True)
    gate.add_argument("--baseline", required=True)
    gate.add_argument("--output", required=True)
    gate.add_argument("--goodput-multiplier", type=float, default=1.3)
    gate.add_argument("--min-attainment", type=float, default=0.8)
    gate.add_argument("--expected-draft-tp", type=int, default=1)
    gate.add_argument("--expected-target-tp", type=int, default=3)
    gate.add_argument("--expected-baseline-tp", type=int, default=4)
    gate.add_argument("--expected-gamma", type=int, default=4)
    gate.add_argument("--require-pass", action="store_true")
    gate.set_defaults(callback=_gate_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return int(args.callback(args))


if __name__ == "__main__":
    raise SystemExit(main())
