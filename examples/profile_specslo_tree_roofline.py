# SPDX-License-Identifier: Apache-2.0
"""Build a SpecSLO packed-tree roofline from actual target-forward samples.

Example (no model execution occurs in the aggregation command)::

    python examples/profile_specslo_tree_roofline.py \
        --measurements measured_tree_samples.json --epsilon-relative 0.10 \
        --output measured_tree_roofline.json

Input schema::

    {
      "metadata": {
        "model": "/data/shared-models/Qwen3-32B",
        "target_tensor_parallel_size": 3,
        "execution_mode": "graph",
        "hardware": "Ascend910B3",
        "measurement_scope": "target_forward",
        "verification_attention_backends": ["fused_infer_attention_causal_v1", "fused_infer_attention_tree_v1"],
        "ar_attention_backend": "paged_attention_v1",
        "ar_comparator": "standard_decode_full_active_batch",
        "native_model_source_sha256": "actual 64-character source SHA256",
        "native_graph_source_sha256": "actual 64-character source SHA256",
        "tree_source_sha256": "actual 64-character source SHA256",
        "tree_fia_sparse_mode": 1,
        "tree_fia_inner_precise": 1,
        "max_model_len": 4096,
        "tree_width": 2,
        "tree_depth": 4
      },
      "measurements": [
        {"kind": "ar", "batch_size": 8, "verification_requests": 8,
         "attention_backend": "paged_attention_v1",
         "context_len": 512, "candidate_counts": [0, 0, 0, 0, 0, 0, 0, 0],
         "physical_query_tokens": 8, "latency_ms": [10.0, 10.1, 10.0],
         "warmup_iterations": 2, "timed_graph_captures": 0, "timed_graph_replays": 3},
        {"kind": "packed_tree", "batch_size": 8, "verification_requests": 4,
         "attention_backend": "fused_infer_attention_tree_v1",
         "context_len": 512, "candidate_counts": [1, 1, 1, 1],
         "physical_query_tokens": 8, "latency_ms": [10.5, 10.4, 10.6],
         "warmup_iterations": 2, "timed_graph_captures": 0, "timed_graph_replays": 3}
      ]
    }

The numbers above illustrate the schema; they are not a measured device table.
``batch_size`` is the active request count used as the scheduler lookup key.
``verification_requests`` is the physical forward request count. The standard
AR control always uses all ``batch_size`` active requests; a packed tree uses
the request count in the logical slot being verified. This matches paper
section 5.3: compare SpecRhythm verification against standard batched decoding
at the same active batch size, not against half-sized AR.
Warmup/capture is excluded only for this offline steady-state capacity probe;
this file does not define end-to-end benchmark timing or SLO measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


def measure_target_forward(
    forward: Callable[[], Any],
    synchronize: Callable[[], None],
    *,
    iterations: int = 10,
    warmup_iterations: int = 2,
    graph_capture_count: Callable[[], int] | None = None,
    graph_replay_count: Callable[[], int] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Measure one stable AR or packed-tree shape, including device completion.

    The callback must reuse the same model/inputs/cache shape. Pass target-role
    synchronization (and aggregate the slowest target TP rank externally), not
    a draft/target world barrier that would charge peer waiting to target work.
    Graph callers must provide the real runner's capture/replay counters. A capture in
    the timed samples invalidates the steady-state probe instead of silently
    entering the roofline table. All measured samples, not the best run, are
    returned for the aggregator.
    """

    if iterations <= 0 or warmup_iterations < 0:
        raise ValueError("Measurement iterations must be positive and warmup non-negative.")
    for _ in range(warmup_iterations):
        forward()
        synchronize()
    before_captures = graph_capture_count() if graph_capture_count is not None else 0
    before_replays = graph_replay_count() if graph_replay_count is not None else 0
    samples = []
    for _ in range(iterations):
        synchronize()
        started = clock()
        forward()
        synchronize()
        elapsed_ms = (clock() - started) * 1000.0
        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0.0:
            raise ValueError("Target-forward measurements must have finite positive durations.")
        samples.append(elapsed_ms)
    after_captures = graph_capture_count() if graph_capture_count is not None else 0
    if after_captures != before_captures:
        raise ValueError("Graph capture occurred inside the timed steady-state samples.")
    timed_replays = graph_replay_count() - before_replays if graph_replay_count is not None else None
    if timed_replays is not None and timed_replays < iterations:
        raise ValueError("Not every timed target forward performed graph replay.")
    return {
        "latency_ms": samples,
        "warmup_iterations": int(warmup_iterations),
        "timed_graph_captures": 0 if graph_capture_count is not None else None,
        "timed_graph_replays": timed_replays,
    }


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Linear quantile using all repeated samples, not the minimum latency."""

    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _canonical_count_distributions(requests: int, maximum: int, budget: int) -> set[tuple[int, ...]]:
    """Return every permutation-equivalent positive count histogram."""
    result: set[tuple[int, ...]] = set()

    def visit(prefix: list[int], minimum: int, slots: int, remaining: int) -> None:
        if slots == 0:
            if remaining == 0:
                result.add(tuple(prefix))
            return
        lower = max(minimum, remaining - maximum * (slots - 1))
        upper = min(maximum, remaining // slots)
        for value in range(lower, upper + 1):
            prefix.append(value)
            visit(prefix, value, slots - 1, remaining - value)
            prefix.pop()

    visit([], 1, requests, budget)
    return result


def build_roofline_table(
    document: Mapping[str, Any],
    *,
    epsilon_relative: float = 0.10,
    latency_percentile: float = 95.0,
    minimum_samples: int = 3,
    context_bucket_size: int = 512,
) -> dict[str, Any]:
    """Validate samples and derive conservative model-specific B_roof entries.

    For every physical batch/context, budgets are scanned in increasing order
    until the first latency-limit violation. A later lucky/shape-specific fast
    point cannot bypass that inflection. For multiple context values or tree
    distributions in one lookup bucket, the most conservative measured bound
    wins. Missing comparisons or an unmeasured bucket never create an entry.
    """

    if not math.isfinite(epsilon_relative) or epsilon_relative < 0.0:
        raise ValueError("epsilon_relative must be finite and non-negative.")
    if not math.isfinite(latency_percentile) or not 0.0 <= latency_percentile <= 100.0:
        raise ValueError("latency_percentile must be in [0, 100].")
    _positive_int(minimum_samples, "minimum_samples")
    _positive_int(context_bucket_size, "context_bucket_size")
    if not isinstance(document, Mapping) or not isinstance(document.get("metadata"), Mapping):
        raise ValueError("Roofline input requires an object with measurement metadata.")
    if "status" in document and document["status"] != "complete":
        raise ValueError("An interrupted or failed collector run is not a complete roofline measurement.")
    metadata = dict(document["metadata"])
    for key in ("model", "hardware"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"Measurement metadata requires a non-empty {key}.")
    _positive_int(metadata.get("target_tensor_parallel_size"), "target_tensor_parallel_size")
    if metadata.get("execution_mode") not in ("eager", "graph"):
        raise ValueError("execution_mode must identify eager or graph measurements.")
    if metadata.get("measurement_scope") != "target_forward":
        raise ValueError("Roofline samples must measure target_forward, not end-to-end or cycle latency.")
    if metadata.get("verification_layout", "packed_tree") != "packed_tree":
        raise ValueError("This profiler only accepts the packed_tree verification layout.")
    # The AR/packed_tree row schema below unambiguously identifies this
    # layout, including legacy raw sample documents lacking this field.
    metadata["verification_layout"] = "packed_tree"
    from vllm_ascend.spec_decode.pearl.roofline import validate_attention_provenance

    validate_attention_provenance(metadata)
    tree_capacity = metadata["tree_width"] * metadata["tree_depth"]
    rows = document.get("measurements")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Roofline generation requires actual non-empty measurements.")
    # Standard batched decoding advances every active request. SpecRhythm
    # verifies one logical slot, so the two forwards intentionally have
    # different physical request counts while sharing active batch/context.
    ar_samples: dict[tuple[int, int], list[float]] = defaultdict(list)
    tree_samples: dict[tuple[int, int, int], dict[tuple[tuple[int, ...], str], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Each measurement must be a JSON object.")
        batch = _positive_int(row.get("batch_size"), "batch_size")
        requests = _positive_int(row.get("verification_requests"), "verification_requests")
        context = _positive_int(row.get("context_len"), "context_len")
        physical = _positive_int(row.get("physical_query_tokens"), "physical_query_tokens")
        if requests > batch:
            raise ValueError("Physical verification requests cannot exceed the active request batch.")
        counts = row.get("candidate_counts")
        if not isinstance(counts, list) or len(counts) != requests:
            raise ValueError("candidate_counts must describe every physical verification request.")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
            raise ValueError("Candidate counts must be non-negative integers.")
        if any(value > tree_capacity for value in counts):
            raise ValueError("Candidate counts cannot exceed the measured tree_width*tree_depth capacity.")
        kind = row.get("kind")
        if kind == "ar":
            if requests != batch or any(counts) or physical != batch:
                raise ValueError(
                    "Standard AR samples must contain one query for every active batch request."
                )
        elif kind == "packed_tree":
            if not all(counts) or physical != requests + sum(counts):
                raise ValueError(
                    "Packed-tree physical queries must equal roots plus actual candidates, without padding."
                )
        else:
            raise ValueError("Measurement kind must be ar or packed_tree.")
        expected_backends = (
            {metadata["ar_attention_backend"]}
            if kind == "ar"
            else set(metadata["verification_attention_backends"])
        )
        if row.get("attention_backend") not in expected_backends:
            raise ValueError("Every roofline sample must record the actual matching attention_backend")
        warmup = row.get("warmup_iterations")
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
            raise ValueError("Every sample must state its excluded warmup_iterations.")
        if metadata["execution_mode"] == "graph" and row.get("timed_graph_captures") != 0:
            raise ValueError("Graph samples require proof of zero timed_graph_captures.")
        timings = row.get("latency_ms")
        if not isinstance(timings, list) or not timings:
            raise ValueError("Measurements must contain repeated latency_ms samples.")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0
            for value in timings
        ):
            raise ValueError("Measured latencies must be finite and strictly positive.")
        if metadata["execution_mode"] == "graph":
            replays = row.get("timed_graph_replays")
            if isinstance(replays, bool) or not isinstance(replays, int) or replays < len(timings):
                raise ValueError("Graph samples must show timed_graph_replays for every measured forward.")
        if kind == "ar":
            ar_samples[(batch, context)].extend(float(value) for value in timings)
        else:
            tree_samples[(batch, requests, context)][
                (tuple(counts), str(row["attention_backend"]))
            ].extend(float(value) for value in timings)

    bucket_roofs: dict[str, list[int]] = defaultdict(list)
    evidence = []
    for (batch, requests, context), shapes in sorted(tree_samples.items()):
        ar = ar_samples.get((batch, context))
        if ar is None:
            raise ValueError(f"Missing full-active AR measurement for batch={batch}, context={context}.")
        if len(ar) < minimum_samples or any(len(values) < minimum_samples for values in shapes.values()):
            raise ValueError(f"Every measured shape needs at least {minimum_samples} latency samples.")
        ar_latency = _percentile(ar, latency_percentile)
        threshold = ar_latency * (1.0 + epsilon_relative)
        by_budget: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for (counts, backend), timings in sorted(shapes.items()):
            by_budget[sum(counts)].append(
                {
                    "candidate_counts": list(counts),
                    "attention_backend": backend,
                    "latency_ms": _percentile(timings, latency_percentile),
                    "samples": len(timings),
                }
            )
        if not by_budget:
            raise ValueError(f"Missing packed-tree sweep for batch={batch}, rows={requests}, context={context}.")
        roof = 0
        stopped = False
        candidates = []
        expected_budget = requests
        for budget, shapes_for_budget in sorted(by_budget.items()):
            worst_latency = max(shape["latency_ms"] for shape in shapes_for_budget)
            within_limit = worst_latency <= threshold
            if not stopped and metadata.get("candidate_distribution_protocol") == (
                "all_canonical_histograms_until_first_violation"
            ):
                if budget != expected_budget:
                    raise ValueError(
                        f"Packed-tree sweep skipped candidate budget {expected_budget} before {budget}."
                    )
                # A passing budget establishes a global B only when every
                # possible request-count histogram passed.  At the first
                # failing budget one concrete counterexample is sufficient.
                if within_limit:
                    observed = {
                        tuple(sorted(int(value) for value in shape["candidate_counts"]))
                        for shape in shapes_for_budget
                    }
                    expected = _canonical_count_distributions(requests, tree_capacity, budget)
                    if observed != expected:
                        raise ValueError(
                            "A passing candidate budget lacks all canonical per-request count distributions"
                        )
                expected_budget += 1
            candidates.append(
                {
                    "candidate_tokens": budget,
                    "worst_shape_latency_ms": worst_latency,
                    "relative_to_ar": worst_latency / ar_latency,
                    "within_limit": within_limit,
                    "reachable_before_first_violation": not stopped,
                    "shapes": shapes_for_budget,
                }
            )
            if not stopped and within_limit:
                roof = budget
            elif not within_limit:
                stopped = True
        key = f"{batch}:{math.ceil(context / context_bucket_size)}"
        bucket_roofs[key].append(roof)
        evidence.append(
            {
                "lookup_key": key,
                "batch_size": batch,
                "ar_requests": batch,
                "verification_requests": requests,
                "context_len": context,
                "ar_latency_ms": ar_latency,
                "verification_limit_ms": threshold,
                "measured_roof": roof,
                "candidate_sweep": candidates,
            }
        )
    # Zero is a measured result, not missing evidence.  It means that even the
    # minimum packed-tree shape exceeded the AR envelope and the runtime must
    # execute one target-only cycle for this exact lookup key.  Omitting the
    # key would make production indistinguishable from an incomplete profile.
    roofline = {key: min(values) for key, values in bucket_roofs.items()}
    fallback_keys = sorted(key for key, budget in roofline.items() if budget == 0)
    return {
        "schema_version": 3,
        "metadata": metadata,
        "epsilon_relative": epsilon_relative,
        "latency_percentile": latency_percentile,
        "minimum_samples": minimum_samples,
        "context_bucket_size": context_bucket_size,
        "roofline": roofline,
        "unqualified_lookup_keys": [],
        "target_only_fallback_lookup_keys": fallback_keys,
        "evidence": evidence,
        "usage": (
            "Use roofline only for this exact model/TP/hardware/mode. A zero entry requires the measured "
            "target-only fallback; missing keys remain unprofiled and cannot use a gamma-derived fallback. "
            "Context bucket coverage consists "
            "only of the explicit measured context points in evidence."
        ),
    }


def merge_measurement_documents(documents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Combine measured batch coverage, never synthesize an unmeasured key.

    Require identical complete experiment metadata (including backend, model
    source, finite guards and timing scope). Each original row and sample is
    retained for the ordinary conservative aggregation below.
    """
    if not documents:
        raise ValueError("At least one measurement document is required")
    first = documents[0]
    if len(documents) == 1:
        return dict(first)
    if not isinstance(first.get("metadata"), Mapping):
        raise ValueError("Merged measurements require metadata")
    rows = []
    for document in documents:
        if document.get("status") != "complete":
            raise ValueError("Every merged collector run must be complete")
        if document.get("metadata") != first["metadata"]:
            raise ValueError("Cannot merge measurements with different experiment metadata")
        if not isinstance(document.get("measurements"), list):
            raise ValueError("Merged measurements require actual measurement rows")
        rows.extend(document["measurements"])
    return {"status": "complete", "metadata": dict(first["metadata"]), "measurements": rows}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--measurements", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epsilon-relative", type=float, default=0.10)
    parser.add_argument("--latency-percentile", type=float, default=95.0)
    parser.add_argument("--minimum-samples", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        result = build_roofline_table(
            merge_measurement_documents([json.loads(path.read_text()) for path in args.measurements]),
            epsilon_relative=args.epsilon_relative,
            latency_percentile=args.latency_percentile,
            minimum_samples=args.minimum_samples,
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
