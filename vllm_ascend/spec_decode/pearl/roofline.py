# SPDX-License-Identifier: Apache-2.0
"""Validated, identity-bound SpecSLO profiling tables and legacy mappings."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STRICT_ROOFLINE_SCHEMA_VERSION = 3
TREE_FIA_SPARSE_MODE = 1
TREE_FIA_INNER_PRECISE = 1
_ROOFLINE_SOURCE_FILES = {
    "native_engine_source_sha256": "native_engine.py",
    "native_model_source_sha256": "native_model.py",
    "native_graph_source_sha256": "native_graph.py",
    "tree_source_sha256": "tree.py",
}


def runtime_source_fingerprints() -> dict[str, str]:
    """Hash every local implementation file that defines measured target work."""
    directory = Path(__file__).resolve().parent
    return {
        field: hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        for field, filename in _ROOFLINE_SOURCE_FILES.items()
    }


def attention_backend_identity(metadata: Any) -> str:
    """Identify the prepared attention route, not the requested graph flag."""
    if metadata.use_fused_infer_attention:
        return "fused_infer_attention_tree_v1" if metadata.tree_attention else "fused_infer_attention_causal_v1"
    if metadata.attention_mask is not None:
        return "dense_sdpa_tree_v1"
    return "paged_attention_v1"


def validate_attention_provenance(metadata: Mapping[str, Any]) -> None:
    backends = metadata.get("verification_attention_backends")
    supported = {
        "fused_infer_attention_causal_v1",
        "fused_infer_attention_tree_v1",
        "dense_sdpa_tree_v1",
    }
    if (
        not isinstance(backends, list)
        or not backends
        or any(not isinstance(value, str) or value not in supported for value in backends)
        or backends != sorted(set(backends))
    ):
        raise ValueError("Strict roofline requires sorted supported verification_attention_backends")
    if metadata.get("ar_attention_backend") != "paged_attention_v1":
        raise ValueError("Strict roofline requires ar_attention_backend=paged_attention_v1")
    if metadata.get("ar_comparator") != "standard_decode_full_active_batch":
        raise ValueError(
            "Strict roofline requires ar_comparator=standard_decode_full_active_batch"
        )
    distribution_protocol = metadata.get("candidate_distribution_protocol")
    if distribution_protocol not in (None, "all_canonical_histograms_until_first_violation"):
        raise ValueError(
            "Strict roofline contains an unsupported candidate-distribution protocol"
        )
    for field in _ROOFLINE_SOURCE_FILES:
        if not isinstance(metadata.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", metadata[field]):
            raise ValueError(f"Strict roofline requires a recorded {field}")
    if metadata.get("tree_fia_sparse_mode") != TREE_FIA_SPARSE_MODE:
        raise ValueError(f"Strict roofline requires tree_fia_sparse_mode={TREE_FIA_SPARSE_MODE}")
    if metadata.get("tree_fia_inner_precise") != TREE_FIA_INNER_PRECISE:
        raise ValueError(f"Strict roofline requires tree_fia_inner_precise={TREE_FIA_INNER_PRECISE}")
    _positive_integer(metadata.get("max_model_len"), "max_model_len")
    _positive_integer(metadata.get("tree_width"), "tree_width")
    _positive_integer(metadata.get("tree_depth"), "tree_depth")


@dataclass(frozen=True)
class ProfiledRoofline(Mapping[str, int]):
    """Strict measured table. Missing buckets never imply a measured fallback.

    Identity validation establishes compatibility with the recorded experiment,
    not the truth of its measurements or an absolute latency guarantee. Evidence
    retains the exact sampled contexts; interpolation inside a bucket is still
    the profiling protocol's responsibility.
    """

    entries: Mapping[str, int]
    metadata: Mapping[str, Any]
    evidence: tuple[Mapping[str, Any], ...]
    context_bucket_size: int = 512
    strict: bool = True

    def __getitem__(self, key: str) -> int:
        return self.entries[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def validate_attention_backend(self, actual_backend: str, *, target_only: bool = False) -> None:
        """Source SHA is audit provenance; backend compatibility is mandatory.

        A measured zero candidate budget is an explicit target-only cycle.  It
        must use the AR backend recorded by the same profiler document rather
        than pretending that a packed-tree operator executed with no nodes.
        """
        expected = (
            {self.metadata["ar_attention_backend"]}
            if target_only
            else set(self.metadata["verification_attention_backends"])
        )
        if actual_backend not in expected:
            raise ValueError(
                f"SpecSLO roofline attention backends {sorted(expected)!r} "
                f"does not match actual target metadata route {actual_backend!r}"
            )

    def lookup(self, batch_size: int, context_len: int) -> int:
        key = self.lookup_key(batch_size, context_len)
        if key not in self.entries:
            raise ValueError(
                f"Unprofiled SpecSLO roofline key {key!r}; strict measured tables "
                "cannot use a gamma-derived fallback. Profile this batch/context first."
            )
        return self.entries[key]

    def lookup_key(self, batch_size: int, context_len: int) -> str:
        return (
            f"{max(1, int(batch_size))}:"
            f"{max(1, math.ceil(max(1, int(context_len)) / self.context_bucket_size))}"
        )

    def lookup_optional(self, batch_size: int, context_len: int) -> int | None:
        """Return measured B or None; zero remains a real measured value."""
        return self.entries.get(self.lookup_key(batch_size, context_len))

    def covers_execution(self, batch_size: int, context_len: int, verification_requests: int) -> bool:
        key = self.lookup_key(batch_size, context_len)
        return key in self.entries and any(
            row["lookup_key"] == key and row["verification_requests"] == verification_requests
            for row in self.evidence
        )

    def validate_execution(self, batch_size: int, context_len: int, verification_requests: int) -> None:
        """Require evidence for the actual target partition, not only active B.

        A half-home sample does not establish full/merged-home or tail-batch
        latency. Context interpolation remains the declared 512-token bucket
        convention; only the concrete context points in evidence were measured.
        """
        self.lookup(batch_size, context_len)
        bucket = max(1, math.ceil(max(1, int(context_len)) / self.context_bucket_size))
        if not self.covers_execution(batch_size, context_len, verification_requests):
            raise ValueError(
                f"Unprofiled SpecSLO physical target rows: active batch={batch_size}, "
                f"context bucket={bucket}, verification_requests={verification_requests}; "
                "a different dual-batch partition cannot reuse measured coverage."
            )


def parse_roofline_argument(value: str) -> dict[str, Any]:
    """Read a JSON object or a local profile path for the existing CLI option."""
    try:
        if value.lstrip().startswith("{"):
            result = json.loads(value)
        else:
            with Path(value).open(encoding="utf-8") as stream:
                result = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read SpecSLO roofline JSON/profile: {error}") from error
    if not isinstance(result, dict):
        raise ValueError("SpecSLO roofline must be a JSON object")
    return result


def _model_identity(value: str) -> str:
    return os.path.realpath(value) if os.path.isabs(value) or os.path.exists(value) else value.rstrip("/")


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Strict roofline {name} must be a positive integer")
    return value


def validate_roofline_identity(
    profile: ProfiledRoofline,
    *,
    model: str,
    target_tp_size: int,
    enforce_eager: bool,
    max_model_len: int,
    tree_width: int,
    tree_depth: int,
) -> None:
    metadata = profile.metadata
    if _model_identity(metadata["model"]) != _model_identity(str(model)):
        raise ValueError("SpecSLO roofline model does not match the configured target model")
    if metadata["target_tensor_parallel_size"] != target_tp_size:
        raise ValueError("SpecSLO roofline target TP does not match the configured target TP")
    mode = "eager" if enforce_eager else "graph"
    if metadata["execution_mode"] != mode:
        raise ValueError(f"SpecSLO roofline execution mode does not match {mode!r}")
    if metadata["max_model_len"] != max_model_len:
        raise ValueError("SpecSLO roofline max_model_len does not match the configured FULL-mask capacity")
    if metadata["tree_width"] != tree_width or metadata["tree_depth"] != tree_depth:
        raise ValueError("SpecSLO roofline tree topology does not match the configured width/depth")
    for field, actual in runtime_source_fingerprints().items():
        if metadata[field] != actual:
            raise ValueError(
                f"SpecSLO roofline {field} does not match the running implementation; "
                "measure B again after every target/graph/tree source change"
            )


def validate_roofline_hardware(profile: Mapping[str, int] | None, actual_hardware: str) -> None:
    """Compare real runtime device identity, not a user echo of the profile."""
    if not isinstance(profile, ProfiledRoofline):
        return

    def normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(value).lower())

    if not actual_hardware or normalize(profile.metadata["hardware"]) != normalize(actual_hardware):
        raise ValueError(
            f"SpecSLO roofline hardware {profile.metadata['hardware']!r} does not "
            f"match detected device {actual_hardware!r}; no hardware compatibility was assumed."
        )


def normalize_roofline(
    value: Mapping[str, Any] | str | Path,
    *,
    model: str,
    target_tp_size: int,
    enforce_eager: bool,
    max_model_len: int,
    tree_width: int,
    tree_depth: int,
) -> Mapping[str, int]:
    """Validate full profiler documents; preserve explicitly legacy bare maps."""
    if isinstance(value, (str, Path)):
        value = parse_roofline_argument(str(value))
    if isinstance(value, ProfiledRoofline):
        validate_attention_provenance(value.metadata)
        validate_roofline_identity(
            value,
            model=model,
            target_tp_size=target_tp_size,
            enforce_eager=enforce_eager,
            max_model_len=max_model_len,
            tree_width=tree_width,
            tree_depth=tree_depth,
        )
        return value
    if not isinstance(value, Mapping):
        raise ValueError("SpecSLO roofline requires a mapping or local profile file")
    if not any(key in value for key in ("schema_version", "metadata", "roofline", "evidence")):
        # This compatibility path is deliberately not marked profiled. Its
        # default/gamma fallback remains available to existing experiments.
        entries = {str(key): int(budget) for key, budget in value.items()}
        if any(budget <= 0 for budget in entries.values()):
            raise ValueError("SpecRhythm roofline values must be positive token budgets.")
        return entries
    if value.get("schema_version") != STRICT_ROOFLINE_SCHEMA_VERSION:
        raise ValueError("Unsupported or missing strict roofline schema_version")
    metadata = value.get("metadata")
    table = value.get("roofline")
    evidence = value.get("evidence")
    if not isinstance(metadata, Mapping) or not isinstance(table, Mapping) or not table:
        raise ValueError("Strict roofline requires metadata and a non-empty roofline table")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Strict roofline requires non-empty measurement evidence")
    for key in ("model", "hardware"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"Strict roofline metadata requires {key}")
    _positive_integer(metadata.get("target_tensor_parallel_size"), "target_tensor_parallel_size")
    if (
        metadata.get("execution_mode") not in ("eager", "graph")
        or metadata.get("measurement_scope") != "target_forward"
    ):
        raise ValueError("Strict roofline requires eager/graph target_forward measurements")
    if metadata.get("verification_layout") != "packed_tree":
        raise ValueError("Strict roofline requires an explicit supported packed_tree verification layout")
    validate_attention_provenance(metadata)
    bucket_size = _positive_integer(value.get("context_bucket_size"), "context_bucket_size")
    if bucket_size != 512:
        raise ValueError("Native SpecSLO currently requires 512-token roofline context buckets")
    entries = {}
    for key, budget in table.items():
        if not isinstance(key, str) or re.fullmatch(r"[1-9]\d*:[1-9]\d*", key) is None:
            raise ValueError("Strict roofline entries require exact batch:context keys")
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError("Strict roofline candidate budget must be a non-negative integer")
        entries[key] = budget
    if set(entries).intersection(value.get("unqualified_lookup_keys", ())):
        raise ValueError("An unqualified profile bucket cannot be used as a measured roof")
    fallback_keys = set(value.get("target_only_fallback_lookup_keys", ()))
    if fallback_keys != {key for key, budget in entries.items() if budget == 0}:
        raise ValueError("Strict roofline target-only fallback keys must exactly identify zero budgets")
    roofs: dict[str, list[int]] = {}
    for row in evidence:
        if not isinstance(row, Mapping):
            raise ValueError("Strict roofline evidence rows must be objects")
        batch = _positive_integer(row.get("batch_size"), "evidence batch_size")
        context = _positive_integer(row.get("context_len"), "evidence context_len")
        requests = _positive_integer(row.get("verification_requests"), "evidence verification_requests")
        if row.get("ar_requests") != batch:
            raise ValueError("Strict roofline evidence requires full-active standard AR comparison")
        key = f"{batch}:{math.ceil(context / bucket_size)}"
        if row.get("lookup_key") != key or requests > batch:
            raise ValueError("Strict roofline evidence has inconsistent batch/context identity")
        measured_roof = row.get("measured_roof")
        if isinstance(measured_roof, bool) or not isinstance(measured_roof, int) or measured_roof < 0:
            raise ValueError("Strict roofline evidence requires a non-negative measured roof")
        tree_capacity = metadata["tree_width"] * metadata["tree_depth"]
        if measured_roof > requests * tree_capacity:
            raise ValueError("Strict roofline evidence exceeds the configured tree topology capacity")
        for candidate in row.get("candidate_sweep", ()):
            for shape in candidate.get("shapes", ()):
                if shape.get("attention_backend") not in metadata["verification_attention_backends"]:
                    raise ValueError("Strict roofline evidence contains an unbound verification attention backend")
        roofs.setdefault(key, []).append(measured_roof)
    if any(key not in roofs or budget > min(roofs[key]) for key, budget in entries.items()):
        raise ValueError("Strict roofline budget lacks supporting evidence or exceeds its measured roof")
    profile = ProfiledRoofline(dict(entries), dict(metadata), tuple(dict(row) for row in evidence), bucket_size)
    validate_roofline_identity(
        profile,
        model=model,
        target_tp_size=target_tp_size,
        enforce_eager=enforce_eager,
        max_model_len=max_model_len,
        tree_width=tree_width,
        tree_depth=tree_depth,
    )
    return profile
