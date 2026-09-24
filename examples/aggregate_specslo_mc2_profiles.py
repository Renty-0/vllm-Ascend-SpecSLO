# SPDX-License-Identifier: Apache-2.0
"""Aggregate real-layer MC2 measurements into a conservative profile.

This command performs no model execution and does not access an NPU.  Every
input must be a complete ``measure_specslo_mc2.py`` result for the same exact
shape set, collected from a different real-model input directory::

    python examples/aggregate_specslo_mc2_profiles.py \
        --input layer-0.json layer-31.json layer-63.json \
        --output qualified-envelope.json

Repeat ``--input`` to safely union disjoint exact-shape envelopes, such as the
attention and FFN down projections consumed by one production model runner::

    python examples/aggregate_specslo_mc2_profiles.py \
        --input attention-layer-0.json attention-layer-31.json attention-layer-63.json \
        --input down-layer-0.json down-layer-31.json down-layer-63.json \
        --output attention-and-down-envelope.json

For each exact shape, the output keeps the complete sample array from the
fastest baseline source and the complete sample array from the slowest fused
source, as ranked by the configured latency percentile.  It never pools,
duplicates, or synthesizes timing samples.  Numerical errors use the maximum
observed value across all inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from vllm_ascend.spec_decode.pearl.mc2 import (
    MC2_TP3_NATIVE_EPILOGUE_OPERATOR,
    normalize_mc2_profile,
)

_IDENTITY_METADATA_FIELDS = (
    "operator",
    "hardware",
    "tensor_parallel_size",
    "source_sha256",
    "rms_norm_epsilon",
    "runtime_binding",
)
_GATE_FIELDS = (
    "latency_percentile",
    "minimum_samples",
    "minimum_speedup",
)
_ROW_IDENTITY_FIELDS = (
    "m",
    "k",
    "n",
    "dtype",
    "weight_format",
    "is_trans_b",
    "activation_layout",
    "weight_layout",
    "residual_layout",
    "gamma_layout",
)
_OMITTED_ROW_FIELDS = {
    "baseline_latency_ms",
    "fused_latency_ms",
    "projection_reduction_diagnostics_by_rank",
    "aggregation_provenance",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        action="append",
        required=True,
        help=(
            "One same-shape real-layer group. Repeat the option to union "
            "disjoint shapes, for example attention and down projections."
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination MC2 qualification profile.")
    return parser


def _sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _shape_key(row: Mapping[str, Any]) -> tuple[int, int, int, str, str, bool]:
    return (
        int(row["m"]),
        int(row["k"]),
        int(row["n"]),
        str(row["dtype"]),
        str(row["weight_format"]),
        bool(row["is_trans_b"]),
    )


def _shape_text(shape: tuple[int, int, int, str, str, bool]) -> str:
    m, k, n, dtype, weight_format, is_trans_b = shape
    return f"M{m}/K{k}/N{n}/{dtype}/{weight_format}/transB={is_trans_b}"


def _is_numerical_error_field(name: str) -> bool:
    return (
        name.startswith("max_abs_") or name.startswith("max_scaled_") or "_max_abs_" in name or "_max_scaled_" in name
    )


def _finite_number(value: Any, field: str, location: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{location} {field} must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{location} {field} must be >= {minimum:g}")
    return result


def _latency_samples(
    row: Mapping[str, Any],
    field: str,
    *,
    minimum_samples: int,
    location: str,
) -> list[float]:
    values = row.get(field)
    if not isinstance(values, list) or len(values) < minimum_samples:
        raise ValueError(f"{location} {field} needs at least {minimum_samples} samples")
    samples = [_finite_number(value, field, location) for value in values]
    if any(value <= 0.0 for value in samples):
        raise ValueError(f"{location} {field} must contain only positive samples")
    return samples


def _validate_projection_diagnostics(
    row: Mapping[str, Any],
    *,
    tp_size: int,
    location: str,
) -> None:
    diagnostics = row.get("projection_reduction_diagnostics_by_rank")
    if not isinstance(diagnostics, list) or len(diagnostics) != tp_size:
        raise ValueError(f"{location} requires complete projection diagnostics for {tp_size} ranks")
    ranks: set[int] = set()
    for item in diagnostics:
        if not isinstance(item, Mapping):
            raise ValueError(f"{location} contains an invalid projection diagnostic")
        rank = item.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < tp_size:
            raise ValueError(f"{location} projection diagnostic has an invalid rank")
        if rank in ranks:
            raise ValueError(f"{location} projection diagnostics duplicate rank {rank}")
        ranks.add(rank)
        if item.get("exact_match") is not True:
            raise ValueError(f"{location} projection diagnostic rank {rank} is not bit-exact")
        actual_hash = item.get("actual_sha256")
        expected_hash = item.get("expected_sha256")
        if (
            not isinstance(actual_hash, str)
            or len(actual_hash) != 64
            or any(character not in "0123456789abcdef" for character in actual_hash)
            or actual_hash != expected_hash
        ):
            raise ValueError(f"{location} projection diagnostic rank {rank} has inconsistent output hashes")
        mismatch_count = item.get("mismatch_count")
        if isinstance(mismatch_count, bool) or not isinstance(mismatch_count, int) or mismatch_count != 0:
            raise ValueError(f"{location} projection diagnostic rank {rank} reports mismatches")
        element_count = item.get("element_count")
        if isinstance(element_count, bool) or not isinstance(element_count, int) or element_count <= 0:
            raise ValueError(f"{location} projection diagnostic rank {rank} has no compared elements")
        if _finite_number(item.get("mismatch_fraction"), "mismatch_fraction", location, minimum=0.0) != 0.0:
            raise ValueError(f"{location} projection diagnostic rank {rank} has a nonzero mismatch fraction")
        if _finite_number(item.get("max_abs_error"), "max_abs_error", location, minimum=0.0) != 0.0:
            raise ValueError(f"{location} projection diagnostic rank {rank} has a nonzero maximum error")
        if item.get("first_mismatch") is not None or item.get("worst_mismatch") is not None:
            raise ValueError(f"{location} projection diagnostic rank {rank} retains mismatch details")
        chunk_counts = item.get("hccl_chunk_mismatch_counts")
        if (
            not isinstance(chunk_counts, list)
            or not chunk_counts
            or any(isinstance(value, bool) or not isinstance(value, int) or value != 0 for value in chunk_counts)
        ):
            raise ValueError(f"{location} projection diagnostic rank {rank} has HCCL chunk mismatches")
        if item.get("mismatching_64_block_ranges") != []:
            raise ValueError(f"{location} projection diagnostic rank {rank} has mismatching blocks")
    if ranks != set(range(tp_size)):
        raise ValueError(f"{location} projection diagnostic ranks {sorted(ranks)} are incomplete")


def _validate_production_row(
    row: Mapping[str, Any],
    *,
    latency_percentile: float,
    minimum_samples: int,
    minimum_speedup: float,
    tp_size: int,
    location: str,
    require_projection_diagnostics: bool,
) -> None:
    invalid_layouts = [
        name
        for name in (
            "activation_layout",
            "weight_layout",
            "residual_layout",
            "gamma_layout",
        )
        if row.get(name) != "contiguous"
    ]
    if invalid_layouts:
        raise ValueError(
            f"{location} must bind contiguous tensor layouts: {', '.join(invalid_layouts)}"
        )
    required_scaled_fields = (
        "max_scaled_norm",
        "max_scaled_added",
        "changed_input_max_scaled_norm",
        "changed_input_max_scaled_added",
    )
    missing_scaled = [name for name in required_scaled_fields if name not in row]
    if missing_scaled:
        raise ValueError(f"{location} lacks required scaled numerical gates: {', '.join(missing_scaled)}")
    scaled_fields = sorted(name for name in row if "max_scaled_" in name)
    for name in scaled_fields:
        value = _finite_number(row[name], name, location, minimum=0.0)
        if value > 1.0:
            raise ValueError(f"{location} {name}={value:g} exceeds the production numerical gate 1.0")

    changed_input_replays = row.get("changed_input_replays")
    if (
        isinstance(changed_input_replays, bool)
        or not isinstance(changed_input_replays, int)
        or changed_input_replays < 2
    ):
        raise ValueError(f"{location} changed_input_replays must be an integer >=2")
    changed_input_delta = _finite_number(row.get("changed_input_delta"), "changed_input_delta", location)
    if changed_input_delta == 0.0:
        raise ValueError(f"{location} changed_input_delta must be nonzero")

    required_absolute_fields = (
        "max_abs_norm",
        "max_abs_added",
        "changed_input_max_abs_norm",
        "changed_input_max_abs_added",
    )
    missing_absolute = [name for name in required_absolute_fields if name not in row]
    if missing_absolute:
        raise ValueError(f"{location} lacks changed-input absolute gates: {', '.join(missing_absolute)}")
    absolute = {
        name: _finite_number(row[name], name, location, minimum=0.0)
        for name in required_absolute_fields
    }
    if float(row["max_scaled_norm"]) < float(row["changed_input_max_scaled_norm"]):
        raise ValueError(f"{location} max_scaled_norm does not include the changed-input worst case")
    if float(row["max_scaled_added"]) < float(row["changed_input_max_scaled_added"]):
        raise ValueError(f"{location} max_scaled_added does not include the changed-input worst case")
    if absolute["max_abs_norm"] < absolute["changed_input_max_abs_norm"]:
        raise ValueError(f"{location} max_abs_norm does not include the changed-input worst case")
    if absolute["max_abs_added"] < absolute["changed_input_max_abs_added"]:
        raise ValueError(f"{location} max_abs_added does not include the changed-input worst case")

    baseline = _latency_samples(
        row,
        "baseline_latency_ms",
        minimum_samples=minimum_samples,
        location=location,
    )
    fused = _latency_samples(
        row,
        "fused_latency_ms",
        minimum_samples=minimum_samples,
        location=location,
    )
    baseline_p = _percentile(baseline, latency_percentile)
    fused_p = _percentile(fused, latency_percentile)
    if fused_p * minimum_speedup > baseline_p:
        raise ValueError(
            f"{location} p{latency_percentile:g} latency gate failed: "
            f"baseline={baseline_p:.9g} ms, fused={fused_p:.9g} ms, "
            f"minimum_speedup={minimum_speedup:g}"
        )

    if require_projection_diagnostics:
        _validate_projection_diagnostics(row, tp_size=tp_size, location=location)


def validate_mc2_measurement_for_production(
    document: Mapping[str, Any],
    *,
    source: str = "<memory>",
    require_projection_diagnostics: bool = True,
) -> None:
    """Validate numerical, replay, projection and latency production gates.

    This intentionally does not validate the source SHA.  Callers can surface
    a failed real-layer measurement before a later identity check explains
    that the artifact was produced by an older source tree.
    """

    if document.get("status") != "measured":
        raise ValueError(f"MC2 measurement {source} is not a complete measured result")
    metadata = document.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"MC2 measurement {source} lacks metadata")
    tp_size = metadata.get("tensor_parallel_size")
    if isinstance(tp_size, bool) or not isinstance(tp_size, int) or tp_size < 2:
        raise ValueError(f"MC2 measurement {source} has an invalid tensor_parallel_size")
    latency_percentile = _finite_number(
        document.get("latency_percentile"),
        "latency_percentile",
        f"MC2 measurement {source}",
        minimum=0.0,
    )
    if latency_percentile > 100.0:
        raise ValueError(f"MC2 measurement {source} latency_percentile must be <=100")
    minimum_samples = document.get("minimum_samples")
    if isinstance(minimum_samples, bool) or not isinstance(minimum_samples, int) or minimum_samples < 3:
        raise ValueError(f"MC2 measurement {source} minimum_samples must be an integer >=3")
    minimum_speedup = _finite_number(
        document.get("minimum_speedup"),
        "minimum_speedup",
        f"MC2 measurement {source}",
    )
    if minimum_speedup <= 1.0:
        raise ValueError(f"MC2 measurement {source} minimum_speedup must be >1")
    measurements = document.get("measurements")
    if not isinstance(measurements, list) or not measurements:
        raise ValueError(f"MC2 measurement {source} has no measurement rows")
    for index, row in enumerate(measurements):
        if not isinstance(row, Mapping):
            raise ValueError(f"MC2 measurement {source} row {index} is not an object")
        shape_fields = tuple(row.get(name) for name in _ROW_IDENTITY_FIELDS)
        shape = _shape_text(_shape_key(row)) if all(value is not None for value in shape_fields) else f"row-{index}"
        _validate_production_row(
            row,
            latency_percentile=latency_percentile,
            minimum_samples=minimum_samples,
            minimum_speedup=minimum_speedup,
            tp_size=tp_size,
            location=f"MC2 measurement {source} {shape}",
            require_projection_diagnostics=require_projection_diagnostics,
        )


def _validate_equal(documents: Sequence[Mapping[str, Any]], field: str, *, metadata: bool = False) -> Any:
    values = []
    for document in documents:
        container = document.get("metadata") if metadata else document
        location = "metadata." if metadata else ""
        if not isinstance(container, Mapping) or field not in container:
            raise ValueError(f"MC2 input is missing required field {location}{field}")
        values.append(container[field])
    if any(value != values[0] for value in values[1:]):
        location = "metadata." if metadata else ""
        raise ValueError(f"MC2 inputs disagree on {location}{field}")
    return deepcopy(values[0])


def _load_inputs(paths: Sequence[Path]) -> list[dict[str, Any]]:
    if len(paths) < 2:
        raise ValueError("MC2 cross-layer aggregation requires at least two input files")
    resolved = [path.expanduser().resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("MC2 aggregation input paths must be unique")

    loaded: list[dict[str, Any]] = []
    seen_document_hashes: dict[str, Path] = {}
    seen_input_sources: dict[str, Path] = {}
    for path in resolved:
        try:
            contents = path.read_bytes()
            document = json.loads(contents)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read MC2 measurement {path}: {error}") from error
        if not isinstance(document, dict):
            raise ValueError(f"MC2 measurement {path} must contain a JSON object")
        document_hash = _sha256(contents)
        if document_hash in seen_document_hashes:
            raise ValueError(f"MC2 inputs {seen_document_hashes[document_hash]} and {path} have duplicate content")
        seen_document_hashes[document_hash] = path
        if document.get("status") != "measured":
            raise ValueError(f"MC2 measurement {path} is not a complete measured result")
        metadata = document.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"MC2 measurement {path} lacks metadata")
        if metadata.get("measurement_scope") != "captured_operator_replay":
            raise ValueError(f"MC2 measurement {path} has an unsupported measurement scope")
        input_source = metadata.get("input_source")
        if (
            not isinstance(input_source, str)
            or not input_source.strip()
            or input_source != input_source.strip()
            or input_source.casefold() == "synthetic"
        ):
            raise ValueError(f"MC2 measurement {path} must identify a real-model input source")
        if input_source in seen_input_sources:
            raise ValueError(
                f"MC2 inputs {seen_input_sources[input_source]} and {path} reuse input_source {input_source!r}"
            )
        seen_input_sources[input_source] = path

        loaded.append(
            {
                "path": str(path),
                "document_sha256": document_hash,
                "input_source": input_source,
                "document": document,
            }
        )
    return loaded


def _measurement_map(record: Mapping[str, Any]) -> dict[tuple[int, int, int, str, str, bool], Mapping[str, Any]]:
    path = record["path"]
    measurements = record["document"].get("measurements")
    if not isinstance(measurements, list) or not measurements:
        raise ValueError(f"MC2 measurement {path} has no measurement rows")
    result: dict[tuple[int, int, int, str, str, bool], Mapping[str, Any]] = {}
    for row in measurements:
        if not isinstance(row, Mapping):
            raise ValueError(f"MC2 measurement {path} contains a non-object row")
        missing = [name for name in _ROW_IDENTITY_FIELDS if name not in row]
        if missing:
            raise ValueError(f"MC2 measurement {path} row is missing {', '.join(missing)}")
        if not isinstance(row["is_trans_b"], bool):
            raise ValueError(f"MC2 measurement {path} row is_trans_b must be boolean")
        shape = _shape_key(row)
        if shape in result:
            raise ValueError(f"MC2 measurement {path} contains duplicate shape {_shape_text(shape)}")
        result[shape] = row
    return result


def _provenance_ref(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(record["path"]),
        "sha256": str(record["document_sha256"]),
        "input_source": str(record["input_source"]),
    }


def _aggregate_shape(
    shape: tuple[int, int, int, str, str, bool],
    records: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    latency_percentile: float,
) -> dict[str, Any]:
    first = rows[0]
    error_fields = {name for name in first if _is_numerical_error_field(name)}
    for row in rows[1:]:
        if {name for name in row if _is_numerical_error_field(name)} != error_fields:
            raise ValueError(f"MC2 shape {_shape_text(shape)} has inconsistent numerical error fields")

    preserved_fields = {
        name for name in first if name not in _OMITTED_ROW_FIELDS and not _is_numerical_error_field(name)
    }
    for row in rows[1:]:
        row_preserved_fields = {
            name for name in row if name not in _OMITTED_ROW_FIELDS and not _is_numerical_error_field(name)
        }
        if row_preserved_fields != preserved_fields:
            raise ValueError(f"MC2 shape {_shape_text(shape)} has inconsistent row fields")
        for name in preserved_fields:
            if row[name] != first[name]:
                raise ValueError(f"MC2 shape {_shape_text(shape)} disagrees on row field {name}")
    for name in error_fields:
        for row in rows:
            value = row[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"MC2 shape {_shape_text(shape)} has invalid numerical error field {name}")

    baseline_index = min(
        range(len(rows)),
        key=lambda index: _percentile(rows[index]["baseline_latency_ms"], latency_percentile),
    )
    fused_index = max(
        range(len(rows)),
        key=lambda index: _percentile(rows[index]["fused_latency_ms"], latency_percentile),
    )
    result = {
        name: deepcopy(value)
        for name, value in first.items()
        if name not in _OMITTED_ROW_FIELDS and not _is_numerical_error_field(name)
    }
    result["baseline_latency_ms"] = list(rows[baseline_index]["baseline_latency_ms"])
    result["fused_latency_ms"] = list(rows[fused_index]["fused_latency_ms"])

    numerical_sources: dict[str, dict[str, Any]] = {}
    for name in sorted(error_fields):
        selected_index = max(range(len(rows)), key=lambda index: float(rows[index][name]))
        result[name] = float(rows[selected_index][name])
        numerical_sources[name] = _provenance_ref(records[selected_index])

    baseline_ref = _provenance_ref(records[baseline_index])
    baseline_ref["selected_percentile_ms"] = _percentile(
        rows[baseline_index]["baseline_latency_ms"], latency_percentile
    )
    fused_ref = _provenance_ref(records[fused_index])
    fused_ref["selected_percentile_ms"] = _percentile(rows[fused_index]["fused_latency_ms"], latency_percentile)
    result["aggregation_provenance"] = {
        "method": "conservative_cross_layer_envelope_v1",
        "baseline_latency_source": baseline_ref,
        "fused_latency_source": fused_ref,
        "numerical_error_sources": numerical_sources,
    }
    return result


def aggregate_mc2_profiles(paths: Sequence[Path]) -> dict[str, Any]:
    """Return a strict conservative envelope from real-layer profiles."""

    records = _load_inputs(paths)
    documents = [record["document"] for record in records]
    if any(
        isinstance(document.get("schema_version"), bool) or document.get("schema_version") != 1
        for document in documents
    ):
        raise ValueError("MC2 aggregation requires schema_version=1 in every input")
    # Structural identity and gate configuration still fail with their exact
    # field names. Source SHA is deliberately deferred until after production
    # gates: a stale real-layer artifact may also expose a genuine kernel
    # failure, which must not be hidden by the expected source change.
    for field in _IDENTITY_METADATA_FIELDS:
        if field == "source_sha256":
            continue
        _validate_equal(documents, field, metadata=True)
    gates = {field: _validate_equal(documents, field) for field in _GATE_FIELDS}
    latency_percentile = gates["latency_percentile"]
    minimum_samples = gates["minimum_samples"]
    minimum_speedup = gates["minimum_speedup"]
    if (
        isinstance(latency_percentile, bool)
        or not isinstance(latency_percentile, (int, float))
        or not math.isfinite(float(latency_percentile))
        or float(latency_percentile) != 95.0
    ):
        raise ValueError("MC2 conservative aggregation requires latency_percentile=95")
    if isinstance(minimum_samples, bool) or not isinstance(minimum_samples, int) or minimum_samples < 3:
        raise ValueError("MC2 conservative aggregation requires integer minimum_samples>=3")
    if (
        isinstance(minimum_speedup, bool)
        or not isinstance(minimum_speedup, (int, float))
        or not math.isfinite(float(minimum_speedup))
        or float(minimum_speedup) <= 1.0
    ):
        raise ValueError("MC2 conservative aggregation requires finite minimum_speedup>1")

    require_projection_diagnostics = (
        documents[0]["metadata"].get("operator") != MC2_TP3_NATIVE_EPILOGUE_OPERATOR
    )
    for record, document in zip(records, documents):
        validate_mc2_measurement_for_production(
            document,
            source=str(record["path"]),
            # v126 keeps F.linear native and qualifies the complete composition
            # bit-exactly; the old fused-MatMul chunk diagnostic does not apply.
            require_projection_diagnostics=require_projection_diagnostics,
        )
    _validate_equal(documents, "source_sha256", metadata=True)

    # Bind every otherwise-qualified input to the current adapter/kernel
    # sources before allowing it to contribute samples.
    for document in documents:
        normalize_mc2_profile(document)

    measurement_maps = [_measurement_map(record) for record in records]
    expected_shapes = set(measurement_maps[0])
    for record, measurements in zip(records[1:], measurement_maps[1:]):
        actual_shapes = set(measurements)
        if actual_shapes != expected_shapes:
            missing = sorted(_shape_text(shape) for shape in expected_shapes - actual_shapes)
            extra = sorted(_shape_text(shape) for shape in actual_shapes - expected_shapes)
            raise ValueError(
                f"MC2 measurement {record['path']} has an incomplete shape set; missing={missing}, extra={extra}"
            )

    metadata = deepcopy(documents[0]["metadata"])
    metadata["input_source"] = "conservative_cross_layer_envelope"
    metadata["aggregation_provenance"] = {
        "method": "conservative_cross_layer_envelope_v1",
        "inputs": [_provenance_ref(record) for record in records],
    }
    measurements = [
        _aggregate_shape(
            shape,
            records,
            [measurement_map[shape] for measurement_map in measurement_maps],
            latency_percentile=float(gates["latency_percentile"]),
        )
        for shape in sorted(expected_shapes)
    ]
    result = {
        "schema_version": 1,
        "status": "measured",
        "metadata": metadata,
        **gates,
        "measurements": measurements,
    }
    validate_mc2_measurement_for_production(
        result,
        source="conservative cross-layer envelope",
        require_projection_diagnostics=False,
    )
    # Keep this invariant executable: any future schema change must update the
    # aggregator before it can emit a profile that production silently ignores.
    normalize_mc2_profile(result)
    return result


def _union_mc2_envelopes(envelopes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Union independently aggregated, disjoint exact-shape envelopes."""

    if len(envelopes) < 2:
        raise ValueError("MC2 profile union requires at least two shape envelopes")
    if any(
        isinstance(envelope.get("schema_version"), bool) or envelope.get("schema_version") != 1
        for envelope in envelopes
    ):
        raise ValueError("MC2 profile union requires schema_version=1 in every envelope")
    if any(envelope.get("status") != "measured" for envelope in envelopes):
        raise ValueError("MC2 profile union requires complete measured envelopes")
    for field in _IDENTITY_METADATA_FIELDS:
        _validate_equal(envelopes, field, metadata=True)
    gates = {field: _validate_equal(envelopes, field) for field in _GATE_FIELDS}
    for envelope in envelopes:
        validate_mc2_measurement_for_production(
            envelope,
            source="conservative exact-shape envelope",
            require_projection_diagnostics=False,
        )
        normalize_mc2_profile(envelope)

    measurements: list[dict[str, Any]] = []
    seen_shapes: set[tuple[int, int, int, str, str, bool]] = set()
    seen_input_paths: set[str] = set()
    seen_input_hashes: set[str] = set()
    seen_input_sources: set[str] = set()
    inputs: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for envelope in envelopes:
        envelope_metadata = envelope["metadata"]
        provenance = envelope_metadata.get("aggregation_provenance")
        if not isinstance(provenance, Mapping) or provenance.get("method") != "conservative_cross_layer_envelope_v1":
            raise ValueError("MC2 profile union accepts only conservative cross-layer envelopes")
        group_inputs = provenance.get("inputs")
        if not isinstance(group_inputs, list) or not group_inputs:
            raise ValueError("MC2 cross-layer envelope lacks input provenance")
        normalized_inputs: list[dict[str, Any]] = []
        for item in group_inputs:
            if not isinstance(item, Mapping):
                raise ValueError("MC2 cross-layer envelope has invalid input provenance")
            normalized = {
                "path": item.get("path"),
                "sha256": item.get("sha256"),
                "input_source": item.get("input_source"),
            }
            if (
                not isinstance(normalized["path"], str)
                or not normalized["path"]
                or not isinstance(normalized["sha256"], str)
                or len(normalized["sha256"]) != 64
                or not isinstance(normalized["input_source"], str)
                or not normalized["input_source"]
            ):
                raise ValueError("MC2 cross-layer envelope has incomplete input provenance")
            path = normalized["path"]
            document_sha = normalized["sha256"]
            input_source = normalized["input_source"]
            if path in seen_input_paths or document_sha in seen_input_hashes or input_source in seen_input_sources:
                raise ValueError("MC2 profile union reuses cross-layer input evidence")
            seen_input_paths.add(path)
            seen_input_hashes.add(document_sha)
            seen_input_sources.add(input_source)
            normalized_inputs.append(normalized)
            inputs.append(normalized)

        group_shapes: list[str] = []
        for row in envelope["measurements"]:
            shape = _shape_key(row)
            if shape in seen_shapes:
                raise ValueError(f"MC2 profile union contains duplicate shape {_shape_text(shape)}")
            seen_shapes.add(shape)
            group_shapes.append(_shape_text(shape))
            measurements.append(deepcopy(row))
        groups.append({"shapes": sorted(group_shapes), "inputs": normalized_inputs})

    metadata = deepcopy(envelopes[0]["metadata"])
    metadata["input_source"] = "conservative_exact_shape_union"
    metadata["aggregation_provenance"] = {
        "method": "conservative_exact_shape_union_v1",
        "inputs": inputs,
        "groups": groups,
    }
    result = {
        "schema_version": 1,
        "status": "measured",
        "metadata": metadata,
        **gates,
        "measurements": sorted(measurements, key=_shape_key),
    }
    validate_mc2_measurement_for_production(
        result,
        source="conservative exact-shape union",
        require_projection_diagnostics=False,
    )
    normalize_mc2_profile(result)
    return result


def aggregate_mc2_profile_groups(groups: Sequence[Sequence[Path]]) -> dict[str, Any]:
    """Aggregate each real-layer group, then union disjoint exact shapes."""

    if not groups:
        raise ValueError("MC2 aggregation requires at least one input group")
    envelopes = [aggregate_mc2_profiles(group) for group in groups]
    return envelopes[0] if len(envelopes) == 1 else _union_mc2_envelopes(envelopes)


def main() -> None:
    args = _parser().parse_args()
    result = aggregate_mc2_profile_groups(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
