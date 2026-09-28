# SPDX-License-Identifier: Apache-2.0
"""Strictly union projection-aware MC2 qualification runs without an NPU.

New input runs must have been finalized by ``qualify_mc2_multishape.sh`` and
contain ``production-projection-rows.json``.  A legacy finalized run may be
used only through its conservative two-projection intersection recorded in
``cross-layer-screen.json`` and ``qualified-rows.txt``.  Partial legacy rows
are intentionally unrecoverable.  Every accepted projection/row is rebound to
raw ACLGraph qualification on layers 0, 31 and 63 and revalidated with the
public MC2 aggregator before disjoint exact shapes are unioned.  Numerical,
changed-input, latency, source and runtime-binding gates are never relaxed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from examples.aggregate_specslo_mc2_profiles import (
    aggregate_mc2_profile_groups,
    aggregate_mc2_profiles,
)
from vllm_ascend.spec_decode.pearl.mc2 import normalize_mc2_profile

_PROJECTION_K = {"attention": 3072, "down": 8576}
_QUALIFICATION_LAYERS = (0, 31, 63)
_MANIFEST_NAME = "production-projection-rows.json"
_SOURCE_MANIFEST_NAME = "qualified-projection-rows.json"
_LEGACY_REPORT_NAME = "cross-layer-screen.json"
_LEGACY_ROWS_NAME = "qualified-rows.txt"
_TOOL_PATH = Path(__file__).resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help="Finalized projection-aware qualification run; repeat to union runs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        help="Union audit report (default: <output-stem>.union-report.json).",
    )
    parser.add_argument(
        "--sha256-output",
        type=Path,
        help="Profile checksum file (default: <output-name>.sha256).",
    )
    return parser


def _sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _read_json_object(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        contents = path.read_bytes()
        document = json.loads(contents)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{label} {path} must contain a JSON object")
    return contents, document


def _validate_projection_rows(
    document: Mapping[str, Any],
    *,
    path: Path,
    expected_stage: str,
) -> dict[str, tuple[int, ...]]:
    location = f"projection manifest {path}"
    schema_version = document.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 1:
        raise ValueError(f"{location} requires schema_version=1")
    if document.get("kind") != "specslo_mc2_projection_rows":
        raise ValueError(f"{location} has an invalid kind")
    if document.get("stage") != expected_stage:
        raise ValueError(f"{location} has an invalid stage")
    if document.get("execution_mode") != "graph":
        raise ValueError(f"{location} is not ACLGraph evidence")
    operations_per_graph = document.get("operations_per_graph")
    if isinstance(operations_per_graph, bool) or operations_per_graph != 1:
        raise ValueError(f"{location} must contain one operation per graph")
    if document.get("qualification_layers") != list(_QUALIFICATION_LAYERS):
        raise ValueError(f"{location} does not cover layers 0/31/63")
    if document.get("projection_k") != _PROJECTION_K:
        raise ValueError(f"{location} has an invalid projection mapping")

    raw_rows = document.get("projection_rows")
    if not isinstance(raw_rows, Mapping) or set(raw_rows) != set(_PROJECTION_K):
        raise ValueError(f"{location} has invalid projection rows")
    projection_rows: dict[str, tuple[int, ...]] = {}
    for kind in _PROJECTION_K:
        values = raw_rows[kind]
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ValueError(f"{location} {kind} rows must be an array")
        rows = tuple(values)
        if any(
            isinstance(row, bool) or not isinstance(row, int) or row <= 0
            for row in rows
        ):
            raise ValueError(f"{location} {kind} rows must be positive integers")
        if tuple(sorted(set(rows))) != rows:
            raise ValueError(f"{location} {kind} rows must be sorted and unique")
        projection_rows[kind] = rows

    intersection = sorted(
        set(projection_rows["attention"]).intersection(projection_rows["down"])
    )
    if document.get("intersection_rows") != intersection:
        raise ValueError(f"{location} has an inconsistent intersection_rows field")
    return projection_rows


def _validate_legacy_rows(value: Any, *, location: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{location} must be an array")
    rows = tuple(value)
    if any(
        isinstance(row, bool) or not isinstance(row, int) or row <= 0
        for row in rows
    ):
        raise ValueError(f"{location} must contain positive integers")
    if tuple(sorted(set(rows))) != rows:
        raise ValueError(f"{location} must be sorted and unique")
    return rows


def _load_legacy_qualification_run(resolved_run: Path) -> dict[str, Any]:
    """Load only the strict two-projection intersection from an old run."""

    report_path = resolved_run / _LEGACY_REPORT_NAME
    report_bytes, report = _read_json_object(
        report_path,
        label="legacy conservative cross-layer report",
    )
    location = f"legacy conservative cross-layer report {report_path}"
    schema_version = report.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 1:
        raise ValueError(f"{location} requires schema_version=1")
    if report.get("kind") != "specslo_mc2_conservative_cross_layer_screen":
        raise ValueError(f"{location} has an invalid kind")
    candidate_rows = _validate_legacy_rows(
        report.get("candidate_rows"),
        location=f"{location} candidate_rows",
    )
    qualified_rows = _validate_legacy_rows(
        report.get("qualified_rows"),
        location=f"{location} qualified_rows",
    )
    if not set(qualified_rows).issubset(candidate_rows):
        raise ValueError(f"{location} qualified rows are absent from candidate_rows")

    raw_groups = report.get("groups")
    if not isinstance(raw_groups, list):
        raise ValueError(f"{location} groups must be an array")
    expected_groups = {
        (kind, row) for kind in _PROJECTION_K for row in candidate_rows
    }
    observed_groups: set[tuple[str, int]] = set()
    passed = {kind: set() for kind in _PROJECTION_K}
    for index, group in enumerate(raw_groups):
        group_location = f"{location} group {index}"
        if not isinstance(group, Mapping):
            raise ValueError(f"{group_location} must be an object")
        kind = group.get("kind")
        row = group.get("m")
        if (
            not isinstance(kind, str)
            or kind not in _PROJECTION_K
            or isinstance(row, bool)
            or not isinstance(row, int)
            or (kind, row) not in expected_groups
        ):
            raise ValueError(f"{group_location} has an unexpected projection/row")
        shape = (kind, row)
        if shape in observed_groups:
            raise ValueError(f"{location} contains duplicate group {kind}/M{row}")
        observed_groups.add(shape)

        expected_inputs = [
            str(
                (
                    resolved_run
                    / "measurements"
                    / kind
                    / f"layer-{layer}"
                    / f"m{row}"
                    / "profile.json"
                ).resolve()
            )
            for layer in _QUALIFICATION_LAYERS
        ]
        if group.get("inputs") != expected_inputs:
            raise ValueError(f"{group_location} has stale or incomplete layer paths")
        status = group.get("status")
        error = group.get("error")
        if status == "pass" and error is None:
            passed[kind].add(row)
        elif status != "fail" or not isinstance(error, str) or not error:
            raise ValueError(f"{group_location} has an invalid status/error pair")

    if observed_groups != expected_groups:
        missing = sorted(expected_groups.difference(observed_groups))
        extra = sorted(observed_groups.difference(expected_groups))
        raise ValueError(
            f"{location} group coverage mismatch: missing={missing}, extra={extra}"
        )
    passed_intersection = tuple(sorted(passed["attention"].intersection(passed["down"])))
    if qualified_rows != passed_intersection:
        raise ValueError(
            f"{location} qualified_rows does not equal the two-projection pass intersection"
        )
    if not qualified_rows:
        raise ValueError(f"{location} contains no qualified two-projection row")

    rows_path = resolved_run / _LEGACY_ROWS_NAME
    try:
        rows_bytes = rows_path.read_bytes()
        expected_contents = " ".join(str(row) for row in qualified_rows) + "\n"
        contents = rows_bytes.decode()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read legacy qualified rows {rows_path}: {error}") from error
    if contents != expected_contents:
        raise ValueError(
            f"legacy qualified rows {rows_path} disagrees with {report_path}"
        )

    projection_rows = {kind: qualified_rows for kind in _PROJECTION_K}
    return {
        "run_dir": resolved_run,
        "manifest_format": "legacy-intersection-v1",
        "manifest_path": report_path,
        "manifest_sha256": _sha256(report_bytes),
        "source_manifest_path": rows_path,
        "source_manifest_sha256": _sha256(rows_bytes),
        "projection_rows": projection_rows,
    }


def _load_qualification_run(run_dir: Path) -> dict[str, Any]:
    try:
        resolved_run = run_dir.expanduser().resolve(strict=True)
    except OSError as error:
        raise ValueError(f"cannot resolve qualification run {run_dir}: {error}") from error
    if not resolved_run.is_dir():
        raise ValueError(f"qualification run is not a directory: {resolved_run}")

    manifest_path = resolved_run / _MANIFEST_NAME
    if not manifest_path.exists():
        return _load_legacy_qualification_run(resolved_run)
    manifest_bytes, manifest = _read_json_object(
        manifest_path,
        label="production projection manifest",
    )
    projection_rows = _validate_projection_rows(
        manifest,
        path=manifest_path,
        expected_stage="conservative-cross-layer-production",
    )
    if not any(projection_rows.values()):
        raise ValueError(f"projection manifest {manifest_path} contains no qualified shape")

    source_path = resolved_run / _SOURCE_MANIFEST_NAME
    if manifest.get("source_manifest") != str(source_path):
        raise ValueError(
            f"production projection manifest {manifest_path} is not bound to "
            f"{source_path}"
        )
    source_bytes, source_manifest = _read_json_object(
        source_path,
        label="per-layer projection manifest",
    )
    if manifest.get("source_manifest_sha256") != _sha256(source_bytes):
        raise ValueError(
            f"production projection manifest {manifest_path} source hash mismatch"
        )
    source_rows = _validate_projection_rows(
        source_manifest,
        path=source_path,
        expected_stage="per-layer-qualification",
    )
    for kind in _PROJECTION_K:
        if not set(projection_rows[kind]).issubset(source_rows[kind]):
            raise ValueError(
                f"production projection manifest {manifest_path} contains a {kind} "
                "row absent from the per-layer qualification"
            )

    return {
        "run_dir": resolved_run,
        "manifest_format": "projection-aware-v1",
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256(manifest_bytes),
        "source_manifest_path": source_path,
        "source_manifest_sha256": _sha256(source_bytes),
        "projection_rows": projection_rows,
    }


def _validate_graph_evidence(
    profile_path: Path,
    *,
    kind: str,
    row: int,
    layer: int,
) -> dict[str, str]:
    raw_path = profile_path.with_name("raw.json")
    if not raw_path.is_file():
        raise ValueError(
            f"qualified shape {kind}/M{row} is missing ACLGraph evidence for "
            f"layer {layer}: {raw_path}"
        )
    raw_bytes, raw = _read_json_object(raw_path, label="ACLGraph measurement")
    profile_bytes, profile = _read_json_object(
        profile_path,
        label="production projection measurement",
    )
    schema_version = raw.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 1:
        raise ValueError(f"ACLGraph measurement {raw_path} requires schema_version=1")
    if raw.get("status") != "measured":
        raise ValueError(f"ACLGraph measurement {raw_path} is incomplete")
    metadata = raw.get("metadata")
    profile_metadata = profile.get("metadata")
    if not isinstance(metadata, Mapping) or not isinstance(profile_metadata, Mapping):
        raise ValueError(f"ACLGraph measurement {raw_path} lacks metadata")
    operations_per_graph = metadata.get("operations_per_graph")
    if (
        metadata.get("measurement_scope")
        != "graph_native_matmul_plus_rank3_epilogue"
        or metadata.get("execution_mode") != "graph"
        or isinstance(operations_per_graph, bool)
        or operations_per_graph != 1
    ):
        raise ValueError(f"ACLGraph measurement {raw_path} is not graph/one-operation evidence")
    if metadata.get("chained_flush") is not True:
        raise ValueError(f"ACLGraph measurement {raw_path} did not flush the chained epilogue")
    if metadata.get("tensor_parallel_size") != 3:
        raise ValueError(f"ACLGraph measurement {raw_path} is not TP3 evidence")
    for field in ("source_sha256", "input_source"):
        if metadata.get(field) != profile_metadata.get(field):
            raise ValueError(
                f"ACLGraph measurement {raw_path} disagrees with its profile on {field}"
            )

    expected_shape = {"m": row, "k": _PROJECTION_K[kind], "n": 5120}
    if raw.get("shape") != expected_shape:
        raise ValueError(
            f"ACLGraph measurement {raw_path} has shape {raw.get('shape')}, "
            f"expected {expected_shape}"
        )
    correctness = raw.get("correctness")
    if not isinstance(correctness, Mapping):
        raise ValueError(f"ACLGraph measurement {raw_path} lacks correctness evidence")
    if correctness.get("exact_all_ranks") is not True:
        raise ValueError(f"ACLGraph measurement {raw_path} is not bit-exact")
    if correctness.get("changed_input_exact_all_ranks") is not True:
        raise ValueError(f"ACLGraph measurement {raw_path} changed-input replay is not bit-exact")
    changed_replays = correctness.get("changed_input_replays")
    changed_delta = correctness.get("changed_input_delta")
    if (
        isinstance(changed_replays, bool)
        or not isinstance(changed_replays, int)
        or changed_replays < 2
        or isinstance(changed_delta, bool)
        or not isinstance(changed_delta, (int, float))
        or not math.isfinite(float(changed_delta))
        or float(changed_delta) == 0.0
    ):
        raise ValueError(f"ACLGraph measurement {raw_path} has invalid changed-input evidence")

    samples = raw.get("samples")
    measurements = profile.get("measurements")
    if (
        not isinstance(samples, Mapping)
        or not isinstance(measurements, list)
        or len(measurements) != 1
        or not isinstance(measurements[0], Mapping)
        or measurements[0].get("baseline_latency_ms")
        != samples.get("baseline_latency_ms")
        or measurements[0].get("fused_latency_ms")
        != samples.get("fused_latency_ms")
    ):
        raise ValueError(
            f"ACLGraph measurement {raw_path} latency samples differ from its profile"
        )
    return {
        "raw_path": str(raw_path),
        "raw_sha256": _sha256(raw_bytes),
        "profile_path": str(profile_path),
        "profile_sha256": _sha256(profile_bytes),
    }


def collect_qualification_groups(
    run_dirs: Sequence[Path],
) -> tuple[list[list[Path]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load manifests and return disjoint, independently validated groups."""

    if not run_dirs:
        raise ValueError("at least one qualification run is required")
    resolved_runs = [path.expanduser().resolve() for path in run_dirs]
    if len(set(resolved_runs)) != len(resolved_runs):
        raise ValueError("qualification run directories must be unique")

    groups: list[list[Path]] = []
    group_records: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    seen_shapes: dict[tuple[str, int], Path] = {}
    for run_dir in run_dirs:
        run = _load_qualification_run(run_dir)
        run_records.append(
            {
                "run_dir": str(run["run_dir"]),
                "manifest_format": run["manifest_format"],
                "manifest": str(run["manifest_path"]),
                "manifest_sha256": run["manifest_sha256"],
                "source_manifest": str(run["source_manifest_path"]),
                "source_manifest_sha256": run["source_manifest_sha256"],
                "projection_rows": {
                    kind: list(rows)
                    for kind, rows in run["projection_rows"].items()
                },
            }
        )
        for kind in _PROJECTION_K:
            for row in run["projection_rows"][kind]:
                shape = (kind, row)
                if shape in seen_shapes:
                    raise ValueError(
                        f"duplicate exact shape {kind}/M{row} in "
                        f"{seen_shapes[shape]} and {run['run_dir']}"
                    )
                paths = [
                    run["run_dir"]
                    / "measurements"
                    / kind
                    / f"layer-{layer}"
                    / f"m{row}"
                    / "profile.json"
                    for layer in _QUALIFICATION_LAYERS
                ]
                missing = [path for path in paths if not path.is_file()]
                if missing:
                    raise ValueError(
                        f"qualified shape {kind}/M{row} is missing layer evidence: "
                        + ", ".join(str(path) for path in missing)
                    )
                graph_evidence = [
                    _validate_graph_evidence(
                        path,
                        kind=kind,
                        row=row,
                        layer=layer,
                    )
                    for layer, path in zip(_QUALIFICATION_LAYERS, paths, strict=True)
                ]
                # Validate the exact projection/M envelope before it can enter
                # the cross-run union.  The public function enforces numerical,
                # changed-input, p95, source and runtime binding contracts.
                envelope = aggregate_mc2_profiles(paths)
                measurements = envelope.get("measurements")
                if not isinstance(measurements, list) or len(measurements) != 1:
                    raise ValueError(
                        f"qualified shape {kind}/M{row} does not contain one exact row"
                    )
                measurement_row = measurements[0]
                expected_shape = (row, _PROJECTION_K[kind], 5120)
                actual_shape = tuple(
                    measurement_row.get(field) for field in ("m", "k", "n")
                )
                if actual_shape != expected_shape:
                    raise ValueError(
                        f"qualified shape {kind}/M{row} contains {actual_shape}, "
                        f"expected {expected_shape}"
                    )
                seen_shapes[shape] = run["run_dir"]
                groups.append(paths)
                group_records.append(
                    {
                        "kind": kind,
                        "m": row,
                        "k": _PROJECTION_K[kind],
                        "layers": list(_QUALIFICATION_LAYERS),
                        "inputs": [str(path) for path in paths],
                        "aclgraph_evidence": graph_evidence,
                    }
                )
    return groups, group_records, run_records


def aggregate_qualification_runs(
    run_dirs: Sequence[Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    groups, group_records, run_records = collect_qualification_groups(run_dirs)
    profile = aggregate_mc2_profile_groups(groups)
    normalized = normalize_mc2_profile(profile)
    report = {
        "schema_version": 1,
        "kind": "specslo_mc2_qualification_run_union",
        "status": "pass",
        "execution_mode": "graph",
        "operations_per_graph": 1,
        "qualification_layers": list(_QUALIFICATION_LAYERS),
        "runs": run_records,
        "groups": group_records,
        "shape_count": len(normalized.entries),
        "source_sha256": profile["metadata"]["source_sha256"],
        "runtime_binding": profile["metadata"]["runtime_binding"],
        "tool": {
            "path": str(_TOOL_PATH),
            "sha256": _sha256(_TOOL_PATH.read_bytes()),
        },
    }
    return profile, report


def _atomic_write(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _default_report_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.union-report.json")


def _default_sha256_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.sha256")


def main() -> int:
    args = _parser().parse_args()
    output = args.output.expanduser().resolve()
    report_path = (
        args.report.expanduser().resolve()
        if args.report is not None
        else _default_report_path(output)
    )
    sha256_path = (
        args.sha256_output.expanduser().resolve()
        if args.sha256_output is not None
        else _default_sha256_path(output)
    )
    destinations = (output, report_path, sha256_path)
    if len(set(destinations)) != len(destinations):
        print("output, report and SHA256 paths must be distinct", file=sys.stderr)
        return 2
    existing = [path for path in destinations if path.exists()]
    if existing:
        print(
            "refusing to overwrite existing union artifact: "
            + ", ".join(str(path) for path in existing),
            file=sys.stderr,
        )
        return 2

    try:
        profile, report = aggregate_qualification_runs(args.run_dir)
        profile_bytes = (
            json.dumps(profile, ensure_ascii=False, indent=2) + "\n"
        ).encode()
        profile_sha256 = _sha256(profile_bytes)
        report.update(
            {
                "output": str(output),
                "profile_sha256": profile_sha256,
                "sha256_output": str(sha256_path),
            }
        )
    except (OSError, TypeError, ValueError) as error:
        failure_report = {
            "schema_version": 1,
            "kind": "specslo_mc2_qualification_run_union",
            "status": "fail",
            "run_dirs": [str(path.expanduser().resolve()) for path in args.run_dir],
            "error": f"{type(error).__name__}: {error}",
            "tool": {
                "path": str(_TOOL_PATH),
                "sha256": _sha256(_TOOL_PATH.read_bytes()),
            },
        }
        _atomic_write(
            report_path,
            (json.dumps(failure_report, indent=2, sort_keys=True) + "\n").encode(),
        )
        print(failure_report["error"], file=sys.stderr)
        return 1

    _atomic_write(output, profile_bytes)
    _atomic_write(
        sha256_path,
        f"{profile_sha256}  {output}\n".encode(),
    )
    _atomic_write(
        report_path,
        (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(f"production profile: {output}")
    print(f"profile SHA256: {profile_sha256}")
    print(f"union report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
