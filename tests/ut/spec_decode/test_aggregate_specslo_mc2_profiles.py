# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for conservative cross-layer MC2 aggregation."""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

import vllm_ascend.spec_decode.pearl.mc2 as mc2_module
from examples.aggregate_specslo_mc2_profiles import (
    _parser,
    aggregate_mc2_profile_groups,
    aggregate_mc2_profiles,
    validate_mc2_measurement_for_production,
)
from examples.aggregate_specslo_mc2_qualification_runs import (
    aggregate_qualification_runs,
    collect_qualification_groups,
)
from examples.aggregate_specslo_mc2_qualification_runs import (
    main as qualification_union_main,
)
from vllm_ascend.spec_decode.pearl.mc2 import mc2_source_sha256, normalize_mc2_profile
from vllm_ascend.spec_decode.pearl.native_engine import NativePearlConfig

_TEST_VENDOR_SHA256 = "a" * 64
_TEST_ADAPTER_SHA256 = "c" * 64


def _projection_diagnostics(m: int, n: int = 5120) -> list[dict]:
    digest = "a" * 64
    return [
        {
            "actual_sha256": digest,
            "expected_sha256": digest,
            "exact_match": True,
            "mismatch_count": 0,
            "element_count": m * n,
            "mismatch_fraction": 0.0,
            "max_abs_error": 0.0,
            "first_mismatch": None,
            "worst_mismatch": None,
            "hccl_chunk_mismatch_counts": [0] * 6,
            "mismatching_64_block_ranges": [],
            "rank": rank,
        }
        for rank in range(3)
    ]


def _row(m: int, *, k: int = 8576, offset: float = 0.0) -> dict:
    return {
        "m": m,
        "k": k,
        "n": 5120,
        "dtype": "bfloat16",
        "weight_format": "ND",
        "requested_weight_format": "ND",
        "is_trans_b": True,
        "activation_layout": "contiguous",
        "weight_layout": "contiguous",
        "residual_layout": "contiguous",
        "gamma_layout": "contiguous",
        "baseline_latency_ms": [1.0 + offset, 1.1 + offset, 1.2 + offset],
        "fused_latency_ms": [0.3 + offset, 0.4 + offset, 0.5 + offset],
        "max_abs_norm": 0.01 + offset / 100.0,
        "norm_atol": 0.05,
        "norm_rtol": 0.0,
        "max_scaled_norm": 0.2 + offset / 100.0,
        "max_abs_added": 0.001 + offset / 1000.0,
        "added_atol": 0.01,
        "added_rtol": 0.0,
        "max_scaled_added": 0.1 + offset / 100.0,
        "changed_input_replays": 2,
        "changed_input_delta": 0.015625,
        "changed_input_max_abs_norm": 0.01 + offset / 100.0,
        "changed_input_max_abs_added": 0.001 + offset / 1000.0,
        "changed_input_max_scaled_norm": 0.2 + offset / 100.0,
        "changed_input_max_scaled_added": 0.1 + offset / 100.0,
        "projection_reduction_diagnostics_by_rank": _projection_diagnostics(m),
    }


def _document(layer: int, rows: list[dict], *, projection: str = "down") -> dict:
    return {
        "schema_version": 1,
        "status": "measured",
        "metadata": {
            "operator": "tp3_matmul_allreduce_add_rmsnorm",
            "standalone_chained_flush": True,
            "hardware": "Ascend910B3",
            "tensor_parallel_size": 3,
            "source_sha256": mc2_source_sha256(),
            "rms_norm_epsilon": 1e-6,
            "runtime_binding": {
                "hccl_deterministic": "true",
                "hccl_op_expansion_mode": "AIV",
                "reduction_mode": "global_01",
                "cann_version": "9.0.0",
                "hccl_version": "9.0.0",
                "vendor_payload_sha256": _TEST_VENDOR_SHA256,
                "opapi_symbol_provider_sha256": {
                    symbol: ("e" * 64 if index < 2 else None)
                    for index, symbol in enumerate(mc2_module._MC2_OPAPI_SYMBOLS)
                },
                "adapter_binary_sha256": _TEST_ADAPTER_SHA256,
                "tp_rank_device_mapping": ["5", "6", "7"],
            },
            "measurement_scope": "captured_operator_replay",
            "input_source": f"/captures/{projection}/layer-{layer}/m100",
        },
        "latency_percentile": 95.0,
        "minimum_samples": 3,
        "minimum_speedup": 1.02,
        "measurements": rows,
    }


def _write(path: Path, document: dict) -> Path:
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def _qualification_run(
    root: Path,
    name: str,
    projection_rows: dict[str, list[int]],
    *,
    vendor_sha256: str = _TEST_VENDOR_SHA256,
) -> Path:
    run_dir = root / name
    run_dir.mkdir()
    normalized = {
        kind: sorted(projection_rows.get(kind, []))
        for kind in ("attention", "down")
    }
    projection_k = {"attention": 3072, "down": 8576}
    for kind, rows in normalized.items():
        for row in rows:
            for layer in (0, 31, 63):
                path = (
                    run_dir
                    / "measurements"
                    / kind
                    / f"layer-{layer}"
                    / f"m{row}"
                    / "profile.json"
                )
                path.parent.mkdir(parents=True)
                document = _document(
                    layer,
                    [_row(row, k=projection_k[kind])],
                    projection=kind,
                )
                document["metadata"]["input_source"] = (
                    f"/captures/{name}/{kind}/layer-{layer}/m{row}"
                )
                document["metadata"]["runtime_binding"][
                    "vendor_payload_sha256"
                ] = vendor_sha256
                _write(path, document)
                measurement = document["measurements"][0]
                _write(
                    path.with_name("raw.json"),
                    {
                        "schema_version": 1,
                        "status": "measured",
                        "metadata": {
                            "measurement_scope": (
                                "graph_native_matmul_plus_rank3_epilogue"
                            ),
                            "execution_mode": "graph",
                            "operator": "torch.ops._C_ascend.allreduce_add_rmsnorm_chained",
                            "chained_flush": True,
                            "operations_per_graph": 1,
                            "tensor_parallel_size": 3,
                            "source_sha256": document["metadata"]["source_sha256"],
                            "input_source": document["metadata"]["input_source"],
                        },
                        "shape": {
                            "m": row,
                            "k": projection_k[kind],
                            "n": 5120,
                        },
                        "correctness": {
                            "exact_all_ranks": True,
                            "changed_input_exact_all_ranks": True,
                            "changed_input_replays": measurement[
                                "changed_input_replays"
                            ],
                            "changed_input_delta": measurement["changed_input_delta"],
                        },
                        "samples": {
                            "baseline_latency_ms": measurement[
                                "baseline_latency_ms"
                            ],
                            "fused_latency_ms": measurement["fused_latency_ms"],
                        },
                    },
                )

    intersection = sorted(set(normalized["attention"]).intersection(normalized["down"]))
    source_document = {
        "schema_version": 1,
        "kind": "specslo_mc2_projection_rows",
        "stage": "per-layer-qualification",
        "execution_mode": "graph",
        "operations_per_graph": 1,
        "qualification_layers": [0, 31, 63],
        "projection_k": projection_k,
        "projection_rows": normalized,
        "intersection_rows": intersection,
    }
    source_path = _write(
        run_dir / "qualified-projection-rows.json",
        source_document,
    )
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    _write(
        run_dir / "production-projection-rows.json",
        {
            **source_document,
            "stage": "conservative-cross-layer-production",
            "source_manifest": str(source_path.resolve()),
            "source_manifest_sha256": source_sha256,
        },
    )
    return run_dir


def _legacy_qualification_run(
    root: Path,
    name: str,
    rows: list[int],
) -> Path:
    """Create the old intersection-only finalized layout."""

    run_dir = _qualification_run(
        root,
        name,
        {"attention": rows, "down": rows},
    )
    (run_dir / "qualified-projection-rows.json").unlink()
    (run_dir / "production-projection-rows.json").unlink()
    groups = []
    for kind in ("attention", "down"):
        for row in rows:
            groups.append(
                {
                    "m": row,
                    "kind": kind,
                    "status": "pass",
                    "error": None,
                    "inputs": [
                        str(
                            (
                                run_dir
                                / "measurements"
                                / kind
                                / f"layer-{layer}"
                                / f"m{row}"
                                / "profile.json"
                            ).resolve()
                        )
                        for layer in (0, 31, 63)
                    ],
                }
            )
    _write(
        run_dir / "cross-layer-screen.json",
        {
            "schema_version": 1,
            "kind": "specslo_mc2_conservative_cross_layer_screen",
            "candidate_rows": rows,
            "qualified_rows": rows,
            "groups": groups,
        },
    )
    (run_dir / "qualified-rows.txt").write_text(
        " ".join(str(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _native_epilogue_document(
    layer: int,
    rows: list[dict],
    *,
    projection: str,
) -> dict:
    document = _document(layer, rows, projection=projection)
    document["metadata"]["operator"] = mc2_module.MC2_TP3_NATIVE_EPILOGUE_OPERATOR
    symbols = mc2_module._mc2_opapi_symbols(
        mc2_module.MC2_TP3_NATIVE_EPILOGUE_OPERATOR
    )
    document["metadata"]["runtime_binding"]["opapi_symbol_provider_sha256"] = {
        symbol: (
            None
            if symbol in mc2_module._MC2_OPTIONAL_OPAPI_SYMBOLS
            else "e" * 64
        )
        for symbol in symbols
    }
    return document


def _chain_report(
    path: Path,
    *,
    row_count: int,
    attention_source: str,
    down_source: str,
    source_sha256: str | None = None,
) -> Path:
    zero_errors = {
        "attention_norm": 0.0,
        "attention_added": 0.0,
        "down_norm": 0.0,
        "down_added": 0.0,
        "chain_state": 0.0,
    }
    document = {
        "schema_version": 1,
        "status": "measured",
        "execution_mode": "graph",
        "tensor_parallel_size": 3,
        "chain_scope": "whole_model",
        "layer_count": 64,
        "operations_per_chain": 128,
        "row_count": row_count,
        "source_sha256": source_sha256 or mc2_source_sha256(),
        "tp_rank_device_mapping": ["5", "6", "7"],
        "attention_input_source": attention_source,
        "down_input_source": down_source,
        "correctness": {
            "initial": {
                "fully_flushed": {
                    "exact_all_ranks": True,
                    "max_abs": zero_errors,
                },
                "deferred_chain": {
                    "exact_all_ranks": True,
                    "max_abs": zero_errors,
                },
            },
            "changed_input_exact_all_ranks": {
                "fully_flushed": True,
                "deferred_chain": True,
            },
            "changed_input_max_abs": {
                "fully_flushed": zero_errors,
                "deferred_chain": zero_errors,
            },
            "changed_input_replays": 8,
            "changed_input_output_delta": 0.125,
            "final_chain_state_zero_all_ranks": True,
        },
        "samples": {
            "fully_flushed_latency_ms": [1.0, 1.0, 1.0],
            "deferred_chain_latency_ms": [0.8, 0.8, 0.8],
        },
        "summary": {
            "deferred_vs_fully_flushed_p50": 1.25,
            "deferred_vs_fully_flushed_p95": 1.25,
        },
    }
    return _write(path, document)


def test_aggregate_uses_real_conservative_arrays_and_max_errors(tmp_path):
    layer0 = _document(0, [_row(100, offset=0.0), _row(64, offset=0.2)])
    layer31 = _document(31, [_row(100, offset=0.3), _row(64, offset=-0.1)])
    layer63 = _document(63, [_row(100, offset=-0.2), _row(64, offset=0.1)])
    paths = [
        _write(tmp_path / "layer-0.json", layer0),
        _write(tmp_path / "layer-31.json", layer31),
        _write(tmp_path / "layer-63.json", layer63),
    ]

    result = aggregate_mc2_profiles(paths)

    profile = normalize_mc2_profile(result)
    assert len(profile.entries) == 2
    rows = {row["m"]: row for row in result["measurements"]}
    # M100: layer 63 has the fastest real baseline and layer 31 the slowest
    # real fused samples. The arrays are selected whole, never pooled/padded.
    assert rows[100]["baseline_latency_ms"] == layer63["measurements"][0]["baseline_latency_ms"]
    assert rows[100]["fused_latency_ms"] == layer31["measurements"][0]["fused_latency_ms"]
    assert len(rows[100]["baseline_latency_ms"]) == 3
    assert len(rows[100]["fused_latency_ms"]) == 3
    assert rows[100]["max_abs_norm"] == layer31["measurements"][0]["max_abs_norm"]
    assert rows[100]["changed_input_max_scaled_added"] == layer31["measurements"][0]["changed_input_max_scaled_added"]
    assert "projection_reduction_diagnostics_by_rank" not in rows[100]

    provenance = result["metadata"]["aggregation_provenance"]
    assert [item["path"] for item in provenance["inputs"]] == [str(path.resolve()) for path in paths]
    for path, item in zip(paths, provenance["inputs"]):
        assert item["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert rows[100]["aggregation_provenance"]["baseline_latency_source"]["path"] == str(paths[2].resolve())
    assert rows[100]["aggregation_provenance"]["fused_latency_source"]["path"] == str(paths[1].resolve())


@pytest.mark.parametrize(
    ("location", "field", "value", "message"),
    (
        ("top", "schema_version", 2, "schema_version=1"),
        ("metadata", "operator", "different", "operator"),
        (
            "metadata",
            "standalone_chained_flush",
            False,
            "standalone_chained_flush",
        ),
        ("metadata", "hardware", "different", "hardware"),
        ("metadata", "tensor_parallel_size", 4, "tensor_parallel_size"),
        ("metadata", "source_sha256", "0" * 64, "source_sha256"),
        ("runtime", "cann_version", "different", "runtime_binding"),
        ("runtime", "vendor_payload_sha256", "b" * 64, "runtime_binding"),
        ("runtime", "adapter_binary_sha256", "d" * 64, "runtime_binding"),
        ("top", "latency_percentile", 50.0, "latency_percentile"),
        ("top", "minimum_samples", 4, "minimum_samples"),
        ("top", "minimum_speedup", 1.1, "minimum_speedup"),
    ),
)
def test_aggregate_fails_closed_on_identity_or_gate_mismatch(tmp_path, location, field, value, message):
    first = _document(0, [_row(100)])
    second = _document(31, [_row(100, offset=0.1)])
    if location == "metadata":
        second["metadata"][field] = value
    elif location == "runtime":
        second["metadata"]["runtime_binding"][field] = value
    else:
        second[field] = value

    paths = [_write(tmp_path / "first.json", first), _write(tmp_path / "second.json", second)]
    with pytest.raises(ValueError, match=message):
        aggregate_mc2_profiles(paths)


def test_aggregate_rejects_duplicate_shape_inside_one_input(tmp_path):
    first = _document(0, [_row(100), deepcopy(_row(100))])
    second = _document(31, [_row(100, offset=0.1)])
    paths = [_write(tmp_path / "first.json", first), _write(tmp_path / "second.json", second)]

    with pytest.raises(ValueError, match="[Dd]uplicate MC2 measurement shape|duplicate shape"):
        aggregate_mc2_profiles(paths)


def test_aggregate_rejects_missing_shape_in_any_layer(tmp_path):
    first = _document(0, [_row(64), _row(100)])
    second = _document(31, [_row(100, offset=0.1)])
    paths = [_write(tmp_path / "first.json", first), _write(tmp_path / "second.json", second)]

    with pytest.raises(ValueError, match="incomplete shape set.*missing"):
        aggregate_mc2_profiles(paths)


def test_aggregate_rejects_copied_evidence_and_reused_real_input(tmp_path):
    first = _document(0, [_row(100)])
    first_path = _write(tmp_path / "first.json", first)
    copied_path = _write(tmp_path / "copied.json", first)
    with pytest.raises(ValueError, match="duplicate content"):
        aggregate_mc2_profiles([first_path, copied_path])

    second = _document(31, [_row(100, offset=0.1)])
    second["metadata"]["input_source"] = first["metadata"]["input_source"]
    second_path = _write(tmp_path / "second.json", second)
    with pytest.raises(ValueError, match="reuse input_source"):
        aggregate_mc2_profiles([first_path, second_path])


def test_aggregate_rejects_inconsistent_row_configuration(tmp_path):
    first = _document(0, [_row(100)])
    second = _document(31, [_row(100, offset=0.1)])
    second["measurements"][0]["changed_input_replays"] = 0
    paths = [_write(tmp_path / "first.json", first), _write(tmp_path / "second.json", second)]

    with pytest.raises(ValueError, match="changed_input_replays"):
        aggregate_mc2_profiles(paths)


def test_reusable_production_validator_accepts_complete_measurement():
    validate_mc2_measurement_for_production(_document(0, [_row(100)]), source="unit-test")


@pytest.mark.parametrize(
    "field",
    (
        "max_scaled_norm",
        "max_scaled_added",
        "changed_input_max_scaled_norm",
        "changed_input_max_scaled_added",
    ),
)
def test_aggregate_rejects_any_scaled_numerical_gate_failure_before_stale_source(
    tmp_path,
    field,
):
    first = _document(0, [_row(100)])
    second = _document(31, [_row(100, offset=0.1)])
    first["measurements"][0][field] = 1.01
    # Both artifacts deliberately carry an old but mutually consistent source
    # hash. The numerical failure must be reported before source identity.
    first["metadata"]["source_sha256"] = second["metadata"]["source_sha256"] = "0" * 64
    paths = [_write(tmp_path / "first.json", first), _write(tmp_path / "second.json", second)]

    with pytest.raises(ValueError, match=rf"{field}=1.01.*numerical gate"):
        aggregate_mc2_profiles(paths)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("changed_input_replays", 1, "changed_input_replays.*>=2"),
        ("changed_input_delta", 0.0, "changed_input_delta.*nonzero"),
    ),
)
def test_aggregate_rejects_degenerate_changed_input_gate(tmp_path, field, value, message):
    first = _document(0, [_row(100)])
    second = _document(31, [_row(100, offset=0.1)])
    first["measurements"][0][field] = value
    second["measurements"][0][field] = value
    paths = [_write(tmp_path / "first.json", first), _write(tmp_path / "second.json", second)]

    with pytest.raises(ValueError, match=message):
        aggregate_mc2_profiles(paths)


def test_aggregate_rejects_incomplete_changed_input_gate(tmp_path):
    first = _document(0, [_row(100)])
    second = _document(31, [_row(100, offset=0.1)])
    first["measurements"][0].pop("changed_input_max_scaled_added")
    second["measurements"][0].pop("changed_input_max_scaled_added")
    paths = [_write(tmp_path / "first.json", first), _write(tmp_path / "second.json", second)]

    with pytest.raises(ValueError, match="lacks required scaled numerical gates"):
        aggregate_mc2_profiles(paths)


def test_aggregate_rejects_changed_input_error_not_folded_into_main_gate(tmp_path):
    first = _document(0, [_row(100)])
    second = _document(31, [_row(100, offset=0.1)])
    for document in (first, second):
        row = document["measurements"][0]
        row["max_scaled_added"] = 0.1
        row["changed_input_max_scaled_added"] = 0.2
    paths = [_write(tmp_path / "first.json", first), _write(tmp_path / "second.json", second)]

    with pytest.raises(ValueError, match="max_scaled_added does not include"):
        aggregate_mc2_profiles(paths)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "complete projection diagnostics"),
        ("duplicate_rank", "duplicate rank"),
        ("non_exact", "not bit-exact"),
        ("mismatch_count", "reports mismatches"),
        ("hash_mismatch", "inconsistent output hashes"),
    ),
)
def test_aggregate_rejects_incomplete_or_nonexact_projection_diagnostics(
    tmp_path,
    mutation,
    message,
):
    first = _document(0, [_row(100)])
    second = _document(31, [_row(100, offset=0.1)])
    for document in (first, second):
        diagnostics = document["measurements"][0]["projection_reduction_diagnostics_by_rank"]
        if mutation == "missing":
            diagnostics.pop()
        elif mutation == "duplicate_rank":
            diagnostics[-1]["rank"] = 1
        elif mutation == "non_exact":
            diagnostics[0]["exact_match"] = False
        elif mutation == "mismatch_count":
            diagnostics[0]["mismatch_count"] = 1
        else:
            diagnostics[0]["actual_sha256"] = "b" * 64
    paths = [_write(tmp_path / "first.json", first), _write(tmp_path / "second.json", second)]

    with pytest.raises(ValueError, match=message):
        aggregate_mc2_profiles(paths)


def test_aggregate_rejects_input_p95_latency_gate_failure(tmp_path):
    first = _document(0, [_row(100)])
    second = _document(31, [_row(100, offset=0.1)])
    first["measurements"][0]["fused_latency_ms"] = [1.0, 1.1, 1.2]
    paths = [_write(tmp_path / "first.json", first), _write(tmp_path / "second.json", second)]

    with pytest.raises(ValueError, match="p95 latency gate failed"):
        aggregate_mc2_profiles(paths)


def test_aggregate_rejects_cross_layer_envelope_latency_failure(tmp_path):
    first = _document(0, [_row(100)])
    second = _document(31, [_row(100, offset=0.1)])
    first_row = first["measurements"][0]
    second_row = second["measurements"][0]
    first_row["baseline_latency_ms"] = [1.0, 1.0, 1.0]
    first_row["fused_latency_ms"] = [0.5, 0.5, 0.5]
    second_row["baseline_latency_ms"] = [2.0, 2.0, 2.0]
    second_row["fused_latency_ms"] = [1.5, 1.5, 1.5]
    paths = [_write(tmp_path / "first.json", first), _write(tmp_path / "second.json", second)]

    with pytest.raises(ValueError, match="conservative cross-layer envelope.*p95 latency gate failed"):
        aggregate_mc2_profiles(paths)


def test_aggregate_requires_p95_even_when_inputs_agree(tmp_path):
    first = _document(0, [_row(100)])
    second = _document(31, [_row(100, offset=0.1)])
    first["latency_percentile"] = second["latency_percentile"] = 50.0
    paths = [_write(tmp_path / "first.json", first), _write(tmp_path / "second.json", second)]

    with pytest.raises(ValueError, match="latency_percentile=95"):
        aggregate_mc2_profiles(paths)


def test_aggregate_reports_missing_required_gate(tmp_path):
    first = _document(0, [_row(100)])
    second = _document(31, [_row(100, offset=0.1)])
    second.pop("minimum_speedup")
    paths = [_write(tmp_path / "first.json", first), _write(tmp_path / "second.json", second)]

    with pytest.raises(ValueError, match="missing required field minimum_speedup"):
        aggregate_mc2_profiles(paths)


def test_repeated_cli_input_builds_attention_down_union_for_native_config(tmp_path):
    attention_paths = [
        _write(
            tmp_path / f"attention-layer-{layer}.json",
            _document(layer, [_row(100, k=3072, offset=offset)], projection="attention"),
        )
        for layer, offset in ((0, 0.0), (31, 0.1), (63, -0.1))
    ]
    down_paths = [
        _write(
            tmp_path / f"down-layer-{layer}.json",
            _document(layer, [_row(100, k=8576, offset=offset)], projection="down"),
        )
        for layer, offset in ((0, 0.0), (31, 0.2), (63, -0.2))
    ]
    arguments = _parser().parse_args(
        [
            "--input",
            *(str(path) for path in attention_paths),
            "--input",
            *(str(path) for path in down_paths),
            "--output",
            str(tmp_path / "combined.json"),
        ]
    )

    result = aggregate_mc2_profile_groups(arguments.input)
    output = tmp_path / "combined.json"
    output.write_text(json.dumps(result), encoding="utf-8")
    config = NativePearlConfig(
        "draft",
        "target",
        1,
        3,
        4,
        512,
        32,
        enable_mc2=True,
        mc2_profile=str(output),
    )

    assert set(config.mc2_profile.entries) == {
        (100, 3072, 5120, "bfloat16", "ND", True),
        (100, 8576, 5120, "bfloat16", "ND", True),
    }
    provenance = config.mc2_profile.metadata["aggregation_provenance"]
    assert config.mc2_profile.metadata["standalone_chained_flush"] is True
    assert provenance["method"] == "conservative_exact_shape_union_v1"
    assert len(provenance["inputs"]) == 6
    assert len(provenance["groups"]) == 2


def test_attention_down_union_rejects_duplicate_exact_shape(tmp_path):
    first_group = [_write(tmp_path / f"first-{layer}.json", _document(layer, [_row(100)])) for layer in (0, 31)]
    second_group = [
        _write(
            tmp_path / f"second-{layer}.json",
            _document(layer + 100, [_row(100)], projection="attention"),
        )
        for layer in (0, 31)
    ]

    with pytest.raises(ValueError, match="duplicate shape"):
        aggregate_mc2_profile_groups([first_group, second_group])


def test_qualification_run_union_preserves_projection_specific_rows(tmp_path):
    small = _qualification_run(
        tmp_path,
        "small",
        {"attention": [4], "down": [8]},
    )
    large = _qualification_run(
        tmp_path,
        "large",
        {"attention": [60], "down": [64]},
    )

    profile, report = aggregate_qualification_runs((small, large))

    assert {
        (row["m"], row["k"])
        for row in profile["measurements"]
    } == {(4, 3072), (8, 8576), (60, 3072), (64, 8576)}
    assert report["status"] == "pass"
    assert report["execution_mode"] == "graph"
    assert report["qualification_layers"] == [0, 31, 63]
    assert report["shape_count"] == 4
    assert len(report["runs"]) == 2
    assert len(report["groups"]) == 4


def test_qualification_run_union_accepts_only_legacy_pass_intersection(tmp_path):
    legacy = _legacy_qualification_run(tmp_path, "legacy-large", [40, 44])
    projection_aware = _qualification_run(
        tmp_path,
        "projection-aware-small",
        {"attention": [4], "down": [8]},
    )

    profile, report = aggregate_qualification_runs((legacy, projection_aware))

    assert {
        (row["m"], row["k"])
        for row in profile["measurements"]
    } == {
        (4, 3072),
        (8, 8576),
        (40, 3072),
        (40, 8576),
        (44, 3072),
        (44, 8576),
    }
    assert [run["manifest_format"] for run in report["runs"]] == [
        "legacy-intersection-v1",
        "projection-aware-v1",
    ]


@pytest.mark.parametrize("corruption", ("rows-file", "one-projection-failed"))
def test_qualification_run_union_rejects_unbound_legacy_intersection(
    tmp_path,
    corruption,
):
    legacy = _legacy_qualification_run(tmp_path, "legacy-corrupt", [40])
    if corruption == "rows-file":
        (legacy / "qualified-rows.txt").write_text("44\n", encoding="utf-8")
        message = "disagrees"
    else:
        report_path = legacy / "cross-layer-screen.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        down = next(group for group in report["groups"] if group["kind"] == "down")
        down["status"] = "fail"
        down["error"] = "conservative p95 gate failed"
        _write(report_path, report)
        message = "two-projection pass intersection"

    with pytest.raises(ValueError, match=message):
        collect_qualification_groups((legacy,))


def test_qualification_run_union_rejects_missing_layer_evidence(tmp_path):
    run_dir = _qualification_run(
        tmp_path,
        "missing-layer",
        {"attention": [4], "down": [8]},
    )
    missing = run_dir / "measurements/attention/layer-63/m4/profile.json"
    missing.unlink()

    with pytest.raises(ValueError, match="missing layer evidence.*layer-63"):
        collect_qualification_groups((run_dir,))


def test_qualification_run_union_rejects_non_graph_manifest(tmp_path):
    run_dir = _qualification_run(
        tmp_path,
        "eager-manifest",
        {"attention": [4], "down": [8]},
    )
    manifest_path = run_dir / "production-projection-rows.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["execution_mode"] = "eager"
    _write(manifest_path, manifest)

    with pytest.raises(ValueError, match="not ACLGraph evidence"):
        collect_qualification_groups((run_dir,))


def test_qualification_run_union_rejects_non_graph_raw_evidence(tmp_path):
    run_dir = _qualification_run(
        tmp_path,
        "eager-raw-evidence",
        {"attention": [4], "down": []},
    )
    raw_path = run_dir / "measurements/attention/layer-31/m4/raw.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw["metadata"]["execution_mode"] = "eager"
    _write(raw_path, raw)

    with pytest.raises(ValueError, match="not graph/one-operation evidence"):
        collect_qualification_groups((run_dir,))


def test_qualification_run_union_rejects_duplicate_shape_across_runs(tmp_path):
    first = _qualification_run(
        tmp_path,
        "first",
        {"attention": [4], "down": [8]},
    )
    second = _qualification_run(
        tmp_path,
        "second",
        {"attention": [4], "down": [12]},
    )

    with pytest.raises(ValueError, match="duplicate exact shape attention/M4"):
        collect_qualification_groups((first, second))


def test_qualification_run_union_rejects_runtime_binding_mismatch(tmp_path):
    first = _qualification_run(
        tmp_path,
        "first-binding",
        {"attention": [4], "down": []},
    )
    second = _qualification_run(
        tmp_path,
        "second-binding",
        {"attention": [], "down": [8]},
        vendor_sha256="b" * 64,
    )

    with pytest.raises(ValueError, match="runtime_binding"):
        aggregate_qualification_runs((first, second))


def test_qualification_run_union_cli_writes_profile_report_and_sha(
    monkeypatch,
    tmp_path,
):
    first = _qualification_run(
        tmp_path,
        "first-cli",
        {"attention": [4], "down": []},
    )
    second = _qualification_run(
        tmp_path,
        "second-cli",
        {"attention": [], "down": [8]},
    )
    output = tmp_path / "union/profile.json"
    report = tmp_path / "union/report.json"
    checksum = tmp_path / "union/profile.sha256"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_specslo_mc2_qualification_runs.py",
            "--run-dir",
            str(first),
            "--run-dir",
            str(second),
            "--output",
            str(output),
            "--report",
            str(report),
            "--sha256-output",
            str(checksum),
        ],
    )

    assert qualification_union_main() == 0
    profile_bytes = output.read_bytes()
    profile_sha256 = hashlib.sha256(profile_bytes).hexdigest()
    assert checksum.read_text(encoding="utf-8") == f"{profile_sha256}  {output.resolve()}\n"
    audit = json.loads(report.read_text(encoding="utf-8"))
    assert audit["status"] == "pass"
    assert audit["profile_sha256"] == profile_sha256
    assert audit["shape_count"] == 2


def test_qualification_run_union_cli_records_rejected_missing_layer(
    monkeypatch,
    tmp_path,
):
    run_dir = _qualification_run(
        tmp_path,
        "failed-cli",
        {"attention": [4], "down": [8]},
    )
    (run_dir / "measurements/down/layer-0/m8/profile.json").unlink()
    output = tmp_path / "failed/profile.json"
    report = tmp_path / "failed/report.json"
    checksum = tmp_path / "failed/profile.sha256"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_specslo_mc2_qualification_runs.py",
            "--run-dir",
            str(run_dir),
            "--output",
            str(output),
            "--report",
            str(report),
            "--sha256-output",
            str(checksum),
        ],
    )

    assert qualification_union_main() == 1
    assert not output.exists()
    assert not checksum.exists()
    audit = json.loads(report.read_text(encoding="utf-8"))
    assert audit["status"] == "fail"
    assert "missing layer evidence" in audit["error"]


def test_multishape_aggregate_attaches_only_explicit_chain_rows(tmp_path):
    attention_paths = [
        _write(
            tmp_path / f"attention-{layer}.json",
            _native_epilogue_document(
                layer,
                [_row(100, k=3072, offset=offset)],
                projection="attention",
            ),
        )
        for layer, offset in ((0, 0.0), (31, 0.1), (63, -0.1))
    ]
    down_paths = [
        _write(
            tmp_path / f"down-{layer}.json",
            _native_epilogue_document(
                layer,
                [_row(100, k=8576, offset=offset)],
                projection="down",
            ),
        )
        for layer, offset in ((0, 0.0), (31, 0.2), (63, -0.2))
    ]
    chain = _chain_report(
        tmp_path / "chain-m100.json",
        row_count=100,
        attention_source="/captures/attention/layer-31/m100",
        down_source="/captures/down/layer-31/m100",
    )

    result = aggregate_mc2_profile_groups(
        (attention_paths, down_paths),
        deferred_chain_paths=(chain,),
        minimum_chain_samples=3,
        minimum_chain_speedup=1.0,
    )

    evidence = result["metadata"]["deferred_read_done_chain"]
    assert evidence["source_sha256"] == result["metadata"]["source_sha256"]
    assert evidence["qualified_row_counts"] == [100]
    assert evidence["layer_count"] == 64
    assert evidence["operations_per_chain"] == 128
    assert evidence["reports"][0]["row_count"] == 100
    normalize_mc2_profile(result)


@pytest.mark.parametrize(
    ("row_count", "source_sha256", "message"),
    (
        (64, None, "row_count 64 is absent"),
        (100, "0" * 64, "source SHA"),
    ),
)
def test_multishape_aggregate_rejects_unbound_chain_evidence(
    tmp_path,
    row_count,
    source_sha256,
    message,
):
    attention_paths = [
        _write(
            tmp_path / f"attention-{layer}.json",
            _native_epilogue_document(
                layer,
                [_row(100, k=3072)],
                projection="attention",
            ),
        )
        for layer in (0, 31)
    ]
    down_paths = [
        _write(
            tmp_path / f"down-{layer}.json",
            _native_epilogue_document(
                layer,
                [_row(100, k=8576)],
                projection="down",
            ),
        )
        for layer in (0, 31)
    ]
    chain = _chain_report(
        tmp_path / "chain.json",
        row_count=row_count,
        attention_source="/captures/attention/layer-31/m100",
        down_source="/captures/down/layer-31/m100",
        source_sha256=source_sha256,
    )

    with pytest.raises(ValueError, match=message):
        aggregate_mc2_profile_groups(
            (attention_paths, down_paths),
            deferred_chain_paths=(chain,),
            minimum_chain_samples=3,
        )
