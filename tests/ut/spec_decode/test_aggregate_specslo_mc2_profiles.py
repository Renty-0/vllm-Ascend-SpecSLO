# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for conservative cross-layer MC2 aggregation."""

from __future__ import annotations

import hashlib
import json
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
