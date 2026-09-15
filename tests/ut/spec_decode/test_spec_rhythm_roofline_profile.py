# SPDX-License-Identifier: Apache-2.0

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from vllm_ascend.spec_decode.pearl.roofline import runtime_source_fingerprints

_PATH = Path(__file__).resolve().parents[3] / "examples/profile_specslo_tree_roofline.py"
_SPEC = importlib.util.spec_from_file_location("specslo_roofline_example", _PATH)
profiler = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(profiler)


def _sample(counts, latency=10.0, context=512, **kwargs):
    return {
        "kind": "packed_tree" if any(counts) else "ar",
        "attention_backend": "fused_infer_attention_tree_v1" if any(counts) else "paged_attention_v1",
        "batch_size": 8,
        "verification_requests": len(counts),
        "context_len": context,
        "candidate_counts": counts,
        "physical_query_tokens": len(counts) + sum(counts),
        "latency_ms": [latency] * 3,
        "warmup_iterations": 2,
        "timed_graph_captures": 0,
        "timed_graph_replays": 3,
        **kwargs,
    }


def _document(rows=None):
    return {
        "metadata": {
            "model": "fixture-target-model",
            "target_tensor_parallel_size": 3,
            "execution_mode": "graph",
            "hardware": "fixture-device",
            "measurement_scope": "target_forward",
            "verification_attention_backends": ["fused_infer_attention_tree_v1"],
            "ar_attention_backend": "paged_attention_v1",
            "ar_comparator": "standard_decode_full_active_batch",
            **runtime_source_fingerprints(),
            "tree_fia_sparse_mode": 1,
            "tree_fia_inner_precise": 1,
            "max_model_len": 1024,
            "tree_width": 2,
            "tree_depth": 2,
        },
        "measurements": rows
        or [
            _sample([0] * 8, 10.0),
            _sample([1, 1], 10.5),
            _sample([2, 2], 10.8),
            _sample([3, 3], 12.0),
        ],
    }


def test_profile_table_uses_largest_measured_budget_before_latency_cliff():
    table = profiler.build_roofline_table(_document())
    assert table["roofline"] == {"8:1": 4}
    assert table["metadata"]["target_tensor_parallel_size"] == 3
    assert table["evidence"][0]["ar_requests"] == 8
    assert table["evidence"][0]["verification_requests"] == 2
    assert table["evidence"][0]["verification_limit_ms"] == pytest.approx(11.0)


@pytest.mark.parametrize(
    "field",
    [
        "verification_attention_backends",
        "ar_attention_backend",
        "ar_comparator",
        "native_engine_source_sha256",
        "native_model_source_sha256",
        "native_graph_source_sha256",
        "tree_source_sha256",
        "tree_fia_sparse_mode",
        "tree_fia_inner_precise",
        "max_model_len",
        "tree_width",
        "tree_depth",
    ],
)
def test_old_full_samples_without_backend_provenance_are_rejected(field):
    document = _document()
    document["metadata"].pop(field)
    with pytest.raises(ValueError, match=field):
        profiler.build_roofline_table(document)


def test_dense_sample_cannot_be_labeled_as_fia_by_top_level_metadata():
    document = _document()
    document["measurements"][1]["attention_backend"] = "dense_sdpa_tree_v1"
    with pytest.raises(ValueError, match="actual matching attention_backend"):
        profiler.build_roofline_table(document)


def test_half_sized_ar_cannot_masquerade_as_standard_active_batch_decode():
    document = _document()
    document["measurements"][0] = _sample([0, 0], 10.0)
    with pytest.raises(ValueError, match="every active batch request"):
        profiler.build_roofline_table(document)


def test_profile_rejects_candidate_shape_larger_than_bound_tree_topology():
    document = _document()
    document["measurements"].append(_sample([5, 1], 10.4))
    with pytest.raises(ValueError, match=r"tree_width\*tree_depth"):
        profiler.build_roofline_table(document)


def test_profile_cannot_select_lucky_large_point_beyond_first_failure():
    document = _document()
    document["measurements"].append(_sample([4, 4], 10.4))
    table = profiler.build_roofline_table(document)
    assert table["roofline"] == {"8:1": 4}
    last = table["evidence"][0]["candidate_sweep"][-1]
    assert last["within_limit"]
    assert not last["reachable_before_first_violation"]


def test_same_budget_uses_worst_measured_candidate_distribution():
    document = _document()
    document["measurements"].append(_sample([1, 3], 12.0))
    assert profiler.build_roofline_table(document)["roofline"] == {"8:1": 2}


def test_exhaustive_protocol_rejects_a_passing_budget_with_a_missing_histogram():
    document = _document(
        [
            _sample([0] * 8, 10.0),
            _sample([1, 1], 10.0),
            _sample([1, 2], 10.0),
            _sample([2, 2], 10.0),
        ]
    )
    document["metadata"]["candidate_distribution_protocol"] = (
        "all_canonical_histograms_until_first_violation"
    )
    with pytest.raises(ValueError, match="lacks all canonical"):
        profiler.build_roofline_table(document)


def test_exhaustive_protocol_accepts_one_concrete_first_failing_distribution():
    document = _document(
        [
            _sample([0] * 8, 10.0),
            _sample([1, 1], 10.0),
            _sample([1, 2], 10.0),
            _sample([1, 3], 12.0),
        ]
    )
    document["metadata"]["candidate_distribution_protocol"] = (
        "all_canonical_histograms_until_first_violation"
    )
    assert profiler.build_roofline_table(document)["roofline"] == {"8:1": 3}


def test_context_bucket_uses_most_conservative_measured_context():
    document = _document()
    document["measurements"].extend(
        [
            _sample([0] * 8, 10.0, context=256),
            _sample([1, 1], 10.4, context=256),
            _sample([2, 2], 12.0, context=256),
        ]
    )
    assert profiler.build_roofline_table(document)["roofline"] == {"8:1": 2}


def test_profile_requires_full_active_ar_batch_and_context():
    document = _document([_sample([1, 1], 10.4)])
    with pytest.raises(ValueError, match="Missing full-active AR"):
        profiler.build_roofline_table(document)


def test_profile_rejects_padding_mislabeled_as_packed_candidates():
    document = _document()
    document["measurements"][1]["physical_query_tokens"] = 20
    with pytest.raises(ValueError, match="without padding"):
        profiler.build_roofline_table(document)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("latency_ms", [1.0], "at least 3"),
        ("latency_ms", [float("nan")] * 3, "finite"),
        ("timed_graph_captures", 1, "zero timed_graph_captures"),
        ("timed_graph_replays", 0, "timed_graph_replays"),
        ("warmup_iterations", None, "warmup_iterations"),
        ("candidate_counts", [True, 1], "non-negative integers"),
    ],
)
def test_profile_rejects_invalid_measurement_evidence(field, value, message):
    document = _document()
    document["measurements"][1][field] = value
    with pytest.raises(ValueError, match=message):
        profiler.build_roofline_table(document)


def test_no_qualified_candidate_budget_records_target_only_fallback():
    document = _document([_sample([0] * 8, 10.0), _sample([1, 1], 20.0)])
    table = profiler.build_roofline_table(document)
    assert table["roofline"] == {"8:1": 0}
    assert table["target_only_fallback_lookup_keys"] == ["8:1"]
    assert table["unqualified_lookup_keys"] == []


def test_unqualified_context_is_not_hidden_by_success_elsewhere_in_bucket():
    document = _document()
    document["measurements"].extend(
        [
            _sample([0] * 8, 10.0, context=256),
            _sample([1, 1], 20.0, context=256),
            _sample([0] * 8, 10.0, context=1024),
            _sample([1, 1], 10.1, context=1024),
        ]
    )
    table = profiler.build_roofline_table(document)
    assert table["roofline"] == {"8:1": 0, "8:2": 2}
    assert table["target_only_fallback_lookup_keys"] == ["8:1"]
    assert table["unqualified_lookup_keys"] == []


def test_profile_does_not_select_minimum_latency_from_repeats():
    document = _document()
    document["measurements"][2]["latency_ms"] = [5.0, 10.0, 20.0]
    assert profiler.build_roofline_table(document)["roofline"] == {"8:1": 2}


def test_target_forward_measurement_excludes_warmup_and_waits_for_device():
    calls = []
    clocks = iter([0.0, 0.01, 1.0, 1.02, 2.0, 2.03])
    sample = profiler.measure_target_forward(
        lambda: calls.append("forward"),
        lambda: calls.append("sync"),
        iterations=3,
        warmup_iterations=2,
        clock=lambda: next(clocks),
    )
    assert calls[:4] == ["forward", "sync", "forward", "sync"]
    assert calls[4:] == ["sync", "forward", "sync"] * 3
    assert sample["latency_ms"] == pytest.approx([10.0, 20.0, 30.0])
    assert sample["warmup_iterations"] == 2


def test_target_forward_measurement_rejects_timed_graph_capture():
    captures = iter([3, 4])
    clocks = iter([0.0, 0.01])
    with pytest.raises(ValueError, match="Graph capture occurred"):
        profiler.measure_target_forward(
            lambda: None,
            lambda: None,
            iterations=1,
            graph_capture_count=lambda: next(captures),
            clock=lambda: next(clocks),
        )


def test_target_forward_measurement_does_not_claim_unobserved_graph_replay():
    clocks = iter([0.0, 0.01])
    sample = profiler.measure_target_forward(
        lambda: None,
        lambda: None,
        iterations=1,
        clock=lambda: next(clocks),
    )
    assert sample["timed_graph_captures"] is None
    assert sample["timed_graph_replays"] is None


def test_cli_writes_only_from_supplied_measurement_file(tmp_path):
    source = tmp_path / "fixture_measurements.json"
    output = tmp_path / "fixture_roofline.json"
    source.write_text(json.dumps(_document()))
    assert profiler.main(["--measurements", str(source), "--output", str(output)]) == 0
    table = json.loads(output.read_text())
    assert table["roofline"] == {"8:1": 4}
    assert table["metadata"]["model"] == "fixture-target-model"


def test_mixed_measurement_scope_cannot_masquerade_as_target_forward():
    document = copy.deepcopy(_document())
    document["metadata"]["measurement_scope"] = "end_to_end"
    with pytest.raises(ValueError, match="target_forward"):
        profiler.build_roofline_table(document)
