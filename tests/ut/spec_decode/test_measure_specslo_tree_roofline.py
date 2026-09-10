# SPDX-License-Identifier: Apache-2.0
"""CPU checks of collector accounting/protocol; not fabricated NPU samples."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from examples import measure_specslo_tree_roofline as collector
from vllm_ascend.spec_decode.pearl.roofline import runtime_source_fingerprints


def _args(*extra):
    return collector._build_parser().parse_args(["--mode", "graph", "--output", "unused.json", *extra])


def test_shape_sweep_keeps_physical_ar_rows_distinct_from_active_batch():
    args = _args(
        "--batch-size",
        "8",
        "--verification-requests",
        "2",
        "--candidate-budgets",
        "2",
        "3",
        "4",
        "--candidate-counts",
        "1,3",
    )
    shapes = collector._shape_counts(args)
    assert shapes == [[0, 0], [1, 1], [1, 2], [1, 3], [2, 2]]
    assert [len(row) + sum(row) for row in shapes] == [2, 4, 5, 6, 6]


def test_default_shape_sweep_measures_every_total_candidate_budget():
    args = _args("--tree-width", "2", "--tree-depth", "2", "--verification-requests", "2")
    shapes = collector._shape_counts(args)
    assert sorted(set(sum(row) for row in shapes)) == [0, 2, 3, 4, 5, 6, 7, 8]
    assert [1, 3] in shapes and [2, 2] in shapes


def test_canonical_sweep_covers_linear_and_branching_histograms_without_permutations():
    assert collector._canonical_count_distributions(4, 4, 6) == [
        [1, 1, 1, 3],
        [1, 1, 2, 2],
    ]


def test_measurement_matrix_compares_one_slot_tree_to_full_active_ar():
    tree_shapes = collector._shape_counts(
        _args("--batch-size", "8", "--verification-requests", "4", "--candidate-budgets", "4", "5")
    )
    measured = collector._measurement_shapes(8, tree_shapes)
    assert measured == [[0] * 8, [1, 1, 1, 1], [1, 1, 1, 2]]


def test_batch_shape_matrix_is_validated_and_sizes_engine_once():
    args = _args(
        "--batch-shape",
        "8:4",
        "--batch-shape",
        "16:8",
        "--batch-shape",
        "64:32",
    )
    assert collector._batch_shapes(args) == [(8, 4), (16, 8), (64, 32)]
    config = collector._engine_config(args)
    assert config.max_num_seqs == 64
    assert config.max_num_batched_tokens >= 32 * args.max_model_len


@pytest.mark.parametrize("raw", ["8", "8:0", "8:9", "a:4", "8:4:2"])
def test_invalid_batch_shape_is_rejected(raw):
    with pytest.raises(ValueError, match="batch-shape|Batch shapes"):
        collector._batch_shapes(_args("--batch-shape", raw))


@pytest.mark.parametrize(
    "extra,match",
    [
        (("--verification-requests", "9"), "must not exceed"),
        (("--candidate-counts", "1,2"), "match physical"),
        (("--candidate-counts", "0,1,1,1"), "match physical"),
        (("--candidate-budgets", "1"), "give each physical"),
        (("--contexts", "1024"), "physical KV slots"),
        (("--warmup-iterations", "1"), "at least two"),
        (("--epsilon-relative", "nan"), "epsilon_relative"),
        (("--latency-percentile", "101"), "latency_percentile"),
    ],
)
def test_collector_rejects_invalid_shapes_before_loading_models(extra, match):
    with pytest.raises(ValueError, match=match):
        collector._shape_counts(_args(*extra))


def test_dry_run_writes_unmeasured_manifest_not_a_fake_profile(tmp_path):
    path = tmp_path / "dry.json"
    assert collector.main(["--mode", "eager", "--output", str(path), "--dry-run"]) == 0
    payload = json.loads(path.read_text())
    assert payload["status"] == "unmeasured_dry_run"
    assert payload["batch_shapes"][0]["candidate_shapes"][0] == [0, 0, 0, 0]
    assert "measurements" not in payload and "roofline" not in payload
    shapes = payload["batch_shapes"][0]["candidate_shapes"]
    assert sorted(set(sum(row) for row in shapes)) == [0, *range(4, 33)]
    assert [1, 1, 1, 3] in shapes and [1, 1, 2, 2] in shapes


def test_collector_percentile_uses_all_repeated_samples():
    assert collector._percentile([1.0, 3.0, 2.0], 95.0) == pytest.approx(2.9)


def test_metadata_names_real_detected_hardware_and_exact_timing_contract():
    metadata = collector._metadata(
        _args(), "Ascend910B2", source_fingerprints=runtime_source_fingerprints()
    )
    assert metadata["hardware"] == "Ascend910B2"
    assert metadata["target_tensor_parallel_size"] == 3
    assert metadata["draft_tensor_parallel_size"] == 1
    assert metadata["execution_mode"] == "graph"
    assert metadata["measurement_scope"] == "target_forward"
    assert metadata["ar_comparator"] == "standard_decode_full_active_batch"
    assert metadata["candidate_distribution_protocol"] == "all_canonical_histograms_until_first_violation"
    assert metadata["output_head"] == "greedy"
    assert metadata["finiteness_guard"] == "spec_rhythm_tree_full_model"
    assert metadata["max_model_len"] == 1024
    assert metadata["tree_width"] == 2
    assert metadata["tree_depth"] == 4
    assert "prefix_prefill" in metadata["timing_excludes"]
    assert "graph_capture" in metadata["timing_excludes"]
    assert "greedy_output_head" in metadata["timing_includes"]
    assert {key: metadata[key] for key in runtime_source_fingerprints()} == runtime_source_fingerprints()
    assert metadata["tree_fia_sparse_mode"] == 1
    assert metadata["tree_fia_inner_precise"] == 2
    assert "verification_attention_backends" not in metadata  # Actual route has not yet been observed.


@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_actual_engine_config_enables_full_guard_and_required_scheduler_flags(mode):
    config = collector._engine_config(_args("--mode", mode))
    assert config.enable_spec_rhythm and config.enable_continuous_batching and config.enable_preemptive_scheduling
    assert config.spec_rhythm_tree_width == 2 and config.spec_rhythm_tree_depth == 4
    assert config.enforce_eager == (mode == "eager")


def test_roofline_prefix_prefill_is_incremental_and_outside_shape_callbacks():
    engine = SimpleNamespace(_run_packed_hidden=Mock())
    collector._prefill_prefix(engine, [[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]], 2)
    assert engine._run_packed_hidden.call_count == 2
    assert engine._run_packed_hidden.call_args_list[0].args == (
        [10, 11, 20, 21],
        [0, 0, 1, 1],
        [0, 1, 0, 1],
    )
    assert engine._run_packed_hidden.call_args_list[1].args == (
        [12, 13, 22, 23],
        [0, 0, 1, 1],
        [2, 3, 2, 3],
    )
    assert all(
        call.kwargs == {"use_aclgraph": False, "use_fused_infer_attention": True}
        for call in engine._run_packed_hidden.call_args_list
    )


def test_merge_preserves_all_real_rows_and_samples_without_relabeling_batch():
    from examples.profile_specslo_tree_roofline import merge_measurement_documents

    rows = [{"batch_size": 1, "latency_ms": [10, 11, 12]}, {"batch_size": 2, "latency_ms": [13, 14, 15]}]
    documents = [{"status": "complete", "metadata": {"model": "same"}, "measurements": [row]} for row in rows]
    merged = merge_measurement_documents(documents)
    assert merged["measurements"] == rows
    assert [row["batch_size"] for row in merged["measurements"]] == [1, 2]


@pytest.mark.parametrize("different", [{"status": "running"}, {"metadata": {"model": "different"}}])
def test_merge_refuses_partial_or_mismatched_experiments(different):
    from examples.profile_specslo_tree_roofline import merge_measurement_documents

    document = {"status": "complete", "metadata": {"model": "same"}, "measurements": []}
    with pytest.raises(ValueError):
        merge_measurement_documents([document, {**document, **different}])


def test_backend_provenance_is_observed_per_shape_and_never_silently_mixed():
    metadata = {}
    collector._record_backend(metadata, {"kind": "ar", "attention_backend": "paged_attention_v1"})
    collector._record_backend(metadata, {"kind": "packed_tree", "attention_backend": "fused_infer_attention_tree_v1"})
    collector._record_backend(metadata, {"kind": "packed_tree", "attention_backend": "dense_sdpa_tree_v1"})
    assert metadata == {
        "ar_attention_backend": "paged_attention_v1",
        "verification_attention_backends": ["dense_sdpa_tree_v1", "fused_infer_attention_tree_v1"],
    }
    with pytest.raises(ValueError, match="different AR attention backends"):
        collector._record_backend(metadata, {"kind": "ar", "attention_backend": "dense_sdpa_tree_v1"})


@pytest.mark.parametrize("status", ["running", "failed", "unmeasured_dry_run"])
def test_incomplete_collector_cannot_be_aggregated_as_a_complete_table(status):
    from examples.profile_specslo_tree_roofline import build_roofline_table

    with pytest.raises(ValueError, match="not a complete roofline measurement"):
        build_roofline_table(
            {"metadata": collector._metadata(_args(), "Ascend910B2"), "status": status, "measurements": []}
        )


def test_per_iteration_slowest_rank_reduction_uses_hccl_int64_not_float64(monkeypatch):
    group = object()

    def reduce(tensor, *, op, group):
        assert tensor.dtype == torch.int64
        assert tensor.tolist() == [1011, 2500, 3999]
        assert op == torch.distributed.ReduceOp.MAX
        assert group is expected_group
        tensor.copy_(torch.tensor([1015, 2500, 4500], dtype=torch.int64))

    expected_group = group
    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    result = collector._reduce_sample_latencies([1.011, 2.5, 3.999], device="cpu", group=group)
    assert result == [1.015, 2.5, 4.5]


@pytest.mark.parametrize("counts", [[0, 0], [1, 3]])
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_forward_callback_reuses_prepared_inputs_and_never_prefills(counts, mode):
    args = _args("--mode", mode, "--verification-requests", "2")
    metadata = SimpleNamespace(
        use_fused_infer_attention=bool(any(counts)),
        tree_attention=bool(any(counts)),
        attention_mask=torch.ones(1) if any(counts) else None,
    )
    model = Mock(return_value=torch.zeros(1))
    model.compute_greedy_tokens.return_value = torch.tensor([7])

    def make_tree(plans, roots, candidates, tables, sequence_ids):
        assert [plan.parent_indices.numel() for plan in plans] == counts
        assert [len(row) for row in candidates] == counts
        assert sequence_ids == [0, 1]
        return (
            torch.tensor([token for root, row in zip(roots, candidates) for token in [root, *row]]),
            torch.cat([plan.positions for plan in plans]),
            metadata,
        )

    model.make_tree_attention_metadata.side_effect = make_tree
    engine = SimpleNamespace(
        device="cpu",
        target_vocab_size=128,
        model=model,
        graph_runner=Mock(),
        cache_block_tables=torch.tensor([[0] * 8, [1] * 8]),
        _prepare_attention_metadata=Mock(return_value=(torch.tensor([31, 31]), metadata)),
        _ensure_cache_capacity=Mock(),
        _prefill_and_sample_target_batch=Mock(),
    )
    forward, physical, backend = collector._prepare_shape(engine, args, 32, counts, 7)
    assert backend == ("fused_infer_attention_tree_v1" if any(counts) else "paged_attention_v1")
    assert physical == 2 + sum(counts)
    forward()
    forward()
    if any(counts):
        model.make_tree_attention_metadata.assert_called_once()
        engine._ensure_cache_capacity.assert_called_once()
        engine._prepare_attention_metadata.assert_not_called()
    else:
        engine._prepare_attention_metadata.assert_called_once_with([0, 1], [31, 31], False)
        model.make_tree_attention_metadata.assert_not_called()
    engine._prefill_and_sample_target_batch.assert_not_called()
    if mode == "graph":
        assert engine.graph_runner.run_target_greedy.call_count == 2
        first, second = [call.args for call in engine.graph_runner.run_target_greedy.call_args_list]
        assert first[0][0] is second[0][0]
        assert first[1][0] is second[1][0]
        assert first[2][0] is second[2][0] is metadata
        model.compute_greedy_tokens.assert_not_called()
    else:
        assert model.call_count == 2
        assert model.compute_greedy_tokens.call_count == 2
        assert model.call_args_list[0].args[0] is model.call_args_list[1].args[0]
        engine.graph_runner.run_target_greedy.assert_not_called()


@pytest.mark.parametrize("failure", [None, "missing_replay", "timed_capture"])
def test_collector_uses_actual_graph_counter_evidence_not_mode_flag(monkeypatch, failure):
    from examples import profile_specslo_tree_roofline as profiler

    args = _args("--iterations", "3", "--warmup-iterations", "3")
    clock = [0.0]
    graph = SimpleNamespace(capture_count=0, replay_count=0)
    calls = []

    def forward():
        calls.append(len(calls))
        clock[0] += 0.01
        graph.capture_count += int(len(calls) == 1 or (failure == "timed_capture" and len(calls) == 4))
        graph.replay_count += int(failure != "missing_replay")

    original_measure = profiler.measure_target_forward
    monkeypatch.setattr(
        profiler,
        "measure_target_forward",
        lambda *positional, **kwargs: original_measure(*positional, **kwargs, clock=lambda: clock[0]),
    )
    monkeypatch.setattr(collector, "_prepare_shape", lambda *args: (forward, 8, "fused_infer_attention_tree_v1"))
    monkeypatch.setattr(torch.npu, "synchronize", lambda: None)
    group = object()
    reductions = []

    def reduce(tensor, *, op, group):
        assert tensor.dtype == torch.int64
        assert group is target_group
        reductions.append((tensor.tolist(), op))

    target_group = group
    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    engine = SimpleNamespace(graph_runner=graph, device="cpu", groups=SimpleNamespace(target_group=group))
    if failure:
        expected = "Not every timed" if failure == "missing_replay" else "Graph capture occurred"
        with pytest.raises(ValueError, match=expected):
            collector._measure_shape(engine, args, 8, 128, [1, 1, 1, 1], 42)
        assert reductions[0][0] == [1]
    else:
        row = collector._measure_shape(engine, args, 8, 128, [1, 1, 1, 1], 42)
        assert row["latency_ms"] == [10.0, 10.0, 10.0]
        assert row["timed_graph_captures"] == 0
        assert row["timed_graph_replays"] == 3
        assert row["warmup_iterations"] == 3
        assert row["physical_query_tokens"] == 8
        assert row["verification_requests"] == 4
        assert reductions[0][0] == [0]
    assert len(calls) == 6
