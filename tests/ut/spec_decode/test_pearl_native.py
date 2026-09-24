# SPDX-License-Identifier: Apache-2.0

import os
from itertools import combinations
from multiprocessing import Pipe
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch
import torch.nn.functional as F

from examples.benchmark_nano_pearl_speculative import (
    _aggregate_decode_host_profile,
    _aggregate_decode_profile,
    _graph_qualification_can_prune,
    _graph_qualification_fixed_point,
    _parse_target_graph_post_counts,
    _require_no_graph_fallback,
    _worker_aclgraph_deltas,
)
from examples.benchmark_nano_pearl_speculative import (
    _build_parser as _build_benchmark_parser,
)
from examples.benchmark_nano_pearl_target_only import (
    _build_parser as _build_target_only_benchmark_parser,
)
from examples.offline_inference_nano_pearl import parse_args as _parse_offline_args
from examples.serve_specslo import _build_parser as _build_specslo_server_parser
from vllm_ascend.spec_decode.pearl import native_engine as native_engine_module
from vllm_ascend.spec_decode.pearl.api import PEARLConfig, PEARLEngine
from vllm_ascend.spec_decode.pearl.native_cache import NativeCacheAllocation, NativePrefixCache
from vllm_ascend.spec_decode.pearl.native_engine import (
    NativePearlConfig,
    NativePearlEngine,
    NativeSpecRhythmDevicePayload,
    PearlPipelineState,
    SamplingParams,
    _bucket_target_verification_widths,
    _bucket_variable_target_verification_widths,
    _build_greedy_verdict,
    _build_greedy_verdict_cpu,
    _build_greedy_verdict_with_layout,
    _build_parser,
    _build_stochastic_verdict,
    _build_verification_layout,
    _can_reuse_rank_local_greedy_verdict,
    _canonical_active_indices,
    _continuous_bucket_indices,
    _continuous_result_states,
    _finished,
    _gamma_from_decode_speeds,
    _linear_draft_fia_lifetime_limits,
    _linear_draft_graph_buckets,
    _linear_window_target_sample_ms,
    _next_linear_draft_graph_bucket,
    _next_mixed_target_verify_capacity,
    _next_power_of_two_graph_bucket,
    _next_stable_target_verify_capacity,
    _normalize_sampling_params,
    _power_of_two_graph_buckets,
    _ranked_linear_draft_fia_plan,
    _rebase_kv_ready_arrivals,
    _record_linear_draft_full_chain_metrics,
    _record_mixed_target_graph_outcome,
    _record_stable_target_verify_graph_outcome,
    _restore_completed_states,
    _restore_ranked_linear_draft_rows,
    _sample_logits,
    _select_preemptive_continuous_indices,
    _set_default_npu_environment,
    _target_graph_precompile_shapes,
    _truncate_completion,
    plan_mixed_target_graph_envelope,
    plan_mixed_target_graph_layout,
    plan_stable_target_verify_graph_envelope,
    plan_stable_target_verify_graph_layout,
    resolve_mixed_target_graph_buckets,
    resolve_mixed_target_graph_verify_capacities,
)
from vllm_ascend.spec_decode.pearl.native_graph import NativeACLGraphRunner, NativeGraphExecution
from vllm_ascend.spec_decode.pearl.native_model import (
    MIN_PAGED_ATTENTION_BLOCKS,
    PAGED_ATTENTION_BLOCK_SIZE,
    NativeAttention,
    NativeAttentionMetadata,
    NativeColumnLinear,
    NativeLMHead,
    NativeQwen2ForCausalLM,
    NativeRMSNorm,
    NativeRowLinear,
    NativeTPContext,
    _is_contiguous_linear_tree_plan,
    _maybe_convert_linear_weights_to_nz,
    _maybe_untie_lm_head_for_nz,
    _pad_token_rows,
    _use_native_fused_mm_all_reduce,
    load_native_qwen2_weights,
    prepare_native_model_config,
)
from vllm_ascend.spec_decode.pearl.spec_rhythm import SpecRhythmProposalTicket
from vllm_ascend.spec_decode.pearl.topology import PearlTopology
from vllm_ascend.spec_decode.pearl.tree import build_tree_speculation_plan, pack_selected_tree_plan


@pytest.mark.parametrize("gamma", [2, 3])
def test_generic_full_window_verdict_uses_target_leader_authority(gamma):
    assert not _can_reuse_rank_local_greedy_verdict(
        [0.0, 0.0],
        gamma=gamma,
        linear_full_window=True,
    )


def test_fixed_gamma4_full_window_may_reuse_rank_local_verdict():
    assert _can_reuse_rank_local_greedy_verdict(
        [0.0, 0.0],
        gamma=4,
        linear_full_window=True,
    )
    assert not _can_reuse_rank_local_greedy_verdict(
        [0.0, 0.1],
        gamma=4,
        linear_full_window=True,
    )


def test_linear_window_ignores_transient_mixed_prefill_latency():
    assert _linear_window_target_sample_ms(17.5, mixed_prefill=False) == 17.5
    assert _linear_window_target_sample_ms(93.0, mixed_prefill=True) == 0.0
    with pytest.raises(ValueError, match="finite and non-negative"):
        _linear_window_target_sample_ms(float("nan"), mixed_prefill=False)


def test_kv_ready_arrival_rebase_preserves_offsets_and_immediate_rows():
    params = [
        SamplingParams(arrival_ts=100.0),
        SamplingParams(arrival_ts=100.25),
        SamplingParams(arrival_ts=None),
        SamplingParams(arrival_ts=101.5),
    ]

    rebased = _rebase_kv_ready_arrivals(params, 700.0)

    assert [value.arrival_ts for value in rebased] == [700.0, 700.25, None, 701.5]
    assert [value.arrival_ts for value in params] == [100.0, 100.25, None, 101.5]


def test_kv_ready_host_snapshot_restores_logical_blocks_to_new_pages():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.config = SimpleNamespace(kvcache_block_size=2)
    engine.device = torch.device("cpu")
    engine.cache_allocation = NativeCacheAllocation(
        block_tables=[[2, 0, -1]],
        num_cached_tokens=[0],
    )
    engine.cache_block_tables = torch.tensor([[2, 0, -1]], dtype=torch.int32)
    first_key = torch.arange(8, dtype=torch.float32).reshape(4, 2, 1, 1)
    first_value = first_key + 100
    second_key = first_key + 200
    second_value = first_key + 300
    engine._tree_layer_caches = [(first_key, first_value), (second_key, second_value)]

    snapshot = engine._snapshot_prompt_kv_to_host(0, 3)
    expected = snapshot.layer_kv.clone()
    engine.cache_allocation.block_tables[0][:2] = [1, 3]
    engine.cache_block_tables[0, :2] = torch.tensor([1, 3], dtype=torch.int32)
    for key_cache, value_cache in engine._tree_layer_caches:
        key_cache.zero_()
        value_cache.zero_()

    engine._restore_prompt_kv_from_host(0, snapshot)

    for layer_index, (key_cache, value_cache) in enumerate(engine._tree_layer_caches):
        assert torch.equal(key_cache.index_select(0, torch.tensor([1, 3])), expected[layer_index, 0])
        assert torch.equal(value_cache.index_select(0, torch.tensor([1, 3])), expected[layer_index, 1])


def test_kv_ready_host_snapshot_batches_gathers_and_restores_multiple_rows():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.config = SimpleNamespace(kvcache_block_size=2)
    engine.device = torch.device("cpu")
    engine.cache_allocation = NativeCacheAllocation(
        block_tables=[[5, 1, -1], [3, -1, -1]],
        num_cached_tokens=[0, 0],
    )
    engine.cache_block_tables = torch.tensor(
        [[5, 1, -1], [3, -1, -1]],
        dtype=torch.int32,
    )
    first_key = torch.arange(12, dtype=torch.float32).reshape(6, 2, 1, 1)
    first_value = first_key + 100
    second_key = first_key + 200
    second_value = first_key + 300
    engine._tree_layer_caches = [(first_key, first_value), (second_key, second_value)]

    snapshots = engine._snapshot_prompt_kv_batch_to_host([0, 1], [3, 2])
    expected = {index: snapshot.layer_kv.clone() for index, snapshot in snapshots.items()}
    assert snapshots[0].layer_kv.untyped_storage().data_ptr() == snapshots[1].layer_kv.untyped_storage().data_ptr()

    engine.cache_allocation.block_tables[0][:2] = [0, 2]
    engine.cache_allocation.block_tables[1][:1] = [4]
    engine.cache_block_tables[0, :2] = torch.tensor([0, 2], dtype=torch.int32)
    engine.cache_block_tables[1, :1] = torch.tensor([4], dtype=torch.int32)
    for key_cache, value_cache in engine._tree_layer_caches:
        key_cache.zero_()
        value_cache.zero_()

    restored_bytes = engine._restore_prompt_kv_batch_from_host(
        [0, 1],
        [snapshots[0], snapshots[1]],
    )

    assert restored_bytes == sum(snapshot.numel() * snapshot.element_size() for snapshot in expected.values())
    for layer_index, (key_cache, value_cache) in enumerate(engine._tree_layer_caches):
        assert torch.equal(key_cache.index_select(0, torch.tensor([0, 2])), expected[0][layer_index, 0])
        assert torch.equal(value_cache.index_select(0, torch.tensor([0, 2])), expected[0][layer_index, 1])
        assert torch.equal(key_cache.index_select(0, torch.tensor([4])), expected[1][layer_index, 0])
        assert torch.equal(value_cache.index_select(0, torch.tensor([4])), expected[1][layer_index, 1])


def test_kv_ready_host_snapshot_restores_contiguous_pages_without_index_copy(monkeypatch):
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.config = SimpleNamespace(kvcache_block_size=2)
    engine.device = torch.device("cpu")
    engine.cache_allocation = NativeCacheAllocation(
        block_tables=[[0, 1, -1], [2, -1, -1]],
        num_cached_tokens=[0, 0],
    )
    engine.cache_block_tables = torch.tensor(
        [[0, 1, -1], [2, -1, -1]],
        dtype=torch.int32,
    )
    key_cache = torch.arange(12, dtype=torch.float32).reshape(6, 2, 1, 1)
    value_cache = key_cache + 100
    engine._tree_layer_caches = [(key_cache, value_cache)]
    snapshots = engine._snapshot_prompt_kv_batch_to_host([0, 1], [3, 2])
    expected = torch.cat(
        [snapshots[0].layer_kv, snapshots[1].layer_kv],
        dim=2,
    ).clone()
    engine.cache_allocation.block_tables[0][:2] = [3, 4]
    engine.cache_allocation.block_tables[1][:1] = [5]
    engine.cache_block_tables[0, :2] = torch.tensor([3, 4], dtype=torch.int32)
    engine.cache_block_tables[1, :1] = torch.tensor([5], dtype=torch.int32)
    key_cache.zero_()
    value_cache.zero_()

    # A dense destination must use the direct host-to-cache slice copy.  Make
    # any accidental fallback to the old scatter path fail the test.
    monkeypatch.setattr(torch.Tensor, "index_copy_", MagicMock(side_effect=AssertionError("unexpected scatter")))
    engine._restore_prompt_kv_batch_from_host(
        [0, 1],
        [snapshots[0], snapshots[1]],
    )

    assert torch.equal(key_cache[3:6], expected[0, 0])
    assert torch.equal(value_cache[3:6], expected[0, 1])


@pytest.mark.parametrize("config_type", [NativePearlConfig, PEARLConfig])
def test_kv_ready_arrivals_reject_online_prefill(config_type):
    common = dict(
        gamma=4,
        enable_continuous_batching=True,
        enable_preemptive_scheduling=True,
        enable_spec_rhythm=True,
        spec_rhythm_online_prefill=True,
        spec_rhythm_kv_ready_arrivals=True,
    )
    if config_type is NativePearlConfig:
        with pytest.raises(ValueError, match="mutually exclusive"):
            config_type("draft", "target", 1, 3, max_model_len=512, max_tokens=32, **common)
    else:
        with pytest.raises(ValueError, match="mutually exclusive"):
            config_type("draft", "target", **common)
    assert not _can_reuse_rank_local_greedy_verdict(
        [0.0, 0.0],
        gamma=4,
        linear_full_window=False,
    )


def test_selected_tree_uses_causal_fast_path_only_for_a_contiguous_spine():
    tree = build_tree_speculation_plan(2, 2, 5, 32)
    assert not _is_contiguous_linear_tree_plan(tree)
    assert _is_contiguous_linear_tree_plan(pack_selected_tree_plan(tree, [0]))
    assert _is_contiguous_linear_tree_plan(pack_selected_tree_plan(tree, [0, 1]))
    assert not _is_contiguous_linear_tree_plan(pack_selected_tree_plan(tree, [0, 2]))


def test_linear_draft_graph_buckets_cover_service_capacity_without_large_low_load_padding():
    assert _power_of_two_graph_buckets(64) == [1, 2, 4, 8, 16, 32, 64]
    assert _power_of_two_graph_buckets(48) == [1, 2, 4, 8, 16, 32, 48]
    assert _next_power_of_two_graph_bucket(1, 64) == 1
    assert _next_power_of_two_graph_bucket(11, 64) == 16
    assert _next_power_of_two_graph_bucket(40, 48) == 48


def test_fixed_gamma_linear_draft_uses_bounded_dense_service_buckets():
    assert _linear_draft_graph_buckets(64) == [
        1,
        2,
        4,
        8,
        12,
        16,
        20,
        24,
        28,
        32,
        36,
        40,
        44,
        48,
        64,
    ]
    assert _linear_draft_graph_buckets(48) == [
        1,
        2,
        4,
        8,
        12,
        16,
        20,
        24,
        28,
        32,
        36,
        40,
        44,
        48,
    ]
    assert _linear_draft_graph_buckets(3) == [1, 2, 3]
    assert _next_linear_draft_graph_bucket(1, 64) == 1
    assert _next_linear_draft_graph_bucket(11, 64) == 12
    assert _next_linear_draft_graph_bucket(17, 64) == 20
    assert _next_linear_draft_graph_bucket(25, 64) == 28
    assert _next_linear_draft_graph_bucket(33, 64) == 36
    assert _next_linear_draft_graph_bucket(37, 64) == 40
    assert _next_linear_draft_graph_bucket(41, 64) == 44
    assert _next_linear_draft_graph_bucket(45, 64) == 48
    assert _next_linear_draft_graph_bucket(49, 64) == 64


def test_stable_target_verify_uses_every_exact_capacity():
    assert [_next_stable_target_verify_capacity(count) for count in range(1, 33)] == list(range(1, 33))
    for invalid in (0, 33):
        with pytest.raises(ValueError, match="provisioned graph family"):
            _next_stable_target_verify_capacity(invalid)


def test_stable_target_verify_accepts_configured_exact_capacity_48():
    capacities = tuple(range(1, 49))
    assert _next_stable_target_verify_capacity(43, capacities) == 43
    assert _next_stable_target_verify_capacity(48, capacities) == 48
    with pytest.raises(ValueError, match="provisioned graph family"):
        _next_stable_target_verify_capacity(49, capacities)


def test_stable_target_verify_uses_smallest_bucketed_capacity():
    capacities = tuple(range(4, 33, 4))
    assert [_next_stable_target_verify_capacity(count, capacities) for count in range(1, 9)] == [
        4,
        4,
        4,
        4,
        8,
        8,
        8,
        8,
    ]


def test_mixed_target_verify_uses_smallest_resident_capacity():
    assert [_next_mixed_target_verify_capacity(count) for count in (1, 8, 9, 16, 17, 24, 25, 32, 33, 48, 49, 64)] == [
        8,
        8,
        16,
        16,
        24,
        24,
        32,
        32,
        48,
        48,
        64,
        64,
    ]
    for invalid in (0, 65):
        with pytest.raises(ValueError, match="resident capacity"):
            _next_mixed_target_verify_capacity(invalid)


@pytest.mark.parametrize(
    ("verification_rows", "prompt_lengths", "capacity", "bucket"),
    [
        (1, [15], 8, 64),
        (16, [32, 31], 16, 64),
        (17, [32, 31], 24, 64),
        (24, [32, 31], 24, 64),
        (25, [32, 31], 32, 64),
        (31, [64, 1], 32, 128),
        (32, [128, 128, 128, 128], 32, 512),
        (32, [160, 159], 32, 320),
        (32, [224, 223], 32, 448),
        (64, [128, 128, 128, 128], 64, 512),
        (32, [512], 32, 512),
        (32, [513], 32, 768),
        (32, [769], 32, 1024),
        (32, [1025], 32, 1536),
        (32, [1570], 32, 2048),
    ],
)
def test_mixed_target_graph_layout_has_fixed_safe_shape(
    verification_rows,
    prompt_lengths,
    capacity,
    bucket,
):
    layout = plan_mixed_target_graph_layout(
        verification_rows,
        prompt_lengths,
        gamma=4,
    )

    assert layout.prompt_token_bucket == bucket
    assert layout.verification_capacity == capacity
    assert layout.request_segment_count == capacity + 5
    assert layout.total_query_tokens == capacity * 4 + bucket + 5
    assert layout.query_lengths[:capacity] == (4,) * capacity
    assert layout.query_lengths[capacity : capacity + len(prompt_lengths)] == tuple(prompt_lengths)
    assert all(length > 0 for length in layout.query_lengths)
    assert layout.verification_output_count == verification_rows * 4
    expected_prompt_indices = []
    cursor = capacity * 4
    for length in prompt_lengths:
        cursor += length
        expected_prompt_indices.append(cursor - 1)
    assert layout.prompt_output_indices == tuple(expected_prompt_indices)
    assert f"|verify:{capacity}|" in layout.graph_key
    assert layout.graph_key.endswith(f"|prompt-tokens:{bucket}")


def test_mixed_target_graph_layout_supports_eight_prompt_cohort():
    prompt_lengths = [64, 63, 62, 61, 60, 59, 58, 57]
    layout = plan_mixed_target_graph_layout(
        19,
        prompt_lengths,
        gamma=4,
        prompt_capacity=8,
    )
    envelope = plan_mixed_target_graph_envelope(
        layout,
        list(range(19)),
        [100 + index for index in range(19)],
        list(range(32, 40)),
        [0] * 8,
        list(range(64, 73)),
    )

    assert layout.prompt_capacity == 8
    assert layout.prompt_token_bucket == 512
    assert layout.request_segment_count == 24 + 8 + 1
    assert layout.total_query_tokens == 24 * 4 + 512 + 8 + 1
    assert layout.graph_key.endswith("|prompt:8+pad1|prompt-tokens:512")
    assert len(envelope.token_sequence_ids) == layout.total_query_tokens
    assert envelope.segment_sequence_ids[24:32] == tuple(range(32, 40))
    assert envelope.segment_sequence_ids[-1] == 72


def test_mixed_target_graph_bucket_subset_is_sorted_and_validated():
    assert resolve_mixed_target_graph_buckets("") == (
        64,
        128,
        192,
        256,
        320,
        384,
        448,
        512,
        768,
        1024,
        1536,
        2048,
        3072,
    )
    assert resolve_mixed_target_graph_buckets("3072,2048,448,384,320,192,1536,768,256,1024,512") == (
        192,
        256,
        320,
        384,
        448,
        512,
        768,
        1024,
        1536,
        2048,
        3072,
    )
    for invalid in ("1", "192,192", "256,nope"):
        with pytest.raises(ValueError, match="GRAPH_BUCKETS"):
            resolve_mixed_target_graph_buckets(invalid)


def test_mixed_target_graph_verify_capacity_subset_is_sorted_and_validated():
    assert resolve_mixed_target_graph_verify_capacities("") == (8, 16, 24, 32, 48, 64)
    assert resolve_mixed_target_graph_verify_capacities("32") == (32,)
    assert resolve_mixed_target_graph_verify_capacities("64,32,16,8") == (8, 16, 32, 64)
    assert _next_mixed_target_verify_capacity(1, (32,)) == 32
    assert _next_mixed_target_verify_capacity(31, (32,)) == 32
    for invalid in ("0", "7", "16,16", "33", "65", "broken"):
        with pytest.raises(ValueError):
            resolve_mixed_target_graph_verify_capacities(invalid)
    with pytest.raises(ValueError):
        _next_mixed_target_verify_capacity(17, (16,))


@pytest.mark.parametrize(
    ("verification_rows", "prompt_lengths", "match"),
    [
        (0, [1], "verification rows"),
        (65, [1], "verification rows"),
        (1, [], "prompt rows"),
        (1, [1, 1, 1, 1, 1], "prompt rows"),
        (1, [0], "prompt lengths"),
        (1, [3073], "largest stable bucket"),
    ],
)
def test_mixed_target_graph_layout_rejects_unsafe_shape_before_device_work(
    verification_rows,
    prompt_lengths,
    match,
):
    with pytest.raises(ValueError, match=match):
        plan_mixed_target_graph_layout(
            verification_rows,
            prompt_lengths,
            gamma=4,
        )


def test_mixed_target_graph_key_ignores_exact_partitions_within_bucket():
    left = plan_mixed_target_graph_layout(1, [20], gamma=4)
    right = plan_mixed_target_graph_layout(8, [31, 32], gamma=4)

    assert left.query_lengths != right.query_lengths
    assert left.graph_key == right.graph_key
    assert left.total_query_tokens == right.total_query_tokens == 101

    larger = plan_mixed_target_graph_layout(17, [20], gamma=4)
    assert larger.verification_capacity == 24
    assert larger.graph_key != left.graph_key


def test_mixed_target_graph_envelope_uses_disjoint_scratch_slots():
    layout = plan_mixed_target_graph_layout(2, [20, 11], gamma=4)
    envelope = plan_mixed_target_graph_envelope(
        layout,
        [3, 7],
        [100, 200],
        [11, 12],
        [5, 9],
        [128, 129, 130, 131, 132],
    )

    assert len(envelope.token_sequence_ids) == layout.total_query_tokens
    assert layout.verification_capacity == 8
    assert len(envelope.segment_sequence_ids) == 13
    assert envelope.segment_sequence_ids[:2] == (3, 7)
    assert envelope.positions[:8] == (
        100,
        101,
        102,
        103,
        200,
        201,
        202,
        203,
    )
    # Six dummy verification rows share scratch row 128, but write
    # consecutive non-overlapping ranges rather than aliases.
    dummy_verify_start = 2 * 4
    dummy_verify_end = 8 * 4
    assert set(envelope.token_sequence_ids[dummy_verify_start:dummy_verify_end]) == {128}
    assert envelope.positions[dummy_verify_start:dummy_verify_end] == tuple(
        range(dummy_verify_end - dummy_verify_start)
    )
    assert envelope.segment_sequence_ids[8:10] == (11, 12)
    assert envelope.segment_sequence_ids[10:] == (131, 132, 132)
    # The last prompt scratch row owns one dummy at position zero; the
    # residual starts at one, so no two writes alias.
    assert envelope.positions[-layout.query_lengths[-1] - 1] == 0
    assert envelope.positions[-layout.query_lengths[-1]] == 1


def test_stable_target_verify_graph_envelope_is_exact_q_without_padding():
    layout = plan_stable_target_verify_graph_layout(
        2,
        query_width=4,
        verification_capacity=2,
    )
    envelope = plan_stable_target_verify_graph_envelope(
        layout,
        [3, 7],
        [100, 200],
        128,
    )

    assert layout.verification_output_count == 8
    assert layout.total_query_tokens == 2 * 4
    assert layout.request_segment_count == 2
    assert envelope.segment_sequence_ids[:2] == (3, 7)
    assert envelope.positions[:8] == (100, 101, 102, 103, 200, 201, 202, 203)
    assert layout.dummy_verification_rows == 0


def test_stable_target_verify_exact_shape_metadata_is_cached():
    native_engine_module._cached_stable_target_verify_graph_layout.cache_clear()
    native_engine_module._stable_target_verify_cumulative_q.cache_clear()

    first = native_engine_module._cached_stable_target_verify_graph_layout(
        7,
        4,
        7,
    )
    second = native_engine_module._cached_stable_target_verify_graph_layout(
        7,
        4,
        7,
    )

    assert first is second
    assert native_engine_module._stable_target_verify_cumulative_q(
        4,
        7,
    ) == (4, 8, 12, 16, 20, 24, 28)


@pytest.mark.parametrize(
    "broken",
    ["verify-count", "prompt-count", "scratch-count", "scratch-alias", "real-alias"],
)
def test_mixed_target_graph_envelope_rejects_unsafe_row_ownership(broken):
    layout = plan_mixed_target_graph_layout(2, [20], gamma=4)
    verify_ids = [3, 7]
    verify_starts = [10, 20]
    prompt_ids = [11]
    prompt_starts = [0]
    scratch_ids = [128, 129, 130, 131, 132]
    if broken == "verify-count":
        verify_ids.pop()
    elif broken == "prompt-count":
        prompt_starts.clear()
    elif broken == "scratch-count":
        scratch_ids.pop()
    elif broken == "scratch-alias":
        scratch_ids[-1] = scratch_ids[-2]
    elif broken == "real-alias":
        prompt_ids[0] = verify_ids[0]

    with pytest.raises(ValueError, match="Mixed-target graph"):
        plan_mixed_target_graph_envelope(
            layout,
            verify_ids,
            verify_starts,
            prompt_ids,
            prompt_starts,
            scratch_ids,
        )


def test_mixed_target_scratch_rows_extend_storage_not_service_capacity():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.config = SimpleNamespace(max_num_queued_seqs=128, max_num_seqs=64)
    engine._mixed_target_graph_enabled = True

    assert engine._cache_sequence_capacity == 128
    assert engine._cache_storage_sequence_capacity == 133
    assert engine._mixed_target_scratch_sequence_ids == (128, 129, 130, 131, 132)


def test_tree_graph_scratch_rows_follow_mixed_rows_without_extending_service_capacity():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.config = SimpleNamespace(
        max_num_queued_seqs=128,
        max_num_seqs=64,
        enable_spec_rhythm=True,
        spec_rhythm_stable_graphs=True,
        spec_rhythm_tree_width=2,
        spec_rhythm_tree_depth=2,
        enforce_eager=False,
    )
    engine._mixed_target_graph_enabled = True

    assert engine._cache_sequence_capacity == 128
    assert engine._mixed_target_scratch_sequence_ids == (128, 129, 130, 131, 132)
    assert engine._tree_graph_scratch_sequence_ids == tuple(range(133, 197))
    assert engine._cache_storage_sequence_capacity == 197


def test_prepare_mixed_target_graph_call_materializes_positive_scratch_slots():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.device = torch.device("cpu")
    engine._mixed_target_graph_enabled = True
    engine.config = SimpleNamespace(
        max_num_queued_seqs=4,
        max_num_seqs=4,
        max_model_len=4096,
    )
    engine.cache_allocation = SimpleNamespace()
    engine.cache_block_tables = torch.arange(9 * 8, dtype=torch.int32).reshape(9, 8)
    engine.model = SimpleNamespace(attention_mask=torch.zeros((1, 1), dtype=torch.bool))
    engine._ensure_cache_capacity = MagicMock()
    engine._cache_slot_mapping = MagicMock(return_value=list(range(101)))
    layout = plan_mixed_target_graph_layout(2, [2, 2], gamma=4)
    envelope = plan_mixed_target_graph_envelope(
        layout,
        [0, 1],
        [5, 7],
        [2, 3],
        [1, 0],
        engine._mixed_target_scratch_sequence_ids,
    )

    positions, metadata = engine._prepare_mixed_target_graph_call(
        layout,
        envelope,
        torch.zeros(101, dtype=torch.long),
    )

    assert positions.tolist() == list(envelope.positions)
    assert metadata.actual_seq_lengths_q[-1] == 101
    assert len(metadata.actual_seq_lengths_q) == 13
    assert metadata.sequence_lens == envelope.sequence_lens
    assert metadata.request_block_tables.shape == (13, 8)
    assert metadata.slot_mapping.tolist() == list(range(101))
    assert min(metadata.slot_mapping.tolist()) >= 0
    engine._ensure_cache_capacity.assert_called_once_with(
        list(envelope.token_sequence_ids),
        list(envelope.positions),
    )


def test_prepare_stable_target_verify_graph_call_materializes_exact_q():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.device = torch.device("cpu")
    engine._mixed_target_graph_enabled = True
    engine.config = SimpleNamespace(
        max_num_queued_seqs=64,
        max_num_seqs=64,
        max_model_len=4096,
    )
    engine.cache_allocation = SimpleNamespace()
    engine.cache_block_tables = torch.arange(
        69 * 8,
        dtype=torch.int32,
    ).reshape(69, 8)
    engine.model = SimpleNamespace(attention_mask=torch.zeros((1, 1), dtype=torch.bool))
    engine._ensure_cache_capacity = MagicMock()
    engine._cache_slot_mapping = MagicMock(return_value=list(range(8)))
    layout = plan_stable_target_verify_graph_layout(
        2,
        query_width=4,
        verification_capacity=2,
    )
    envelope = plan_stable_target_verify_graph_envelope(
        layout,
        [0, 1],
        [5, 7],
        engine._mixed_target_scratch_sequence_ids[0],
    )

    positions, metadata = engine._prepare_stable_target_verify_graph_call(
        layout,
        envelope,
        torch.zeros(8, dtype=torch.long),
    )

    assert positions.tolist() == list(envelope.positions)
    assert metadata.actual_seq_lengths_q == (4, 8)
    assert metadata.sequence_lens == envelope.sequence_lens
    assert metadata.request_block_tables.shape == (2, 8)
    assert metadata.slot_mapping.tolist() == list(range(8))
    engine._ensure_cache_capacity.assert_called_once_with(
        list(envelope.token_sequence_ids),
        list(envelope.positions),
    )


def _stable_target_verify_numerical_harness(stable_tokens, restored_tokens):
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.device = torch.device("cpu")
    engine.draft_vocab_size = 64
    engine._stable_target_verify_numerical_validation_attempts = 0
    engine._stable_target_verify_numerical_validation_passes = 0
    engine._stable_target_verify_numerical_validation_failures = 0
    engine._stable_target_verify_numerical_restore_failures = 0
    exact_tokens = torch.arange(8, dtype=torch.long) + 100
    engine._run_device_packed_greedy = MagicMock(side_effect=[exact_tokens, restored_tokens])
    engine._prepare_stable_target_verify_graph_call = MagicMock(return_value=(torch.arange(8), SimpleNamespace()))
    engine.model = MagicMock()
    engine.model.return_value = torch.arange(16, dtype=torch.float32).reshape(
        8,
        2,
    )
    engine.model.compute_greedy_tokens.return_value = stable_tokens
    layout = plan_stable_target_verify_graph_layout(
        2,
        query_width=4,
        verification_capacity=2,
    )
    envelope = plan_stable_target_verify_graph_envelope(
        layout,
        [0, 1],
        [1, 1],
        64,
    )
    inputs = torch.arange(1, 9, dtype=torch.long)
    return engine, layout, envelope, inputs, exact_tokens


def test_stable_target_verify_numerics_compares_and_restores_exact_kv():
    expected = torch.arange(8, dtype=torch.long) + 100
    engine, layout, envelope, inputs, _ = _stable_target_verify_numerical_harness(expected, expected)

    engine._validate_stable_target_verify_numerics(
        layout,
        envelope,
        inputs,
    )

    assert engine._run_device_packed_greedy.call_count == 2
    for exact_call in engine._run_device_packed_greedy.call_args_list:
        assert exact_call.kwargs == {
            "use_aclgraph": False,
            "use_fused_infer_attention": True,
        }
    assert engine.model.call_count == 1
    assert engine._stable_target_verify_numerical_validation_attempts == 1
    assert engine._stable_target_verify_numerical_validation_passes == 1
    assert engine._stable_target_verify_numerical_validation_failures == 0
    assert engine._stable_target_verify_numerical_restore_failures == 0


def test_stable_target_verify_numerics_fails_closed_on_envelope_mismatch():
    exact = torch.arange(8, dtype=torch.long) + 100
    stable = exact.clone()
    stable[-1] += 1
    engine, layout, envelope, inputs, _ = _stable_target_verify_numerical_harness(stable, exact)

    with pytest.raises(RuntimeError, match="changed greedy tokens"):
        engine._validate_stable_target_verify_numerics(
            layout,
            envelope,
            inputs,
        )

    assert engine._stable_target_verify_numerical_validation_attempts == 1
    assert engine._stable_target_verify_numerical_validation_passes == 0
    assert engine._stable_target_verify_numerical_validation_failures == 1
    assert engine._stable_target_verify_numerical_restore_failures == 0


def test_stable_target_verify_numerics_fails_closed_on_restore_mismatch():
    exact = torch.arange(8, dtype=torch.long) + 100
    restored = exact.clone()
    restored[0] += 1
    engine, layout, envelope, inputs, _ = _stable_target_verify_numerical_harness(exact, restored)

    with pytest.raises(RuntimeError, match="KV restoration is not trustworthy"):
        engine._validate_stable_target_verify_numerics(
            layout,
            envelope,
            inputs,
        )

    assert engine._stable_target_verify_numerical_validation_attempts == 1
    assert engine._stable_target_verify_numerical_validation_passes == 0
    assert engine._stable_target_verify_numerical_validation_failures == 1
    assert engine._stable_target_verify_numerical_restore_failures == 1


def test_mixed_target_graph_qualification_changes_real_and_scratch_rows():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.gamma = 4
    engine.device = torch.device("cpu")
    engine._mixed_target_graph_enabled = True
    engine._mixed_target_graph_prompt_buckets = (64, 128)
    engine.config = SimpleNamespace(
        max_num_queued_seqs=64,
        max_num_seqs=64,
        max_model_len=4096,
        spec_rhythm_linear_bonus_token=False,
    )
    engine.cache_allocation = SimpleNamespace()
    engine.cache_block_tables = torch.zeros((69, 4), dtype=torch.int32)
    engine._prepare_mixed_target_graph_call = MagicMock(
        side_effect=lambda layout, envelope, inputs: (
            torch.arange(inputs.numel()),
            SimpleNamespace(),
        )
    )
    engine.graph_runner = MagicMock()
    engine.graph_runner.max_graph_tokens = 261
    engine.graph_runner.entries = {}
    engine.graph_runner.execution_counters = {
        "generic": {
            "changed_input_validation_calls": 0,
            "runtime_validation_failures": 0,
        }
    }

    calls_by_key = {}

    def qualify_on_second_call(*args, **kwargs):
        key = (f"hidden|fia-stable:{kwargs['graph_key']}", kwargs["expected_tokens"])
        entry = engine.graph_runner.entries.setdefault(
            key,
            SimpleNamespace(runtime_validated=False),
        )
        calls_by_key[key] = calls_by_key.get(key, 0) + 1
        if calls_by_key[key] == 2:
            entry.runtime_validated = True
            engine.graph_runner.execution_counters["generic"]["changed_input_validation_calls"] += 1
        return torch.zeros((kwargs["expected_tokens"], 2))

    engine.graph_runner.run_stable_fia_hidden.side_effect = qualify_on_second_call

    engine._qualify_stable_target_verify_graph = MagicMock()
    engine._qualify_mixed_target_graph_buckets()

    assert engine.graph_runner.run_stable_fia_hidden.call_count == 18
    envelopes = [call.args[1] for call in engine._prepare_mixed_target_graph_call.call_args_list]
    assert [
        (
            envelope.layout.prompt_token_bucket,
            envelope.layout.verification_capacity,
            envelope.layout.verification_rows,
        )
        for envelope in envelopes
    ] == [
        (128, 32, 32),
        (128, 32, 17),
        (128, 24, 24),
        (128, 24, 17),
        (128, 16, 16),
        (128, 16, 15),
        (128, 8, 8),
        (128, 8, 7),
        (64, 48, 48),
        (64, 48, 17),
        (64, 32, 32),
        (64, 32, 17),
        (64, 24, 24),
        (64, 24, 17),
        (64, 16, 16),
        (64, 16, 15),
        (64, 8, 8),
        (64, 8, 7),
    ]
    for first_envelope, changed_envelope in zip(
        envelopes[::2],
        envelopes[1::2],
    ):
        capacity = first_envelope.layout.verification_capacity
        changed_rows = changed_envelope.layout.verification_rows
        assert first_envelope.layout.graph_key == changed_envelope.layout.graph_key
        assert changed_envelope.layout.prompt_lengths[-3:] == (1, 1, 1)
        assert changed_envelope.segment_sequence_ids[:changed_rows] == tuple(range(changed_rows))
        assert changed_envelope.segment_sequence_ids[changed_rows:capacity] == (64,) * (capacity - changed_rows)
        assert changed_envelope.segment_sequence_ids[capacity : capacity + 4] == tuple(range(capacity, capacity + 4))
    for first_call, changed_call in zip(
        engine.graph_runner.run_stable_fia_hidden.call_args_list[::2],
        engine.graph_runner.run_stable_fia_hidden.call_args_list[1::2],
    ):
        assert int(first_call.args[0].min()) == 1
        assert int(changed_call.args[0].min()) == 2
    assert engine._mixed_target_graph_qualified_buckets == 9
    assert engine.graph_runner.execution_counters["generic"]["changed_input_validation_calls"] == 9
    engine._qualify_stable_target_verify_graph.assert_called_once_with()


def test_mixed_target_graph_prefill_only_skips_redundant_decode_graph_family():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.gamma = 4
    engine.device = torch.device("cpu")
    engine._mixed_target_graph_enabled = True
    engine._mixed_target_graph_prefill_only = True
    # An empty selected set isolates the family-routing assertion from the
    # mixed-envelope capture mechanics covered by the test above.
    engine._mixed_target_graph_prompt_buckets = ()
    engine.config = SimpleNamespace(
        max_num_queued_seqs=64,
        max_num_seqs=64,
        max_model_len=4096,
        spec_rhythm_linear_bonus_token=False,
    )
    engine.cache_allocation = SimpleNamespace()
    engine.cache_block_tables = torch.zeros((69, 4), dtype=torch.int32)
    engine.graph_runner = MagicMock(max_graph_tokens=4096)
    engine._qualify_stable_target_verify_graph = MagicMock()

    engine._qualify_mixed_target_graph_buckets()

    engine._qualify_stable_target_verify_graph.assert_not_called()


def test_stable_target_verify_graph_qualifies_every_exact_capacity_largest_first():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.gamma = 4
    engine.draft_vocab_size = 64
    engine.device = torch.device("cpu")
    engine._mixed_target_graph_enabled = True
    engine.config = SimpleNamespace(
        max_num_queued_seqs=64,
        max_num_seqs=64,
        max_model_len=4096,
        spec_rhythm_linear_bonus_token=False,
    )
    engine.cache_allocation = SimpleNamespace()
    engine.cache_block_tables = torch.zeros((69, 4), dtype=torch.int32)
    engine._prepare_stable_target_verify_graph_call = MagicMock(
        side_effect=lambda layout, envelope, inputs: (
            torch.arange(inputs.numel()),
            SimpleNamespace(),
        )
    )
    numerical_shapes = []

    def validate_numerics(layout, envelope, inputs):
        numerical_shapes.append((layout.verification_rows, layout.verification_capacity))
        engine._stable_target_verify_numerical_validation_attempts += 1
        engine._stable_target_verify_numerical_validation_passes += 1

    engine._validate_stable_target_verify_numerics = MagicMock(side_effect=validate_numerics)
    engine.graph_runner = MagicMock()
    engine.graph_runner.max_graph_tokens = 128
    engine.graph_runner.entries = {}
    engine.graph_runner.execution_counters = {
        "generic": {
            "changed_input_validation_calls": 0,
            "runtime_validation_failures": 0,
        }
    }

    calls_by_key = {}

    def qualify_on_second_call(*args, **kwargs):
        key = (
            f"greedy:{engine.draft_vocab_size}|fia-stable:{kwargs['graph_key']}",
            kwargs["expected_tokens"],
        )
        entry = engine.graph_runner.entries.setdefault(
            key,
            SimpleNamespace(runtime_validated=False),
        )
        calls_by_key[key] = calls_by_key.get(key, 0) + 1
        if calls_by_key[key] == 2:
            entry.runtime_validated = True
            engine.graph_runner.execution_counters["generic"]["changed_input_validation_calls"] += 1
        return torch.zeros(kwargs["expected_tokens"], dtype=torch.long)

    engine.graph_runner.run_stable_fia_greedy.side_effect = qualify_on_second_call

    engine._qualify_stable_target_verify_graph()

    assert engine.graph_runner.run_stable_fia_greedy.call_count == 64
    assert all(
        call.args[3] == engine.draft_vocab_size for call in engine.graph_runner.run_stable_fia_greedy.call_args_list
    )
    graph_capacities = [
        call.kwargs["expected_tokens"] // 4 for call in engine.graph_runner.run_stable_fia_greedy.call_args_list
    ]
    assert graph_capacities == [capacity for capacity in range(32, 0, -1) for _ in range(2)]
    assert numerical_shapes == [(capacity, capacity) for capacity in range(32, 0, -1) for _ in range(2)]
    assert engine._stable_target_verify_graph_qualified == 1
    assert engine._stable_target_verify_graph_qualified_capacities == tuple(range(1, 33))
    assert engine._stable_target_verify_graph_capacity_map == {value: value for value in range(1, 33)}


@pytest.mark.parametrize(
    ("outcome", "changed_counter"),
    [
        (("graph", 19, 311), "spec_rhythm_mixed_target_graph_batches"),
        (
            ("bounded_shape", 19, 311),
            "spec_rhythm_mixed_target_graph_bounded_shape_fallback_batches",
        ),
        (
            ("token_capacity", 19, 311),
            "spec_rhythm_mixed_target_graph_token_capacity_fallback_batches",
        ),
        (
            ("execution_fallback", 19, 311),
            "spec_rhythm_mixed_target_graph_execution_fallback_batches",
        ),
    ],
)
def test_mixed_target_graph_outcome_updates_service_counters(
    outcome,
    changed_counter,
):
    keys = (
        "spec_rhythm_mixed_target_graph_batches",
        "spec_rhythm_mixed_target_graph_requests",
        "spec_rhythm_mixed_target_graph_prompt_tokens",
        "spec_rhythm_mixed_target_graph_bounded_shape_fallback_batches",
        "spec_rhythm_mixed_target_graph_token_capacity_fallback_batches",
        "spec_rhythm_mixed_target_graph_execution_fallback_batches",
    )
    counters = dict.fromkeys(keys, 0)

    _record_mixed_target_graph_outcome(counters, outcome)

    assert counters[changed_counter] == 1
    if outcome[0] == "graph":
        assert counters["spec_rhythm_mixed_target_graph_requests"] == 19
        assert counters["spec_rhythm_mixed_target_graph_prompt_tokens"] == 311
    else:
        assert counters["spec_rhythm_mixed_target_graph_batches"] == 0
        assert counters["spec_rhythm_mixed_target_graph_requests"] == 0
        assert counters["spec_rhythm_mixed_target_graph_prompt_tokens"] == 0


@pytest.mark.parametrize(
    ("outcome", "changed_counter"),
    [
        (("graph", 17), "spec_rhythm_stable_target_verify_graph_batches"),
        (
            ("bounded_shape", 17),
            "spec_rhythm_stable_target_verify_graph_bounded_shape_fallback_batches",
        ),
        (
            ("token_capacity", 17),
            "spec_rhythm_stable_target_verify_graph_token_capacity_fallback_batches",
        ),
        (
            ("execution_fallback", 17),
            "spec_rhythm_stable_target_verify_graph_execution_fallback_batches",
        ),
    ],
)
def test_stable_target_verify_graph_outcome_has_independent_service_counters(
    outcome,
    changed_counter,
):
    keys = (
        "spec_rhythm_stable_target_verify_graph_batches",
        "spec_rhythm_stable_target_verify_graph_requests",
        "spec_rhythm_stable_target_verify_graph_bounded_shape_fallback_batches",
        "spec_rhythm_stable_target_verify_graph_token_capacity_fallback_batches",
        "spec_rhythm_stable_target_verify_graph_execution_fallback_batches",
    )
    counters = dict.fromkeys(keys, 0)

    _record_stable_target_verify_graph_outcome(counters, outcome)

    assert counters[changed_counter] == 1
    assert counters["spec_rhythm_stable_target_verify_graph_requests"] == (17 if outcome[0] == "graph" else 0)
    assert "spec_rhythm_mixed_target_graph_batches" not in counters


def test_precompile_changed_input_qualifies_every_full_serial_draft_bucket():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.config = SimpleNamespace(
        max_num_seqs=3,
        enable_continuous_batching=True,
        max_tokens=32,
        enable_spec_rhythm=True,
        spec_rhythm_tree_width=1,
        spec_rhythm_tree_depth=1,
        spec_rhythm_stable_graphs=True,
    )
    engine.is_draft = True
    engine.gamma = 4
    engine.draft_vocab_size = 8
    draft_counters = {
        "runtime_validation_calls": 0,
        "changed_input_validation_calls": 0,
        "runtime_validation_failures": 0,
    }
    graph_runner = SimpleNamespace(
        draft_entries={},
        execution_counters={"draft": draft_counters},
        set_expected_fia_batch_size=MagicMock(),
        task_update_skip_replay_count=0,
        last_draft_entry_key=None,
    )
    engine.graph_runner = graph_runner
    engine.precompiled_decode_batch_sizes = frozenset()
    engine._allocate_cache = MagicMock()
    engine._release_cache = MagicMock()
    engine._prefill_and_sample_target_batch = MagicMock()

    state_snapshots: list[tuple[int, list[list[int]]]] = []
    calls_by_bucket: dict[int, int] = {}

    def draft_batch(states, active_indices, budgets, **kwargs):
        batch_size = len(active_indices)
        calls_by_bucket[batch_size] = calls_by_bucket.get(batch_size, 0) + 1
        state_snapshots.append((batch_size, [list(state.token_ids) for state in states]))
        assert active_indices == list(range(batch_size))
        assert budgets == [engine.gamma] * batch_size
        assert kwargs["verification_sizes"] == [engine.gamma] * batch_size
        assert kwargs["full_window"] is True
        entry_key = (
            f"draft-greedy:{engine.draft_vocab_size}|steps:{engine.gamma}|paged",
            batch_size,
        )
        if calls_by_bucket[batch_size] == 1:
            graph_runner.draft_entries[entry_key] = SimpleNamespace(
                runtime_validated=False,
                validated_real_row_count=batch_size,
            )
        else:
            assert calls_by_bucket[batch_size] == 2
            entry = graph_runner.draft_entries[entry_key]
            assert not entry.runtime_validated
            entry.runtime_validated = True
            draft_counters["runtime_validation_calls"] += 1
            draft_counters["changed_input_validation_calls"] += 1
        graph_runner.last_draft_entry_key = entry_key
        next_windows = torch.full(
            (batch_size, engine.gamma),
            10 + batch_size,
            dtype=torch.long,
        )
        return None, next_windows, None

    engine._draft_spec_rhythm_device_batch = MagicMock(side_effect=draft_batch)

    with patch("vllm_ascend.spec_decode.pearl.native_engine.torch.npu.synchronize") as synchronize:
        engine._precompile_decode_graphs(include_target_graphs=False)

    assert calls_by_bucket == {1: 2, 2: 2, 3: 2}
    assert state_snapshots == [
        (1, [[0, 0]]),
        (1, [[0, 0, 11]]),
        (2, [[0, 0], [0, 0]]),
        (2, [[0, 0, 12], [0, 0, 12]]),
        (3, [[0, 0], [0, 0], [0, 0]]),
        (3, [[0, 0, 13], [0, 0, 13], [0, 0, 13]]),
    ]
    assert draft_counters == {
        "runtime_validation_calls": 3,
        "changed_input_validation_calls": 3,
        "runtime_validation_failures": 0,
    }

    assert all(
        entry.runtime_validated and entry.validated_real_row_count == entry_key[1]
        for entry_key, entry in graph_runner.draft_entries.items()
    )
    assert [args.args[0] for args in graph_runner.set_expected_fia_batch_size.call_args_list] == [1, 2, 3]
    assert engine.precompiled_decode_batch_sizes == frozenset()
    engine._allocate_cache.assert_called_once_with(
        [[0], [0], [0]],
        enable_prefix_caching=False,
    )
    engine._release_cache.assert_called_once_with()
    synchronize.assert_called_once_with()


def test_stable_task_barrier_qualification_keeps_common_kv_signature_fixed():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.gamma = 4
    engine.draft_vocab_size = 8
    counters = {
        "runtime_validation_calls": 0,
        "changed_input_validation_calls": 0,
        "runtime_validation_failures": 0,
    }
    graph_runner = SimpleNamespace(
        draft_entries={},
        execution_counters={"draft": counters},
        task_update_skip_replay_count=0,
        last_draft_entry_key=None,
    )
    engine.graph_runner = graph_runner
    state = PearlPipelineState(
        [1, 2],
        prompt_length=2,
        max_tokens=32,
    )
    state.committed_length = len(state.token_ids)
    entry_key = (
        "draft-greedy:8|steps:4|fia:1|lane:0|stable-task-barrier|kv-cap:38",
        1,
    )
    calls = 0

    def draft_batch(states, active_indices, budgets, **kwargs):
        nonlocal calls
        calls += 1
        assert active_indices == [0]
        assert budgets == [engine.gamma]
        assert kwargs["full_window"] is True
        assert kwargs["graph_lane"] == 0
        # The stable barrier must never qualify by mutating the common-KV host
        # signature. Input tensors still change as tokens are appended.
        assert states[0].max_tokens == 32
        if calls == 1:
            graph_runner.draft_entries[entry_key] = SimpleNamespace(
                runtime_validated=False,
                validated_real_row_count=1,
            )
        else:
            entry = graph_runner.draft_entries[entry_key]
            assert not entry.runtime_validated
            entry.runtime_validated = True
            counters["runtime_validation_calls"] += 1
            counters["changed_input_validation_calls"] += 1
            graph_runner.task_update_skip_replay_count += 1
        graph_runner.last_draft_entry_key = entry_key
        return None, torch.full((1, engine.gamma), 7, dtype=torch.long), None

    engine._draft_spec_rhythm_device_batch = MagicMock(side_effect=draft_batch)
    with patch.dict(
        os.environ,
        {
            "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_BUCKET": "1",
            "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_COMMON_KV": "1",
            "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_STABLE_TASK_BARRIER": "1",
        },
    ):
        engine._qualify_linear_draft_graph_bucket(
            [state],
            1,
            graph_lane=0,
        )

    assert calls == 3
    assert state.token_ids == [1, 2]
    assert state.committed_length == 2
    assert state.max_tokens == 32
    assert graph_runner.task_update_skip_replay_count == 2


def test_draft_only_precompile_participates_in_prefill_without_capturing_target_graphs():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.config = SimpleNamespace(
        max_num_seqs=3,
        enable_continuous_batching=True,
        max_tokens=32,
    )
    engine.is_draft = False
    engine.precompiled_decode_batch_sizes = frozenset()
    engine.graph_runner = SimpleNamespace(
        set_expected_fia_batch_size=MagicMock(),
    )
    engine._allocate_cache = MagicMock()
    engine._release_cache = MagicMock()
    engine._prefill_and_sample_target_batch = MagicMock()
    engine._run_device_packed_greedy = MagicMock()
    engine._draft_spec_rhythm_device_batch = MagicMock()

    with patch("vllm_ascend.spec_decode.pearl.native_engine.torch.npu.synchronize") as synchronize:
        engine._precompile_decode_graphs(include_target_graphs=False)

    engine._prefill_and_sample_target_batch.assert_called_once()
    engine._run_device_packed_greedy.assert_not_called()
    engine._draft_spec_rhythm_device_batch.assert_not_called()
    engine.graph_runner.set_expected_fia_batch_size.assert_not_called()
    assert engine.precompiled_decode_batch_sizes == frozenset()
    engine._release_cache.assert_called_once_with()
    synchronize.assert_called_once_with()


def test_target_only_precompile_does_not_capture_serial_draft_graphs():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.config = SimpleNamespace(
        max_num_seqs=3,
        enable_continuous_batching=True,
        max_tokens=32,
    )
    engine.is_draft = True
    engine.precompiled_decode_batch_sizes = frozenset()
    engine.graph_runner = SimpleNamespace(
        set_expected_fia_batch_size=MagicMock(),
    )
    engine._allocate_cache = MagicMock()
    engine._release_cache = MagicMock()
    engine._prefill_and_sample_target_batch = MagicMock()
    engine._draft_spec_rhythm_device_batch = MagicMock()

    with patch("vllm_ascend.spec_decode.pearl.native_engine.torch.npu.synchronize") as synchronize:
        engine._precompile_decode_graphs(
            include_target_graphs=True,
            include_draft_graphs=False,
        )

    engine._prefill_and_sample_target_batch.assert_called_once()
    engine._draft_spec_rhythm_device_batch.assert_not_called()
    engine.graph_runner.set_expected_fia_batch_size.assert_not_called()
    assert engine.precompiled_decode_batch_sizes == frozenset()
    engine._release_cache.assert_called_once_with()
    synchronize.assert_called_once_with()


def test_serial_draft_precompile_also_qualifies_opted_in_mixed_target_graph():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.config = SimpleNamespace(
        max_num_seqs=3,
        enable_continuous_batching=True,
        max_tokens=32,
    )
    engine.is_draft = False
    engine._mixed_target_graph_enabled = True
    engine.precompiled_decode_batch_sizes = frozenset()
    engine.graph_runner = SimpleNamespace(
        set_expected_fia_batch_size=MagicMock(),
    )
    engine._allocate_cache = MagicMock()
    engine._release_cache = MagicMock()
    engine._prefill_and_sample_target_batch = MagicMock()
    engine._run_device_packed_greedy = MagicMock()
    engine._qualify_mixed_target_graph_buckets = MagicMock()

    with patch("vllm_ascend.spec_decode.pearl.native_engine.torch.npu.synchronize") as synchronize:
        engine._precompile_decode_graphs(include_target_graphs=False)

    engine._qualify_mixed_target_graph_buckets.assert_called_once_with()
    engine._run_device_packed_greedy.assert_not_called()
    engine._allocate_cache.assert_called_once_with(
        [[0], [0], [0]],
        enable_prefix_caching=False,
        reserve_sequence_capacity=True,
    )
    assert engine.precompiled_decode_batch_sizes == frozenset()
    engine._release_cache.assert_called_once_with()
    synchronize.assert_called_once_with()


@pytest.mark.parametrize(
    ("execution", "expected_counter"),
    [
        (
            NativeGraphExecution(
                "capture_replay",
                capture_attempted=True,
                replay_executed=True,
            ),
            "_linear_draft_full_chain_capture_replay_calls",
        ),
        (
            NativeGraphExecution("replay", replay_executed=True),
            "_linear_draft_full_chain_replay_calls",
        ),
        (
            NativeGraphExecution("eager", "disabled_entry"),
            "_linear_draft_full_chain_eager_fallback_calls",
        ),
    ],
)
def test_linear_full_chain_telemetry_distinguishes_runner_outcomes(
    execution,
    expected_counter,
):
    owner = SimpleNamespace()

    _record_linear_draft_full_chain_metrics(
        owner,
        logical_rows=3,
        padded_rows=4,
        gamma=4,
        execution=execution,
        eager_rows=2,
    )

    assert owner._linear_draft_full_chain_bucket_calls == {4: 1}
    assert owner._linear_draft_full_chain_logical_row_calls == {3: 1}
    assert owner._linear_draft_full_chain_calls == 1
    assert owner._linear_draft_full_chain_logical_rows == 3
    assert owner._linear_draft_full_chain_logical_tokens == 12
    assert owner._linear_draft_full_chain_padded_rows == 4
    assert owner._linear_draft_full_chain_padded_tokens == 16
    assert owner._linear_draft_full_chain_padding_rows == 1
    assert owner._linear_draft_full_chain_padding_tokens == 4
    assert owner._linear_draft_full_chain_eager_rows == 2
    assert owner._linear_draft_full_chain_multi_eager_calls == 1
    assert owner._linear_draft_full_chain_max_eager_rows == 2
    assert getattr(owner, expected_counter) == 1
    assert (
        sum(
            getattr(owner, name, 0)
            for name in (
                "_linear_draft_full_chain_capture_replay_calls",
                "_linear_draft_full_chain_replay_calls",
                "_linear_draft_full_chain_eager_fallback_calls",
                "_linear_draft_full_chain_unclassified_calls",
            )
        )
        == owner._linear_draft_full_chain_calls
    )


def test_pearl_defaults_to_tp3_deterministic_aiv_without_overriding_user_configuration():
    with patch.dict(os.environ, {}, clear=True):
        _set_default_npu_environment(target_tp_size=3)
        assert os.environ["TASK_QUEUE_ENABLE"] == "1"
        assert os.environ["HCCL_OP_EXPANSION_MODE"] == "AIV"
        assert os.environ["HCCL_DETERMINISTIC"] == "true"

        os.environ["HCCL_OP_EXPANSION_MODE"] = "user-mode"
        os.environ["HCCL_DETERMINISTIC"] = "false"
        os.environ["TASK_QUEUE_ENABLE"] = "0"
        _set_default_npu_environment(target_tp_size=3)
        assert os.environ["TASK_QUEUE_ENABLE"] == "0"
        assert os.environ["HCCL_OP_EXPANSION_MODE"] == "user-mode"
        assert os.environ["HCCL_DETERMINISTIC"] == "false"


def test_native_mm_allreduce_never_activates_from_symbol_presence_alone():
    with (
        patch("vllm_ascend.spec_decode.pearl.native_model.ascend_envs.VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE", False),
        patch(
            "vllm_ascend.spec_decode.pearl.native_model.ascend_envs.VLLM_ASCEND_PEARL_ENABLE_TP3_MM_ALL_REDUCE",
            False,
        ),
    ):
        assert not _use_native_fused_mm_all_reduce(2, "npu")
        assert not _use_native_fused_mm_all_reduce(3, "npu")
        assert not _use_native_fused_mm_all_reduce(4, "npu")
        assert not _use_native_fused_mm_all_reduce(2, "cpu")


def test_native_mm_allreduce_matches_explicit_vllm_and_tp3_opt_ins():
    with patch("vllm_ascend.spec_decode.pearl.native_model.ascend_envs.VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE", True):
        assert _use_native_fused_mm_all_reduce(2, "npu")
        assert _use_native_fused_mm_all_reduce(4, "npu")
        assert _use_native_fused_mm_all_reduce(8, "npu")
        assert not _use_native_fused_mm_all_reduce(3, "npu")
    with patch(
        "vllm_ascend.spec_decode.pearl.native_model.ascend_envs.VLLM_ASCEND_PEARL_ENABLE_TP3_MM_ALL_REDUCE", True
    ):
        assert _use_native_fused_mm_all_reduce(3, "npu")


def test_sampling_params_support_an_independent_draft_temperature():
    params = SamplingParams(
        temperature=0.0,
        draft_temperature=0.7,
        top_p=0.9,
        top_k=8,
        draft_top_p=0.8,
        draft_top_k=4,
    )
    assert params.temperature == 0.0
    assert params.draft_temperature == 0.7
    assert (params.top_p, params.top_k, params.draft_top_p, params.draft_top_k) == (0.9, 8, 0.8, 4)
    with pytest.raises(ValueError, match="temperatures"):
        SamplingParams(draft_temperature=-0.1)
    with pytest.raises(ValueError, match="top-p"):
        SamplingParams(top_p=0.0)
    with pytest.raises(ValueError, match="top-k"):
        SamplingParams(draft_top_k=-1)


def test_pearl_does_not_force_deterministic_hccl_for_power_of_two_target_tp():
    with patch.dict(os.environ, {}, clear=True):
        _set_default_npu_environment(target_tp_size=4)
        assert os.environ["TASK_QUEUE_ENABLE"] == "1"
        assert os.environ["HCCL_OP_EXPANSION_MODE"] == "AIV"
        assert "HCCL_DETERMINISTIC" not in os.environ


def test_active_requests_are_canonicalized_by_verification_width():
    states = [
        PearlPipelineState([1], prompt_length=1, pre_verify=True),
        PearlPipelineState([2], prompt_length=1, pre_verify=False),
        PearlPipelineState([3], prompt_length=1, pre_verify=True),
        PearlPipelineState([4], prompt_length=1, pre_verify=False),
    ]

    assert _canonical_active_indices(states, [0, 1, 2, 3]) == [1, 3, 0, 2]


def test_target_verification_widths_are_quantized_without_exposing_padding():
    widths, valid_indices = _bucket_target_verification_widths(
        [False, False, False, *([True] * 13)],
        gamma=4,
        num_buckets=8,
    )

    assert widths == [4, 4, 4, 4, *([1] * 12)]
    assert len(valid_indices) == 25
    assert valid_indices[:13] == list(range(13))
    assert valid_indices[13:] == list(range(16, 28))


def test_variable_target_widths_retain_only_real_candidate_rows():
    widths, valid_indices = _bucket_variable_target_verification_widths([4, 2, 1, 1], gamma=5, num_buckets=2)
    assert widths == [5, 5, 1, 1]
    assert valid_indices == [0, 1, 2, 3, 5, 6, 10, 11]


def test_target_graph_precompile_shapes_include_nondivisible_maximum():
    shapes = _target_graph_precompile_shapes([5], gamma=4, num_buckets=2)

    assert shapes == [(5, 5, 0), (14, 5, 3), (20, 5, 5)]


def test_target_verification_widths_use_explicit_hotspot_buckets():
    widths, valid_indices = _bucket_target_verification_widths(
        [False, False, False, False, True],
        gamma=4,
        num_buckets=2,
        configured_post_counts=((5, (0, 3, 5)),),
    )

    assert widths == [4, 4, 4, 4, 4]
    assert valid_indices == [*range(16), 16]


def test_target_graph_precompile_shapes_use_explicit_hotspot_buckets():
    shapes = _target_graph_precompile_shapes(
        [5],
        gamma=4,
        num_buckets=2,
        configured_post_counts=((5, (0, 2, 5)),),
    )

    assert shapes == [(5, 5, 0), (11, 5, 2), (20, 5, 5)]


def test_target_graph_post_counts_reject_invalid_ranges():
    with pytest.raises(ValueError, match="strictly increasing"):
        NativePearlConfig(
            "draft",
            "target",
            1,
            2,
            4,
            512,
            32,
            target_verification_graph_post_counts=((8, (0, 4, 7)),),
        )


def test_benchmark_parses_explicit_target_graph_post_counts():
    assert _parse_target_graph_post_counts(["128:0,26,75,128", "64:0,47,64"]) == (
        (128, (0, 26, 75, 128)),
        (64, (0, 47, 64)),
    )


class _RecordingHostView:
    def __init__(self, tensor, events, bounds):
        self.tensor = tensor
        self.events = events
        self.bounds = bounds

    def copy_(self, source, *, non_blocking=False):
        self.events.append(("copy", *self.bounds, non_blocking))
        self.tensor.copy_(source)
        return self

    def tolist(self):
        self.events.append(("tolist", *self.bounds))
        return self.tensor.tolist()


class _RecordingHostBuffer:
    def __init__(self, size, events):
        self.tensor = torch.full((size,), -999, dtype=torch.long)
        self.events = events

    def __getitem__(self, key):
        assert isinstance(key, slice)
        start = 0 if key.start is None else key.start
        stop = self.tensor.numel() if key.stop is None else key.stop
        return _RecordingHostView(self.tensor[key], self.events, (start, stop))


class _RecordingCopyEvent:
    def __init__(self, events):
        self.events = events

    def record(self):
        self.events.append("record")

    def synchronize(self):
        self.events.append("synchronize")


def _fixed_full_window_correction_engine(*, is_draft, events):
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.rank = 0 if is_draft else 1
    engine.is_draft = is_draft
    engine.gamma = 4
    engine.device = SimpleNamespace(type="npu")
    engine.config = SimpleNamespace(
        spec_rhythm_linear_full_window=True,
        spec_rhythm_linear_bonus_token=False,
    )
    engine.topology = PearlTopology(draft_ranks=(0,), target_ranks=(1, 2))
    engine.groups = SimpleNamespace(correction_group=object())
    engine._fixed_full_window_host_correction_staging = native_engine_module._FixedFullWindowHostCorrectionStaging(
        verdict_receive=torch.full((64,), -777, dtype=torch.long),
        host_values=_RecordingHostBuffer(192, events),
        copy_done_event=_RecordingCopyEvent(events),
    )
    return engine


@pytest.mark.parametrize("batch_size", [1, 16, 32])
def test_fixed_full_window_target_staging_overlaps_continuation_copy_and_collective_wait(batch_size):
    events = []
    engine = _fixed_full_window_correction_engine(is_draft=False, events=events)
    verdict = torch.stack(
        (
            torch.full((batch_size,), 4, dtype=torch.long),
            torch.full((batch_size,), -1, dtype=torch.long),
        ),
        dim=1,
    )
    local_next = torch.arange(batch_size * 4, dtype=torch.long).reshape(batch_size, 4)
    work = MagicMock()
    work.wait.side_effect = lambda: events.append("wait")

    def broadcast(*_args, **_kwargs):
        events.append("broadcast")
        return work

    with (
        patch.dict(os.environ, {"VLLM_ASCEND_SPECRHYTHM_GLOO_CORRECTION": "0"}),
        patch("vllm_ascend.spec_decode.pearl.native_engine.dist.broadcast", side_effect=broadcast),
    ):
        accepted, corrections, next_windows = engine._broadcast_device_round_result(
            verdict,
            None,
            verification_size=batch_size * 4,
            batch_size=batch_size,
            local_next_windows=local_next,
            replicated_target_verdict=True,
            next_window_sizes=[4] * batch_size,
        )

    continuation_size = batch_size * 4
    packed_size = continuation_size + batch_size * 2
    assert events == [
        "broadcast",
        ("copy", 0, continuation_size, True),
        "wait",
        ("copy", continuation_size, packed_size, True),
        "record",
        "synchronize",
        ("tolist", 0, packed_size),
    ]
    assert accepted == [4] * batch_size
    assert corrections == [None] * batch_size
    assert next_windows == local_next.tolist()
    assert engine._fixed_full_window_host_correction_staging_eligible_calls == 1
    assert engine._fixed_full_window_host_correction_staging_calls == 1


def test_fixed_full_window_draft_staging_reuses_receive_buffer_and_keeps_windows_on_device():
    events = []
    engine = _fixed_full_window_correction_engine(is_draft=True, events=events)
    receive_storage = engine._fixed_full_window_host_correction_staging.verdict_receive.untyped_storage()
    published_storage_pointers = []
    work = MagicMock()
    work.wait.side_effect = lambda: events.append("wait")

    def broadcast(published, **_kwargs):
        events.append("broadcast")
        published_storage_pointers.append(published.untyped_storage().data_ptr())
        rows = published.numel() // 2
        published.copy_(torch.tensor([4, -1] * rows, dtype=torch.long))
        return work

    with (
        patch.dict(os.environ, {"VLLM_ASCEND_SPECRHYTHM_GLOO_CORRECTION": "0"}),
        patch("vllm_ascend.spec_decode.pearl.native_engine.dist.broadcast", side_effect=broadcast),
    ):
        for batch_size in (32, 1):
            local_next = torch.arange(batch_size * 4, dtype=torch.long).reshape(batch_size, 4)
            accepted, corrections, next_windows = engine._broadcast_device_round_result(
                None,
                None,
                verification_size=batch_size * 4,
                batch_size=batch_size,
                local_next_windows=local_next,
                replicated_target_verdict=True,
                next_window_sizes=[4] * batch_size,
            )
            assert accepted == [4] * batch_size
            assert corrections == [None] * batch_size
            assert all(torch.equal(window, local_next[row]) for row, window in enumerate(next_windows))

    assert published_storage_pointers == [receive_storage.data_ptr()] * 2
    assert engine._fixed_full_window_host_correction_staging_eligible_calls == 2
    assert engine._fixed_full_window_host_correction_staging_calls == 2
    assert events == [
        "broadcast",
        "wait",
        ("copy", 0, 64, True),
        "record",
        "synchronize",
        ("tolist", 0, 64),
        "broadcast",
        "wait",
        ("copy", 0, 2, True),
        "record",
        "synchronize",
        ("tolist", 0, 2),
    ]


def test_fixed_full_window_staging_ignores_stale_tail_when_target_batch_shrinks():
    events = []
    engine = _fixed_full_window_correction_engine(is_draft=False, events=events)
    work = MagicMock()

    with (
        patch.dict(os.environ, {"VLLM_ASCEND_SPECRHYTHM_GLOO_CORRECTION": "0"}),
        patch("vllm_ascend.spec_decode.pearl.native_engine.dist.broadcast", return_value=work),
    ):
        engine._broadcast_device_round_result(
            torch.tensor([[4, -1]] * 32),
            None,
            verification_size=128,
            batch_size=32,
            local_next_windows=torch.arange(128).reshape(32, 4),
            replicated_target_verdict=True,
            next_window_sizes=[4] * 32,
        )
        accepted, corrections, next_windows = engine._broadcast_device_round_result(
            torch.tensor([[0, 12345]]),
            None,
            verification_size=4,
            batch_size=1,
            local_next_windows=torch.tensor([[91, 92, 93, 94]]),
            replicated_target_verdict=True,
            next_window_sizes=[4],
        )

    assert accepted == [0]
    assert corrections == [12345]
    assert next_windows == [[91, 92, 93, 94]]
    assert events[-4:] == [
        ("copy", 4, 6, True),
        "record",
        "synchronize",
        ("tolist", 0, 6),
    ]


def test_fixed_full_window_staging_falls_back_for_partial_windows():
    events = []
    engine = _fixed_full_window_correction_engine(is_draft=False, events=events)
    work = MagicMock()

    with (
        patch.dict(os.environ, {"VLLM_ASCEND_SPECRHYTHM_GLOO_CORRECTION": "0"}),
        patch("vllm_ascend.spec_decode.pearl.native_engine.dist.broadcast", return_value=work),
    ):
        accepted, corrections, next_windows = engine._broadcast_device_round_result(
            torch.tensor([[3, -1]]),
            None,
            verification_size=3,
            batch_size=1,
            local_next_windows=torch.tensor([[31, 32, 33, 999]]),
            replicated_target_verdict=True,
            next_window_sizes=[3],
        )

    assert events == []
    assert accepted == [3]
    assert corrections == [None]
    assert next_windows == [[31, 32, 33]]
    assert getattr(engine, "_fixed_full_window_host_correction_staging_eligible_calls", 0) == 0
    assert getattr(engine, "_fixed_full_window_host_correction_staging_calls", 0) == 0


def test_target_follower_builds_replicated_greedy_result_without_broadcast():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.rank = 2
    engine.is_draft = False
    engine.gamma = 3
    engine.device = torch.device("cpu")
    engine.topology = PearlTopology(draft_ranks=(0,), target_ranks=(1, 2))
    engine.groups = SimpleNamespace(correction_group=MagicMock())
    verdict = torch.tensor([[2, -1], [0, 42]])
    draft_message = torch.tensor([10, 11, 12, 20, 21, 22, 30, 31, 32])

    with patch("vllm_ascend.spec_decode.pearl.native_engine.dist.broadcast") as broadcast:
        accepted, corrections, next_windows = engine._broadcast_round_result(
            verdict,
            draft_message,
            verification_size=3,
            batch_size=2,
            replicated_target_verdict=True,
        )

    broadcast.assert_not_called()
    assert accepted == [2, 0]
    assert corrections == [None, 42]
    assert next_windows == [[20, 21, 22], [30, 31, 32]]


def test_target_follower_uses_rank_local_continuations_with_verification_only_message():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.rank = 2
    engine.is_draft = False
    engine.gamma = 3
    engine.device = torch.device("cpu")
    engine.topology = PearlTopology(draft_ranks=(0,), target_ranks=(1, 2))
    engine.groups = SimpleNamespace(correction_group=MagicMock())
    verdict = torch.tensor([[3, -1], [0, 42]])
    verification_only = torch.tensor([10, 11, 12, 13])
    local_next = torch.tensor([[20, 21, 22], [30, 31, 32]])

    with patch("vllm_ascend.spec_decode.pearl.native_engine.dist.broadcast") as broadcast:
        accepted, corrections, next_windows = engine._broadcast_device_round_result(
            verdict,
            verification_only,
            verification_size=4,
            batch_size=2,
            local_next_windows=local_next,
            replicated_target_verdict=True,
            next_window_sizes=[3, 3],
        )

    broadcast.assert_not_called()
    assert accepted == [3, 0]
    assert corrections == [None, 42]
    assert next_windows == [[20, 21, 22], [30, 31, 32]]


def test_device_round_result_classifies_second_channel_as_correction_or_bonus():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.rank = 2
    engine.is_draft = False
    engine.gamma = 4
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(spec_rhythm_linear_bonus_token=True)
    engine.topology = PearlTopology(draft_ranks=(0,), target_ranks=(1, 2))
    engine.groups = SimpleNamespace(correction_group=MagicMock())
    verdict = torch.tensor(
        [
            [4, 88],
            [2, 91],
            [4, -1],
        ],
        dtype=torch.long,
    )
    local_next = torch.arange(12, dtype=torch.long).reshape(3, 4)

    with patch("vllm_ascend.spec_decode.pearl.native_engine.dist.broadcast") as broadcast:
        accepted, corrections, next_windows = engine._broadcast_device_round_result(
            verdict,
            None,
            verification_size=12,
            batch_size=3,
            local_next_windows=local_next,
            replicated_target_verdict=True,
            next_window_sizes=[4, 4, 4],
        )

    broadcast.assert_not_called()
    assert accepted == [4, 2, 4]
    assert corrections == [None, 91, None]
    assert engine._last_device_round_bonus_tokens == [88, None, None]
    assert next_windows == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ]


def test_gloo_correction_materializes_only_target_leader_verdict():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.rank = 1
    engine.is_draft = False
    engine.gamma = 3
    engine.device = torch.device("cpu")
    engine.topology = PearlTopology(draft_ranks=(0,), target_ranks=(1, 2))
    coordination_group = object()
    engine.groups = SimpleNamespace(
        correction_group=object(),
        verification_coordination_group=coordination_group,
    )
    verdict = torch.tensor([[3, -1], [0, 42]])
    local_next = torch.tensor([[20, 21, 22], [30, 31, 32]])

    with (
        patch.dict(
            os.environ,
            {"VLLM_ASCEND_SPECRHYTHM_GLOO_CORRECTION": "1"},
        ),
        patch("vllm_ascend.spec_decode.pearl.native_engine.dist.broadcast") as broadcast,
    ):
        accepted, corrections, next_windows = engine._broadcast_device_round_result(
            verdict,
            None,
            verification_size=4,
            batch_size=2,
            local_next_windows=local_next,
            replicated_target_verdict=True,
        )

    (published,), kwargs = broadcast.call_args
    assert published.device.type == "cpu"
    assert published.tolist() == [3, -1, 0, 42]
    assert kwargs == {"src": 1, "group": coordination_group}
    assert accepted == [3, 0]
    assert corrections == [None, 42]
    assert next_windows == [[20, 21, 22], [30, 31, 32]]


def test_gloo_correction_delivers_target_verdict_to_draft():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.rank = 0
    engine.is_draft = True
    engine.gamma = 3
    engine.device = torch.device("cpu")
    engine.topology = PearlTopology(draft_ranks=(0,), target_ranks=(1, 2))
    coordination_group = object()
    engine.groups = SimpleNamespace(
        correction_group=object(),
        verification_coordination_group=coordination_group,
    )
    local_next = torch.tensor([[20, 21, 22], [30, 31, 32]])

    def publish_verdict(result, **_kwargs):
        result.copy_(torch.tensor([3, -1, 0, 42]))

    with (
        patch.dict(
            os.environ,
            {"VLLM_ASCEND_SPECRHYTHM_GLOO_CORRECTION": "1"},
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_engine.dist.broadcast",
            side_effect=publish_verdict,
        ) as broadcast,
    ):
        accepted, corrections, next_windows = engine._broadcast_device_round_result(
            None,
            None,
            verification_size=4,
            batch_size=2,
            local_next_windows=local_next,
            replicated_target_verdict=True,
        )

    broadcast.assert_called_once()
    assert broadcast.call_args.kwargs == {
        "src": 1,
        "group": coordination_group,
    }
    assert accepted == [3, 0]
    assert corrections == [None, 42]
    assert torch.equal(next_windows[0], torch.tensor([20, 21, 22]))
    assert torch.equal(next_windows[1], torch.tensor([30, 31, 32]))


def test_gloo_correction_subgroup_keeps_target_follower_local_verdict():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.rank = 2
    engine.is_draft = False
    engine.gamma = 3
    engine.device = torch.device("cpu")
    engine.topology = PearlTopology(draft_ranks=(0,), target_ranks=(1, 2))
    coordination_group = object()
    engine.groups = SimpleNamespace(
        correction_group=object(),
        verification_coordination_group=coordination_group,
        correction_coordination_group=object(),
    )
    follower_verdict = torch.tensor([[0, 999], [0, 998]])
    local_next = torch.tensor([[20, 21, 22], [30, 31, 32]])

    with (
        patch.dict(
            os.environ,
            {"VLLM_ASCEND_SPECRHYTHM_GLOO_CORRECTION": "1"},
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_engine.dist.broadcast",
        ) as broadcast,
    ):
        accepted, corrections, _ = engine._broadcast_device_round_result(
            follower_verdict,
            None,
            verification_size=4,
            batch_size=2,
            local_next_windows=local_next,
            replicated_target_verdict=True,
        )

    broadcast.assert_not_called()
    assert accepted == [0, 0]
    assert corrections == [999, 998]


def test_gloo_correction_falls_back_for_wider_draft_tp():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.rank = 2
    engine.is_draft = False
    engine.gamma = 3
    engine.device = torch.device("cpu")
    engine.topology = PearlTopology(
        draft_ranks=(0, 1),
        target_ranks=(2, 3),
    )
    correction_group = object()
    engine.groups = SimpleNamespace(
        correction_group=correction_group,
        verification_coordination_group=object(),
    )
    verdict = torch.tensor([[3, -1]])
    local_next = torch.tensor([[20, 21, 22]])
    work = MagicMock()

    with (
        patch.dict(
            os.environ,
            {"VLLM_ASCEND_SPECRHYTHM_GLOO_CORRECTION": "1"},
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_engine.dist.broadcast",
            return_value=work,
        ) as broadcast,
    ):
        accepted, corrections, _ = engine._broadcast_device_round_result(
            verdict,
            None,
            verification_size=3,
            batch_size=1,
            local_next_windows=local_next,
            replicated_target_verdict=True,
        )

    (published,), kwargs = broadcast.call_args
    assert torch.equal(published, verdict.flatten())
    assert kwargs == {
        "src": 2,
        "group": correction_group,
        "async_op": True,
    }
    work.wait.assert_called_once_with()
    assert accepted == [3]
    assert corrections == [None]


def test_gloo_correction_does_not_change_stochastic_world_broadcast():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.rank = 1
    engine.is_draft = False
    engine.gamma = 3
    engine.device = torch.device("cpu")
    engine.topology = PearlTopology(draft_ranks=(0,), target_ranks=(1, 2))
    engine.groups = SimpleNamespace(
        correction_group=object(),
        verification_coordination_group=object(),
    )
    verdict = torch.tensor([[2, 42]])
    local_next = torch.tensor([[20, 21, 22]])
    work = MagicMock()

    with (
        patch.dict(
            os.environ,
            {"VLLM_ASCEND_SPECRHYTHM_GLOO_CORRECTION": "1"},
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_engine.dist.broadcast",
            return_value=work,
        ) as broadcast,
    ):
        accepted, corrections, _ = engine._broadcast_device_round_result(
            verdict,
            None,
            verification_size=3,
            batch_size=1,
            local_next_windows=local_next,
            replicated_target_verdict=False,
        )

    (published,), kwargs = broadcast.call_args
    assert torch.equal(published, verdict.flatten())
    assert kwargs == {"src": 1, "async_op": True}
    work.wait.assert_called_once_with()
    assert accepted == [2]
    assert corrections == [42]


def test_tree_verdict_target_leader_broadcasts_to_target_and_draft_groups():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.rank = 1
    engine.device = torch.device("cpu")
    engine.topology = PearlTopology(draft_ranks=(0,), target_ranks=(1, 2))
    engine.groups = SimpleNamespace(
        target_group=object(),
        correction_group=object(),
    )
    plan = build_tree_speculation_plan(
        width=2,
        depth=2,
        prefix_len=1,
        max_model_len=8,
        candidate_budget=3,
    )
    output = SimpleNamespace(
        token_ids=torch.tensor([[10, 11, 12]]),
        accepted_node_indices=torch.tensor([[0, -1]]),
    )

    with patch("vllm_ascend.spec_decode.pearl.native_engine.dist.broadcast") as broadcast:
        tokens, accepted, target_ms = engine._broadcast_spec_rhythm_tree_verdict(output, [plan])

    assert tokens == [[10, 11, 12]]
    assert accepted == [[0, -1]]
    assert target_ms == 0.0
    assert [call.kwargs["group"] for call in broadcast.call_args_list] == [
        engine.groups.target_group,
        engine.groups.correction_group,
    ]


def test_static_padding_restores_completed_rows_without_growing_their_state():
    completed = PearlPipelineState(
        [10, 20, 21],
        prompt_length=1,
        max_tokens=2,
        ignore_eos=True,
    )
    active = PearlPipelineState(
        [10, 30],
        prompt_length=1,
        max_tokens=3,
        ignore_eos=True,
    )
    states = [completed, active]
    snapshots: dict[int, PearlPipelineState] = {}

    _restore_completed_states(states, snapshots, frozenset())
    assert snapshots[0].token_ids == [10, 20, 21]

    states[0].token_ids.extend([22, 23, 24, 25])
    states[0].committed_length = len(states[0].token_ids)
    _restore_completed_states(states, snapshots, frozenset())

    assert states[0].token_ids == [10, 20, 21]
    assert states[0] is not snapshots[0]
    assert states[1] is active


def test_continuous_tail_uses_only_full_and_half_batch_graph_buckets():
    completed = {index: PearlPipelineState([index], prompt_length=1) for index in range(64)}

    assert len(_continuous_bucket_indices(list(range(40)), completed, 64)) == 64
    assert len(_continuous_bucket_indices(list(range(20)), completed, 64)) == 32
    assert _continuous_bucket_indices([], completed, 64) == []


def test_preemptive_continuous_scheduling_explores_least_served_requests():
    states = [PearlPipelineState([index], prompt_length=1) for index in range(6)]
    for state, rounds in zip(states, [2, 0, 1, 0, 3, 1]):
        state.verification_rounds = rounds

    assert _select_preemptive_continuous_indices(states, [0, 1, 2, 3, 4, 5], 4) == [1, 3, 2, 5]


def test_preemptive_continuous_scheduling_prioritizes_estimated_remaining_work():
    slow = PearlPipelineState([0, *range(17)], prompt_length=1, max_tokens=100)
    fast = PearlPipelineState([0, *range(33)], prompt_length=1, max_tokens=100)
    slow.verification_rounds = fast.verification_rounds = 8

    assert _select_preemptive_continuous_indices([slow, fast], [0, 1], 1) == [0]


def test_preemptive_continuous_scheduling_uses_recent_slowdown():
    first = PearlPipelineState([0, *range(33)], prompt_length=1, max_tokens=100)
    second = PearlPipelineState([0, *range(25)], prompt_length=1, max_tokens=100)
    first.verification_rounds = second.verification_rounds = 8

    selected = _select_preemptive_continuous_indices(
        [first, second],
        [0, 1],
        1,
        recent_token_gains=[[1, 1, 1, 1], [3, 3, 3, 3]],
    )

    assert selected == [0]


def test_spec_rhythm_preemptive_scheduling_prioritizes_tpot_debt():
    relaxed = PearlPipelineState([0, *range(8)], prompt_length=1, max_tokens=32, slo_tpot_ms=100.0)
    urgent = PearlPipelineState([0, *range(8)], prompt_length=1, max_tokens=32, slo_tpot_ms=10.0)
    relaxed.verification_rounds = urgent.verification_rounds = 8

    selected = _select_preemptive_continuous_indices(
        [relaxed, urgent],
        [0, 1],
        1,
        decode_elapsed_ms=500.0,
        slo_aware=True,
    )

    assert selected == [1]


def test_sampling_params_validate_optional_tpot_slo():
    params = SamplingParams(temperature=0.0, max_tokens=4, slo_tpot_ms=25.0, slo_class="tight")
    assert params.slo_tpot_ms == 25.0
    assert params.slo_class == "tight"
    with pytest.raises(ValueError, match="TPOT SLO"):
        SamplingParams(slo_tpot_ms=0.0)


def test_preverify_acceptance_keeps_draft_and_target_states_in_sync():
    target = PearlPipelineState([1, 2, 3], prompt_length=2)
    draft = target.clone()
    next_window = [4, 5, 6, 7]
    draft.token_ids.extend(next_window)

    draft.apply_draft_verification(
        gamma=4,
        accepted=1,
        correction_token_id=None,
        next_round_token_ids=next_window,
    )
    target.apply_target_verification(
        gamma=4,
        accepted=1,
        correction_token_id=None,
        next_round_token_ids=next_window,
    )

    assert draft.token_ids == target.token_ids == [1, 2, 3, 4, 5, 6, 7]
    assert draft.committed_completion_token_ids == target.committed_completion_token_ids == [3, 4]
    assert not draft.pre_verify
    assert draft.accepted_draft_tokens == 1
    assert draft.verified_draft_tokens == 1
    assert draft.verification_rounds == target.verification_rounds == 1


def test_postverify_rejection_rolls_back_the_same_pipeline_suffix_on_both_sides():
    target = PearlPipelineState(
        [1, 2, 3, 4, 5],
        prompt_length=2,
        pre_verify=False,
        committed_length=2,
    )
    draft = target.clone()
    next_window = [6, 7, 8, 9]
    draft.token_ids.extend(next_window)

    draft.apply_draft_verification(
        gamma=4,
        accepted=2,
        correction_token_id=42,
        next_round_token_ids=next_window,
    )
    target.apply_target_verification(
        gamma=4,
        accepted=2,
        correction_token_id=42,
        next_round_token_ids=next_window,
    )

    assert draft.token_ids == target.token_ids == [1, 2, 3, 4, 42]
    assert draft.committed_completion_token_ids == target.committed_completion_token_ids == [3, 4, 42]
    assert draft.pre_verify
    assert draft.accepted_draft_tokens == 2
    assert draft.verified_draft_tokens == 4
    assert draft.verification_rounds == target.verification_rounds == 1


def test_variable_gamma_transition_tracks_current_and_next_window_sizes():
    target = PearlPipelineState([1, 2, 3], prompt_length=2)
    draft = target.clone()
    first_window = [4, 5]
    draft.token_ids.extend(first_window)
    for state, apply in (
        (draft, draft.apply_draft_verification),
        (target, target.apply_target_verification),
    ):
        apply(
            gamma=5,
            accepted=1,
            correction_token_id=None,
            next_round_token_ids=first_window,
            verification_size=1,
        )
        assert state.pending_window_size == 2

    second_window = [6, 7, 8, 9, 10]
    draft.token_ids.extend(second_window)
    draft.apply_draft_verification(
        gamma=5,
        accepted=1,
        correction_token_id=42,
        next_round_token_ids=second_window,
        verification_size=2,
    )
    target.apply_target_verification(
        gamma=5,
        accepted=1,
        correction_token_id=42,
        next_round_token_ids=second_window,
        verification_size=2,
    )

    assert draft.token_ids == target.token_ids == [1, 2, 3, 4, 5, 42]
    assert draft.committed_length == target.committed_length == 6
    assert draft.pending_window_size == target.pending_window_size == 0
    assert draft.continuation_epoch == target.continuation_epoch == 2


def test_rolling_eager_suffix_is_guarded_until_parent_verification():
    draft = PearlPipelineState(
        [1, 2, 3, 4, 5],
        prompt_length=2,
        pre_verify=False,
        committed_length=4,
        pending_window_size=2,
    )
    target = draft.clone()
    next_window = [6, 7, 8]
    eager_window = [9, 10]
    draft.token_ids.extend(next_window)
    draft.token_ids.extend(eager_window)

    # A rejection invalidates the eager continuation before the ordinary
    # PEARL rollback consumes the current next-window suffix.
    del draft.token_ids[-len(eager_window) :]
    draft.apply_draft_verification(
        gamma=4,
        accepted=1,
        correction_token_id=42,
        next_round_token_ids=next_window,
        verification_size=2,
    )
    target.apply_target_verification(
        gamma=4,
        accepted=1,
        correction_token_id=42,
        next_round_token_ids=next_window,
        verification_size=2,
    )
    assert draft.token_ids == target.token_ids
    assert draft.committed_length == target.committed_length


@pytest.mark.parametrize("accepted", [0, 1, 2, 3, 4])
def test_full_window_state_transition_matches_target_and_draft_committed_prefix(accepted):
    prefix = [10, 11, 12]
    proposal = [20, 21, 22, 23]
    staged_eager = [30, 31, 32, 33]
    correction = None if accepted == len(proposal) else 90 + accepted
    target = PearlPipelineState(prefix.copy(), prompt_length=2)
    draft = target.clone()
    draft.token_ids.extend([*proposal, *staged_eager])

    target.apply_target_full_window_verification(
        proposal_token_ids=proposal,
        accepted=accepted,
        correction_token_id=correction,
    )
    draft.apply_draft_full_window_verification(
        proposal_token_ids=proposal,
        accepted=accepted,
        correction_token_id=correction,
        staged_eager_token_ids=staged_eager,
    )

    committed = [*prefix, *proposal[:accepted]]
    if correction is not None:
        committed.append(correction)
    expected_draft = [*committed, *staged_eager] if correction is None else committed
    assert target.token_ids == committed
    assert draft.token_ids == expected_draft
    assert target.committed_length == draft.committed_length == len(committed)
    assert target.committed_completion_token_ids == draft.committed_completion_token_ids
    assert target.accepted_draft_tokens == draft.accepted_draft_tokens == accepted
    assert target.verified_draft_tokens == draft.verified_draft_tokens == len(proposal)
    assert target.verification_rounds == draft.verification_rounds == 1
    assert target.continuation_epoch == draft.continuation_epoch == 1
    assert target.pre_verify and draft.pre_verify
    assert target.pending_window_size == draft.pending_window_size == 0


def test_full_window_draft_full_accept_without_eager_ends_at_committed_frontier():
    state = PearlPipelineState([1, 2, 3], prompt_length=2)
    proposal = torch.tensor([4, 5, 6, 7])
    state.token_ids.extend(proposal.tolist())

    state.apply_draft_full_window_verification(
        proposal_token_ids=proposal,
        accepted=4,
        correction_token_id=None,
    )

    assert state.token_ids == [1, 2, 3, 4, 5, 6, 7]
    assert state.committed_length == len(state.token_ids)
    assert state.acceptance_lengths == [4]


def test_full_window_bonus_commits_identically_on_target_and_draft():
    prefix = [10, 11, 12]
    proposal = [20, 21, 22, 23]
    bonus = 99
    target = PearlPipelineState(prefix.copy(), prompt_length=2)
    draft = target.clone()
    draft.token_ids.extend(proposal)

    target.apply_target_full_window_verification(
        proposal_token_ids=proposal,
        accepted=len(proposal),
        correction_token_id=None,
        bonus_token_id=bonus,
    )
    draft.apply_draft_full_window_verification(
        proposal_token_ids=proposal,
        accepted=len(proposal),
        correction_token_id=None,
        bonus_token_id=bonus,
    )

    expected = [*prefix, *proposal, bonus]
    assert target.token_ids == draft.token_ids == expected
    assert target.committed_length == draft.committed_length == len(expected)
    assert target.committed_completion_token_ids == [12, *proposal, bonus]
    assert draft.committed_completion_token_ids == [12, *proposal, bonus]
    assert target.accepted_draft_tokens == draft.accepted_draft_tokens == 4
    assert target.verified_draft_tokens == draft.verified_draft_tokens == 4
    assert target.verification_rounds == draft.verification_rounds == 1


def test_full_window_rejection_commits_correction_but_never_bonus():
    prefix = [10, 11, 12]
    proposal = [20, 21, 22, 23]
    target = PearlPipelineState(prefix.copy(), prompt_length=2)
    draft = target.clone()
    draft.token_ids.extend(proposal)

    target.apply_target_full_window_verification(
        proposal_token_ids=proposal,
        accepted=2,
        correction_token_id=90,
    )
    draft.apply_draft_full_window_verification(
        proposal_token_ids=proposal,
        accepted=2,
        correction_token_id=90,
    )

    assert target.token_ids == draft.token_ids == [*prefix, 20, 21, 90]
    assert target.committed_length == draft.committed_length == 6
    assert target.accepted_draft_tokens == draft.accepted_draft_tokens == 2
    assert target.verified_draft_tokens == draft.verified_draft_tokens == 4


def test_full_window_bonus_and_staged_eager_are_mutually_exclusive():
    state = PearlPipelineState([10, 11, 12], prompt_length=2)
    proposal = [20, 21, 22, 23]
    eager = [30, 31, 32, 33]
    state.token_ids.extend([*proposal, *eager])
    before = state.clone()

    with pytest.raises(ValueError, match="cannot retain a staged eager"):
        state.apply_draft_full_window_verification(
            proposal_token_ids=proposal,
            accepted=4,
            correction_token_id=None,
            staged_eager_token_ids=eager,
            bonus_token_id=99,
        )

    assert state == before


def test_full_window_transitions_reject_misaligned_role_local_tails_without_mutation():
    target = PearlPipelineState([1, 2, 3, 99], prompt_length=2, committed_length=3)
    target_before = target.clone()
    with pytest.raises(RuntimeError, match="exact committed token frontier"):
        target.apply_target_full_window_verification(
            proposal_token_ids=[4, 5, 6, 7],
            accepted=0,
            correction_token_id=42,
        )
    assert target == target_before

    draft = PearlPipelineState([1, 2, 3, 4, 5, 6, 8], prompt_length=2, committed_length=3)
    draft_before = draft.clone()
    with pytest.raises(RuntimeError, match=r"proposal \+ staged eager"):
        draft.apply_draft_full_window_verification(
            proposal_token_ids=[4, 5, 6, 7],
            accepted=0,
            correction_token_id=42,
        )
    assert draft == draft_before


def test_device_mailbox_validates_prefix_epoch_and_variable_shapes():
    state = PearlPipelineState([1, 2], prompt_length=1)
    ticket = SpecRhythmProposalTicket(
        proposal_id=3,
        request_index=0,
        home_batch_id=0,
        gamma=2,
        required_prefix_epoch=0,
    )
    payload = NativeSpecRhythmDevicePayload(
        ticket=ticket,
        verification_tokens=torch.tensor([4]),
        next_tokens=torch.tensor([4, 5]),
        verification_size=1,
        draft_confidence=0.8,
        host_next_tokens=(4, 5),
    )
    payload.validate_for(state)
    state.continuation_epoch = 1
    with pytest.raises(RuntimeError, match="stale device mailbox"):
        payload.validate_for(state)

    bad_width = NativeSpecRhythmDevicePayload(
        ticket=ticket,
        verification_tokens=torch.tensor([4, 5, 6]),
        next_tokens=torch.tensor([4, 5]),
        verification_size=3,
        draft_confidence=0.8,
    )
    state.continuation_epoch = 0
    with pytest.raises(RuntimeError, match="does not match"):
        bad_width.validate_for(state)

    bad_host = NativeSpecRhythmDevicePayload(
        ticket=ticket,
        verification_tokens=torch.tensor([4]),
        next_tokens=torch.tensor([4, 5]),
        verification_size=1,
        draft_confidence=0.8,
        host_next_tokens=(4,),
    )
    with pytest.raises(RuntimeError, match="host continuation"):
        bad_host.validate_for(state)


def test_native_qwen2_model_runs_with_a_single_tensor_parallel_rank_on_cpu():
    config = SimpleNamespace(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        rope_theta=10_000.0,
        rms_norm_eps=1e-6,
        intermediate_size=32,
        tie_word_embeddings=False,
        num_hidden_layers=1,
    )
    model = NativeQwen2ForCausalLM(
        config,
        NativeTPContext(group=None, rank=0, size=1, leader_rank=0),
    )
    # Native parameters are torch.empty until checkpoint loading. This unit
    # test has no checkpoint; initialize them explicitly instead of depending
    # on allocator contents left by previous tests (which may contain NaNs).
    generator = torch.Generator().manual_seed(123)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.05, 0.05, generator=generator)
    model.configure_cache(16)

    hidden_states = model(torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2]))
    logits = model.compute_logits(hidden_states)
    greedy_tokens = model.compute_greedy_tokens(hidden_states, vocabulary_size=31)
    confidence_tokens, confidences = model.compute_greedy_tokens_with_confidence(hidden_states, vocabulary_size=31)

    assert hidden_states.shape == (3, 16)
    assert logits.shape == (3, 32)
    assert torch.equal(greedy_tokens, logits[:, :31].argmax(dim=-1))
    assert torch.equal(confidence_tokens, greedy_tokens)
    expected_confidences = torch.softmax(logits[:, :31].float(), dim=-1).max(dim=-1).values
    assert torch.allclose(confidences, expected_confidences)


@pytest.mark.parametrize("architecture", ["Qwen2ForCausalLM", "Qwen3ForCausalLM", "LlamaForCausalLM"])
def test_native_model_runs_every_upstream_architecture_on_cpu(architecture):
    config = SimpleNamespace(
        architectures=[architecture],
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        rope_theta=10_000.0,
        rope_parameters={"rope_theta": 10_000.0, "rope_type": "default"},
        rms_norm_eps=1e-6,
        intermediate_size=32,
        hidden_act="silu",
        attention_bias=architecture == "LlamaForCausalLM",
        mlp_bias=architecture == "LlamaForCausalLM",
        tie_word_embeddings=False,
        num_hidden_layers=1,
    )
    model = NativeQwen2ForCausalLM(
        config,
        NativeTPContext(group=None, rank=0, size=1, leader_rank=0),
    )
    model.configure_cache(16)

    hidden_states = model(torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2]))

    assert hidden_states.shape == (3, 16)
    attention = model.layers[0].self_attn
    assert isinstance(attention.q_norm, NativeRMSNorm) == (architecture == "Qwen3ForCausalLM")
    assert (attention.o_proj.bias is not None) == (architecture == "LlamaForCausalLM")
    assert (model.layers[0].mlp.gate_up_proj.bias is not None) == (architecture == "LlamaForCausalLM")


def test_native_qwen3_attention_admits_production_qknorm_rope_fusion():
    config = SimpleNamespace(
        architectures=["Qwen3ForCausalLM"],
        hidden_size=256,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=32,
        rope_theta=10_000.0,
        rope_parameters={"rope_theta": 10_000.0, "rope_type": "default"},
        rms_norm_eps=1e-6,
    )

    attention = NativeAttention(
        config,
        NativeTPContext(group=None, rank=0, size=1, leader_rank=0),
    )

    assert attention.use_qknorm_rope_fusion == hasattr(torch.ops.vllm, "qkv_rmsnorm_rope")
    assert attention.rotary_emb.cos_sin_cache.shape == (32, 128)


def test_dynamic_tp_config_matches_upstream_padding_rules():
    config = SimpleNamespace(
        architectures=["Qwen2ForCausalLM"],
        vocab_size=101,
        hidden_size=1280,
        num_attention_heads=10,
        num_key_value_heads=2,
        intermediate_size=1000,
    )

    prepared = prepare_native_model_config(config, tensor_parallel_size=3)

    assert prepared is not config
    assert prepared.head_dim == 128
    assert prepared.num_attention_heads == 15
    assert prepared.num_key_value_heads == 3
    assert prepared.intermediate_size == 1152
    assert prepared.vocab_size == 102
    assert prepared.valid_vocab_size == 101


def test_tp3_balanced_shards_remove_attention_padding_and_rebalance_ffn():
    config = SimpleNamespace(
        architectures=["Qwen3ForCausalLM"],
        vocab_size=102,
        hidden_size=32,
        num_attention_heads=16,
        num_key_value_heads=8,
        intermediate_size=1152,
        max_position_embeddings=32,
        rope_theta=10_000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        num_hidden_layers=1,
        pearl_tp3_balanced_ffn_shift=128,
    )

    prepared = prepare_native_model_config(config, tensor_parallel_size=3)

    assert prepared.pearl_q_head_partitions == (6, 6, 4)
    assert prepared.pearl_kv_head_partitions == (3, 3, 2)
    assert prepared.pearl_intermediate_partitions == (256, 256, 640)
    models = [
        NativeQwen2ForCausalLM(
            prepared,
            NativeTPContext(group=None, rank=rank, size=3, leader_rank=0),
        )
        for rank in range(3)
    ]

    assert [model.layers[0].self_attn.num_heads for model in models] == [6, 6, 4]
    assert [model.layers[0].self_attn.num_kv_heads for model in models] == [3, 3, 2]
    assert [model.layers[0].self_attn.qkv_proj.weight.shape[0] for model in models] == [24, 24, 16]
    assert [model.layers[0].self_attn.o_proj.weight.shape[1] for model in models] == [12, 12, 8]
    assert [model.layers[0].mlp.gate_up_proj.weight.shape[0] for model in models] == [512, 512, 1280]
    assert [model.layers[0].mlp.down_proj.weight.shape[1] for model in models] == [256, 256, 640]

    loaded_gate = torch.arange(1152 * 32, dtype=torch.float32).view(1152, 32)
    loaded_down = torch.arange(32 * 1152, dtype=torch.float32).view(32, 1152)
    for model in models:
        model.layers[0].mlp.gate_up_proj.load_shard(loaded_gate, 0)
        model.layers[0].mlp.down_proj.load_weight(loaded_down)
    assert torch.equal(
        torch.cat([model.layers[0].mlp.gate_up_proj.weight[:size] for model, size in zip(models, (256, 256, 640))]),
        loaded_gate,
    )
    assert torch.equal(
        torch.cat([model.layers[0].mlp.down_proj.weight for model in models], dim=1),
        loaded_down,
    )


def test_tp3_light_rank_rotates_exact_attention_and_ffn_to_leader():
    config = SimpleNamespace(
        architectures=["Qwen3ForCausalLM"],
        vocab_size=151936,
        hidden_size=5120,
        num_attention_heads=64,
        num_key_value_heads=8,
        intermediate_size=25600,
        max_position_embeddings=4096,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        num_hidden_layers=1,
        pearl_tp3_balanced_ffn_shift=0,
        pearl_tp3_light_rank=0,
    )

    prepared = prepare_native_model_config(config, tensor_parallel_size=3)

    assert prepared.pearl_q_head_partitions == (16, 24, 24)
    assert prepared.pearl_kv_head_partitions == (2, 3, 3)
    assert prepared.pearl_intermediate_partitions == (8448, 8576, 8576)
    assert sum(prepared.pearl_q_head_partitions) == config.num_attention_heads
    assert sum(prepared.pearl_kv_head_partitions) == config.num_key_value_heads
    assert sum(prepared.pearl_intermediate_partitions) == config.intermediate_size


def test_dynamic_tp_weight_loaders_zero_pad_the_final_partition():
    context = NativeTPContext(group=None, rank=2, size=3, leader_rank=0)
    column = NativeColumnLinear(2, 12, context)
    row = NativeRowLinear(12, 2, context)
    loaded_column = torch.arange(20, dtype=torch.float32).view(10, 2)
    loaded_row = torch.arange(20, dtype=torch.float32).view(2, 10)

    column.load_weight(loaded_column)
    row.load_weight(loaded_row)

    assert torch.equal(column.weight[:2], loaded_column[8:10])
    assert torch.count_nonzero(column.weight[2:]) == 0
    assert torch.equal(row.weight[:, :2], loaded_row[:, 8:10])
    assert torch.count_nonzero(row.weight[:, 2:]) == 0


def test_target_token_row_padding_is_projection_local_and_shape_preserving():
    states = torch.arange(6, dtype=torch.float32).view(3, 2)
    padded, real_rows = _pad_token_rows(states, 4)

    assert real_rows == 3
    assert padded.shape == (4, 2)
    assert torch.equal(padded[:real_rows], states)
    assert torch.count_nonzero(padded[real_rows:]) == 0

    context = NativeTPContext(group=None, rank=0, size=1, leader_rank=0)
    column = NativeColumnLinear(2, 3, context)
    row = NativeRowLinear(3, 2, context)
    column.token_pad_multiple = 4
    row.token_pad_multiple = 4
    column.weight.data.copy_(torch.arange(6, dtype=torch.float32).view(3, 2))
    row.weight.data.copy_(torch.arange(6, dtype=torch.float32).view(2, 3))

    projected = column(states)
    output = row(projected)

    assert projected.shape == (3, 3)
    assert output.shape == (3, 2)
    assert torch.equal(projected, F.linear(states, column.weight))
    assert torch.equal(output, F.linear(projected, row.weight))


def test_native_model_propagates_token_row_padding_to_linear_and_lm_head():
    config = SimpleNamespace(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=512,
        rope_theta=10_000.0,
        rms_norm_eps=1e-6,
        intermediate_size=32,
        tie_word_embeddings=False,
        num_hidden_layers=1,
        pearl_token_pad_multiple=16,
    )
    model = NativeQwen2ForCausalLM(
        config,
        NativeTPContext(group=None, rank=0, size=1, leader_rank=0),
    )

    padded_modules = [
        module for module in model.modules() if isinstance(module, (NativeColumnLinear, NativeRowLinear, NativeLMHead))
    ]
    assert padded_modules
    assert all(module.token_pad_multiple == 16 for module in padded_modules)


def test_native_attention_allocates_vllm_compatible_paged_cache():
    config = SimpleNamespace(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        rope_theta=10_000.0,
    )
    attention = NativeAttention(
        config,
        NativeTPContext(group=None, rank=0, size=1, leader_rank=0),
    )

    attention.configure_cache(PAGED_ATTENTION_BLOCK_SIZE + 1, use_paged_attention=True)

    assert attention.uses_paged_attention
    assert attention.key_cache is not None
    assert attention.key_cache.shape == (
        MIN_PAGED_ATTENTION_BLOCKS,
        PAGED_ATTENTION_BLOCK_SIZE,
        1,
        8,
    )
    assert attention.block_table is not None
    assert attention.block_table.dtype == torch.int32
    assert attention.block_table.tolist() == [[0, 1]]
    assert attention.context_lens is not None
    assert attention.context_lens.device.type == "cpu"


def test_native_attention_writes_a_packed_gqa_batch_with_one_cann_call():
    config = SimpleNamespace(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        rope_theta=10_000.0,
    )
    attention = NativeAttention(
        config,
        NativeTPContext(group=None, rank=0, size=1, leader_rank=0),
    )
    attention.configure_cache(16, use_paged_attention=True)
    packed_qkv = torch.randn(4, 32)
    key = packed_qkv[:, 16:24].view(4, 1, 8)
    value = packed_qkv[:, 24:].view(4, 1, 8)

    with patch("vllm_ascend.spec_decode.pearl.native_model.DeviceOperator.reshape_and_cache") as cache_op:
        attention._write_to_cache(torch.arange(4), key, value)

    cache_op.assert_called_once()
    assert cache_op.call_args.kwargs["key"] is key
    assert cache_op.call_args.kwargs["value"] is value
    assert cache_op.call_args.kwargs["slot_mapping"].dtype == torch.int32


def test_native_lm_head_greedy_uses_one_tensor_parallel_collective():
    context = NativeTPContext(group=MagicMock(), rank=0, size=2, leader_rank=0)
    lm_head = NativeLMHead(vocab_size=8, hidden_size=2, context=context)
    lm_head.weight.data.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]]))

    def copy_local_candidate(outputs, candidate, **_kwargs):
        outputs[0].copy_(candidate)
        outputs[1].copy_(candidate)

    with patch(
        "vllm_ascend.spec_decode.pearl.native_model.dist.all_gather",
        side_effect=copy_local_candidate,
    ) as all_gather:
        tokens = lm_head.greedy(torch.tensor([[1.0, 0.0]]), vocabulary_size=8)

    assert tokens.tolist() == [2]
    all_gather.assert_called_once()


def test_native_rmsnorm_mc2_requests_residual_output():
    context = NativeTPContext(group=MagicMock(), rank=1, size=3, leader_rank=0)
    rmsnorm = NativeRMSNorm(hidden_size=5, eps=1e-6)
    hidden = torch.randn(2, 3)
    residual = torch.randn(2, 5)
    weight = torch.randn(5, 3)
    expected = (torch.randn_like(residual), torch.randn_like(residual))

    with (
        patch(
            "vllm_ascend.spec_decode.pearl.native_model.resolve_hccl_comm_name",
            return_value="test-hcomm",
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_model.matmul_allreduce_add_rmsnorm_or_fallback",
            return_value=expected,
        ) as fused,
    ):
        result = rmsnorm.forward_mc2(hidden, residual, weight, context)

    assert result is expected
    assert fused.call_args.kwargs["is_gather_add_out"] is True


def test_native_model_builds_disjoint_paged_metadata_for_a_static_batch():
    config = SimpleNamespace(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=512,
        rope_theta=10_000.0,
        rms_norm_eps=1e-6,
        intermediate_size=32,
        tie_word_embeddings=False,
        num_hidden_layers=1,
    )
    model = NativeQwen2ForCausalLM(
        config,
        NativeTPContext(group=None, rank=0, size=1, leader_rank=0),
    )
    model.configure_cache(PAGED_ATTENTION_BLOCK_SIZE + 1, max_num_seqs=2)

    positions, metadata = model.make_attention_metadata([0, 1, 1], [128, 0, 128])

    assert positions.tolist() == [128, 0, 128]
    assert metadata.slot_mapping.tolist() == [128, 256, 384]
    assert metadata.context_lens.tolist() == [129, 1, 129]
    assert metadata.context_lens.device.type == "cpu"
    assert metadata.block_tables.tolist() == [[0, 1], [2, 3], [2, 3]]
    assert metadata.actual_seq_lengths_q == (1, 3)
    assert metadata.sequence_lens == (129, 129)
    assert metadata.request_block_tables is None

    _, fia_metadata = model.make_attention_metadata(
        [0, 1, 1],
        [128, 0, 128],
        use_fused_infer_attention=True,
    )
    assert fia_metadata.request_block_tables.tolist() == [[0, 1], [2, 3]]
    assert fia_metadata.block_tables is fia_metadata.request_block_tables

    _, remapped = model.make_attention_metadata(
        [0, 1],
        [128, 0],
        block_tables=[[4, 6], [3, 5]],
    )
    assert remapped.slot_mapping.tolist() == [6 * PAGED_ATTENTION_BLOCK_SIZE, 3 * PAGED_ATTENTION_BLOCK_SIZE]
    assert remapped.block_tables.tolist() == [[4, 6], [3, 5]]

    _, direct_slots = model.make_attention_metadata(
        [0, 1],
        [128, 0],
        block_tables=[[4, 6], [3, 5]],
        slot_mapping=[777, 888],
    )
    assert direct_slots.slot_mapping.tolist() == [777, 888]


def test_fia_graph_input_copy_skips_unused_token_block_tables():
    entry = SimpleNamespace(
        input_ids=torch.empty(2, dtype=torch.long),
        positions=torch.empty(2, dtype=torch.long),
        slot_mapping=torch.empty(2, dtype=torch.long),
        context_lens=MagicMock(),
        block_tables=MagicMock(),
        request_block_tables=torch.empty((2, 2), dtype=torch.int32),
        actual_seq_lengths_q=(),
        sequence_lens=(),
    )
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([1, 2]),
        context_lens=torch.tensor([4, 5]),
        block_tables=torch.tensor([[0, 1], [2, 3]]),
        request_block_tables=torch.tensor([[4, 5], [6, 7]], dtype=torch.int32),
        actual_seq_lengths_q=(1, 2),
        sequence_lens=(4, 5),
    )

    NativeACLGraphRunner._copy_inputs(
        entry,
        torch.tensor([10, 11]),
        torch.tensor([3, 4]),
        metadata,
    )

    entry.context_lens.copy_.assert_not_called()
    entry.block_tables.copy_.assert_not_called()
    assert torch.equal(entry.request_block_tables, metadata.request_block_tables)
    assert entry.actual_seq_lengths_q == (1, 2)
    assert entry.sequence_lens == (4, 5)


def test_paged_draft_graph_input_copy_updates_context_and_token_block_tables():
    entry = SimpleNamespace(
        input_ids=torch.empty(2, dtype=torch.long),
        positions=(torch.empty(2, dtype=torch.long),),
        slot_mappings=(torch.empty(2, dtype=torch.long),),
        context_lens=(torch.empty(2, dtype=torch.int32),),
        block_tables=(torch.empty((2, 2), dtype=torch.int32),),
        request_block_tables=(None,),
        actual_seq_lengths_q=((1, 2),),
        sequence_lens=((3, 4),),
        attention_masks=(None,),
        tree_attention_masks=(None,),
        tree_attention_modes=(False,),
    )
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([1, 2]),
        context_lens=torch.tensor([4, 5], dtype=torch.int32),
        block_tables=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        request_block_tables=None,
        attention_mask=None,
        tree_attention=False,
        tree_attention_mask=None,
        use_fused_infer_attention=False,
        actual_seq_lengths_q=(1, 2),
        sequence_lens=(4, 5),
    )

    NativeACLGraphRunner._copy_draft_inputs(
        entry,
        torch.tensor([10, 11]),
        [torch.tensor([3, 4])],
        [metadata],
    )

    assert torch.equal(entry.context_lens[0], metadata.context_lens)
    assert torch.equal(entry.block_tables[0], metadata.block_tables)
    assert entry.actual_seq_lengths_q == ((1, 2),)
    assert entry.sequence_lens == ((4, 5),)


def test_native_prefix_cache_shares_full_prompt_blocks_within_and_across_batches():
    cache = NativePrefixCache(num_blocks=8, blocks_per_sequence=2, block_size=4)
    shared_prefix = [1, 2, 3, 4]

    first = cache.allocate([shared_prefix + [5], shared_prefix + [9]])

    assert first.num_cached_tokens == [0, 4]
    assert first.block_tables[0][0] == first.block_tables[1][0]
    cache.release()

    second = cache.allocate([shared_prefix + [7]])

    assert second.num_cached_tokens == [4]
    assert second.block_tables[0][0] == first.block_tables[0][0]
    cache.release()


def test_native_prefix_cache_allocates_decode_pages_lazily():
    cache = NativePrefixCache(num_blocks=3, blocks_per_sequence=4, block_size=4)
    allocation = cache.allocate([[1], [2]])

    assert allocation.block_tables == [[0, -1, -1, -1], [1, -1, -1, -1]]
    updates = cache.ensure_capacity([0], [4])
    assert updates == [(0, 1, 2)]
    assert allocation.block_tables[0][1] == 2
    with pytest.raises(RuntimeError, match="no free physical blocks"):
        cache.ensure_capacity([1], [4])
    cache.release()


def test_native_prefix_cache_activates_reserved_live_sequence():
    cache = NativePrefixCache(num_blocks=6, blocks_per_sequence=3, block_size=4)
    allocation = cache.allocate([[1]], sequence_capacity=3)

    assert allocation.block_tables == [
        [0, -1, -1],
        [-1, -1, -1],
        [-1, -1, -1],
    ]
    cached_tokens = cache.activate_sequence(2, [2, 3, 4, 5, 6])

    assert cached_tokens == 0
    assert allocation.block_tables[2][:2] == [1, 2]
    with pytest.raises(RuntimeError, match="activated twice"):
        cache.activate_sequence(2, [7])
    cache.release()


def test_native_prefix_cache_recycles_completed_online_sequence_pages():
    # There are exactly enough physical pages for one live row.  Activating
    # the distant reserved row can therefore succeed only if row 0 returned
    # both pages to the pool.
    cache = NativePrefixCache(num_blocks=2, blocks_per_sequence=3, block_size=4)
    allocation = cache.allocate([], sequence_capacity=128)

    cache.activate_sequence(0, [1, 2, 3, 4, 5])
    first_blocks = tuple(allocation.block_tables[0][:2])
    assert all(block_id >= 0 for block_id in first_blocks)
    assert cache.release_sequence(0) == 2
    assert allocation.block_tables[0] == [-1, -1, -1]

    cache.activate_sequence(127, [6, 7, 8, 9, 10])
    assert set(allocation.block_tables[127][:2]) == set(first_blocks)
    assert cache.release_sequence(0) == 0
    cache.release()


def test_target_ar_draft_rank_returns_without_allocating_or_running_the_model():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.config = SimpleNamespace(
        max_num_seqs=1,
        max_num_batched_tokens=16,
        max_tokens=4,
        max_model_len=16,
    )
    engine.is_draft = True
    engine._allocate_cache = MagicMock()

    result = engine.generate_target_ar_batch(
        [[1, 2]],
        SamplingParams(temperature=0, max_tokens=4),
    )

    assert result is None
    engine._allocate_cache.assert_not_called()


def test_target_ar_prefill_preserves_global_sequence_ids_across_chunks():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.cache_allocation = SimpleNamespace(num_cached_tokens=[0, 4, 8])
    engine.topology = SimpleNamespace(target_leader_rank=1)
    engine.groups = SimpleNamespace(target_group=object())
    engine._run_packed_sample = MagicMock(return_value=torch.tensor([31, 41]))
    states = [
        PearlPipelineState([1], prompt_length=1, temperature=0),
        PearlPipelineState([2], prompt_length=1, temperature=0),
    ]

    result = engine._target_ar_prefill(
        [[10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]],
        states,
        sequence_ids=[1, 2],
    )

    assert result == [31, 41]
    engine._run_packed_sample.assert_called_once_with(
        [14, 15, 28, 29],
        [1, 1, 2, 2],
        [4, 5, 8, 9],
        [0, 0],
        use_aclgraph=False,
        logit_indices=[1, 3],
    )


def test_native_engine_builds_slots_from_the_cpu_cache_page_table():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.config = SimpleNamespace(kvcache_block_size=128)
    engine.cache_allocation = SimpleNamespace(block_tables=[[3, 7], [4, 9]])

    slots = engine._cache_slot_mapping([0, 1], [129, 2])

    assert slots == [7 * 128 + 1, 4 * 128 + 2]


@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_target_postverify_uses_speculative_fia_aclgraph(temperature):
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.gamma = 4
    engine.draft_vocab_size = 32
    engine.config = SimpleNamespace(
        enforce_eager=False,
        target_verification_graph_buckets=32,
        target_use_paged_attention=False,
    )
    state = PearlPipelineState(
        [1, 2, 3, 4, 5, 6],
        prompt_length=2,
        pre_verify=False,
        temperature=temperature,
    )
    engine._run_packed_greedy = MagicMock(return_value=torch.tensor([1, 2, 3, 4]))
    engine._run_packed_model = MagicMock(return_value=torch.randn(4, 32))

    engine._target_round_outputs_batch([state], [0])

    runner = engine._run_packed_greedy if temperature == 0 else engine._run_packed_model
    assert runner.call_args.kwargs["use_aclgraph"] is True
    assert runner.call_args.kwargs["use_fused_infer_attention"] is True


def test_target_preverify_uses_speculative_fia_aclgraph():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.gamma = 4
    engine.draft_vocab_size = 32
    engine.config = SimpleNamespace(
        enforce_eager=False,
        target_verification_graph_buckets=32,
        target_use_paged_attention=False,
    )
    state = PearlPipelineState([1, 2, 3], prompt_length=2, pre_verify=True)
    engine._run_packed_greedy = MagicMock(return_value=torch.tensor([4]))

    engine._target_round_outputs_batch([state], [0])

    assert engine._run_packed_greedy.call_args.kwargs["use_aclgraph"] is True
    assert engine._run_packed_greedy.call_args.kwargs["use_fused_infer_attention"] is True


def test_target_verification_can_select_paged_attention_aclgraph():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.gamma = 4
    engine.draft_vocab_size = 32
    engine.config = SimpleNamespace(
        enforce_eager=False,
        target_verification_graph_buckets=32,
        target_use_paged_attention=True,
    )
    state = PearlPipelineState([1, 2, 3], prompt_length=2, pre_verify=True)
    engine._run_packed_greedy = MagicMock(return_value=torch.tensor([4]))

    engine._target_round_outputs_batch([state], [0])

    assert engine._run_packed_greedy.call_args.kwargs["use_aclgraph"] is True
    assert engine._run_packed_greedy.call_args.kwargs["use_fused_infer_attention"] is False


@pytest.mark.parametrize("force_stepwise", [False, True])
def test_target_full_window_preserves_request_rows_and_queries_complete_proposals(force_stepwise):
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.gamma = 4
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(
        enforce_eager=True,
        target_use_paged_attention=True,
        spec_rhythm_stable_graphs=True,
    )
    engine._run_device_packed_greedy = MagicMock()
    states = [
        PearlPipelineState([1, 101], prompt_length=1),
        PearlPipelineState([4, 5, 102], prompt_length=1),
        PearlPipelineState([7, 8, 9, 103], prompt_length=1),
    ]
    active_indices = [2, 0]
    proposals = [
        torch.tensor([20, 21, 22, 23]),
        torch.tensor([30, 31, 32, 33]),
    ]
    payloads = [
        NativeSpecRhythmDevicePayload(
            ticket=SpecRhythmProposalTicket(
                proposal_id=row,
                request_index=request_index,
                home_batch_id=row,
                gamma=engine.gamma,
                required_prefix_epoch=0,
            ),
            verification_tokens=proposal,
            next_tokens=proposal,
            verification_size=engine.gamma,
            draft_confidence=1.0,
        )
        for row, (request_index, proposal) in enumerate(zip(active_indices, proposals))
    ]
    expected_rows = [[200, 201, 202, 203], [300, 301, 302, 303]]
    if force_stepwise:
        engine._run_device_packed_greedy.side_effect = [
            torch.tensor([200 + step, 300 + step]) for step in range(engine.gamma)
        ]
    else:
        engine._run_device_packed_greedy.return_value = torch.tensor(expected_rows).flatten()

    with patch.dict(
        os.environ,
        {
            "VLLM_ASCEND_SPECRHYTHM_FORCE_STEPWISE_TARGET": "1" if force_stepwise else "0",
            "VLLM_ASCEND_SPECRHYTHM_DISABLE_TARGET_ACLGRAPH": "0",
            # Keep the packed branch covered only as an explicit backend
            # diagnostic. Production paged-attention full-window verification
            # uses the causal multi-step path even inside B<=64/gamma<=8.
            "VLLM_ASCEND_SPECRHYTHM_PACKED_TARGET": "0" if force_stepwise else "1",
        },
    ):
        output, logits = engine._target_full_window_outputs_batch(states, active_indices, payloads)

    assert logits is None
    assert output.reshape(len(active_indices), engine.gamma).tolist() == expected_rows
    calls = engine._run_device_packed_greedy.call_args_list
    if force_stepwise:
        assert len(calls) == engine.gamma
        expected_inputs = [
            torch.tensor([103, 101]),
            torch.tensor([20, 30]),
            torch.tensor([21, 31]),
            torch.tensor([22, 32]),
        ]
        expected_positions = [[3, 1], [4, 2], [5, 3], [6, 4]]
        for call, expected_input, expected_position in zip(calls, expected_inputs, expected_positions):
            assert torch.equal(call.args[0], expected_input)
            assert call.args[1] == active_indices
            assert call.args[2] == expected_position
    else:
        assert len(calls) == 1
        assert torch.equal(
            calls[0].args[0],
            torch.tensor([103, 20, 21, 22, 101, 30, 31, 32]),
        )
        assert calls[0].args[1] == [2, 2, 2, 2, 0, 0, 0, 0]
        assert calls[0].args[2] == [3, 4, 5, 6, 1, 2, 3, 4]


def test_target_full_window_captures_causal_pa_chain_in_one_exact_row_graph():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.gamma = 4
    engine.device = torch.device("cpu")
    engine.draft_vocab_size = 32
    engine.config = SimpleNamespace(
        enforce_eager=False,
        target_use_paged_attention=True,
        spec_rhythm_stable_graphs=True,
    )
    engine._prepare_attention_metadata = MagicMock(
        side_effect=[(torch.tensor([3 + step, 1 + step]), SimpleNamespace(step=step)) for step in range(engine.gamma)]
    )
    engine._run_device_packed_greedy = MagicMock()
    engine.graph_runner = MagicMock()
    engine.graph_runner.run_target_greedy.return_value = tuple(
        torch.tensor([200 + step, 300 + step]) for step in range(engine.gamma)
    )
    states = [
        PearlPipelineState([1, 101], prompt_length=1),
        PearlPipelineState([4, 5, 102], prompt_length=1),
        PearlPipelineState([7, 8, 9, 103], prompt_length=1),
    ]
    active_indices = [2, 0]
    proposals = [
        torch.tensor([20, 21, 22, 23]),
        torch.tensor([30, 31, 32, 33]),
    ]
    payloads = [
        NativeSpecRhythmDevicePayload(
            ticket=SpecRhythmProposalTicket(
                proposal_id=row,
                request_index=request_index,
                home_batch_id=row,
                gamma=engine.gamma,
                required_prefix_epoch=0,
            ),
            verification_tokens=proposal,
            next_tokens=proposal,
            verification_size=engine.gamma,
            draft_confidence=1.0,
        )
        for row, (request_index, proposal) in enumerate(zip(active_indices, proposals))
    ]

    with patch.dict(
        os.environ,
        {
            "VLLM_ASCEND_SPECRHYTHM_FORCE_STEPWISE_TARGET": "0",
            "VLLM_ASCEND_SPECRHYTHM_PACKED_TARGET": "0",
            "VLLM_ASCEND_SPECRHYTHM_DISABLE_TARGET_ACLGRAPH": "0",
        },
    ):
        output, logits = engine._target_full_window_outputs_batch(
            states,
            active_indices,
            payloads,
        )

    assert logits is None
    assert output.reshape(2, 4).tolist() == [
        [200, 201, 202, 203],
        [300, 301, 302, 303],
    ]
    graph_inputs = engine.graph_runner.run_target_greedy.call_args.args[0]
    assert [value.tolist() for value in graph_inputs] == [
        [103, 101],
        [20, 30],
        [21, 31],
        [22, 32],
    ]
    assert all(value.shape == (2,) for value in graph_inputs)
    assert engine._prepare_attention_metadata.call_args_list == [
        call(active_indices, [3 + step, 1 + step], use_fused_infer_attention=False) for step in range(engine.gamma)
    ]
    engine._run_device_packed_greedy.assert_not_called()


def test_target_full_window_keeps_packed_fia_with_stable_draft_graphs():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.gamma = 4
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(
        enforce_eager=False,
        target_use_paged_attention=False,
        spec_rhythm_stable_graphs=True,
    )
    engine._run_device_packed_greedy = MagicMock(return_value=torch.arange(8))
    states = [
        PearlPipelineState([1, 101], prompt_length=1),
        PearlPipelineState([2, 102], prompt_length=1),
    ]
    payloads = [
        NativeSpecRhythmDevicePayload(
            ticket=SpecRhythmProposalTicket(
                proposal_id=row,
                request_index=row,
                home_batch_id=row,
                gamma=4,
                required_prefix_epoch=0,
            ),
            verification_tokens=torch.tensor([200 + row * 4 + step for step in range(4)]),
            next_tokens=torch.tensor([200 + row * 4 + step for step in range(4)]),
            verification_size=4,
            draft_confidence=1.0,
        )
        for row in range(2)
    ]

    output, _ = engine._target_full_window_outputs_batch(
        states,
        [0, 1],
        payloads,
    )

    assert output.tolist() == list(range(8))
    call_args = engine._run_device_packed_greedy.call_args
    assert call_args.kwargs == {
        "use_aclgraph": True,
        "use_fused_infer_attention": True,
    }
    assert call_args.args[1] == [0, 0, 0, 0, 1, 1, 1, 1]


def test_target_full_window_fia_uses_stable_decode_only_graph_envelope(
    monkeypatch,
):
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.gamma = 4
    engine.device = torch.device("cpu")
    engine.draft_vocab_size = 64
    engine._mixed_target_graph_enabled = True
    engine._mixed_target_graph_prompt_buckets = (256, 512, 1024, 2048)
    engine._stable_target_verify_graph_capacity_map = {value: value for value in range(1, 33)}
    engine.config = SimpleNamespace(
        enforce_eager=False,
        target_use_paged_attention=False,
        spec_rhythm_stable_graphs=True,
        spec_rhythm_linear_bonus_token=False,
        max_num_queued_seqs=4,
        max_num_seqs=4,
        max_model_len=4096,
    )
    engine._validate_packed_causal_fia_leakage_once = MagicMock()
    engine._prepare_stable_target_verify_graph_call = MagicMock(return_value=(torch.arange(8), SimpleNamespace()))
    engine._run_device_packed_greedy = MagicMock()
    engine.graph_runner = MagicMock()
    engine.graph_runner.max_graph_tokens = 2181
    engine.graph_runner.last_generic_execution = SimpleNamespace(used_aclgraph=True)
    engine.graph_runner.run_stable_fia_greedy.return_value = torch.arange(8, dtype=torch.long) + 1000
    engine.model = MagicMock()
    monkeypatch.setattr(
        native_engine_module.envs,
        "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH",
        True,
    )
    states = [
        PearlPipelineState([1, 101], prompt_length=1),
        PearlPipelineState([2, 102], prompt_length=1),
    ]
    payloads = [
        NativeSpecRhythmDevicePayload(
            ticket=SpecRhythmProposalTicket(
                proposal_id=row,
                request_index=row,
                home_batch_id=row,
                gamma=4,
                required_prefix_epoch=0,
            ),
            verification_tokens=torch.tensor([200 + row * 4 + step for step in range(4)]),
            next_tokens=torch.tensor([200 + row * 4 + step for step in range(4)]),
            verification_size=4,
            draft_confidence=1.0,
        )
        for row in range(2)
    ]

    with patch.object(
        native_engine_module.torch,
        "cat",
        wraps=torch.cat,
    ) as cat_mock:
        output, logits = engine._target_full_window_outputs_batch(
            states,
            [0, 1],
            payloads,
        )

    assert logits is None
    assert output.tolist() == list(range(1000, 1008))
    # Exact-Q service builds the proposal window once. It must not allocate an
    # empty padding tensor and copy the complete window through a second cat.
    assert cat_mock.call_count == 1
    engine._run_device_packed_greedy.assert_not_called()
    graph_call = engine.graph_runner.run_stable_fia_greedy.call_args
    graph_inputs = graph_call.args[0]
    assert graph_inputs.shape == (8,)
    assert graph_inputs[:8].tolist() == [
        101,
        200,
        201,
        202,
        102,
        204,
        205,
        206,
    ]
    assert graph_call.kwargs == {
        "graph_key": ("stable-target-verify-hidden|width:4|verify:2"),
        "expected_tokens": 8,
        "expected_request_segments": 2,
    }
    assert graph_call.args[3] == 64
    engine.graph_runner.run_stable_fia_greedy.assert_called_once()
    engine.graph_runner.run_stable_fia_hidden.assert_not_called()
    engine.model.compute_greedy_tokens.assert_not_called()
    envelope = engine._prepare_stable_target_verify_graph_call.call_args.args[1]
    assert envelope.layout.total_query_tokens == 8
    assert envelope.layout.verification_capacity == 2
    assert envelope.segment_sequence_ids == (0, 1)
    assert engine._last_stable_target_verify_graph_outcome == ("graph", 2)


def test_sealed_exact_target_service_uses_persistent_graph_staging(monkeypatch):
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.device = torch.device("cpu")
    engine.draft_vocab_size = 64
    engine._mixed_target_graph_enabled = True
    engine._stable_target_verify_graph_capacity_map = {2: 2}
    engine.config = SimpleNamespace(
        enforce_eager=False,
        max_model_len=4096,
        max_num_queued_seqs=3,
        max_num_seqs=3,
    )
    engine.cache_block_tables = torch.tensor(
        [[10, 11], [20, 21], [30, 31], [40, 41]],
        dtype=torch.int32,
    )
    engine._ensure_cache_capacity = MagicMock()
    engine._cache_slot_mapping = MagicMock(return_value=[13, 14, 15, 16, 27, 28, 29, 30])
    engine._prepare_stable_target_verify_graph_call = MagicMock()
    engine.graph_runner = MagicMock()
    engine.graph_runner.max_graph_tokens = 2181
    engine.graph_runner.graph_cache_sealed = True
    engine.graph_runner.last_generic_execution = SimpleNamespace(used_aclgraph=True)
    engine.graph_runner.run_stable_fia_greedy_staged.return_value = torch.arange(8, dtype=torch.long) + 500
    engine.model = SimpleNamespace(attention_mask=torch.zeros((4, 4), dtype=torch.int8))
    monkeypatch.setattr(
        native_engine_module.envs,
        "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH",
        True,
    )

    output = engine._target_full_window_stable_fia_graph(
        torch.arange(8, dtype=torch.long),
        (0, 1),
        (3, 7),
        query_width=4,
    )

    assert output.tolist() == list(range(500, 508))
    engine._prepare_stable_target_verify_graph_call.assert_not_called()
    engine.graph_runner.run_stable_fia_greedy.assert_not_called()
    staged_call = engine.graph_runner.run_stable_fia_greedy_staged.call_args
    assert staged_call.args[0].tolist() == list(range(8))
    assert staged_call.args[1] == 64
    assert staged_call.kwargs == {
        "positions": (3, 4, 5, 6, 7, 8, 9, 10),
        "slot_mapping": [13, 14, 15, 16, 27, 28, 29, 30],
        "actual_seq_lengths_q": (4, 8),
        "sequence_lens": (7, 11),
        "segment_sequence_ids": (0, 1),
        "cache_block_tables": engine.cache_block_tables,
        "attention_mask": engine.model.attention_mask,
        "graph_key": "stable-target-verify-hidden|width:4|verify:2",
        "expected_tokens": 8,
        "expected_request_segments": 2,
        "real_tokens": 8,
    }
    engine._ensure_cache_capacity.assert_called_once_with(
        [0, 0, 0, 0, 1, 1, 1, 1],
        [3, 4, 5, 6, 7, 8, 9, 10],
    )
    assert engine._last_stable_target_verify_graph_outcome == ("graph", 2)


def test_target_full_window_bonus_queries_fifth_token_and_splits_verdict_rows():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.rank = 1
    engine.is_draft = False
    engine.gamma = 4
    engine.device = torch.device("cpu")
    engine.draft_vocab_size = 64
    engine.topology = PearlTopology(draft_ranks=(0,), target_ranks=(1, 2, 3))
    engine.config = SimpleNamespace(
        enforce_eager=True,
        target_use_paged_attention=False,
        spec_rhythm_stable_graphs=True,
        spec_rhythm_linear_bonus_token=True,
        spec_rhythm_cpu_verdict=False,
        spec_rhythm_linear_full_window=True,
    )
    engine.greedy_verification_layouts = {}
    target_rows = torch.tensor(
        [
            [20, 21, 22, 23, 90],
            [30, 91, 32, 33, 99],
        ],
        dtype=torch.long,
    )
    engine._run_device_packed_greedy = MagicMock(return_value=target_rows.flatten())
    engine._validate_packed_causal_fia_leakage_once = MagicMock()
    states = [
        PearlPipelineState([1, 101], prompt_length=1),
        PearlPipelineState([4, 5, 102], prompt_length=1),
        PearlPipelineState([7, 8, 9, 103], prompt_length=1),
    ]
    active_indices = [2, 0]
    proposals = [
        torch.tensor([20, 21, 22, 23]),
        torch.tensor([30, 31, 32, 33]),
    ]
    payloads = [
        NativeSpecRhythmDevicePayload(
            ticket=SpecRhythmProposalTicket(
                proposal_id=row,
                request_index=request_index,
                home_batch_id=row,
                gamma=4,
                required_prefix_epoch=0,
            ),
            verification_tokens=proposal,
            next_tokens=proposal,
            verification_size=4,
            draft_confidence=1.0,
        )
        for row, (request_index, proposal) in enumerate(zip(active_indices, proposals))
    ]

    with patch.dict(
        os.environ,
        {
            "VLLM_ASCEND_SPECRHYTHM_FORCE_STEPWISE_TARGET": "0",
            "VLLM_ASCEND_SPECRHYTHM_DISABLE_TARGET_ACLGRAPH": "0",
        },
    ):
        target_tokens, logits = engine._target_full_window_outputs_batch(
            states,
            active_indices,
            payloads,
        )

    assert logits is None
    assert target_tokens.reshape(2, 5).tolist() == target_rows.tolist()
    target_call = engine._run_device_packed_greedy.call_args
    assert target_call.args[0].tolist() == [
        103,
        20,
        21,
        22,
        23,
        101,
        30,
        31,
        32,
        33,
    ]
    assert target_call.args[1] == [2, 2, 2, 2, 2, 0, 0, 0, 0, 0]
    assert target_call.args[2] == [3, 4, 5, 6, 7, 1, 2, 3, 4, 5]

    verdict = engine._verify_target_tokens_batch(
        target_tokens,
        None,
        torch.stack(proposals).flatten(),
        [4, 4],
        [0.0, 0.0],
        bonus_enabled=[True, True],
    )

    assert verdict is not None
    assert verdict.tolist() == [[4, 90], [1, 91]]


def test_mixed_target_prefill_fuses_disjoint_rows_in_one_eager_fia_forward():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.gamma = 4
    engine.device = torch.device("cpu")
    engine.draft_vocab_size = 64
    engine.config = SimpleNamespace(
        target_use_paged_attention=False,
        max_num_batched_tokens=32,
    )
    engine.cache_allocation = SimpleNamespace(num_cached_tokens=[0, 0, 1, 0])
    engine._run_device_packed_hidden = MagicMock(return_value=torch.arange(24, dtype=torch.float32).reshape(12, 2))
    engine.model = MagicMock()
    engine.model.compute_greedy_tokens.return_value = torch.arange(10, dtype=torch.long) + 1000
    states = [
        PearlPipelineState([10, 100], prompt_length=1),
        PearlPipelineState([20, 21, 101], prompt_length=2),
        PearlPipelineState([30, 31, 32], prompt_length=3),
        PearlPipelineState([40, 41], prompt_length=2),
    ]
    proposals = [
        torch.tensor([200, 201, 202, 203]),
        torch.tensor([210, 211, 212, 213]),
    ]
    payloads = [
        NativeSpecRhythmDevicePayload(
            ticket=SpecRhythmProposalTicket(
                proposal_id=row,
                request_index=row,
                home_batch_id=row,
                gamma=4,
                required_prefix_epoch=0,
            ),
            verification_tokens=proposal,
            next_tokens=proposal,
            verification_size=4,
            draft_confidence=1.0,
        )
        for row, proposal in enumerate(proposals)
    ]

    with patch.object(
        native_engine_module.torch,
        "cat",
        wraps=torch.cat,
    ) as cat_mock:
        target_tokens, logits, first_tokens = engine._target_full_window_outputs_with_prefill_batch(
            states,
            [0, 1],
            payloads,
            [2, 3],
        )

    assert logits is None
    assert target_tokens.tolist() == list(range(1000, 1008))
    assert first_tokens.tolist() == [1008, 1009]
    # One cat assembles real verification rows and one assembles the compact
    # eager mixed tensor.
    assert cat_mock.call_count == 2
    call_args = engine._run_device_packed_hidden.call_args
    assert torch.equal(
        call_args.args[0],
        torch.tensor([100, 200, 201, 202, 101, 210, 211, 212, 31, 32, 40, 41]),
    )
    assert call_args.args[1] == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        3,
        3,
    ]
    assert call_args.args[2] == [1, 2, 3, 4, 2, 3, 4, 5, 1, 2, 0, 1]
    assert call_args.kwargs == {
        "use_aclgraph": False,
        "use_fused_infer_attention": True,
    }
    selected_hidden = engine.model.compute_greedy_tokens.call_args.args[0]
    expected_hidden = torch.arange(24, dtype=torch.float32).reshape(12, 2)[[*range(8), 9, 11]]
    assert torch.equal(selected_hidden, expected_hidden)
    assert engine._last_mixed_target_graph_outcome == ("disabled", 0, 0)


def test_mixed_target_prefill_uses_stable_graph_and_real_output_indices(
    monkeypatch,
):
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.gamma = 4
    engine.device = torch.device("cpu")
    engine.draft_vocab_size = 64
    engine._mixed_target_graph_enabled = True
    engine.config = SimpleNamespace(
        target_use_paged_attention=False,
        max_num_batched_tokens=256,
        max_num_queued_seqs=4,
        max_num_seqs=4,
        max_model_len=4096,
        enforce_eager=False,
        spec_rhythm_linear_bonus_token=False,
    )
    engine.cache_allocation = SimpleNamespace(num_cached_tokens=[0, 0, 1, 0])
    engine._run_device_packed_hidden = MagicMock()
    engine._prepare_mixed_target_graph_call = MagicMock(return_value=(torch.arange(101), SimpleNamespace()))
    engine.graph_runner = MagicMock()
    engine.graph_runner.max_graph_tokens = 101
    engine.graph_runner.run_stable_fia_hidden.return_value = torch.arange(
        202,
        dtype=torch.float32,
    ).reshape(101, 2)
    engine.model = MagicMock()
    engine.model.compute_greedy_tokens.return_value = torch.arange(10, dtype=torch.long) + 1000
    monkeypatch.setattr(
        native_engine_module.envs,
        "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH",
        True,
    )
    states = [
        PearlPipelineState([10, 100], prompt_length=1),
        PearlPipelineState([20, 21, 101], prompt_length=2),
        PearlPipelineState([30, 31, 32], prompt_length=3),
        PearlPipelineState([40, 41], prompt_length=2),
    ]
    proposals = [
        torch.tensor([200, 201, 202, 203]),
        torch.tensor([210, 211, 212, 213]),
    ]
    payloads = [
        NativeSpecRhythmDevicePayload(
            ticket=SpecRhythmProposalTicket(
                proposal_id=row,
                request_index=row,
                home_batch_id=row,
                gamma=4,
                required_prefix_epoch=0,
            ),
            verification_tokens=proposal,
            next_tokens=proposal,
            verification_size=4,
            draft_confidence=1.0,
        )
        for row, proposal in enumerate(proposals)
    ]

    with patch.object(
        native_engine_module.torch,
        "cat",
        wraps=torch.cat,
    ) as cat_mock:
        target_tokens, logits, first_tokens = engine._target_full_window_outputs_with_prefill_batch(
            states,
            [0, 1],
            payloads,
            [2, 3],
        )

    assert logits is None
    assert target_tokens.tolist() == list(range(1000, 1008))
    assert first_tokens.tolist() == [1008, 1009]
    # One cat assembles real verification rows and one assembles the fixed
    # graph envelope. The compact eager-only mixed tensor must stay unbuilt.
    assert cat_mock.call_count == 2
    engine._run_device_packed_hidden.assert_not_called()
    graph_call = engine.graph_runner.run_stable_fia_hidden.call_args
    graph_inputs = graph_call.args[0]
    assert graph_inputs.shape == (101,)
    assert graph_inputs[:8].tolist() == [
        100,
        200,
        201,
        202,
        101,
        210,
        211,
        212,
    ]
    assert graph_inputs[32:36].tolist() == [31, 32, 40, 41]
    assert torch.count_nonzero(graph_inputs[8:32]) == 0
    assert graph_call.kwargs == {
        "graph_key": ("mixed-target-hidden|gamma:4|verify:8|prompt:4+pad1|prompt-tokens:64"),
        "expected_tokens": 101,
        "expected_request_segments": 13,
    }
    selected_hidden = engine.model.compute_greedy_tokens.call_args.args[0]
    expected_hidden = torch.arange(202, dtype=torch.float32).reshape(101, 2)[[*range(8), 33, 35]]
    assert torch.equal(selected_hidden, expected_hidden)
    assert engine._last_mixed_target_graph_outcome == ("graph", 4, 4)


def test_mixed_target_prefill_graph_accepts_partial_token_chunks(monkeypatch):
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.gamma = 4
    engine.device = torch.device("cpu")
    engine.draft_vocab_size = 64
    engine._mixed_target_graph_enabled = True
    engine.config = SimpleNamespace(
        target_use_paged_attention=False,
        max_num_batched_tokens=256,
        max_num_queued_seqs=4,
        max_num_seqs=4,
        max_model_len=4096,
        enforce_eager=False,
        spec_rhythm_linear_bonus_token=False,
    )
    engine.cache_allocation = SimpleNamespace(num_cached_tokens=[0, 0, 0, 0])
    engine._prepare_mixed_target_graph_call = MagicMock(return_value=(torch.arange(101), SimpleNamespace()))
    engine.graph_runner = MagicMock()
    engine.graph_runner.max_graph_tokens = 101
    engine.graph_runner.run_stable_fia_hidden.return_value = torch.arange(
        202,
        dtype=torch.float32,
    ).reshape(101, 2)
    engine.model = MagicMock()
    engine.model.compute_greedy_tokens.return_value = torch.arange(9, dtype=torch.long) + 1000
    monkeypatch.setattr(
        native_engine_module.envs,
        "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH",
        True,
    )
    states = [
        PearlPipelineState([10, 100], prompt_length=1),
        PearlPipelineState([20, 21, 101], prompt_length=2),
        PearlPipelineState([30, 31, 32], prompt_length=3),
        PearlPipelineState([40, 41], prompt_length=2),
    ]
    proposals = [
        torch.tensor([200, 201, 202, 203]),
        torch.tensor([210, 211, 212, 213]),
    ]
    payloads = [
        NativeSpecRhythmDevicePayload(
            ticket=SpecRhythmProposalTicket(
                proposal_id=row,
                request_index=row,
                home_batch_id=row,
                gamma=4,
                required_prefix_epoch=0,
            ),
            verification_tokens=proposal,
            next_tokens=proposal,
            verification_size=4,
            draft_confidence=1.0,
        )
        for row, proposal in enumerate(proposals)
    ]
    chunks = (
        native_engine_module.SpecRhythmPrefillTokenChunk(2, 1, 2, 3),
        native_engine_module.SpecRhythmPrefillTokenChunk(3, 0, 2, 2),
    )

    target_tokens, logits, first_tokens = engine._target_full_window_outputs_with_prefill_batch(
        states,
        [0, 1],
        payloads,
        [2, 3],
        prefill_chunks=chunks,
    )

    assert logits is None
    assert target_tokens.tolist() == list(range(1000, 1008))
    assert first_tokens.tolist() == [1008]
    graph_inputs = engine.graph_runner.run_stable_fia_hidden.call_args.args[0]
    assert graph_inputs[32:35].tolist() == [31, 40, 41]
    selected_hidden = engine.model.compute_greedy_tokens.call_args.args[0]
    expected_hidden = torch.arange(202, dtype=torch.float32).reshape(101, 2)[[*range(8), 34]]
    assert torch.equal(selected_hidden, expected_hidden)
    envelope = engine._prepare_mixed_target_graph_call.call_args.args[1]
    assert envelope.positions[32:35] == (1, 0, 1)
    assert engine._last_mixed_target_graph_outcome == ("graph", 4, 3)


def test_mixed_target_graph_capacity_shortfall_keeps_unpadded_eager_path(
    monkeypatch,
):
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.gamma = 4
    engine.device = torch.device("cpu")
    engine.draft_vocab_size = 64
    engine._mixed_target_graph_enabled = True
    engine.config = SimpleNamespace(
        target_use_paged_attention=False,
        max_num_batched_tokens=32,
        max_num_queued_seqs=4,
        max_num_seqs=4,
        max_model_len=4096,
        enforce_eager=False,
        spec_rhythm_linear_bonus_token=False,
    )
    engine.cache_allocation = SimpleNamespace(num_cached_tokens=[0, 1])
    engine.graph_runner = MagicMock(max_graph_tokens=100)
    engine._run_device_packed_hidden = MagicMock(return_value=torch.arange(12, dtype=torch.float32).reshape(6, 2))
    engine.model = MagicMock()
    engine.model.compute_greedy_tokens.return_value = torch.arange(5)
    monkeypatch.setattr(
        native_engine_module.envs,
        "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH",
        True,
    )
    states = [
        PearlPipelineState([10, 100], prompt_length=1),
        PearlPipelineState([20, 21], prompt_length=2),
    ]
    proposal = torch.tensor([200, 201, 202, 203])
    payload = NativeSpecRhythmDevicePayload(
        ticket=SpecRhythmProposalTicket(
            proposal_id=0,
            request_index=0,
            home_batch_id=0,
            gamma=4,
            required_prefix_epoch=0,
        ),
        verification_tokens=proposal,
        next_tokens=proposal,
        verification_size=4,
        draft_confidence=1.0,
    )

    engine._target_full_window_outputs_with_prefill_batch(
        states,
        [0],
        [payload],
        [1],
    )

    engine.graph_runner.run_stable_fia_hidden.assert_not_called()
    eager_inputs = engine._run_device_packed_hidden.call_args.args[0]
    assert eager_inputs.tolist() == [100, 200, 201, 202, 21]
    assert engine._last_mixed_target_graph_outcome == (
        "token_capacity",
        2,
        1,
    )


def test_prefill_broadcast_uses_precomputed_mixed_target_tokens(monkeypatch):
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.device = torch.device("cpu")
    engine.cache_allocation = SimpleNamespace(num_cached_tokens=[0, 0])
    engine.topology = SimpleNamespace(target_leader_rank=0)
    engine._run_packed_sample = MagicMock()
    engine._vote_spec_rhythm_prefill_finiteness = MagicMock()
    monkeypatch.setattr(native_engine_module.dist, "broadcast", MagicMock())
    states = [
        PearlPipelineState([10, 11], prompt_length=2),
        PearlPipelineState([20, 21], prompt_length=2),
    ]

    tokens = engine._prefill_and_sample_target_batch(
        [[10, 11], [20, 21]],
        states,
        [0, 1],
        precomputed_target_tokens=torch.tensor([31, 41]),
    )

    assert tokens == [31, 41]
    engine._run_packed_sample.assert_not_called()
    engine._vote_spec_rhythm_prefill_finiteness.assert_called_once_with()


def test_draft_prefill_hidden_batch_populates_only_uncached_prompt_suffixes():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.device = torch.device("cpu")
    engine.cache_allocation = SimpleNamespace(num_cached_tokens=[0, 2, 1])
    expected = torch.tensor([[1.0], [2.0]])
    engine._run_packed_hidden = MagicMock(return_value=expected)

    hidden = engine._draft_prefill_hidden_batch(
        [[10, 11, 12, 13], [20, 21, 22]],
        [1, 2],
    )

    assert hidden is expected
    engine._run_packed_hidden.assert_called_once_with(
        [12, 13, 21, 22],
        [1, 1, 2, 2],
        [2, 3, 1, 2],
        use_aclgraph=False,
        logit_indices=[1, 3],
        use_fused_infer_attention=False,
    )


def test_mixed_draft_prefill_uses_one_fia_first_step_and_three_step_pa_graph():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.gamma = 4
    engine.device = torch.device("cpu")
    engine.draft_vocab_size = 64
    engine.config = SimpleNamespace(
        draft_use_paged_attention=True,
        enforce_eager=False,
        max_num_batched_tokens=32,
        max_num_seqs=8,
    )
    engine.cache_allocation = SimpleNamespace(num_cached_tokens=[0, 0, 1, 0])
    engine._run_device_packed_hidden = MagicMock(return_value=torch.arange(12, dtype=torch.float32).reshape(6, 2))
    engine.model = MagicMock()
    engine.model.compute_greedy_tokens_with_confidence.return_value = (
        torch.tensor([500, 501]),
        torch.tensor([0.8, 0.4]),
    )

    def metadata(_rows, positions, use_fused_infer_attention):
        assert use_fused_infer_attention is False
        return (
            torch.tensor(positions),
            SimpleNamespace(
                slot_mapping=torch.arange(2, dtype=torch.int32),
                context_lens=torch.tensor(positions, dtype=torch.int32) + 1,
                block_tables=torch.zeros((2, 2), dtype=torch.int32),
                request_block_tables=None,
                actual_seq_lengths_q=(1, 2),
                sequence_lens=tuple(position + 1 for position in positions),
                attention_mask=None,
                use_fused_infer_attention=False,
            ),
        )

    engine._prepare_attention_metadata = MagicMock(side_effect=metadata)
    engine.graph_runner = MagicMock()
    engine.graph_runner.run_draft_greedy.return_value = torch.tensor(
        [
            [[600, 700_000], [601, 600_000], [602, 500_000]],
            [[610, 300_000], [611, 200_000], [612, 100_000]],
        ]
    )
    states = [
        PearlPipelineState([10, 100], prompt_length=1),
        PearlPipelineState([20, 21, 101], prompt_length=2),
        PearlPipelineState([30, 31, 32], prompt_length=3),
        PearlPipelineState([40, 41], prompt_length=2),
    ]

    verification, windows, confidence = engine._draft_spec_rhythm_device_batch_with_prefill(
        states,
        [0, 1],
        [4, 4],
        [2, 3],
    )

    assert windows.tolist() == [
        [500, 600, 601, 602],
        [501, 610, 611, 612],
    ]
    assert torch.equal(verification, windows.flatten())
    assert torch.allclose(confidence, torch.tensor([0.65, 0.25]))
    mixed_call = engine._run_device_packed_hidden.call_args
    assert torch.equal(
        mixed_call.args[0],
        torch.tensor([100, 101, 31, 32, 40, 41]),
    )
    assert mixed_call.args[1] == [0, 1, 2, 2, 3, 3]
    assert mixed_call.args[2] == [1, 2, 1, 2, 0, 1]
    assert mixed_call.kwargs == {
        "use_aclgraph": False,
        "use_fused_infer_attention": True,
    }
    graph_call = engine.graph_runner.run_draft_greedy.call_args
    assert torch.equal(graph_call.args[0], torch.tensor([500, 501]))
    assert len(graph_call.args[1]) == 3
    assert len(graph_call.args[2]) == 3
    assert graph_call.kwargs == {
        "valid_row_count": 2,
        "return_confidence": True,
    }


def test_packed_causal_fia_leakage_probe_is_explicit_and_runs_once():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.gamma = 4
    engine.draft_vocab_size = 64
    engine.rank = 1
    engine.model = MagicMock()
    reference_hidden = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    perturbed_hidden = reference_hidden.clone()
    perturbed_hidden[[3, 7]] += 10
    reference_tokens = torch.arange(8, dtype=torch.long)
    perturbed_tokens = reference_tokens.clone()
    perturbed_tokens[[3, 7]] += 10
    engine._run_device_packed_hidden = MagicMock(
        side_effect=[
            reference_hidden,
            perturbed_hidden,
            reference_hidden.clone(),
        ]
    )
    engine.model.compute_greedy_tokens.side_effect = [
        reference_tokens,
        perturbed_tokens,
        reference_tokens.clone(),
    ]
    query_tokens = torch.tensor([10, 11, 12, 13, 20, 21, 22, 23])
    sequence_ids = [2, 2, 2, 2, 0, 0, 0, 0]
    positions = [7, 8, 9, 10, 3, 4, 5, 6]

    with patch.dict(
        os.environ,
        {"VLLM_ASCEND_SPECRHYTHM_VALIDATE_PACKED_CAUSAL_LEAKAGE": "0"},
    ):
        engine._validate_packed_causal_fia_leakage_once(
            query_tokens,
            sequence_ids,
            positions,
        )
    engine._run_device_packed_hidden.assert_not_called()

    with patch.dict(
        os.environ,
        {"VLLM_ASCEND_SPECRHYTHM_VALIDATE_PACKED_CAUSAL_LEAKAGE": "1"},
    ):
        engine._validate_packed_causal_fia_leakage_once(
            query_tokens,
            sequence_ids,
            positions,
        )
        engine._validate_packed_causal_fia_leakage_once(
            query_tokens,
            sequence_ids,
            positions,
        )

    assert engine._run_device_packed_hidden.call_count == 3
    calls = engine._run_device_packed_hidden.call_args_list
    assert torch.equal(calls[0].args[0], query_tokens)
    assert torch.equal(calls[2].args[0], query_tokens)
    assert calls[1].args[0].tolist() == [10, 11, 12, 14, 20, 21, 22, 24]
    assert all(call.kwargs["use_aclgraph"] is False for call in calls)
    assert all(call.kwargs["use_fused_infer_attention"] is True for call in calls)
    assert engine._packed_causal_leakage_probe_completed is True
    assert engine._packed_causal_leakage_probe_attempts == 1
    assert engine._packed_causal_leakage_probe_passes == 1
    assert engine._packed_causal_leakage_probe_failures == 0
    assert engine._packed_causal_leakage_probe_hidden_mismatch_steps == 0
    assert engine._packed_causal_leakage_probe_token_mismatch_steps == 0
    assert engine._packed_causal_leakage_probe_max_abs_diff == 0.0


def test_packed_causal_fia_leakage_probe_restores_kv_before_failure():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.gamma = 4
    engine.draft_vocab_size = 64
    engine.rank = 1
    engine.model = MagicMock()
    reference_hidden = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    leaked_hidden = reference_hidden.clone()
    leaked_hidden[0, 0] += 1
    reference_tokens = torch.arange(4, dtype=torch.long)
    leaked_tokens = reference_tokens.clone()
    leaked_tokens[1] += 1
    engine._run_device_packed_hidden = MagicMock(
        side_effect=[
            reference_hidden,
            leaked_hidden,
            reference_hidden.clone(),
        ]
    )
    engine.model.compute_greedy_tokens.side_effect = [
        reference_tokens,
        leaked_tokens,
        reference_tokens.clone(),
    ]
    query_tokens = torch.tensor([10, 11, 12, 13])

    with (
        patch.dict(
            os.environ,
            {"VLLM_ASCEND_SPECRHYTHM_VALIDATE_PACKED_CAUSAL_LEAKAGE": "1"},
        ),
        pytest.raises(RuntimeError, match="hidden_mismatch_steps=1"),
    ):
        engine._validate_packed_causal_fia_leakage_once(
            query_tokens,
            [0, 0, 0, 0],
            [3, 4, 5, 6],
        )

    calls = engine._run_device_packed_hidden.call_args_list
    assert len(calls) == 3
    # The last call replays the original token row even though validation is
    # about to fail, restoring the KV slot changed by the perturbation.
    assert torch.equal(calls[-1].args[0], query_tokens)
    assert engine._packed_causal_leakage_probe_attempts == 1
    assert engine._packed_causal_leakage_probe_passes == 0
    assert engine._packed_causal_leakage_probe_failures == 1
    assert engine._packed_causal_leakage_probe_hidden_mismatch_steps == 1
    assert engine._packed_causal_leakage_probe_token_mismatch_steps == 1
    assert engine._packed_causal_leakage_probe_restore_hidden_mismatch_steps == 0
    assert engine._packed_causal_leakage_probe_restore_token_mismatch_steps == 0


def test_target_full_window_invokes_packed_causal_probe_before_normal_fia():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.gamma = 4
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(
        enforce_eager=False,
        target_use_paged_attention=False,
        spec_rhythm_stable_graphs=True,
    )
    engine._validate_packed_causal_fia_leakage_once = MagicMock()
    engine._run_device_packed_greedy = MagicMock(return_value=torch.arange(4))
    state = PearlPipelineState([1, 101], prompt_length=1)
    proposal = torch.tensor([20, 21, 22, 23])
    payload = NativeSpecRhythmDevicePayload(
        ticket=SpecRhythmProposalTicket(
            proposal_id=0,
            request_index=0,
            home_batch_id=0,
            gamma=4,
            required_prefix_epoch=0,
        ),
        verification_tokens=proposal,
        next_tokens=proposal,
        verification_size=4,
        draft_confidence=1.0,
    )

    output, _ = engine._target_full_window_outputs_batch(
        [state],
        [0],
        [payload],
    )

    query_tokens = torch.tensor([101, 20, 21, 22])
    probe_call = engine._validate_packed_causal_fia_leakage_once.call_args
    assert torch.equal(probe_call.args[0], query_tokens)
    assert probe_call.args[1:] == ([0, 0, 0, 0], [1, 2, 3, 4])
    production_call = engine._run_device_packed_greedy.call_args
    assert torch.equal(production_call.args[0], query_tokens)
    assert production_call.kwargs == {
        "use_aclgraph": True,
        "use_fused_infer_attention": True,
    }
    assert output.tolist() == [0, 1, 2, 3]


def test_target_full_window_rejects_an_uncommitted_target_suffix_before_forward():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.gamma = 4
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(
        enforce_eager=True,
        target_use_paged_attention=True,
        spec_rhythm_stable_graphs=True,
    )
    engine._run_device_packed_greedy = MagicMock()
    state = PearlPipelineState([1, 2, 3, 99], prompt_length=2, committed_length=3)
    proposal = torch.tensor([10, 11, 12, 13])
    payload = NativeSpecRhythmDevicePayload(
        ticket=SpecRhythmProposalTicket(
            proposal_id=0,
            request_index=0,
            home_batch_id=0,
            gamma=engine.gamma,
            required_prefix_epoch=0,
        ),
        verification_tokens=proposal,
        next_tokens=proposal,
        verification_size=engine.gamma,
        draft_confidence=1.0,
    )

    with pytest.raises(RuntimeError, match="uncommitted token suffix"):
        engine._target_full_window_outputs_batch([state], [0], [payload])

    engine._run_device_packed_greedy.assert_not_called()


@pytest.mark.parametrize(("batch_size", "gamma"), [(65, 4), (1, 9)])
def test_target_full_window_uses_causal_pa_outside_validated_packed_boundary(
    batch_size,
    gamma,
):
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.gamma = gamma
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(
        enforce_eager=True,
        target_use_paged_attention=True,
        spec_rhythm_stable_graphs=True,
    )
    engine._run_device_packed_greedy = MagicMock(side_effect=[torch.arange(batch_size) for _ in range(gamma)])
    states = [PearlPipelineState([1, 100 + row], prompt_length=1) for row in range(batch_size)]
    active_indices = list(range(batch_size))
    payloads = [
        NativeSpecRhythmDevicePayload(
            ticket=SpecRhythmProposalTicket(
                proposal_id=row,
                request_index=row,
                home_batch_id=row % 2,
                gamma=gamma,
                required_prefix_epoch=0,
            ),
            verification_tokens=torch.arange(gamma) + 200 + row * gamma,
            next_tokens=torch.arange(gamma) + 200 + row * gamma,
            verification_size=gamma,
            draft_confidence=1.0,
        )
        for row in range(batch_size)
    ]

    with patch.dict(
        os.environ,
        {
            "VLLM_ASCEND_SPECRHYTHM_FORCE_STEPWISE_TARGET": "0",
            "VLLM_ASCEND_SPECRHYTHM_PACKED_TARGET": "0",
        },
    ):
        output, _ = engine._target_full_window_outputs_batch(
            states,
            active_indices,
            payloads,
        )

    assert output.shape == (batch_size * gamma,)
    assert engine._run_device_packed_greedy.call_count == gamma


@pytest.mark.parametrize("return_logits", [False, True])
def test_tree_eager_diagnostic_logits_are_optional(return_logits):
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.device = torch.device("cpu")
    engine.target_vocab_size = 32
    engine.config = SimpleNamespace(enforce_eager=True)
    engine.cache_allocation = SimpleNamespace(block_tables=torch.zeros((1, 1), dtype=torch.int32))
    engine.cache_block_tables = torch.zeros((1, 1), dtype=torch.int32)
    engine._ensure_cache_capacity = MagicMock()
    engine.model = MagicMock()
    engine.model.make_tree_attention_metadata.return_value = (
        torch.tensor([1, 2]),
        torch.tensor([1, 2]),
        SimpleNamespace(
            slot_mapping=torch.tensor([0, 1], dtype=torch.int32),
            use_fused_infer_attention=False,
            tree_attention=True,
            attention_mask=torch.tensor([[False, True], [False, False]]),
        ),
    )
    engine.model.compute_greedy_tokens.return_value = torch.tensor([7, 8])
    plan = build_tree_speculation_plan(1, 1, 1, 8)

    result = engine.target_tree_forward([plan], [1], [[2]], return_logits=return_logits)

    assert result["target_query_token_ids"].tolist() == [7, 8]
    assert result["attention_backend"] == "dense_sdpa_tree_v1"
    engine.model.compute_greedy_tokens.assert_called_once()
    assert engine.model.compute_logits.call_count == int(return_logits)
    if not return_logits:
        assert result["target_logits"] is None


def test_target_tree_forward_uses_aclgraph_for_a_static_tree_shape():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.device = torch.device("cpu")
    engine.target_vocab_size = 32
    engine.config = SimpleNamespace(enforce_eager=False)
    engine.cache_allocation = SimpleNamespace(block_tables=torch.zeros((1, 1), dtype=torch.int32))
    engine.cache_block_tables = torch.zeros((1, 1), dtype=torch.int32)
    engine._ensure_cache_capacity = MagicMock()
    engine.graph_runner = MagicMock()
    engine.graph_runner.run_target_greedy.return_value = (torch.tensor([7, 8, 9]),)
    engine.graph_runner.last_target_execution = SimpleNamespace(used_aclgraph=True)
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int32),
        use_fused_infer_attention=True,
        tree_attention=True,
        attention_mask=torch.zeros((1, 1, 3, 8), dtype=torch.bool),
    )
    engine.model = MagicMock()
    engine.model.make_tree_attention_metadata.return_value = (
        torch.tensor([1, 2, 3]),
        torch.tensor([1, 2, 3]),
        metadata,
    )
    plan = build_tree_speculation_plan(
        width=2,
        depth=1,
        prefix_len=1,
        max_model_len=8,
        candidate_budget=2,
    )

    result = engine.target_tree_forward([plan], [1], [[2, 3]])

    assert result["target_token_ids"].tolist() == [7, 8]
    assert result["bonus_token_ids"].tolist() == [9]
    assert result["used_aclgraph"] is True
    assert result["attention_backend"] == "fused_infer_attention_tree_v1"
    assert result["used_tree_attention"] is True
    assert engine.model.make_tree_attention_metadata.call_args.kwargs["allow_causal_fast_path"] is True
    engine.graph_runner.run_target_greedy.assert_called_once()
    engine.model.assert_not_called()


def test_target_tree_forward_samples_graph_logits_from_common_vocabulary():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.device = torch.device("cpu")
    engine.draft_vocab_size = 16
    engine.target_vocab_size = 32
    engine.config = SimpleNamespace(enforce_eager=False)
    engine.cache_allocation = SimpleNamespace(block_tables=torch.zeros((1, 1), dtype=torch.int32))
    engine.cache_block_tables = torch.zeros((1, 1), dtype=torch.int32)
    engine._ensure_cache_capacity = MagicMock()
    engine.graph_runner = MagicMock()
    logits = torch.full((3, 16), -100.0)
    logits[0, 2] = logits[1, 9] = logits[2, 10] = 100.0
    engine.graph_runner.run_tree_logits.return_value = logits
    engine.graph_runner.last_target_execution = SimpleNamespace(used_aclgraph=True)
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int32),
        use_fused_infer_attention=True,
        tree_attention=True,
        attention_mask=torch.zeros((1, 1, 3, 8), dtype=torch.bool),
    )
    engine.model = MagicMock()
    engine.model.make_tree_attention_metadata.return_value = (
        torch.tensor([1, 2, 3]),
        torch.tensor([1, 2, 2]),
        metadata,
    )
    plan = build_tree_speculation_plan(2, 1, 1, 8, candidate_budget=2)

    result = engine.target_tree_forward([plan], [1], [[2, 3]], temperatures=[0.7])
    verdict = engine.verify_tree_outputs(
        torch.tensor([2, 3]),
        plan.parent_indices,
        result["target_query_token_ids"],
        result["bonus_token_ids"],
        [2],
        1,
    )

    assert result["target_query_token_ids"].tolist() == [2, 9, 10]
    assert verdict.token_ids.tolist() == [[2, 9]]
    assert verdict.accepted_node_indices.tolist() == [[0]]
    engine.graph_runner.run_tree_logits.assert_called_once()
    assert engine.graph_runner.run_tree_logits.call_args.args[-1] == 16
    engine.graph_runner.run_target_greedy.assert_not_called()


def test_target_tree_forward_packs_only_budgeted_nodes_and_uses_frontier_bonus():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.device = torch.device("cpu")
    engine.target_vocab_size = 32
    engine.config = SimpleNamespace(enforce_eager=False)
    engine.cache_allocation = SimpleNamespace(block_tables=torch.zeros((1, 1), dtype=torch.int32))
    engine.cache_block_tables = torch.zeros((1, 1), dtype=torch.int32)
    engine._ensure_cache_capacity = MagicMock()
    engine.graph_runner = MagicMock()
    # A one-node candidate budget requires exactly two physical queries:
    # root + active node.  The inactive sibling must not reach the model.
    engine.graph_runner.run_target_greedy.return_value = (torch.tensor([7, 8]),)
    engine.graph_runner.last_target_execution = SimpleNamespace(used_aclgraph=True)
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([1, 2], dtype=torch.int32),
        use_fused_infer_attention=True,
        tree_attention=True,
        attention_mask=torch.zeros((1, 1, 2, 8), dtype=torch.bool),
    )
    engine.model = MagicMock()
    engine.model.make_tree_attention_metadata.return_value = (
        torch.tensor([1, 2]),
        torch.tensor([1, 2]),
        metadata,
    )
    plan = build_tree_speculation_plan(
        width=2,
        depth=1,
        prefix_len=1,
        max_model_len=8,
        candidate_budget=1,
    )

    result = engine.target_tree_forward([plan], [1], [[2]])

    assert result["target_token_ids"].tolist() == [7]
    assert result["bonus_token_ids"].tolist() == [8]
    packed_plan = engine.model.make_tree_attention_metadata.call_args.args[0][0]
    packed_row = engine.model.make_tree_attention_metadata.call_args.args[2][0]
    assert packed_row == [2]
    assert packed_plan.parent_indices.tolist() == [-1]
    assert packed_plan.positions.tolist() == [1, 2]
    assert packed_plan.cache_positions.tolist() == [1, 2]
    assert result["query_count"] == 2
    assert result["cache_slot_mapping"].tolist() == [1, 2]
    assert result["target_query_token_ids"].tolist() == [7, 8]
    engine._ensure_cache_capacity.assert_called_once_with([0, 0], [1, 2])


def test_target_tree_forward_reports_graph_fallback_as_eager():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.device = torch.device("cpu")
    engine.target_vocab_size = 32
    engine.config = SimpleNamespace(enforce_eager=False)
    engine.cache_allocation = SimpleNamespace(block_tables=[[0]])
    engine.cache_block_tables = torch.zeros((1, 1), dtype=torch.int32)
    engine._ensure_cache_capacity = MagicMock()
    engine.graph_runner = MagicMock()
    engine.graph_runner.run_target_greedy.return_value = (torch.tensor([7, 8]),)
    engine.graph_runner.last_target_execution = SimpleNamespace(used_aclgraph=False)
    engine.model = MagicMock()
    engine.model.make_tree_attention_metadata.return_value = (
        torch.tensor([1, 2]),
        torch.tensor([1, 2]),
        SimpleNamespace(
            slot_mapping=torch.tensor([1, 2], dtype=torch.int32),
            use_fused_infer_attention=True,
            tree_attention=True,
            attention_mask=torch.zeros((1, 1, 2, 8), dtype=torch.bool),
        ),
    )
    plan = build_tree_speculation_plan(2, 1, 1, 8, candidate_budget=1)

    result = engine.target_tree_forward([plan], [1], [[2]])

    assert result["used_aclgraph"] is False
    assert result["query_count"] == 2


def test_target_tree_forward_packs_variable_request_budgets_into_one_forward():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.device = torch.device("cpu")
    engine.target_vocab_size = 32
    engine.config = SimpleNamespace(enforce_eager=False)
    engine.cache_allocation = SimpleNamespace(block_tables=[[0], [1]])
    engine.cache_block_tables = torch.zeros((2, 1), dtype=torch.int32)
    engine._ensure_cache_capacity = MagicMock()
    engine.graph_runner = MagicMock()
    engine.graph_runner.run_target_greedy.return_value = (torch.arange(7, 13),)
    engine.graph_runner.last_target_execution = SimpleNamespace(used_aclgraph=True)
    engine.model = MagicMock()
    engine.model.make_tree_attention_metadata.return_value = (
        torch.tensor([1, 2, 10, 20, 21, 22]),
        torch.tensor([1, 2, 6, 7, 8, 7]),
        SimpleNamespace(
            slot_mapping=torch.tensor([1, 2, 6, 7, 8, 9], dtype=torch.int32),
            use_fused_infer_attention=True,
            tree_attention=True,
            attention_mask=torch.zeros((2, 1, 4, 16), dtype=torch.bool),
        ),
    )
    plans = [
        build_tree_speculation_plan(2, 2, 1, 16, candidate_budget=1),
        build_tree_speculation_plan(2, 2, 6, 16, candidate_budget=3),
    ]

    result = engine.target_tree_forward(plans, [1, 10], [[2], [20, 21, 22]], sequence_ids=[0, 1])

    packed = engine.model.make_tree_attention_metadata.call_args.args[0]
    assert [plan.parent_indices.numel() for plan in packed] == [1, 3]
    assert result["query_count"] == 6  # Four candidates plus two request roots.
    assert result["num_draft_tokens"] == [1, 3]
    assert result["target_token_ids"].tolist() == [7, 9, 10, 11]
    assert result["target_query_token_ids"].tolist() == list(range(7, 13))
    assert result["bonus_token_ids"].tolist() == [8, 12]
    assert result["cache_slot_mapping"].tolist() == [1, 2, 6, 7, 8, 9]
    engine._ensure_cache_capacity.assert_called_once_with([0, 0, 1, 1, 1, 1], [1, 2, 6, 7, 8, 9])


def test_tree_target_graph_padding_preserves_real_rows_and_fills_exact_buckets():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.config = SimpleNamespace(
        enforce_eager=False,
        spec_rhythm_stable_graphs=True,
        spec_rhythm_tree_width=2,
        spec_rhythm_tree_depth=2,
        max_model_len=64,
    )
    states = [PearlPipelineState([index + 1] * (index + 2), prompt_length=index + 2) for index in range(4)]
    plans = [
        engine._spec_rhythm_tree_plan(states[0], 4),
        engine._spec_rhythm_tree_plan(states[1], 4),
    ]

    padded = engine._pad_spec_rhythm_target_tree_graph(
        plans,
        [11, 12],
        [[21, 22, 23, 24], [31, 32, 33, 34]],
        [0, 1],
        [0, 1, 2, 3],
        states,
    )

    padded_plans, roots, rows, request_ids, real_count = padded
    assert real_count == 2
    assert request_ids == [0, 1, 2, 3]
    assert roots[:2] == [11, 12]
    assert rows[:2] == [[21, 22, 23, 24], [31, 32, 33, 34]]
    # Two real five-query trees plus two three-query dummy trees fill the
    # four-token-aligned 16-query graph envelope exactly.
    assert [plan.candidate_budget for plan in padded_plans] == [4, 4, 2, 2]
    assert sum(plan.candidate_budget + 1 for plan in padded_plans) == 16
    assert padded_plans[2].parent_indices.tolist() == [-1, 0]
    assert padded_plans[3].parent_indices.tolist() == [-1, 0]


def test_tree_target_graph_padding_uses_reserved_rows_when_opposite_home_is_empty():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.config = SimpleNamespace(
        enforce_eager=False,
        enable_spec_rhythm=True,
        spec_rhythm_stable_graphs=True,
        spec_rhythm_tree_width=2,
        spec_rhythm_tree_depth=2,
        max_model_len=64,
        max_num_queued_seqs=4,
        max_num_seqs=4,
    )
    engine._mixed_target_graph_enabled = False
    states = [PearlPipelineState([index + 1] * (index + 2), prompt_length=index + 2) for index in range(2)]
    plans = [
        engine._spec_rhythm_tree_plan(states[0], 4),
        engine._spec_rhythm_tree_plan(states[1], 4),
    ]

    padded = engine._pad_spec_rhythm_target_tree_graph(
        plans,
        [11, 12],
        [[21, 22, 23, 24], [31, 32, 33, 34]],
        [0, 1],
        [0, 1],
        states,
    )

    padded_plans, roots, rows, request_ids, real_count = padded
    assert real_count == 2
    assert request_ids == [0, 1, 4, 5]
    assert roots == [11, 12, 0, 0]
    assert rows[-2:] == [[0, 0], [0, 0]]
    assert [plan.candidate_budget for plan in padded_plans] == [4, 4, 2, 2]
    assert sum(plan.candidate_budget + 1 for plan in padded_plans) == 16


def test_draft_tree_graph_padding_fixes_both_request_and_query_extents():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.config = SimpleNamespace(
        enforce_eager=False,
        enable_spec_rhythm=True,
        spec_rhythm_stable_graphs=True,
        spec_rhythm_tree_width=2,
        spec_rhythm_tree_depth=2,
        max_model_len=64,
        max_num_queued_seqs=8,
        max_num_seqs=8,
    )
    engine._mixed_target_graph_enabled = False
    engine._ensure_cache_capacity = MagicMock()
    query_lengths = [3, 3, 1, 1, 3]
    requests = []
    for request_id, query_count in enumerate(query_lengths):
        depth = max(1, query_count - 1)
        plan = build_tree_speculation_plan(1, depth, 2, 64, candidate_budget=depth)
        requests.append(
            (
                plan,
                request_id,
                list(range(-1, query_count - 1)) if query_count > 1 else -1,
                [request_id] * query_count if query_count > 1 else request_id,
            )
        )

    padded, padding_queries = engine._pad_tree_draft_graph_requests(requests)

    assert len(padded) == 8
    assert [request[1] for request in padded[5:]] == [8, 9, 10]
    scratch_lengths = [1 if isinstance(request[3], int) else len(request[3]) for request in padded[5:]]
    assert scratch_lengths == [5, 4, 4]
    assert padding_queries == 13
    assert sum(1 if isinstance(request[3], int) else len(request[3]) for request in padded) == 24
    assert engine._ensure_cache_capacity.call_count == 3


def test_tree_target_graph_padding_grows_bucket_for_shallow_tree():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.config = SimpleNamespace(
        enforce_eager=False,
        spec_rhythm_stable_graphs=True,
        spec_rhythm_tree_width=2,
        spec_rhythm_tree_depth=1,
        max_model_len=64,
    )
    states = [PearlPipelineState([index + 1] * (index + 2), prompt_length=index + 2) for index in range(4)]
    plan = engine._spec_rhythm_tree_plan(states[0], 2)

    padded = engine._pad_spec_rhythm_target_tree_graph([plan], [11], [[21, 22]], [0], [0, 1, 2, 3], states)

    padded_plans, _, _, request_ids, real_count = padded
    assert real_count == 1
    # One dummy row cannot fill the next four-query bucket for a 2x1 tree.
    # Four request rows can: 3 real queries + 3 * 3 dummy queries = 12.
    assert request_ids == [0, 1, 2, 3]
    assert [plan.candidate_budget for plan in padded_plans] == [2, 2, 2, 2]
    assert sum(plan.candidate_budget + 1 for plan in padded_plans) == 12


def test_target_tree_forward_discards_graph_padding_outputs_and_mappings():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.device = torch.device("cpu")
    engine.target_vocab_size = 32
    engine.config = SimpleNamespace(
        enforce_eager=False,
        spec_rhythm_tree_width=2,
        spec_rhythm_tree_depth=2,
    )
    engine.cache_allocation = SimpleNamespace(block_tables=[[0], [1], [2]])
    engine.cache_block_tables = torch.zeros((3, 1), dtype=torch.int32)
    engine._ensure_cache_capacity = MagicMock()
    engine.graph_runner = MagicMock()
    engine.graph_runner.run_target_greedy.return_value = (torch.arange(7, 14),)
    engine.graph_runner.last_target_execution = SimpleNamespace(used_aclgraph=True)
    engine.model = MagicMock()
    engine.model.make_tree_attention_metadata.return_value = (
        torch.arange(7),
        torch.arange(7),
        SimpleNamespace(
            slot_mapping=torch.arange(7, dtype=torch.int32),
            use_fused_infer_attention=True,
            tree_attention=True,
            tree_attention_mask=torch.zeros((3, 1, 3, 16), dtype=torch.bool),
            attention_mask=torch.zeros((7, 16), dtype=torch.bool),
        ),
    )
    plans = [
        build_tree_speculation_plan(2, 2, 1, 16, candidate_budget=1),
        build_tree_speculation_plan(2, 2, 2, 16, candidate_budget=1),
        build_tree_speculation_plan(2, 2, 3, 16, candidate_budget=2),
    ]

    result = engine.target_tree_forward(
        plans,
        [1, 2, 3],
        [[4], [5], [0, 0]],
        sequence_ids=[0, 1, 2],
        real_tree_count=2,
    )

    assert result["target_token_ids"].tolist() == [7, 9]
    assert result["target_query_token_ids"].tolist() == [7, 8, 9, 10]
    assert result["bonus_token_ids"].tolist() == [8, 10]
    assert result["num_draft_tokens"] == [1, 1]
    assert result["tree_count"] == 2
    assert result["logical_query_count"] == 4
    assert result["query_count"] == 7
    assert result["graph_padding_query_count"] == 3
    assert result["cache_slot_mapping"].tolist() == [0, 1, 2, 3]
    padded_metadata = engine.graph_runner.run_target_greedy.call_args.args[2][0]
    assert padded_metadata.tree_attention_mask.shape == (3, 1, 5, 16)


def test_tree_state_commits_path_and_records_budget_acceptance():
    state = PearlPipelineState([11], prompt_length=1, max_tokens=8)

    state.apply_tree_verification(
        output_token_ids=[21, 22, -1],
        accepted=2,
        proposed_tokens=3,
    )

    assert state.completion_token_ids == [21, 22]
    assert state.committed_length == 3
    assert state.accepted_draft_tokens == 2
    assert state.verified_draft_tokens == 3
    assert state.acceptance_lengths == [3, 0]
    assert state.pre_verify is True


def test_draft_tree_forward_expands_spine_and_broadcast_shape():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.device = torch.device("cpu")
    engine.draft_vocab_size = 4
    engine.cache_allocation = SimpleNamespace(block_tables=torch.zeros((2, 2), dtype=torch.int32))
    engine.cache_block_tables = torch.zeros((2, 2), dtype=torch.int32)
    engine._ensure_cache_capacity = MagicMock()
    engine.model = MagicMock()
    engine.model.make_tree_level_attention_metadata.side_effect = (
        lambda plan, sequence_id, node_indices, input_token_ids, block_tables: (
            torch.tensor(input_token_ids),
            torch.tensor(node_indices),
            SimpleNamespace(),
        )
    )
    engine.model.make_tree_attention_metadata.return_value = (
        torch.tensor([1, 2, 3, 4, 5]),
        torch.tensor([1, 2, 3, 4, 5]),
        SimpleNamespace(slot_mapping=torch.arange(5, dtype=torch.int32)),
    )
    engine.model.compute_logits.side_effect = [
        torch.tensor([[0.1, 0.9, 0.2, 0.3]]),
        torch.tensor([[0.8, 0.1, 0.2, 0.3]]),
    ]
    plan = build_tree_speculation_plan(
        width=2,
        depth=2,
        prefix_len=1,
        max_model_len=16,
        candidate_budget=3,
    )

    result = engine.draft_tree_forward([plan], [1], [1])

    assert len(result["draft_token_ids"]) == 1
    assert len(result["draft_token_ids"][0]) == 4
    assert result["parent_indices"].numel() == 3
    assert result["num_draft_tokens"] == [3]
    assert engine.model.make_tree_attention_metadata.call_args.kwargs["sequence_ids"] == [1]


def test_draft_tree_forward_fuses_committed_catchup_into_root_call():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.device = torch.device("cpu")
    engine.draft_vocab_size = 4
    engine.config = SimpleNamespace(enforce_eager=True, max_model_len=16)
    engine.cache_allocation = SimpleNamespace(block_tables=torch.zeros((1, 4), dtype=torch.int32))
    engine.cache_block_tables = torch.zeros((1, 4), dtype=torch.int32)
    engine._ensure_cache_capacity = MagicMock()
    engine.model = MagicMock()
    engine.model.make_tree_level_attention_metadata.side_effect = (
        lambda plan, sequence_id, node_indices, input_token_ids, block_tables: (
            torch.tensor(input_token_ids),
            torch.tensor(node_indices),
            SimpleNamespace(),
        )
    )
    engine.model.make_tree_attention_metadata.return_value = (
        torch.tensor([8, 1, 2]),
        torch.tensor([3, 4, 4]),
        SimpleNamespace(slot_mapping=torch.arange(3, dtype=torch.int32)),
    )
    # The first row only restores accepted KV.  Candidate sampling must use
    # the final (current-root) row from the same packed model invocation.
    engine.model.compute_logits.return_value = torch.tensor([[9.0, 0.1, 0.2, 0.3], [0.1, 9.0, 0.2, 0.3]])
    plan = build_tree_speculation_plan(
        width=2,
        depth=1,
        prefix_len=3,
        max_model_len=16,
        candidate_budget=2,
    )

    result = engine.draft_tree_forward(
        [plan],
        [8],
        [0],
        committed_catchup_token_ids=[[7, 8]],
    )

    packed_call = engine.model.make_tree_level_attention_metadata.call_args_list[0]
    catchup_plan = packed_call.args[0]
    assert catchup_plan.width == 1
    assert catchup_plan.depth == 1
    assert catchup_plan.prefix_len == 2
    assert packed_call.args[2] == [-1, 0]
    assert packed_call.args[3] == [7, 8]
    assert result["draft_token_ids"] == [[1, 3]]
    assert result["model_calls"] == 2  # One fused logits call plus final KV write.


def test_linear_tree_reuses_one_full_chain_draft_graph():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.device = torch.device("cpu")
    engine.gamma = 4
    engine.config = SimpleNamespace(max_num_seqs=8, max_model_len=32, enforce_eager=True)
    engine.cache_allocation = SimpleNamespace(block_tables=torch.zeros((2, 2), dtype=torch.int32))
    engine.cache_block_tables = torch.zeros((2, 2), dtype=torch.int32)
    engine._ensure_cache_capacity = MagicMock()
    engine.graph_runner = MagicMock()
    engine.graph_runner.last_draft_execution = SimpleNamespace(used_aclgraph=True)
    engine._pad_tree_draft_graph_requests = MagicMock(side_effect=lambda requests: (requests, 0))
    engine._pack_tree_draft_level = MagicMock(
        return_value=(torch.tensor([5, 6, 7, 8, 9]), torch.arange(5), SimpleNamespace())
    )
    engine.model = MagicMock(return_value=torch.zeros((5, 4)))
    proposals = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]])
    confidence = torch.tensor([0.8, 0.6])
    engine._draft_spec_rhythm_device_batch = MagicMock(return_value=(proposals.flatten(), proposals, confidence))
    plans = [build_tree_speculation_plan(1, 4, prefix_len, 32, candidate_budget=4) for prefix_len in (3, 5)]

    result = engine.draft_tree_forward(
        plans,
        [7, 9],
        [0, 1],
        committed_catchup_token_ids=[[5, 6, 7], [8, 9]],
    )

    engine._draft_spec_rhythm_device_batch.assert_called_once()
    assert engine._draft_spec_rhythm_device_batch.call_args.kwargs["prepare_final_kv"] is True
    assert result["draft_token_ids"] == proposals.tolist()
    assert result["num_draft_tokens"] == [4, 4]
    assert result["model_calls"] == 1
    assert result["graph_calls"] == 1
    assert result["linear_full_chain"] is True
    assert result["linear_final_kv_prepared"] is True
    assert result["materialization_deferred"] is True
    assert result["query_count"] == 10
    assert result["graph_padding_query_count"] == 0
    engine._pad_tree_draft_graph_requests.assert_not_called()
    assert engine._spec_rhythm_linear_adopted_kv_rounds == 1
    assert engine._spec_rhythm_linear_adopted_kv_queries == 3
    assert [row.tolist() for row in result["node_confidences"]] == [
        pytest.approx([0.8] * 4),
        pytest.approx([0.6] * 4),
    ]


def test_linear_eager_tree_batches_frontier_then_reuses_full_chain_graph():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.device = torch.device("cpu")
    engine.gamma = 4
    engine.draft_vocab_size = 4
    engine.config = SimpleNamespace(max_num_seqs=8, enforce_eager=True)
    engine.cache_allocation = SimpleNamespace(block_tables=torch.zeros((1, 8), dtype=torch.int32))
    engine.cache_block_tables = torch.zeros((1, 8), dtype=torch.int32)
    engine._ensure_cache_capacity = MagicMock()
    engine._pad_tree_draft_graph_requests = MagicMock(side_effect=lambda requests: (requests, 0))
    engine._pack_tree_draft_level = MagicMock(return_value=(torch.tensor([14]), torch.tensor([7]), SimpleNamespace()))
    engine.model = MagicMock(return_value=torch.zeros((1, 4)))
    engine.model.compute_logits.return_value = torch.tensor([[0.1, 0.2, 0.3, 9.0]])
    engine.graph_runner = MagicMock()
    engine.graph_runner.last_draft_execution = SimpleNamespace(used_aclgraph=True)
    engine._cache_slot_mapping = MagicMock(side_effect=lambda _ids, positions: list(range(len(positions))))
    proposals = torch.tensor([[21, 22, 23, 24]])
    engine._draft_spec_rhythm_device_batch = MagicMock(
        return_value=(proposals.flatten(), proposals, torch.tensor([0.75]))
    )
    parent = build_tree_speculation_plan(1, 4, 3, 32, candidate_budget=4)
    eager = build_tree_speculation_plan(1, 4, 8, 32, candidate_budget=4)

    result = engine.draft_tree_forward(
        [eager],
        [99],
        [0],
        eager_parent_sources={0: (parent, [11, 12, 13, 14])},
    )

    engine._draft_spec_rhythm_device_batch.assert_called_once()
    assert engine._draft_spec_rhythm_device_batch.call_args.kwargs["prepare_final_kv"] is True
    assert result["eager_frontier_tokens"] == {0: 3}
    assert result["root_token_ids"] == [3]
    assert result["draft_token_ids"] == proposals.tolist()
    assert result["model_calls"] == 2
    assert result["linear_full_chain"] is True
    assert result["cache_slot_mapping"].numel() == 5


def test_draft_tree_eager_writes_frontier_root_into_tree_kv():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.device = torch.device("cpu")
    engine.draft_vocab_size = 4
    engine.cache_allocation = SimpleNamespace(block_tables=torch.zeros((1, 4), dtype=torch.int32))
    engine.cache_block_tables = torch.zeros((1, 4), dtype=torch.int32)
    engine._ensure_cache_capacity = MagicMock()
    engine.model = MagicMock()
    engine.model.make_tree_level_attention_metadata.side_effect = (
        lambda plan, sequence_id, node_indices, input_token_ids, block_tables: (
            torch.tensor(input_token_ids),
            torch.tensor(node_indices),
            SimpleNamespace(),
        )
    )
    engine.model.make_tree_attention_metadata.return_value = (
        torch.tensor([1, 2, 3, 4, 5]),
        torch.tensor([1, 2, 3, 4, 5]),
        SimpleNamespace(slot_mapping=torch.arange(5, dtype=torch.int32)),
    )
    # Frontier query, root query, and the second spine query.  The final
    # masked write does not request logits.
    engine.model.compute_logits.side_effect = [
        torch.tensor([[0.1, 0.2, 0.3, 9.0]]),
        torch.tensor([[0.1, 9.0, 0.2, 0.3]]),
        torch.tensor([[0.1, 0.2, 9.0, 0.3]]),
    ]
    parent_plan = build_tree_speculation_plan(
        width=2,
        depth=2,
        prefix_len=1,
        max_model_len=16,
        candidate_budget=3,
    )
    eager_plan = build_tree_speculation_plan(
        width=2,
        depth=2,
        prefix_len=5,
        max_model_len=16,
        candidate_budget=3,
    )

    result = engine.draft_tree_forward(
        [eager_plan],
        [99],
        [0],
        eager_parent_sources={0: (parent_plan, [10, 11, 12, 13])},
    )

    assert result["root_token_ids"] == [3]
    final_call = engine.model.make_tree_attention_metadata.call_args
    assert final_call.args[1] == [3]


def test_tree_metadata_separates_logical_rope_and_physical_kv_positions():
    model = NativeQwen2ForCausalLM.__new__(NativeQwen2ForCausalLM)
    model.embed_tokens = SimpleNamespace(weight=torch.empty(1))
    model.layers = [SimpleNamespace(self_attn=SimpleNamespace(block_size=4))]
    model.max_model_len = 16
    plan = build_tree_speculation_plan(
        width=2,
        depth=2,
        prefix_len=1,
        max_model_len=16,
        candidate_budget=4,
    )
    _, positions, metadata = model.make_tree_attention_metadata(
        [plan], [7], [[10, 11, 20, 21]], [[0, 0, 0, 0]], sequence_ids=[0]
    )
    assert positions.tolist() == plan.positions.tolist()
    assert metadata.context_lens.tolist() == [2, 3, 4, 5, 6]
    assert metadata.slot_mapping.tolist() == [1, 2, 3, 0, 1]

    _, level_positions, level_metadata = model.make_tree_level_attention_metadata(
        plan, 0, [0, 2], [10, 20], [[0, 0, 0, 0]]
    )
    assert level_positions.tolist() == [2, 2]
    assert level_metadata.context_lens.tolist() == [3, 5]
    assert level_metadata.slot_mapping.tolist() == [2, 0]


def test_greedy_graph_forwards_the_speculative_fia_backend():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.device = torch.device("cpu")
    engine._run_device_packed_greedy = MagicMock(return_value=torch.tensor([4]))

    engine._run_packed_greedy(
        [3],
        [0],
        [2],
        use_aclgraph=True,
        use_fused_infer_attention=True,
    )

    assert engine._run_device_packed_greedy.call_args.kwargs == {
        "use_aclgraph": True,
        "use_fused_infer_attention": True,
    }


def test_draft_round_uses_speculative_fia_aclgraph():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.gamma = 2
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(enforce_eager=False)
    engine._run_device_packed_greedy = MagicMock(
        side_effect=[torch.tensor([4]), torch.tensor([5])],
    )
    state = PearlPipelineState([1, 2, 3], prompt_length=2)

    verification, continuation = engine._draft_round_batch([state], [0])

    assert verification == [[4]]
    assert continuation == [[4, 5]]
    assert all(call.kwargs["use_aclgraph"] is True for call in engine._run_device_packed_greedy.call_args_list)
    assert all(
        call.kwargs["use_fused_infer_attention"] is True for call in engine._run_device_packed_greedy.call_args_list
    )


def test_draft_round_uses_one_gamma_step_aclgraph_when_available():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.gamma = 2
    engine.draft_vocab_size = 32
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(enforce_eager=False)
    engine.graph_runner = MagicMock()
    engine.graph_runner.run_draft_greedy.return_value = torch.tensor([[4, 5]])
    metadata = SimpleNamespace(use_fused_infer_attention=True)
    engine._prepare_attention_metadata = MagicMock(
        side_effect=[
            (torch.tensor([2]), metadata),
            (torch.tensor([3]), metadata),
        ]
    )
    state = PearlPipelineState([1, 2, 3], prompt_length=2)

    verification, continuation = engine._draft_round_batch([state], [0])

    assert verification == [[4]]
    assert continuation == [[4, 5]]
    assert engine._prepare_attention_metadata.call_args_list[0].args == ([0], [2])
    assert engine._prepare_attention_metadata.call_args_list[1].args == ([0], [3])
    engine.graph_runner.run_draft_greedy.assert_called_once()


def test_draft_round_can_select_paged_attention_aclgraph():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.gamma = 2
    engine.draft_vocab_size = 32
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(enforce_eager=False, draft_use_paged_attention=True)
    engine.graph_runner = MagicMock()
    engine.graph_runner.run_draft_greedy.return_value = torch.tensor([[4, 5]])
    metadata = SimpleNamespace(use_fused_infer_attention=False)
    engine._prepare_attention_metadata = MagicMock(
        side_effect=[
            (torch.tensor([2]), metadata),
            (torch.tensor([3]), metadata),
        ]
    )
    state = PearlPipelineState([1, 2, 3], prompt_length=2)

    verification, continuation = engine._draft_round_batch([state], [0])

    assert verification == [[4]]
    assert continuation == [[4, 5]]
    assert all(
        call.kwargs["use_fused_infer_attention"] is False for call in engine._prepare_attention_metadata.call_args_list
    )
    engine.graph_runner.run_draft_greedy.assert_called_once()


def test_draft_aclgraph_captures_paged_attention_steps():
    model = MagicMock()
    runner = NativeACLGraphRunner(model, enabled=False)
    runner.enabled = True
    runner.expected_fia_batch_size = 2
    runner._capture_draft = MagicMock(return_value=torch.tensor([[4, 5], [6, 7]]))
    metadata = SimpleNamespace(use_fused_infer_attention=False)

    output = runner.run_draft_greedy(
        torch.tensor([2, 3]),
        [torch.tensor([1, 1]), torch.tensor([2, 2])],
        [metadata, metadata],
        vocabulary_size=32,
    )

    assert output.tolist() == [[4, 5], [6, 7]]
    assert runner._capture_draft.call_args.args[0] == (
        "draft-greedy:32|steps:2|paged",
        2,
    )
    assert runner._capture_draft.call_args.kwargs["valid_row_count"] == 2


def test_draft_aclgraph_eager_chain_feeds_each_token_to_the_next_step():
    model = MagicMock()
    model.side_effect = [torch.tensor([[10.0]]), torch.tensor([[20.0]])]
    model.compute_greedy_tokens.side_effect = [torch.tensor([4]), torch.tensor([5])]
    runner = NativeACLGraphRunner(model, enabled=False)

    output = runner.run_draft_greedy(
        torch.tensor([3]),
        [torch.tensor([2]), torch.tensor([3])],
        [
            SimpleNamespace(use_fused_infer_attention=True),
            SimpleNamespace(use_fused_infer_attention=True),
        ],
        vocabulary_size=32,
    )

    assert output.tolist() == [[4, 5]]
    assert runner.last_draft_execution.mode == "eager"
    assert runner.last_draft_execution.fallback_reason == "disabled"
    assert torch.equal(model.call_args_list[0].args[0], torch.tensor([3]))
    assert torch.equal(model.call_args_list[1].args[0], torch.tensor([4]))


def test_draft_aclgraph_kv_only_tail_skips_unused_lm_head():
    model = MagicMock()
    model.side_effect = [
        torch.tensor([[10.0]]),
        torch.tensor([[20.0]]),
        torch.tensor([[30.0]]),
    ]
    model.compute_greedy_tokens.side_effect = [
        torch.tensor([4]),
        torch.tensor([5]),
    ]
    runner = NativeACLGraphRunner(model, enabled=False)

    output = runner.run_draft_greedy(
        torch.tensor([3]),
        [torch.tensor([2]), torch.tensor([3]), torch.tensor([4])],
        [
            SimpleNamespace(use_fused_infer_attention=True),
            SimpleNamespace(use_fused_infer_attention=True),
            SimpleNamespace(use_fused_infer_attention=True),
        ],
        vocabulary_size=32,
        final_kv_only=True,
    )

    assert output.tolist() == [[4, 5]]
    assert model.call_count == 3
    assert model.compute_greedy_tokens.call_count == 2
    assert torch.equal(model.call_args_list[0].args[0], torch.tensor([3]))
    assert torch.equal(model.call_args_list[1].args[0], torch.tensor([4]))
    assert torch.equal(model.call_args_list[2].args[0], torch.tensor([5]))


def test_draft_aclgraph_kv_only_tail_uses_separate_graph_entry():
    model = MagicMock()
    runner = NativeACLGraphRunner(model, enabled=False)
    runner.enabled = True
    runner.expected_fia_batch_size = 1
    runner._capture_draft = MagicMock(return_value=torch.tensor([[4, 5]]))
    metadata = SimpleNamespace(use_fused_infer_attention=False)

    output = runner.run_draft_greedy(
        torch.tensor([3]),
        [torch.tensor([2]), torch.tensor([3]), torch.tensor([4])],
        [metadata, metadata, metadata],
        vocabulary_size=32,
        final_kv_only=True,
    )

    assert output.tolist() == [[4, 5]]
    assert runner._capture_draft.call_args.args[0] == (
        "draft-greedy:32|steps:3|paged|final-kv",
        1,
    )
    assert runner._capture_draft.call_args.kwargs["final_kv_only"] is True


def test_target_aclgraph_eager_path_keeps_fixed_verification_inputs():
    model = MagicMock()
    model.side_effect = [torch.tensor([[10.0]]), torch.tensor([[20.0]])]
    model.compute_greedy_tokens.side_effect = [torch.tensor([4]), torch.tensor([5])]
    runner = NativeACLGraphRunner(model, enabled=False)
    metadata = SimpleNamespace(use_fused_infer_attention=False)

    outputs = runner.run_target_greedy(
        [torch.tensor([3]), torch.tensor([9])],
        [torch.tensor([2]), torch.tensor([3])],
        [metadata, metadata],
        vocabulary_size=32,
    )

    assert [output.tolist() for output in outputs] == [[4], [5]]
    assert torch.equal(model.call_args_list[0].args[0], torch.tensor([3]))
    assert torch.equal(model.call_args_list[1].args[0], torch.tensor([9]))


def test_target_aclgraph_requires_matching_step_metadata():
    runner = NativeACLGraphRunner(MagicMock(), enabled=False)
    metadata = SimpleNamespace(use_fused_infer_attention=False)

    with pytest.raises(ValueError, match="matching non-empty"):
        runner.run_target_greedy(
            [torch.tensor([3])],
            [],
            [metadata],
            vocabulary_size=32,
        )


def test_draft_aclgraph_uses_eager_for_a_dynamic_tail_batch():
    model = MagicMock()
    runner = NativeACLGraphRunner(model, enabled=False)
    runner.enabled = True
    runner.expected_fia_batch_size = 4
    runner._execute_draft = MagicMock(return_value=torch.tensor([[4, 5], [6, 7]]))
    metadata = SimpleNamespace(use_fused_infer_attention=True)

    output = runner.run_draft_greedy(
        torch.tensor([2, 3]),
        [torch.tensor([1, 1]), torch.tensor([2, 2])],
        [metadata, metadata],
        vocabulary_size=32,
    )

    assert output.tolist() == [[4, 5], [6, 7]]
    assert runner.shape_fallback_count == 1
    assert runner.last_draft_execution.mode == "eager"
    assert runner.last_draft_execution.fallback_reason == "shape"
    runner._execute_draft.assert_called_once()


def test_draft_round_snapshots_a_reused_aclgraph_output_buffer():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.gamma = 2
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(enforce_eager=False)
    shared_output = torch.tensor([0])

    def replay(*_args, **_kwargs):
        shared_output.add_(1)
        return shared_output

    engine._run_device_packed_greedy = MagicMock(side_effect=replay)
    state = PearlPipelineState([1, 2, 3], prompt_length=2)

    _, continuation = engine._draft_round_batch([state], [0])

    assert continuation == [[1, 2]]


def test_draft_round_executes_only_rows_with_remaining_variable_budget():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.gamma = 4
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(enforce_eager=True, draft_use_paged_attention=False)
    engine._run_device_packed_greedy = MagicMock(
        side_effect=[
            torch.tensor([10, 20]),
            torch.tensor([11]),
            torch.tensor([12]),
        ]
    )
    states = [
        PearlPipelineState([1, 2], prompt_length=1),
        PearlPipelineState([3, 4], prompt_length=1),
    ]

    verification, continuation = engine._draft_round_device_batch(states, [0, 1], draft_budgets=[3, 1])

    assert verification.tolist() == [10, 20]
    assert continuation.tolist() == [[10, 11, 12, -1], [20, -1, -1, -1]]
    assert [call.args[1] for call in engine._run_device_packed_greedy.call_args_list] == [
        [0, 1],
        [0],
        [0],
    ]


def test_spec_rhythm_eager_verification_keeps_the_parent_window_prefix():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.gamma = 4
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(
        enforce_eager=True,
        draft_use_paged_attention=False,
    )
    engine._run_device_packed_greedy_with_confidence = MagicMock(
        side_effect=[
            (torch.tensor([10, 20]), torch.tensor([0.8, 0.7])),
            (torch.tensor([11, 21]), torch.tensor([0.6, 0.5])),
        ]
    )
    states = [
        PearlPipelineState([1, 2], prompt_length=1),
        PearlPipelineState([3, 4, 30, 31, 32, 33], prompt_length=1),
    ]

    verification, continuation, confidence = engine._draft_spec_rhythm_device_batch(
        states,
        [0, 1],
        [2, 2],
        verification_sizes=[1, 4],
        verification_prefixes=[None, torch.tensor([31, 32, 33])],
    )

    assert verification.tolist() == [10, 31, 32, 33, 20]
    assert continuation.tolist() == [[10, 11, -1, -1], [20, 21, -1, -1]]
    assert confidence.tolist() == pytest.approx([0.7, 0.6])
    assert engine._linear_draft_stepwise_calls == 1
    assert engine._linear_draft_stepwise_model_calls == 2
    assert getattr(engine, "_linear_draft_full_chain_bucket_calls", {}) == {}


def test_spec_rhythm_full_window_draft_payload_is_the_complete_next_window():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.gamma = 4
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(
        enforce_eager=True,
        draft_use_paged_attention=False,
        enable_spec_rhythm=True,
        spec_rhythm_stable_graphs=False,
        max_num_seqs=8,
    )
    engine._run_device_packed_greedy_with_confidence = MagicMock(
        side_effect=[
            (torch.tensor([10, 20]), torch.tensor([0.9, 0.8])),
            (torch.tensor([11, 21]), torch.tensor([0.7, 0.6])),
            (torch.tensor([12, 22]), torch.tensor([0.5, 0.4])),
            (torch.tensor([13, 23]), torch.tensor([0.3, 0.2])),
        ]
    )
    states = [
        PearlPipelineState([1, 2], prompt_length=1),
        PearlPipelineState([3, 4], prompt_length=1),
    ]

    verification, continuation, confidence = engine._draft_spec_rhythm_device_batch(
        states,
        [0, 1],
        [engine.gamma, engine.gamma],
        verification_sizes=[engine.gamma, engine.gamma],
        full_window=True,
    )

    expected = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    assert torch.equal(continuation, expected)
    assert torch.equal(verification.reshape(2, engine.gamma), expected)
    assert confidence.tolist() == pytest.approx([0.6, 0.5])
    step_inputs = [call.args[0] for call in engine._run_device_packed_greedy_with_confidence.call_args_list]
    assert len(step_inputs) == engine.gamma
    assert all(
        torch.equal(actual, expected_input)
        for actual, expected_input in zip(
            step_inputs,
            [
                torch.tensor([2, 4]),
                torch.tensor([10, 20]),
                torch.tensor([11, 21]),
                torch.tensor([12, 22]),
            ],
        )
    )


def test_linear_draft_metadata_only_padding_matches_generic_graph_padding():
    input_ids = torch.tensor([3, 4], dtype=torch.long)
    positions = torch.tensor([9, 10], dtype=torch.long)
    metadata = NativeAttentionMetadata(
        slot_mapping=torch.tensor([17, 18], dtype=torch.int32),
        context_lens=torch.tensor([10, 11], dtype=torch.int32),
        block_tables=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        actual_seq_lengths_q=(1, 2),
        sequence_lens=(10, 11),
    )

    _, expected_positions, expected_metadata = NativeACLGraphRunner._pad_inputs(
        input_ids,
        positions,
        metadata,
        capture_size=4,
    )
    actual_positions, actual_metadata = native_engine_module._pad_linear_draft_graph_metadata(
        positions,
        metadata,
        capture_size=4,
    )

    assert torch.equal(actual_positions, expected_positions)
    assert torch.equal(actual_metadata.slot_mapping, expected_metadata.slot_mapping)
    assert torch.equal(actual_metadata.context_lens, expected_metadata.context_lens)
    assert torch.equal(actual_metadata.block_tables, expected_metadata.block_tables)
    assert actual_metadata.actual_seq_lengths_q == expected_metadata.actual_seq_lengths_q
    assert actual_metadata.sequence_lens == expected_metadata.sequence_lens


def test_bucketed_linear_draft_fia_masks_exact_prefix_and_pads_safe_rows():
    positions = torch.tensor([9, 32], dtype=torch.long)
    request_tables = torch.tensor(
        [[5, -1, -1], [7, 8, 9]],
        dtype=torch.int32,
    )
    metadata = NativeAttentionMetadata(
        slot_mapping=torch.tensor([17, 44], dtype=torch.int32),
        context_lens=torch.tensor([10, 33], dtype=torch.int32),
        block_tables=request_tables,
        actual_seq_lengths_q=(1, 2),
        sequence_lens=(10, 33),
        request_block_tables=request_tables,
        attention_mask=torch.zeros((1, 1), dtype=torch.int8),
        use_fused_infer_attention=True,
    )

    padded_positions, padded = native_engine_module._pad_bucketed_linear_draft_fia_graph_metadata(
        positions,
        metadata,
        4,
        block_size=16,
        host_request_block_tables=[[5, -1, -1], [7, 8, 9]],
        kv_length_limits=[20, 48],
    )

    assert torch.equal(padded_positions, torch.tensor([9, 32, 0, 0]))
    assert torch.equal(
        padded.slot_mapping,
        torch.tensor([17, 44, -1, -1], dtype=torch.int32),
    )
    assert padded.actual_seq_lengths_q == (1, 2, 3, 4)
    assert padded.sequence_lens == (20, 48, 1, 1)
    assert padded.context_lens.tolist() == [20, 48, 1, 1]
    assert padded.tree_attention
    assert padded.attention_mask is None
    assert padded.tree_attention_mask.shape == (4, 1, 1, 48)
    assert not padded.tree_attention_mask[0, 0, 0, :10].any()
    assert padded.tree_attention_mask[0, 0, 0, 10:].all()
    assert not padded.tree_attention_mask[1, 0, 0, :33].any()
    assert padded.tree_attention_mask[1, 0, 0, 33:].all()
    assert not padded.tree_attention_mask[2, 0, 0, 0]
    assert padded.tree_attention_mask[2, 0, 0, 1:].all()
    assert padded.request_block_tables.tolist() == [
        [5, 5, 5],
        [7, 8, 9],
        [5, 5, 5],
        [5, 5, 5],
    ]


def test_bucketed_linear_draft_fia_accepts_batched_masks_and_shared_table():
    request_tables = torch.tensor(
        [[5, -1, -1], [7, 8, 9]],
        dtype=torch.int32,
    )
    shared_table = native_engine_module.sanitize_and_pad_linear_draft_fia_request_tables(
        request_tables,
        capture_size=4,
    )
    masks = native_engine_module.LinearDraftFIAFullMaskBuilder().build(
        [9, 32],
        step_count=4,
        capture_size=4,
        capacity=48,
        device="cpu",
    )

    padded_metadatas = []
    for step in range(4):
        positions = torch.tensor([9 + step, 32 + step], dtype=torch.long)
        metadata = NativeAttentionMetadata(
            slot_mapping=torch.tensor([17 + step, 44 + step], dtype=torch.int32),
            context_lens=(positions + 1).to(torch.int32),
            block_tables=request_tables,
            actual_seq_lengths_q=(1, 2),
            sequence_lens=tuple((positions + 1).tolist()),
            request_block_tables=request_tables,
            attention_mask=torch.zeros((1, 1), dtype=torch.int8),
            use_fused_infer_attention=True,
        )
        _, padded = native_engine_module._pad_bucketed_linear_draft_fia_graph_metadata(
            positions,
            metadata,
            4,
            block_size=16,
            exact_positions=positions.tolist(),
            host_request_block_tables=[[5, -1, -1], [7, 8, 9]],
            kv_length_limits=[20, 48],
            precomputed_full_mask=masks[step],
            shared_request_block_tables=shared_table,
        )
        padded_metadatas.append(padded)

    assert all(metadata.request_block_tables is shared_table for metadata in padded_metadatas)
    assert all(metadata.tree_attention_mask is masks[step] for step, metadata in enumerate(padded_metadatas))


def test_persistent_linear_draft_fia_envelope_reuses_lane_local_metadata():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.device = torch.device("cpu")
    engine._linear_draft_fia_persistent_staging_pool = {}

    kwargs = {
        "capture_size": 4,
        "step_count": 4,
        "block_size": 4,
        "table_width": 3,
        "sequence_lens": (12, 12, 12, 12),
        "input_dtype": torch.long,
        "table_dtype": torch.int32,
    }
    lane_zero = engine._get_linear_draft_fia_persistent_envelope(
        graph_lane=0,
        **kwargs,
    )
    same_lane = engine._get_linear_draft_fia_persistent_envelope(
        graph_lane=0,
        **kwargs,
    )
    lane_one = engine._get_linear_draft_fia_persistent_envelope(
        graph_lane=1,
        **kwargs,
    )

    assert same_lane is lane_zero
    assert lane_one is not lane_zero
    assert lane_one.staging.full_masks.data_ptr() != lane_zero.staging.full_masks.data_ptr()
    assert all(
        metadata.request_block_tables is lane_zero.staging.request_block_tables
        for metadata in lane_zero.attention_metadatas
    )
    assert all(
        metadata.tree_attention_mask is lane_zero.staging.full_mask_views[step]
        for step, metadata in enumerate(lane_zero.attention_metadatas)
    )
    assert engine._linear_draft_fia_persistent_staging_hits == 1
    assert engine._linear_draft_fia_persistent_staging_misses == 2


def test_linear_draft_fia_common_kv_uses_service_wide_lifetime_limit():
    states = [
        PearlPipelineState([1, 2], prompt_length=2, max_tokens=32),
        # This inactive request must still determine the stable service-wide
        # envelope so later admission cannot change a graph lane's literals.
        PearlPipelineState([3] * 65, prompt_length=65, max_tokens=128),
        PearlPipelineState([4] * 17, prompt_length=17, max_tokens=64),
    ]

    assert _linear_draft_fia_lifetime_limits(
        states,
        [2, 0],
        4,
        common_kv=False,
    ) == [85, 38]
    assert _linear_draft_fia_lifetime_limits(
        states,
        [2, 0],
        4,
        common_kv=True,
    ) == [197, 197]


def test_linear_draft_fia_common_kv_is_independently_opt_in(monkeypatch):
    variable = "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_COMMON_KV"
    monkeypatch.delenv(variable, raising=False)
    assert native_engine_module.envs.VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_COMMON_KV is False
    monkeypatch.setenv(variable, "1")
    assert native_engine_module.envs.VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_COMMON_KV is True


def test_linear_draft_fia_stable_task_barrier_is_independently_opt_in(
    monkeypatch,
):
    variable = "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_STABLE_TASK_BARRIER"
    monkeypatch.delenv(variable, raising=False)
    assert native_engine_module.envs.VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_STABLE_TASK_BARRIER is False
    monkeypatch.setenv(variable, "1")
    assert native_engine_module.envs.VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_STABLE_TASK_BARRIER is True


def test_linear_draft_fia_common_kv_keeps_30_to_31_row_bucket_signature():
    capture_size = 32
    common_limit = 48

    def padded_metadata(num_tokens: int) -> NativeAttentionMetadata:
        request_tables = torch.tensor(
            [[100 + row, -1, -1, -1] for row in range(num_tokens)],
            dtype=torch.int32,
        )
        positions = torch.full((num_tokens,), 9, dtype=torch.long)
        metadata = NativeAttentionMetadata(
            slot_mapping=torch.arange(num_tokens, dtype=torch.int32),
            context_lens=torch.full((num_tokens,), 10, dtype=torch.int32),
            block_tables=request_tables,
            actual_seq_lengths_q=tuple(range(1, num_tokens + 1)),
            sequence_lens=(10,) * num_tokens,
            request_block_tables=request_tables,
            attention_mask=torch.zeros((1, 1), dtype=torch.int8),
            use_fused_infer_attention=True,
        )
        _, padded = native_engine_module._pad_bucketed_linear_draft_fia_graph_metadata(
            positions,
            metadata,
            capture_size,
            block_size=16,
            exact_positions=positions.tolist(),
            host_request_block_tables=request_tables.tolist(),
            kv_length_limits=[common_limit] * num_tokens,
            dummy_kv_length_limits=[common_limit] * (capture_size - num_tokens),
        )
        return padded

    rows_30 = padded_metadata(30)
    rows_31 = padded_metadata(31)
    expected_signature = (common_limit,) * capture_size
    assert rows_30.sequence_lens == expected_signature
    assert rows_31.sequence_lens == expected_signature
    assert rows_30.context_lens.tolist() == list(expected_signature)
    assert rows_31.context_lens.tolist() == list(expected_signature)
    for padded, first_dummy in ((rows_30, 30), (rows_31, 31)):
        assert (padded.slot_mapping[first_dummy:] < 0).all()
        for row in range(first_dummy, capture_size):
            dummy_mask = padded.tree_attention_mask[row, 0, 0]
            assert not dummy_mask[0]
            assert dummy_mask[1:].all()


def test_ranked_linear_draft_fia_capacity_slots_dominate_any_subset():
    states = [
        PearlPipelineState(
            [row + 1] * (row + 1),
            prompt_length=row + 1,
            max_tokens=8 + (row * 7) % 23,
        )
        for row in range(9)
    ]
    gamma = 4
    capture_size = 8
    expected_capacities = tuple(
        sorted(
            (state.prompt_length + state.max_tokens + gamma for state in states),
            reverse=True,
        )[:capture_size]
    )

    for subset_size in range(1, 6):
        for subset in combinations(range(len(states)), subset_size):
            plan = _ranked_linear_draft_fia_plan(
                states,
                subset,
                gamma,
                capture_size,
            )
            ranked_active_limits = [
                states[subset[caller_row]].prompt_length + states[subset[caller_row]].max_tokens + gamma
                for caller_row in plan.graph_to_caller_rows
            ]
            assert plan.capacity_limits == expected_capacities
            assert all(
                active_limit <= capacity_limit
                for active_limit, capacity_limit in zip(
                    ranked_active_limits,
                    plan.capacity_limits,
                )
            )


def test_ranked_linear_draft_fia_keeps_30_to_31_row_bucket_signature():
    capture_size = 32
    gamma = 4
    states = [
        PearlPipelineState(
            [row + 1, row + 101],
            prompt_length=2,
            max_tokens=8 + row,
        )
        for row in range(40)
    ]

    def pad_subset(active_indices: list[int]) -> NativeAttentionMetadata:
        plan = _ranked_linear_draft_fia_plan(
            states,
            active_indices,
            gamma,
            capture_size,
        )
        graph_indices = [active_indices[row] for row in plan.graph_to_caller_rows]
        num_tokens = len(graph_indices)
        request_tables = torch.tensor(
            [[100 + index, -1, -1, -1] for index in graph_indices],
            dtype=torch.int32,
        )
        positions = torch.ones(num_tokens, dtype=torch.long)
        metadata = NativeAttentionMetadata(
            slot_mapping=torch.arange(num_tokens, dtype=torch.int32),
            context_lens=torch.full((num_tokens,), 2, dtype=torch.int32),
            block_tables=request_tables,
            actual_seq_lengths_q=tuple(range(1, num_tokens + 1)),
            sequence_lens=(2,) * num_tokens,
            request_block_tables=request_tables,
            attention_mask=torch.zeros((1, 1), dtype=torch.int8),
            use_fused_infer_attention=True,
        )
        _, padded = native_engine_module._pad_bucketed_linear_draft_fia_graph_metadata(
            positions,
            metadata,
            capture_size,
            block_size=16,
            exact_positions=positions.tolist(),
            host_request_block_tables=request_tables.tolist(),
            kv_length_limits=plan.capacity_limits[:num_tokens],
            dummy_kv_length_limits=plan.capacity_limits[num_tokens:],
        )
        return padded

    rows_30 = pad_subset(list(range(0, 40, 1))[:30])
    rows_31 = pad_subset(list(range(39, -1, -1))[:31])
    assert rows_30.sequence_lens == rows_31.sequence_lens
    assert rows_30.context_lens.tolist() == rows_31.context_lens.tolist()
    for padded, first_dummy in ((rows_30, 30), (rows_31, 31)):
        assert (padded.slot_mapping[first_dummy:] < 0).all()
        for row in range(first_dummy, capture_size):
            dummy_mask = padded.tree_attention_mask[row, 0, 0]
            assert not dummy_mask[0]
            assert dummy_mask[1:].all()


def test_ranked_linear_draft_fia_restores_graph_rows_to_caller_order():
    states = [
        PearlPipelineState([1], prompt_length=1, max_tokens=10),
        PearlPipelineState([2], prompt_length=1, max_tokens=30),
        PearlPipelineState([3], prompt_length=1, max_tokens=20),
    ]
    plan = _ranked_linear_draft_fia_plan(states, [0, 1, 2], 4, 4)
    assert plan.graph_to_caller_rows == (1, 2, 0)
    assert plan.caller_to_graph_rows == (2, 0, 1)

    graph_windows = torch.tensor(
        [[100, 101], [200, 201], [300, 301]],
    )
    graph_confidence = torch.tensor([0.9, 0.8, 0.7])
    assert _restore_ranked_linear_draft_rows(
        graph_windows,
        plan.caller_to_graph_rows,
    ).tolist() == [[300, 301], [100, 101], [200, 201]]
    assert _restore_ranked_linear_draft_rows(
        graph_confidence,
        plan.caller_to_graph_rows,
    ).tolist() == pytest.approx([0.7, 0.9, 0.8])


def test_ranked_linear_draft_fia_rejects_capacity_above_page_table():
    states = [
        PearlPipelineState([1, 2], prompt_length=2, max_tokens=80),
    ]
    plan = _ranked_linear_draft_fia_plan(states, [0], 4, 1)
    request_tables = torch.tensor([[7, 8, 9, 10]], dtype=torch.int32)
    metadata = NativeAttentionMetadata(
        slot_mapping=torch.tensor([1], dtype=torch.int32),
        context_lens=torch.tensor([2], dtype=torch.int32),
        block_tables=request_tables,
        actual_seq_lengths_q=(1,),
        sequence_lens=(2,),
        request_block_tables=request_tables,
        attention_mask=torch.zeros((1, 1), dtype=torch.int8),
        use_fused_infer_attention=True,
    )

    with pytest.raises(ValueError, match="exceeding cache capacity"):
        native_engine_module._pad_bucketed_linear_draft_fia_graph_metadata(
            torch.tensor([1]),
            metadata,
            capture_size=1,
            block_size=16,
            exact_positions=[1],
            host_request_block_tables=request_tables.tolist(),
            kv_length_limits=plan.capacity_limits,
            dummy_kv_length_limits=(),
        )


def test_ranked_linear_draft_fia_full_window_restores_all_public_rows():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.gamma = 4
    engine.device = torch.device("cpu")
    engine.draft_vocab_size = 1000
    engine.config = SimpleNamespace(
        enforce_eager=False,
        draft_use_paged_attention=True,
        enable_spec_rhythm=True,
        spec_rhythm_stable_graphs=True,
        spec_rhythm_linear_bonus_token=False,
        max_num_seqs=4,
    )
    engine.model = SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(block_size=16))])
    block_tables = [[20 + row, -1, -1, -1] for row in range(3)]
    engine.cache_allocation = SimpleNamespace(block_tables=block_tables)
    states = [
        PearlPipelineState([1, 10], prompt_length=2, max_tokens=10),
        PearlPipelineState([2, 20], prompt_length=2, max_tokens=30),
        PearlPipelineState([3, 30], prompt_length=2, max_tokens=20),
    ]

    def prepare(sequence_ids, positions, *, use_fused_infer_attention):
        assert sequence_ids == [1, 2, 0]
        assert use_fused_infer_attention
        tables = torch.tensor(
            [block_tables[index] for index in sequence_ids],
            dtype=torch.int32,
        )
        lengths = tuple(position + 1 for position in positions)
        return (
            torch.tensor(positions, dtype=torch.long),
            NativeAttentionMetadata(
                slot_mapping=torch.arange(3, dtype=torch.int32),
                context_lens=torch.tensor(lengths, dtype=torch.int32),
                block_tables=tables,
                actual_seq_lengths_q=(1, 2, 3),
                sequence_lens=lengths,
                request_block_tables=tables,
                attention_mask=torch.zeros((1, 1), dtype=torch.int8),
                use_fused_infer_attention=True,
            ),
        )

    engine._prepare_attention_metadata = MagicMock(side_effect=prepare)
    graph_rows = torch.tensor(
        [
            [100, 101, 102, 103],
            [200, 201, 202, 203],
            [300, 301, 302, 303],
            [-1, -1, -1, -1],
        ],
    )
    engine.graph_runner = SimpleNamespace(
        run_draft_greedy=MagicMock(return_value=graph_rows),
        last_draft_execution=None,
    )

    with patch.dict(
        os.environ,
        {
            "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_BUCKET": "1",
            "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_COMMON_KV": "0",
            "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_RANKED_KV": "1",
        },
    ):
        verification, next_windows, confidence = engine._draft_spec_rhythm_device_batch(
            states,
            [0, 1, 2],
            [4, 4, 4],
            verification_sizes=[4, 4, 4],
            full_window=True,
            graph_lane=0,
            prepare_final_kv=True,
        )

    expected = torch.tensor(
        [
            [300, 301, 302, 303],
            [100, 101, 102, 103],
            [200, 201, 202, 203],
        ]
    )
    assert torch.equal(next_windows, expected)
    assert torch.equal(verification, expected.flatten())
    assert confidence.tolist() == [1.0, 1.0, 1.0]
    graph_input = engine.graph_runner.run_draft_greedy.call_args.args[0]
    assert graph_input.tolist() == [20, 30, 10, 0]
    graph_call = engine.graph_runner.run_draft_greedy.call_args
    assert len(graph_call.args[1]) == 5
    assert len(graph_call.args[2]) == 5
    assert graph_call.kwargs["final_kv_only"] is True
    for metadata in graph_call.args[2]:
        assert metadata.sequence_lens == (36, 26, 16, 1)
        dummy_mask = metadata.tree_attention_mask[3, 0, 0]
        assert not dummy_mask[0]
        assert dummy_mask[1:].all()


def test_ranked_and_common_linear_draft_fia_are_mutually_exclusive():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.gamma = 4
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(
        draft_use_paged_attention=True,
        enable_spec_rhythm=True,
        spec_rhythm_stable_graphs=True,
    )
    state = PearlPipelineState([1, 2], prompt_length=1)
    with (
        patch.dict(
            os.environ,
            {
                "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_BUCKET": "1",
                "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_COMMON_KV": "1",
                "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_RANKED_KV": "1",
            },
        ),
        pytest.raises(ValueError, match="mutually exclusive"),
    ):
        engine._draft_spec_rhythm_device_batch(
            [state],
            [0],
            [4],
            verification_sizes=[4],
            full_window=True,
        )


def test_bucketed_linear_draft_fia_rejects_undercovered_kv_limit():
    metadata = NativeAttentionMetadata(
        slot_mapping=torch.tensor([17], dtype=torch.int32),
        context_lens=torch.tensor([10], dtype=torch.int32),
        block_tables=torch.tensor([[5]], dtype=torch.int32),
        actual_seq_lengths_q=(1,),
        sequence_lens=(10,),
        request_block_tables=torch.tensor([[5]], dtype=torch.int32),
        attention_mask=torch.zeros((1, 1), dtype=torch.int8),
        use_fused_infer_attention=True,
    )

    with pytest.raises(ValueError, match="must cover every exact request length"):
        native_engine_module._pad_bucketed_linear_draft_fia_graph_metadata(
            torch.tensor([9]),
            metadata,
            1,
            block_size=16,
            host_request_block_tables=[[5]],
            kv_length_limits=[9],
        )


def test_bucketed_linear_draft_fia_covers_context_bucket_boundaries():
    block_size = 128
    capacity = 4096
    contexts = (1, 127, 128, 129, 2047, 2048, 2049, 4095)
    table_width = capacity // block_size
    request_tables = []
    for request_index, context_length in enumerate(contexts):
        visible_pages = (context_length + block_size - 1) // block_size
        first_page = 100 + request_index * table_width
        request_tables.append(
            list(range(first_page, first_page + visible_pages)) + [-1] * (table_width - visible_pages)
        )
    request_tables_tensor = torch.tensor(request_tables, dtype=torch.int32)
    positions = torch.tensor(
        [context_length - 1 for context_length in contexts],
        dtype=torch.long,
    )
    metadata = NativeAttentionMetadata(
        slot_mapping=torch.arange(len(contexts), dtype=torch.int32),
        context_lens=torch.tensor(contexts, dtype=torch.int32),
        block_tables=request_tables_tensor,
        actual_seq_lengths_q=tuple(range(1, len(contexts) + 1)),
        sequence_lens=contexts,
        request_block_tables=request_tables_tensor,
        attention_mask=torch.zeros((1, 1), dtype=torch.int8),
        use_fused_infer_attention=True,
    )

    padded_positions, padded = native_engine_module._pad_bucketed_linear_draft_fia_graph_metadata(
        positions,
        metadata,
        capture_size=12,
        block_size=block_size,
        exact_positions=positions.tolist(),
        host_request_block_tables=request_tables,
    )

    expected_buckets = (1, 128, 128, 256, 2048, 2048, 4096, 4096)
    assert padded.sequence_lens == (*expected_buckets, 1, 1, 1, 1)
    assert padded.context_lens.tolist() == list(padded.sequence_lens)
    assert torch.equal(padded_positions[: len(contexts)], positions)
    assert not padded_positions[len(contexts) :].any()
    assert padded.tree_attention_mask.shape == (12, 1, 1, capacity)
    for request_index, context_length in enumerate(contexts):
        request_mask = padded.tree_attention_mask[request_index, 0, 0]
        assert not request_mask[:context_length].any()
        assert request_mask[context_length:].all()


@pytest.mark.parametrize(
    ("context_length", "host_table"),
    [
        (129, [11, -1, 13, 14]),
        (1, [-1, -1, -1, -1]),
    ],
    ids=("visible-page-hole", "all-pages-unallocated"),
)
def test_bucketed_linear_draft_fia_rejects_unallocated_visible_pages(
    context_length,
    host_table,
):
    request_tables = torch.tensor([host_table], dtype=torch.int32)
    metadata = NativeAttentionMetadata(
        slot_mapping=torch.tensor([17], dtype=torch.int32),
        context_lens=torch.tensor([context_length], dtype=torch.int32),
        block_tables=request_tables,
        actual_seq_lengths_q=(1,),
        sequence_lens=(context_length,),
        request_block_tables=request_tables,
        attention_mask=torch.zeros((1, 1), dtype=torch.int8),
        use_fused_infer_attention=True,
    )

    with pytest.raises(RuntimeError, match="unallocated page"):
        native_engine_module._pad_bucketed_linear_draft_fia_graph_metadata(
            torch.tensor([context_length - 1]),
            metadata,
            capture_size=1,
            block_size=128,
            exact_positions=[context_length - 1],
            host_request_block_tables=[host_table],
        )


def test_bucketed_linear_draft_fia_sanitizes_each_tail_and_dummy_row_safely():
    request_tables = torch.tensor(
        [
            [17, -1, -1, -1],
            [29, 30, -1, -1],
        ],
        dtype=torch.int32,
    )
    metadata = NativeAttentionMetadata(
        slot_mapping=torch.tensor([71, 72], dtype=torch.int32),
        context_lens=torch.tensor([1, 129], dtype=torch.int32),
        block_tables=request_tables,
        actual_seq_lengths_q=(1, 2),
        sequence_lens=(1, 129),
        request_block_tables=request_tables,
        attention_mask=torch.zeros((1, 1), dtype=torch.int8),
        use_fused_infer_attention=True,
    )

    _, padded = native_engine_module._pad_bucketed_linear_draft_fia_graph_metadata(
        torch.tensor([0, 128]),
        metadata,
        capture_size=4,
        block_size=128,
        exact_positions=[0, 128],
        host_request_block_tables=[
            [17, -1, -1, -1],
            [29, 30, -1, -1],
        ],
    )

    assert padded.request_block_tables.tolist() == [
        [17, 17, 17, 17],
        [29, 30, 29, 29],
        [17, 17, 17, 17],
        [17, 17, 17, 17],
    ]
    assert (padded.slot_mapping[2:] < 0).all()
    for dummy_index in (2, 3):
        dummy_mask = padded.tree_attention_mask[dummy_index, 0, 0]
        assert not dummy_mask[0]
        assert dummy_mask[1:].all()


@pytest.mark.parametrize(
    ("work_rows", "expected_graph_rows"),
    [
        (1, 1),
        (3, 4),
        (8, 8),
        (16, 16),
        (17, 20),
        (25, 28),
        (32, 32),
        (33, 36),
        (37, 40),
        (41, 44),
        (45, 48),
        (48, 48),
        (64, 64),
    ],
)
def test_spec_rhythm_full_draft_graph_uses_bounded_dense_buckets(
    work_rows,
    expected_graph_rows,
):
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.gamma = 4
    engine.draft_vocab_size = 128
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(
        enforce_eager=False,
        draft_use_paged_attention=False,
        enable_spec_rhythm=True,
        spec_rhythm_stable_graphs=True,
        max_num_seqs=64,
    )
    engine.graph_runner = MagicMock()
    engine.graph_runner.last_draft_execution = NativeGraphExecution(
        "replay",
        replay_executed=True,
    )
    engine.graph_runner.run_draft_greedy.return_value = torch.arange(
        expected_graph_rows * engine.gamma,
        dtype=torch.long,
    ).reshape(expected_graph_rows, engine.gamma)
    engine._run_device_packed_greedy_with_confidence = MagicMock()
    engine.rank = 0
    engine.last_worker_decode_phase_seconds = {}
    engine.last_worker_decode_profile_seconds = {}
    engine.last_worker_decode_profile_detail_seconds = {}
    engine.last_worker_decode_counters = {}
    engine.last_worker_decode_host_timeline = []
    engine.last_worker_profiled_decode_steps = 0

    def metadata(indices, positions, *, use_fused_infer_attention):
        assert len(indices) == work_rows
        assert not use_fused_infer_attention
        return torch.tensor(positions, dtype=torch.long), SimpleNamespace(
            slot_mapping=torch.arange(work_rows, dtype=torch.long),
            context_lens=torch.ones(work_rows, dtype=torch.int32),
            block_tables=torch.zeros((work_rows, 1), dtype=torch.int32),
        )

    engine._prepare_attention_metadata = MagicMock(side_effect=metadata)
    states = [PearlPipelineState([1, 2], prompt_length=1) for _ in range(work_rows)]

    with (
        patch.object(
            NativeACLGraphRunner,
            "_pad_inputs",
            wraps=NativeACLGraphRunner._pad_inputs,
        ) as pad_inputs,
        patch.object(
            native_engine_module.torch,
            "ones",
            wraps=torch.ones,
        ) as ones,
        patch.object(
            native_engine_module.torch,
            "full",
            wraps=torch.full,
        ) as full,
    ):
        verification, continuation, confidence = engine._draft_spec_rhythm_device_batch(
            states,
            list(range(work_rows)),
            [engine.gamma] * work_rows,
            verification_sizes=[1] * work_rows,
        )

    graph_call = engine.graph_runner.run_draft_greedy.call_args
    assert pad_inputs.call_count == 1
    confidence_call = call(
        work_rows,
        dtype=torch.float32,
        device=engine.device,
    )
    assert ones.call_args_list.count(confidence_call) == 1
    assert (
        call(
            (work_rows, engine.gamma),
            -1,
            dtype=torch.long,
            device=engine.device,
        )
        not in full.call_args_list
    )
    assert graph_call.args[0].shape == (expected_graph_rows,)
    assert all(value.shape == (expected_graph_rows,) for value in graph_call.args[1])
    assert all(value.slot_mapping.shape == (expected_graph_rows,) for value in graph_call.args[2])
    assert graph_call.kwargs["valid_row_count"] == work_rows
    assert graph_call.kwargs["return_confidence"] is True
    assert continuation.shape == (work_rows, engine.gamma)
    assert verification.shape == (work_rows,)
    assert confidence.tolist() == [1.0] * work_rows
    engine._run_device_packed_greedy_with_confidence.assert_not_called()
    assert engine._linear_draft_full_chain_bucket_calls == {expected_graph_rows: 1}
    assert engine._linear_draft_full_chain_calls == 1
    assert getattr(engine, "_linear_draft_full_chain_capture_replay_calls", 0) == 0
    assert engine._linear_draft_full_chain_replay_calls == 1
    assert getattr(engine, "_linear_draft_full_chain_eager_fallback_calls", 0) == 0
    assert getattr(engine, "_linear_draft_full_chain_unclassified_calls", 0) == 0
    assert engine._linear_draft_full_chain_logical_rows == work_rows
    assert engine._linear_draft_full_chain_logical_tokens == work_rows * engine.gamma
    assert engine._linear_draft_full_chain_padded_rows == expected_graph_rows
    assert engine._linear_draft_full_chain_padded_tokens == expected_graph_rows * engine.gamma
    assert engine._linear_draft_full_chain_padding_rows == expected_graph_rows - work_rows
    assert engine._linear_draft_full_chain_padding_tokens == (expected_graph_rows - work_rows) * engine.gamma
    assert getattr(engine, "_linear_draft_stepwise_calls", 0) == 0
    engine._fixed_full_window_host_correction_staging_eligible_calls = 7
    engine._fixed_full_window_host_correction_staging_calls = 6
    metrics = engine.graph_metrics()
    assert metrics[f"spec_rhythm_linear_draft_full_chain_bucket_{expected_graph_rows}_calls"] == 1
    assert metrics["spec_rhythm_linear_draft_full_chain_calls"] == 1
    assert metrics["spec_rhythm_linear_draft_full_chain_replay_calls"] == 1
    assert metrics["spec_rhythm_linear_draft_full_chain_logical_rows"] == work_rows
    assert metrics["spec_rhythm_linear_draft_full_chain_padded_rows"] == expected_graph_rows
    assert metrics["spec_rhythm_linear_draft_stepwise_calls"] == 0
    assert metrics["spec_rhythm_fixed_full_window_host_correction_staging_eligible_calls"] == 7
    assert metrics["spec_rhythm_fixed_full_window_host_correction_staging_calls"] == 6
    assert {
        "mc2_dispatch_fused_attempt",
        "mc2_dispatch_fused_success",
        "mc2_dispatch_fallback",
        "mc2_dispatch_exception",
    } <= metrics.keys()


def test_linear_acceptance_only_full_chain_omits_softmax_confidence(monkeypatch):
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.gamma = 4
    engine.draft_vocab_size = 128
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(
        enforce_eager=False,
        draft_use_paged_attention=False,
        enable_spec_rhythm=True,
        spec_rhythm_stable_graphs=True,
        max_num_seqs=8,
    )
    engine.graph_runner = MagicMock()
    engine.graph_runner.last_draft_execution = NativeGraphExecution(
        "replay",
        replay_executed=True,
    )
    engine.graph_runner.run_draft_greedy.return_value = torch.tensor(
        [[11, 12, 13, 14]],
        dtype=torch.long,
    )
    engine._prepare_attention_metadata = MagicMock(
        side_effect=lambda indices, positions, *, use_fused_infer_attention: (
            torch.tensor(positions, dtype=torch.long),
            SimpleNamespace(
                slot_mapping=torch.arange(len(indices), dtype=torch.long),
                context_lens=torch.ones(len(indices), dtype=torch.int32),
                block_tables=torch.zeros((len(indices), 1), dtype=torch.int32),
            ),
        )
    )
    monkeypatch.setattr(
        native_engine_module.envs,
        "VLLM_ASCEND_SPECRHYTHM_LINEAR_ACCEPTANCE_ONLY",
        True,
    )

    verification, continuation, confidence = engine._draft_spec_rhythm_device_batch(
        [PearlPipelineState([1, 2], prompt_length=1)],
        [0],
        [4],
        verification_sizes=[4],
        full_window=True,
    )

    graph_call = engine.graph_runner.run_draft_greedy.call_args
    assert graph_call.kwargs["return_confidence"] is False
    assert verification.tolist() == [11, 12, 13, 14]
    assert continuation.tolist() == [[11, 12, 13, 14]]
    assert confidence.tolist() == [1.0]


def test_linear_spec_rhythm_capture_prewarms_paged_home_and_full_draft_graphs():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.gamma = 4
    engine.draft_vocab_size = 128
    engine.device = torch.device("cpu")
    engine.precompiled_decode_batch_sizes = frozenset()
    engine.config = SimpleNamespace(
        enforce_eager=False,
        target_use_paged_attention=False,
        draft_use_paged_attention=False,
        enable_spec_rhythm=True,
        spec_rhythm_tree_width=1,
        spec_rhythm_tree_depth=1,
        spec_rhythm_stable_graphs=True,
        max_num_seqs=64,
    )
    engine.graph_runner = MagicMock()
    engine.graph_runner.run_draft_greedy.side_effect = (
        lambda input_ids, positions, metadatas, vocabulary_size, *, valid_row_count: torch.zeros(
            (input_ids.shape[0], len(positions)),
            dtype=torch.long,
        )
    )
    metadata_modes = []

    def metadata(indices, positions, *, use_fused_infer_attention):
        metadata_modes.append(use_fused_infer_attention)
        rows = len(indices)
        return torch.tensor(positions, dtype=torch.long), SimpleNamespace(
            slot_mapping=torch.arange(rows, dtype=torch.long),
            context_lens=torch.ones(rows, dtype=torch.int32),
            block_tables=torch.zeros((rows, 1), dtype=torch.int32),
        )

    engine._prepare_attention_metadata = MagicMock(side_effect=metadata)
    prompts = [[1]] * 40

    with patch("vllm_ascend.spec_decode.pearl.native_engine.dist.barrier"):
        engine._capture_decode_graphs(prompts, [2] * len(prompts))

    graph_calls = engine.graph_runner.run_draft_greedy.call_args_list
    assert [call.args[0].shape[0] for call in graph_calls] == [32, 64]
    assert [call.kwargs["valid_row_count"] for call in graph_calls] == [32, 40]
    assert all(mode is False for mode in metadata_modes)
    assert graph_calls[1].args[2][0].slot_mapping[40:].tolist() == [-1] * 24


def test_ordinary_pearl_capture_keeps_its_single_home_fia_graph():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.gamma = 4
    engine.draft_vocab_size = 128
    engine.device = torch.device("cpu")
    engine.precompiled_decode_batch_sizes = frozenset()
    engine.config = SimpleNamespace(
        enforce_eager=False,
        target_use_paged_attention=False,
        draft_use_paged_attention=False,
        enable_spec_rhythm=False,
        spec_rhythm_tree_width=1,
        spec_rhythm_tree_depth=1,
        spec_rhythm_stable_graphs=True,
        max_num_seqs=64,
    )
    engine.graph_runner = MagicMock()
    metadata_modes = []

    def metadata(indices, positions, *, use_fused_infer_attention):
        metadata_modes.append(use_fused_infer_attention)
        return torch.tensor(positions), SimpleNamespace(
            use_fused_infer_attention=use_fused_infer_attention,
        )

    engine._prepare_attention_metadata = MagicMock(side_effect=metadata)

    with patch("vllm_ascend.spec_decode.pearl.native_engine.dist.barrier"):
        engine._capture_decode_graphs([[1]] * 40, [2] * 40)

    graph_call = engine.graph_runner.run_draft_greedy.call_args
    assert graph_call.args[0].shape == (32,)
    assert graph_call.kwargs["valid_row_count"] == 32
    assert metadata_modes == [True] * engine.gamma


@pytest.mark.parametrize(
    ("disable_target_graph", "expected_calls"),
    [("1", 0), ("0", 2)],
)
def test_static_target_graph_precapture_respects_diagnostic_disable(
    monkeypatch,
    disable_target_graph,
    expected_calls,
):
    monkeypatch.setenv(
        "VLLM_ASCEND_SPECRHYTHM_DISABLE_TARGET_ACLGRAPH",
        disable_target_graph,
    )
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.gamma = 4
    engine.device = torch.device("cpu")
    engine.precompiled_decode_batch_sizes = frozenset()
    engine.config = SimpleNamespace(
        enforce_eager=False,
        target_use_paged_attention=False,
    )
    engine._run_device_packed_greedy = MagicMock()

    with patch("vllm_ascend.spec_decode.pearl.native_engine.dist.barrier"):
        engine._capture_decode_graphs([[1], [2]], [3, 4])

    assert engine._run_device_packed_greedy.call_count == expected_calls


def test_greedy_verdict_supports_mixed_per_request_gamma():
    target = torch.tensor([1, 2, 9, 4, 5, 6])
    draft = torch.tensor([1, 2, 3, 4, 0, 0])
    verdict = _build_greedy_verdict(target, draft, verification_sizes=[3, 1, 2], gamma=5)
    assert verdict.tolist() == [[2, 9], [1, -1], [0, 5]]


def test_completion_is_truncated_at_the_first_eos_or_token_limit():
    assert _truncate_completion([10, 11, 99, 12], frozenset((99,)), 4) == [10, 11, 99]
    assert _truncate_completion([10, 11, 12], frozenset(), 2) == [10, 11]
    assert _truncate_completion([10, 99, 12], frozenset((99,)), 3, ignore_eos=True) == [10, 99, 12]


def test_sampling_params_are_per_request_and_reject_mixed_temperature_modes():
    params = SamplingParams(temperature=0.7, max_tokens=12, ignore_eos=True)

    assert _normalize_sampling_params(2, params, 64) == [params, params]
    with pytest.raises(ValueError, match="all zero or all non-zero"):
        _normalize_sampling_params(
            2,
            [SamplingParams(temperature=0), SamplingParams(temperature=1)],
            64,
        )
    with pytest.raises(ValueError, match="draft temperatures"):
        _normalize_sampling_params(
            2,
            [SamplingParams(temperature=0, draft_temperature=0), SamplingParams(temperature=0, draft_temperature=1)],
            64,
        )

    state = PearlPipelineState(
        [1, 99],
        prompt_length=1,
        committed_length=2,
        max_tokens=2,
        ignore_eos=True,
    )
    assert not _finished(state, frozenset((99,)))
    state.token_ids.append(2)
    state.committed_length = 3
    assert _finished(state, frozenset((99,)))


def test_target_sampler_supports_greedy_and_exponential_race_sampling():
    logits = torch.tensor([[1.0, 3.0, 2.0], [2.0, 1.0, 3.0]])

    assert _sample_logits(logits, [0.0, 0.0]).tolist() == [1, 2]
    assert _sample_logits(
        logits,
        [1.0, 1.0],
        exponential_noise=torch.ones_like(logits),
    ).tolist() == [1, 2]


def test_target_sampler_applies_per_row_top_k_and_top_p_before_sampling():
    logits = torch.tensor([[5.0, 4.0, 3.0], [5.0, 4.0, 3.0]])
    # Noise would make token 2 win without filtering. Both filters retain
    # only token 0 for their respective row.
    noise = torch.tensor([[10.0, 10.0, 0.001], [10.0, 10.0, 0.001]])
    tokens = _sample_logits(
        logits,
        [1.0, 1.0],
        top_ps=[1.0, 0.5],
        top_ks=[1, 0],
        exponential_noise=noise,
    )
    assert tokens.tolist() == [0, 0]


def test_stochastic_verdict_accepts_prefix_and_samples_masked_correction():
    target_logits = torch.tensor([[8.0, 1.0, 0.0]] * 5)
    draft_tokens = torch.zeros(5, dtype=torch.long)

    verdict = _build_stochastic_verdict(
        target_logits,
        draft_tokens,
        verification_sizes=[1, 4],
        gamma=4,
        temperatures=[1.0] * 5,
        random_values=torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
        exponential_noise=torch.ones_like(target_logits),
    )

    assert verdict.tolist() == [[1, -1], [2, 1]]


def test_pipeline_state_reports_upstream_mat_segments():
    state = PearlPipelineState([1, 2, 3], prompt_length=2)
    assert state.acceptance_lengths == []
    state.apply_target_verification(
        gamma=4,
        accepted=1,
        correction_token_id=None,
        next_round_token_ids=[4, 5, 6, 7],
    )
    state.apply_target_verification(
        gamma=4,
        accepted=2,
        correction_token_id=42,
        next_round_token_ids=[8, 9, 10, 11],
    )

    assert state.acceptance_lengths == [4, 0]
    assert sum(state.acceptance_lengths) / len(state.acceptance_lengths) == 2.0


def test_auto_gamma_uses_the_upstream_draft_to_target_speed_ratio():
    assert _gamma_from_decode_speeds(700.0, 100.0) == 7
    assert _gamma_from_decode_speeds(50.0, 100.0) == 1
    assert _gamma_from_decode_speeds(10_000.0, 100.0) == 100

    config = NativePearlConfig("draft", "target", 1, 2, -1, 512, 32)
    assert config.gamma == -1


def test_direct_worker_cli_exposes_cache_capacity_controls():
    args = _build_parser().parse_args(
        [
            "--draft-model",
            "draft",
            "--target-model",
            "target",
            "--prompt",
            "hello",
            "--gpu-memory-utilization",
            "0.98",
            "--num-kvcache-blocks",
            "32",
            "--max-aclgraph-entries",
            "8",
            "--spec-rhythm-online-prefill",
            "--spec-rhythm-prefill-coalesce-min-requests",
            "2",
            "--spec-rhythm-prefill-coalesce-max-wait-ms",
            "600",
            "--spec-rhythm-prefill-token-chunk-size",
            "128",
            "--spec-rhythm-merge-ready-homes",
            "--spec-rhythm-slo-home-partition",
            "--spec-rhythm-priority-mode",
            "--spec-rhythm-priority-burst",
            "3",
            "--spec-rhythm-target-fallback-max-batch",
            "2",
            "--spec-rhythm-cpu-verdict",
        ]
    )

    assert args.gpu_memory_utilization == 0.98
    assert args.num_kvcache_blocks == 32
    assert args.max_aclgraph_entries == 8
    assert args.target_verification_graph_buckets == 8
    assert args.disable_cpu_binding is False
    assert args.draft_use_production_rope is True
    assert args.target_use_production_rope is True
    assert args.spec_rhythm_min_gamma == 1
    assert args.spec_rhythm_max_eager_tokens == 0
    assert args.spec_rhythm_eager_reserve_tokens == 0
    assert args.spec_rhythm_tree_width == 1
    assert args.spec_rhythm_tree_depth == 1
    assert args.spec_rhythm_online_prefill is True
    assert args.spec_rhythm_prefill_coalesce_min_requests == 2
    assert args.spec_rhythm_prefill_coalesce_max_wait_ms == 600.0
    assert args.spec_rhythm_prefill_token_chunk_size == 128
    assert args.spec_rhythm_merge_ready_homes is True
    assert args.spec_rhythm_slo_home_partition is True
    assert args.spec_rhythm_priority_mode is True
    assert args.spec_rhythm_priority_burst == 3
    assert args.spec_rhythm_target_fallback_max_batch == 2
    assert args.spec_rhythm_cpu_verdict is True


def test_benchmark_cli_defaults_production_rope_with_independent_fallbacks():
    required = [
        "--draft-model",
        "draft",
        "--target-model",
        "target",
        "--prompt",
        "hello",
    ]
    defaults = _build_benchmark_parser().parse_args(required)
    fallback = _build_benchmark_parser().parse_args(
        [
            *required,
            "--no-draft-use-production-rope",
            "--no-target-use-production-rope",
        ]
    )

    assert defaults.draft_use_production_rope is True
    assert defaults.target_use_production_rope is True
    assert fallback.draft_use_production_rope is False
    assert fallback.target_use_production_rope is False


def test_linear_full_window_cli_is_opt_in_across_public_entrypoints():
    benchmark_required = [
        "--draft-model",
        "draft",
        "--target-model",
        "target",
        "--prompt",
        "hello",
    ]
    benchmark_default = _build_benchmark_parser().parse_args(benchmark_required)
    benchmark_enabled = _build_benchmark_parser().parse_args(
        [
            *benchmark_required,
            "--spec-rhythm-linear-full-window",
            "--spec-rhythm-linear-eager-cross-graph-bucket",
        ]
    )
    server_required = [
        "--draft-model",
        "draft",
        "--target-model",
        "target",
        "--spec-rhythm-roofline",
        "roofline.json",
    ]
    server_default = _build_specslo_server_parser().parse_args(server_required)
    server_enabled = _build_specslo_server_parser().parse_args([*server_required, "--spec-rhythm-linear-full-window"])
    with patch(
        "sys.argv",
        [
            "offline_inference_nano_pearl.py",
            "--draft-model",
            "draft",
            "--target-model",
            "target",
            "hello",
        ],
    ):
        offline_default = _parse_offline_args()
    with patch(
        "sys.argv",
        [
            "offline_inference_nano_pearl.py",
            "--draft-model",
            "draft",
            "--target-model",
            "target",
            "--spec-rhythm-linear-full-window",
            "hello",
        ],
    ):
        offline_enabled = _parse_offline_args()

    assert benchmark_default.spec_rhythm_linear_full_window is False
    assert benchmark_enabled.spec_rhythm_linear_full_window is True
    assert benchmark_default.spec_rhythm_linear_eager_cross_graph_bucket is False
    assert benchmark_enabled.spec_rhythm_linear_eager_cross_graph_bucket is True
    assert server_default.spec_rhythm_linear_full_window is False
    assert server_enabled.spec_rhythm_linear_full_window is True
    assert offline_default.spec_rhythm_linear_full_window is False
    assert offline_enabled.spec_rhythm_linear_full_window is True


def test_prefill_coalescing_cli_defaults_and_values_across_online_entrypoints():
    benchmark_required = [
        "--draft-model",
        "draft",
        "--target-model",
        "target",
        "--prompt",
        "hello",
    ]
    benchmark_default = _build_benchmark_parser().parse_args(benchmark_required)
    benchmark_enabled = _build_benchmark_parser().parse_args(
        [
            *benchmark_required,
            "--spec-rhythm-prefill-coalesce-min-requests",
            "2",
            "--spec-rhythm-prefill-coalesce-max-wait-ms",
            "600",
            "--spec-rhythm-prefill-token-chunk-size",
            "128",
        ]
    )
    server_required = [
        "--draft-model",
        "draft",
        "--target-model",
        "target",
        "--spec-rhythm-roofline",
        "roofline.json",
    ]
    server_default = _build_specslo_server_parser().parse_args(server_required)
    server_enabled = _build_specslo_server_parser().parse_args(
        [
            *server_required,
            "--prefill-coalesce-min-requests",
            "3",
            "--prefill-coalesce-max-wait-ms",
            "450",
        ]
    )
    native_required = [
        "--draft-model",
        "draft",
        "--target-model",
        "target",
        "--prompt",
        "hello",
    ]
    native_default = _build_parser().parse_args(native_required)
    native_enabled = _build_parser().parse_args(
        [
            *native_required,
            "--spec-rhythm-prefill-coalesce-min-requests",
            "4",
            "--spec-rhythm-prefill-coalesce-max-wait-ms",
            "700",
            "--spec-rhythm-prefill-token-chunk-size",
            "96",
        ]
    )

    assert benchmark_default.spec_rhythm_prefill_coalesce_min_requests == 1
    assert benchmark_default.spec_rhythm_prefill_coalesce_max_wait_ms == 0.0
    assert benchmark_enabled.spec_rhythm_prefill_coalesce_min_requests == 2
    assert benchmark_enabled.spec_rhythm_prefill_coalesce_max_wait_ms == 600.0
    assert benchmark_default.spec_rhythm_prefill_token_chunk_size == 0
    assert benchmark_enabled.spec_rhythm_prefill_token_chunk_size == 128
    assert server_default.prefill_coalesce_min_requests == 1
    assert server_default.prefill_coalesce_max_wait_ms == 0.0
    assert server_enabled.prefill_coalesce_min_requests == 3
    assert server_enabled.prefill_coalesce_max_wait_ms == 450.0
    assert native_default.spec_rhythm_prefill_coalesce_min_requests == 1
    assert native_default.spec_rhythm_prefill_coalesce_max_wait_ms == 0.0
    assert native_enabled.spec_rhythm_prefill_coalesce_min_requests == 4
    assert native_enabled.spec_rhythm_prefill_coalesce_max_wait_ms == 700.0
    assert native_default.spec_rhythm_prefill_token_chunk_size == 0
    assert native_enabled.spec_rhythm_prefill_token_chunk_size == 96


def test_serial_draft_graph_precompile_cli_is_opt_in_across_public_entrypoints():
    benchmark_required = [
        "--draft-model",
        "draft",
        "--target-model",
        "target",
        "--prompt",
        "hello",
    ]
    benchmark_default = _build_benchmark_parser().parse_args(benchmark_required)
    benchmark_enabled = _build_benchmark_parser().parse_args([*benchmark_required, "--precompile-serial-draft-graphs"])
    server_required = [
        "--draft-model",
        "draft",
        "--target-model",
        "target",
        "--spec-rhythm-roofline",
        "roofline.json",
    ]
    server_default = _build_specslo_server_parser().parse_args(server_required)
    server_enabled = _build_specslo_server_parser().parse_args([*server_required, "--precompile-serial-draft-graphs"])
    native_required = [
        "--draft-model",
        "draft",
        "--target-model",
        "target",
        "--prompt",
        "hello",
    ]
    native_default = _build_parser().parse_args(native_required)
    native_enabled = _build_parser().parse_args([*native_required, "--precompile-serial-draft-graphs"])
    with patch(
        "sys.argv",
        [
            "offline_inference_nano_pearl.py",
            "--draft-model",
            "draft",
            "--target-model",
            "target",
            "hello",
        ],
    ):
        offline_default = _parse_offline_args()
    with patch(
        "sys.argv",
        [
            "offline_inference_nano_pearl.py",
            "--draft-model",
            "draft",
            "--target-model",
            "target",
            "--precompile-serial-draft-graphs",
            "hello",
        ],
    ):
        offline_enabled = _parse_offline_args()

    for args in (
        benchmark_default,
        server_default,
        native_default,
        offline_default,
    ):
        assert args.precompile_serial_draft_graphs is False
    for args in (
        benchmark_enabled,
        server_enabled,
        native_enabled,
        offline_enabled,
    ):
        assert args.precompile_serial_draft_graphs is True


def test_benchmark_clis_accept_disjoint_warmup_prompt_offsets():
    speculative = _build_benchmark_parser().parse_args(
        [
            "--draft-model",
            "draft",
            "--target-model",
            "target",
            "--prompt",
            "hello",
            "--warmup-prompt-offset",
            "200",
        ]
    )
    target_only = _build_target_only_benchmark_parser().parse_args(
        [
            "--model",
            "target",
            "--prompt",
            "hello",
            "--warmup-prompt-offset",
            "200",
        ]
    )

    assert speculative.warmup_prompt_offset == 200
    assert target_only.warmup_prompt_offset == 200


def test_speculative_benchmark_accepts_ablation_mode_and_respect_eos():
    args = _build_benchmark_parser().parse_args(
        [
            "--draft-model",
            "draft",
            "--target-model",
            "target",
            "--prompt",
            "hello",
            "--respect-eos",
            "--spec-rhythm-ablation-mode",
            "dual_batch_rolling",
        ]
    )
    assert args.respect_eos is True
    assert args.spec_rhythm_ablation_mode == "dual_batch_rolling"


def test_speculative_benchmark_accepts_saturated_manifest_arrivals():
    args = _build_benchmark_parser().parse_args(
        [
            "--draft-model",
            "draft",
            "--target-model",
            "target",
            "--request-manifest",
            "workload.jsonl",
            "--saturated-arrivals",
        ]
    )

    assert args.saturated_arrivals is True


def test_target_only_benchmark_accepts_shared_prefill_coalesce_policy():
    args = _build_target_only_benchmark_parser().parse_args(
        [
            "--model",
            "target",
            "--prompt",
            "hello",
            "--online-arrivals",
            "--prefill-coalesce-min-requests",
            "4",
            "--prefill-coalesce-max-wait-ms",
            "1100",
        ]
    )

    assert args.prefill_coalesce_min_requests == 4
    assert args.prefill_coalesce_max_wait_ms == 1100.0


def test_target_only_benchmark_accepts_shared_mixed_graph_envelope():
    args = _build_target_only_benchmark_parser().parse_args(
        [
            "--model",
            "target",
            "--prompt",
            "hello",
            "--max-num-batched-tokens",
            "640",
            "--max-cudagraph-capture-size",
            "640",
            "--cudagraph-capture-sizes",
            "1,2,4,8,16,24,32,40,48,56,64,128,192,256,320,384,448,512,640",
            "--cudagraph-copy-inputs",
        ]
    )

    assert args.max_num_batched_tokens == 640
    assert args.max_cudagraph_capture_size == 640
    assert args.cudagraph_capture_sizes[-3:] == [512 - 64, 512, 640]
    assert args.cudagraph_copy_inputs is True


def test_public_pearl_config_maps_upstream_fields_to_native_runtime():
    model_config = SimpleNamespace(
        architectures=["Qwen2ForCausalLM"],
        eos_token_id=[1, 2],
    )
    with patch(
        "vllm_ascend.spec_decode.pearl.api.AutoConfig.from_pretrained",
        side_effect=[model_config, model_config],
    ):
        config = PEARLConfig(
            "draft",
            "target",
            draft_tensor_parallel_size=1,
            target_tensor_parallel_size=2,
            max_model_len=512,
            max_num_batched_tokens=1024,
            max_num_seqs=8,
            draft_dtype="bfloat16",
            target_dtype="float16",
            prefill_chunk_size=4,
            max_num_queued_seqs=12,
            gamma=4,
            enable_continuous_batching=True,
            enable_preemptive_scheduling=True,
            enable_spec_rhythm=True,
            spec_rhythm_slo_home_partition=True,
            spec_rhythm_min_gamma=2,
            spec_rhythm_max_eager_tokens=3,
            spec_rhythm_eager_reserve_tokens=12,
            spec_rhythm_roofline={"8:1": 24},
            spec_rhythm_draft_token_budget=32,
            spec_rhythm_tree_width=2,
            spec_rhythm_tree_depth=2,
            pad_finished_requests=True,
            draft_use_paged_attention=True,
            target_use_paged_attention=True,
            draft_use_production_rope=True,
            target_use_production_rope=True,
            precompile_decode_graphs=True,
            target_verification_graph_post_counts=((8, (0, 4, 8)),),
            enable_cpu_binding=False,
            profile_decode_steps=5,
            profile_host_decode_steps=7,
        )

    native = config.to_native()
    assert config.world_size == 3
    assert config.eos == [1, 2]
    assert config.draft_config.model == "draft"
    assert config.draft_config.devices == [0]
    assert config.target_config.model == "target"
    assert config.target_config.devices == [1, 2]
    assert config.target_config.master_rank == 1
    assert native.draft_model == "draft"
    assert native.target_tp_size == 2
    assert native.draft_dtype == "bfloat16"
    assert native.target_dtype == "float16"
    assert native.max_num_batched_tokens == 1024
    assert native.max_aclgraph_entries == 32
    assert native.target_verification_graph_buckets == 8
    assert native.target_verification_graph_post_counts == ((8, (0, 4, 8)),)
    assert native.max_num_seqs == 8
    assert native.prefill_chunk_size == 4
    assert native.spec_rhythm_slo_home_partition is True
    assert native.max_num_queued_seqs == 12
    assert native.enable_continuous_batching is True
    assert native.enable_preemptive_scheduling is True
    assert native.enable_spec_rhythm is True
    assert native.spec_rhythm_min_gamma == 2
    assert native.spec_rhythm_max_eager_tokens == 3
    assert native.spec_rhythm_eager_reserve_tokens == 12
    assert native.spec_rhythm_roofline == {"8:1": 24}
    assert native.spec_rhythm_draft_token_budget == 32
    assert native.spec_rhythm_tree_width == 2
    assert native.spec_rhythm_tree_depth == 2
    assert native.pad_finished_requests is True
    assert native.draft_use_paged_attention is True
    assert native.target_use_paged_attention is True
    assert native.draft_use_production_rope is True
    assert native.target_use_production_rope is True
    assert native.precompile_decode_graphs is True
    assert native.enable_cpu_binding is False
    assert native.profile_decode_steps == 5
    assert native.profile_host_decode_steps == 7


def test_public_pearl_config_maps_opt_in_linear_full_window_to_native_runtime():
    model_config = SimpleNamespace(
        architectures=["Qwen2ForCausalLM"],
        eos_token_id=1,
    )
    with patch(
        "vllm_ascend.spec_decode.pearl.api.AutoConfig.from_pretrained",
        side_effect=[model_config, model_config],
    ):
        config = PEARLConfig(
            "draft",
            "target",
            gamma=4,
            enable_continuous_batching=True,
            enable_preemptive_scheduling=True,
            enable_spec_rhythm=True,
            spec_rhythm_linear_full_window=True,
            spec_rhythm_linear_eager_cross_graph_bucket=True,
            spec_rhythm_online_prefill=True,
            spec_rhythm_prefill_coalesce_min_requests=2,
            spec_rhythm_prefill_coalesce_max_wait_ms=600.0,
            spec_rhythm_prefill_token_chunk_size=128,
            spec_rhythm_min_gamma=4,
            spec_rhythm_tree_width=1,
            spec_rhythm_tree_depth=1,
            enable_prefix_caching=False,
        )

    native = config.to_native()
    assert config.spec_rhythm_linear_full_window is True
    assert native.spec_rhythm_linear_full_window is True
    assert config.spec_rhythm_linear_eager_cross_graph_bucket is True
    assert native.spec_rhythm_linear_eager_cross_graph_bucket is True
    assert native.spec_rhythm_online_prefill is True
    assert native.spec_rhythm_prefill_coalesce_min_requests == 2
    assert native.spec_rhythm_prefill_coalesce_max_wait_ms == 600.0
    assert native.spec_rhythm_prefill_token_chunk_size == 128


@pytest.mark.parametrize(
    "config_type",
    [NativePearlConfig, PEARLConfig],
)
def test_cross_graph_bucket_linear_eager_requires_full_window(config_type):
    common = {
        "gamma": 4,
        "max_num_seqs": 8,
        "enable_continuous_batching": True,
        "enable_preemptive_scheduling": True,
        "enable_spec_rhythm": True,
        "spec_rhythm_linear_eager_cross_graph_bucket": True,
        "spec_rhythm_min_gamma": 4,
    }
    if config_type is NativePearlConfig:
        with pytest.raises(ValueError, match="Cross-graph-bucket"):
            config_type(
                "draft",
                "target",
                1,
                3,
                max_model_len=512,
                max_tokens=32,
                **common,
            )
    else:
        with pytest.raises(ValueError, match="Cross-graph-bucket"):
            config_type("draft", "target", **common)


@pytest.mark.parametrize(
    "overrides",
    [
        {"enable_spec_rhythm": False},
        {"spec_rhythm_online_prefill": False},
        {"spec_rhythm_linear_full_window": False},
        {"spec_rhythm_tree_width": 2},
    ],
)
def test_native_prefill_coalescing_rejects_other_modes(overrides):
    values = {
        "draft_model": "draft",
        "target_model": "target",
        "draft_tp_size": 1,
        "target_tp_size": 3,
        "gamma": 4,
        "max_model_len": 512,
        "max_tokens": 32,
        "max_num_seqs": 8,
        "enable_continuous_batching": True,
        "enable_preemptive_scheduling": True,
        "enable_spec_rhythm": True,
        "spec_rhythm_linear_full_window": True,
        "spec_rhythm_online_prefill": True,
        "spec_rhythm_prefill_coalesce_min_requests": 2,
        "spec_rhythm_prefill_coalesce_max_wait_ms": 600.0,
        "spec_rhythm_min_gamma": 4,
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        NativePearlConfig(**values)


def test_public_prefill_coalescing_rejects_non_online_mode_before_model_load():
    with pytest.raises(ValueError, match="prefill coalescing"):
        PEARLConfig(
            "draft",
            "target",
            gamma=4,
            max_num_seqs=8,
            enable_continuous_batching=True,
            enable_preemptive_scheduling=True,
            enable_spec_rhythm=True,
            spec_rhythm_linear_full_window=True,
            spec_rhythm_online_prefill=False,
            spec_rhythm_prefill_coalesce_min_requests=2,
            spec_rhythm_prefill_coalesce_max_wait_ms=600.0,
            spec_rhythm_min_gamma=4,
        )


def test_native_prefill_coalescing_accepts_tree_scheduler():
    config = NativePearlConfig(
        "draft",
        "target",
        1,
        3,
        4,
        512,
        32,
        max_num_seqs=8,
        enable_continuous_batching=True,
        enable_preemptive_scheduling=True,
        enable_spec_rhythm=True,
        spec_rhythm_online_prefill=True,
        spec_rhythm_prefill_coalesce_min_requests=4,
        spec_rhythm_prefill_coalesce_max_wait_ms=1100.0,
        spec_rhythm_tree_width=2,
        spec_rhythm_tree_depth=2,
    )

    assert config.spec_rhythm_prefill_coalesce_min_requests == 4
    assert config.spec_rhythm_prefill_coalesce_max_wait_ms == 1100.0


@pytest.mark.parametrize(
    ("minimum", "wait_ms"),
    [(0, 0.0), (2, 0.0), (1, 1.0), (2, -1.0), (2, float("inf"))],
)
def test_native_prefill_coalescing_rejects_invalid_bounds(minimum, wait_ms):
    with pytest.raises(ValueError, match="prefill coalescing"):
        NativePearlConfig(
            "draft",
            "target",
            1,
            3,
            4,
            512,
            32,
            max_num_seqs=8,
            enable_continuous_batching=True,
            enable_preemptive_scheduling=True,
            enable_spec_rhythm=True,
            spec_rhythm_linear_full_window=True,
            spec_rhythm_online_prefill=True,
            spec_rhythm_prefill_coalesce_min_requests=minimum,
            spec_rhythm_prefill_coalesce_max_wait_ms=wait_ms,
            spec_rhythm_min_gamma=4,
        )


def test_public_pearl_config_maps_serial_draft_graph_precompile_to_native_runtime():
    model_config = SimpleNamespace(
        architectures=["Qwen2ForCausalLM"],
        eos_token_id=1,
    )
    with patch(
        "vllm_ascend.spec_decode.pearl.api.AutoConfig.from_pretrained",
        side_effect=[model_config, model_config],
    ):
        config = PEARLConfig(
            "draft",
            "target",
            gamma=4,
            max_num_seqs=8,
            enable_continuous_batching=True,
            enable_preemptive_scheduling=True,
            enable_spec_rhythm=True,
            draft_use_paged_attention=True,
            precompile_serial_draft_graphs=True,
        )

    native = config.to_native()
    assert config.precompile_serial_draft_graphs is True
    assert native.precompile_serial_draft_graphs is True
    assert native.precompile_decode_graphs is False


@pytest.mark.parametrize("tree_depth", [3, 4])
def test_serial_draft_graph_precompile_rejects_tree_chain_synthetic_kv(tree_depth):
    with pytest.raises(ValueError, match="1x1 serial topology"):
        NativePearlConfig(
            "draft",
            "target",
            1,
            3,
            4,
            512,
            32,
            max_num_seqs=8,
            enable_continuous_batching=True,
            enable_preemptive_scheduling=True,
            enable_spec_rhythm=True,
            spec_rhythm_tree_width=1,
            spec_rhythm_tree_depth=tree_depth,
            draft_use_paged_attention=True,
            precompile_serial_draft_graphs=True,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"enable_spec_rhythm": False},
        {"spec_rhythm_min_gamma": 3},
        {"spec_rhythm_tree_width": 2},
        {"spec_rhythm_tree_depth": 2},
    ],
)
def test_public_pearl_config_restricts_linear_full_window_to_fixed_gamma_serial_linear_path(
    overrides,
):
    values = {
        "gamma": 4,
        "enable_continuous_batching": True,
        "enable_preemptive_scheduling": True,
        "enable_spec_rhythm": True,
        "spec_rhythm_linear_full_window": True,
        "spec_rhythm_min_gamma": 4,
        "spec_rhythm_tree_width": 1,
        "spec_rhythm_tree_depth": 1,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match="linear full-window"):
        PEARLConfig("draft", "target", **values)


def test_public_pearl_config_rejects_partial_linear_full_window_eager_cap():
    with pytest.raises(ValueError, match="eager-token cap must be 0 or gamma"):
        PEARLConfig(
            "draft",
            "target",
            gamma=4,
            enable_continuous_batching=True,
            enable_preemptive_scheduling=True,
            enable_spec_rhythm=True,
            spec_rhythm_linear_full_window=True,
            spec_rhythm_min_gamma=4,
            spec_rhythm_max_eager_tokens=2,
        )


@pytest.mark.parametrize(
    "mode",
    ["serial", "nano_pearl", "dual_batch", "dual_batch_rolling"],
)
def test_native_fixed_gamma_ablation_modes_require_online_full_window(mode):
    common = dict(
        enable_continuous_batching=True,
        enable_preemptive_scheduling=True,
        enable_spec_rhythm=True,
        spec_rhythm_linear_full_window=True,
        spec_rhythm_online_prefill=True,
        spec_rhythm_min_gamma=4,
        spec_rhythm_ablation_mode=mode,
    )
    config = NativePearlConfig("draft", "target", 1, 4, 4, 512, 32, **common)
    assert config.spec_rhythm_ablation_mode == mode

    common["spec_rhythm_online_prefill"] = False
    with pytest.raises(ValueError, match="ablation modes"):
        NativePearlConfig("draft", "target", 1, 4, 4, 512, 32, **common)


def test_public_fixed_gamma_ablation_mode_maps_to_native_runtime():
    model_config = SimpleNamespace(
        architectures=["Qwen2ForCausalLM"],
        eos_token_id=1,
    )
    with patch(
        "vllm_ascend.spec_decode.pearl.api.AutoConfig.from_pretrained",
        side_effect=[model_config, model_config],
    ):
        config = PEARLConfig(
            "draft",
            "target",
            draft_tensor_parallel_size=1,
            target_tensor_parallel_size=4,
            gamma=4,
            enable_continuous_batching=True,
            enable_preemptive_scheduling=True,
            enable_spec_rhythm=True,
            spec_rhythm_linear_full_window=True,
            spec_rhythm_online_prefill=True,
            spec_rhythm_min_gamma=4,
            spec_rhythm_ablation_mode="nano_pearl",
        )
    assert config.to_native().spec_rhythm_ablation_mode == "nano_pearl"


def test_native_pearl_config_rejects_partial_linear_full_window_eager_cap():
    with pytest.raises(ValueError, match="eager-token cap must be 0 or gamma"):
        NativePearlConfig(
            "draft",
            "target",
            1,
            3,
            4,
            512,
            32,
            enable_continuous_batching=True,
            enable_preemptive_scheduling=True,
            enable_spec_rhythm=True,
            spec_rhythm_linear_full_window=True,
            spec_rhythm_min_gamma=4,
            spec_rhythm_max_eager_tokens=2,
        )


def test_public_pearl_config_rejects_full_window_draft_budget_below_gamma():
    with pytest.raises(ValueError, match="draft-token budget must be at least gamma"):
        PEARLConfig(
            "draft",
            "target",
            gamma=4,
            enable_continuous_batching=True,
            enable_preemptive_scheduling=True,
            enable_spec_rhythm=True,
            spec_rhythm_linear_full_window=True,
            spec_rhythm_min_gamma=4,
            spec_rhythm_draft_token_budget=3,
        )


def test_native_pearl_config_rejects_full_window_draft_budget_below_gamma():
    with pytest.raises(ValueError, match="draft-token budget must be at least gamma"):
        NativePearlConfig(
            "draft",
            "target",
            1,
            3,
            4,
            512,
            32,
            enable_continuous_batching=True,
            enable_preemptive_scheduling=True,
            enable_spec_rhythm=True,
            spec_rhythm_linear_full_window=True,
            spec_rhythm_min_gamma=4,
            spec_rhythm_draft_token_budget=3,
        )


def test_public_pearl_config_rejects_nonpositive_worker_timeout():
    with pytest.raises(ValueError, match="worker_timeout_seconds"):
        PEARLConfig("draft", "target", worker_timeout_seconds=0)


def test_native_pearl_config_rejects_unsupported_dtype():
    with pytest.raises(ValueError, match="model dtype"):
        NativePearlConfig("draft", "target", 1, 2, 4, 512, 32, target_dtype="float32")


def test_public_pearl_config_requires_continuous_batching_for_preemption():
    with pytest.raises(ValueError, match="preemptive scheduling"):
        PEARLConfig("draft", "target", enable_preemptive_scheduling=True)


def test_public_pearl_config_rejects_negative_profile_decode_steps():
    with pytest.raises(ValueError, match="profile_decode_steps"):
        PEARLConfig("draft", "target", profile_decode_steps=-1)


def test_public_pearl_config_rejects_negative_host_profile_decode_steps():
    with pytest.raises(ValueError, match="profile_host_decode_steps"):
        PEARLConfig("draft", "target", profile_host_decode_steps=-1)


def test_public_pearl_config_requires_steps_for_profiling_only():
    with pytest.raises(ValueError, match="profiling-only"):
        PEARLConfig("draft", "target", stop_after_profiled_decode_steps=True)


def test_native_graph_precompilation_requires_fixed_paged_configuration():
    common = {
        "draft_model": "draft",
        "target_model": "target",
        "draft_tp_size": 1,
        "target_tp_size": 1,
        "max_model_len": 32,
        "max_tokens": 8,
        "precompile_decode_graphs": True,
    }

    with pytest.raises(ValueError, match="fixed gamma"):
        NativePearlConfig(**common, gamma=-1)
    with pytest.raises(ValueError, match="paged attention"):
        NativePearlConfig(**common, gamma=4)
    with pytest.raises(ValueError, match="max_aclgraph_entries"):
        NativePearlConfig(
            **common,
            gamma=4,
            max_num_seqs=8,
            enable_continuous_batching=True,
            draft_use_paged_attention=True,
            target_use_paged_attention=True,
            max_aclgraph_entries=8,
        )


def test_serial_draft_graph_precompilation_requires_stable_paged_configuration():
    common = {
        "draft_model": "draft",
        "target_model": "target",
        "draft_tp_size": 1,
        "target_tp_size": 1,
        "gamma": 4,
        "max_model_len": 32,
        "max_tokens": 8,
        "max_num_seqs": 8,
        "enable_continuous_batching": True,
        "enable_preemptive_scheduling": True,
        "enable_spec_rhythm": True,
        "draft_use_paged_attention": True,
        "precompile_serial_draft_graphs": True,
    }

    config = NativePearlConfig(**common)
    assert config.precompile_serial_draft_graphs is True
    assert config.precompile_decode_graphs is False

    without_draft_paged = {**common, "draft_use_paged_attention": False}
    with pytest.raises(ValueError, match="paged attention on the draft model"):
        NativePearlConfig(**without_draft_paged)
    nonserial_tree = {**common, "spec_rhythm_tree_width": 2}
    with pytest.raises(ValueError, match="fixed-gamma stable SpecRhythm"):
        NativePearlConfig(**nonserial_tree)
    eager = {**common, "enforce_eager": True}
    with pytest.raises(ValueError, match="incompatible with enforce_eager"):
        NativePearlConfig(**eager)
    over_capacity = {
        **common,
        "max_num_seqs": 64,
        "max_aclgraph_entries": 8,
    }
    with pytest.raises(ValueError, match="exceeds max_aclgraph_entries"):
        NativePearlConfig(**over_capacity)


def test_profiling_only_continuous_results_can_include_unfinished_requests():
    completed = PearlPipelineState([1, 2], prompt_length=1)
    unfinished = PearlPipelineState([3, 4], prompt_length=1)
    completed_request_states = {0: completed}
    local_states = [PearlPipelineState([9], prompt_length=1), unfinished]

    result_states = _continuous_result_states(local_states, completed_request_states)

    assert result_states == [completed, unfinished]


@pytest.mark.parametrize("prefill_chunk_size", [0, 9])
def test_public_pearl_config_rejects_invalid_prefill_chunk_size(prefill_chunk_size):
    with pytest.raises(ValueError, match="prefill_chunk_size"):
        PEARLConfig(
            "draft",
            "target",
            max_num_seqs=8,
            prefill_chunk_size=prefill_chunk_size,
        )


def test_public_engine_collects_worker_replies_in_rank_order():
    engine = PEARLEngine.__new__(PEARLEngine)
    engine.config = SimpleNamespace(worker_timeout_seconds=1.0)
    engine._processes = [MagicMock(), MagicMock()]
    for process in engine._processes:
        process.is_alive.return_value = True
    rank0, worker0 = Pipe(duplex=True)
    rank1, worker1 = Pipe(duplex=True)
    engine._connections = [rank0, rank1]
    try:
        worker1.send(("ready", 1))
        worker0.send(("ready", 0))

        assert engine._receive_all("test") == [("ready", 0), ("ready", 1)]
    finally:
        for connection in (rank0, rank1, worker0, worker1):
            connection.close()


def test_public_engine_preserves_sampling_request_metadata_when_not_overridden():
    engine = PEARLEngine.__new__(PEARLEngine)
    engine.config = SimpleNamespace(max_model_len=64)
    engine._requests = []
    engine._next_request_id = 0
    params = SamplingParams(
        max_tokens=4,
        request_id="trace-7",
        arrival_ts=123.5,
        slo_tpot_ms=20.0,
        slo_class="tight",
        spec_rhythm_max_gamma=3,
    )

    engine.add_request([1, 2], params)

    queued = engine._requests[0][2]
    assert queued.request_id == "trace-7"
    assert queued.arrival_ts == 123.5
    assert queued.slo_tpot_ms == 20.0
    assert queued.slo_class == "tight"
    assert queued.spec_rhythm_max_gamma == 3


def test_public_engine_times_out_with_pending_worker_ranks():
    engine = PEARLEngine.__new__(PEARLEngine)
    engine.config = SimpleNamespace(worker_timeout_seconds=0.01)
    process = MagicMock()
    process.is_alive.return_value = True
    parent, worker = Pipe(duplex=True)
    engine._processes = [process]
    engine._connections = [parent]
    try:
        with pytest.raises(TimeoutError, match=r"pending ranks: \[0\]"):
            engine._receive_all("test timeout")
    finally:
        parent.close()
        worker.close()


def test_public_engine_broadcasts_upstream_log_command():
    engine = PEARLEngine.__new__(PEARLEngine)
    engine._send_all = MagicMock()
    engine._receive_all = MagicMock(return_value=[("logged", 0), ("logged", 1)])

    engine.log("ready")

    engine._send_all.assert_called_once_with(("log", "ready", None, None))
    engine._receive_all.assert_called_once_with("worker logging")


def test_public_engine_configures_decode_profiling_on_every_worker():
    engine = PEARLEngine.__new__(PEARLEngine)
    engine._send_all = MagicMock()
    engine._receive_all = MagicMock(return_value=[("configured", 0), ("configured", 1)])

    engine.configure_decode_profiling(5, True, 7)

    engine._send_all.assert_called_once_with(("configure_decode_profiling", 5, True, 7))
    engine._receive_all.assert_called_once_with("worker profiling configuration")


def test_public_engine_seals_and_unseals_graph_cache_on_every_worker():
    engine = PEARLEngine.__new__(PEARLEngine)
    engine._send_all = MagicMock()
    sealed_metrics = [{"rank": 0, "aclgraph_sealed": 1}]
    unsealed_metrics = [{"rank": 0, "aclgraph_sealed": 0}]
    pruned_metrics = [{"rank": 0, "aclgraph_pruned_unvalidated_entries": 2}]
    engine._receive_all = MagicMock(
        side_effect=[
            [("graph_cache_sealed", 0, sealed_metrics[0])],
            [("graph_cache_unsealed", 0, unsealed_metrics[0])],
            [("graph_cache_pruned", 0, pruned_metrics[0])],
        ]
    )

    assert engine.seal_graph_cache() == sealed_metrics
    assert engine.unseal_graph_cache() == unsealed_metrics
    assert engine.prune_unvalidated_graph_entries() == pruned_metrics

    assert engine._send_all.call_args_list == [
        call(("seal_graph_cache", None, None, None)),
        call(("unseal_graph_cache", None, None, None)),
        call(("prune_unvalidated_graph_entries", None, None, None)),
    ]
    assert engine._receive_all.call_args_list == [
        call("worker graph-cache sealing"),
        call("worker graph-cache unsealing"),
        call("worker graph-cache pruning"),
    ]
    assert engine.last_worker_metrics == pruned_metrics


def test_public_engine_publishes_live_admission_with_monotonic_epoch_sequence():
    engine = PEARLEngine.__new__(PEARLEngine)
    receiver, sender = Pipe(duplex=False)
    engine.config = SimpleNamespace(
        spec_rhythm_tree_width=2,
        spec_rhythm_tree_depth=2,
        max_model_len=64,
        max_num_queued_seqs=4,
        max_num_seqs=2,
    )
    engine._admission_connections = [sender]
    engine._live_lock = __import__("threading").Lock()
    engine._live_epoch = 7
    engine._live_accepting = True
    engine._live_admission_seq = 0
    engine._live_total_requests = 1
    engine._live_sampling_signature = (False, False)
    params = SamplingParams(temperature=0, max_tokens=2, request_id="live-1")
    try:
        engine.admit_live_requests([([1, 2], params)])
        message = receiver.recv()
        assert message[:3] == ("admit", 7, 1)
        assert message[3] == [([1, 2], params)]
        assert engine._live_total_requests == 2
    finally:
        receiver.close()
        sender.close()


def test_public_engine_idle_fence_does_not_overtake_pending_live_admission():
    engine = PEARLEngine.__new__(PEARLEngine)
    receiver, sender = Pipe(duplex=False)
    engine._admission_connections = [sender]
    engine._live_lock = __import__("threading").Lock()
    engine._live_epoch = 3
    engine._live_accepting = True
    engine._live_admission_seq = 2
    try:
        engine._resolve_live_idle(3, worker_admission_seq=1)
        assert engine._live_accepting is True
        assert receiver.poll() is False

        engine._resolve_live_idle(3, worker_admission_seq=2)
        assert engine._live_accepting is False
        assert receiver.recv() == ("close_live", 3, 2, ())
    finally:
        receiver.close()
        sender.close()


def test_native_engine_reconfigures_decode_profiling_without_model_reload():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.config = NativePearlConfig(
        draft_model="draft",
        target_model="target",
        draft_tp_size=1,
        target_tp_size=1,
        gamma=4,
        max_model_len=32,
        max_tokens=8,
        profile_decode_steps=0,
    )

    engine.configure_decode_profiling(5, True)

    assert engine.config.profile_decode_steps == 5
    assert engine.config.stop_after_profiled_decode_steps is True


def test_public_engine_chunks_queued_requests_by_sequence_and_token_limits():
    engine = PEARLEngine.__new__(PEARLEngine)
    engine.config = SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=5)
    params = SamplingParams(temperature=0, max_tokens=4)
    requests = [
        (0, [1, 2, 3], params),
        (1, [4, 5], params),
        (2, [6, 7, 8], params),
    ]

    chunks = engine._request_chunks(requests)

    assert [[request[0] for request in chunk] for chunk in chunks] == [[0, 1], [2]]


def test_public_engine_validates_pipeline_and_benchmark_cache_capacity_before_dispatch():
    engine = PEARLEngine.__new__(PEARLEngine)
    engine.config = SimpleNamespace(gamma=4, max_model_len=10)
    engine._requests = [(0, [1, 2, 3, 4, 5], SamplingParams(temperature=0, max_tokens=2))]

    with pytest.raises(ValueError, match="verification window"):
        engine.generate()
    with pytest.raises(ValueError, match="benchmark steps"):
        engine.bench_generate(num_pearl_steps=1)


def test_public_engine_aggregates_aclgraph_metrics_from_every_worker():
    engine = PEARLEngine.__new__(PEARLEngine)
    engine.config = SimpleNamespace(
        gamma=4,
        max_model_len=32,
        max_num_seqs=1,
        max_num_batched_tokens=32,
        worker_timeout_seconds=1,
    )
    engine._requests = [(0, [1], SamplingParams(max_tokens=1))]
    engine.last_metrics = []
    engine.tokenizer = MagicMock()
    engine.tokenizer.decode.return_value = "result"
    engine._send_all = MagicMock()
    leader_result = {
        "completion_token_ids": [2],
        "num_acc_tokens": [1],
        "elapsed_seconds": 1.0,
    }
    base_metrics = {
        "aclgraph_captures": 2,
        "aclgraph_capture_attempts": 2,
        "aclgraph_replays": 3,
        "aclgraph_failed_captures": 0,
        "aclgraph_capacity_fallbacks": 0,
        "aclgraph_shape_fallbacks": 0,
    }
    failed_rank_metrics = {
        **base_metrics,
        "aclgraph_capture_attempts": 8,
        "aclgraph_failed_captures": 6,
        "aclgraph_capacity_fallbacks": 9,
        "aclgraph_shape_fallbacks": 11,
        "worker_spec_rhythm_draft_materialized_nodes": 7,
    }
    engine._receive_all = MagicMock(
        return_value=[
            ("result", None, failed_rank_metrics),
            ("result", [leader_result], base_metrics),
        ]
    )

    _, num_tokens, _, elapsed = engine._generate("pearl")

    assert num_tokens == [1]
    assert elapsed == 1.0
    assert engine.last_metrics[0]["aclgraph_capture_attempts"] == 8
    assert engine.last_metrics[0]["aclgraph_failed_captures"] == 6
    assert engine.last_metrics[0]["aclgraph_capacity_fallbacks"] == 9
    assert engine.last_metrics[0]["aclgraph_shape_fallbacks"] == 11
    assert engine.last_metrics[0]["worker_spec_rhythm_draft_materialized_nodes"] == 7
    assert engine.last_worker_metrics_by_chunk == [
        {
            "batch_size": 1,
            "num_requests": 1,
            "worker_metrics": [failed_rank_metrics, base_metrics],
        }
    ]


def test_decode_profile_aggregates_only_full_batch_chunks_per_step():
    def worker(rank, is_draft, scale):
        return {
            "rank": rank,
            "is_draft_rank": int(is_draft),
            "worker_profiled_decode_steps": 2,
            "worker_profile_draft_compute_seconds": 0.01 * scale,
            "worker_profile_draft_to_target_communication_seconds": 0.02 * scale,
            "worker_profile_target_compute_seconds": 0.03 * scale,
            "worker_profile_target_verdict_seconds": 0.04 * scale,
            "worker_profile_target_to_draft_communication_seconds": 0.05 * scale,
            "worker_profile_wait_sync_seconds": 0.06 * scale,
            "worker_profile_state_update_seconds": 0.07 * scale,
            "worker_profile_detail_commit_consensus_seconds": 0.08 * scale,
            "worker_profile_detail_kv_compaction_seconds": 0.09 * scale,
        }

    chunks = [
        {
            "batch_size": 8,
            "worker_metrics": [worker(0, True, 1), worker(1, False, 2), worker(2, False, 3)],
        },
        {
            "batch_size": 3,
            "worker_metrics": [worker(0, True, 100), worker(1, False, 100)],
        },
    ]

    profile = _aggregate_decode_profile(chunks, batch_size=8)

    assert profile["profiled_full_batch_chunks"] == 1
    assert profile["profiled_decode_steps"] == 2
    assert profile["phase_milliseconds_per_decode_step"] == pytest.approx(
        {
            "draft_compute": 5.0,
            "draft_to_target_communication": 30.0,
            "target_verify": 105.0,
            "target_to_draft_communication": 75.0,
            "wait_sync_state_update": 195.0,
        }
    )
    assert profile["detail_milliseconds_per_decode_step"] == pytest.approx(
        {"commit_consensus": 120.0, "kv_compaction": 135.0}
    )


def test_host_decode_profile_merges_rank_local_timestamps_without_sync():
    def trace(rank, is_draft, *, draft=(1.01, 1.03), target=(1.012, 1.052)):
        return {
            "rank": rank,
            "is_draft_rank": int(is_draft),
            "worker_host_timeline": [
                {
                    "step": 0,
                    "rank": rank,
                    "is_draft_rank": int(is_draft),
                    "cycle_start_seconds": 1.0,
                    "scheduler_roof_end_seconds": 1.002,
                    "scheduler_plan_end_seconds": 1.005,
                    "scheduler_budget_end_seconds": 1.007,
                    "scheduler_end_seconds": 1.01,
                    "draft_start_seconds": draft[0],
                    "draft_end_seconds": draft[1],
                    "target_start_seconds": target[0],
                    "target_end_seconds": target[1],
                    "linear_w_observed_draft_ms": 18.0,
                    "linear_w_observed_target_ms": 40.0,
                    "compute_coordination_start_seconds": 1.03,
                    "compute_coordination_end_seconds": 1.052,
                    "compact_submit_start_seconds": 1.052,
                    "compact_submit_end_seconds": 1.053,
                    "compact_wait_start_seconds": 1.053,
                    "compact_wait_end_seconds": 1.054,
                    "overlap_prefill_start_seconds": 1.054,
                    "overlap_prefill_end_seconds": 1.056,
                    "exchange_start_seconds": 1.052,
                    "exchange_end_seconds": 1.057 + rank * 0.001,
                    "verdict_start_seconds": 1.057,
                    "verdict_end_seconds": 1.059,
                    "correction_start_seconds": 1.059,
                    "correction_end_seconds": 1.063 + rank * 0.001,
                    "state_start_seconds": 1.064,
                    "state_preflight_end_seconds": 1.0645,
                    "state_consensus_end_seconds": 1.065,
                    "state_compaction_end_seconds": 1.0652,
                    "state_request_commit_end_seconds": 1.066,
                    "state_end_seconds": 1.066,
                    "cycle_end_seconds": 1.07 + rank * 0.001,
                    "accounted_tail_ms": 1.25,
                    "new_activation_tail_request_ms": 0.5,
                    "target_requests": 4,
                    "draft_requests": 4,
                    "verify_candidates": 8,
                }
            ],
        }

    profile = _aggregate_decode_host_profile(
        [
            {
                "batch_size": 8,
                "worker_metrics": [
                    trace(0, True),
                    trace(1, False),
                    trace(2, False),
                    trace(3, False),
                ],
            }
        ],
        batch_size=8,
    )

    cycle = profile["cycles"][0]
    assert cycle["cycle_wall_ms"] == pytest.approx(73.0)
    assert cycle["scheduler_roof_critical_ms"] == pytest.approx(2.0)
    assert cycle["scheduler_plan_critical_ms"] == pytest.approx(3.0)
    assert cycle["scheduler_budget_critical_ms"] == pytest.approx(2.0)
    assert cycle["scheduler_tree_plan_critical_ms"] == pytest.approx(3.0)
    assert cycle["draft_compute_host_ms"] == pytest.approx(20.0)
    assert cycle["target_compute_host_ms"] == pytest.approx(40.0)
    assert cycle["host_compute_overlap_ms"] == pytest.approx(18.0)
    assert cycle["draft_compute_device_ms"] == pytest.approx(18.0)
    assert cycle["target_verify_device_ms"] == pytest.approx(40.0)
    assert cycle["draft_device_below_target"] is True
    assert cycle["compute_coordination_critical_ms"] == pytest.approx(22.0)
    assert cycle["compact_submit_critical_ms"] == pytest.approx(1.0)
    assert cycle["compact_wait_critical_ms"] == pytest.approx(1.0)
    assert cycle["staged_prefill_critical_ms"] == pytest.approx(2.0)
    assert cycle["draft_to_target_critical_ms"] == pytest.approx(8.0)
    assert cycle["target_to_draft_critical_ms"] == pytest.approx(7.0)
    assert cycle["state_update_critical_ms"] == pytest.approx(2.0)
    assert cycle["state_preflight_critical_ms"] == pytest.approx(0.5)
    assert cycle["state_consensus_critical_ms"] == pytest.approx(0.5)
    assert cycle["state_compaction_critical_ms"] == pytest.approx(0.2)
    assert cycle["state_request_commit_critical_ms"] == pytest.approx(0.8)
    assert cycle["accounted_tail_ms"] == pytest.approx(1.25)
    assert cycle["new_activation_tail_request_ms"] == pytest.approx(0.5)


def test_worker_aclgraph_deltas_match_workers_by_rank():
    before = [
        {
            "rank": 1,
            "aclgraph_captures": 2,
            "aclgraph_capture_attempts": 2,
            "aclgraph_replays": 4,
            "aclgraph_failed_captures": 0,
            "aclgraph_capacity_fallbacks": 0,
            "aclgraph_shape_fallbacks": 0,
        },
        {
            "rank": 0,
            "aclgraph_captures": 1,
            "aclgraph_capture_attempts": 1,
            "aclgraph_replays": 3,
            "aclgraph_failed_captures": 0,
            "aclgraph_capacity_fallbacks": 0,
            "aclgraph_shape_fallbacks": 0,
        },
    ]
    after = [
        {**before[1], "aclgraph_replays": 8},
        {
            **before[0],
            "aclgraph_captures": 3,
            "aclgraph_capture_attempts": 3,
            "aclgraph_replays": 9,
        },
    ]

    deltas = _worker_aclgraph_deltas(before, after)

    assert deltas[0]["rank"] == 0
    assert deltas[0]["aclgraph_capture_attempts_delta"] == 0
    assert deltas[0]["aclgraph_replays_delta"] == 5
    assert deltas[1]["rank"] == 1
    assert deltas[1]["aclgraph_capture_attempts_delta"] == 1
    assert deltas[1]["aclgraph_replays_delta"] == 5


def test_worker_aclgraph_deltas_count_cold_measurement_without_warmup():
    after = [
        {
            "rank": 0,
            "aclgraph_captures": 2,
            "aclgraph_capture_attempts": 2,
            "aclgraph_replays": 7,
            "aclgraph_failed_captures": 0,
            "aclgraph_capacity_fallbacks": 0,
            "aclgraph_shape_fallbacks": 0,
        }
    ]

    deltas = _worker_aclgraph_deltas([], after)

    assert len(deltas) == 1
    assert deltas[0]["rank"] == 0
    assert deltas[0]["is_draft_rank"] == 0
    assert deltas[0]["aclgraph_captures_delta"] == 2
    assert deltas[0]["aclgraph_capture_attempts_delta"] == 2
    assert deltas[0]["aclgraph_replays_delta"] == 7
    assert deltas[0]["aclgraph_failed_captures_delta"] == 0
    assert deltas[0]["aclgraph_runtime_validation_replays_delta"] == 0
    assert deltas[0]["aclgraph_generic_total_calls_delta"] == 0


def test_worker_aclgraph_deltas_isolate_measured_draft_taskless_pa_counters():
    before = [
        {
            "rank": 0,
            "is_draft_rank": 1,
            "aclgraph_task_update_replays": 7,
            "aclgraph_task_update_skipped_replays": 2,
            "aclgraph_draft_task_update_replays": 7,
            "aclgraph_draft_task_update_tasks": 784,
            "aclgraph_draft_taskless_replays": 3,
            "aclgraph_draft_step_major_pa_replays": 5,
            "aclgraph_draft_step_major_pa_fallback_replays": 1,
            "aclgraph_pa_workspace_host_key_tasks": 784,
            "aclgraph_pa_workspace_tensor_key_tasks": 4,
            "aclgraph_pa_workspace_get_calls": 28,
            "aclgraph_pa_workspace_cache_hits": 756,
            "aclgraph_pa_task_update_profiled_tasks": 784,
            "aclgraph_pa_task_update_host_key_ns": 1000,
            "aclgraph_pa_task_update_host_get_workspace_ns": 2000,
            "aclgraph_pa_task_update_host_task_update_ns": 3000,
        }
    ]
    after = [
        {
            **before[0],
            "aclgraph_task_update_replays": 12,
            "aclgraph_task_update_skipped_replays": 3,
            "aclgraph_draft_task_update_replays": 12,
            "aclgraph_draft_task_update_tasks": 1344,
            "aclgraph_draft_taskless_replays": 8,
            "aclgraph_draft_step_major_pa_replays": 9,
            "aclgraph_draft_step_major_pa_fallback_replays": 2,
            "aclgraph_pa_workspace_host_key_tasks": 1344,
            "aclgraph_pa_workspace_tensor_key_tasks": 8,
            "aclgraph_pa_workspace_get_calls": 48,
            "aclgraph_pa_workspace_cache_hits": 1296,
            "aclgraph_pa_task_update_profiled_tasks": 1344,
            "aclgraph_pa_task_update_host_key_ns": 1700,
            "aclgraph_pa_task_update_host_get_workspace_ns": 3200,
            "aclgraph_pa_task_update_host_task_update_ns": 5100,
        }
    ]

    delta = _worker_aclgraph_deltas(before, after)[0]

    assert delta["aclgraph_task_update_replays_delta"] == 5
    assert delta["aclgraph_task_update_skipped_replays_delta"] == 1
    assert delta["aclgraph_draft_task_update_replays_delta"] == 5
    assert delta["aclgraph_draft_task_update_tasks_delta"] == 560
    assert delta["aclgraph_draft_taskless_replays_delta"] == 5
    assert delta["aclgraph_draft_step_major_pa_replays_delta"] == 4
    assert delta["aclgraph_draft_step_major_pa_fallback_replays_delta"] == 1
    assert delta["aclgraph_pa_workspace_host_key_tasks_delta"] == 560
    assert delta["aclgraph_pa_workspace_tensor_key_tasks_delta"] == 4
    assert delta["aclgraph_pa_workspace_get_calls_delta"] == 20
    assert delta["aclgraph_pa_workspace_cache_hits_delta"] == 540
    assert delta["aclgraph_pa_task_update_profiled_tasks_delta"] == 560
    assert delta["aclgraph_pa_task_update_host_key_ns_delta"] == 700
    assert delta["aclgraph_pa_task_update_host_get_workspace_ns_delta"] == 1200
    assert delta["aclgraph_pa_task_update_host_task_update_ns_delta"] == 2100


def test_worker_aclgraph_deltas_include_mc2_dispatch_counters():
    before = [
        {
            "rank": 1,
            "mc2_dispatch_fused_attempt": 7,
            "mc2_dispatch_fused_success": 5,
            "mc2_dispatch_fallback": 2,
            "mc2_dispatch_exception": 0,
        }
    ]
    after = [
        {
            "rank": 1,
            "mc2_dispatch_fused_attempt": 11,
            "mc2_dispatch_fused_success": 8,
            "mc2_dispatch_fallback": 3,
            "mc2_dispatch_exception": 1,
        },
        {
            "rank": 2,
            "mc2_dispatch_fused_attempt": 2,
            "mc2_dispatch_fused_success": 2,
        },
    ]

    deltas = _worker_aclgraph_deltas(before, after)

    assert deltas[0]["mc2_dispatch_fused_attempt_delta"] == 4
    assert deltas[0]["mc2_dispatch_fused_success_delta"] == 3
    assert deltas[0]["mc2_dispatch_fallback_delta"] == 1
    assert deltas[0]["mc2_dispatch_exception_delta"] == 1
    assert deltas[1]["mc2_dispatch_fused_attempt_delta"] == 2
    assert deltas[1]["mc2_dispatch_fused_success_delta"] == 2
    assert deltas[1]["mc2_dispatch_fallback_delta"] == 0
    assert deltas[1]["mc2_dispatch_exception_delta"] == 0


def test_worker_aclgraph_deltas_include_mixed_target_service_contract():
    before = [
        {
            "rank": 1,
            "is_draft_rank": 0,
            "spec_rhythm_mixed_target_graph_enabled": 1,
            "spec_rhythm_mixed_target_graph_qualified_buckets": 6,
            "spec_rhythm_mixed_target_graph_prefill_only": 0,
            "spec_rhythm_stable_target_verify_graph_enabled": 1,
            "spec_rhythm_stable_target_verify_graph_qualified": 1,
            "spec_rhythm_stable_target_verify_graph_qualified_capacities": 32,
            "spec_rhythm_stable_target_verify_graph_qualified_capacity_mask": 0xFFFFFFFF,
            "spec_rhythm_stable_target_verify_graph_max_qualified_capacity": 32,
            "spec_rhythm_stable_target_verify_graph_exact_routes": 32,
            "spec_rhythm_stable_target_verify_numerical_validation_attempts": 64,
            "spec_rhythm_stable_target_verify_numerical_validation_passes": 64,
            "spec_rhythm_stable_target_verify_numerical_validation_failures": 0,
            "spec_rhythm_stable_target_verify_numerical_restore_failures": 0,
            "worker_spec_rhythm_serial_protocol": 0,
            "worker_spec_rhythm_single_batch_overlap_protocol": 0,
            "worker_spec_rhythm_dual_batch_overlap_protocol": 1,
            "worker_spec_rhythm_concurrent_draft_target_submission_cycles": 11,
            "worker_spec_rhythm_single_batch_overlap_submission_cycles": 0,
            "worker_spec_rhythm_dual_batch_overlap_submission_cycles": 11,
            "worker_spec_rhythm_mixed_target_prefill_batches": 3,
            "worker_spec_rhythm_mixed_target_graph_batches": 3,
            "worker_spec_rhythm_mixed_target_graph_requests": 7,
            "worker_spec_rhythm_mixed_target_graph_prompt_tokens": 511,
            "worker_spec_rhythm_mixed_target_graph_token_capacity_fallback_batches": 0,
        }
    ]
    after = [
        {
            **before[0],
            "worker_spec_rhythm_concurrent_draft_target_submission_cycles": 17,
            "worker_spec_rhythm_dual_batch_overlap_submission_cycles": 17,
            "worker_spec_rhythm_mixed_target_prefill_batches": 8,
            "worker_spec_rhythm_mixed_target_graph_batches": 8,
            "worker_spec_rhythm_mixed_target_graph_requests": 19,
            "worker_spec_rhythm_mixed_target_graph_prompt_tokens": 2047,
            "worker_spec_rhythm_mixed_target_graph_token_capacity_fallback_batches": 1,
        }
    ]

    delta = _worker_aclgraph_deltas(before, after)[0]

    assert delta["spec_rhythm_mixed_target_graph_enabled"] == 1
    assert delta["spec_rhythm_mixed_target_graph_qualified_buckets"] == 6
    assert delta["spec_rhythm_mixed_target_graph_prefill_only"] == 0
    assert delta["spec_rhythm_stable_target_verify_graph_qualified"] == 1
    assert delta["spec_rhythm_stable_target_verify_graph_qualified_capacity_mask"] == 0xFFFFFFFF
    assert delta["spec_rhythm_stable_target_verify_numerical_validation_passes"] == 64
    assert delta["worker_spec_rhythm_serial_protocol"] == 0
    assert delta["worker_spec_rhythm_single_batch_overlap_protocol"] == 0
    assert delta["worker_spec_rhythm_dual_batch_overlap_protocol"] == 1
    assert delta["worker_spec_rhythm_concurrent_draft_target_submission_cycles"] == 17
    assert delta["worker_spec_rhythm_single_batch_overlap_submission_cycles"] == 0
    assert delta["worker_spec_rhythm_dual_batch_overlap_submission_cycles"] == 17
    assert delta["worker_spec_rhythm_mixed_target_prefill_batches"] == 8
    assert delta["worker_spec_rhythm_mixed_target_graph_batches"] == 8
    assert delta["worker_spec_rhythm_mixed_target_graph_requests"] == 19
    assert delta["worker_spec_rhythm_mixed_target_graph_prompt_tokens"] == 2047
    assert delta["worker_spec_rhythm_mixed_target_graph_token_capacity_fallback_batches"] == 1


def test_graph_only_benchmark_gate_accepts_active_workers_without_fallback():
    deltas = [
        {
            "rank": rank,
            "aclgraph_captures_delta": 0,
            "aclgraph_capture_attempts_delta": 0,
            "aclgraph_replays_delta": 4,
            "aclgraph_failed_captures_delta": 0,
            "aclgraph_capacity_fallbacks_delta": 0,
            "aclgraph_shape_fallbacks_delta": 0,
        }
        for rank in range(4)
    ]
    _require_no_graph_fallback(deltas)


def _stable_target_verify_graph_contract(capacity=32):
    return {
        "spec_rhythm_stable_target_verify_graph_enabled": 1,
        "spec_rhythm_stable_target_verify_graph_qualified": 1,
        "spec_rhythm_stable_target_verify_graph_qualified_capacities": capacity,
        "spec_rhythm_stable_target_verify_graph_qualified_capacity_mask": (1 << capacity) - 1,
        "spec_rhythm_stable_target_verify_graph_max_qualified_capacity": capacity,
        "spec_rhythm_stable_target_verify_graph_exact_routes": capacity,
        "spec_rhythm_stable_target_verify_graph_routes": capacity,
        "spec_rhythm_stable_target_verify_request_bucket": 1,
        "spec_rhythm_stable_target_verify_numerical_validation_attempts": 2 * capacity,
        "spec_rhythm_stable_target_verify_numerical_validation_passes": 2 * capacity,
        "spec_rhythm_stable_target_verify_numerical_validation_failures": 0,
        "spec_rhythm_stable_target_verify_numerical_restore_failures": 0,
        "worker_spec_rhythm_stable_target_verify_graph_batches": 7,
        "worker_spec_rhythm_stable_target_verify_graph_requests": 19,
        "worker_spec_rhythm_stable_target_verify_graph_bounded_shape_fallback_batches": 0,
        "worker_spec_rhythm_stable_target_verify_graph_token_capacity_fallback_batches": 0,
        "worker_spec_rhythm_stable_target_verify_graph_execution_fallback_batches": 0,
    }


def test_graph_only_benchmark_gate_accepts_all_mixed_target_prefills_on_graph():
    deltas = _full_window_graph_deltas()
    deltas[0].update(
        {
            "spec_rhythm_mixed_target_graph_enabled": 0,
            "spec_rhythm_mixed_target_graph_qualified_buckets": 0,
        }
    )
    deltas[1].update(
        {
            "spec_rhythm_mixed_target_graph_enabled": 1,
            "spec_rhythm_mixed_target_graph_qualified_buckets": 6,
            "worker_spec_rhythm_mixed_target_prefill_batches": 3,
            "worker_spec_rhythm_mixed_target_graph_batches": 3,
            "worker_spec_rhythm_mixed_target_graph_bounded_shape_fallback_batches": 0,
            "worker_spec_rhythm_mixed_target_graph_token_capacity_fallback_batches": 0,
            "worker_spec_rhythm_mixed_target_graph_execution_fallback_batches": 0,
            **_stable_target_verify_graph_contract(),
        }
    )

    _require_no_graph_fallback(deltas, require_full_window=True)


def test_graph_only_benchmark_gate_accepts_kv_ready_without_mixed_prefill():
    deltas = _full_window_graph_deltas()
    deltas[0].update(
        {
            "spec_rhythm_mixed_target_graph_enabled": 0,
            "spec_rhythm_mixed_target_graph_qualified_buckets": 0,
        }
    )
    deltas[1].update(
        {
            "spec_rhythm_mixed_target_graph_enabled": 1,
            "spec_rhythm_mixed_target_graph_qualified_buckets": 6,
            "worker_spec_rhythm_mixed_target_prefill_batches": 0,
            "worker_spec_rhythm_mixed_target_graph_batches": 0,
            "worker_spec_rhythm_mixed_target_graph_bounded_shape_fallback_batches": 0,
            "worker_spec_rhythm_mixed_target_graph_token_capacity_fallback_batches": 0,
            "worker_spec_rhythm_mixed_target_graph_execution_fallback_batches": 0,
            **_stable_target_verify_graph_contract(),
        }
    )

    _require_no_graph_fallback(
        deltas,
        require_full_window=True,
        require_mixed_prefill=False,
    )


def test_graph_only_benchmark_gate_accepts_exact_verify_family_through_48():
    deltas = _full_window_graph_deltas()
    deltas[0].update(
        {
            "spec_rhythm_mixed_target_graph_enabled": 0,
            "spec_rhythm_mixed_target_graph_qualified_buckets": 0,
        }
    )
    deltas[1].update(
        {
            "spec_rhythm_mixed_target_graph_enabled": 1,
            "spec_rhythm_mixed_target_graph_qualified_buckets": 11,
            "worker_spec_rhythm_mixed_target_prefill_batches": 3,
            "worker_spec_rhythm_mixed_target_graph_batches": 3,
            "worker_spec_rhythm_mixed_target_graph_bounded_shape_fallback_batches": 0,
            "worker_spec_rhythm_mixed_target_graph_token_capacity_fallback_batches": 0,
            "worker_spec_rhythm_mixed_target_graph_execution_fallback_batches": 0,
            **_stable_target_verify_graph_contract(48),
        }
    )

    _require_no_graph_fallback(deltas, require_full_window=True)


def test_graph_only_benchmark_gate_accepts_prefill_only_mixed_graph_family():
    deltas = _full_window_graph_deltas()
    deltas[0].update(
        {
            "spec_rhythm_mixed_target_graph_enabled": 0,
            "spec_rhythm_mixed_target_graph_qualified_buckets": 0,
            "spec_rhythm_mixed_target_graph_prefill_only": 0,
        }
    )
    deltas[1].update(
        {
            "spec_rhythm_mixed_target_graph_enabled": 1,
            "spec_rhythm_mixed_target_graph_qualified_buckets": 9,
            "spec_rhythm_mixed_target_graph_prefill_only": 1,
            "worker_spec_rhythm_mixed_target_prefill_batches": 3,
            "worker_spec_rhythm_mixed_target_graph_batches": 3,
            "worker_spec_rhythm_mixed_target_graph_bounded_shape_fallback_batches": 0,
            "worker_spec_rhythm_mixed_target_graph_token_capacity_fallback_batches": 0,
            "worker_spec_rhythm_mixed_target_graph_execution_fallback_batches": 0,
            "spec_rhythm_stable_target_verify_graph_enabled": 0,
            "spec_rhythm_stable_target_verify_graph_qualified": 0,
            "spec_rhythm_stable_target_verify_graph_qualified_capacities": 0,
            "spec_rhythm_stable_target_verify_graph_qualified_capacity_mask": 0,
            "spec_rhythm_stable_target_verify_graph_max_qualified_capacity": 0,
            "spec_rhythm_stable_target_verify_graph_exact_routes": 0,
            "spec_rhythm_stable_target_verify_numerical_validation_attempts": 0,
            "spec_rhythm_stable_target_verify_numerical_validation_passes": 0,
            "spec_rhythm_stable_target_verify_numerical_validation_failures": 0,
            "spec_rhythm_stable_target_verify_numerical_restore_failures": 0,
            "worker_spec_rhythm_stable_target_verify_graph_batches": 0,
            "worker_spec_rhythm_stable_target_verify_graph_requests": 0,
            "worker_spec_rhythm_stable_target_verify_graph_bounded_shape_fallback_batches": 0,
            "worker_spec_rhythm_stable_target_verify_graph_token_capacity_fallback_batches": 0,
            "worker_spec_rhythm_stable_target_verify_graph_execution_fallback_batches": 0,
        }
    )

    _require_no_graph_fallback(deltas, require_full_window=True)


@pytest.mark.parametrize(
    "counter",
    [
        "worker_spec_rhythm_mixed_target_graph_bounded_shape_fallback_batches",
        "worker_spec_rhythm_mixed_target_graph_token_capacity_fallback_batches",
        "worker_spec_rhythm_mixed_target_graph_execution_fallback_batches",
        "worker_spec_rhythm_stable_target_verify_graph_bounded_shape_fallback_batches",
        "worker_spec_rhythm_stable_target_verify_graph_token_capacity_fallback_batches",
        "worker_spec_rhythm_stable_target_verify_graph_execution_fallback_batches",
    ],
)
def test_graph_only_benchmark_gate_rejects_mixed_target_fallback(counter):
    deltas = _full_window_graph_deltas()
    deltas[0]["spec_rhythm_mixed_target_graph_enabled"] = 0
    deltas[1].update(
        {
            "spec_rhythm_mixed_target_graph_enabled": 1,
            "spec_rhythm_mixed_target_graph_qualified_buckets": 6,
            "worker_spec_rhythm_mixed_target_prefill_batches": 3,
            "worker_spec_rhythm_mixed_target_graph_batches": 2,
            **_stable_target_verify_graph_contract(),
            counter: 1,
        }
    )

    with pytest.raises(RuntimeError, match="mixed_target_failures"):
        _require_no_graph_fallback(deltas, require_full_window=True)


@pytest.mark.parametrize(
    ("counter", "value"),
    [
        ("spec_rhythm_stable_target_verify_graph_qualified", 0),
        ("spec_rhythm_stable_target_verify_graph_qualified_capacities", 31),
        ("spec_rhythm_stable_target_verify_graph_qualified_capacity_mask", 0x7FFFFFFF),
        ("spec_rhythm_stable_target_verify_graph_max_qualified_capacity", 31),
        ("spec_rhythm_stable_target_verify_graph_exact_routes", 31),
        ("spec_rhythm_stable_target_verify_numerical_validation_passes", 63),
        ("spec_rhythm_stable_target_verify_numerical_validation_failures", 1),
        ("worker_spec_rhythm_stable_target_verify_graph_batches", 0),
    ],
)
def test_graph_only_benchmark_gate_rejects_incomplete_stable_verify_contract(
    counter,
    value,
):
    deltas = _full_window_graph_deltas()
    deltas[0]["spec_rhythm_mixed_target_graph_enabled"] = 0
    deltas[1].update(
        {
            "spec_rhythm_mixed_target_graph_enabled": 1,
            "spec_rhythm_mixed_target_graph_qualified_buckets": 6,
            "worker_spec_rhythm_mixed_target_prefill_batches": 3,
            "worker_spec_rhythm_mixed_target_graph_batches": 3,
            "worker_spec_rhythm_mixed_target_graph_requests": 7,
            **_stable_target_verify_graph_contract(),
            counter: value,
        }
    )

    with pytest.raises(RuntimeError, match="mixed_target_failures"):
        _require_no_graph_fallback(deltas, require_full_window=True)


def _qualification_worker_metrics(**updates):
    metrics = {
        "rank": 0,
        "is_draft_rank": 0,
        "aclgraph_entries": 1,
        "aclgraph_generic_entries": 1,
        "aclgraph_draft_entries": 0,
        "aclgraph_target_entries": 0,
        "aclgraph_unvalidated_entries": 0,
        "aclgraph_generic_unvalidated_entries": 0,
        "aclgraph_draft_unvalidated_entries": 0,
        "aclgraph_target_unvalidated_entries": 0,
        "aclgraph_disabled_entries": 0,
        "aclgraph_captures": 1,
        "aclgraph_capture_attempts": 1,
        "aclgraph_replays": 2,
        "aclgraph_failed_captures": 0,
        "aclgraph_capacity_fallbacks": 0,
        "aclgraph_shape_fallbacks": 0,
        "aclgraph_runtime_validation_replays": 1,
        "aclgraph_generic_total_calls": 2,
        "aclgraph_generic_capture_replay_calls": 1,
        "aclgraph_generic_replay_calls": 1,
        "aclgraph_generic_eager_fallback_calls": 0,
        "aclgraph_generic_disabled_entry_calls": 0,
        "aclgraph_generic_unclassified_calls": 0,
        "aclgraph_generic_runtime_validation_calls": 1,
        "aclgraph_generic_runtime_validation_failures": 0,
        "aclgraph_generic_changed_input_validation_calls": 1,
        "aclgraph_generic_logical_row_expansion_validation_calls": 0,
    }
    metrics.update(updates)
    return metrics


def test_graph_qualification_fixed_point_requires_a_validation_free_hot_trace():
    before = _qualification_worker_metrics()
    complete, issues = _graph_qualification_fixed_point(
        [before],
        [
            _qualification_worker_metrics(
                aclgraph_replays=3,
                aclgraph_generic_total_calls=3,
                aclgraph_generic_replay_calls=2,
            )
        ],
    )

    assert complete
    assert issues == []


def test_graph_qualification_fixed_point_requires_all_exact_verify_graphs():
    stable_contract = _stable_target_verify_graph_contract()
    before = _qualification_worker_metrics(**stable_contract)
    after = _qualification_worker_metrics(
        **stable_contract,
        aclgraph_replays=3,
        aclgraph_generic_total_calls=3,
        aclgraph_generic_replay_calls=2,
    )

    complete, issues = _graph_qualification_fixed_point([before], [after])
    assert complete
    assert issues == []

    after["spec_rhythm_stable_target_verify_graph_exact_routes"] = 31
    complete, issues = _graph_qualification_fixed_point([before], [after])
    assert not complete
    assert any("stable target-verify graph contract mismatch" in issue for issue in issues)


@pytest.mark.parametrize(
    "updates",
    [
        {
            "aclgraph_entries": 2,
            "aclgraph_generic_entries": 2,
            "aclgraph_captures": 2,
            "aclgraph_capture_attempts": 2,
        },
        {
            "aclgraph_unvalidated_entries": 1,
            "aclgraph_generic_unvalidated_entries": 1,
        },
        {"aclgraph_disabled_entries": 1},
        {
            "aclgraph_shape_fallbacks": 1,
            "aclgraph_generic_eager_fallback_calls": 1,
        },
        {
            "aclgraph_replays": 3,
            "aclgraph_runtime_validation_replays": 2,
            "aclgraph_generic_total_calls": 3,
            "aclgraph_generic_replay_calls": 2,
            "aclgraph_generic_runtime_validation_calls": 2,
            "aclgraph_generic_changed_input_validation_calls": 2,
        },
    ],
)
def test_graph_qualification_fixed_point_rejects_open_or_fallback_state(
    updates,
):
    before = _qualification_worker_metrics()
    after = _qualification_worker_metrics(**updates)

    complete, issues = _graph_qualification_fixed_point([before], [after])

    assert not complete
    assert issues


def test_graph_qualification_prunes_only_a_capture_free_stale_set():
    before = _qualification_worker_metrics()
    stale = _qualification_worker_metrics(
        aclgraph_replays=3,
        aclgraph_generic_total_calls=3,
        aclgraph_generic_replay_calls=2,
        aclgraph_unvalidated_entries=1,
        aclgraph_generic_unvalidated_entries=1,
    )

    assert _graph_qualification_can_prune([before], [stale])
    assert not _graph_qualification_can_prune(
        [before],
        [
            {
                **stale,
                "aclgraph_entries": 2,
                "aclgraph_generic_entries": 2,
                "aclgraph_captures": 2,
                "aclgraph_capture_attempts": 2,
            }
        ],
    )
    assert not _graph_qualification_can_prune(
        [before],
        [{**stale, "aclgraph_disabled_entries": 1}],
    )


def _full_window_graph_deltas(target_graph_kind="target"):
    if target_graph_kind not in ("target", "generic"):
        raise ValueError("Unsupported test target graph kind.")
    common = {
        "aclgraph_captures_delta": 0,
        "aclgraph_capture_attempts_delta": 0,
        "aclgraph_failed_captures_delta": 0,
        "aclgraph_capacity_fallbacks_delta": 0,
        "aclgraph_shape_fallbacks_delta": 0,
        "aclgraph_runtime_validation_replays_delta": 0,
    }
    return [
        {
            **common,
            "rank": 0,
            "is_draft_rank": 1,
            "aclgraph_replays_delta": 5,
            "aclgraph_draft_total_calls_delta": 5,
            "aclgraph_draft_replay_calls_delta": 5,
            "spec_rhythm_linear_draft_full_chain_calls_delta": 5,
            "spec_rhythm_linear_draft_full_chain_capture_replay_calls_delta": 0,
            "spec_rhythm_linear_draft_full_chain_replay_calls_delta": 5,
            "spec_rhythm_linear_draft_full_chain_eager_fallback_calls_delta": 0,
            "spec_rhythm_linear_draft_full_chain_unclassified_calls_delta": 0,
            "spec_rhythm_linear_draft_stepwise_calls_delta": 0,
        },
        {
            **common,
            "rank": 1,
            "is_draft_rank": 0,
            "aclgraph_replays_delta": 5,
            f"aclgraph_{target_graph_kind}_total_calls_delta": 5,
            f"aclgraph_{target_graph_kind}_replay_calls_delta": 5,
        },
    ]


def test_graph_only_benchmark_gate_accepts_audited_full_window_calls():
    _require_no_graph_fallback(
        _full_window_graph_deltas(),
        require_full_window=True,
    )


def test_graph_only_benchmark_gate_accepts_additional_graph_only_draft_calls():
    deltas = _full_window_graph_deltas()
    deltas[0]["aclgraph_replays_delta"] += 2
    deltas[0]["aclgraph_draft_total_calls_delta"] += 2
    deltas[0]["aclgraph_draft_replay_calls_delta"] += 2

    _require_no_graph_fallback(
        deltas,
        require_full_window=True,
    )


def test_graph_only_benchmark_gate_can_explicitly_select_packed_target_backend():
    _require_no_graph_fallback(
        _full_window_graph_deltas("generic"),
        require_full_window=True,
        full_window_target_graph_kind="generic",
    )


@pytest.mark.parametrize("target_graph_kind", ["target", "generic"])
def test_graph_only_benchmark_gate_auto_selects_one_target_backend(
    target_graph_kind,
):
    _require_no_graph_fallback(
        _full_window_graph_deltas(target_graph_kind),
        require_full_window=True,
        full_window_target_graph_kind="auto",
    )


@pytest.mark.parametrize(
    ("rank", "counter"),
    [
        (0, "spec_rhythm_linear_draft_stepwise_calls_delta"),
        (0, "aclgraph_draft_eager_fallback_calls_delta"),
        (1, "aclgraph_target_unclassified_calls_delta"),
        (1, "aclgraph_target_runtime_validation_calls_delta"),
        (1, "aclgraph_target_changed_input_validation_calls_delta"),
        (1, "aclgraph_target_logical_row_expansion_validation_calls_delta"),
    ],
)
def test_graph_only_benchmark_gate_rejects_non_replay_full_window_calls(
    rank,
    counter,
):
    deltas = _full_window_graph_deltas()
    deltas[rank][counter] = 1
    if counter == "aclgraph_draft_eager_fallback_calls_delta":
        deltas[rank]["aclgraph_draft_replay_calls_delta"] -= 1
    elif counter == "aclgraph_target_unclassified_calls_delta":
        deltas[rank]["aclgraph_target_replay_calls_delta"] -= 1
    with pytest.raises(RuntimeError, match="Graph-only benchmark invariant failed"):
        _require_no_graph_fallback(deltas, require_full_window=True)


def test_graph_only_benchmark_gate_rejects_target_calls_not_equal_to_replays():
    deltas = _full_window_graph_deltas()
    deltas[1]["aclgraph_target_replay_calls_delta"] -= 1

    with pytest.raises(RuntimeError, match="Graph-only benchmark invariant failed"):
        _require_no_graph_fallback(deltas, require_full_window=True)


@pytest.mark.parametrize(
    "target_graph_kind",
    ["target", "generic"],
)
def test_graph_only_benchmark_gate_rejects_mixed_target_backends(
    target_graph_kind,
):
    deltas = _full_window_graph_deltas(target_graph_kind)
    other_kind = "generic" if target_graph_kind == "target" else "target"
    deltas[1][f"aclgraph_{other_kind}_total_calls_delta"] = 1
    deltas[1][f"aclgraph_{other_kind}_replay_calls_delta"] = 1
    deltas[1]["aclgraph_replays_delta"] += 1

    with pytest.raises(RuntimeError, match="Graph-only benchmark invariant failed"):
        _require_no_graph_fallback(
            deltas,
            require_full_window=True,
            full_window_target_graph_kind="auto",
        )


def test_graph_only_benchmark_gate_rejects_wrong_explicit_target_backend():
    with pytest.raises(RuntimeError, match="Graph-only benchmark invariant failed"):
        _require_no_graph_fallback(
            _full_window_graph_deltas("target"),
            require_full_window=True,
            full_window_target_graph_kind="generic",
        )


def test_graph_only_benchmark_gate_rejects_measured_runtime_validation():
    deltas = _full_window_graph_deltas()
    deltas[1]["aclgraph_runtime_validation_replays_delta"] = 1

    with pytest.raises(RuntimeError, match="Graph-only benchmark invariant failed"):
        _require_no_graph_fallback(deltas, require_full_window=True)


def test_graph_only_benchmark_gate_rejects_unknown_target_backend():
    with pytest.raises(
        ValueError,
        match="full_window_target_graph_kind must be auto, target, or generic",
    ):
        _require_no_graph_fallback(
            _full_window_graph_deltas(),
            require_full_window=True,
            full_window_target_graph_kind="tree",
        )


@pytest.mark.parametrize("failure", ["fallback", "capture", "inactive"])
def test_graph_only_benchmark_gate_rejects_any_rank_without_graph_contract(failure):
    delta = {
        "rank": 2,
        "aclgraph_captures_delta": 0,
        "aclgraph_replays_delta": 3,
        "aclgraph_failed_captures_delta": 0,
        "aclgraph_capacity_fallbacks_delta": 0,
        "aclgraph_shape_fallbacks_delta": 0,
    }
    if failure == "fallback":
        delta["aclgraph_shape_fallbacks_delta"] = 1
    elif failure == "capture":
        delta["aclgraph_capture_attempts_delta"] = 1
    else:
        delta["aclgraph_replays_delta"] = 0
    with pytest.raises(RuntimeError, match="Graph-only benchmark invariant failed"):
        _require_no_graph_fallback([delta])


def test_greedy_verdict_finds_first_mismatch_in_packed_mixed_windows():
    target_tokens = torch.tensor([5, 10, 21, 32, 40, 51, 61, 71, 81])
    draft_tokens = torch.tensor([5, 10, 20, 30, 40, 50, 60, 70, 80])

    verdict = _build_greedy_verdict(
        target_tokens,
        draft_tokens,
        verification_sizes=[1, 4, 4],
        gamma=4,
    )

    assert verdict.tolist() == [[1, -1], [1, 21], [0, 51]]


def test_layout_greedy_verdict_matches_general_packed_windows():
    target_tokens = torch.tensor([5, 10, 21, 32, 40, 51, 61, 71, 81])
    draft_tokens = torch.tensor([5, 10, 20, 30, 40, 50, 60, 70, 80])
    expected_sizes, dense_positions, correction_positions = _build_verification_layout(
        [1, 4, 4],
        4,
        target_tokens.device,
    )

    verdict = _build_greedy_verdict_with_layout(
        target_tokens,
        draft_tokens,
        expected_sizes,
        dense_positions,
        correction_positions,
        gamma=4,
    )

    assert verdict.tolist() == [[1, -1], [1, 21], [0, 51]]


def test_cpu_greedy_verdict_matches_device_verdict_for_mixed_windows():
    target = torch.tensor([10, 11, 12, 20, 21, 30, 31], dtype=torch.long)
    draft = torch.tensor([10, 99, 12, 20, 22, 30, 31], dtype=torch.long)
    sizes = [3, 2, 2]
    expected = _build_greedy_verdict(target, draft, sizes, gamma=3)
    actual = _build_greedy_verdict_cpu(target, draft, sizes)
    assert actual.tolist() == expected.tolist()


def test_native_aclgraph_padding_preserves_tokens_and_uses_inactive_cache_slots():
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([9, 10]),
        context_lens=torch.tensor([3, 4], dtype=torch.int32),
        block_tables=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
    )

    input_ids, positions, padded = NativeACLGraphRunner._pad_inputs(
        torch.tensor([5, 6]),
        torch.tensor([1, 2]),
        metadata,
        4,
    )

    assert input_ids.tolist() == [5, 6, 0, 0]
    assert positions.tolist() == [1, 2, 0, 0]
    assert padded.slot_mapping.tolist() == [9, 10, -1, -1]
    assert padded.context_lens.tolist() == [3, 4, 1, 1]
    assert padded.block_tables.tolist() == [[1, 2], [3, 4], [1, 2], [1, 2]]


def test_native_aclgraph_eager_greedy_path_includes_lm_head_sampling():
    model = MagicMock()
    hidden_states = torch.randn(2, 4)
    model.return_value = hidden_states
    model.compute_greedy_tokens.return_value = torch.tensor([3, 5])
    runner = NativeACLGraphRunner(model, enabled=False)
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0, 1], dtype=torch.int32),
        context_lens=torch.tensor([1, 1], dtype=torch.int32),
        block_tables=torch.tensor([[0], [1]], dtype=torch.int32),
    )

    tokens = runner.run_greedy(
        torch.tensor([1, 2]),
        torch.tensor([0, 0]),
        metadata,
        vocabulary_size=8,
    )

    assert tokens.tolist() == [3, 5]
    model.compute_greedy_tokens.assert_called_once_with(hidden_states, 8)


def test_native_aclgraph_capacity_uses_eager_for_new_shapes():
    model = MagicMock()
    metadata = SimpleNamespace(
        use_fused_infer_attention=True,
        actual_seq_lengths_q=(1,),
    )
    with patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream"):
        runner = NativeACLGraphRunner(model, enabled=True, max_graph_entries=1)
    runner.entries[("existing", 1)] = MagicMock()
    runner._execute = MagicMock(return_value=torch.tensor([7]))

    output = runner.run_greedy(
        torch.tensor([1]),
        torch.tensor([0]),
        metadata,
        vocabulary_size=8,
    )

    assert output.tolist() == [7]
    assert runner.capacity_fallback_count == 1
    runner._execute.assert_called_once()


def test_native_aclgraph_capacity_counts_failed_capture_attempts():
    model = MagicMock()
    metadata = SimpleNamespace(
        use_fused_infer_attention=True,
        actual_seq_lengths_q=(1,),
    )
    with patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream"):
        runner = NativeACLGraphRunner(model, enabled=True, max_graph_entries=1)
    runner.capture_attempt_count = 1
    runner._capture_budget_used = 1
    runner._execute = MagicMock(return_value=torch.tensor([7]))

    output = runner.run_greedy(
        torch.tensor([1]),
        torch.tensor([0]),
        metadata,
        vocabulary_size=8,
    )

    assert output.tolist() == [7]
    assert runner.entries == {}
    assert runner.capacity_fallback_count == 1
    runner._execute.assert_called_once()


def test_native_aclgraph_releases_target_graph_resources_and_reopens_resident_capacity():
    with patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream"):
        runner = NativeACLGraphRunner(MagicMock(), enabled=True, max_graph_entries=2)
    first = MagicMock()
    second = MagicMock()
    runner.target_entries[("first", (1,))] = first
    runner.target_entries[("second", (2,))] = second
    runner._capture_budget_used = 2
    runner.capture_attempt_count = 7
    runner.capture_count = 5
    runner.replay_count = 20

    with patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.synchronize") as synchronize:
        assert runner.release_target_graph_entries() == 2

    synchronize.assert_called_once_with()
    first.graph.reset.assert_called_once_with()
    second.graph.reset.assert_called_once_with()
    assert not runner.target_entries
    assert runner._capture_budget_used == 0
    assert runner.capture_attempt_count == 7
    assert runner.capture_count == 5
    assert runner.replay_count == 20


def test_native_aclgraph_qualification_status_and_seal_require_every_entry():
    with patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream"):
        runner = NativeACLGraphRunner(MagicMock(), enabled=True)
    runner.entries[("generic", 1)] = SimpleNamespace(runtime_validated=True)
    runner.draft_entries[("draft", 1)] = SimpleNamespace(runtime_validated=False)
    runner.target_entries[("target", (1,))] = SimpleNamespace(runtime_validated=True)

    status = runner.graph_qualification_status()

    assert status == {
        "generic_entries": 1,
        "draft_entries": 1,
        "target_entries": 1,
        "generic_unvalidated_entries": 0,
        "draft_unvalidated_entries": 1,
        "target_unvalidated_entries": 0,
        "unvalidated_entries": 1,
        "disabled_entries": 0,
        "generic_pruned_unvalidated_entries": 0,
        "draft_pruned_unvalidated_entries": 0,
        "target_pruned_unvalidated_entries": 0,
        "pruned_unvalidated_entries": 0,
        "shared_memory_pool": 0,
        "sealed": 0,
    }
    with pytest.raises(RuntimeError, match="unqualified resident entries"):
        runner.seal_graph_cache()

    runner.draft_entries[("draft", 1)].runtime_validated = True
    sealed = runner.seal_graph_cache()
    assert sealed["sealed"] == 1
    assert runner.unseal_graph_cache()["sealed"] == 0


def test_native_aclgraph_seal_rejects_disabled_entries():
    with patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream"):
        runner = NativeACLGraphRunner(MagicMock(), enabled=True)
    runner.disabled_entry_keys.add(("greedy:8", 1))

    with pytest.raises(RuntimeError, match="containing disabled entries"):
        runner.seal_graph_cache()


def test_native_aclgraph_prunes_only_unvalidated_entries_and_reopens_capacity():
    with patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream"):
        runner = NativeACLGraphRunner(MagicMock(), enabled=True)
    retained = SimpleNamespace(runtime_validated=True, graph=MagicMock())
    generic_stale = SimpleNamespace(runtime_validated=False, graph=MagicMock())
    draft_stale = SimpleNamespace(runtime_validated=False, graph=MagicMock())
    target_stale = SimpleNamespace(runtime_validated=False, graph=MagicMock())
    runner.entries[("retained", 1)] = retained
    runner.entries[("generic-stale", 1)] = generic_stale
    runner.draft_entries[("draft-stale", 1)] = draft_stale
    runner.target_entries[("target-stale", (1,))] = target_stale
    runner._capture_budget_used = 4

    with patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.synchronize") as synchronize:
        status = runner.prune_unvalidated_graph_entries()

    synchronize.assert_called_once_with()
    retained.graph.reset.assert_not_called()
    for entry in (generic_stale, draft_stale, target_stale):
        entry.graph.reset.assert_called_once_with()
    assert list(runner.entries) == [("retained", 1)]
    assert not runner.draft_entries
    assert not runner.target_entries
    assert runner._capture_budget_used == 1
    assert status["unvalidated_entries"] == 0
    assert status["pruned_unvalidated_entries"] == 3
    assert status["generic_pruned_unvalidated_entries"] == 1
    assert status["draft_pruned_unvalidated_entries"] == 1
    assert status["target_pruned_unvalidated_entries"] == 1


def test_sealed_native_aclgraph_refuses_missing_generic_entry_before_capture():
    model = MagicMock()
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0], dtype=torch.int32),
        context_lens=torch.tensor([1], dtype=torch.int32),
        block_tables=torch.tensor([[0]], dtype=torch.int32),
        actual_seq_lengths_q=(),
        sequence_lens=(),
        request_block_tables=None,
        attention_mask=None,
        use_fused_infer_attention=False,
    )
    with patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream"):
        runner = NativeACLGraphRunner(model, enabled=True)
    runner.graph_cache_sealed = True

    with pytest.raises(RuntimeError, match="reason=missing_entry"):
        runner.run_greedy(
            torch.tensor([1]),
            torch.tensor([0]),
            metadata,
            vocabulary_size=8,
        )

    assert runner.capture_attempt_count == 0
    assert runner.execution_counters["generic"]["total_calls"] == 0
    model.assert_not_called()


def test_sealed_native_aclgraph_refuses_generic_logical_row_expansion():
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int32),
        context_lens=torch.tensor([1, 1, 1], dtype=torch.int32),
        block_tables=torch.zeros((3, 1), dtype=torch.int32),
        actual_seq_lengths_q=(),
        sequence_lens=(),
        request_block_tables=None,
        attention_mask=None,
        use_fused_infer_attention=False,
    )
    with patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream"):
        runner = NativeACLGraphRunner(MagicMock(), enabled=True)
    runner.entries[("greedy:8", 4)] = SimpleNamespace(
        input_ids=torch.zeros(4, dtype=torch.long),
        positions=torch.zeros(4, dtype=torch.long),
        slot_mapping=torch.full((4,), -1, dtype=torch.int32),
        context_lens=torch.zeros(4, dtype=torch.int32),
        block_tables=torch.zeros((4, 1), dtype=torch.int32),
        request_block_tables=None,
        attention_mask=None,
        actual_seq_lengths_q=(),
        sequence_lens=(),
        runtime_validated=True,
        validated_real_row_count=2,
    )
    runner.graph_cache_sealed = True

    with pytest.raises(RuntimeError, match="reason=logical_row_expansion"):
        runner.run_greedy(
            torch.tensor([1, 2, 3]),
            torch.tensor([0, 0, 0]),
            metadata,
            vocabulary_size=8,
        )


def test_native_aclgraph_replay_orders_task_updates_without_host_sync():
    model = MagicMock()
    metadata = SimpleNamespace(
        use_fused_infer_attention=True,
        actual_seq_lengths_q=(1,),
    )
    update_stream = MagicMock()
    current_stream = MagicMock()
    with (
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream",
            return_value=update_stream,
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.current_stream",
            return_value=current_stream,
        ),
    ):
        runner = NativeACLGraphRunner(model, enabled=True)
        entry = MagicMock(
            output=torch.tensor([7]),
            runtime_validated=True,
        )
        runner.entries[("greedy:8|fia:1", 1)] = entry
        runner._copy_inputs = MagicMock()
        runner._update_attention_tasks = MagicMock()

        output = runner.run_greedy(
            torch.tensor([1]),
            torch.tensor([0]),
            metadata,
            vocabulary_size=8,
        )

    assert output.tolist() == [7]
    update_stream.wait_stream.assert_called_once_with(current_stream)
    current_stream.synchronize.assert_not_called()
    runner._update_attention_tasks.assert_called_once_with(entry)
    entry.graph.replay.assert_called_once_with()


def test_native_aclgraph_first_changed_generic_input_is_runtime_validated():
    model = MagicMock()
    update_stream = MagicMock()
    current_stream = MagicMock()
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0], dtype=torch.int32),
        context_lens=torch.tensor([1], dtype=torch.int32),
        block_tables=torch.tensor([[0]], dtype=torch.int32),
        actual_seq_lengths_q=(),
        sequence_lens=(),
        request_block_tables=None,
        attention_mask=None,
        use_fused_infer_attention=False,
    )
    with patch(
        "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream",
        return_value=update_stream,
    ):
        runner = NativeACLGraphRunner(model, enabled=True)
    entry = SimpleNamespace(
        input_ids=torch.tensor([1]),
        positions=torch.tensor([0]),
        slot_mapping=metadata.slot_mapping.clone(),
        context_lens=metadata.context_lens.clone(),
        block_tables=metadata.block_tables.clone(),
        request_block_tables=None,
        attention_mask=None,
        actual_seq_lengths_q=(),
        sequence_lens=(),
        graph=MagicMock(),
        output=torch.tensor([7]),
        # Ordinary generic PA/FIA entries must retain their captured task
        # metadata.  Only an explicitly marked device-position PA graph is
        # allowed to be taskless.
        tasks=[SimpleNamespace(event=MagicMock())],
        runtime_validated=False,
        validated_real_row_count=1,
    )
    runner.entries[("greedy:8", 1)] = entry
    runner._execute = MagicMock(return_value=torch.tensor([7]))
    runner._update_attention_tasks = MagicMock()
    with (
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.current_stream",
            return_value=current_stream,
        ),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.synchronize"),
    ):
        output = runner.run_greedy(
            torch.tensor([2]),
            torch.tensor([0]),
            metadata,
            vocabulary_size=8,
        )

    assert output.tolist() == [7]
    assert entry.runtime_validated
    assert runner.runtime_validation_replay_count == 1
    counters = runner.execution_counters["generic"]
    assert counters["total_calls"] == 1
    assert counters["replay_calls"] == 1
    assert counters["runtime_validation_calls"] == 1
    assert counters["changed_input_validation_calls"] == 1


def test_generic_graph_capture_tracks_logical_rows_not_bucket_size():
    update_stream = MagicMock()
    current_stream = MagicMock()
    with patch(
        "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream",
        return_value=update_stream,
    ):
        runner = NativeACLGraphRunner(MagicMock(), enabled=True)
    runner._execute = MagicMock(side_effect=[torch.tensor([7, 8, 0, 0]), torch.tensor([7, 8, 0, 0])])
    runner._update_attention_tasks = MagicMock()
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0, 1, -1, -1], dtype=torch.int32),
        context_lens=torch.tensor([1, 1, 0, 0], dtype=torch.int32),
        block_tables=torch.zeros((4, 1), dtype=torch.int32),
        actual_seq_lengths_q=(),
        sequence_lens=(),
        request_block_tables=None,
        attention_mask=None,
        use_fused_infer_attention=False,
    )
    with (
        patch("torch.npu.NPUGraph", return_value=MagicMock()),
        patch("torch.npu.graph", return_value=MagicMock()),
        patch("torch.npu.current_stream", return_value=current_stream),
        patch("torch.npu.synchronize"),
    ):
        runner._capture(
            ("greedy:8", 4),
            torch.tensor([1, 2, 0, 0]),
            torch.tensor([0, 0, 0, 0]),
            metadata,
            None,
            valid_row_count=2,
        )

    entry = runner.entries[("greedy:8", 4)]
    assert entry.validated_real_row_count == 2
    assert not entry.runtime_validated


def test_native_aclgraph_greedy_validation_requires_exact_integer_output():
    reference = torch.arange(128)
    sparse = reference.clone()
    sparse[:7] = -1
    dense = sparse.clone()
    dense[7] = -1

    assert NativeACLGraphRunner._outputs_match(reference.clone(), reference)
    assert not NativeACLGraphRunner._outputs_match(sparse, reference)
    assert not NativeACLGraphRunner._outputs_match(dense, reference)


def test_native_aclgraph_greedy_validation_rejects_any_small_batch_mismatch():
    reference = torch.arange(16)
    one_mismatch = reference.clone()
    one_mismatch[0] = -1
    two_mismatches = one_mismatch.clone()
    two_mismatches[1] = -1

    assert not NativeACLGraphRunner._outputs_match(one_mismatch, reference)
    assert not NativeACLGraphRunner._outputs_match(two_mismatches, reference)


def test_draft_aclgraph_validation_ignores_only_consistent_padding_rows():
    reference = torch.arange(16, dtype=torch.long).reshape(4, 4)
    graph = reference.clone()
    graph[2:] = -1
    metadatas = [SimpleNamespace(slot_mapping=torch.tensor([10, 11, -1, -1])) for _ in range(4)]

    assert NativeACLGraphRunner._draft_outputs_match(
        graph,
        reference,
        metadatas,
        valid_row_count=2,
    )

    graph[1, 3] = -1
    assert not NativeACLGraphRunner._draft_outputs_match(
        graph,
        reference,
        metadatas,
        valid_row_count=2,
    )


@pytest.mark.parametrize(
    "bad_slot_mapping",
    [
        # A true logical row is consistently lost in every proposal step.
        torch.tensor([10, -1, -1, -1]),
        # The valid set contains a hole and is not the declared prefix.
        torch.tensor([10, -1, 12, -1]),
        # A graph-padding row is incorrectly materialized.
        torch.tensor([10, 11, 12, -1]),
    ],
)
def test_draft_aclgraph_validation_rejects_noncanonical_slot_prefix(
    bad_slot_mapping,
):
    reference = torch.arange(16, dtype=torch.long).reshape(4, 4)
    metadatas = [SimpleNamespace(slot_mapping=bad_slot_mapping) for _ in range(4)]

    assert not NativeACLGraphRunner._draft_outputs_match(
        reference,
        reference,
        metadatas,
        valid_row_count=2,
    )


def test_draft_aclgraph_validation_rejects_any_real_row_difference():
    reference = torch.arange(16, dtype=torch.long).reshape(4, 4)
    graph = reference.clone()
    graph[0, 0] = -1
    metadatas = [SimpleNamespace(slot_mapping=torch.tensor([10, 11, -1, -1])) for _ in range(4)]

    assert not NativeACLGraphRunner._draft_outputs_match(
        graph,
        reference,
        metadatas,
        valid_row_count=2,
    )


def test_native_aclgraph_skips_dynamic_fia_tail_without_spending_capture_budget():
    model = MagicMock()
    metadata = SimpleNamespace(
        use_fused_infer_attention=True,
        actual_seq_lengths_q=(1, 2, 3, 4, 5, 6, 7),
    )
    with patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream"):
        runner = NativeACLGraphRunner(model, enabled=True, max_graph_entries=1)
    runner.set_expected_fia_batch_size(8)
    runner._execute = MagicMock(return_value=torch.tensor([7]))

    output = runner.run_greedy(
        torch.tensor([1]),
        torch.tensor([0]),
        metadata,
        vocabulary_size=8,
    )

    assert output.tolist() == [7]
    assert runner.capture_attempt_count == 0
    assert runner.shape_fallback_count == 1
    runner._execute.assert_called_once()


@pytest.mark.parametrize(
    ("actual_seq_lengths_q", "expected"),
    [
        ((1, 2, 3, 4), True),
        ((4, 8, 12, 16), True),
        ((1, 5, 6, 10), True),
        ((1, 5, 5, 9), False),
    ],
)
def test_native_aclgraph_reuses_full_batch_fia_segments(actual_seq_lengths_q, expected):
    with patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream"):
        runner = NativeACLGraphRunner(MagicMock(), enabled=True)
    runner.set_expected_fia_batch_size(4)

    assert runner._is_reusable_fia_shape(actual_seq_lengths_q) is expected


def test_native_aclgraph_rejects_nonpositive_entry_capacity():
    with pytest.raises(ValueError, match="max_graph_entries"):
        NativeACLGraphRunner(MagicMock(), enabled=False, max_graph_entries=0)


def test_native_weight_loader_uses_safe_open_keys_api(tmp_path):
    weight_file = tmp_path / "model.safetensors"
    weight_file.touch()
    checkpoint = MagicMock()
    checkpoint.keys.return_value = []
    model = MagicMock()
    model.packed_modules_mapping = {}

    with patch("vllm_ascend.spec_decode.pearl.native_model.safe_open") as safe_open:
        safe_open.return_value.__enter__.return_value = checkpoint
        load_native_qwen2_weights(model, str(tmp_path))

    checkpoint.keys.assert_called_once_with()


def test_native_nz_conversion_keeps_tied_lm_head_nd_but_converts_transformer():
    model = MagicMock()
    model.config.tie_word_embeddings = True
    lm_head = MagicMock(spec=NativeLMHead)
    lm_head.bias = None
    embedding_weight = MagicMock()
    embedding_weight.device.type = "npu"
    lm_head.weight = embedding_weight
    model.lm_head = lm_head
    model.embed_tokens.weight = embedding_weight
    transformer = MagicMock(spec=NativeRowLinear)
    transformer.bias = None
    transformer.weight = MagicMock()
    transformer.weight.device.type = "npu"
    transformer_weight = transformer.weight.data
    model.modules.return_value = [lm_head, transformer]
    converted = MagicMock()

    with (
        patch(
            "vllm_ascend.spec_decode.pearl.native_model.ascend_envs.VLLM_ASCEND_ENABLE_NZ",
            2,
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_model.torch_npu.npu_format_cast",
            return_value=converted,
        ) as format_cast,
    ):
        _maybe_convert_linear_weights_to_nz(model)

    format_cast.assert_called_once_with(transformer_weight, 29)
    assert model.embed_tokens.weight is embedding_weight
    assert lm_head.weight is embedding_weight
    assert transformer.weight.data is converted


def test_native_nz_conversion_keeps_biased_linear_weights_in_nd():
    model = MagicMock()
    model.config.tie_word_embeddings = False
    biased = MagicMock(spec=NativeColumnLinear)
    biased.bias = MagicMock()
    biased.weight = MagicMock()
    biased.weight.device.type = "npu"
    biased_weight = biased.weight.data
    unbiased = MagicMock(spec=NativeColumnLinear)
    unbiased.bias = None
    unbiased.weight = MagicMock()
    unbiased.weight.device.type = "npu"
    unbiased_weight = unbiased.weight.data
    model.modules.return_value = [biased, unbiased]
    converted = MagicMock()

    with (
        patch(
            "vllm_ascend.spec_decode.pearl.native_model.ascend_envs.VLLM_ASCEND_ENABLE_NZ",
            2,
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_model.torch_npu.npu_format_cast",
            return_value=converted,
        ) as format_cast,
    ):
        _maybe_convert_linear_weights_to_nz(model)

    format_cast.assert_called_once_with(unbiased_weight, 29)
    assert biased.weight.data is biased_weight
    assert unbiased.weight.data is converted


def test_native_nz_conversion_prefers_model_local_mode():
    model = MagicMock()
    model.config.tie_word_embeddings = False
    model.config.pearl_weight_nz_mode = 2
    linear = MagicMock(spec=NativeColumnLinear)
    linear.bias = None
    linear.weight = MagicMock()
    linear.weight.device.type = "npu"
    original = linear.weight.data
    model.modules.return_value = [linear]
    converted = MagicMock()

    with (
        patch(
            "vllm_ascend.spec_decode.pearl.native_model.ascend_envs.VLLM_ASCEND_ENABLE_NZ",
            1,
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_model.torch_npu.npu_format_cast",
            return_value=converted,
        ) as format_cast,
    ):
        _maybe_convert_linear_weights_to_nz(model)

    format_cast.assert_called_once_with(original, 29)
    assert linear.weight.data is converted


def test_native_lm_head_nz_mode_unties_loaded_embedding_values():
    model = MagicMock()
    model.config.pearl_weight_nz_mode = 9
    tied_weight = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4))
    model.embed_tokens.weight = tied_weight
    model.lm_head.weight = tied_weight

    _maybe_untie_lm_head_for_nz(model)

    assert model.lm_head.weight is not model.embed_tokens.weight
    assert not model.lm_head.weight.requires_grad
    torch.testing.assert_close(model.lm_head.weight, model.embed_tokens.weight)


def test_native_lm_head_nz_mode_converts_only_private_output_weight():
    model = MagicMock()
    model.config.pearl_weight_nz_mode = 9
    lm_head = MagicMock(spec=NativeLMHead)
    lm_head.bias = None
    lm_head.weight = MagicMock()
    lm_head.weight.device.type = "npu"
    model.lm_head = lm_head
    model.embed_tokens.weight = MagicMock()
    model.named_modules.return_value = [("lm_head", lm_head)]
    original = lm_head.weight.data
    converted = MagicMock()

    with patch(
        "vllm_ascend.spec_decode.pearl.native_model.torch_npu.npu_format_cast",
        return_value=converted,
    ) as format_cast:
        _maybe_convert_linear_weights_to_nz(model)

    format_cast.assert_called_once_with(original, 29)
    assert lm_head.weight.data is converted


def test_native_selective_target_nz_converts_only_ffn_projections():
    model = MagicMock()
    model.config.tie_word_embeddings = False
    model.config.pearl_weight_nz_mode = 3
    modules = {}
    for name in (
        "model.layers.0.self_attn.qkv_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_up_proj",
        "model.layers.0.mlp.down_proj",
        "lm_head",
    ):
        module = MagicMock(spec=NativeColumnLinear)
        module.bias = None
        module.weight = MagicMock()
        module.weight.device.type = "npu"
        modules[name] = module
    model.named_modules.return_value = list(modules.items())
    gate_up_weight = modules["model.layers.0.mlp.gate_up_proj"].weight.data
    down_weight = modules["model.layers.0.mlp.down_proj"].weight.data
    converted = [MagicMock(), MagicMock()]

    with patch(
        "vllm_ascend.spec_decode.pearl.native_model.torch_npu.npu_format_cast",
        side_effect=converted,
    ) as format_cast:
        _maybe_convert_linear_weights_to_nz(model)

    assert format_cast.call_count == 2
    assert format_cast.call_args_list[0].args == (
        gate_up_weight,
        29,
    )
    assert format_cast.call_args_list[1].args == (
        down_weight,
        29,
    )
    assert modules["model.layers.0.mlp.gate_up_proj"].weight.data is converted[0]
    assert modules["model.layers.0.mlp.down_proj"].weight.data is converted[1]


@pytest.mark.parametrize(
    ("mode", "selected"),
    (
        (4, "model.layers.0.mlp.down_proj"),
        (5, "model.layers.0.mlp.gate_up_proj"),
    ),
)
def test_native_split_target_nz_converts_one_ffn_projection(mode, selected):
    model = MagicMock()
    model.config.tie_word_embeddings = False
    model.config.pearl_weight_nz_mode = mode
    modules = {}
    for name in (
        "model.layers.0.self_attn.qkv_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_up_proj",
        "model.layers.0.mlp.down_proj",
        "lm_head",
    ):
        module = MagicMock(spec=NativeColumnLinear)
        module.bias = None
        module.weight = MagicMock()
        module.weight.device.type = "npu"
        modules[name] = module
    model.named_modules.return_value = list(modules.items())
    originals = {name: module.weight.data for name, module in modules.items()}
    converted = MagicMock()

    with patch(
        "vllm_ascend.spec_decode.pearl.native_model.torch_npu.npu_format_cast",
        return_value=converted,
    ) as format_cast:
        _maybe_convert_linear_weights_to_nz(model)

    format_cast.assert_called_once_with(originals[selected], 29)
    for name, module in modules.items():
        assert module.weight.data is (converted if name == selected else originals[name])


def test_native_tp3_hybrid_nz_excludes_gate_up_projection():
    model = MagicMock()
    model.config.tie_word_embeddings = False
    model.config.pearl_weight_nz_mode = 8
    modules = {}
    for name in (
        "model.layers.0.self_attn.qkv_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_up_proj",
        "model.layers.0.mlp.down_proj",
        "lm_head",
    ):
        module = MagicMock(spec=NativeColumnLinear)
        module.bias = None
        module.weight = MagicMock()
        module.weight.device.type = "npu"
        modules[name] = module
    model.named_modules.return_value = list(modules.items())
    originals = {name: module.weight.data for name, module in modules.items()}
    converted = [MagicMock() for _ in range(4)]

    with patch(
        "vllm_ascend.spec_decode.pearl.native_model.torch_npu.npu_format_cast",
        side_effect=converted,
    ) as format_cast:
        _maybe_convert_linear_weights_to_nz(model)

    assert format_cast.call_count == 4
    assert modules["model.layers.0.mlp.gate_up_proj"].weight.data is originals["model.layers.0.mlp.gate_up_proj"]
    selected = (
        "model.layers.0.self_attn.qkv_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.down_proj",
        "lm_head",
    )
    for name, value in zip(selected, converted):
        assert modules[name].weight.data is value


def test_native_tp3_small_m_nz_converts_qkv_and_down_only():
    model = MagicMock()
    model.config.pearl_weight_nz_mode = 10
    modules = {}
    for name in (
        "model.layers.0.self_attn.qkv_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_up_proj",
        "model.layers.0.mlp.down_proj",
        "lm_head",
    ):
        module = MagicMock(spec=NativeColumnLinear)
        module.bias = None
        module.weight = MagicMock()
        module.weight.device.type = "npu"
        modules[name] = module
    model.lm_head = modules["lm_head"]
    model.embed_tokens.weight = MagicMock()
    model.named_modules.return_value = list(modules.items())
    originals = {name: module.weight.data for name, module in modules.items()}
    converted = [MagicMock(), MagicMock()]

    with patch(
        "vllm_ascend.spec_decode.pearl.native_model.torch_npu.npu_format_cast",
        side_effect=converted,
    ) as format_cast:
        _maybe_convert_linear_weights_to_nz(model)

    assert format_cast.call_count == 2
    assert modules["model.layers.0.self_attn.qkv_proj"].weight.data is converted[0]
    assert modules["model.layers.0.mlp.down_proj"].weight.data is converted[1]
    for name in (
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_up_proj",
        "lm_head",
    ):
        assert modules[name].weight.data is originals[name]


def test_native_tp3_large_m_nz_retains_nd_down_and_stores_nz_copy():
    model = MagicMock()
    model.config.pearl_weight_nz_mode = 11
    down = MagicMock(spec=NativeRowLinear)
    down.bias = None
    down.weight = MagicMock()
    down.weight.device.type = "npu"
    original = down.weight.data
    model.lm_head = MagicMock()
    model.embed_tokens.weight = MagicMock()
    model.named_modules.return_value = [("model.layers.0.mlp.down_proj", down)]
    converted = MagicMock()

    with patch(
        "vllm_ascend.spec_decode.pearl.native_model.torch_npu.npu_format_cast",
        return_value=converted,
    ) as format_cast:
        _maybe_convert_linear_weights_to_nz(model)

    format_cast.assert_called_once_with(original, 29)
    assert down.weight.data is original
    assert down.large_m_nz_weight is converted


@pytest.mark.parametrize(("mode", "selected_layer"), ((6, 0), (7, 1)))
def test_native_parity_target_nz_converts_down_projection_on_selected_layers(mode, selected_layer):
    model = MagicMock()
    model.config.tie_word_embeddings = False
    model.config.pearl_weight_nz_mode = mode
    modules = {}
    for layer_index in range(2):
        for projection in ("gate_up_proj", "down_proj"):
            name = f"model.layers.{layer_index}.mlp.{projection}"
            module = MagicMock(spec=NativeColumnLinear)
            module.bias = None
            module.weight = MagicMock()
            module.weight.device.type = "npu"
            modules[name] = module
    model.named_modules.return_value = list(modules.items())
    originals = {name: module.weight.data for name, module in modules.items()}
    converted = MagicMock()

    with patch(
        "vllm_ascend.spec_decode.pearl.native_model.torch_npu.npu_format_cast",
        return_value=converted,
    ) as format_cast:
        _maybe_convert_linear_weights_to_nz(model)

    selected = f"model.layers.{selected_layer}.mlp.down_proj"
    format_cast.assert_called_once_with(originals[selected], 29)
    for name, module in modules.items():
        assert module.weight.data is (converted if name == selected else originals[name])
