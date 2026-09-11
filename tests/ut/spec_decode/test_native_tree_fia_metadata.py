# SPDX-License-Identifier: Apache-2.0
"""Independent packing/cache invariants for native FULL-mask FIA."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tests.ut.spec_decode.test_spec_rhythm_kv_preflight import _fixture, _preflight
from tests.ut.spec_decode.test_tree_draft_batch import _engine
from vllm_ascend.spec_decode.pearl.native_engine import NativePearlEngine
from vllm_ascend.spec_decode.pearl.native_model import NativeAttention, make_tree_fia_mask
from vllm_ascend.spec_decode.pearl.tree import (
    build_tree_speculation_plan,
    pack_selected_tree_plan,
    verify_greedy_tree_batch,
)


def test_heterogeneous_full_mask_pads_only_mask_not_queries():
    engine = _engine()
    plans = [pack_selected_tree_plan(build_tree_speculation_plan(2, 2, 3, 32), range(count)) for count in (1, 4)]
    tokens, _, metadata = engine.model.make_tree_attention_metadata(
        plans, [7, 8], [[9], [10, 11, 12, 13]], engine.cache_block_tables
    )
    assert tokens.numel() == 7
    assert metadata.actual_seq_lengths_q == (2, 7)
    assert metadata.tree_attention_mask.shape == (2, 1, 5, 32)
    assert metadata.tree_attention_mask[0, 0, 2:].all()
    assert torch.equal(metadata.tree_attention_mask[0, 0, :2], plans[0].attention_mask)
    assert torch.equal(metadata.tree_attention_mask[1, 0], plans[1].attention_mask)
    assert metadata.attention_mask.shape == (7, 32)
    assert metadata.tree_attention and not metadata.use_fused_infer_attention


def test_multi_query_draft_level_has_one_cumulative_request():
    engine = _engine()
    plan = build_tree_speculation_plan(2, 2, 3, 32)
    tokens, _, metadata = engine.model.make_tree_level_attention_metadata(
        plan, 1, [0, 2], [7, 8], engine.cache_block_tables
    )
    assert tokens.numel() == 2
    assert metadata.actual_seq_lengths_q == (2,)
    assert len(metadata.sequence_lens) == 1
    assert torch.equal(metadata.request_block_tables, engine.cache_block_tables[1:2])
    assert torch.equal(metadata.tree_attention_mask[0, 0], plan.attention_mask[[1, 3]])


def test_unified_q1_draft_has_one_full_mask_per_request():
    engine = _engine()
    plan = build_tree_speculation_plan(2, 2, 3, 32)
    _, _, metadata = engine._pack_tree_draft_level([(plan, 0, -1, 7), (plan, 1, 0, 8)])
    assert metadata.actual_seq_lengths_q == (1, 2)
    assert metadata.tree_attention_mask.shape == (2, 1, 1, 32)
    assert torch.equal(metadata.tree_attention_mask[:, 0, 0], metadata.attention_mask)
    assert torch.equal(metadata.request_block_tables, engine.cache_block_tables)


def test_host_verdict_matches_tensor_verifier_for_heterogeneous_trees():
    base = build_tree_speculation_plan(2, 2, 3, 32)
    plans = [
        pack_selected_tree_plan(base, [0, 1]),
        pack_selected_tree_plan(base, [0, 2]),
    ]
    drafts = [[7, 8], [10, 11]]
    targets = torch.tensor([7, 8, 9, 11, 12, 13])
    expected = verify_greedy_tree_batch(
        torch.tensor([7, 8, 10, 11]),
        torch.cat([plan.parent_indices for plan in plans]),
        [2, 2],
        targets,
        torch.tensor([9, 13]),
        max_depth=2,
    )

    actual = NativePearlEngine._verify_tree_outputs_host(
        drafts, plans, targets, max_depth=2, output_device=torch.device("cpu")
    )

    assert torch.equal(actual.token_ids, expected.token_ids)
    assert torch.equal(actual.accepted_node_indices, expected.accepted_node_indices.to(torch.long))


@pytest.mark.parametrize("rows", [[], [torch.ones(0, 4)], [torch.ones(2)], [torch.ones(1, 4), torch.ones(2, 5)]])
def test_invalid_full_mask_layout_fails_before_operator(rows):
    with pytest.raises(ValueError):
        make_tree_fia_mask(rows)


def test_cache_holes_are_finite_and_sticky_fault_resets_on_reallocation():
    attention = SimpleNamespace(qkv_proj=SimpleNamespace(weight=torch.ones(1)), num_kv_heads=1, head_dim=2)
    NativeAttention.configure_cache(attention, 16, 2, block_size=4, use_paged_attention=True)
    assert not attention.key_cache.any()
    assert not attention.value_cache.any()
    assert not attention.tree_cache_nonfinite
    attention.tree_cache_nonfinite.fill_(True)
    NativeAttention.configure_cache(attention, 16, 2, block_size=4, use_paged_attention=True)
    assert not attention.tree_cache_nonfinite


@pytest.mark.parametrize("draft", [False, True])
def test_nonfinite_attention_flag_is_preserved_for_the_shared_commit_vote(draft):
    fixture = _fixture(draft=draft)
    engine, controller, *_ = fixture
    for layer in engine.model.layers:
        layer.self_attn.tree_cache_nonfinite = torch.tensor(False)
    engine.model.layers[-1].self_attn.tree_cache_nonfinite.fill_(True)
    ready_before = dict(controller.ready)
    # Local structural preflight is intentionally non-mutating and does not
    # synchronize device flags. The production loop folds this sticky bit
    # into its following world all-reduce commit vote.
    _preflight(fixture)
    assert engine._spec_rhythm_nonfinite_flag()
    assert controller.ready == ready_before


def test_fia_dispatch_uses_full_mask_while_dense_reference_is_preserved():
    engine = _engine()
    plan = build_tree_speculation_plan(2, 2, 3, 32)
    _, _, metadata = engine.model.make_tree_attention_metadata([plan], [7], [[8, 9, 10, 11]], engine.cache_block_tables)
    attention = SimpleNamespace(
        key_cache=torch.zeros(8, 4, 1, 2),
        value_cache=torch.zeros(8, 4, 1, 2),
        block_size=4,
        num_kv_heads=1,
        num_heads=2,
        scale=0.5,
    )
    with patch("vllm_ascend.spec_decode.pearl.native_model.run_native_fused_infer_attention") as run:
        NativeAttention._fused_infer_attention(attention, torch.zeros(5, 2, 2), metadata)
    assert run.call_args.kwargs["attention_mask"] is metadata.tree_attention_mask
    assert run.call_args.kwargs["tree_attention"] is True
    assert metadata.attention_mask.ndim == 2
