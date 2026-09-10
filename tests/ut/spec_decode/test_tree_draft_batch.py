# SPDX-License-Identifier: Apache-2.0
"""Structural regressions for unified tree drafting and eager KV isolation."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm_ascend.spec_decode.pearl.native_engine import NativePearlEngine
from vllm_ascend.spec_decode.pearl.native_model import NativeQwen2ForCausalLM
from vllm_ascend.spec_decode.pearl.tree import build_tree_speculation_plan, tree_primary_path


class _DraftModel:
    make_tree_level_attention_metadata = NativeQwen2ForCausalLM.make_tree_level_attention_metadata
    make_tree_attention_metadata = NativeQwen2ForCausalLM.make_tree_attention_metadata

    def __init__(self):
        self.embed_tokens = SimpleNamespace(weight=torch.empty(1))
        self.layers = [SimpleNamespace(self_attn=SimpleNamespace(block_size=4))]
        self.max_model_len = 32
        self.calls = []

    def __call__(self, input_ids, positions, metadata):
        self.calls.append((input_ids.clone(), positions.clone(), metadata))
        return input_ids.float().unsqueeze(1)

    def compute_logits(self, hidden):
        logits = torch.zeros((hidden.shape[0], 32))
        tokens = (hidden[:, 0].to(torch.long) + 1) % 32
        logits.scatter_(1, tokens.unsqueeze(1), 10)
        logits.scatter_(1, ((tokens + 1) % 32).unsqueeze(1), 5)
        return logits


def _engine(requests=2):
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.device = torch.device("cpu")
    engine.draft_vocab_size = 32
    engine.config = SimpleNamespace(enforce_eager=True)
    engine.cache_block_tables = torch.arange(requests * 8, dtype=torch.int32).view(requests, 8)
    engine.cache_allocation = SimpleNamespace(block_tables=engine.cache_block_tables)
    engine._ensure_cache_capacity = MagicMock()
    engine.model = _DraftModel()
    return engine


def test_multiple_normal_trees_share_one_model_invocation_per_depth():
    engine = _engine()
    plans = [build_tree_speculation_plan(2, 2, prefix_len=2, max_model_len=32) for _ in range(2)]
    output = engine.draft_tree_forward(plans, [5, 9], [0, 1])
    assert [call[0].numel() for call in engine.model.calls] == [2, 2, 10]
    assert output["model_calls"] == 3
    assert output["graph_calls"] == 0
    assert output["draft_token_ids"] == [[6, 7, 7, 8], [10, 11, 11, 12]]
    assert [row.numel() for row in output["node_confidences"]] == [4, 4]


def test_normal_and_eager_share_levels_but_eager_scratch_does_not_overwrite_parent():
    engine = _engine()
    parent = build_tree_speculation_plan(2, 2, prefix_len=2, max_model_len=32)
    normal = build_tree_speculation_plan(2, 2, prefix_len=2, max_model_len=32)
    eager = build_tree_speculation_plan(2, 2, prefix_len=5, max_model_len=32)
    output = engine.draft_tree_forward(
        [normal, eager], [5, 99], [0, 1], eager_parent_sources={1: (parent, [10, 11, 12, 13])}
    )
    assert [call[0].numel() for call in engine.model.calls] == [1, 2, 2, 10]
    assert output["root_token_ids"] == [5, 12]
    assert output["eager_frontier_tokens"] == {1: 12}
    final = engine.model.calls[-1][2]
    # Request 1 owns physical slots 32..63. Its parent is at logical 2..6;
    # the new eager root and all candidates live at logical 7..11.
    assert output["cache_slot_mapping"][5:].tolist() == [39, 40, 41, 42, 43]
    eager_root_mask = final.attention_mask[5]
    assert not eager_root_mask[[0, 1, 2, 3, 4, 7]].any()
    assert eager_root_mask[[5, 6, 8, 9, 10, 11]].all()
    # Logical RoPE position is 5, despite its scratch physical location 7.
    assert engine.model.calls[-1][1][5] == 5


def test_eager_frontier_uses_actual_noncontiguous_dependency_node_index():
    engine = _engine(requests=1)
    base = build_tree_speculation_plan(2, 2, prefix_len=2, max_model_len=32)
    # The primary branch follows indices 0 -> 2, not 0 -> 1.
    parent = replace(base, parent_indices=torch.tensor([-1, -1, 0, 1], dtype=torch.int32))
    assert tree_primary_path(parent) == [0, 2]
    eager = build_tree_speculation_plan(2, 1, prefix_len=5, max_model_len=32)
    output = engine.draft_tree_forward([eager], [99], [0], eager_parent_sources={0: (parent, [10, 20, 30, 25])})
    assert engine.model.calls[0][0].tolist() == [30]
    assert output["eager_frontier_tokens"] == {0: 31}


def test_scratch_mask_blocks_parent_siblings_for_every_new_tree_query():
    parent = build_tree_speculation_plan(3, 2, prefix_len=2, max_model_len=32)
    eager = build_tree_speculation_plan(2, 2, prefix_len=5, max_model_len=32)
    scratch = NativePearlEngine._tree_eager_scratch_plan(eager, parent)
    assert scratch.cache_positions.tolist() == [9, 10, 11, 12, 13]
    assert scratch.positions.tolist() == eager.positions.tolist()
    assert scratch.attention_mask[:, 5:9].all()
    assert not scratch.attention_mask[:, :5].any()
    assert scratch.attention_mask[2, 12:14].all()
    assert not scratch.attention_mask[2, 9:12].any()


def test_mixed_depth_tree_batch_removes_only_finished_expansions():
    engine = _engine()
    plans = [
        build_tree_speculation_plan(2, 1, prefix_len=2, max_model_len=32),
        build_tree_speculation_plan(2, 3, prefix_len=2, max_model_len=32),
    ]
    output = engine.draft_tree_forward(plans, [5, 9], [0, 1])
    assert [call[0].numel() for call in engine.model.calls] == [2, 1, 1, 10]
    assert output["model_calls"] == 4


def test_nonzero_draft_temperature_samples_distinct_siblings():
    engine = _engine(requests=1)
    plan = build_tree_speculation_plan(2, 2, prefix_len=2, max_model_len=32)
    torch.manual_seed(7)
    output = engine.draft_tree_forward([plan], [5], [0], draft_temperatures=[0.7])

    # The exponential-race sampler draws siblings without replacement; this
    # checks that contract instead of binding the test to torch.multinomial.
    # Spine-first layout stores the depth-0 sibling pair at nodes 0 and 2.
    assert len({output["draft_token_ids"][0][0], output["draft_token_ids"][0][2]}) == 2
    assert all(0.0 <= value <= 1.0 for value in output["node_confidences"][0].tolist())
