# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.spec_decode.pearl.spec_rhythm import SpecRhythmBudgetPlan
from vllm_ascend.spec_decode.pearl.tree import (
    SpecRhythmTreeCoordinator,
    build_tree_attention_mask,
    build_tree_speculation_plan,
    make_spine_first_parents,
    select_tree_candidates,
    tree_budget_from_spec_rhythm,
    verify_greedy_tree,
    verify_greedy_tree_batch,
    verify_sampled_tree,
)
from vllm_ascend.spec_decode.tree_kv import build_tree_kv_compaction_plan


def test_tree_mask_exposes_prefix_root_and_ancestors_only():
    mask = build_tree_attention_mask(2, 2, prefix_len=3, max_model_len=8)
    assert mask.shape == (5, 8)
    assert not mask[0, :4].any()
    assert not mask[1, :5].any()
    assert not mask[2, 5]
    assert mask[3, 4]
    assert mask[3, 5]
    assert not mask[3, 6]
    assert not mask[4, 4]
    assert mask[4, 6]


def test_tree_plan_positions_and_budget():
    plan = build_tree_speculation_plan(2, 3, prefix_len=4, max_model_len=12, candidate_budget=4)
    assert plan.positions.tolist() == [4, 5, 6, 7, 5, 6, 7]
    assert plan.candidate_budget == 4
    assert torch.equal(plan.parent_indices, make_spine_first_parents(2, 3))
    assert plan.cache_positions is not None
    assert plan.cache_positions.tolist() == list(range(4, 11))


def test_selection_preserves_ancestor_chain():
    parents = make_spine_first_parents(2, 3)
    tokens = torch.arange(6)
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 8.0])
    selected = select_tree_candidates(tokens, parents, scores, budget=3)
    assert selected.indices.tolist() == [0, 1, 5]
    assert selected.token_ids.tolist() == [0, 1, 5]


def test_selection_rejects_invalid_parent_order():
    with pytest.raises(ValueError):
        select_tree_candidates(torch.arange(2), torch.tensor([1, -1]), torch.ones(2), 1)


def test_tree_budget_reads_normal_and_eager_allocations():
    plan = SpecRhythmBudgetPlan(
        plan_id=1,
        normal_budgets={2: 3},
        eager_budgets={5: 2},
        progress_gaps={2: 3, 5: 2},
        eager_priorities={5: 1.0},
        verification_roof=8,
        draft_token_budget=8,
        allocated_draft_tokens=5,
    )
    assert tree_budget_from_spec_rhythm(plan, 2, 2, 3) == 3
    assert tree_budget_from_spec_rhythm(plan, 5, 2, 3) == 2


def test_tree_coordinator_uses_budget_to_choose_depth():
    plan = SpecRhythmBudgetPlan(
        plan_id=1,
        normal_budgets={2: 3},
        eager_budgets={},
        progress_gaps={2: 3},
        eager_priorities={},
        verification_roof=4,
        draft_token_budget=4,
        allocated_draft_tokens=3,
    )
    tree = SpecRhythmTreeCoordinator(width=2, max_depth=3).for_request(plan, 2, prefix_len=2, max_model_len=8)
    assert tree is not None
    assert (tree.width, tree.depth, tree.candidate_budget) == (2, 2, 3)


def test_device_tree_verifier_follows_ancestor_chain_and_bonus():
    parents = make_spine_first_parents(2, 2)
    result = verify_greedy_tree(
        torch.tensor([10, 11, 20, 21]),
        parents,
        torch.tensor([10, 11, 42, 99]),
        torch.tensor(7),
        max_depth=2,
    )
    assert result.token_ids.tolist() == [10, 11, 7]
    assert result.accepted_node_indices.tolist() == [0, 1]


def test_device_tree_verifier_selects_bonus_from_accepted_sibling_frontier():
    # Nodes 0 and 1 are the first-level siblings.  The target chooses node 1
    # and its output row (query root + query node) supplies the bonus token.
    result = verify_greedy_tree(
        torch.tensor([10, 20]),
        torch.tensor([-1, -1], dtype=torch.int32),
        torch.tensor([20, 0, 77]),
        torch.tensor(0),
        max_depth=1,
    )
    assert result.token_ids.tolist() == [20, 77]
    assert result.accepted_node_indices.tolist() == [1]


def test_sampled_tree_commits_the_target_sample_on_hit_and_miss():
    hit = verify_sampled_tree(
        torch.tensor([10, 20]),
        torch.tensor([-1, -1], dtype=torch.int32),
        torch.tensor([20, 0, 77]),
        torch.tensor(0),
        max_depth=1,
    )
    miss = verify_sampled_tree(
        torch.tensor([10, 20]),
        torch.tensor([-1, -1], dtype=torch.int32),
        torch.tensor([30, 0, 0]),
        torch.tensor(0),
        max_depth=1,
    )
    assert hit.token_ids.tolist() == [20, 77]
    assert hit.accepted_node_indices.tolist() == [1]
    assert miss.token_ids.tolist() == [30, -1]
    assert miss.accepted_node_indices.tolist() == [-1]


def test_device_tree_verifier_handles_short_variable_budget_without_index_error():
    result = verify_greedy_tree(
        torch.tensor([10]),
        torch.tensor([-1], dtype=torch.int32),
        torch.tensor([10, 42]),
        torch.tensor(0),
        max_depth=3,
    )
    assert result.token_ids.tolist() == [10, 42, -1, -1]
    assert result.accepted_node_indices.tolist() == [0, -1, -1]


def test_device_tree_verifier_rejects_first_token_without_host_traversal():
    result = verify_greedy_tree(
        torch.tensor([10, 11]),
        torch.tensor([-1, 0], dtype=torch.int32),
        torch.tensor([99, 11]),
        torch.tensor(7),
        max_depth=2,
    )
    assert result.token_ids.tolist() == [99, -1, -1]
    assert result.accepted_node_indices.tolist() == [-1, -1]


def test_variable_width_tree_batch_and_kv_compaction_plan():
    drafts = torch.tensor([10, 11, 30, 31, 32])
    parents = torch.tensor([-1, 0, -1, 0, 0], dtype=torch.int32)
    targets = torch.tensor([10, 11, 30, 31, 99])
    result = verify_greedy_tree_batch(
        drafts,
        parents,
        [2, 3],
        targets,
        torch.tensor([7, 8]),
        max_depth=2,
    )
    assert result.token_ids.shape == (2, 3)
    assert result.accepted_node_indices.tolist() == [[0, 1], [0, 1]]
    compaction = build_tree_kv_compaction_plan(
        torch.tensor([100, 101, 102, 103]),
        torch.tensor([0, 2, -1], dtype=torch.int32),
    )
    assert compaction.source_slots.tolist() == [100, 102]
    assert compaction.destination_slots.tolist() == [100, 101]
    shifted = build_tree_kv_compaction_plan(
        torch.tensor([100, 101, 102]),
        torch.tensor([0, 2], dtype=torch.int32),
        destination_start=200,
    )
    assert shifted.destination_slots.tolist() == [200, 201]


def test_variable_width_tree_batch_accepts_dynamic_frontier_rows():
    result = verify_greedy_tree_batch(
        torch.tensor([10, 20, 30]),
        torch.tensor([-1, -1, -1], dtype=torch.int32),
        [1, 2],
        # Request 0: root -> node 0 -> bonus 41.
        # Request 1: root selects sibling node 1 -> bonus 42.
        torch.tensor([10, 41, 30, 0, 42]),
        torch.tensor([0, 0]),
        max_depth=2,
    )
    assert result.token_ids.tolist() == [[10, 41, -1], [30, 42, -1]]
    assert result.accepted_node_indices.tolist() == [[0, -1], [1, -1]]
