# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.spec_decode.tree import (
    make_full_tree_parents,
    make_spine_first_tree_parents,
    verify_greedy_tree,
)


def test_full_tree_parent_layout():
    assert make_full_tree_parents(2, 3).tolist() == [
        -1, -1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5
    ]


def test_spine_first_tree_parent_layout():
    assert make_spine_first_tree_parents(2, 3).tolist() == [-1, 0, 1, -1, 0, 1]


def test_greedy_tree_selects_non_contiguous_longest_path():
    # root children: nodes 0,1; node 1 children: nodes 4,5.
    drafts = torch.tensor([10, 20, 11, 12, 21, 22])
    parents = torch.tensor([-1, -1, 0, 0, 1, 1])
    # Root predicts 20, then node 1 predicts 22, then node 5 predicts 99.
    predictions = torch.tensor([20, 0, 22, 0, 0, 0, 99])
    result = verify_greedy_tree(drafts, parents, predictions, max_depth=2)
    assert result.token_ids.tolist() == [20, 22, 99]
    assert result.accepted_node_indices.tolist() == [1, 5]


def test_greedy_tree_recovers_when_no_child_matches():
    result = verify_greedy_tree(
        torch.tensor([10, 20]),
        torch.tensor([-1, -1]),
        torch.tensor([30, 31, 32]),
        max_depth=1,
    )
    assert result.token_ids.tolist() == [30, -1]
    assert result.accepted_node_indices.tolist() == [-1]


@pytest.mark.parametrize("width,depth", [(0, 1), (1, 0)])
def test_full_tree_rejects_non_positive_shape(width, depth):
    with pytest.raises(ValueError):
        make_full_tree_parents(width, depth)
