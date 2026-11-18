"""Causal KV routing checks for selective target-verification envelopes."""

import pytest
import torch

from vllm_ascend.spec_decode.pearl.mixed_width_layout import (
    plan_selective_verify_layout,
    route_selective_verify_envelope,
)
from vllm_ascend.spec_decode.pearl.mixed_width_ops import pack_selective_target_queries


def test_mixed_width_rows_and_padding_have_disjoint_causal_slots():
    layout = plan_selective_verify_layout((4, 2, 4), capacities=(4,), token_buckets=(16,))
    envelope = route_selective_verify_envelope(
        layout,
        (10, 11, 12),
        (100, 250, 400),
        scratch_sequence_id=99,
    )
    assert layout.real_query_tokens == 10
    assert layout.query_lengths == (4, 2, 4, 1, 5)
    assert envelope.token_sequence_ids[:10] == (10,) * 4 + (11,) * 2 + (12,) * 4
    assert envelope.positions[:10] == (100, 101, 102, 103, 250, 251, 400, 401, 402, 403)
    assert envelope.token_sequence_ids[10:] == (99,) * 6
    assert envelope.positions[10:] == tuple(range(6))
    assert len(set(zip(envelope.token_sequence_ids, envelope.positions))) == layout.token_bucket


def test_same_graph_shape_can_route_different_width_patterns_without_aliasing():
    first = plan_selective_verify_layout((4, 2, 4), capacities=(4,), token_buckets=(16,))
    second = plan_selective_verify_layout((2, 4, 4), capacities=(4,), token_buckets=(16,))
    assert first.graph_key == second.graph_key
    assert first.query_lengths != second.query_lengths
    for layout in (first, second):
        envelope = route_selective_verify_envelope(layout, (1, 2, 3), (7, 10, 5), scratch_sequence_id=100)
        assert len(set(zip(envelope.token_sequence_ids, envelope.positions))) == 16


def test_invalid_capacity_or_overlapping_sequence_ids_are_rejected():
    with pytest.raises(ValueError, match="capacity"):
        plan_selective_verify_layout((4, 2, 4), capacities=(2,))
    with pytest.raises(ValueError, match="token buckets"):
        plan_selective_verify_layout((4, 2, 4), capacities=(4,), token_buckets=(8,))
    layout = plan_selective_verify_layout((2, 4), capacities=(2,), token_buckets=(8,))
    with pytest.raises(ValueError, match="unique"):
        route_selective_verify_envelope(layout, (1, 1), (0, 0), scratch_sequence_id=99)


def test_selective_query_and_verdict_labels_follow_the_same_causal_rows():
    roots = torch.tensor([7, 8, 9], dtype=torch.long)
    proposals = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]])
    query, labels = pack_selective_target_queries(roots, proposals, (4, 2, 4))
    assert query.tolist() == [7, 10, 11, 12, 8, 20, 9, 30, 31, 32]
    assert labels.tolist() == [10, 11, 12, 13, 20, 21, 30, 31, 32, 33]
    assert len(query) == len(labels) == 10
