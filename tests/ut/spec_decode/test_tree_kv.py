# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import torch

from vllm_ascend.spec_decode import tree_kv
from vllm_ascend.spec_decode.tree_kv import move_kv_cache_slots


def test_move_kv_cache_slots_handles_overlap():
    cache = torch.arange(2 * 2 * 4 * 3).reshape(2, 2, 4, 3).clone()
    original = cache.clone()
    move_kv_cache_slots([cache], torch.tensor([3, 6]), torch.tensor([1, 3]))
    for plane in range(2):
        flat = cache[plane].flatten(0, 1)
        original_flat = original[plane].flatten(0, 1)
        assert torch.equal(flat[1], original_flat[3])
        assert torch.equal(flat[3], original_flat[6])


def test_move_tuple_kv_cache_slots():
    key = torch.arange(8).reshape(2, 4, 1).clone()
    value = key + 100
    move_kv_cache_slots([(key, value)], torch.tensor([5]), torch.tensor([0]))
    assert key.flatten(0, 1)[0].item() == 5
    assert value.flatten(0, 1)[0].item() == 105


def test_move_npu_tuple_uses_fused_reshape_and_cache(monkeypatch):
    key = Mock(device=Mock(type="npu"))
    value = Mock(device=Mock(type="npu"))
    fused = Mock()
    monkeypatch.setattr(tree_kv, "_move_npu_kv_pair_slots", fused)
    monkeypatch.setattr(tree_kv, "_is_npu_kv_pair", lambda pair: True)

    move_kv_cache_slots([(key, value)], torch.tensor([5]), torch.tensor([0]))

    fused.assert_called_once()
