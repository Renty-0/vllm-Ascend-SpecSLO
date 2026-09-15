# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from unittest.mock import Mock, call

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
    assert fused.call_args.args[3].dtype == torch.int32


def _cpu_graph_runner(monkeypatch, *, max_graph_entries=16):
    monkeypatch.setattr(tree_kv, "_is_npu_kv_pair", lambda pair: True)
    key = torch.arange(8).reshape(2, 4, 1).clone()
    value = key + 100
    runner = tree_kv.TreeKVCompactionGraphRunner(
        [(key, value)],
        max_graph_entries=max_graph_entries,
    )
    return runner, key, value


def test_tree_kv_graph_runner_caches_exact_move_counts(monkeypatch):
    runner, _, _ = _cpu_graph_runner(monkeypatch)
    two = tree_kv._TreeKVCompactionGraphEntry(
        torch.empty(2, dtype=torch.long),
        torch.empty(2, dtype=torch.int32),
        Mock(),
        validated=True,
    )
    one = tree_kv._TreeKVCompactionGraphEntry(
        torch.empty(1, dtype=torch.long),
        torch.empty(1, dtype=torch.int32),
        Mock(),
        validated=True,
    )
    capture = Mock(side_effect=[two, one])
    monkeypatch.setattr(runner, "_capture_entry", capture)

    assert runner.move(torch.tensor([3, 4]), torch.tensor([0, 1]))
    assert runner.move(torch.tensor([5, 6]), torch.tensor([1, 2]))
    assert runner.move(torch.tensor([7]), torch.tensor([0]))

    assert capture.call_args_list == [call(2), call(1)]
    assert two.graph.replay.call_count == 2
    assert one.graph.replay.call_count == 1
    assert runner.capture_count == 2
    assert runner.replay_count == 3


def test_tree_kv_graph_runner_capacity_falls_back_to_eager(monkeypatch):
    runner, _, _ = _cpu_graph_runner(monkeypatch, max_graph_entries=1)
    entry = tree_kv._TreeKVCompactionGraphEntry(
        torch.empty(1, dtype=torch.long),
        torch.empty(1, dtype=torch.int32),
        Mock(),
        validated=True,
    )
    monkeypatch.setattr(runner, "_capture_entry", Mock(return_value=entry))
    eager = Mock()
    monkeypatch.setattr(runner, "_eager_move", eager)

    assert runner.move(torch.tensor([3]), torch.tensor([0]))
    assert not runner.move(torch.tensor([4, 5]), torch.tensor([1, 2]))

    eager.assert_called_once()
    assert runner.capacity_fallback_count == 1
    assert set(runner.entries) == {1}


def test_tree_kv_graph_capture_failure_executes_real_move_once(monkeypatch):
    runner, _, _ = _cpu_graph_runner(monkeypatch)
    monkeypatch.setattr(
        runner,
        "_capture_entry",
        Mock(side_effect=RuntimeError("unsupported graph operator")),
    )
    eager = Mock()
    monkeypatch.setattr(runner, "_eager_move", eager)
    source = torch.tensor([3, 4])
    destination = torch.tensor([0, 1])

    assert not runner.move(source, destination)
    assert not runner.move(source, destination)

    assert eager.call_count == 2
    assert runner.capture_attempt_count == 1
    assert runner.failed_capture_count == 1
    assert runner.disabled_move_counts == {2}


def test_tree_kv_graph_capture_warms_and_captures_identity(monkeypatch):
    runner, _, _ = _cpu_graph_runner(monkeypatch)
    eager = Mock()
    graph = Mock()
    monkeypatch.setattr(runner, "_eager_move", eager)
    monkeypatch.setattr(torch.npu, "synchronize", Mock())
    monkeypatch.setattr(torch.npu, "NPUGraph", Mock(return_value=graph))
    monkeypatch.setattr(torch.npu, "graph", lambda captured: nullcontext())

    entry = runner._capture_entry(3)

    assert eager.call_count == 2
    for args in (invocation.args for invocation in eager.call_args_list):
        assert torch.equal(args[0], torch.tensor([0, 1, 2]))
        assert torch.equal(args[1], torch.tensor([0, 1, 2], dtype=torch.int32))
    assert entry.graph is graph
    assert entry.source_slots.dtype == torch.long
    assert entry.destination_slots.dtype == torch.int32


def test_tree_kv_graph_first_real_move_replays_only_once(monkeypatch):
    runner, key, value = _cpu_graph_runner(monkeypatch)
    graph = Mock()
    entry = tree_kv._TreeKVCompactionGraphEntry(
        torch.empty(2, dtype=torch.long),
        torch.empty(2, dtype=torch.int32),
        graph,
    )
    monkeypatch.setattr(runner, "_capture_entry", Mock(return_value=entry))

    def cpu_move(source, destination):
        destination = destination.to(torch.long)
        for cache in (key, value):
            flat = cache.flatten(0, 1)
            flat.index_copy_(0, destination, flat.index_select(0, source))

    monkeypatch.setattr(runner, "_eager_move", cpu_move)
    graph.replay.side_effect = lambda: cpu_move(
        entry.source_slots,
        entry.destination_slots,
    )
    original_key = key.clone()
    original_value = value.clone()

    assert runner.move(torch.tensor([3, 6]), torch.tensor([1, 3]))

    assert graph.replay.call_count == 1
    assert entry.validated
    assert torch.equal(key.flatten(0, 1)[1], original_key.flatten(0, 1)[3])
    assert torch.equal(key.flatten(0, 1)[3], original_key.flatten(0, 1)[6])
    assert torch.equal(value.flatten(0, 1)[1], original_value.flatten(0, 1)[3])
    assert torch.equal(value.flatten(0, 1)[3], original_value.flatten(0, 1)[6])


def test_tree_kv_graph_runner_release_resets_entries(monkeypatch):
    runner, _, _ = _cpu_graph_runner(monkeypatch)
    first = tree_kv._TreeKVCompactionGraphEntry(torch.empty(1), torch.empty(1), Mock(), validated=True)
    second = tree_kv._TreeKVCompactionGraphEntry(torch.empty(2), torch.empty(2), Mock(), validated=True)
    runner.entries.update({1: first, 2: second})
    synchronize = Mock()
    monkeypatch.setattr(torch.npu, "synchronize", synchronize)

    assert runner.release() == 2
    assert not runner.entries
    synchronize.assert_called_once()
    first.graph.reset.assert_called_once()
    second.graph.reset.assert_called_once()
