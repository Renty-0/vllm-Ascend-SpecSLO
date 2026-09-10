# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for production FULL-mask FIA capture and replay."""

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.ut.spec_decode.test_spec_rhythm_tree_graph import _runner
from vllm_ascend.spec_decode.pearl import native_graph as graph
from vllm_ascend.spec_decode.pearl.native_model import NativeAttentionMetadata


def _metadata(counts=(2, 5), lengths=(12, 13), columns=32):
    cumulative, total = [], 0
    for count in counts:
        total += count
        cumulative.append(total)
    mask = torch.ones(len(counts), 1, max(counts), columns, dtype=torch.bool)
    packed_masks = []
    for request, count in enumerate(counts):
        mask[request, 0, :count, :lengths[request]] = False
        mask[request, 0, :count, 1] = True
        packed_masks.append(mask[request, 0, :count].clone())
    tables = torch.arange(len(counts) * (columns // 4)).reshape(len(counts), -1).int()
    return NativeAttentionMetadata(
        slot_mapping=torch.arange(total, dtype=torch.int32),
        context_lens=torch.tensor([length for count, length in zip(counts, lengths) for _ in range(count)]),
        block_tables=tables.repeat_interleave(torch.tensor(counts), dim=0),
        actual_seq_lengths_q=tuple(cumulative), sequence_lens=tuple(lengths), request_block_tables=tables,
        attention_mask=torch.cat(packed_masks), use_fused_infer_attention=True,
        tree_attention=True, tree_attention_mask=mask,
    )


def _call(runner, metadata):
    size = metadata.slot_mapping.numel()
    return runner.run_target_greedy([torch.arange(size)], [torch.arange(size)], [metadata], vocabulary_size=32)


def _fia_arguments(metadata, *, tree=True):
    rows = metadata.slot_mapping.numel()
    return {
        "query": torch.zeros(rows, 4, 8), "key_cache": torch.zeros(16, 4, 16),
        "value_cache": torch.zeros(16, 4, 16), "num_kv_heads": 2, "num_heads": 4,
        "scale": 0.125, "block_table": metadata.request_block_tables,
        "attention_mask": metadata.tree_attention_mask if tree else metadata.attention_mask,
        "actual_seq_lengths_q": list(metadata.actual_seq_lengths_q),
        "actual_seq_lengths_kv": list(metadata.sequence_lens), "block_size": 4,
        "output": torch.zeros(rows, 4, 8),
        **({"tree_attention": True} if tree else {}),
    }


@pytest.mark.parametrize("tree", [False, True])
def test_fia_eager_and_capture_keep_distinct_operator_contracts(tree):
    args = _fia_arguments(_metadata(), tree=tree)
    with (
        patch.object(graph.torch_npu.npu_fused_infer_attention_score, "out") as operator,
        patch.object(
            graph.torch_npu, "_npu_fused_infer_attention_score_get_max_workspace", return_value=torch.empty(8),
        ),
        patch("torch.npu.current_stream", return_value=MagicMock()),
        patch("torch.npu.ExternalEvent", return_value=MagicMock()),
        patch("torch.npu.graph_task_group_begin"),
        patch("torch.npu.graph_task_group_end", return_value="handle"),
    ):
        graph.run_native_fused_infer_attention(**args)
        with graph._collect_graph_tasks() as tasks:
            graph.run_native_fused_infer_attention(**args)
        assert len(tasks) == 1 and tasks[0].tree_attention == tree
        for call in operator.call_args_list:
            kwargs = call.kwargs
            assert kwargs["actual_seq_lengths"] == [2, 7]
            if tree:
                assert kwargs["sparse_mode"] == 1 and kwargs["inner_precise"] == 2
                assert "next_tokens" not in kwargs and "pre_tokens" not in kwargs
                assert kwargs["atten_mask"].shape == (2, 1, 5, 32)
            else:
                assert kwargs["sparse_mode"] == 3
                assert kwargs["next_tokens"] == kwargs["pre_tokens"] == 2147483647
                assert "inner_precise" not in kwargs


def test_fia_workspace_cache_separates_linear_tree_and_full_mask_shape():
    with (
        patch.object(graph.torch_npu.npu_fused_infer_attention_score, "out"),
        patch.object(graph.torch_npu, "_npu_fused_infer_attention_score_get_max_workspace",
                     return_value=torch.empty(8)) as workspace,
        patch("torch.npu.current_stream", return_value=MagicMock()),
        patch("torch.npu.ExternalEvent", return_value=MagicMock()),
        patch("torch.npu.graph_task_group_begin"), patch("torch.npu.graph_task_group_end"),
        graph._collect_graph_tasks(),
    ):
        for tree, columns in [(False, 32), (True, 32), (True, 32), (True, 64)]:
            graph.run_native_fused_infer_attention(**_fia_arguments(_metadata(columns=columns), tree=tree))
        assert workspace.call_count == 3


@pytest.mark.parametrize("tree", [False, True])
def test_fia_task_update_preserves_mode_and_updates_heterogeneous_lengths(tree):
    args = _fia_arguments(_metadata(), tree=tree)
    task = graph.NativeFusedInferAttentionGraphTask(
        query=args["query"], key_cache=args["key_cache"], value_cache=args["value_cache"],
        num_kv_heads=2, num_heads=4, scale=0.125, block_table=args["block_table"],
        attention_mask=args["attention_mask"], output=args["output"], softmax_lse=torch.zeros(1),
        block_size=4, workspace=torch.zeros(1), handle="handle", event=MagicMock(), tree_attention=tree,
    )
    stream = MagicMock()
    with (
        patch.object(graph.torch_npu.npu_fused_infer_attention_score, "out") as operator,
        patch("torch.npu.graph_task_update_begin") as begin,
        patch("torch.npu.graph_task_update_end") as end,
    ):
        _runner()._update_fused_infer_attention_task(task, [5, 7], [20, 16], stream=stream)
        begin.assert_called_once_with(stream, "handle")
        end.assert_called_once_with(stream)
        task.event.record.assert_called_once_with(stream)
        kwargs = operator.call_args.kwargs
        assert kwargs["actual_seq_lengths"] == [5, 7]
        assert kwargs["actual_seq_lengths_kv"] == [20, 16]
        assert kwargs["atten_mask"] is args["attention_mask"]
        if tree:
            assert kwargs["sparse_mode"] == 1 and kwargs["inner_precise"] == 2
            assert "next_tokens" not in kwargs and "pre_tokens" not in kwargs
        else:
            assert kwargs["sparse_mode"] == 3 and kwargs["next_tokens"] == 0
            assert "inner_precise" not in kwargs and "pre_tokens" not in kwargs


def test_tree_fia_graph_key_reuses_true_length_changes_without_context_bucketing():
    runner = _runner()
    runner._capture_target = MagicMock(return_value=(torch.arange(7),))
    first = _metadata()
    second = _metadata(counts=(5, 2), lengths=(20, 19))
    with patch.object(graph, "bucket_tree_attention_metadata", side_effect=AssertionError("no dense bucketing")):
        _call(runner, first)
        _call(runner, second)
        _call(runner, replace(first, tree_attention=False, tree_attention_mask=None))
        _call(runner, _metadata(counts=(3, 4)))
    calls = runner._capture_target.call_args_list
    assert calls[0].args[0] == calls[1].args[0]
    assert calls[0].args[0] != calls[2].args[0]
    assert calls[0].args[0] != calls[3].args[0]
    assert "tree-fia-full" in calls[0].args[0][0]
    assert calls[0].kwargs["reference_metadatas"][0] is first
    assert first.context_lens.tolist() == [12, 12, 13, 13, 13, 13, 13]


def test_tree_fia_capture_owns_full_mask_and_replay_copies_every_new_value():
    runner = _runner()
    runner._execute_target = MagicMock(return_value=(torch.arange(7),))
    runner._update_target_attention_tasks = MagicMock()
    first = _metadata()
    expected_first = first.tree_attention_mask.clone()
    with (
        patch("torch.npu.NPUGraph", return_value=MagicMock()), patch("torch.npu.graph", return_value=MagicMock()),
        patch("torch.npu.current_stream", return_value=MagicMock()), patch("torch.npu.synchronize"),
    ):
        _call(runner, first)
        entry = next(iter(runner.target_entries.values()))
        captured = runner._execute_target.call_args_list[1].args[2][0]
        assert captured.tree_attention
        assert entry.tree_attention_modes == (True,)
        assert captured.tree_attention_mask is entry.tree_attention_masks[0]
        assert captured.tree_attention_mask.data_ptr() != first.tree_attention_mask.data_ptr()
        first.tree_attention_mask.fill_(False)
        assert torch.equal(captured.tree_attention_mask, expected_first)
        second = _metadata(counts=(5, 2), lengths=(20, 19))
        second.tree_attention_mask[1, 0, 0, 3] = True
        _call(runner, second)
        assert runner.last_target_execution.mode == "replay"
        assert runner.capture_count == 1
        assert torch.equal(entry.tree_attention_masks[0], second.tree_attention_mask)
        assert torch.equal(entry.attention_masks[0], second.attention_mask)
        assert entry.actual_seq_lengths_q == ((5, 7),)
        assert entry.sequence_lens == ((20, 19),)


@pytest.mark.parametrize("corruption", ["missing", "shape", "dtype", "mode"])
def test_full_mask_replay_contract_failure_precedes_any_buffer_copy(corruption):
    runner = _runner()
    runner._execute_target = MagicMock(return_value=(torch.arange(7),))
    runner._update_target_attention_tasks = MagicMock()
    metadata = _metadata()
    with (
        patch("torch.npu.NPUGraph", return_value=MagicMock()), patch("torch.npu.graph", return_value=MagicMock()),
        patch("torch.npu.current_stream", return_value=MagicMock()), patch("torch.npu.synchronize"),
    ):
        _call(runner, metadata)
    entry = next(iter(runner.target_entries.values()))
    original_ids = entry.input_ids[0].clone()
    changes = {
        "missing": {"tree_attention_mask": None},
        "shape": {"tree_attention_mask": torch.zeros(2, 1, 4, 32, dtype=torch.bool)},
        "dtype": {"tree_attention_mask": metadata.tree_attention_mask.float()},
        "mode": {"tree_attention": False},
    }
    broken = replace(metadata, **changes[corruption])
    with pytest.raises(RuntimeError, match="FULL tree mask|tree/linear FIA contract"):
        graph.NativeACLGraphRunner._copy_target_inputs(entry, [torch.full((7,), 999)], [torch.arange(7)], [broken])
    assert torch.equal(entry.input_ids[0], original_ids)


@pytest.mark.parametrize("mask,lengths,kv,query_count", [
    (torch.zeros(7, 32, dtype=torch.bool), [2, 7], [12, 13], 7),
    (torch.zeros(2, 1, 5, 32), [2, 7], [12, 13], 7),
    (torch.zeros(2, 1, 5, 32, dtype=torch.bool), [2, 6], [12, 13], 7),
    (torch.zeros(2, 1, 5, 32, dtype=torch.bool), [2, 7], [12, 33], 7),
    (torch.zeros(2, 1, 4, 32, dtype=torch.bool), [2, 7], [12, 13], 7),
])
def test_full_mask_abi_rejects_padding_bad_shape_and_out_of_range_lengths(mask, lengths, kv, query_count):
    with pytest.raises(ValueError, match="Tree FIA"):
        graph._validate_tree_fia_mask(query_count, mask, torch.zeros(2, 8), lengths, kv, 4)
