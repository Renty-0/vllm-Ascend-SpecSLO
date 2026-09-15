# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from vllm_ascend.spec_decode.pearl.native_graph import (
    NativeACLGraphRunner,
    NativePagedAttentionGraphTask,
)


def _draft_replay_runner():
    update_stream = MagicMock()
    with patch(
        "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream",
        return_value=update_stream,
    ):
        runner = NativeACLGraphRunner(MagicMock(), enabled=True)
    entry = MagicMock(
        output=torch.tensor([[4, 5]]),
        graph=MagicMock(),
    )
    runner.draft_entries[("draft-greedy:8|steps:2|paged", 1)] = entry
    runner._copy_draft_inputs = MagicMock()
    runner._update_draft_attention_tasks = MagicMock()
    return runner, entry, update_stream


def _run_draft_replay(runner):
    metadata = SimpleNamespace(use_fused_infer_attention=False)
    return runner.run_draft_greedy(
        torch.tensor([3]),
        [torch.tensor([2]), torch.tensor([3])],
        [metadata, metadata],
        vocabulary_size=8,
    )


def _generic_replay_runner():
    update_stream = MagicMock()
    with patch(
        "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream",
        return_value=update_stream,
    ):
        runner = NativeACLGraphRunner(MagicMock(), enabled=True)
    entry = MagicMock(
        output=torch.tensor([[4, 5]]),
        graph=MagicMock(),
    )
    entry.runtime_validated = True
    entry.validated_real_row_count = 1
    entry.tasks = [
        SimpleNamespace(event=MagicMock()),
        SimpleNamespace(event=MagicMock()),
    ]
    entry.replay_first_copy_done_event = MagicMock()
    runner.entries[("hidden", 1)] = entry
    runner._copy_inputs = MagicMock()
    runner._update_attention_tasks = MagicMock()
    return runner, entry, update_stream


def _run_generic_replay(runner):
    metadata = SimpleNamespace(use_fused_infer_attention=False)
    return runner(
        torch.tensor([3]),
        torch.tensor([2]),
        metadata,
    )


def _paged_task(*, context_lens=(8, 8)):
    return NativePagedAttentionGraphTask(
        query=torch.zeros((2, 4, 8)),
        key_cache=torch.zeros((4, 16, 2, 8)),
        value_cache=torch.zeros((4, 16, 2, 8)),
        num_kv_heads=2,
        num_heads=4,
        scale=0.125,
        block_table=torch.zeros((2, 1), dtype=torch.int32),
        context_lens=torch.tensor(context_lens, dtype=torch.int32),
        output=torch.zeros((2, 4, 8)),
        workspace=MagicMock(name="old_workspace"),
        handle=MagicMock(name="handle"),
        event=MagicMock(name="event"),
    )


def _draft_capture_metadata():
    return SimpleNamespace(
        slot_mapping=torch.tensor([0]),
        context_lens=torch.tensor([1], dtype=torch.int32),
        block_tables=torch.tensor([[0]], dtype=torch.int32),
        actual_seq_lengths_q=(),
        sequence_lens=(),
        request_block_tables=None,
        attention_mask=None,
        use_fused_infer_attention=False,
    )


@pytest.mark.parametrize(
    ("capture_output", "expected_mode", "expected_reason"),
    [
        (torch.tensor([[4, 5]]), "capture_replay", None),
        (torch.tensor([[9, 9]]), "eager", "capture_validation"),
    ],
)
def test_draft_graph_reports_capture_outcome(
    capture_output,
    expected_mode,
    expected_reason,
):
    update_stream = MagicMock()
    with patch(
        "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream",
        return_value=update_stream,
    ):
        runner = NativeACLGraphRunner(MagicMock(), enabled=True)
    runner._execute_draft = MagicMock(
        side_effect=[torch.tensor([[4, 5]]), capture_output]
    )
    runner._update_draft_attention_tasks = MagicMock()
    warning_context = (
        pytest.warns(RuntimeWarning, match="gamma-step draft ACLGraph")
        if expected_reason is not None
        else patch("warnings.warn")
    )
    with (
        patch("torch.npu.NPUGraph", return_value=MagicMock()),
        patch("torch.npu.graph", return_value=MagicMock()),
        patch("torch.npu.current_stream", return_value=MagicMock()),
        patch("torch.npu.synchronize"),
        warning_context,
    ):
        output = runner.run_draft_greedy(
            torch.tensor([3]),
            [torch.tensor([2]), torch.tensor([3])],
            [_draft_capture_metadata(), _draft_capture_metadata()],
            vocabulary_size=8,
        )

    assert runner.last_draft_execution.mode == expected_mode
    assert runner.last_draft_execution.fallback_reason == expected_reason
    assert runner.last_draft_execution.capture_attempted
    assert runner.last_draft_execution.replay_executed
    if expected_reason is None:
        assert output.tolist() == [[4, 5]]
        assert runner.last_draft_execution.used_aclgraph
    else:
        assert output.tolist() == [[4, 5]]
        assert not runner.last_draft_execution.used_aclgraph


def test_draft_graph_auxiliary_task_update_has_reverse_stream_dependency():
    runner, entry, update_stream = _draft_replay_runner()
    current_stream = MagicMock()
    with (
        patch.dict(
            os.environ,
            {"VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE": "0"},
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.current_stream",
            return_value=current_stream,
        ),
    ):
        output = _run_draft_replay(runner)

    assert output.tolist() == [[4, 5]]
    update_stream.wait_stream.assert_called_once_with(current_stream)
    current_stream.wait_stream.assert_called_once_with(update_stream)
    runner._update_draft_attention_tasks.assert_called_once_with(entry)
    entry.graph.replay.assert_called_once_with()
    assert runner.last_draft_execution.mode == "replay"
    assert runner.last_draft_execution.replay_executed


def test_draft_graph_can_update_tasks_inline_on_cann_runtimes_that_require_it():
    runner, entry, update_stream = _draft_replay_runner()
    current_stream = MagicMock()
    with (
        patch.dict(
            os.environ,
            {"VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE": "1"},
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.current_stream",
            return_value=current_stream,
        ),
    ):
        output = _run_draft_replay(runner)

    assert output.tolist() == [[4, 5]]
    update_stream.wait_stream.assert_not_called()
    current_stream.wait_stream.assert_not_called()
    runner._update_draft_attention_tasks.assert_called_once_with(
        entry,
        stream=current_stream,
    )
    entry.graph.replay.assert_called_once_with()
    assert runner.last_draft_execution.mode == "replay"
    assert runner.last_draft_execution.replay_executed


def test_draft_graph_replay_first_orders_copy_dependency_before_task_update():
    runner, entry, update_stream = _draft_replay_runner()
    current_stream = MagicMock()
    copy_done_event = MagicMock()
    entry.tasks_per_step = 1
    entry.positions = [torch.tensor([2]), torch.tensor([3])]
    entry.tasks = [
        SimpleNamespace(event=MagicMock()),
        SimpleNamespace(event=MagicMock()),
    ]
    entry.replay_first_copy_done_event = copy_done_event

    calls = MagicMock()
    runner._copy_draft_inputs.side_effect = lambda *args: calls.copy()
    copy_done_event.record.side_effect = lambda *args: calls.record()
    update_stream.wait_event.side_effect = lambda *args: calls.wait()
    entry.graph.replay.side_effect = lambda: calls.replay()
    runner._update_draft_attention_tasks.side_effect = lambda *args, **kwargs: calls.update()

    with (
        patch.dict(
            os.environ,
            {
                "VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE": "0",
                "VLLM_ASCEND_PEARL_DRAFT_REPLAY_FIRST_TASK_UPDATE": "1",
            },
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.current_stream",
            return_value=current_stream,
        ),
    ):
        output = _run_draft_replay(runner)

    assert output.tolist() == [[4, 5]]
    assert calls.mock_calls == [
        call.copy(),
        call.record(),
        call.wait(),
        call.replay(),
        call.update(),
    ]
    copy_done_event.record.assert_called_once_with(current_stream)
    update_stream.wait_event.assert_called_once_with(copy_done_event)
    current_stream.wait_stream.assert_not_called()
    runner._update_draft_attention_tasks.assert_called_once_with(entry)
    assert runner.last_draft_execution.mode == "replay"
    assert runner.last_draft_execution.replay_executed


def test_draft_graph_replay_first_fails_closed_after_graph_submission():
    runner, entry, _ = _draft_replay_runner()
    current_stream = MagicMock()
    copy_done_event = MagicMock()
    entry.tasks_per_step = 1
    entry.positions = [torch.tensor([2]), torch.tensor([3])]
    entry.tasks = [
        SimpleNamespace(event=MagicMock()),
        SimpleNamespace(event=MagicMock()),
    ]
    entry.replay_first_copy_done_event = copy_done_event
    runner._update_draft_attention_tasks.side_effect = ValueError("update failed")

    with (
        patch.dict(
            os.environ,
            {
                "VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE": "0",
                "VLLM_ASCEND_PEARL_DRAFT_REPLAY_FIRST_TASK_UPDATE": "1",
            },
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.current_stream",
            return_value=current_stream,
        ),
        pytest.raises(
            RuntimeError,
            match="cannot safely fall back",
        ),
    ):
        _run_draft_replay(runner)

    entry.graph.replay.assert_called_once_with()


def test_generic_target_replay_first_orders_graph_before_task_update():
    runner, entry, update_stream = _generic_replay_runner()
    current_stream = MagicMock()
    copy_done_event = entry.replay_first_copy_done_event

    calls = MagicMock()
    runner._copy_inputs.side_effect = lambda *args: calls.copy()
    copy_done_event.record.side_effect = lambda *args: calls.record()
    update_stream.wait_event.side_effect = lambda *args: calls.wait()
    entry.graph.replay.side_effect = lambda: calls.replay()
    runner._update_attention_tasks.side_effect = (
        lambda *args, **kwargs: calls.update()
    )

    with (
        patch.dict(
            os.environ,
            {
                "VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE": "0",
                "VLLM_ASCEND_PEARL_TARGET_REPLAY_FIRST_TASK_UPDATE": "1",
                "VLLM_ASCEND_PEARL_SYNC_GRAPH_INPUTS": "0",
                "VLLM_ASCEND_PEARL_SYNC_GRAPH_TASK_UPDATE": "0",
                "VLLM_ASCEND_PEARL_SYNC_GRAPH_REPLAY": "0",
            },
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.current_stream",
            return_value=current_stream,
        ),
    ):
        output = _run_generic_replay(runner)

    assert output.tolist() == [[4, 5]]
    assert calls.mock_calls == [
        call.copy(),
        call.record(),
        call.wait(),
        call.replay(),
        call.update(),
    ]
    copy_done_event.record.assert_called_once_with(current_stream)
    update_stream.wait_event.assert_called_once_with(copy_done_event)
    update_stream.wait_stream.assert_not_called()
    current_stream.wait_stream.assert_not_called()
    runner._update_attention_tasks.assert_called_once_with(entry)
    assert runner.last_generic_execution.mode == "replay"
    assert runner.last_generic_execution.replay_executed
    assert runner.replay_count == 1
    metrics = runner.graph_execution_metrics()
    assert metrics["generic_replay_calls"] == 1
    assert metrics["generic_task_update_replays"] == 1
    assert metrics["generic_task_update_tasks"] == 2


def test_generic_target_replay_first_is_opt_in():
    runner, entry, update_stream = _generic_replay_runner()
    current_stream = MagicMock()
    calls = MagicMock()
    runner._copy_inputs.side_effect = lambda *args: calls.copy()
    update_stream.wait_stream.side_effect = lambda *args: calls.forward_wait()
    runner._update_attention_tasks.side_effect = (
        lambda *args, **kwargs: calls.update()
    )
    current_stream.wait_stream.side_effect = lambda *args: calls.reverse_wait()
    entry.graph.replay.side_effect = lambda: calls.replay()

    with (
        patch.dict(
            os.environ,
            {
                "VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE": "0",
                "VLLM_ASCEND_PEARL_TARGET_REPLAY_FIRST_TASK_UPDATE": "0",
                "VLLM_ASCEND_PEARL_SYNC_GRAPH_INPUTS": "0",
                "VLLM_ASCEND_PEARL_SYNC_GRAPH_TASK_UPDATE": "0",
                "VLLM_ASCEND_PEARL_SYNC_GRAPH_REPLAY": "0",
            },
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.current_stream",
            return_value=current_stream,
        ),
    ):
        output = _run_generic_replay(runner)

    assert output.tolist() == [[4, 5]]
    assert calls.mock_calls == [
        call.copy(),
        call.forward_wait(),
        call.update(),
        call.reverse_wait(),
        call.replay(),
    ]
    update_stream.wait_event.assert_not_called()
    entry.replay_first_copy_done_event.record.assert_not_called()


def test_generic_target_replay_first_fails_closed_after_graph_submission():
    runner, entry, update_stream = _generic_replay_runner()
    current_stream = MagicMock()
    runner._update_attention_tasks.side_effect = ValueError("update failed")

    with (
        patch.dict(
            os.environ,
            {
                "VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE": "0",
                "VLLM_ASCEND_PEARL_TARGET_REPLAY_FIRST_TASK_UPDATE": "1",
                "VLLM_ASCEND_PEARL_SYNC_GRAPH_INPUTS": "0",
                "VLLM_ASCEND_PEARL_SYNC_GRAPH_TASK_UPDATE": "0",
                "VLLM_ASCEND_PEARL_SYNC_GRAPH_REPLAY": "0",
            },
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.current_stream",
            return_value=current_stream,
        ),
        pytest.raises(RuntimeError, match="cannot safely fall back"),
    ):
        _run_generic_replay(runner)

    entry.graph.replay.assert_called_once_with()
    update_stream.wait_event.assert_called_once_with(
        entry.replay_first_copy_done_event
    )
    assert runner.replay_count == 0
    metrics = runner.graph_execution_metrics()
    assert metrics["generic_replay_calls"] == 0
    assert metrics["generic_task_update_replays"] == 0
    assert metrics["generic_task_update_tasks"] == 0


def test_generic_target_replay_first_is_mutually_exclusive_with_inline_update():
    runner, entry, _ = _generic_replay_runner()
    with (
        patch.dict(
            os.environ,
            {
                "VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE": "1",
                "VLLM_ASCEND_PEARL_TARGET_REPLAY_FIRST_TASK_UPDATE": "1",
            },
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.current_stream",
            return_value=MagicMock(),
        ),
        pytest.raises(RuntimeError, match="mutually exclusive"),
    ):
        _run_generic_replay(runner)

    runner._copy_inputs.assert_not_called()
    runner._update_attention_tasks.assert_not_called()
    entry.graph.replay.assert_not_called()


def test_draft_graph_first_changed_input_replay_is_exactly_validated():
    runner, entry, _ = _draft_replay_runner()
    entry.runtime_validated = False
    entry.validated_real_row_count = 1
    runner._draft_inputs_changed = MagicMock(return_value=True)
    runner._execute_draft = MagicMock(return_value=torch.tensor([[4, 5]]))
    runner._draft_outputs_match = MagicMock(return_value=True)
    with (
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.current_stream",
            return_value=MagicMock(),
        ),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.synchronize"),
    ):
        output = _run_draft_replay(runner)

    assert output.tolist() == [[4, 5]]
    assert entry.runtime_validated
    assert runner.runtime_validation_replay_count == 1
    assert runner.execution_counters["draft"]["runtime_validation_calls"] == 1
    assert runner.execution_counters["draft"]["changed_input_validation_calls"] == 1
    assert runner.execution_counters["draft"]["replay_calls"] == 1


def test_draft_graph_logical_row_expansion_is_revalidated_once():
    runner, entry, _ = _draft_replay_runner()
    del runner.draft_entries[("draft-greedy:8|steps:2|paged", 1)]
    runner.draft_entries[("draft-greedy:8|steps:2|paged", 2)] = entry
    entry.output = torch.tensor([[4, 5], [6, 7]])
    entry.runtime_validated = True
    entry.validated_real_row_count = 1
    runner._draft_inputs_changed = MagicMock(return_value=False)
    runner._execute_draft = MagicMock(return_value=entry.output.clone())
    runner._draft_outputs_match = MagicMock(return_value=True)
    metadata = SimpleNamespace(use_fused_infer_attention=False)
    with (
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.current_stream",
            return_value=MagicMock(),
        ),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.synchronize"),
    ):
        output = runner.run_draft_greedy(
            torch.tensor([3, 4]),
            [torch.tensor([2, 2]), torch.tensor([3, 3])],
            [metadata, metadata],
            vocabulary_size=8,
            valid_row_count=2,
        )

    assert output.tolist() == [[4, 5], [6, 7]]
    assert entry.validated_real_row_count == 2
    counters = runner.execution_counters["draft"]
    assert counters["runtime_validation_calls"] == 1
    assert counters["changed_input_validation_calls"] == 0
    assert counters["logical_row_expansion_validation_calls"] == 1


def test_paged_attention_task_update_refreshes_workspace_for_current_lengths():
    runner = NativeACLGraphRunner(MagicMock(), enabled=False)
    update_stream = MagicMock()
    new_workspace = MagicMock(name="new_workspace")
    task = _paged_task()
    with (
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.stream",
            return_value=MagicMock(),
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch_npu._npu_paged_attention_get_workspace",
            return_value=new_workspace,
        ) as get_workspace,
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch_npu._npu_paged_attention"
        ) as paged_attention,
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_update_begin"
        ) as update_begin,
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_update_end"
        ) as update_end,
    ):
        runner._update_attention_task_list(
            [task],
            [((), ())],
            stream=update_stream,
        )

    get_workspace.assert_called_once_with(
        query=task.query,
        key_cache=task.key_cache,
        value_cache=task.value_cache,
        num_kv_heads=task.num_kv_heads,
        num_heads=task.num_heads,
        scale_value=task.scale,
        block_table=task.block_table,
        context_lens=task.context_lens,
        out=task.output,
    )
    assert task.workspace is new_workspace
    update_begin.assert_called_once_with(update_stream, task.handle)
    assert paged_attention.call_args.kwargs["workspace"] is new_workspace
    update_end.assert_called_once_with(update_stream)
    task.event.record.assert_called_once_with(update_stream)


def test_paged_attention_task_update_shares_workspace_only_for_equal_lengths():
    runner = NativeACLGraphRunner(MagicMock(), enabled=False)
    update_stream = MagicMock()
    first = _paged_task(context_lens=(8, 8))
    second = _paged_task(context_lens=(8, 8))
    third = _paged_task(context_lens=(9, 9))
    workspaces = [MagicMock(name="workspace_8"), MagicMock(name="workspace_9")]
    with (
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.stream",
            return_value=MagicMock(),
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch_npu._npu_paged_attention_get_workspace",
            side_effect=workspaces,
        ) as get_workspace,
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch_npu._npu_paged_attention"
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_update_begin"
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_update_end"
        ),
    ):
        runner._update_attention_task_list(
            [first, second, third],
            [((), ()), ((), ()), ((), ())],
            stream=update_stream,
        )

    assert get_workspace.call_count == 2
    assert first.workspace is workspaces[0]
    assert second.workspace is workspaces[0]
    assert third.workspace is workspaces[1]
