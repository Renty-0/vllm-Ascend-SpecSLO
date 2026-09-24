# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch

import vllm_ascend.spec_decode.pearl.native_graph as native_graph
from vllm_ascend.spec_decode.pearl.native_graph import (
    NativeACLGraphEntry,
    NativeACLGraphRunner,
    NativeDraftACLGraphEntry,
    NativeFusedInferAttentionGraphTask,
    NativePagedAttentionGraphTask,
)


def _exact_stable_fia_entry(attention_mask):
    entry = NativeACLGraphEntry(
        input_ids=torch.zeros(4, dtype=torch.long),
        positions=torch.zeros(4, dtype=torch.long),
        slot_mapping=torch.zeros(4, dtype=torch.int32),
        context_lens=torch.zeros(4, dtype=torch.int32),
        block_tables=torch.zeros((2, 2), dtype=torch.int32),
        request_block_tables=torch.zeros((2, 2), dtype=torch.int32),
        actual_seq_lengths_q=(2, 4),
        sequence_lens=(4, 6),
        graph=MagicMock(),
        output=torch.tensor([10, 11, 12, 13]),
        tasks=[SimpleNamespace(event=MagicMock())],
        attention_mask=attention_mask.clone(),
        runtime_validated=True,
        validated_real_row_count=4,
    )
    NativeACLGraphRunner._provision_stable_fia_staging(
        entry,
        SimpleNamespace(attention_mask=attention_mask),
    )
    return entry


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
    entry.positions = (torch.tensor([2]), torch.tensor([3]))
    entry.tasks = [SimpleNamespace(event=MagicMock()), SimpleNamespace(event=MagicMock())]
    entry.tasks_per_step = 1
    entry.taskless_device_attention = False
    entry.actual_seq_lengths_q = ((), ())
    entry.sequence_lens = ((), ())
    runner.draft_entries[("draft-greedy:8|steps:2|paged", 1)] = entry
    runner._copy_draft_inputs = MagicMock()
    runner._update_draft_attention_tasks = MagicMock()
    return runner, entry, update_stream


def _stable_task_barrier_draft_replay_runner():
    update_stream = MagicMock(name="update_stream")
    with patch(
        "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream",
        return_value=update_stream,
    ):
        runner = NativeACLGraphRunner(MagicMock(), enabled=True)
    barrier = MagicMock(name="stable_task_barrier")
    copy_done_event = MagicMock(name="copy_done_event")
    entry = SimpleNamespace(
        output=torch.tensor([[4, 5, 6, 7]]),
        graph=MagicMock(name="draft_graph"),
        actual_seq_lengths_q=((1,),) * 4,
        sequence_lens=((3,),) * 4,
        positions=tuple(torch.tensor([step + 2]) for step in range(4)),
        tasks=[SimpleNamespace(event=barrier) for _ in range(112)],
        tasks_per_step=28,
        runtime_validated=True,
        validated_real_row_count=1,
        replay_first_copy_done_event=copy_done_event,
        stable_task_barrier=True,
        stable_task_barrier_event=barrier,
    )
    entry_key = (
        "draft-greedy:8|steps:4|fia:1|stable-task-barrier|kv-cap:3",
        1,
    )
    runner.draft_entries[entry_key] = entry
    runner._copy_draft_inputs = MagicMock()
    runner._update_draft_attention_tasks = MagicMock()
    return runner, entry, update_stream, barrier, copy_done_event


def _run_stable_task_barrier_draft_replay(runner, *, sequence_length=3):
    metadatas = [
        SimpleNamespace(
            use_fused_infer_attention=True,
            tree_attention=False,
            actual_seq_lengths_q=(1,),
            sequence_lens=(sequence_length,),
        )
        for _ in range(4)
    ]
    return runner.run_draft_greedy(
        torch.tensor([3]),
        [torch.tensor([step + 2]) for step in range(4)],
        metadatas,
        vocabulary_size=8,
        stable_task_barrier=True,
    )


def test_draft_graph_output_carries_real_per_step_confidence() -> None:
    runner = NativeACLGraphRunner.__new__(NativeACLGraphRunner)
    model = MagicMock()
    model.side_effect = [torch.tensor([[1.0]]), torch.tensor([[2.0]])]
    model.compute_greedy_tokens_with_confidence.side_effect = [
        (torch.tensor([11]), torch.tensor([0.25])),
        (torch.tensor([12]), torch.tensor([0.875])),
    ]
    runner.model = model

    output = runner._execute_draft(
        torch.tensor([10]),
        [torch.tensor([1]), torch.tensor([2])],
        [SimpleNamespace(), SimpleNamespace()],
        vocabulary_size=32,
        return_confidence=True,
    )

    assert output.dtype == torch.long
    assert output.tolist() == [[[11, 250000], [12, 875000]]]
    assert model.call_args_list[1].args[0].tolist() == [11]


@pytest.mark.parametrize("priority", [-1, 0])
def test_graph_update_stream_uses_configured_priority(priority):
    with (
        patch.object(
            __import__(
                "vllm_ascend.spec_decode.pearl.native_graph",
                fromlist=["envs"],
            ).envs,
            "VLLM_ASCEND_PEARL_GRAPH_UPDATE_STREAM_PRIORITY",
            priority,
        ),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream") as stream,
    ):
        runner = NativeACLGraphRunner(MagicMock(), enabled=True)

    stream.assert_called_once_with(priority=priority)
    assert runner.update_stream_priority == priority


def test_graph_update_stream_rejects_unknown_priority():
    with (
        patch.object(
            __import__(
                "vllm_ascend.spec_decode.pearl.native_graph",
                fromlist=["envs"],
            ).envs,
            "VLLM_ASCEND_PEARL_GRAPH_UPDATE_STREAM_PRIORITY",
            -2,
        ),
        pytest.raises(ValueError, match="must be -1 or 0"),
    ):
        NativeACLGraphRunner(MagicMock(), enabled=True)


@pytest.mark.parametrize("group_size", [1, 2, 4])
def test_target_fia_task_event_group_size_is_validated_at_runner_init(
    group_size,
):
    envs = __import__("vllm_ascend.spec_decode.pearl.native_graph", fromlist=["envs"]).envs
    with (
        patch.object(
            envs,
            "VLLM_ASCEND_PEARL_TARGET_FIA_TASK_EVENT_GROUP_SIZE",
            group_size,
        ),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream"),
    ):
        runner = NativeACLGraphRunner(MagicMock(), enabled=True)

    assert runner.target_fia_task_event_group_size == group_size
    assert runner.graph_execution_metrics()["target_fia_task_event_group_size"] == group_size


def test_target_fia_task_event_group_size_rejects_unknown_value():
    envs = __import__("vllm_ascend.spec_decode.pearl.native_graph", fromlist=["envs"]).envs
    with (
        patch.object(
            envs,
            "VLLM_ASCEND_PEARL_TARGET_FIA_TASK_EVENT_GROUP_SIZE",
            3,
        ),
        pytest.raises(ValueError, match="must be 1, 2, or 4"),
    ):
        NativeACLGraphRunner(MagicMock(), enabled=False)


def test_target_fia_task_prefix_event_group_is_validated_at_runner_init():
    envs = __import__("vllm_ascend.spec_decode.pearl.native_graph", fromlist=["envs"]).envs
    with (
        patch.object(envs, "VLLM_ASCEND_PEARL_TARGET_FIA_TASK_EVENT_GROUP_SIZE", 4),
        patch.object(
            envs,
            "VLLM_ASCEND_PEARL_TARGET_FIA_TASK_PREFIX_EVENT_GROUP_SIZE",
            2,
        ),
    ):
        runner = NativeACLGraphRunner(MagicMock(), enabled=False)

    assert runner.target_fia_task_prefix_event_group_size == 2
    assert runner.graph_execution_metrics()["target_fia_task_prefix_event_group_size"] == 2


def test_target_fia_task_prefix_event_group_must_be_shorter_than_steady_group():
    envs = __import__("vllm_ascend.spec_decode.pearl.native_graph", fromlist=["envs"]).envs
    with (
        patch.object(envs, "VLLM_ASCEND_PEARL_TARGET_FIA_TASK_EVENT_GROUP_SIZE", 2),
        patch.object(
            envs,
            "VLLM_ASCEND_PEARL_TARGET_FIA_TASK_PREFIX_EVENT_GROUP_SIZE",
            2,
        ),
        pytest.raises(ValueError, match="must be smaller"),
    ):
        NativeACLGraphRunner(MagicMock(), enabled=False)


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_target_fia_task_update_worker_count_is_validated_at_runner_init(
    workers,
):
    envs = __import__("vllm_ascend.spec_decode.pearl.native_graph", fromlist=["envs"]).envs
    with (
        patch.object(
            envs,
            "VLLM_ASCEND_PEARL_TARGET_FIA_TASK_UPDATE_WORKERS",
            workers,
        ),
        patch.object(
            envs,
            "VLLM_ASCEND_PEARL_TARGET_FIA_TASK_EVENT_GROUP_SIZE",
            1,
        ),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream") as stream,
    ):
        runner = NativeACLGraphRunner(
            MagicMock(),
            enabled=True,
            is_target_worker=True,
        )

    assert runner.target_fia_task_update_workers == workers
    assert stream.call_count == workers
    if runner._target_fia_update_executor is not None:
        runner._target_fia_update_executor.shutdown()


@pytest.mark.parametrize("group_size", [2, 4])
def test_parallel_target_fia_task_update_accepts_complete_shared_event_groups(
    group_size,
):
    envs = __import__("vllm_ascend.spec_decode.pearl.native_graph", fromlist=["envs"]).envs
    with (
        patch.object(
            envs,
            "VLLM_ASCEND_PEARL_TARGET_FIA_TASK_UPDATE_WORKERS",
            2,
        ),
        patch.object(
            envs,
            "VLLM_ASCEND_PEARL_TARGET_FIA_TASK_EVENT_GROUP_SIZE",
            group_size,
        ),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream") as stream,
    ):
        runner = NativeACLGraphRunner(
            MagicMock(),
            enabled=True,
            is_target_worker=True,
        )

    assert runner.target_fia_task_update_workers == 2
    assert runner.target_fia_task_event_group_size == group_size
    assert stream.call_count == 2
    assert runner._target_fia_update_executor is not None
    runner._target_fia_update_executor.shutdown()


def test_shared_graph_pool_is_created_only_when_opted_in():
    envs = __import__("vllm_ascend.spec_decode.pearl.native_graph", fromlist=["envs"]).envs
    pool = object()
    with (
        patch.object(envs, "VLLM_ASCEND_PEARL_SHARED_GRAPH_POOL", True),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream"),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_pool_handle",
            return_value=pool,
        ) as pool_handle,
    ):
        runner = NativeACLGraphRunner(MagicMock(), enabled=True)

    pool_handle.assert_called_once_with()
    assert runner.shared_graph_pool_enabled
    assert runner.graph_pool is pool
    assert runner.graph_qualification_status()["shared_memory_pool"] == 1


def test_shared_graph_pool_is_not_created_for_eager_runner():
    envs = __import__("vllm_ascend.spec_decode.pearl.native_graph", fromlist=["envs"]).envs
    with (
        patch.object(envs, "VLLM_ASCEND_PEARL_SHARED_GRAPH_POOL", True),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_pool_handle") as pool_handle,
    ):
        runner = NativeACLGraphRunner(MagicMock(), enabled=False)

    pool_handle.assert_not_called()
    assert not runner.shared_graph_pool_enabled
    assert runner.graph_pool is None


@pytest.mark.parametrize("priority", [-1, 0])
def test_graph_update_stream_explicit_priority_overrides_environment(priority):
    with (
        patch.object(
            __import__(
                "vllm_ascend.spec_decode.pearl.native_graph",
                fromlist=["envs"],
            ).envs,
            "VLLM_ASCEND_PEARL_GRAPH_UPDATE_STREAM_PRIORITY",
            -2,
        ),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream") as stream,
    ):
        runner = NativeACLGraphRunner(
            MagicMock(),
            enabled=True,
            update_stream_priority=priority,
        )

    stream.assert_called_once_with(priority=priority)
    assert runner.update_stream_priority == priority


def _run_draft_replay(runner):
    metadata = SimpleNamespace(
        use_fused_infer_attention=False,
        actual_seq_lengths_q=(1,),
        sequence_lens=(3,),
    )
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
    entry.taskless_device_attention = False
    entry.tasks = [
        SimpleNamespace(event=MagicMock()),
        SimpleNamespace(event=MagicMock()),
    ]
    entry.replay_first_copy_done_event = MagicMock()
    runner.entries[("hidden", 1)] = entry
    runner._copy_inputs = MagicMock()
    runner._update_attention_tasks = MagicMock()
    return runner, entry, update_stream


def test_generic_taskless_device_attention_skips_all_update_dependencies():
    runner, entry, update_stream = _generic_replay_runner()
    current_stream = MagicMock()
    runner.update_stream = None
    entry.tasks = []
    entry.taskless_device_attention = True

    with (
        patch.dict(
            os.environ,
            {
                # Neither task-update policy applies to taskless device PA;
                # even a globally conflicting pair must not gate its replay.
                "VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE": "1",
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
    runner._copy_inputs.assert_called_once()
    runner._update_attention_tasks.assert_not_called()
    entry.graph.replay.assert_called_once_with()
    update_stream.wait_event.assert_not_called()
    update_stream.wait_stream.assert_not_called()
    current_stream.wait_stream.assert_not_called()
    assert runner.generic_taskless_replays == 1
    assert runner.task_update_skip_replay_count == 0


def test_generic_empty_task_list_without_device_attention_origin_fails_closed():
    runner, entry, _ = _generic_replay_runner()
    entry.tasks = []
    entry.taskless_device_attention = False

    with pytest.raises(RuntimeError, match="not an explicit device-position"):
        _run_generic_replay(runner)

    runner._copy_inputs.assert_not_called()
    entry.graph.replay.assert_not_called()


def _stable_fia_metadata(query_lengths, *, total_tokens):
    cumulative = []
    for length in query_lengths:
        cumulative.append(length + (cumulative[-1] if cumulative else 0))
    assert cumulative[-1] == total_tokens
    return SimpleNamespace(
        use_fused_infer_attention=True,
        actual_seq_lengths_q=tuple(cumulative),
        sequence_lens=tuple(max(1, length) for length in query_lengths),
        request_block_tables=torch.zeros(
            (len(query_lengths), 4),
            dtype=torch.int32,
        ),
        slot_mapping=torch.arange(total_tokens, dtype=torch.int32),
    )


def test_stable_fia_hidden_uses_one_key_for_changed_query_partitions():
    runner = NativeACLGraphRunner.__new__(NativeACLGraphRunner)
    runner._run = MagicMock(return_value=torch.zeros((197, 8)))
    input_ids = torch.zeros(197, dtype=torch.long)
    positions = torch.arange(197, dtype=torch.long)
    first_lengths = [4] * 32 + [20, 1, 1, 1, 46]
    second_lengths = [4] * 32 + [31, 32, 1, 1, 4]

    envs = __import__(
        "vllm_ascend.spec_decode.pearl.native_graph",
        fromlist=["envs"],
    ).envs
    with patch.object(
        envs,
        "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH",
        True,
    ):
        for lengths in (first_lengths, second_lengths):
            runner.run_stable_fia_hidden(
                input_ids,
                positions,
                _stable_fia_metadata(lengths, total_tokens=197),
                graph_key="mixed-target|prompt-tokens:64",
                expected_tokens=197,
                expected_request_segments=37,
            )

    assert runner._run.call_count == 2
    for call_args in runner._run.call_args_list:
        assert call_args.kwargs["stable_fia_key"] == ("mixed-target|prompt-tokens:64")
        assert call_args.kwargs["stable_fia_capture_size"] == 197
        assert call_args.kwargs["output_kind"] == "hidden"
        assert call_args.kwargs["output_transform"] is None


def test_stable_fia_greedy_captures_token_selection_in_the_same_graph():
    runner = NativeACLGraphRunner.__new__(NativeACLGraphRunner)
    runner.model = MagicMock()
    runner.model.compute_greedy_tokens.return_value = torch.arange(197)
    runner._run = MagicMock(return_value=torch.arange(197))
    input_ids = torch.zeros(197, dtype=torch.long)
    positions = torch.arange(197, dtype=torch.long)
    metadata = _stable_fia_metadata(
        [4] * 32 + [20, 1, 1, 1, 46],
        total_tokens=197,
    )

    envs = __import__(
        "vllm_ascend.spec_decode.pearl.native_graph",
        fromlist=["envs"],
    ).envs
    with patch.object(
        envs,
        "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH",
        True,
    ):
        output = runner.run_stable_fia_greedy(
            input_ids,
            positions,
            metadata,
            151936,
            graph_key="stable-target-verify|width:4|verify:32",
            expected_tokens=197,
            expected_request_segments=37,
        )

    assert torch.equal(output, torch.arange(197))
    call_args = runner._run.call_args
    assert call_args.kwargs["output_kind"] == "greedy:151936"
    assert call_args.kwargs["stable_fia_key"] == ("stable-target-verify|width:4|verify:32")
    assert call_args.kwargs["stable_fia_capture_size"] == 197
    hidden = torch.randn(197, 8)
    assert torch.equal(
        call_args.kwargs["output_transform"](hidden),
        torch.arange(197),
    )
    runner.model.compute_greedy_tokens.assert_called_once_with(hidden, 151936)


def test_exact_stable_fia_sealed_service_reuses_graph_owned_staging():
    update_stream = MagicMock()
    current_stream = MagicMock()
    model = MagicMock()
    attention_mask = torch.zeros((4, 4), dtype=torch.int8)
    with patch(
        "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream",
        return_value=update_stream,
    ):
        runner = NativeACLGraphRunner(model, enabled=True)
    entry = _exact_stable_fia_entry(attention_mask)
    key = (
        "greedy:64|fia-stable:stable-target-verify-hidden|width:2|verify:2",
        4,
    )
    runner.entries[key] = entry
    runner.graph_cache_sealed = True
    positions_staging_id = id(entry.stable_positions_cpu)
    slots_staging_id = id(entry.stable_slot_mapping_cpu)
    segment_staging_id = id(entry.stable_segment_ids_device)
    cache_block_tables = torch.tensor(
        [[10, 11], [20, 21], [30, 31]],
        dtype=torch.int32,
    )
    envs = __import__(
        "vllm_ascend.spec_decode.pearl.native_graph",
        fromlist=["envs"],
    ).envs

    with (
        patch.object(
            envs,
            "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH",
            True,
        ),
        patch.object(
            envs,
            "VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE",
            False,
        ),
        patch.object(
            envs,
            "VLLM_ASCEND_PEARL_TARGET_REPLAY_FIRST_TASK_UPDATE",
            False,
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.current_stream",
            return_value=current_stream,
        ),
        patch.object(runner, "_copy_inputs") as copy_inputs,
        patch.object(runner, "_update_attention_tasks"),
    ):
        first = runner.run_stable_fia_greedy_staged(
            torch.tensor([1, 2, 3, 4]),
            64,
            positions=(3, 4, 8, 9),
            slot_mapping=(13, 14, 28, 29),
            actual_seq_lengths_q=(2, 4),
            sequence_lens=(5, 10),
            segment_sequence_ids=(2, 0),
            cache_block_tables=cache_block_tables,
            attention_mask=attention_mask,
            graph_key="stable-target-verify-hidden|width:2|verify:2",
            expected_tokens=4,
            expected_request_segments=2,
        )
        second = runner.run_stable_fia_greedy_staged(
            torch.tensor([5, 6, 7, 8]),
            64,
            positions=(4, 5, 9, 10),
            slot_mapping=(14, 15, 29, 30),
            actual_seq_lengths_q=(2, 4),
            sequence_lens=(6, 11),
            segment_sequence_ids=(1, 2),
            cache_block_tables=cache_block_tables,
            attention_mask=attention_mask,
            graph_key="stable-target-verify-hidden|width:2|verify:2",
            expected_tokens=4,
            expected_request_segments=2,
        )

    assert first.tolist() == [10, 11, 12, 13]
    assert second.tolist() == [10, 11, 12, 13]
    assert entry.input_ids.tolist() == [5, 6, 7, 8]
    assert entry.positions.tolist() == [4, 5, 9, 10]
    assert entry.slot_mapping.tolist() == [14, 15, 29, 30]
    assert entry.context_lens.tolist() == [5, 6, 10, 11]
    assert entry.request_block_tables.tolist() == [[20, 21], [30, 31]]
    assert id(entry.stable_positions_cpu) == positions_staging_id
    assert id(entry.stable_slot_mapping_cpu) == slots_staging_id
    assert id(entry.stable_segment_ids_device) == segment_staging_id
    copy_inputs.assert_not_called()
    assert entry.graph.replay.call_count == 2


@pytest.mark.parametrize("broken_proof", ["unsealed", "unvalidated", "mask"])
def test_exact_stable_fia_staging_fails_closed_before_buffer_mutation(
    broken_proof,
):
    attention_mask = torch.zeros((4, 4), dtype=torch.int8)
    runner = NativeACLGraphRunner.__new__(NativeACLGraphRunner)
    runner.enabled = True
    runner.graph_cache_sealed = True
    runner.disabled_entry_keys = set()
    runner.entries = {}
    entry = _exact_stable_fia_entry(attention_mask)
    key = (
        "greedy:64|fia-stable:stable-target-verify-hidden|width:2|verify:2",
        4,
    )
    runner.entries[key] = entry
    runner._run = MagicMock()
    if broken_proof == "unsealed":
        runner.graph_cache_sealed = False
    elif broken_proof == "unvalidated":
        entry.runtime_validated = False
    else:
        attention_mask.add_(1)
    original_inputs = entry.input_ids.clone()
    envs = __import__(
        "vllm_ascend.spec_decode.pearl.native_graph",
        fromlist=["envs"],
    ).envs

    with (
        patch.object(
            envs,
            "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH",
            True,
        ),
        pytest.raises(RuntimeError),
    ):
        runner.run_stable_fia_greedy_staged(
            torch.tensor([1, 2, 3, 4]),
            64,
            positions=(3, 4, 8, 9),
            slot_mapping=(13, 14, 28, 29),
            actual_seq_lengths_q=(2, 4),
            sequence_lens=(5, 10),
            segment_sequence_ids=(2, 0),
            cache_block_tables=torch.zeros((3, 2), dtype=torch.int32),
            attention_mask=attention_mask,
            graph_key="stable-target-verify-hidden|width:2|verify:2",
            expected_tokens=4,
            expected_request_segments=2,
        )

    assert torch.equal(entry.input_ids, original_inputs)
    runner._run.assert_not_called()


@pytest.mark.parametrize(
    "mutation",
    [
        "token_count",
        "slot_count",
        "segment_count",
        "non_monotonic_q",
        "request_table_rows",
        "non_positive_kv",
        "not_fia",
    ],
)
def test_stable_fia_hidden_validates_atomically_before_graph_call(mutation):
    runner = NativeACLGraphRunner.__new__(NativeACLGraphRunner)
    runner._run = MagicMock()
    total_tokens = 197
    input_ids = torch.zeros(total_tokens, dtype=torch.long)
    positions = torch.arange(total_tokens, dtype=torch.long)
    lengths = [4] * 32 + [20, 1, 1, 1, 46]
    metadata = _stable_fia_metadata(lengths, total_tokens=total_tokens)
    if mutation == "token_count":
        input_ids = input_ids[:-1]
    elif mutation == "slot_count":
        metadata.slot_mapping = metadata.slot_mapping[:-1]
    elif mutation == "segment_count":
        metadata.actual_seq_lengths_q = metadata.actual_seq_lengths_q[:-1]
        metadata.sequence_lens = metadata.sequence_lens[:-1]
        metadata.request_block_tables = metadata.request_block_tables[:-1]
    elif mutation == "non_monotonic_q":
        values = list(metadata.actual_seq_lengths_q)
        values[-2] = values[-3]
        metadata.actual_seq_lengths_q = tuple(values)
    elif mutation == "request_table_rows":
        metadata.request_block_tables = metadata.request_block_tables[:-1]
    elif mutation == "non_positive_kv":
        metadata.sequence_lens = (*metadata.sequence_lens[:-1], 0)
    elif mutation == "not_fia":
        metadata.use_fused_infer_attention = False

    envs = __import__(
        "vllm_ascend.spec_decode.pearl.native_graph",
        fromlist=["envs"],
    ).envs
    with (
        patch.object(
            envs,
            "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH",
            True,
        ),
        pytest.raises(ValueError, match="Stable FIA"),
    ):
        runner.run_stable_fia_hidden(
            input_ids,
            positions,
            metadata,
            graph_key="mixed-target|prompt-tokens:64",
            expected_tokens=total_tokens,
            expected_request_segments=37,
        )
    runner._run.assert_not_called()


def test_stable_fia_hidden_is_independently_opt_in():
    runner = NativeACLGraphRunner.__new__(NativeACLGraphRunner)
    runner._run = MagicMock()
    total_tokens = 197
    metadata = _stable_fia_metadata(
        [4] * 32 + [20, 1, 1, 1, 46],
        total_tokens=total_tokens,
    )
    envs = __import__(
        "vllm_ascend.spec_decode.pearl.native_graph",
        fromlist=["envs"],
    ).envs
    with (
        patch.object(
            envs,
            "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH",
            False,
        ),
        pytest.raises(RuntimeError, match="explicit.*opt-in"),
    ):
        runner.run_stable_fia_hidden(
            torch.zeros(total_tokens, dtype=torch.long),
            torch.arange(total_tokens, dtype=torch.long),
            metadata,
            graph_key="mixed-target|prompt-tokens:64",
            expected_tokens=total_tokens,
            expected_request_segments=37,
        )
    runner._run.assert_not_called()


def _run_generic_replay(runner):
    metadata = SimpleNamespace(
        use_fused_infer_attention=False,
        actual_seq_lengths_q=(1, 2),
        sequence_lens=(3, 3),
    )
    return runner(
        torch.tensor([3]),
        torch.tensor([2]),
        metadata,
    )


@pytest.mark.parametrize(
    ("inline_update", "replay_first_update"),
    [(False, False), (True, False), (False, True)],
)
def test_generic_graph_stable_host_lengths_skip_task_rebuild_in_all_modes(
    inline_update,
    replay_first_update,
):
    runner, entry, update_stream = _generic_replay_runner()
    current_stream = MagicMock()
    entry.actual_seq_lengths_q = (1, 2)
    entry.sequence_lens = (3, 3)

    with (
        patch.dict(
            os.environ,
            {
                "VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE": str(int(inline_update)),
                "VLLM_ASCEND_PEARL_TARGET_REPLAY_FIRST_TASK_UPDATE": str(int(replay_first_update)),
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

    runner._update_attention_tasks.assert_not_called()
    event_stream = current_stream if inline_update else update_stream
    for task in entry.tasks:
        task.event.record.assert_called_once_with(event_stream)
    assert runner.task_update_skip_replay_count == 1
    assert runner.task_update_replay_count == 0
    assert runner.task_update_replay_counts["generic"] == 0
    assert runner.task_update_task_counts["generic"] == 0
    entry.graph.replay.assert_called_once_with()
    assert output.tolist() == [[4, 5]]

    if replay_first_update:
        entry.replay_first_copy_done_event.record.assert_called_once_with(current_stream)
        update_stream.wait_event.assert_called_once_with(entry.replay_first_copy_done_event)
        update_stream.wait_stream.assert_not_called()
        current_stream.wait_stream.assert_not_called()
    elif inline_update:
        update_stream.wait_event.assert_not_called()
        update_stream.wait_stream.assert_not_called()
        current_stream.wait_stream.assert_not_called()
    else:
        update_stream.wait_stream.assert_called_once_with(current_stream)
        current_stream.wait_stream.assert_called_once_with(update_stream)


@pytest.mark.parametrize(
    ("inline_update", "replay_first_update"),
    [(False, False), (True, False), (False, True)],
)
def test_generic_graph_changed_host_lengths_rebuild_tasks_in_all_modes(
    inline_update,
    replay_first_update,
):
    runner, entry, _ = _generic_replay_runner()
    current_stream = MagicMock()
    entry.actual_seq_lengths_q = (9,)
    entry.sequence_lens = (9,)

    with (
        patch.dict(
            os.environ,
            {
                "VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE": str(int(inline_update)),
                "VLLM_ASCEND_PEARL_TARGET_REPLAY_FIRST_TASK_UPDATE": str(int(replay_first_update)),
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
        _run_generic_replay(runner)

    if inline_update:
        runner._update_attention_tasks.assert_called_once_with(
            entry,
            stream=current_stream,
        )
    else:
        runner._update_attention_tasks.assert_called_once_with(entry)
    for task in entry.tasks:
        task.event.record.assert_not_called()
    assert runner.task_update_skip_replay_count == 0
    assert runner.task_update_replay_count == 1
    assert runner.task_update_replay_counts["generic"] == 1
    assert runner.task_update_task_counts["generic"] == len(entry.tasks)


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


def _fia_task():
    return NativeFusedInferAttentionGraphTask(
        query=torch.zeros((2, 4, 8)),
        key_cache=torch.zeros((4, 16, 2, 8)),
        value_cache=torch.zeros((4, 16, 2, 8)),
        num_kv_heads=2,
        num_heads=4,
        scale=0.125,
        block_table=torch.zeros((2, 1), dtype=torch.int32),
        attention_mask=torch.zeros((2, 1, 2, 16), dtype=torch.bool),
        output=torch.zeros((2, 4, 8)),
        softmax_lse=torch.zeros(1),
        block_size=16,
        workspace=MagicMock(name="workspace"),
        handle=MagicMock(name="handle"),
        event=MagicMock(name="event"),
    )


@dataclass(frozen=True)
class _PaddingMetadata:
    slot_mapping: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    actual_seq_lengths_q: tuple[int, ...]
    sequence_lens: tuple[int, ...]
    use_fused_infer_attention: bool = False
    sentinel: str = "preserved"


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


def _atomic_draft_copy_case():
    input_ids = torch.tensor([7, 8], dtype=torch.long)
    positions = [
        torch.tensor([20, 21], dtype=torch.long),
        torch.tensor([21, 22], dtype=torch.long),
    ]
    attention_metadatas = []
    captured_positions = []
    captured_slot_mappings = []
    captured_context_lens = []
    captured_block_tables = []
    captured_request_tables = []
    captured_attention_masks = []
    captured_tree_masks = []
    for step in range(2):
        captured_positions.append(torch.full((2,), -10 - step, dtype=torch.long))
        captured_slot_mappings.append(torch.full((2,), -20 - step, dtype=torch.int32))
        captured_context_lens.append(torch.full((2,), 30 + step, dtype=torch.int32))
        captured_block_tables.append(torch.full((2, 2), 40 + step, dtype=torch.int32))
        captured_request_tables.append(torch.full((2, 2), 50 + step, dtype=torch.int32))
        captured_attention_masks.append(torch.zeros((2, 2), dtype=torch.bool))
        captured_tree_masks.append(torch.zeros((2, 1, 1, 32), dtype=torch.bool))
        attention_metadatas.append(
            SimpleNamespace(
                slot_mapping=torch.tensor([100 + step * 2, 101 + step * 2], dtype=torch.int32),
                context_lens=torch.tensor([22 + step, 23 + step], dtype=torch.int32),
                block_tables=torch.tensor(
                    [[60 + step, 61 + step], [62 + step, 63 + step]],
                    dtype=torch.int32,
                ),
                request_block_tables=torch.tensor(
                    [[70 + step, 71 + step], [72 + step, 73 + step]],
                    dtype=torch.int32,
                ),
                attention_mask=torch.ones((2, 2), dtype=torch.bool),
                tree_attention_mask=torch.ones((2, 1, 1, 32), dtype=torch.bool),
                actual_seq_lengths_q=(1, 2),
                sequence_lens=(22 + step, 23 + step),
                use_fused_infer_attention=True,
                tree_attention=True,
            )
        )
    entry = NativeDraftACLGraphEntry(
        input_ids=torch.tensor([-1, -2], dtype=torch.long),
        positions=tuple(captured_positions),
        slot_mappings=tuple(captured_slot_mappings),
        context_lens=tuple(captured_context_lens),
        block_tables=tuple(captured_block_tables),
        request_block_tables=tuple(captured_request_tables),
        actual_seq_lengths_q=((1, 2), (1, 2)),
        sequence_lens=((16, 17), (17, 18)),
        graph=MagicMock(),
        output=torch.zeros((2, 2), dtype=torch.long),
        tasks=[],
        tasks_per_step=0,
        attention_masks=tuple(captured_attention_masks),
        tree_attention_masks=tuple(captured_tree_masks),
        tree_attention_modes=(True, True),
    )
    return entry, input_ids, positions, attention_metadatas


def _snapshot_draft_copy_entry(entry):
    tensor_groups = (
        entry.positions,
        entry.slot_mappings,
        entry.context_lens,
        entry.block_tables,
        entry.request_block_tables,
        entry.attention_masks,
        entry.tree_attention_masks,
    )
    return {
        "input_ids": entry.input_ids.clone(),
        "tensor_groups": tuple(
            tuple(value.clone() if value is not None else None for value in group) for group in tensor_groups
        ),
        "actual_seq_lengths_q": entry.actual_seq_lengths_q,
        "sequence_lens": entry.sequence_lens,
    }


def _assert_draft_copy_entry_unchanged(entry, snapshot):
    assert torch.equal(entry.input_ids, snapshot["input_ids"])
    tensor_groups = (
        entry.positions,
        entry.slot_mappings,
        entry.context_lens,
        entry.block_tables,
        entry.request_block_tables,
        entry.attention_masks,
        entry.tree_attention_masks,
    )
    for current_group, saved_group in zip(tensor_groups, snapshot["tensor_groups"]):
        for current, saved in zip(current_group, saved_group):
            if saved is None:
                assert current is None
            else:
                assert torch.equal(current, saved)
    assert entry.actual_seq_lengths_q == snapshot["actual_seq_lengths_q"]
    assert entry.sequence_lens == snapshot["sequence_lens"]


def test_copy_draft_inputs_updates_a_fully_preflighted_tree_fia_replay():
    entry, input_ids, positions, metadatas = _atomic_draft_copy_case()
    old_context_lens = tuple(value.clone() for value in entry.context_lens)
    old_block_tables = tuple(value.clone() for value in entry.block_tables)

    NativeACLGraphRunner._copy_draft_inputs(entry, input_ids, positions, metadatas)

    assert torch.equal(entry.input_ids, input_ids)
    for step, metadata in enumerate(metadatas):
        assert torch.equal(entry.positions[step], positions[step])
        assert torch.equal(entry.slot_mappings[step], metadata.slot_mapping)
        assert torch.equal(entry.request_block_tables[step], metadata.request_block_tables)
        assert torch.equal(entry.attention_masks[step], metadata.attention_mask)
        assert torch.equal(entry.tree_attention_masks[step], metadata.tree_attention_mask)
        # FIA does not bind these compatibility-only metadata tensors into
        # the replay; its request-major table is the graph-owned page table.
        assert torch.equal(entry.context_lens[step], old_context_lens[step])
        assert torch.equal(entry.block_tables[step], old_block_tables[step])
    assert entry.actual_seq_lengths_q == ((1, 2), (1, 2))
    assert entry.sequence_lens == ((22, 23), (23, 24))


def test_copy_draft_inputs_updates_one_shared_request_table_once():
    entry, input_ids, positions, metadatas = _atomic_draft_copy_case()
    shared_captured_table = entry.request_block_tables[0]
    entry.request_block_tables = (
        shared_captured_table,
        shared_captured_table,
    )
    shared_input_table = metadatas[0].request_block_tables
    metadatas[1].request_block_tables = shared_input_table
    version_before = shared_captured_table._version

    NativeACLGraphRunner._copy_draft_inputs(entry, input_ids, positions, metadatas)

    assert shared_captured_table._version == version_before + 1
    assert torch.equal(shared_captured_table, shared_input_table)


def test_copy_draft_inputs_rejects_inconsistent_shared_request_table_alias():
    entry, input_ids, positions, metadatas = _atomic_draft_copy_case()
    shared_captured_table = entry.request_block_tables[0]
    entry.request_block_tables = (
        shared_captured_table,
        shared_captured_table,
    )
    snapshot = _snapshot_draft_copy_entry(entry)

    with pytest.raises(RuntimeError, match="must also share one replay input"):
        NativeACLGraphRunner._copy_draft_inputs(entry, input_ids, positions, metadatas)

    _assert_draft_copy_entry_unchanged(entry, snapshot)


def test_capture_draft_preserves_shared_request_table_alias():
    shared_request_table = torch.tensor(
        [[7, 8], [9, 10]],
        dtype=torch.int32,
    )
    metadatas = [
        SimpleNamespace(
            slot_mapping=torch.tensor([step * 2, step * 2 + 1]),
            context_lens=torch.tensor([4 + step, 5 + step]),
            block_tables=shared_request_table,
            request_block_tables=shared_request_table,
            attention_mask=None,
            tree_attention=True,
            tree_attention_mask=torch.zeros(
                (2, 1, 1, 16),
                dtype=torch.bool,
            ),
            use_fused_infer_attention=True,
            actual_seq_lengths_q=(1, 2),
            sequence_lens=(8, 8),
        )
        for step in range(2)
    ]
    runner = NativeACLGraphRunner(MagicMock(), enabled=False)
    runner.update_stream = MagicMock()
    output = torch.tensor([[11, 12], [13, 14]])
    runner._execute_draft = MagicMock(return_value=output)
    runner._update_draft_attention_tasks = MagicMock()
    runner._draft_outputs_match = MagicMock(return_value=True)
    graph = MagicMock()
    current_stream = MagicMock()

    with (
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.synchronize"),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.NPUGraph",
            return_value=graph,
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph",
            return_value=MagicMock(),
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.current_stream",
            return_value=current_stream,
        ),
    ):
        runner._capture_draft(
            ("draft-greedy:32|steps:2|fia|tree", 2),
            torch.tensor([1, 2]),
            [torch.tensor([3, 4]), torch.tensor([4, 5])],
            metadatas,
            vocabulary_size=32,
            valid_row_count=2,
        )

    captured_tables = runner.draft_entries[("draft-greedy:32|steps:2|fia|tree", 2)].request_block_tables
    assert captured_tables[0] is captured_tables[1]
    assert captured_tables[0] is not shared_request_table
    assert torch.equal(captured_tables[0], shared_request_table)


@pytest.mark.parametrize(
    "broken_contract",
    [
        "step-count",
        "captured-step-count",
        "input-shape",
        "input-layout",
        "input-dtype",
        "input-device",
        "position-shape",
        "position-dtype",
        "position-device",
        "slot-shape",
        "slot-dtype",
        "slot-device",
        "attention-mask-shape",
        "attention-mask-dtype",
        "attention-mask-device",
        "attention-mask-presence",
        "full-mask-shape",
        "full-mask-dtype",
        "full-mask-device",
        "block-table-shape",
        "block-table-dtype",
        "block-table-device",
        "request-table-shape",
        "request-table-dtype",
        "request-table-device",
        "request-table-presence",
        "fia-mode",
        "tree-mode",
        "query-partition",
        "query-extent",
        "kv-count",
        "kv-value",
        "request-row-count",
        "full-mask-kv-coverage",
    ],
)
def test_copy_draft_inputs_preflight_failure_is_atomic(broken_contract):
    entry, input_ids, positions, metadatas = _atomic_draft_copy_case()

    if broken_contract == "step-count":
        positions.pop()
    elif broken_contract == "captured-step-count":
        entry.tree_attention_modes = (True,)
    elif broken_contract == "input-shape":
        input_ids = torch.tensor([7, 8, 9], dtype=torch.long)
    elif broken_contract == "input-layout":
        entry.input_ids = entry.input_ids.view(1, 2)
        input_ids = input_ids.view(1, 2)
    elif broken_contract == "input-dtype":
        input_ids = input_ids.to(torch.int32)
    elif broken_contract == "input-device":
        input_ids = torch.empty(input_ids.shape, dtype=input_ids.dtype, device="meta")
    elif broken_contract == "position-shape":
        positions[1] = torch.tensor([1, 2, 3], dtype=torch.long)
    elif broken_contract == "position-dtype":
        positions[1] = positions[1].to(torch.int32)
    elif broken_contract == "position-device":
        positions[1] = torch.empty(positions[1].shape, dtype=positions[1].dtype, device="meta")
    elif broken_contract == "slot-shape":
        metadatas[1].slot_mapping = torch.zeros(3, dtype=torch.int32)
    elif broken_contract == "slot-dtype":
        metadatas[1].slot_mapping = metadatas[1].slot_mapping.to(torch.int64)
    elif broken_contract == "slot-device":
        metadatas[1].slot_mapping = torch.empty((2,), dtype=torch.int32, device="meta")
    elif broken_contract == "attention-mask-shape":
        metadatas[1].attention_mask = torch.ones((2, 3), dtype=torch.bool)
    elif broken_contract == "attention-mask-dtype":
        metadatas[1].attention_mask = metadatas[1].attention_mask.to(torch.int8)
    elif broken_contract == "attention-mask-device":
        metadatas[1].attention_mask = torch.empty((2, 2), dtype=torch.bool, device="meta")
    elif broken_contract == "attention-mask-presence":
        metadatas[1].attention_mask = None
    elif broken_contract == "full-mask-shape":
        metadatas[1].tree_attention_mask = torch.ones((2, 1, 2, 32), dtype=torch.bool)
    elif broken_contract == "full-mask-dtype":
        metadatas[1].tree_attention_mask = metadatas[1].tree_attention_mask.to(torch.int8)
    elif broken_contract == "full-mask-device":
        metadatas[1].tree_attention_mask = torch.empty((2, 1, 1, 32), dtype=torch.bool, device="meta")
    elif broken_contract == "block-table-shape":
        metadatas[1].block_tables = torch.zeros((2, 3), dtype=torch.int32)
    elif broken_contract == "block-table-dtype":
        metadatas[1].block_tables = metadatas[1].block_tables.to(torch.int64)
    elif broken_contract == "block-table-device":
        metadatas[1].block_tables = torch.empty((2, 2), dtype=torch.int32, device="meta")
    elif broken_contract == "request-table-shape":
        metadatas[1].request_block_tables = torch.zeros((2, 3), dtype=torch.int32)
    elif broken_contract == "request-table-dtype":
        metadatas[1].request_block_tables = metadatas[1].request_block_tables.to(torch.int64)
    elif broken_contract == "request-table-device":
        metadatas[1].request_block_tables = torch.empty((2, 2), dtype=torch.int32, device="meta")
    elif broken_contract == "request-table-presence":
        metadatas[1].request_block_tables = None
    elif broken_contract == "fia-mode":
        metadatas[1].use_fused_infer_attention = False
    elif broken_contract == "tree-mode":
        metadatas[1].tree_attention = False
    elif broken_contract == "query-partition":
        metadatas[1].actual_seq_lengths_q = (1, 1)
    elif broken_contract == "query-extent":
        metadatas[1].actual_seq_lengths_q = (1, 3)
    elif broken_contract == "kv-count":
        metadatas[1].sequence_lens = (23,)
    elif broken_contract == "kv-value":
        metadatas[1].sequence_lens = (23, 0)
    elif broken_contract == "request-row-count":
        replacement = torch.zeros((1, 2), dtype=torch.int32)
        entry.request_block_tables = (
            entry.request_block_tables[0],
            replacement,
        )
        metadatas[1].request_block_tables = replacement.clone()
    elif broken_contract == "full-mask-kv-coverage":
        replacement = torch.zeros((2, 1, 1, 16), dtype=torch.bool)
        entry.tree_attention_masks = (
            entry.tree_attention_masks[0],
            replacement,
        )
        metadatas[1].tree_attention_mask = replacement.clone()
    else:  # pragma: no cover - keeps new parameter entries fail-closed.
        raise AssertionError(f"Unhandled broken contract: {broken_contract}")

    snapshot = _snapshot_draft_copy_entry(entry)
    with pytest.raises(RuntimeError, match="Draft ACLGraph"):
        NativeACLGraphRunner._copy_draft_inputs(entry, input_ids, positions, metadatas)
    _assert_draft_copy_entry_unchanged(entry, snapshot)


def test_copy_draft_inputs_accepts_finite_suffix_for_paged_graph_padding():
    entry, input_ids, positions, metadatas = _atomic_draft_copy_case()
    entry.taskless_device_attention = True
    entry.request_block_tables = (None, None)
    entry.tree_attention_masks = (None, None)
    entry.tree_attention_modes = (False, False)
    entry.actual_seq_lengths_q = ((1,), (1,))
    entry.sequence_lens = ((11, 1), (12, 1))
    for step, metadata in enumerate(metadatas):
        metadata.request_block_tables = None
        metadata.tree_attention_mask = None
        metadata.tree_attention = False
        metadata.use_fused_infer_attention = False
        metadata.actual_seq_lengths_q = (1,)
        metadata.sequence_lens = (21 + step, 1)

    NativeACLGraphRunner._copy_draft_inputs(entry, input_ids, positions, metadatas)

    assert entry.sequence_lens == ((21, 1), (22, 1))
    for step, metadata in enumerate(metadatas):
        assert torch.equal(entry.context_lens[step], metadata.context_lens)
        assert torch.equal(entry.block_tables[step], metadata.block_tables)


def test_copy_taskless_device_attention_rejects_missing_padding_kv_suffix():
    entry, input_ids, positions, metadatas = _atomic_draft_copy_case()
    entry.taskless_device_attention = True
    entry.request_block_tables = (None, None)
    entry.tree_attention_masks = (None, None)
    entry.tree_attention_modes = (False, False)
    entry.actual_seq_lengths_q = ((1,), (1,))
    entry.sequence_lens = ((11, 1), (12, 1))
    for step, metadata in enumerate(metadatas):
        metadata.request_block_tables = None
        metadata.tree_attention_mask = None
        metadata.tree_attention = False
        metadata.use_fused_infer_attention = False
        metadata.actual_seq_lengths_q = (1,)
        # Missing the exact length-one dummy-row suffix.
        metadata.sequence_lens = (21 + step,)

    snapshot = _snapshot_draft_copy_entry(entry)
    with pytest.raises(RuntimeError, match="one KV length per real or padded row"):
        NativeACLGraphRunner._copy_draft_inputs(entry, input_ids, positions, metadatas)
    _assert_draft_copy_entry_unchanged(entry, snapshot)


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
    runner.model.layers = [
        SimpleNamespace(
            self_attn=SimpleNamespace(
                uses_paged_attention=True,
                use_device_paged_attention=True,
            )
        )
    ]
    runner._execute_draft = MagicMock(side_effect=[torch.tensor([[4, 5]]), capture_output])
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
    assert runner.draft_taskless_replays == 1
    runner._update_draft_attention_tasks.assert_not_called()
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


def test_taskless_device_attention_replays_without_update_stream_dependency():
    runner, entry, update_stream = _draft_replay_runner()
    current_stream = MagicMock()
    runner.update_stream = None
    entry.tasks = []
    entry.positions = [torch.tensor([2]), torch.tensor([3])]
    entry.tasks_per_step = 0
    entry.taskless_device_attention = True

    with (
        patch.dict(
            os.environ,
            {
                # Taskless device PA has no task-update policy to conflict.
                "VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE": "1",
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
    entry.graph.replay.assert_called_once_with()
    runner._copy_draft_inputs.assert_called_once()
    runner._update_draft_attention_tasks.assert_not_called()
    update_stream.wait_stream.assert_not_called()
    update_stream.wait_event.assert_not_called()
    current_stream.wait_stream.assert_not_called()
    assert runner.draft_taskless_replays == 1
    assert runner.task_update_replay_counts["draft"] == 0


def test_empty_task_list_without_device_attention_origin_fails_before_replay():
    runner, entry, update_stream = _draft_replay_runner()
    entry.tasks = []
    entry.tasks_per_step = 0
    entry.taskless_device_attention = False

    with pytest.raises(RuntimeError, match="not an explicit device-position"):
        _run_draft_replay(runner)

    runner._copy_draft_inputs.assert_not_called()
    runner._update_draft_attention_tasks.assert_not_called()
    entry.graph.replay.assert_not_called()
    update_stream.wait_stream.assert_not_called()
    assert runner.draft_taskless_replays == 0


def test_taskless_device_attention_origin_rejects_captured_tasks():
    runner, entry, _ = _draft_replay_runner()
    entry.taskless_device_attention = True

    with pytest.raises(RuntimeError, match="unexpectedly owns attention tasks"):
        _run_draft_replay(runner)

    runner._copy_draft_inputs.assert_not_called()
    entry.graph.replay.assert_not_called()


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


def test_draft_graph_stable_host_lengths_skip_task_rebuild_and_signal_events():
    runner, entry, update_stream = _draft_replay_runner()
    current_stream = MagicMock()
    entry.runtime_validated = True
    entry.validated_real_row_count = 1
    entry.actual_seq_lengths_q = ((1,), (1,))
    entry.sequence_lens = ((3,), (3,))
    entry.tasks = [
        SimpleNamespace(event=MagicMock()),
        SimpleNamespace(event=MagicMock()),
    ]
    with (
        patch.dict(
            os.environ,
            {
                "VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE": "0",
                "VLLM_ASCEND_PEARL_DRAFT_REPLAY_FIRST_TASK_UPDATE": "0",
            },
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.current_stream",
            return_value=current_stream,
        ),
    ):
        output = _run_draft_replay(runner)

    runner._update_draft_attention_tasks.assert_not_called()
    for task in entry.tasks:
        task.event.record.assert_called_once_with(update_stream)
    assert runner.task_update_skip_replay_count == 1
    assert runner.task_update_replay_count == 0
    entry.graph.replay.assert_called_once_with()
    assert output.tolist() == [[4, 5]]


def test_stable_task_barrier_capture_inserts_one_wait_reset_for_112_tasks():
    stream = MagicMock(name="capture_stream")
    barrier = MagicMock(name="stable_task_barrier")
    query = torch.zeros((1, 2, 8))
    key_cache = torch.zeros((4, 16, 1, 8))
    value_cache = torch.zeros_like(key_cache)
    block_table = torch.zeros((1, 1), dtype=torch.int32)
    attention_mask = torch.zeros((1, 1, 1, 16), dtype=torch.bool)
    output = torch.zeros_like(query)
    with (
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.current_stream",
            return_value=stream,
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.ExternalEvent",
            return_value=barrier,
        ) as event_factory,
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_group_begin") as group_begin,
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_group_end",
            side_effect=[MagicMock(name=f"handle_{index}") for index in range(112)],
        ) as group_end,
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch_npu._npu_fused_infer_attention_score_get_max_workspace",
            return_value=MagicMock(name="workspace"),
        ),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch_npu.npu_fused_infer_attention_score.out"),
        native_graph._collect_graph_tasks(
            stable_task_barrier=True,
        ) as tasks,
    ):
        for _ in range(112):
            native_graph.run_native_fused_infer_attention(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                num_kv_heads=1,
                num_heads=2,
                scale=0.125,
                block_table=block_table,
                attention_mask=attention_mask,
                actual_seq_lengths_q=[1],
                actual_seq_lengths_kv=[3],
                block_size=16,
                output=output,
            )

    assert len(tasks) == 112
    assert all(task.event is barrier for task in tasks)
    event_factory.assert_called_once_with()
    barrier.wait.assert_called_once_with(stream)
    barrier.reset.assert_called_once_with(stream)
    assert group_begin.call_count == group_end.call_count == 112


@pytest.mark.parametrize(
    ("group_size", "task_count", "expected_event_count"),
    [(2, 5, 3), (4, 9, 3)],
)
def test_target_fia_event_groups_keep_one_handle_per_operator(
    group_size,
    task_count,
    expected_event_count,
):
    stream = MagicMock(name="capture_stream")
    events = [MagicMock(name=f"event_{index}") for index in range(expected_event_count)]
    query = torch.zeros((1, 2, 8))
    key_cache = torch.zeros((4, 16, 1, 8))
    value_cache = torch.zeros_like(key_cache)
    block_table = torch.zeros((1, 1), dtype=torch.int32)
    attention_mask = torch.zeros((1, 1, 1, 16), dtype=torch.bool)
    output = torch.zeros_like(query)
    with (
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.current_stream",
            return_value=stream,
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.ExternalEvent",
            side_effect=events,
        ) as event_factory,
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_group_begin") as group_begin,
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_group_end",
            side_effect=[MagicMock(name=f"handle_{index}") for index in range(task_count)],
        ) as group_end,
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch_npu._npu_fused_infer_attention_score_get_max_workspace",
            return_value=MagicMock(name="workspace"),
        ),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch_npu.npu_fused_infer_attention_score.out"),
        native_graph._collect_graph_tasks(
            task_event_group_size=group_size,
        ) as tasks,
    ):
        for _ in range(task_count):
            native_graph.run_native_fused_infer_attention(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                num_kv_heads=1,
                num_heads=2,
                scale=0.125,
                block_table=block_table,
                attention_mask=attention_mask,
                actual_seq_lengths_q=[1],
                actual_seq_lengths_kv=[3],
                block_size=16,
                output=output,
            )

    assert len(tasks) == task_count
    assert event_factory.call_count == expected_event_count
    for group_index, start in enumerate(range(0, task_count, group_size)):
        group = tasks[start : start + group_size]
        assert all(task.event is events[group_index] for task in group)
    for event in events:
        event.wait.assert_called_once_with(stream)
        event.reset.assert_called_once_with(stream)
    # CANN supports only a single operator per task-group handle.  This path
    # deliberately coalesces ExternalEvents while retaining every handle.
    assert group_begin.call_count == group_end.call_count == task_count


def test_target_fia_prefix_event_group_releases_two_then_four_layers():
    stream = MagicMock(name="capture_stream")
    events = [MagicMock(name=f"event_{index}") for index in range(3)]
    query = torch.zeros((1, 2, 8))
    key_cache = torch.zeros((4, 16, 1, 8))
    value_cache = torch.zeros_like(key_cache)
    block_table = torch.zeros((1, 1), dtype=torch.int32)
    attention_mask = torch.zeros((1, 1, 1, 16), dtype=torch.bool)
    output = torch.zeros_like(query)
    with (
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.current_stream",
            return_value=stream,
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.ExternalEvent",
            side_effect=events,
        ),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_group_begin"),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_group_end",
            side_effect=[MagicMock(name=f"handle_{index}") for index in range(10)],
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch_npu._npu_fused_infer_attention_score_get_max_workspace",
            return_value=MagicMock(name="workspace"),
        ),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch_npu.npu_fused_infer_attention_score.out"),
        native_graph._collect_graph_tasks(
            task_event_group_size=4,
            task_event_prefix_group_size=2,
        ) as tasks,
    ):
        for _ in range(10):
            native_graph.run_native_fused_infer_attention(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                num_kv_heads=1,
                num_heads=2,
                scale=0.125,
                block_table=block_table,
                attention_mask=attention_mask,
                actual_seq_lengths_q=[1],
                actual_seq_lengths_kv=[3],
                block_size=16,
                output=output,
            )

    expected_groups = ((0, 2), (2, 6), (6, 10))
    for event, (start, stop) in zip(events, expected_groups):
        assert all(task.event is event for task in tasks[start:stop])
        event.wait.assert_called_once_with(stream)
        event.reset.assert_called_once_with(stream)


def test_stable_task_barrier_refresh_updates_all_handles_before_one_record():
    runner, entry, update_stream, barrier, _ = _stable_task_barrier_draft_replay_runner()
    calls = MagicMock()
    runner._update_draft_attention_tasks.side_effect = lambda *args, **kwargs: calls.update()
    barrier.record.side_effect = lambda *args: calls.record()

    runner._refresh_draft_stable_task_barrier(entry)

    assert calls.mock_calls == [call.update(), call.record()]
    runner._update_draft_attention_tasks.assert_called_once_with(
        entry,
        stream=update_stream,
        record_events=False,
    )
    barrier.record.assert_called_once_with(update_stream)
    assert runner.draft_stable_task_barrier_records == 1


def test_stable_task_barrier_replay_first_records_once_for_112_tasks():
    runner, entry, update_stream, barrier, copy_done_event = _stable_task_barrier_draft_replay_runner()
    current_stream = MagicMock(name="current_stream")
    calls = MagicMock()
    runner._copy_draft_inputs.side_effect = lambda *args: calls.copy()
    copy_done_event.record.side_effect = lambda *args: calls.copy_done()
    update_stream.wait_event.side_effect = lambda *args: calls.wait()
    entry.graph.replay.side_effect = lambda: calls.replay()
    barrier.record.side_effect = lambda *args: calls.barrier()
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
        output = _run_stable_task_barrier_draft_replay(runner)

    assert output.tolist() == [[4, 5, 6, 7]]
    assert calls.mock_calls == [
        call.copy(),
        call.copy_done(),
        call.wait(),
        call.replay(),
        call.barrier(),
    ]
    runner._update_draft_attention_tasks.assert_not_called()
    barrier.record.assert_called_once_with(update_stream)
    assert runner.task_update_skip_replay_count == 1
    assert runner.draft_stable_task_barrier_records == 1


def test_stable_task_barrier_sealed_cache_rejects_unknown_kv_capacity():
    runner, entry, _, barrier, _ = _stable_task_barrier_draft_replay_runner()
    runner.graph_cache_sealed = True
    with pytest.raises(RuntimeError, match="reason=missing_entry"):
        _run_stable_task_barrier_draft_replay(
            runner,
            sequence_length=4,
        )

    runner._copy_draft_inputs.assert_not_called()
    entry.graph.replay.assert_not_called()
    runner._update_draft_attention_tasks.assert_not_called()
    barrier.record.assert_not_called()


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
    runner._update_attention_tasks.side_effect = lambda *args, **kwargs: calls.update()

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
    runner._update_attention_tasks.side_effect = lambda *args, **kwargs: calls.update()
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
    update_stream.wait_event.assert_called_once_with(entry.replay_first_copy_done_event)
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
    metadata = SimpleNamespace(
        use_fused_infer_attention=False,
        actual_seq_lengths_q=(1, 2),
        sequence_lens=(3, 3),
    )
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
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch_npu._npu_paged_attention") as paged_attention,
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_update_begin") as update_begin,
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_update_end") as update_end,
    ):
        runner._update_attention_task_list(
            [task],
            [((1, 2), (8, 8))],
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
    metrics = runner.graph_execution_metrics()
    assert metrics["pa_workspace_host_key_tasks"] == 1
    assert metrics["pa_workspace_tensor_key_tasks"] == 0
    assert metrics["pa_workspace_get_calls"] == 1
    assert metrics["pa_workspace_cache_hits"] == 0


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
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch_npu._npu_paged_attention"),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_update_begin"),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_update_end"),
    ):
        runner._update_attention_task_list(
            [first, second, third],
            [
                ((1, 2), (8, 8)),
                ((1, 2), (8, 8)),
                ((1, 2), (9, 9)),
            ],
            stream=update_stream,
        )

    assert get_workspace.call_count == 2
    assert first.workspace is workspaces[0]
    assert second.workspace is workspaces[0]
    assert third.workspace is workspaces[1]
    metrics = runner.graph_execution_metrics()
    assert metrics["pa_workspace_host_key_tasks"] == 3
    assert metrics["pa_workspace_tensor_key_tasks"] == 0
    assert metrics["pa_workspace_get_calls"] == 2
    assert metrics["pa_workspace_cache_hits"] == 1


def test_draft_step_major_paged_attention_queries_one_workspace_per_step():
    runner = NativeACLGraphRunner(MagicMock(), enabled=False)
    runner.update_stream = MagicMock()
    # The third row is graph padding.  It must retain the production
    # length-one KV contract rather than forcing this fast path to fall back.
    step0 = [_paged_task(context_lens=(8, 8, 1)) for _ in range(2)]
    step1 = [_paged_task(context_lens=(9, 9, 1)) for _ in range(2)]
    entry = SimpleNamespace(
        tasks=[*step0, *step1],
        tasks_per_step=2,
        actual_seq_lengths_q=((1, 2), (1, 2)),
        sequence_lens=((8, 8, 1), (9, 9, 1)),
    )
    workspaces = [MagicMock(name="workspace_8"), MagicMock(name="workspace_9")]
    with (
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.envs.VLLM_ASCEND_PEARL_DRAFT_STEP_MAJOR_PA_TASK_UPDATE",
            True,
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.stream",
            return_value=MagicMock(),
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch_npu._npu_paged_attention_get_workspace",
            side_effect=workspaces,
        ) as get_workspace,
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch_npu._npu_paged_attention") as paged_attention,
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_update_begin") as update_begin,
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_update_end") as update_end,
    ):
        runner._update_draft_attention_tasks(entry)

    assert get_workspace.call_count == 2
    assert paged_attention.call_count == 4
    assert update_begin.call_count == update_end.call_count == 4
    assert all(task.workspace is workspaces[0] for task in step0)
    assert all(task.workspace is workspaces[1] for task in step1)
    metrics = runner.graph_execution_metrics()
    assert metrics["draft_step_major_pa_replays"] == 1
    assert metrics["draft_step_major_pa_fallback_replays"] == 0
    assert metrics["pa_workspace_get_calls"] == 2
    assert metrics["pa_workspace_cache_hits"] == 2


def test_paged_attention_host_lengths_avoid_per_layer_tensor_materialization():
    runner = NativeACLGraphRunner(MagicMock(), enabled=False)
    update_stream = MagicMock()
    task = _paged_task(context_lens=(8, 8))
    context_lens = MagicMock(name="cpu_context_lens")
    context_lens.numel.return_value = 2
    context_lens.tolist.side_effect = AssertionError("the aligned host tuple must avoid Tensor.tolist")
    task.context_lens = context_lens
    with (
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.stream",
            return_value=MagicMock(),
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch_npu._npu_paged_attention_get_workspace",
            return_value=MagicMock(),
        ),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch_npu._npu_paged_attention"),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_update_begin"),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_update_end"),
    ):
        runner._update_attention_task_list(
            [task],
            [((1, 2), (8, 8))],
            stream=update_stream,
        )

    context_lens.tolist.assert_not_called()


def test_paged_attention_finite_padding_uses_host_lengths_without_tensor_materialization():
    runner = NativeACLGraphRunner(MagicMock(), enabled=False)
    update_stream = MagicMock()
    task = _paged_task(context_lens=(8, 8, 1))
    context_lens = MagicMock(name="cpu_context_lens")
    context_lens.numel.return_value = 3
    context_lens.tolist.side_effect = AssertionError("the finite padded host tuple must avoid Tensor.tolist")
    task.context_lens = context_lens
    with (
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.stream",
            return_value=MagicMock(),
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch_npu._npu_paged_attention_get_workspace",
            return_value=MagicMock(),
        ),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch_npu._npu_paged_attention"),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_update_begin"),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_update_end"),
    ):
        runner._update_attention_task_list(
            [task],
            [((1, 2), (8, 8, 1))],
            stream=update_stream,
        )

    context_lens.tolist.assert_not_called()
    metrics = runner.graph_execution_metrics()
    assert metrics["pa_workspace_host_key_tasks"] == 1
    assert metrics["pa_workspace_tensor_key_tasks"] == 0


def test_paged_attention_packed_queries_keep_tensor_length_fallback():
    runner = NativeACLGraphRunner(MagicMock(), enabled=False)
    update_stream = MagicMock()
    task = _paged_task(context_lens=(7, 8))
    with (
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.stream",
            return_value=MagicMock(),
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch_npu._npu_paged_attention_get_workspace",
            return_value=MagicMock(),
        ),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch_npu._npu_paged_attention"),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_update_begin"),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_update_end"),
    ):
        runner._update_attention_task_list(
            [task],
            [((2,), (8,))],
            stream=update_stream,
        )

    metrics = runner.graph_execution_metrics()
    assert metrics["pa_workspace_host_key_tasks"] == 0
    assert metrics["pa_workspace_tensor_key_tasks"] == 1


def test_paged_attention_host_timing_is_opt_in_and_phase_separated():
    runner = NativeACLGraphRunner(MagicMock(), enabled=False)
    runner.profile_pa_task_update = True
    update_stream = MagicMock()
    task = _paged_task(context_lens=(8, 8))
    with (
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.stream",
            return_value=MagicMock(),
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch_npu._npu_paged_attention_get_workspace",
            return_value=MagicMock(),
        ),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch_npu._npu_paged_attention"),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_update_begin"),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_update_end"),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.time.perf_counter_ns",
            side_effect=[10, 20, 30, 50, 60, 100],
        ),
    ):
        runner._update_attention_task_list(
            [task],
            [((1, 2), (8, 8))],
            stream=update_stream,
        )

    metrics = runner.graph_execution_metrics()
    assert metrics["pa_task_update_profiled_tasks"] == 1
    assert metrics["pa_task_update_host_key_ns"] == 10
    assert metrics["pa_task_update_host_get_workspace_ns"] == 20
    assert metrics["pa_task_update_host_task_update_ns"] == 40


@pytest.mark.parametrize("group_size", [2, 4])
def test_fia_task_update_records_one_event_per_group(group_size):
    runner = NativeACLGraphRunner(MagicMock(), enabled=False)
    update_stream = MagicMock(name="update_stream")
    task_count = group_size * 2
    group_events = [
        MagicMock(name="group_event_0"),
        MagicMock(name="group_event_1"),
    ]
    tasks = [_fia_task() for _ in range(task_count)]
    for task_index, task in enumerate(tasks):
        task.event = group_events[task_index // group_size]
    with (
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.stream",
            return_value=MagicMock(),
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch_npu.npu_fused_infer_attention_score.out"
        ) as fused_attention,
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_update_begin") as update_begin,
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_update_end") as update_end,
    ):
        runner._update_attention_task_list(
            tasks,
            [((1, 2), (8, 8))] * task_count,
            stream=update_stream,
            event_group_size=group_size,
        )

    # Every FIA operator keeps its independently captured CANN handle; only
    # the ExternalEvent release is coalesced after each complete group update.
    assert update_begin.call_count == update_end.call_count == task_count
    assert fused_attention.call_count == task_count
    for event in group_events:
        event.record.assert_called_once_with(update_stream)


def test_parallel_fia_chunk_keeps_complete_event_groups_on_one_stream():
    runner = NativeACLGraphRunner(MagicMock(), enabled=False)
    update_stream = MagicMock(name="parallel_update_stream")
    group_events = [MagicMock(name=f"group_event_{index}") for index in range(4)]
    tasks = [_fia_task() for _ in range(8)]
    for task_index, task in enumerate(tasks):
        task.event = group_events[task_index // 2]
    lengths = [([1, 2], [8, 8])] * len(tasks)
    with (
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.set_device"),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.stream",
            return_value=MagicMock(),
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch_npu.npu_fused_infer_attention_score.out"
        ) as fused_attention,
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_update_begin") as update_begin,
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_update_end") as update_end,
    ):
        runner._update_fia_task_chunk(
            tasks,
            lengths,
            (0, 1, 4, 5),
            update_stream,
            frozenset((1, 3, 5, 7)),
        )

    assert update_begin.call_count == update_end.call_count == 4
    assert fused_attention.call_count == 4
    group_events[0].record.assert_called_once_with(update_stream)
    group_events[2].record.assert_called_once_with(update_stream)
    group_events[1].record.assert_not_called()
    group_events[3].record.assert_not_called()


def test_fia_task_event_group_layout_fails_before_any_update():
    runner = NativeACLGraphRunner(MagicMock(), enabled=False)
    tasks = [_fia_task(), _fia_task()]
    with (
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_update_begin") as update_begin,
        pytest.raises(RuntimeError, match="do not share"),
    ):
        runner._update_attention_task_list(
            tasks,
            [((1,), (8,))] * 2,
            stream=MagicMock(),
            event_group_size=2,
        )

    update_begin.assert_not_called()


def test_fia_host_timing_is_disabled_without_profile_switch():
    runner = NativeACLGraphRunner(MagicMock(), enabled=False)
    update_stream = MagicMock()
    task = _fia_task()
    with (
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.stream",
            return_value=MagicMock(),
        ),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch_npu.npu_fused_infer_attention_score.out"),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_update_begin"),
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_update_end"),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.time.perf_counter_ns",
            side_effect=AssertionError("profiling clock must stay off"),
        ) as clock,
    ):
        runner._update_attention_task_list(
            [task],
            [((1, 2), (8, 8))],
            stream=update_stream,
        )

    clock.assert_not_called()
    metrics = runner.graph_execution_metrics()
    assert metrics["fia_task_update_profiled_tasks"] == 0
    assert metrics["fia_task_update_host_key_ns"] == 0
    assert metrics["fia_task_update_host_submit_ns"] == 0
    assert metrics["fia_task_update_host_event_ns"] == 0


def test_fia_host_timing_is_opt_in_and_phase_separated():
    runner = NativeACLGraphRunner(MagicMock(), enabled=False)
    runner.profile_pa_task_update = True
    update_stream = MagicMock()
    task = _fia_task()
    with (
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.stream",
            return_value=MagicMock(),
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch_npu.npu_fused_infer_attention_score.out"
        ) as fused_attention,
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_update_begin") as update_begin,
        patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.graph_task_update_end") as update_end,
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.time.perf_counter_ns",
            side_effect=[10, 20, 30, 50, 60, 90],
        ),
    ):
        runner._update_attention_task_list(
            [task],
            [((1, 2), (8, 8))],
            stream=update_stream,
        )

    update_begin.assert_called_once_with(update_stream, task.handle)
    update_end.assert_called_once_with(update_stream)
    task.event.record.assert_called_once_with(update_stream)
    assert fused_attention.call_args.kwargs["actual_seq_lengths"] == [1, 2]
    assert fused_attention.call_args.kwargs["actual_seq_lengths_kv"] == [8, 8]
    metrics = runner.graph_execution_metrics()
    assert metrics["fia_task_update_profiled_tasks"] == 1
    assert metrics["fia_task_update_host_key_ns"] == 10
    assert metrics["fia_task_update_host_submit_ns"] == 20
    assert metrics["fia_task_update_host_event_ns"] == 30


def test_padding_preserves_token_aligned_host_sequence_lengths():
    metadata = _PaddingMetadata(
        slot_mapping=torch.tensor([9, 10]),
        context_lens=torch.tensor([3, 4], dtype=torch.int32),
        block_tables=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        actual_seq_lengths_q=(1, 2),
        sequence_lens=(3, 4),
    )

    _, _, padded = NativeACLGraphRunner._pad_inputs(
        torch.tensor([5, 6]),
        torch.tensor([1, 2]),
        metadata,
        4,
    )

    assert padded.context_lens.tolist() == [3, 4, 1, 1]
    assert padded.actual_seq_lengths_q == (1, 2)
    assert padded.sequence_lens == (3, 4, 1, 1)
    assert padded.block_tables.tolist() == [[1, 2], [3, 4], [1, 2], [1, 2]]
    assert padded.sentinel == "preserved"
