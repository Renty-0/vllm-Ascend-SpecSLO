# SPDX-License-Identifier: Apache-2.0

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from examples.benchmark_nano_pearl_speculative import _worker_aclgraph_deltas
from vllm_ascend.spec_decode.pearl.native_engine import NativePearlEngine
from vllm_ascend.spec_decode.pearl.native_graph import (
    NativeACLGraphRunner,
    NativeGraphExecution,
)


def test_replay_histograms_record_generic_and_target_token_rows() -> None:
    runner = NativeACLGraphRunner(MagicMock(), enabled=False)
    generic_key = (
        "hidden|fia-stable:stable-target-verify-hidden|requests:16",
        64,
    )
    target_key = ("target-hidden:0|steps:2|paged", (64, 128))

    for _ in range(2):
        runner._record_execution(
            "generic",
            NativeGraphExecution("replay", replay_executed=True),
            None,
            entry_key=generic_key,
            token_rows=(64,),
        )
    runner._record_execution(
        "generic",
        NativeGraphExecution(
            "capture_replay",
            capture_attempted=True,
            replay_executed=True,
        ),
        None,
        entry_key=generic_key,
        token_rows=(64,),
    )
    runner._record_execution(
        "target",
        NativeGraphExecution("replay", replay_executed=True),
        None,
        entry_key=target_key,
        token_rows=(64, 128),
    )

    metrics = runner.graph_execution_metrics()

    assert metrics["generic_replay_calls"] == 2
    assert metrics["generic_capture_replay_calls"] == 1
    assert metrics["generic_replay_entry_key_histogram"] == {
        repr(generic_key): 2,
    }
    assert metrics["generic_replay_token_rows_histogram"] == {"64": 2}
    assert metrics["target_replay_entry_key_histogram"] == {
        repr(target_key): 1,
    }
    assert metrics["target_replay_token_rows_histogram"] == {
        "64": 1,
        "128": 1,
    }
    json.dumps(metrics)


def test_generic_replay_records_physical_capture_size_as_model_m() -> None:
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
    entry.validated_real_row_count = 64
    entry.actual_seq_lengths_q = (1, 2)
    entry.sequence_lens = (3, 3)
    entry.tasks = [SimpleNamespace(event=MagicMock())]
    entry.replay_first_copy_done_event = MagicMock()
    entry_key = ("hidden", 64)
    runner.entries[entry_key] = entry
    runner._copy_inputs = MagicMock()
    runner._pad_inputs = MagicMock(
        side_effect=lambda input_ids, positions, metadata, capture_size: (
            input_ids,
            positions,
            metadata,
        )
    )
    metadata = SimpleNamespace(
        use_fused_infer_attention=False,
        actual_seq_lengths_q=(1, 2),
        sequence_lens=(3, 3),
    )

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
            return_value=MagicMock(),
        ),
    ):
        output = runner(
            torch.tensor([3]),
            torch.tensor([2]),
            metadata,
        )

    assert output.tolist() == [[4, 5]]
    assert runner.graph_execution_metrics()["generic_replay_entry_key_histogram"] == {repr(entry_key): 1}
    assert runner.graph_execution_metrics()["generic_replay_token_rows_histogram"] == {"64": 1}
    assert runner._pad_inputs.call_args.args[3] == 64


def test_worker_graph_metrics_expose_replay_histograms() -> None:
    runner = NativeACLGraphRunner(MagicMock(), enabled=False)
    entry_key = ("greedy:32", 64)
    runner._record_execution(
        "generic",
        NativeGraphExecution("replay", replay_executed=True),
        None,
        entry_key=entry_key,
        token_rows=(64,),
    )
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.rank = 1
    engine.is_draft = False
    engine.graph_runner = runner
    engine.last_worker_decode_phase_seconds = {}
    engine.last_worker_profiled_decode_steps = 0
    engine.last_worker_decode_profile_seconds = {}
    engine.last_worker_decode_profile_detail_seconds = {}
    engine.last_worker_decode_counters = {}
    engine.last_worker_decode_host_timeline = []

    metrics = engine.graph_metrics()

    assert metrics["is_draft_rank"] == 0
    assert metrics["aclgraph_generic_replay_entry_key_histogram"] == {
        repr(entry_key): 1,
    }
    assert metrics["aclgraph_generic_replay_token_rows_histogram"] == {
        "64": 1,
    }
    json.dumps(metrics)


def test_benchmark_deltas_subtract_replay_histograms_by_rank() -> None:
    first_key = "('hidden|fia-stable:first', 64)"
    second_key = "('hidden|fia-stable:second', 160)"
    retired_key = "('hidden|fia-stable:retired', 128)"
    before = [
        {
            "rank": 1,
            "is_draft_rank": 0,
            "aclgraph_generic_replay_entry_key_histogram": {
                first_key: 3,
                retired_key: 2,
            },
            "aclgraph_generic_replay_token_rows_histogram": {
                "64": 3,
                "128": 2,
            },
        }
    ]
    after = [
        {
            "rank": 1,
            "is_draft_rank": 0,
            "aclgraph_generic_replay_entry_key_histogram": {
                first_key: 8,
                second_key: 4,
            },
            "aclgraph_generic_replay_token_rows_histogram": {
                "64": 8,
                "160": 4,
            },
        }
    ]

    delta = _worker_aclgraph_deltas(before, after)[0]

    assert delta["is_draft_rank"] == 0
    assert delta["aclgraph_generic_replay_entry_key_histogram_delta"] == {
        first_key: 5,
        retired_key: -2,
        second_key: 4,
    }
    assert delta["aclgraph_generic_replay_token_rows_histogram_delta"] == {
        "128": -2,
        "160": 4,
        "64": 5,
    }
    assert delta["aclgraph_target_replay_entry_key_histogram_delta"] == {}
    assert delta["aclgraph_target_replay_token_rows_histogram_delta"] == {}
    json.dumps(delta)
