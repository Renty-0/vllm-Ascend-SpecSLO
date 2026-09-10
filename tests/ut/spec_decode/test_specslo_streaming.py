# SPDX-License-Identifier: Apache-2.0
"""CPU pipe regressions for guarded SpecSLO output delivery."""

import os
import threading
from multiprocessing import Pipe
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from vllm_ascend.spec_decode.pearl import api
from vllm_ascend.spec_decode.pearl.native_engine import SamplingParams


@pytest.fixture
def pipe_engine():
    engine = api.PEARLEngine.__new__(api.PEARLEngine)
    engine.config = SimpleNamespace(worker_timeout_seconds=2.0)
    connections = [Pipe(duplex=True) for _ in range(4)]
    engine._connections = [parent for parent, _ in connections]
    engine._processes = [SimpleNamespace(is_alive=lambda: True, exitcode=None) for _ in connections]
    try:
        yield engine, [worker for _, worker in connections]
    finally:
        for parent, worker in connections:
            parent.close()
            worker.close()


def _event(tokens=(11, 12), *, finished=False):
    return {
        "request_index": 0,
        "request_id": "trace-17",
        "token_ids": list(tokens),
        "finished": finished,
        "elapsed_seconds": 0.25,
    }


def test_commit_is_delivered_before_worker_finishes_generation(pipe_engine):
    engine, workers = pipe_engine
    delivered = threading.Event()
    finished = threading.Event()
    seen = []

    def worker():
        workers[1].send(("commit", _event()))
        assert delivered.wait(timeout=1.0)
        workers[1].send(("result", [{"completion_token_ids": [11, 12]}]))
        finished.set()

    def on_commit(event):
        assert not finished.is_set()
        seen.append(event)
        delivered.set()

    for rank in (0, 2, 3):
        workers[rank].send(("result", None))
    thread = threading.Thread(target=worker)
    thread.start()
    try:
        replies = engine._receive_all("stream test", on_token_commit=on_commit)
    finally:
        delivered.set()
        thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert seen == [_event()]
    assert len(replies) == 4
    assert replies[1][1] == [{"completion_token_ids": [11, 12]}]


def test_callback_exception_drains_all_commits_and_worker_results(pipe_engine):
    engine, workers = pipe_engine
    for rank, worker in enumerate(workers):
        if rank == 1:
            worker.send(("commit", _event()))
            worker.send(("commit", _event((13,), finished=True)))
        worker.send(("result", [] if rank == 1 else None))

    callback = Mock(side_effect=ValueError("client failed"))
    with pytest.raises(api._TokenCommitDeliveryError, match="workers were drained") as error:
        engine._receive_all("stream failure", on_token_commit=callback)
    assert isinstance(error.value.__cause__, ValueError)
    callback.assert_called_once()
    assert not any(connection.poll() for connection in engine._connections)
    for rank, worker in enumerate(workers):
        worker.send(("ready", rank))
    assert engine._receive_all("next command") == [("ready", rank) for rank in range(4)]


def test_commit_messages_do_not_replace_terminal_replies_without_a_callback(pipe_engine):
    engine, workers = pipe_engine
    workers[1].send(("commit", _event()))
    for rank, worker in enumerate(workers):
        worker.send(("result", rank))
    assert engine._receive_all("discard commit") == [("result", rank) for rank in range(4)]


def _controller_for_chunks():
    engine = api.PEARLEngine.__new__(api.PEARLEngine)
    engine.config = SimpleNamespace(
        gamma=2,
        max_model_len=128,
        max_num_seqs=1,
        max_num_batched_tokens=128,
        enable_continuous_batching=False,
        enable_spec_rhythm=True,
    )
    engine._requests = [
        (7, [1], SamplingParams(temperature=0.0, max_tokens=2)),
        (12, [2], SamplingParams(temperature=0.0, max_tokens=2, request_id="external-12")),
    ]
    engine._send_all = Mock()
    engine.tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: str(tokens))
    return engine


def test_public_generate_streaming_maps_chunk_local_indices_and_preserves_return():
    engine = _controller_for_chunks()
    seen = []

    def receive(operation, *, on_token_commit):
        chunk_number = engine._send_all.call_count
        event = _event((10 + chunk_number,), finished=True)
        event["request_id"] = None if chunk_number == 1 else "external-12"
        on_token_commit(event)
        return [
            (
                "result",
                [
                    {
                        "completion_token_ids": event["token_ids"],
                        "num_acc_tokens": [1],
                        "elapsed_seconds": 0.5,
                    }
                ],
            )
        ]

    engine._receive_all = Mock(side_effect=receive)
    text, counts, accepted, elapsed = engine.generate(on_token_commit=seen.append)
    assert [call.args[0][0] for call in engine._send_all.call_args_list] == ["pearl_stream", "pearl_stream"]
    assert [event["request_index"] for event in seen] == [7, 12]
    assert [event["request_id"] for event in seen] == [7, "external-12"]
    assert text == ["[11]", "[12]"]
    assert counts == [1, 1]
    assert accepted == ([1], [1])
    assert elapsed == 1.0
    assert engine._requests == []


def test_callback_failure_does_not_requeue_consumed_or_delivered_requests():
    engine = _controller_for_chunks()
    engine._requests.append((15, [3], SamplingParams(max_tokens=2)))
    engine._receive_all = Mock(
        side_effect=[
            [("result", [{"completion_token_ids": [11], "num_acc_tokens": [1], "elapsed_seconds": 0.1}])],
            api._TokenCommitDeliveryError("callback failed after drain"),
        ]
    )
    with pytest.raises(api._TokenCommitDeliveryError):
        engine.generate(on_token_commit=lambda event: None)
    assert [request[0] for request in engine._requests] == [15]
    assert engine._send_all.call_count == 2


def test_public_generate_rejects_noncallable_callback_before_consuming_requests():
    engine = _controller_for_chunks()
    with pytest.raises(TypeError, match="callable"):
        engine.generate(on_token_commit=123)
    assert len(engine._requests) == 2
    engine._send_all.assert_not_called()


def test_public_generate_rejects_unsupported_pearl_streaming_before_dispatch():
    engine = _controller_for_chunks()
    engine.config.enable_spec_rhythm = False
    with pytest.raises(ValueError, match="requires enable_spec_rhythm"):
        engine.generate(on_token_commit=lambda event: None)
    assert len(engine._requests) == 2
    engine._send_all.assert_not_called()


def test_worker_sends_commit_events_before_final_result(monkeypatch):
    event = _event(finished=True)
    engine = Mock()
    engine.graph_metrics.return_value = {}

    def generate_batch(prompts, params, *, on_token_commit):
        on_token_commit(event)
        return [{"completion_token_ids": event["token_ids"]}]

    engine.generate_batch.side_effect = generate_batch
    monkeypatch.setattr(api, "NativePearlEngine", lambda config: engine)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)
    connection = Mock()
    connection.recv.side_effect = [
        ("pearl_stream", [[1]], [SamplingParams(max_tokens=2)], None),
        ("exit", None, None, None),
    ]
    config = SimpleNamespace(draft_tp_size=1, target_tp_size=3)
    with patch.dict(os.environ):
        api._pearl_worker(config, 1, 12345, connection)
    assert [call.args[0][0] for call in connection.send.call_args_list] == ["ready", "commit", "result"]
    assert connection.send.call_args_list[1].args[0][1] == event
    connection.close.assert_called_once()
