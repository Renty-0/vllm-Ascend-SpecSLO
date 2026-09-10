# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from vllm_ascend.spec_decode.pearl.http_server import (
    SpecSLOHTTPService,
    create_specslo_app,
)
from vllm_ascend.spec_decode.pearl.native_engine import SamplingParams


class _FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, token_ids, *, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr(int(token_id)) for token_id in token_ids)

    def apply_chat_template(self, messages, *, tokenize: bool, add_generation_prompt: bool):
        assert tokenize and add_generation_prompt
        text = "|".join(f"{message['role']}:{message['content']}" for message in messages)
        return self.encode(text)


class _FakeEngine:
    instances: list[_FakeEngine] = []

    def __init__(self, config) -> None:
        self.config = config
        self.tokenizer = _FakeTokenizer()
        self.requests: list[tuple[list[int], SamplingParams]] = []
        self.generate_batches: list[list[tuple[list[int], SamplingParams]]] = []
        self.last_metrics: list[dict[str, Any]] = []
        self.exited = False
        self.aborted_request_ids: list[str] = []
        self._processes = ()
        self.instances.append(self)

    def add_request(self, prompt, sampling_params) -> None:
        self.requests.append((list(prompt), sampling_params))

    def generate(self, *, on_token_commit=None):
        batch = self.requests
        self.requests = []
        self.generate_batches.append(batch)
        texts = []
        self.last_metrics = []
        for _, params in batch:
            tokens = [ord("O"), ord("K")]
            if on_token_commit is not None:
                on_token_commit(
                    {
                        "request_id": params.request_id,
                        "token_ids": tokens,
                        "finished": True,
                    }
                )
            texts.append("OK")
            self.last_metrics.append(
                {
                    "request_id": params.request_id,
                    "completion_token_ids": tokens,
                    "num_acc_tokens": [1],
                }
            )
        return texts, [2] * len(batch), tuple([1] for _ in batch), 0.01

    def generate_live(self, *, on_token_commit=None):
        return self.generate(on_token_commit=on_token_commit)

    def admit_live_requests(self, requests) -> None:
        self.requests.extend(requests)

    def abort_live_requests(self, request_ids) -> bool:
        self.aborted_request_ids.extend(str(value) for value in request_ids)
        return True

    def exit(self) -> None:
        self.exited = True


class _FailingFakeEngine(_FakeEngine):
    def generate(self, *, on_token_commit=None):
        del on_token_commit
        self.requests = []
        raise RuntimeError("synthetic collective failure")


class _SlowLiveFakeEngine(_FakeEngine):
    started = threading.Event()

    def __init__(self, config) -> None:
        super().__init__(config)
        self._request_lock = threading.Lock()

    def add_request(self, prompt, sampling_params) -> None:
        with self._request_lock:
            super().add_request(prompt, sampling_params)

    def admit_live_requests(self, requests) -> None:
        with self._request_lock:
            self.requests.extend(requests)

    def generate_live(self, *, on_token_commit=None):
        self.started.set()
        # Leave a deterministic window in which the event-loop batcher can
        # publish a request after decode has begun.
        time.sleep(0.05)
        with self._request_lock:
            return super().generate(on_token_commit=on_token_commit)


def _config():
    return SimpleNamespace(
        max_num_queued_seqs=8,
        max_num_seqs=4,
        target_model_path="/models/Qwen3-32B",
    )


def test_service_persists_engine_and_microbatches_compatible_requests() -> None:
    async def scenario() -> None:
        _FakeEngine.instances.clear()
        service = SpecSLOHTTPService(
            _config(),
            engine_factory=_FakeEngine,
            batch_wait_ms=20,
            max_batch_size=4,
        )
        await service.start()
        try:
            first = await service.submit([1], SamplingParams(temperature=0, max_tokens=2), stream=True)
            second = await service.submit([2], SamplingParams(temperature=0, max_tokens=2))
            first_result, second_result = await asyncio.gather(first.future, second.future)
            assert first_result.text == second_result.text == "OK"
            assert first_result.token_ids == second_result.token_ids == (79, 75)
            assert [event async for event in first.events()] == [
                {
                    "request_id": first.request_id,
                    "token_ids": (79, 75),
                    "finished": True,
                    "text": "OK",
                }
            ]
            engine = _FakeEngine.instances[-1]
            assert len(engine.generate_batches) == 1
            assert len(engine.generate_batches[0]) == 2
            assert service.health()["completed_requests"] == 2
            assert service.health()["inflight_requests"] == 0
        finally:
            await service.stop()
        assert _FakeEngine.instances[-1].exited

    asyncio.run(scenario())


def test_service_separates_target_and_draft_sampling_modes() -> None:
    async def scenario() -> None:
        _FakeEngine.instances.clear()
        service = SpecSLOHTTPService(
            _config(),
            engine_factory=_FakeEngine,
            batch_wait_ms=20,
            max_batch_size=4,
        )
        await service.start()
        try:
            handles = [
                await service.submit([1], SamplingParams(temperature=0, draft_temperature=0, max_tokens=2)),
                await service.submit([2], SamplingParams(temperature=1, draft_temperature=0, max_tokens=2)),
                await service.submit([3], SamplingParams(temperature=1, draft_temperature=1, max_tokens=2)),
            ]
            await asyncio.gather(*(handle.future for handle in handles))
            engine = _FakeEngine.instances[-1]
            assert [len(batch) for batch in engine.generate_batches] == [1, 1, 1]
        finally:
            await service.stop()

    asyncio.run(scenario())


def test_service_admits_arrival_while_live_decode_is_running() -> None:
    async def scenario() -> None:
        _SlowLiveFakeEngine.instances.clear()
        _SlowLiveFakeEngine.started.clear()
        service = SpecSLOHTTPService(
            _config(),
            engine_factory=_SlowLiveFakeEngine,
            batch_wait_ms=0,
            max_batch_size=4,
        )
        await service.start()
        try:
            first = await service.submit([1], SamplingParams(temperature=0, max_tokens=2))
            await asyncio.to_thread(_SlowLiveFakeEngine.started.wait, 1.0)
            second = await service.submit([2], SamplingParams(temperature=0, max_tokens=2))
            await asyncio.gather(first.future, second.future)
            engine = _SlowLiveFakeEngine.instances[-1]
            assert len(engine.generate_batches) == 1
            assert len(engine.generate_batches[0]) == 2
        finally:
            await service.stop()

    asyncio.run(scenario())


def test_service_propagates_abort_to_a_live_worker_epoch_once() -> None:
    async def scenario() -> None:
        _SlowLiveFakeEngine.instances.clear()
        _SlowLiveFakeEngine.started.clear()
        service = SpecSLOHTTPService(
            _config(),
            engine_factory=_SlowLiveFakeEngine,
            batch_wait_ms=0,
            max_batch_size=4,
        )
        await service.start()
        try:
            handle = await service.submit([1], SamplingParams(temperature=0, max_tokens=2))
            await asyncio.to_thread(_SlowLiveFakeEngine.started.wait, 1.0)
            assert await service.abort_requests((handle.request_id,)) == 1
            assert await service.abort_requests((handle.request_id,)) == 0
            assert handle.future.cancelled()
            await asyncio.sleep(0.1)
            engine = _SlowLiveFakeEngine.instances[-1]
            assert engine.aborted_request_ids == [handle.request_id]
            assert service.health()["aborted_requests"] == 1
            assert service.health()["inflight_requests"] == 0
        finally:
            await service.stop()

    asyncio.run(scenario())


def test_service_propagates_batch_failure_and_shuts_down_workers() -> None:
    async def scenario() -> None:
        _FailingFakeEngine.instances.clear()
        service = SpecSLOHTTPService(
            _config(),
            engine_factory=_FailingFakeEngine,
            batch_wait_ms=0,
        )
        await service.start()
        handle = await service.submit([1], SamplingParams(temperature=0, max_tokens=2), stream=True)
        try:
            try:
                await handle.future
            except RuntimeError as error:
                assert str(error) == "synthetic collective failure"
            else:
                raise AssertionError("the synthetic worker failure was not propagated")
            assert [event async for event in handle.events()] == [{"error": "synthetic collective failure"}]
            assert service.health()["failed_requests"] == 1
            assert service.health()["inflight_requests"] == 0
        finally:
            await service.stop()
        assert _FailingFakeEngine.instances[-1].exited

    asyncio.run(scenario())


def test_openai_completion_chat_health_and_streaming_endpoints() -> None:
    _FakeEngine.instances.clear()
    app = create_specslo_app(
        _config(),
        engine_factory=_FakeEngine,
        batch_wait_ms=0,
    )
    with TestClient(app) as client:
        assert client.get("/health").json()["healthy"]
        assert client.get("/v1/models").json()["data"][0]["id"] == "Qwen3-32B"

        completion = client.post(
            "/v1/completions",
            json={
                "model": "Qwen3-32B",
                "prompt": "x",
                "max_tokens": 2,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 32,
                "draft_temperature": 0.5,
                "draft_top_p": 0.8,
                "draft_top_k": 16,
            },
        )
        assert completion.status_code == 200
        assert completion.json()["choices"][0]["text"] == "OK"
        assert completion.json()["usage"] == {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        }
        params = _FakeEngine.instances[-1].generate_batches[0][0][1]
        assert (params.temperature, params.top_p, params.top_k) == (0.7, 0.9, 32)
        assert (params.draft_temperature, params.draft_top_p, params.draft_top_k) == (0.5, 0.8, 16)

        chat = client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen3-32B",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 2,
                "temperature": 0,
            },
        )
        assert chat.status_code == 200
        assert chat.json()["choices"][0]["message"] == {"role": "assistant", "content": "OK"}

        with client.stream(
            "POST",
            "/v1/completions",
            json={"prompt": [1], "max_tokens": 2, "temperature": 0, "stream": True},
        ) as streamed:
            body = "".join(streamed.iter_text())
        assert '"text": "OK"' in body
        assert "data: [DONE]" in body

        unknown = client.post(
            "/v1/completions",
            json={"model": "not-served", "prompt": "x", "temperature": 0},
        )
        assert unknown.status_code == 404
    assert _FakeEngine.instances[-1].exited
