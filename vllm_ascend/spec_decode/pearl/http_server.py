# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent OpenAI-compatible HTTP lifecycle for native SpecSLO.

The HTTP layer deliberately owns only admission, micro-batching and response
delivery.  SpecRhythm scheduling, dual-batch execution, tree verification and
guarded commits remain inside :class:`PEARLEngine` and its HCCL workers.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vllm_ascend.spec_decode.pearl.api import PEARLConfig, PEARLEngine, SamplingParams


class SpecSLOCompletionRequest(BaseModel):
    """Supported subset of the OpenAI completion request plus SLO controls."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    prompt: str | list[int]
    max_tokens: int = Field(default=16, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    draft_temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    draft_top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    draft_top_k: int = Field(default=0, ge=0)
    stream: bool = False
    ignore_eos: bool = False
    slo_tpot_ms: float | None = Field(default=None, gt=0.0)
    slo_class: str | None = None
    spec_rhythm_max_gamma: int | None = Field(default=None, ge=1)
    user: str | None = None


class SpecSLOChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class SpecSLOChatRequest(BaseModel):
    """Supported OpenAI chat request with native SpecRhythm extensions."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    messages: list[SpecSLOChatMessage]
    max_tokens: int = Field(default=16, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    draft_temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    draft_top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    draft_top_k: int = Field(default=0, ge=0)
    stream: bool = False
    ignore_eos: bool = False
    slo_tpot_ms: float | None = Field(default=None, gt=0.0)
    slo_class: str | None = None
    spec_rhythm_max_gamma: int | None = Field(default=None, ge=1)
    user: str | None = None

    @model_validator(mode="after")
    def validate_messages(self) -> SpecSLOChatRequest:
        if not self.messages:
            raise ValueError("messages must not be empty")
        return self


@dataclass(frozen=True)
class SpecSLOGenerationResult:
    request_id: str
    text: str
    token_ids: tuple[int, ...]
    prompt_tokens: int
    metrics: dict[str, Any]


@dataclass
class _GenerationJob:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    future: asyncio.Future[SpecSLOGenerationResult]
    stream_queue: asyncio.Queue[dict[str, Any] | None] | None = None
    cancelled: bool = False
    abort_notified: bool = False
    admitted_at: float = field(default_factory=time.time)

    @property
    def sampling_signature(self) -> tuple[bool, bool]:
        """Return the native batcher's two greedy/stochastic mode bits."""
        return (
            self.sampling_params.temperature > 0,
            self.sampling_params.draft_temperature > 0,
        )


@dataclass
class _LiveEpoch:
    jobs: dict[str, _GenerationJob]
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, job: _GenerationJob) -> None:
        with self.lock:
            if job.request_id in self.jobs:
                raise ValueError(f"duplicate SpecSLO request id {job.request_id!r}")
            self.jobs[job.request_id] = job

    def remove(self, request_id: str) -> None:
        with self.lock:
            self.jobs.pop(request_id, None)

    def get(self, request_id: str) -> _GenerationJob | None:
        with self.lock:
            return self.jobs.get(request_id)

    def snapshot(self) -> list[_GenerationJob]:
        with self.lock:
            return list(self.jobs.values())


@dataclass(frozen=True)
class SpecSLOGenerationHandle:
    """One admitted HTTP request and its optional commit stream."""

    request_id: str
    future: asyncio.Future[SpecSLOGenerationResult]
    stream_queue: asyncio.Queue[dict[str, Any] | None] | None
    _job: _GenerationJob
    _cancel_callback: Callable[[str], None]

    def cancel_delivery(self) -> None:
        # Worker invalidation occurs at the next collective commit boundary;
        # no rank abandons an in-flight HCCL operation independently.
        if not self._job.cancelled:
            self._job.cancelled = True
            self._cancel_callback(self.request_id)

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        if self.stream_queue is None:
            raise RuntimeError("This SpecSLO request was not submitted with stream=True")
        while True:
            event = await self.stream_queue.get()
            if event is None:
                break
            yield event


class SpecSLOHTTPService:
    """Own one persistent dual-model worker set and serialize model batches."""

    def __init__(
        self,
        config: PEARLConfig,
        *,
        engine_factory: Callable[[PEARLConfig], PEARLEngine] = PEARLEngine,
        batch_wait_ms: float = 2.0,
        max_batch_size: int | None = None,
        shutdown_timeout_seconds: float = 30.0,
    ) -> None:
        if batch_wait_ms < 0:
            raise ValueError("SpecSLO HTTP batch_wait_ms must be non-negative")
        if shutdown_timeout_seconds <= 0:
            raise ValueError("SpecSLO HTTP shutdown timeout must be positive")
        configured_capacity = config.max_num_queued_seqs or config.max_num_seqs
        self.max_batch_size = configured_capacity if max_batch_size is None else int(max_batch_size)
        if not 0 < self.max_batch_size <= configured_capacity:
            raise ValueError("SpecSLO HTTP max batch size must fit the configured request capacity")
        self.config = config
        self.engine_factory = engine_factory
        self.batch_wait_seconds = batch_wait_ms / 1000.0
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self.engine: PEARLEngine | None = None
        self._queue: asyncio.Queue[_GenerationJob | None] = asyncio.Queue()
        self._deferred: deque[_GenerationJob] = deque()
        self._batch_task: asyncio.Task[None] | None = None
        self._accepting = False
        # Mutated only by the service event loop. Worker threads publish
        # terminal states back through call_soon_threadsafe, so /health can
        # never observe a resolved Future with stale in-flight accounting.
        self._active_request_ids: set[str] = set()
        self._batches = 0
        self._completed = 0
        self._failed = 0
        self._aborted = 0
        self._jobs: dict[str, _GenerationJob] = {}

    @property
    def tokenizer(self):
        if self.engine is None:
            raise RuntimeError("SpecSLO HTTP service has not started")
        return self.engine.tokenizer

    async def start(self) -> None:
        if self._batch_task is not None:
            return
        self.engine = await asyncio.to_thread(self.engine_factory, self.config)
        self._accepting = True
        self._batch_task = asyncio.create_task(self._batch_loop(), name="specslo-http-batcher")

    async def stop(self) -> None:
        self._accepting = False
        task = self._batch_task
        if task is not None:
            await self._queue.put(None)
            try:
                await asyncio.wait_for(task, timeout=self.shutdown_timeout_seconds)
            except TimeoutError:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            self._batch_task = None
        engine, self.engine = self.engine, None
        if engine is not None:
            await asyncio.to_thread(engine.exit)

    def health(self) -> dict[str, Any]:
        workers = []
        if self.engine is not None:
            workers = [
                {
                    "rank": rank,
                    "pid": process.pid,
                    "alive": process.is_alive(),
                    "exitcode": process.exitcode,
                }
                for rank, process in enumerate(getattr(self.engine, "_processes", ()))
            ]
        return {
            "accepting": self._accepting,
            "queue_depth": self._queue.qsize() + len(self._deferred),
            "inflight_requests": len(self._active_request_ids),
            "batches": self._batches,
            "completed_requests": self._completed,
            "failed_requests": self._failed,
            "aborted_requests": self._aborted,
            "workers": workers,
            "healthy": self._accepting and all(worker["alive"] for worker in workers),
        }

    def encode_completion(self, prompt: str | list[int]) -> list[int]:
        if isinstance(prompt, str):
            tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            tokens = [int(token_id) for token_id in prompt]
        if not tokens:
            raise ValueError("SpecSLO completion prompt must not be empty")
        return list(tokens)

    def encode_chat(self, messages: Sequence[SpecSLOChatMessage]) -> list[int]:
        payload = [{"role": message.role, "content": message.content} for message in messages]
        tokens = self.tokenizer.apply_chat_template(payload, tokenize=True, add_generation_prompt=True)
        if not tokens:
            raise ValueError("SpecSLO chat template produced an empty prompt")
        return [int(token_id) for token_id in tokens]

    async def submit(
        self,
        prompt_token_ids: Sequence[int],
        sampling_params: SamplingParams,
        *,
        request_id: str | None = None,
        stream: bool = False,
    ) -> SpecSLOGenerationHandle:
        if not self._accepting or self.engine is None:
            raise RuntimeError("SpecSLO HTTP service is not accepting requests")
        tokens = [int(token_id) for token_id in prompt_token_ids]
        if not tokens:
            raise ValueError("SpecSLO requests require a non-empty prompt")
        external_id = request_id or f"specslo-{uuid.uuid4().hex}"
        if external_id in self._jobs:
            raise ValueError(f"duplicate SpecSLO request id {external_id!r}")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[SpecSLOGenerationResult] = loop.create_future()
        stream_queue = asyncio.Queue() if stream else None
        job = _GenerationJob(
            request_id=external_id,
            prompt_token_ids=tokens,
            sampling_params=replace(
                sampling_params,
                request_id=external_id,
                arrival_ts=time.time() if sampling_params.arrival_ts is None else sampling_params.arrival_ts,
            ),
            future=future,
            stream_queue=stream_queue,
        )
        self._jobs[external_id] = job
        await self._queue.put(job)
        return SpecSLOGenerationHandle(external_id, future, stream_queue, job, self._schedule_abort)

    def _schedule_abort(self, request_id: str) -> None:
        loop = asyncio.get_running_loop()
        loop.create_task(self.abort_requests((request_id,)))

    async def abort_requests(self, request_ids: Sequence[str]) -> int:
        """Cancel queued work and invalidate live worker state at a safe fence."""

        jobs = [self._jobs[value] for value in dict.fromkeys(request_ids) if value in self._jobs]
        newly_aborted = 0
        notify_workers: list[_GenerationJob] = []
        for job in jobs:
            if not job.cancelled:
                job.cancelled = True
            if not job.future.done():
                job.future.cancel()
            if job.stream_queue is not None:
                await job.stream_queue.put(None)
            if not job.abort_notified:
                job.abort_notified = True
                notify_workers.append(job)
                newly_aborted += 1
        self._aborted += newly_aborted
        engine = self.engine
        abort_live = None if engine is None else getattr(engine, "abort_live_requests", None)
        if notify_workers and callable(abort_live):
            await asyncio.to_thread(abort_live, [job.request_id for job in notify_workers])
        return newly_aborted

    async def _next_job(self) -> _GenerationJob | None:
        while True:
            job = self._deferred.popleft() if self._deferred else await self._queue.get()
            if job is None or not job.cancelled:
                return job
            self._jobs.pop(job.request_id, None)

    async def _collect_batch(self, first: _GenerationJob) -> list[_GenerationJob]:
        jobs = [first]
        deadline = asyncio.get_running_loop().time() + self.batch_wait_seconds
        while len(jobs) < self.max_batch_size:
            timeout = deadline - asyncio.get_running_loop().time()
            if timeout <= 0:
                break
            try:
                candidate = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            except TimeoutError:
                break
            if candidate is None:
                # Preserve shutdown ordering after every already-admitted job.
                await self._queue.put(None)
                break
            if candidate.cancelled:
                self._jobs.pop(candidate.request_id, None)
                continue
            # Native PEARL intentionally rejects mixed greedy/stochastic
            # rows. Keep each model batch sampling-homogeneous while leaving
            # incompatible jobs admitted for the next cycle.
            if candidate.sampling_signature != first.sampling_signature:
                self._deferred.append(candidate)
                continue
            jobs.append(candidate)
        return jobs

    async def _batch_loop(self) -> None:
        while True:
            first = await self._next_job()
            if first is None:
                if self._deferred:
                    continue
                break
            jobs = await self._collect_batch(first)
            self._active_request_ids.update(job.request_id for job in jobs)
            self._batches += 1
            if callable(getattr(self.engine, "generate_live", None)):
                stop = await self._run_live_epoch(jobs)
                if stop:
                    break
            else:
                await asyncio.to_thread(self._run_batch, jobs, asyncio.get_running_loop())

    async def _run_live_epoch(self, initial_jobs: list[_GenerationJob]) -> bool:
        """Feed compatible arrivals into workers while their decode is live."""
        assert self.engine is not None
        loop = asyncio.get_running_loop()
        epoch = _LiveEpoch({job.request_id: job for job in initial_jobs})
        generation = asyncio.create_task(
            asyncio.to_thread(self._run_batch, initial_jobs, loop, epoch),
            name="specslo-live-generation",
        )
        stop_requested = False
        while not generation.done():
            arrival = asyncio.create_task(self._queue.get())
            done, _ = await asyncio.wait((generation, arrival), return_when=asyncio.FIRST_COMPLETED)
            if generation in done:
                if arrival.done() and not arrival.cancelled():
                    candidate = arrival.result()
                    if candidate is None:
                        stop_requested = True
                    else:
                        self._deferred.append(candidate)
                else:
                    arrival.cancel()
                    with suppress(asyncio.CancelledError):
                        await arrival
                break
            candidate = arrival.result()
            if candidate is None:
                stop_requested = True
                break
            if candidate.sampling_signature != initial_jobs[0].sampling_signature:
                self._deferred.append(candidate)
                continue
            epoch.add(candidate)
            try:
                await asyncio.to_thread(
                    self.engine.admit_live_requests,
                    [(candidate.prompt_token_ids, candidate.sampling_params)],
                )
            except RuntimeError:
                # The worker idle fence won the race. Preserve the request for
                # a new epoch instead of writing after the close marker.
                epoch.remove(candidate.request_id)
                self._deferred.append(candidate)
                break
            self._active_request_ids.add(candidate.request_id)
        await generation
        return stop_requested

    def _publish_result(self, job: _GenerationJob, result: SpecSLOGenerationResult) -> None:
        """Atomically publish one worker result on the service event loop."""

        self._active_request_ids.discard(job.request_id)
        self._jobs.pop(job.request_id, None)
        if job.cancelled:
            return
        self._completed += 1
        if not job.future.done():
            job.future.set_result(result)
        if job.stream_queue is not None:
            job.stream_queue.put_nowait(None)

    def _publish_failure(self, job: _GenerationJob, error: BaseException) -> None:
        """Atomically publish one worker failure on the service event loop."""

        self._active_request_ids.discard(job.request_id)
        self._jobs.pop(job.request_id, None)
        if job.cancelled:
            return
        self._failed += 1
        if not job.future.done():
            job.future.set_exception(error)
        if job.stream_queue is not None:
            job.stream_queue.put_nowait({"error": str(error)})
            job.stream_queue.put_nowait(None)

    def _run_batch(
        self,
        jobs: list[_GenerationJob],
        loop: asyncio.AbstractEventLoop,
        live_epoch: _LiveEpoch | None = None,
    ) -> None:
        engine = self.engine
        if engine is None:
            error = RuntimeError("SpecSLO engine stopped before an admitted batch ran")
            self._fail_jobs(jobs, error, loop)
            return
        by_id = {job.request_id: job for job in jobs}
        try:
            for job in jobs:
                engine.add_request(job.prompt_token_ids, job.sampling_params)

            def deliver(event: dict[str, Any]) -> None:
                request_id = str(event.get("request_id"))
                job = live_epoch.get(request_id) if live_epoch is not None else by_id.get(request_id)
                if job is None or job.stream_queue is None or job.cancelled:
                    return
                token_ids = tuple(int(value) for value in event.get("token_ids", ()))
                payload = {
                    **event,
                    "token_ids": token_ids,
                    "text": engine.tokenizer.decode(token_ids, skip_special_tokens=False),
                }
                loop.call_soon_threadsafe(job.stream_queue.put_nowait, payload)

            generate = engine.generate_live if live_epoch is not None else engine.generate
            texts, _, _, _ = generate(on_token_commit=deliver)
            metrics = list(engine.last_metrics)
            expected_jobs = live_epoch.snapshot() if live_epoch is not None else jobs
            if len(texts) != len(expected_jobs) or len(metrics) != len(expected_jobs):
                raise RuntimeError("SpecSLO workers returned a result count different from the admitted batch")
            completed_ids: set[str] = set()
            completed: list[tuple[_GenerationJob, SpecSLOGenerationResult]] = []
            for text, row in zip(texts, metrics):
                request_id = str(row.get("request_id"))
                job = live_epoch.get(request_id) if live_epoch is not None else by_id.get(request_id)
                if job is None:
                    raise RuntimeError(f"SpecSLO workers returned unknown request {request_id!r}")
                completed_ids.add(request_id)
                token_ids = tuple(int(value) for value in row.get("completion_token_ids", ()))
                completed.append(
                    (
                        job,
                        SpecSLOGenerationResult(
                            request_id=job.request_id,
                            text=str(text),
                            token_ids=token_ids,
                            prompt_tokens=len(job.prompt_token_ids),
                            metrics=dict(row),
                        ),
                    )
                )
            if completed_ids != {job.request_id for job in expected_jobs}:
                raise RuntimeError("SpecSLO workers omitted one or more admitted request ids")
            for job, result in completed:
                loop.call_soon_threadsafe(self._publish_result, job, result)
        except BaseException as error:
            self._fail_jobs(live_epoch.snapshot() if live_epoch is not None else jobs, error, loop)

    def _fail_jobs(
        self,
        jobs: Sequence[_GenerationJob],
        error: BaseException,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        for job in jobs:
            loop.call_soon_threadsafe(self._publish_failure, job, error)


def _sampling_params(request: SpecSLOCompletionRequest | SpecSLOChatRequest) -> SamplingParams:
    return SamplingParams(
        temperature=request.temperature,
        draft_temperature=request.draft_temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        draft_top_p=request.draft_top_p,
        draft_top_k=request.draft_top_k,
        max_tokens=request.max_tokens,
        ignore_eos=request.ignore_eos,
        slo_tpot_ms=request.slo_tpot_ms,
        slo_class=request.slo_class,
        spec_rhythm_max_gamma=request.spec_rhythm_max_gamma,
    )


def _validate_requested_model(config: PEARLConfig, model: str | None) -> None:
    accepted = {
        config.target_model_path,
        config.target_model_path.rstrip("/").rsplit("/", 1)[-1],
    }
    if model is not None and model not in accepted:
        raise HTTPException(status_code=404, detail=f"SpecSLO does not serve model {model!r}")


def _completion_payload(result: SpecSLOGenerationResult, model: str, created: int) -> dict[str, Any]:
    completion_tokens = len(result.token_ids)
    finish_reason = str(result.metrics.get("finish_reason", "stop"))
    return {
        "id": result.request_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "text": result.text, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": result.prompt_tokens + completion_tokens,
        },
        "specslo": result.metrics,
    }


def _chat_payload(result: SpecSLOGenerationResult, model: str, created: int) -> dict[str, Any]:
    payload = _completion_payload(result, model, created)
    payload["object"] = "chat.completion"
    payload["choices"] = [
        {
            "index": 0,
            "message": {"role": "assistant", "content": result.text},
            "finish_reason": str(result.metrics.get("finish_reason", "stop")),
        }
    ]
    return payload


async def _stream_response(
    handle: SpecSLOGenerationHandle,
    *,
    raw_request: Request,
    model: str,
    chat: bool,
) -> AsyncIterator[str]:
    created = int(time.time())
    try:
        async for event in handle.events():
            if await raw_request.is_disconnected():
                handle.cancel_delivery()
                break
            if "error" in event:
                yield f"data: {json.dumps({'error': event['error']})}\n\n"
                break
            choice = {
                "index": 0,
                "finish_reason": None,
            }
            if chat:
                choice["delta"] = {"content": event["text"]}
                object_name = "chat.completion.chunk"
            else:
                choice["text"] = event["text"]
                object_name = "text_completion"
            payload = {
                "id": handle.request_id,
                "object": object_name,
                "created": created,
                "model": model,
                "choices": [choice],
                "specslo": {"token_ids": event["token_ids"]},
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        if not handle._job.cancelled:
            yield "data: [DONE]\n\n"
    except asyncio.CancelledError:
        handle.cancel_delivery()
        raise


def create_specslo_app(
    config: PEARLConfig,
    *,
    engine_factory: Callable[[PEARLConfig], PEARLEngine] = PEARLEngine,
    batch_wait_ms: float = 2.0,
    max_batch_size: int | None = None,
) -> FastAPI:
    """Create an OpenAI-compatible app backed by one persistent SpecSLO engine."""

    service = SpecSLOHTTPService(
        config,
        engine_factory=engine_factory,
        batch_wait_ms=batch_wait_ms,
        max_batch_size=max_batch_size,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await service.start()
        try:
            yield
        finally:
            await service.stop()

    app = FastAPI(title="vLLM-Ascend SpecSLO", lifespan=lifespan)
    app.state.specslo_service = service

    @app.get("/health")
    async def health() -> JSONResponse:
        status = service.health()
        return JSONResponse(status, status_code=200 if status["healthy"] else 503)

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": config.target_model_path.rstrip("/").rsplit("/", 1)[-1],
                    "object": "model",
                    "owned_by": "vllm-ascend-specslo",
                }
            ],
        }

    @app.post("/v1/completions")
    async def completions(payload: SpecSLOCompletionRequest, raw_request: Request):
        _validate_requested_model(config, payload.model)
        try:
            prompt = service.encode_completion(payload.prompt)
            handle = await service.submit(prompt, _sampling_params(payload), stream=payload.stream)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        model = payload.model or config.target_model_path.rstrip("/").rsplit("/", 1)[-1]
        if payload.stream:
            return StreamingResponse(
                _stream_response(handle, raw_request=raw_request, model=model, chat=False),
                media_type="text/event-stream",
            )
        try:
            result = await handle.future
        except Exception as error:
            raise HTTPException(status_code=500, detail=str(error)) from error
        return _completion_payload(result, model, int(time.time()))

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: SpecSLOChatRequest, raw_request: Request):
        _validate_requested_model(config, payload.model)
        try:
            prompt = service.encode_chat(payload.messages)
            handle = await service.submit(prompt, _sampling_params(payload), stream=payload.stream)
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        model = payload.model or config.target_model_path.rstrip("/").rsplit("/", 1)[-1]
        if payload.stream:
            return StreamingResponse(
                _stream_response(handle, raw_request=raw_request, model=model, chat=True),
                media_type="text/event-stream",
            )
        try:
            result = await handle.future
        except Exception as error:
            raise HTTPException(status_code=500, detail=str(error)) from error
        return _chat_payload(result, model, int(time.time()))

    return app


__all__ = [
    "SpecSLOChatMessage",
    "SpecSLOChatRequest",
    "SpecSLOCompletionRequest",
    "SpecSLOGenerationHandle",
    "SpecSLOGenerationResult",
    "SpecSLOHTTPService",
    "create_specslo_app",
]
