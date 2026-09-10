# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM V1 EngineCore protocol adapter for the native SpecSLO runtime.

The stock V1 core owns one model/executor.  SpecSLO instead owns one HCCL
world containing separate draft and target TP groups, so it cannot be modeled
as an ordinary speculative proposer inside a single V1 model runner.  This
adapter terminates the V1 request/output protocol at the frontend boundary and
keeps the dual-model worker lifecycle inside :class:`PEARLEngine`.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import math
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import suppress

from vllm.sampling_params import SamplingParams as V1SamplingParams
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.engine.core_client import EngineCoreClient

from vllm_ascend.spec_decode.pearl.api import PEARLConfig, SamplingParams
from vllm_ascend.spec_decode.pearl.http_server import (
    SpecSLOGenerationHandle,
    SpecSLOHTTPService,
)


def _reject_unsupported_request(request: EngineCoreRequest) -> V1SamplingParams:
    if request.prompt_token_ids is None or not request.prompt_token_ids:
        raise ValueError("SpecSLO V1 requires a non-empty tokenized prompt")
    if request.mm_features or request.prompt_embeds is not None:
        raise NotImplementedError("SpecSLO V1 does not yet support multimodal inputs or prompt embeddings")
    if request.pooling_params is not None:
        raise NotImplementedError("SpecSLO V1 is a generation-only engine")
    if request.lora_request is not None:
        raise NotImplementedError("SpecSLO V1 does not yet support LoRA requests")
    params = request.sampling_params
    if params is None:
        raise ValueError("SpecSLO V1 requires sampling_params")
    if params.n != 1:
        raise NotImplementedError("SpecSLO V1 currently supports n=1 only")
    if params.logprobs is not None or params.prompt_logprobs is not None:
        raise NotImplementedError("SpecSLO V1 does not yet return logprobs")
    if params.structured_outputs is not None:
        raise NotImplementedError("SpecSLO V1 does not yet support structured output")
    if params.stop or params.stop_token_ids or params.min_tokens:
        raise NotImplementedError("SpecSLO V1 currently supports model EOS and max_tokens stopping only")
    if (
        params.presence_penalty != 0
        or params.frequency_penalty != 0
        or params.repetition_penalty != 1
        or params.min_p != 0
        or params.logit_bias
        or params.allowed_token_ids
        or params.bad_words
    ):
        raise NotImplementedError("SpecSLO V1 does not silently ignore logits processors")
    if params.max_tokens is None or params.max_tokens <= 0:
        raise ValueError("SpecSLO V1 max_tokens must be positive")
    return params


def sampling_params_from_v1(request: EngineCoreRequest) -> SamplingParams:
    """Translate the supported V1 sampling surface without changing meaning."""

    params = _reject_unsupported_request(request)
    extra = dict(params.extra_args or {})
    supported_extra = {
        "draft_temperature",
        "draft_top_p",
        "draft_top_k",
        "slo_tpot_ms",
        "slo_class",
        "spec_rhythm_max_gamma",
    }
    unknown = set(extra) - supported_extra
    if unknown:
        raise NotImplementedError(f"Unsupported SpecSLO V1 extra_args: {sorted(unknown)}")
    arrival = float(request.arrival_time)
    if not math.isfinite(arrival):
        raise ValueError("SpecSLO V1 arrival_time must be finite")
    # V1 frontends normally use wall time, but tolerate a monotonic timestamp
    # from custom callers without delaying admission until an impossible epoch.
    if arrival < 1_000_000_000:
        arrival = time.time()
    return SamplingParams(
        temperature=float(params.temperature),
        draft_temperature=float(extra.get("draft_temperature", 0.0)),
        top_p=float(params.top_p),
        top_k=int(params.top_k),
        draft_top_p=float(extra.get("draft_top_p", 1.0)),
        draft_top_k=int(extra.get("draft_top_k", 0)),
        max_tokens=int(params.max_tokens),
        ignore_eos=bool(params.ignore_eos),
        slo_tpot_ms=(None if extra.get("slo_tpot_ms") is None else float(extra["slo_tpot_ms"])),
        slo_class=(None if extra.get("slo_class") is None else str(extra["slo_class"])),
        spec_rhythm_max_gamma=(
            None if extra.get("spec_rhythm_max_gamma") is None else int(extra["spec_rhythm_max_gamma"])
        ),
        request_id=request.request_id,
        arrival_ts=arrival,
    )


class SpecSLOV1EngineCoreClient(EngineCoreClient):
    """Protocol-compatible V1 client backed by persistent SpecSLO workers.

    A private asyncio loop owns the service so both synchronous V1 callers and
    ``AsyncLLM``-style callers use the same lifecycle.  No model is recreated
    between requests or HTTP batches.
    """

    def __init__(
        self,
        config: PEARLConfig,
        *,
        service_factory: Callable[[PEARLConfig], SpecSLOHTTPService] = SpecSLOHTTPService,
        startup_timeout_seconds: float = 600.0,
    ) -> None:
        self.config = config
        self._service = service_factory(config)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="specslo-v1-core", daemon=True)
        self._thread.start()
        self._closed = False
        self._consumers: dict[str, asyncio.Task[None]] = {}
        self._outputs: asyncio.Queue[EngineCoreOutputs] | None = None
        self._submit(self._start()).result(timeout=startup_timeout_seconds)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coroutine) -> concurrent.futures.Future:
        if self._closed:
            raise RuntimeError("SpecSLO V1 client is closed")
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    async def _start(self) -> None:
        self._outputs = asyncio.Queue()
        await self._service.start()

    async def _await(self, future: concurrent.futures.Future):
        return await asyncio.wrap_future(future)

    async def _add_request(self, request: EngineCoreRequest) -> None:
        if request.request_id in self._consumers:
            raise ValueError(f"duplicate V1 request id {request.request_id!r}")
        if request.abort_immediately:
            await self._put_terminal(request.request_id, FinishReason.ABORT, request.trace_headers)
            return
        params = sampling_params_from_v1(request)
        handle = await self._service.submit(
            request.prompt_token_ids or (),
            params,
            request_id=request.request_id,
            stream=True,
        )
        task = asyncio.create_task(
            self._consume(request, handle),
            name=f"specslo-v1-request-{request.request_id}",
        )
        self._consumers[request.request_id] = task
        task.add_done_callback(lambda _task, request_id=request.request_id: self._consumers.pop(request_id, None))

    async def _put(self, output: EngineCoreOutput, *, finished: bool = False) -> None:
        assert self._outputs is not None
        await self._outputs.put(
            EngineCoreOutputs(
                outputs=[output],
                finished_requests={output.request_id} if finished else None,
            )
        )

    async def _put_terminal(
        self,
        request_id: str,
        reason: FinishReason,
        trace_headers,
        *,
        stop_reason: str | None = None,
        token_ids: Sequence[int] = (),
    ) -> None:
        await self._put(
            EngineCoreOutput(
                request_id=request_id,
                new_token_ids=[int(value) for value in token_ids],
                finish_reason=reason,
                stop_reason=stop_reason,
                trace_headers=trace_headers,
            ),
            finished=True,
        )

    async def _consume(self, request: EngineCoreRequest, handle: SpecSLOGenerationHandle) -> None:
        delivered = 0
        try:
            async for event in handle.events():
                token_ids = [int(value) for value in event.get("token_ids", ())]
                delivered += len(token_ids)
                await self._put(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=token_ids,
                        trace_headers=request.trace_headers,
                    )
                )
            result = await handle.future
            remaining = result.token_ids[delivered:]
            finish = str(result.metrics.get("finish_reason", "stop"))
            reason = FinishReason.LENGTH if finish == "length" else FinishReason.STOP
            await self._put_terminal(
                request.request_id,
                reason,
                request.trace_headers,
                token_ids=remaining,
            )
        except asyncio.CancelledError:
            await self._put_terminal(request.request_id, FinishReason.ABORT, request.trace_headers)
        except Exception as error:
            await self._put_terminal(
                request.request_id,
                FinishReason.ERROR,
                request.trace_headers,
                stop_reason=str(error),
            )

    def add_request(self, request: EngineCoreRequest) -> None:
        self._submit(self._add_request(request)).result()

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        await self._await(self._submit(self._add_request(request)))

    def get_output(self) -> EngineCoreOutputs:
        return self._submit(self._get_output()).result()

    async def get_output_async(self) -> EngineCoreOutputs:
        return await self._await(self._submit(self._get_output()))

    async def _get_output(self) -> EngineCoreOutputs:
        assert self._outputs is not None
        return await self._outputs.get()

    def abort_requests(self, request_ids: list[str]) -> None:
        self._submit(self._service.abort_requests(request_ids)).result()

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        await self._await(self._submit(self._service.abort_requests(request_ids)))

    def get_supported_tasks(self) -> tuple[str, ...]:
        return ("generate",)

    async def get_supported_tasks_async(self) -> tuple[str, ...]:
        return self.get_supported_tasks()

    def shutdown(self, timeout: float | None = None) -> None:
        if self._closed:
            return
        deadline = 30.0 if timeout is None else float(timeout)
        future = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        with suppress(Exception):
            future.result(timeout=deadline)
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=deadline)
        if self._thread.is_alive():
            raise TimeoutError("SpecSLO V1 event loop did not stop before the shutdown deadline")
        self._loop.close()

    async def _shutdown(self) -> None:
        await self._service.abort_requests(tuple(self._consumers))
        await self._service.stop()
        consumers = list(self._consumers.values())
        for task in consumers:
            task.cancel()
        if consumers:
            await asyncio.gather(*consumers, return_exceptions=True)


__all__ = ["SpecSLOV1EngineCoreClient", "sampling_params_from_v1"]
