# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import time
from types import SimpleNamespace

from vllm.sampling_params import SamplingParams as V1SamplingParams
from vllm.v1.engine import EngineCoreRequest, FinishReason

from vllm_ascend.spec_decode.pearl.http_server import SpecSLOHTTPService
from vllm_ascend.spec_decode.pearl.v1_client import (
    SpecSLOV1EngineCoreClient,
    sampling_params_from_v1,
)

from .test_specslo_http_server import _FakeEngine


def _config():
    return SimpleNamespace(
        max_num_queued_seqs=8,
        max_num_seqs=4,
        target_model_path="/models/Qwen3-32B",
    )


def _request(*, extra_args=None, abort_immediately=False) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="v1-request",
        prompt_token_ids=[1, 2],
        mm_features=None,
        sampling_params=V1SamplingParams(
            temperature=0.7,
            top_p=0.9,
            top_k=20,
            max_tokens=2,
            extra_args=extra_args,
        ),
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        abort_immediately=abort_immediately,
    )


def test_v1_sampling_translation_preserves_specslo_controls() -> None:
    params = sampling_params_from_v1(
        _request(
            extra_args={
                "draft_temperature": 0.6,
                "draft_top_p": 0.8,
                "draft_top_k": 10,
                "slo_tpot_ms": 40,
                "slo_class": "tight",
                "spec_rhythm_max_gamma": 5,
            }
        )
    )
    assert (params.temperature, params.top_p, params.top_k) == (0.7, 0.9, 20)
    assert (params.draft_temperature, params.draft_top_p, params.draft_top_k) == (0.6, 0.8, 10)
    assert (params.slo_tpot_ms, params.slo_class, params.spec_rhythm_max_gamma) == (40, "tight", 5)


def test_v1_client_owns_workers_and_streams_engine_core_outputs() -> None:
    _FakeEngine.instances.clear()

    def service_factory(config):
        return SpecSLOHTTPService(config, engine_factory=_FakeEngine, batch_wait_ms=0)

    client = SpecSLOV1EngineCoreClient(_config(), service_factory=service_factory)
    try:
        client.add_request(_request())
        token_ids = []
        finish_reason = None
        while finish_reason is None:
            outputs = client.get_output()
            assert len(outputs.outputs) == 1
            output = outputs.outputs[0]
            token_ids.extend(output.new_token_ids)
            finish_reason = output.finish_reason
        assert token_ids == [ord("O"), ord("K")]
        assert finish_reason == FinishReason.STOP
        assert outputs.finished_requests == {"v1-request"}
        assert len(_FakeEngine.instances) == 1
    finally:
        client.shutdown()
    assert _FakeEngine.instances[-1].exited


def test_v1_abort_immediately_never_enters_model_queue() -> None:
    _FakeEngine.instances.clear()

    def service_factory(config):
        return SpecSLOHTTPService(config, engine_factory=_FakeEngine, batch_wait_ms=0)

    client = SpecSLOV1EngineCoreClient(_config(), service_factory=service_factory)
    try:
        client.add_request(_request(abort_immediately=True))
        output = client.get_output().outputs[0]
        assert output.finish_reason == FinishReason.ABORT
        assert not _FakeEngine.instances[-1].generate_batches
    finally:
        client.shutdown()
