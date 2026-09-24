# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the native target-only benchmark contract."""

from __future__ import annotations

import hashlib
import json

import pytest

from examples import benchmark_nano_pearl_native_target_only as benchmark
from vllm_ascend.spec_decode import pearl


def test_graph_cache_seal_requires_one_batch_point() -> None:
    with pytest.raises(ValueError, match="requires exactly one batch size"):
        benchmark.main(
            [
                "--draft-model",
                "draft",
                "--target-model",
                "target",
                "--batch-sizes",
                "1",
                "2",
                "--seal-graph-cache-after-warmup",
                "--prompt",
                "hello",
            ]
        )


def test_graph_cache_is_sealed_between_warmup_and_measurement(monkeypatch, tmp_path) -> None:
    calls: list[str] = []

    class FakeTokenizer:
        @staticmethod
        def apply_chat_template(*_args, **_kwargs):
            return "formatted"

        @staticmethod
        def encode(_prompt):
            return [1, 2]

    class FakeEngine:
        tokenizer = FakeTokenizer()

        def __init__(self, _config):
            self.last_worker_metrics = [{"rank": 0, "aclgraph_sealed": 0}]
            self.last_metrics = []
            self.generations = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def add_request(self, _prompt, _params):
            calls.append("add")

        def AR_generate(self):
            self.generations += 1
            calls.append(f"generate-{self.generations}")
            self.last_metrics = [
                {
                    "completion_token_ids": [10, 11],
                    "prefill_elapsed_seconds": 0.01,
                    "decode_elapsed_seconds": 0.02,
                }
            ]
            if self.generations == 2:
                self.last_worker_metrics = [{"rank": 0, "aclgraph_sealed": 1, "aclgraph_replays": 2}]
            return None, [2], None, 0.1

        def seal_graph_cache(self):
            calls.append("seal")
            self.last_worker_metrics = [{"rank": 0, "aclgraph_sealed": 1, "aclgraph_replays": 1}]
            return self.last_worker_metrics

    class FakeConfig:
        def __init__(self, **_kwargs):
            pass

    class FakeSamplingParams:
        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(pearl, "PEARLConfig", FakeConfig)
    monkeypatch.setattr(pearl, "PEARLEngine", FakeEngine)
    monkeypatch.setattr(pearl, "SamplingParams", FakeSamplingParams)
    output = tmp_path / "result.json"

    benchmark.main(
        [
            "--draft-model",
            "draft",
            "--target-model",
            "target",
            "--batch-sizes",
            "1",
            "--num-prompts",
            "1",
            "--warmup-prompts",
            "1",
            "--max-tokens",
            "2",
            "--seal-graph-cache-after-warmup",
            "--prompt",
            "hello",
            "--output-json",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    row = payload["results"][0]
    assert calls == ["add", "generate-1", "seal", "add", "generate-2"]
    assert payload["seal_graph_cache_after_warmup"] is True
    assert row["worker_metrics_before_measurement"] == [{"rank": 0, "aclgraph_sealed": 1, "aclgraph_replays": 1}]
    assert row["worker_metrics_after_measurement"] == [{"rank": 0, "aclgraph_sealed": 1, "aclgraph_replays": 2}]


def test_mc2_profile_and_execution_mode_provenance(monkeypatch, tmp_path) -> None:
    profile = tmp_path / "profile.json"
    profile_bytes = json.dumps(
        {"metadata": {"rms_norm_epsilon": 1e-6}},
        separators=(",", ":"),
    ).encode()
    profile.write_bytes(profile_bytes)

    class FakeTokenizer:
        @staticmethod
        def apply_chat_template(*_args, **_kwargs):
            return "formatted"

        @staticmethod
        def encode(_prompt):
            return [1]

    class FakeEngine:
        tokenizer = FakeTokenizer()

        def __init__(self, _config):
            self.last_worker_metrics = []
            self.last_metrics = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def add_request(self, _prompt, _params):
            pass

        def AR_generate(self):
            self.last_metrics = [
                {
                    "completion_token_ids": [10],
                    "prefill_elapsed_seconds": 0.01,
                    "decode_elapsed_seconds": 0.02,
                }
            ]
            return None, [1], None, 0.1

    class FakeConfig:
        def __init__(self, **_kwargs):
            pass

    class FakeSamplingParams:
        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(pearl, "PEARLConfig", FakeConfig)
    monkeypatch.setattr(pearl, "PEARLEngine", FakeEngine)
    monkeypatch.setattr(pearl, "SamplingParams", FakeSamplingParams)
    output = tmp_path / "result.json"

    benchmark.main(
        [
            "--draft-model",
            "draft",
            "--target-model",
            "target",
            "--batch-sizes",
            "1",
            "--max-tokens",
            "1",
            "--enable-mc2",
            "--mc2-profile",
            str(profile),
            "--enforce-eager",
            "--prompt",
            "hello",
            "--output-json",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["execution_mode"] == "eager"
    assert payload["enforce_eager"] is True
    assert payload["mc2_profile_resolved"] == str(profile.resolve())
    assert payload["mc2_profile_sha256"] == hashlib.sha256(profile_bytes).hexdigest()
    assert payload["mc2_profile_rms_norm_epsilon"] == 1e-6
