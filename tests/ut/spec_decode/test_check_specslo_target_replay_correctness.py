# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the static target replay correctness diagnostic."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from examples import check_specslo_target_replay_correctness as diagnostic
from vllm_ascend.spec_decode.pearl import api as pearl_api
from vllm_ascend.spec_decode.pearl.api import (
    PEARLEngine,
    _configure_worker_target_graph_correctness,
    _target_graph_correctness_environment,
)


@pytest.fixture(autouse=True)
def _restore_correctness_environment():
    """Keep diagnostic mode changes local to each CPU test.

    The worker-opcode tests intentionally exercise ``os.environ.update``.
    Saving the complete diagnostic key set here prevents those process-global
    changes from selecting replay-first or validate-every behavior in a later
    graph-runner test from the same pytest process.
    """
    keys = set(_target_graph_correctness_environment("packed_fia_eager"))
    original = {key: os.environ.get(key) for key in keys}
    try:
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _args(*extra: str):
    return diagnostic._build_parser().parse_args(["--output", "/tmp/target-replay-correctness.json", *extra])


def test_static_correctness_defaults_are_frozen_and_repeat_candidate_three_times():
    args = _args()
    diagnostic._validate_args(args)
    assert args.batch_size == 8
    assert args.num_prompts == 8
    assert args.max_tokens == 32
    assert args.gamma == 4
    assert args.candidate_repeats == 3


@pytest.mark.parametrize(
    "extra",
    [
        ["--batch-size", "4"],
        ["--num-prompts", "7"],
        ["--max-tokens", "31"],
        ["--gamma", "3"],
        ["--candidate-repeats", "2"],
    ],
)
def test_static_correctness_rejects_shape_or_repeat_overrides(extra):
    with pytest.raises(ValueError):
        diagnostic._validate_args(_args(*extra))


def test_case_comparison_covers_tokens_rounds_and_acceptance():
    oracle = {
        "name": "oracle",
        "output_token_ids": [[1, 2], [3, 4]],
        "output_token_ids_sha256": "token-hash",
        "decoded_outputs_sha256": "text-hash",
        "token_counts": [2, 2],
        "global_round_count": 3,
        "request_verification_rounds": [2, 2],
        "accepted_draft_tokens": [4, 5],
        "verified_draft_tokens": [6, 6],
        "acceptance_rates": [4 / 6, 5 / 6],
        "num_acc_tokens": [[2, 2], [3, 2]],
    }
    exact = dict(oracle, name="candidate")
    assert diagnostic._compare_case(exact, oracle)["passed"]

    changed = dict(exact)
    changed["output_token_ids"] = [[1, 9], [3, 4]]
    changed["output_token_ids_sha256"] = "changed"
    changed["request_verification_rounds"] = [3, 2]
    changed["accepted_draft_tokens"] = [3, 5]
    result = diagnostic._compare_case(changed, oracle)
    assert not result["passed"]
    assert result["token_mismatches"][0]["first_difference"] == 1
    assert "request_verification_rounds" in result["unequal_fields"]
    assert "accepted_draft_tokens" in result["unequal_fields"]


class _FakeEngine:
    def __init__(self):
        self.pending = []
        self.modes = []
        self.generate_calls = 0
        self.last_metrics = []
        self.last_worker_metrics = [
            {"rank": 0, "is_draft_rank": 1},
            {"rank": 1, "is_draft_rank": 0},
        ]

    def configure_target_graph_correctness_mode(
        self,
        mode,
        *,
        validate_every_replay=False,
    ):
        assert not self.pending
        self.modes.append((mode, validate_every_replay))
        return [{"mode": mode}, {"mode": mode}]

    def add_request(self, tokens, params, request_id=None):
        assert request_id == f"static-{len(self.pending)}"
        assert params.max_tokens == 32
        self.pending.append(list(tokens))

    def generate(self):
        assert len(self.pending) == 8
        self.generate_calls += 1
        token_rows = [[index] * 32 for index in range(8)]
        self.last_metrics = [
            {
                "completion_token_ids": row,
                "round_count": 8,
                "verification_rounds": 7,
                "accepted_draft_tokens": 20,
                "verified_draft_tokens": 28,
                "acceptance_rate": 20 / 28,
                "num_acc_tokens": [4] * 7,
            }
            for row in token_rows
        ]
        self.pending = []
        return [f"output-{index}" for index in range(8)], [32] * 8, (), 0.0


def test_case_matrix_reuses_one_engine_and_clears_requests_between_modes():
    engine = _FakeEngine()
    prompts = [[index + 1] for index in range(8)]
    cases = diagnostic._run_case_matrix(
        engine,
        prompts,
        candidate_repeats=3,
        validate_every_replay=True,
    )
    assert len(cases) == 6
    assert engine.generate_calls == 6
    assert engine.modes == [
        ("packed_fia_eager", False),
        ("graph_update_first", False),
        ("graph_replay_first", False),
        ("graph_replay_first", False),
        ("graph_replay_first", False),
        ("graph_replay_first", True),
    ]
    assert not engine.pending


def test_target_graph_correctness_environment_is_hermetic(monkeypatch):
    eager = _target_graph_correctness_environment("packed_fia_eager")
    control = _target_graph_correctness_environment("graph_update_first")
    candidate = _target_graph_correctness_environment(
        "graph_replay_first",
        validate_every_replay=True,
    )
    assert eager["VLLM_ASCEND_SPECRHYTHM_DISABLE_TARGET_ACLGRAPH"] == "1"
    assert control["VLLM_ASCEND_SPECRHYTHM_DISABLE_TARGET_ACLGRAPH"] == "0"
    assert control["VLLM_ASCEND_PEARL_TARGET_REPLAY_FIRST_TASK_UPDATE"] == "0"
    assert candidate["VLLM_ASCEND_PEARL_TARGET_REPLAY_FIRST_TASK_UPDATE"] == "1"
    assert candidate["VLLM_ASCEND_PEARL_VALIDATE_GRAPH_REPLAYS"] == "1"
    assert set(eager) == set(control) == set(candidate)
    with pytest.raises(ValueError, match="Unknown target graph correctness mode"):
        _target_graph_correctness_environment("unknown")

    for key in candidate:
        monkeypatch.delenv(key, raising=False)
    idle = SimpleNamespace(cache_allocation=None, cache_block_tables=None)
    assert (
        _configure_worker_target_graph_correctness(
            idle,
            "graph_replay_first",
            validate_every_replay=True,
        )
        == candidate
    )
    busy = SimpleNamespace(cache_allocation=object(), cache_block_tables=None)
    with pytest.raises(RuntimeError, match="released its KV cache"):
        _configure_worker_target_graph_correctness(busy, "packed_fia_eager")


def test_public_engine_dispatches_mode_only_when_controller_is_idle():
    engine = PEARLEngine.__new__(PEARLEngine)
    engine._requests = []
    engine._live_epoch = None
    engine._send_all = MagicMock()
    expected = _target_graph_correctness_environment(
        "graph_replay_first",
        validate_every_replay=True,
    )
    engine._receive_all = MagicMock(
        return_value=[
            ("target_graph_correctness_configured", 0, expected),
            ("target_graph_correctness_configured", 1, expected),
        ]
    )
    assert engine.configure_target_graph_correctness_mode(
        "graph_replay_first",
        validate_every_replay=True,
    ) == [expected, expected]
    engine._send_all.assert_called_once_with(
        (
            "configure_target_graph_correctness",
            "graph_replay_first",
            True,
            None,
        )
    )

    queued = PEARLEngine.__new__(PEARLEngine)
    queued._requests = [(0, [1], object())]
    queued._live_epoch = None
    queued._send_all = MagicMock()
    with pytest.raises(RuntimeError, match="queued requests"):
        queued.configure_target_graph_correctness_mode("packed_fia_eager")
    queued._send_all.assert_not_called()

    live = PEARLEngine.__new__(PEARLEngine)
    live._requests = []
    live._live_epoch = 4
    live._send_all = MagicMock()
    with pytest.raises(RuntimeError, match="live generation"):
        live.configure_target_graph_correctness_mode("packed_fia_eager")
    live._send_all.assert_not_called()


def _target_delta(rank, **overrides):
    row = {
        "rank": rank,
        "is_draft_rank": 0,
        "aclgraph_captures": 0,
        "aclgraph_capture_attempts": 0,
        "aclgraph_replays": 5,
        "aclgraph_failed_captures": 0,
        "aclgraph_capacity_fallbacks": 0,
        "aclgraph_shape_fallbacks": 0,
        "aclgraph_runtime_validation_replays": 0,
        "aclgraph_disabled_entries": 0,
        "aclgraph_generic_total_calls": 5,
        "aclgraph_generic_capture_replay_calls": 0,
        "aclgraph_generic_replay_calls": 5,
        "aclgraph_generic_eager_fallback_calls": 0,
        "aclgraph_generic_runtime_validation_calls": 0,
        "aclgraph_generic_runtime_validation_failures": 0,
        "aclgraph_generic_task_update_replays": 5,
        "aclgraph_generic_task_update_tasks": 320,
    }
    row.update(overrides)
    return row


def test_route_gate_requires_resident_replay_on_each_target_rank_without_recapture():
    case = {
        "name": "candidate",
        "mode": "graph_replay_first",
        "validate_every_replay": False,
        "graph_metric_deltas": [
            {"rank": 0, "is_draft_rank": 1},
            *[_target_delta(rank) for rank in (1, 2, 3)],
        ],
    }
    assert diagnostic._route_check(case)["passed"]

    recaptured = dict(case)
    recaptured["graph_metric_deltas"] = [
        {"rank": 0, "is_draft_rank": 1},
        _target_delta(
            1,
            aclgraph_capture_attempts=1,
            aclgraph_captures=1,
            aclgraph_generic_total_calls=5,
            aclgraph_generic_capture_replay_calls=1,
            aclgraph_generic_replay_calls=4,
        ),
        _target_delta(2),
        _target_delta(3),
    ]
    assert not diagnostic._route_check(recaptured)["passed"]

    missing_rank = dict(case)
    missing_rank["graph_metric_deltas"] = case["graph_metric_deltas"][:-1]
    assert not diagnostic._route_check(missing_rank)["passed"]


def test_validate_every_replay_gate_requires_validation_on_each_target_replay():
    strict = {
        "name": "strict-candidate",
        "mode": "graph_replay_first",
        "validate_every_replay": True,
        "graph_metric_deltas": [
            {"rank": 0, "is_draft_rank": 1},
            *[
                _target_delta(
                    rank,
                    aclgraph_runtime_validation_replays=5,
                    aclgraph_generic_runtime_validation_calls=5,
                )
                for rank in (1, 2, 3)
            ],
        ],
    }
    assert diagnostic._route_check(strict)["passed"]
    strict["graph_metric_deltas"][2]["aclgraph_generic_runtime_validation_calls"] = 4
    assert not diagnostic._route_check(strict)["passed"]


class _ScriptedConnection:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent = []
        self.closed = False

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)

    def close(self):
        self.closed = True


def test_worker_opcode_applies_environment_and_acknowledges_after_cache_check(monkeypatch):
    class FakeNativeEngine:
        def __init__(self, _config):
            self.cache_allocation = None
            self.cache_block_tables = None

        def graph_metrics(self):
            return {"rank": 2}

    connection = _ScriptedConnection(
        [
            (
                "configure_target_graph_correctness",
                "graph_replay_first",
                True,
                None,
            ),
            ("exit", None, None, None),
        ]
    )
    isolated_environment = {}
    monkeypatch.setattr(pearl_api, "NativePearlEngine", FakeNativeEngine)
    monkeypatch.setattr(pearl_api.os, "environ", isolated_environment)
    monkeypatch.setattr(pearl_api.dist, "is_initialized", lambda: False)
    config = SimpleNamespace(draft_tp_size=1, target_tp_size=3)

    pearl_api._pearl_worker(config, 2, 12345, connection)

    expected = _target_graph_correctness_environment(
        "graph_replay_first",
        validate_every_replay=True,
    )
    assert connection.sent == [
        ("ready", 2, {"rank": 2}),
        ("target_graph_correctness_configured", 2, expected),
    ]
    assert all(isolated_environment[key] == value for key, value in expected.items())
    assert connection.closed
