# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the MC2 eager/graph quadrant gate."""

from __future__ import annotations

import copy
import hashlib
import json

from examples import check_specslo_mc2_graph_quadrants as diagnostic


def _digest(rows):
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _worker(rank, *, graph, mc2_on, after):
    graph_replays = (7 if after else 5) if graph else 0
    generic_replays = (7 if after else 5) if graph else 0
    if mc2_on and graph:
        mc2 = (384, 384, 0, 0)
    elif mc2_on:
        mc2 = (20, 20, 0, 0) if after else (10, 10, 0, 0)
    else:
        mc2 = (0, 0, 0, 0)
    worker = {
        "rank": rank,
        "is_draft_rank": 0,
        "aclgraph_sealed": int(graph),
        "aclgraph_entries": int(graph),
        "aclgraph_captures": int(graph),
        "aclgraph_capture_attempts": int(graph),
        "aclgraph_replays": graph_replays,
        "aclgraph_generic_replay_calls": generic_replays,
        **{field: 0 for field in diagnostic.GRAPH_ZERO_FIELDS},
    }
    worker.update(dict(zip(diagnostic.MC2_COUNTERS, mc2)))
    chained = (384, 384, 3) if mc2_on and graph else (0, 0, 0)
    worker.update(dict(zip(diagnostic.MC2_CHAIN_COUNTERS, chained)))
    return worker


def _payload(*, eager, mc2_on):
    rows = [[1, 2], [3, 4]]
    graph = not eager
    payload = {
        "backend": "nano-pearl-native-target-only",
        "draft_model": "draft",
        "target_model": "target",
        "draft_tensor_parallel_size": 1,
        "target_tensor_parallel_size": 3,
        "prefill_chunk_size": None,
        "enable_prefix_caching": False,
        "max_tokens": 2,
        "warmup_prompts": 2,
        "first_prompt_token_ids": [7],
        "target_rms_norm_epsilon": 1e-6,
        "enforce_eager": eager,
        "execution_mode": "eager" if eager else "aclgraph",
        "enable_mc2": mc2_on,
        "mc2_profile": "profile.json" if mc2_on else None,
        "mc2_profile_resolved": "/tmp/profile.json" if mc2_on else None,
        "mc2_profile_sha256": "a" * 64 if mc2_on else None,
        "mc2_profile_rms_norm_epsilon": 1e-6 if mc2_on else None,
        "seal_graph_cache_after_warmup": graph,
        "results": [
            {
                "batch_size": 2,
                "num_prompts": 2,
                "num_static_chunks": 1,
                "output_tokens": 4,
                "output_token_ids": rows,
                "output_token_ids_sha256": _digest(rows),
                "worker_metrics_before_measurement": [
                    _worker(rank, graph=graph, mc2_on=mc2_on, after=False)
                    for rank in (1, 2, 3)
                ],
                "worker_metrics_after_measurement": [
                    _worker(rank, graph=graph, mc2_on=mc2_on, after=True)
                    for rank in (1, 2, 3)
                ],
            }
        ],
    }
    return payload


def _quadrants():
    return {
        name: _payload(eager=eager, mc2_on=mc2_on)
        for name, (eager, mc2_on) in diagnostic.QUADRANTS.items()
    }


def test_complete_four_quadrant_evidence_passes():
    report = diagnostic.validate_quadrants(
        _quadrants(),
        expected_graph_on_resident_dispatches_per_rank=384,
        expected_chain_operations_per_flush=128,
    )

    assert report["passed"]
    assert report["status"] == "pass"
    assert report["expected_graph_on_resident_dispatches_per_rank"] == 384
    assert report["expected_chain_operations_per_flush"] == 128
    assert len(set(report["complete_output_sha256"].values())) == 1
    assert report["route_evidence"]["graph_on"]["target_workers"][0]["mc2_delta"] == {
        field: 0 for field in diagnostic.MC2_COUNTERS
    }


def test_graph_on_rejects_host_redispatch_during_replay():
    payloads = _quadrants()
    payloads["graph_on"]["results"][0]["worker_metrics_after_measurement"][1][
        "mc2_dispatch_fused_attempt"
    ] += 1

    report = diagnostic.validate_quadrants(payloads)

    assert not report["passed"]
    assert any("re-entered Python MC2 dispatch" in error for error in report["errors"])
    assert any("divergent MC2 dispatch state" in error for error in report["errors"])


def test_graph_on_rejects_incomplete_resident_mc2_coverage():
    report = diagnostic.validate_quadrants(
        _quadrants(),
        expected_graph_on_resident_dispatches_per_rank=383,
    )

    assert not report["passed"]
    assert any("resident graph MC2 attempt/success" in error for error in report["errors"])


def test_graph_on_rejects_per_layer_flush_when_whole_model_chain_is_required():
    payloads = _quadrants()
    for phase in ("worker_metrics_before_measurement", "worker_metrics_after_measurement"):
        for worker in payloads["graph_on"]["results"][0][phase]:
            worker["mc2_dispatch_native_epilogue_chain_flush"] = 192

    report = diagnostic.validate_quadrants(
        payloads,
        expected_chain_operations_per_flush=128,
    )

    assert not report["passed"]
    assert any("resident chain attempt/success/flush" in error for error in report["errors"])


def test_output_and_epsilon_mismatches_fail():
    payloads = _quadrants()
    payloads["eager_on"]["mc2_profile_rms_norm_epsilon"] = 1e-5
    payloads["graph_on"]["results"][0]["output_token_ids"][0][0] = 99
    payloads["graph_on"]["results"][0]["output_token_ids_sha256"] = _digest(
        payloads["graph_on"]["results"][0]["output_token_ids"]
    )

    report = diagnostic.validate_quadrants(payloads)

    assert not report["passed"]
    assert any("epsilon mismatch" in error for error in report["errors"])
    assert any("output tokens differ" in error for error in report["errors"])


def test_contract_mismatch_and_unsealed_graph_fail():
    payloads = _quadrants()
    payloads["graph_off"]["max_tokens"] = 3
    for worker in payloads["graph_off"]["results"][0]["worker_metrics_before_measurement"]:
        worker["aclgraph_sealed"] = 0

    report = diagnostic.validate_quadrants(copy.deepcopy(payloads))

    assert not report["passed"]
    assert any("run contract differs at max_tokens" in error for error in report["errors"])
    assert any("graph cache is not sealed" in error for error in report["errors"])
