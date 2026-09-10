# SPDX-License-Identifier: Apache-2.0
"""Synthetic CPU schema/observer tests, not measured NPU roofline evidence."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from examples import check_specslo_profile_online as probe
from tests.ut.spec_decode.test_spec_rhythm_tree_loop import _TreeLoopHarness
from vllm_ascend.spec_decode.pearl.roofline import (
    STRICT_ROOFLINE_SCHEMA_VERSION,
    ProfiledRoofline,
    normalize_roofline,
    runtime_source_fingerprints,
)
from vllm_ascend.spec_decode.pearl.spec_rhythm import SpecRhythmBudgetShaper


def _document():
    # Explicitly fabricated only for CPU contract testing; never used by an
    # NPU acceptance invocation or reported as capacity/performance evidence.
    evidence = []
    for batch, roof in ((1, 1), (2, 4)):
        for context in (128, 256, 512):
            evidence.append(
                {
                    "lookup_key": f"{batch}:1",
                    "batch_size": batch,
                    "context_len": context,
                    "ar_requests": batch,
                    "verification_requests": 1,
                    "measured_roof": roof,
                    "candidate_sweep": [
                        {
                            "candidate_tokens": count,
                            "within_limit": True,
                            "shapes": [
                                {
                                    "candidate_counts": [count],
                                    "attention_backend": "fused_infer_attention_tree_v1",
                                    "latency_ms": 10.0,
                                    "samples": 3,
                                }
                            ],
                        }
                        for count in (1, 2, 4)
                    ],
                }
            )
    return {
        "schema_version": STRICT_ROOFLINE_SCHEMA_VERSION,
        "context_bucket_size": 512,
        "metadata": {
            "model": "/models/target",
            "target_tensor_parallel_size": 3,
            "execution_mode": "graph",
            "hardware": "CPU-test-fixture-not-real-hardware",
            "measurement_scope": "target_forward",
            "verification_layout": "packed_tree",
            "verification_attention_backends": ["fused_infer_attention_tree_v1"],
            "ar_attention_backend": "paged_attention_v1",
            "ar_comparator": "standard_decode_full_active_batch",
            **runtime_source_fingerprints(),
            "tree_fia_sparse_mode": 1,
            "tree_fia_inner_precise": 2,
            "max_model_len": 1024,
            "tree_width": 2,
            "tree_depth": 2,
        },
        "roofline": {"1:1": 1, "2:1": 4},
        "evidence": evidence,
    }


def _profile(document=None):
    return normalize_roofline(
        _document() if document is None else document,
        model="/models/target",
        target_tp_size=3,
        enforce_eager=False,
        max_model_len=1024,
        tree_width=2,
        tree_depth=2,
    )


def _args(tmp_path, document=None, *extra):
    path = tmp_path / "synthetic-cpu-only.json"
    path.write_text(json.dumps(_document() if document is None else document))
    return probe._build_parser().parse_args(
        [
            "--profile",
            str(path),
            "--target-model",
            "/models/target",
            "--output",
            str(tmp_path / "output.json"),
            *extra,
        ]
    )


def _execution(batch=2, context=70):
    return {
        "batch_size": batch,
        "context_len": context,
        "lookup_key": f"{batch}:1",
        "candidate_roof": 4 if batch == 2 else 1,
        "execution_index": 0,
        "verification_requests": 1,
    }


def _output(count=4, graph=True):
    return {"query_count": count + 1, "attention_backend": "fused_infer_attention_tree_v1", "used_aclgraph": graph}


def test_dry_run_preserves_multiple_real_lookup_keys_without_claiming_execution(tmp_path):
    args = _args(tmp_path)
    assert (
        probe.main(
            [
                "--profile",
                str(args.profile),
                "--target-model",
                args.target_model,
                "--output",
                str(args.output),
                "--dry-run",
            ]
        )
        == 0
    )
    report = json.loads(args.output.read_text())
    assert report["status"] == "dry_run" and report["passed"] is None
    assert not report["npu_executed"] and not report["worker_records"]
    assert report["capacity"] == 2 and report["prompt_count"] == 4
    assert report["measured_roofline"] == {"1:1": 1, "2:1": 4}
    assert report["required_coverage"] == [
        {"lookup_key": "1:1", "candidate_roof": 1, "verification_requests": 1},
        {"lookup_key": "2:1", "candidate_roof": 4, "verification_requests": 1},
    ]


def test_missing_tail_profile_fails_without_filling_the_table(tmp_path):
    document = _document()
    document["roofline"].pop("1:1")
    document["unqualified_lookup_keys"] = ["1:1"]
    original = copy.deepcopy(document)
    args = _args(tmp_path, document)
    profile, _ = probe._load_profile(args)
    with pytest.raises(ValueError, match="Unprofiled.*1:1"):
        probe._required_coverage(profile, args.prompt_lengths, args.max_tokens)
    assert json.loads(args.profile.read_text()) == original
    assert dict(profile) == {"2:1": 4}


def test_missing_later_context_bucket_is_not_interpolated():
    with pytest.raises(ValueError, match="Unprofiled.*1:2"):
        probe._required_coverage(_profile(), [500, 501, 502, 503], [20] * 4)


def test_legacy_budget_map_cannot_enter_measured_acceptance(tmp_path):
    args = _args(tmp_path, {"1:1": 1, "2:1": 4}, "--mode", "graph")
    with pytest.raises(ValueError, match="full measured profile"):
        probe._load_profile(args)


def test_wrong_profile_mode_is_rejected(tmp_path):
    args = _args(tmp_path, None, "--mode", "eager")
    with pytest.raises(ValueError, match="execution mode"):
        probe._load_profile(args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("arrival_offsets", [0.0] * 4),
        ("arrival_offsets", [0.0, 1.0, 0.5, 2.0]),
        ("arrival_offsets", [0.0, 0.0, float("nan"), 1.0]),
        ("max_tokens", [1, 2, 3, 4]),
        ("max_model_len", 100),
    ],
)
def test_invalid_online_cases_rejected_before_runtime(tmp_path, field, value):
    args = _args(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        probe._validate_args(args)


def test_observer_retains_strict_mapping_and_original_lookup_behavior():
    original = _profile()
    observed = probe._ObservedProfile(original)
    shaper = SpecRhythmBudgetShaper(min_gamma=1, max_gamma=4, roofline=observed)
    assert shaper.roofline is observed and isinstance(observed, ProfiledRoofline)
    assert shaper.verification_roof(2, 65) == 4
    observed.validate_execution(1, 98, 1)
    assert observed.executions == [
        {
            "batch_size": 1,
            "context_len": 98,
            "lookup_key": "1:1",
            "candidate_roof": 1,
            "execution_index": 0,
            "verification_requests": 1,
        }
    ]
    before = list(observed.lookups)
    with pytest.raises(ValueError, match="Unprofiled"):
        shaper.verification_roof(3, 98)
    assert observed.lookups == before and dict(observed) == dict(original)


def test_actual_forward_links_budget_shape_backend_graph_and_sampled_contexts():
    result = probe._forward_evidence(_profile(), _execution(), [4], _output())
    assert result["passed"] and result["actual_candidates"] == 4
    assert result["physical_query_tokens"] == 5
    assert not result["actual_context_was_sampled"]
    assert [row["sampled_context_len"] for row in result["profile_evidence"]] == [128, 256, 512]
    assert all(row["matching_candidate_shapes"][0]["samples"] == 3 for row in result["profile_evidence"])
    assert probe._forward_evidence(_profile(), _execution(context=128), [4], _output())["actual_context_was_sampled"]


@pytest.mark.parametrize(
    "fault", ["budget", "fake_roof", "shape", "backend", "fallback", "queries", "missing_validation"]
)
def test_invalid_actual_consumption_cannot_be_reported_as_profiled(fault):
    execution, counts, output = _execution(), [4], _output()
    if fault == "budget":
        execution = _execution(batch=1)
    elif fault == "fake_roof":
        execution["candidate_roof"] = 100
    elif fault == "shape":
        counts, output = [3], _output(3)
    elif fault == "backend":
        output["attention_backend"] = "dense_sdpa_tree_v1"
    elif fault == "fallback":
        output["used_aclgraph"] = False
    elif fault == "queries":
        output["query_count"] = 8
    else:
        execution = None
    assert not probe._forward_evidence(_profile(), execution, counts, output)["passed"]


def test_instance_observers_do_not_change_forward_results_and_can_be_removed():
    profile = probe._ObservedProfile(_profile())
    profile.validate_execution(2, 70, 1)
    expected = _output()
    target = lambda *args, **kwargs: expected
    prefill = lambda *args, **kwargs: [7]
    engine = SimpleNamespace(is_draft=False, target_tree_forward=target, _prefill_and_sample_target_batch=prefill)
    forwards, prefills = [], []
    restore = probe._install_observers(engine, profile, forwards, prefills)
    assert engine.target_tree_forward([SimpleNamespace(candidate_budget=4)], [1], [[2, 3, 4, 5]]) is expected
    assert engine._prefill_and_sample_target_batch([[1]], [None], [3]) == [7]
    assert forwards[0]["passed"] and prefills[0]["request_indices"] == [3]
    restore()
    assert engine.target_tree_forward is target and engine._prefill_and_sample_target_batch is prefill


def _records():
    profile = _profile()
    ids = [f"request-{index}" for index in range(4)]
    arrivals = dict(zip(ids, [10.0, 10.0, 10.05, 10.1]))
    outputs = [[7, 8], [8, 9, 10], [9, 10], [10, 11, 12]]
    forwards = [
        probe._forward_evidence(profile, _execution(), [4], _output()),
        probe._forward_evidence(profile, _execution(1, 99), [1], _output(1)),
    ]
    events = [
        {
            "request_index": index,
            "request_id": ids[index],
            "token_ids": row,
            "finished": True,
            "host_received_wall_time": 11.0,
            "elapsed_seconds": 1.0,
        }
        for index, row in enumerate(outputs)
    ]
    records = [
        {
            "rank": rank,
            "error": None,
            "lookups": [{"batch_size": 2}, {"batch_size": 1}],
            "validated_executions": [{}, {}],
            "target_forwards": [] if rank == 0 else copy.deepcopy(forwards),
            "prefills": [
                {"request_indices": [0, 1], "started_wall_time": 10.0},
                {"request_indices": [2, 3], "started_wall_time": 10.2},
            ],
            "outputs": outputs if rank == 1 else None,
            "stream_events": events if rank == 1 else [],
        }
        for rank in range(4)
    ]
    return profile, records, list(map(len, outputs)), ids, arrivals


def test_full_online_acceptance_checks_stream_refill_tail_and_target_rank_agreement():
    result = probe._summarize(*_records())
    assert result["passed"]
    assert result["consumed_lookup_keys"] == ["1:1", "2:1"]


@pytest.mark.parametrize("fault", ["tail", "rank", "stream", "arrival", "refill", "unmatched_forward", "error"])
def test_online_acceptance_rejects_missing_coverage_or_stream_regressions(fault):
    profile, records, limits, ids, arrivals = _records()
    if fault == "tail":
        records[2]["target_forwards"][1]["batch_size"] = 2
    elif fault == "rank":
        records[3]["target_forwards"][1]["context_len"] += 1
    elif fault == "stream":
        records[1]["stream_events"][0]["token_ids"] = [17]
    elif fault == "arrival":
        records[0]["prefills"][1]["started_wall_time"] = 9.0
    elif fault == "refill":
        records[0]["prefills"].pop()
    elif fault == "unmatched_forward":
        records[2]["validated_executions"].append({})
    else:
        records[0]["error"] = "real runtime exception"
    assert not probe._summarize(profile, records, limits, ids, arrivals)["passed"]


def test_real_cpu_service_loop_observes_profile_shrink_at_tail_without_output_loss(monkeypatch):
    harness = _TreeLoopHarness(monkeypatch, requests=4, capacity=2, max_tokens=8, online_prefill=True)
    document = _document()
    document["metadata"].update(model="cpu-target", execution_mode="eager", max_model_len=256)
    observed = probe._ObservedProfile(
        normalize_roofline(
            document,
            model="cpu-target",
            target_tp_size=3,
            enforce_eager=True,
            max_model_len=256,
            tree_width=2,
            tree_depth=2,
        )
    )
    harness.engine.config = replace(
        harness.engine.config, spec_rhythm_roofline=observed, spec_rhythm_verification_budget=None
    )
    forwards, prefills = [], []
    restore = probe._install_observers(harness.engine, observed, forwards, prefills)
    try:
        results = harness.run()
    finally:
        restore()
    assert [len(row["completion_token_ids"]) for row in results] == [8] * 4
    assert {row["lookup_key"] for row in forwards} == {"1:1", "2:1"}
    assert all(row["passed"] for row in forwards)
    assert all(row["actual_candidates"] <= row["candidate_roof"] for row in forwards)
    assert all(row["actual_candidates"] <= 1 for row in forwards if row["lookup_key"] == "1:1")
    assert len(observed.executions) == len(forwards)
    assert sorted(index for event in prefills for index in event["request_indices"]) == [0, 1, 2, 3]


@pytest.mark.parametrize("fail", [False, True])
def test_fixed_reference_is_separate_and_restores_real_configuration(fail):
    from vllm_ascend.spec_decode.pearl.native_engine import NativePearlConfig, NativeSamplingParams

    config = NativePearlConfig(
        "draft",
        "target",
        1,
        3,
        4,
        1024,
        8,
        enable_continuous_batching=True,
        enable_preemptive_scheduling=True,
        enable_spec_rhythm=True,
        spec_rhythm_roofline={"1:1": 1, "2:1": 4},
        spec_rhythm_tree_width=2,
        spec_rhythm_tree_depth=2,
    )
    original_params = [NativeSamplingParams(max_tokens=8, arrival_ts=1234.5, slo_tpot_ms=40)]
    engine = SimpleNamespace(config=config)
    prompts = [[1, 2]]

    def generate(actual_prompts, actual_params):
        assert actual_prompts is prompts
        assert engine.config.spec_rhythm_roofline is None
        assert engine.config.spec_rhythm_verification_budget == 1
        assert engine.config.enforce_eager == config.enforce_eager
        assert actual_params[0].arrival_ts is None
        assert actual_params[0].slo_tpot_ms == 40
        if fail:
            raise RuntimeError("reference forward failed")
        return [{"completion_token_ids": [7, 8]}]

    engine.generate_batch = generate
    result = probe._fixed_budget_reference(engine, prompts, original_params)
    assert engine.config is config
    assert original_params[0].arrival_ts == 1234.5
    assert (result["error"] is not None) is fail
    assert result["outputs"] == (None if fail else [[7, 8]])
