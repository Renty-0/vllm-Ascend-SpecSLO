# SPDX-License-Identifier: Apache-2.0
"""CPU timing fixtures only; none of these numbers are benchmark results."""

import json
from types import SimpleNamespace

import pytest

from examples.benchmark_nano_pearl_speculative import (
    _host_continuation_measurement_issues,
    _summarize_slo_metrics,
    _worker_aclgraph_deltas,
)
from examples.benchmark_nano_pearl_target_only import _online_slo_summary
from examples.specslo_slo_metrics import summarize_slo_rows


def _baseline(first=10.0, last=10.31, count=8, metrics=True, bound=40):
    output = SimpleNamespace(
        request_id="r",
        outputs=[SimpleNamespace(token_ids=[7] * count)],
        metrics=SimpleNamespace(first_token_ts=first, last_token_ts=last, arrival_time=100.0) if metrics else None,
    )
    return _online_slo_summary([output], [{"request_id": "r", "slo_class": "tight", "slo_tpot_ms": bound}], -91, 2.0)


def test_baseline_retains_two_denominators_and_raw_request_evidence():
    result = _baseline()
    assert result["attainment"] == 0
    assert result["paper"]["attainment"] == 1
    assert result["paper"]["goodput_tokens_per_e2e_second"] == 4
    row = result["request_metrics"][0]
    assert row["observed_tpot_ms"] == pytest.approx(310 / 7)
    assert row["paper_tpot_ms"] == pytest.approx(310 / 8)
    assert row["request_e2e_ms"] == pytest.approx(1310)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "args",
    [
        {"metrics": False},
        {"first": 0},
        {"first": None},
        {"last": None},
        {"last": 0},
        {"last": 9},
        {"first": float("nan")},
        {"last": float("inf")},
        {"count": 0},
    ],
)
def test_missing_reversed_or_nonfinite_timestamps_never_become_zero_latency_pass(args):
    result = _baseline(**args)
    for report in (result, result["paper"]):
        assert report["attained_requests"] == 0 and report["goodput_tokens"] == 0
        assert report["missing_timing_requests"] == 1
    json.dumps(result, allow_nan=False)


def test_unconstrained_request_is_not_in_slo_denominator():
    result = _baseline(bound=None)
    assert result["constrained_requests"] == 0 and result["attainment"] is None


@pytest.mark.parametrize("bound", [0, -1, float("nan"), float("inf")])
def test_invalid_slo_constraints_fail_instead_of_inf_comparison(bound):
    with pytest.raises(ValueError, match="finite positive"):
        _baseline(bound=bound)


def test_native_paper_and_baseline_use_identical_denominator_and_goodput_window():
    row = {
        "request_id": "r",
        "completion_token_ids": [7] * 8,
        "slo_tpot_ms": 40,
        "slo_class": "tight",
        "observed_tpot_ms": 310 / 7,
        "paper_tpot_ms": 310 / 8,
        "slo_attained": False,
        "slo_goodput_tokens": 0,
    }
    native = _summarize_slo_metrics([row], 1.5, 2.0)
    baseline = _baseline()
    for key in (
        "constrained_requests",
        "attained_requests",
        "attainment",
        "goodput_tokens",
        "goodput_tokens_per_e2e_second",
    ):
        assert native["paper"][key] == baseline["paper"][key]
    assert native["paper"]["by_class"]["tight"]["mean_tpot_ms"] == pytest.approx(
        baseline["paper"]["by_class"]["tight"]["mean_tpot_ms"]
    )
    del row["paper_tpot_ms"]
    legacy = _summarize_slo_metrics([row], 1.5, 2.0)
    assert legacy["paper"]["missing_timing_requests"] == 1
    assert legacy["paper"]["attained_requests"] == 0


def test_native_paper_reports_explicit_decode_stage_goodput():
    row = {
        "request_id": "r",
        "completion_token_ids": [7] * 8,
        "slo_tpot_ms": 40,
        "slo_class": "tight",
        "observed_tpot_ms": 20,
        "paper_tpot_ms": 20,
        "slo_attained": True,
        "slo_goodput_tokens": 8,
    }
    result = _summarize_slo_metrics(
        [row],
        1.5,
        2.0,
        paper_decode_elapsed=1.0,
    )
    assert result["paper"]["goodput_tokens_per_e2e_second"] == 4
    assert result["paper"]["decode_stage_measurement_seconds"] == 1
    assert result["paper"]["goodput_tokens_per_decode_second"] == 8


def test_worker_graph_deltas_report_measured_staged_correction_path():
    before = [
        {
            "rank": 1,
            "spec_rhythm_staged_gloo_correction_submit_calls": 10,
            "spec_rhythm_staged_gloo_correction_fallback_calls": 2,
            "spec_rhythm_staged_gloo_correction_host_continuation_reuses": 7,
            "spec_rhythm_staged_gloo_correction_device_continuation_copies": 3,
            "worker_spec_rhythm_gloo_proposal_host_continuation_reuse_requested": 0,
            "worker_spec_rhythm_gloo_proposal_host_continuation_reuse_active": 0,
        }
    ]
    after = [
        {
            "rank": 1,
            "spec_rhythm_staged_gloo_correction_submit_calls": 16,
            "spec_rhythm_staged_gloo_correction_fallback_calls": 4,
            "spec_rhythm_staged_gloo_correction_host_continuation_reuses": 12,
            "spec_rhythm_staged_gloo_correction_device_continuation_copies": 4,
            "worker_spec_rhythm_gloo_proposal_host_continuation_reuse_requested": 1,
            "worker_spec_rhythm_gloo_proposal_host_continuation_reuse_active": 1,
        }
    ]

    delta = _worker_aclgraph_deltas(before, after)[0]

    assert delta["spec_rhythm_staged_gloo_correction_submit_calls_delta"] == 6
    assert delta["spec_rhythm_staged_gloo_correction_fallback_calls_delta"] == 2
    assert delta["spec_rhythm_staged_gloo_correction_host_continuation_reuses_delta"] == 5
    assert delta["spec_rhythm_staged_gloo_correction_device_continuation_copies_delta"] == 1
    assert delta["worker_spec_rhythm_gloo_proposal_host_continuation_reuse_requested"] == 1
    assert delta["worker_spec_rhythm_gloo_proposal_host_continuation_reuse_active"] == 1
    assert "worker_spec_rhythm_gloo_proposal_host_continuation_reuse_active_delta" not in delta


def test_worker_graph_deltas_preserve_measured_draft_continuation_path():
    before = [
        {
            "rank": 0,
            "is_draft_rank": 1,
            "worker_spec_rhythm_gloo_proposal_host_continuation_reuses": 99,
            "worker_spec_rhythm_gloo_proposal_device_continuation_reads": 88,
        }
    ]
    after = [
        {
            "rank": 0,
            "is_draft_rank": 1,
            "worker_spec_rhythm_gloo_proposal_host_continuation_reuses": 17,
            "worker_spec_rhythm_gloo_proposal_device_continuation_reads": 0,
        }
    ]

    delta = _worker_aclgraph_deltas(before, after)[0]

    assert delta["worker_spec_rhythm_gloo_proposal_host_continuation_reuses"] == 17
    assert delta["worker_spec_rhythm_gloo_proposal_device_continuation_reads"] == 0
    assert "worker_spec_rhythm_gloo_proposal_host_continuation_reuses_delta" not in delta


def test_worker_graph_deltas_include_measured_target_fia_parallel_updates():
    before = [
        {
            "rank": 1,
            "is_draft_rank": 0,
            "aclgraph_target_fia_task_update_workers": 2,
            "aclgraph_target_fia_parallel_update_replays": 5,
            "aclgraph_target_fia_parallel_update_tasks": 320,
        }
    ]
    after = [
        {
            **before[0],
            "aclgraph_target_fia_parallel_update_replays": 12,
            "aclgraph_target_fia_parallel_update_tasks": 768,
        }
    ]

    delta = _worker_aclgraph_deltas(before, after)[0]

    assert delta["aclgraph_target_fia_task_update_workers"] == 2
    assert delta["aclgraph_target_fia_parallel_update_replays_delta"] == 7
    assert delta["aclgraph_target_fia_parallel_update_tasks_delta"] == 448


def test_worker_graph_deltas_include_measured_draft_pa_parallel_updates():
    before = [
        {
            "rank": 0,
            "is_draft_rank": 1,
            "aclgraph_draft_pa_task_update_workers": 2,
            "aclgraph_draft_pa_parallel_update_replays": 5,
            "aclgraph_draft_pa_parallel_update_tasks": 1280,
            "aclgraph_draft_pa_parallel_update_fallback_replays": 1,
            "aclgraph_draft_pa_parallel_update_rejections": {
                "invalid_step_lengths": 1,
            },
        }
    ]
    after = [
        {
            **before[0],
            "aclgraph_draft_pa_parallel_update_replays": 12,
            "aclgraph_draft_pa_parallel_update_tasks": 3072,
            "aclgraph_draft_pa_parallel_update_fallback_replays": 3,
            "aclgraph_draft_pa_parallel_update_rejections": {
                "invalid_step_lengths": 1,
                "non_linear_entry": 2,
            },
        }
    ]

    delta = _worker_aclgraph_deltas(before, after)[0]

    assert delta["aclgraph_draft_pa_task_update_workers"] == 2
    assert delta["aclgraph_draft_pa_parallel_update_replays_delta"] == 7
    assert delta["aclgraph_draft_pa_parallel_update_tasks_delta"] == 1792
    assert delta["aclgraph_draft_pa_parallel_update_fallback_replays_delta"] == 2
    assert delta["aclgraph_draft_pa_parallel_update_rejections_delta"] == {
        "non_linear_entry": 2,
    }


def _continuation_evidence(*, reuse: bool):
    selected_draft = {
        "worker_spec_rhythm_gloo_proposal_host_continuation_reuses": 7 if reuse else 0,
        "worker_spec_rhythm_gloo_proposal_device_continuation_reads": 0 if reuse else 7,
    }
    workers = [
        {
            "rank": 0,
            "is_draft_rank": 1,
            "worker_spec_rhythm_gloo_proposal_host_continuation_reuse_requested": int(reuse),
            "worker_spec_rhythm_gloo_proposal_host_continuation_reuse_active": int(reuse),
            **selected_draft,
        }
    ]
    for rank in (1, 2, 3):
        workers.append(
            {
                "rank": rank,
                "is_draft_rank": 0,
                "worker_spec_rhythm_gloo_proposal_host_continuation_reuse_requested": int(reuse),
                "worker_spec_rhythm_gloo_proposal_host_continuation_reuse_active": int(reuse),
                "worker_spec_rhythm_gloo_proposal_host_continuation_reuses": 0,
                "worker_spec_rhythm_gloo_proposal_device_continuation_reads": 0,
                "spec_rhythm_staged_gloo_correction_submit_calls_delta": 5,
                "spec_rhythm_staged_gloo_correction_fallback_calls_delta": 0,
                "spec_rhythm_staged_gloo_correction_host_continuation_reuses_delta": (
                    5 if reuse else 0
                ),
                "spec_rhythm_staged_gloo_correction_device_continuation_copies_delta": (
                    0 if reuse else 5
                ),
            }
        )
    return workers


@pytest.mark.parametrize("reuse", [False, True])
def test_host_continuation_measurement_accepts_only_selected_path(reuse):
    assert not _host_continuation_measurement_issues(
        _continuation_evidence(reuse=reuse),
        expected_reuse=reuse,
    )


def test_host_continuation_measurement_rejects_inactive_and_mixed_paths():
    workers = _continuation_evidence(reuse=True)
    workers[0]["worker_spec_rhythm_gloo_proposal_device_continuation_reads"] = 1
    workers[1]["worker_spec_rhythm_gloo_proposal_host_continuation_reuse_active"] = 0
    workers[2]["spec_rhythm_staged_gloo_correction_host_continuation_reuses_delta"] = 4
    workers[3]["spec_rhythm_staged_gloo_correction_device_continuation_copies_delta"] = 1
    workers[3]["spec_rhythm_staged_gloo_correction_fallback_calls_delta"] = 1

    issues = _host_continuation_measurement_issues(
        workers,
        expected_reuse=True,
    )

    assert any("rank 1 host-continuation active=0" in issue for issue in issues)
    assert any("draft rank 0 measured 1 unexpected device read" in issue for issue in issues)
    assert any("target rank 2 measured 4 host reuse calls for 5" in issue for issue in issues)
    assert any("target rank 3 measured 1 unexpected device copy" in issue for issue in issues)
    assert any("target rank 3 measured 1 unstaged correction fallbacks" in issue for issue in issues)


def test_partial_missing_group_timing_keeps_missing_requests_in_attainment_denominator():
    result = summarize_slo_rows(
        [
            {"output_tokens": 10, "slo_tpot_ms": 40, "paper_tpot_ms": 30},
            {"output_tokens": 10, "slo_tpot_ms": 40, "paper_tpot_ms": None},
        ],
        tpot_field="paper_tpot_ms",
        definition="CPU fixture",
        elapsed_seconds=4,
    )
    assert result["attainment"] == 0.5
    assert result["goodput_tokens_per_e2e_second"] == 2.5
    assert result["by_class"]["unspecified"]["missing_timing_requests"] == 1
