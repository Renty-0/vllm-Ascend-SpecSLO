# SPDX-License-Identifier: Apache-2.0
"""CPU timing fixtures only; none of these numbers are benchmark results."""

import json
from types import SimpleNamespace

import pytest

from examples.benchmark_nano_pearl_speculative import _summarize_slo_metrics
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
