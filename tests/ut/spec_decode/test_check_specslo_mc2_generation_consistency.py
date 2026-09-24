# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the MC2 generation consistency diagnostic."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from examples import check_specslo_mc2_generation_consistency as diagnostic


def _digest(rows):
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _payload(rows, *, enable_mc2, batch_size=8):
    payload = {field: f"fixed-{field}" for field in diagnostic.RUN_CONTRACT_FIELDS}
    payload.update(
        {
            "enable_mc2": enable_mc2,
            "mc2_profile": "profile.json" if enable_mc2 else None,
            "results": [
                {
                    "batch_size": batch_size,
                    "num_prompts": len(rows),
                    "prompt_token_ids_sha256": "prompt-hash",
                    "request_output_token_limits": [len(row) for row in rows],
                    "output_token_ids": rows,
                    "output_token_ids_sha256": _digest(rows),
                }
            ],
        }
    )
    return payload


def test_equal_complete_outputs_pass_and_recompute_hashes():
    rows = [[1, 2, 3], [4, 5]]
    report = diagnostic.compare_payloads(
        _payload(rows, enable_mc2=False),
        _payload(copy.deepcopy(rows), enable_mc2=True),
    )

    assert report["passed"]
    assert report["comparable"]
    assert report["complete_output_hashes"] == {
        "mc2_off_recomputed_sha256": _digest(rows),
        "mc2_on_recomputed_sha256": _digest(rows),
        "equal": True,
        "mc2_off_stored_sha256": _digest(rows),
        "mc2_on_stored_sha256": _digest(rows),
        "mc2_off_stored_matches_recomputed": True,
        "mc2_on_stored_matches_recomputed": True,
    }
    assert report["output_comparison"]["first_difference"] is None
    assert report["target_at_first_difference"]["status"] == "not_applicable_outputs_are_equal"


def test_first_difference_is_earliest_completion_offset_not_request_order():
    off_rows = [[1, 2, 3], [4, 5, 6]]
    on_rows = [[1, 2, 9], [8, 5, 6]]
    report = diagnostic.compare_payloads(
        _payload(off_rows, enable_mc2=False),
        _payload(on_rows, enable_mc2=True),
    )

    assert not report["passed"]
    comparison = report["output_comparison"]
    assert comparison["requests_with_differences"] == 2
    assert comparison["positional_token_differences"] == 2
    assert comparison["first_difference"] == {
        "request_index": 1,
        "first_difference": 0,
        "mc2_off_token": 4,
        "mc2_on_token": 8,
        "mc2_off_length": 3,
        "mc2_on_length": 3,
    }
    assert comparison["first_difference_by_request_order"]["request_index"] == 0
    assert report["target_at_first_difference"]["coordinate"]["watch"] == "1:0"


def test_length_and_row_count_differences_are_not_dropped():
    comparison = diagnostic._compare_outputs(
        [[1, 2], [3]],
        [[1], [3], [4, 5]],
    )

    assert not comparison["equal"]
    assert comparison["requests_with_differences"] == 2
    assert comparison["positional_token_differences"] == 3
    assert comparison["request_mismatches"] == [
        {
            "request_index": 0,
            "first_difference": 1,
            "mc2_off_token": 2,
            "mc2_on_token": None,
            "mc2_off_length": 2,
            "mc2_on_length": 1,
        },
        {
            "request_index": 2,
            "first_difference": 0,
            "mc2_off_token": None,
            "mc2_on_token": 4,
            "mc2_off_length": 0,
            "mc2_on_length": 2,
        },
    ]


def test_target_traces_report_argmax_top2_margin_at_first_difference():
    off_rows = [[10, 11]]
    on_rows = [[10, 12]]
    off_trace = {
        "target_token_diagnostics": [
            {
                "request_index": 0,
                "output_offset": 1,
                "logits": [0.0, 3.0, 3.0, -1.0],
            }
        ]
    }
    on_trace = {
        "target_token_diagnostics": [
            {
                "request_index": 0,
                "output_offset": 1,
                "argmax_token_id": 2,
                "top1": {"token_id": 2, "logit": 4.25},
                "top2": [1, 4.0],
            }
        ]
    }

    report = diagnostic.compare_payloads(
        _payload(off_rows, enable_mc2=False),
        _payload(on_rows, enable_mc2=True),
        off_trace=off_trace,
        on_trace=on_trace,
    )
    target = report["target_at_first_difference"]
    assert target["status"] == "available_for_both_runs"
    # Exact ties use the smaller token ID, matching torch/CPU argmax ordering.
    assert target["mc2_off"]["target_argmax_token_id"] == 1
    assert target["mc2_off"]["top1"] == {"token_id": 1, "logit": 3.0}
    assert target["mc2_off"]["top2"] == {"token_id": 2, "logit": 3.0}
    assert target["mc2_off"]["top1_top2_margin"] == 0.0
    assert target["mc2_off"]["target_argmax_matches_generated_token"] is False
    assert target["mc2_on"]["target_argmax_token_id"] == 2
    assert target["mc2_on"]["top1_top2_margin"] == pytest.approx(0.25)
    assert target["argmax_equal"] is False


def test_missing_trace_never_implies_logit_evidence():
    report = diagnostic.compare_payloads(
        _payload([[1]], enable_mc2=False),
        _payload([[2]], enable_mc2=True),
    )
    target = report["target_at_first_difference"]
    assert target["status"] == "not_recorded_by_source_artifacts"
    assert not target["mc2_off"]["available"]
    assert not target["mc2_on"]["available"]
    assert target["argmax_equal"] is None


def test_contract_or_switch_mismatch_fails_comparability():
    off = _payload([[1]], enable_mc2=False)
    on = _payload([[1]], enable_mc2=True)
    on["seed"] = "changed"
    report = diagnostic.compare_payloads(off, on)
    assert not report["passed"]
    assert not report["comparable"]
    assert report["contract_differences"]["seed"] == {
        "mc2_off": "fixed-seed",
        "mc2_on": "changed",
    }

    on["seed"] = off["seed"]
    on["enable_mc2"] = False
    report = diagnostic.compare_payloads(off, on)
    assert not report["comparable"]
    assert not report["mc2_switch_contract"]["passed"]


def test_multiple_results_require_batch_selection():
    off = _payload([[1]], enable_mc2=False)
    on = _payload([[1]], enable_mc2=True)
    off["results"].append({**off["results"][0], "batch_size": 16})
    on["results"].append({**on["results"][0], "batch_size": 16})

    with pytest.raises(ValueError, match="multiple results"):
        diagnostic.compare_payloads(off, on)
    assert diagnostic.compare_payloads(off, on, batch_size=16)["passed"]


def test_malformed_or_duplicate_target_trace_is_rejected():
    off = _payload([[1]], enable_mc2=False)
    on = _payload([[2]], enable_mc2=True)
    duplicate = {
        "target_token_diagnostics": [
            {"request_index": 0, "output_offset": 0, "argmax_token_id": 1},
            {"request_index": 0, "output_offset": 0, "argmax_token_id": 1},
        ]
    }
    with pytest.raises(ValueError, match="duplicate rows"):
        diagnostic.compare_payloads(off, on, off_trace=duplicate)

    nonfinite = {
        "target_token_diagnostics": [
            {"request_index": 0, "output_offset": 0, "logits": [0.0, float("nan")]},
        ]
    }
    with pytest.raises(ValueError, match="finite"):
        diagnostic.compare_payloads(off, on, off_trace=nonfinite)
