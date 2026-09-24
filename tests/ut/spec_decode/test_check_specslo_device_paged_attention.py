# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage for the device-position PagedAttention qualifier."""

import argparse
import json
import math

import pytest
import torch

from examples import check_specslo_device_paged_attention as check


def _args(**overrides):
    values = {
        "device": "npu:0",
        "rows": 4,
        "qheads": 8,
        "kvheads": 4,
        "head_dim": 8,
        "page": 16,
        "tile": 16,
        "seed": 7,
        "atol": 0.02,
        "rtol": 0.02,
        "output": "unused.json",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_defaults_cover_gqa2_cross_page_and_changed_positions():
    args = _args()
    check._validate_args(args)
    inputs = check._build_inputs(args)

    assert args.qheads // args.kvheads == 2
    assert bool((inputs["positions"] >= args.page).all())
    assert bool((inputs["changed_positions"] > inputs["positions"]).all())
    assert int(inputs["changed_positions"].max()) < inputs["block_table"].shape[1] * args.page
    assert inputs["query"].dtype == inputs["key_cache"].dtype == torch.bfloat16
    assert inputs["positions"].dtype == torch.int64
    assert inputs["block_table"].dtype == torch.int32
    assert not torch.equal(inputs["query"], inputs["changed_query"])
    scale = 1 / math.sqrt(args.head_dim)
    initial_reference = check._cpu_reference(
        inputs["query"],
        inputs["key_cache"],
        inputs["value_cache"],
        inputs["block_table"],
        inputs["positions"],
        scale=scale,
    )
    changed_position_reference = check._cpu_reference(
        inputs["query"],
        inputs["key_cache"],
        inputs["value_cache"],
        inputs["block_table"],
        inputs["changed_positions"],
        scale=scale,
    )
    assert not torch.equal(initial_reference, changed_position_reference)


def test_cpu_reference_obeys_page_table_and_gqa_head_mapping():
    query = torch.zeros(1, 4, 2, dtype=torch.bfloat16)
    keys = torch.zeros(2, 2, 2, 2, dtype=torch.bfloat16)
    values = torch.tensor(
        [
            [[[10, 20], [100, 200]], [[30, 40], [300, 400]]],
            [[[1, 2], [11, 12]], [[3, 4], [13, 14]]],
        ],
        dtype=torch.bfloat16,
    )
    # Three visible tokens: page 1 contributes two, then page 0 contributes one.
    output = check._cpu_reference(
        query,
        keys,
        values,
        torch.tensor([[1, 0]], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int64),
        scale=1 / math.sqrt(2),
    )
    expected = torch.tensor(
        [[[14 / 3, 26 / 3], [14 / 3, 26 / 3], [124 / 3, 226 / 3], [124 / 3, 226 / 3]]],
        dtype=torch.float64,
    )
    assert output.dtype == torch.float64
    assert torch.allclose(output, expected, atol=1e-12, rtol=1e-12)


def test_comparison_fails_closed_on_nonfinite_and_tolerance_miss():
    nonfinite = check._comparison(torch.tensor([float("nan")]), torch.ones(1), atol=0.1, rtol=0.1)
    assert nonfinite == {
        "finite": False,
        "within_tolerance": False,
        "nonfinite_actual": 1,
        "nonfinite_reference": 0,
    }

    mismatch = check._comparison(torch.tensor([2.0]), torch.tensor([1.0]), atol=0.01, rtol=0.01)
    assert mismatch["finite"]
    assert not mismatch["within_tolerance"]
    assert mismatch["outside_tolerance_count"] == 1


def test_main_persists_failure_and_raises_for_tolerance_miss(monkeypatch, tmp_path):
    output = tmp_path / "failed.json"
    failed = {
        "finite": True,
        "within_tolerance": False,
        "exact_equal": False,
        "outside_tolerance_count": 1,
    }
    monkeypatch.setattr(
        check,
        "_run_npu",
        lambda _args, _inputs: {
            "comparisons": {"eager_vs_cpu_reference": failed},
            "position_change_effect": {"finite": True, "exact_equal": False, "observed": True},
            "changed_input_effect": {"finite": True, "exact_equal": False, "observed": True},
        },
    )

    with pytest.raises(AssertionError, match="numerical comparisons failed"):
        check.main(["--head-dim", "8", "--page", "16", "--tile", "16", "--output", str(output)])

    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report["error_type"] == "AssertionError"


def test_main_fails_closed_when_position_only_replay_has_no_effect(monkeypatch, tmp_path):
    output = tmp_path / "frozen-position.json"
    passed = {
        "finite": True,
        "within_tolerance": True,
        "exact_equal": True,
        "outside_tolerance_count": 0,
    }
    monkeypatch.setattr(
        check,
        "_run_npu",
        lambda _args, _inputs: {
            "comparisons": {"position_only_aclgraph_replay_vs_cpu_reference": passed},
            "position_change_effect": {"finite": True, "exact_equal": True, "observed": False},
            "changed_input_effect": {"finite": True, "exact_equal": False, "observed": True},
        },
    )

    with pytest.raises(AssertionError, match="changed position replay"):
        check.main(["--head-dim", "8", "--page", "16", "--tile", "16", "--output", str(output)])

    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert not report["position_change_effect"]["observed"]


def test_main_persists_npu_runtime_exception(monkeypatch, tmp_path):
    output = tmp_path / "runtime-error.json"

    def fail(_args, _inputs):
        raise RuntimeError("capture exploded")

    monkeypatch.setattr(check, "_run_npu", fail)
    with pytest.raises(RuntimeError, match="capture exploded"):
        check.main(["--head-dim", "8", "--page", "16", "--tile", "16", "--output", str(output)])

    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report["error_type"] == "RuntimeError"
    assert report["error"] == "capture exploded"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"qheads": 7}, "divisible"),
        ({"qheads": 16, "kvheads": 2}, "GQA groups"),
        ({"tile": 8}, "tile must be"),
        ({"page": 24, "tile": 16}, "divide page"),
        ({"head_dim": 7}, "power of two"),
        ({"device": "cpu"}, "explicit npu"),
    ],
)
def test_invalid_contract_rejected_before_npu_initialization(overrides, message):
    with pytest.raises(ValueError, match=message):
        check._validate_args(_args(**overrides))
