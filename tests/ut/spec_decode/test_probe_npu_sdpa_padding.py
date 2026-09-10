# SPDX-License-Identifier: Apache-2.0
"""Check the isolated probe's reference and exact input accounting on CPU."""

import json

import pytest
import torch

from examples import probe_npu_sdpa_padding as probe


@pytest.mark.parametrize("mask_kind", ["tail", "holes"])
def test_probe_padding_and_holes_preserve_exact_visible_qkv(mask_kind):
    full = probe._inputs(9, 16, 4, 2, 8, mask_kind, 7, 1.0)
    query, keys, values, mask = full
    assert query.shape == (1, 4, 1, 8)
    assert keys.shape == values.shape == (1, 4, 16, 8)
    assert query.dtype == keys.dtype == values.dtype == torch.bfloat16
    assert not mask[..., 9:].any()
    assert not keys[..., 9:, :].count_nonzero()
    assert not values[..., 9:, :].count_nonzero()
    assert mask.sum() == (9 if mask_kind == "tail" else 6)
    assert not keys.masked_select(~mask.reshape(1, 1, 16, 1).expand_as(keys)).count_nonzero()
    original = query, keys[..., :9, :], values[..., :9, :], mask[..., :9]
    reference = probe._reference(*original)
    padded_reference = probe._reference(*full)
    assert reference.dtype == padded_reference.dtype == torch.float64
    assert torch.allclose(reference, padded_reference, atol=1e-12, rtol=1e-12)


def test_reference_is_independent_of_sdpa_and_manual_float32_matches_it(monkeypatch):
    data = probe._inputs(7, 16, 4, 2, 8, "holes", 123, 1.0)
    monkeypatch.setattr(
        torch.nn.functional,
        "scaled_dot_product_attention",
        lambda *args, **kwargs: pytest.fail("reference/manual route must not call SDPA"),
    )
    reference = probe._reference(*data)
    actual = probe._execute("manual_fp32", *data)
    assert torch.allclose(actual.double(), reference, atol=1e-6, rtol=1e-6)


def test_comparison_reports_instead_of_hiding_nonfinite_values():
    report = probe._comparison(torch.tensor([float("nan")]), torch.tensor([1.0]), atol=0.001, rtol=0.001)
    assert report == {"finite": False, "within_tolerance": False, "nonfinite_actual": 1, "nonfinite_reference": 0}


def test_cpu_probe_has_explicit_non_npu_provenance_and_all_methods(tmp_path):
    path = tmp_path / "probe.json"
    assert (
        probe.main(
            [
                "--device",
                "cpu",
                "--lengths",
                "7",
                "--pad-to",
                "16",
                "--heads",
                "4",
                "--kv-heads",
                "2",
                "--head-dims",
                "8",
                "--scales",
                "1",
                "--masks",
                "holes",
                "--output",
                str(path),
            ]
        )
        == 0
    )
    report = json.loads(path.read_text())
    assert report["status"] == "complete"
    assert report["torch_npu_version"] is None
    assert report["device_name"] == "CPU float32 test"
    assert {row["method"] for row in report["cases"]} == {
        "sdpa_bf16",
        "sdpa_fp32",
        "sdpa_fp32_additive",
        "manual_fp32",
    }
    assert all(row["requested_hf32"] == "not_applicable" for row in report["cases"])
    assert all(row["cpu_reference_padding_invariance"]["within_tolerance"] for row in report["cases"])
    manual = next(row for row in report["cases"] if row["method"] == "manual_fp32")
    assert manual["original_vs_cpu64"]["max_abs_error"] < 1e-6
    assert manual["padded_vs_cpu64"]["max_abs_error"] < 1e-6


def test_bad_gqa_or_padding_dimensions_fail_before_device_initialization():
    args = probe._build_parser().parse_args(["--output", "unused", "--heads", "7", "--kv-heads", "2"])
    with pytest.raises(ValueError, match="Query heads"):
        probe._validate(args)
