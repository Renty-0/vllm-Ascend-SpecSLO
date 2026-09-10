# SPDX-License-Identifier: Apache-2.0
"""CPU structure/oracle tests; NPU reduction tie behavior requires a real run."""

from types import SimpleNamespace

import pytest
import torch

from examples.check_specslo_greedy_ties import (
    _build_parser,
    _native_head_probe,
    _operator_probes,
    _validate_args,
    compare_logit_ties,
    compare_native_head,
)


def test_exact_bfloat16_ties_compare_device_and_explicit_min_id():
    logits = torch.full((2, 1500), -1.0, dtype=torch.bfloat16)
    logits[0, 1052] = logits[0, 1449] = 38.0
    logits[1, 594] = logits[1, 752] = 34.5
    result = compare_logit_ties(logits)
    assert [row["max_index"] for row in result["rows"]] == [1052, 594]
    assert [row["exact_tie_count"] for row in result["rows"]] == [2, 2]
    assert all(row["explicit_matches_cpu"] for row in result["rows"])


def test_exact_tie_test_does_not_turn_small_margin_into_tie():
    result = compare_logit_ties(torch.tensor([[38.01, 38.02]], dtype=torch.float32))
    assert result["rows"][0]["exact_tie_count"] == 1
    assert result["rows"][0]["explicit_min_exact_tie_id"] == 1
    quantized = compare_logit_ties(torch.tensor([[38.01, 38.02]], dtype=torch.bfloat16))
    assert quantized["rows"][0]["exact_tie_count"] == 2
    assert quantized["rows"][0]["explicit_min_exact_tie_id"] == 0


def test_token_offset_keeps_global_tie_ids():
    result = compare_logit_ties(torch.tensor([[1.0, 2.0, 2.0]]), token_offset=100)
    assert result["rows"][0]["exact_tie_ids_first32"] == [101, 102]
    assert result["rows"][0]["max_index"] == 101


@pytest.mark.parametrize("logits", [torch.empty(1, 0), torch.ones(2), torch.tensor([[float("nan")]])])
def test_bad_logit_shape_and_nonfinite_input_rejected(logits):
    with pytest.raises(ValueError):
        compare_logit_ties(logits)


def test_probe_includes_truncated_and_strided_layouts():
    args = SimpleNamespace(dtypes=["bfloat16"], vocab_sizes=[17])
    result = _operator_probes(args, torch.device("cpu"))
    assert [row["layout"] for row in result] == ["contiguous", "truncated", "stride2", "quantized_near_tie"]
    assert not result[1]["contiguous"] and not result[2]["contiguous"]
    assert result[2]["stride"][1] == 2


def test_tiny_native_head_compares_same_hidden_and_preserves_production_behavior():
    args = SimpleNamespace(head_vocab_size=37, nz_weight=False)
    result = _native_head_probe(args, torch.device("cpu"), 0, 1, torch.bfloat16)
    assert result["same_hidden_local_vs_full_projection_max_abs_error"] == 0
    assert result["production_greedy_matches_full_argmax"]
    assert result["production_greedy_matches_explicit_min_ties"]
    assert result["production_greedy_ids"] == [0, 0, 0, 0]


def test_diagnostic_reports_greedy_difference_without_rewriting_it():
    class Head:
        context = SimpleNamespace(rank=0, size=1)
        weight = torch.tensor([[1.0], [1.0]])
        vocab_start, vocab_end = 0, 2

        def __call__(self, hidden):
            return hidden @ self.weight.t()

        def greedy(self, hidden, vocabulary_size):
            return torch.ones(hidden.shape[0], dtype=torch.long)

    result = compare_native_head(Head(), torch.ones(1, 1), 2)
    assert result["production_greedy_ids"] == [1]
    assert result["full_head_argmax_ids"] == [0]
    assert not result["production_greedy_matches_full_argmax"]
    assert result["same_hidden_local_vs_full_projection_max_abs_error"] == 0


def test_cli_rejects_invalid_sizes_and_cpu_nz():
    for extra in (["--vocab-sizes", "1"], ["--nz-weight"]):
        args = _build_parser().parse_args(["--output", "unused.json", *extra])
        with pytest.raises(ValueError):
            _validate_args(args)
