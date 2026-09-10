# SPDX-License-Identifier: Apache-2.0
"""CPU structural tests; these do not substitute for the NPU logit probe."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from examples.check_specslo_tree_numerics import (
    _ancestor_path,
    _build_parser,
    _candidate_rows,
    _logit_report,
    _passed,
    _prefix_prefill_length,
    _tree_logits,
    _validate_args,
)


def test_head_diagnostics_reject_graph_cli_and_accept_eager():
    args = _build_parser().parse_args(
        [
            "--output",
            "/report.json",
            "--prefix-output-tokens",
            "0",
            "--head-diagnostics",
        ]
    )
    _validate_args(args)
    args.graph = True
    with pytest.raises(ValueError, match="requires eager"):
        _validate_args(args)


def test_tree_head_diagnostics_reuses_original_hidden_without_new_model_forward():
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    model = MagicMock(return_value=hidden)
    model.lm_head = SimpleNamespace(context=SimpleNamespace(size=1))
    model.make_tree_attention_metadata.return_value = (
        torch.tensor([1, 2]),
        torch.tensor([3, 4]),
        object(),
    )
    model.compute_logits.return_value = torch.ones(2, 4)
    engine = SimpleNamespace(
        model=model, target_vocab_size=3, cache_block_tables=[[0]], _ensure_cache_capacity=MagicMock()
    )
    plan = SimpleNamespace(cache_positions=torch.tensor([3, 4]))
    with (
        patch("torch.npu.synchronize"),
        patch("examples.check_specslo_greedy_ties.compare_native_head", return_value={"tp_rank": 0}) as compare,
    ):
        logits, outcome = _tree_logits(engine, plan, 1, [2], False, head_diagnostics=True)
    model.assert_called_once()
    assert compare.call_args.args[1] is hidden
    assert compare.call_args.args[0] is model.lm_head
    assert logits.shape == (2, 3)
    assert outcome["head_diagnostics"]["additional_transformer_forwards"] == 0
    assert outcome["head_diagnostics"]["ranks"] == [{"tp_rank": 0}]


def test_tree_head_diagnostics_not_run_without_opt_in():
    model = MagicMock(return_value=torch.ones(1, 2))
    model.make_tree_attention_metadata.return_value = (torch.tensor([1]), torch.tensor([3]), object())
    model.compute_logits.return_value = torch.ones(1, 3)
    engine = SimpleNamespace(
        model=model, target_vocab_size=3, cache_block_tables=[[0]], _ensure_cache_capacity=MagicMock()
    )
    with patch("torch.npu.synchronize"), patch("examples.check_specslo_greedy_ties.compare_native_head") as compare:
        _, outcome = _tree_logits(engine, SimpleNamespace(cache_positions=torch.tensor([3])), 1, [], False)
    compare.assert_not_called()
    assert "head_diagnostics" not in outcome


def test_prefix_mode_keeps_last_token_for_probe_and_original_prompt_boundary():
    assert _prefix_prefill_length([1, 2, 3, 4, 5], 2, "prefill") == 4
    assert _prefix_prefill_length([1, 2, 3, 4, 5], 2, "incremental") == 2
    assert _prefix_prefill_length([1, 2], 2, "incremental") == 1
    with pytest.raises(ValueError, match="original prompt length"):
        _prefix_prefill_length([1, 2], None, "incremental")


def test_ancestor_paths_exclude_siblings_and_handle_noncontiguous_path():
    parents = [-1, 0, -1, 2, 1, 3]
    assert _ancestor_path(parents, -1) == []
    assert _ancestor_path(parents, 4) == [0, 1, 4]
    assert _ancestor_path(parents, 5) == [2, 3, 5]


@pytest.mark.parametrize("parents,node", [([1, 0], 0), ([-2], 0), ([-1], 1)])
def test_ancestor_path_rejects_invalid_graph(parents, node):
    with pytest.raises(ValueError):
        _ancestor_path(parents, node)


def test_fixed_candidate_layout_retains_spine_then_per_depth_siblings():
    row = _candidate_rows([10, 20, 30], width=3, depth=3, vocabulary_size=100)
    assert row == [10, 20, 30, 12, 13, 22, 23, 32, 33]


def test_small_margin_is_reported_but_does_not_hide_argmax_change():
    reference = torch.tensor([1.0, 0.999, -3.0])
    actual = torch.tensor([0.999, 1.0, -3.0])
    result = _logit_report(actual, reference, atol=0.01, rtol=0.001, top_k=2)
    assert result["within_tolerance"]
    assert not result["argmax_exact"]
    assert result["reference_margin_within_2_max_error"]
    assert not _passed(result, require_argmax=True)
    assert _passed(result, require_argmax=False)
    assert result["actual_argmax"] == 1
    assert result["reference_argmax"] == 0


def test_same_argmax_does_not_hide_large_full_vocabulary_error():
    result = _logit_report([8, 0.5, -3], [8, 0.1, -3], atol=0.01, rtol=0.001, top_k=2)
    assert result["argmax_exact"]
    assert not result["within_tolerance"]
    assert result["outside_tolerance_count"] == 1
    assert result["max_error_token_id"] == 1
    assert result["total_variation"] > 0
    assert result["jensen_shannon_divergence"] > 0
    assert result["max_abs_probability_error"] > 0
    assert result["top_k_id_overlap"] == 2
    assert not _passed(result, require_argmax=True)


def test_nonfinite_logits_fail_instead_of_generating_nan_report_numbers():
    result = _logit_report([float("nan"), 1], [0, 1], atol=0.01, rtol=0.001, top_k=2)
    assert not result["finite"]
    assert not _passed(result, require_argmax=False)


def test_probe_requires_baseline_for_requested_output_prefix():
    args = _build_parser().parse_args(["--output", "/report.json"])
    with pytest.raises(ValueError, match="baseline-json"):
        _validate_args(args)
    args = _build_parser().parse_args(["--output", "/report.json", "--prefix-output-tokens", "0"])
    _validate_args(args)


def test_probe_prefix_offsets_must_align_with_rows():
    args = _build_parser().parse_args(
        [
            "--output",
            "/report.json",
            "--baseline-json",
            "/baseline.json",
            "--request-indices",
            "1",
            "2",
            "3",
            "--prefix-output-tokens",
            "3",
            "4",
        ]
    )
    with pytest.raises(ValueError, match="one per request"):
        _validate_args(args)
