# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the independent FULL-mask FIA operator probe."""

import json

import pytest
import torch

from examples import check_specslo_tree_fia as probe


def _args():
    return probe._parser().parse_args([
        "--device", "cpu", "--heads", "4", "--kv-heads", "2", "--head-dim", "8",
        "--block-size", "4", "--max-context", "32", "--contexts", "3", "4", "--output", "unused",
    ])


@pytest.mark.parametrize("counts", [[1, 1], [1, 5], [2, 5], [5, 2]])
@pytest.mark.parametrize("scratch", [False, True])
def test_fia_query_and_candidate_accounting_has_no_query_padding(counts, scratch):
    case = probe._make_case([3, 4], counts, scratch=scratch, args=_args(), seed=7)
    assert case["query"].shape[0] == sum(counts)
    assert case["candidate_count"] == sum(counts) - len(counts)
    assert case["actual_seq_lengths"] == [counts[0], sum(counts)]
    assert case["mask"].shape == (2, 1, max(counts), 32)
    row = 0
    for request, count in enumerate(counts):
        assert case["mask"][request, 0, count:].all()
        assert len(set(case["positions"][request])) == count
        for query in range(count):
            assert torch.where(~case["mask"][request, 0, query])[0].tolist() == case["visible_rows"][row]
            row += 1


def test_fia_nonprefix_selected_tree_masks_exclude_siblings():
    case = probe._make_case([3, 4], [2, 5], scratch=True, args=_args(), seed=9)
    assert case["parents"][1] == [-1, 0, -1, 2]
    primary_deep_row = case["visible_rows"][2 + 2]
    positions = case["positions"][1]
    assert positions[0] in primary_deep_row
    assert positions[1] in primary_deep_row and positions[2] in primary_deep_row
    assert positions[3] not in primary_deep_row and positions[4] not in primary_deep_row


@pytest.mark.parametrize("counts", [[1, 1], [1, 5], [2, 5]])
@pytest.mark.parametrize("nan", [False, True])
def test_explicit_ancestor_oracle_is_invariant_to_blocked_kv(counts, nan):
    case = probe._make_case([3, 4], counts, scratch=True, args=_args(), seed=5)
    poisoned = probe._poison_blocked(case, nan=nan)
    assert poisoned is not None and poisoned["poisoned_logical_positions"]
    row = poisoned["checked_row"]
    reference = probe._oracle(case)[row]
    changed = probe._oracle(poisoned)[row]
    assert reference.dtype == torch.float64
    assert torch.isfinite(changed).all()
    assert torch.equal(reference, changed)


def test_float64_oracle_ignores_fia_mask_and_sdpa(monkeypatch):
    case = probe._make_case([3, 4], [2, 5], scratch=True, args=_args(), seed=5)
    original = probe._oracle(case)
    case["mask"].fill_(False)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", lambda *args, **kwargs: pytest.fail())
    assert torch.equal(probe._oracle(case), original)


@pytest.mark.parametrize("layout", ["TND", "BNSD"])
def test_single_query_full_mask_abi_uses_explicit_lengths_and_precision(layout):
    case = probe._make_case([3, 4], [1, 1], scratch=True, args=_args(), seed=5)
    tensors = probe._device_tensors(case, "cpu", layout)
    kwargs = probe._fia_kwargs(case, tensors, layout)
    assert kwargs["sparse_mode"] == 1 and kwargs["inner_precise"] == 1
    assert kwargs["atten_mask"].shape == (2, 1, 1, 32)
    assert kwargs["actual_seq_lengths"] == ([1, 2] if layout == "TND" else [1, 1])
    assert kwargs["actual_seq_lengths_kv"] == [9, 10]
    assert kwargs["query"].shape == ((2, 4, 8) if layout == "TND" else (2, 4, 1, 8))
    assert kwargs["key"].shape == (16, 4, 16)


def test_heterogeneous_bnsd_rejected_instead_of_padding_queries():
    case = probe._make_case([3, 4], [2, 5], scratch=False, args=_args(), seed=5)
    with pytest.raises(ValueError, match="unpadded TND"):
        probe._device_tensors(case, "cpu", "BNSD")


def test_cpu_probe_records_honest_provenance_and_q1_holes(tmp_path):
    output = tmp_path / "fia-cpu.json"
    args = [
        "--device", "cpu", "--heads", "4", "--kv-heads", "2", "--head-dim", "8",
        "--block-size", "4", "--max-context", "32", "--contexts", "3", "4", "--output", str(output),
    ]
    assert probe.main(args) == 0
    document = json.loads(output.read_text())
    assert document["status"] == "complete" and document["passed"]
    assert not document["npu_executed"]
    assert not document["graphs"]
    assert len(document["cases"]) == 16
    single = [row for row in document["cases"] if row["query_counts"] == [1, 1] and row["scratch"]]
    assert single and all(len(row["blocked_kv_perturbations"]) == 2 for row in single)
    assert all("fia_vs_cpu64" not in row for row in document["cases"])


def test_cpu_cannot_request_graph_evidence():
    args = _args()
    args.graph = True
    with pytest.raises(ValueError, match="cannot claim an NPU graph"):
        probe._validate(args)
