# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the world4/target-TP3 HCCL tree probe."""

from __future__ import annotations

import gzip

import pytest
import torch

from examples import probe_specslo_mc2_subgroup_hccl_tree as probe
from examples.specslo_hccl_bf16_tree_report import (
    CANDIDATE_NAMES,
    build_signature_map,
    make_candidates,
    summarize_output,
    tensor_sha256,
    write_element_signature_artifact,
)


def test_target_rank_to_real_input_mapping_and_physical_device_records():
    assert [probe._target_rank_for_global_rank(rank) for rank in range(4)] == [
        None,
        0,
        1,
        2,
    ]

    records = probe._physical_device_records(("1", "2", "4", "5"))

    assert records[0] == {
        "global_rank": 0,
        "local_rank": 0,
        "physical_device": "1",
        "role": "draft_coordinator",
        "target_rank": None,
        "input_rank": None,
    }
    assert [record["physical_device"] for record in records] == ["1", "2", "4", "5"]
    assert [record["input_rank"] for record in records[1:]] == [0, 1, 2]


def test_physical_device_cli_and_environment_must_describe_same_four_cards():
    assert probe._resolve_physical_devices("1,2,4,5", "1,2,4,5") == (
        "1",
        "2",
        "4",
        "5",
    )
    assert probe._resolve_physical_devices(None, "0,3,6,7") == (
        "0",
        "3",
        "6",
        "7",
    )
    with pytest.raises(ValueError, match="does not match"):
        probe._resolve_physical_devices("1,2,4,5", "0,1,2,3")
    with pytest.raises(ValueError, match="contain 4 entries"):
        probe._parse_physical_devices("1,2,4")
    with pytest.raises(ValueError, match="duplicates"):
        probe._parse_physical_devices("1,2,2,4")


def test_every_rank_uses_one_explicit_group_creation_order(monkeypatch):
    calls: list[tuple[tuple[int, ...], str]] = []

    def fake_new_group(*, ranks, backend):
        calls.append((tuple(ranks), backend))
        return f"group-{len(calls)}"

    monkeypatch.setattr(probe.dist, "new_group", fake_new_group)

    groups = probe._create_process_groups()

    assert calls == [
        ((1, 2, 3), "hccl"),
        ((1, 2, 3), "gloo"),
        ((0, 1, 2, 3), "gloo"),
    ]
    assert groups == {
        "target_hccl": "group-1",
        "target_gloo": "group-2",
        "world_gloo": "group-3",
    }


def test_single_node_layout_rejects_wrong_world_or_local_rank():
    probe._validate_layout(global_rank=3, local_rank=3, world_size=4)
    with pytest.raises(ValueError, match="full world size 4"):
        probe._validate_layout(global_rank=2, local_rank=2, world_size=3)
    with pytest.raises(ValueError, match="LOCAL_RANK == global rank"):
        probe._validate_layout(global_rank=2, local_rank=0, world_size=4)


def test_probe_requires_deterministic_aiv_environment(monkeypatch, tmp_path):
    args = probe._parser().parse_args(
        [
            "--input-dir",
            str(tmp_path),
            "--output-json",
            str(tmp_path / "report.json"),
            "--physical-devices",
            "1,2,4,5",
        ]
    )
    monkeypatch.setenv("HCCL_DETERMINISTIC", "true")
    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", "AIV")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "1,2,4,5")
    assert probe._validate_args(args) == ("1", "2", "4", "5")

    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU")
    with pytest.raises(RuntimeError, match="HCCL_OP_EXPANSION_MODE=AIV"):
        probe._validate_args(args)


def test_cpu_candidate_signature_and_artifact_match_standalone_contract(tmp_path):
    projections = [
        torch.tensor([[1.0, 0.5], [2.0, -3.0]], dtype=torch.bfloat16),
        torch.tensor([[0.25, 2.0], [-1.0, 4.0]], dtype=torch.bfloat16),
        torch.tensor([[3.0, -0.5], [0.5, 1.0]], dtype=torch.bfloat16),
    ]
    candidates = make_candidates(projections)
    actual = candidates["tree_01_then_2"].clone()

    membership, signature = build_signature_map(actual, candidates)
    summary = summarize_output(
        actual,
        candidates,
        tile_widths=(1, 2),
        row_block_size=1,
        max_flat_ranges=100,
    )
    output_sha = tensor_sha256(actual)
    artifact = write_element_signature_artifact(
        signature,
        tmp_path / "report.json",
        output_sha,
    )

    assert tuple(candidates) == CANDIDATE_NAMES
    assert bool(membership[0].all())
    assert bool(((signature & 1) == 1).all())
    assert summary["whole_tensor"]["candidate_mismatch_counts"]["tree_01_then_2"] == 0
    assert summary["flat_element_signature_ranges"]["truncated"] is False
    with gzip.open(artifact["path"], "rb") as stream:
        raw = stream.read()
    assert raw == signature.contiguous().numpy().tobytes()
    assert artifact["raw_sha256"] == tensor_sha256(signature)
