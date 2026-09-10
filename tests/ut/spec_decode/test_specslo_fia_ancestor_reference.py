# SPDX-License-Identifier: Apache-2.0
"""CPU oracle construction only; no NPU result or model equivalence claim."""

import json
from copy import deepcopy
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
import torch

from examples.check_specslo_fia_ancestor_reference import (
    _engine_config,
    _parser,
    _scenario,
    ancestor_nodes,
    ancestor_spec,
    load_prefix_cases,
    oracle_metadata,
    selected_parents,
)


def test_real_native_config_validates_all_spec_rhythm_startup_dependencies_without_loading_devices():
    from vllm_ascend.spec_decode.pearl.native_engine import NativePearlConfig

    args = _parser().parse_args(["--output", "unused.json"])
    config = _engine_config(args)  # Real __post_init__ performs all validation.
    assert isinstance(config, NativePearlConfig)
    assert config.enable_spec_rhythm
    assert config.enable_continuous_batching and config.enable_preemptive_scheduling
    assert (config.draft_tp_size, config.target_tp_size) == (1, 3)
    assert (config.spec_rhythm_tree_width, config.spec_rhythm_tree_depth) == (2, 3)
    assert config.max_num_seqs == 2 and not config.enforce_eager
    for field in ("enable_continuous_batching", "enable_preemptive_scheduling"):
        with pytest.raises(ValueError, match="requires"):
            replace(config, **{field: False})


def test_real_diagnostic_config_large_context_has_sufficient_prefill_token_capacity():
    config = _engine_config(_parser().parse_args(["--output", "unused.json", "--max-model-len", "32768"]))
    assert config.max_num_batched_tokens >= config.max_model_len


def _specs(scratch=False):
    specs = []
    for sequence, (prefix, parents) in enumerate(((3, [-1]), (7, [-1, 0, -1, 1]))):
        start = prefix + 7 if scratch else prefix
        logical = prefix + 4 if scratch else prefix
        specs.append(
            {
                "sequence_id": sequence,
                "parents": parents,
                "prefix_count": prefix,
                "dependencies": list(range(prefix, prefix + 4)) if scratch else [],
                "physical": list(range(start, start + len(parents) + 1)),
                "positions": [logical, *[logical + len(ancestor_nodes(parents, node)) for node in range(len(parents))]],
                "tokens": list(range(10, 11 + len(parents))),
            }
        )
    return specs


@pytest.mark.parametrize("scratch", [False, True])
def test_independent_heterogeneous_full_mask_and_noncontiguous_page_addresses(scratch):
    specs = _specs(scratch)
    tables = [[3, 0, 6, 2, 4, 5, 1, 7], [12, 11, 8, 10, 15, 9, 14, 13]]
    inputs, positions, metadata = oracle_metadata(specs, tables, 4, 32)
    assert inputs.numel() == positions.numel() == 7
    assert metadata.actual_seq_lengths_q == (2, 7)
    assert metadata.tree_attention_mask.shape == (2, 1, 5, 32)
    assert metadata.tree_attention_mask[0, 0, 2:].all()  # Envelope padding is not a query.
    assert metadata.context_lens.device.type == "cpu"
    assert metadata.tree_attention and metadata.use_fused_infer_attention
    second = specs[1]
    root, branch_root, branch_child, other_root, branch_leaf = second["physical"]
    expected = sorted([*range(7), *second["dependencies"], root, branch_root, branch_child, branch_leaf])
    visible = (~metadata.tree_attention_mask[1, 0, 4]).nonzero().reshape(-1).tolist()
    assert visible == expected
    assert other_root not in visible
    assert metadata.slot_mapping.tolist() == [
        tables[spec["sequence_id"]][value // 4] * 4 + value % 4 for spec in specs for value in spec["physical"]
    ]
    assert metadata.request_block_tables.tolist() == tables
    assert metadata.sequence_lens == tuple(max(spec["physical"]) + 1 for spec in specs)


def test_noncontiguous_selection_is_ancestor_closed_and_positions_not_consecutive_depths():
    from vllm_ascend.spec_decode.pearl.tree import build_tree_speculation_plan, pack_selected_tree_plan

    full = build_tree_speculation_plan(2, 3, 7, 32, device="cpu")
    parents = [-1, 0, 1, -1, 0, 1]
    selected = [0, 1, 3, 5]
    assert full.parent_indices.tolist() == parents
    packed = pack_selected_tree_plan(full, selected)
    assert packed.parent_indices.tolist() == selected_parents(parents, selected) == [-1, 0, -1, 1]
    assert packed.cache_positions.tolist() == [7, 8, 9, 10, 11]
    assert packed.positions.tolist() == [7, 8, 9, 8, 10]
    assert [len(ancestor_nodes(parents, node)) for node in selected] == [1, 2, 1, 3]
    # The old diagnostic selection is valid for its imagined two-chain
    # tree, but NOT for the production spine-first tree. Keep the counterexample.
    with pytest.raises(ValueError, match="ancestor closed"):
        pack_selected_tree_plan(full, [0, 3, 4, 5])
    for invalid in ([4], [0, 5], [3, 5], [3, 3], [3, 0]):
        with pytest.raises(ValueError):
            selected_parents(parents, invalid)


@pytest.mark.parametrize("layout", ["normal", "scratch"])
@pytest.mark.parametrize("saved_prefix", [False, True])
def test_actual_scenario_builds_packs_and_matches_independent_oracle_on_cpu(layout, saved_prefix):
    """Exercise the actual diagnostic path, mocking only device allocation/compute.

    The real production builder, packer, eager scratch helper and metadata
    builder run here. This test establishes topology/mask/address agreement,
    NOT model numerical equivalence or graph execution.
    """
    from tests.ut.spec_decode.test_tree_draft_batch import _engine

    engine = _engine()
    engine.target_vocab_size = 32
    engine.model.block_size = 4
    engine._allocate_cache = MagicMock()
    engine._run_packed_hidden = MagicMock()
    # Reordered pages catch logical-position-as-physical-slot mistakes.
    engine.cache_block_tables = torch.tensor(
        [[3, 0, 6, 2, 4, 5, 1, 7], [12, 11, 8, 10, 15, 9, 14, 13]], dtype=torch.int32
    )
    args = _parser().parse_args(["--output", "unused.json", "--contexts", "4", "8", "--max-model-len", "32"])
    cases = (
        [
            {"prefix_token_ids": [2, 3, 4, 5], "candidate_token_ids": [10, 11, 12, 13, 14, 15]},
            {"prefix_token_ids": [2, 3, 4, 5, 6, 7, 8, 9], "candidate_token_ids": [20, 21, 22, 23, 24, 25]},
        ]
        if saved_prefix
        else None
    )
    specs, production, oracle = _scenario(engine, args, layout, [2, 3], cases)
    assert [spec["original_selected_ids"] for spec in specs] == [[0], [0, 1, 3, 5]]
    assert [spec["parents"] for spec in specs] == [[-1], [-1, 0, -1, 1]]
    assert [len(spec["tokens"]) for spec in specs] == [2, 5]
    assert torch.equal(production[0], oracle[0]) and torch.equal(production[1], oracle[1])
    for field in (
        "slot_mapping",
        "context_lens",
        "block_tables",
        "request_block_tables",
        "attention_mask",
        "tree_attention_mask",
    ):
        assert torch.equal(getattr(production[2], field), getattr(oracle[2], field)), field
    assert production[2].actual_seq_lengths_q == oracle[2].actual_seq_lengths_q == (2, 7)
    assert production[2].sequence_lens == oracle[2].sequence_lens
    for spec in specs:
        base = spec["prefix_count"]
        if layout == "scratch":
            assert spec["dependencies"] == [base, base + 1, base + 2, base + 3]
            assert spec["physical"][0] == base + 7
            assert spec["positions"][0] == base + 4
            # Every query blocks all three parent's rejected side branches.
            row_start = 0 if spec["sequence_id"] == 0 else 2
            count = len(spec["tokens"])
            assert oracle[2].attention_mask[row_start : row_start + count, base + 4 : base + 7].all()
        else:
            assert not spec["dependencies"]
            assert spec["physical"][0] == spec["positions"][0] == base
    if saved_prefix:
        assert specs[1]["tokens"][1:] == [20, 21, 23, 25]


@pytest.mark.parametrize("parents,node", [([0], 0), ([1, 0], 1), ([-2], 0), ([-1], 1)])
def test_invalid_or_cyclic_parent_relation_cannot_enter_oracle(parents, node):
    with pytest.raises(ValueError):
        ancestor_nodes(parents, node)


def test_ancestor_only_keeps_real_scratch_slots_and_excludes_other_branch():
    spec = _specs(scratch=True)[1]
    subset = ancestor_spec(spec, 4)
    assert subset["parents"] == [-1, 0, 1]
    assert subset["physical"] == [spec["physical"][row] for row in [0, 1, 2, 4]]
    assert subset["dependencies"] == spec["dependencies"]
    assert ancestor_spec(spec, 0)["parents"] == []
    tables = [[0] * 8, list(range(8))]
    _, _, full = oracle_metadata([spec], tables, 4, 32)
    _, _, compact = oracle_metadata([subset], tables, 4, 32)
    assert compact.actual_seq_lengths_q == (4,)
    assert torch.equal(full.attention_mask[-1], compact.attention_mask[-1])


def test_oracle_rejects_overlapping_dependency_before_execution():
    specs = deepcopy(_specs(scratch=True))
    specs[0]["dependencies"].append(specs[0]["physical"][0])
    with pytest.raises(ValueError, match="overlaps immutable"):
        oracle_metadata(specs, [list(range(8)), list(range(8, 16))], 4, 32)


def test_saved_prefix_source_preserves_exact_tokens_and_does_not_hide_extra_cases(tmp_path):
    path = tmp_path / "old.json"
    normal = {
        "layout": "normal",
        "prefix_token_ids": [1, 2, 3],
        "root_token_id": 3,
        "candidate_token_ids": [4, 5, 6, 7, 8, 9],
    }
    cases = [normal, {**normal, "layout": "scratch"}, {**normal, "prefix_token_ids": [8, 3]}]
    path.write_text(json.dumps({"cases": cases}))
    assert load_prefix_cases(path, 32) == [cases[0], cases[2]]
    path.write_text(json.dumps({"cases": [normal] * 3}))
    with pytest.raises(ValueError, match="exactly two"):
        load_prefix_cases(path, 32)


def test_saved_prefix_root_mismatch_is_rejected_before_model_load(tmp_path):
    path = tmp_path / "bad.json"
    case = {
        "layout": "normal",
        "prefix_token_ids": [1, 2],
        "root_token_id": 3,
        "candidate_token_ids": [4, 5, 6, 7, 8, 9],
    }
    path.write_text(json.dumps({"cases": [case, case]}))
    with pytest.raises(ValueError, match="do not fit"):
        load_prefix_cases(path, 32)
