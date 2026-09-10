# SPDX-License-Identifier: Apache-2.0
"""CPU checks for actual-cache diagnostic isolation; not NPU numeric claims."""

from types import SimpleNamespace

import pytest
import torch

from examples.check_specslo_online_tree_numerics import (
    _annotate_final_outputs,
    _build_parser,
    _engine_config_values,
    _independent_mask,
    _install_observer,
    _kv_equal,
    _parse_watch,
    _physical_prefix_slots,
    _request_parameter_rows,
    _restore_kv,
    _selected_queries,
    _snapshot_kv,
    _teacher_comparisons,
)
from vllm_ascend.spec_decode.pearl.native_model import NativeAttentionMetadata, NativeLMHead, NativeTPContext
from vllm_ascend.spec_decode.pearl.tree import build_tree_speculation_plan, pack_selected_tree_plan


def test_boundary_budget_is_distinct_from_online_capacity_and_queue():
    args = _build_parser().parse_args(
        [
            "--output",
            "unused.json",
            "--num-prompts",
            "4",
            "--online-capacity",
            "2",
            "--verification-budget",
            "4",
            "--max-tokens",
            "32",
            "--prefill-chunk-size",
            "2",
        ]
    )
    config = _engine_config_values(args)
    assert config["max_num_seqs"] == config["prefill_chunk_size"] == 2
    assert config["max_num_queued_seqs"] == config["spec_rhythm_verification_budget"] == 4
    assert config["gamma"] == config["spec_rhythm_max_eager_tokens"] == 4
    assert config["gpu_memory_utilization"] == 0.85


def test_legacy_gamma_and_eager_cap_do_not_silently_become_tree_capacity():
    args = _build_parser().parse_args(
        [
            "--output",
            "unused.json",
            "--tree-depth",
            "3",
            "--gamma",
            "4",
            "--max-eager-tokens",
            "4",
            "--no-online-prefill",
            "--acceptance-floor",
            "0.1",
            "--urgency-threshold",
            "0.75",
        ]
    )
    config = _engine_config_values(args)
    assert config["gamma"] == config["spec_rhythm_max_eager_tokens"] == 4
    assert config["spec_rhythm_tree_width"] * config["spec_rhythm_tree_depth"] == 6
    assert not config["spec_rhythm_online_prefill"]
    assert config["spec_rhythm_acceptance_floor"] == 0.1


def test_sampling_explicitly_greedy_and_rebases_manifest_arrivals_without_retokenization():
    args = _build_parser().parse_args(
        [
            "--output",
            "unused.json",
            "--max-tokens",
            "32",
            "--arrival-interval",
            "0.05",
            "--arrival-lead",
            "0.1",
            "--slo-tpot-ms",
            "40",
            "50",
            "150",
        ]
    )
    rows = [{"arrival_ts": 99999, "arrival_offset_sec": 0.2, "slo_tpot_ms": 25}, {}, {}, {}]
    parameters = _request_parameter_rows(args, rows, 200.0)
    assert [row["arrival_ts"] for row in parameters] == [200.2, 200.05, 200.1, 200.15]
    assert [row["slo_tpot_ms"] for row in parameters] == [25, 50, 150, 40]
    assert all(row["temperature"] == row["draft_temperature"] == 0 for row in parameters)
    assert all(row["max_tokens"] == 32 and row["ignore_eos"] for row in parameters)


def test_no_arrival_or_slo_constraints_are_invented_by_default():
    args = _build_parser().parse_args(["--output", "unused.json"])
    parameters = _request_parameter_rows(args, [{}, {}], 200.0)
    assert all(row["arrival_ts"] is None and row["slo_tpot_ms"] is None for row in parameters)


@pytest.mark.parametrize("row", [{"slo_tpot_ms": 0}, {"slo_tpot_ms": float("nan")}, {"arrival_offset_sec": -1}])
def test_bad_manifest_constraints_are_rejected(row):
    args = _build_parser().parse_args(["--output", "unused.json"])
    with pytest.raises(ValueError):
        _request_parameter_rows(args, [row], 100.0)


def test_watch_indices_are_zero_based_and_range_checked():
    assert _parse_watch(["5:27", "6:61", "5:27"], 8, 64) == {(5, 27), (6, 61)}
    for values in (["5"], ["8:0"], ["0:64"], ["-1:0"]):
        with pytest.raises(ValueError):
            _parse_watch(values, 8, 64)


def test_watch_selection_uses_actual_logical_depth_and_packed_count():
    full = build_tree_speculation_plan(2, 2, 10, 32)
    packed = pack_selected_tree_plan(full, [0, 2])
    selected = _selected_queries([packed], [0], [8], {(0, 4)})
    assert len(selected) == 2
    assert [item["query_row"] for item in selected] == [1, 2]
    assert [item["ancestor_nodes"] for item in selected] == [[0], [1]]
    assert all(item["root_output_offset"] == 3 for item in selected)


def test_independent_ancestor_mask_blocks_siblings():
    plan = build_tree_speculation_plan(2, 2, 3, 16)
    for node in range(-1, 4):
        torch.testing.assert_close(_independent_mask(plan, node, device="cpu"), plan.attention_mask[node + 1])
    assert _independent_mask(plan, 1, device="cpu")[6]


def test_prefix_physical_mapping_spans_noncontiguous_pages():
    slots = _physical_prefix_slots(SimpleNamespace(prefix_len=6), torch.tensor([3, 1, -1]), 4)
    assert slots.tolist() == [12, 13, 14, 15, 4, 5]
    with pytest.raises(ValueError, match="unallocated"):
        _physical_prefix_slots(SimpleNamespace(prefix_len=9), torch.tensor([3, 1, -1]), 4)


@pytest.mark.parametrize("paged", [False, True])
def test_snapshot_restore_preserves_every_layer_and_cache_layout(paged):
    shape = (4, 4, 1, 1) if paged else (16, 1, 1)
    layers = [
        SimpleNamespace(
            self_attn=SimpleNamespace(
                uses_paged_attention=paged,
                key_cache=torch.arange(16).float().view(shape),
                value_cache=(torch.arange(16).float() + index).view(shape),
            )
        )
        for index in range(2)
    ]
    model = SimpleNamespace(layers=layers)
    slots = torch.tensor([2, 5, 12])
    snapshot = _snapshot_kv(model, slots)
    for layer in layers:
        layer.self_attn.key_cache.fill_(-1)
        layer.self_attn.value_cache.fill_(-2)
    assert not _kv_equal(model, slots, snapshot)
    _restore_kv(model, slots, snapshot)
    assert _kv_equal(model, slots, snapshot)


def _teacher_fixture(fail=False):
    plan = build_tree_speculation_plan(2, 2, 3, 16)
    inputs = torch.tensor([1, 2, 3, 4, 5])
    slots = plan.cache_positions.long()
    attention = SimpleNamespace(
        uses_paged_attention=False,
        block_size=4,
        key_cache=torch.arange(16).float().view(16, 1, 1),
        value_cache=torch.arange(16).float().view(16, 1, 1),
    )
    metadata = NativeAttentionMetadata(
        slot_mapping=slots,
        context_lens=(slots + 1).int(),
        block_tables=torch.arange(4).expand(5, -1),
        attention_mask=plan.attention_mask,
    )

    class Model:
        layers = [SimpleNamespace(self_attn=attention)]

        def __call__(self, input_ids, positions, single):
            assert input_ids.numel() == 1
            attention.key_cache.index_fill_(0, single.slot_mapping, 99.0)
            attention.value_cache.index_fill_(0, single.slot_mapping, 77.0)
            if fail:
                attention.key_cache[0] = -100
                raise RuntimeError("intentional singleton fault after cache mutation")
            return input_ids.float().unsqueeze(1)

        def compute_logits(self, hidden):
            return hidden * torch.arange(8).view(1, -1)

    engine = SimpleNamespace(model=Model(), target_vocab_size=8)
    hidden = inputs.float().unsqueeze(1)
    selected = _selected_queries([plan], [0], [3], {(0, 2)})
    captured = (inputs, plan.positions, metadata, hidden)
    args = SimpleNamespace(atol=0.01, rtol=0.001, top_k=5)
    return engine, plan, selected, captured, args, engine.model.compute_logits(hidden)


def test_teacher_path_reads_actual_prefix_and_restores_original_query_kv():
    engine, plan, selected, captured, args, logits = _teacher_fixture()
    snapshot = _snapshot_kv(engine.model, torch.arange(16))
    reports = _teacher_comparisons(engine, [plan], selected, captured, args, logits)
    assert reports
    assert all(row["logits"]["max_abs_error"] == 0 for row in reports)
    assert all(row["committed_prefix_kv_unchanged"] for row in reports)
    assert all(row["query_kv_restored"] and row["prefix_kv_restored"] for row in reports)
    assert reports[0]["query_kv_by_layer"][0]["key_cache_max_abs_error"] > 0
    assert _kv_equal(engine.model, torch.arange(16), snapshot)


def test_teacher_fault_restores_both_query_and_prefix_in_finally():
    engine, plan, selected, captured, args, logits = _teacher_fixture(fail=True)
    snapshot = _snapshot_kv(engine.model, torch.arange(16))
    with pytest.raises(RuntimeError, match="intentional singleton fault"):
        _teacher_comparisons(engine, [plan], selected, captured, args, logits)
    assert _kv_equal(engine.model, torch.arange(16), snapshot)


def test_final_prefix_annotation_does_not_treat_every_sibling_as_accepted():
    records = [
        {
            "input_token_ids": [9, 10, 20],
            "original_production_query_tokens": [10, 11, 99],
            "selected_queries": [
                {
                    "request_index": 0,
                    "output_offset": 2,
                    "query_row": row,
                    "tree_row_start": 0,
                    "ancestor_nodes": [row - 1],
                    "root_output_offset": 1,
                }
                for row in (1, 2)
            ],
        }
    ]
    _annotate_final_outputs(records, [[9, 10, 11]])
    first, second = records[0]["selected_queries"]
    assert first["ancestor_tokens_match_final_prefix"] and first["production_prediction_matches_final_output"]
    assert not second["ancestor_tokens_match_final_prefix"]


def test_online_forward_hook_captures_actual_packed_hidden_only_once_and_uninstalls():
    plan = build_tree_speculation_plan(2, 2, 3, 16)
    inputs = torch.tensor([1, 2, 3, 4, 5])
    metadata = NativeAttentionMetadata(
        slot_mapping=plan.cache_positions.long(),
        context_lens=(plan.cache_positions + 1).int(),
        block_tables=torch.arange(4).expand(5, -1),
        attention_mask=plan.attention_mask,
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.lm_head = NativeLMHead(8, 1, NativeTPContext(group=None, rank=0, size=1, leader_rank=0))
            self.lm_head.weight.data.copy_(torch.arange(8).float().view(8, 1))

        def forward(self, input_ids, positions, attention_metadata):
            self.calls += 1
            return input_ids.float().unsqueeze(1)

    model = Model()

    def original_target(plans, roots, candidates, sequence_ids, return_logits):
        hidden = model(inputs, plan.positions, metadata)
        return {"target_query_token_ids": model.lm_head.greedy(hidden, 8)}

    engine = SimpleNamespace(
        model=model,
        is_draft=False,
        rank=1,
        topology=SimpleNamespace(target_leader_rank=1),
        target_vocab_size=8,
        target_tree_forward=original_target,
    )
    records = []
    restore = _install_observer(engine, [[7, 8, 9]], {(0, 2)}, SimpleNamespace(teacher_paths=False), records)
    result = engine.target_tree_forward([plan], [1], [[2, 3, 4, 5]], [0], False)
    assert result["target_query_token_ids"].tolist() == [7] * 5
    assert model.calls == 1
    assert records[0]["actual_packed_query_count"] == 5
    assert records[0]["original_production_matches_same_hidden_greedy"]
    assert not model._forward_hooks
    restore()
    assert engine.target_tree_forward is original_target
