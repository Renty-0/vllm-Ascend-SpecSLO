# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for tree graph metadata and truthful execution outcomes.

These tests do not claim to exercise the Ascend graph runtime. Hardware graph
replay/numerical regression remains a separate end-to-end acceptance gate.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.spec_decode.pearl.native_graph import (
    NativeACLGraphRunner,
    bucket_tree_attention_metadata,
)
from vllm_ascend.spec_decode.pearl.native_model import (
    NativeAttention,
    NativeAttentionMetadata,
    NativeQwen2ForCausalLM,
)


def _metadata(lengths=(5, 6), *, mask_columns=32, block_count=8):
    rows = len(lengths)
    return NativeAttentionMetadata(
        slot_mapping=torch.arange(rows, dtype=torch.int32),
        context_lens=torch.tensor(lengths, dtype=torch.int32),
        block_tables=torch.arange(block_count, dtype=torch.int32).expand(rows, -1).clone(),
        actual_seq_lengths_q=(rows,),
        sequence_lens=(max(lengths),),
        attention_mask=torch.zeros((rows, mask_columns), dtype=torch.bool),
    )


def _runner(**kwargs):
    model = MagicMock()
    model.layers = [SimpleNamespace(self_attn=SimpleNamespace(block_size=4))]
    runner = NativeACLGraphRunner(model, enabled=False, **kwargs)
    runner.enabled = True
    runner.update_stream = MagicMock()
    return runner


def _target_call(runner, metadata):
    rows = metadata.context_lens.numel()
    return runner.run_target_greedy([torch.arange(rows)], [torch.arange(rows)], [metadata], vocabulary_size=32)


def test_tree_bucket_is_stable_across_growing_contexts_without_new_query_rows():
    first = bucket_tree_attention_metadata(_metadata((5, 6)), block_size=4)
    second = bucket_tree_attention_metadata(_metadata((6, 7)), block_size=4)
    assert first.context_lens.tolist() == second.context_lens.tolist() == [8, 8]
    assert first.slot_mapping.numel() == second.slot_mapping.numel() == 2
    assert first.attention_mask[0, 5:].all()
    assert not first.attention_mask[0, :5].any()
    assert second.attention_mask[0, 6:].all()
    assert not second.attention_mask[0, :6].any()


def test_tree_bucket_preserves_sibling_mask_and_sanitizes_unallocated_tail_pages():
    original = _metadata()
    original.attention_mask[1, 2] = True
    original.block_tables[:, 2:] = -1
    bucketed = bucket_tree_attention_metadata(original, block_size=4)
    assert bucketed.attention_mask[1, 2]
    assert bucketed.block_tables.min() == 0
    assert original.block_tables[0, 2] == -1
    assert original.context_lens.tolist() == [5, 6]


def test_tree_bucket_caps_non_power_of_two_context_capacity():
    metadata = _metadata((9, 10), mask_columns=10, block_count=3)
    assert bucket_tree_attention_metadata(metadata, block_size=4).context_lens.tolist() == [10, 10]


@pytest.mark.parametrize("lengths", [(0, 2), (5, 33)])
def test_tree_bucket_rejects_invalid_context_lengths(lengths):
    with pytest.raises(ValueError, match="context length"):
        bucket_tree_attention_metadata(_metadata(lengths), block_size=4)


def test_dense_bucket_matches_exact_context_with_nan_padding_and_masked_sibling():
    attention = NativeAttention.__new__(NativeAttention)
    torch.nn.Module.__init__(attention)
    attention.num_heads = 2
    attention.num_kv_heads = 1
    attention.scale = 0.5
    attention.block_size = 4
    attention.uses_paged_attention = False
    generator = torch.Generator().manual_seed(3)
    attention.key_cache = torch.randn(32, 1, 4, generator=generator)
    attention.value_cache = torch.randn(32, 1, 4, generator=generator)
    attention.key_cache[6:] = float("nan")
    attention.value_cache[6:] = float("nan")
    attention.key_cache[2] = float("nan")
    attention.value_cache[2] = float("nan")
    query = torch.randn(2, 2, 4, generator=generator)
    metadata = _metadata()
    metadata.attention_mask[:, 2] = True
    eager = attention._dense_attention(query, metadata)
    bucketed = attention._dense_attention(query, bucket_tree_attention_metadata(metadata, block_size=4))
    assert torch.isfinite(eager).all()
    assert torch.isfinite(bucketed).all()
    torch.testing.assert_close(bucketed, eager, rtol=1e-6, atol=1e-6)


def test_bfloat16_tree_attention_accumulates_float32_across_holes_and_bucket_tail():
    attention = NativeAttention.__new__(NativeAttention)
    torch.nn.Module.__init__(attention)
    attention.num_heads = 4
    attention.num_kv_heads = 2
    attention.scale = 0.125
    attention.block_size = 128
    attention.uses_paged_attention = False
    generator = torch.Generator().manual_seed(43)
    attention.key_cache = torch.randn(512, 2, 64, generator=generator).bfloat16()
    attention.value_cache = torch.randn(512, 2, 64, generator=generator).bfloat16()
    query = torch.randn(2, 4, 64, generator=generator).bfloat16()
    metadata = _metadata((259, 260), mask_columns=512, block_count=4)
    metadata.attention_mask[:, [128, 254, 255, 257]] = True
    attention.key_cache[260:] = float("nan")
    attention.value_cache[260:] = float("nan")
    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    with patch("torch.nn.functional.scaled_dot_product_attention", wraps=original_sdpa) as sdpa:
        eager = attention._dense_attention(query, metadata)
        bucketed = attention._dense_attention(query, bucket_tree_attention_metadata(metadata, block_size=128))
        assert all(
            call.args[0].dtype == call.args[1].dtype == call.args[2].dtype == torch.float32
            for call in sdpa.call_args_list
        )
    assert eager.dtype == bucketed.dtype == torch.bfloat16
    assert torch.isfinite(bucketed).all()
    torch.testing.assert_close(bucketed, eager, rtol=1e-3, atol=1e-3)


def test_target_graph_key_reuses_context_bucket_and_keeps_original_reference_metadata():
    runner = _runner()
    runner._capture_target = MagicMock(return_value=(torch.tensor([1, 2]),))
    first = _metadata((5, 6))
    second = _metadata((6, 7))
    third = _metadata((8, 9))
    _target_call(runner, first)
    _target_call(runner, second)
    _target_call(runner, third)
    calls = runner._capture_target.call_args_list
    assert calls[0].args[0] == calls[1].args[0]
    assert calls[1].args[0] != calls[2].args[0]
    assert calls[0].kwargs["reference_metadatas"][0] is first
    assert calls[0].args[3][0].context_lens.tolist() == [8, 8]


def test_target_graph_reports_capacity_fallback_not_graph_enabled():
    runner = _runner(max_graph_entries=1)
    runner.capture_attempt_count = 1
    runner._capture_budget_used = 1
    runner._execute_target = MagicMock(return_value=(torch.tensor([1, 2]),))
    original = _metadata()
    _target_call(runner, original)
    assert not runner.last_target_execution.used_aclgraph
    assert runner.last_target_execution.fallback_reason == "entry_capacity"
    assert not runner.last_target_execution.replay_executed
    assert runner._execute_target.call_args.args[2][0] is original


def test_target_graph_enforces_maximum_packed_token_count():
    runner = _runner(max_graph_tokens=1)
    runner._execute_target = MagicMock(return_value=(torch.tensor([1, 2]),))
    _target_call(runner, _metadata())
    assert runner.last_target_execution.fallback_reason == "token_capacity"
    assert runner.capture_attempt_count == 0


def test_target_graph_reports_disabled_execution():
    runner = _runner()
    runner.enabled = False
    runner._execute_target = MagicMock(return_value=(torch.tensor([1, 2]),))
    _target_call(runner, _metadata())
    assert runner.last_target_execution.mode == "eager"
    assert runner.last_target_execution.fallback_reason == "disabled"


def test_tree_logits_and_hidden_graph_interfaces_preserve_requested_outputs():
    runner = _runner()
    runner.enabled = False
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    logits = torch.arange(12, dtype=torch.float32).view(2, 6)
    runner.model.return_value = hidden
    runner.model.compute_logits.return_value = logits
    inputs = torch.tensor([1, 2])
    positions = torch.tensor([4, 5])
    actual = runner.run_tree_logits(inputs, positions, _metadata(), vocabulary_size=4)
    torch.testing.assert_close(actual, logits[:, :4])
    runner.model.compute_greedy_tokens.assert_not_called()
    runner.model.compute_logits.reset_mock()
    torch.testing.assert_close(runner.run_tree_hidden(inputs, positions, _metadata()), hidden)
    runner.model.compute_logits.assert_not_called()


def test_tree_float_graph_validation_rejects_wrong_output_shape():
    assert not NativeACLGraphRunner._target_outputs_match((torch.ones(2, 3),), (torch.ones(2, 4),))


def test_target_graph_reports_capture_then_real_replay_and_copies_new_mask():
    runner = _runner()
    runner._execute_target = MagicMock(return_value=(torch.tensor([1, 2]),))
    runner._update_target_attention_tasks = MagicMock()
    with (
        patch("torch.npu.NPUGraph", return_value=MagicMock()),
        patch("torch.npu.graph", return_value=MagicMock()),
        patch("torch.npu.current_stream", return_value=MagicMock()),
        patch("torch.npu.synchronize"),
    ):
        _target_call(runner, _metadata((5, 6)))
        assert runner.last_target_execution.mode == "capture_replay"
        assert runner.last_target_execution.capture_attempted
        assert runner.last_target_execution.used_aclgraph
        _target_call(runner, _metadata((6, 7)))
        assert runner.last_target_execution.mode == "replay"
        assert runner.last_target_execution.replay_executed
        assert not runner.last_target_execution.capture_attempted
        entry = next(iter(runner.target_entries.values()))
        assert not entry.attention_masks[0][0, 5]
        assert entry.attention_masks[0][0, 6]
        assert runner.capture_count == 1
        assert runner.replay_count == 2


def test_target_graph_validation_failure_reports_eager_return_after_replay():
    runner = _runner()
    runner._execute_target = MagicMock(
        side_effect=[
            (torch.tensor([1, 2]),),
            (torch.tensor([9, 9]),),
            (torch.tensor([9, 9]),),
            (torch.tensor([1, 2]),),
        ]
    )
    runner._update_target_attention_tasks = MagicMock()
    with (
        patch("torch.npu.NPUGraph", return_value=MagicMock()),
        patch("torch.npu.graph", return_value=MagicMock()),
        patch("torch.npu.current_stream", return_value=MagicMock()),
        patch("torch.npu.synchronize"),
        pytest.warns(RuntimeWarning, match="first replay"),
    ):
        output = _target_call(runner, _metadata())
    assert output[0].tolist() == [1, 2]
    assert runner.last_target_execution.replay_executed
    assert runner.last_target_execution.capture_attempted
    assert not runner.last_target_execution.used_aclgraph
    assert runner.last_target_execution.fallback_reason == "capture_validation"
    assert not runner.target_entries
    diagnostic = runner.last_target_validation_error
    assert diagnostic["eager_kv_restored"]
    assert diagnostic["graph_vs_same_bucket_eager"][0]["max_abs_error"] == 0
    assert diagnostic["bucket_eager_vs_original"][0]["mismatch_elements"] == 2
    assert diagnostic["restored_eager_vs_original"][0]["max_abs_error"] == 0
    assert runner._execute_target.call_args.args[2][0].context_lens.tolist() == [5, 6]


def test_tree_graph_failure_diagnostic_does_not_relax_float_tolerance():
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    perturbed = original.clone()
    perturbed[0, 0] += 0.003
    assert not NativeACLGraphRunner._target_outputs_match((perturbed,), (original,))
    summary = NativeACLGraphRunner._target_difference_summary((perturbed,), (original,))[0]
    assert summary["mismatch_elements"] == 1
    assert summary["per_row_max_abs_error_first32"][1] == 0
    assert summary["nonfinite_output"] == 0


def test_tree_graph_failure_diagnostic_identifies_shape_and_nonfinite_errors():
    mismatched = NativeACLGraphRunner._target_difference_summary((torch.ones(2),), (torch.ones(3),))[0]
    assert mismatched["shape"] != mismatched["reference_shape"]
    nonfinite = NativeACLGraphRunner._target_difference_summary((torch.tensor([float("nan"), 1.0]),), (torch.ones(2),))[
        0
    ]
    assert nonfinite["nonfinite_output"] == 1
    assert nonfinite["mismatch_elements"] == 1


def test_tree_metadata_packs_selected_nodes_not_exploration_envelope():
    model = NativeQwen2ForCausalLM.__new__(NativeQwen2ForCausalLM)
    model.embed_tokens = SimpleNamespace(weight=torch.empty(1))
    model.layers = [SimpleNamespace(self_attn=SimpleNamespace(block_size=4))]
    model.max_model_len = 16
    plan = SimpleNamespace(
        width=3,
        depth=3,
        parent_indices=torch.tensor([-1, 0]),
        positions=torch.tensor([2, 3, 4]),
        cache_positions=torch.tensor([2, 3, 4]),
        attention_mask=torch.ones((3, 16), dtype=torch.bool),
    )
    inputs, _, metadata = model.make_tree_attention_metadata([plan], [1], [[2, 3]], [[0, 1, 2, 3]])
    assert inputs.tolist() == [1, 2, 3]
    assert metadata.slot_mapping.numel() == 3
    assert metadata.actual_seq_lengths_q == (3,)
