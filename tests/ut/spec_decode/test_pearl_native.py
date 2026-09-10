# SPDX-License-Identifier: Apache-2.0

import os
from multiprocessing import Pipe
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from examples.benchmark_nano_pearl_speculative import (
    _aggregate_decode_profile,
    _parse_target_graph_post_counts,
    _worker_aclgraph_deltas,
)
from examples.benchmark_nano_pearl_speculative import (
    _build_parser as _build_benchmark_parser,
)
from examples.benchmark_nano_pearl_target_only import (
    _build_parser as _build_target_only_benchmark_parser,
)
from vllm_ascend.spec_decode.pearl.api import PEARLConfig, PEARLEngine
from vllm_ascend.spec_decode.pearl.native_cache import NativePrefixCache
from vllm_ascend.spec_decode.pearl.native_engine import (
    NativePearlConfig,
    NativePearlEngine,
    NativeSpecRhythmDevicePayload,
    PearlPipelineState,
    SamplingParams,
    _bucket_target_verification_widths,
    _bucket_variable_target_verification_widths,
    _build_greedy_verdict,
    _build_greedy_verdict_cpu,
    _build_greedy_verdict_with_layout,
    _build_parser,
    _build_stochastic_verdict,
    _build_verification_layout,
    _canonical_active_indices,
    _continuous_bucket_indices,
    _continuous_result_states,
    _finished,
    _gamma_from_decode_speeds,
    _normalize_sampling_params,
    _restore_completed_states,
    _sample_logits,
    _select_preemptive_continuous_indices,
    _set_default_npu_environment,
    _target_graph_precompile_shapes,
    _truncate_completion,
)
from vllm_ascend.spec_decode.pearl.native_graph import NativeACLGraphRunner
from vllm_ascend.spec_decode.pearl.native_model import (
    MIN_PAGED_ATTENTION_BLOCKS,
    PAGED_ATTENTION_BLOCK_SIZE,
    NativeAttention,
    NativeColumnLinear,
    NativeLMHead,
    NativeQwen2ForCausalLM,
    NativeRMSNorm,
    NativeRowLinear,
    NativeTPContext,
    _is_contiguous_linear_tree_plan,
    _maybe_convert_linear_weights_to_nz,
    _use_native_fused_mm_all_reduce,
    load_native_qwen2_weights,
    prepare_native_model_config,
)
from vllm_ascend.spec_decode.pearl.spec_rhythm import SpecRhythmProposalTicket
from vllm_ascend.spec_decode.pearl.topology import PearlTopology
from vllm_ascend.spec_decode.pearl.tree import build_tree_speculation_plan, pack_selected_tree_plan


def test_selected_tree_uses_causal_fast_path_only_for_a_contiguous_spine():
    tree = build_tree_speculation_plan(2, 2, 5, 32)
    assert not _is_contiguous_linear_tree_plan(tree)
    assert _is_contiguous_linear_tree_plan(pack_selected_tree_plan(tree, [0]))
    assert _is_contiguous_linear_tree_plan(pack_selected_tree_plan(tree, [0, 1]))
    assert not _is_contiguous_linear_tree_plan(pack_selected_tree_plan(tree, [0, 2]))


def test_pearl_defaults_to_tp3_deterministic_aiv_without_overriding_user_configuration():
    with patch.dict(os.environ, {}, clear=True):
        _set_default_npu_environment(target_tp_size=3)
        assert os.environ["TASK_QUEUE_ENABLE"] == "1"
        assert os.environ["HCCL_OP_EXPANSION_MODE"] == "AIV"
        assert os.environ["HCCL_DETERMINISTIC"] == "true"

        os.environ["HCCL_OP_EXPANSION_MODE"] = "user-mode"
        os.environ["HCCL_DETERMINISTIC"] = "false"
        os.environ["TASK_QUEUE_ENABLE"] = "0"
        _set_default_npu_environment(target_tp_size=3)
        assert os.environ["TASK_QUEUE_ENABLE"] == "0"
        assert os.environ["HCCL_OP_EXPANSION_MODE"] == "user-mode"
        assert os.environ["HCCL_DETERMINISTIC"] == "false"


def test_native_mm_allreduce_never_activates_from_symbol_presence_alone():
    with (
        patch("vllm_ascend.spec_decode.pearl.native_model.ascend_envs.VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE", False),
        patch(
            "vllm_ascend.spec_decode.pearl.native_model.ascend_envs.VLLM_ASCEND_PEARL_ENABLE_TP3_MM_ALL_REDUCE",
            False,
        ),
    ):
        assert not _use_native_fused_mm_all_reduce(2, "npu")
        assert not _use_native_fused_mm_all_reduce(3, "npu")
        assert not _use_native_fused_mm_all_reduce(4, "npu")
        assert not _use_native_fused_mm_all_reduce(2, "cpu")


def test_native_mm_allreduce_matches_explicit_vllm_and_tp3_opt_ins():
    with patch(
        "vllm_ascend.spec_decode.pearl.native_model.ascend_envs.VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE", True
    ):
        assert _use_native_fused_mm_all_reduce(2, "npu")
        assert _use_native_fused_mm_all_reduce(4, "npu")
        assert _use_native_fused_mm_all_reduce(8, "npu")
        assert not _use_native_fused_mm_all_reduce(3, "npu")
    with patch(
        "vllm_ascend.spec_decode.pearl.native_model.ascend_envs.VLLM_ASCEND_PEARL_ENABLE_TP3_MM_ALL_REDUCE", True
    ):
        assert _use_native_fused_mm_all_reduce(3, "npu")


def test_sampling_params_support_an_independent_draft_temperature():
    params = SamplingParams(
        temperature=0.0,
        draft_temperature=0.7,
        top_p=0.9,
        top_k=8,
        draft_top_p=0.8,
        draft_top_k=4,
    )
    assert params.temperature == 0.0
    assert params.draft_temperature == 0.7
    assert (params.top_p, params.top_k, params.draft_top_p, params.draft_top_k) == (0.9, 8, 0.8, 4)
    with pytest.raises(ValueError, match="temperatures"):
        SamplingParams(draft_temperature=-0.1)
    with pytest.raises(ValueError, match="top-p"):
        SamplingParams(top_p=0.0)
    with pytest.raises(ValueError, match="top-k"):
        SamplingParams(draft_top_k=-1)


def test_pearl_does_not_force_deterministic_hccl_for_power_of_two_target_tp():
    with patch.dict(os.environ, {}, clear=True):
        _set_default_npu_environment(target_tp_size=4)
        assert os.environ["TASK_QUEUE_ENABLE"] == "1"
        assert os.environ["HCCL_OP_EXPANSION_MODE"] == "AIV"
        assert "HCCL_DETERMINISTIC" not in os.environ


def test_active_requests_are_canonicalized_by_verification_width():
    states = [
        PearlPipelineState([1], prompt_length=1, pre_verify=True),
        PearlPipelineState([2], prompt_length=1, pre_verify=False),
        PearlPipelineState([3], prompt_length=1, pre_verify=True),
        PearlPipelineState([4], prompt_length=1, pre_verify=False),
    ]

    assert _canonical_active_indices(states, [0, 1, 2, 3]) == [1, 3, 0, 2]


def test_target_verification_widths_are_quantized_without_exposing_padding():
    widths, valid_indices = _bucket_target_verification_widths(
        [False, False, False, *([True] * 13)],
        gamma=4,
        num_buckets=8,
    )

    assert widths == [4, 4, 4, 4, *([1] * 12)]
    assert len(valid_indices) == 25
    assert valid_indices[:13] == list(range(13))
    assert valid_indices[13:] == list(range(16, 28))


def test_variable_target_widths_retain_only_real_candidate_rows():
    widths, valid_indices = _bucket_variable_target_verification_widths([4, 2, 1, 1], gamma=5, num_buckets=2)
    assert widths == [5, 5, 1, 1]
    assert valid_indices == [0, 1, 2, 3, 5, 6, 10, 11]


def test_target_graph_precompile_shapes_include_nondivisible_maximum():
    shapes = _target_graph_precompile_shapes([5], gamma=4, num_buckets=2)

    assert shapes == [(5, 5, 0), (14, 5, 3), (20, 5, 5)]


def test_target_verification_widths_use_explicit_hotspot_buckets():
    widths, valid_indices = _bucket_target_verification_widths(
        [False, False, False, False, True],
        gamma=4,
        num_buckets=2,
        configured_post_counts=((5, (0, 3, 5)),),
    )

    assert widths == [4, 4, 4, 4, 4]
    assert valid_indices == [*range(16), 16]


def test_target_graph_precompile_shapes_use_explicit_hotspot_buckets():
    shapes = _target_graph_precompile_shapes(
        [5],
        gamma=4,
        num_buckets=2,
        configured_post_counts=((5, (0, 2, 5)),),
    )

    assert shapes == [(5, 5, 0), (11, 5, 2), (20, 5, 5)]


def test_target_graph_post_counts_reject_invalid_ranges():
    with pytest.raises(ValueError, match="strictly increasing"):
        NativePearlConfig(
            "draft",
            "target",
            1,
            2,
            4,
            512,
            32,
            target_verification_graph_post_counts=((8, (0, 4, 7)),),
        )


def test_benchmark_parses_explicit_target_graph_post_counts():
    assert _parse_target_graph_post_counts(["128:0,26,75,128", "64:0,47,64"]) == (
        (128, (0, 26, 75, 128)),
        (64, (0, 47, 64)),
    )


def test_target_follower_builds_replicated_greedy_result_without_broadcast():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.rank = 2
    engine.gamma = 3
    engine.device = torch.device("cpu")
    engine.topology = PearlTopology(draft_ranks=(0,), target_ranks=(1, 2))
    engine.groups = SimpleNamespace(correction_group=MagicMock())
    verdict = torch.tensor([[2, -1], [0, 42]])
    draft_message = torch.tensor([10, 11, 12, 20, 21, 22, 30, 31, 32])

    with patch("vllm_ascend.spec_decode.pearl.native_engine.dist.broadcast") as broadcast:
        accepted, corrections, next_windows = engine._broadcast_round_result(
            verdict,
            draft_message,
            verification_size=3,
            batch_size=2,
            replicated_target_verdict=True,
        )

    broadcast.assert_not_called()
    assert accepted == [2, 0]
    assert corrections == [None, 42]
    assert next_windows == [[20, 21, 22], [30, 31, 32]]


def test_tree_verdict_target_leader_broadcasts_to_target_and_draft_groups():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.rank = 1
    engine.device = torch.device("cpu")
    engine.topology = PearlTopology(draft_ranks=(0,), target_ranks=(1, 2))
    engine.groups = SimpleNamespace(
        target_group=object(),
        correction_group=object(),
    )
    plan = build_tree_speculation_plan(
        width=2,
        depth=2,
        prefix_len=1,
        max_model_len=8,
        candidate_budget=3,
    )
    output = SimpleNamespace(
        token_ids=torch.tensor([[10, 11, 12]]),
        accepted_node_indices=torch.tensor([[0, -1]]),
    )

    with patch("vllm_ascend.spec_decode.pearl.native_engine.dist.broadcast") as broadcast:
        tokens, accepted = engine._broadcast_spec_rhythm_tree_verdict(output, [plan])

    assert tokens == [[10, 11, 12]]
    assert accepted == [[0, -1]]
    assert [call.kwargs["group"] for call in broadcast.call_args_list] == [
        engine.groups.target_group,
        engine.groups.correction_group,
    ]


def test_static_padding_restores_completed_rows_without_growing_their_state():
    completed = PearlPipelineState(
        [10, 20, 21],
        prompt_length=1,
        max_tokens=2,
        ignore_eos=True,
    )
    active = PearlPipelineState(
        [10, 30],
        prompt_length=1,
        max_tokens=3,
        ignore_eos=True,
    )
    states = [completed, active]
    snapshots: dict[int, PearlPipelineState] = {}

    _restore_completed_states(states, snapshots, frozenset())
    assert snapshots[0].token_ids == [10, 20, 21]

    states[0].token_ids.extend([22, 23, 24, 25])
    states[0].committed_length = len(states[0].token_ids)
    _restore_completed_states(states, snapshots, frozenset())

    assert states[0].token_ids == [10, 20, 21]
    assert states[0] is not snapshots[0]
    assert states[1] is active


def test_continuous_tail_uses_only_full_and_half_batch_graph_buckets():
    completed = {index: PearlPipelineState([index], prompt_length=1) for index in range(64)}

    assert len(_continuous_bucket_indices(list(range(40)), completed, 64)) == 64
    assert len(_continuous_bucket_indices(list(range(20)), completed, 64)) == 32
    assert _continuous_bucket_indices([], completed, 64) == []


def test_preemptive_continuous_scheduling_explores_least_served_requests():
    states = [PearlPipelineState([index], prompt_length=1) for index in range(6)]
    for state, rounds in zip(states, [2, 0, 1, 0, 3, 1]):
        state.verification_rounds = rounds

    assert _select_preemptive_continuous_indices(states, [0, 1, 2, 3, 4, 5], 4) == [1, 3, 2, 5]


def test_preemptive_continuous_scheduling_prioritizes_estimated_remaining_work():
    slow = PearlPipelineState([0, *range(17)], prompt_length=1, max_tokens=100)
    fast = PearlPipelineState([0, *range(33)], prompt_length=1, max_tokens=100)
    slow.verification_rounds = fast.verification_rounds = 8

    assert _select_preemptive_continuous_indices([slow, fast], [0, 1], 1) == [0]


def test_preemptive_continuous_scheduling_uses_recent_slowdown():
    first = PearlPipelineState([0, *range(33)], prompt_length=1, max_tokens=100)
    second = PearlPipelineState([0, *range(25)], prompt_length=1, max_tokens=100)
    first.verification_rounds = second.verification_rounds = 8

    selected = _select_preemptive_continuous_indices(
        [first, second],
        [0, 1],
        1,
        recent_token_gains=[[1, 1, 1, 1], [3, 3, 3, 3]],
    )

    assert selected == [0]


def test_spec_rhythm_preemptive_scheduling_prioritizes_tpot_debt():
    relaxed = PearlPipelineState([0, *range(8)], prompt_length=1, max_tokens=32, slo_tpot_ms=100.0)
    urgent = PearlPipelineState([0, *range(8)], prompt_length=1, max_tokens=32, slo_tpot_ms=10.0)
    relaxed.verification_rounds = urgent.verification_rounds = 8

    selected = _select_preemptive_continuous_indices(
        [relaxed, urgent],
        [0, 1],
        1,
        decode_elapsed_ms=500.0,
        slo_aware=True,
    )

    assert selected == [1]


def test_sampling_params_validate_optional_tpot_slo():
    params = SamplingParams(temperature=0.0, max_tokens=4, slo_tpot_ms=25.0, slo_class="tight")
    assert params.slo_tpot_ms == 25.0
    assert params.slo_class == "tight"
    with pytest.raises(ValueError, match="TPOT SLO"):
        SamplingParams(slo_tpot_ms=0.0)


def test_preverify_acceptance_keeps_draft_and_target_states_in_sync():
    target = PearlPipelineState([1, 2, 3], prompt_length=2)
    draft = target.clone()
    next_window = [4, 5, 6, 7]
    draft.token_ids.extend(next_window)

    draft.apply_draft_verification(
        gamma=4,
        accepted=1,
        correction_token_id=None,
        next_round_token_ids=next_window,
    )
    target.apply_target_verification(
        gamma=4,
        accepted=1,
        correction_token_id=None,
        next_round_token_ids=next_window,
    )

    assert draft.token_ids == target.token_ids == [1, 2, 3, 4, 5, 6, 7]
    assert draft.committed_completion_token_ids == target.committed_completion_token_ids == [3, 4]
    assert not draft.pre_verify
    assert draft.accepted_draft_tokens == 1
    assert draft.verified_draft_tokens == 1
    assert draft.verification_rounds == target.verification_rounds == 1


def test_postverify_rejection_rolls_back_the_same_pipeline_suffix_on_both_sides():
    target = PearlPipelineState(
        [1, 2, 3, 4, 5],
        prompt_length=2,
        pre_verify=False,
        committed_length=2,
    )
    draft = target.clone()
    next_window = [6, 7, 8, 9]
    draft.token_ids.extend(next_window)

    draft.apply_draft_verification(
        gamma=4,
        accepted=2,
        correction_token_id=42,
        next_round_token_ids=next_window,
    )
    target.apply_target_verification(
        gamma=4,
        accepted=2,
        correction_token_id=42,
        next_round_token_ids=next_window,
    )

    assert draft.token_ids == target.token_ids == [1, 2, 3, 4, 42]
    assert draft.committed_completion_token_ids == target.committed_completion_token_ids == [3, 4, 42]
    assert draft.pre_verify
    assert draft.accepted_draft_tokens == 2
    assert draft.verified_draft_tokens == 4
    assert draft.verification_rounds == target.verification_rounds == 1


def test_variable_gamma_transition_tracks_current_and_next_window_sizes():
    target = PearlPipelineState([1, 2, 3], prompt_length=2)
    draft = target.clone()
    first_window = [4, 5]
    draft.token_ids.extend(first_window)
    for state, apply in (
        (draft, draft.apply_draft_verification),
        (target, target.apply_target_verification),
    ):
        apply(
            gamma=5,
            accepted=1,
            correction_token_id=None,
            next_round_token_ids=first_window,
            verification_size=1,
        )
        assert state.pending_window_size == 2

    second_window = [6, 7, 8, 9, 10]
    draft.token_ids.extend(second_window)
    draft.apply_draft_verification(
        gamma=5,
        accepted=1,
        correction_token_id=42,
        next_round_token_ids=second_window,
        verification_size=2,
    )
    target.apply_target_verification(
        gamma=5,
        accepted=1,
        correction_token_id=42,
        next_round_token_ids=second_window,
        verification_size=2,
    )

    assert draft.token_ids == target.token_ids == [1, 2, 3, 4, 5, 42]
    assert draft.committed_length == target.committed_length == 6
    assert draft.pending_window_size == target.pending_window_size == 0
    assert draft.continuation_epoch == target.continuation_epoch == 2


def test_rolling_eager_suffix_is_guarded_until_parent_verification():
    draft = PearlPipelineState(
        [1, 2, 3, 4, 5],
        prompt_length=2,
        pre_verify=False,
        committed_length=4,
        pending_window_size=2,
    )
    target = draft.clone()
    next_window = [6, 7, 8]
    eager_window = [9, 10]
    draft.token_ids.extend(next_window)
    draft.token_ids.extend(eager_window)

    # A rejection invalidates the eager continuation before the ordinary
    # PEARL rollback consumes the current next-window suffix.
    del draft.token_ids[-len(eager_window) :]
    draft.apply_draft_verification(
        gamma=4,
        accepted=1,
        correction_token_id=42,
        next_round_token_ids=next_window,
        verification_size=2,
    )
    target.apply_target_verification(
        gamma=4,
        accepted=1,
        correction_token_id=42,
        next_round_token_ids=next_window,
        verification_size=2,
    )
    assert draft.token_ids == target.token_ids
    assert draft.committed_length == target.committed_length


def test_device_mailbox_validates_prefix_epoch_and_variable_shapes():
    state = PearlPipelineState([1, 2], prompt_length=1)
    ticket = SpecRhythmProposalTicket(
        proposal_id=3,
        request_index=0,
        home_batch_id=0,
        gamma=2,
        required_prefix_epoch=0,
    )
    payload = NativeSpecRhythmDevicePayload(
        ticket=ticket,
        verification_tokens=torch.tensor([4]),
        next_tokens=torch.tensor([4, 5]),
        verification_size=1,
        draft_confidence=0.8,
    )
    payload.validate_for(state)
    state.continuation_epoch = 1
    with pytest.raises(RuntimeError, match="stale device mailbox"):
        payload.validate_for(state)

    bad_width = NativeSpecRhythmDevicePayload(
        ticket=ticket,
        verification_tokens=torch.tensor([4, 5, 6]),
        next_tokens=torch.tensor([4, 5]),
        verification_size=3,
        draft_confidence=0.8,
    )
    state.continuation_epoch = 0
    with pytest.raises(RuntimeError, match="does not match"):
        bad_width.validate_for(state)


def test_native_qwen2_model_runs_with_a_single_tensor_parallel_rank_on_cpu():
    config = SimpleNamespace(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        rope_theta=10_000.0,
        rms_norm_eps=1e-6,
        intermediate_size=32,
        tie_word_embeddings=False,
        num_hidden_layers=1,
    )
    model = NativeQwen2ForCausalLM(
        config,
        NativeTPContext(group=None, rank=0, size=1, leader_rank=0),
    )
    # Native parameters are torch.empty until checkpoint loading. This unit
    # test has no checkpoint; initialize them explicitly instead of depending
    # on allocator contents left by previous tests (which may contain NaNs).
    generator = torch.Generator().manual_seed(123)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.05, 0.05, generator=generator)
    model.configure_cache(16)

    hidden_states = model(torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2]))
    logits = model.compute_logits(hidden_states)
    greedy_tokens = model.compute_greedy_tokens(hidden_states, vocabulary_size=31)
    confidence_tokens, confidences = model.compute_greedy_tokens_with_confidence(hidden_states, vocabulary_size=31)

    assert hidden_states.shape == (3, 16)
    assert logits.shape == (3, 32)
    assert torch.equal(greedy_tokens, logits[:, :31].argmax(dim=-1))
    assert torch.equal(confidence_tokens, greedy_tokens)
    expected_confidences = torch.softmax(logits[:, :31].float(), dim=-1).max(dim=-1).values
    assert torch.allclose(confidences, expected_confidences)


@pytest.mark.parametrize("architecture", ["Qwen2ForCausalLM", "Qwen3ForCausalLM", "LlamaForCausalLM"])
def test_native_model_runs_every_upstream_architecture_on_cpu(architecture):
    config = SimpleNamespace(
        architectures=[architecture],
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        rope_theta=10_000.0,
        rope_parameters={"rope_theta": 10_000.0, "rope_type": "default"},
        rms_norm_eps=1e-6,
        intermediate_size=32,
        hidden_act="silu",
        attention_bias=architecture == "LlamaForCausalLM",
        mlp_bias=architecture == "LlamaForCausalLM",
        tie_word_embeddings=False,
        num_hidden_layers=1,
    )
    model = NativeQwen2ForCausalLM(
        config,
        NativeTPContext(group=None, rank=0, size=1, leader_rank=0),
    )
    model.configure_cache(16)

    hidden_states = model(torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2]))

    assert hidden_states.shape == (3, 16)
    attention = model.layers[0].self_attn
    assert isinstance(attention.q_norm, NativeRMSNorm) == (architecture == "Qwen3ForCausalLM")
    assert (attention.o_proj.bias is not None) == (architecture == "LlamaForCausalLM")
    assert (model.layers[0].mlp.gate_up_proj.bias is not None) == (architecture == "LlamaForCausalLM")


def test_native_qwen3_attention_admits_production_qknorm_rope_fusion():
    config = SimpleNamespace(
        architectures=["Qwen3ForCausalLM"],
        hidden_size=256,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=32,
        rope_theta=10_000.0,
        rope_parameters={"rope_theta": 10_000.0, "rope_type": "default"},
        rms_norm_eps=1e-6,
    )

    attention = NativeAttention(
        config,
        NativeTPContext(group=None, rank=0, size=1, leader_rank=0),
    )

    assert attention.use_qknorm_rope_fusion == hasattr(torch.ops.vllm, "qkv_rmsnorm_rope")
    assert attention.rotary_emb.cos_sin_cache.shape == (32, 128)


def test_dynamic_tp_config_matches_upstream_padding_rules():
    config = SimpleNamespace(
        architectures=["Qwen2ForCausalLM"],
        vocab_size=101,
        hidden_size=1280,
        num_attention_heads=10,
        num_key_value_heads=2,
        intermediate_size=1000,
    )

    prepared = prepare_native_model_config(config, tensor_parallel_size=3)

    assert prepared is not config
    assert prepared.head_dim == 128
    assert prepared.num_attention_heads == 15
    assert prepared.num_key_value_heads == 3
    assert prepared.intermediate_size == 1152
    assert prepared.vocab_size == 102
    assert prepared.valid_vocab_size == 101


def test_dynamic_tp_weight_loaders_zero_pad_the_final_partition():
    context = NativeTPContext(group=None, rank=2, size=3, leader_rank=0)
    column = NativeColumnLinear(2, 12, context)
    row = NativeRowLinear(12, 2, context)
    loaded_column = torch.arange(20, dtype=torch.float32).view(10, 2)
    loaded_row = torch.arange(20, dtype=torch.float32).view(2, 10)

    column.load_weight(loaded_column)
    row.load_weight(loaded_row)

    assert torch.equal(column.weight[:2], loaded_column[8:10])
    assert torch.count_nonzero(column.weight[2:]) == 0
    assert torch.equal(row.weight[:, :2], loaded_row[:, 8:10])
    assert torch.count_nonzero(row.weight[:, 2:]) == 0


def test_native_attention_allocates_vllm_compatible_paged_cache():
    config = SimpleNamespace(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        rope_theta=10_000.0,
    )
    attention = NativeAttention(
        config,
        NativeTPContext(group=None, rank=0, size=1, leader_rank=0),
    )

    attention.configure_cache(PAGED_ATTENTION_BLOCK_SIZE + 1, use_paged_attention=True)

    assert attention.uses_paged_attention
    assert attention.key_cache is not None
    assert attention.key_cache.shape == (
        MIN_PAGED_ATTENTION_BLOCKS,
        PAGED_ATTENTION_BLOCK_SIZE,
        1,
        8,
    )
    assert attention.block_table is not None
    assert attention.block_table.dtype == torch.int32
    assert attention.block_table.tolist() == [[0, 1]]
    assert attention.context_lens is not None
    assert attention.context_lens.device.type == "cpu"


def test_native_attention_writes_a_packed_gqa_batch_with_one_cann_call():
    config = SimpleNamespace(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        rope_theta=10_000.0,
    )
    attention = NativeAttention(
        config,
        NativeTPContext(group=None, rank=0, size=1, leader_rank=0),
    )
    attention.configure_cache(16, use_paged_attention=True)
    packed_qkv = torch.randn(4, 32)
    key = packed_qkv[:, 16:24].view(4, 1, 8)
    value = packed_qkv[:, 24:].view(4, 1, 8)

    with patch("vllm_ascend.spec_decode.pearl.native_model.DeviceOperator.reshape_and_cache") as cache_op:
        attention._write_to_cache(torch.arange(4), key, value)

    cache_op.assert_called_once()
    assert cache_op.call_args.kwargs["key"] is key
    assert cache_op.call_args.kwargs["value"] is value
    assert cache_op.call_args.kwargs["slot_mapping"].dtype == torch.int32


def test_native_lm_head_greedy_uses_one_tensor_parallel_collective():
    context = NativeTPContext(group=MagicMock(), rank=0, size=2, leader_rank=0)
    lm_head = NativeLMHead(vocab_size=8, hidden_size=2, context=context)
    lm_head.weight.data.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]]))

    def copy_local_candidate(outputs, candidate, **_kwargs):
        outputs[0].copy_(candidate)
        outputs[1].copy_(candidate)

    with patch(
        "vllm_ascend.spec_decode.pearl.native_model.dist.all_gather",
        side_effect=copy_local_candidate,
    ) as all_gather:
        tokens = lm_head.greedy(torch.tensor([[1.0, 0.0]]), vocabulary_size=8)

    assert tokens.tolist() == [2]
    all_gather.assert_called_once()


def test_native_model_builds_disjoint_paged_metadata_for_a_static_batch():
    config = SimpleNamespace(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=512,
        rope_theta=10_000.0,
        rms_norm_eps=1e-6,
        intermediate_size=32,
        tie_word_embeddings=False,
        num_hidden_layers=1,
    )
    model = NativeQwen2ForCausalLM(
        config,
        NativeTPContext(group=None, rank=0, size=1, leader_rank=0),
    )
    model.configure_cache(PAGED_ATTENTION_BLOCK_SIZE + 1, max_num_seqs=2)

    positions, metadata = model.make_attention_metadata([0, 1, 1], [128, 0, 128])

    assert positions.tolist() == [128, 0, 128]
    assert metadata.slot_mapping.tolist() == [128, 256, 384]
    assert metadata.context_lens.tolist() == [129, 1, 129]
    assert metadata.context_lens.device.type == "cpu"
    assert metadata.block_tables.tolist() == [[0, 1], [2, 3], [2, 3]]
    assert metadata.actual_seq_lengths_q == (1, 3)
    assert metadata.sequence_lens == (129, 129)
    assert metadata.request_block_tables is None

    _, fia_metadata = model.make_attention_metadata(
        [0, 1, 1],
        [128, 0, 128],
        use_fused_infer_attention=True,
    )
    assert fia_metadata.request_block_tables.tolist() == [[0, 1], [2, 3]]
    assert fia_metadata.block_tables is fia_metadata.request_block_tables

    _, remapped = model.make_attention_metadata(
        [0, 1],
        [128, 0],
        block_tables=[[4, 6], [3, 5]],
    )
    assert remapped.slot_mapping.tolist() == [6 * PAGED_ATTENTION_BLOCK_SIZE, 3 * PAGED_ATTENTION_BLOCK_SIZE]
    assert remapped.block_tables.tolist() == [[4, 6], [3, 5]]

    _, direct_slots = model.make_attention_metadata(
        [0, 1],
        [128, 0],
        block_tables=[[4, 6], [3, 5]],
        slot_mapping=[777, 888],
    )
    assert direct_slots.slot_mapping.tolist() == [777, 888]


def test_fia_graph_input_copy_skips_unused_token_block_tables():
    entry = SimpleNamespace(
        input_ids=torch.empty(2, dtype=torch.long),
        positions=torch.empty(2, dtype=torch.long),
        slot_mapping=torch.empty(2, dtype=torch.long),
        context_lens=MagicMock(),
        block_tables=MagicMock(),
        request_block_tables=torch.empty((2, 2), dtype=torch.int32),
        actual_seq_lengths_q=(),
        sequence_lens=(),
    )
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([1, 2]),
        context_lens=torch.tensor([4, 5]),
        block_tables=torch.tensor([[0, 1], [2, 3]]),
        request_block_tables=torch.tensor([[4, 5], [6, 7]], dtype=torch.int32),
        actual_seq_lengths_q=(1, 2),
        sequence_lens=(4, 5),
    )

    NativeACLGraphRunner._copy_inputs(
        entry,
        torch.tensor([10, 11]),
        torch.tensor([3, 4]),
        metadata,
    )

    entry.context_lens.copy_.assert_not_called()
    entry.block_tables.copy_.assert_not_called()
    assert torch.equal(entry.request_block_tables, metadata.request_block_tables)
    assert entry.actual_seq_lengths_q == (1, 2)
    assert entry.sequence_lens == (4, 5)


def test_paged_draft_graph_input_copy_updates_context_and_token_block_tables():
    entry = SimpleNamespace(
        input_ids=torch.empty(2, dtype=torch.long),
        positions=(torch.empty(2, dtype=torch.long),),
        slot_mappings=(torch.empty(2, dtype=torch.long),),
        context_lens=(torch.empty(2, dtype=torch.int32),),
        block_tables=(torch.empty((2, 2), dtype=torch.int32),),
        request_block_tables=(None,),
        actual_seq_lengths_q=(),
        sequence_lens=(),
    )
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([1, 2]),
        context_lens=torch.tensor([4, 5], dtype=torch.int32),
        block_tables=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        request_block_tables=torch.tensor([[4, 5], [6, 7]], dtype=torch.int32),
        actual_seq_lengths_q=(1, 2),
        sequence_lens=(4, 5),
    )

    NativeACLGraphRunner._copy_draft_inputs(
        entry,
        torch.tensor([10, 11]),
        [torch.tensor([3, 4])],
        [metadata],
    )

    assert torch.equal(entry.context_lens[0], metadata.context_lens)
    assert torch.equal(entry.block_tables[0], metadata.block_tables)
    assert entry.actual_seq_lengths_q == ((1, 2),)
    assert entry.sequence_lens == ((4, 5),)


def test_native_prefix_cache_shares_full_prompt_blocks_within_and_across_batches():
    cache = NativePrefixCache(num_blocks=8, blocks_per_sequence=2, block_size=4)
    shared_prefix = [1, 2, 3, 4]

    first = cache.allocate([shared_prefix + [5], shared_prefix + [9]])

    assert first.num_cached_tokens == [0, 4]
    assert first.block_tables[0][0] == first.block_tables[1][0]
    cache.release()

    second = cache.allocate([shared_prefix + [7]])

    assert second.num_cached_tokens == [4]
    assert second.block_tables[0][0] == first.block_tables[0][0]
    cache.release()


def test_native_prefix_cache_allocates_decode_pages_lazily():
    cache = NativePrefixCache(num_blocks=3, blocks_per_sequence=4, block_size=4)
    allocation = cache.allocate([[1], [2]])

    assert allocation.block_tables == [[0, -1, -1, -1], [1, -1, -1, -1]]
    updates = cache.ensure_capacity([0], [4])
    assert updates == [(0, 1, 2)]
    assert allocation.block_tables[0][1] == 2
    with pytest.raises(RuntimeError, match="no free physical blocks"):
        cache.ensure_capacity([1], [4])
    cache.release()


def test_native_prefix_cache_activates_reserved_live_sequence():
    cache = NativePrefixCache(num_blocks=6, blocks_per_sequence=3, block_size=4)
    allocation = cache.allocate([[1]], sequence_capacity=3)

    assert allocation.block_tables == [
        [0, -1, -1],
        [-1, -1, -1],
        [-1, -1, -1],
    ]
    cached_tokens = cache.activate_sequence(2, [2, 3, 4, 5, 6])

    assert cached_tokens == 0
    assert allocation.block_tables[2][:2] == [1, 2]
    with pytest.raises(RuntimeError, match="activated twice"):
        cache.activate_sequence(2, [7])
    cache.release()


def test_target_ar_draft_rank_returns_without_allocating_or_running_the_model():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.config = SimpleNamespace(
        max_num_seqs=1,
        max_num_batched_tokens=16,
        max_tokens=4,
        max_model_len=16,
    )
    engine.is_draft = True
    engine._allocate_cache = MagicMock()

    result = engine.generate_target_ar_batch(
        [[1, 2]],
        SamplingParams(temperature=0, max_tokens=4),
    )

    assert result is None
    engine._allocate_cache.assert_not_called()


def test_target_ar_prefill_preserves_global_sequence_ids_across_chunks():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.cache_allocation = SimpleNamespace(num_cached_tokens=[0, 4, 8])
    engine.topology = SimpleNamespace(target_leader_rank=1)
    engine.groups = SimpleNamespace(target_group=object())
    engine._run_packed_sample = MagicMock(return_value=torch.tensor([31, 41]))
    states = [
        PearlPipelineState([1], prompt_length=1, temperature=0),
        PearlPipelineState([2], prompt_length=1, temperature=0),
    ]

    result = engine._target_ar_prefill(
        [[10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]],
        states,
        sequence_ids=[1, 2],
    )

    assert result == [31, 41]
    engine._run_packed_sample.assert_called_once_with(
        [14, 15, 28, 29],
        [1, 1, 2, 2],
        [4, 5, 8, 9],
        [0, 0],
        use_aclgraph=False,
        logit_indices=[1, 3],
    )


def test_native_engine_builds_slots_from_the_cpu_cache_page_table():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.config = SimpleNamespace(kvcache_block_size=128)
    engine.cache_allocation = SimpleNamespace(block_tables=[[3, 7], [4, 9]])

    slots = engine._cache_slot_mapping([0, 1], [129, 2])

    assert slots == [7 * 128 + 1, 4 * 128 + 2]


@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_target_postverify_uses_speculative_fia_aclgraph(temperature):
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.gamma = 4
    engine.draft_vocab_size = 32
    engine.config = SimpleNamespace(
        enforce_eager=False,
        target_verification_graph_buckets=32,
        target_use_paged_attention=False,
    )
    state = PearlPipelineState(
        [1, 2, 3, 4, 5, 6],
        prompt_length=2,
        pre_verify=False,
        temperature=temperature,
    )
    engine._run_packed_greedy = MagicMock(return_value=torch.tensor([1, 2, 3, 4]))
    engine._run_packed_model = MagicMock(return_value=torch.randn(4, 32))

    engine._target_round_outputs_batch([state], [0])

    runner = engine._run_packed_greedy if temperature == 0 else engine._run_packed_model
    assert runner.call_args.kwargs["use_aclgraph"] is True
    assert runner.call_args.kwargs["use_fused_infer_attention"] is True


def test_target_preverify_uses_speculative_fia_aclgraph():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.gamma = 4
    engine.draft_vocab_size = 32
    engine.config = SimpleNamespace(
        enforce_eager=False,
        target_verification_graph_buckets=32,
        target_use_paged_attention=False,
    )
    state = PearlPipelineState([1, 2, 3], prompt_length=2, pre_verify=True)
    engine._run_packed_greedy = MagicMock(return_value=torch.tensor([4]))

    engine._target_round_outputs_batch([state], [0])

    assert engine._run_packed_greedy.call_args.kwargs["use_aclgraph"] is True
    assert engine._run_packed_greedy.call_args.kwargs["use_fused_infer_attention"] is True


def test_target_verification_can_select_paged_attention_aclgraph():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.gamma = 4
    engine.draft_vocab_size = 32
    engine.config = SimpleNamespace(
        enforce_eager=False,
        target_verification_graph_buckets=32,
        target_use_paged_attention=True,
    )
    state = PearlPipelineState([1, 2, 3], prompt_length=2, pre_verify=True)
    engine._run_packed_greedy = MagicMock(return_value=torch.tensor([4]))

    engine._target_round_outputs_batch([state], [0])

    assert engine._run_packed_greedy.call_args.kwargs["use_aclgraph"] is True
    assert engine._run_packed_greedy.call_args.kwargs["use_fused_infer_attention"] is False


@pytest.mark.parametrize("return_logits", [False, True])
def test_tree_eager_diagnostic_logits_are_optional(return_logits):
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.device = torch.device("cpu")
    engine.target_vocab_size = 32
    engine.config = SimpleNamespace(enforce_eager=True)
    engine.cache_allocation = SimpleNamespace(block_tables=torch.zeros((1, 1), dtype=torch.int32))
    engine.cache_block_tables = torch.zeros((1, 1), dtype=torch.int32)
    engine._ensure_cache_capacity = MagicMock()
    engine.model = MagicMock()
    engine.model.make_tree_attention_metadata.return_value = (
        torch.tensor([1, 2]),
        torch.tensor([1, 2]),
        SimpleNamespace(
            slot_mapping=torch.tensor([0, 1], dtype=torch.int32),
            use_fused_infer_attention=False,
            tree_attention=True,
            attention_mask=torch.tensor([[False, True], [False, False]]),
        ),
    )
    engine.model.compute_greedy_tokens.return_value = torch.tensor([7, 8])
    plan = build_tree_speculation_plan(1, 1, 1, 8)

    result = engine.target_tree_forward([plan], [1], [[2]], return_logits=return_logits)

    assert result["target_query_token_ids"].tolist() == [7, 8]
    assert result["attention_backend"] == "dense_sdpa_tree_v1"
    engine.model.compute_greedy_tokens.assert_called_once()
    assert engine.model.compute_logits.call_count == int(return_logits)
    if not return_logits:
        assert result["target_logits"] is None


def test_target_tree_forward_uses_aclgraph_for_a_static_tree_shape():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.device = torch.device("cpu")
    engine.target_vocab_size = 32
    engine.config = SimpleNamespace(enforce_eager=False)
    engine.cache_allocation = SimpleNamespace(block_tables=torch.zeros((1, 1), dtype=torch.int32))
    engine.cache_block_tables = torch.zeros((1, 1), dtype=torch.int32)
    engine._ensure_cache_capacity = MagicMock()
    engine.graph_runner = MagicMock()
    engine.graph_runner.run_target_greedy.return_value = (torch.tensor([7, 8, 9]),)
    engine.graph_runner.last_target_execution = SimpleNamespace(used_aclgraph=True)
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int32),
        use_fused_infer_attention=True,
        tree_attention=True,
        attention_mask=torch.zeros((1, 1, 3, 8), dtype=torch.bool),
    )
    engine.model = MagicMock()
    engine.model.make_tree_attention_metadata.return_value = (
        torch.tensor([1, 2, 3]),
        torch.tensor([1, 2, 3]),
        metadata,
    )
    plan = build_tree_speculation_plan(
        width=2,
        depth=1,
        prefix_len=1,
        max_model_len=8,
        candidate_budget=2,
    )

    result = engine.target_tree_forward([plan], [1], [[2, 3]])

    assert result["target_token_ids"].tolist() == [7, 8]
    assert result["bonus_token_ids"].tolist() == [9]
    assert result["used_aclgraph"] is True
    assert result["attention_backend"] == "fused_infer_attention_tree_v1"
    engine.graph_runner.run_target_greedy.assert_called_once()
    engine.model.assert_not_called()


def test_target_tree_forward_samples_graph_logits_from_common_vocabulary():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.device = torch.device("cpu")
    engine.draft_vocab_size = 16
    engine.target_vocab_size = 32
    engine.config = SimpleNamespace(enforce_eager=False)
    engine.cache_allocation = SimpleNamespace(block_tables=torch.zeros((1, 1), dtype=torch.int32))
    engine.cache_block_tables = torch.zeros((1, 1), dtype=torch.int32)
    engine._ensure_cache_capacity = MagicMock()
    engine.graph_runner = MagicMock()
    logits = torch.full((3, 16), -100.0)
    logits[0, 2] = logits[1, 9] = logits[2, 10] = 100.0
    engine.graph_runner.run_tree_logits.return_value = logits
    engine.graph_runner.last_target_execution = SimpleNamespace(used_aclgraph=True)
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int32),
        use_fused_infer_attention=True,
        tree_attention=True,
        attention_mask=torch.zeros((1, 1, 3, 8), dtype=torch.bool),
    )
    engine.model = MagicMock()
    engine.model.make_tree_attention_metadata.return_value = (
        torch.tensor([1, 2, 3]),
        torch.tensor([1, 2, 2]),
        metadata,
    )
    plan = build_tree_speculation_plan(2, 1, 1, 8, candidate_budget=2)

    result = engine.target_tree_forward([plan], [1], [[2, 3]], temperatures=[0.7])
    verdict = engine.verify_tree_outputs(
        torch.tensor([2, 3]),
        plan.parent_indices,
        result["target_query_token_ids"],
        result["bonus_token_ids"],
        [2],
        1,
    )

    assert result["target_query_token_ids"].tolist() == [2, 9, 10]
    assert verdict.token_ids.tolist() == [[2, 9]]
    assert verdict.accepted_node_indices.tolist() == [[0]]
    engine.graph_runner.run_tree_logits.assert_called_once()
    assert engine.graph_runner.run_tree_logits.call_args.args[-1] == 16
    engine.graph_runner.run_target_greedy.assert_not_called()


def test_target_tree_forward_packs_only_budgeted_nodes_and_uses_frontier_bonus():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.device = torch.device("cpu")
    engine.target_vocab_size = 32
    engine.config = SimpleNamespace(enforce_eager=False)
    engine.cache_allocation = SimpleNamespace(block_tables=torch.zeros((1, 1), dtype=torch.int32))
    engine.cache_block_tables = torch.zeros((1, 1), dtype=torch.int32)
    engine._ensure_cache_capacity = MagicMock()
    engine.graph_runner = MagicMock()
    # A one-node candidate budget requires exactly two physical queries:
    # root + active node.  The inactive sibling must not reach the model.
    engine.graph_runner.run_target_greedy.return_value = (torch.tensor([7, 8]),)
    engine.graph_runner.last_target_execution = SimpleNamespace(used_aclgraph=True)
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([1, 2], dtype=torch.int32),
        use_fused_infer_attention=True,
        tree_attention=True,
        attention_mask=torch.zeros((1, 1, 2, 8), dtype=torch.bool),
    )
    engine.model = MagicMock()
    engine.model.make_tree_attention_metadata.return_value = (
        torch.tensor([1, 2]),
        torch.tensor([1, 2]),
        metadata,
    )
    plan = build_tree_speculation_plan(
        width=2,
        depth=1,
        prefix_len=1,
        max_model_len=8,
        candidate_budget=1,
    )

    result = engine.target_tree_forward([plan], [1], [[2]])

    assert result["target_token_ids"].tolist() == [7]
    assert result["bonus_token_ids"].tolist() == [8]
    packed_plan = engine.model.make_tree_attention_metadata.call_args.args[0][0]
    packed_row = engine.model.make_tree_attention_metadata.call_args.args[2][0]
    assert packed_row == [2]
    assert packed_plan.parent_indices.tolist() == [-1]
    assert packed_plan.positions.tolist() == [1, 2]
    assert packed_plan.cache_positions.tolist() == [1, 2]
    assert result["query_count"] == 2
    assert result["cache_slot_mapping"].tolist() == [1, 2]
    assert result["target_query_token_ids"].tolist() == [7, 8]
    engine._ensure_cache_capacity.assert_called_once_with([0, 0], [1, 2])


def test_target_tree_forward_reports_graph_fallback_as_eager():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.device = torch.device("cpu")
    engine.target_vocab_size = 32
    engine.config = SimpleNamespace(enforce_eager=False)
    engine.cache_allocation = SimpleNamespace(block_tables=[[0]])
    engine.cache_block_tables = torch.zeros((1, 1), dtype=torch.int32)
    engine._ensure_cache_capacity = MagicMock()
    engine.graph_runner = MagicMock()
    engine.graph_runner.run_target_greedy.return_value = (torch.tensor([7, 8]),)
    engine.graph_runner.last_target_execution = SimpleNamespace(used_aclgraph=False)
    engine.model = MagicMock()
    engine.model.make_tree_attention_metadata.return_value = (
        torch.tensor([1, 2]),
        torch.tensor([1, 2]),
        SimpleNamespace(
            slot_mapping=torch.tensor([1, 2], dtype=torch.int32),
            use_fused_infer_attention=True,
            tree_attention=True,
            attention_mask=torch.zeros((1, 1, 2, 8), dtype=torch.bool),
        ),
    )
    plan = build_tree_speculation_plan(2, 1, 1, 8, candidate_budget=1)

    result = engine.target_tree_forward([plan], [1], [[2]])

    assert result["used_aclgraph"] is False
    assert result["query_count"] == 2


def test_target_tree_forward_packs_variable_request_budgets_into_one_forward():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.device = torch.device("cpu")
    engine.target_vocab_size = 32
    engine.config = SimpleNamespace(enforce_eager=False)
    engine.cache_allocation = SimpleNamespace(block_tables=[[0], [1]])
    engine.cache_block_tables = torch.zeros((2, 1), dtype=torch.int32)
    engine._ensure_cache_capacity = MagicMock()
    engine.graph_runner = MagicMock()
    engine.graph_runner.run_target_greedy.return_value = (torch.arange(7, 13),)
    engine.graph_runner.last_target_execution = SimpleNamespace(used_aclgraph=True)
    engine.model = MagicMock()
    engine.model.make_tree_attention_metadata.return_value = (
        torch.tensor([1, 2, 10, 20, 21, 22]),
        torch.tensor([1, 2, 6, 7, 8, 7]),
        SimpleNamespace(
            slot_mapping=torch.tensor([1, 2, 6, 7, 8, 9], dtype=torch.int32),
            use_fused_infer_attention=True,
            tree_attention=True,
            attention_mask=torch.zeros((2, 1, 4, 16), dtype=torch.bool),
        ),
    )
    plans = [
        build_tree_speculation_plan(2, 2, 1, 16, candidate_budget=1),
        build_tree_speculation_plan(2, 2, 6, 16, candidate_budget=3),
    ]

    result = engine.target_tree_forward(plans, [1, 10], [[2], [20, 21, 22]], sequence_ids=[0, 1])

    packed = engine.model.make_tree_attention_metadata.call_args.args[0]
    assert [plan.parent_indices.numel() for plan in packed] == [1, 3]
    assert result["query_count"] == 6  # Four candidates plus two request roots.
    assert result["num_draft_tokens"] == [1, 3]
    assert result["target_token_ids"].tolist() == [7, 9, 10, 11]
    assert result["target_query_token_ids"].tolist() == list(range(7, 13))
    assert result["bonus_token_ids"].tolist() == [8, 12]
    assert result["cache_slot_mapping"].tolist() == [1, 2, 6, 7, 8, 9]
    engine._ensure_cache_capacity.assert_called_once_with([0, 0, 1, 1, 1, 1], [1, 2, 6, 7, 8, 9])


def test_tree_state_commits_path_and_records_budget_acceptance():
    state = PearlPipelineState([11], prompt_length=1, max_tokens=8)

    state.apply_tree_verification(
        output_token_ids=[21, 22, -1],
        accepted=2,
        proposed_tokens=3,
    )

    assert state.completion_token_ids == [21, 22]
    assert state.committed_length == 3
    assert state.accepted_draft_tokens == 2
    assert state.verified_draft_tokens == 3
    assert state.acceptance_lengths == [3, 0]
    assert state.pre_verify is True


def test_draft_tree_forward_expands_spine_and_broadcast_shape():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.device = torch.device("cpu")
    engine.draft_vocab_size = 4
    engine.cache_allocation = SimpleNamespace(block_tables=torch.zeros((2, 2), dtype=torch.int32))
    engine.cache_block_tables = torch.zeros((2, 2), dtype=torch.int32)
    engine._ensure_cache_capacity = MagicMock()
    engine.model = MagicMock()
    engine.model.make_tree_level_attention_metadata.side_effect = (
        lambda plan, sequence_id, node_indices, input_token_ids, block_tables: (
            torch.tensor(input_token_ids),
            torch.tensor(node_indices),
            SimpleNamespace(),
        )
    )
    engine.model.make_tree_attention_metadata.return_value = (
        torch.tensor([1, 2, 3, 4, 5]),
        torch.tensor([1, 2, 3, 4, 5]),
        SimpleNamespace(slot_mapping=torch.arange(5, dtype=torch.int32)),
    )
    engine.model.compute_logits.side_effect = [
        torch.tensor([[0.1, 0.9, 0.2, 0.3]]),
        torch.tensor([[0.8, 0.1, 0.2, 0.3]]),
    ]
    plan = build_tree_speculation_plan(
        width=2,
        depth=2,
        prefix_len=1,
        max_model_len=16,
        candidate_budget=3,
    )

    result = engine.draft_tree_forward([plan], [1], [1])

    assert len(result["draft_token_ids"]) == 1
    assert len(result["draft_token_ids"][0]) == 4
    assert result["parent_indices"].numel() == 3
    assert result["num_draft_tokens"] == [3]
    assert engine.model.make_tree_attention_metadata.call_args.kwargs["sequence_ids"] == [1]


def test_draft_tree_eager_writes_frontier_root_into_tree_kv():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.device = torch.device("cpu")
    engine.draft_vocab_size = 4
    engine.cache_allocation = SimpleNamespace(block_tables=torch.zeros((1, 4), dtype=torch.int32))
    engine.cache_block_tables = torch.zeros((1, 4), dtype=torch.int32)
    engine._ensure_cache_capacity = MagicMock()
    engine.model = MagicMock()
    engine.model.make_tree_level_attention_metadata.side_effect = (
        lambda plan, sequence_id, node_indices, input_token_ids, block_tables: (
            torch.tensor(input_token_ids),
            torch.tensor(node_indices),
            SimpleNamespace(),
        )
    )
    engine.model.make_tree_attention_metadata.return_value = (
        torch.tensor([1, 2, 3, 4, 5]),
        torch.tensor([1, 2, 3, 4, 5]),
        SimpleNamespace(slot_mapping=torch.arange(5, dtype=torch.int32)),
    )
    # Frontier query, root query, and the second spine query.  The final
    # masked write does not request logits.
    engine.model.compute_logits.side_effect = [
        torch.tensor([[0.1, 0.2, 0.3, 9.0]]),
        torch.tensor([[0.1, 9.0, 0.2, 0.3]]),
        torch.tensor([[0.1, 0.2, 9.0, 0.3]]),
    ]
    parent_plan = build_tree_speculation_plan(
        width=2,
        depth=2,
        prefix_len=1,
        max_model_len=16,
        candidate_budget=3,
    )
    eager_plan = build_tree_speculation_plan(
        width=2,
        depth=2,
        prefix_len=5,
        max_model_len=16,
        candidate_budget=3,
    )

    result = engine.draft_tree_forward(
        [eager_plan],
        [99],
        [0],
        eager_parent_sources={0: (parent_plan, [10, 11, 12, 13])},
    )

    assert result["root_token_ids"] == [3]
    final_call = engine.model.make_tree_attention_metadata.call_args
    assert final_call.args[1] == [3]


def test_tree_metadata_separates_logical_rope_and_physical_kv_positions():
    model = NativeQwen2ForCausalLM.__new__(NativeQwen2ForCausalLM)
    model.embed_tokens = SimpleNamespace(weight=torch.empty(1))
    model.layers = [SimpleNamespace(self_attn=SimpleNamespace(block_size=4))]
    model.max_model_len = 16
    plan = build_tree_speculation_plan(
        width=2,
        depth=2,
        prefix_len=1,
        max_model_len=16,
        candidate_budget=4,
    )
    _, positions, metadata = model.make_tree_attention_metadata(
        [plan], [7], [[10, 11, 20, 21]], [[0, 0, 0, 0]], sequence_ids=[0]
    )
    assert positions.tolist() == plan.positions.tolist()
    assert metadata.context_lens.tolist() == [2, 3, 4, 5, 6]
    assert metadata.slot_mapping.tolist() == [1, 2, 3, 0, 1]

    _, level_positions, level_metadata = model.make_tree_level_attention_metadata(
        plan, 0, [0, 2], [10, 20], [[0, 0, 0, 0]]
    )
    assert level_positions.tolist() == [2, 2]
    assert level_metadata.context_lens.tolist() == [3, 5]
    assert level_metadata.slot_mapping.tolist() == [2, 0]


def test_greedy_graph_forwards_the_speculative_fia_backend():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.device = torch.device("cpu")
    engine._run_device_packed_greedy = MagicMock(return_value=torch.tensor([4]))

    engine._run_packed_greedy(
        [3],
        [0],
        [2],
        use_aclgraph=True,
        use_fused_infer_attention=True,
    )

    assert engine._run_device_packed_greedy.call_args.kwargs == {
        "use_aclgraph": True,
        "use_fused_infer_attention": True,
    }


def test_draft_round_uses_speculative_fia_aclgraph():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.gamma = 2
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(enforce_eager=False)
    engine._run_device_packed_greedy = MagicMock(
        side_effect=[torch.tensor([4]), torch.tensor([5])],
    )
    state = PearlPipelineState([1, 2, 3], prompt_length=2)

    verification, continuation = engine._draft_round_batch([state], [0])

    assert verification == [[4]]
    assert continuation == [[4, 5]]
    assert all(call.kwargs["use_aclgraph"] is True for call in engine._run_device_packed_greedy.call_args_list)
    assert all(
        call.kwargs["use_fused_infer_attention"] is True for call in engine._run_device_packed_greedy.call_args_list
    )


def test_draft_round_uses_one_gamma_step_aclgraph_when_available():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.gamma = 2
    engine.draft_vocab_size = 32
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(enforce_eager=False)
    engine.graph_runner = MagicMock()
    engine.graph_runner.run_draft_greedy.return_value = torch.tensor([[4, 5]])
    metadata = SimpleNamespace(use_fused_infer_attention=True)
    engine._prepare_attention_metadata = MagicMock(
        side_effect=[
            (torch.tensor([2]), metadata),
            (torch.tensor([3]), metadata),
        ]
    )
    state = PearlPipelineState([1, 2, 3], prompt_length=2)

    verification, continuation = engine._draft_round_batch([state], [0])

    assert verification == [[4]]
    assert continuation == [[4, 5]]
    assert engine._prepare_attention_metadata.call_args_list[0].args == ([0], [2])
    assert engine._prepare_attention_metadata.call_args_list[1].args == ([0], [3])
    engine.graph_runner.run_draft_greedy.assert_called_once()


def test_draft_round_can_select_paged_attention_aclgraph():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.gamma = 2
    engine.draft_vocab_size = 32
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(enforce_eager=False, draft_use_paged_attention=True)
    engine.graph_runner = MagicMock()
    engine.graph_runner.run_draft_greedy.return_value = torch.tensor([[4, 5]])
    metadata = SimpleNamespace(use_fused_infer_attention=False)
    engine._prepare_attention_metadata = MagicMock(
        side_effect=[
            (torch.tensor([2]), metadata),
            (torch.tensor([3]), metadata),
        ]
    )
    state = PearlPipelineState([1, 2, 3], prompt_length=2)

    verification, continuation = engine._draft_round_batch([state], [0])

    assert verification == [[4]]
    assert continuation == [[4, 5]]
    assert all(
        call.kwargs["use_fused_infer_attention"] is False for call in engine._prepare_attention_metadata.call_args_list
    )
    engine.graph_runner.run_draft_greedy.assert_called_once()


def test_draft_aclgraph_captures_paged_attention_steps():
    model = MagicMock()
    runner = NativeACLGraphRunner(model, enabled=False)
    runner.enabled = True
    runner.expected_fia_batch_size = 2
    runner._capture_draft = MagicMock(return_value=torch.tensor([[4, 5], [6, 7]]))
    metadata = SimpleNamespace(use_fused_infer_attention=False)

    output = runner.run_draft_greedy(
        torch.tensor([2, 3]),
        [torch.tensor([1, 1]), torch.tensor([2, 2])],
        [metadata, metadata],
        vocabulary_size=32,
    )

    assert output.tolist() == [[4, 5], [6, 7]]
    assert runner._capture_draft.call_args.args[0] == (
        "draft-greedy:32|steps:2|paged",
        2,
    )


def test_draft_aclgraph_eager_chain_feeds_each_token_to_the_next_step():
    model = MagicMock()
    model.side_effect = [torch.tensor([[10.0]]), torch.tensor([[20.0]])]
    model.compute_greedy_tokens.side_effect = [torch.tensor([4]), torch.tensor([5])]
    runner = NativeACLGraphRunner(model, enabled=False)

    output = runner.run_draft_greedy(
        torch.tensor([3]),
        [torch.tensor([2]), torch.tensor([3])],
        [
            SimpleNamespace(use_fused_infer_attention=True),
            SimpleNamespace(use_fused_infer_attention=True),
        ],
        vocabulary_size=32,
    )

    assert output.tolist() == [[4, 5]]
    assert torch.equal(model.call_args_list[0].args[0], torch.tensor([3]))
    assert torch.equal(model.call_args_list[1].args[0], torch.tensor([4]))


def test_target_aclgraph_eager_path_keeps_fixed_verification_inputs():
    model = MagicMock()
    model.side_effect = [torch.tensor([[10.0]]), torch.tensor([[20.0]])]
    model.compute_greedy_tokens.side_effect = [torch.tensor([4]), torch.tensor([5])]
    runner = NativeACLGraphRunner(model, enabled=False)
    metadata = SimpleNamespace(use_fused_infer_attention=False)

    outputs = runner.run_target_greedy(
        [torch.tensor([3]), torch.tensor([9])],
        [torch.tensor([2]), torch.tensor([3])],
        [metadata, metadata],
        vocabulary_size=32,
    )

    assert [output.tolist() for output in outputs] == [[4], [5]]
    assert torch.equal(model.call_args_list[0].args[0], torch.tensor([3]))
    assert torch.equal(model.call_args_list[1].args[0], torch.tensor([9]))


def test_target_aclgraph_requires_matching_step_metadata():
    runner = NativeACLGraphRunner(MagicMock(), enabled=False)
    metadata = SimpleNamespace(use_fused_infer_attention=False)

    with pytest.raises(ValueError, match="matching non-empty"):
        runner.run_target_greedy(
            [torch.tensor([3])],
            [],
            [metadata],
            vocabulary_size=32,
        )


def test_draft_aclgraph_uses_eager_for_a_dynamic_tail_batch():
    model = MagicMock()
    runner = NativeACLGraphRunner(model, enabled=False)
    runner.enabled = True
    runner.expected_fia_batch_size = 4
    runner._execute_draft = MagicMock(return_value=torch.tensor([[4, 5], [6, 7]]))
    metadata = SimpleNamespace(use_fused_infer_attention=True)

    output = runner.run_draft_greedy(
        torch.tensor([2, 3]),
        [torch.tensor([1, 1]), torch.tensor([2, 2])],
        [metadata, metadata],
        vocabulary_size=32,
    )

    assert output.tolist() == [[4, 5], [6, 7]]
    assert runner.shape_fallback_count == 1
    runner._execute_draft.assert_called_once()


def test_draft_round_snapshots_a_reused_aclgraph_output_buffer():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.gamma = 2
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(enforce_eager=False)
    shared_output = torch.tensor([0])

    def replay(*_args, **_kwargs):
        shared_output.add_(1)
        return shared_output

    engine._run_device_packed_greedy = MagicMock(side_effect=replay)
    state = PearlPipelineState([1, 2, 3], prompt_length=2)

    _, continuation = engine._draft_round_batch([state], [0])

    assert continuation == [[1, 2]]


def test_draft_round_executes_only_rows_with_remaining_variable_budget():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.gamma = 4
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(enforce_eager=True, draft_use_paged_attention=False)
    engine._run_device_packed_greedy = MagicMock(
        side_effect=[
            torch.tensor([10, 20]),
            torch.tensor([11]),
            torch.tensor([12]),
        ]
    )
    states = [
        PearlPipelineState([1, 2], prompt_length=1),
        PearlPipelineState([3, 4], prompt_length=1),
    ]

    verification, continuation = engine._draft_round_device_batch(states, [0, 1], draft_budgets=[3, 1])

    assert verification.tolist() == [10, 20]
    assert continuation.tolist() == [[10, 11, 12, -1], [20, -1, -1, -1]]
    assert [call.args[1] for call in engine._run_device_packed_greedy.call_args_list] == [
        [0, 1],
        [0],
        [0],
    ]


def test_spec_rhythm_eager_verification_keeps_the_parent_window_prefix():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.gamma = 4
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(
        enforce_eager=True,
        draft_use_paged_attention=False,
    )
    engine._run_device_packed_greedy_with_confidence = MagicMock(
        side_effect=[
            (torch.tensor([10, 20]), torch.tensor([0.8, 0.7])),
            (torch.tensor([11, 21]), torch.tensor([0.6, 0.5])),
        ]
    )
    states = [
        PearlPipelineState([1, 2], prompt_length=1),
        PearlPipelineState([3, 4, 30, 31, 32, 33], prompt_length=1),
    ]

    verification, continuation, confidence = engine._draft_spec_rhythm_device_batch(
        states,
        [0, 1],
        [2, 2],
        verification_sizes=[1, 4],
        verification_prefixes=[None, torch.tensor([31, 32, 33])],
    )

    assert verification.tolist() == [10, 31, 32, 33, 20]
    assert continuation.tolist() == [[10, 11, -1, -1], [20, 21, -1, -1]]
    assert confidence.tolist() == pytest.approx([0.7, 0.6])


def test_greedy_verdict_supports_mixed_per_request_gamma():
    target = torch.tensor([1, 2, 9, 4, 5, 6])
    draft = torch.tensor([1, 2, 3, 4, 0, 0])
    verdict = _build_greedy_verdict(target, draft, verification_sizes=[3, 1, 2], gamma=5)
    assert verdict.tolist() == [[2, 9], [1, -1], [0, 5]]


def test_completion_is_truncated_at_the_first_eos_or_token_limit():
    assert _truncate_completion([10, 11, 99, 12], frozenset((99,)), 4) == [10, 11, 99]
    assert _truncate_completion([10, 11, 12], frozenset(), 2) == [10, 11]
    assert _truncate_completion([10, 99, 12], frozenset((99,)), 3, ignore_eos=True) == [10, 99, 12]


def test_sampling_params_are_per_request_and_reject_mixed_temperature_modes():
    params = SamplingParams(temperature=0.7, max_tokens=12, ignore_eos=True)

    assert _normalize_sampling_params(2, params, 64) == [params, params]
    with pytest.raises(ValueError, match="all zero or all non-zero"):
        _normalize_sampling_params(
            2,
            [SamplingParams(temperature=0), SamplingParams(temperature=1)],
            64,
        )
    with pytest.raises(ValueError, match="draft temperatures"):
        _normalize_sampling_params(
            2,
            [SamplingParams(temperature=0, draft_temperature=0), SamplingParams(temperature=0, draft_temperature=1)],
            64,
        )

    state = PearlPipelineState(
        [1, 99],
        prompt_length=1,
        committed_length=2,
        max_tokens=2,
        ignore_eos=True,
    )
    assert not _finished(state, frozenset((99,)))
    state.token_ids.append(2)
    state.committed_length = 3
    assert _finished(state, frozenset((99,)))


def test_target_sampler_supports_greedy_and_exponential_race_sampling():
    logits = torch.tensor([[1.0, 3.0, 2.0], [2.0, 1.0, 3.0]])

    assert _sample_logits(logits, [0.0, 0.0]).tolist() == [1, 2]
    assert _sample_logits(
        logits,
        [1.0, 1.0],
        exponential_noise=torch.ones_like(logits),
    ).tolist() == [1, 2]


def test_target_sampler_applies_per_row_top_k_and_top_p_before_sampling():
    logits = torch.tensor([[5.0, 4.0, 3.0], [5.0, 4.0, 3.0]])
    # Noise would make token 2 win without filtering. Both filters retain
    # only token 0 for their respective row.
    noise = torch.tensor([[10.0, 10.0, 0.001], [10.0, 10.0, 0.001]])
    tokens = _sample_logits(
        logits,
        [1.0, 1.0],
        top_ps=[1.0, 0.5],
        top_ks=[1, 0],
        exponential_noise=noise,
    )
    assert tokens.tolist() == [0, 0]


def test_stochastic_verdict_accepts_prefix_and_samples_masked_correction():
    target_logits = torch.tensor([[8.0, 1.0, 0.0]] * 5)
    draft_tokens = torch.zeros(5, dtype=torch.long)

    verdict = _build_stochastic_verdict(
        target_logits,
        draft_tokens,
        verification_sizes=[1, 4],
        gamma=4,
        temperatures=[1.0] * 5,
        random_values=torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
        exponential_noise=torch.ones_like(target_logits),
    )

    assert verdict.tolist() == [[1, -1], [2, 1]]


def test_pipeline_state_reports_upstream_mat_segments():
    state = PearlPipelineState([1, 2, 3], prompt_length=2)
    assert state.acceptance_lengths == []
    state.apply_target_verification(
        gamma=4,
        accepted=1,
        correction_token_id=None,
        next_round_token_ids=[4, 5, 6, 7],
    )
    state.apply_target_verification(
        gamma=4,
        accepted=2,
        correction_token_id=42,
        next_round_token_ids=[8, 9, 10, 11],
    )

    assert state.acceptance_lengths == [4, 0]
    assert sum(state.acceptance_lengths) / len(state.acceptance_lengths) == 2.0


def test_auto_gamma_uses_the_upstream_draft_to_target_speed_ratio():
    assert _gamma_from_decode_speeds(700.0, 100.0) == 7
    assert _gamma_from_decode_speeds(50.0, 100.0) == 1
    assert _gamma_from_decode_speeds(10_000.0, 100.0) == 100

    config = NativePearlConfig("draft", "target", 1, 2, -1, 512, 32)
    assert config.gamma == -1


def test_direct_worker_cli_exposes_cache_capacity_controls():
    args = _build_parser().parse_args(
        [
            "--draft-model",
            "draft",
            "--target-model",
            "target",
            "--prompt",
            "hello",
            "--gpu-memory-utilization",
            "0.98",
            "--num-kvcache-blocks",
            "32",
            "--max-aclgraph-entries",
            "8",
            "--spec-rhythm-online-prefill",
            "--spec-rhythm-merge-ready-homes",
            "--spec-rhythm-priority-mode",
            "--spec-rhythm-priority-burst",
            "3",
            "--spec-rhythm-target-fallback-max-batch",
            "2",
            "--spec-rhythm-cpu-verdict",
        ]
    )

    assert args.gpu_memory_utilization == 0.98
    assert args.num_kvcache_blocks == 32
    assert args.max_aclgraph_entries == 8
    assert args.target_verification_graph_buckets == 8
    assert args.disable_cpu_binding is False
    assert args.draft_use_production_rope is True
    assert args.target_use_production_rope is True
    assert args.spec_rhythm_min_gamma == 1
    assert args.spec_rhythm_max_eager_tokens == 0
    assert args.spec_rhythm_tree_width == 1
    assert args.spec_rhythm_tree_depth == 1
    assert args.spec_rhythm_online_prefill is True
    assert args.spec_rhythm_merge_ready_homes is True
    assert args.spec_rhythm_priority_mode is True
    assert args.spec_rhythm_priority_burst == 3
    assert args.spec_rhythm_target_fallback_max_batch == 2
    assert args.spec_rhythm_cpu_verdict is True


def test_benchmark_cli_defaults_production_rope_with_independent_fallbacks():
    required = [
        "--draft-model",
        "draft",
        "--target-model",
        "target",
        "--prompt",
        "hello",
    ]
    defaults = _build_benchmark_parser().parse_args(required)
    fallback = _build_benchmark_parser().parse_args(
        [
            *required,
            "--no-draft-use-production-rope",
            "--no-target-use-production-rope",
        ]
    )

    assert defaults.draft_use_production_rope is True
    assert defaults.target_use_production_rope is True
    assert fallback.draft_use_production_rope is False
    assert fallback.target_use_production_rope is False


def test_benchmark_clis_accept_disjoint_warmup_prompt_offsets():
    speculative = _build_benchmark_parser().parse_args(
        [
            "--draft-model",
            "draft",
            "--target-model",
            "target",
            "--prompt",
            "hello",
            "--warmup-prompt-offset",
            "200",
        ]
    )
    target_only = _build_target_only_benchmark_parser().parse_args(
        [
            "--model",
            "target",
            "--prompt",
            "hello",
            "--warmup-prompt-offset",
            "200",
        ]
    )

    assert speculative.warmup_prompt_offset == 200
    assert target_only.warmup_prompt_offset == 200


def test_public_pearl_config_maps_upstream_fields_to_native_runtime():
    model_config = SimpleNamespace(
        architectures=["Qwen2ForCausalLM"],
        eos_token_id=[1, 2],
    )
    with patch(
        "vllm_ascend.spec_decode.pearl.api.AutoConfig.from_pretrained",
        side_effect=[model_config, model_config],
    ):
        config = PEARLConfig(
            "draft",
            "target",
            draft_tensor_parallel_size=1,
            target_tensor_parallel_size=2,
            max_model_len=512,
            max_num_batched_tokens=1024,
            max_num_seqs=8,
            draft_dtype="bfloat16",
            target_dtype="float16",
            prefill_chunk_size=4,
            max_num_queued_seqs=12,
            gamma=4,
            enable_continuous_batching=True,
            enable_preemptive_scheduling=True,
            enable_spec_rhythm=True,
            spec_rhythm_min_gamma=2,
            spec_rhythm_max_eager_tokens=3,
            spec_rhythm_roofline={"8:1": 24},
            spec_rhythm_draft_token_budget=32,
            spec_rhythm_tree_width=2,
            spec_rhythm_tree_depth=2,
            pad_finished_requests=True,
            draft_use_paged_attention=True,
            target_use_paged_attention=True,
            draft_use_production_rope=True,
            target_use_production_rope=True,
            precompile_decode_graphs=True,
            target_verification_graph_post_counts=((8, (0, 4, 8)),),
            enable_cpu_binding=False,
            profile_decode_steps=5,
        )

    native = config.to_native()
    assert config.world_size == 3
    assert config.eos == [1, 2]
    assert config.draft_config.model == "draft"
    assert config.draft_config.devices == [0]
    assert config.target_config.model == "target"
    assert config.target_config.devices == [1, 2]
    assert config.target_config.master_rank == 1
    assert native.draft_model == "draft"
    assert native.target_tp_size == 2
    assert native.draft_dtype == "bfloat16"
    assert native.target_dtype == "float16"
    assert native.max_num_batched_tokens == 1024
    assert native.max_aclgraph_entries == 32
    assert native.target_verification_graph_buckets == 8
    assert native.target_verification_graph_post_counts == ((8, (0, 4, 8)),)
    assert native.max_num_seqs == 8
    assert native.prefill_chunk_size == 4
    assert native.max_num_queued_seqs == 12
    assert native.enable_continuous_batching is True
    assert native.enable_preemptive_scheduling is True
    assert native.enable_spec_rhythm is True
    assert native.spec_rhythm_min_gamma == 2
    assert native.spec_rhythm_max_eager_tokens == 3
    assert native.spec_rhythm_roofline == {"8:1": 24}
    assert native.spec_rhythm_draft_token_budget == 32
    assert native.spec_rhythm_tree_width == 2
    assert native.spec_rhythm_tree_depth == 2
    assert native.pad_finished_requests is True
    assert native.draft_use_paged_attention is True
    assert native.target_use_paged_attention is True
    assert native.draft_use_production_rope is True
    assert native.target_use_production_rope is True
    assert native.precompile_decode_graphs is True
    assert native.enable_cpu_binding is False
    assert native.profile_decode_steps == 5


def test_public_pearl_config_rejects_nonpositive_worker_timeout():
    with pytest.raises(ValueError, match="worker_timeout_seconds"):
        PEARLConfig("draft", "target", worker_timeout_seconds=0)


def test_native_pearl_config_rejects_unsupported_dtype():
    with pytest.raises(ValueError, match="model dtype"):
        NativePearlConfig("draft", "target", 1, 2, 4, 512, 32, target_dtype="float32")


def test_public_pearl_config_requires_continuous_batching_for_preemption():
    with pytest.raises(ValueError, match="preemptive scheduling"):
        PEARLConfig("draft", "target", enable_preemptive_scheduling=True)


def test_public_pearl_config_rejects_negative_profile_decode_steps():
    with pytest.raises(ValueError, match="profile_decode_steps"):
        PEARLConfig("draft", "target", profile_decode_steps=-1)


def test_public_pearl_config_requires_steps_for_profiling_only():
    with pytest.raises(ValueError, match="profiling-only"):
        PEARLConfig("draft", "target", stop_after_profiled_decode_steps=True)


def test_native_graph_precompilation_requires_fixed_paged_configuration():
    common = {
        "draft_model": "draft",
        "target_model": "target",
        "draft_tp_size": 1,
        "target_tp_size": 1,
        "max_model_len": 32,
        "max_tokens": 8,
        "precompile_decode_graphs": True,
    }

    with pytest.raises(ValueError, match="fixed gamma"):
        NativePearlConfig(**common, gamma=-1)
    with pytest.raises(ValueError, match="paged attention"):
        NativePearlConfig(**common, gamma=4)
    with pytest.raises(ValueError, match="max_aclgraph_entries"):
        NativePearlConfig(
            **common,
            gamma=4,
            max_num_seqs=8,
            enable_continuous_batching=True,
            draft_use_paged_attention=True,
            target_use_paged_attention=True,
            max_aclgraph_entries=8,
        )


def test_profiling_only_continuous_results_can_include_unfinished_requests():
    completed = PearlPipelineState([1, 2], prompt_length=1)
    unfinished = PearlPipelineState([3, 4], prompt_length=1)
    completed_request_states = {0: completed}
    local_states = [PearlPipelineState([9], prompt_length=1), unfinished]

    result_states = _continuous_result_states(local_states, completed_request_states)

    assert result_states == [completed, unfinished]


@pytest.mark.parametrize("prefill_chunk_size", [0, 9])
def test_public_pearl_config_rejects_invalid_prefill_chunk_size(prefill_chunk_size):
    with pytest.raises(ValueError, match="prefill_chunk_size"):
        PEARLConfig(
            "draft",
            "target",
            max_num_seqs=8,
            prefill_chunk_size=prefill_chunk_size,
        )


def test_public_engine_collects_worker_replies_in_rank_order():
    engine = PEARLEngine.__new__(PEARLEngine)
    engine.config = SimpleNamespace(worker_timeout_seconds=1.0)
    engine._processes = [MagicMock(), MagicMock()]
    for process in engine._processes:
        process.is_alive.return_value = True
    rank0, worker0 = Pipe(duplex=True)
    rank1, worker1 = Pipe(duplex=True)
    engine._connections = [rank0, rank1]
    try:
        worker1.send(("ready", 1))
        worker0.send(("ready", 0))

        assert engine._receive_all("test") == [("ready", 0), ("ready", 1)]
    finally:
        for connection in (rank0, rank1, worker0, worker1):
            connection.close()


def test_public_engine_preserves_sampling_request_metadata_when_not_overridden():
    engine = PEARLEngine.__new__(PEARLEngine)
    engine.config = SimpleNamespace(max_model_len=64)
    engine._requests = []
    engine._next_request_id = 0
    params = SamplingParams(
        max_tokens=4,
        request_id="trace-7",
        arrival_ts=123.5,
        slo_tpot_ms=20.0,
        slo_class="tight",
        spec_rhythm_max_gamma=3,
    )

    engine.add_request([1, 2], params)

    queued = engine._requests[0][2]
    assert queued.request_id == "trace-7"
    assert queued.arrival_ts == 123.5
    assert queued.slo_tpot_ms == 20.0
    assert queued.slo_class == "tight"
    assert queued.spec_rhythm_max_gamma == 3


def test_public_engine_times_out_with_pending_worker_ranks():
    engine = PEARLEngine.__new__(PEARLEngine)
    engine.config = SimpleNamespace(worker_timeout_seconds=0.01)
    process = MagicMock()
    process.is_alive.return_value = True
    parent, worker = Pipe(duplex=True)
    engine._processes = [process]
    engine._connections = [parent]
    try:
        with pytest.raises(TimeoutError, match=r"pending ranks: \[0\]"):
            engine._receive_all("test timeout")
    finally:
        parent.close()
        worker.close()


def test_public_engine_broadcasts_upstream_log_command():
    engine = PEARLEngine.__new__(PEARLEngine)
    engine._send_all = MagicMock()
    engine._receive_all = MagicMock(return_value=[("logged", 0), ("logged", 1)])

    engine.log("ready")

    engine._send_all.assert_called_once_with(("log", "ready", None, None))
    engine._receive_all.assert_called_once_with("worker logging")


def test_public_engine_configures_decode_profiling_on_every_worker():
    engine = PEARLEngine.__new__(PEARLEngine)
    engine._send_all = MagicMock()
    engine._receive_all = MagicMock(return_value=[("configured", 0), ("configured", 1)])

    engine.configure_decode_profiling(5, True)

    engine._send_all.assert_called_once_with(("configure_decode_profiling", 5, True, None))
    engine._receive_all.assert_called_once_with("worker profiling configuration")


def test_public_engine_publishes_live_admission_with_monotonic_epoch_sequence():
    engine = PEARLEngine.__new__(PEARLEngine)
    receiver, sender = Pipe(duplex=False)
    engine.config = SimpleNamespace(
        spec_rhythm_tree_width=2,
        spec_rhythm_tree_depth=2,
        max_model_len=64,
        max_num_queued_seqs=4,
        max_num_seqs=2,
    )
    engine._admission_connections = [sender]
    engine._live_lock = __import__("threading").Lock()
    engine._live_epoch = 7
    engine._live_accepting = True
    engine._live_admission_seq = 0
    engine._live_total_requests = 1
    engine._live_sampling_signature = (False, False)
    params = SamplingParams(temperature=0, max_tokens=2, request_id="live-1")
    try:
        engine.admit_live_requests([([1, 2], params)])
        message = receiver.recv()
        assert message[:3] == ("admit", 7, 1)
        assert message[3] == [([1, 2], params)]
        assert engine._live_total_requests == 2
    finally:
        receiver.close()
        sender.close()


def test_public_engine_idle_fence_does_not_overtake_pending_live_admission():
    engine = PEARLEngine.__new__(PEARLEngine)
    receiver, sender = Pipe(duplex=False)
    engine._admission_connections = [sender]
    engine._live_lock = __import__("threading").Lock()
    engine._live_epoch = 3
    engine._live_accepting = True
    engine._live_admission_seq = 2
    try:
        engine._resolve_live_idle(3, worker_admission_seq=1)
        assert engine._live_accepting is True
        assert receiver.poll() is False

        engine._resolve_live_idle(3, worker_admission_seq=2)
        assert engine._live_accepting is False
        assert receiver.recv() == ("close_live", 3, 2, ())
    finally:
        receiver.close()
        sender.close()


def test_native_engine_reconfigures_decode_profiling_without_model_reload():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.config = NativePearlConfig(
        draft_model="draft",
        target_model="target",
        draft_tp_size=1,
        target_tp_size=1,
        gamma=4,
        max_model_len=32,
        max_tokens=8,
        profile_decode_steps=0,
    )

    engine.configure_decode_profiling(5, True)

    assert engine.config.profile_decode_steps == 5
    assert engine.config.stop_after_profiled_decode_steps is True


def test_public_engine_chunks_queued_requests_by_sequence_and_token_limits():
    engine = PEARLEngine.__new__(PEARLEngine)
    engine.config = SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=5)
    params = SamplingParams(temperature=0, max_tokens=4)
    requests = [
        (0, [1, 2, 3], params),
        (1, [4, 5], params),
        (2, [6, 7, 8], params),
    ]

    chunks = engine._request_chunks(requests)

    assert [[request[0] for request in chunk] for chunk in chunks] == [[0, 1], [2]]


def test_public_engine_validates_pipeline_and_benchmark_cache_capacity_before_dispatch():
    engine = PEARLEngine.__new__(PEARLEngine)
    engine.config = SimpleNamespace(gamma=4, max_model_len=10)
    engine._requests = [(0, [1, 2, 3, 4, 5], SamplingParams(temperature=0, max_tokens=2))]

    with pytest.raises(ValueError, match="verification window"):
        engine.generate()
    with pytest.raises(ValueError, match="benchmark steps"):
        engine.bench_generate(num_pearl_steps=1)


def test_public_engine_aggregates_aclgraph_metrics_from_every_worker():
    engine = PEARLEngine.__new__(PEARLEngine)
    engine.config = SimpleNamespace(
        gamma=4,
        max_model_len=32,
        max_num_seqs=1,
        max_num_batched_tokens=32,
        worker_timeout_seconds=1,
    )
    engine._requests = [(0, [1], SamplingParams(max_tokens=1))]
    engine.last_metrics = []
    engine.tokenizer = MagicMock()
    engine.tokenizer.decode.return_value = "result"
    engine._send_all = MagicMock()
    leader_result = {
        "completion_token_ids": [2],
        "num_acc_tokens": [1],
        "elapsed_seconds": 1.0,
    }
    base_metrics = {
        "aclgraph_captures": 2,
        "aclgraph_capture_attempts": 2,
        "aclgraph_replays": 3,
        "aclgraph_failed_captures": 0,
        "aclgraph_capacity_fallbacks": 0,
        "aclgraph_shape_fallbacks": 0,
    }
    failed_rank_metrics = {
        **base_metrics,
        "aclgraph_capture_attempts": 8,
        "aclgraph_failed_captures": 6,
        "aclgraph_capacity_fallbacks": 9,
        "aclgraph_shape_fallbacks": 11,
    }
    engine._receive_all = MagicMock(
        return_value=[
            ("result", None, failed_rank_metrics),
            ("result", [leader_result], base_metrics),
        ]
    )

    _, num_tokens, _, elapsed = engine._generate("pearl")

    assert num_tokens == [1]
    assert elapsed == 1.0
    assert engine.last_metrics[0]["aclgraph_capture_attempts"] == 8
    assert engine.last_metrics[0]["aclgraph_failed_captures"] == 6
    assert engine.last_metrics[0]["aclgraph_capacity_fallbacks"] == 9
    assert engine.last_metrics[0]["aclgraph_shape_fallbacks"] == 11
    assert engine.last_worker_metrics_by_chunk == [
        {
            "batch_size": 1,
            "num_requests": 1,
            "worker_metrics": [failed_rank_metrics, base_metrics],
        }
    ]


def test_decode_profile_aggregates_only_full_batch_chunks_per_step():
    def worker(rank, is_draft, scale):
        return {
            "rank": rank,
            "is_draft_rank": int(is_draft),
            "worker_profiled_decode_steps": 2,
            "worker_profile_draft_compute_seconds": 0.01 * scale,
            "worker_profile_draft_to_target_communication_seconds": 0.02 * scale,
            "worker_profile_target_compute_seconds": 0.03 * scale,
            "worker_profile_target_verdict_seconds": 0.04 * scale,
            "worker_profile_target_to_draft_communication_seconds": 0.05 * scale,
            "worker_profile_wait_sync_seconds": 0.06 * scale,
            "worker_profile_state_update_seconds": 0.07 * scale,
        }

    chunks = [
        {
            "batch_size": 8,
            "worker_metrics": [worker(0, True, 1), worker(1, False, 2), worker(2, False, 3)],
        },
        {
            "batch_size": 3,
            "worker_metrics": [worker(0, True, 100), worker(1, False, 100)],
        },
    ]

    profile = _aggregate_decode_profile(chunks, batch_size=8)

    assert profile["profiled_full_batch_chunks"] == 1
    assert profile["profiled_decode_steps"] == 2
    assert profile["phase_milliseconds_per_decode_step"] == pytest.approx(
        {
            "draft_compute": 5.0,
            "draft_to_target_communication": 30.0,
            "target_verify": 105.0,
            "target_to_draft_communication": 75.0,
            "wait_sync_state_update": 195.0,
        }
    )


def test_worker_aclgraph_deltas_match_workers_by_rank():
    before = [
        {
            "rank": 1,
            "aclgraph_captures": 2,
            "aclgraph_capture_attempts": 2,
            "aclgraph_replays": 4,
            "aclgraph_failed_captures": 0,
            "aclgraph_capacity_fallbacks": 0,
            "aclgraph_shape_fallbacks": 0,
        },
        {
            "rank": 0,
            "aclgraph_captures": 1,
            "aclgraph_capture_attempts": 1,
            "aclgraph_replays": 3,
            "aclgraph_failed_captures": 0,
            "aclgraph_capacity_fallbacks": 0,
            "aclgraph_shape_fallbacks": 0,
        },
    ]
    after = [
        {**before[1], "aclgraph_replays": 8},
        {
            **before[0],
            "aclgraph_captures": 3,
            "aclgraph_capture_attempts": 3,
            "aclgraph_replays": 9,
        },
    ]

    deltas = _worker_aclgraph_deltas(before, after)

    assert deltas[0]["rank"] == 0
    assert deltas[0]["aclgraph_capture_attempts_delta"] == 0
    assert deltas[0]["aclgraph_replays_delta"] == 5
    assert deltas[1]["rank"] == 1
    assert deltas[1]["aclgraph_capture_attempts_delta"] == 1
    assert deltas[1]["aclgraph_replays_delta"] == 5


def test_worker_aclgraph_deltas_count_cold_measurement_without_warmup():
    after = [
        {
            "rank": 0,
            "aclgraph_captures": 2,
            "aclgraph_capture_attempts": 2,
            "aclgraph_replays": 7,
            "aclgraph_failed_captures": 0,
            "aclgraph_capacity_fallbacks": 0,
            "aclgraph_shape_fallbacks": 0,
        }
    ]

    deltas = _worker_aclgraph_deltas([], after)

    assert deltas == [
        {
            "rank": 0,
            "aclgraph_captures_delta": 2,
            "aclgraph_capture_attempts_delta": 2,
            "aclgraph_replays_delta": 7,
            "aclgraph_failed_captures_delta": 0,
            "aclgraph_capacity_fallbacks_delta": 0,
            "aclgraph_shape_fallbacks_delta": 0,
        }
    ]


def test_greedy_verdict_finds_first_mismatch_in_packed_mixed_windows():
    target_tokens = torch.tensor([5, 10, 21, 32, 40, 51, 61, 71, 81])
    draft_tokens = torch.tensor([5, 10, 20, 30, 40, 50, 60, 70, 80])

    verdict = _build_greedy_verdict(
        target_tokens,
        draft_tokens,
        verification_sizes=[1, 4, 4],
        gamma=4,
    )

    assert verdict.tolist() == [[1, -1], [1, 21], [0, 51]]


def test_layout_greedy_verdict_matches_general_packed_windows():
    target_tokens = torch.tensor([5, 10, 21, 32, 40, 51, 61, 71, 81])
    draft_tokens = torch.tensor([5, 10, 20, 30, 40, 50, 60, 70, 80])
    expected_sizes, dense_positions, correction_positions = _build_verification_layout(
        [1, 4, 4],
        4,
        target_tokens.device,
    )

    verdict = _build_greedy_verdict_with_layout(
        target_tokens,
        draft_tokens,
        expected_sizes,
        dense_positions,
        correction_positions,
        gamma=4,
    )

    assert verdict.tolist() == [[1, -1], [1, 21], [0, 51]]


def test_cpu_greedy_verdict_matches_device_verdict_for_mixed_windows():
    target = torch.tensor([10, 11, 12, 20, 21, 30, 31], dtype=torch.long)
    draft = torch.tensor([10, 99, 12, 20, 22, 30, 31], dtype=torch.long)
    sizes = [3, 2, 2]
    expected = _build_greedy_verdict(target, draft, sizes, gamma=3)
    actual = _build_greedy_verdict_cpu(target, draft, sizes)
    assert actual.tolist() == expected.tolist()


def test_native_aclgraph_padding_preserves_tokens_and_uses_inactive_cache_slots():
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([9, 10]),
        context_lens=torch.tensor([3, 4], dtype=torch.int32),
        block_tables=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
    )

    input_ids, positions, padded = NativeACLGraphRunner._pad_inputs(
        torch.tensor([5, 6]),
        torch.tensor([1, 2]),
        metadata,
        4,
    )

    assert input_ids.tolist() == [5, 6, 0, 0]
    assert positions.tolist() == [1, 2, 0, 0]
    assert padded.slot_mapping.tolist() == [9, 10, -1, -1]
    assert padded.context_lens.tolist() == [3, 4, 0, 0]
    assert padded.block_tables.tolist() == [[1, 2], [3, 4], [0, 0], [0, 0]]


def test_native_aclgraph_eager_greedy_path_includes_lm_head_sampling():
    model = MagicMock()
    hidden_states = torch.randn(2, 4)
    model.return_value = hidden_states
    model.compute_greedy_tokens.return_value = torch.tensor([3, 5])
    runner = NativeACLGraphRunner(model, enabled=False)
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0, 1], dtype=torch.int32),
        context_lens=torch.tensor([1, 1], dtype=torch.int32),
        block_tables=torch.tensor([[0], [1]], dtype=torch.int32),
    )

    tokens = runner.run_greedy(
        torch.tensor([1, 2]),
        torch.tensor([0, 0]),
        metadata,
        vocabulary_size=8,
    )

    assert tokens.tolist() == [3, 5]
    model.compute_greedy_tokens.assert_called_once_with(hidden_states, 8)


def test_native_aclgraph_capacity_uses_eager_for_new_shapes():
    model = MagicMock()
    metadata = SimpleNamespace(
        use_fused_infer_attention=True,
        actual_seq_lengths_q=(1,),
    )
    with patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream"):
        runner = NativeACLGraphRunner(model, enabled=True, max_graph_entries=1)
    runner.entries[("existing", 1)] = MagicMock()
    runner._execute = MagicMock(return_value=torch.tensor([7]))

    output = runner.run_greedy(
        torch.tensor([1]),
        torch.tensor([0]),
        metadata,
        vocabulary_size=8,
    )

    assert output.tolist() == [7]
    assert runner.capacity_fallback_count == 1
    runner._execute.assert_called_once()


def test_native_aclgraph_capacity_counts_failed_capture_attempts():
    model = MagicMock()
    metadata = SimpleNamespace(
        use_fused_infer_attention=True,
        actual_seq_lengths_q=(1,),
    )
    with patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream"):
        runner = NativeACLGraphRunner(model, enabled=True, max_graph_entries=1)
    runner.capture_attempt_count = 1
    runner._capture_budget_used = 1
    runner._execute = MagicMock(return_value=torch.tensor([7]))

    output = runner.run_greedy(
        torch.tensor([1]),
        torch.tensor([0]),
        metadata,
        vocabulary_size=8,
    )

    assert output.tolist() == [7]
    assert runner.entries == {}
    assert runner.capacity_fallback_count == 1
    runner._execute.assert_called_once()


def test_native_aclgraph_releases_target_graph_resources_and_reopens_resident_capacity():
    with patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream"):
        runner = NativeACLGraphRunner(MagicMock(), enabled=True, max_graph_entries=2)
    first = MagicMock()
    second = MagicMock()
    runner.target_entries[("first", (1,))] = first
    runner.target_entries[("second", (2,))] = second
    runner._capture_budget_used = 2
    runner.capture_attempt_count = 7
    runner.capture_count = 5
    runner.replay_count = 20

    with patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.synchronize") as synchronize:
        assert runner.release_target_graph_entries() == 2

    synchronize.assert_called_once_with()
    first.graph.reset.assert_called_once_with()
    second.graph.reset.assert_called_once_with()
    assert not runner.target_entries
    assert runner._capture_budget_used == 0
    assert runner.capture_attempt_count == 7
    assert runner.capture_count == 5
    assert runner.replay_count == 20


def test_native_aclgraph_replay_orders_task_updates_without_host_sync():
    model = MagicMock()
    metadata = SimpleNamespace(
        use_fused_infer_attention=True,
        actual_seq_lengths_q=(1,),
    )
    update_stream = MagicMock()
    current_stream = MagicMock()
    with (
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream",
            return_value=update_stream,
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_graph.torch.npu.current_stream",
            return_value=current_stream,
        ),
    ):
        runner = NativeACLGraphRunner(model, enabled=True)
        entry = MagicMock(
            output=torch.tensor([7]),
            runtime_validated=True,
        )
        runner.entries[("greedy:8|fia:1", 1)] = entry
        runner._copy_inputs = MagicMock()
        runner._update_attention_tasks = MagicMock()

        output = runner.run_greedy(
            torch.tensor([1]),
            torch.tensor([0]),
            metadata,
            vocabulary_size=8,
        )

    assert output.tolist() == [7]
    update_stream.wait_stream.assert_called_once_with(current_stream)
    current_stream.synchronize.assert_not_called()
    runner._update_attention_tasks.assert_called_once_with(entry)
    entry.graph.replay.assert_called_once_with()


def test_native_aclgraph_greedy_validation_accepts_only_sparse_graph_divergence():
    reference = torch.arange(128)
    sparse = reference.clone()
    sparse[:7] = -1
    dense = sparse.clone()
    dense[7] = -1

    assert NativeACLGraphRunner._outputs_match(sparse, reference)
    assert not NativeACLGraphRunner._outputs_match(dense, reference)


def test_native_aclgraph_greedy_validation_rejects_multiple_small_batch_mismatches():
    reference = torch.arange(16)
    one_mismatch = reference.clone()
    one_mismatch[0] = -1
    two_mismatches = one_mismatch.clone()
    two_mismatches[1] = -1

    assert NativeACLGraphRunner._outputs_match(one_mismatch, reference)
    assert not NativeACLGraphRunner._outputs_match(two_mismatches, reference)


def test_native_aclgraph_skips_dynamic_fia_tail_without_spending_capture_budget():
    model = MagicMock()
    metadata = SimpleNamespace(
        use_fused_infer_attention=True,
        actual_seq_lengths_q=(1, 2, 3, 4, 5, 6, 7),
    )
    with patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream"):
        runner = NativeACLGraphRunner(model, enabled=True, max_graph_entries=1)
    runner.set_expected_fia_batch_size(8)
    runner._execute = MagicMock(return_value=torch.tensor([7]))

    output = runner.run_greedy(
        torch.tensor([1]),
        torch.tensor([0]),
        metadata,
        vocabulary_size=8,
    )

    assert output.tolist() == [7]
    assert runner.capture_attempt_count == 0
    assert runner.shape_fallback_count == 1
    runner._execute.assert_called_once()


@pytest.mark.parametrize(
    ("actual_seq_lengths_q", "expected"),
    [
        ((1, 2, 3, 4), True),
        ((4, 8, 12, 16), True),
        ((1, 5, 6, 10), True),
        ((1, 5, 5, 9), False),
    ],
)
def test_native_aclgraph_reuses_full_batch_fia_segments(actual_seq_lengths_q, expected):
    with patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream"):
        runner = NativeACLGraphRunner(MagicMock(), enabled=True)
    runner.set_expected_fia_batch_size(4)

    assert runner._is_reusable_fia_shape(actual_seq_lengths_q) is expected


def test_native_aclgraph_rejects_nonpositive_entry_capacity():
    with pytest.raises(ValueError, match="max_graph_entries"):
        NativeACLGraphRunner(MagicMock(), enabled=False, max_graph_entries=0)


def test_native_weight_loader_uses_safe_open_keys_api(tmp_path):
    weight_file = tmp_path / "model.safetensors"
    weight_file.touch()
    checkpoint = MagicMock()
    checkpoint.keys.return_value = []
    model = MagicMock()
    model.packed_modules_mapping = {}

    with patch("vllm_ascend.spec_decode.pearl.native_model.safe_open") as safe_open:
        safe_open.return_value.__enter__.return_value = checkpoint
        load_native_qwen2_weights(model, str(tmp_path))

    checkpoint.keys.assert_called_once_with()


def test_native_nz_conversion_keeps_tied_lm_head_nd_but_converts_transformer():
    model = MagicMock()
    model.config.tie_word_embeddings = True
    lm_head = MagicMock(spec=NativeLMHead)
    lm_head.bias = None
    embedding_weight = MagicMock()
    embedding_weight.device.type = "npu"
    lm_head.weight = embedding_weight
    model.lm_head = lm_head
    model.embed_tokens.weight = embedding_weight
    transformer = MagicMock(spec=NativeRowLinear)
    transformer.bias = None
    transformer.weight = MagicMock()
    transformer.weight.device.type = "npu"
    transformer_weight = transformer.weight.data
    model.modules.return_value = [lm_head, transformer]
    converted = MagicMock()

    with (
        patch(
            "vllm_ascend.spec_decode.pearl.native_model.ascend_envs.VLLM_ASCEND_ENABLE_NZ",
            2,
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_model.torch_npu.npu_format_cast",
            return_value=converted,
        ) as format_cast,
    ):
        _maybe_convert_linear_weights_to_nz(model)

    format_cast.assert_called_once_with(transformer_weight, 29)
    assert model.embed_tokens.weight is embedding_weight
    assert lm_head.weight is embedding_weight
    assert transformer.weight.data is converted


def test_native_nz_conversion_keeps_biased_linear_weights_in_nd():
    model = MagicMock()
    model.config.tie_word_embeddings = False
    biased = MagicMock(spec=NativeColumnLinear)
    biased.bias = MagicMock()
    biased.weight = MagicMock()
    biased.weight.device.type = "npu"
    biased_weight = biased.weight.data
    unbiased = MagicMock(spec=NativeColumnLinear)
    unbiased.bias = None
    unbiased.weight = MagicMock()
    unbiased.weight.device.type = "npu"
    unbiased_weight = unbiased.weight.data
    model.modules.return_value = [biased, unbiased]
    converted = MagicMock()

    with (
        patch(
            "vllm_ascend.spec_decode.pearl.native_model.ascend_envs.VLLM_ASCEND_ENABLE_NZ",
            2,
        ),
        patch(
            "vllm_ascend.spec_decode.pearl.native_model.torch_npu.npu_format_cast",
            return_value=converted,
        ) as format_cast,
    ):
        _maybe_convert_linear_weights_to_nz(model)

    format_cast.assert_called_once_with(unbiased_weight, 29)
    assert biased.weight.data is biased_weight
    assert unbiased.weight.data is converted
