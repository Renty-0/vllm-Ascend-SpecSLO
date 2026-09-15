# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.spec_decode.pearl import native_engine as native
from vllm_ascend.spec_decode.pearl.native_engine import (
    NativePearlConfig,
    NativePearlEngine,
    NativeSamplingParams,
    NativeSpecRhythmDevicePayload,
    PearlPipelineState,
)
from vllm_ascend.spec_decode.pearl.spec_rhythm import (
    ProposalLifecycle,
    SpecRhythmPipelineController,
    SpecRhythmRuntimeState,
)
from vllm_ascend.spec_decode.pearl.tree import (
    build_tree_speculation_plan,
    pack_selected_tree_plan,
    tree_primary_path,
)


def _fixture(count=1, width=2, depth=4):
    states = {
        index: SpecRhythmRuntimeState(index, index % 2, delivered_tokens=7, prefix_epoch=4) for index in range(count)
    }
    controller = SpecRhythmPipelineController(states)
    payloads = {}
    cache_mappings = {}
    for index in states:
        size = width * depth
        ticket = controller.new_ticket(index, gamma=size, eager=False)
        controller.publish([ticket])
        plan = build_tree_speculation_plan(width, depth, prefix_len=7, max_model_len=32)
        payloads[ticket.proposal_id] = {
            "ticket": ticket,
            "plan": pack_selected_tree_plan(plan, range(size)),
            "row": list(range(10, 10 + size)),
            "confidence": 0.8,
        }
        # Logical packed rows can map to non-adjacent physical pages. This is
        # deliberately a strided tensor view, not just a non-monotone list.
        cache_mappings[ticket.proposal_id] = torch.arange(200, dtype=torch.int32)[::3][: size + 1]
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine._move_tree_cache_slots = MagicMock()
    return engine, controller, payloads, cache_mappings


def _stage_eager(controller, payloads, cache_mappings, index=0):
    parent = controller.ready[index]
    parent_payload = payloads[parent.proposal_id]
    dependencies = tuple(parent_payload["row"][node] for node in tree_primary_path(parent_payload["plan"]))
    eager = controller.new_ticket(index, gamma=2, eager=True)
    controller.publish([eager])
    plan = build_tree_speculation_plan(1, 2, prefix_len=7 + len(dependencies) + 1, max_model_len=32)
    payloads[eager.proposal_id] = {
        "ticket": eager,
        "plan": plan,
        "row": [50, 51],
        "confidence": 0.9,
        "eager_parent_proposal_id": parent.proposal_id,
        "eager_dependency_tokens": dependencies,
        "eager_dependency_length": len(dependencies),
        "eager_frontier_token": 99,
    }
    cache_mappings[eager.proposal_id] = torch.tensor([90, 94, 95], dtype=torch.int32)
    return eager


def test_dynamic_tree_roof_shrink_keeps_ticket_state_and_noncontiguous_kv_slots():
    engine, controller, payloads, mappings = _fixture()
    ticket = controller.ready[0]
    before_state = asdict(controller.request_states[0])
    old_mapping = mappings[ticket.proposal_id]
    old_mapping_values = old_mapping.clone()
    assert not old_mapping.is_contiguous()
    counts = engine._bound_spec_rhythm_tree_ready(controller, payloads, mappings, [0], 3)
    assert counts == {"pruned_proposals": 1, "pruned_candidates": 5, "invalidated_eager": 0}
    assert controller.ready[0] is ticket
    assert ticket.gamma == 3
    assert ticket.proposal_id == 0
    assert ticket.home_batch_id == 0
    assert ticket.required_prefix_epoch == 4
    assert ticket.lifecycle is ProposalLifecycle.AVAILABLE
    assert asdict(controller.request_states[0]) == before_state
    payload = payloads[ticket.proposal_id]
    assert payload["row"] == [10, 11, 12]
    assert payload["plan"].candidate_budget == 3
    assert payload["plan"].parent_indices.tolist() == [-1, 0, 1]
    assert payload["plan"].cache_positions.tolist() == [7, 8, 9, 10]
    assert mappings[ticket.proposal_id].tolist() == [0, 3, 6, 9]
    assert mappings[ticket.proposal_id].data_ptr() == old_mapping.data_ptr()
    assert torch.equal(old_mapping, old_mapping_values)
    engine._move_tree_cache_slots.assert_not_called()
    schedule = controller.build_plan([0], verification_budget=3)
    assert schedule.target_candidate_budgets == {0: 3}
    controller.finish_verification(
        0,
        fully_accepted=True,
        proposed_tokens=3,
        accepted_tokens=3,
        delivered_tokens=4,
        draft_confidence=0.8,
        ema_alpha=0.2,
    )
    assert controller.request_states[0].prefix_epoch == 5
    assert controller.request_states[0].delivered_tokens == 11


def test_dynamic_roof_invalidates_eager_with_truncated_dependency_immediately():
    engine, controller, payloads, mappings = _fixture()
    parent = controller.ready[0]
    eager = _stage_eager(controller, payloads, mappings)
    counts = engine._bound_spec_rhythm_tree_ready(controller, payloads, mappings, [0], 3)
    assert counts["invalidated_eager"] == 1
    assert controller.ready[0] is parent
    assert 0 not in controller.staged_eager
    assert eager.lifecycle is ProposalLifecycle.INVALIDATED
    assert eager.proposal_id not in payloads
    assert eager.proposal_id not in mappings


def test_pruning_only_siblings_preserves_exact_eager_dependency_and_promotion():
    engine, controller, payloads, mappings = _fixture(width=4, depth=2)
    eager = _stage_eager(controller, payloads, mappings)
    eager_payload = payloads[eager.proposal_id]
    counts = engine._bound_spec_rhythm_tree_ready(controller, payloads, mappings, [0], 3)
    assert counts["invalidated_eager"] == 0
    assert controller.staged_eager[0] is eager
    assert payloads[eager.proposal_id] is eager_payload
    assert eager.lifecycle is ProposalLifecycle.STAGED_EAGER
    promoted = controller.finish_verification(
        0,
        fully_accepted=True,
        proposed_tokens=3,
        accepted_tokens=2,
        delivered_tokens=3,
        draft_confidence=0.8,
        ema_alpha=0.2,
    )
    assert promoted is eager
    assert controller.ready[0] is eager
    assert eager.lifecycle is ProposalLifecycle.AVAILABLE


@pytest.mark.parametrize("corrupt", ["parent_id", "epoch", "home", "lifecycle", "missing_frontier"])
def test_roof_shrink_never_preserves_mismatched_eager_guard(corrupt):
    engine, controller, payloads, mappings = _fixture(width=4, depth=2)
    eager = _stage_eager(controller, payloads, mappings)
    if corrupt == "parent_id":
        payloads[eager.proposal_id]["eager_parent_proposal_id"] = 999
    elif corrupt == "epoch":
        eager.required_prefix_epoch += 1
    elif corrupt == "home":
        eager.home_batch_id = 1
    elif corrupt == "lifecycle":
        eager.lifecycle = ProposalLifecycle.CONSUMED
    else:
        payloads[eager.proposal_id]["eager_frontier_token"] = None
    counts = engine._bound_spec_rhythm_tree_ready(controller, payloads, mappings, [0], 3)
    assert counts["invalidated_eager"] == 1
    assert eager.lifecycle is ProposalLifecycle.INVALIDATED


def test_dynamic_roof_preflights_all_payloads_before_mutating_any():
    engine, controller, payloads, mappings = _fixture(count=2)
    eager = _stage_eager(controller, payloads, mappings)
    payloads[controller.ready[1].proposal_id]["row"] = [10]
    with pytest.raises(RuntimeError, match="aligned, packed"):
        engine._bound_spec_rhythm_tree_ready(controller, payloads, mappings, [0, 1], 3)
    assert controller.ready[0].gamma == 8
    assert controller.ready[1].gamma == 8
    assert controller.staged_eager[0] is eager
    assert eager.lifecycle is ProposalLifecycle.STAGED_EAGER
    assert len(mappings[controller.ready[0].proposal_id]) == 9


def test_dynamic_roof_rejects_bad_cache_mapping_before_mutation():
    engine, controller, payloads, mappings = _fixture()
    ticket = controller.ready[0]
    mappings[ticket.proposal_id] = torch.tensor([1, 2, 3])
    with pytest.raises(RuntimeError, match="cache mapping is shorter"):
        engine._bound_spec_rhythm_tree_ready(controller, payloads, mappings, [0], 3)
    assert ticket.gamma == 8


def test_target_ready_payload_without_existing_kv_mapping_can_be_pruned():
    engine, controller, payloads, mappings = _fixture()
    mappings.clear()
    counts = engine._bound_spec_rhythm_tree_ready(controller, payloads, mappings, [0], 3)
    assert counts["pruned_candidates"] == 5
    assert mappings == {}


def test_roof_shrink_is_idempotent_and_does_not_touch_inactive_proposals():
    engine, controller, payloads, mappings = _fixture(count=2)
    engine._bound_spec_rhythm_tree_ready(controller, payloads, mappings, [0], 3)
    assert controller.ready[1].gamma == 8
    counts = engine._bound_spec_rhythm_tree_ready(controller, payloads, mappings, [0], 3)
    assert counts == {"pruned_proposals": 0, "pruned_candidates": 0, "invalidated_eager": 0}


def _linear_fixture(is_draft=False, count=1):
    states = [
        PearlPipelineState(
            token_ids=[101, 102, 201, 202, *range(901, 909 if is_draft else 908)],
            prompt_length=2,
            committed_length=4,
            pre_verify=False,
            pending_window_size=8,
            continuation_epoch=4,
            max_tokens=5,
            accepted_draft_tokens=10,
            verified_draft_tokens=20,
            verification_rounds=4,
        )
        for _ in range(count)
    ]
    controller = SpecRhythmPipelineController(
        {index: SpecRhythmRuntimeState(index, 0, delivered_tokens=2, prefix_epoch=4) for index in range(count)}
    )
    payloads = {}
    for index in range(count):
        ticket = controller.new_ticket(index, gamma=1, eager=False)
        controller.publish([ticket])
        payloads[ticket.proposal_id] = NativeSpecRhythmDevicePayload(
            ticket,
            torch.arange(901, 909),
            torch.tensor([908]),
            8,
            0.8,
        )
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.gamma = 8
    engine.is_draft = is_draft
    return engine, controller, payloads, states


def _linear_stage_eager(controller, payloads, states, is_draft):
    ticket = controller.new_ticket(0, gamma=2, eager=True)
    controller.publish([ticket])
    payloads[ticket.proposal_id] = NativeSpecRhythmDevicePayload(
        ticket,
        torch.tensor([990]),
        torch.tensor([990, 991]),
        1,
        0.9,
    )
    if is_draft:
        states[0].token_ids.extend([990, 991])
    return ticket


def test_linear_actual_verification_size_not_next_gamma_consumes_budget():
    engine, controller, payloads, states = _linear_fixture(count=2)
    counts = engine._bound_spec_rhythm_linear_ready(controller, payloads, states, [0, 1], 8)
    assert counts["rebuilt_requests"] == 0
    assert [ticket.gamma for ticket in controller.ready.values()] == [1, 1]
    actual_counts = {
        index: payloads[ticket.proposal_id].verification_size for index, ticket in controller.ready.items()
    }
    plan = controller.build_plan([0, 1], verification_budget=8, ready_candidate_counts=actual_counts)
    assert plan.target_candidate_budgets == {0: 8}
    assert plan.deferred_target_request_indices == (1,)


@pytest.mark.parametrize("is_draft", [False, True])
def test_linear_roof_rebuild_drops_only_unverified_work_and_old_eager(is_draft):
    engine, controller, payloads, states = _linear_fixture(is_draft)
    ready = controller.ready[0]
    eager = _linear_stage_eager(controller, payloads, states, is_draft)
    runtime_before = asdict(controller.request_states[0])
    state_before = states[0].clone()
    counts = engine._bound_spec_rhythm_linear_ready(controller, payloads, states, [0], 3)
    assert counts["rebuilt_requests"] == counts["discarded_ready"] == counts["invalidated_eager"] == 1
    assert counts["discarded_unverified_tokens"] == len(state_before.token_ids) - 4
    assert states[0].token_ids == [101, 102, 201, 202]
    assert states[0].committed_completion_token_ids == [201, 202]
    assert states[0].committed_length == 4
    assert states[0].continuation_epoch == 4
    assert states[0].pre_verify
    assert states[0].pending_window_size == 0
    assert states[0].accepted_draft_tokens == state_before.accepted_draft_tokens
    assert states[0].verified_draft_tokens == state_before.verified_draft_tokens
    assert states[0].verification_rounds == state_before.verification_rounds
    assert asdict(controller.request_states[0]) == runtime_before
    assert not controller.ready and not controller.staged_eager and not payloads
    assert ready.lifecycle is eager.lifecycle is ProposalLifecycle.INVALIDATED
    fresh = controller.new_ticket(0, gamma=1, eager=False)
    assert fresh.proposal_id > eager.proposal_id
    assert fresh.required_prefix_epoch == ready.required_prefix_epoch
    assert fresh.home_batch_id == ready.home_batch_id


@pytest.mark.parametrize("is_draft", [False, True])
def test_linear_rebuilt_request_continues_to_reference_output_without_loss(is_draft):
    engine, controller, payloads, states = _linear_fixture(is_draft)
    old_eager = _linear_stage_eager(controller, payloads, states, is_draft)
    engine._bound_spec_rhythm_linear_ready(controller, payloads, states, [0], 3)
    state = states[0]
    for token in (203, 204, 205):
        ticket = controller.new_ticket(0, gamma=1, eager=False)
        controller.publish([ticket])
        payloads[ticket.proposal_id] = NativeSpecRhythmDevicePayload(
            ticket,
            torch.tensor([token]),
            torch.tensor([token]),
            1,
            0.9,
        )
        engine._bound_spec_rhythm_linear_ready(controller, payloads, states, [0], 3)
        plan = controller.build_plan([0], verification_budget=3, ready_candidate_counts={0: 1})
        assert plan.target_request_indices == (0,)
        if is_draft:
            state.token_ids.append(token)
            state.apply_draft_verification(
                gamma=8,
                accepted=1,
                correction_token_id=None,
                next_round_token_ids=[token],
                verification_size=1,
            )
        else:
            state.apply_target_verification(
                gamma=8,
                accepted=1,
                correction_token_id=None,
                next_round_token_ids=[token],
                verification_size=1,
            )
        promoted = controller.finish_verification(
            0,
            fully_accepted=True,
            proposed_tokens=1,
            accepted_tokens=1,
            delivered_tokens=1,
            draft_confidence=0.9,
            ema_alpha=0.2,
        )
        assert promoted is None
        payloads.pop(ticket.proposal_id)
    assert state.committed_completion_token_ids == [201, 202, 203, 204, 205]
    assert native._finished(state, set())
    assert old_eager.lifecycle is ProposalLifecycle.INVALIDATED
    assert controller.request_states[0].delivered_tokens == 5


def test_linear_rebuild_handles_oversized_pending_window_without_ready_ticket():
    engine, controller, payloads, states = _linear_fixture()
    controller.invalidate_request(0)
    payloads.clear()
    counts = engine._bound_spec_rhythm_linear_ready(controller, payloads, states, [0], 3)
    assert counts["rebuilt_requests"] == 1
    assert counts["discarded_ready"] == 0
    assert states[0].pre_verify
    assert states[0].committed_completion_token_ids == [201, 202]


def test_linear_rebuild_does_not_trim_next_gamma_when_actual_verify_fits():
    engine, controller, payloads, states = _linear_fixture()
    state = states[0]
    state.pre_verify = True
    state.pending_window_size = 0
    ticket = controller.ready[0]
    ticket.gamma = 8
    payloads[ticket.proposal_id] = NativeSpecRhythmDevicePayload(
        ticket,
        torch.tensor([901]),
        torch.arange(901, 909),
        1,
        0.8,
    )
    before = asdict(state)
    counts = engine._bound_spec_rhythm_linear_ready(controller, payloads, states, [0], 3)
    assert counts["rebuilt_requests"] == 0
    assert asdict(state) == before


def test_linear_rebuild_preflights_all_requests_before_any_discard():
    engine, controller, payloads, states = _linear_fixture(count=2)
    states[1].continuation_epoch += 1
    with pytest.raises(RuntimeError, match="stale device mailbox"):
        engine._bound_spec_rhythm_linear_ready(controller, payloads, states, [0, 1], 3)
    assert len(states[0].token_ids) > states[0].committed_length
    assert controller.ready[0].lifecycle is ProposalLifecycle.AVAILABLE
    assert len(payloads) == 2


@pytest.mark.parametrize("is_draft", [False, True])
def test_native_linear_loop_uses_actual_counts_and_shared_committed_context(monkeypatch, is_draft):
    """Run real loop admission/planning on CPU, stopping before model execution."""
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.gamma = 8
    engine.is_draft = is_draft
    engine.device = torch.device("cpu")
    engine.rank = 0 if is_draft else 1
    engine.topology = SimpleNamespace(target_leader_rank=1)
    engine.eos_token_ids = frozenset()
    engine.config = NativePearlConfig(
        "fixture-draft",
        "fixture-target",
        1,
        3,
        8,
        1024,
        5,
        max_num_seqs=2,
        spec_rhythm_roofline={"2:1": 8, "2:2": 1},
    )
    # Target has 512 tokens while draft has 513. Neither unverified suffix may
    # push the shared committed context (505 tokens) into a different bucket.
    prefix = [101] * 503 + [201, 202]
    draft_states = [
        PearlPipelineState(
            [*prefix, *range(901, 909)],
            503,
            committed_length=505,
            pre_verify=False,
            pending_window_size=8,
        )
        for _ in range(2)
    ]
    target_states = [
        PearlPipelineState(
            [*prefix, *range(901, 908)],
            503,
            committed_length=505,
            pre_verify=False,
            pending_window_size=8,
        )
        for _ in range(2)
    ]
    recorded = {}

    class StopBeforeModel(RuntimeError):
        pass

    def controller_factory(states):
        controller = SpecRhythmPipelineController(states)
        original_build = controller.build_plan

        def stop_after_plan(active, **kwargs):
            recorded["kwargs"] = kwargs
            recorded["plan"] = original_build(active, **kwargs)
            raise StopBeforeModel

        controller.build_plan = stop_after_plan
        return controller

    original_bound = engine._bound_spec_rhythm_linear_ready

    def inject_prior_ready(
        controller,
        payloads,
        local_states,
        active,
        roof,
        *,
        full_window=False,
    ):
        for index in active:
            controller.request_states[index].home_batch_id = 0
            ticket = controller.new_ticket(index, gamma=1, eager=False)
            controller.publish([ticket])
            payloads[ticket.proposal_id] = NativeSpecRhythmDevicePayload(
                ticket,
                torch.arange(901, 909),
                torch.tensor([908]),
                8,
                0.8,
            )
        recorded["roof"] = roof
        return original_bound(
            controller,
            payloads,
            local_states,
            active,
            roof,
            full_window=full_window,
        )

    engine._bound_spec_rhythm_linear_ready = inject_prior_ready
    monkeypatch.setattr(native, "SpecRhythmPipelineController", controller_factory)
    monkeypatch.setattr(native.dist, "broadcast", lambda *args, **kwargs: None)
    monkeypatch.delenv("VLLM_ASCEND_PEARL_NPU_PROFILE_DIR", raising=False)
    with pytest.raises(StopBeforeModel):
        engine._generate_spec_rhythm_decode(
            draft_states=draft_states,
            target_states=target_states,
            request_params=[NativeSamplingParams(), NativeSamplingParams()],
            initial_batch_size=2,
            continuous_batching=True,
            prefill_elapsed=0.0,
            started=0.0,
            max_rounds=1,
            prefilled_indices={0, 1},
        )
    assert recorded["roof"] == 8
    assert recorded["kwargs"]["ready_candidate_counts"] == {0: 8, 1: 8}
    assert recorded["plan"].target_candidate_budgets == {0: 8}
