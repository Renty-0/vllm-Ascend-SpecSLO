# SPDX-License-Identifier: Apache-2.0
"""CPU fault injection at the tree verdict/KV commit boundary."""

import multiprocessing
from dataclasses import asdict
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

from tests.ut.spec_decode.test_spec_rhythm_tree_loop import _TreeLoopHarness
from vllm_ascend.spec_decode.pearl.native_engine import NativePearlEngine
from vllm_ascend.spec_decode.pearl.spec_rhythm import (
    SpecRhythmPipelineController,
    SpecRhythmRuntimeState,
)
from vllm_ascend.spec_decode.pearl.tree import build_tree_speculation_plan, pack_selected_tree_plan


def _model(capacity=128):
    return SimpleNamespace(
        layers=[
            SimpleNamespace(
                self_attn=SimpleNamespace(
                    key_cache=torch.zeros(capacity // 4, 4, 1, 1),
                    value_cache=torch.zeros(capacity // 4, 4, 1, 1),
                )
            )
            for _ in range(2)
        ]
    )


def _fixture(*, draft=False, eager=False):
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.device = torch.device("cpu")
    engine.is_draft = draft
    engine.model = _model()
    engine._cache_slot_mapping = Mock(return_value=[72, 76, 75])
    controller = SpecRhythmPipelineController({index: SpecRhythmRuntimeState(index, index % 2) for index in range(2)})
    plans, payloads, mappings = [], {}, {}
    for index in range(2):
        ticket = controller.new_ticket(index, gamma=4, eager=False)
        controller.publish([ticket])
        plan = pack_selected_tree_plan(build_tree_speculation_plan(2, 2, 7, 64), range(4))
        plans.append(plan)
        payloads[ticket.proposal_id] = {"ticket": ticket, "plan": plan}
        mappings[ticket.proposal_id] = torch.tensor([3, 7, 6, 9, 11]) + 16 * index
    if eager:
        ticket = controller.new_ticket(1, gamma=2, eager=True)
        controller.publish([ticket])
        payloads[ticket.proposal_id] = {
            "ticket": ticket,
            "plan": build_tree_speculation_plan(1, 2, 10, 64),
        }
        mappings[ticket.proposal_id] = torch.tensor([60, 63, 69])
    output = None if draft else {"cache_slot_mapping": torch.cat(list(mappings.values()))}
    return engine, controller, payloads, mappings, plans, output


def _preflight(fixture):
    engine, controller, payloads, mappings, plans, output = fixture
    return engine._preflight_spec_rhythm_tree_cache_commit(
        controller,
        payloads,
        mappings,
        [0, 1],
        plans,
        output,
    )


def test_kv_preflight_unindexed_npu_resolves_only_to_current_card(monkeypatch):
    current_device = Mock(return_value=2)
    monkeypatch.setattr(torch.npu, "current_device", current_device)
    resolved = NativePearlEngine._spec_rhythm_tree_cache_device(torch.device("npu"))
    assert resolved == torch.device("npu:2")
    assert resolved != torch.device("npu:3")
    assert NativePearlEngine._spec_rhythm_tree_cache_device(torch.device("npu:3")) == torch.device("npu:3")
    assert NativePearlEngine._spec_rhythm_tree_cache_device(torch.device("cpu")) == torch.device("cpu")
    current_device.assert_called_once()


@pytest.mark.parametrize("draft,eager", [(False, False), (True, False), (True, True)])
def test_kv_preflight_accepts_nonmonotone_slots_without_mutation(draft, eager):
    fixture = _fixture(draft=draft, eager=eager)
    _, controller, payloads, mappings, _, _ = fixture
    original_states = {index: asdict(state) for index, state in controller.request_states.items()}
    original_tickets = {key: asdict(value["ticket"]) for key, value in payloads.items()}
    original_mappings = {key: value.clone() for key, value in mappings.items()}
    pending = _preflight(fixture)
    assert {index: asdict(state) for index, state in controller.request_states.items()} == original_states
    assert {key: asdict(value["ticket"]) for key, value in payloads.items()} == original_tickets
    assert all(torch.equal(mappings[key], value) for key, value in original_mappings.items())
    assert set(pending) == (set() if draft else set(controller.ready[index].proposal_id for index in range(2)))


@pytest.mark.parametrize("kind", ["missing", "short", "long", "rank", "float", "negative", "large", "alias"])
@pytest.mark.parametrize("location", ["target", "draft", "eager"])
def test_kv_preflight_rejects_bad_last_proposal_mapping(kind, location):
    fixture = _fixture(draft=location != "target", eager=location == "eager")
    _, controller, _, mappings, _, output = fixture
    ticket = controller.staged_eager[1] if location == "eager" else controller.ready[1]
    value = mappings[ticket.proposal_id].clone()
    if kind == "missing":
        value = None
    elif kind == "short":
        value = value[:-1]
    elif kind == "long":
        value = torch.cat((value, torch.tensor([99])))
    elif kind == "rank":
        value = value.reshape(1, -1)
    elif kind == "float":
        value = value.to(torch.float32)
    elif kind == "negative":
        value[-1] = -1
    elif kind == "large":
        value[-1] = 128
    else:
        value[-1] = value[0]
    if location == "target":
        if value is None or value.ndim != 1:
            output["cache_slot_mapping"] = value
        else:
            first = mappings[controller.ready[0].proposal_id]
            output["cache_slot_mapping"] = torch.cat((first, value))
    else:
        mappings[ticket.proposal_id] = value
    with pytest.raises(RuntimeError, match="KV mapping"):
        _preflight(fixture)
    assert all(state.delivered_tokens == 0 for state in controller.request_states.values())
    assert len(controller.ready) == 2
    assert len(controller.staged_eager) == int(location == "eager")


@pytest.mark.parametrize("destination", [[72], [72, -1, 75], [72, 128, 75], [72, 72, 75]])
def test_kv_preflight_checks_promotion_destination_before_commit(destination):
    fixture = _fixture(draft=True, eager=True)
    fixture[0]._cache_slot_mapping.return_value = destination
    with pytest.raises(RuntimeError, match="draft eager destination.*KV mapping"):
        _preflight(fixture)
    assert fixture[1].request_states[1].prefix_epoch == 0
    assert 1 in fixture[1].staged_eager


@pytest.mark.parametrize("cache", [None, torch.zeros(128, 1, 1), torch.zeros(4, 4, 1, 1)])
def test_kv_preflight_validates_last_layer_key_and_value_capacity(cache):
    fixture = _fixture()
    fixture[0].model.layers[-1].self_attn.value_cache = cache
    with pytest.raises(RuntimeError, match="paged layer caches|out-of-bounds"):
        _preflight(fixture)


def test_kv_preflight_accepts_strided_slot_mapping_views():
    fixture = _fixture()
    fixture[-1]["cache_slot_mapping"] = torch.arange(40)[::4]
    assert not fixture[-1]["cache_slot_mapping"].is_contiguous()
    pending = _preflight(fixture)
    assert [value.tolist() for value in pending.values()] == [[0, 4, 8, 12, 16], [20, 24, 28, 32, 36]]


@pytest.mark.parametrize("failure", ["truncated", "extra", "negative", "overflow", "alias"])
def test_tree_loop_bad_final_kv_row_never_commits_first_row(monkeypatch, failure):
    harness = _TreeLoopHarness(monkeypatch, requests=4, capacity=4, max_tokens=6)
    original_states = [state.token_ids[:] for state in harness.target_states]
    harness.engine.model = _model(64)
    harness.engine._move_tree_cache_slots = Mock()
    # Keep the real compaction method: the original truncated-last-row bug
    # committed prior rows, then failed index_select in this implementation.
    harness.engine.compact_tree_round = Mock(
        wraps=lambda *args, **kwargs: NativePearlEngine.compact_tree_round(harness.engine, *args, **kwargs)
    )

    def malformed_target(*args, **kwargs):
        output = harness.target(*args, **kwargs)
        mapping = output["cache_slot_mapping"]
        if failure == "truncated":
            mapping = mapping[: args[0][0].parent_indices.numel() + 2]
        elif failure == "extra":
            mapping = torch.cat((mapping, torch.tensor([63])))
        elif failure == "negative":
            mapping[-1] = -1
        elif failure == "overflow":
            mapping[-1] = 64
        else:
            mapping[-1] = mapping[-2]
        output["cache_slot_mapping"] = mapping
        return output

    harness.engine.target_tree_forward = malformed_target
    with pytest.raises(RuntimeError, match="KV mapping"):
        harness.run(max_rounds=2)
    assert [state.token_ids for state in harness.target_states] == original_states
    assert [state.token_ids for state in harness.draft_states] == original_states
    assert all(state.verification_rounds == 0 for state in harness.target_states)
    assert all(
        state.prefix_epoch == state.delivered_tokens == 0 for state in harness.controllers[0].request_states.values()
    )
    harness.engine.compact_tree_round.assert_not_called()
    harness.engine._move_tree_cache_slots.assert_not_called()


def test_tree_loop_bad_eager_destination_is_detected_before_first_commit(monkeypatch):
    harness = _TreeLoopHarness(monkeypatch, requests=2, capacity=2, slo=0.001)
    harness.engine.is_draft = True
    harness.engine.model = _model(512)
    harness.engine._move_tree_cache_slots = Mock()
    original_states = [state.token_ids[:] for state in harness.target_states]

    def slots(indices, positions):
        # Normal publication starts at the original prefix. The continuation
        # promotion destination follows its dependency path and frontier.
        if min(positions) > 1:
            return [-1] * len(positions)
        return [index * 100 + position for index, position in zip(indices, positions)]

    harness.engine._cache_slot_mapping = slots
    moves_before_verdict = []
    harness.before_verdict = lambda: moves_before_verdict.append(harness.engine._move_tree_cache_slots.call_count)
    with pytest.raises(RuntimeError, match="draft eager destination.*KV mapping"):
        harness.run(max_rounds=2)
    assert harness.controllers[0].staged_eager
    assert [state.token_ids for state in harness.target_states] == original_states
    assert [state.token_ids for state in harness.draft_states] == original_states
    assert all(state.delivered_tokens == 0 for state in harness.controllers[0].request_states.values())
    harness.engine.compact_tree_round.assert_not_called()
    assert harness.engine._move_tree_cache_slots.call_count == moves_before_verdict[-1]


def _observe_preflight_votes(monkeypatch, harness, *, remote_failure=False):
    previous_all_reduce = dist.all_reduce
    votes = []

    def all_reduce(value, *args, **kwargs):
        # Earlier role timings and strict graph checks are independent. Only
        # inject after the verdict, at the collective commit boundary.
        if value.numel() == 1 and harness.events[-1][0] == "verdict":
            assert value.dtype == torch.int64
            assert kwargs.get("op") == dist.ReduceOp.MAX
            votes.append(value.tolist())
            previous_all_reduce(value, *args, **kwargs)
            if remote_failure:
                value.fill_(1)
        else:
            previous_all_reduce(value, *args, **kwargs)

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    return votes


def _record_stream_chunks(harness):
    chunks = []
    harness.engine._token_commit_callback = chunks.append
    harness.engine._stream_delivered_counts = {}
    harness.engine._stream_started = 0.0
    return chunks


@pytest.mark.parametrize("failure", ["semantic", "mapping"])
def test_local_preflight_error_still_reaches_collective_vote(monkeypatch, failure):
    harness = _TreeLoopHarness(monkeypatch, requests=4, capacity=4)
    original_states = [state.token_ids[:] for state in harness.target_states]
    votes = _observe_preflight_votes(monkeypatch, harness)
    chunks = _record_stream_chunks(harness)
    if failure == "semantic":

        def stale_final_row():
            verified = next(indices for event, indices in reversed(harness.events) if event == "target")
            harness.controllers[0].ready[verified[-1]].required_prefix_epoch += 1

        harness.before_verdict = stale_final_row
    else:

        def broken_mapping(*args, **kwargs):
            output = harness.target(*args, **kwargs)
            output["cache_slot_mapping"] = output["cache_slot_mapping"][:-1]
            return output

        harness.engine.target_tree_forward = broken_mapping
    with pytest.raises(RuntimeError, match="commit preflight failed on rank") as caught:
        harness.run(max_rounds=2)
    assert caught.value.__cause__ is not None
    assert votes == [[1]], "a local exception must not strand peers waiting for their vote"
    assert not chunks
    assert [state.token_ids for state in harness.target_states] == original_states
    assert all(state.delivered_tokens == 0 for state in harness.controllers[0].request_states.values())
    harness.engine.compact_tree_round.assert_not_called()


def test_remote_preflight_error_prevents_leader_commit_mapping_and_stream_chunks(monkeypatch):
    harness = _TreeLoopHarness(monkeypatch, requests=4, capacity=4)
    original_states = [state.token_ids[:] for state in harness.target_states]
    votes = _observe_preflight_votes(monkeypatch, harness, remote_failure=True)
    chunks = _record_stream_chunks(harness)
    original_preflight = harness.engine._preflight_spec_rhythm_tree_cache_commit
    inspected_mappings = []

    def record_pending(*args, **kwargs):
        pending = original_preflight(*args, **kwargs)
        inspected_mappings.append((args[2], dict(args[2]), pending))
        return pending

    harness.engine._preflight_spec_rhythm_tree_cache_commit = record_pending
    with pytest.raises(RuntimeError, match="preflight failed on another rank; this step was not committed"):
        harness.run(max_rounds=2)
    assert votes == [[0]], "the simulated target leader must itself pass preflight"
    assert inspected_mappings and inspected_mappings[0][2]
    for mappings, original, pending in inspected_mappings:
        assert set(mappings) == set(original)
        assert not set(pending).intersection(mappings), "remote failure must not publish local pending mappings"
    assert not chunks
    assert not harness.engine._stream_delivered_counts
    assert [state.token_ids for state in harness.target_states] == original_states
    assert [state.token_ids for state in harness.draft_states] == original_states
    assert all(
        state.prefix_epoch == state.delivered_tokens == 0 for state in harness.controllers[0].request_states.values()
    )
    harness.engine.compact_tree_round.assert_not_called()


def test_collective_success_keeps_streamed_tokens_equal_to_committed_outputs(monkeypatch):
    harness = _TreeLoopHarness(monkeypatch, requests=4, capacity=4)
    votes = _observe_preflight_votes(monkeypatch, harness)
    chunks = _record_stream_chunks(harness)
    results = harness.run()
    assert votes and all(value == [0] for value in votes)
    assert chunks
    for index, result in enumerate(results):
        streamed = [token for chunk in chunks if chunk["request_index"] == index for token in chunk["token_ids"]]
        assert streamed == result["completion_token_ids"]


def _distributed_preflight_worker(rank, world_size, store_path, result_queue):
    """Real Gloo vote with CPU substitutes only for compute and tree wire."""
    real_all_reduce = dist.all_reduce
    real_destroy_process_group = dist.destroy_process_group
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{store_path}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=15),
        )
        with pytest.MonkeyPatch.context() as monkeypatch:
            harness = _TreeLoopHarness(monkeypatch, requests=4, capacity=4)
            harness.engine.rank = rank
            chunks = _record_stream_chunks(harness)
            simulated_all_reduce = dist.all_reduce
            votes = []

            def collective(value, *args, **kwargs):
                if value.numel() == 1 and harness.events[-1][0] == "verdict":
                    local_vote = value.tolist()[0]
                    real_all_reduce(value, *args, **kwargs)
                    votes.append((local_vote, value.tolist()[0]))
                else:
                    simulated_all_reduce(value, *args, **kwargs)

            monkeypatch.setattr(dist, "all_reduce", collective)
            target_calls = []

            def target(*args, **kwargs):
                output = harness.target(*args, **kwargs)
                target_calls.append((args, output))
                return output

            def verdict(output, plans):
                if output is None:
                    # The protocol fixture supplies the same valid target
                    # verdict to nonleader ranks without model/TP transport.
                    args, target_output = target_calls[-1]
                    output = harness.engine.verify_tree_outputs(
                        torch.tensor([token for row in args[2] for token in row]),
                        torch.cat([plan.parent_indices for plan in plans]),
                        target_output["target_query_token_ids"],
                        target_output["bonus_token_ids"],
                        [int(plan.candidate_budget) for plan in plans],
                        max(plan.depth for plan in plans),
                    )
                return harness.verdict(output, plans)

            harness.engine.target_tree_forward = target
            harness.engine._broadcast_spec_rhythm_tree_verdict = verdict
            if rank == 2:

                def stale_last_row():
                    verified = next(indices for event, indices in reversed(harness.events) if event == "target")
                    harness.controllers[0].ready[verified[-1]].required_prefix_epoch += 1

                harness.before_verdict = stale_last_row
            error_message = None
            try:
                harness.run(max_rounds=2)
            except RuntimeError as error:
                error_message = str(error)
            result_queue.put(
                {
                    "rank": rank,
                    "error": error_message,
                    "votes": votes,
                    "chunks": chunks,
                    "committed": [state.committed_completion_token_ids for state in harness.target_states],
                    "delivered": [state.delivered_tokens for state in harness.controllers[0].request_states.values()],
                    "compactions": harness.engine.compact_tree_round.call_count,
                }
            )
    except Exception as error:
        result_queue.put({"rank": rank, "fatal": repr(error)})
    finally:
        if dist.is_initialized():
            real_destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU distributed preflight test requires Gloo")
def test_real_four_rank_vote_aborts_all_commits_for_one_stale_rank(tmp_path):
    context = multiprocessing.get_context("fork")
    results = context.Queue()
    processes = [
        context.Process(
            target=_distributed_preflight_worker,
            args=(rank, 4, str(tmp_path / "kv-preflight-gloo-store"), results),
        )
        for rank in range(4)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=20)
        assert all(not process.is_alive() for process in processes), "preflight vote stranded a peer"
        assert all(process.exitcode == 0 for process in processes)
        records = [results.get(timeout=2) for _ in processes]
        assert {record["rank"] for record in records} == set(range(4))
        for record in records:
            assert "fatal" not in record, record
            assert "commit preflight failed" in record["error"]
            assert record["votes"] == [(int(record["rank"] == 2), 1)]
            assert record["committed"] == [[], [], [], []]
            assert record["delivered"] == [0, 0, 0, 0]
            assert record["compactions"] == 0
            assert not record["chunks"]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        results.close()
