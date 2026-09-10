# SPDX-License-Identifier: Apache-2.0
"""Exercise the real tree service loop with CPU model/transport substitutes.

These are integration regressions for scheduling and commit ordering, not NPU
overlap, numerical-model, or ACLGraph performance claims.  Planning, tree
verification, proposal lifecycle and request-state updates remain production
implementations; only model compute, distributed transport and KV movement are
replaced at the device boundary.
"""

import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from examples.profile_specslo_tree_roofline import build_roofline_table
from vllm_ascend.spec_decode.pearl import native_engine as native
from vllm_ascend.spec_decode.pearl.roofline import ProfiledRoofline, runtime_source_fingerprints
from vllm_ascend.spec_decode.pearl.spec_rhythm import SpecRhythmPipelineController
from vllm_ascend.spec_decode.pearl.topology import PearlTopology
from vllm_ascend.spec_decode.pearl.tree_budget import DraftWindowEstimator


class _TreeLoopHarness:
    def __init__(
        self,
        monkeypatch,
        *,
        requests=4,
        capacity=2,
        max_tokens=6,
        online_prefill=False,
        arrivals=None,
        slo=None,
        budget=8,
        draft_window_ms=32.0,
        calibrate_window=True,
    ):
        self.events = []
        self.active_snapshots = []
        self.home_snapshots = []
        self.controllers = []
        self.capacity = capacity
        self.before_verdict = None
        self.prefilled = set()
        self.estimators = []
        self.window_observations = []
        self.last_explored_requests = 0
        harness = self

        class RecordingController(SpecRhythmPipelineController):
            def __init__(self, states):
                super().__init__(states)
                harness.controllers.append(self)

            def build_plan(self, active_indices, **kwargs):
                active = list(active_indices)
                harness.active_snapshots.append(active)
                harness.home_snapshots.append({index: self.request_states[index].home_batch_id for index in active})
                return super().build_plan(active, **kwargs)

        class CalibratedWindowEstimator(DraftWindowEstimator):
            def __init__(self, ema_alpha):
                super().__init__(ema_alpha)
                harness.estimators.append(self)
                if calibrate_window:
                    self.observe(draft_compute_ms=4.0, drafted_tokens=4, target_verify_ms=draft_window_ms)

        def synchronize_role_timings(timing, *args, **kwargs):
            # Model-boundary timing substitute: one millisecond per explored
            # node on draft, a configurable target window. The production
            # estimator and its EMA/window admission calculation stay real.
            # HCCL all-reduce does not support float64. The real control
            # envelope uses int64 microseconds, distinct from broadcast.
            assert timing.dtype == torch.int64
            if timing.numel() == 1:
                # The collective preflight vote is a failure bit, not a
                # timing envelope. With no injected peer error MAX keeps it.
                assert kwargs.get("op") == native.dist.ReduceOp.MAX
                return
            assert timing.numel() == 6
            timing[0] = self.last_explored_requests * 4000
            timing[1] = round(draft_window_ms * 1000.0) if timing[1] > 0 else 0
            self.window_observations.append(timing.tolist())

        monkeypatch.setattr(native, "SpecRhythmPipelineController", RecordingController)
        monkeypatch.setattr(native, "DraftWindowEstimator", CalibratedWindowEstimator)
        monkeypatch.setattr(native.dist, "broadcast", lambda *args, **kwargs: None)
        monkeypatch.setattr(native.dist, "barrier", lambda *args, **kwargs: None)
        monkeypatch.setattr(native.dist, "all_reduce", synchronize_role_timings)
        monkeypatch.setattr(torch.npu, "synchronize", lambda: None)
        self.engine = native.NativePearlEngine.__new__(native.NativePearlEngine)
        self.engine.config = native.NativePearlConfig(
            "cpu-draft",
            "cpu-target",
            1,
            3,
            4,
            256,
            max_tokens,
            max_num_seqs=capacity,
            enable_spec_rhythm=True,
            enable_continuous_batching=True,
            enable_preemptive_scheduling=True,
            spec_rhythm_tree_width=2,
            spec_rhythm_tree_depth=2,
            spec_rhythm_verification_budget=budget,
            spec_rhythm_online_prefill=online_prefill,
            spec_rhythm_urgency_threshold=0.0,
            spec_rhythm_acceptance_floor=0.0,
            enforce_eager=True,
        )
        self.engine.topology = PearlTopology.from_tensor_parallel_sizes(1, 3)
        self.engine.rank = self.engine.topology.target_leader_rank
        self.engine.is_draft = False
        self.engine.device = torch.device("cpu")
        self.engine.eos_token_ids = set()
        self.engine.groups = SimpleNamespace(is_verification_worker=True)
        self.engine.gamma = 4
        self.engine._release_cache = Mock()
        self.engine.graph_metrics = lambda: {}
        self.engine.compact_tree_round = Mock()
        self.engine.draft_tree_forward = self.draft
        self.engine.target_tree_forward = self.target
        self.engine._exchange_spec_rhythm_tree_candidates = self.exchange
        self.engine._broadcast_spec_rhythm_tree_verdict = self.verdict
        self.engine._prefill_and_sample_target_batch = self.prefill
        self.params = [
            native.NativeSamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
                ignore_eos=True,
                slo_tpot_ms=slo,
                arrival_ts=None if arrivals is None else arrivals[index],
                request_id=f"request-{index}",
            )
            for index in range(requests)
        ]
        self.draft_states = [
            native.PearlPipelineState(
                [100 + index, 200 + index],
                prompt_length=2,
                max_tokens=max_tokens,
                ignore_eos=True,
                slo_tpot_ms=slo,
            )
            for index in range(requests)
        ]
        self.target_states = [state.clone() for state in self.draft_states]

    def draft(self, plans, roots, indices, *, eager_parent_sources=None):
        self.events.append(("draft", tuple(indices)))
        rows = [
            [10 * (index + 1) + position for position in range(plan.parent_indices.numel())]
            for index, plan in zip(indices, plans)
        ]
        return {
            "draft_token_ids": rows,
            "draft_confidence": torch.tensor([0.625] * len(plans)),
            "node_confidences": [torch.full((plan.parent_indices.numel(),), 0.625) for plan in plans],
            "cache_slot_mapping": torch.arange(sum(plan.parent_indices.numel() + 1 for plan in plans)),
            "selected_indices": [list(range(plan.candidate_budget)) for plan in plans],
            "model_calls": 1,
            "graph_calls": 0,
            "eager_frontier_tokens": {index: 999 for index in (eager_parent_sources or {})},
        }

    def exchange(self, rows, plans, frontiers=None, confidences=None, *args, **kwargs):
        self.events.append(("exchange", tuple(plan.candidate_budget for plan in plans)))
        self.last_explored_requests = len(plans)
        if not plans:
            return None
        # Use the real wire encoder on the simulated draft leader; the
        # target-side native loop still invokes the real decoder.
        previous_rank = self.engine.rank
        self.engine.rank = self.engine.topology.draft_leader_rank
        try:
            return native.NativePearlEngine._exchange_spec_rhythm_tree_candidates(
                self.engine, rows, plans, frontiers, confidences, *args, **kwargs
            )
        finally:
            self.engine.rank = previous_rank

    def target(self, plans, roots, rows, *, sequence_ids, return_logits=True):
        self.events.append(("target", tuple(sequence_ids)))
        if self.engine.config.spec_rhythm_online_prefill:
            assert set(sequence_ids) <= self.prefilled, "decode reached unprefilled request"
        predictions = []
        for plan, row in zip(plans, rows):
            parents = plan.parent_indices.tolist()
            # Follow the first available child from each query.  This remains
            # valid after confidence selection removes/reorders siblings.
            for query_parent in [-1, *range(len(row))]:
                child = next((index for index, parent in enumerate(parents) if parent == query_parent), None)
                predictions.append(999 if child is None else row[child])
        return {
            "target_query_token_ids": torch.tensor(predictions, dtype=torch.long),
            "bonus_token_ids": torch.tensor([999] * len(plans), dtype=torch.long),
            "query_count": sum(plan.parent_indices.numel() + 1 for plan in plans),
            "attention_backend": "fused_infer_attention_tree_v1",
            "cache_slot_mapping": torch.arange(sum(plan.parent_indices.numel() + 1 for plan in plans)),
        }

    def verdict(self, output, plans):
        self.events.append(("verdict", tuple(plan.candidate_budget for plan in plans)))
        if self.before_verdict is not None:
            self.before_verdict()
        return output.token_ids.tolist(), output.accepted_node_indices.tolist()

    def prefill(self, rows, states, indices):
        self.events.append(("prefill", tuple(indices)))
        self.prefilled.update(indices)
        return [301 + index for index in indices]

    def run(self, *, max_rounds=40):
        return self.engine._generate_spec_rhythm_tree_decode(
            draft_states=self.draft_states,
            target_states=self.target_states,
            request_params=self.params,
            initial_batch_size=self.capacity,
            continuous_batching=True,
            prefill_elapsed=0.0,
            started=time.perf_counter(),
            max_rounds=max_rounds,
            prefilled_indices=(None if self.engine.config.spec_rhythm_online_prefill else set(range(len(self.params)))),
        )


@pytest.mark.parametrize("max_tokens", [1, 2, 4, 5])
def test_tree_commits_only_delivered_tokens_at_output_limit(monkeypatch, max_tokens):
    harness = _TreeLoopHarness(monkeypatch, requests=4, capacity=4, max_tokens=max_tokens)
    results = harness.run()
    assert all(len(result["completion_token_ids"]) == max_tokens for result in results)
    for index, runtime in harness.controllers[0].request_states.items():
        assert runtime.delivered_tokens == max_tokens
        assert len(harness.target_states[index].committed_completion_token_ids) == max_tokens
        assert len(harness.draft_states[index].committed_completion_token_ids) == max_tokens
    assert not harness.controllers[0].ready
    assert not harness.controllers[0].staged_eager


def test_tree_loop_completes_and_matches_replicated_request_states(monkeypatch):
    harness = _TreeLoopHarness(monkeypatch, requests=4, capacity=4)
    results = harness.run()
    assert len(results) == 4
    assert all(len(result["completion_token_ids"]) == 6 for result in results)
    assert [state.token_ids for state in harness.draft_states] == [state.token_ids for state in harness.target_states]
    harness.engine._release_cache.assert_called_once()


def test_tree_loop_caps_active_requests_and_refills_after_guarded_completion(monkeypatch):
    harness = _TreeLoopHarness(monkeypatch, requests=5, capacity=2)
    results = harness.run()
    assert max(map(len, harness.active_snapshots)) <= 2
    assert set().union(*map(set, harness.active_snapshots)) == set(range(5))
    assert all(len(result["completion_token_ids"]) == 6 for result in results)
    homes_by_request = {}
    for homes in harness.home_snapshots:
        for index, home in homes.items():
            homes_by_request.setdefault(index, set()).add(home)
    assert all(len(homes) == 1 for homes in homes_by_request.values())


def test_tree_loop_submits_both_computes_before_new_candidate_exchange(monkeypatch):
    harness = _TreeLoopHarness(monkeypatch, requests=4, capacity=4)
    harness.run()
    # In a steady step, the target consumes an already-ready proposal, so it
    # must not wait for this step's new draft exchange.  This asserts only
    # rank-local protocol ordering; actual device overlap needs an NPU trace.
    prior_verdict = -1
    steady_steps = 0
    for position, (event, _) in enumerate(harness.events):
        if event != "verdict":
            continue
        step = [name for name, _ in harness.events[prior_verdict + 1 : position]]
        if "draft" in step and "target" in step:
            draft_position = max(index for index, name in enumerate(step) if name == "draft")
            target_position = max(index for index, name in enumerate(step) if name == "target")
            exchange_position = max(index for index, name in enumerate(step) if name == "exchange")
            assert max(draft_position, target_position) < exchange_position
            steady_steps += 1
        prior_verdict = position
    assert steady_steps > 0


@pytest.mark.parametrize("stale_row", [0, -1])
def test_tree_loop_rejects_stale_verdict_before_mutating_any_request(monkeypatch, stale_row):
    harness = _TreeLoopHarness(monkeypatch, requests=4, capacity=4)
    original_states = [state.clone() for state in harness.target_states]

    def stale_epoch():
        controller = harness.controllers[0]
        verified = next(indices for event, indices in reversed(harness.events) if event == "target")
        controller.ready[verified[stale_row]].required_prefix_epoch += 1

    harness.before_verdict = stale_epoch
    with pytest.raises(RuntimeError, match="stale|epoch|prefix"):
        harness.run()
    assert [state.token_ids for state in harness.target_states] == [state.token_ids for state in original_states]
    assert all(state.verification_rounds == 0 for state in harness.target_states)
    assert all(state.continuation_epoch == 0 for state in harness.target_states)


def test_tree_loop_online_admission_prefills_before_decode(monkeypatch):
    harness = _TreeLoopHarness(monkeypatch, requests=3, capacity=2, online_prefill=True)
    results = harness.run()
    assert harness.prefilled == {0, 1, 2}
    assert all(len(result["completion_token_ids"]) == 6 for result in results)


def test_tree_loop_accepts_unknown_request_during_active_decode(monkeypatch):
    harness = _TreeLoopHarness(
        monkeypatch,
        requests=1,
        capacity=2,
        max_tokens=4,
        online_prefill=True,
    )
    admitted = False
    calls = []
    live_params = native.NativeSamplingParams(
        temperature=0,
        max_tokens=4,
        ignore_eos=True,
        slo_tpot_ms=50.0,
        request_id="live-request",
        arrival_ts=time.time(),
    )

    def poll_admission(block):
        nonlocal admitted
        calls.append(block)
        if not admitted:
            admitted = True
            return False, [([777, 778], live_params)]
        return (True, ()) if block else (False, ())

    harness.engine._request_admission_callback = poll_admission
    harness.engine._activate_cache_sequence = Mock()
    harness.engine.graph_runner = SimpleNamespace(set_expected_fia_batch_size=Mock())
    results = harness.run()

    assert len(results) == 2
    assert results[1]["request_id"] == "live-request"
    assert len(results[1]["completion_token_ids"]) == 4
    assert harness.prefilled == {0, 1}
    assert harness.engine._activate_cache_sequence.call_args.args[:2] == (1, [777, 778])
    assert calls[-1] is True
    assert results[0]["spec_rhythm"]["spec_rhythm_live_admitted_requests"] == 1


def test_tree_loop_replicates_actual_draft_confidence_into_runtime_tracker(monkeypatch):
    harness = _TreeLoopHarness(monkeypatch, requests=2, capacity=2)
    harness.run(max_rounds=2)
    controller = harness.controllers[0]
    assert controller.request_states[0].draft_confidence_ema == pytest.approx(0.925)


def test_tree_confidence_envelope_roundtrip_is_identical_on_every_rank(monkeypatch):
    harness = _TreeLoopHarness(monkeypatch, requests=2, capacity=2)
    plans = [
        harness.engine._spec_rhythm_tree_plan(harness.target_states[0], 2),
        harness.engine._spec_rhythm_tree_plan(harness.target_states[1], 4),
    ]
    captured = []

    def broadcast(message, *, src, **kwargs):
        if harness.engine.rank == src:
            captured.append(message.clone())
        else:
            message.copy_(captured[0])

    monkeypatch.setattr(native.dist, "broadcast", broadcast)
    decoded = []
    for rank in range(4):
        harness.engine.rank = rank
        message = native.NativePearlEngine._exchange_spec_rhythm_tree_candidates(
            harness.engine,
            [[151665, 42], [7, 8, 9, 10]] if rank == 0 else None,
            plans,
            [None, 123] if rank == 0 else None,
            [0.123456789, 0.987654321] if rank == 0 else None,
        )
        decoded.append(harness.engine._split_tree_candidates(message, plans))
    assert all(value == decoded[0] for value in decoded)
    assert decoded[0] == (
        [[151665, 42], [7, 8, 9, 10]],
        [None, 123],
        [0.123456789, 0.987654321],
        [[0, 1], [0, 1, 2, 3]],
    )


@pytest.mark.parametrize("corrupt_parent_id", [False, True])
def test_tree_loop_eager_promotion_requires_matching_parent_identity(monkeypatch, corrupt_parent_id):
    harness = _TreeLoopHarness(monkeypatch, requests=2, capacity=2, slo=0.001)
    inspected_eager = []

    def inspect_dependency():
        controller = harness.controllers[0]
        inspected_eager.extend(controller.staged_eager)
        if corrupt_parent_id:
            for index in controller.staged_eager:
                controller.ready[index].proposal_id += 100000

    harness.before_verdict = inspect_dependency
    results = harness.run(max_rounds=2)
    assert inspected_eager, "fixture did not exercise an eager continuation"
    counters = results[0]["spec_rhythm"]
    if corrupt_parent_id:
        assert counters["spec_rhythm_tree_eager_promoted"] == 0
        assert counters["spec_rhythm_tree_eager_dependency_rejections"] > 0
        assert counters["spec_rhythm_tree_eager_invalidated"] > 0
    else:
        assert counters["spec_rhythm_tree_eager_promoted"] > 0
    # Parent verification remains valid even when a continuation is discarded.
    assert len(results[0]["completion_token_ids"]) > 0


def test_tree_loop_respects_online_arrivals_with_a_deterministic_clock(monkeypatch):
    harness = _TreeLoopHarness(
        monkeypatch,
        requests=3,
        capacity=2,
        online_prefill=True,
        arrivals=[100.0, 100.0, 101.0],
    )
    clock = [100.0]
    monkeypatch.setattr(native.time, "time", lambda: clock[0])

    def advance_wait(seconds):
        clock[0] += seconds

    monkeypatch.setattr(native.time, "sleep", advance_wait)
    original_prefill = harness.prefill

    def checked_prefill(rows, states, indices):
        assert all(harness.params[index].arrival_ts <= clock[0] for index in indices)
        return original_prefill(rows, states, indices)

    harness.engine._prefill_and_sample_target_batch = checked_prefill
    results = harness.run()
    assert harness.prefilled == {0, 1, 2}
    assert clock[0] >= 101.0
    assert len(results) == 3


def test_tree_kv_compaction_uses_page_table_slots_not_physical_arithmetic():
    engine = native.NativePearlEngine.__new__(native.NativePearlEngine)
    key_cache = torch.arange(16, dtype=torch.float32).reshape(4, 4, 1, 1)
    value_cache = key_cache.clone() + 100
    engine.model = SimpleNamespace(
        layers=[SimpleNamespace(self_attn=SimpleNamespace(key_cache=key_cache, value_cache=value_cache))]
    )
    # Request 0 crosses from the end of physical page 0 to page 3, then page
    # 1.  Its accepted sibling branch occupies slots 4/5, but logical commit
    # destinations are 12/13.  Request 1 has a different variable node count.
    query_slots = torch.tensor([3, 12, 13, 4, 5, 2, 8, 9], dtype=torch.long)
    accepted = torch.tensor([[2, 3], [1, -1]], dtype=torch.int32)
    engine.compact_tree_round(query_slots, accepted, [4, 2])
    flat_keys = key_cache.flatten()
    flat_values = value_cache.flatten()
    assert flat_keys[[12, 13, 8]].tolist() == [4, 5, 9]
    assert flat_values[[12, 13, 8]].tolist() == [104, 105, 109]
    assert flat_keys[[3, 2]].tolist() == [3, 2]


@pytest.mark.parametrize("online_prefill", [False, True])
def test_tree_loop_delivers_only_committed_chunks_before_final_result(monkeypatch, online_prefill):
    harness = _TreeLoopHarness(monkeypatch, requests=3, capacity=2, max_tokens=5, online_prefill=online_prefill)
    chunks = {index: [] for index in range(3)}
    events = []

    def on_commit(event):
        index = event["request_index"]
        assert event["token_ids"]
        state = harness.target_states[index]
        chunks[index].extend(event["token_ids"])
        assert chunks[index] == state.committed_completion_token_ids[: len(chunks[index])]
        assert len(chunks[index]) <= 5
        assert event["elapsed_seconds"] >= 0.0
        events.append(event)

    harness.engine._token_commit_callback = on_commit
    harness.engine._stream_delivered_counts = {}
    harness.engine._stream_started = time.perf_counter()
    results = harness.run()
    for index, result in enumerate(results):
        assert chunks[index] == result["completion_token_ids"]
        request_events = [event for event in events if event["request_index"] == index]
        assert request_events[-1]["finished"] is True
        assert all(event["request_id"] == f"request-{index}" for event in request_events)
    assert len(events) > len(results)


def test_guarded_delivery_helper_clips_eos_and_does_not_repeat_tokens(monkeypatch):
    harness = _TreeLoopHarness(monkeypatch, requests=1, capacity=1)
    engine = harness.engine
    engine.eos_token_ids = {2}
    engine._stream_delivered_counts = {}
    engine._stream_started = time.perf_counter()
    events = []
    engine._token_commit_callback = events.append
    state = native.PearlPipelineState([1, 11, 2, 13], prompt_length=1, max_tokens=8, ignore_eos=False)

    engine._deliver_committed_tokens(0, harness.params[0], state)
    engine._deliver_committed_tokens(0, harness.params[0], state)

    assert len(events) == 1
    assert events[0]["token_ids"] == [11, 2]
    assert events[0]["finished"] is True
    engine.rank = engine.topology.draft_leader_rank
    engine._stream_delivered_counts = {}
    engine._deliver_committed_tokens(0, harness.params[0], state)
    assert len(events) == 1


@pytest.mark.parametrize("window_ms, expects_eager", [(2.0, False), (16.0, True)])
def test_tree_loop_eager_admission_respects_the_measured_residual_window(monkeypatch, window_ms, expects_eager):
    harness = _TreeLoopHarness(
        monkeypatch,
        requests=2,
        capacity=2,
        max_tokens=12,
        slo=0.001,
        draft_window_ms=window_ms,
    )
    staged = []
    harness.before_verdict = lambda: staged.extend(harness.controllers[0].staged_eager)

    results = harness.run(max_rounds=4)

    assert bool(staged) is expects_eager
    counters = results[0]["spec_rhythm"]
    if expects_eager:
        assert counters["spec_rhythm_tree_eager_promoted"] > 0
    else:
        assert counters["spec_rhythm_tree_eager_promoted"] == 0
        assert counters["spec_rhythm_tree_eager_invalidated"] == 0
    assert harness.window_observations
    assert harness.estimators[0].draft_ms_per_token == pytest.approx(1.0)
    assert harness.estimators[0].target_verify_ms == pytest.approx(window_ms)


def test_tree_loop_does_not_admit_eager_before_window_calibration(monkeypatch):
    harness = _TreeLoopHarness(
        monkeypatch,
        requests=2,
        capacity=2,
        max_tokens=12,
        slo=0.001,
        draft_window_ms=16.0,
        calibrate_window=False,
    )
    staged = []
    harness.before_verdict = lambda: staged.extend(harness.controllers[0].staged_eager)

    harness.run(max_rounds=2)

    assert staged == []
    # The first target observation calibrates the estimator only after that
    # step's work was selected, making it available for the following step.
    assert harness.estimators[0].estimate(normal_tokens=4, max_draft_tokens=8).calibrated


def test_tree_loop_tpot_includes_first_verification_after_prefill_token(monkeypatch):
    harness = _TreeLoopHarness(monkeypatch, requests=2, capacity=2, max_tokens=4, online_prefill=True)
    clock = [0.0]
    original_draft = harness.draft
    original_target = harness.target
    monkeypatch.setattr(native.time, "perf_counter", lambda: clock[0])

    def timed_draft(*args, **kwargs):
        clock[0] += 0.02
        return original_draft(*args, **kwargs)

    def timed_target(*args, **kwargs):
        clock[0] += 0.03
        return original_target(*args, **kwargs)

    harness.engine.draft_tree_forward = timed_draft
    harness.engine.target_tree_forward = timed_target

    results = harness.run()

    # Prefill delivers one token. Warmup costs 20 ms, the first verified
    # cycle costs 20 + 30 ms in this sequential CPU substitute, and commits
    # three more tokens. None of the first cycle may disappear from TPOT.
    assert results[0]["completion_token_ids"][0] == 301
    assert len(results[0]["completion_token_ids"]) == 4
    assert results[0]["observed_tpot_ms"] == pytest.approx(70.0 / 3.0)
    runtime = harness.controllers[0].request_states[0]
    assert runtime.delivered_tokens == 4
    assert runtime.decode_start_elapsed_ms == 0.0


def test_online_refill_pause_counts_for_old_active_requests_not_new_prefill_tokens(monkeypatch):
    harness = _TreeLoopHarness(monkeypatch, requests=3, capacity=2, max_tokens=4, online_prefill=True)
    clock = [0.0]
    original_draft = harness.draft
    original_target = harness.target
    original_prefill = harness.prefill
    monkeypatch.setattr(native.time, "perf_counter", lambda: clock[0])

    def timed_draft(*args, **kwargs):
        clock[0] += 0.02
        return original_draft(*args, **kwargs)

    def timed_target(*args, **kwargs):
        clock[0] += 0.03
        return original_target(*args, **kwargs)

    def timed_prefill(rows, states, indices):
        if indices == [2]:
            clock[0] += 0.1
        return original_prefill(rows, states, indices)

    harness.engine.draft_tree_forward = timed_draft
    harness.engine.target_tree_forward = timed_target
    harness.engine._prefill_and_sample_target_batch = timed_prefill

    results = harness.run()

    # Request 0 finishes after 20 ms warmup + 50 ms verify cycle. Its
    # replacement request 2 takes 100 ms to prefill while request 1 waits.
    # The next 50 ms verification must charge that pause to request 1,
    # but not to request 2 before its first output or the finished request 0.
    assert [(kind, indices) for kind, indices in harness.events if kind == "prefill"] == [
        ("prefill", (0, 1)),
        ("prefill", (2,)),
    ]
    assert [len(row["completion_token_ids"]) for row in results] == [4, 4, 4]
    assert results[0]["observed_tpot_ms"] == pytest.approx(70.0 / 3.0)
    assert results[1]["observed_tpot_ms"] == pytest.approx(220.0 / 3.0)
    assert results[2]["observed_tpot_ms"] == pytest.approx(80.0 / 3.0)
    assert results[1]["spec_rhythm"]["spec_rhythm_online_refill_ms"] == pytest.approx(100.0)


def test_native_callback_failure_is_deferred_until_real_tree_loop_drains(monkeypatch):
    harness = _TreeLoopHarness(monkeypatch, requests=2, capacity=2, max_tokens=6)
    # Use the public native wrapper with the actual tree loop as its compute
    # implementation; device boundaries remain the same CPU substitutes.
    harness.engine._generate_batch_impl = lambda *args, **kwargs: harness.run()
    callback = Mock(side_effect=ValueError("client callback failed"))

    with pytest.raises(RuntimeError, match="after collective drain") as error:
        harness.engine.generate_batch([[1], [2]], harness.params, on_token_commit=callback)

    assert isinstance(error.value.__cause__, ValueError)
    callback.assert_called_once()
    assert all(len(state.committed_completion_token_ids) >= 6 for state in harness.target_states)
    harness.engine._release_cache.assert_called_once()
    assert harness.engine._token_commit_callback is None
    assert harness.engine._token_commit_error is None


def test_native_generate_clears_callback_after_model_error(monkeypatch):
    harness = _TreeLoopHarness(monkeypatch, requests=1, capacity=1)
    harness.engine._generate_batch_impl = Mock(side_effect=RuntimeError("model failure"))
    callback = Mock()

    with pytest.raises(RuntimeError, match="model failure"):
        harness.engine.generate_batch([[1]], harness.params, on_token_commit=callback)

    callback.assert_not_called()
    assert harness.engine._token_commit_callback is None
    assert harness.engine._token_commit_error is None


def test_native_generate_does_not_reuse_callback_from_previous_batch(monkeypatch):
    harness = _TreeLoopHarness(monkeypatch, requests=1, capacity=1)
    state = native.PearlPipelineState([1, 11], prompt_length=1, max_tokens=1)

    def completed_batch(*args, **kwargs):
        harness.engine._deliver_committed_tokens(0, harness.params[0], state)
        return [{"completion_token_ids": [11]}]

    harness.engine._generate_batch_impl = completed_batch
    callback = Mock()
    result = harness.engine.generate_batch([[1]], harness.params, on_token_commit=callback)
    assert result == [{"completion_token_ids": [11]}]
    callback.assert_called_once()
    assert harness.engine._token_commit_callback is None
    result = harness.engine.generate_batch([[1]], harness.params)
    assert result == [{"completion_token_ids": [11]}]
    callback.assert_called_once()


def test_native_online_prefill_respects_arrivals_when_requests_fit_one_batch(monkeypatch):
    harness = _TreeLoopHarness(
        monkeypatch,
        requests=2,
        capacity=2,
        max_tokens=4,
        online_prefill=True,
        arrivals=[100.0, 101.0],
    )
    harness.engine.graph_runner = SimpleNamespace(set_expected_fia_batch_size=Mock())
    harness.engine._allocate_cache = Mock()
    harness.engine._capture_decode_graphs = Mock()
    clock = [100.0]
    monkeypatch.setattr(native.time, "time", lambda: clock[0])

    def advance_wait(seconds):
        clock[0] += seconds

    monkeypatch.setattr(native.time, "sleep", advance_wait)
    original_prefill = harness.prefill

    def checked_prefill(rows, states, indices):
        assert all(harness.params[index].arrival_ts <= clock[0] for index in indices)
        return original_prefill(rows, states, indices)

    harness.engine._prefill_and_sample_target_batch = checked_prefill
    results = harness.engine.generate_batch([[100, 200], [101, 201]], harness.params)

    assert clock[0] >= 101.0
    assert [indices for event, indices in harness.events if event == "prefill"] == [(0,), (1,)]
    assert all(len(result["completion_token_ids"]) == 4 for result in results)
    harness.engine._capture_decode_graphs.assert_not_called()


@pytest.mark.parametrize("temperature,draft_temperature", [(0.7, 0.0), (0.0, 0.7)])
def test_native_tree_routes_nonzero_sampling_to_tree_workers(monkeypatch, temperature, draft_temperature):
    harness = _TreeLoopHarness(monkeypatch, requests=1, capacity=1)
    harness.params[0] = replace(
        harness.params[0],
        temperature=temperature,
        draft_temperature=draft_temperature,
    )
    for state in (harness.draft_states[0], harness.target_states[0]):
        state.temperature = temperature
        state.draft_temperature = draft_temperature
    observed = {}
    original_draft = harness.draft
    original_target = harness.target

    def sampled_draft(*args, draft_temperatures=None, **kwargs):
        observed["draft_temperatures"] = draft_temperatures
        return original_draft(*args, **kwargs)

    def sampled_target(*args, temperatures=None, **kwargs):
        observed["target_temperatures"] = temperatures
        return original_target(*args, **kwargs)

    harness.engine.draft_tree_forward = sampled_draft
    harness.engine.target_tree_forward = sampled_target
    results = harness.run()

    assert len(results) == 1
    assert observed.get("target_temperatures") == ([temperature] if temperature else None)
    assert observed.get("draft_temperatures") == ([draft_temperature] if draft_temperature else None)


@pytest.mark.parametrize("temperature,draft_temperature", [(0.7, 0.0), (0.0, 0.7)])
def test_ordinary_linear_pearl_still_accepts_nonzero_sampling(monkeypatch, temperature, draft_temperature):
    harness = _TreeLoopHarness(monkeypatch, requests=1, capacity=1)
    harness.engine.config = replace(
        harness.engine.config,
        enable_spec_rhythm=False,
        spec_rhythm_tree_width=1,
        spec_rhythm_tree_depth=1,
    )
    harness.engine.graph_runner = SimpleNamespace(set_expected_fia_batch_size=Mock())
    harness.engine._allocate_cache = Mock(side_effect=RuntimeError("reached cache allocation"))
    params = native.NativeSamplingParams(temperature=temperature, draft_temperature=draft_temperature, max_tokens=2)

    with pytest.raises(RuntimeError, match="reached cache allocation"):
        harness.engine.generate_batch([[1]], params)

    harness.engine._allocate_cache.assert_called_once()


@pytest.mark.parametrize("reason", ["explicit_budget", "streaming_callback"])
def test_linear_specslo_keeps_control_plane_for_budget_or_streaming(monkeypatch, reason):
    harness = _TreeLoopHarness(monkeypatch, requests=1, capacity=1)
    harness.engine.config = replace(
        harness.engine.config,
        spec_rhythm_tree_width=1,
        spec_rhythm_tree_depth=1,
        spec_rhythm_min_gamma=4,
        spec_rhythm_verification_budget=4 if reason == "explicit_budget" else None,
    )
    harness.engine.graph_runner = SimpleNamespace(set_expected_fia_batch_size=Mock())
    harness.engine._allocate_cache = Mock()
    harness.engine._capture_decode_graphs = Mock()
    harness.engine._generate_spec_rhythm_decode = Mock(return_value=[])

    result = harness.engine.generate_batch(
        [[1]],
        native.NativeSamplingParams(temperature=0.0, max_tokens=2),
        on_token_commit=Mock() if reason == "streaming_callback" else None,
    )

    assert result == []
    harness.engine._generate_spec_rhythm_decode.assert_called_once()


def _synthetic_tree_profile(verification_requests, mode="eager", max_model_len=256, tree_latency=10.1):
    common = {
        "batch_size": 2,
        "verification_requests": verification_requests,
        "context_len": 512,
        "warmup_iterations": 2,
        "timed_graph_captures": 0,
        "timed_graph_replays": 3,
    }
    counts = [4 // verification_requests] * verification_requests
    return build_roofline_table(
        {
            "metadata": {
                "model": "cpu-target",
                "target_tensor_parallel_size": 3,
                "execution_mode": mode,
                "hardware": "cpu-fixture",
                "measurement_scope": "target_forward",
                "verification_attention_backends": ["fused_infer_attention_tree_v1"],
                "ar_attention_backend": "paged_attention_v1",
                "ar_comparator": "standard_decode_full_active_batch",
                **runtime_source_fingerprints(),
                "tree_fia_sparse_mode": 1,
                "tree_fia_inner_precise": 2,
                "max_model_len": max_model_len,
                "tree_width": 2,
                "tree_depth": 2,
            },
            "measurements": [
                {
                    **common,
                    "kind": "ar",
                    "attention_backend": "paged_attention_v1",
                    "verification_requests": 2,
                    "candidate_counts": [0, 0],
                    "physical_query_tokens": 2,
                    "latency_ms": [10.0] * 3,
                },
                {
                    **common,
                    "kind": "packed_tree",
                    "attention_backend": "fused_infer_attention_tree_v1",
                    "candidate_counts": counts,
                    "physical_query_tokens": verification_requests + sum(counts),
                    "latency_ms": [tree_latency] * 3,
                },
            ],
        }
    )


def test_measured_zero_tree_roof_runs_target_only_without_gamma_fallback(monkeypatch):
    harness = _TreeLoopHarness(monkeypatch, requests=2, capacity=2, max_tokens=3)
    profile = _synthetic_tree_profile(1, tree_latency=20.0)
    assert profile["roofline"] == {"2:1": 0}
    tail_evidence = dict(profile["evidence"][0])
    tail_evidence.update(lookup_key="1:1", batch_size=1, ar_requests=1, verification_requests=1)
    profile["roofline"]["1:1"] = 0
    profile["target_only_fallback_lookup_keys"] = ["1:1", "2:1"]
    profile["evidence"].append(tail_evidence)
    harness.engine.config = replace(
        harness.engine.config,
        spec_rhythm_verification_budget=None,
        spec_rhythm_roofline=profile,
    )
    harness.engine._prepare_attention_metadata = lambda rows, positions, use_fia: (
        torch.tensor(positions),
        SimpleNamespace(use_fused_infer_attention=False, attention_mask=None),
    )

    class TargetModel:
        def __call__(self, input_ids, positions, metadata):
            return input_ids

        @staticmethod
        def compute_greedy_tokens(hidden, vocabulary_size):
            return hidden + 1

    harness.engine.model = TargetModel()
    harness.engine.draft_vocab_size = 1000
    results = harness.run(max_rounds=10)

    assert [row["completion_token_ids"] for row in results] == [[201, 202, 203], [202, 203, 204]]
    assert all(row["spec_rhythm"]["spec_rhythm_last_verification_roof"] == 0 for row in results)
    assert results[0]["spec_rhythm"]["spec_rhythm_target_only_fallback_rounds"] == 6
    assert results[0]["spec_rhythm"]["spec_rhythm_target_only_fallback_tokens"] == 6
    assert not any(event in ("draft", "target", "exchange", "verdict") for event, _ in harness.events)
    assert [state.token_ids for state in harness.draft_states] == [state.token_ids for state in harness.target_states]


@pytest.mark.parametrize("profiled_rows", [1, 2])
def test_real_tree_loop_binds_strict_profile_to_actual_target_rows_and_only_active_context(monkeypatch, profiled_rows):
    harness = _TreeLoopHarness(monkeypatch, requests=3, capacity=2, max_tokens=12)
    harness.engine.config = replace(
        harness.engine.config,
        max_model_len=1024,
        spec_rhythm_verification_budget=None,
        spec_rhythm_roofline=_synthetic_tree_profile(profiled_rows, max_model_len=1024),
    )
    # A queued request has a much longer prompt, but is not admitted yet.
    # It must not change the active B=2 lookup from bucket 1 to bucket 2.
    pending = native.PearlPipelineState([77] * 600, prompt_length=600, max_tokens=12, ignore_eos=True)
    harness.draft_states[2] = pending
    harness.target_states[2] = pending.clone()
    calls = []
    validate = ProfiledRoofline.validate_execution

    def checked(profile, batch_size, context_len, verification_requests):
        calls.append((batch_size, context_len, verification_requests))
        return validate(profile, batch_size, context_len, verification_requests)

    monkeypatch.setattr(ProfiledRoofline, "validate_execution", checked)
    if profiled_rows == 1:
        results = harness.run(max_rounds=2)
        assert results[0]["completion_token_ids"] == [10, 11, 999]
        assert [indices for event, indices in harness.events if event == "target"] == [(0,)]
    else:
        with pytest.raises(ValueError, match="Unprofiled.*physical target rows"):
            harness.run(max_rounds=2)
        assert not any(event == "target" for event, _ in harness.events)
        assert harness.target_states[0].committed_completion_token_ids == []
        harness.engine.compact_tree_round.assert_not_called()
    assert calls == [(2, 2, 1)]


def test_real_tree_loop_uses_target_only_for_unprofiled_physical_home_when_profile_requests_it(monkeypatch):
    harness = _TreeLoopHarness(monkeypatch, requests=2, capacity=2, max_tokens=3)
    profile = _synthetic_tree_profile(2)
    profile["metadata"]["unprofiled_execution_policy"] = "target_only"
    harness.engine.config = replace(
        harness.engine.config,
        spec_rhythm_verification_budget=None,
        spec_rhythm_roofline=profile,
    )
    harness.engine._prepare_attention_metadata = lambda rows, positions, use_fia: (
        torch.tensor(positions),
        SimpleNamespace(use_fused_infer_attention=False, attention_mask=None),
    )

    class TargetModel:
        def __call__(self, input_ids, positions, metadata):
            return input_ids

        @staticmethod
        def compute_greedy_tokens(hidden, vocabulary_size):
            return hidden + 1

    harness.engine.model = TargetModel()
    harness.engine.draft_vocab_size = 1000
    results = harness.run(max_rounds=10)

    assert [row["completion_token_ids"] for row in results] == [[201, 202, 203], [202, 203, 204]]
    counters = results[0]["spec_rhythm"]
    assert counters["spec_rhythm_unprofiled_target_only_fallback_rounds"] == 6
    assert counters["spec_rhythm_unprofiled_target_only_fallback_tokens"] == 6


def test_tree_loop_eos_clips_runtime_state_and_invalidates_unneeded_eager(monkeypatch):
    harness = _TreeLoopHarness(monkeypatch, requests=2, capacity=2, max_tokens=12, slo=0.001)
    harness.engine.eos_token_ids = {11, 21}
    harness.params = [replace(params, ignore_eos=False) for params in harness.params]
    for state in [*harness.draft_states, *harness.target_states]:
        state.ignore_eos = False
    commits = []
    harness.engine._token_commit_callback = commits.append
    harness.engine._stream_delivered_counts = {}
    harness.engine._stream_started = time.perf_counter()

    results = harness.run()

    assert [row["completion_token_ids"] for row in results] == [[10, 11], [20, 21]]
    assert [row.committed_completion_token_ids for row in harness.target_states] == [[10, 11], [20, 21]]
    assert [row.committed_completion_token_ids for row in harness.draft_states] == [[10, 11], [20, 21]]
    assert all(state.delivered_tokens == 2 for state in harness.controllers[0].request_states.values())
    assert [event["token_ids"] for event in commits] == [[10, 11], [20, 21]]
    assert all(event["finished"] for event in commits)
    assert not harness.controllers[0].ready
    assert not harness.controllers[0].staged_eager
    assert results[0]["spec_rhythm"]["spec_rhythm_tree_eager_promoted"] == 0
    assert results[0]["spec_rhythm"]["spec_rhythm_tree_eager_invalidated"] > 0


def test_real_loop_strict_graph_fallback_fails_before_new_proposal_or_verdict_publication(monkeypatch):
    harness = _TreeLoopHarness(monkeypatch, requests=2, capacity=2, max_tokens=12)
    profile = _synthetic_tree_profile(1, mode="graph")
    harness.engine.config = replace(
        harness.engine.config,
        enforce_eager=False,
        spec_rhythm_verification_budget=None,
        spec_rhythm_roofline=profile,
    )
    original_target = harness.target

    def fallback_target(*args, **kwargs):
        return {**original_target(*args, **kwargs), "used_aclgraph": False}

    harness.engine.target_tree_forward = fallback_target
    with pytest.raises(RuntimeError, match="Strict SpecSLO graph roofline.*target eager fallback"):
        harness.run(max_rounds=2)

    # Warmup published request 0. The next step computes draft 1 / target 0,
    # but a failed target graph cannot publish draft 1 or commit target 0.
    assert harness.events[-1] == ("target", (0,))
    assert not any(event == "verdict" for event, _ in harness.events)
    assert set(harness.controllers[0].ready) == {0}
    assert all(not state.committed_completion_token_ids for state in harness.target_states)
    harness.engine.compact_tree_round.assert_not_called()
