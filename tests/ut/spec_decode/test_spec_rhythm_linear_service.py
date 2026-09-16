# SPDX-License-Identifier: Apache-2.0
"""CPU service-loop regressions for linear SpecRhythm.

The model and transport boundaries are substituted; admission, timing,
controller planning, request-state transitions and result construction are the
production implementations.  These tests make no NPU-overlap claim.
"""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.spec_decode.pearl import native_engine as native
from vllm_ascend.spec_decode.pearl.spec_rhythm import (
    PipelinePhase,
    SpecRhythmPipelineController,
    SpecRhythmRuntimeState,
)
from vllm_ascend.spec_decode.pearl.topology import PearlTopology


class _LinearLoopHarness:
    def __init__(
        self,
        monkeypatch,
        *,
        max_tokens=(6, 6, 6, 6),
        capacity=4,
        online_prefill=False,
        arrivals=None,
        target_fallback: bool | int = False,
        slo=None,
        profile_host_steps=0,
        remote_draft_compute_ms=4.0,
        full_window=False,
        eager_cross_graph_bucket=False,
        idle_residual_eager=False,
        prefill_coalesce_min_requests=1,
        prefill_coalesce_max_wait_ms=0.0,
        prefill_token_chunk_size=0,
        prompt_lengths=None,
        is_draft=False,
        bonus_token=False,
    ):
        self.events = []
        self.prefilled = set()
        self.synchronize = Mock()
        self.broadcast_sizes = []
        self.remote_draft_compute_ms = remote_draft_compute_ms
        self.proposal_batches = []
        request_count = len(max_tokens)

        def broadcast(value, *args, **kwargs):
            self.broadcast_sizes.append(int(value.numel()))
            if kwargs.get("async_op"):
                if not is_draft:
                    value.fill_(777)
                    value[-1] = round(self.remote_draft_compute_ms * 1000.0)
                return SimpleNamespace(wait=lambda: None)
            if is_draft and value.dtype == torch.long and value.ndim == 1:
                # Emulate the exact token produced by target leader during
                # the tiny-batch AR fallback.
                value.fill_(888)
            return None

        monkeypatch.setattr(native.dist, "broadcast", broadcast)
        monkeypatch.setattr(native.dist, "barrier", lambda *args, **kwargs: None)
        monkeypatch.setattr(torch.npu, "synchronize", self.synchronize)
        monkeypatch.delenv("VLLM_ASCEND_PEARL_NPU_PROFILE_DIR", raising=False)

        self.engine = native.NativePearlEngine.__new__(native.NativePearlEngine)
        self.engine.config = native.NativePearlConfig(
            "cpu-draft",
            "cpu-target",
            1,
            3,
            4,
            256,
            max(max_tokens),
            max_num_seqs=capacity,
            enable_prefix_caching=not bool(prefill_token_chunk_size),
            enable_spec_rhythm=True,
            enable_continuous_batching=True,
            enable_preemptive_scheduling=True,
            spec_rhythm_linear_full_window=full_window,
            spec_rhythm_linear_bonus_token=bonus_token,
            spec_rhythm_linear_eager_cross_graph_bucket=(eager_cross_graph_bucket),
            spec_rhythm_linear_idle_residual_eager=idle_residual_eager,
            spec_rhythm_online_prefill=online_prefill,
            spec_rhythm_prefill_coalesce_min_requests=(prefill_coalesce_min_requests),
            spec_rhythm_prefill_coalesce_max_wait_ms=(prefill_coalesce_max_wait_ms),
            spec_rhythm_prefill_token_chunk_size=prefill_token_chunk_size,
            spec_rhythm_min_gamma=4,
            spec_rhythm_target_fallback_max_batch=(capacity if target_fallback is True else int(target_fallback)),
            spec_rhythm_cpu_verdict=True,
            profile_host_decode_steps=profile_host_steps,
            enforce_eager=True,
        )
        self.engine.topology = PearlTopology.from_tensor_parallel_sizes(1, 3)
        self.engine.rank = (
            self.engine.topology.draft_leader_rank if is_draft else self.engine.topology.target_leader_rank
        )
        self.engine.is_draft = is_draft
        self.engine.device = torch.device("cpu")
        self.engine.eos_token_ids = frozenset()
        self.engine.groups = SimpleNamespace(
            is_verification_worker=True,
            verification_group=None,
            correction_group=None,
        )
        self.engine.gamma = 4
        self.engine.graph_runner = SimpleNamespace(set_expected_fia_batch_size=Mock())
        self.engine.greedy_verification_layouts = {}
        self.engine.cache_allocation = SimpleNamespace(num_cached_tokens=[0] * request_count)
        # The scheduler harness is intentionally device-free.  Exercise the
        # online cache lifecycle boundaries without constructing physical NPU
        # page tables; NativePrefixCache has dedicated allocation/recycling
        # tests below.
        self.engine._activate_cache_sequence = Mock()
        self.engine._release_cache_sequence = Mock(return_value=0)
        self.engine._release_cache = Mock()
        self.engine.graph_metrics = lambda: {}
        self.engine._deliver_committed_tokens = lambda *args, **kwargs: None
        self.engine._prefill_and_sample_target_batch = self.prefill
        self.engine._prefill_spec_rhythm_token_chunk_batch = self.prefill_token_chunk
        self.engine._target_round_outputs_batch = self.target
        self.engine._target_full_window_outputs_batch = self.target_full_window
        self.engine._exchange_spec_rhythm_device_proposals = self.exchange
        self.engine._broadcast_device_round_result = self.correction
        self.engine._run_packed_sample = self.target_only
        self.engine._run_packed_hidden = Mock()
        if is_draft:
            self.engine._draft_spec_rhythm_device_batch = self.draft

        self.params = [
            native.NativeSamplingParams(
                temperature=0.0,
                max_tokens=limit,
                ignore_eos=True,
                slo_tpot_ms=slo,
                arrival_ts=None if arrivals is None else arrivals[index],
                request_id=f"request-{index}",
            )
            for index, limit in enumerate(max_tokens)
        ]
        prompt_lengths = prompt_lengths or (1,) * request_count
        if len(prompt_lengths) != request_count:
            raise ValueError("prompt_lengths must be request-aligned")
        initial_token = [] if online_prefill else [301]
        self.draft_states = [
            native.PearlPipelineState(
                [
                    *(1000 * (index + 1) + offset for offset in range(prompt_lengths[index])),
                    *initial_token,
                ],
                prompt_length=prompt_lengths[index],
                max_tokens=max_tokens[index],
                ignore_eos=True,
                slo_tpot_ms=slo,
            )
            for index in range(request_count)
        ]
        self.target_states = [state.clone() for state in self.draft_states]

    def prefill(self, rows, states, indices, **_kwargs):
        self.events.append(("prefill", tuple(indices)))
        self.prefilled.update(indices)
        return [301 + index for index in indices]

    def prefill_token_chunk(
        self,
        prompts,
        states,
        chunks,
        **_kwargs,
    ):
        spans = tuple(
            (
                chunk.request_index,
                chunk.start,
                chunk.end,
                chunk.completes_prompt,
            )
            for chunk in chunks
        )
        self.events.append(("prefill-token-chunk", spans))
        for chunk in chunks:
            state = states[chunk.request_index]
            assert state.committed_length == state.prompt_length
            assert len(state.token_ids) == state.prompt_length
            assert len(prompts[chunk.request_index]) == chunk.prompt_length
        completed = {chunk.request_index: 301 + chunk.request_index for chunk in chunks if chunk.completes_prompt}
        self.prefilled.update(completed)
        return completed

    def target_only(self, input_ids, indices, positions, temperatures, **kwargs):
        self.events.append(("target-only", tuple(indices)))
        return torch.full((len(indices),), 888, dtype=torch.long)

    def draft(
        self,
        states,
        indices,
        budgets,
        *,
        verification_sizes,
        verification_prefixes,
        full_window,
    ):
        self.events.append(("draft", tuple(indices)))
        windows = torch.full((len(indices), self.engine.gamma), 777, dtype=torch.long)
        return windows.flatten(), windows, torch.ones(len(indices))

    def target(self, states, indices, verification_sizes):
        self.events.append(("target", tuple(indices)))
        size = sum(verification_sizes)
        return torch.full((size,), 777, dtype=torch.long), None

    def target_full_window(self, states, indices, payloads, *, proposal_matrix=None):
        self.events.append(("target-full-window", tuple(indices)))
        if proposal_matrix is not None:
            assert torch.equal(
                proposal_matrix,
                torch.stack([payload.next_tokens for payload in payloads]),
            )
        rows = [payload.verification_tokens for payload in payloads]
        if self.engine.config.spec_rhythm_linear_bonus_token:
            rows = [
                torch.cat(
                    (
                        row,
                        torch.tensor(
                            [900 + payload.ticket.request_index],
                            dtype=row.dtype,
                            device=row.device,
                        ),
                    )
                )
                for row, payload in zip(rows, payloads)
            ]
        return torch.cat(rows), None

    def exchange(
        self,
        verification,
        windows,
        confidences,
        tickets,
        verification_sizes,
        *,
        draft_compute_ms=None,
    ):
        self.events.append(("exchange", tuple(ticket.request_index for ticket in tickets)))
        self.proposal_batches.append(tuple((ticket.request_index, ticket.gamma, ticket.eager) for ticket in tickets))
        count = len(tickets)
        if not count:
            return None, [], None
        verification_count = sum(verification_sizes)
        include_timing = draft_compute_ms is not None
        message = torch.zeros(
            7 * count + verification_count + count * self.engine.gamma + int(include_timing),
            dtype=torch.long,
        )
        message[7 * count : 7 * count + verification_count] = 777
        continuation_start = 7 * count + verification_count
        message[continuation_start : continuation_start + count * self.engine.gamma] = 777
        timing = None
        if include_timing:
            message[-1] = round(self.remote_draft_compute_ms * 1000.0)
            timing = message[-1]
        return message, [torch.tensor(1_000_000) for _ in tickets], timing

    def correction(
        self,
        verdict,
        draft_message,
        verification_size,
        batch_size,
        local_next_windows,
        *,
        next_window_sizes,
        **kwargs,
    ):
        self.events.append(("correction", batch_size))
        self.engine._last_device_round_extra_values = [1.0] * batch_size
        if verdict is None:
            accepted = list(next_window_sizes)
            corrections = [None] * batch_size
            bonuses = [None] * batch_size
        else:
            accepted = [int(value) for value in verdict[:, 0].tolist()]
            result_tokens = [int(value) for value in verdict[:, 1].tolist()]
            if self.engine.config.spec_rhythm_linear_bonus_token:
                corrections = [
                    token_id if accepted_count < expected else None
                    for accepted_count, expected, token_id in zip(
                        accepted,
                        next_window_sizes,
                        result_tokens,
                    )
                ]
                bonuses = [
                    token_id if accepted_count == expected and token_id >= 0 else None
                    for accepted_count, expected, token_id in zip(
                        accepted,
                        next_window_sizes,
                        result_tokens,
                    )
                ]
            else:
                corrections = [None if token_id < 0 else token_id for token_id in result_tokens]
                bonuses = [None] * batch_size
        self.engine._last_device_round_bonus_tokens = bonuses
        return accepted, corrections, [[777] * int(size) for size in next_window_sizes]

    def run(self, *, continuous_batching=False, max_rounds=None):
        return self.engine._generate_spec_rhythm_decode(
            draft_states=self.draft_states,
            target_states=self.target_states,
            request_params=self.params,
            initial_batch_size=self.engine.config.max_num_seqs,
            continuous_batching=continuous_batching,
            prefill_elapsed=0.0,
            started=native.time.perf_counter(),
            max_rounds=max_rounds,
            prefilled_indices=(None if self.engine.config.spec_rhythm_online_prefill else set(range(len(self.params)))),
        )


def test_underfilled_online_trace_still_honors_arrival_gate(monkeypatch):
    wall = [100.0]
    sleeps = []
    monkeypatch.setattr(native.time, "time", lambda: wall[0])

    def advance(_delay):
        sleeps.append(_delay)
        wall[0] = 112.0

    monkeypatch.setattr(native.time, "sleep", advance)
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(2,),
        capacity=4,
        online_prefill=True,
        arrivals=(112.0,),
        target_fallback=True,
        slo=40.0,
    )

    results = harness.run(continuous_batching=False)

    assert sleeps, "an under-filled online batch must wait for its replay arrival"
    assert harness.events[0] == ("prefill", (0,))
    diagnostics = results[0]["spec_rhythm"]
    assert diagnostics["spec_rhythm_initial_min_arrival_delta_ms"] == pytest.approx(12_000.0)
    assert diagnostics["spec_rhythm_initial_max_arrival_delta_ms"] == pytest.approx(12_000.0)
    assert diagnostics["spec_rhythm_first_ready_requests"] == 1
    assert diagnostics["spec_rhythm_max_admission_batch_size"] == 1
    assert diagnostics["spec_rhythm_prefill_coalesce_enabled"] == 0
    assert diagnostics["spec_rhythm_prefill_coalesce_deferred_polls"] == 0
    assert results[0]["arrival_ts"] == 112.0


def test_token_chunk_planner_is_fair_capped_and_cursor_pure():
    lengths = {0: 7, 1: 2, 2: 5}
    cursors = {0: 0, 1: 0, 2: 0}

    first = native.plan_spec_rhythm_prefill_token_chunk(
        [0, 1, 2],
        prompt_lengths=lengths,
        cursors=cursors,
        token_cap=6,
    )

    assert cursors == {0: 0, 1: 0, 2: 0}
    assert [(chunk.request_index, chunk.start, chunk.end, chunk.completes_prompt) for chunk in first] == [
        (0, 0, 2, False),
        (1, 0, 2, True),
        (2, 0, 2, False),
    ]
    assert sum(chunk.token_count for chunk in first) == 6

    cursors.update({chunk.request_index: chunk.end for chunk in first})
    second = native.plan_spec_rhythm_prefill_token_chunk(
        [0, 1, 2],
        prompt_lengths=lengths,
        cursors=cursors,
        token_cap=6,
    )
    assert [(chunk.request_index, chunk.start, chunk.end, chunk.completes_prompt) for chunk in second] == [
        (0, 2, 5, False),
        (2, 2, 5, True),
    ]


def test_fixed_gamma_identity_budget_preserves_selected_row_order():
    normal, eager = native._fixed_gamma_identity_budgets(
        [3, 1],
        [4, 2],
        gamma=4,
        verification_roof=20,
    )

    assert list(normal.items()) == [(3, 4), (1, 4)]
    assert list(eager.items()) == [(4, 4), (2, 4)]


@pytest.mark.parametrize(
    ("normal", "eager", "roof", "message"),
    [
        ([0, 0], [], 8, "duplicate"),
        ([0], [0], 8, "both normal and eager"),
        ([0, 1], [2], 8, "exceeds"),
    ],
)
def test_fixed_gamma_identity_budget_rejects_invalid_envelopes(
    normal,
    eager,
    roof,
    message,
):
    with pytest.raises(ValueError, match=message):
        native._fixed_gamma_identity_budgets(
            normal,
            eager,
            gamma=4,
            verification_roof=roof,
        )


def test_slo_fixed_gamma_full_window_skips_general_bound_and_shape(monkeypatch):
    monkeypatch.delenv(
        "VLLM_ASCEND_SPECRHYTHM_VALIDATE_MAILBOX",
        raising=False,
    )
    bound = Mock(side_effect=AssertionError("fixed path called linear rebuild"))
    shape = Mock(side_effect=AssertionError("fixed path called general shaper"))
    monkeypatch.setattr(
        native.NativePearlEngine,
        "_bound_spec_rhythm_linear_ready",
        bound,
    )
    monkeypatch.setattr(native.SpecRhythmBudgetShaper, "shape", shape)
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(9, 9, 9, 9),
        capacity=4,
        slo=40.0,
        full_window=True,
    )

    result = harness.run(max_rounds=6)[0]
    metrics = result["spec_rhythm"]

    bound.assert_not_called()
    shape.assert_not_called()
    assert harness.proposal_batches
    assert all(gamma == 4 for batch in harness.proposal_batches for _, gamma, _ in batch)
    assert metrics["spec_rhythm_fixed_gamma_scheduler_fast_path"] == 1
    assert metrics["spec_rhythm_fixed_gamma_scheduler_fast_path_cycles"] == 5
    assert metrics["spec_rhythm_linear_rebuilt_requests"] == 0
    assert metrics["spec_rhythm_linear_discarded_ready"] == 0
    assert metrics["spec_rhythm_linear_invalidated_eager"] == 0
    assert metrics["spec_rhythm_linear_discarded_unverified_tokens"] == 0


def test_token_chunk_planner_rotates_when_cap_is_smaller_than_row_count():
    lengths = {0: 2, 1: 2, 2: 2}
    cursors = {0: 0, 1: 0, 2: 0}
    observed = []

    for _ in range(3):
        chunks = native.plan_spec_rhythm_prefill_token_chunk(
            [0, 1, 2],
            prompt_lengths=lengths,
            cursors=cursors,
            token_cap=1,
        )
        assert len(chunks) == 1
        observed.append(chunks[0].request_index)
        cursors[chunks[0].request_index] = chunks[0].end

    assert observed == [0, 1, 2]


def test_token_chunk_prefill_samples_only_prompt_final_rows(monkeypatch):
    monkeypatch.setattr(native.dist, "broadcast", lambda *_args, **_kwargs: None)
    engine = native.NativePearlEngine.__new__(native.NativePearlEngine)
    engine.config = SimpleNamespace(
        enable_prefix_caching=False,
        spec_rhythm_prefill_token_chunk_size=4,
    )
    engine.cache_allocation = SimpleNamespace(num_cached_tokens=[0, 0])
    engine.device = torch.device("cpu")
    engine.is_draft = False
    engine._run_packed_hidden = Mock()
    engine._run_packed_sample = Mock(return_value=torch.tensor([901]))
    engine._vote_spec_rhythm_prefill_finiteness = Mock()
    engine.topology = PearlTopology.from_tensor_parallel_sizes(1, 3)
    engine.groups = SimpleNamespace(verification_coordination_group=None)
    states = {
        index: native.PearlPipelineState(tokens, len(tokens))
        for index, tokens in {0: [10, 11, 12], 1: [20, 21, 22]}.items()
    }
    prompts = {index: state.token_ids for index, state in states.items()}

    partial = engine._prefill_spec_rhythm_token_chunk_batch(
        prompts,
        states,
        (
            native.SpecRhythmPrefillTokenChunk(0, 0, 2, 3),
            native.SpecRhythmPrefillTokenChunk(1, 0, 2, 3),
        ),
    )

    assert partial == {}
    engine._run_packed_hidden.assert_called_once()
    engine._run_packed_sample.assert_not_called()

    engine._run_packed_hidden.reset_mock()
    completed = engine._prefill_spec_rhythm_token_chunk_batch(
        prompts,
        states,
        (native.SpecRhythmPrefillTokenChunk(0, 2, 3, 3),),
    )

    assert completed == {0: 901}
    engine._run_packed_hidden.assert_not_called()
    sample_args = engine._run_packed_sample.call_args
    assert sample_args.args[:3] == ([12], [0], [2])
    assert sample_args.kwargs["logit_indices"] == [0]


def test_cross_cycle_token_chunk_prefill_is_private_until_final_chunk(
    monkeypatch,
):
    monkeypatch.setattr(native.time, "time", lambda: 100.0)
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(30, 5),
        capacity=1,
        online_prefill=True,
        arrivals=(100.0, 100.0),
        slo=40.0,
        full_window=True,
        prefill_token_chunk_size=2,
        prompt_lengths=(1, 5),
    )
    publications = []

    def record_delivery(index, _params, state):
        request_chunks = [
            event
            for event in harness.events
            if event[0] == "prefill-token-chunk" and any(span[0] == index for span in event[1])
        ]
        publications.append(
            (
                index,
                len(request_chunks),
                state.committed_length,
                list(state.committed_completion_token_ids),
            )
        )

    harness.engine._deliver_committed_tokens = record_delivery

    results = harness.run(continuous_batching=True)

    request_one_chunks = [
        event[1]
        for event in harness.events
        if event[0] == "prefill-token-chunk" and any(span[0] == 1 for span in event[1])
    ]
    assert request_one_chunks == [
        ((1, 0, 2, False),),
        ((1, 2, 4, False),),
        ((1, 4, 5, True),),
    ]
    assert [row for row in publications if row[0] == 1][0] == (
        1,
        3,
        6,
        [302],
    )
    assert results[1]["completion_token_ids"][0] == 302
    counters = results[1]["spec_rhythm"]
    assert counters["spec_rhythm_prefill_token_chunk_enabled"] == 1
    assert counters["spec_rhythm_prefill_token_chunk_size"] == 2
    assert counters["spec_rhythm_prefill_token_chunk_partial_submissions"] >= 2
    assert counters["spec_rhythm_prefill_token_chunk_max_submission_tokens"] <= 2


def test_partial_token_chunk_drains_safely_after_incumbent_finishes(
    monkeypatch,
):
    monkeypatch.setattr(native.time, "time", lambda: 100.0)
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(6, 2),
        capacity=1,
        online_prefill=True,
        arrivals=(100.0, 100.0),
        slo=40.0,
        full_window=True,
        prefill_token_chunk_size=2,
        prompt_lengths=(1, 5),
    )

    results = harness.run(continuous_batching=True)

    request_one_chunks = [
        event[1]
        for event in harness.events
        if event[0] == "prefill-token-chunk" and any(span[0] == 1 for span in event[1])
    ]
    assert request_one_chunks == [
        ((1, 0, 2, False),),
        ((1, 2, 4, False),),
        ((1, 4, 5, True),),
    ]
    assert results[1]["completion_token_ids"] == [302, 777]


def test_token_chunk_prefill_config_fails_closed():
    common = {
        "draft_model": "draft",
        "target_model": "target",
        "draft_tp_size": 1,
        "target_tp_size": 3,
        "gamma": 4,
        "max_model_len": 256,
        "max_tokens": 8,
        "max_num_batched_tokens": 256,
        "enable_continuous_batching": True,
        "enable_preemptive_scheduling": True,
        "enable_spec_rhythm": True,
        "spec_rhythm_online_prefill": True,
        "spec_rhythm_linear_full_window": True,
        "spec_rhythm_min_gamma": 4,
        "spec_rhythm_prefill_token_chunk_size": 8,
    }

    with pytest.raises(ValueError, match="prefix caching"):
        native.NativePearlConfig(**common)
    with pytest.raises(ValueError, match="online prefill"):
        native.NativePearlConfig(
            **{
                **common,
                "enable_prefix_caching": False,
                "spec_rhythm_online_prefill": False,
            }
        )
    with pytest.raises(ValueError, match="must not exceed"):
        native.NativePearlConfig(
            **{
                **common,
                "enable_prefix_caching": False,
                "spec_rhythm_prefill_token_chunk_size": 257,
            }
        )
    enabled = native.NativePearlConfig(**{**common, "enable_prefix_caching": False})
    assert enabled.spec_rhythm_prefill_token_chunk_size == 8


def test_online_prefill_coalesces_two_ready_arrivals(monkeypatch):
    wall = [99.9]

    def advance_wall():
        wall[0] += 0.1
        return wall[0]

    monkeypatch.setattr(native.time, "time", advance_wall)
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(30, 5, 5),
        capacity=4,
        online_prefill=True,
        arrivals=(100.0, 100.15, 100.25),
        full_window=True,
        prefill_coalesce_min_requests=2,
        prefill_coalesce_max_wait_ms=600.0,
    )

    results = harness.run()

    prefills = [event for event in harness.events if event[0] == "prefill"]
    assert prefills[:2] == [("prefill", (0,)), ("prefill", (1, 2))]
    assert [result["completion_token_ids"][0] for result in results] == [301, 302, 303]
    diagnostics = results[0]["spec_rhythm"]
    assert diagnostics["spec_rhythm_prefill_coalesce_enabled"] == 1
    assert diagnostics["spec_rhythm_prefill_coalesce_deferred_polls"] >= 1
    assert diagnostics["spec_rhythm_prefill_coalesce_size_releases"] >= 1
    assert diagnostics["spec_rhythm_prefill_pair_batches"] >= 1


def test_online_prefill_coalescing_releases_one_request_at_timeout(monkeypatch):
    wall = [99.9]

    def advance_wall():
        wall[0] += 0.1
        return wall[0]

    monkeypatch.setattr(native.time, "time", advance_wall)
    monkeypatch.setattr(native.time, "sleep", lambda _delay: None)
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(30, 5, 5),
        capacity=4,
        online_prefill=True,
        arrivals=(100.0, 100.15, 105.0),
        full_window=True,
        prefill_coalesce_min_requests=2,
        prefill_coalesce_max_wait_ms=140.0,
    )

    results = harness.run()

    prefills = [event for event in harness.events if event[0] == "prefill"]
    assert ("prefill", (1,)) in prefills
    diagnostics = results[0]["spec_rhythm"]
    assert diagnostics["spec_rhythm_prefill_coalesce_deferred_polls"] >= 1
    assert diagnostics["spec_rhythm_prefill_coalesce_timeout_releases"] >= 1
    assert diagnostics["spec_rhythm_prefill_singleton_batches"] >= 2
    assert diagnostics["spec_rhythm_prefill_coalesce_max_wait_observed_ms"] >= 140.0


def test_online_prefill_coalescing_never_delays_an_empty_service(monkeypatch):
    wall = [100.0]
    monkeypatch.setattr(native.time, "time", lambda: wall[0])
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(2,),
        capacity=2,
        online_prefill=True,
        arrivals=(100.0,),
        full_window=True,
        prefill_coalesce_min_requests=2,
        prefill_coalesce_max_wait_ms=600.0,
    )

    result = harness.run()[0]

    assert harness.events[0] == ("prefill", (0,))
    diagnostics = result["spec_rhythm"]
    assert diagnostics["spec_rhythm_prefill_coalesce_deferred_polls"] == 0
    assert diagnostics["spec_rhythm_prefill_coalesce_forced_releases"] >= 1


def test_online_prefill_coalescing_preserves_tokens_and_paper_tpot_scope(
    monkeypatch,
):
    monkeypatch.setattr(native.time, "perf_counter", lambda: 0.0)

    def run(minimum, wait_ms):
        wall = [99.9]

        def advance_wall():
            wall[0] += 0.1
            return wall[0]

        monkeypatch.setattr(native.time, "time", advance_wall)
        harness = _LinearLoopHarness(
            monkeypatch,
            max_tokens=(30, 5, 5),
            capacity=4,
            online_prefill=True,
            arrivals=(100.0, 100.05, 100.15),
            full_window=True,
            prefill_coalesce_min_requests=minimum,
            prefill_coalesce_max_wait_ms=wait_ms,
        )
        return harness.run()

    immediate = run(1, 0.0)
    coalesced = run(2, 600.0)

    assert [result["completion_token_ids"] for result in coalesced] == [
        result["completion_token_ids"] for result in immediate
    ]
    assert [result["paper_tpot_ms"] for result in coalesced] == [result["paper_tpot_ms"] for result in immediate]


def test_online_fallback_admits_arrival_before_long_incumbent_finishes(monkeypatch):
    wall = [100.0]
    monkeypatch.setattr(native.time, "time", lambda: wall[0])
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(5, 2),
        capacity=2,
        online_prefill=True,
        arrivals=(100.0, 100.03),
        target_fallback=True,
        slo=40.0,
        full_window=True,
    )
    original_target_only = harness.target_only

    def advance_target_only(input_ids, indices, positions, temperatures, **kwargs):
        wall[0] += 0.02
        return original_target_only(
            input_ids,
            indices,
            positions,
            temperatures,
            **kwargs,
        )

    harness.engine._run_packed_sample = advance_target_only

    results = harness.run()

    assert results[0]["completion_token_ids"][-1] == 888
    assert results[1]["completion_token_ids"] == [302, 888]
    # Request 1 arrives after two fallback cycles, while request 0 still has
    # two tokens left.  Admission must therefore not wait for request 0 to
    # complete.
    assert ("prefill", (1,)) in harness.events
    second_prefill = harness.events.index(("prefill", (1,)))
    assert harness.events[second_prefill - 1] == ("target-only", (0,))


@pytest.mark.parametrize(
    ("disable_target_graph", "expected_use_aclgraph"),
    [("1", False), ("0", True)],
)
def test_target_fallback_respects_target_aclgraph_diagnostic_switch(
    monkeypatch,
    disable_target_graph,
    expected_use_aclgraph,
):
    monkeypatch.setenv(
        "VLLM_ASCEND_SPECRHYTHM_DISABLE_TARGET_ACLGRAPH",
        disable_target_graph,
    )
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(2,),
        capacity=1,
        target_fallback=True,
    )
    harness.engine.config = replace(harness.engine.config, enforce_eager=False)
    observed = []

    def target_only(input_ids, indices, positions, temperatures, **kwargs):
        observed.append(bool(kwargs["use_aclgraph"]))
        return harness.target_only(
            input_ids,
            indices,
            positions,
            temperatures,
            **kwargs,
        )

    harness.engine._run_packed_sample = target_only

    results = harness.run()

    assert results[0]["completion_token_ids"] == [301, 888]
    assert observed == [expected_use_aclgraph]


def test_full_fallback_batch_carries_clock_for_same_cycle_replacement(
    monkeypatch,
):
    monkeypatch.setattr(native.time, "time", lambda: 100.0)
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(2, 4, 2),
        capacity=2,
        online_prefill=True,
        arrivals=(100.0, 100.0, 100.0),
        target_fallback=True,
    )

    harness.run()

    first_fallback = harness.events.index(("target-only", (0, 1)))
    # Request 0 finishes while the starting batch is full.  The existing
    # fallback-clock broadcast must still carry a usable wall timestamp so
    # request 2 can fill that slot before the next target-only round.
    assert harness.events[first_fallback + 1] == ("prefill", (2,))


def test_fallback_last_max_round_does_not_prefill_a_replacement(monkeypatch):
    monkeypatch.setattr(native.time, "time", lambda: 100.0)
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(2, 2),
        capacity=1,
        online_prefill=True,
        arrivals=(100.0, 100.0),
        target_fallback=True,
    )

    results = harness.run(max_rounds=1)

    assert results[0]["completion_token_ids"] == [301, 888]
    assert results[1]["completion_token_ids"] == []
    assert [event for event in harness.events if event[0] == "prefill"] == [("prefill", (0,))]


def test_fallback_profile_stop_does_not_prefill_a_replacement(monkeypatch):
    monkeypatch.setattr(native.time, "time", lambda: 100.0)
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(2, 2),
        capacity=1,
        online_prefill=True,
        arrivals=(100.0, 100.0),
        target_fallback=True,
    )
    harness.engine.config = replace(
        harness.engine.config,
        profile_decode_steps=1,
        stop_after_profiled_decode_steps=True,
    )

    results = harness.run()

    assert results[0]["completion_token_ids"] == [301, 888]
    assert results[1]["completion_token_ids"] == []
    assert [event for event in harness.events if event[0] == "prefill"] == [("prefill", (0,))]
    assert harness.engine.last_worker_profiled_decode_steps == 1


def test_staged_prefill_arrival_debt_runs_through_activation_boundary(monkeypatch):
    wall = [99.9]

    def advance_wall():
        wall[0] += 0.1
        return wall[0]

    monkeypatch.setattr(native.time, "time", advance_wall)
    runtime_states = {}
    controller_type = native.SpecRhythmPipelineController

    def capture_controller(states):
        runtime_states.update(states)
        return controller_type(states)

    monkeypatch.setattr(native, "SpecRhythmPipelineController", capture_controller)
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(9, 5),
        capacity=2,
        online_prefill=True,
        arrivals=(100.0, 100.05),
        slo=40.0,
        full_window=True,
    )

    harness.run()

    # Every time() call advances this synthetic clock by 100 ms.  Request 1 is
    # reserved earlier, but its own callback-complete publication timestamp is
    # t=100.4.  Its scheduler debt must run through that post-prefill boundary
    # rather than stopping at reservation/cycle selection.
    assert runtime_states[1].arrival_wait_ms == pytest.approx(350.0)


def test_online_prefill_ready_selection_ignores_role_local_cache_hits(
    monkeypatch,
):
    """Draft/target prefix-cache residency must not change admission shape."""

    monkeypatch.setattr(native.time, "time", lambda: 100.0)

    def first_prefill(cache_hits):
        harness = _LinearLoopHarness(
            monkeypatch,
            max_tokens=(2, 2),
            capacity=2,
            online_prefill=True,
            arrivals=(100.0, 100.0),
            target_fallback=True,
        )
        harness.engine.config = replace(
            harness.engine.config,
            max_num_batched_tokens=256,
        )
        harness.engine.cache_allocation.num_cached_tokens = list(cache_hits)
        for states in (harness.draft_states, harness.target_states):
            for index, state in enumerate(states):
                state.token_ids = [100 + index] * 200
                state.prompt_length = 200
                state.committed_length = 200
                state.draft_synced_length = 200
        harness.run()
        return next(event for event in harness.events if event[0] == "prefill")

    # The old role-local uncached-token calculation admitted both rows for the
    # second cache layout, while the target's cold layout admitted only one.
    assert first_prefill((0, 0)) == ("prefill", (0,))
    assert first_prefill((128, 128)) == ("prefill", (0,))


def test_fallback_arrival_debt_uses_post_prefill_publication_time(monkeypatch):
    wall = [100.0]
    monkeypatch.setattr(native.time, "time", lambda: wall[0])
    runtime_states = {}
    controller_type = native.SpecRhythmPipelineController

    def capture_controller(states):
        runtime_states.update(states)
        return controller_type(states)

    monkeypatch.setattr(native, "SpecRhythmPipelineController", capture_controller)
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(4, 2),
        capacity=2,
        online_prefill=True,
        arrivals=(100.0, 100.01),
        target_fallback=True,
    )
    original_prefill = harness.prefill
    original_target_only = harness.target_only

    def delayed_prefill(rows, states, indices):
        if indices == [1]:
            wall[0] += 0.2
        return original_prefill(rows, states, indices)

    def advance_target(input_ids, indices, positions, temperatures, **kwargs):
        wall[0] += 0.02
        return original_target_only(input_ids, indices, positions, temperatures, **kwargs)

    harness.engine._prefill_and_sample_target_batch = delayed_prefill
    harness.engine._run_packed_sample = advance_target

    result = harness.run()[1]

    # Selected at 100.02, published only after the 200 ms prefill at 100.22.
    assert runtime_states[1].arrival_wait_ms == pytest.approx(210.0)
    assert result["spec_rhythm"]["spec_rhythm_first_token_clock_broadcasts"] >= 2


def test_new_activation_tail_starts_after_first_token_publication(monkeypatch):
    perf = [0.0]
    wall = [100.0]
    monkeypatch.setattr(native.time, "perf_counter", lambda: perf[0])
    monkeypatch.setattr(native.time, "time", lambda: wall[0])
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(4, 2),
        capacity=2,
        online_prefill=True,
        arrivals=(100.0, 100.01),
        target_fallback=True,
    )
    original_broadcast = native.dist.broadcast
    publication_broadcasts = [0]

    def delayed_broadcast(value, *args, **kwargs):
        result = original_broadcast(value, *args, **kwargs)
        if not kwargs.get("async_op") and value.numel() == 3:
            publication_broadcasts[0] += 1
            if publication_broadcasts[0] == 2:
                # This happens after request 1's per-row publication timestamp
                # but before refill_end, so it belongs to request 1's TPOT.
                perf[0] += 0.05
        return result

    monkeypatch.setattr(native.dist, "broadcast", delayed_broadcast)
    original_target_only = harness.target_only

    def timed_target(input_ids, indices, positions, temperatures, **kwargs):
        wall[0] += 0.02
        perf[0] += 0.02
        return original_target_only(input_ids, indices, positions, temperatures, **kwargs)

    harness.engine._run_packed_sample = timed_target

    def publish(index, _params, state):
        if index == 1 and len(state.committed_completion_token_ids) == 1:
            # Publication begins at callback invocation.  Time spent delivering
            # that first event therefore lies after the first-token timestamp
            # and must not disappear before the next token.
            perf[0] += 0.1

    harness.engine._deliver_committed_tokens = publish

    result = harness.run()[1]

    assert result["observed_tpot_ms"] == pytest.approx(170.0)
    diagnostics = result["spec_rhythm"]
    assert diagnostics["spec_rhythm_new_request_tail_ms"] == pytest.approx(150.0)
    assert diagnostics["spec_rhythm_new_request_tail_segments"] >= 1


def test_fallback_finished_row_uses_its_own_publication_endpoint(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(native.time, "perf_counter", lambda: clock[0])
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(2, 2),
        capacity=2,
        target_fallback=True,
    )
    original_target_only = harness.target_only

    def timed_target(input_ids, indices, positions, temperatures, **kwargs):
        clock[0] += 0.01
        return original_target_only(input_ids, indices, positions, temperatures, **kwargs)

    harness.engine._run_packed_sample = timed_target

    def delayed_second_callback(index, _params, state):
        if index == 1 and len(state.committed_completion_token_ids) == 2:
            clock[0] += 0.1

    harness.engine._deliver_committed_tokens = delayed_second_callback

    results = harness.run()

    assert [result["observed_tpot_ms"] for result in results] == pytest.approx([10.0, 110.0])


def test_verification_finished_row_uses_its_own_publication_endpoint(
    monkeypatch,
):
    clock = [0.0]
    monkeypatch.setattr(native.time, "perf_counter", lambda: clock[0])
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(5, 5, 5, 5),
        capacity=4,
        full_window=True,
    )
    original_target = harness.target_full_window

    def timed_target(states, indices, payloads, *, proposal_matrix=None):
        output = original_target(
            states,
            indices,
            payloads,
            proposal_matrix=proposal_matrix,
        )
        if indices:
            clock[0] += 0.01
        return output

    harness.engine._target_full_window_outputs_batch = timed_target

    def delayed_later_row(index, _params, state):
        if index == 2 and len(state.committed_completion_token_ids) == 5:
            clock[0] += 0.1

    harness.engine._deliver_committed_tokens = delayed_later_row

    results = harness.run()

    # Requests 0 and 2 share a home and finish in the same verification batch.
    # Request 0 must stop at its own callback rather than inheriting request 2's
    # subsequent 100 ms callback delay.
    assert results[0]["observed_tpot_ms"] == pytest.approx(2.5)
    assert results[2]["observed_tpot_ms"] == pytest.approx(27.5)


def test_finish_envelope_rejects_cross_rank_active_order_mismatch(monkeypatch):
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(5, 5),
        capacity=2,
        slo=40.0,
        full_window=True,
    )
    original_broadcast = native.dist.broadcast

    def corrupt_active_order(value, *args, **kwargs):
        result = original_broadcast(value, *args, **kwargs)
        # fixed-W clock (4) + active IDs (B=2) + finish offsets (B=2)
        if value.dtype == torch.float64 and value.numel() == 8:
            value[4], value[5] = value[5].clone(), value[4].clone()
        return result

    monkeypatch.setattr(native.dist, "broadcast", corrupt_active_order)

    with pytest.raises(RuntimeError, match="active-row order diverged"):
        harness.run()


@pytest.mark.parametrize(
    ("terminal_max_tokens", "eos_token_ids"),
    [
        (1, frozenset()),
        (5, frozenset({302})),
    ],
)
def test_terminal_staged_prefill_completes_without_consuming_decode_capacity(
    monkeypatch,
    terminal_max_tokens,
    eos_token_ids,
):
    wall = [99.9]

    def advance_wall():
        wall[0] += 0.1
        return wall[0]

    monkeypatch.setattr(native.time, "time", advance_wall)
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(9, terminal_max_tokens, 5),
        capacity=2,
        online_prefill=True,
        arrivals=(100.0, 100.05, 100.05),
        slo=40.0,
        full_window=True,
    )
    harness.engine.eos_token_ids = eos_token_ids
    if eos_token_ids:
        harness.params[1] = replace(harness.params[1], ignore_eos=False)
        harness.draft_states[1].ignore_eos = False
        harness.target_states[1].ignore_eos = False
    deliveries = []

    def record_delivery(index, _params, state):
        deliveries.append(
            (
                index,
                list(state.committed_completion_token_ids),
                native._finished(state, harness.engine.eos_token_ids),
            )
        )

    harness.engine._deliver_committed_tokens = record_delivery

    results = harness.run(continuous_batching=True)

    assert results[1]["completion_token_ids"] == [302]
    assert results[1]["accepted_draft_tokens"] == 0
    assert results[1]["verified_draft_tokens"] == 0
    assert results[1]["verification_rounds"] == 0
    assert [delivery for delivery in deliveries if delivery[0] == 1] == [(1, [302], True)]
    assert all(request_index != 1 for batch in harness.proposal_batches for request_index, _gamma, _eager in batch)
    assert [event for event in harness.events if event[0] == "prefill"][:3] == [
        ("prefill", (0,)),
        ("prefill", (1,)),
        ("prefill", (2,)),
    ]
    counters = results[1]["spec_rhythm"]
    assert counters["spec_rhythm_admitted_requests"] == 3
    assert counters["spec_rhythm_prefill_requests"] == 3


def test_linear_online_refill_pause_charges_only_surviving_incumbent(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(native.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(native.time, "time", lambda: 100.0)
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(2, 4, 4),
        capacity=2,
        online_prefill=True,
        target_fallback=True,
        slo=100.0,
    )
    original_prefill = harness.prefill

    def timed_prefill(rows, states, indices):
        if indices == [2]:
            clock[0] += 0.1
        return original_prefill(rows, states, indices)

    def timed_target_only(input_ids, indices, positions, temperatures, **kwargs):
        clock[0] += 0.02
        return harness.target_only(input_ids, indices, positions, temperatures, **kwargs)

    harness.engine._prefill_and_sample_target_batch = timed_prefill
    harness.engine._run_packed_sample = timed_target_only

    results = harness.run(continuous_batching=True)

    assert [result["observed_tpot_ms"] for result in results] == pytest.approx([20.0, 160.0 / 3.0, 20.0])
    # The completed request is not charged for its replacement's prefill, and
    # the replacement itself does not inherit a pre-admission pause.
    assert results[0]["spec_rhythm"]["spec_rhythm_online_refill_ms"] == pytest.approx(100.0)


def test_linear_result_has_paper_metrics_and_mode_identity(monkeypatch):
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(2,),
        capacity=1,
        online_prefill=True,
        target_fallback=True,
        slo=40.0,
    )

    result = harness.run()[0]

    assert result["paper_tpot_ms"] == pytest.approx(result["observed_tpot_ms"] / 2)
    assert result["paper_tpot_definition"] == "same_decode_elapsed_ms / output_tokens"
    assert result["paper_slo_attained"] is True
    assert result["paper_slo_goodput_tokens"] == 2
    assert result["finish_reason"] == "length"
    assert result["tree_mode"] is False
    assert (result["tree_width"], result["tree_depth"]) == (1, 1)


def test_linear_tpot_includes_first_speculative_cycle_after_prefill(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(native.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(native.time, "time", lambda: 100.0)
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(2,),
        capacity=1,
        online_prefill=True,
        target_fallback=False,
        slo=40.0,
    )
    original_target = harness.target

    def timed_target(states, indices, verification_sizes):
        values = original_target(states, indices, verification_sizes)
        if indices:
            clock[0] += 0.02
        return values

    harness.engine._target_round_outputs_batch = timed_target

    result = harness.run()[0]

    assert result["completion_token_ids"] == [301, 777]
    assert result["observed_tpot_ms"] == pytest.approx(20.0)
    assert result["paper_tpot_ms"] == pytest.approx(10.0)


def test_linear_cycle_clock_charges_tail_and_carries_accounting_collective(
    monkeypatch,
):
    """Every wait between the first and last token enters request TPOT."""

    clock = [0.0]
    monkeypatch.setattr(native.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(native.time, "time", lambda: 100.0)
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(9,),
        capacity=1,
        slo=40.0,
        full_window=True,
    )
    original_broadcast = native.dist.broadcast

    def delayed_broadcast(value, *args, **kwargs):
        result = original_broadcast(value, *args, **kwargs)
        if not kwargs.get("async_op"):
            if value.numel() == 6:
                # Existing elapsed/wall/W broadcast plus fixed active-ID and
                # finish-offset envelopes: it occurs after the primary endpoint
                # and therefore belongs to the cycle tail.
                clock[0] += 0.007
            elif value.numel() == 12:
                # Tail accounting itself is causally too late for its own
                # payload; its remaining fields carry the target monotonic
                # endpoint and replicated service frontier without adding
                # another collective.  The persistent boundary carries it
                # into the next cycle's primary interval.
                clock[0] += 0.011
        return result

    monkeypatch.setattr(native.dist, "broadcast", delayed_broadcast)
    original_target = harness.target_full_window

    def timed_target(states, indices, payloads, *, proposal_matrix=None):
        output = original_target(
            states,
            indices,
            payloads,
            proposal_matrix=proposal_matrix,
        )
        if indices:
            clock[0] += 0.013
        return output

    harness.engine._target_full_window_outputs_batch = timed_target

    result = harness.run()[0]

    assert result["completion_token_ids"] == [301, *([777] * 8)]
    # From the initial prefill token to the final commit the one-home pipeline
    # runs warmup, verify, draft, verify.  Three carried tail broadcasts cost
    # 11 ms, the three surviving-cycle primary broadcasts cost 7 ms, and the
    # two target calls cost 13 ms.  Work after the final token is deliberately
    # excluded: (3 * 11 + 3 * 7 + 2 * 13) / (9 - 1).
    assert result["observed_tpot_ms"] == pytest.approx(10.0)
    counters = result["spec_rhythm"]
    assert counters["spec_rhythm_cycle_accounted_tail_cycles"] == 3
    assert counters["spec_rhythm_cycle_accounted_tail_ms"] == pytest.approx(21.0)


def test_linear_full_window_service_verifies_complete_serial_proposals(monkeypatch):
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(9, 9, 9, 9),
        capacity=4,
        slo=40.0,
        full_window=True,
    )

    results = harness.run()

    assert all(result["linear_full_window"] is True for result in results)
    assert all(result["completion_token_ids"] == [301, *([777] * 8)] for result in results)
    assert all(result["verified_draft_tokens"] % 4 == 0 for result in results)
    assert all(result["accepted_draft_tokens"] == result["verified_draft_tokens"] for result in results)
    assert any(event[0] == "target-full-window" for event in harness.events)
    assert all(gamma == 4 for batch in harness.proposal_batches for _, gamma, _ in batch)


def test_linear_full_window_bonus_service_commits_fifth_target_token(monkeypatch):
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(6,),
        capacity=1,
        full_window=True,
        bonus_token=True,
    )

    result = harness.run()[0]

    assert result["completion_token_ids"] == [301, 777, 777, 777, 777, 900]
    assert result["accepted_draft_tokens"] == 4
    assert result["verified_draft_tokens"] == 4
    assert harness.target_states[0].committed_completion_token_ids == [
        301,
        777,
        777,
        777,
        777,
        900,
    ]
    counters = result["spec_rhythm"]
    assert counters["spec_rhythm_linear_bonus_eligible_rows"] == 1
    assert counters["spec_rhythm_linear_bonus_committed_tokens"] == 1
    assert counters["spec_rhythm_linear_bonus_suppressed_eager_rows"] == 0


@pytest.mark.parametrize(
    ("committed_outputs", "max_tokens", "expected"),
    [
        (1, 5, False),
        (1, 6, True),
        (5, 5, False),
    ],
)
def test_full_window_eager_requires_horizon_beyond_its_parent(
    committed_outputs,
    max_tokens,
    expected,
):
    state = native.PearlPipelineState(
        [100, *range(200, 200 + committed_outputs)],
        prompt_length=1,
        max_tokens=max_tokens,
    )

    assert native._full_window_eager_has_useful_horizon(state, 4) is expected


def test_linear_full_window_does_not_draft_eager_child_for_final_parent(
    monkeypatch,
):
    admitted_candidates = []

    def admit_all_complete_rows(eligible_indices, **kwargs):
        admitted_candidates.extend(eligible_indices)
        count = len(eligible_indices)
        return list(eligible_indices), native.DraftWindowBudget(
            draft_window_ms=100.0,
            draft_ms_per_token=1.0,
            normal_tokens=0,
            draft_token_budget=count * 4,
            eager_token_budget=count * 4,
            residual_window_ms=100.0,
            predicted_exposed_draft_ms=0.0,
            calibrated=True,
        )

    monkeypatch.setattr(native, "_gate_linear_eager_candidates", admit_all_complete_rows)
    monkeypatch.setattr(
        native.SpecRhythmRuntimeState,
        "projected_progress_gap",
        lambda self, projected_wait_ms: 1,
    )
    monkeypatch.setattr(
        native.SpecRhythmRuntimeState,
        "urgency",
        lambda self, projected_wait_ms: 1.0,
    )
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(5,),
        capacity=1,
        slo=40.0,
        full_window=True,
    )

    result = harness.run()[0]

    assert len(result["completion_token_ids"]) == 5
    assert admitted_candidates == []
    assert all(not eager for batch in harness.proposal_batches for _, _, eager in batch)


def test_linear_full_window_rejects_partial_budget_before_collective(monkeypatch):
    original_shape = native.SpecRhythmBudgetShaper.shape

    def partial_shape(shaper, **kwargs):
        plan = original_shape(shaper, **kwargs)
        normal_budgets = dict(plan.normal_budgets)
        first = next(iter(normal_budgets))
        normal_budgets[first] = 2
        return replace(
            plan,
            normal_budgets=normal_budgets,
            allocated_draft_tokens=sum(normal_budgets.values()),
        )

    monkeypatch.setattr(native.SpecRhythmBudgetShaper, "shape", partial_shape)
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(9, 9),
        capacity=2,
        full_window=True,
    )

    with pytest.raises(RuntimeError, match="full-window scheduling"):
        harness.run(max_rounds=1)

    assert harness.broadcast_sizes == []
    assert harness.proposal_batches == []


def test_linear_full_window_fallback_discards_uncommitted_draft_tail(monkeypatch):
    """Shrinking into exact AR must leave both model homes on one frontier."""

    target_harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(5, 5, 14, 14),
        capacity=4,
        target_fallback=2,
        slo=40.0,
        full_window=True,
    )
    target_results = target_harness.run(max_rounds=20)

    # This second harness represents a non-leader rank without a real
    # broadcast peer.  Keep its local monotonic clock at the target-leader
    # value carried by the mocked primary-clock collective.
    monkeypatch.setattr(native.time, "perf_counter", lambda: 0.0)
    draft_harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(5, 5, 14, 14),
        capacity=4,
        target_fallback=2,
        slo=40.0,
        full_window=True,
        is_draft=True,
    )
    assert draft_harness.run(max_rounds=20) is None

    # Requests 2/3 retain ready or eager proposal suffixes when requests 0/1
    # finish and the resident batch crosses the fallback threshold.  Exact
    # target tokens (888) must replace, rather than trail, those suffixes.
    for request_index in (2, 3):
        target_tokens = target_results[request_index]["completion_token_ids"]
        draft_tokens = draft_harness.draft_states[request_index].committed_completion_token_ids
        assert draft_tokens == target_tokens
        assert 888 in draft_tokens
        assert draft_tokens[-1] == 888


def test_target_fallback_preflights_all_rows_before_mutating_any_state(monkeypatch):
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(3, 3),
        capacity=2,
        target_fallback=True,
        slo=40.0,
        full_window=True,
    )
    first_before = harness.target_states[0].clone()
    harness.target_states[1].continuation_epoch = 1

    with pytest.raises(RuntimeError, match="stale committed prefix epoch"):
        harness.run(max_rounds=1)

    assert harness.target_states[0] == first_before


def test_linear_host_timeline_adds_no_profiling_synchronize(monkeypatch):
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(8, 8, 8, 8),
        capacity=4,
        profile_host_steps=2,
    )

    harness.run(max_rounds=2)

    traces = harness.engine.last_worker_decode_host_timeline
    assert len(traces) == 2
    assert traces[1]["target_requests"] > 0
    assert traces[1]["draft_requests"] > 0
    for key in (
        "scheduler_roof_end_seconds",
        "scheduler_plan_end_seconds",
        "scheduler_budget_end_seconds",
        "scheduler_end_seconds",
        "draft_start_seconds",
        "draft_end_seconds",
        "target_start_seconds",
        "target_end_seconds",
        "exchange_start_seconds",
        "exchange_end_seconds",
        "verdict_start_seconds",
        "verdict_end_seconds",
        "correction_start_seconds",
        "correction_end_seconds",
        "state_start_seconds",
        "state_end_seconds",
        "accounting_split_seconds",
        "accounted_tail_ms",
        "accounting_boundary_seconds",
        "cycle_end_seconds",
    ):
        assert key in traces[1]
    assert traces[0]["accounting_boundary_seconds"] == traces[1]["cycle_start_seconds"]
    # The only synchronize is the pre-existing service-finalization fence.
    harness.synchronize.assert_called_once_with()
    # Active-cycle accounting carries elapsed monotonic time, the target-leader
    # wall clock, fixed-capacity active-ID and finish-endpoint envelopes in one
    # broadcast.
    assert 10 in harness.broadcast_sizes


def test_full_window_host_trace_records_async_submit_and_wait_boundaries(monkeypatch):
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(9, 9, 9, 9),
        capacity=4,
        slo=40.0,
        profile_host_steps=2,
        full_window=True,
    )

    harness.run(max_rounds=2)

    steady = harness.engine.last_worker_decode_host_timeline[1]
    assert steady["draft_end_seconds"] <= steady["compact_submit_start_seconds"]
    assert steady["verdict_end_seconds"] <= steady["compact_submit_start_seconds"]
    assert steady["compact_submit_start_seconds"] <= steady["compact_submit_end_seconds"]
    assert steady["compact_submit_end_seconds"] <= steady["compact_wait_start_seconds"]
    assert steady["compact_wait_start_seconds"] <= steady["compact_wait_end_seconds"]


@pytest.mark.parametrize(
    ("target_window_ms", "expected_indices"),
    [
        (4.0, []),
        (8.0, [2]),
        (12.0, [2, 1]),
    ],
)
def test_linear_fixed_gamma_w_admits_only_complete_ranked_rows(
    target_window_ms,
    expected_indices,
):
    states = {
        1: native.SpecRhythmRuntimeState(
            request_index=1,
            home_batch_id=1,
            slo_tpot_ms=10.0,
            delivered_tokens=4,
            decode_elapsed_ms=50.0,
        ),
        2: native.SpecRhythmRuntimeState(
            request_index=2,
            home_batch_id=0,
            slo_tpot_ms=10.0,
            delivered_tokens=1,
            decode_elapsed_ms=30.0,
            acceptance_ema=0.9,
        ),
    }
    estimator = native.DraftWindowEstimator(ema_alpha=1.0)
    estimator.observe(
        draft_compute_ms=4.0,
        drafted_tokens=4,
        target_verify_ms=target_window_ms,
        eager_work=False,
    )

    selected, window = native._gate_linear_eager_candidates(
        [1, 2],
        states=states,
        projected_wait_ms=1.0,
        normal_request_count=1,
        gamma=4,
        estimator=estimator,
    )

    assert selected == expected_indices
    assert window.eager_token_budget % 4 == 0
    assert window.eager_token_budget == len(expected_indices) * 4


def test_linear_fixed_gamma_w_bootstrap_defers_all_optional_rows():
    states = {
        index: native.SpecRhythmRuntimeState(
            request_index=index,
            home_batch_id=index % 2,
            slo_tpot_ms=1.0,
            decode_elapsed_ms=10.0,
        )
        for index in (1, 2)
    }

    selected, window = native._gate_linear_eager_candidates(
        [1, 2],
        states=states,
        projected_wait_ms=1.0,
        normal_request_count=1,
        gamma=4,
        estimator=native.DraftWindowEstimator(),
    )

    assert selected == []
    assert window.calibrated is False
    assert window.draft_token_budget == 4
    assert window.eager_token_budget == 0


@pytest.mark.parametrize(
    ("normal_rows", "expected_count"),
    [
        (17, 3),
        (20, 0),
        (21, 3),
        (32, 0),
    ],
)
def test_linear_fixed_gamma_w_never_promotes_optional_eager_work_to_next_graph_bucket(
    normal_rows,
    expected_count,
):
    states = {
        index: native.SpecRhythmRuntimeState(
            request_index=index,
            home_batch_id=index % 2,
            slo_tpot_ms=10.0,
            delivered_tokens=1,
            decode_elapsed_ms=100.0 + index,
            acceptance_ema=0.9,
        )
        for index in range(8)
    }
    estimator = native.DraftWindowEstimator(ema_alpha=1.0)
    estimator.observe(
        draft_compute_ms=1.0,
        drafted_tokens=4,
        target_verify_ms=100.0,
        eager_work=False,
    )

    selected, window = native._gate_linear_eager_candidates(
        list(states),
        states=states,
        projected_wait_ms=1.0,
        normal_request_count=normal_rows,
        gamma=4,
        estimator=estimator,
        max_rows=64,
    )

    assert window.calibrated
    assert len(selected) == expected_count
    selected_bucket = native._next_linear_draft_graph_bucket(
        max(1, normal_rows),
        64,
    )
    assert normal_rows + len(selected) <= selected_bucket


def test_linear_fixed_gamma_w_can_explicitly_cross_graph_bucket_within_w_and_capacity():
    states = {
        index: native.SpecRhythmRuntimeState(
            request_index=index,
            home_batch_id=index % 2,
            slo_tpot_ms=10.0,
            delivered_tokens=1,
            decode_elapsed_ms=100.0 + index,
            acceptance_ema=0.9,
        )
        for index in range(8)
    }
    estimator = native.DraftWindowEstimator(ema_alpha=1.0)
    estimator.observe(
        draft_compute_ms=1.0,
        drafted_tokens=4,
        target_verify_ms=100.0,
        eager_work=False,
    )

    selected, window = native._gate_linear_eager_candidates(
        list(states),
        states=states,
        projected_wait_ms=1.0,
        normal_request_count=20,
        gamma=4,
        estimator=estimator,
        max_rows=24,
        allow_cross_graph_bucket=True,
    )

    assert window.calibrated
    assert len(selected) == 4
    assert 20 + len(selected) <= 24
    assert 20 + len(selected) > native._next_linear_draft_graph_bucket(20, 24)


def test_linear_fixed_gamma_w_does_not_fill_paid_rows_beyond_scalar_w():
    states = {
        index: native.SpecRhythmRuntimeState(
            request_index=index,
            home_batch_id=index % 2,
            slo_tpot_ms=10.0,
            delivered_tokens=1,
            decode_elapsed_ms=100.0 + index,
            acceptance_ema=0.9,
        )
        for index in range(8)
    }
    estimator = native.DraftWindowEstimator(ema_alpha=1.0)
    estimator.observe(
        draft_compute_ms=4.0,
        drafted_tokens=4,
        target_verify_ms=4.0,
        eager_work=False,
    )

    selected, window = native._gate_linear_eager_candidates(
        list(states),
        states=states,
        projected_wait_ms=1.0,
        normal_request_count=17,
        gamma=4,
        estimator=estimator,
        max_rows=64,
    )

    assert window.eager_token_budget == 0
    assert selected == []


def test_linear_fixed_gamma_w_eager_only_work_is_not_pinned_to_virtual_bucket_one():
    states = {
        index: native.SpecRhythmRuntimeState(
            request_index=index,
            home_batch_id=index % 2,
            slo_tpot_ms=10.0,
            delivered_tokens=1,
            decode_elapsed_ms=100.0 + index,
            acceptance_ema=0.9,
        )
        for index in range(8)
    }
    estimator = native.DraftWindowEstimator(ema_alpha=1.0)
    estimator.observe(
        draft_compute_ms=4.0,
        drafted_tokens=4,
        target_verify_ms=16.0,
        eager_work=False,
    )

    selected, window = native._gate_linear_eager_candidates(
        list(states),
        states=states,
        projected_wait_ms=1.0,
        normal_request_count=0,
        gamma=4,
        estimator=estimator,
        max_rows=64,
    )

    assert window.eager_token_budget == 16
    assert len(selected) == 4


def test_linear_fixed_gamma_w_keeps_urgent_rows_ahead_of_idle_residual_rows():
    states = {
        0: native.SpecRhythmRuntimeState(
            request_index=0,
            home_batch_id=0,
            slo_tpot_ms=10.0,
            delivered_tokens=1,
            decode_elapsed_ms=100.0,
            acceptance_ema=0.5,
            full_acceptance_ema=0.1,
        ),
        1: native.SpecRhythmRuntimeState(
            request_index=1,
            home_batch_id=1,
            slo_tpot_ms=1000.0,
            delivered_tokens=10,
            decode_elapsed_ms=1.0,
            acceptance_ema=1.0,
            full_acceptance_ema=1.0,
        ),
    }
    estimator = native.DraftWindowEstimator(ema_alpha=1.0)
    estimator.observe(
        draft_compute_ms=4.0,
        drafted_tokens=4,
        target_verify_ms=4.0,
        eager_work=False,
    )

    selected, _ = native._gate_linear_eager_candidates(
        [1, 0],
        states=states,
        projected_wait_ms=1.0,
        normal_request_count=0,
        gamma=4,
        estimator=estimator,
        max_rows=1,
        residual_indices=frozenset({1}),
    )

    assert selected == [0]


def test_linear_idle_residual_eager_turns_single_home_target_only_into_continuation(
    monkeypatch,
):
    clock = [0.0]
    monkeypatch.setattr(native.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(native.time, "time", lambda: 100.0)

    def ample_window(_self, *, normal_tokens, max_draft_tokens, eager_work=False):
        del eager_work
        return native.DraftWindowBudget(
            draft_window_ms=16.0,
            draft_ms_per_token=1.0,
            normal_tokens=normal_tokens,
            draft_token_budget=max_draft_tokens,
            eager_token_budget=max_draft_tokens - normal_tokens,
            residual_window_ms=16.0 - normal_tokens,
            predicted_exposed_draft_ms=0.0,
            calibrated=True,
        )

    monkeypatch.setattr(native.DraftWindowEstimator, "estimate", ample_window)
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(24,),
        capacity=1,
        slo=1000.0,
        full_window=True,
        idle_residual_eager=True,
        remote_draft_compute_ms=4.0,
        profile_host_steps=8,
    )
    original_target = harness.target_full_window

    def timed_target(states, indices, payloads, *, proposal_matrix=None):
        output = original_target(
            states,
            indices,
            payloads,
            proposal_matrix=proposal_matrix,
        )
        if indices:
            clock[0] += 0.016
        return output

    harness.engine._target_full_window_outputs_batch = timed_target

    results = harness.run(max_rounds=8)

    counters = results[0]["spec_rhythm"]
    assert counters["spec_rhythm_linear_idle_residual_eligible_rows"] > 0
    assert counters["spec_rhythm_linear_idle_residual_admitted_rows"] > 0


def test_linear_fixed_gamma_w_telemetry_and_cycle_bootstrap(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(native.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(native.time, "time", lambda: 100.0)
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(12, 12, 12, 12),
        capacity=4,
        slo=0.001,
        profile_host_steps=6,
        remote_draft_compute_ms=4.0,
    )
    original_target = harness.target

    def timed_target(states, indices, verification_sizes):
        output = original_target(states, indices, verification_sizes)
        if indices:
            clock[0] += 0.016
        return output

    harness.engine._target_round_outputs_batch = timed_target

    results = harness.run(max_rounds=6)

    traces = harness.engine.last_worker_decode_host_timeline
    assert traces[0]["linear_w_calibrated"] == 0
    assert traces[0]["linear_w_admitted_eager_rows"] == 0
    assert traces[1]["linear_w_calibrated"] == 0
    assert traces[1]["linear_w_admitted_eager_rows"] == 0
    # Two normal home rows exactly fill bucket 2 in this four-request
    # harness.  W is calibrated, but optional eager work must not force the
    # serial draft graph into bucket 4.
    assert all(trace["linear_w_admitted_eager_rows"] == 0 for trace in traces[2:])
    # W is a row gate only: it never creates a partial gamma proposal.
    assert all(gamma == 4 for batch in harness.proposal_batches for _, gamma, _ in batch)
    counters = results[0]["spec_rhythm"]
    assert counters["spec_rhythm_linear_w_enabled"] == 1
    assert counters["spec_rhythm_linear_w_calibrated_steps"] > 0
    assert counters["spec_rhythm_linear_w_eligible_eager_rows"] > 0
    assert counters["spec_rhythm_linear_w_admitted_eager_rows"] == 0
    assert counters["spec_rhythm_linear_w_bucket_deferred_eager_rows"] > 0
    assert counters["spec_rhythm_linear_w_last_observed_draft_ms"] == pytest.approx(4.0)
    assert counters["spec_rhythm_linear_w_last_observed_target_ms"] == pytest.approx(16.0)
    # Fixed-W retains the four-scalar primary clock and appends fixed-capacity
    # active-ID and finish-endpoint envelopes.  The tail clock also
    # carries one target-leader monotonic endpoint plus ten service-frontier
    # fingerprint scalars, making refill time and rank divergence auditable
    # without another collective.
    assert set(harness.broadcast_sizes) == {12}
    harness.synchronize.assert_called_once_with()


def test_linear_fixed_gamma_cross_bucket_w_telemetry_distinguishes_admission(
    monkeypatch,
):
    clock = [0.0]
    monkeypatch.setattr(native.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(native.time, "time", lambda: 100.0)
    harness = _LinearLoopHarness(
        monkeypatch,
        max_tokens=(12, 12, 12, 12),
        capacity=4,
        slo=0.001,
        profile_host_steps=6,
        remote_draft_compute_ms=4.0,
        full_window=True,
        eager_cross_graph_bucket=True,
    )
    original_target = harness.target_full_window

    def timed_target(states, indices, payloads, *, proposal_matrix=None):
        output = original_target(
            states,
            indices,
            payloads,
            proposal_matrix=proposal_matrix,
        )
        if indices:
            clock[0] += 0.016
        return output

    harness.engine._target_full_window_outputs_batch = timed_target

    results = harness.run(max_rounds=6)

    traces = harness.engine.last_worker_decode_host_timeline
    counters = results[0]["spec_rhythm"]
    assert counters["spec_rhythm_linear_w_cross_graph_bucket_enabled"] == 1
    assert counters["spec_rhythm_linear_w_cross_bucket_candidate_rows"] > 0
    assert counters["spec_rhythm_linear_w_bucket_blocked_eager_rows"] == 0
    assert counters["spec_rhythm_linear_w_cross_bucket_admitted_eager_rows"] > 0
    assert any(
        trace.get("linear_w_cross_bucket_candidate_rows", 0) > 0
        and trace.get("linear_w_bucket_blocked_eager_rows", 0) == 0
        and trace.get("linear_w_cross_bucket_admitted_eager_rows", 0) > 0
        for trace in traces
    )


def test_linear_proposal_timing_tail_does_not_enter_continuation(monkeypatch):
    monkeypatch.setattr(native.dist, "broadcast", lambda *args, **kwargs: None)
    engine = native.NativePearlEngine.__new__(native.NativePearlEngine)
    engine.gamma = 4
    engine.device = torch.device("cpu")
    engine.topology = PearlTopology.from_tensor_parallel_sizes(1, 3)
    engine.rank = engine.topology.draft_leader_rank
    engine.is_draft = True
    engine.groups = SimpleNamespace(is_verification_worker=True, verification_group=None)
    ticket = native.SpecRhythmProposalTicket(
        proposal_id=7,
        request_index=0,
        home_batch_id=0,
        gamma=4,
        required_prefix_epoch=0,
    )
    verification = torch.tensor([101, 102], dtype=torch.long)
    continuations = torch.tensor([[201, 202, 203, 204]], dtype=torch.long)

    message, _, timing = engine._exchange_spec_rhythm_device_proposals(
        verification,
        continuations,
        torch.tensor([0.75]),
        [ticket],
        [2],
        draft_compute_ms=1.25,
    )

    assert message is not None
    assert message.numel() == 7 + 2 + 4 + 1
    assert timing is not None and timing.item() == 1250
    engine.is_draft = False
    payload = engine._materialize_spec_rhythm_payloads(
        tickets=[ticket],
        verification_sizes=[2],
        local_verification=None,
        local_next_windows=None,
        exchanged_message=message,
        draft_confidences=[message[6]],
    )[0]
    assert payload.verification_tokens.tolist() == [101, 102]
    assert payload.next_tokens.tolist() == [201, 202, 203, 204]


def test_draft_tp_follower_materializes_warmup_payload_for_next_cycle(monkeypatch):
    """A non-verification draft follower must retain its local proposal."""

    broadcast = Mock()
    monkeypatch.setattr(native.dist, "broadcast", broadcast)
    engine = native.NativePearlEngine.__new__(native.NativePearlEngine)
    engine.gamma = 4
    engine.device = torch.device("cpu")
    engine.topology = PearlTopology.from_tensor_parallel_sizes(2, 1)
    engine.rank = engine.topology.draft_ranks[1]
    engine.is_draft = True
    engine.groups = SimpleNamespace(
        is_verification_worker=False,
        verification_group=None,
    )

    runtime = SpecRhythmRuntimeState(request_index=0, home_batch_id=0)
    controller = SpecRhythmPipelineController({0: runtime})
    warmup = controller.build_plan([0])
    assert warmup.phase is PipelinePhase.WARMUP
    assert warmup.normal_draft_request_indices == (0,)

    state = native.PearlPipelineState([10, 11], prompt_length=1, max_tokens=16)
    ticket = controller.new_ticket(0, gamma=engine.gamma, eager=False)
    continuations = torch.tensor([[201, 202, 203, 204]], dtype=torch.long)
    verification = continuations.flatten()
    state.token_ids.extend(continuations[0].tolist())
    local_confidences = torch.tensor([0.75])
    message, confidence_values, timing = engine._exchange_spec_rhythm_device_proposals(
        verification,
        continuations,
        local_confidences,
        [ticket],
        [engine.gamma],
    )
    payloads = engine._materialize_spec_rhythm_payloads(
        tickets=[ticket],
        verification_sizes=[engine.gamma],
        local_verification=verification,
        local_next_windows=continuations,
        exchanged_message=message,
        draft_confidences=confidence_values,
    )
    assert len(payloads) == 1
    controller.publish([ticket])
    payload_by_id = {payload.ticket.proposal_id: payload for payload in payloads}

    assert engine._bound_spec_rhythm_linear_ready(
        controller,
        payload_by_id,
        [state],
        [0],
        engine.gamma,
        full_window=True,
    ) == {
        "rebuilt_requests": 0,
        "discarded_ready": 0,
        "invalidated_eager": 0,
        "discarded_unverified_tokens": 0,
    }
    next_cycle = controller.build_plan(
        [0],
        verification_budget=engine.gamma,
        ready_candidate_counts={0: payload_by_id[controller.ready[0].proposal_id].verification_size},
    )
    assert next_cycle.phase is PipelinePhase.STEADY
    assert next_cycle.target_request_indices == (0,)
    payload = payload_by_id[ticket.proposal_id]
    assert payload.ticket is controller.ready[0]
    payload.validate_for(state, full_window=True)
    assert payload.verification_tokens.tolist() == continuations[0].tolist()
    assert payload.next_tokens.tolist() == continuations[0].tolist()
    assert float(payload.draft_confidence) == pytest.approx(0.75)
    assert message is None
    assert timing is None
    assert torch.equal(confidence_values, local_confidences)
    broadcast.assert_not_called()


def test_slo_summary_preserves_worker_arrival_clock():
    from examples.benchmark_nano_pearl_speculative import _summarize_slo_metrics

    summary = _summarize_slo_metrics(
        [
            {
                "request_id": "a",
                "completion_token_ids": [1, 2],
                "slo_class": "tight",
                "slo_tpot_ms": 40.0,
                "observed_tpot_ms": 20.0,
                "paper_tpot_ms": 10.0,
                "slo_attained": True,
                "slo_goodput_tokens": 2,
                "arrival_ts": 100.0,
            },
            {
                "request_id": "b",
                "completion_token_ids": [1, 2],
                "slo_class": "tight",
                "slo_tpot_ms": 40.0,
                "observed_tpot_ms": 20.0,
                "paper_tpot_ms": 10.0,
                "slo_attained": True,
                "slo_goodput_tokens": 2,
                "arrival_ts": 112.0,
            },
        ],
        1.0,
        1.0,
    )

    assert [row["arrival_ts"] for row in summary["request_metrics"]] == [100.0, 112.0]
    assert [row["arrival_offset_sec"] for row in summary["request_metrics"]] == [0.0, 12.0]
