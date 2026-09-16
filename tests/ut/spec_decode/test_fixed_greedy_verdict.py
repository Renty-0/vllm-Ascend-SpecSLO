# SPDX-License-Identifier: Apache-2.0
"""Contracts for PEARL's fused fixed-γ greedy verdict."""

from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.spec_decode.pearl.native_engine as native_engine
from vllm_ascend.ops.triton.spec_decode.fixed_greedy_verdict import (
    fixed_greedy_full_window_bonus_verdict,
    fixed_greedy_full_window_verdict,
)


def test_fixed_greedy_full_window_verdict_covers_every_rejection_position():
    draft = torch.tensor(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
            [40, 41, 42, 43],
            [50, 51, 52, 53],
        ],
        dtype=torch.long,
    ).flatten()
    target = torch.tensor(
        [
            [90, 11, 12, 13],
            [20, 91, 22, 23],
            [30, 31, 92, 23],
            [40, 41, 42, 93],
            [50, 51, 52, 53],
        ],
        dtype=torch.long,
    ).flatten()

    verdict = fixed_greedy_full_window_verdict(target, draft)

    assert verdict.dtype == torch.long
    assert verdict.tolist() == [[0, 90], [1, 91], [2, 92], [3, 93], [4, -1]]


def test_fixed_greedy_full_window_verdict_returns_enabled_bonus_only_on_full_match():
    draft = torch.tensor(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
            [40, 41, 42, 43],
        ],
        dtype=torch.long,
    ).flatten()
    target = draft.clone()
    target[5] = 91
    bonus_token_ids = torch.tensor([100, 101, 102, 103], dtype=torch.long)
    bonus_enabled_mask = torch.tensor([True, True, False, True], dtype=torch.bool)

    verdict = fixed_greedy_full_window_verdict(
        target,
        draft,
        bonus_token_ids,
        bonus_enabled_mask,
    )

    assert verdict.shape == (4, 2)
    assert verdict.tolist() == [[4, 100], [1, 91], [4, -1], [4, 103]]


@pytest.mark.parametrize("flat_draft", [False, True])
def test_fixed_greedy_full_window_bonus_verdict_consumes_strided_target_rows(
    flat_draft,
):
    draft = torch.tensor(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
            [40, 41, 42, 43],
        ],
        dtype=torch.long,
    )
    target_rows = torch.tensor(
        [
            [10, 11, 12, 13, 100],
            [20, 91, 22, 23, 101],
            [30, 31, 32, 33, 102],
            [40, 41, 42, 43, 103],
        ],
        dtype=torch.long,
    )
    bonus_enabled_mask = torch.tensor(
        [True, True, False, True],
        dtype=torch.bool,
    )

    verdict = fixed_greedy_full_window_bonus_verdict(
        target_rows,
        draft.flatten() if flat_draft else draft,
        bonus_enabled_mask,
    )

    assert verdict.shape == (4, 2)
    assert verdict.tolist() == [[4, 100], [1, 91], [4, -1], [4, 103]]


def test_fixed_greedy_full_window_bonus_verdict_accepts_an_empty_batch():
    verdict = fixed_greedy_full_window_bonus_verdict(
        torch.empty((0, 5), dtype=torch.long),
        torch.empty((0, 4), dtype=torch.long),
        torch.empty(0, dtype=torch.bool),
    )

    assert verdict.shape == (0, 2)
    assert verdict.dtype == torch.long


@pytest.mark.parametrize(
    ("target_rows", "draft", "mask", "match"),
    [
        (
            torch.ones(10, dtype=torch.long),
            torch.ones(8, dtype=torch.long),
            torch.ones(2, dtype=torch.bool),
            "target tokens must have shape",
        ),
        (
            torch.ones((2, 4), dtype=torch.long),
            torch.ones(8, dtype=torch.long),
            torch.ones(2, dtype=torch.bool),
            "target tokens must have shape",
        ),
        (
            torch.ones((2, 5), dtype=torch.long),
            torch.ones((2, 5), dtype=torch.long),
            torch.ones(2, dtype=torch.bool),
            "draft tokens must have shape",
        ),
        (
            torch.ones((2, 5), dtype=torch.long),
            torch.ones(8, dtype=torch.long),
            torch.ones((2, 1), dtype=torch.bool),
            "enabled mask must have shape",
        ),
        (
            torch.ones((2, 5), dtype=torch.int32),
            torch.ones(8, dtype=torch.long),
            torch.ones(2, dtype=torch.bool),
            "token tensors must use torch.long",
        ),
        (
            torch.ones((2, 5), dtype=torch.long),
            torch.ones(8, dtype=torch.int32),
            torch.ones(2, dtype=torch.bool),
            "token tensors must use torch.long",
        ),
        (
            torch.ones((2, 5), dtype=torch.long),
            torch.ones(8, dtype=torch.long),
            torch.ones(2, dtype=torch.int32),
            "enabled mask must use torch.bool",
        ),
        (
            torch.ones((2, 10), dtype=torch.long)[:, ::2],
            torch.ones(8, dtype=torch.long),
            torch.ones(2, dtype=torch.bool),
            "contiguous",
        ),
        (
            torch.ones((2, 5), dtype=torch.long),
            torch.ones((2, 8), dtype=torch.long)[:, ::2],
            torch.ones(2, dtype=torch.bool),
            "contiguous",
        ),
        (
            torch.ones((2, 5), dtype=torch.long),
            torch.ones(8, dtype=torch.long),
            torch.tensor([True, False, True, False])[::2],
            "contiguous",
        ),
    ],
)
def test_fixed_greedy_full_window_bonus_verdict_rejects_invalid_contract(
    target_rows,
    draft,
    mask,
    match,
):
    with pytest.raises(ValueError, match=match):
        fixed_greedy_full_window_bonus_verdict(target_rows, draft, mask)


def test_fixed_greedy_full_window_bonus_verdict_requires_one_device():
    target_rows = torch.ones((2, 5), dtype=torch.long)
    draft = torch.ones(8, dtype=torch.long)
    mask = torch.ones(2, dtype=torch.bool, device="meta")

    with pytest.raises(ValueError, match="share one device"):
        fixed_greedy_full_window_bonus_verdict(target_rows, draft, mask)


@pytest.mark.parametrize(
    ("target", "draft", "match"),
    [
        (torch.ones((2, 4), dtype=torch.long), torch.ones(8, dtype=torch.long), "one-dimensional"),
        (torch.ones(8, dtype=torch.int32), torch.ones(8, dtype=torch.int32), "torch.long"),
        (torch.ones(5, dtype=torch.long), torch.ones(5, dtype=torch.long), "divisible by four"),
        (torch.arange(16, dtype=torch.long).reshape(4, 4)[:, 0], torch.ones(4, dtype=torch.long), "contiguous"),
    ],
)
def test_fixed_greedy_full_window_verdict_rejects_invalid_contract(target, draft, match):
    with pytest.raises(ValueError, match=match):
        fixed_greedy_full_window_verdict(target, draft)


def test_fixed_greedy_full_window_verdict_accepts_an_empty_cpu_batch():
    verdict = fixed_greedy_full_window_verdict(
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
    )

    assert verdict.shape == (0, 2)
    assert verdict.dtype == torch.long


def test_fixed_greedy_full_window_verdict_accepts_an_empty_bonus_batch():
    verdict = fixed_greedy_full_window_verdict(
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.bool),
    )

    assert verdict.shape == (0, 2)
    assert verdict.dtype == torch.long


@pytest.mark.parametrize(
    ("bonus_token_ids", "bonus_enabled_mask", "match"),
    [
        (torch.ones(2, dtype=torch.long), None, "supplied together"),
        (None, torch.ones(2, dtype=torch.bool), "supplied together"),
        (torch.ones((2, 1), dtype=torch.long), torch.ones(2, dtype=torch.bool), "shape"),
        (torch.ones(3, dtype=torch.long), torch.ones(3, dtype=torch.bool), "shape"),
        (torch.ones(2, dtype=torch.int32), torch.ones(2, dtype=torch.bool), "torch.long"),
        (torch.ones(2, dtype=torch.long), torch.ones(2, dtype=torch.int32), "torch.bool"),
        (
            torch.arange(4, dtype=torch.long)[::2],
            torch.tensor([True, False]),
            "contiguous",
        ),
        (
            torch.ones(2, dtype=torch.long),
            torch.tensor([True, False, True, False])[::2],
            "contiguous",
        ),
    ],
)
def test_fixed_greedy_full_window_verdict_rejects_invalid_bonus_contract(
    bonus_token_ids,
    bonus_enabled_mask,
    match,
):
    target = torch.arange(8, dtype=torch.long)
    draft = target.clone()

    with pytest.raises(ValueError, match=match):
        fixed_greedy_full_window_verdict(
            target,
            draft,
            bonus_token_ids,
            bonus_enabled_mask,
        )


@pytest.mark.skipif(
    not hasattr(torch, "device"),
    reason="torch device support is required",
)
def test_fixed_greedy_full_window_verdict_rejects_bonus_on_another_device():
    target = torch.arange(8, dtype=torch.long)
    draft = target.clone()
    bonus_token_ids = torch.ones(2, dtype=torch.long, device="meta")
    bonus_enabled_mask = torch.ones(2, dtype=torch.bool, device="meta")

    with pytest.raises(ValueError, match="share one device"):
        fixed_greedy_full_window_verdict(
            target,
            draft,
            bonus_token_ids,
            bonus_enabled_mask,
        )


def _verification_engine(*, full_window: bool):
    return SimpleNamespace(
        groups=SimpleNamespace(is_verification_worker=True),
        rank=1,
        topology=SimpleNamespace(target_leader_rank=1),
        is_draft=False,
        gamma=4,
        config=SimpleNamespace(
            spec_rhythm_cpu_verdict=False,
            spec_rhythm_linear_full_window=full_window,
        ),
        greedy_verification_layouts={},
    )


def test_native_verifier_routes_only_full_gamma4_windows_to_fused_builder(monkeypatch):
    target = torch.arange(8, dtype=torch.long)
    draft = target.clone()
    sentinel = torch.tensor([[4, -1], [4, -1]], dtype=torch.long)
    calls = []

    def fake_fused(target_tokens, draft_tokens):
        calls.append((target_tokens, draft_tokens))
        return sentinel

    monkeypatch.setattr(native_engine, "fixed_greedy_full_window_verdict", fake_fused)
    engine = _verification_engine(full_window=True)

    verdict = native_engine.NativePearlEngine._verify_target_tokens_batch(
        engine,
        target,
        None,
        draft,
        [4, 4],
        [0.0, 0.0],
    )

    assert verdict is sentinel
    assert len(calls) == 1
    assert calls[0][0] is target
    assert torch.equal(calls[0][1], draft)


def test_native_verifier_keeps_legacy_uniform_gamma4_route_outside_full_window(monkeypatch):
    target = torch.tensor([1, 2, 9, 4], dtype=torch.long)
    draft = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("legacy PEARL must not use the full-window-only kernel")

    monkeypatch.setattr(native_engine, "fixed_greedy_full_window_verdict", fail_if_called)
    engine = _verification_engine(full_window=False)

    verdict = native_engine.NativePearlEngine._verify_target_tokens_batch(
        engine,
        target,
        None,
        draft,
        [4],
        [0.0],
    )

    assert verdict.tolist() == [[2, 9]]
