# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the compact fixed-gamma proposal transport."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.spec_decode.pearl import fixed_greedy_transport as transport


def test_unindexed_logical_device_matches_its_indexed_runtime_device() -> None:
    assert transport._device_matches(torch.device("npu:0"), torch.device("npu"))
    assert not transport._device_matches(torch.device("npu:1"), torch.device("npu:0"))
    assert not transport._device_matches(torch.device("cpu"), torch.device("npu"))


def test_full_window_layout_removes_per_row_metadata() -> None:
    layout = transport.CompactFixedGreedyEnvelopeLayout.full_window(
        8,
        4,
        include_draft_timing=True,
    )

    assert layout.verification_sizes == (4,) * 8
    assert layout.verification_token_count == 32
    assert layout.continuation_token_count == 32
    assert layout.message_numel == 33
    assert layout.legacy_message_numel == 121
    assert layout.share_full_window_tokens is True


@pytest.mark.parametrize(
    ("verification_sizes", "gamma"),
    [
        ((), 4),
        ((1,), 0),
        ((0,), 4),
        ((5,), 4),
        ((True,), 4),
    ],
)
def test_layout_rejects_invalid_dimensions(verification_sizes, gamma) -> None:
    with pytest.raises(ValueError):
        transport.CompactFixedGreedyEnvelopeLayout(verification_sizes, gamma)


def test_pack_and_unpack_return_zero_copy_views() -> None:
    layout = transport.CompactFixedGreedyEnvelopeLayout(
        (1, 3),
        4,
        include_draft_timing=True,
    )
    verification = torch.tensor([11, 21, 22, 23], dtype=torch.long)
    continuations = torch.tensor(
        [[31, 32, 33, 34], [41, 42, 43, 44]],
        dtype=torch.long,
    )
    out = torch.empty(layout.message_numel, dtype=torch.long)

    message = transport.pack_compact_fixed_greedy_envelope(
        layout,
        verification,
        continuations,
        draft_compute_us=torch.tensor(1234, dtype=torch.long),
        out=out,
    )
    views = transport.unpack_compact_fixed_greedy_envelope(layout, message)

    assert message is out
    assert torch.equal(views.verification_tokens, verification)
    assert torch.equal(views.continuation_tokens, continuations)
    assert [row.tolist() for row in views.verification_rows] == [[11], [21, 22, 23]]
    assert views.draft_compute_us is not None
    assert views.draft_compute_us.item() == 1234
    message[0] = 99
    assert views.verification_rows[0].item() == 99


def test_pack_validates_payload_contract_without_reading_device_values() -> None:
    layout = transport.CompactFixedGreedyEnvelopeLayout((2,), 4, include_draft_timing=True)
    verification = torch.tensor([1, 2], dtype=torch.long)
    continuation = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)

    with pytest.raises(ValueError, match="verification_tokens must use torch.long"):
        transport.pack_compact_fixed_greedy_envelope(
            layout,
            verification.float(),
            continuation,
            draft_compute_us=3,
        )
    with pytest.raises(ValueError, match="continuation_tokens must have shape"):
        transport.pack_compact_fixed_greedy_envelope(
            layout,
            verification,
            continuation[:, :3],
            draft_compute_us=3,
        )
    with pytest.raises(ValueError, match="draft_compute_us must be one torch.long value"):
        transport.pack_compact_fixed_greedy_envelope(
            layout,
            verification,
            continuation,
            draft_compute_us=torch.tensor([1, 2], dtype=torch.long),
        )


class _DeferredCopyWork:
    def __init__(self, destination: torch.Tensor, source: torch.Tensor):
        self.destination = destination
        self.source = source
        self.wait_count = 0

    def wait(self) -> None:
        self.wait_count += 1
        self.destination.copy_(self.source)


def test_receiver_does_not_expose_envelope_until_wait(monkeypatch) -> None:
    layout = transport.CompactFixedGreedyEnvelopeLayout.full_window(
        2,
        2,
        include_draft_timing=True,
    )
    expected = torch.tensor([10, 11, 20, 21, 900], dtype=torch.long)
    work_holder = {}

    def broadcast(destination, *, src, group, async_op):
        assert src == 0
        assert group == "verification-group"
        assert async_op is True
        work = _DeferredCopyWork(destination, expected)
        work_holder["work"] = work
        return work

    monkeypatch.setattr(transport.dist, "broadcast", broadcast)
    pending = transport.begin_compact_fixed_greedy_broadcast(
        layout,
        rank=1,
        source_rank=0,
        group="verification-group",
        device="cpu",
    )

    assert pending.completed is False
    assert pending.message_numel == expected.numel()
    assert work_holder["work"].wait_count == 0
    views = pending.wait()
    assert pending.completed is True
    assert work_holder["work"].wait_count == 1
    assert views.verification_tokens.tolist() == [10, 11, 20, 21]
    assert views.continuation_tokens.tolist() == [[10, 11], [20, 21]]
    assert views.draft_compute_us is not None
    assert views.draft_compute_us.item() == 900
    assert pending.wait() is views
    assert work_holder["work"].wait_count == 1


def test_source_starts_async_broadcast_and_retains_buffer(monkeypatch) -> None:
    layout = transport.CompactFixedGreedyEnvelopeLayout((1,), 2)
    work = Mock()

    def broadcast(message, *, src, group, async_op):
        assert message.tolist() == [7, 8, 9]
        assert src == 3
        assert group == "verification-group"
        assert async_op is True
        return work

    monkeypatch.setattr(transport.dist, "broadcast", broadcast)
    pending = transport.begin_compact_fixed_greedy_broadcast(
        layout,
        rank=3,
        source_rank=3,
        group="verification-group",
        device="cpu",
        verification_tokens=torch.tensor([7], dtype=torch.long),
        continuation_tokens=torch.tensor([[8, 9]], dtype=torch.long),
    )

    assert pending.completed is False
    work.wait.assert_not_called()
    views = pending.wait()
    work.wait.assert_called_once_with()
    assert views.verification_tokens.tolist() == [7]
    assert views.continuation_tokens.tolist() == [[8, 9]]
    assert views.draft_compute_us is None


def test_receiver_rejects_source_payload() -> None:
    layout = transport.CompactFixedGreedyEnvelopeLayout((1,), 2)
    with pytest.raises(ValueError, match="receivers must not provide"):
        transport.begin_compact_fixed_greedy_broadcast(
            layout,
            rank=1,
            source_rank=0,
            group=None,
            device="cpu",
            verification_tokens=torch.tensor([1], dtype=torch.long),
        )
