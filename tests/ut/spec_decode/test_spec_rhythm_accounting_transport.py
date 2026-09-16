# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only checks for reusable SpecRhythm accounting envelopes."""

from __future__ import annotations

import pytest
import torch

from vllm_ascend.spec_decode.pearl import accounting_transport as transport


def test_full_rewrite_reuses_buffer_and_leaves_no_stale_fields() -> None:
    envelope = transport.ReusableSpecRhythmEnvelope(8, device="cpu")
    buffer = envelope.buffer
    staging = envelope.host_staging
    data_ptr = buffer.data_ptr()

    first = [4.5, 99.0, 7, 8, 9, 0.1, 0.2, 0.3]
    second = [0.0, 0.0, -1, -1, -1, -1.0, -1.0, -1.0]
    assert envelope.rewrite(first) is buffer
    assert torch.equal(buffer, torch.tensor(first, dtype=torch.float64))
    assert envelope.rewrite(second) is buffer

    assert envelope.buffer is buffer
    assert envelope.host_staging is staging
    assert envelope.buffer.data_ptr() == data_ptr
    assert torch.equal(buffer, torch.tensor(second, dtype=torch.float64))


def test_wrong_size_is_rejected_before_persistent_content_changes() -> None:
    envelope = transport.ReusableSpecRhythmEnvelope(4, device="cpu")
    envelope.rewrite([1.0, 2.0, 3.0, 4.0])
    before = envelope.buffer.clone()

    with pytest.raises(ValueError, match="expected 4, got 3"):
        envelope.rewrite([9.0, 8.0, 7.0])

    assert torch.equal(envelope.buffer, before)


def test_accounting_and_tail_keep_collective_envelopes_and_order(monkeypatch) -> None:
    # B=2 primary layout: [elapsed, wall, active x2, finish x2]. Tail layout:
    # [tail elapsed, boundary, frontier x3]. These are the exact fixed-field
    # contracts consumed by the native loop; reuse must alter neither.
    accounting = transport.ReusableSpecRhythmEnvelope(6, device="cpu")
    tail = transport.ReusableSpecRhythmEnvelope(5, device="cpu")
    group = object()
    calls: list[tuple[int, object, int, list[float]]] = []

    def broadcast(message, *, src, group):
        calls.append((src, group, message.data_ptr(), message.tolist()))

    monkeypatch.setattr(transport.dist, "broadcast", broadcast)
    first_accounting = [0.04, 100.0, 3, 7, -1.0, 0.025]
    first_tail = [0.002, 101.0, 1, 2, 3]
    second_accounting = [0.05, 102.0, 9, -1, 0.03, -1.0]
    second_tail = [0.0, 103.0, 4, 0, -1]

    accounting.rewrite(first_accounting)
    assert accounting.broadcast_and_materialize(source_rank=1, group=group) == first_accounting
    tail.rewrite(first_tail)
    assert tail.broadcast_and_materialize(source_rank=1, group=group) == first_tail
    accounting.rewrite(second_accounting)
    assert accounting.broadcast_and_materialize(source_rank=1, group=group) == second_accounting
    tail.rewrite(second_tail)
    assert tail.broadcast_and_materialize(source_rank=1, group=group) == second_tail

    assert [(src, seen_group) for src, seen_group, _, _ in calls] == [
        (1, group),
        (1, group),
        (1, group),
        (1, group),
    ]
    assert [payload for _, _, _, payload in calls] == [
        first_accounting,
        first_tail,
        second_accounting,
        second_tail,
    ]
    assert calls[0][2] == calls[2][2]
    assert calls[1][2] == calls[3][2]
