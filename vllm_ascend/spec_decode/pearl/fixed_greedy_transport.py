# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact asynchronous transport for fixed-gamma greedy proposals.

This module deliberately contains no scheduler or model-runner integration.
It provides the data-plane contract needed by callers that can safely overlap
a draft-to-target HCCL broadcast with verification of the previous proposal.
The current native CANN/HCCL engine deliberately does not early-post that
broadcast: it completes the local model streams and Gloo coordination first to
avoid a cross-communicator stream-wait cycle.  The caller remains responsible
for choosing the safe ordering and for keeping the control plane (ticket order,
verification sizes, and gamma) identical on every participating rank.

Unlike the self-describing SpecRhythm mailbox, the compact envelope carries no
per-row routing metadata or confidence values.  It is therefore suitable only
for deterministic greedy execution after both sides have agreed on a
``CompactFixedGreedyEnvelopeLayout``.  Mailbox validation and dynamic-gamma
execution must continue to use the self-describing envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import torch
import torch.distributed as dist


def _device_matches(actual: torch.device, expected: torch.device) -> bool:
    """Treat an unindexed logical device as the current device of that type."""
    return actual.type == expected.type and (
        expected.index is None or actual.index == expected.index
    )


@dataclass(frozen=True)
class CompactFixedGreedyEnvelopeLayout:
    """Locally agreed layout of one compact proposal envelope.

    ``verification_sizes`` contains one positive width per proposal row.  The
    continuation matrix always has the fixed shape ``(row_count, gamma)``.
    Keeping the verification widths in the local control plane permits the
    compact message to omit the old seven-int64 metadata row without losing
    the boundaries needed by the receiver.
    """

    verification_sizes: tuple[int, ...]
    gamma: int
    include_draft_timing: bool = False
    share_full_window_tokens: bool = False

    def __post_init__(self) -> None:
        verification_sizes = tuple(self.verification_sizes)
        object.__setattr__(self, "verification_sizes", verification_sizes)
        if not verification_sizes:
            raise ValueError("A compact fixed-greedy envelope requires at least one proposal row.")
        if not isinstance(self.gamma, Integral) or isinstance(self.gamma, bool) or self.gamma <= 0:
            raise ValueError("Compact fixed-greedy gamma must be a positive integer.")
        if any(
            not isinstance(size, Integral) or isinstance(size, bool) or size <= 0 or size > self.gamma
            for size in verification_sizes
        ):
            raise ValueError("Compact verification sizes must be integers in [1, gamma].")
        if self.share_full_window_tokens and any(
            size != self.gamma for size in verification_sizes
        ):
            raise ValueError(
                "A shared full-window envelope requires every verification row to use gamma tokens."
            )

    @classmethod
    def full_window(
        cls,
        row_count: int,
        gamma: int,
        *,
        include_draft_timing: bool = False,
    ) -> CompactFixedGreedyEnvelopeLayout:
        """Build the common full-window layout with ``gamma`` tokens per row."""
        if not isinstance(row_count, Integral) or isinstance(row_count, bool) or row_count <= 0:
            raise ValueError("Compact fixed-greedy row count must be a positive integer.")
        return cls(
            verification_sizes=(gamma,) * int(row_count),
            gamma=gamma,
            include_draft_timing=include_draft_timing,
            share_full_window_tokens=True,
        )

    @property
    def row_count(self) -> int:
        return len(self.verification_sizes)

    @property
    def verification_token_count(self) -> int:
        return sum(self.verification_sizes)

    @property
    def continuation_token_count(self) -> int:
        return self.row_count * self.gamma

    @property
    def message_numel(self) -> int:
        proposal_tokens = (
            self.continuation_token_count
            if self.share_full_window_tokens
            else self.verification_token_count + self.continuation_token_count
        )
        return proposal_tokens + int(self.include_draft_timing)

    @property
    def legacy_message_numel(self) -> int:
        """Size of the current seven-metadata-word self-describing envelope."""
        return (
            self.verification_token_count
            + self.continuation_token_count
            + int(self.include_draft_timing)
            + 7 * self.row_count
        )


@dataclass(frozen=True)
class CompactFixedGreedyEnvelopeViews:
    """Zero-copy views into a completed compact envelope."""

    verification_tokens: torch.Tensor
    continuation_tokens: torch.Tensor
    draft_compute_us: torch.Tensor | None
    verification_sizes: tuple[int, ...]

    @property
    def verification_rows(self) -> tuple[torch.Tensor, ...]:
        """Return zero-copy per-request verification slices."""
        rows: list[torch.Tensor] = []
        offset = 0
        for size in self.verification_sizes:
            rows.append(self.verification_tokens.narrow(0, offset, size))
            offset += size
        return tuple(rows)


def _validate_message_buffer(
    message: torch.Tensor,
    layout: CompactFixedGreedyEnvelopeLayout,
    *,
    name: str,
) -> None:
    if message.dtype != torch.long:
        raise ValueError(f"{name} must use torch.long.")
    if message.ndim != 1 or message.numel() != layout.message_numel:
        raise ValueError(f"{name} must have shape ({layout.message_numel},).")
    if not message.is_contiguous():
        raise ValueError(f"{name} must be contiguous for collective communication.")


def _validate_payload_tensor(
    value: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if value.dtype != torch.long:
        raise ValueError(f"{name} must use torch.long.")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}.")


def pack_compact_fixed_greedy_envelope(
    layout: CompactFixedGreedyEnvelopeLayout,
    verification_tokens: torch.Tensor,
    continuation_tokens: torch.Tensor,
    *,
    draft_compute_us: int | torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack one compact proposal without any device-to-host synchronization.

    Passing ``out`` allows a caller to reuse a fixed-shape device allocation.
    That buffer must not be reused until the pending collective has completed.
    """
    device = verification_tokens.device
    _validate_payload_tensor(
        verification_tokens,
        name="verification_tokens",
        shape=(layout.verification_token_count,),
        device=device,
    )
    _validate_payload_tensor(
        continuation_tokens,
        name="continuation_tokens",
        shape=(layout.row_count, layout.gamma),
        device=device,
    )
    timing: torch.Tensor | None = None
    if layout.include_draft_timing:
        if draft_compute_us is None:
            raise ValueError("This compact envelope layout requires draft_compute_us.")
        if torch.is_tensor(draft_compute_us):
            if draft_compute_us.dtype != torch.long or draft_compute_us.numel() != 1:
                raise ValueError("draft_compute_us must be one torch.long value.")
            if draft_compute_us.device != device:
                raise ValueError(f"draft_compute_us must be on {device}, got {draft_compute_us.device}.")
            timing = draft_compute_us.reshape(())
        else:
            if not isinstance(draft_compute_us, Integral) or isinstance(draft_compute_us, bool) or draft_compute_us < 0:
                raise ValueError("draft_compute_us must be a non-negative integer.")
            timing = torch.tensor(draft_compute_us, dtype=torch.long, device=device)
    elif draft_compute_us is not None:
        raise ValueError("draft_compute_us was provided for a layout without timing.")

    flat_continuations = continuation_tokens.reshape(-1)
    if out is None:
        parts = (
            [flat_continuations]
            if layout.share_full_window_tokens
            else [verification_tokens, flat_continuations]
        )
        if timing is not None:
            parts.append(timing.reshape(1))
        # The allocation path deliberately retains torch.cat: on device it is
        # one packing launch, whereas allocating first and issuing two or three
        # slice copies can cost more than the tiny message itself.
        message = torch.cat(parts)
    else:
        _validate_message_buffer(out, layout, name="out")
        if out.device != device:
            raise ValueError(f"out must be on {device}, got {out.device}.")
        message = out
        if layout.share_full_window_tokens:
            verification_end = 0
            continuation_end = layout.continuation_token_count
            message[:continuation_end].copy_(flat_continuations)
        else:
            verification_end = layout.verification_token_count
            continuation_end = verification_end + layout.continuation_token_count
            message[:verification_end].copy_(verification_tokens)
            message[verification_end:continuation_end].copy_(flat_continuations)
        if timing is not None:
            message[continuation_end].copy_(timing)
    return message


def unpack_compact_fixed_greedy_envelope(
    layout: CompactFixedGreedyEnvelopeLayout,
    message: torch.Tensor,
) -> CompactFixedGreedyEnvelopeViews:
    """Return zero-copy views into a completed compact proposal message."""
    _validate_message_buffer(message, layout, name="message")
    verification_end = layout.verification_token_count
    if layout.share_full_window_tokens:
        proposal_tokens = message[: layout.continuation_token_count]
        verification_tokens = proposal_tokens
        continuation_tokens = proposal_tokens.reshape(
            layout.row_count,
            layout.gamma,
        )
        continuation_end = layout.continuation_token_count
    else:
        continuation_end = verification_end + layout.continuation_token_count
        verification_tokens = message[:verification_end]
        continuation_tokens = message[verification_end:continuation_end].reshape(
            layout.row_count,
            layout.gamma,
        )
    timing = message[continuation_end].reshape(()) if layout.include_draft_timing else None
    return CompactFixedGreedyEnvelopeViews(
        verification_tokens=verification_tokens,
        continuation_tokens=continuation_tokens,
        draft_compute_us=timing,
        verification_sizes=layout.verification_sizes,
    )


@dataclass
class PendingCompactFixedGreedyEnvelope:
    """Own an in-flight collective and its message buffer until completion."""

    layout: CompactFixedGreedyEnvelopeLayout
    _message: torch.Tensor
    _work: Any
    _views: CompactFixedGreedyEnvelopeViews | None = None

    @property
    def completed(self) -> bool:
        """Whether :meth:`wait` has established that the buffer is readable."""
        return self._views is not None

    @property
    def message_numel(self) -> int:
        return self._message.numel()

    def wait(self) -> CompactFixedGreedyEnvelopeViews:
        """Wait once for HCCL and expose the completed, zero-copy views."""
        if self._views is None:
            if self._work is not None:
                self._work.wait()
            self._views = unpack_compact_fixed_greedy_envelope(self.layout, self._message)
        return self._views


def begin_compact_fixed_greedy_broadcast(
    layout: CompactFixedGreedyEnvelopeLayout,
    *,
    rank: int,
    source_rank: int,
    group: dist.ProcessGroup,
    device: torch.device | str,
    verification_tokens: torch.Tensor | None = None,
    continuation_tokens: torch.Tensor | None = None,
    draft_compute_us: int | torch.Tensor | None = None,
    buffer: torch.Tensor | None = None,
) -> PendingCompactFixedGreedyEnvelope:
    """Start a compact proposal broadcast and return without waiting.

    Only ``source_rank`` supplies payload tensors.  Receivers provide a device
    (and optionally a reusable ``buffer``) and wait immediately before the new
    proposal is materialized into scheduler state.  Starting the receive before
    target verification is what permits D->T communication to overlap that
    independent work.
    """
    if not isinstance(rank, Integral) or isinstance(rank, bool) or rank < 0:
        raise ValueError("rank must be a non-negative integer.")
    if not isinstance(source_rank, Integral) or isinstance(source_rank, bool) or source_rank < 0:
        raise ValueError("source_rank must be a non-negative integer.")
    actual_device = torch.device(device)
    is_source = rank == source_rank
    if is_source:
        if verification_tokens is None or continuation_tokens is None:
            raise ValueError("The compact-envelope source must provide both payload tensors.")
        message = pack_compact_fixed_greedy_envelope(
            layout,
            verification_tokens,
            continuation_tokens,
            draft_compute_us=draft_compute_us,
            out=buffer,
        )
        if not _device_matches(message.device, actual_device):
            raise ValueError(f"Source payload must be on {actual_device}, got {message.device}.")
    else:
        if verification_tokens is not None or continuation_tokens is not None or draft_compute_us is not None:
            raise ValueError("Compact-envelope receivers must not provide source payload values.")
        if buffer is None:
            message = torch.empty(layout.message_numel, dtype=torch.long, device=actual_device)
        else:
            _validate_message_buffer(buffer, layout, name="buffer")
            if not _device_matches(buffer.device, actual_device):
                raise ValueError(f"buffer must be on {actual_device}, got {buffer.device}.")
            message = buffer

    work = dist.broadcast(
        message,
        src=int(source_rank),
        group=group,
        async_op=True,
    )
    return PendingCompactFixedGreedyEnvelope(layout, message, work)
