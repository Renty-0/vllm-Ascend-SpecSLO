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
per-row routing metadata.  It is therefore suitable only for deterministic
greedy execution after both sides have agreed on a
``CompactFixedGreedyEnvelopeLayout``.  Mailbox validation and dynamic-gamma
execution must continue to use the self-describing envelope.
"""

from __future__ import annotations

import time
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
    include_draft_confidence: bool = False
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
        include_draft_confidence: bool = False,
    ) -> CompactFixedGreedyEnvelopeLayout:
        """Build the common full-window layout with ``gamma`` tokens per row."""
        if not isinstance(row_count, Integral) or isinstance(row_count, bool) or row_count <= 0:
            raise ValueError("Compact fixed-greedy row count must be a positive integer.")
        return cls(
            verification_sizes=(gamma,) * int(row_count),
            gamma=gamma,
            include_draft_timing=include_draft_timing,
            include_draft_confidence=include_draft_confidence,
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
        return (
            proposal_tokens
            + self.row_count * int(self.include_draft_confidence)
            + int(self.include_draft_timing)
        )

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
    # Quantized to one part per million so the complete envelope can retain a
    # single HCCL-friendly int64 dtype without a second collective.
    draft_confidences: torch.Tensor | None
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


@dataclass(frozen=True)
class CompactFixedGreedyCPUBroadcastTiming:
    """Host timing for one CPU/Gloo proposal publication."""

    device_to_host_seconds: float
    broadcast_seconds: float
    host_to_device_submit_seconds: float
    # Every participant already owns the complete host proposal envelope
    # after Gloo. Expose the continuation view so the service can retain it
    # with the guarded proposal and avoid a later NPU->CPU round trip during
    # correction/state update.
    host_continuation_tokens: torch.Tensor | None = None


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
    draft_confidences: torch.Tensor | None = None,
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
    quantized_confidences: torch.Tensor | None = None
    if layout.include_draft_confidence:
        if draft_confidences is None:
            raise ValueError("This compact envelope layout requires draft_confidences.")
        if (
            not draft_confidences.dtype.is_floating_point
            or tuple(draft_confidences.shape) != (layout.row_count,)
            or draft_confidences.device != device
        ):
            raise ValueError(
                "draft_confidences must be one floating-point value per row "
                f"on {device}."
            )
        quantized_confidences = torch.round(
            draft_confidences.float().clamp(0.0, 1.0) * 1_000_000
        ).to(torch.long)
    elif draft_confidences is not None:
        raise ValueError("draft_confidences were provided for a layout without confidence.")

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
        if quantized_confidences is not None:
            parts.append(quantized_confidences)
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
        confidence_end = continuation_end
        if quantized_confidences is not None:
            confidence_end += layout.row_count
            message[continuation_end:confidence_end].copy_(quantized_confidences)
        if timing is not None:
            message[confidence_end].copy_(timing)
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
    confidence_end = continuation_end
    confidences = None
    if layout.include_draft_confidence:
        confidence_end += layout.row_count
        confidences = message[continuation_end:confidence_end]
    timing = message[confidence_end].reshape(()) if layout.include_draft_timing else None
    return CompactFixedGreedyEnvelopeViews(
        verification_tokens=verification_tokens,
        continuation_tokens=continuation_tokens,
        draft_confidences=confidences,
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
    draft_confidences: torch.Tensor | None = None,
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
            draft_confidences=draft_confidences,
            draft_compute_us=draft_compute_us,
            out=buffer,
        )
        if not _device_matches(message.device, actual_device):
            raise ValueError(f"Source payload must be on {actual_device}, got {message.device}.")
    else:
        if (
            verification_tokens is not None
            or continuation_tokens is not None
            or draft_confidences is not None
            or draft_compute_us is not None
        ):
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


def broadcast_compact_fixed_greedy_via_cpu(
    layout: CompactFixedGreedyEnvelopeLayout,
    *,
    rank: int,
    source_rank: int,
    group: dist.ProcessGroup,
    device: torch.device | str,
    verification_tokens: torch.Tensor | None = None,
    continuation_tokens: torch.Tensor | None = None,
    draft_confidences: torch.Tensor | None = None,
    draft_compute_us: int | torch.Tensor | None = None,
    source_device_buffer: torch.Tensor | None = None,
    source_cpu_buffer: torch.Tensor | None = None,
) -> tuple[
    CompactFixedGreedyEnvelopeViews,
    CompactFixedGreedyCPUBroadcastTiming,
]:
    """Publish a compact proposal through a CPU process group.

    Only the draft source waits for its proposal graph while copying the tiny
    envelope to host. Target ranks can participate in Gloo while their TP
    model work remains queued. Receiver copies are then submitted to each
    current device stream, preserving the dependency before next-cycle use
    without a global target-stream fence.
    """

    if not isinstance(rank, Integral) or isinstance(rank, bool) or rank < 0:
        raise ValueError("rank must be a non-negative integer.")
    if not isinstance(source_rank, Integral) or isinstance(source_rank, bool) or source_rank < 0:
        raise ValueError("source_rank must be a non-negative integer.")
    actual_device = torch.device(device)
    is_source = rank == source_rank
    device_message: torch.Tensor | None = None
    device_to_host_started = time.perf_counter()
    if is_source:
        if verification_tokens is None or continuation_tokens is None:
            raise ValueError("The CPU compact-envelope source must provide both payload tensors.")
        if source_device_buffer is not None:
            _validate_message_buffer(
                source_device_buffer,
                layout,
                name="source_device_buffer",
            )
            if not _device_matches(source_device_buffer.device, actual_device):
                raise ValueError(
                    "source_device_buffer must be on "
                    f"{actual_device}, got {source_device_buffer.device}."
                )
        if source_cpu_buffer is not None:
            _validate_message_buffer(
                source_cpu_buffer,
                layout,
                name="source_cpu_buffer",
            )
            if source_cpu_buffer.device.type != "cpu":
                raise ValueError("source_cpu_buffer must be on CPU.")
        device_message = pack_compact_fixed_greedy_envelope(
            layout,
            verification_tokens,
            continuation_tokens,
            draft_confidences=draft_confidences,
            draft_compute_us=draft_compute_us,
            out=source_device_buffer,
        )
        if not _device_matches(device_message.device, actual_device):
            raise ValueError(
                f"Source payload must be on {actual_device}, got {device_message.device}."
            )
        if source_cpu_buffer is None:
            cpu_message = device_message.detach().cpu()
        else:
            cpu_message = source_cpu_buffer
            async_device_to_host = bool(
                actual_device.type == "npu" and cpu_message.is_pinned()
            )
            cpu_message.copy_(
                device_message,
                non_blocking=async_device_to_host,
            )
            # Gloo reads host memory immediately, so only this tiny copy must
            # complete before publication. The allocation survives cycles.
            if async_device_to_host:
                torch.npu.current_stream(device=actual_device).synchronize()
    else:
        if source_device_buffer is not None or source_cpu_buffer is not None:
            raise ValueError("CPU compact-envelope receivers cannot provide source buffers.")
        if (
            verification_tokens is not None
            or continuation_tokens is not None
            or draft_confidences is not None
            or draft_compute_us is not None
        ):
            raise ValueError("CPU compact-envelope receivers must not provide source payload values.")
        # Gloo writes directly into a pinned receiver buffer.  The following
        # tiny H2D transfer can then be queued behind the already-submitted
        # target verification instead of blocking Python until that target
        # stream becomes idle.
        cpu_message = torch.empty(
            layout.message_numel,
            dtype=torch.long,
            pin_memory=actual_device.type == "npu",
        )
    device_to_host_ended = time.perf_counter()

    broadcast_started = device_to_host_ended
    dist.broadcast(
        cpu_message,
        src=int(source_rank),
        group=group,
    )
    broadcast_ended = time.perf_counter()

    host_to_device_started = broadcast_ended
    if device_message is None:
        device_message = cpu_message.to(
            device=actual_device,
            non_blocking=cpu_message.is_pinned(),
        )
    host_to_device_ended = time.perf_counter()
    host_continuation_tokens = unpack_compact_fixed_greedy_envelope(
        layout,
        cpu_message,
    ).continuation_tokens
    return (
        unpack_compact_fixed_greedy_envelope(layout, device_message),
        CompactFixedGreedyCPUBroadcastTiming(
            device_to_host_seconds=(
                device_to_host_ended - device_to_host_started if is_source else 0.0
            ),
            broadcast_seconds=broadcast_ended - broadcast_started,
            host_to_device_submit_seconds=(
                host_to_device_ended - host_to_device_started if not is_source else 0.0
            ),
            host_continuation_tokens=host_continuation_tokens,
        ),
    )
