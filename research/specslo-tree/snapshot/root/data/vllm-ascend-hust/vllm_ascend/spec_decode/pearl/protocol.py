# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-shape HCCL payloads for PEARL proposal and verification exchange."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

_HEADER_WIDTH = 2
_NO_CORRECTION_TOKEN_ID = -1
_PADDING_TOKEN_ID = -1


def _normalize_token_rows(token_rows: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(token_id for token_id in row) for row in token_rows)


@dataclass(frozen=True)
class PearlProposalBatch:
    """Candidate token windows produced by the draft leader.

    ``request_slots`` are stable integer identifiers owned by the future
    vLLM-worker bridge. They deliberately avoid serializing vLLM request IDs
    through HCCL. Candidate windows may have different lengths, although a
    production PEARL scheduler normally uses a common gamma per batch.
    """

    request_slots: tuple[int, ...]
    candidate_token_ids: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        request_slots = tuple(self.request_slots)
        candidate_token_ids = _normalize_token_rows(tuple(self.candidate_token_ids))
        object.__setattr__(self, "request_slots", request_slots)
        object.__setattr__(self, "candidate_token_ids", candidate_token_ids)

        if not request_slots:
            raise ValueError("A PEARL proposal batch must contain at least one request.")
        if len(request_slots) != len(candidate_token_ids):
            raise ValueError("Proposal request slots and candidate windows must have the same length.")
        if len(set(request_slots)) != len(request_slots) or any(slot < 0 for slot in request_slots):
            raise ValueError("Proposal request slots must be unique non-negative integers.")
        if any(not window for window in candidate_token_ids):
            raise ValueError("Every PEARL proposal must contain at least one token.")
        if any(token_id < 0 for window in candidate_token_ids for token_id in window):
            raise ValueError("Candidate token IDs must be non-negative.")

    @property
    def batch_size(self) -> int:
        """Number of requests in the batch."""
        return len(self.request_slots)

    @property
    def window_size(self) -> int:
        """Width of the padded candidate-token tensor."""
        return max(len(window) for window in self.candidate_token_ids)

    def to_tensors(self, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the batch into fixed-shape tensors suitable for HCCL broadcast."""
        request_slots = torch.tensor(self.request_slots, dtype=torch.int64, device=device)
        candidate_lengths = torch.tensor(
            [len(window) for window in self.candidate_token_ids], dtype=torch.int64, device=device
        )
        candidate_token_ids = torch.full(
            (self.batch_size, self.window_size), _PADDING_TOKEN_ID, dtype=torch.int64, device=device
        )
        for row_index, window in enumerate(self.candidate_token_ids):
            candidate_token_ids[row_index, : len(window)] = torch.tensor(window, dtype=torch.int64, device=device)
        header = torch.tensor([self.batch_size, self.window_size], dtype=torch.int64, device=device)
        return header, request_slots, candidate_lengths, candidate_token_ids

    @classmethod
    def from_tensors(
        cls,
        header: torch.Tensor,
        request_slots: torch.Tensor,
        candidate_lengths: torch.Tensor,
        candidate_token_ids: torch.Tensor,
    ) -> PearlProposalBatch:
        """Decode tensors emitted by :meth:`to_tensors`."""
        batch_size, window_size = _read_header(header)
        _validate_tensor_shape(request_slots, (batch_size,), "request_slots")
        _validate_tensor_shape(candidate_lengths, (batch_size,), "candidate_lengths")
        _validate_tensor_shape(candidate_token_ids, (batch_size, window_size), "candidate_token_ids")

        request_slot_values = tuple(int(value) for value in request_slots.cpu().tolist())
        length_values = tuple(int(value) for value in candidate_lengths.cpu().tolist())
        token_rows = candidate_token_ids.cpu().tolist()
        if any(length <= 0 or length > window_size for length in length_values):
            raise ValueError("PEARL proposal candidate lengths must be in [1, window_size].")

        return cls(
            request_slots=request_slot_values,
            candidate_token_ids=tuple(
                tuple(int(token_id) for token_id in row[:length]) for row, length in zip(token_rows, length_values)
            ),
        )


@dataclass(frozen=True)
class PearlVerificationBatch:
    """Target verdicts for the candidate windows of a proposal batch.

    A partial acceptance carries one target-sampled correction token. A fully
    accepted window carries no correction token because the next PEARL round
    provides the next candidate window.
    """

    request_slots: tuple[int, ...]
    accepted_prefix_lengths: tuple[int, ...]
    correction_token_ids: tuple[int | None, ...]
    finished: tuple[bool, ...]

    def __post_init__(self) -> None:
        request_slots = tuple(self.request_slots)
        accepted_prefix_lengths = tuple(self.accepted_prefix_lengths)
        correction_token_ids = tuple(self.correction_token_ids)
        finished = tuple(self.finished)
        object.__setattr__(self, "request_slots", request_slots)
        object.__setattr__(self, "accepted_prefix_lengths", accepted_prefix_lengths)
        object.__setattr__(self, "correction_token_ids", correction_token_ids)
        object.__setattr__(self, "finished", finished)

        batch_size = len(request_slots)
        if not request_slots:
            raise ValueError("A PEARL verification batch must contain at least one request.")
        if len(set(request_slots)) != batch_size or any(slot < 0 for slot in request_slots):
            raise ValueError("Verification request slots must be unique non-negative integers.")
        if any(len(values) != batch_size for values in (accepted_prefix_lengths, correction_token_ids, finished)):
            raise ValueError("Every PEARL verification field must have the same batch size.")
        if any(length < 0 for length in accepted_prefix_lengths):
            raise ValueError("Accepted prefix lengths must be non-negative.")
        if any(token_id is not None and token_id < 0 for token_id in correction_token_ids):
            raise ValueError("Correction token IDs must be non-negative when present.")

    @property
    def batch_size(self) -> int:
        """Number of target verdicts in the batch."""
        return len(self.request_slots)

    def validate_against(self, proposals: PearlProposalBatch) -> None:
        """Validate request order and correction semantics against proposals."""
        if self.request_slots != proposals.request_slots:
            raise ValueError("PEARL verification request slots must match the proposal order.")
        for accepted, correction, window in zip(
            self.accepted_prefix_lengths,
            self.correction_token_ids,
            proposals.candidate_token_ids,
        ):
            if accepted > len(window):
                raise ValueError("Accepted prefix length cannot exceed the candidate-window length.")
            if accepted == len(window) and correction is not None:
                raise ValueError("A fully accepted PEARL window must not carry a correction token.")
            if accepted < len(window) and correction is None:
                raise ValueError("A partially accepted PEARL window requires a correction token.")

    def to_tensors(
        self, device: torch.device | str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode the batch into fixed-shape tensors suitable for HCCL broadcast."""
        header = torch.tensor([self.batch_size, 0], dtype=torch.int64, device=device)
        request_slots = torch.tensor(self.request_slots, dtype=torch.int64, device=device)
        accepted_prefix_lengths = torch.tensor(self.accepted_prefix_lengths, dtype=torch.int64, device=device)
        correction_token_ids = torch.tensor(
            [token_id if token_id is not None else _NO_CORRECTION_TOKEN_ID for token_id in self.correction_token_ids],
            dtype=torch.int64,
            device=device,
        )
        finished = torch.tensor(self.finished, dtype=torch.int64, device=device)
        return header, request_slots, accepted_prefix_lengths, correction_token_ids, finished

    @classmethod
    def from_tensors(
        cls,
        header: torch.Tensor,
        request_slots: torch.Tensor,
        accepted_prefix_lengths: torch.Tensor,
        correction_token_ids: torch.Tensor,
        finished: torch.Tensor,
    ) -> PearlVerificationBatch:
        """Decode tensors emitted by :meth:`to_tensors`."""
        batch_size, reserved = _read_header(header)
        if reserved != 0:
            raise ValueError("PEARL verification header contains an unsupported reserved value.")
        for tensor, name in (
            (request_slots, "request_slots"),
            (accepted_prefix_lengths, "accepted_prefix_lengths"),
            (correction_token_ids, "correction_token_ids"),
            (finished, "finished"),
        ):
            _validate_tensor_shape(tensor, (batch_size,), name)

        correction_values = tuple(int(value) for value in correction_token_ids.cpu().tolist())
        return cls(
            request_slots=tuple(int(value) for value in request_slots.cpu().tolist()),
            accepted_prefix_lengths=tuple(int(value) for value in accepted_prefix_lengths.cpu().tolist()),
            correction_token_ids=tuple(
                None if value == _NO_CORRECTION_TOKEN_ID else value for value in correction_values
            ),
            finished=tuple(bool(value) for value in finished.cpu().tolist()),
        )


def broadcast_proposals(
    proposals: PearlProposalBatch | None,
    *,
    source_rank: int,
    group: dist.ProcessGroup,
    device: torch.device | str,
) -> PearlProposalBatch:
    """Broadcast a draft proposal batch to the PEARL verification group.

    The source rank supplies ``proposals``; all other verification-group ranks
    pass ``None``. Reading the two-element header requires one device-to-host
    synchronization to allocate the variable-size batch buffers. It is outside
    the per-token model hot path and replaces nano-PEARL's Python object IPC.
    """
    is_source = dist.get_rank() == source_rank
    if is_source:
        if proposals is None:
            raise ValueError("The PEARL proposal source rank must provide a proposal batch.")
        header, request_slots, candidate_lengths, candidate_token_ids = proposals.to_tensors(device)
    else:
        if proposals is not None:
            raise ValueError("Only the PEARL proposal source rank may provide a proposal batch.")
        header = torch.empty(_HEADER_WIDTH, dtype=torch.int64, device=device)

    dist.broadcast(header, src=source_rank, group=group)
    batch_size, window_size = _read_header(header)
    if not is_source:
        request_slots = torch.empty(batch_size, dtype=torch.int64, device=device)
        candidate_lengths = torch.empty(batch_size, dtype=torch.int64, device=device)
        candidate_token_ids = torch.empty((batch_size, window_size), dtype=torch.int64, device=device)

    dist.broadcast(request_slots, src=source_rank, group=group)
    dist.broadcast(candidate_lengths, src=source_rank, group=group)
    dist.broadcast(candidate_token_ids, src=source_rank, group=group)
    return PearlProposalBatch.from_tensors(header, request_slots, candidate_lengths, candidate_token_ids)


def broadcast_verifications(
    verifications: PearlVerificationBatch | None,
    *,
    source_rank: int,
    device: torch.device | str,
) -> PearlVerificationBatch:
    """Broadcast target verification results to every PEARL worker.

    This uses the initialized default world group, matching the original
    PEARL correction path: all draft and target ranks need the same rollback
    decision before their next decode step.
    """
    is_source = dist.get_rank() == source_rank
    if is_source:
        if verifications is None:
            raise ValueError("The PEARL verification source rank must provide a verification batch.")
        header, request_slots, accepted_prefix_lengths, correction_token_ids, finished = verifications.to_tensors(
            device
        )
    else:
        if verifications is not None:
            raise ValueError("Only the PEARL verification source rank may provide a verification batch.")
        header = torch.empty(_HEADER_WIDTH, dtype=torch.int64, device=device)

    dist.broadcast(header, src=source_rank)
    batch_size, reserved = _read_header(header)
    if reserved != 0:
        raise ValueError("PEARL verification header contains an unsupported reserved value.")
    if not is_source:
        request_slots = torch.empty(batch_size, dtype=torch.int64, device=device)
        accepted_prefix_lengths = torch.empty(batch_size, dtype=torch.int64, device=device)
        correction_token_ids = torch.empty(batch_size, dtype=torch.int64, device=device)
        finished = torch.empty(batch_size, dtype=torch.int64, device=device)

    dist.broadcast(request_slots, src=source_rank)
    dist.broadcast(accepted_prefix_lengths, src=source_rank)
    dist.broadcast(correction_token_ids, src=source_rank)
    dist.broadcast(finished, src=source_rank)
    return PearlVerificationBatch.from_tensors(
        header,
        request_slots,
        accepted_prefix_lengths,
        correction_token_ids,
        finished,
    )


def _read_header(header: torch.Tensor) -> tuple[int, int]:
    _validate_tensor_shape(header, (_HEADER_WIDTH,), "header")
    batch_size, width = (int(value) for value in header.cpu().tolist())
    if batch_size <= 0 or width < 0:
        raise ValueError("PEARL protocol header contains an invalid batch size or width.")
    return batch_size, width


def _validate_tensor_shape(tensor: torch.Tensor, expected_shape: tuple[int, ...], name: str) -> None:
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"PEARL {name} has shape {tuple(tensor.shape)}, expected {expected_shape}.")
