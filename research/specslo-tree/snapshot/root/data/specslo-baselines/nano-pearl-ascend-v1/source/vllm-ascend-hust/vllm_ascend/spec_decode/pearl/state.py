# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-local PEARL acceptance and correction state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from vllm_ascend.spec_decode.pearl.protocol import PearlProposalBatch, PearlVerificationBatch


class PearlPhase(str, Enum):
    """Synchronization state for a request in the PEARL decode loop."""

    PRIMING = "priming"
    PIPELINED = "pipelined"
    FINISHED = "finished"


@dataclass(frozen=True)
class PearlRequestState:
    """Target-validated token state for one request.

    The future worker bridge owns the target and draft KV-cache bookkeeping.
    This class owns only the shared logical result so both engines apply the
    same accepted prefix and target correction before the next PEARL round.
    """

    request_slot: int
    token_ids: tuple[int, ...]
    phase: PearlPhase = PearlPhase.PRIMING
    accepted_draft_tokens: int = 0

    def __post_init__(self) -> None:
        token_ids = tuple(self.token_ids)
        object.__setattr__(self, "token_ids", token_ids)
        if self.request_slot < 0:
            raise ValueError("PEARL request slots must be non-negative.")
        if any(token_id < 0 for token_id in token_ids):
            raise ValueError("PEARL request token IDs must be non-negative.")
        if self.accepted_draft_tokens < 0:
            raise ValueError("Accepted PEARL draft-token count must be non-negative.")

    def apply_verification(
        self,
        candidate_token_ids: tuple[int, ...],
        accepted_prefix_length: int,
        correction_token_id: int | None,
        finished: bool,
    ) -> PearlRequestState:
        """Apply one validated candidate window to this request state.

        Args:
            candidate_token_ids: Draft candidate window in target-tokenizer IDs.
            accepted_prefix_length: Number of leading candidates accepted by
                the target model.
            correction_token_id: Target-sampled token replacing the first
                rejected candidate, or ``None`` after full acceptance.
            finished: Whether the target model ended this request.

        Returns:
            A new state with only target-validated output tokens appended.
        """
        if self.phase is PearlPhase.FINISHED:
            raise ValueError("Cannot apply PEARL verification to a finished request.")
        if not candidate_token_ids:
            raise ValueError("PEARL candidate windows must not be empty.")
        if not 0 <= accepted_prefix_length <= len(candidate_token_ids):
            raise ValueError("Accepted PEARL prefix length is outside the candidate window.")

        full_acceptance = accepted_prefix_length == len(candidate_token_ids)
        if full_acceptance and correction_token_id is not None:
            raise ValueError("Fully accepted PEARL windows must not include a correction token.")
        if not full_acceptance and correction_token_id is None:
            raise ValueError("Rejected PEARL windows require a correction token.")

        appended_tokens = list(candidate_token_ids[:accepted_prefix_length])
        if correction_token_id is not None:
            appended_tokens.append(correction_token_id)

        return PearlRequestState(
            request_slot=self.request_slot,
            token_ids=(*self.token_ids, *appended_tokens),
            phase=PearlPhase.FINISHED if finished else PearlPhase.PIPELINED if full_acceptance else PearlPhase.PRIMING,
            accepted_draft_tokens=self.accepted_draft_tokens + accepted_prefix_length,
        )


def advance_request_states(
    request_states: Mapping[int, PearlRequestState],
    proposals: PearlProposalBatch,
    verifications: PearlVerificationBatch,
) -> dict[int, PearlRequestState]:
    """Apply a target verification batch to a mapping of PEARL request state."""
    verifications.validate_against(proposals)
    next_states = dict(request_states)
    for request_slot, candidates, accepted, correction, finished in zip(
        proposals.request_slots,
        proposals.candidate_token_ids,
        verifications.accepted_prefix_lengths,
        verifications.correction_token_ids,
        verifications.finished,
    ):
        try:
            request_state = request_states[request_slot]
        except KeyError as exc:
            raise KeyError(f"Missing PEARL request state for slot {request_slot}.") from exc
        next_states[request_slot] = request_state.apply_verification(candidates, accepted, correction, finished)
    return next_states
