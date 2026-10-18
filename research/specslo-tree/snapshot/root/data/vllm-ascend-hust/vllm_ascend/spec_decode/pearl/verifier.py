# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Target-side PEARL candidate verification."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import torch

from vllm_ascend.spec_decode.pearl.protocol import PearlProposalBatch, PearlVerificationBatch
from vllm_ascend.spec_decode.pearl.vocab import PearlVocabProjection


@dataclass(frozen=True)
class PearlTargetVerifier:
    """Convert target logits into PEARL acceptance and correction messages.

    ``target_logits`` contains one row for every candidate token, in proposal
    order. The target leader owns this operation and broadcasts the returned
    :class:`PearlVerificationBatch` to draft and target workers.
    """

    vocab_projection: PearlVocabProjection
    eos_token_ids: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        if any(token_id < 0 for token_id in self.eos_token_ids):
            raise ValueError("PEARL EOS token IDs must be non-negative.")

    @classmethod
    def with_eos_tokens(
        cls,
        vocab_projection: PearlVocabProjection,
        eos_token_ids: int | Collection[int] | None,
    ) -> PearlTargetVerifier:
        """Construct a verifier from one or more model EOS token IDs."""
        if eos_token_ids is None:
            normalized_eos: frozenset[int] = frozenset()
        elif isinstance(eos_token_ids, int):
            normalized_eos = frozenset((eos_token_ids,))
        else:
            normalized_eos = frozenset(eos_token_ids)
        return cls(vocab_projection=vocab_projection, eos_token_ids=normalized_eos)

    @torch.inference_mode()
    def verify(
        self,
        proposals: PearlProposalBatch,
        target_logits: torch.Tensor,
        *,
        temperature: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> PearlVerificationBatch:
        """Verify candidates using the original PEARL greedy/random rules.

        Greedy mode accepts consecutive candidates equal to target argmax.
        Positive-temperature mode follows nano-PEARL's rule: accept each
        candidate with its normalized target probability and sample a target
        correction after masking the rejected candidate.
        """
        if temperature < 0:
            raise ValueError("PEARL temperature must be non-negative.")
        candidate_rows = proposals.candidate_token_ids
        candidate_count = sum(len(row) for row in candidate_rows)
        if target_logits.ndim != 2 or target_logits.shape[0] != candidate_count:
            raise ValueError(
                "PEARL target logits must have shape "
                f"({candidate_count}, target_vocab_size); got {tuple(target_logits.shape)}."
            )

        candidate_ids = torch.tensor(
            [token_id for row in candidate_rows for token_id in row],
            device=target_logits.device,
            dtype=torch.long,
        )
        self.vocab_projection.validate_draft_token_ids(candidate_ids)
        projected_logits = self.vocab_projection.project_target_logits(target_logits)
        if temperature == 0:
            accepted_mask = projected_logits.argmax(dim=-1).eq(candidate_ids)
            correction_ids = projected_logits.argmax(dim=-1)
        else:
            probabilities = torch.softmax(projected_logits.float() / temperature, dim=-1)
            candidate_probabilities = probabilities.gather(1, candidate_ids.unsqueeze(1)).squeeze(1)
            accepted_mask = torch.rand(
                candidate_count,
                device=target_logits.device,
                generator=generator,
            ).le(candidate_probabilities)
            correction_logits = projected_logits.clone()
            correction_logits.scatter_(1, candidate_ids.unsqueeze(1), float("-inf"))
            correction_probabilities = torch.softmax(correction_logits.float() / temperature, dim=-1)
            correction_ids = torch.multinomial(correction_probabilities, 1, generator=generator).squeeze(1)

        accepted_prefix_lengths: list[int] = []
        correction_token_ids: list[int | None] = []
        finished: list[bool] = []
        cursor = 0
        for candidates in candidate_rows:
            candidate_length = len(candidates)
            accepted = accepted_mask[cursor : cursor + candidate_length]
            rejected_offsets = (~accepted).nonzero(as_tuple=False)
            accepted_length = candidate_length if rejected_offsets.numel() == 0 else int(rejected_offsets[0, 0])
            correction = None if accepted_length == candidate_length else int(correction_ids[cursor + accepted_length])
            committed_tokens = (
                candidates[:accepted_length] if correction is None else (*candidates[:accepted_length], correction)
            )
            accepted_prefix_lengths.append(accepted_length)
            correction_token_ids.append(correction)
            finished.append(any(token_id in self.eos_token_ids for token_id in committed_tokens))
            cursor += candidate_length

        return PearlVerificationBatch(
            request_slots=proposals.request_slots,
            accepted_prefix_lengths=tuple(accepted_prefix_lengths),
            correction_token_ids=tuple(correction_token_ids),
            finished=tuple(finished),
        )
