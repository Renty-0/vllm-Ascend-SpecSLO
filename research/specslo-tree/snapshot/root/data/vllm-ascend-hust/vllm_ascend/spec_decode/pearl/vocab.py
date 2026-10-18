# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vocabulary compatibility checks for PEARL draft/target model pairs."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PearlVocabProjection:
    """Project target logits into an identical draft-token-ID prefix.

    PEARL passes draft token IDs directly to the target verifier and feeds a
    target correction back to the draft model. Therefore this optimization is
    only valid when the draft vocabulary is an exact token-ID prefix of the
    target vocabulary. It is deliberately stricter than vLLM's generic
    heterogeneous-vocabulary mapping: PEARL does not translate token IDs.
    """

    draft_vocab_size: int
    target_vocab_size: int

    def __post_init__(self) -> None:
        if self.draft_vocab_size <= 0:
            raise ValueError("The PEARL draft vocabulary size must be positive.")
        if self.target_vocab_size < self.draft_vocab_size:
            raise ValueError("PEARL prefix projection requires target_vocab_size to be at least draft_vocab_size.")

    @property
    def requires_projection(self) -> bool:
        """Whether target logits have an unsupported suffix to remove."""
        return self.target_vocab_size != self.draft_vocab_size

    @classmethod
    def from_tokenizers(
        cls,
        *,
        draft_tokenizer,
        target_tokenizer,
        draft_vocab_size: int,
        target_vocab_size: int,
    ) -> PearlVocabProjection:
        """Build a projection after proving token IDs share a common prefix."""
        projection = cls(
            draft_vocab_size=draft_vocab_size,
            target_vocab_size=target_vocab_size,
        )
        mismatches: list[tuple[int, str | None, str | None]] = []
        for token_id in range(draft_vocab_size):
            draft_token = draft_tokenizer.convert_ids_to_tokens(token_id)
            target_token = target_tokenizer.convert_ids_to_tokens(token_id)
            if draft_token != target_token:
                mismatches.append((token_id, draft_token, target_token))
                if len(mismatches) == 3:
                    break
        if mismatches:
            details = ", ".join(
                f"id={token_id}: draft={draft_token!r}, target={target_token!r}"
                for token_id, draft_token, target_token in mismatches
            )
            raise ValueError(
                "PEARL prefix vocabulary projection requires identical token IDs "
                f"for [0, {draft_vocab_size}); found mismatches: {details}."
            )
        return projection

    def project_target_logits(self, target_logits: torch.Tensor) -> torch.Tensor:
        """Crop target logits before PEARL sampling or acceptance comparison."""
        if target_logits.ndim < 1:
            raise ValueError("PEARL target logits must have a vocabulary dimension.")
        if target_logits.shape[-1] != self.target_vocab_size:
            raise ValueError(
                "PEARL target logits have an unexpected vocabulary size: "
                f"got {target_logits.shape[-1]}, expected {self.target_vocab_size}."
            )
        if not self.requires_projection:
            return target_logits
        # The Ascend rejection sampler requires a contiguous vocabulary axis.
        return target_logits[..., : self.draft_vocab_size].contiguous()

    def validate_draft_token_ids(self, token_ids: torch.Tensor) -> None:
        """Reject candidates that cannot be embedded by the draft model."""
        if token_ids.numel() == 0:
            return
        if token_ids.dtype.is_floating_point:
            raise ValueError("PEARL draft token IDs must use an integer dtype.")
        minimum = int(token_ids.min().item())
        maximum = int(token_ids.max().item())
        if minimum < 0 or maximum >= self.draft_vocab_size:
            raise ValueError(
                "PEARL draft token IDs must be in the projected vocabulary range "
                f"[0, {self.draft_vocab_size}); got [{minimum}, {maximum}]."
            )
