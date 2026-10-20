# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM ``custom_class`` proposer backed by a separate PEARL draft engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.distributed.parallel_state import get_tp_group

from vllm_ascend.spec_decode.pearl.transport import PearlTransportError, exchange_unix_message
from vllm_ascend.spec_decode.pearl.vocab import PearlVocabProjection

if TYPE_CHECKING:
    import numpy as np
    import torch
    from vllm.config import VllmConfig

PEARL_CONFIG_KEY = "pearl"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0


class PearlRemoteProposer:
    """Request greedy candidates from a separate vLLM draft-model process.

    The target tensor-parallel leader owns the Unix-socket request; candidate
    rows are then broadcast through vLLM's existing CPU tensor-parallel group.
    The target's regular speculative scheduler verifies the returned rows and
    performs paged-KV rollback, so this class deliberately owns no cache state.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        pearl_config = vllm_config.additional_config.get(PEARL_CONFIG_KEY)
        if not isinstance(pearl_config, dict):
            raise ValueError("PEARL requires VllmConfig.additional_config['pearl'] to be a JSON object.")
        socket_path = pearl_config.get("draft_socket_path")
        if not isinstance(socket_path, str) or not socket_path:
            raise ValueError("PEARL requires a non-empty 'draft_socket_path'.")
        timeout_seconds = pearl_config.get("request_timeout_seconds", DEFAULT_REQUEST_TIMEOUT_SECONDS)
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            raise ValueError("PEARL request_timeout_seconds must be a positive number.")

        self.socket_path = socket_path
        self.timeout_seconds = float(timeout_seconds)
        self.num_speculative_tokens = speculative_config.num_speculative_tokens
        draft_vocab_size = pearl_config.get("draft_vocab_size")
        target_vocab_size = pearl_config.get("target_vocab_size")
        if draft_vocab_size is None and target_vocab_size is None:
            self.pearl_vocab_projection = None
        elif (
            isinstance(draft_vocab_size, int)
            and isinstance(target_vocab_size, int)
            and draft_vocab_size > 0
            and target_vocab_size >= draft_vocab_size
        ):
            self.pearl_vocab_projection = PearlVocabProjection(
                draft_vocab_size=draft_vocab_size,
                target_vocab_size=target_vocab_size,
            )
        else:
            raise ValueError("PEARL vocab sizes must be positive integers with target >= draft.")

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    ) -> list[list[int]]:
        """Return one candidate row for each active target request.

        ``sampled_token_ids`` is used only as vLLM's active-request marker.
        ``num_tokens_no_spec`` already includes the target's most recent sampled
        token, therefore the prompt is copied directly from ``token_ids_cpu``.
        """
        del slot_mappings
        if len(sampled_token_ids) > len(num_tokens_no_spec):
            raise ValueError("PEARL received more sampled-token rows than target requests.")

        prompt_token_ids: list[list[int]] = []
        active_request_indices: list[int] = []
        candidate_rows: list[list[int]] = [[] for _ in sampled_token_ids]
        for request_index, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                continue
            num_tokens = int(num_tokens_no_spec[request_index])
            if num_tokens <= 0:
                raise ValueError("PEARL cannot draft from an empty target context.")
            prompt = [int(token_id) for token_id in token_ids_cpu[request_index, :num_tokens]]
            if any(token_id < 0 for token_id in prompt):
                raise ValueError("PEARL target context contains an invalid token ID.")
            prompt_token_ids.append(prompt)
            active_request_indices.append(request_index)

        if not active_request_indices:
            return candidate_rows

        tp_group = get_tp_group()
        if tp_group.rank_in_group == 0:
            proposals = self._request_candidates(prompt_token_ids)
        else:
            proposals = None
        proposals = tp_group.broadcast_object(proposals, src=0)
        if not isinstance(proposals, list) or len(proposals) != len(active_request_indices):
            raise PearlTransportError("PEARL draft leader broadcast malformed candidate rows.")
        for request_index, candidate_row in zip(active_request_indices, proposals):
            candidate_rows[request_index] = self._validate_candidate_row(candidate_row)
        return candidate_rows

    def _request_candidates(self, prompt_token_ids: list[list[int]]) -> list[list[int]]:
        response = exchange_unix_message(
            self.socket_path,
            {
                "op": "propose",
                "prompt_token_ids": prompt_token_ids,
                "num_speculative_tokens": self.num_speculative_tokens,
            },
            self.timeout_seconds,
        )
        if response.get("ok") is not True:
            raise PearlTransportError(
                f"PEARL draft service rejected proposal request: {response.get('error', 'unknown error')}"
            )
        candidates = response.get("candidate_token_ids")
        if not isinstance(candidates, list):
            raise PearlTransportError("PEARL draft service omitted candidate_token_ids.")
        return candidates

    def _validate_candidate_row(self, candidate_row: Any) -> list[int]:
        if not isinstance(candidate_row, list):
            raise PearlTransportError("PEARL draft service returned a non-list candidate row.")
        if len(candidate_row) > self.num_speculative_tokens:
            raise PearlTransportError("PEARL draft service returned an invalid candidate-window length.")
        if any(not isinstance(token_id, int) or token_id < 0 for token_id in candidate_row):
            raise PearlTransportError("PEARL draft service returned invalid token IDs.")
        return candidate_row
