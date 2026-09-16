# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contracts and qualification gates for native PARD parallel drafting.

The device implementation lives in :mod:`native_engine`; this module keeps the
packed-layout, checkpoint and public experimental-envelope checks independently
CPU-testable.  PARD draft execution is currently eager-only, while the target
verification worker may use its separately qualified ACLGraphs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

SERIAL_LINEAR_DRAFT_MODE = "serial_linear"
PARD_PARALLEL_DRAFT_MODE = "pard_parallel"
NATIVE_DRAFT_MODES = frozenset(
    {
        SERIAL_LINEAR_DRAFT_MODE,
        PARD_PARALLEL_DRAFT_MODE,
    }
)
PARD_PARALLEL_GAMMA = 4
PARD_ARCHITECTURE = "Qwen3ForCausalLM"
PARD_MODEL_TYPE = "qwen3"


@dataclass(frozen=True)
class PardParallelDraftLayout:
    """Packed PARD inputs and the hidden-state rows consumed by the LM head.

    Each request contributes a non-empty suffix of real committed tokens which
    ends at its current root, followed by ``gamma - 1`` PARD mask tokens.  The
    root row and the mask rows produce the four parallel proposals.  Earlier
    suffix rows exist only to repair canonical draft KV state.
    """

    input_token_ids: tuple[int, ...]
    sequence_ids: tuple[int, ...]
    positions: tuple[int, ...]
    sample_row_indices: tuple[int, ...]
    query_lengths: tuple[int, ...]
    gamma: int = PARD_PARALLEL_GAMMA

    @property
    def batch_size(self) -> int:
        return len(self.query_lengths)

    @property
    def proposal_shape(self) -> tuple[int, int]:
        return (self.batch_size, self.gamma)

    def attention_allowed(self, query_row: int, key_row: int) -> bool:
        """Return the packed causal-mask value without materializing ``T x T``.

        PARD uses ordinary causal attention inside each request segment.  A row
        must never see another request, and it may only see earlier (or its own)
        rows in its segment.  Keeping this predicate explicit makes the native
        metadata contract CPU-testable while avoiding a quadratic host mask.
        """

        num_rows = len(self.input_token_ids)
        if not 0 <= query_row < num_rows or not 0 <= key_row < num_rows:
            raise IndexError("PARD attention row is outside the packed input.")
        return self.sequence_ids[query_row] == self.sequence_ids[key_row] and key_row <= query_row


def validate_native_draft_mode(draft_mode: str, gamma: int) -> None:
    """Validate the public native draft-mode envelope without loading models."""

    if draft_mode not in NATIVE_DRAFT_MODES:
        choices = ", ".join(sorted(NATIVE_DRAFT_MODES))
        raise ValueError(f"Unknown native draft mode {draft_mode!r}; expected one of {choices}.")
    if draft_mode == PARD_PARALLEL_DRAFT_MODE and gamma != PARD_PARALLEL_GAMMA:
        raise ValueError(
            f"Native PARD parallel drafting currently has a fixed gamma=4 contract; received gamma={gamma}."
        )


def validate_pard_parallel_model_pair(
    draft_config: Any,
    target_config: Any,
    *,
    gamma: int,
) -> int:
    """Validate the strict checkpoint contract and return the PARD mask ID."""

    validate_native_draft_mode(PARD_PARALLEL_DRAFT_MODE, gamma)
    if str(getattr(draft_config, "spd_type", "")).lower() != "pard":
        raise ValueError('pard_parallel requires draft config `spd_type="pard"`.')

    draft_architectures = tuple(getattr(draft_config, "architectures", ()) or ())
    target_architectures = tuple(getattr(target_config, "architectures", ()) or ())
    if PARD_ARCHITECTURE not in draft_architectures or getattr(draft_config, "model_type", None) != PARD_MODEL_TYPE:
        raise ValueError(
            "pard_parallel requires a Qwen3 draft checkpoint with architecture "
            f"{PARD_ARCHITECTURE!r} and model_type={PARD_MODEL_TYPE!r}."
        )
    if PARD_ARCHITECTURE not in target_architectures or getattr(target_config, "model_type", None) != PARD_MODEL_TYPE:
        raise ValueError(
            "pard_parallel currently requires a Qwen3 target with architecture "
            f"{PARD_ARCHITECTURE!r} and model_type={PARD_MODEL_TYPE!r}."
        )

    draft_vocab_size = getattr(draft_config, "vocab_size", None)
    target_vocab_size = getattr(target_config, "vocab_size", None)
    if (
        isinstance(draft_vocab_size, bool)
        or not isinstance(draft_vocab_size, int)
        or draft_vocab_size <= 0
        or isinstance(target_vocab_size, bool)
        or not isinstance(target_vocab_size, int)
        or target_vocab_size <= 0
    ):
        raise ValueError("pard_parallel requires positive integer draft and target vocab sizes.")
    if draft_vocab_size != target_vocab_size:
        raise ValueError(
            "pard_parallel requires identical draft and target vocab sizes; "
            f"received {draft_vocab_size} and {target_vocab_size}."
        )

    pard_token = getattr(draft_config, "pard_token", None)
    if isinstance(pard_token, bool) or not isinstance(pard_token, int):
        raise ValueError("pard_parallel requires an integer `pard_token` in the draft config.")
    if not 0 <= pard_token < draft_vocab_size:
        raise ValueError(
            "pard_parallel draft `pard_token` must lie inside the shared vocabulary; "
            f"received pard_token={pard_token}, vocab_size={draft_vocab_size}."
        )
    return pard_token


def require_experimental_pard_eager(config: Any, *, enabled: bool) -> None:
    """Open the public PARD gate only for the qualified eager envelope."""

    if not enabled:
        raise pard_parallel_execution_unavailable()
    requirements = {
        "draft_mode=pard_parallel": getattr(config, "draft_mode", None) == PARD_PARALLEL_DRAFT_MODE,
        "gamma=4": getattr(config, "gamma", None) == PARD_PARALLEL_GAMMA,
        "enable_spec_rhythm=True": bool(getattr(config, "enable_spec_rhythm", False)),
        "linear_full_window=True": bool(getattr(config, "spec_rhythm_linear_full_window", False)),
        "fixed min_gamma=4": getattr(config, "spec_rhythm_min_gamma", None) == PARD_PARALLEL_GAMMA,
        "linear tree shape 1x1": (
            getattr(config, "spec_rhythm_tree_width", None) == 1
            and getattr(config, "spec_rhythm_tree_depth", None) == 1
        ),
        "rolling eager disabled": (
            getattr(config, "spec_rhythm_max_eager_tokens", None) == 0
            and getattr(config, "spec_rhythm_eager_reserve_tokens", None) == 0
            and not bool(getattr(config, "spec_rhythm_linear_eager_cross_graph_bucket", False))
            and not bool(getattr(config, "spec_rhythm_linear_idle_residual_eager", False))
        ),
        "bonus token disabled": not bool(getattr(config, "spec_rhythm_linear_bonus_token", False)),
        # ``precompile_decode_graphs`` is target-only unless the independent
        # serial-draft switch below is enabled.  Allowing it is essential for
        # the qualified hybrid path: one eager PARD forward on rank 0 and
        # ACLGraph target verification on the TP ranks.  PARD must still never
        # enter the serial-draft graph family silently.
        "serial draft graph precompilation disabled": not bool(
            getattr(config, "precompile_serial_draft_graphs", False)
        ),
    }
    missing = [label for label, satisfied in requirements.items() if not satisfied]
    if missing:
        raise ValueError("Experimental native PARD eager qualification requires " + ", ".join(missing) + ".")


def build_pard_parallel_draft_layout(
    committed_suffix_token_ids: Sequence[Sequence[int]],
    sequence_ids: Sequence[int],
    first_positions: Sequence[int],
    *,
    pard_token: int,
    vocab_size: int,
    gamma: int = PARD_PARALLEL_GAMMA,
) -> PardParallelDraftLayout:
    """Build a request-isolated packed PARD layout without touching a device.

    ``committed_suffix_token_ids[row]`` must end with the current real root
    token.  Prefix entries repair draft KV state; the root plus ``gamma - 1``
    internal PARD mask tokens are the LM-head sample rows.
    """

    validate_native_draft_mode(PARD_PARALLEL_DRAFT_MODE, gamma)
    batch_size = len(committed_suffix_token_ids)
    if batch_size == 0:
        raise ValueError("A PARD parallel draft batch must contain at least one request.")
    if len(sequence_ids) != batch_size or len(first_positions) != batch_size:
        raise ValueError("Every PARD request needs one sequence ID and first position.")
    if len(set(sequence_ids)) != batch_size:
        raise ValueError("PARD packed sequence IDs must be unique per request.")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("PARD vocab_size must be a positive integer.")
    if isinstance(pard_token, bool) or not isinstance(pard_token, int) or not 0 <= pard_token < vocab_size:
        raise ValueError("PARD mask token must be an integer inside the draft vocabulary.")

    input_token_ids: list[int] = []
    packed_sequence_ids: list[int] = []
    positions: list[int] = []
    sample_row_indices: list[int] = []
    query_lengths: list[int] = []
    mask_count = gamma - 1

    for row, (suffix, sequence_id, first_position) in enumerate(
        zip(committed_suffix_token_ids, sequence_ids, first_positions)
    ):
        suffix_values = tuple(suffix)
        if not suffix_values:
            raise ValueError(f"PARD request {row} must include its current committed root token.")
        if isinstance(sequence_id, bool) or not isinstance(sequence_id, int) or sequence_id < 0:
            raise ValueError(f"PARD request {row} has an invalid sequence ID.")
        if isinstance(first_position, bool) or not isinstance(first_position, int) or first_position < 0:
            raise ValueError(f"PARD request {row} has an invalid first position.")
        if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in suffix_values):
            raise ValueError(f"PARD request {row} contains a non-integer token ID.")
        if any(token_id < 0 or token_id >= vocab_size for token_id in suffix_values):
            raise ValueError(f"PARD request {row} contains a token outside the draft vocabulary.")

        request_start = len(input_token_ids)
        request_tokens = (*suffix_values, *((pard_token,) * mask_count))
        query_length = len(request_tokens)
        root_row = request_start + len(suffix_values) - 1
        input_token_ids.extend(request_tokens)
        packed_sequence_ids.extend([sequence_id] * query_length)
        positions.extend(range(first_position, first_position + query_length))
        sample_row_indices.extend(range(root_row, root_row + gamma))
        query_lengths.append(query_length)

    return PardParallelDraftLayout(
        input_token_ids=tuple(input_token_ids),
        sequence_ids=tuple(packed_sequence_ids),
        positions=tuple(positions),
        sample_row_indices=tuple(sample_row_indices),
        query_lengths=tuple(query_lengths),
    )


def gather_pard_parallel_proposals(
    packed_token_predictions: Sequence[int],
    layout: PardParallelDraftLayout,
) -> tuple[tuple[int, ...], ...]:
    """Gather packed per-row predictions into the fixed ``[B, 4]`` contract."""

    if len(packed_token_predictions) != len(layout.input_token_ids):
        raise ValueError("Packed PARD predictions must cover every input row.")
    gathered = tuple(packed_token_predictions[index] for index in layout.sample_row_indices)
    expected = layout.batch_size * layout.gamma
    if len(gathered) != expected:
        raise RuntimeError("PARD sample-row layout does not match its fixed proposal shape.")
    return tuple(tuple(gathered[offset : offset + layout.gamma]) for offset in range(0, expected, layout.gamma))


def pard_parallel_execution_unavailable() -> RuntimeError:
    """Return the fail-closed error used by public native execution entrypoints."""

    return RuntimeError(
        "Native pard_parallel execution is experimental and disabled by default. "
        "Set VLLM_ASCEND_SPECSLO_ENABLE_EXPERIMENTAL_PARD_EAGER=1 only for the "
        "strict qualified envelope; PARD draft remains eager-only and never falls "
        "back to serial drafting. Use draft_mode='serial_linear' otherwise."
    )
