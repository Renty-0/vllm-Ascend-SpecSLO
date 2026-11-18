"""Host-only shape and KV routing for selective linear target verification.

The existing fixed-width service uses one query per proposed token.  A
selective service may verify fewer tokens for a relaxed request, but every
real query must still have a unique causal KV position.  This module plans a
fixed graph envelope without importing torch or mutating a model cache.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SelectiveVerifyLayout:
    widths: tuple[int, ...]
    capacity: int
    token_bucket: int
    query_lengths: tuple[int, ...]

    @property
    def real_query_tokens(self) -> int:
        return sum(self.widths)

    @property
    def verification_rows(self) -> int:
        return len(self.widths)

    @property
    def verification_capacity(self) -> int:
        return self.capacity

    @property
    def verification_output_count(self) -> int:
        return self.real_query_tokens

    @property
    def request_segment_count(self) -> int:
        return len(self.query_lengths)

    @property
    def total_query_tokens(self) -> int:
        return self.token_bucket

    @property
    def graph_key(self) -> str:
        return f"stable-target-selective|capacity:{self.capacity}|query:{self.token_bucket}"


@dataclass(frozen=True)
class SelectiveVerifyEnvelope:
    layout: SelectiveVerifyLayout
    token_sequence_ids: tuple[int, ...]
    positions: tuple[int, ...]
    segment_sequence_ids: tuple[int, ...]
    sequence_lens: tuple[int, ...]


def plan_selective_verify_layout(
    widths: Sequence[int],
    *,
    capacities: Sequence[int] = (8, 16, 24, 32),
    token_buckets: Sequence[int] = tuple(range(8, 145, 8)),
) -> SelectiveVerifyLayout:
    """Choose a bounded graph shape while preserving all real query lengths.

    Empty dummy segments are forbidden: each unused request slot consumes one
    scratch token, and one final scratch segment fills the selected Q bucket.
    Keeping this residual segment makes both Q and segment count invariant for
    every replay of a graph key.
    """

    real_widths = tuple(int(width) for width in widths)
    if not real_widths or any(width not in (2, 4) for width in real_widths):
        raise ValueError("Selective verification requires non-empty 2/4-token widths.")
    valid_capacities = sorted(set(int(value) for value in capacities))
    capacity = next((value for value in valid_capacities if value >= len(real_widths)), None)
    if capacity is None or capacity <= 0:
        raise ValueError("Selective verification exceeds the provisioned request capacity.")
    dummy_count = capacity - len(real_widths)
    minimum_tokens = sum(real_widths) + dummy_count + 1
    valid_buckets = sorted(set(int(value) for value in token_buckets))
    bucket = next((value for value in valid_buckets if value >= minimum_tokens), None)
    if bucket is None or bucket <= 0:
        raise ValueError("Selective verification exceeds the provisioned token buckets.")
    residual = bucket - sum(real_widths) - dummy_count
    query_lengths = (*real_widths, *((1,) * dummy_count), residual)
    assert len(query_lengths) == capacity + 1 and sum(query_lengths) == bucket
    return SelectiveVerifyLayout(real_widths, capacity, bucket, query_lengths)


def route_selective_verify_envelope(
    layout: SelectiveVerifyLayout,
    sequence_ids: Sequence[int],
    first_positions: Sequence[int],
    *,
    scratch_sequence_id: int,
) -> SelectiveVerifyEnvelope:
    """Assign real and dummy query tokens to disjoint causal KV slots."""

    real_ids = tuple(int(value) for value in sequence_ids)
    real_starts = tuple(int(value) for value in first_positions)
    scratch_id = int(scratch_sequence_id)
    if len(real_ids) != len(layout.widths) or len(real_starts) != len(layout.widths):
        raise ValueError("Selective verification real rows and starts must match widths.")
    if len(set(real_ids)) != len(real_ids) or scratch_id in real_ids:
        raise ValueError("Selective verification sequence IDs must be disjoint and unique.")
    if scratch_id < 0 or any(value < 0 for value in (*real_ids, *real_starts)):
        raise ValueError("Selective verification sequence IDs and positions must be non-negative.")

    token_ids: list[int] = []
    positions: list[int] = []
    segment_ids: list[int] = []
    sequence_lens: list[int] = []
    scratch_position = 0
    for row, width in enumerate(layout.query_lengths):
        real = row < len(real_ids)
        sequence_id = real_ids[row] if real else scratch_id
        start = real_starts[row] if real else scratch_position
        token_ids.extend((sequence_id,) * width)
        positions.extend(range(start, start + width))
        segment_ids.append(sequence_id)
        sequence_lens.append(start + width)
        if not real:
            scratch_position += width
    if len(token_ids) != layout.token_bucket or len(segment_ids) != layout.capacity + 1:
        raise RuntimeError("Selective verification envelope does not match its graph layout.")
    return SelectiveVerifyEnvelope(
        layout,
        tuple(token_ids),
        tuple(positions),
        tuple(segment_ids),
        tuple(sequence_lens),
    )
