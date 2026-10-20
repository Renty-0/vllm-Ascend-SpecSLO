# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reusable tensor builders for fixed-gamma linear draft FIA."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def validate_linear_draft_fia_final_pages(
    exact_first_positions: Sequence[int],
    host_request_block_tables: Sequence[Sequence[int]],
    *,
    step_count: int,
    block_size: int,
    capacity: int,
) -> None:
    """Validate that a shared table snapshot covers the final serial step."""

    positions = tuple(exact_first_positions)
    tables = tuple(tuple(int(page) for page in table) for table in host_request_block_tables)
    if not positions or len(positions) != len(tables) or step_count <= 0 or block_size <= 0 or capacity <= 0:
        raise ValueError("Linear draft FIA final-page validation requires aligned positive inputs.")
    if any(
        not isinstance(position, int) or isinstance(position, bool) or position < 0 or position + step_count > capacity
        for position in positions
    ):
        raise ValueError("Linear draft FIA final proposal positions must fit the cache capacity.")
    if any(len(table) * block_size != capacity for table in tables):
        raise ValueError("Linear draft FIA request tables must span the FULL-mask capacity.")

    for position, table in zip(positions, tables):
        final_visible_length = position + step_count
        final_visible_pages = (final_visible_length + block_size - 1) // block_size
        if any(page < 0 for page in table[:final_visible_pages]):
            raise RuntimeError("Linear draft FIA shared table is missing a final proposal page.")


class LinearDraftFIAFullMaskBuilder:
    """Build every serial-draft FULL mask with one broadcast comparison.

    Only the immutable key-position vector is cached.  Mask values are rebuilt
    for every call because both the exact request positions and the active row
    order may change between graph replays.
    """

    def __init__(self) -> None:
        self._key_positions: dict[tuple[torch.device, int], torch.Tensor] = {}

    def _get_key_positions(
        self,
        *,
        device: torch.device | str,
        capacity: int,
    ) -> torch.Tensor:
        normalized_device = torch.device(device)
        key = (normalized_device, int(capacity))
        positions = self._key_positions.get(key)
        if positions is None:
            positions = torch.arange(
                capacity,
                dtype=torch.long,
                device=normalized_device,
            )
            self._key_positions[key] = positions
        return positions

    def build(
        self,
        exact_first_positions: Sequence[int],
        *,
        step_count: int,
        capture_size: int,
        capacity: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, ...]:
        """Return one exact causal FULL mask view for each serial step.

        Real row ``r`` at step ``s`` can attend exactly through
        ``exact_first_positions[r] + s``.  Padding rows always expose key zero
        only, independent of the step, so no stale KV value can become visible.
        """

        positions = tuple(exact_first_positions)
        real_rows = len(positions)
        if step_count <= 0 or capture_size <= 0 or capacity <= 0:
            raise ValueError("Linear draft FIA masks require positive steps, rows and capacity.")
        if real_rows <= 0 or real_rows > capture_size:
            raise ValueError("Linear draft FIA masks require a non-empty real batch no larger than the capture size.")
        if any(not isinstance(position, int) or isinstance(position, bool) or position < 0 for position in positions):
            raise ValueError("Linear draft FIA exact positions must be non-negative integers.")
        if any(position + step_count > capacity for position in positions):
            raise ValueError("Linear draft FIA proposal positions exceed the FULL-mask capacity.")

        # Construct all visible lengths with one host-to-device transfer.  A
        # single device comparison then materializes the complete gamma-step
        # mask family.  The returned unbound tensors are views that retain the
        # common backing storage until every graph-input copy has been queued.
        visible_lengths = torch.tensor(
            [
                [
                    *(position + step + 1 for position in positions),
                    *((1,) * (capture_size - real_rows)),
                ]
                for step in range(step_count)
            ],
            dtype=torch.long,
            device=torch.device(device),
        )
        key_positions = self._get_key_positions(
            device=device,
            capacity=capacity,
        )
        masks = key_positions.view(1, 1, capacity).ge(visible_lengths.view(step_count, capture_size, 1))
        masks = masks.view(step_count, capture_size, 1, 1, capacity)
        return tuple(masks.unbind(0))


class LinearDraftFIAPersistentStaging:
    """Reusable fixed-shape inputs for one serial-draft FIA graph entry.

    The request identities, exact positions and page-table contents remain
    dynamic.  Only their fixed graph envelope is persistent.  ``refresh``
    rewrites every mutable value before the returned views are consumed, so a
    caller must not retain snapshots across refreshes.

    CPU staging tensors deliberately remain pageable and are copied with the
    default blocking semantics.  This keeps their reuse safe without adding a
    second host buffer or an NPU completion event.  Device writes are expected
    to run on the same current stream as the subsequent ACLGraph input copies.
    """

    def __init__(
        self,
        *,
        step_count: int,
        capture_size: int,
        capacity: int,
        table_width: int,
        device: torch.device | str,
        input_dtype: torch.dtype = torch.long,
        position_dtype: torch.dtype = torch.long,
        slot_dtype: torch.dtype = torch.int32,
        table_dtype: torch.dtype = torch.int32,
    ) -> None:
        if step_count <= 0 or capture_size <= 0 or capacity <= 0 or table_width <= 0:
            raise ValueError(
                "Persistent linear draft FIA staging requires positive steps, rows, capacity and table width."
            )
        self.step_count = int(step_count)
        self.capture_size = int(capture_size)
        self.capacity = int(capacity)
        self.table_width = int(table_width)
        self.device = torch.device(device)

        self.input_ids = torch.zeros(
            self.capture_size,
            dtype=input_dtype,
            device=self.device,
        )
        self.positions = torch.zeros(
            (self.step_count, self.capture_size),
            dtype=position_dtype,
            device=self.device,
        )
        self.slot_mappings = torch.full(
            (self.step_count, self.capture_size),
            -1,
            dtype=slot_dtype,
            device=self.device,
        )
        self.visible_lengths = torch.ones(
            (self.step_count, self.capture_size),
            dtype=position_dtype,
            device=self.device,
        )
        self.full_masks = torch.empty(
            (
                self.step_count,
                self.capture_size,
                1,
                1,
                self.capacity,
            ),
            dtype=torch.bool,
            device=self.device,
        )
        self.request_block_tables = torch.empty(
            (self.capture_size, self.table_width),
            dtype=table_dtype,
            device=self.device,
        )
        self._request_table_valid = torch.empty(
            (self.capture_size, self.table_width),
            dtype=torch.bool,
            device=self.device,
        )
        self._key_positions = torch.arange(
            self.capacity,
            dtype=position_dtype,
            device=self.device,
        )

        # Populate the two small CPU sources in-place.  Keeping their NumPy
        # views avoids constructing four temporary device tensors per cycle.
        self._host_positions = torch.zeros(
            (self.step_count, self.capture_size),
            dtype=position_dtype,
        )
        self._host_slot_mappings = torch.full(
            (self.step_count, self.capture_size),
            -1,
            dtype=slot_dtype,
        )
        self._host_positions_array = self._host_positions.numpy()
        self._host_slot_mappings_array = self._host_slot_mappings.numpy()

        self.position_views = tuple(self.positions.unbind(0))
        self.slot_mapping_views = tuple(self.slot_mappings.unbind(0))
        self.full_mask_views = tuple(self.full_masks.unbind(0))

    def refresh(
        self,
        exact_first_positions: Sequence[int],
        step_slot_mappings: Sequence[Sequence[int]],
        request_block_tables: torch.Tensor,
        input_root_ids: torch.Tensor,
    ) -> None:
        """Rewrite all dynamic rows while preserving every tensor identity."""

        positions = tuple(exact_first_positions)
        real_rows = len(positions)
        slot_rows = tuple(tuple(int(slot) for slot in row) for row in step_slot_mappings)
        if real_rows <= 0 or real_rows > self.capture_size:
            raise ValueError(
                "Persistent linear draft FIA staging needs a non-empty real batch no larger than its capture size."
            )
        if len(slot_rows) != self.step_count or any(len(row) != real_rows for row in slot_rows):
            raise ValueError("Persistent linear draft FIA staging needs one aligned slot mapping row per serial step.")
        if any(
            not isinstance(position, int)
            or isinstance(position, bool)
            or position < 0
            or position + self.step_count > self.capacity
            for position in positions
        ):
            raise ValueError("Persistent linear draft FIA positions must fit the complete serial window.")
        if (
            request_block_tables.ndim != 2
            or tuple(request_block_tables.shape) != (real_rows, self.table_width)
            or request_block_tables.dtype != self.request_block_tables.dtype
            or request_block_tables.device != self.request_block_tables.device
        ):
            raise ValueError(
                "Persistent linear draft FIA request tables must preserve their real-row shape, dtype and device."
            )
        if (
            input_root_ids.ndim != 1
            or input_root_ids.numel() != real_rows
            or input_root_ids.dtype != self.input_ids.dtype
            or input_root_ids.device != self.input_ids.device
        ):
            raise ValueError(
                "Persistent linear draft FIA root IDs must preserve their real-row shape, dtype and device."
            )

        self._host_positions_array.fill(0)
        self._host_slot_mappings_array.fill(-1)
        for step in range(self.step_count):
            self._host_positions_array[step, :real_rows] = [position + step for position in positions]
            self._host_slot_mappings_array[step, :real_rows] = slot_rows[step]
        self.positions.copy_(self._host_positions)
        self.slot_mappings.copy_(self._host_slot_mappings)

        self.input_ids.zero_()
        self.input_ids[:real_rows].copy_(input_root_ids)

        valid_rows = self._request_table_valid[:real_rows]
        output_rows = self.request_block_tables[:real_rows]
        torch.ge(request_block_tables, 0, out=valid_rows)
        torch.where(
            valid_rows,
            request_block_tables,
            request_block_tables[:, :1],
            out=output_rows,
        )
        if real_rows < self.capture_size:
            self.request_block_tables[real_rows:].copy_(output_rows[:1].expand(self.capture_size - real_rows, -1))

        # Padding positions are zero, hence their visible length is one and
        # their FULL-mask row can read only key zero.
        torch.add(self.positions, 1, out=self.visible_lengths)
        torch.ge(
            self._key_positions.view(1, 1, self.capacity),
            self.visible_lengths.view(
                self.step_count,
                self.capture_size,
                1,
            ),
            out=self.full_masks.view(
                self.step_count,
                self.capture_size,
                self.capacity,
            ),
        )


def sanitize_and_pad_linear_draft_fia_request_tables(
    request_block_tables: torch.Tensor,
    *,
    capture_size: int,
) -> torch.Tensor:
    """Sanitize one final request-table snapshot for every serial step.

    The caller must first allocate the page containing the final proposal
    position and validate that column zero is allocated for every real row.
    Later logical pages may still be negative; FULL masks make them invisible,
    while the FIA ABI requires their physical page indices to be valid.
    """

    if request_block_tables.ndim != 2:
        raise ValueError("Linear draft FIA request block tables must be two-dimensional.")
    real_rows, table_width = request_block_tables.shape
    if real_rows <= 0 or real_rows > capture_size or table_width <= 0:
        raise ValueError(
            "Linear draft FIA request tables require a non-empty real batch, "
            "positive width and sufficient capture rows."
        )

    fallback_pages = request_block_tables[:, :1]
    sanitized_tables = torch.where(
        request_block_tables >= 0,
        request_block_tables,
        fallback_pages,
    )
    if real_rows == capture_size:
        return sanitized_tables

    padded_tables = torch.empty(
        (capture_size, table_width),
        dtype=request_block_tables.dtype,
        device=request_block_tables.device,
    )
    padded_tables[:real_rows].copy_(sanitized_tables)
    padded_tables[real_rows:].copy_(sanitized_tables[:1].expand(capture_size - real_rows, -1))
    return padded_tables
