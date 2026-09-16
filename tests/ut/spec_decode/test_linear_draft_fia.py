# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the fixed-gamma linear-draft FIA tensor builders."""

import pytest
import torch

from vllm_ascend import envs
from vllm_ascend.spec_decode.pearl.linear_fia import (
    LinearDraftFIAFullMaskBuilder,
    LinearDraftFIAPersistentStaging,
    sanitize_and_pad_linear_draft_fia_request_tables,
    validate_linear_draft_fia_final_pages,
)


def test_batched_full_masks_are_independently_opt_in(monkeypatch):
    variable = "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_BATCHED_MASKS"
    monkeypatch.delenv(variable, raising=False)
    assert envs.VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_BATCHED_MASKS is False
    monkeypatch.setenv(variable, "1")
    assert envs.VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_BATCHED_MASKS is True


def test_persistent_staging_is_independently_opt_in(monkeypatch):
    variable = "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_PERSISTENT_STAGING"
    monkeypatch.delenv(variable, raising=False)
    assert envs.VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_PERSISTENT_STAGING is False
    monkeypatch.setenv(variable, "1")
    assert envs.VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_PERSISTENT_STAGING is True


def test_batched_full_masks_preserve_exact_visibility_and_dummy_rows():
    builder = LinearDraftFIAFullMaskBuilder()

    masks = builder.build(
        [2, 4],
        step_count=4,
        capture_size=4,
        capacity=12,
        device="cpu",
    )

    assert len(masks) == 4
    assert all(mask.shape == (4, 1, 1, 12) for mask in masks)
    assert all(mask.dtype == torch.bool for mask in masks)
    for step, mask in enumerate(masks):
        for row, first_position in enumerate((2, 4)):
            visible_length = first_position + step + 1
            assert not mask[row, 0, 0, :visible_length].any()
            assert mask[row, 0, 0, visible_length:].all()
        for dummy_row in (2, 3):
            assert not mask[dummy_row, 0, 0, 0]
            assert mask[dummy_row, 0, 0, 1:].all()


def test_full_mask_builder_caches_only_key_positions_not_mask_values():
    builder = LinearDraftFIAFullMaskBuilder()

    first_masks = builder.build(
        [1],
        step_count=4,
        capture_size=2,
        capacity=8,
        device="cpu",
    )
    cached_keys = builder._key_positions[(torch.device("cpu"), 8)]
    first_snapshot = torch.stack(first_masks).clone()

    second_masks = builder.build(
        [2],
        step_count=4,
        capture_size=2,
        capacity=8,
        device="cpu",
    )

    assert builder._key_positions[(torch.device("cpu"), 8)] is cached_keys
    assert torch.equal(torch.stack(first_masks), first_snapshot)
    assert not torch.equal(torch.stack(second_masks), first_snapshot)
    assert not second_masks[0][0, 0, 0, 2]
    assert first_masks[0][0, 0, 0, 2]


def test_persistent_staging_refreshes_exact_values_without_reallocation():
    staging = LinearDraftFIAPersistentStaging(
        step_count=4,
        capture_size=4,
        capacity=12,
        table_width=3,
        device="cpu",
    )
    tensor_pointers = {
        "input_ids": staging.input_ids.data_ptr(),
        "positions": staging.positions.data_ptr(),
        "slots": staging.slot_mappings.data_ptr(),
        "masks": staging.full_masks.data_ptr(),
        "tables": staging.request_block_tables.data_ptr(),
    }

    staging.refresh(
        [2, 4],
        [
            [10, 20],
            [11, 21],
            [12, 22],
            [13, 23],
        ],
        torch.tensor(
            [
                [7, 11, -1],
                [13, 17, 19],
            ],
            dtype=torch.int32,
        ),
        torch.tensor([101, 202]),
    )

    assert staging.input_ids.tolist() == [101, 202, 0, 0]
    assert staging.positions.tolist() == [
        [2, 4, 0, 0],
        [3, 5, 0, 0],
        [4, 6, 0, 0],
        [5, 7, 0, 0],
    ]
    assert staging.slot_mappings.tolist() == [
        [10, 20, -1, -1],
        [11, 21, -1, -1],
        [12, 22, -1, -1],
        [13, 23, -1, -1],
    ]
    assert staging.request_block_tables.tolist() == [
        [7, 11, 7],
        [13, 17, 19],
        [7, 11, 7],
        [7, 11, 7],
    ]
    for step, mask in enumerate(staging.full_mask_views):
        for row, first_position in enumerate((2, 4)):
            visible_length = first_position + step + 1
            assert not mask[row, 0, 0, :visible_length].any()
            assert mask[row, 0, 0, visible_length:].all()
        for dummy_row in (2, 3):
            assert not mask[dummy_row, 0, 0, 0]
            assert mask[dummy_row, 0, 0, 1:].all()

    staging.refresh(
        [5],
        [[31], [32], [33], [34]],
        torch.tensor([[23, -1, -1]], dtype=torch.int32),
        torch.tensor([303]),
    )

    assert tensor_pointers == {
        "input_ids": staging.input_ids.data_ptr(),
        "positions": staging.positions.data_ptr(),
        "slots": staging.slot_mappings.data_ptr(),
        "masks": staging.full_masks.data_ptr(),
        "tables": staging.request_block_tables.data_ptr(),
    }
    assert staging.input_ids.tolist() == [303, 0, 0, 0]
    assert staging.positions.tolist() == [
        [5, 0, 0, 0],
        [6, 0, 0, 0],
        [7, 0, 0, 0],
        [8, 0, 0, 0],
    ]
    assert staging.slot_mappings.tolist() == [
        [31, -1, -1, -1],
        [32, -1, -1, -1],
        [33, -1, -1, -1],
        [34, -1, -1, -1],
    ]
    assert staging.request_block_tables.tolist() == [
        [23, 23, 23],
        [23, 23, 23],
        [23, 23, 23],
        [23, 23, 23],
    ]


def test_persistent_staging_rejects_misaligned_dynamic_inputs():
    staging = LinearDraftFIAPersistentStaging(
        step_count=4,
        capture_size=2,
        capacity=8,
        table_width=2,
        device="cpu",
    )

    with pytest.raises(ValueError, match="slot mapping row"):
        staging.refresh(
            [1],
            [[4], [5], [6]],
            torch.tensor([[7, -1]], dtype=torch.int32),
            torch.tensor([11]),
        )
    with pytest.raises(ValueError, match="request tables"):
        staging.refresh(
            [1],
            [[4], [5], [6], [7]],
            torch.tensor([[7, -1]], dtype=torch.int64),
            torch.tensor([11]),
        )
    with pytest.raises(ValueError, match="complete serial window"):
        staging.refresh(
            [5],
            [[4], [5], [6], [7]],
            torch.tensor([[7, -1]], dtype=torch.int32),
            torch.tensor([11]),
        )


@pytest.mark.parametrize(
    ("positions", "step_count", "capture_size", "capacity"),
    [
        ([], 4, 4, 8),
        ([0, 1], 4, 1, 8),
        ([True], 4, 1, 8),
        ([-1], 4, 1, 8),
        ([5], 4, 1, 8),
    ],
)
def test_full_mask_builder_rejects_invalid_envelopes(
    positions,
    step_count,
    capture_size,
    capacity,
):
    with pytest.raises(ValueError):
        LinearDraftFIAFullMaskBuilder().build(
            positions,
            step_count=step_count,
            capture_size=capture_size,
            capacity=capacity,
            device="cpu",
        )


def test_final_request_table_snapshot_is_sanitized_once_and_shared_exactly():
    final_request_tables = torch.tensor(
        [
            [7, 11, -1, -1],
            [13, 17, 19, -1],
        ],
        dtype=torch.int32,
    )

    shared_table = sanitize_and_pad_linear_draft_fia_request_tables(
        final_request_tables,
        capture_size=4,
    )
    step_tables = (shared_table,) * 4

    assert shared_table.tolist() == [
        [7, 11, 7, 7],
        [13, 17, 19, 13],
        [7, 11, 7, 7],
        [7, 11, 7, 7],
    ]
    assert all(table is shared_table for table in step_tables)
    assert all(torch.equal(table, step_tables[0]) for table in step_tables[1:])
    assert final_request_tables.tolist() == [
        [7, 11, -1, -1],
        [13, 17, 19, -1],
    ]


def test_shared_request_table_requires_the_final_proposal_page():
    with pytest.raises(RuntimeError, match="final proposal page"):
        validate_linear_draft_fia_final_pages(
            [15],
            [[7, -1]],
            step_count=4,
            block_size=16,
            capacity=32,
        )

    validate_linear_draft_fia_final_pages(
        [15],
        [[7, 11]],
        step_count=4,
        block_size=16,
        capacity=32,
    )


@pytest.mark.parametrize(
    ("shape", "capture_size"),
    [
        ((2,), 2),
        ((0, 2), 2),
        ((2, 0), 2),
        ((3, 2), 2),
    ],
)
def test_request_table_builder_rejects_invalid_shapes(shape, capture_size):
    with pytest.raises(ValueError):
        sanitize_and_pad_linear_draft_fia_request_tables(
            torch.empty(shape, dtype=torch.int32),
            capture_size=capture_size,
        )
