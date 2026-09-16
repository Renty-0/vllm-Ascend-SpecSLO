# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
"""Fused verdict construction for fixed-width greedy speculation."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

FIXED_GREEDY_FULL_WINDOW_WIDTH = 4


@triton.jit
def _fixed_greedy_full_window_verdict_kernel(
    target_token_ids_ptr,
    draft_token_ids_ptr,
    bonus_token_ids_ptr,
    bonus_enabled_mask_ptr,
    verdict_ptr,
    WIDTH: tl.constexpr,
    TARGET_STRIDE: tl.constexpr,
    HAS_BONUS: tl.constexpr,
    BONUS_IN_TARGET_ROWS: tl.constexpr,
):
    """Write ``(accepted_prefix_length, correction_or_bonus_token)``."""
    row = tl.program_id(axis=0)
    lane_offsets = tl.arange(0, WIDTH)
    target_offsets = row * TARGET_STRIDE + lane_offsets
    draft_offsets = row * WIDTH + lane_offsets
    target_token_ids = tl.load(target_token_ids_ptr + target_offsets)
    draft_token_ids = tl.load(draft_token_ids_ptr + draft_offsets)

    mismatches = (target_token_ids != draft_token_ids).to(tl.int32)
    all_match = tl.sum(mismatches, axis=0) == 0
    # argmax is only semantically consumed when at least one lane mismatches;
    # for a fully accepted row the selected token is replaced by -1 below.
    first_mismatch = tl.argmax(mismatches, axis=0)
    accepted = tl.where(all_match, WIDTH, first_mismatch)
    correction = tl.load(target_token_ids_ptr + row * TARGET_STRIDE + first_mismatch)
    correction = tl.where(all_match, -1, correction)
    if HAS_BONUS:
        # Triton-Ascend currently materializes torch.bool loads as int8.  Make
        # the predicate explicit so ``tl.where`` does not rely on the
        # deprecated implicit integer-to-bool conversion.
        bonus_enabled = tl.load(bonus_enabled_mask_ptr + row) != 0
        if BONUS_IN_TARGET_ROWS:
            bonus_token_id = tl.load(target_token_ids_ptr + row * TARGET_STRIDE + WIDTH)
        else:
            bonus_token_id = tl.load(bonus_token_ids_ptr + row)
        correction = tl.where(all_match & bonus_enabled, bonus_token_id, correction)

    output_offset = row * 2
    tl.store(verdict_ptr + output_offset, accepted)
    tl.store(verdict_ptr + output_offset + 1, correction)


def _fixed_greedy_full_window_verdict_reference(
    target_token_ids: torch.Tensor,
    draft_token_ids: torch.Tensor,
    bonus_token_ids: torch.Tensor | None = None,
    bonus_enabled_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Torch fallback with the same fixed-width verdict contract."""
    target_matrix = target_token_ids.reshape(-1, FIXED_GREEDY_FULL_WINDOW_WIDTH)
    draft_matrix = draft_token_ids.reshape(-1, FIXED_GREEDY_FULL_WINDOW_WIDTH)
    matches = target_matrix == draft_matrix
    all_match = matches.all(dim=1)
    first_mismatch = (~matches).to(dtype=torch.int32).argmax(dim=1)
    accepted = torch.where(
        all_match,
        torch.full_like(first_mismatch, FIXED_GREEDY_FULL_WINDOW_WIDTH),
        first_mismatch,
    ).to(dtype=torch.long)
    correction = target_matrix.gather(
        1,
        first_mismatch.to(dtype=torch.long).unsqueeze(1),
    ).squeeze(1)
    correction = torch.where(all_match, torch.full_like(correction, -1), correction)
    if bonus_token_ids is not None:
        assert bonus_enabled_mask is not None
        correction = torch.where(
            all_match & bonus_enabled_mask,
            bonus_token_ids,
            correction,
        )
    return torch.stack((accepted, correction), dim=1)


def fixed_greedy_full_window_verdict(
    target_token_ids: torch.Tensor,
    draft_token_ids: torch.Tensor,
    bonus_token_ids: torch.Tensor | None = None,
    bonus_enabled_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the fixed-γ=4 greedy verdict in one kernel on Ascend.

    Both inputs contain contiguous, row-major complete proposal windows.  Row
    ``i`` is accepted through its first mismatch; a rejected row carries the
    target token at that mismatch.  By default a fully accepted row carries
    ``-1``, preserving the original contract.  When both optional bonus inputs
    are supplied, a fully accepted row carries its bonus token if that row's
    mask is true and ``-1`` otherwise.  The result remains a ``[B, 2]`` tensor.
    CPU and other device types intentionally retain a plain Torch reference
    path so protocol/unit tests do not require Triton-Ascend hardware.
    """
    if target_token_ids.ndim != 1 or draft_token_ids.shape != target_token_ids.shape:
        raise ValueError("Fixed greedy verdict expects aligned one-dimensional token tensors.")
    if target_token_ids.dtype != torch.long or draft_token_ids.dtype != torch.long:
        raise ValueError("Fixed greedy verdict token tensors must use torch.long.")
    if target_token_ids.device != draft_token_ids.device:
        raise ValueError("Fixed greedy verdict token tensors must share one device.")
    if target_token_ids.numel() % FIXED_GREEDY_FULL_WINDOW_WIDTH:
        raise ValueError("Fixed greedy verdict token count must be divisible by four.")
    if not target_token_ids.is_contiguous() or not draft_token_ids.is_contiguous():
        raise ValueError("Fixed greedy verdict token tensors must be contiguous.")

    has_bonus = bonus_token_ids is not None or bonus_enabled_mask is not None
    if has_bonus and (bonus_token_ids is None or bonus_enabled_mask is None):
        raise ValueError("Fixed greedy verdict bonus token ids and enabled mask must be supplied together.")
    batch_size = target_token_ids.numel() // FIXED_GREEDY_FULL_WINDOW_WIDTH
    if bonus_token_ids is not None:
        assert bonus_enabled_mask is not None
        expected_shape = (batch_size,)
        if bonus_token_ids.shape != expected_shape or bonus_enabled_mask.shape != expected_shape:
            raise ValueError("Fixed greedy verdict bonus tensors must have shape (batch_size,).")
        if bonus_token_ids.dtype != torch.long:
            raise ValueError("Fixed greedy verdict bonus token ids must use torch.long.")
        if bonus_enabled_mask.dtype != torch.bool:
            raise ValueError("Fixed greedy verdict bonus enabled mask must use torch.bool.")
        if bonus_token_ids.device != target_token_ids.device or bonus_enabled_mask.device != target_token_ids.device:
            raise ValueError("Fixed greedy verdict inputs must share one device.")
        if not bonus_token_ids.is_contiguous() or not bonus_enabled_mask.is_contiguous():
            raise ValueError("Fixed greedy verdict bonus tensors must be contiguous.")
    if not target_token_ids.numel():
        return torch.empty((0, 2), dtype=torch.long, device=target_token_ids.device)
    if target_token_ids.device.type != "npu":
        return _fixed_greedy_full_window_verdict_reference(
            target_token_ids,
            draft_token_ids,
            bonus_token_ids,
            bonus_enabled_mask,
        )

    verdict = torch.empty((batch_size, 2), dtype=torch.long, device=target_token_ids.device)
    if batch_size:
        _fixed_greedy_full_window_verdict_kernel[(batch_size,)](
            target_token_ids,
            draft_token_ids,
            bonus_token_ids if bonus_token_ids is not None else target_token_ids,
            bonus_enabled_mask if bonus_enabled_mask is not None else draft_token_ids,
            verdict,
            WIDTH=FIXED_GREEDY_FULL_WINDOW_WIDTH,
            TARGET_STRIDE=FIXED_GREEDY_FULL_WINDOW_WIDTH,
            HAS_BONUS=bonus_token_ids is not None,
            BONUS_IN_TARGET_ROWS=False,
        )
    return verdict


def fixed_greedy_full_window_bonus_verdict(
    target_token_rows: torch.Tensor,
    draft_token_ids: torch.Tensor,
    bonus_enabled_mask: torch.Tensor,
) -> torch.Tensor:
    """Build fixed-γ=4 verdicts directly from ``[B, 5]`` target rows.

    Each target row contains four verification token IDs followed by its
    bonus token ID.  ``draft_token_ids`` may be either ``[B, 4]`` or flat
    ``[B * 4]``; both layouts are consumed without a materializing copy when
    contiguous.  A full match publishes column five only for a row whose
    bonus mask is true.  Rejection keeps the correction-token behavior of
    :func:`fixed_greedy_full_window_verdict` and the output remains ``[B, 2]``.
    """
    bonus_width = FIXED_GREEDY_FULL_WINDOW_WIDTH + 1
    if target_token_rows.ndim != 2 or target_token_rows.shape[1:] != (bonus_width,):
        raise ValueError("Fixed greedy bonus verdict target tokens must have shape (batch_size, 5).")
    batch_size = target_token_rows.shape[0]
    valid_draft_shapes = {
        (batch_size, FIXED_GREEDY_FULL_WINDOW_WIDTH),
        (batch_size * FIXED_GREEDY_FULL_WINDOW_WIDTH,),
    }
    if tuple(draft_token_ids.shape) not in valid_draft_shapes:
        raise ValueError(
            "Fixed greedy bonus verdict draft tokens must have shape (batch_size, 4) or (batch_size * 4,)."
        )
    if bonus_enabled_mask.shape != (batch_size,):
        raise ValueError("Fixed greedy bonus verdict enabled mask must have shape (batch_size,).")
    if target_token_rows.dtype != torch.long or draft_token_ids.dtype != torch.long:
        raise ValueError("Fixed greedy bonus verdict token tensors must use torch.long.")
    if bonus_enabled_mask.dtype != torch.bool:
        raise ValueError("Fixed greedy bonus verdict enabled mask must use torch.bool.")
    if target_token_rows.device != draft_token_ids.device or target_token_rows.device != bonus_enabled_mask.device:
        raise ValueError("Fixed greedy bonus verdict inputs must share one device.")
    if (
        not target_token_rows.is_contiguous()
        or not draft_token_ids.is_contiguous()
        or not bonus_enabled_mask.is_contiguous()
    ):
        raise ValueError("Fixed greedy bonus verdict inputs must be contiguous.")
    if not batch_size:
        return torch.empty(
            (0, 2),
            dtype=torch.long,
            device=target_token_rows.device,
        )
    if target_token_rows.device.type != "npu":
        return _fixed_greedy_full_window_verdict_reference(
            target_token_rows[:, :FIXED_GREEDY_FULL_WINDOW_WIDTH],
            draft_token_ids,
            target_token_rows[:, FIXED_GREEDY_FULL_WINDOW_WIDTH],
            bonus_enabled_mask,
        )

    verdict = torch.empty(
        (batch_size, 2),
        dtype=torch.long,
        device=target_token_rows.device,
    )
    _fixed_greedy_full_window_verdict_kernel[(batch_size,)](
        target_token_rows,
        draft_token_ids,
        target_token_rows,
        bonus_enabled_mask,
        verdict,
        WIDTH=FIXED_GREEDY_FULL_WINDOW_WIDTH,
        TARGET_STRIDE=bonus_width,
        HAS_BONUS=True,
        BONUS_IN_TARGET_ROWS=True,
    )
    return verdict


__all__ = [
    "FIXED_GREEDY_FULL_WINDOW_WIDTH",
    "fixed_greedy_full_window_bonus_verdict",
    "fixed_greedy_full_window_verdict",
]
