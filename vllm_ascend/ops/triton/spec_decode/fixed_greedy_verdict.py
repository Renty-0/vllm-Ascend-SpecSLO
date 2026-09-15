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
    verdict_ptr,
    WIDTH: tl.constexpr,
):
    """Write ``(accepted_prefix_length, correction_token_id)`` per row."""
    row = tl.program_id(axis=0)
    offsets = row * WIDTH + tl.arange(0, WIDTH)
    target_token_ids = tl.load(target_token_ids_ptr + offsets)
    draft_token_ids = tl.load(draft_token_ids_ptr + offsets)

    mismatches = (target_token_ids != draft_token_ids).to(tl.int32)
    all_match = tl.sum(mismatches, axis=0) == 0
    # argmax is only semantically consumed when at least one lane mismatches;
    # for a fully accepted row the selected token is replaced by -1 below.
    first_mismatch = tl.argmax(mismatches, axis=0)
    accepted = tl.where(all_match, WIDTH, first_mismatch)
    correction = tl.load(target_token_ids_ptr + row * WIDTH + first_mismatch)
    correction = tl.where(all_match, -1, correction)

    output_offset = row * 2
    tl.store(verdict_ptr + output_offset, accepted)
    tl.store(verdict_ptr + output_offset + 1, correction)


def _fixed_greedy_full_window_verdict_reference(
    target_token_ids: torch.Tensor,
    draft_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Torch fallback with the same no-bonus-token verdict contract."""
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
    return torch.stack((accepted, correction), dim=1)


def fixed_greedy_full_window_verdict(
    target_token_ids: torch.Tensor,
    draft_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Return the fixed-γ=4 greedy verdict in one kernel on Ascend.

    Both inputs contain contiguous, row-major complete proposal windows.  Row
    ``i`` is accepted through its first mismatch; a rejected row carries the
    target token at that mismatch, while a fully accepted row carries ``-1``.
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
    if not target_token_ids.numel():
        return torch.empty((0, 2), dtype=torch.long, device=target_token_ids.device)
    if target_token_ids.device.type != "npu":
        return _fixed_greedy_full_window_verdict_reference(target_token_ids, draft_token_ids)

    batch_size = target_token_ids.numel() // FIXED_GREEDY_FULL_WINDOW_WIDTH
    verdict = torch.empty((batch_size, 2), dtype=torch.long, device=target_token_ids.device)
    if batch_size:
        _fixed_greedy_full_window_verdict_kernel[(batch_size,)](
            target_token_ids,
            draft_token_ids,
            verdict,
            WIDTH=FIXED_GREEDY_FULL_WINDOW_WIDTH,
        )
    return verdict


__all__ = [
    "FIXED_GREEDY_FULL_WINDOW_WIDTH",
    "fixed_greedy_full_window_verdict",
]
