"""Device tensor packing for selective linear target verification."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def pack_selective_target_queries(
    roots: torch.Tensor,
    proposals: torch.Tensor,
    widths: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack causal target inputs and comparison labels in identical row order.

    A width-``w`` proposal ``d1..dw`` asks the target about
    ``[root,d1,..,d(w-1)]`` and compares the result with ``[d1,..,dw]``.
    The unused draft suffix remains outside the target query and verdict.
    """

    row_widths = tuple(int(value) for value in widths)
    if roots.ndim != 1 or proposals.ndim != 2:
        raise ValueError("Selective target roots/proposals must be rank one/two.")
    rows, gamma = proposals.shape
    if rows == 0 or roots.shape != (rows,) or len(row_widths) != rows:
        raise ValueError("Selective target inputs must contain aligned non-empty rows.")
    if gamma != 4 or any(width not in (2, 4) for width in row_widths):
        raise ValueError("Selective target verification currently supports gamma four and widths two/four.")
    if roots.dtype != torch.long or proposals.dtype != torch.long or roots.device != proposals.device:
        raise ValueError("Selective target tokens must share one int64 device.")

    query_rows = torch.cat((roots.unsqueeze(1), proposals[:, :3]), dim=1).reshape(-1)
    proposal_rows = proposals.reshape(-1)
    query_offsets = [row * 4 + step for row, width in enumerate(row_widths) for step in range(width)]
    index = torch.tensor(query_offsets, dtype=torch.long, device=roots.device)
    return query_rows.index_select(0, index), proposal_rows.index_select(0, index)
