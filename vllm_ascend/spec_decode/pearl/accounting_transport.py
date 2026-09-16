# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reusable fixed-shape control-plane envelopes for SpecRhythm.

The linear service publishes two float64 envelopes per cycle: the cycle/finish
accounting record and the post-refill service frontier.  Their shapes are fixed
for the lifetime of one ``generate`` call, so allocating a fresh device tensor
for every publication only adds allocator and H2D bookkeeping to the critical
path.  This module owns the deliberately small, protocol-agnostic buffer used
by both the WORLD/HCCL compatibility path and the opt-in CPU/Gloo path.

The caller still owns collective ordering and all semantic validation.  A
rewrite must cover the complete envelope; accepting partial writes here would
make stale identity or finish fields possible after a shorter logical cycle.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real

import torch
import torch.distributed as dist


class ReusableSpecRhythmEnvelope:
    """One fixed-size float64 broadcast buffer with full-rewrite semantics."""

    def __init__(self, numel: int, *, device: torch.device | str) -> None:
        if not isinstance(numel, int) or isinstance(numel, bool) or numel <= 0:
            raise ValueError("A SpecRhythm envelope must contain at least one value.")
        self._device = torch.device(device)
        self._buffer = torch.empty(
            int(numel),
            dtype=torch.float64,
            device=self._device,
        )
        # A persistent CPU staging tensor avoids allocating the destination of
        # the host-to-device copy on every HCCL publication.  The CPU/Gloo path
        # writes the collective buffer itself and therefore needs no duplicate
        # staging allocation.
        self._host_staging = (
            self._buffer if self._device.type == "cpu" else torch.empty(int(numel), dtype=torch.float64, device="cpu")
        )

    @property
    def buffer(self) -> torch.Tensor:
        """Return the stable tensor passed to ``dist.broadcast``."""
        return self._buffer

    @property
    def host_staging(self) -> torch.Tensor:
        """Return the stable host-side staging tensor (primarily for tests)."""
        return self._host_staging

    def rewrite(self, values: Sequence[Real]) -> torch.Tensor:
        """Overwrite every field and return the same collective buffer.

        Length is checked before either persistent tensor is mutated.  A tiny
        temporary CPU conversion may still be created by PyTorch, but both the
        collective allocation and the H2D destination are retained across
        cycles, which are the expensive allocations on the NPU path.
        """
        if len(values) != self._buffer.numel():
            raise ValueError(
                "SpecRhythm envelope rewrite has the wrong number of fields: "
                f"expected {self._buffer.numel()}, got {len(values)}."
            )
        converted = torch.as_tensor(values, dtype=torch.float64, device="cpu")
        self._host_staging.copy_(converted)
        if self._host_staging is not self._buffer:
            self._buffer.copy_(self._host_staging)
        return self._buffer

    def broadcast_and_materialize(
        self,
        *,
        source_rank: int,
        group: dist.ProcessGroup | None = None,
    ) -> list[float]:
        """Publish once, then return the complete host-visible envelope.

        This intentionally remains a blocking broadcast followed by one
        materialization.  Keeping those operations together makes it harder
        for an optimization to accidentally reorder either accounting
        collective around the scheduler/frontier validations.
        """
        dist.broadcast(self._buffer, src=int(source_rank), group=group)
        return [float(value) for value in self._buffer.cpu().tolist()]
