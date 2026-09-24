# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PEARL process-group topology for independent draft and target workers."""

from __future__ import annotations

from dataclasses import dataclass

import torch.distributed as dist


@dataclass(frozen=True)
class PearlTopology:
    """Describes the rank layout used by a disaggregated PEARL deployment.

    Every worker belongs to exactly one model-parallel group. The draft leader
    joins the verification group with every target rank, matching PEARL's
    proposal flow while keeping the two model groups independent.
    """

    draft_ranks: tuple[int, ...]
    target_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        draft_ranks = tuple(self.draft_ranks)
        target_ranks = tuple(self.target_ranks)
        object.__setattr__(self, "draft_ranks", draft_ranks)
        object.__setattr__(self, "target_ranks", target_ranks)

        if not draft_ranks or not target_ranks:
            raise ValueError("PEARL requires at least one draft rank and one target rank.")
        if any(rank < 0 for rank in (*draft_ranks, *target_ranks)):
            raise ValueError("PEARL ranks must be non-negative.")
        if len(set(draft_ranks)) != len(draft_ranks):
            raise ValueError("Draft ranks must be unique.")
        if len(set(target_ranks)) != len(target_ranks):
            raise ValueError("Target ranks must be unique.")
        if set(draft_ranks).intersection(target_ranks):
            raise ValueError("Draft and target ranks must not overlap.")

    @classmethod
    def from_tensor_parallel_sizes(cls, draft_tp_size: int, target_tp_size: int) -> PearlTopology:
        """Create the contiguous rank layout used by the PEARL launcher."""
        if draft_tp_size <= 0 or target_tp_size <= 0:
            raise ValueError("Draft and target tensor-parallel sizes must both be positive.")
        return cls(
            draft_ranks=tuple(range(draft_tp_size)),
            target_ranks=tuple(range(draft_tp_size, draft_tp_size + target_tp_size)),
        )

    @property
    def world_size(self) -> int:
        """Number of workers required by this topology."""
        return len(self.draft_ranks) + len(self.target_ranks)

    @property
    def draft_leader_rank(self) -> int:
        """Global rank that broadcasts a proposal batch."""
        return self.draft_ranks[0]

    @property
    def target_leader_rank(self) -> int:
        """Global rank that broadcasts target verification results."""
        return self.target_ranks[0]

    @property
    def verification_ranks(self) -> tuple[int, ...]:
        """Ranks receiving candidate proposals from the draft leader."""
        return (self.draft_leader_rank, *self.target_ranks)

    @property
    def correction_ranks(self) -> tuple[int, ...]:
        """Ranks receiving corrections from the target leader."""
        return (*self.draft_ranks, self.target_leader_rank)

    def validate_world_size(self, world_size: int) -> None:
        """Ensure a torch distributed world can realize this rank layout."""
        expected_ranks = set(range(world_size))
        configured_ranks = set(self.draft_ranks).union(self.target_ranks)
        if configured_ranks != expected_ranks:
            raise ValueError(
                "PEARL topology must cover every distributed rank exactly once: "
                f"configured={sorted(configured_ranks)}, world_size={world_size}."
            )

    def is_draft_rank(self, rank: int) -> bool:
        """Return whether ``rank`` belongs to the draft model group."""
        return rank in self.draft_ranks

    def is_verification_rank(self, rank: int) -> bool:
        """Return whether ``rank`` receives draft proposals for verification."""
        return rank in self.verification_ranks


@dataclass(frozen=True)
class PearlProcessGroups:
    """HCCL process groups required by a PEARL worker.

    Group construction is collective. All ranks must call :meth:`create` in
    the same order before either model starts its request loop.
    """

    topology: PearlTopology
    rank: int
    draft_group: dist.ProcessGroup
    target_group: dist.ProcessGroup
    verification_group: dist.ProcessGroup
    correction_group: dist.ProcessGroup
    world_coordination_group: dist.ProcessGroup
    verification_coordination_group: dist.ProcessGroup
    correction_coordination_group: dist.ProcessGroup

    @classmethod
    def create(cls, topology: PearlTopology, backend: str | None = None) -> PearlProcessGroups:
        """Create model-parallel and cross-model PEARL groups.

        Args:
            topology: Complete PEARL rank layout.
            backend: Distributed backend. Defaults to the initialized backend,
                which is ``hccl`` for the Ascend launcher.

        Raises:
            RuntimeError: If torch distributed has not been initialized.
            ValueError: If the initialized world does not match ``topology``.
        """
        if not dist.is_initialized():
            raise RuntimeError("Initialize torch.distributed before creating PEARL process groups.")

        world_size = dist.get_world_size()
        topology.validate_world_size(world_size)
        selected_backend = backend or str(dist.get_backend())

        # torch.distributed.new_group is collective even for ranks outside a
        # subgroup, so keep this globally identical order across all workers.
        draft_group = dist.new_group(ranks=list(topology.draft_ranks), backend=selected_backend)
        target_group = dist.new_group(ranks=list(topology.target_ranks), backend=selected_backend)
        verification_group = dist.new_group(ranks=list(topology.verification_ranks), backend=selected_backend)
        correction_group = dist.new_group(ranks=list(topology.correction_ranks), backend=selected_backend)
        # MC2 startup admission must include every draft and target rank.  It
        # cannot reuse ``verification_coordination_group`` because that group
        # contains only the draft leader when draft TP > 1.  Keep a dedicated
        # all-world CPU group so startup failures never submit HCCL work and
        # every worker executes the same group-construction order.
        world_coordination_group = dist.new_group(
            ranks=list(range(world_size)),
            backend="gloo",
        )
        # HCCL collectives from an overlapping process group must not be
        # submitted while target TP all-reduces are still in flight.  A tiny
        # host-side rendezvous lets every verification rank announce that its
        # local model stream is complete without consuming an NPU/HCCL stream
        # itself.  This is particularly important for TP3 stepwise target
        # verification, where target compute can outlive the draft chain.
        verification_coordination_group = dist.new_group(
            ranks=list(topology.verification_ranks),
            backend="gloo",
        )
        # Greedy target verification is replicated across the target TP
        # ranks.  Only the authoritative target leader and draft ranks need
        # to exchange the compact correction envelope; target followers can
        # consume their identical rank-local verdict.  Keep this CPU group
        # separate from the wider proposal/compute rendezvous so a two-value
        # correction does not serialize every target follower each cycle.
        correction_coordination_group = dist.new_group(
            ranks=list(topology.correction_ranks),
            backend="gloo",
        )

        return cls(
            topology=topology,
            rank=dist.get_rank(),
            draft_group=draft_group,
            target_group=target_group,
            verification_group=verification_group,
            correction_group=correction_group,
            world_coordination_group=world_coordination_group,
            verification_coordination_group=verification_coordination_group,
            correction_coordination_group=correction_coordination_group,
        )

    @property
    def is_draft_worker(self) -> bool:
        """Return whether this worker belongs to the draft model group."""
        return self.topology.is_draft_rank(self.rank)

    @property
    def model_group(self) -> dist.ProcessGroup:
        """Return the local model-parallel group for this worker."""
        return self.draft_group if self.is_draft_worker else self.target_group

    @property
    def is_verification_worker(self) -> bool:
        """Return whether this worker participates in proposal broadcasts."""
        return self.topology.is_verification_rank(self.rank)
