#!/usr/bin/env python3
"""Static dependency audit for the SpecSLO TP3 direct MC2 protocol.

This tool does not import torch and does not touch an NPU.  It models the two
cross-rank phases in the small-M direct path:

* payload-ready, before a consumer may read the three projection windows;
* read-complete, before a producer may overwrite its projection window.

The production protocol publishes a source-owned slot for every source and
destination pair, including the local ``source == destination`` pair.  The
experimental self-IPC-elision patch retains all remote pairs and replaces the
local pair with the already-required local AIC/AIV ordering.  Exhaustively
checking every observable event subset proves that both formulations expose
the same reduce/overwrite predicates; it is not a substitute for the NPU
eager/ACLGraph qualification required before enabling the patch.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path


@dataclass(frozen=True)
class ProtocolCounts:
    rank_size: int
    phases: int
    reset_dma_per_cluster: int
    reset_slots_per_rank: int
    publish_dma_per_cluster: int
    poll_lanes_per_cluster: int
    cross_rank_rendezvous: int
    local_sync_all: int


def protocol_counts(rank_size: int, *, elide_self_ipc: bool) -> ProtocolCounts:
    if rank_size < 2:
        raise ValueError("rank_size must be at least 2")
    phases = 2
    peers_per_rank = rank_size - int(elide_self_ipc)
    return ProtocolCounts(
        rank_size=rank_size,
        phases=phases,
        # ResetTp3DirectIpcFlags coalesces both 3-slot phase rows into one
        # local DMA.  The experimental patch deliberately leaves this intact.
        reset_dma_per_cluster=rank_size,
        reset_slots_per_rank=phases * rank_size,
        publish_dma_per_cluster=phases * rank_size * peers_per_rank,
        poll_lanes_per_cluster=phases * rank_size * peers_per_rank,
        # One entry rendezvous remains mandatory to order reset before remote
        # publication.  Ready and read-complete are the other two phases.
        cross_rank_rendezvous=3,
        # Direct Process contains five local SyncAll points; the candidate
        # does not remove any of them.
        local_sync_all=5,
    )


def _full_ready(local_aic_done: bool, ready_slots: tuple[bool, ...], rank: int) -> bool:
    slots = list(ready_slots)
    slots[rank] = local_aic_done
    return all(slots)


def _elided_ready(local_aic_done: bool, ready_slots: tuple[bool, ...], rank: int) -> bool:
    return local_aic_done and all(value for source, value in enumerate(ready_slots) if source != rank)


def _full_overwrite(local_read_done: bool, done_slots: tuple[bool, ...], rank: int) -> bool:
    slots = list(done_slots)
    slots[rank] = local_read_done
    return all(slots)


def _elided_overwrite(local_read_done: bool, done_slots: tuple[bool, ...], rank: int) -> bool:
    return local_read_done and all(value for consumer, value in enumerate(done_slots) if consumer != rank)


def audit_dependency_equivalence(rank_size: int) -> dict[str, int | bool]:
    if rank_size < 2:
        raise ValueError("rank_size must be at least 2")
    checked = 0
    for rank in range(rank_size):
        for local_state in (False, True):
            for remote_state in product((False, True), repeat=rank_size - 1):
                slots = []
                remote_iter = iter(remote_state)
                for peer in range(rank_size):
                    slots.append(local_state if peer == rank else next(remote_iter))
                slot_tuple = tuple(slots)
                assert _full_ready(local_state, slot_tuple, rank) == _elided_ready(
                    local_state, slot_tuple, rank
                )
                assert _full_overwrite(local_state, slot_tuple, rank) == _elided_overwrite(
                    local_state, slot_tuple, rank
                )
                checked += 2
    return {
        "rank_size": rank_size,
        "equivalent": True,
        "predicate_cases_checked": checked,
    }


def build_report(rank_size: int = 3) -> dict[str, object]:
    baseline = protocol_counts(rank_size, elide_self_ipc=False)
    candidate = protocol_counts(rank_size, elide_self_ipc=True)
    return {
        "status": "static_only",
        "npu_used": False,
        "dependency_audit": audit_dependency_equivalence(rank_size),
        "baseline": asdict(baseline),
        "experimental_self_ipc_elision": asdict(candidate),
        "delta": {
            "publish_dma_per_cluster": (
                candidate.publish_dma_per_cluster - baseline.publish_dma_per_cluster
            ),
            "poll_lanes_per_cluster": (
                candidate.poll_lanes_per_cluster - baseline.poll_lanes_per_cluster
            ),
            "reset_dma_per_cluster": 0,
            "cross_rank_rendezvous": 0,
            "local_sync_all": 0,
        },
        "qualification_required": [
            "compile the opt-in candidate in a separate vendor package",
            "eager changed-input bit-exact test for full epsilon=1e-6 epilogue",
            "eager changed-input bit-exact projection-only test",
            "ACLGraph changed-input replay without timeout or graph fallback",
            "interleaved baseline/candidate latency samples on uncontended TP3 devices",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-size", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.rank_size)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
