from pathlib import Path

import pytest

from examples.audit_specslo_tp3_mc2_sync import (
    audit_dependency_equivalence,
    build_report,
    protocol_counts,
)

ROOT = Path(__file__).resolve().parents[3]
AIV_SOURCE = ROOT / "csrc/mc2/matmul_allreduce_add_rmsnorm/op_kernel" / "matmul_allreduce_add_rmsnorm_aiv_kernel.h"
TILING_SOURCE = ROOT / "csrc/mc2/matmul_allreduce_add_rmsnorm/op_host" / "matmul_allreduce_add_rmsnorm_tiling.cpp"
ADAPTER_SOURCE = ROOT / "csrc/mc2/matmul_allreduce_add_rmsnorm" / "matmul_allreduce_add_rmsnorm_torch_adpt.h"
EXPERIMENTAL_PATCH = ROOT / "docs/patches/specslo_tp3_mc2_elide_self_ipc_experimental.patch"


@pytest.mark.parametrize("rank_size", range(2, 9))
def test_eliding_only_self_ipc_preserves_ready_and_overwrite_predicates(rank_size: int):
    result = audit_dependency_equivalence(rank_size)

    assert result["equivalent"] is True
    assert result["predicate_cases_checked"] == rank_size * 2 * (2 ** (rank_size - 1)) * 2


def test_tp3_candidate_removes_only_redundant_local_slots():
    baseline = protocol_counts(3, elide_self_ipc=False)
    candidate = protocol_counts(3, elide_self_ipc=True)

    assert baseline.publish_dma_per_cluster == 18
    assert candidate.publish_dma_per_cluster == 12
    assert baseline.poll_lanes_per_cluster == 18
    assert candidate.poll_lanes_per_cluster == 12
    assert candidate.reset_dma_per_cluster == baseline.reset_dma_per_cluster == 3
    assert candidate.cross_rank_rendezvous == baseline.cross_rank_rendezvous == 3
    assert candidate.local_sync_all == baseline.local_sync_all == 5


def test_static_report_is_explicitly_non_npu_and_requires_graph_qualification():
    report = build_report()

    assert report["status"] == "static_only"
    assert report["npu_used"] is False
    assert any("ACLGraph" in gate for gate in report["qualification_required"])
    assert report["delta"]["publish_dma_per_cluster"] == -6
    assert report["delta"]["poll_lanes_per_cluster"] == -6


def test_production_protocol_retains_all_mandatory_ordering_edges():
    source = AIV_SOURCE.read_text(encoding="utf-8")

    wait = source.index("WaitEvent(0);")
    reset = source.index("ResetTp3DirectIpcFlags(false);", wait)
    entry = source.index("CrossRankSyncEx(FLAG_NUM, true);", reset)
    ready = source.index("Tp3PublishAndWait(FLAG_ZERO_IDX, 1, true);", entry)
    compute = source.index("ParallelWithSplitStepOneAddNorm", ready)
    done = source.index("Tp3PublishAndWait(FLAG_ONE_IDX, 1, true);", compute)

    assert wait < reset < entry < ready < compute < done
    assert "PipeBarrier<PIPE_ALL>();\n            AscendC::SyncAll<true>();" in source[compute:done]
    assert "PipeSync<HardEvent::MTE3_S>();" in source


def test_experimental_patch_is_self_contained_and_symmetric_for_publish_and_poll():
    patch = EXPERIMENTAL_PATCH.read_text(encoding="utf-8")

    # The patch itself is the opt-in boundary: it is intentionally not applied
    # to the production source tree, but produces an enabled candidate vendor
    # when applied in a disposable worktree.
    assert "#define VLLM_ASCEND_EXPERIMENTAL_TP3_ELIDE_SELF_IPC 1" in patch
    assert "VLLM_ASCEND_EXPERIMENTAL_TP3_ELIDE_SELF_IPC" not in AIV_SOURCE.read_text(encoding="utf-8")
    assert patch.count("Tp3DirectIpcPeerRequired(core_idx, concurrent_publish_poll)") == 2
    assert "return peer_rank != rank;" in patch
    assert "AscendC::SyncAll<true>();" not in "".join(
        line[1:] for line in patch.splitlines() if line.startswith("-") and not line.startswith("---")
    )


def test_v119_numerical_contract_remains_fail_closed_in_production_source():
    tiling = TILING_SOURCE.read_text(encoding="utf-8")
    adapter = ADAPTER_SOURCE.read_text(encoding="utf-8")
    aiv = AIV_SOURCE.read_text(encoding="utf-8")

    assert "ppTilingData.projectionOnly ? epsilon : 1.0e-6F" in tiling
    assert "projection_only || epsilon_f == 1.0e-6F" in adapter
    assert "if (!projection_only)" in aiv
    assert "if (projection_only)" in aiv
