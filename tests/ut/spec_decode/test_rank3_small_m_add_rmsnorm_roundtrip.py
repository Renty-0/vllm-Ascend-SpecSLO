# SPDX-License-Identifier: Apache-2.0
"""Static CPU contracts for the TP3 BF16 SingleN rounding boundary."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TILER = (
    REPO_ROOT
    / "csrc/mc2/allreduce_add_rmsnorm/op_host/allreduce_add_rmsnorm_tiling.cpp"
)
KERNEL = (
    REPO_ROOT
    / "csrc/mc2/allreduce_add_rmsnorm/op_kernel/allreduce_add_rmsnorm_aiv_kernel.h"
)


def test_host_marks_only_the_baseline_single_n_row_regime():
    source = TILER.read_text(encoding="utf-8")

    assert "constexpr uint32_t kBaselineSingleNRoundTrip = 1U << 0;" in source
    assignment = (
        "pp.tilingKey = m <= static_cast<int64_t>(aivNum) ?\n"
        "        kBaselineSingleNRoundTrip : 0U;"
    )
    assert assignment in source
    assert source.index(assignment) < source.index("context->SetBlockDim(launchedAiv);")


def test_kernel_roundtrip_is_fail_closed_to_dedicated_tp3_bf16_n5120():
    source = KERNEL.read_text(encoding="utf-8")
    gate_start = source.index("bool baseline_single_n_roundtrip = false;")
    gate_end = source.index("LocalTensor<MmadDtype> x_local;", gate_start)
    gate = source[gate_start:gate_end]

    assert "if constexpr (std::is_same<MmadDtype, bfloat16_t>::value)" in gate
    assert "dedicated_tp3_reduce &&" in gate
    assert "rank_size == TP3_AIV_ONLY_RANKS &&" in gate
    assert "n == 5120 &&" in gate
    assert "tiling_key & TP3_BASELINE_SINGLE_N_ROUNDTRIP" in gate


def test_small_m_roundtrip_precedes_rms_and_has_both_casts():
    source = KERNEL.read_text(encoding="utf-8")
    branch_start = source.index("} else if (baseline_single_n_roundtrip) {")
    rms_start = source.index("Mul(sqx, norm_fp32, norm_fp32, n);", branch_start)
    branch = source[branch_start:rms_start]

    cast_to_bf16 = "Cast(out_local, norm_fp32, RoundMode::CAST_RINT, n);"
    cast_back_to_fp32 = "Cast(norm_fp32, out_local, RoundMode::CAST_NONE, n);"
    assert branch.count(cast_to_bf16) == 1
    assert branch.count(cast_back_to_fp32) == 1
    assert branch.index(cast_to_bf16) < branch.index(cast_back_to_fp32)
