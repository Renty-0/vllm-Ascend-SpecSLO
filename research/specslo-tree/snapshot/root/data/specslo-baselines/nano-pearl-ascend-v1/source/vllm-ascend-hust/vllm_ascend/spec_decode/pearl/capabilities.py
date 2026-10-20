# SPDX-License-Identifier: Apache-2.0
"""Runtime capability inventory for the vLLM-Ascend-SpecSLO port."""

from __future__ import annotations

from typing import Any

import torch

from vllm_ascend.spec_decode.pearl.mc2 import capability_dict


def _has_torch_npu(name: str) -> bool:
    try:
        import torch_npu

        return hasattr(torch_npu, name)
    except (ImportError, RuntimeError):
        return False


def collect_specslo_capabilities(
    device: torch.device | str | None = None,
    *,
    tp_size: int = 1,
) -> dict[str, Any]:
    """Collect operator and graph support without initializing a model."""

    resolved = torch.device(device or ("npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cpu"))
    if resolved.type == "npu":
        npu_graph = bool(hasattr(torch.npu, "NPUGraph"))
    else:
        npu_graph = False
    is_npu = resolved.type == "npu"
    report: dict[str, Any] = {
        "device": str(resolved),
        "aclgraph": is_npu and npu_graph,
        "fused_infer_attention_score": is_npu and _has_torch_npu("npu_fused_infer_attention_score"),
        "paged_attention": is_npu and _has_torch_npu("_npu_paged_attention"),
        "reshape_and_cache": is_npu and _has_torch_npu("_npu_reshape_and_cache"),
        "rotary_embedding": is_npu and _has_torch_npu("npu_rotary_embedding"),
        "production_rope": is_npu and hasattr(getattr(torch.ops, "vllm"), "npu_rotary_embedding"),
        "qkv_rmsnorm_rope": is_npu and hasattr(getattr(torch.ops, "vllm"), "qkv_rmsnorm_rope"),
        "mc2": capability_dict(resolved, tp_size),
        "tree_verification": True,
        "notes": {
            "tree_verification": "device kernel is available when the vLLM V1 tree path is enabled",
            "mc2": "requires the compiled vllm_ascend custom extension and a TP group",
        },
    }
    return report


__all__ = ["collect_specslo_capabilities"]
