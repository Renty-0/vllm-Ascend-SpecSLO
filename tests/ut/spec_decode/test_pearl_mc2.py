# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.spec_decode.pearl.mc2 import (
    MC2Profile,
    detect_mc2_capability,
    matmul_allreduce_add_rmsnorm_or_fallback,
    mc2_source_sha256,
    normalize_mc2_profile,
)


def test_mc2_capability_is_explicit_on_cpu():
    capability = detect_mc2_capability("cpu", tp_size=3)
    assert not capability.available
    assert "Ascend" in capability.reason


def test_mc2_fallback_matches_shapes_and_is_differentiable():
    x = torch.randn(2, 4, 3, requires_grad=True)
    weight = torch.randn(5, 3, requires_grad=True)
    residual = torch.randn(2, 4, 5, requires_grad=True)
    gamma = torch.ones(5, requires_grad=True)
    output, added = matmul_allreduce_add_rmsnorm_or_fallback(
        x, weight, residual, gamma, tp_rank_size=1, use_fused=False
    )
    assert output.shape == residual.shape
    assert added.shape == residual.shape
    output.square().mean().backward()
    assert x.grad is not None


def _profile_document():
    return {
        "schema_version": 1,
        "metadata": {
            "operator": "matmul_allreduce_add_rmsnorm",
            "hardware": "Ascend 910B3",
            "tensor_parallel_size": 3,
            "source_sha256": mc2_source_sha256(),
        },
        "latency_percentile": 95.0,
        "minimum_samples": 3,
        "minimum_speedup": 1.02,
        "measurements": [
            {
                "m": 2,
                "k": 3,
                "n": 5,
                "dtype": "bfloat16",
                "weight_format": "ND",
                "is_trans_b": True,
                "baseline_latency_ms": [1.0, 1.0, 1.0],
                "fused_latency_ms": [0.8, 0.8, 0.8],
                "max_abs_norm": 0.01,
                "norm_atol": 0.05,
                "max_abs_added": 0.001,
                "added_atol": 0.01,
            }
        ],
    }


def test_mc2_profile_requires_current_source_identity():
    profile = normalize_mc2_profile(_profile_document())
    assert isinstance(profile, MC2Profile)
    tampered = _profile_document()
    tampered["metadata"]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source hash"):
        normalize_mc2_profile(tampered)


def test_mc2_profile_qualifies_only_exact_faster_numerical_shape(monkeypatch):
    profile = normalize_mc2_profile(_profile_document())
    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_device_name=lambda _index: "Ascend 910B3"))
    x = torch.empty(2, 3, dtype=torch.bfloat16)
    weight = torch.empty(5, 3, dtype=torch.bfloat16)
    residual = torch.empty(2, 5, dtype=torch.bfloat16)
    assert profile.qualify(x, weight, residual, tp_size=3, is_trans_b=True).qualified
    assert not profile.qualify(x[:1], weight, residual[:1], tp_size=3, is_trans_b=True).qualified


def test_mc2_profile_retains_unqualified_measurement_as_fallback(monkeypatch):
    document = _profile_document()
    document["measurements"][0]["fused_latency_ms"] = [1.5, 1.5, 1.5]
    profile = normalize_mc2_profile(document)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_device_name=lambda _index: "Ascend 910B3"))
    x = torch.empty(2, 3, dtype=torch.bfloat16)
    weight = torch.empty(5, 3, dtype=torch.bfloat16)
    residual = torch.empty(2, 5, dtype=torch.bfloat16)
    result = profile.qualify(x, weight, residual, tp_size=3, is_trans_b=True)
    assert not result.qualified
    assert "not faster" in result.reason
