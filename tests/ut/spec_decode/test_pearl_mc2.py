# SPDX-License-Identifier: Apache-2.0

import logging
import os
import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
from torch import nn

import vllm_ascend.spec_decode.pearl.mc2 as mc2_module
import vllm_ascend.spec_decode.pearl.native_model as native_model_module
from vllm_ascend.spec_decode.pearl.mc2 import (
    MC2_TP3_NATIVE_EPILOGUE_OPERATOR,
    MC2Capability,
    MC2Profile,
    MC2Qualification,
    bind_mc2_static_route,
    build_mc2_static_route,
    build_mc2_static_route_manifest,
    detect_mc2_capability,
    matmul_allreduce_add_rmsnorm_or_fallback,
    mc2_source_sha256,
    normalize_mc2_profile,
    reset_mc2_dispatch_counters,
    snapshot_mc2_dispatch_counters,
)
from vllm_ascend.spec_decode.pearl.native_engine import (
    NativePearlConfig,
    _configure_mc2_model_flags,
    _freeze_mc2_worker_routes,
    _validate_mc2_target_attention_sharding,
    _validate_mc2_worker_admission,
)
from vllm_ascend.spec_decode.pearl.native_model import (
    NativeQwen2DecoderLayer,
    NativeQwen2ForCausalLM,
    NativeQwen2MLP,
    NativeRMSNorm,
    NativeTPContext,
    _enable_tp3_down_mc2,
)

_TEST_VENDOR_SHA256 = "a" * 64
_TEST_ADAPTER_SHA256 = "c" * 64
_TEST_PROVIDER_SHA256 = {
    symbol: ("e" * 64 if index < 2 else None)
    for index, symbol in enumerate(mc2_module._MC2_OPAPI_SYMBOLS)
}
_TEST_NATIVE_EPILOGUE_PROVIDER_SHA256 = {
    symbol: ("f" * 64 if index < 2 else None)
    for index, symbol in enumerate(mc2_module._MC2_NATIVE_EPILOGUE_OPAPI_SYMBOLS)
}
_ACTIVE_VENDOR_PAYLOAD_SHA256 = mc2_module.active_mc2_vendor_payload_sha256


@pytest.fixture(autouse=True)
def _reset_mc2_dispatch_audit(monkeypatch):
    reset_mc2_dispatch_counters()
    mc2_module._resolve_active_mc2_package_cached.cache_clear()
    mc2_module._resolve_mc2_symbol_vendor.cache_clear()
    mc2_module._mc2_vendor_payload_sha256_for_path.cache_clear()
    monkeypatch.setenv("HCCL_DETERMINISTIC", "true")
    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", "AIV")
    monkeypatch.setattr(mc2_module, "_active_physical_device_id", lambda _device: "0")
    monkeypatch.setattr(
        mc2_module,
        "active_mc2_vendor_payload_sha256",
        lambda operator=None: _TEST_VENDOR_SHA256,
    )
    monkeypatch.setattr(
        mc2_module,
        "active_mc2_adapter_binary_sha256",
        lambda: _TEST_ADAPTER_SHA256,
    )
    monkeypatch.setattr(
        mc2_module,
        "active_mc2_opapi_symbol_provider_sha256",
        lambda operator=None: dict(
            _TEST_NATIVE_EPILOGUE_PROVIDER_SHA256
            if operator == MC2_TP3_NATIVE_EPILOGUE_OPERATOR
            else _TEST_PROVIDER_SHA256
        ),
    )
    yield
    reset_mc2_dispatch_counters()
    mc2_module._resolve_active_mc2_package_cached.cache_clear()
    mc2_module._resolve_mc2_symbol_vendor.cache_clear()
    mc2_module._mc2_vendor_payload_sha256_for_path.cache_clear()


def test_mc2_capability_is_explicit_on_cpu():
    capability = detect_mc2_capability("cpu", tp_size=3)
    assert not capability.available
    assert "Ascend" in capability.reason


@pytest.mark.parametrize("value", (None, "", "   ", "UNDEFINED", "unknown", "UNKNOWN", -1))
def test_mc2_weight_format_fails_closed_for_unknown_values(monkeypatch, value):
    import torch_npu

    monkeypatch.setattr(torch_npu, "get_npu_format", lambda _tensor: value)
    tensor = SimpleNamespace(device=SimpleNamespace(type="npu"))

    assert mc2_module._weight_format(tensor) == "unknown"


@pytest.mark.parametrize("error_type", (ImportError, ValueError))
def test_mc2_weight_format_fails_closed_for_format_lookup_errors(monkeypatch, error_type):
    import torch_npu

    def fail_lookup(_tensor):
        raise error_type("format lookup failed")

    monkeypatch.setattr(torch_npu, "get_npu_format", fail_lookup)
    tensor = SimpleNamespace(device=SimpleNamespace(type="npu"))

    assert mc2_module._weight_format(tensor) == "unknown"


def test_mc2_weight_format_preserves_known_format(monkeypatch):
    import torch_npu

    monkeypatch.setattr(torch_npu, "get_npu_format", lambda _tensor: "ND")
    tensor = SimpleNamespace(device=SimpleNamespace(type="npu"))

    assert mc2_module._weight_format(tensor) == "ND"


def test_tp3_full_epilogue_epsilon_is_explicit_and_fail_closed():
    repo_root = Path(__file__).resolve().parents[3]
    adapter = (
        repo_root
        / "csrc/mc2/matmul_allreduce_add_rmsnorm/matmul_allreduce_add_rmsnorm_torch_adpt.h"
    ).read_text(encoding="utf-8")
    tiling = (
        repo_root
        / "csrc/mc2/matmul_allreduce_add_rmsnorm/op_host/matmul_allreduce_add_rmsnorm_tiling.cpp"
    ).read_text(encoding="utf-8")

    assert "projection_only || epsilon_f == 1.0e-6F" in adapter
    assert "qualified only for epsilon=1e-6" in adapter
    assert "ppTilingData.projectionOnly ? epsilon : 1.0e-6F" in tiling


def test_mc2_capability_rejects_incomplete_active_vendor(monkeypatch):
    import vllm_ascend.custom_op_package as custom_op_package

    monkeypatch.setattr(mc2_module, "_MC2_LOAD_ATTEMPTED", True)
    monkeypatch.setattr(mc2_module, "_matmul_op", lambda: object())
    monkeypatch.setattr(
        custom_op_package,
        "resolve_active_matmul_allreduce_add_rmsnorm_package",
        lambda **_kwargs: SimpleNamespace(available=False, reason="missing MC2 kernel manifest"),
    )

    capability = detect_mc2_capability("npu:0", tp_size=3)

    assert not capability.available
    assert capability.reason == "missing MC2 kernel manifest"


def test_mc2_model_flags_are_target_only():
    profile = object()
    config = SimpleNamespace(
        enable_mc2=True,
        target_tp_size=3,
        mc2_profile=profile,
        enforce_eager=True,
    )
    draft_model_config = SimpleNamespace()
    target_model_config = SimpleNamespace()

    _configure_mc2_model_flags(config, draft_model_config, target_model_config)

    assert draft_model_config.pearl_enable_mc2 is False
    assert draft_model_config.pearl_mc2_profile is None
    assert draft_model_config.pearl_enforce_eager is True
    assert target_model_config.pearl_enable_mc2 is True
    assert target_model_config.pearl_mc2_profile is profile
    assert target_model_config.pearl_enforce_eager is True


def test_mc2_model_flags_bind_profile_to_target_epsilon_and_tp_size():
    profile = MC2Profile(
        metadata={"rms_norm_epsilon": 1e-6, "tensor_parallel_size": 3},
        entries={},
        latency_percentile=95.0,
        minimum_samples=3,
        minimum_speedup=1.02,
    )
    config = SimpleNamespace(enable_mc2=True, target_tp_size=3, mc2_profile=profile)

    with pytest.raises(ValueError, match="epsilon does not match"):
        _configure_mc2_model_flags(
            config,
            SimpleNamespace(),
            SimpleNamespace(rms_norm_eps=1e-5),
        )

    mismatched_tp = SimpleNamespace(
        enable_mc2=True,
        target_tp_size=2,
        mc2_profile=profile,
    )
    with pytest.raises(ValueError, match="tensor-parallel size"):
        _configure_mc2_model_flags(
            mismatched_tp,
            SimpleNamespace(),
            SimpleNamespace(rms_norm_eps=1e-6),
        )


@pytest.mark.parametrize(
    ("enable_mc2", "target_tp_size"),
    ((False, 3), (True, 1)),
)
def test_mc2_model_flags_disable_unqualified_targets(enable_mc2, target_tp_size):
    config = SimpleNamespace(
        enable_mc2=enable_mc2,
        target_tp_size=target_tp_size,
        mc2_profile=object(),
    )
    draft_model_config = SimpleNamespace()
    target_model_config = SimpleNamespace()

    _configure_mc2_model_flags(config, draft_model_config, target_model_config)

    assert draft_model_config.pearl_enable_mc2 is False
    assert draft_model_config.pearl_mc2_profile is None
    assert target_model_config.pearl_enable_mc2 is False
    assert target_model_config.pearl_mc2_profile is None


def test_mc2_requires_tensor_parallel_target():
    with pytest.raises(ValueError, match="target tensor parallel size greater than one"):
        NativePearlConfig(
            draft_model="draft",
            target_model="target",
            draft_tp_size=1,
            target_tp_size=1,
            gamma=4,
            max_model_len=128,
            max_tokens=16,
            enable_mc2=True,
        )


def test_mc2_config_rejects_legacy_runtime_profile():
    document = _profile_document()
    document["metadata"].pop("runtime_binding")

    with pytest.raises(ValueError, match="production-bound runtime"):
        NativePearlConfig(
            "draft",
            "target",
            1,
            3,
            4,
            512,
            32,
            enable_mc2=True,
            mc2_profile=document,
        )


def test_mc2_worker_admission_is_local_and_freezes_target_runtime(monkeypatch):
    import vllm_ascend.utils as utils

    profile = object()
    config = SimpleNamespace(enable_mc2=True, mc2_profile=profile)
    freeze = MagicMock(return_value=None)
    monkeypatch.setattr(utils, "enable_custom_op", MagicMock())
    monkeypatch.setattr(
        "vllm_ascend.spec_decode.pearl.native_engine.validate_mc2_runtime_binding",
        freeze,
    )
    all_reduce = MagicMock(side_effect=AssertionError("MC2 local validation cannot submit HCCL"))
    monkeypatch.setattr(dist, "all_reduce", all_reduce)

    _validate_mc2_worker_admission(
        config,
        is_draft=False,
        device=torch.device("cpu"),
        target_rank=2,
    )

    freeze.assert_called_once_with(profile, torch.device("cpu"), tp_rank_id=2)
    all_reduce.assert_not_called()


def test_mc2_worker_admission_reports_local_runtime_failure(monkeypatch):
    import vllm_ascend.utils as utils

    config = SimpleNamespace(enable_mc2=True, mc2_profile=object())
    monkeypatch.setattr(utils, "enable_custom_op", MagicMock())
    monkeypatch.setattr(
        "vllm_ascend.spec_decode.pearl.native_engine.validate_mc2_runtime_binding",
        lambda *_args, **_kwargs: "synthetic identity mismatch",
    )
    all_reduce = MagicMock(side_effect=AssertionError("MC2 local validation cannot submit HCCL"))
    monkeypatch.setattr(dist, "all_reduce", all_reduce)

    with pytest.raises(RuntimeError, match="synthetic identity mismatch"):
        _validate_mc2_worker_admission(
            config,
            is_draft=False,
            device=torch.device("cpu"),
            target_rank=0,
        )
    all_reduce.assert_not_called()


@pytest.mark.parametrize(
    ("balanced_ffn_shift", "light_rank"),
    ((128, -1), (0, 2)),
)
def test_mc2_rejects_nonuniform_tp3_attention_partitions(
    balanced_ffn_shift,
    light_rank,
):
    config = SimpleNamespace(enable_mc2=True, target_tp_size=3)
    qwen3_config = SimpleNamespace(num_key_value_heads=8)

    with pytest.raises(ValueError, match="uniform attention partitions"):
        _validate_mc2_target_attention_sharding(
            config,
            qwen3_config,
            balanced_ffn_shift=balanced_ffn_shift,
            light_rank=light_rank,
        )


def test_mc2_allows_uniform_tp3_attention_partitions():
    config = SimpleNamespace(enable_mc2=True, target_tp_size=3)
    model_config = SimpleNamespace(num_key_value_heads=9)

    _validate_mc2_target_attention_sharding(
        config,
        model_config,
        balanced_ffn_shift=128,
        light_rank=-1,
    )


def _down_mc2_test_mlp(*, partitions=(8576, 8576, 8576), bias=None):
    return SimpleNamespace(
        down_proj=SimpleNamespace(
            input_partition_sizes=partitions,
            weight=torch.empty(5, partitions[0]),
            bias=bias,
        ),
        gate_up_proj=SimpleNamespace(bias=bias),
    )


def test_down_mc2_configuration_is_default_off_and_strict(monkeypatch):
    context = NativeTPContext(group=None, rank=0, size=3, leader_rank=0)
    profile = object()
    monkeypatch.delenv("VLLM_ASCEND_PEARL_MC2_DOWN_PROJ", raising=False)
    assert not _enable_tp3_down_mc2(
        _down_mc2_test_mlp(),
        context,
        enable_mc2=True,
        profile=profile,
    )

    monkeypatch.setenv("VLLM_ASCEND_PEARL_MC2_DOWN_PROJ", "1")
    assert _enable_tp3_down_mc2(
        _down_mc2_test_mlp(),
        context,
        enable_mc2=True,
        profile=profile,
    )
    assert not _enable_tp3_down_mc2(
        _down_mc2_test_mlp(partitions=(8576, 8576, 8448)),
        context,
        enable_mc2=True,
        profile=profile,
    )
    assert not _enable_tp3_down_mc2(
        _down_mc2_test_mlp(bias=torch.empty(5)),
        context,
        enable_mc2=True,
        profile=profile,
    )
    assert not _enable_tp3_down_mc2(
        _down_mc2_test_mlp(),
        NativeTPContext(group=None, rank=0, size=4, leader_rank=0),
        enable_mc2=True,
        profile=profile,
    )
    assert not _enable_tp3_down_mc2(
        _down_mc2_test_mlp(),
        context,
        enable_mc2=False,
        profile=profile,
    )
    assert not _enable_tp3_down_mc2(
        _down_mc2_test_mlp(),
        context,
        enable_mc2=True,
        profile=None,
    )


def test_down_mc2_consumes_qualified_next_norm_without_split_projection(monkeypatch):
    class FixedGateUp(nn.Module):
        def forward(self, hidden_states):
            return torch.cat((hidden_states, hidden_states), dim=-1)

    class RecordingDown(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(5, 3))
            self.bias = None
            self.calls = 0

        def forward(self, hidden_states):
            self.calls += 1
            return torch.nn.functional.linear(hidden_states, self.weight)

    mlp = NativeQwen2MLP.__new__(NativeQwen2MLP)
    nn.Module.__init__(mlp)
    mlp.gate_up_proj = FixedGateUp()
    mlp.down_proj = RecordingDown()
    next_norm = NativeRMSNorm(5, 1e-6)
    profile = object()
    context = NativeTPContext(group=None, rank=0, size=3, leader_rank=0)
    monkeypatch.setattr(
        next_norm,
        "qualify_forward_mc2",
        lambda *_args, **_kwargs: SimpleNamespace(
            qualification=MC2Qualification(True, "qualified")
        ),
    )
    monkeypatch.setattr(
        next_norm,
        "forward_mc2",
        lambda _activation, residual, *_args, **_kwargs: (
            torch.full_like(residual, 7.0),
            torch.full_like(residual, 11.0),
        ),
    )

    output, residual, normalized = mlp.forward_down_mc2(
        torch.randn(2, 3),
        torch.randn(2, 5),
        next_norm,
        context,
        profile=profile,
    )

    assert normalized
    assert torch.equal(output, torch.full_like(output, 7.0))
    assert torch.equal(residual, torch.full_like(residual, 11.0))
    assert mlp.down_proj.calls == 0


def test_down_mc2_unqualified_shape_stays_on_split_projection(monkeypatch):
    class FixedGateUp(nn.Module):
        def forward(self, hidden_states):
            return torch.cat((hidden_states, hidden_states), dim=-1)

    class RecordingDown(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(5, 3))
            self.bias = None
            self.calls = 0

        def forward(self, hidden_states):
            self.calls += 1
            return torch.nn.functional.linear(hidden_states, self.weight)

    mlp = NativeQwen2MLP.__new__(NativeQwen2MLP)
    nn.Module.__init__(mlp)
    mlp.gate_up_proj = FixedGateUp()
    mlp.down_proj = RecordingDown()
    next_norm = NativeRMSNorm(5, 1e-6)
    monkeypatch.setattr(
        next_norm,
        "qualify_forward_mc2",
        lambda *_args, **_kwargs: SimpleNamespace(
            qualification=MC2Qualification(False, "unqualified")
        ),
    )
    residual = torch.randn(2, 5)

    output, next_residual, normalized = mlp.forward_down_mc2(
        torch.randn(2, 3),
        residual,
        next_norm,
        NativeTPContext(group=None, rank=0, size=3, leader_rank=0),
        profile=object(),
    )

    assert not normalized
    assert output.shape == residual.shape
    assert next_residual is residual
    assert mlp.down_proj.calls == 1


def test_decoder_skips_input_norm_for_cross_layer_mc2_state():
    class FailingInputNorm(nn.Module):
        def forward(self, *_args):
            raise AssertionError("next input norm was already consumed")

    class RegularAttention(nn.Module):
        def forward(self, _positions, hidden_states, _metadata):
            return hidden_states

    layer = NativeQwen2DecoderLayer.__new__(NativeQwen2DecoderLayer)
    nn.Module.__init__(layer)
    layer.self_attn = RegularAttention()
    layer.input_layernorm = FailingInputNorm()
    layer.post_attention_layernorm = NativeRMSNorm(5, 1e-6)
    layer.mlp = nn.Identity()
    layer.enable_mc2 = False
    layer.enable_down_mc2 = False
    layer.mc2_input_capture = None
    hidden_states = torch.randn(2, 5)
    residual = torch.randn(2, 5)

    output, next_residual, normalized = layer(
        torch.arange(2),
        hidden_states,
        residual,
        input_is_normalized=True,
        return_normalization_state=True,
    )

    assert output.shape == hidden_states.shape
    assert next_residual.shape == residual.shape
    assert not normalized


def test_model_skips_consumed_next_and_final_norms():
    class FailingFinalNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))

        def forward(self, *_args):
            raise AssertionError("final norm was already consumed by down MC2")

    class CrossLayer(nn.Module):
        def __init__(self, expected_input_state):
            super().__init__()
            self.input_layernorm = NativeRMSNorm(2, 1e-6)
            self.expected_input_state = expected_input_state

        def forward(
            self,
            _positions,
            hidden_states,
            residual,
            _attention_metadata,
            *,
            next_layernorm_gamma,
            next_layernorm,
            input_is_normalized,
            return_normalization_state,
        ):
            assert input_is_normalized is self.expected_input_state
            assert next_layernorm.weight is next_layernorm_gamma
            assert return_normalization_state
            return hidden_states + 1, torch.zeros_like(hidden_states), True

    model = NativeQwen2ForCausalLM.__new__(NativeQwen2ForCausalLM)
    nn.Module.__init__(model)
    model.embed_tokens = nn.Identity()
    model.layers = nn.ModuleList((CrossLayer(False), CrossLayer(True)))
    model.norm = FailingFinalNorm()
    model.track_cache_finiteness = False

    output = model(torch.ones(1, 2), torch.zeros(1, dtype=torch.long))

    assert torch.equal(output, torch.full((1, 2), 3.0))


class _MC2TestProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(5, 3))
        self.bias = None
        self.calls = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.nn.functional.linear(hidden_states, self.weight)


class _MC2TestAttention(nn.Module):
    def __init__(self, local_attended: torch.Tensor) -> None:
        super().__init__()
        self.o_proj = _MC2TestProjection()
        self.local_attended = local_attended

    def forward(
        self,
        _positions,
        _hidden_states,
        _attention_metadata,
        *,
        return_pre_projection: bool = False,
    ) -> torch.Tensor:
        assert return_pre_projection
        return self.local_attended


class _MC2TestInputNorm(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden_states, residual


class _MC2TestPostNorm(NativeRMSNorm):
    def __init__(self) -> None:
        super().__init__(hidden_size=5, eps=1e-6)
        self.normal_calls = 0
        self.mc2_calls = 0

    def forward(self, *args, **kwargs):
        self.normal_calls += 1
        return super().forward(*args, **kwargs)

    def forward_mc2(
        self,
        _local_hidden_states,
        residual,
        _projection_weight,
        _context,
        **_kwargs,
    ):
        self.mc2_calls += 1
        return torch.full_like(residual, 7.0), torch.full_like(residual, 11.0)


def _mc2_test_decoder(profile) -> tuple[NativeQwen2DecoderLayer, _MC2TestPostNorm]:
    layer = NativeQwen2DecoderLayer.__new__(NativeQwen2DecoderLayer)
    nn.Module.__init__(layer)
    local_attended = torch.randn(2, 3)
    layer.self_attn = _MC2TestAttention(local_attended)
    layer.input_layernorm = _MC2TestInputNorm()
    post_norm = _MC2TestPostNorm()
    layer.post_attention_layernorm = post_norm
    layer.mlp = nn.Identity()
    layer.context = NativeTPContext(group=None, rank=0, size=3, leader_rank=0)
    layer.enable_mc2 = True
    layer.mc2_profile = profile
    return layer, post_norm


@pytest.mark.parametrize(
    ("capability_available", "shape_qualified"),
    ((True, False), (False, True)),
)
def test_unqualified_or_unavailable_mc2_uses_original_projection_and_norm(
    monkeypatch,
    capability_available,
    shape_qualified,
):
    profile = SimpleNamespace(
        qualify=MagicMock(
            return_value=MC2Qualification(
                shape_qualified,
                "test qualification",
            )
        )
    )
    layer, post_norm = _mc2_test_decoder(profile)
    monkeypatch.setattr(
        native_model_module,
        "detect_mc2_capability",
        lambda _device, tp_size: MC2Capability(
            capability_available,
            "cpu",
            tp_size,
            "matmul_allreduce_add_rmsnorm",
            "test capability",
        ),
    )
    residual = torch.randn(2, 5)

    output, output_residual = layer(
        torch.arange(2),
        torch.randn(2, 5),
        residual,
    )

    assert output.shape == residual.shape
    assert output_residual.shape == residual.shape
    assert layer.self_attn.o_proj.calls == 1
    assert post_norm.normal_calls == 1
    assert post_norm.mc2_calls == 0
    assert profile.qualify.call_count == int(capability_available)


def test_exact_qualified_mc2_bypasses_original_projection_and_norm(monkeypatch):
    profile = SimpleNamespace(
        qualify=MagicMock(
            return_value=MC2Qualification(
                True,
                "exact shape passed numerical and repeated p95 gates",
                1.0,
                0.8,
            )
        )
    )
    layer, post_norm = _mc2_test_decoder(profile)
    monkeypatch.setattr(
        native_model_module,
        "detect_mc2_capability",
        lambda _device, tp_size: MC2Capability(
            True,
            "cpu",
            tp_size,
            "matmul_allreduce_add_rmsnorm",
            "test capability",
        ),
    )
    residual = torch.randn(2, 5)

    output, output_residual = layer(
        torch.arange(2),
        torch.randn(2, 5),
        residual,
    )

    assert torch.equal(output, torch.full_like(residual, 7.0))
    assert torch.equal(output_residual, torch.full_like(residual, 11.0))
    assert layer.self_attn.o_proj.calls == 0
    assert post_norm.normal_calls == 0
    assert post_norm.mc2_calls == 1
    profile.qualify.assert_called_once()


def test_decoder_static_route_skips_dynamic_qualification_and_comm_resolution(monkeypatch):
    profile = normalize_mc2_profile(_profile_document())
    layer, post_norm = _mc2_test_decoder(profile)
    layer.self_attn.local_attended = layer.self_attn.local_attended.to(torch.bfloat16)
    layer.self_attn.o_proj.to(torch.bfloat16)
    post_norm.to(torch.bfloat16)
    layer.mc2_attention_route = build_mc2_static_route(
        profile,
        projection_kind="attention",
        layer_index=0,
        group_tp="pre-resolved-comm",
        tp_rank_size=3,
        tp_rank_id=0,
        epsilon=1e-6,
        weight=layer.self_attn.o_proj.weight,
        gamma=post_norm.weight,
    )
    monkeypatch.setattr(
        MC2Profile,
        "qualify",
        MagicMock(side_effect=AssertionError("dynamic qualification must not run")),
    )
    monkeypatch.setattr(
        native_model_module,
        "resolve_hccl_comm_name",
        MagicMock(side_effect=AssertionError("communicator must already be resolved")),
    )
    residual = torch.randn(2, 5, dtype=torch.bfloat16)

    output, output_residual = layer(
        torch.arange(2),
        torch.randn(2, 5, dtype=torch.bfloat16),
        residual,
    )

    assert torch.equal(output, torch.full_like(residual, 7.0))
    assert torch.equal(output_residual, torch.full_like(residual, 11.0))
    assert layer.self_attn.o_proj.calls == 0
    assert post_norm.mc2_calls == 1


def test_frozen_decoder_route_missing_fails_without_dynamic_qualification(monkeypatch):
    profile = SimpleNamespace(
        qualify=MagicMock(side_effect=AssertionError("dynamic qualification is forbidden"))
    )
    layer, _post_norm = _mc2_test_decoder(profile)
    layer.mc2_routes_frozen = True
    layer.mc2_attention_route = None
    layer.mc2_down_route = object()
    resolve = MagicMock(side_effect=AssertionError("dynamic communicator resolution is forbidden"))
    monkeypatch.setattr(native_model_module, "resolve_hccl_comm_name", resolve)

    with pytest.raises(RuntimeError, match="missing a static route"):
        layer(
            torch.arange(2),
            torch.randn(2, 5),
            torch.randn(2, 5),
        )

    profile.qualify.assert_not_called()
    resolve.assert_not_called()


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


def test_qualified_native_mc2_rejects_missing_hccl_communicator(monkeypatch):
    norm = NativeRMSNorm(5, 1e-6)
    monkeypatch.setattr(native_model_module, "resolve_hccl_comm_name", lambda *_args, **_kwargs: "")
    adapter = MagicMock()
    monkeypatch.setattr(native_model_module, "matmul_allreduce_add_rmsnorm_or_fallback", adapter)

    with pytest.raises(RuntimeError, match="could not resolve.*HCCL communicator"):
        norm.forward_mc2(
            torch.randn(2, 3),
            torch.randn(2, 5),
            torch.randn(5, 3),
            NativeTPContext(group=None, rank=0, size=3, leader_rank=0),
            profile=object(),
        )

    adapter.assert_not_called()


def _qualified_profile(operator="matmul_allreduce_add_rmsnorm"):
    return SimpleNamespace(
        metadata={"operator": operator},
        qualify=lambda *_args, **_kwargs: MC2Qualification(
            True,
            "exact shape passed numerical and repeated p95 gates",
            1.0,
            0.8,
        ),
    )


def _install_fake_fused(monkeypatch, operator):
    monkeypatch.setattr(
        mc2_module,
        "detect_mc2_capability",
        lambda _device, tp_size: MC2Capability(
            True,
            "npu:0",
            tp_size,
            "matmul_allreduce_add_rmsnorm",
            "test operator",
        ),
    )
    monkeypatch.setattr(mc2_module, "_matmul_op", lambda: operator)


def _dispatch_inputs(rows):
    return (
        torch.randn(rows, 3),
        torch.randn(5, 3),
        torch.randn(rows, 5),
        torch.ones(5),
    )


def test_mc2_dispatch_counters_snapshot_and_reset():
    x, weight, residual, gamma = _dispatch_inputs(4)
    matmul_allreduce_add_rmsnorm_or_fallback(
        x,
        weight,
        residual,
        gamma,
        tp_rank_size=1,
        use_fused=False,
    )

    snapshot = snapshot_mc2_dispatch_counters()
    assert snapshot == {
        "fused_attempt": 0,
        "fused_success": 0,
        "fallback": 1,
        "exception": 0,
        "full_fused_attempt": 0,
        "full_fused_success": 0,
        "native_epilogue_attempt": 0,
        "native_epilogue_success": 0,
        "native_epilogue_chained_attempt": 0,
        "native_epilogue_chained_success": 0,
        "native_epilogue_chain_flush": 0,
    }
    snapshot["fallback"] = 99
    assert snapshot_mc2_dispatch_counters()["fallback"] == 1

    reset_mc2_dispatch_counters()
    assert snapshot_mc2_dispatch_counters() == {
        "fused_attempt": 0,
        "fused_success": 0,
        "fallback": 0,
        "exception": 0,
        "full_fused_attempt": 0,
        "full_fused_success": 0,
        "native_epilogue_attempt": 0,
        "native_epilogue_success": 0,
        "native_epilogue_chained_attempt": 0,
        "native_epilogue_chained_success": 0,
        "native_epilogue_chain_flush": 0,
    }


def test_mc2_dispatch_counters_record_fused_success(monkeypatch):
    x, weight, residual, gamma = _dispatch_inputs(5)
    expected_output = torch.full_like(residual, 3.0)
    expected_added = torch.full_like(residual, 2.0)

    def fused_operator(*_args):
        return expected_output, expected_added

    _install_fake_fused(monkeypatch, fused_operator)
    output, added = matmul_allreduce_add_rmsnorm_or_fallback(
        x,
        weight,
        residual,
        gamma,
        tp_rank_size=3,
        use_fused=True,
        profile=_qualified_profile(),
    )

    assert output is expected_output
    assert added is expected_added
    assert snapshot_mc2_dispatch_counters() == {
        "fused_attempt": 1,
        "fused_success": 1,
        "fallback": 0,
        "exception": 0,
        "full_fused_attempt": 1,
        "full_fused_success": 1,
        "native_epilogue_attempt": 0,
        "native_epilogue_success": 0,
        "native_epilogue_chained_attempt": 0,
        "native_epilogue_chained_success": 0,
        "native_epilogue_chain_flush": 0,
    }


def test_mc2_prequalification_skips_duplicate_capability_and_profile_checks(monkeypatch):
    x, weight, residual, gamma = _dispatch_inputs(5)
    expected_output = torch.full_like(residual, 3.0)
    expected_added = torch.full_like(residual, 2.0)
    profile = _qualified_profile()
    profile.qualify = MagicMock(side_effect=AssertionError("duplicate profile qualification"))
    monkeypatch.setattr(
        mc2_module,
        "detect_mc2_capability",
        MagicMock(side_effect=AssertionError("duplicate capability detection")),
    )
    monkeypatch.setattr(mc2_module, "_matmul_op", lambda: lambda *_args: (expected_output, expected_added))
    ticket = mc2_module._bind_mc2_dispatch_ticket(
        MC2Qualification(
            True,
            "already qualified by NativeRMSNorm",
            1.0,
            0.8,
        ),
        x,
        weight,
        residual,
        gamma,
        tp_rank_size=3,
        tp_rank_id=0,
        epsilon=1e-6,
        is_trans_b=True,
        profile=profile,
    )

    output, added = matmul_allreduce_add_rmsnorm_or_fallback(
        x,
        weight,
        residual,
        gamma,
        tp_rank_size=3,
        use_fused=True,
        strict_fused=True,
        profile=profile,
        prequalification=ticket,
    )

    assert output is expected_output
    assert added is expected_added
    profile.qualify.assert_not_called()


def test_v126_profile_operator_is_normalized_and_frozen_into_route():
    profile = normalize_mc2_profile(_profile_document(MC2_TP3_NATIVE_EPILOGUE_OPERATOR))
    route = build_mc2_static_route(
        profile,
        projection_kind="attention",
        layer_index=0,
        group_tp="test-comm",
        tp_rank_size=3,
        tp_rank_id=0,
        epsilon=1e-6,
        weight=torch.empty(5, 3, dtype=torch.bfloat16),
        gamma=torch.ones(5, dtype=torch.bfloat16),
    )

    assert route.operator_kind == MC2_TP3_NATIVE_EPILOGUE_OPERATOR
    assert route.consensus_document()["operator_kind"] == MC2_TP3_NATIVE_EPILOGUE_OPERATOR

    legacy_profile = normalize_mc2_profile(_profile_document())
    legacy_route = build_mc2_static_route(
        legacy_profile,
        projection_kind="attention",
        layer_index=0,
        group_tp="test-comm",
        tp_rank_size=3,
        tp_rank_id=0,
        epsilon=1e-6,
        weight=torch.empty(5, 3, dtype=torch.bfloat16),
        gamma=torch.ones(5, dtype=torch.bfloat16),
    )
    assert build_mc2_static_route_manifest(profile, (route,)).digest != build_mc2_static_route_manifest(
        legacy_profile, (legacy_route,)
    ).digest


@pytest.mark.parametrize(
    ("key", "tp_size", "epsilon", "message"),
    (
        ((0, 3072, 5120, "bfloat16", "ND", True), 3, 1e-6, "1 <= M <= 160"),
        ((161, 3072, 5120, "bfloat16", "ND", True), 3, 1e-6, "1 <= M <= 160"),
        ((64, 3072, 5121, "bfloat16", "ND", True), 3, 1e-6, "hidden size 5120"),
        ((64, 3072, 5120, "float16", "ND", True), 3, 1e-6, "bfloat16"),
        ((64, 3072, 5120, "bfloat16", "ND", False), 3, 1e-6, "F.linear"),
        ((64, 3072, 5120, "bfloat16", "ND", True), 2, 1e-6, "parallel size 3"),
        ((64, 3072, 5120, "bfloat16", "ND", True), 3, 1e-5, "epsilon=1e-6"),
    ),
)
def test_v126_static_shape_contract_rejects_out_of_envelope(key, tp_size, epsilon, message):
    assert message in mc2_module._native_epilogue_static_shape_error(
        key,
        tp_size=tp_size,
        epsilon=epsilon,
    )


@pytest.mark.parametrize("rows", (1, 64, 160))
@pytest.mark.parametrize("input_size", (3072, 8576))
def test_v126_static_shape_contract_accepts_qwen3_tp3(rows, input_size):
    assert (
        mc2_module._native_epilogue_static_shape_error(
            (rows, input_size, 5120, "bfloat16", "ND", True),
            tp_size=3,
            epsilon=1e-6,
        )
        is None
    )


def test_v126_dispatch_runs_native_linear_then_epilogue_once(monkeypatch):
    x, weight, residual, gamma = _dispatch_inputs(5)
    profile = _qualified_profile(MC2_TP3_NATIVE_EPILOGUE_OPERATOR)
    ticket = mc2_module._bind_mc2_dispatch_ticket(
        MC2Qualification(True, "qualified", 1.0, 0.8),
        x,
        weight,
        residual,
        gamma,
        tp_rank_size=3,
        tp_rank_id=0,
        epsilon=1e-6,
        is_trans_b=True,
        profile=profile,
    )
    order = []
    original_linear = torch.nn.functional.linear

    def linear(*args, **kwargs):
        order.append("linear")
        return original_linear(*args, **kwargs)

    expected_output = torch.full_like(residual, 3.0)
    expected_added = torch.full_like(residual, 2.0)

    def epilogue(local_projection, *args):
        order.append("epilogue")
        assert torch.equal(local_projection, original_linear(x, weight))
        assert args[0] is residual
        assert args[1] is gamma
        assert args[2:] == ("test-comm", 3, 0, 1e-6, True)
        return expected_output, expected_added

    monkeypatch.setattr(mc2_module, "_native_epilogue_dispatch_error", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mc2_module.F, "linear", linear)
    monkeypatch.setattr(mc2_module, "_native_epilogue_op", lambda: epilogue)
    monkeypatch.setattr(
        mc2_module,
        "_matmul_op",
        lambda: (_ for _ in ()).throw(AssertionError("legacy op must not be selected")),
    )

    output, added = matmul_allreduce_add_rmsnorm_or_fallback(
        x,
        weight,
        residual,
        gamma,
        group_tp="test-comm",
        tp_rank_size=3,
        tp_rank_id=0,
        epsilon=1e-6,
        is_gather_add_out=True,
        use_fused=True,
        strict_fused=True,
        profile=profile,
        prequalification=ticket,
    )

    assert output is expected_output
    assert added is expected_added
    assert order == ["linear", "epilogue"]
    counters = snapshot_mc2_dispatch_counters()
    assert counters["fused_attempt"] == counters["native_epilogue_attempt"] == 1
    assert counters["fused_success"] == counters["native_epilogue_success"] == 1
    assert counters["full_fused_attempt"] == counters["fallback"] == 0


def test_v131_chained_dispatch_carries_explicit_state_without_fallback(monkeypatch):
    x, weight, residual, gamma = _dispatch_inputs(5)
    profile = _qualified_profile(MC2_TP3_NATIVE_EPILOGUE_OPERATOR)
    ticket = mc2_module._bind_mc2_dispatch_ticket(
        MC2Qualification(True, "qualified", 1.0, 0.8),
        x,
        weight,
        residual,
        gamma,
        tp_rank_size=3,
        tp_rank_id=0,
        epsilon=1e-6,
        is_trans_b=True,
        profile=profile,
    )
    chain_state = torch.zeros((64, 4), dtype=torch.int64)
    expected_output = torch.full_like(residual, 3.0)
    expected_added = torch.full_like(residual, 2.0)
    expected_state = torch.ones_like(chain_state)

    def chained_epilogue(local_projection, *args):
        assert torch.equal(local_projection, torch.nn.functional.linear(x, weight))
        assert args[0] is residual
        assert args[1] is gamma
        assert args[2] is chain_state
        assert args[3:] == ("test-comm", 3, 0, 1e-6, True, False)
        return expected_output, expected_added, expected_state

    monkeypatch.setattr(mc2_module, "_native_epilogue_dispatch_error", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mc2_module, "_native_epilogue_chained_op", lambda: chained_epilogue)
    monkeypatch.setattr(
        mc2_module,
        "_native_epilogue_op",
        lambda: (_ for _ in ()).throw(AssertionError("standalone epilogue must not be selected")),
    )

    output, added, next_state = matmul_allreduce_add_rmsnorm_or_fallback(
        x,
        weight,
        residual,
        gamma,
        group_tp="test-comm",
        tp_rank_size=3,
        tp_rank_id=0,
        epsilon=1e-6,
        is_gather_add_out=True,
        use_fused=True,
        strict_fused=True,
        profile=profile,
        prequalification=ticket,
        chain_state=chain_state,
        flush_chain=False,
    )

    assert output is expected_output
    assert added is expected_added
    assert next_state is expected_state
    counters = snapshot_mc2_dispatch_counters()
    assert counters["native_epilogue_chained_attempt"] == 1
    assert counters["native_epilogue_chained_success"] == 1
    assert counters["native_epilogue_chain_flush"] == 0
    assert counters["fallback"] == counters["exception"] == 0


def test_mc2_model_chain_requires_order_and_flushes_only_final_down():
    chain = native_model_module._MC2IntraLayerChain(
        torch.zeros((64, 4), dtype=torch.int64),
        layer_count=2,
    )
    attention = SimpleNamespace(layer_index=0, projection_kind="attention")
    down = SimpleNamespace(layer_index=0, projection_kind="down")
    next_attention = SimpleNamespace(layer_index=1, projection_kind="attention")
    next_down = SimpleNamespace(layer_index=1, projection_kind="down")

    state, flush = chain.begin(attention)
    assert not flush
    attention_state = torch.ones_like(state)
    chain.commit(attention, attention_state)
    assert chain.state is attention_state

    state, flush = chain.begin(down)
    assert state is attention_state
    assert not flush
    down_state = torch.full_like(state, 2)
    chain.commit(down, down_state)

    state, flush = chain.begin(next_attention)
    assert state is down_state
    assert not flush
    next_attention_state = torch.full_like(state, 3)
    chain.commit(next_attention, next_attention_state)

    state, flush = chain.begin(next_down)
    assert state is next_attention_state
    assert flush
    final_state = torch.zeros_like(state)
    chain.commit(next_down, final_state)
    chain.finish()
    assert chain.state is final_state


def test_mc2_chained_flush_mode_flushes_every_epilogue():
    chain = native_model_module._MC2IntraLayerChain(
        torch.zeros((64, 4), dtype=torch.int64),
        layer_count=1,
        defer_read_done=False,
    )
    attention = SimpleNamespace(layer_index=0, projection_kind="attention")
    down = SimpleNamespace(layer_index=0, projection_kind="down")

    state, flush = chain.begin(attention)
    assert flush
    chain.commit(attention, torch.zeros_like(state))
    state, flush = chain.begin(down)
    assert flush
    chain.commit(down, torch.zeros_like(state))
    chain.finish()


def test_v126_strict_epilogue_failure_never_uses_split_fallback(monkeypatch):
    x, weight, residual, gamma = _dispatch_inputs(5)
    profile = _qualified_profile(MC2_TP3_NATIVE_EPILOGUE_OPERATOR)
    ticket = mc2_module._bind_mc2_dispatch_ticket(
        MC2Qualification(True, "qualified", 1.0, 0.8),
        x,
        weight,
        residual,
        gamma,
        tp_rank_size=3,
        tp_rank_id=0,
        epsilon=1e-6,
        is_trans_b=True,
        profile=profile,
    )

    def fail(*_args):
        raise RuntimeError("v126 failure")

    monkeypatch.setattr(mc2_module, "_native_epilogue_dispatch_error", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mc2_module, "_native_epilogue_op", lambda: fail)
    monkeypatch.setattr(mc2_module.dist, "all_reduce", MagicMock(side_effect=AssertionError("split fallback")))

    with pytest.raises(RuntimeError, match="failed in strict mode"):
        matmul_allreduce_add_rmsnorm_or_fallback(
            x,
            weight,
            residual,
            gamma,
            group_tp="test-comm",
            tp_rank_size=3,
            tp_rank_id=0,
            epsilon=1e-6,
            is_gather_add_out=True,
            use_fused=True,
            strict_fused=True,
            profile=profile,
            prequalification=ticket,
        )

    counters = snapshot_mc2_dispatch_counters()
    assert counters["fused_attempt"] == counters["native_epilogue_attempt"] == 1
    assert counters["fused_success"] == counters["native_epilogue_success"] == 0
    assert counters["exception"] == 1
    assert counters["fallback"] == 0


def test_mc2_prequalification_rejects_cross_tensor_reuse(monkeypatch):
    x, weight, residual, gamma = _dispatch_inputs(5)
    profile = _qualified_profile()
    ticket = mc2_module._bind_mc2_dispatch_ticket(
        MC2Qualification(True, "qualified", 1.0, 0.8),
        x,
        weight,
        residual,
        gamma,
        tp_rank_size=3,
        tp_rank_id=0,
        epsilon=1e-6,
        is_trans_b=True,
        profile=profile,
    )
    monkeypatch.setattr(mc2_module, "_matmul_op", lambda: lambda *_args: (residual, residual))

    with pytest.raises(RuntimeError, match="ticket does not match"):
        matmul_allreduce_add_rmsnorm_or_fallback(
            x.clone(),
            weight,
            residual,
            gamma,
            tp_rank_size=3,
            use_fused=True,
            strict_fused=True,
            profile=profile,
            prequalification=ticket,
        )


def test_mc2_fused_exception_logs_real_reason_and_falls_back(monkeypatch, caplog):
    x, weight, residual, gamma = _dispatch_inputs(7)

    def failing_operator(*_args):
        raise RuntimeError("synthetic fused invocation failure")

    _install_fake_fused(monkeypatch, failing_operator)
    with caplog.at_level(logging.WARNING, logger=mc2_module.__name__):
        # Populate the ordinary per-shape fallback log guard first. A later
        # operator exception for the same shape must still expose its actual
        # reason rather than being hidden by that earlier qualification log.
        matmul_allreduce_add_rmsnorm_or_fallback(
            x,
            weight,
            residual,
            gamma,
            tp_rank_size=3,
            use_fused=True,
            profile=None,
        )
        caplog.clear()
        reset_mc2_dispatch_counters()
        output, added = matmul_allreduce_add_rmsnorm_or_fallback(
            x,
            weight,
            residual,
            gamma,
            tp_rank_size=3,
            use_fused=True,
            profile=_qualified_profile(),
        )

    assert output.shape == residual.shape
    assert added.shape == residual.shape
    fallback_records = [record.getMessage() for record in caplog.records if "fallback" in record.getMessage()]
    assert len(fallback_records) == 1
    assert "RuntimeError: synthetic fused invocation failure" in fallback_records[0]
    assert "exact shape passed" not in fallback_records[0]
    assert snapshot_mc2_dispatch_counters() == {
        "fused_attempt": 1,
        "fused_success": 0,
        "fallback": 1,
        "exception": 1,
        "full_fused_attempt": 1,
        "full_fused_success": 0,
        "native_epilogue_attempt": 0,
        "native_epilogue_success": 0,
        "native_epilogue_chained_attempt": 0,
        "native_epilogue_chained_success": 0,
        "native_epilogue_chain_flush": 0,
    }


def test_mc2_fused_exception_is_fail_closed_in_strict_mode(monkeypatch):
    x, weight, residual, gamma = _dispatch_inputs(11)

    def failing_operator(*_args):
        raise RuntimeError("strict fused invocation failure")

    _install_fake_fused(monkeypatch, failing_operator)
    with pytest.raises(RuntimeError, match="failed in strict mode") as raised:
        matmul_allreduce_add_rmsnorm_or_fallback(
            x,
            weight,
            residual,
            gamma,
            tp_rank_size=3,
            use_fused=True,
            strict_fused=True,
            profile=_qualified_profile(),
        )

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "strict fused invocation failure" in str(raised.value.__cause__)
    assert snapshot_mc2_dispatch_counters() == {
        "fused_attempt": 1,
        "fused_success": 0,
        "fallback": 0,
        "exception": 1,
        "full_fused_attempt": 1,
        "full_fused_success": 0,
        "native_epilogue_attempt": 0,
        "native_epilogue_success": 0,
        "native_epilogue_chained_attempt": 0,
        "native_epilogue_chained_success": 0,
        "native_epilogue_chain_flush": 0,
    }


def _profile_document(operator="matmul_allreduce_add_rmsnorm"):
    cann_version = mc2_module._installed_component_version("cann")
    hccl_version = mc2_module._installed_component_version("hccl")
    assert cann_version is not None
    assert hccl_version is not None
    provider_symbols = mc2_module._mc2_opapi_symbols(operator)
    provider_sha256 = {
        symbol: ("e" * 64 if index < 2 else None)
        for index, symbol in enumerate(provider_symbols)
    }
    return {
        "schema_version": 1,
        "metadata": {
            "operator": operator,
            "hardware": "Ascend 910B3",
            "tensor_parallel_size": 3,
            "source_sha256": mc2_source_sha256(),
            "rms_norm_epsilon": 1e-6,
            "runtime_binding": {
                "hccl_deterministic": "true",
                "hccl_op_expansion_mode": "AIV",
                "reduction_mode": "global_01",
                "cann_version": cann_version,
                "hccl_version": hccl_version,
                "vendor_payload_sha256": _TEST_VENDOR_SHA256,
                "opapi_symbol_provider_sha256": provider_sha256,
                "adapter_binary_sha256": _TEST_ADAPTER_SHA256,
                "tp_rank_device_mapping": ["0", "1", "2"],
            },
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
                "activation_layout": "contiguous",
                "weight_layout": "contiguous",
                "residual_layout": "contiguous",
                "gamma_layout": "contiguous",
                "changed_input_replays": 8,
                "changed_input_delta": 0.015625,
                "baseline_latency_ms": [1.0, 1.0, 1.0],
                "fused_latency_ms": [0.8, 0.8, 0.8],
                "max_abs_norm": 0.01,
                "norm_atol": 0.05,
                "max_abs_added": 0.001,
                "added_atol": 0.01,
            }
        ],
    }


def test_mc2_static_route_precomputes_profile_gate_and_binds_exact_parameters():
    profile = normalize_mc2_profile(_profile_document())
    weight = torch.empty(5, 3, dtype=torch.bfloat16)
    gamma = torch.ones(5, dtype=torch.bfloat16)
    route = build_mc2_static_route(
        profile,
        projection_kind="attention",
        layer_index=0,
        group_tp="pre-resolved-comm",
        tp_rank_size=3,
        tp_rank_id=0,
        epsilon=1e-6,
        weight=weight,
        gamma=gamma,
    )

    assert route.decisions[2].qualified
    ticket = bind_mc2_static_route(
        route,
        profile,
        torch.empty(2, 3, dtype=torch.bfloat16),
        weight,
        torch.empty(2, 5, dtype=torch.bfloat16),
        gamma,
    )
    assert ticket.qualification.qualified

    with pytest.raises(RuntimeError, match="parameter ownership"):
        bind_mc2_static_route(
            route,
            profile,
            torch.empty(2, 3, dtype=torch.bfloat16),
            weight.clone(),
            torch.empty(2, 5, dtype=torch.bfloat16),
            gamma,
        )


def test_mc2_static_route_rejects_noncontiguous_activation_layout():
    profile = normalize_mc2_profile(_profile_document())
    weight = torch.empty(5, 3, dtype=torch.bfloat16)
    gamma = torch.ones(5, dtype=torch.bfloat16)
    route = build_mc2_static_route(
        profile,
        projection_kind="attention",
        layer_index=0,
        group_tp="pre-resolved-comm",
        tp_rank_size=3,
        tp_rank_id=0,
        epsilon=1e-6,
        weight=weight,
        gamma=gamma,
    )
    activation = torch.empty(3, 2, dtype=torch.bfloat16).t()
    assert not activation.is_contiguous()

    with pytest.raises(RuntimeError, match="requires contiguous"):
        bind_mc2_static_route(
            route,
            profile,
            activation,
            weight,
            torch.empty(2, 5, dtype=torch.bfloat16),
            gamma,
        )

    result = profile.qualify(
        activation,
        weight,
        torch.empty(2, 5, dtype=torch.bfloat16),
        tp_size=3,
        is_trans_b=True,
    )
    assert not result.qualified
    assert "contiguous" in result.reason


def test_mc2_static_route_unprofiled_rows_fail_closed_without_requalification():
    profile = normalize_mc2_profile(_profile_document())
    weight = torch.empty(5, 3, dtype=torch.bfloat16)
    gamma = torch.ones(5, dtype=torch.bfloat16)
    route = build_mc2_static_route(
        profile,
        projection_kind="down",
        layer_index=4,
        group_tp="pre-resolved-comm",
        tp_rank_size=3,
        tp_rank_id=1,
        epsilon=1e-6,
        weight=weight,
        gamma=gamma,
    )

    ticket = bind_mc2_static_route(
        route,
        profile,
        torch.empty(7, 3, dtype=torch.bfloat16),
        weight,
        torch.empty(7, 5, dtype=torch.bfloat16),
        gamma,
    )

    assert not ticket.qualification.qualified
    assert ticket.qualification.reason == "unprofiled MC2 static row count 7"


def test_mc2_static_route_manifest_digest_is_rank_independent():
    profile = normalize_mc2_profile(_profile_document())
    weight = torch.empty(5, 3, dtype=torch.bfloat16)
    gamma = torch.ones(5, dtype=torch.bfloat16)

    def manifest_for_rank(rank):
        route = build_mc2_static_route(
            profile,
            projection_kind="attention",
            layer_index=0,
            group_tp=f"rank-{rank}-local-handle",
            tp_rank_size=3,
            tp_rank_id=rank,
            epsilon=1e-6,
            weight=weight,
            gamma=gamma,
        )
        return build_mc2_static_route_manifest(profile, (route,))

    assert manifest_for_rank(0).digest == manifest_for_rank(2).digest


def test_mc2_disabled_static_route_digest_allows_rank_local_split_shards():
    profile = normalize_mc2_profile(_profile_document())
    gamma = torch.ones(5, dtype=torch.bfloat16)

    def manifest_for_width(width):
        route = build_mc2_static_route(
            profile,
            projection_kind="down",
            layer_index=0,
            group_tp="rank-local-handle",
            tp_rank_size=3,
            tp_rank_id=0,
            epsilon=1e-6,
            weight=torch.empty(5, width, dtype=torch.bfloat16),
            gamma=gamma,
            enabled=False,
            disabled_reason="TP3 down-projection MC2 is disabled by model policy",
        )
        return build_mc2_static_route_manifest(profile, (route,))

    assert manifest_for_width(3).digest == manifest_for_width(4).digest


def _fill_mc2_world_consensus(
    outputs,
    local_payload,
    *,
    target_ranks=(1, 2, 3),
    target_digest=None,
):
    digest = target_digest or local_payload.get("digest") or ("ab" * 32)
    for rank, output in enumerate(outputs):
        output_payload = dict(local_payload)
        output_payload["rank"] = rank
        output_payload["role"] = "target" if rank in target_ranks else "draft"
        output_payload["error"] = None
        output_payload["digest"] = digest if rank in target_ranks else None
        output_payload["target_global_ranks"] = tuple(target_ranks)
        output.clear()
        output.update(output_payload)


def test_mc2_worker_route_consensus_uses_exactly_one_world_gloo_collective(monkeypatch):
    manifest = mc2_module.MC2StaticRouteManifest((), "02" * 32, "01" * 32)
    context = NativeTPContext(group=object(), rank=0, size=3, leader_rank=1)
    coordination_group = object()
    collectives = []
    events = []

    def validate_runtime(*_args, **_kwargs):
        events.append("runtime")

    def freeze_routes():
        events.append("freeze")
        return manifest

    model = SimpleNamespace(freeze_mc2_static_routes=MagicMock(side_effect=freeze_routes))
    monkeypatch.setattr(
        "vllm_ascend.spec_decode.pearl.native_engine._validate_mc2_worker_admission",
        validate_runtime,
    )

    def all_gather_object(outputs, value, **kwargs):
        events.append("gather")
        collectives.append(("all_gather_object", kwargs.get("group")))
        for index in range(len(outputs)):
            outputs[index] = {}
        _fill_mc2_world_consensus(outputs, value)

    monkeypatch.setattr(dist, "all_reduce", MagicMock(side_effect=AssertionError("HCCL vote is forbidden")))
    monkeypatch.setattr(dist, "all_gather", MagicMock(side_effect=AssertionError("HCCL gather is forbidden")))
    monkeypatch.setattr(dist, "all_gather_object", all_gather_object)

    result = _freeze_mc2_worker_routes(
        SimpleNamespace(enable_mc2=True),
        is_draft=False,
        model=model,
        context=context,
        device=torch.device("cpu"),
        coordination_group=coordination_group,
        target_group=context.group,
        global_rank=1,
        world_size=4,
        target_global_ranks=(1, 2, 3),
    )

    assert result is manifest
    assert events == ["runtime", "freeze", "gather"]
    assert collectives == [("all_gather_object", coordination_group)]


def test_mc2_worker_route_consensus_rejects_target_digest_mismatch(monkeypatch):
    manifest = mc2_module.MC2StaticRouteManifest((), "02" * 32, "01" * 32)
    model = SimpleNamespace(freeze_mc2_static_routes=MagicMock(return_value=manifest))
    context = NativeTPContext(group=object(), rank=0, size=3, leader_rank=1)
    monkeypatch.setattr(
        "vllm_ascend.spec_decode.pearl.native_engine._validate_mc2_worker_admission",
        lambda *_args, **_kwargs: None,
    )

    def mismatch(outputs, value, **_kwargs):
        for index in range(len(outputs)):
            outputs[index] = {}
        _fill_mc2_world_consensus(outputs, value)
        outputs[-1]["digest"] = "ff" * 32

    monkeypatch.setattr(dist, "all_gather_object", mismatch)

    with pytest.raises(RuntimeError, match="digest differs"):
        _freeze_mc2_worker_routes(
            SimpleNamespace(enable_mc2=True),
            is_draft=False,
            model=model,
            context=context,
            device=torch.device("cpu"),
            coordination_group=object(),
            target_group=context.group,
            global_rank=1,
            world_size=4,
            target_global_ranks=(1, 2, 3),
        )


def test_mc2_worker_route_consensus_rejects_target_group_identity_mismatch(monkeypatch):
    manifest = mc2_module.MC2StaticRouteManifest((), "02" * 32, "01" * 32)
    model = SimpleNamespace(freeze_mc2_static_routes=MagicMock(return_value=manifest))
    context = NativeTPContext(group=object(), rank=0, size=3, leader_rank=1)
    monkeypatch.setattr(
        "vllm_ascend.spec_decode.pearl.native_engine._validate_mc2_worker_admission",
        lambda *_args, **_kwargs: None,
    )

    def mismatch(outputs, value, **_kwargs):
        for index in range(len(outputs)):
            outputs[index] = {}
        _fill_mc2_world_consensus(outputs, value)
        outputs[-1]["target_global_ranks"] = (1, 3, 2)

    monkeypatch.setattr(dist, "all_gather_object", mismatch)

    with pytest.raises(RuntimeError, match="global-rank tuple differs"):
        _freeze_mc2_worker_routes(
            SimpleNamespace(enable_mc2=True),
            is_draft=False,
            model=model,
            context=context,
            device=torch.device("cpu"),
            coordination_group=object(),
            target_group=context.group,
            global_rank=1,
            world_size=4,
            target_global_ranks=(1, 2, 3),
        )


def test_mc2_worker_route_consensus_propagates_local_construction_error(monkeypatch):
    model = SimpleNamespace(
        freeze_mc2_static_routes=MagicMock(side_effect=RuntimeError("synthetic route failure"))
    )
    context = NativeTPContext(group=object(), rank=0, size=3, leader_rank=1)
    calls = 0
    monkeypatch.setattr(
        "vllm_ascend.spec_decode.pearl.native_engine._validate_mc2_worker_admission",
        lambda *_args, **_kwargs: None,
    )

    def gather_with_error(outputs, value, **_kwargs):
        nonlocal calls
        calls += 1
        for index in range(len(outputs)):
            outputs[index] = {}
        _fill_mc2_world_consensus(outputs, value)
        outputs[1]["error"] = value["error"]
        outputs[1]["digest"] = None

    monkeypatch.setattr(dist, "all_gather_object", gather_with_error)

    with pytest.raises(RuntimeError, match="rank 1: RuntimeError: synthetic route failure"):
        _freeze_mc2_worker_routes(
            SimpleNamespace(enable_mc2=True),
            is_draft=False,
            model=model,
            context=context,
            device=torch.device("cpu"),
            coordination_group=object(),
            target_group=context.group,
            global_rank=1,
            world_size=4,
            target_global_ranks=(1, 2, 3),
        )
    assert calls == 1


def test_mc2_world_consensus_includes_nonleader_draft_rank(monkeypatch):
    context = NativeTPContext(group=object(), rank=1, size=2, leader_rank=0)
    calls = []

    def gather(outputs, value, **kwargs):
        calls.append(kwargs.get("group"))
        for index in range(len(outputs)):
            outputs[index] = {}
        _fill_mc2_world_consensus(
            outputs,
            value,
            target_ranks=(2, 3, 4),
        )

    coordination_group = object()
    monkeypatch.setattr(dist, "all_gather_object", gather)

    result = _freeze_mc2_worker_routes(
        SimpleNamespace(enable_mc2=True),
        is_draft=True,
        model=SimpleNamespace(
            freeze_mc2_static_routes=MagicMock(side_effect=AssertionError("draft must not freeze routes"))
        ),
        context=context,
        device=torch.device("cpu"),
        coordination_group=coordination_group,
        target_group=object(),
        global_rank=1,
        world_size=5,
        target_global_ranks=(2, 3, 4),
    )

    assert result is None
    assert calls == [coordination_group]


def _make_mc2_route_freeze_model(profile):
    o_proj = SimpleNamespace(
        weight=torch.empty(5, 3, dtype=torch.bfloat16),
        bias=None,
        pre_resolved_comm_name=None,
    )
    down_proj = SimpleNamespace(
        weight=torch.empty(5, 3, dtype=torch.bfloat16),
        bias=None,
        pre_resolved_comm_name=None,
        large_m_nz_weight=None,
        large_m_nz_min_rows=0,
    )
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(o_proj=o_proj),
        post_attention_layernorm=SimpleNamespace(
            eps=1e-6,
            weight=torch.ones(5, dtype=torch.bfloat16),
        ),
        input_layernorm=SimpleNamespace(
            eps=1e-6,
            weight=torch.ones(5, dtype=torch.bfloat16),
        ),
        mlp=SimpleNamespace(down_proj=down_proj),
        enable_down_mc2=True,
        mc2_attention_route=None,
        mc2_down_route=None,
        mc2_routes_frozen=False,
    )
    model = SimpleNamespace(
        _mc2_static_route_manifest=None,
        _mc2_static_routes_frozen=False,
        config=SimpleNamespace(
            pearl_enable_mc2=True,
            pearl_mc2_profile=profile,
            rms_norm_eps=1e-6,
        ),
        context=NativeTPContext(group=object(), rank=0, size=3, leader_rank=1),
        embed_tokens=SimpleNamespace(weight=torch.empty(2, 5, dtype=torch.bfloat16)),
        layers=[layer],
        norm=SimpleNamespace(
            eps=1e-6,
            weight=torch.ones(5, dtype=torch.bfloat16),
        ),
    )
    return model, layer, o_proj, down_proj


def test_model_freezes_all_layer_routes_with_one_communicator_lookup(monkeypatch):
    profile = normalize_mc2_profile(_profile_document())
    o_proj = SimpleNamespace(
        weight=torch.empty(5, 3, dtype=torch.bfloat16),
        bias=None,
        pre_resolved_comm_name=None,
    )
    down_proj = SimpleNamespace(
        weight=torch.empty(5, 3, dtype=torch.bfloat16),
        bias=None,
        pre_resolved_comm_name=None,
    )
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(o_proj=o_proj),
        post_attention_layernorm=SimpleNamespace(
            eps=1e-6,
            weight=torch.ones(5, dtype=torch.bfloat16),
        ),
        input_layernorm=SimpleNamespace(
            eps=1e-6,
            weight=torch.ones(5, dtype=torch.bfloat16),
        ),
        mlp=SimpleNamespace(down_proj=down_proj),
        enable_down_mc2=True,
        mc2_attention_route=None,
        mc2_down_route=None,
    )
    model = SimpleNamespace(
        _mc2_static_route_manifest=None,
        config=SimpleNamespace(
            pearl_enable_mc2=True,
            pearl_mc2_profile=profile,
            rms_norm_eps=1e-6,
        ),
        context=NativeTPContext(group=object(), rank=0, size=3, leader_rank=1),
        embed_tokens=SimpleNamespace(weight=torch.empty(2, 5, dtype=torch.bfloat16)),
        layers=[layer],
        norm=SimpleNamespace(
            eps=1e-6,
            weight=torch.ones(5, dtype=torch.bfloat16),
        ),
    )
    resolve = MagicMock(return_value="one-pre-resolved-communicator")
    monkeypatch.setattr(
        native_model_module,
        "detect_mc2_capability",
        lambda device, size: MC2Capability(True, str(device), size, "mc2", "available"),
    )
    monkeypatch.setattr(native_model_module, "validate_mc2_static_environment", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(native_model_module, "resolve_hccl_comm_name", resolve)

    manifest = NativeQwen2ForCausalLM.freeze_mc2_static_routes(model)

    assert len(manifest.routes) == 2
    assert layer.mc2_attention_route is manifest.routes[0]
    assert layer.mc2_down_route is manifest.routes[1]
    assert o_proj.pre_resolved_comm_name == "one-pre-resolved-communicator"
    assert down_proj.pre_resolved_comm_name == "one-pre-resolved-communicator"
    assert layer.mc2_routes_frozen
    resolve.assert_called_once()


def test_explicit_mc2_admission_rejects_zero_qualified_production_rows(monkeypatch):
    document = _profile_document()
    document["measurements"][0]["k"] = 4
    profile = normalize_mc2_profile(document)
    model, layer, o_proj, down_proj = _make_mc2_route_freeze_model(profile)
    monkeypatch.setattr(
        native_model_module,
        "detect_mc2_capability",
        lambda device, size: MC2Capability(True, str(device), size, "mc2", "available"),
    )
    monkeypatch.setattr(native_model_module, "validate_mc2_static_environment", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        native_model_module,
        "resolve_hccl_comm_name",
        lambda *_args, **_kwargs: "one-pre-resolved-communicator",
    )

    with pytest.raises(RuntimeError, match="zero qualified production row"):
        NativeQwen2ForCausalLM.freeze_mc2_static_routes(model)

    assert layer.mc2_attention_route is None
    assert layer.mc2_down_route is None
    assert not layer.mc2_routes_frozen
    assert o_proj.pre_resolved_comm_name is None
    assert down_proj.pre_resolved_comm_name is None


@pytest.mark.parametrize(
    ("with_chain_evidence", "graph_execution"),
    ((False, True), (True, False), (True, True)),
)
def test_model_chains_only_graph_qualified_deferred_epilogues(
    with_chain_evidence,
    graph_execution,
):
    document = _profile_document(MC2_TP3_NATIVE_EPILOGUE_OPERATOR)
    if with_chain_evidence:
        document["metadata"]["deferred_read_done_chain"] = {
            "qualified": True,
            "execution_mode": "graph",
            "chain_scope": "whole_model",
            "layer_count": 2,
            "operations_per_chain": 4,
        }
    profile = normalize_mc2_profile(document)
    qualification = MC2Qualification(True, "qualified", 1.0, 0.8)
    routes = tuple(
        SimpleNamespace(enabled=True, decisions={2: qualification}) for _ in range(4)
    )

    rows = native_model_module._qualified_mc2_chain_row_counts(
        profile,
        routes,
        layer_count=2,
        graph_execution=graph_execution,
    )

    assert rows == ({2} if with_chain_evidence and graph_execution else frozenset())


@pytest.mark.parametrize("graph_execution", (False, True))
def test_model_uses_chained_flush_only_when_both_shapes_were_measured(
    graph_execution,
):
    document = _profile_document(MC2_TP3_NATIVE_EPILOGUE_OPERATOR)
    document["metadata"]["standalone_chained_flush"] = True
    profile = normalize_mc2_profile(document)
    qualification = MC2Qualification(True, "qualified", 1.0, 0.8)
    routes = tuple(
        SimpleNamespace(enabled=True, decisions={2: qualification}) for _ in range(4)
    )

    rows = native_model_module._qualified_mc2_chained_flush_row_counts(
        profile,
        routes,
        layer_count=2,
        graph_execution=graph_execution,
    )

    assert rows == ({2} if graph_execution else frozenset())


def test_mc2_profile_requires_current_source_identity():
    profile = normalize_mc2_profile(_profile_document())
    assert isinstance(profile, MC2Profile)
    renormalized = normalize_mc2_profile(profile)
    assert renormalized is not profile
    assert renormalized.metadata == profile.metadata
    with pytest.raises(TypeError):
        profile.metadata["hardware"] = "mutated"
    restored = pickle.loads(pickle.dumps(profile))
    assert restored == profile
    with pytest.raises(TypeError):
        restored.entries[next(iter(restored.entries))]["m"] = 999
    tampered = _profile_document()
    tampered["metadata"]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source hash"):
        normalize_mc2_profile(tampered)


def test_mc2_source_identity_includes_model_runner_chain_wiring():
    source = Path(mc2_module.__file__).read_text(encoding="utf-8")

    assert 'adapter.with_name("native_model.py")' in source


def test_mc2_profile_requires_explicit_contiguous_layout_binding():
    document = _profile_document()
    document["measurements"][0].pop("activation_layout")

    with pytest.raises(ValueError, match="contiguous.*layouts"):
        normalize_mc2_profile(document)


def test_legacy_mc2_profile_loads_but_cannot_qualify(monkeypatch):
    document = _profile_document()
    document["metadata"].pop("runtime_binding")
    profile = normalize_mc2_profile(document)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_device_name=lambda _index: "Ascend 910B3"))
    x = torch.empty(2, 3, dtype=torch.bfloat16)
    weight = torch.empty(5, 3, dtype=torch.bfloat16)
    residual = torch.empty(2, 5, dtype=torch.bfloat16)

    result = profile.qualify(x, weight, residual, tp_size=3, is_trans_b=True)

    assert not result.qualified
    assert "legacy" in result.reason


def test_legacy_mc2_profile_without_epsilon_loads_but_cannot_qualify(monkeypatch):
    document = _profile_document()
    document["metadata"].pop("rms_norm_epsilon")
    profile = normalize_mc2_profile(document)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_device_name=lambda _index: "Ascend 910B3"))
    x = torch.empty(2, 3, dtype=torch.bfloat16)
    weight = torch.empty(5, 3, dtype=torch.bfloat16)
    residual = torch.empty(2, 5, dtype=torch.bfloat16)

    result = profile.qualify(x, weight, residual, tp_size=3, is_trans_b=True)

    assert not result.qualified
    assert "epsilon binding" in result.reason


def test_mc2_profile_fails_closed_when_model_epsilon_changes(monkeypatch):
    profile = normalize_mc2_profile(_profile_document())
    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_device_name=lambda _index: "Ascend 910B3"))
    x = torch.empty(2, 3, dtype=torch.bfloat16)
    weight = torch.empty(5, 3, dtype=torch.bfloat16)
    residual = torch.empty(2, 5, dtype=torch.bfloat16)

    result = profile.qualify(
        x,
        weight,
        residual,
        tp_size=3,
        is_trans_b=True,
        epsilon=1e-5,
    )

    assert not result.qualified
    assert "epsilon does not match" in result.reason


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("changed_input_replays", None),
        ("changed_input_replays", 1),
        ("changed_input_delta", None),
        ("changed_input_delta", 0.0),
    ),
)
def test_mc2_profile_requires_changed_input_graph_qualification(monkeypatch, field, value):
    document = _profile_document()
    if value is None:
        document["measurements"][0].pop(field)
    else:
        document["measurements"][0][field] = value
    profile = normalize_mc2_profile(document)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_device_name=lambda _index: "Ascend 910B3"))
    x = torch.empty(2, 3, dtype=torch.bfloat16)
    weight = torch.empty(5, 3, dtype=torch.bfloat16)
    residual = torch.empty(2, 5, dtype=torch.bfloat16)

    result = profile.qualify(x, weight, residual, tp_size=3, is_trans_b=True)

    assert not result.qualified
    assert "changed-input ACLGraph qualification" in result.reason


@pytest.mark.parametrize("epsilon", (True, 0.0, -1e-6, float("inf"), float("nan"), "1e-6"))
def test_mc2_profile_rejects_invalid_epsilon_binding(epsilon):
    document = _profile_document()
    document["metadata"]["rms_norm_epsilon"] = epsilon

    with pytest.raises(ValueError, match="rms_norm_epsilon"):
        normalize_mc2_profile(document)


def test_mc2_profile_rejects_partial_runtime_binding():
    document = _profile_document()
    document["metadata"]["runtime_binding"].pop("reduction_mode")

    with pytest.raises(ValueError, match="missing reduction_mode"):
        normalize_mc2_profile(document)


def test_mc2_runtime_binding_builder_refuses_unqualified_environment(monkeypatch):
    monkeypatch.setenv("HCCL_DETERMINISTIC", "false")

    with pytest.raises(RuntimeError, match="HCCL_DETERMINISTIC=true"):
        mc2_module.current_mc2_runtime_binding(tp_rank_device_mapping=("4", "5", "6"))


def test_mc2_runtime_binding_builder_records_exact_runtime_identity():
    binding = mc2_module.current_mc2_runtime_binding(tp_rank_device_mapping=(4, 5, 6))

    assert binding["hccl_deterministic"] == "true"
    assert binding["hccl_op_expansion_mode"] == "AIV"
    assert binding["reduction_mode"] == "global_01"
    assert binding["cann_version"]
    assert binding["hccl_version"]
    assert binding["vendor_payload_sha256"] == _TEST_VENDOR_SHA256
    assert binding["opapi_symbol_provider_sha256"] == _TEST_PROVIDER_SHA256
    assert binding["adapter_binary_sha256"] == _TEST_ADAPTER_SHA256
    assert binding["tp_rank_device_mapping"] == ["4", "5", "6"]


def test_native_epilogue_runtime_binding_uses_operator_specific_provider_identity():
    binding = mc2_module.current_mc2_runtime_binding(
        tp_rank_device_mapping=(4, 5, 6),
        operator=MC2_TP3_NATIVE_EPILOGUE_OPERATOR,
    )

    assert (
        binding["opapi_symbol_provider_sha256"]
        == _TEST_NATIVE_EPILOGUE_PROVIDER_SHA256
    )


def test_mc2_profile_fails_closed_when_vendor_payload_changes(monkeypatch):
    profile = normalize_mc2_profile(_profile_document())
    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_device_name=lambda _index: "Ascend 910B3"))
    monkeypatch.setattr(mc2_module, "active_mc2_vendor_payload_sha256", lambda: "b" * 64)
    x = torch.empty(2, 3, dtype=torch.bfloat16)
    weight = torch.empty(5, 3, dtype=torch.bfloat16)
    residual = torch.empty(2, 5, dtype=torch.bfloat16)

    result = profile.qualify(x, weight, residual, tp_size=3, is_trans_b=True)

    assert not result.qualified
    assert "vendor payload" in result.reason


def test_mc2_profile_fails_closed_when_adapter_binary_changes(monkeypatch):
    profile = normalize_mc2_profile(_profile_document())
    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_device_name=lambda _index: "Ascend 910B3"))
    monkeypatch.setattr(mc2_module, "active_mc2_adapter_binary_sha256", lambda: "d" * 64)
    x = torch.empty(2, 3, dtype=torch.bfloat16)
    weight = torch.empty(5, 3, dtype=torch.bfloat16)
    residual = torch.empty(2, 5, dtype=torch.bfloat16)

    result = profile.qualify(x, weight, residual, tp_size=3, is_trans_b=True)

    assert not result.qualified
    assert "adapter binary" in result.reason


def test_mc2_profile_fails_closed_when_opapi_symbol_provider_changes(monkeypatch):
    profile = normalize_mc2_profile(_profile_document())
    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_device_name=lambda _index: "Ascend 910B3"))
    changed = dict(_TEST_PROVIDER_SHA256)
    changed[mc2_module._MC2_REQUIRED_OPAPI_SYMBOLS[0]] = "f" * 64
    monkeypatch.setattr(
        mc2_module,
        "active_mc2_opapi_symbol_provider_sha256",
        lambda: changed,
    )
    x = torch.empty(2, 3, dtype=torch.bfloat16)
    weight = torch.empty(5, 3, dtype=torch.bfloat16)
    residual = torch.empty(2, 5, dtype=torch.bfloat16)

    result = profile.qualify(x, weight, residual, tp_size=3, is_trans_b=True)

    assert not result.qualified
    assert "OPAPI symbol providers" in result.reason


def test_mc2_profile_rejects_incomplete_opapi_symbol_provider_identity():
    document = _profile_document()
    providers = document["metadata"]["runtime_binding"]["opapi_symbol_provider_sha256"]
    providers.pop(mc2_module._MC2_REQUIRED_OPAPI_SYMBOLS[0])

    with pytest.raises(ValueError, match="provider identity is incomplete"):
        normalize_mc2_profile(document)


@pytest.mark.parametrize("value", (None, "A" * 64, "0" * 63, 1))
def test_mc2_profile_rejects_invalid_vendor_payload_identity(value):
    document = _profile_document()
    if value is None:
        document["metadata"]["runtime_binding"].pop("vendor_payload_sha256")
        message = "missing vendor_payload_sha256"
    else:
        document["metadata"]["runtime_binding"]["vendor_payload_sha256"] = value
        message = "vendor_payload_sha256"

    with pytest.raises(ValueError, match=message):
        normalize_mc2_profile(document)


@pytest.mark.parametrize("value", (None, "A" * 64, "0" * 63, 1))
def test_mc2_profile_rejects_invalid_adapter_binary_identity(value):
    document = _profile_document()
    if value is None:
        document["metadata"]["runtime_binding"].pop("adapter_binary_sha256")
        message = "missing adapter_binary_sha256"
    else:
        document["metadata"]["runtime_binding"]["adapter_binary_sha256"] = value
        message = "adapter_binary_sha256"

    with pytest.raises(ValueError, match=message):
        normalize_mc2_profile(document)


def test_vendor_payload_digest_is_path_independent_and_content_bound(tmp_path):
    def make_vendor(root: Path, object_bytes: bytes) -> Path:
        files = {
            "op_api/lib/libcust_opapi.so": b"opapi",
            "op_api/include/aclnnop/aclnn_matmul_allreduce_add_rmsnorm.h": b"header",
            (
                "op_impl/ai_core/tbe/custom_transformer_impl/ascendc/"
                "matmul_allreduce_add_rmsnorm/matmul_allreduce_add_rmsnorm_aiv_kernel.h"
            ): b"source",
            (
                "op_impl/ai_core/tbe/kernel/ascend910b/matmul_allreduce_add_rmsnorm/"
                "kernel.o"
            ): object_bytes,
            (
                "op_impl/ai_core/tbe/kernel/config/ascend910b/"
                "matmul_allreduce_add_rmsnorm.json"
            ): b"manifest",
        }
        for relative, contents in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
        return root

    first = make_vendor(tmp_path / "first", b"kernel-a")
    second = make_vendor(tmp_path / "second", b"kernel-a")
    changed = make_vendor(tmp_path / "changed", b"kernel-b")

    digest_first = mc2_module._mc2_vendor_payload_sha256_for_path(str(first))
    assert digest_first == mc2_module._mc2_vendor_payload_sha256_for_path(str(second))
    assert digest_first != mc2_module._mc2_vendor_payload_sha256_for_path(str(changed))


def test_native_epilogue_active_payload_accepts_explicit_nondefault_vendor_name(
    monkeypatch,
    tmp_path,
):
    vendor = tmp_path / "specslo_mc2_v126_aiv_epilogue_transformer"
    files = {
        "op_api/lib/libcust_opapi.so": b"opapi-v126",
        "op_api/include/aclnnop/aclnn_allreduce_add_rmsnorm.h": b"header-v126",
        (
            "op_impl/ai_core/tbe/specslo_mc2_v126_aiv_epilogue_transformer_impl/"
            "ascendc/allreduce_add_rmsnorm/allreduce_add_rmsnorm_kernel.h"
        ): b"source-v126",
        (
            "op_impl/ai_core/tbe/kernel/ascend910b/allreduce_add_rmsnorm/"
            "kernel.o"
        ): b"kernel-v126",
        (
            "op_impl/ai_core/tbe/kernel/config/ascend910b/"
            "allreduce_add_rmsnorm.json"
        ): b"manifest-v126",
    }
    for relative, contents in files.items():
        path = vendor / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    library = str((vendor / "op_api/lib/libcust_opapi.so").resolve())
    providers = tuple(
        (symbol, library if symbol in mc2_module._MC2_NATIVE_EPILOGUE_REQUIRED_OPAPI_SYMBOLS else "")
        for symbol in mc2_module._MC2_NATIVE_EPILOGUE_OPAPI_SYMBOLS
    )
    monkeypatch.setattr(
        mc2_module,
        "_freeze_and_get_mc2_opapi_symbol_providers",
        lambda operator=None: providers,
    )

    active_digest = _ACTIVE_VENDOR_PAYLOAD_SHA256(
        MC2_TP3_NATIVE_EPILOGUE_OPERATOR
    )
    direct_digest = mc2_module._mc2_vendor_payload_sha256_for_path(
        str(vendor),
        MC2_TP3_NATIVE_EPILOGUE_OPERATOR,
    )

    assert active_digest == direct_digest


def test_native_epilogue_provider_query_rejects_mixed_required_dsos(
    monkeypatch,
    tmp_path,
):
    first = tmp_path / "first.so"
    second = tmp_path / "second.so"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    workspace_symbol, execute_symbol = (
        mc2_module._MC2_NATIVE_EPILOGUE_REQUIRED_OPAPI_SYMBOLS
    )
    rows = [
        f"{workspace_symbol}={first}",
        f"{execute_symbol}={second}",
        *(f"{symbol}=" for symbol in mc2_module._MC2_OPTIONAL_OPAPI_SYMBOLS),
    ]
    monkeypatch.setattr(
        torch.ops._C_ascend,
        "freeze_and_get_allreduce_add_rmsnorm_opapi_symbol_providers",
        lambda: rows,
        raising=False,
    )
    mc2_module._freeze_and_get_mc2_opapi_symbol_providers.cache_clear()

    with pytest.raises(RuntimeError, match="mixed DSOs"):
        mc2_module._freeze_and_get_mc2_opapi_symbol_providers(
            MC2_TP3_NATIVE_EPILOGUE_OPERATOR
        )


def test_mc2_symbol_resolver_skips_incomplete_vendor_and_matches_cpp_path_semantics(
    monkeypatch,
    tmp_path,
):
    first = tmp_path / "first" / "custom_transformer"
    second = tmp_path / "second" / "custom_transformer"
    libraries = {}
    for vendor in (first, second):
        library = vendor / "op_api/lib/libcust_opapi.so"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"placeholder")
        libraries[str(library.resolve())] = vendor

    class FakeHandle:
        def __init__(self, path, **_kwargs):
            self.path = str(Path(path).resolve())

        def __getattr__(self, symbol):
            # The first vendor is structurally present but exports no MC2
            # symbols. Both symbols come from the second vendor.
            if libraries[self.path] == second and symbol in mc2_module._MC2_OPAPI_SYMBOLS:
                return object()
            raise AttributeError(symbol)

    monkeypatch.setattr(mc2_module.ctypes, "CDLL", FakeHandle)
    search_path = f"{first}{os.pathsep}{second}"

    assert mc2_module._resolve_mc2_symbol_vendor(search_path) == str(second.resolve())


def test_mc2_symbol_resolver_rejects_mixed_opapi_dsos(monkeypatch, tmp_path):
    first = tmp_path / "first" / "custom_transformer"
    second = tmp_path / "second" / "custom_transformer"
    for vendor in (first, second):
        library = vendor / "op_api/lib/libcust_opapi.so"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"placeholder")

    workspace_symbol, execute_symbol = mc2_module._MC2_REQUIRED_OPAPI_SYMBOLS

    class FakeHandle:
        def __init__(self, path, **_kwargs):
            self.vendor = Path(path).resolve().parents[2]

        def __getattr__(self, symbol):
            if (self.vendor == first and symbol == workspace_symbol) or (
                self.vendor == second and symbol == execute_symbol
            ):
                return object()
            raise AttributeError(symbol)

    monkeypatch.setattr(mc2_module.ctypes, "CDLL", FakeHandle)
    search_path = f"{first}{os.pathsep}{second}"

    with pytest.raises(RuntimeError, match="mixed DSOs"):
        mc2_module._resolve_mc2_symbol_vendor(search_path)


@pytest.mark.parametrize(
    ("name", "value", "reason"),
    (
        ("HCCL_DETERMINISTIC", "false", "HCCL_DETERMINISTIC=true"),
        ("HCCL_OP_EXPANSION_MODE", "", "HCCL_OP_EXPANSION_MODE=AIV"),
    ),
)
def test_mc2_profile_fails_closed_when_collective_environment_changes(monkeypatch, name, value, reason):
    profile = normalize_mc2_profile(_profile_document())
    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_device_name=lambda _index: "Ascend 910B3"))
    monkeypatch.setenv(name, value)
    x = torch.empty(2, 3, dtype=torch.bfloat16)
    weight = torch.empty(5, 3, dtype=torch.bfloat16)
    residual = torch.empty(2, 5, dtype=torch.bfloat16)

    result = profile.qualify(x, weight, residual, tp_size=3, is_trans_b=True)

    assert not result.qualified
    assert reason in result.reason


@pytest.mark.parametrize("component", ("cann", "hccl"))
def test_mc2_profile_fails_closed_when_runtime_version_changes(monkeypatch, component):
    profile = normalize_mc2_profile(_profile_document())
    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_device_name=lambda _index: "Ascend 910B3"))
    original = mc2_module._installed_component_version
    monkeypatch.setattr(
        mc2_module,
        "_installed_component_version",
        lambda name: "changed-version" if name == component else original(name),
    )
    x = torch.empty(2, 3, dtype=torch.bfloat16)
    weight = torch.empty(5, 3, dtype=torch.bfloat16)
    residual = torch.empty(2, 5, dtype=torch.bfloat16)

    result = profile.qualify(x, weight, residual, tp_size=3, is_trans_b=True)

    assert not result.qualified
    assert f"{component.upper()} version" in result.reason


def test_mc2_profile_binds_tp_rank_to_physical_device(monkeypatch):
    document = _profile_document()
    document["metadata"]["runtime_binding"]["tp_rank_device_mapping"] = ["4", "5", "6"]
    profile = normalize_mc2_profile(document)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_device_name=lambda _index: "Ascend 910B3"))
    monkeypatch.setattr(mc2_module, "_active_physical_device_id", lambda _device: "5")
    x = torch.empty(2, 3, dtype=torch.bfloat16)
    weight = torch.empty(5, 3, dtype=torch.bfloat16)
    residual = torch.empty(2, 5, dtype=torch.bfloat16)

    assert profile.qualify(
        x,
        weight,
        residual,
        tp_size=3,
        tp_rank_id=1,
        is_trans_b=True,
    ).qualified
    mismatch = profile.qualify(
        x,
        weight,
        residual,
        tp_size=3,
        tp_rank_id=0,
        is_trans_b=True,
    )
    assert not mismatch.qualified
    assert "rank-to-device mapping" in mismatch.reason


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


def test_mc2_native_epilogue_diagnostic_override_bypasses_only_latency(monkeypatch):
    document = _profile_document()
    document["metadata"]["operator"] = "tp3_matmul_allreduce+native_add_rmsnorm"
    document["measurements"][0]["fused_latency_ms"] = [1.5, 1.5, 1.5]
    profile = normalize_mc2_profile(document)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_device_name=lambda _index: "Ascend 910B3"))
    monkeypatch.setenv("VLLM_ASCEND_PEARL_MC2_DIAGNOSTIC_FORCE_NATIVE_EPILOGUE", "1")
    x = torch.empty(2, 3, dtype=torch.bfloat16)
    weight = torch.empty(5, 3, dtype=torch.bfloat16)
    residual = torch.empty(2, 5, dtype=torch.bfloat16)

    result = profile.qualify(x, weight, residual, tp_size=3, is_trans_b=True)

    assert result.qualified
    assert "diagnostic-only" in result.reason
    assert result.baseline_p95_ms == 1.0
    assert result.fused_p95_ms == 1.5
    assert not profile.qualify(x[:1], weight, residual[:1], tp_size=3, is_trans_b=True).qualified
    assert not profile.qualify(x, weight, residual, tp_size=2, is_trans_b=True).qualified


def test_mc2_native_epilogue_remains_fail_closed_by_default(monkeypatch):
    document = _profile_document()
    document["metadata"]["operator"] = "tp3_matmul_allreduce+native_add_rmsnorm"
    document["measurements"][0]["fused_latency_ms"] = [1.5, 1.5, 1.5]
    profile = normalize_mc2_profile(document)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_device_name=lambda _index: "Ascend 910B3"))
    monkeypatch.delenv("VLLM_ASCEND_PEARL_MC2_DIAGNOSTIC_FORCE_NATIVE_EPILOGUE", raising=False)
    x = torch.empty(2, 3, dtype=torch.bfloat16)
    weight = torch.empty(5, 3, dtype=torch.bfloat16)
    residual = torch.empty(2, 5, dtype=torch.bfloat16)

    result = profile.qualify(x, weight, residual, tp_size=3, is_trans_b=True)

    assert not result.qualified
    assert "not faster" in result.reason


def test_mc2_native_epilogue_diagnostic_override_keeps_numerical_gate(monkeypatch):
    document = _profile_document()
    document["metadata"]["operator"] = "tp3_matmul_allreduce+native_add_rmsnorm"
    document["measurements"][0].update(
        {
            "fused_latency_ms": [1.5, 1.5, 1.5],
            "max_abs_added": 0.02,
        }
    )
    profile = normalize_mc2_profile(document)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_device_name=lambda _index: "Ascend 910B3"))
    monkeypatch.setenv("VLLM_ASCEND_PEARL_MC2_DIAGNOSTIC_FORCE_NATIVE_EPILOGUE", "1")
    x = torch.empty(2, 3, dtype=torch.bfloat16)
    weight = torch.empty(5, 3, dtype=torch.bfloat16)
    residual = torch.empty(2, 5, dtype=torch.bfloat16)

    result = profile.qualify(x, weight, residual, tp_size=3, is_trans_b=True)

    assert not result.qualified
    assert "numerical tolerance failed" in result.reason


def test_mc2_native_epilogue_diagnostic_override_does_not_force_custom(monkeypatch):
    document = _profile_document()
    document["measurements"][0]["fused_latency_ms"] = [1.5, 1.5, 1.5]
    profile = normalize_mc2_profile(document)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_device_name=lambda _index: "Ascend 910B3"))
    monkeypatch.setenv("VLLM_ASCEND_PEARL_MC2_DIAGNOSTIC_FORCE_NATIVE_EPILOGUE", "1")
    x = torch.empty(2, 3, dtype=torch.bfloat16)
    weight = torch.empty(5, 3, dtype=torch.bfloat16)
    residual = torch.empty(2, 5, dtype=torch.bfloat16)

    result = profile.qualify(x, weight, residual, tp_size=3, is_trans_b=True)

    assert not result.qualified
    assert "not faster" in result.reason


def test_mc2_profile_uses_scaled_numerical_gate_when_present(monkeypatch):
    document = _profile_document()
    row = document["measurements"][0]
    row.update(
        {
            "max_abs_added": 0.03125,
            "norm_rtol": 0.01,
            "added_rtol": 0.01,
            "max_scaled_norm": 0.25,
            "max_scaled_added": 0.75,
        }
    )
    profile = normalize_mc2_profile(document)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_device_name=lambda _index: "Ascend 910B3"))
    x = torch.empty(2, 3, dtype=torch.bfloat16)
    weight = torch.empty(5, 3, dtype=torch.bfloat16)
    residual = torch.empty(2, 5, dtype=torch.bfloat16)
    assert profile.qualify(x, weight, residual, tp_size=3, is_trans_b=True).qualified

    document["measurements"][0]["max_scaled_added"] = 1.01
    profile = normalize_mc2_profile(document)
    assert not profile.qualify(x, weight, residual, tp_size=3, is_trans_b=True).qualified


def test_mc2_profile_rejects_partial_scaled_numerical_gate():
    document = _profile_document()
    document["measurements"][0]["norm_rtol"] = 0.01
    with pytest.raises(ValueError, match="all scaled-error fields"):
        normalize_mc2_profile(document)
