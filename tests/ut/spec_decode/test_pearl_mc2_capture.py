# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

import vllm_ascend.spec_decode.pearl.native_model as native_model_module
from vllm_ascend.spec_decode.pearl.native_model import (
    NativeQwen2ForCausalLM,
    NativeQwen2MLP,
    NativeTPContext,
    _MC2RealInputCapture,
)


def _context(rank: int = 1) -> NativeTPContext:
    return NativeTPContext(group=None, rank=rank, size=3, leader_rank=0)


def _configure(
    monkeypatch,
    directory: str = "",
    rows: str = "",
    *,
    kind: str = "attention",
    layers: str = "",
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_PEARL_MC2_CAPTURE_DIR", directory)
    monkeypatch.setenv("VLLM_ASCEND_PEARL_MC2_CAPTURE_ROWS", rows)
    monkeypatch.setenv("VLLM_ASCEND_PEARL_MC2_CAPTURE_KIND", kind)
    monkeypatch.setenv("VLLM_ASCEND_PEARL_MC2_CAPTURE_LAYERS", layers)


def test_mc2_capture_is_disabled_without_both_opt_in_values(monkeypatch, tmp_path):
    _configure(monkeypatch)
    assert _MC2RealInputCapture.from_env(SimpleNamespace(), _context()) is None

    _configure(monkeypatch, str(tmp_path), "")
    with pytest.raises(ValueError, match="must be set together"):
        _MC2RealInputCapture.from_env(SimpleNamespace(enforce_eager=True), _context())


@pytest.mark.parametrize("rows", ("0", "64,,128", "64,nope", "64,64"))
def test_mc2_capture_rejects_invalid_rows(monkeypatch, tmp_path, rows):
    _configure(monkeypatch, str(tmp_path), rows)
    with pytest.raises(ValueError, match="rows must"):
        _MC2RealInputCapture.from_env(SimpleNamespace(enforce_eager=True), _context())


def test_mc2_capture_rejects_graph_mode_and_relative_directory(monkeypatch, tmp_path):
    _configure(monkeypatch, "relative/capture", "8")
    with pytest.raises(ValueError, match="absolute, non-root"):
        _MC2RealInputCapture.from_env(SimpleNamespace(enforce_eager=True), _context())

    _configure(monkeypatch, str(tmp_path), "8")
    with pytest.raises(ValueError, match="enforce-eager diagnostics"):
        _MC2RealInputCapture.from_env(SimpleNamespace(enforce_eager=False), _context())


@pytest.mark.parametrize(
    ("kind", "layers", "message"),
    (
        ("mlp", "", "kind must"),
        ("down", "-1", "layers must"),
        ("down", "2,,3", "layers must"),
        ("down", "2,2", "layers must"),
    ),
)
def test_mc2_capture_rejects_invalid_projection_or_layer_filter(
    monkeypatch, tmp_path, kind, layers, message
):
    _configure(monkeypatch, str(tmp_path), "8", kind=kind, layers=layers)
    with pytest.raises(ValueError, match=message):
        _MC2RealInputCapture.from_env(SimpleNamespace(enforce_eager=True), _context())


def test_mc2_capture_rejects_layer_outside_configured_decoder(monkeypatch, tmp_path):
    _configure(monkeypatch, str(tmp_path), "8", kind="down", layers="3")
    with pytest.raises(ValueError, match="outside the configured decoder"):
        _MC2RealInputCapture.from_env(
            SimpleNamespace(enforce_eager=True, num_hidden_layers=3),
            _context(),
        )


def test_mc2_capture_writes_measurement_payload_once_atomically(monkeypatch, tmp_path):
    _configure(monkeypatch, str(tmp_path), "8,16")
    capture = _MC2RealInputCapture.from_env(SimpleNamespace(enforce_eager=True), _context())
    assert capture is not None
    activation = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    weight = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    residual = torch.arange(40, dtype=torch.float32).reshape(2, 4, 5)
    gamma = torch.arange(5, dtype=torch.float32)
    replacements = []
    real_replace = native_model_module.os.replace

    def record_replace(source, destination):
        replacements.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr(native_model_module.os, "replace", record_replace)
    capture.maybe_capture(activation, weight, residual, gamma)

    destination = tmp_path / "attention" / "layer-0" / "m8" / "rank-1.pt"
    payload = torch.load(destination, map_location="cpu", weights_only=True)
    assert set(payload) == {"activation", "weight", "residual", "gamma"}
    assert payload["activation"].shape == (8, 3)
    assert payload["weight"].shape == (5, 3)
    assert payload["residual"].shape == (8, 5)
    assert payload["gamma"].shape == (5,)
    assert torch.equal(payload["activation"], activation.reshape(8, 3))
    assert len(replacements) == 1
    assert not list(destination.parent.glob("*.tmp"))

    activation.add_(1000)
    capture.maybe_capture(activation, weight, residual, gamma)
    unchanged = torch.load(destination, map_location="cpu", weights_only=True)
    assert torch.equal(unchanged["activation"], payload["activation"])
    assert len(replacements) == 1


def test_down_capture_is_not_consumed_by_attention_and_paths_include_layer(monkeypatch, tmp_path):
    _configure(monkeypatch, str(tmp_path), "8", kind="down")
    capture = _MC2RealInputCapture.from_env(SimpleNamespace(enforce_eager=True), _context())
    assert capture is not None
    activation = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    weight = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    residual = torch.arange(40, dtype=torch.float32).reshape(8, 5)
    gamma = torch.arange(5, dtype=torch.float32)

    capture.maybe_capture(
        activation,
        weight,
        residual,
        gamma,
        projection_kind="attention",
        layer_index=0,
    )
    assert not list(tmp_path.rglob("*.pt"))

    capture.maybe_capture(
        activation,
        weight,
        residual,
        gamma,
        projection_kind="down",
        layer_index=7,
    )
    assert (tmp_path / "down" / "layer-7" / "m8" / "rank-1.pt").is_file()
    assert not (tmp_path / "attention").exists()


def test_explicit_layer_filter_captures_each_selected_layer(monkeypatch, tmp_path):
    _configure(monkeypatch, str(tmp_path), "8", kind="down", layers="2,4")
    capture = _MC2RealInputCapture.from_env(SimpleNamespace(enforce_eager=True), _context())
    assert capture is not None
    activation = torch.zeros(8, 3)
    weight = torch.zeros(5, 3)
    residual = torch.zeros(8, 5)
    gamma = torch.ones(5)

    for layer_index in (1, 2, 3, 4):
        capture.maybe_capture(
            activation + layer_index,
            weight,
            residual,
            gamma,
            projection_kind="down",
            layer_index=layer_index,
        )

    paths = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*.pt"))
    assert paths == [
        "down/layer-2/m8/rank-1.pt",
        "down/layer-4/m8/rank-1.pt",
    ]


def test_mlp_passes_post_swiglu_down_projection_inputs_to_capture():
    class FixedGateUp(nn.Module):
        def __init__(self, output):
            super().__init__()
            self.output = output

        def forward(self, _hidden_states):
            return self.output

    class RecordingDown(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.arange(15, dtype=torch.float32).reshape(5, 3))
            self.last_input = None

        def forward(self, activation):
            self.last_input = activation
            return torch.nn.functional.linear(activation, self.weight)

    mlp = NativeQwen2MLP.__new__(NativeQwen2MLP)
    nn.Module.__init__(mlp)
    gate_up = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    mlp.gate_up_proj = FixedGateUp(gate_up)
    mlp.down_proj = RecordingDown()
    capture = Mock(projection_kind="down")
    residual = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    gamma = torch.ones(5)

    mlp(
        torch.zeros(2, 5),
        mc2_input_capture=capture,
        residual=residual,
        next_layernorm_gamma=gamma,
        layer_index=3,
    )

    gate, up = gate_up.chunk(2, dim=-1)
    expected_activation = torch.nn.functional.silu(gate) * up
    assert torch.equal(mlp.down_proj.last_input, expected_activation)
    args = capture.maybe_capture.call_args
    assert torch.equal(args.args[0], expected_activation)
    assert args.args[1] is mlp.down_proj.weight
    assert args.args[2] is residual
    assert args.args[3] is gamma
    assert args.kwargs == {"projection_kind": "down", "layer_index": 3}


def test_model_routes_next_input_norm_and_final_norm_gamma_to_each_layer():
    class Norm(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.weight = nn.Parameter(torch.full((2,), value))

        def forward(self, hidden_states, residual):
            return hidden_states, residual

    class Layer(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.input_layernorm = Norm(value)
            self.seen_gamma = None

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
            self.seen_gamma = next_layernorm_gamma
            assert next_layernorm.weight is next_layernorm_gamma
            assert not input_is_normalized
            assert return_normalization_state
            return hidden_states, hidden_states if residual is None else residual, False

    model = NativeQwen2ForCausalLM.__new__(NativeQwen2ForCausalLM)
    nn.Module.__init__(model)
    model.embed_tokens = nn.Identity()
    model.layers = nn.ModuleList((Layer(1.0), Layer(2.0)))
    model.norm = Norm(3.0)
    model.track_cache_finiteness = False
    model(torch.ones(1, 2), torch.zeros(1, dtype=torch.long))

    assert model.layers[0].seen_gamma is model.layers[1].input_layernorm.weight
    assert model.layers[1].seen_gamma is model.norm.weight


def test_mc2_capture_rejects_active_aclgraph(monkeypatch, tmp_path):
    _configure(monkeypatch, str(tmp_path), "8")
    capture = _MC2RealInputCapture.from_env(SimpleNamespace(), _context())
    assert capture is not None
    monkeypatch.setattr(
        native_model_module.torch,
        "npu",
        SimpleNamespace(is_current_stream_capturing=lambda: True),
        raising=False,
    )
    fake_activation = SimpleNamespace(device=SimpleNamespace(type="npu"))
    with pytest.raises(RuntimeError, match="cannot run during ACLGraph capture"):
        capture._reject_graph_capture(fake_activation)
