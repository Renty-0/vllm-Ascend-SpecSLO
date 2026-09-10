# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as functional

from examples import check_specslo_draft_bucket_layers as diagnostic
from vllm_ascend.spec_decode.pearl.native_model import NativeAttentionMetadata


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_size = 4
        self.key_cache = torch.arange(8, dtype=torch.float32).reshape(4, 1, 2) / 10
        self.value_cache = self.key_cache.clone() + 1

    def _dense_attention(self, query, metadata):
        output = []
        for row, length in enumerate(metadata.context_lens.tolist()):
            keys = self.key_cache[:length].transpose(0, 1).unsqueeze(0)
            values = self.value_cache[:length].transpose(0, 1).unsqueeze(0)
            visible = (~metadata.attention_mask[row, :length]).view(1, 1, 1, -1)
            result = functional.scaled_dot_product_attention(
                query[row].reshape(1, 1, 1, 2), keys, values, attn_mask=visible
            )
            output.append(result.reshape(1, 2))
        return torch.stack(output)

    def forward(self, hidden, metadata):
        return self._dense_attention(hidden.unsqueeze(1), metadata).reshape_as(hidden)


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttention()

    def forward(self, positions, hidden, residual, metadata):
        attended = self.self_attn(hidden, metadata)
        return hidden + attended, hidden


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TinyLayer(), TinyLayer()])
        self.explode = False

    def forward(self, tokens, positions, metadata):
        if self.explode:
            raise RuntimeError("injected model failure")
        hidden = tokens.float().reshape(-1, 1).expand(-1, 2)
        residual = None
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, residual, metadata)
        return hidden


def _metadata():
    return NativeAttentionMetadata(
        slot_mapping=torch.tensor([1, 2]),
        context_lens=torch.tensor([2, 3]),
        block_tables=torch.tensor([[0], [0]]),
        attention_mask=torch.tensor([[False, False, True, True], [False, True, False, True]]),
    )


def test_snapshot_restore_is_exact_and_independent_of_live_mutations():
    model = TinyModel()
    snapshot = diagnostic._snapshot_cache(model)
    expected = [(keys.clone(), values.clone()) for keys, values in snapshot]
    for layer in model.layers:
        layer.self_attn.key_cache.add_(100)
        layer.self_attn.value_cache.fill_(float("nan"))
    diagnostic._restore_cache(model, snapshot)
    for layer, (keys, values), (saved_keys, saved_values) in zip(model.layers, expected, snapshot):
        assert torch.equal(layer.self_attn.key_cache, keys)
        assert torch.equal(layer.self_attn.value_cache, values)
        assert torch.equal(saved_keys, keys)
        assert torch.equal(saved_values, values)


def test_bad_snapshot_shape_is_rejected_before_any_cache_is_restored():
    model = TinyModel()
    snapshot = diagnostic._snapshot_cache(model)
    model.layers[0].self_attn.key_cache.fill_(99)
    snapshot[1] = (torch.zeros(1), snapshot[1][1])
    with pytest.raises(ValueError, match="snapshot shape"):
        diagnostic._restore_cache(model, snapshot)
    assert (model.layers[0].self_attn.key_cache == 99).all()


def test_actual_sdpa_arguments_and_visible_physical_slots_are_captured():
    model = TinyModel()
    original_sdpa = functional.scaled_dot_product_attention
    original = diagnostic._capture_forward(model, torch.tensor([1, 2]), torch.tensor([1, 2]), _metadata())
    bucket = diagnostic._capture_forward(
        model, torch.tensor([1, 2]), torch.tensor([1, 2]), replace(_metadata(), context_lens=torch.tensor([4, 4]))
    )
    assert functional.scaled_dot_product_attention is original_sdpa
    assert set(original["sdpa"]) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    row = original["sdpa"][(0, 1)]
    assert row["physical_slots"] == [0, 1, 2]
    assert row["keys"].shape == (1, 1, 3, 2)
    indices, keys, values = diagnostic._visible(row)
    assert indices.tolist() == [0, 2]
    assert torch.equal(keys, model.layers[0].self_attn.key_cache[[0, 2]].transpose(0, 1).unsqueeze(0))
    summary = diagnostic._summarize(original, bucket, atol=1e-6, rtol=1e-6)
    assert summary["output"]["within_tolerance"]
    assert summary["first_different_visible_inputs"] is None
    assert all(node["identical_visible_inputs"] for layer in summary["layers"] for node in layer["nodes"])


def test_failure_always_restores_global_sdpa_and_module_methods_and_hooks():
    model = TinyModel()
    model.explode = True
    original_sdpa = functional.scaled_dot_product_attention
    with pytest.raises(RuntimeError, match="injected model failure"):
        diagnostic._capture_forward(model, torch.tensor([1, 2]), torch.tensor([1, 2]), _metadata())
    assert functional.scaled_dot_product_attention is original_sdpa
    for layer in model.layers:
        assert not layer._forward_hooks
        assert not layer.self_attn._forward_hooks
        assert "_dense_attention" not in layer.self_attn.__dict__


def test_comparison_detects_visible_inputs_changed_even_with_same_mask():
    data = diagnostic._capture_forward(TinyModel(), torch.tensor([1, 2]), torch.tensor([1, 2]), _metadata())
    original = data["sdpa"][(0, 1)]
    changed = {**original, "keys": original["keys"].clone()}
    changed["keys"][0, 0, 0, 0] += 0.1
    summary = diagnostic._compare_sdpa(original, changed, atol=0.001, rtol=0.001)
    assert summary["visible_logical_positions_equal"]
    assert summary["visible_physical_slots_equal"]
    assert summary["query"]["exact_equal"]
    assert not summary["visible_keys"]["exact_equal"]
    assert not summary["identical_visible_inputs"]


def test_nonfinite_sdpa_is_reported_without_hiding_failure_in_summary():
    original = diagnostic._capture_forward(TinyModel(), torch.tensor([1, 2]), torch.tensor([1, 2]), _metadata())
    changed = diagnostic._capture_forward(TinyModel(), torch.tensor([1, 2]), torch.tensor([1, 2]), _metadata())
    changed["sdpa"][(0, 1)]["output"] = torch.full_like(changed["sdpa"][(0, 1)]["output"], float("nan"))
    summary = diagnostic._summarize(original, changed, atol=0.001, rtol=0.001)
    assert summary["first_different_sdpa"] == (0, 1)
    assert summary["first_different_bf16_sdpa"] == (0, 1)
    assert not summary["layers"][0]["nodes"][1]["sdpa_output"]["finite"]


def test_actual_pre_forward_snapshot_round_trip_compacts_physical_pages(tmp_path):
    model = TinyModel()
    for layer in model.layers:
        layer.self_attn.key_cache = torch.arange(24, dtype=torch.float32).reshape(12, 1, 2)
        layer.self_attn.value_cache = layer.self_attn.key_cache + 10
    metadata = replace(_metadata(), slot_mapping=torch.tensor([9, 10]), block_tables=torch.tensor([[2], [2]]))
    payload = diagnostic.capture_draft_replay_snapshot(model, torch.tensor([1, 2]), torch.tensor([1, 2]), metadata)
    assert payload["original_physical_pages"] == [0, 2]
    assert payload["metadata"]["slot_mapping"].tolist() == [5, 6]
    assert payload["metadata"]["block_tables"].tolist() == [[1], [1]]
    destination = tmp_path / "snapshot.pt"
    torch.save(payload, destination)
    saved = torch.load(destination, map_location="cpu", weights_only=True)
    for layer in model.layers:
        layer.self_attn.key_cache.fill_(-1)
        layer.self_attn.value_cache.fill_(-1)
    inputs, positions, replay = diagnostic.restore_draft_replay_snapshot(model, saved, device="cpu")
    assert inputs.tolist() == [1, 2] and positions.tolist() == [1, 2]
    assert replay.block_tables.tolist() == [[1], [1]]
    assert replay.context_lens.device.type == "cpu"
    assert model.layers[0].self_attn.key_cache[4:8, 0, 0].tolist() == [16, 18, 20, 22]
    assert (model.layers[0].self_attn.key_cache[8:] == -1).all()


def test_snapshot_validation_precedes_any_cache_write():
    model = TinyModel()
    payload = diagnostic.capture_draft_replay_snapshot(model, torch.tensor([1, 2]), torch.tensor([1, 2]), _metadata())
    payload["cache"][-1][-1] = torch.zeros(1, 4, 1, 3)
    model.layers[0].self_attn.key_cache.fill_(-1)
    with pytest.raises(ValueError, match="cache shape"):
        diagnostic.restore_draft_replay_snapshot(model, payload, device="cpu")
    assert (model.layers[0].self_attn.key_cache == -1).all()


def test_failed_capture_hook_saves_pre_forward_cache_only_on_failure(tmp_path):
    model = TinyModel()
    model.context = SimpleNamespace(size=1)

    class Runner:
        failed_capture_count = 0
        last_target_validation_error = {"injected": True}
        fail = False

        def _capture_target(self, *args, **kwargs):
            self.model.layers[0].self_attn.key_cache.fill_(999)
            self.failed_capture_count += int(self.fail)
            return "unchanged output"

    runner = Runner()
    runner.model = model
    path = tmp_path / "actual.pt"
    remove = diagnostic.install_failed_draft_capture_hook(runner, path, minimum_context=1)
    args = ("entry", [torch.tensor([1, 2])], [torch.tensor([1, 2])], [_metadata()], 100)
    assert runner._capture_target(*args, output_kind="hidden") == "unchanged output"
    assert not path.exists()
    model.layers[0].self_attn.key_cache.fill_(123)
    runner.fail = True
    assert runner._capture_target(*args, output_kind="hidden") == "unchanged output"
    saved = torch.load(path, weights_only=True)
    assert (saved["cache"][0][0] == 123).all()
    assert (model.layers[0].self_attn.key_cache == 999).all()
    assert saved["failure"]["validation"] == {"injected": True}
    remove()
    assert "_capture_target" not in runner.__dict__
