# SPDX-License-Identifier: Apache-2.0
"""CPU tree control plans and explicit device-boundary transfer contracts.

The NPU-device spy below records requested devices but redirects allocations
to CPU. It checks Python boundary wiring, not NPU numerical correctness,
asynchronous-copy behavior, or actual device residency.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.ut.spec_decode.test_spec_rhythm_tree_loop import _TreeLoopHarness
from tests.ut.spec_decode.test_tree_draft_batch import _engine
from vllm_ascend.spec_decode.pearl.native_engine import NativePearlEngine
from vllm_ascend.spec_decode.pearl.tree import build_tree_speculation_plan, pack_selected_tree_plan, tree_primary_path


def _assert_cpu_plan(plan):
    for value in (plan.parent_indices, plan.positions, plan.cache_positions, plan.attention_mask):
        assert value.device.type == "cpu"


class _NPUDeviceBoundarySpy:
    """Record requested NPU allocations without allocating an NPU tensor."""

    def __init__(self, monkeypatch):
        self.device = torch.device("npu:3")
        self.transfers = []
        self.device_outputs = {}
        self.factory_calls = []
        original_to = torch.Tensor.to
        original_cat = torch.cat

        def requested(value):
            return isinstance(value, (str, torch.device)) and str(value) == str(self.device)

        def to(tensor, *args, **kwargs):
            requested_device = kwargs.get("device", args[0] if args else None)
            if not requested(requested_device):
                return original_to(tensor, *args, **kwargs)
            if "device" in kwargs:
                kwargs["device"] = torch.device("cpu")
            else:
                args = (torch.device("cpu"), *args[1:])
            result = original_to(tensor, *args, **kwargs).clone()
            self.transfers.append((tensor, result))
            self.device_outputs[id(result)] = result
            return result

        def factory_wrapper(name, original):
            def wrapped(*args, **kwargs):
                on_device = requested(kwargs.get("device"))
                if on_device:
                    kwargs["device"] = torch.device("cpu")
                result = original(*args, **kwargs)
                if on_device:
                    self.factory_calls.append((name, result))
                    self.device_outputs[id(result)] = result
                return result

            return wrapped

        def cat(tensors, *args, **kwargs):
            values = list(tensors)
            result = original_cat(values, *args, **kwargs)
            if values and all(id(value) in self.device_outputs for value in values):
                self.device_outputs[id(result)] = result
            return result

        monkeypatch.setattr(torch.Tensor, "to", to)
        for name in ("tensor", "arange", "empty", "zeros", "ones", "full"):
            monkeypatch.setattr(torch, name, factory_wrapper(name, getattr(torch, name)))
        monkeypatch.setattr(torch, "cat", cat)

    def assert_device_transfer(self, tensor):
        assert id(tensor) in self.device_outputs, "missing explicit execution-device transfer"


def test_normal_and_eager_plan_factories_ignore_execution_device():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    # A meta allocation cannot pass the plan's data-dependent validation.
    # Successful construction proves the factory explicitly selects CPU.
    engine.device = torch.device("meta")
    engine.config = SimpleNamespace(spec_rhythm_tree_width=2, spec_rhythm_tree_depth=2, max_model_len=64)
    state = SimpleNamespace(token_ids=list(range(8)))
    normal = engine._spec_rhythm_tree_plan(state, 3)
    eager = engine._spec_rhythm_tree_eager_plan(normal, state, 2)
    _assert_cpu_plan(normal)
    _assert_cpu_plan(eager)
    assert normal.candidate_budget == 3
    assert normal.prefix_len == 7
    assert normal.positions.tolist() == [7, 8, 9, 8, 9]
    assert eager.candidate_budget == 2
    assert eager.prefix_len == 10


def test_selected_scratch_and_trimmed_plans_keep_cpu_storage_and_dependencies():
    parent = pack_selected_tree_plan(build_tree_speculation_plan(2, 2, 7, 64), [0, 2, 3])
    assert parent.parent_indices.tolist() == [-1, -1, 0]
    assert tree_primary_path(parent) == [0, 2]
    eager = build_tree_speculation_plan(2, 2, 10, 64, candidate_budget=2)
    scratch = NativePearlEngine._tree_eager_scratch_plan(eager, parent)
    trimmed = pack_selected_tree_plan(parent, [0, 1])
    for plan in (parent, eager, scratch, trimmed):
        _assert_cpu_plan(plan)
    assert scratch.cache_positions.tolist() == [11, 12, 13, 14, 15]
    assert torch.equal(scratch.positions, eager.positions)
    assert scratch.attention_mask[:, 9].all()  # Parent's rejected sibling.
    assert not scratch.attention_mask[:, [7, 8, 10, 11]].any()
    assert trimmed.parent_indices.tolist() == [-1, -1]
    assert trimmed.cache_positions.tolist() == [7, 8, 9]
    assert trimmed.candidate_budget == 2


def test_selected_cpu_plan_reuses_immutable_packed_geometry():
    base = build_tree_speculation_plan(2, 3, 7, 64, candidate_budget=4)
    first = pack_selected_tree_plan(base, [0, 1, 2, 4])
    second = pack_selected_tree_plan(base, torch.tensor([0, 1, 2, 4]))

    assert first is second
    assert first.parent_indices.tolist() == [-1, 0, 1, 0]
    assert first.positions.tolist() == [7, 8, 9, 10, 9]
    assert first.cache_positions.tolist() == [7, 8, 9, 10, 11]
    assert torch.equal(first.attention_mask, second.attention_mask)


@pytest.mark.parametrize("level", [False, True])
def test_cpu_plan_metadata_explicitly_targets_model_device(monkeypatch, level):
    engine = _engine()
    boundary = _NPUDeviceBoundarySpy(monkeypatch)
    engine.model.embed_tokens = SimpleNamespace(weight=SimpleNamespace(device=boundary.device))
    engine.model.layers[0].self_attn.uses_paged_attention = True
    plan = pack_selected_tree_plan(build_tree_speculation_plan(2, 2, 3, 32), [0, 2, 3])
    original_mask = plan.attention_mask.clone()
    if level:
        ids, positions, metadata = engine.model.make_tree_level_attention_metadata(
            plan, 1, [0, 2], [7, 8], engine.cache_block_tables
        )
        expected_positions = [4, 5]
        expected_slots = [36, 38]
        expected_mask = plan.attention_mask[[1, 3]]
    else:
        ids, positions, metadata = engine.model.make_tree_attention_metadata(
            [plan], [7], [[8, 9, 10]], engine.cache_block_tables, sequence_ids=[1]
        )
        expected_positions = [3, 4, 4, 5]
        expected_slots = [35, 36, 37, 38]
        expected_mask = plan.attention_mask
    boundary.assert_device_transfer(ids)
    boundary.assert_device_transfer(positions)
    assert positions.tolist() == expected_positions
    assert metadata.slot_mapping.tolist() == expected_slots
    assert metadata.attention_mask is None
    assert torch.equal(
        metadata.tree_attention_mask[0, 0, : expected_mask.shape[0]],
        expected_mask,
    )
    assert metadata.context_lens.device.type == "cpu"
    assert metadata.tree_attention and metadata.use_fused_infer_attention
    # Mask transfers must be explicit too; the plan remains CPU and unchanged.
    mask_copies = [source for source, _ in boundary.transfers if source.dtype == torch.bool]
    assert len(mask_copies) == 1
    assert sum(value.numel() for value in mask_copies) == expected_mask.numel()
    assert torch.equal(plan.attention_mask, original_mask)
    _assert_cpu_plan(plan)


def test_draft_result_parent_indices_are_transferred_from_cpu_plan(monkeypatch):
    engine = _engine(requests=1)
    boundary = _NPUDeviceBoundarySpy(monkeypatch)
    engine.device = boundary.device
    plan = build_tree_speculation_plan(2, 2, 2, 32, candidate_budget=3)
    output = engine.draft_tree_forward([plan], [5], [0])
    boundary.assert_device_transfer(output["parent_indices"])
    assert output["parent_indices"].tolist() == [-1, 0, -1]
    assert output["num_draft_tokens"] == [3]
    _assert_cpu_plan(plan)


def test_execute_tree_round_transfers_cpu_parents_before_device_verifier(monkeypatch):
    engine = NativePearlEngine.__new__(NativePearlEngine)
    boundary = _NPUDeviceBoundarySpy(monkeypatch)
    engine.device = boundary.device
    plan = pack_selected_tree_plan(build_tree_speculation_plan(2, 2, 3, 16), [0, 2, 3])
    engine.target_tree_forward = Mock(
        return_value={"target_query_token_ids": torch.tensor([7, 8, 9, 10]), "bonus_token_ids": torch.tensor([10])}
    )
    original_verify = NativePearlEngine.verify_tree_outputs

    def verify(drafts, parents, *args):
        boundary.assert_device_transfer(drafts)
        boundary.assert_device_transfer(parents)
        return original_verify(drafts, parents, *args)

    engine.verify_tree_outputs = verify
    output = engine.execute_tree_round([plan], [6], [[7, 11, 8]], plan.parent_indices)
    assert output.token_ids.tolist() == [[7, 8, 10]]
    assert output.accepted_node_indices.tolist() == [[0, 2]]
    _assert_cpu_plan(plan)


def test_actual_service_loop_transfers_cpu_plan_parents_at_verification(monkeypatch):
    harness = _TreeLoopHarness(monkeypatch, requests=4, capacity=2, max_tokens=6)
    boundary = _NPUDeviceBoundarySpy(monkeypatch)
    harness.engine.device = boundary.device
    # All spy allocations physically live on CPU, including mocked model KV.
    harness.engine._spec_rhythm_tree_cache_device = lambda _device: torch.device("cpu")
    original_verify = NativePearlEngine.verify_tree_outputs
    verified_parents = []

    def verify(drafts, parents, *args):
        boundary.assert_device_transfer(drafts)
        boundary.assert_device_transfer(parents)
        verified_parents.append(parents.tolist())
        return original_verify(drafts, parents, *args)

    original_draft = harness.engine.draft_tree_forward
    original_target = harness.engine.target_tree_forward

    def check_plans(forward):
        def checked(plans, *args, **kwargs):
            for plan in plans:
                _assert_cpu_plan(plan)
            return forward(plans, *args, **kwargs)

        return checked

    harness.engine.verify_tree_outputs = verify
    harness.engine.draft_tree_forward = check_plans(original_draft)
    harness.engine.target_tree_forward = check_plans(original_target)
    results = harness.run()
    assert len(results) == 4
    assert verified_parents
    assert all(len(result["completion_token_ids"]) == 6 for result in results)
