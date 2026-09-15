# SPDX-License-Identifier: Apache-2.0
"""Strict measured-table provenance versus explicitly legacy roof mappings."""

import copy
import json
import pickle
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from examples.profile_specslo_tree_roofline import build_roofline_table
from vllm_ascend.spec_decode.pearl import native_engine as native_module
from vllm_ascend.spec_decode.pearl.api import PEARLConfig
from vllm_ascend.spec_decode.pearl.native_engine import NativePearlConfig
from vllm_ascend.spec_decode.pearl.roofline import (
    ProfiledRoofline,
    normalize_roofline,
    parse_roofline_argument,
    runtime_source_fingerprints,
    validate_roofline_hardware,
)
from vllm_ascend.spec_decode.pearl.spec_rhythm import SpecRhythmBudgetShaper


@pytest.fixture
def profile_document():
    # Synthetic measurements exercise schema/compatibility checks only, not
    # hardware capacity or the truth of any measured performance claim.
    common = {
        "batch_size": 2,
        "verification_requests": 1,
        "context_len": 512,
        "warmup_iterations": 2,
        "timed_graph_captures": 0,
        "timed_graph_replays": 3,
    }
    return build_roofline_table(
        {
            "metadata": {
                "model": "/models/target",
                "target_tensor_parallel_size": 3,
                "execution_mode": "graph",
                "hardware": "Ascend910B2",
                "measurement_scope": "target_forward",
                "verification_attention_backends": ["fused_infer_attention_tree_v1"],
                "ar_attention_backend": "paged_attention_v1",
                "ar_comparator": "standard_decode_full_active_batch",
                **runtime_source_fingerprints(),
                "tree_fia_sparse_mode": 1,
                "tree_fia_inner_precise": 1,
                "max_model_len": 1024,
                "tree_width": 2,
                "tree_depth": 2,
            },
            "measurements": [
                {
                    **common,
                    "kind": "ar",
                    "attention_backend": "paged_attention_v1",
                    "verification_requests": 2,
                    "candidate_counts": [0, 0],
                    "physical_query_tokens": 2,
                    "latency_ms": [10.0, 10.0, 10.0],
                },
                {
                    **common,
                    "kind": "packed_tree",
                    "attention_backend": "fused_infer_attention_tree_v1",
                    "candidate_counts": [2],
                    "physical_query_tokens": 3,
                    "latency_ms": [10.1, 10.1, 10.1],
                },
            ],
        }
    )


def _normalize(value, **kwargs):
    return normalize_roofline(
        value,
        **{
            "model": "/models/target",
            "target_tp_size": 3,
            "enforce_eager": False,
            "max_model_len": 1024,
            "tree_width": 2,
            "tree_depth": 2,
            **kwargs,
        },
    )


def test_strict_profile_is_loaded_and_preserves_evidence_through_pickle(profile_document):
    profile = _normalize(profile_document)
    assert isinstance(profile, ProfiledRoofline)
    assert profile.strict is True
    assert profile.lookup(2, 512) == 2
    assert profile.metadata["hardware"] == "Ascend910B2"
    assert profile.evidence[0]["verification_requests"] == 1
    assert profile.evidence[0]["context_len"] == 512
    assert profile.evidence[0]["measured_roof"] == 2
    assert profile.evidence[0]["candidate_sweep"][0]["shapes"][0]["candidate_counts"] == [2]
    restored = pickle.loads(pickle.dumps(profile))
    assert isinstance(restored, ProfiledRoofline)
    assert restored.metadata == profile.metadata
    assert restored.evidence == profile.evidence
    assert restored.lookup(2, 512) == 2


@pytest.mark.parametrize(
    "field",
    [
        "verification_attention_backends",
        "ar_attention_backend",
        "ar_comparator",
        "native_engine_source_sha256",
        "native_model_source_sha256",
        "native_graph_source_sha256",
        "tree_source_sha256",
        "tree_fia_sparse_mode",
        "tree_fia_inner_precise",
        "max_model_len",
        "tree_width",
        "tree_depth",
    ],
)
def test_full_profile_without_backend_provenance_cannot_be_treated_as_measured(profile_document, field):
    profile_document["metadata"].pop(field)
    with pytest.raises(ValueError, match=field):
        _normalize(profile_document)


def test_profile_requires_actual_prepared_target_attention_backend(profile_document):
    profile = _normalize(profile_document)
    profile.validate_attention_backend("fused_infer_attention_tree_v1")
    for wrong in ("dense_sdpa_tree_v1", "paged_attention_v1", "unknown"):
        with pytest.raises(ValueError, match="does not match actual target metadata"):
            profile.validate_attention_backend(wrong)


def test_profile_rejects_wrong_tree_fia_inner_precision(profile_document):
    profile_document["metadata"]["tree_fia_inner_precise"] = 2
    with pytest.raises(ValueError, match="tree_fia_inner_precise=1"):
        _normalize(profile_document)


@pytest.mark.parametrize("field", list(runtime_source_fingerprints()))
def test_profile_rejects_source_changed_after_measurement(profile_document, field):
    profile_document["metadata"][field] = "f" * 64
    with pytest.raises(ValueError, match=f"{field} does not match"):
        _normalize(profile_document)


def test_profiled_zero_budget_is_preserved_as_measured_target_only_fallback(profile_document):
    profile_document["roofline"] = {"2:1": 0}
    profile_document["evidence"][0]["measured_roof"] = 0
    profile_document["target_only_fallback_lookup_keys"] = ["2:1"]
    profile = _normalize(profile_document)
    assert profile.lookup(2, 512) == 0
    profile.validate_attention_backend("paged_attention_v1", target_only=True)
    with pytest.raises(ValueError, match="does not match"):
        profile.validate_attention_backend("fused_infer_attention_tree_v1", target_only=True)


def test_profiled_zero_budget_requires_explicit_fallback_identity(profile_document):
    profile_document["roofline"] = {"2:1": 0}
    profile_document["evidence"][0]["measured_roof"] = 0
    with pytest.raises(ValueError, match="fallback keys"):
        _normalize(profile_document)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"model": "/models/wrong"}, "model"),
        ({"target_tp_size": 2}, "target TP"),
        ({"enforce_eager": True}, "execution mode"),
        ({"max_model_len": 2048}, "max_model_len"),
        ({"tree_width": 4}, "tree topology"),
        ({"tree_depth": 4}, "tree topology"),
    ],
)
def test_profile_rejects_wrong_model_tp_or_mode(profile_document, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _normalize(profile_document, **kwargs)


def test_strict_profile_does_not_invent_missing_batch_or_context_budget(profile_document):
    shaper = SpecRhythmBudgetShaper(min_gamma=1, max_gamma=8, roofline=_normalize(profile_document))
    assert isinstance(shaper.roofline, ProfiledRoofline)
    assert shaper.verification_roof(2, 512) == 2
    for batch, context in ((3, 512), (2, 513)):
        with pytest.raises(ValueError, match="Unprofiled.*gamma-derived fallback"):
            shaper.verification_roof(batch, context)


def test_strict_profile_distinguishes_measured_zero_from_unprofiled_lookup(profile_document):
    profile = _normalize(profile_document)
    assert profile.lookup_optional(2, 512) == 2
    assert profile.lookup_optional(3, 512) is None
    assert profile.covers_execution(2, 512, 1)
    assert not profile.covers_execution(2, 512, 2)


def test_legacy_bare_mapping_retains_unprofiled_compatibility_fallback():
    roof = _normalize({"2:1": 2})
    assert not isinstance(roof, ProfiledRoofline)
    shaper = SpecRhythmBudgetShaper(min_gamma=1, max_gamma=4, roofline=roof)
    assert shaper.verification_roof(2, 512) == 2
    assert shaper.verification_roof(3, 513) == 12


def test_profile_half_home_evidence_does_not_cover_merged_or_tail_target_rows(profile_document):
    profile = _normalize(profile_document)
    profile.validate_execution(batch_size=2, context_len=512, verification_requests=1)
    with pytest.raises(ValueError, match="Unprofiled.*physical target rows"):
        profile.validate_execution(batch_size=2, context_len=512, verification_requests=2)
    with pytest.raises(ValueError, match="Unprofiled.*roofline key"):
        profile.validate_execution(batch_size=1, context_len=512, verification_requests=1)


def test_profile_hardware_compares_real_supplied_device_identity(profile_document):
    profile = _normalize(profile_document)
    validate_roofline_hardware(profile, "Ascend 910B2")
    for hardware in ("Ascend910B3", "Ascend910", "", "NVIDIA A100"):
        with pytest.raises(ValueError, match="does not match detected device"):
            validate_roofline_hardware(profile, hardware)


def test_cli_argument_accepts_profile_file_and_legacy_json(tmp_path, profile_document):
    path = tmp_path / "measured roof.json"
    path.write_text(json.dumps(profile_document), encoding="utf-8")
    assert parse_roofline_argument(str(path)) == profile_document
    assert isinstance(_normalize(str(path)), ProfiledRoofline)
    assert parse_roofline_argument('{"2:1": 2}') == {"2:1": 2}
    with pytest.raises(ValueError, match="Cannot read"):
        parse_roofline_argument(str(tmp_path / "missing.json"))


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.update(evidence=[]), "evidence"),
        (lambda value: value.update(context_bucket_size=1024), "512-token"),
        (lambda value: value["roofline"].update({"2:1": 3}), "exceeds its measured roof"),
        (lambda value: value["roofline"].update({"default": 2}), "exact batch:context"),
        (lambda value: value.update(unqualified_lookup_keys=["2:1"]), "unqualified"),
        (lambda value: value["evidence"][0].update(context_len=513), "inconsistent"),
        (lambda value: value["metadata"].update(verification_layout="linear"), "verification layout"),
        (lambda value: value["metadata"].pop("verification_layout"), "verification layout"),
    ],
)
def test_strict_profile_rejects_missing_or_inconsistent_evidence(profile_document, mutation, match):
    document = copy.deepcopy(profile_document)
    mutation(document)
    with pytest.raises(ValueError, match=match):
        _normalize(document)


def test_native_config_keeps_strict_profile_and_rejects_fixed_override(profile_document):
    config = NativePearlConfig(
        "draft",
        "/models/target",
        1,
        3,
        4,
        1024,
        8,
        spec_rhythm_roofline=profile_document,
        spec_rhythm_tree_width=2,
        spec_rhythm_tree_depth=2,
    )
    assert isinstance(config.spec_rhythm_roofline, ProfiledRoofline)
    with pytest.raises(ValueError, match="cannot be overridden"):
        NativePearlConfig(
            "draft",
            "/models/target",
            1,
            3,
            4,
            1024,
            8,
            spec_rhythm_roofline=profile_document,
            spec_rhythm_verification_budget=8,
            spec_rhythm_tree_width=2,
            spec_rhythm_tree_depth=2,
        )
    with pytest.raises(ValueError, match="cannot be overridden"):
        SpecRhythmBudgetShaper(min_gamma=1, max_gamma=4, roofline=_normalize(profile_document), verification_budget=8)
    with pytest.raises(ValueError, match="tree topology"):
        NativePearlConfig("draft", "/models/target", 1, 3, 4, 1024, 8, spec_rhythm_roofline=profile_document)


@pytest.mark.parametrize("role,peer_invalid", [("target", False), ("target", True), ("draft", True)])
def test_native_initialization_validates_detected_hardware_before_model_load(
    monkeypatch,
    profile_document,
    role,
    peer_invalid,
):
    config = NativePearlConfig(
        "draft",
        "/models/target",
        1,
        3,
        4,
        1024,
        8,
        spec_rhythm_roofline=profile_document,
        spec_rhythm_tree_width=2,
        spec_rhythm_tree_depth=2,
    )
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(native_module.torch.npu, "set_device", Mock())
    detected_name = Mock(return_value="Ascend910B2" if peer_invalid else "Ascend910B3")
    monkeypatch.setattr(native_module.torch.npu, "get_device_name", detected_name)
    monkeypatch.setattr(native_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(native_module.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(native_module.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(native_module, "init_device_properties_triton", lambda: None)
    actual_device = native_module.torch.device
    monkeypatch.setattr(native_module.torch, "device", lambda value: actual_device("cpu"))
    monkeypatch.setattr(
        native_module.PearlProcessGroups,
        "create",
        lambda *args, **kwargs: SimpleNamespace(is_draft_worker=role == "draft"),
    )
    flags = []

    def reduce_hardware_flag(flag, **kwargs):
        assert flag.dtype == native_module.torch.int64
        assert kwargs["op"] == native_module.dist.ReduceOp.MAX
        flags.append(int(flag[0]))
        if peer_invalid:
            flag[0] = 1

    monkeypatch.setattr(native_module.dist, "all_reduce", reduce_hardware_flag)
    model_context = Mock(side_effect=AssertionError("model initialization must not begin"))
    monkeypatch.setattr(native_module, "NativeTPContext", model_context)

    match = "failed on another target rank" if peer_invalid else "does not match detected device 'Ascend910B3'"
    with pytest.raises(ValueError, match=match):
        native_module.NativePearlEngine(config)

    assert flags == [0 if peer_invalid else 1]
    if role == "target":
        detected_name.assert_called_once_with(1)
    else:
        detected_name.assert_not_called()
    model_context.assert_not_called()


def test_public_config_passes_strict_profile_to_worker_config(monkeypatch, profile_document):
    monkeypatch.setattr(
        "vllm_ascend.spec_decode.pearl.api.AutoConfig.from_pretrained",
        lambda path: SimpleNamespace(architectures=["Qwen3ForCausalLM"], eos_token_id=1),
    )
    config = PEARLConfig(
        "draft",
        "/models/target",
        draft_tensor_parallel_size=1,
        target_tensor_parallel_size=3,
        gamma=4,
        max_model_len=1024,
        spec_rhythm_roofline=profile_document,
        spec_rhythm_tree_width=2,
        spec_rhythm_tree_depth=2,
    )
    native = config.to_native()
    assert isinstance(native.spec_rhythm_roofline, ProfiledRoofline)
    assert native.spec_rhythm_roofline.metadata == profile_document["metadata"]


@pytest.mark.parametrize(
    "draft,used_graph,peer_failure,raises,local_flag",
    [
        (False, True, False, False, 0),
        (False, False, False, True, 1),
        (False, True, True, True, 0),
        (True, False, False, False, 0),
        (True, False, True, True, 0),
    ],
)
def test_strict_graph_roof_requires_actual_target_graph_on_all_ranks(
    monkeypatch,
    profile_document,
    draft,
    used_graph,
    peer_failure,
    raises,
    local_flag,
):
    engine = native_module.NativePearlEngine.__new__(native_module.NativePearlEngine)
    engine.config = NativePearlConfig(
        "draft",
        "/models/target",
        1,
        3,
        4,
        1024,
        8,
        spec_rhythm_roofline=profile_document,
        spec_rhythm_tree_width=2,
        spec_rhythm_tree_depth=2,
    )
    engine.is_draft = draft
    engine.device = "cpu"
    flags = []

    def reduce(flag, *, op):
        assert flag.dtype == native_module.torch.int64
        assert op == native_module.dist.ReduceOp.MAX
        flags.append(int(flag[0]))
        if peer_failure:
            flag[0] = 1

    monkeypatch.setattr(native_module.dist, "all_reduce", reduce)
    output = None if draft else {"used_aclgraph": used_graph, "attention_backend": "fused_infer_attention_tree_v1"}
    if raises:
        with pytest.raises(RuntimeError, match="target eager fallback"):
            engine._validate_spec_rhythm_target_graph(output, has_target_work=True)
    else:
        engine._validate_spec_rhythm_target_graph(output, has_target_work=True)
    assert flags == [local_flag]
    flags.clear()
    engine._validate_spec_rhythm_target_graph(None, has_target_work=False)
    assert flags == []


def test_legacy_roof_fallback_does_not_add_collectives_or_graph_restrictions(monkeypatch):
    engine = native_module.NativePearlEngine.__new__(native_module.NativePearlEngine)
    engine.config = NativePearlConfig("draft", "target", 1, 3, 4, 1024, 8, spec_rhythm_roofline={"default": 8})
    reduce = Mock()
    monkeypatch.setattr(native_module.dist, "all_reduce", reduce)
    engine._validate_spec_rhythm_target_graph({"used_aclgraph": False}, has_target_work=True)
    reduce.assert_not_called()
