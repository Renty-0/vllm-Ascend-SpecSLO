# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from examples.benchmark_nano_pearl_speculative import _build_parser as build_benchmark_parser
from vllm_ascend.spec_decode.pearl import api as pearl_api
from vllm_ascend.spec_decode.pearl import native_engine
from vllm_ascend.spec_decode.pearl.api import PEARLConfig, PEARLEngine
from vllm_ascend.spec_decode.pearl.native_engine import (
    NativePearlConfig,
    NativePearlEngine,
)
from vllm_ascend.spec_decode.pearl.native_engine import (
    _build_parser as build_native_parser,
)
from vllm_ascend.spec_decode.pearl.pard import (
    PARD_PARALLEL_DRAFT_MODE,
    PARD_PARALLEL_GAMMA,
    SERIAL_LINEAR_DRAFT_MODE,
    build_pard_parallel_draft_layout,
    gather_pard_parallel_proposals,
    require_experimental_pard_eager,
    validate_pard_parallel_model_pair,
)

PARD_TOKEN = 151670
VOCAB_SIZE = 151936


def _model_config(
    *,
    pard: bool,
    architecture: str = "Qwen3ForCausalLM",
    model_type: str = "qwen3",
    vocab_size: int = VOCAB_SIZE,
):
    values = {
        "architectures": [architecture],
        "eos_token_id": 151645,
        "model_type": model_type,
        "vocab_size": vocab_size,
    }
    if pard:
        values.update(spd_type="pard", pard_token=PARD_TOKEN)
    return SimpleNamespace(**values)


def _native_config(**overrides) -> NativePearlConfig:
    values = {
        "draft_model": "draft",
        "target_model": "target",
        "draft_tp_size": 1,
        "target_tp_size": 3,
        "gamma": PARD_PARALLEL_GAMMA,
        "max_model_len": 512,
        "max_tokens": 32,
    }
    values.update(overrides)
    return NativePearlConfig(**values)


def _qualified_pard_overrides() -> dict[str, object]:
    return {
        "draft_mode": PARD_PARALLEL_DRAFT_MODE,
        "gamma": PARD_PARALLEL_GAMMA,
        "enable_continuous_batching": True,
        "enable_preemptive_scheduling": True,
        "enable_spec_rhythm": True,
        "spec_rhythm_linear_full_window": True,
        "spec_rhythm_min_gamma": PARD_PARALLEL_GAMMA,
        "spec_rhythm_tree_width": 1,
        "spec_rhythm_tree_depth": 1,
        "spec_rhythm_max_eager_tokens": 0,
        "spec_rhythm_eager_reserve_tokens": 0,
        "spec_rhythm_linear_eager_cross_graph_bucket": False,
        "spec_rhythm_linear_idle_residual_eager": False,
        "spec_rhythm_linear_bonus_token": False,
        "precompile_decode_graphs": False,
        "precompile_serial_draft_graphs": False,
    }


def test_pard_layout_is_request_isolated_and_gathers_fixed_b4() -> None:
    layout = build_pard_parallel_draft_layout(
        [[10, 11], [20]],
        [3, 7],
        [4, 9],
        pard_token=PARD_TOKEN,
        vocab_size=VOCAB_SIZE,
    )

    assert layout.input_token_ids == (
        10,
        11,
        PARD_TOKEN,
        PARD_TOKEN,
        PARD_TOKEN,
        20,
        PARD_TOKEN,
        PARD_TOKEN,
        PARD_TOKEN,
    )
    assert layout.sequence_ids == (3, 3, 3, 3, 3, 7, 7, 7, 7)
    assert layout.positions == (4, 5, 6, 7, 8, 9, 10, 11, 12)
    assert layout.query_lengths == (5, 4)
    assert layout.sample_row_indices == (1, 2, 3, 4, 5, 6, 7, 8)
    assert layout.proposal_shape == (2, 4)
    assert layout.attention_allowed(4, 1)
    assert not layout.attention_allowed(1, 4)
    assert not layout.attention_allowed(5, 4)
    assert not layout.attention_allowed(4, 5)

    packed_predictions = tuple(range(100, 109))
    assert gather_pard_parallel_proposals(packed_predictions, layout) == (
        (101, 102, 103, 104),
        (105, 106, 107, 108),
    )

    changed = build_pard_parallel_draft_layout(
        [[10, 11], [88, 89, 90]],
        [3, 7],
        [4, 20],
        pard_token=PARD_TOKEN,
        vocab_size=VOCAB_SIZE,
    )
    first_query_size = layout.query_lengths[0]
    assert changed.input_token_ids[:first_query_size] == layout.input_token_ids[:first_query_size]
    assert changed.sequence_ids[:first_query_size] == layout.sequence_ids[:first_query_size]
    assert changed.positions[:first_query_size] == layout.positions[:first_query_size]
    assert changed.sample_row_indices[:4] == layout.sample_row_indices[:4]


@pytest.mark.parametrize(
    ("suffixes", "sequence_ids", "first_positions", "pard_token", "vocab_size", "message"),
    [
        ([], [], [], PARD_TOKEN, VOCAB_SIZE, "at least one request"),
        ([[1], []], [0, 1], [0, 0], PARD_TOKEN, VOCAB_SIZE, "current committed root"),
        ([[1], [2]], [0, 0], [0, 0], PARD_TOKEN, VOCAB_SIZE, "unique"),
        ([[1]], [0], [-1], PARD_TOKEN, VOCAB_SIZE, "first position"),
        ([[VOCAB_SIZE]], [0], [0], PARD_TOKEN, VOCAB_SIZE, "outside"),
        ([[1]], [0], [0], VOCAB_SIZE, VOCAB_SIZE, "inside"),
    ],
)
def test_pard_layout_rejects_invalid_request_contracts(
    suffixes,
    sequence_ids,
    first_positions,
    pard_token,
    vocab_size,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_pard_parallel_draft_layout(
            suffixes,
            sequence_ids,
            first_positions,
            pard_token=pard_token,
            vocab_size=vocab_size,
        )


def test_pard_layout_rejects_non_gamma4() -> None:
    with pytest.raises(ValueError, match="fixed gamma=4"):
        build_pard_parallel_draft_layout(
            [[1]],
            [0],
            [0],
            pard_token=PARD_TOKEN,
            vocab_size=VOCAB_SIZE,
            gamma=3,
        )


@pytest.mark.parametrize(
    ("draft", "target", "message"),
    [
        (_model_config(pard=False), _model_config(pard=False), "spd_type"),
        (
            SimpleNamespace(
                architectures=["Qwen3ForCausalLM"],
                model_type="qwen3",
                spd_type="pard",
                vocab_size=VOCAB_SIZE,
            ),
            _model_config(pard=False),
            "pard_token",
        ),
        (_model_config(pard=True, architecture="Qwen2ForCausalLM"), _model_config(pard=False), "Qwen3 draft"),
        (_model_config(pard=True), _model_config(pard=False, architecture="LlamaForCausalLM"), "Qwen3 target"),
        (_model_config(pard=True, model_type="qwen2"), _model_config(pard=False), "Qwen3 draft"),
        (_model_config(pard=True), _model_config(pard=False, model_type="qwen2"), "Qwen3 target"),
        (_model_config(pard=True), _model_config(pard=False, vocab_size=VOCAB_SIZE - 1), "identical"),
        (
            SimpleNamespace(
                architectures=["Qwen3ForCausalLM"],
                model_type="qwen3",
                spd_type="pard",
                pard_token=VOCAB_SIZE,
                vocab_size=VOCAB_SIZE,
            ),
            _model_config(pard=False),
            "shared vocabulary",
        ),
    ],
)
def test_pard_model_pair_validation_rejects_invalid_configs(draft, target, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_pard_parallel_model_pair(draft, target, gamma=PARD_PARALLEL_GAMMA)


def test_pard_model_pair_validation_returns_internal_mask_id() -> None:
    assert (
        validate_pard_parallel_model_pair(
            _model_config(pard=True),
            _model_config(pard=False),
            gamma=PARD_PARALLEL_GAMMA,
        )
        == PARD_TOKEN
    )


def test_native_config_defaults_serial_and_rejects_bad_modes() -> None:
    assert _native_config().draft_mode == SERIAL_LINEAR_DRAFT_MODE
    with pytest.raises(ValueError, match="Unknown native draft mode"):
        _native_config(draft_mode="unknown")
    with pytest.raises(ValueError, match="fixed gamma=4"):
        _native_config(draft_mode=PARD_PARALLEL_DRAFT_MODE, gamma=3)


def test_api_config_validates_pard_and_execution_fails_before_worker_start(monkeypatch) -> None:
    monkeypatch.delenv(
        "VLLM_ASCEND_SPECSLO_ENABLE_EXPERIMENTAL_PARD_EAGER",
        raising=False,
    )
    configs = {
        "draft": _model_config(pard=True),
        "target": _model_config(pard=False),
    }
    monkeypatch.setattr(pearl_api.AutoConfig, "from_pretrained", lambda path: configs[path])
    config = PEARLConfig(
        "draft",
        "target",
        draft_tensor_parallel_size=1,
        target_tensor_parallel_size=3,
        gamma=PARD_PARALLEL_GAMMA,
        draft_mode=PARD_PARALLEL_DRAFT_MODE,
    )
    assert config.pard_token == PARD_TOKEN
    assert config.to_native().draft_mode == PARD_PARALLEL_DRAFT_MODE
    with pytest.raises(RuntimeError, match="experimental and disabled"):
        PEARLEngine(config)


def test_native_engine_pard_fails_before_npu_initialization(monkeypatch) -> None:
    monkeypatch.delenv(
        "VLLM_ASCEND_SPECSLO_ENABLE_EXPERIMENTAL_PARD_EAGER",
        raising=False,
    )
    configs = {
        "draft": _model_config(pard=True),
        "target": _model_config(pard=False),
    }
    monkeypatch.setattr(native_engine.AutoConfig, "from_pretrained", lambda path: configs[path])
    with pytest.raises(RuntimeError, match="experimental and disabled"):
        NativePearlEngine(_native_config(draft_mode=PARD_PARALLEL_DRAFT_MODE))


def test_experimental_pard_gate_allows_only_the_strict_envelope() -> None:
    config = SimpleNamespace(**_qualified_pard_overrides())
    require_experimental_pard_eager(config, enabled=True)

    invalid = (
        ("gamma", 3, "gamma=4"),
        ("enable_spec_rhythm", False, "enable_spec_rhythm"),
        ("spec_rhythm_linear_full_window", False, "linear_full_window"),
        ("spec_rhythm_min_gamma", 1, "min_gamma"),
        ("spec_rhythm_tree_width", 2, "1x1"),
        ("spec_rhythm_max_eager_tokens", 4, "rolling eager"),
        ("spec_rhythm_eager_reserve_tokens", 4, "rolling eager"),
        ("spec_rhythm_linear_idle_residual_eager", True, "rolling eager"),
        ("spec_rhythm_linear_bonus_token", True, "bonus token"),
        ("precompile_serial_draft_graphs", True, "serial draft graph precompilation"),
    )
    for field, value, message in invalid:
        values = _qualified_pard_overrides()
        values[field] = value
        with pytest.raises(ValueError, match=message):
            require_experimental_pard_eager(SimpleNamespace(**values), enabled=True)


def test_experimental_pard_gate_allows_target_only_graph_precompilation() -> None:
    values = _qualified_pard_overrides()
    values["precompile_decode_graphs"] = True

    require_experimental_pard_eager(SimpleNamespace(**values), enabled=True)


def test_api_experimental_pard_gate_reaches_later_construction_with_opt_in(monkeypatch) -> None:
    configs = {
        "draft": _model_config(pard=True),
        "target": _model_config(pard=False),
    }
    monkeypatch.setenv("VLLM_ASCEND_SPECSLO_ENABLE_EXPERIMENTAL_PARD_EAGER", "1")
    monkeypatch.setattr(pearl_api.AutoConfig, "from_pretrained", lambda path: configs[path])
    monkeypatch.setattr(pearl_api, "validate_model_pair", lambda *_args: None)
    monkeypatch.setattr(pearl_api, "_set_default_npu_environment", lambda *_args: None)
    monkeypatch.setattr(
        pearl_api.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    start_workers = MagicMock()
    monkeypatch.setattr(PEARLEngine, "_start_workers", start_workers)
    config = PEARLConfig(
        "draft",
        "target",
        draft_tensor_parallel_size=1,
        target_tensor_parallel_size=3,
        **_qualified_pard_overrides(),
    )

    engine = PEARLEngine(config)
    try:
        start_workers.assert_called_once_with()
        with pytest.raises(ValueError, match="greedy target and draft"):
            engine.add_request(
                [1],
                native_engine.SamplingParams(temperature=0.1),
            )
    finally:
        engine.exit()


def test_native_experimental_pard_gate_reaches_device_binding_with_opt_in(monkeypatch) -> None:
    class ReachedDeviceBinding(RuntimeError):
        pass

    configs = {
        "draft": _model_config(pard=True),
        "target": _model_config(pard=False),
    }
    monkeypatch.setenv("VLLM_ASCEND_SPECSLO_ENABLE_EXPERIMENTAL_PARD_EAGER", "1")
    monkeypatch.setattr(native_engine.AutoConfig, "from_pretrained", lambda path: configs[path])
    monkeypatch.setattr(
        native_engine.torch.npu,
        "set_device",
        lambda _rank: (_ for _ in ()).throw(ReachedDeviceBinding("after PARD gate")),
    )
    with pytest.raises(ReachedDeviceBinding, match="after PARD gate"):
        NativePearlEngine(_native_config(**_qualified_pard_overrides()))


def test_benchmark_cli_exposes_explicit_draft_mode() -> None:
    parser = build_benchmark_parser()
    common = ["--draft-model", "draft", "--target-model", "target", "--prompt", "hello"]
    default_args = parser.parse_args(common)
    pard_args = parser.parse_args(
        [
            *common,
            "--draft-mode",
            PARD_PARALLEL_DRAFT_MODE,
        ]
    )
    assert default_args.draft_mode == SERIAL_LINEAR_DRAFT_MODE
    assert pard_args.draft_mode == PARD_PARALLEL_DRAFT_MODE


def test_native_cli_exposes_explicit_draft_mode() -> None:
    parser = build_native_parser()
    common = ["--draft-model", "draft", "--target-model", "target"]
    default_args = parser.parse_args(common)
    pard_args = parser.parse_args([*common, "--draft-mode", PARD_PARALLEL_DRAFT_MODE])
    assert default_args.draft_mode == SERIAL_LINEAR_DRAFT_MODE
    assert pard_args.draft_mode == PARD_PARALLEL_DRAFT_MODE


@pytest.mark.parametrize("accepted", range(PARD_PARALLEL_GAMMA + 1))
def test_pard_verdict_keeps_mask_kv_behind_the_real_frontier(accepted: int) -> None:
    prefix = [10, 11, 12]
    proposal = [20, 21, 22, 23]
    correction = None if accepted == PARD_PARALLEL_GAMMA else 90 + accepted
    state = native_engine.PearlPipelineState(
        prefix.copy(),
        prompt_length=2,
        committed_length=len(prefix),
        draft_synced_length=2,
    )

    suffix, first_position = state.pard_repair_suffix()
    assert suffix == [12]
    assert first_position == 2
    state.mark_pard_draft_repaired()
    state.token_ids.extend(proposal)
    state.apply_pard_full_window_verification(
        proposal_token_ids=proposal,
        accepted=accepted,
        correction_token_id=correction,
    )

    expected = [*prefix, *proposal[:accepted]]
    if correction is not None:
        expected.append(correction)
    assert state.token_ids == expected
    assert state.committed_length == len(expected)
    # Only the real root repair advances physical draft KV.  Proposal mask
    # rows are disposable and never become part of the synchronized frontier.
    assert state.draft_synced_length == len(prefix)


def test_pard_eager_batch_uses_one_forward_and_isolates_requests() -> None:
    states = [
        native_engine.PearlPipelineState(
            [10, 11, 12],
            prompt_length=2,
            committed_length=3,
            draft_synced_length=2,
        ),
        native_engine.PearlPipelineState(
            [30, 31, 32, 33],
            prompt_length=2,
            committed_length=4,
            draft_synced_length=2,
        ),
    ]
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.gamma = PARD_PARALLEL_GAMMA
    engine.pard_token = PARD_TOKEN
    engine.draft_vocab_size = VOCAB_SIZE
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(
        draft_mode=PARD_PARALLEL_DRAFT_MODE,
        enforce_eager=True,
        draft_use_paged_attention=True,
    )
    engine._run_packed_hidden = MagicMock(return_value=torch.arange(100, 108, dtype=torch.float32).unsqueeze(1))
    engine.model = SimpleNamespace(compute_greedy_tokens=lambda hidden, _vocab_size: hidden[:, 0].to(torch.long))

    verification, proposals, confidence = engine._draft_pard_parallel_eager_batch(
        states,
        [0, 1],
    )

    assert engine._run_packed_hidden.call_count == 1
    call = engine._run_packed_hidden.call_args
    assert call.args[0] == [
        12,
        PARD_TOKEN,
        PARD_TOKEN,
        PARD_TOKEN,
        32,
        33,
        PARD_TOKEN,
        PARD_TOKEN,
        PARD_TOKEN,
    ]
    assert call.args[1] == [0, 0, 0, 0, 1, 1, 1, 1, 1]
    assert call.args[2] == [2, 3, 4, 5, 2, 3, 4, 5, 6]
    assert call.kwargs == {
        "use_aclgraph": False,
        "logit_indices": [0, 1, 2, 3, 5, 6, 7, 8],
        "use_fused_infer_attention": False,
    }
    assert proposals.tolist() == [[100, 101, 102, 103], [104, 105, 106, 107]]
    assert verification.tolist() == [100, 101, 102, 103, 104, 105, 106, 107]
    assert confidence.tolist() == [1.0, 1.0]
    assert states[0].draft_synced_length == 3
    assert states[1].draft_synced_length == 4
    assert states[0].token_ids == [10, 11, 12]
    assert states[1].token_ids == [30, 31, 32, 33]
    assert engine._pard_parallel_eager_forward_calls == 1
    assert engine._pard_parallel_repair_rows == 3
    assert engine._pard_parallel_mask_rows == 6
    assert engine._pard_parallel_serial_fallback_calls == 0


def test_pard_mode_never_falls_back_to_the_legacy_serial_loop() -> None:
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.gamma = PARD_PARALLEL_GAMMA
    engine.config = SimpleNamespace(draft_mode=PARD_PARALLEL_DRAFT_MODE)
    with pytest.raises(RuntimeError, match="legacy cross-window"):
        engine._draft_round_device_batch(
            [native_engine.PearlPipelineState([1, 2], prompt_length=1)],
            [0],
        )
