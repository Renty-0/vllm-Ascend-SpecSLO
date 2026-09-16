# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from examples.benchmark_nano_pearl_speculative import _build_parser
from vllm_ascend.spec_decode.pearl.api import PEARLConfig
from vllm_ascend.spec_decode.pearl.native_engine import NativePearlConfig
from vllm_ascend.spec_decode.pearl.native_model import NativeLMHead, NativeQwen2ForCausalLM, NativeTPContext


def _head(*, fast_path: bool, tp_size: int = 1) -> NativeLMHead:
    head = NativeLMHead(
        vocab_size=4,
        hidden_size=2,
        context=NativeTPContext(group=None, rank=0, size=tp_size, leader_rank=0),
        tp1_greedy_argmax=fast_path,
    )
    with torch.no_grad():
        head.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 2.0],
                    [100.0, 100.0],
                    [-100.0, -100.0],
                ]
            )[: head.vocab_size_per_rank]
        )
    return head


def test_tp1_greedy_argmax_fast_path_requests_only_token_ids():
    head = _head(fast_path=True)
    real_argmax = torch.argmax

    with patch(
        "vllm_ascend.spec_decode.pearl.native_model.torch.argmax",
        wraps=real_argmax,
    ) as argmax:
        tokens = head.greedy(torch.tensor([[1.0, 1.0]]), vocabulary_size=2)

    assert tokens.tolist() == [1]
    argmax.assert_called_once()
    assert argmax.call_args.kwargs == {"dim": -1}


def test_tp1_greedy_argmax_is_disabled_by_default():
    head = _head(fast_path=False)

    with patch(
        "vllm_ascend.spec_decode.pearl.native_model.torch.argmax",
        side_effect=AssertionError("the default path must retain max-with-value"),
    ):
        tokens = head.greedy(torch.tensor([[1.0, 1.0]]), vocabulary_size=2)

    assert tokens.tolist() == [1]


@pytest.mark.parametrize("method_name", ["greedy", "greedy_with_confidence"])
def test_tp1_greedy_skips_zero_vocabulary_offset_add(method_name):
    head = _head(fast_path=False)

    with patch.object(
        torch.Tensor,
        "__iadd__",
        side_effect=AssertionError("TP1 must not launch an add-zero operation"),
    ):
        result = getattr(head, method_name)(
            torch.tensor([[1.0, 1.0]]),
            vocabulary_size=2,
        )

    tokens = result[0] if isinstance(result, tuple) else result
    assert tokens.tolist() == [1]


def test_tp1_greedy_argmax_matches_legacy_for_vocab_crop_and_exact_tie():
    fast_head = _head(fast_path=True)
    legacy_head = _head(fast_path=False)
    with torch.no_grad():
        fast_head.weight[:2].copy_(torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
        legacy_head.weight.copy_(fast_head.weight)

    hidden_states = torch.tensor([[1.0, 0.0]])
    fast_tokens = fast_head.greedy(hidden_states, vocabulary_size=2)
    legacy_tokens = legacy_head.greedy(hidden_states, vocabulary_size=2)

    assert fast_tokens.tolist() == [0]
    assert torch.equal(fast_tokens, legacy_tokens)


def test_lm_head_rejects_argmax_fast_path_for_tensor_parallel_draft():
    with pytest.raises(ValueError, match="only for TP1"):
        _head(fast_path=True, tp_size=2)


def test_model_config_controls_tp1_greedy_argmax_and_defaults_off():
    base_config = {
        "vocab_size": 4,
        "hidden_size": 4,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "max_position_embeddings": 8,
        "rope_theta": 10_000.0,
        "rms_norm_eps": 1e-6,
        "intermediate_size": 8,
        "tie_word_embeddings": False,
        "num_hidden_layers": 1,
    }
    context = NativeTPContext(group=None, rank=0, size=1, leader_rank=0)

    ordinary = NativeQwen2ForCausalLM(SimpleNamespace(**base_config), context)
    fast = NativeQwen2ForCausalLM(
        SimpleNamespace(**base_config, pearl_tp1_greedy_argmax=True),
        context,
    )

    assert ordinary.lm_head.tp1_greedy_argmax is False
    assert fast.lm_head.tp1_greedy_argmax is True


def test_native_config_rejects_argmax_fast_path_above_tp1():
    with pytest.raises(ValueError, match="requires draft TP size 1"):
        NativePearlConfig(
            "draft",
            "target",
            2,
            1,
            4,
            512,
            32,
            draft_tp1_greedy_argmax=True,
        )


def test_public_config_maps_argmax_fast_path_to_native_runtime():
    model_config = SimpleNamespace(
        architectures=["Qwen2ForCausalLM"],
        eos_token_id=1,
    )
    with patch(
        "vllm_ascend.spec_decode.pearl.api.AutoConfig.from_pretrained",
        side_effect=[model_config, model_config],
    ):
        config = PEARLConfig(
            "draft",
            "target",
            draft_tensor_parallel_size=1,
            target_tensor_parallel_size=1,
            draft_tp1_greedy_argmax=True,
        )

    assert config.to_native().draft_tp1_greedy_argmax is True


def test_public_config_rejects_argmax_fast_path_above_tp1_before_model_load():
    with pytest.raises(ValueError, match="requires draft TP size 1"):
        PEARLConfig("draft", "target", draft_tp1_greedy_argmax=True)


def test_speculative_benchmark_cli_keeps_argmax_fast_path_opt_in():
    parser = _build_parser()
    common = [
        "--draft-model",
        "draft",
        "--target-model",
        "target",
        "--prompt",
        "hello",
    ]

    assert parser.parse_args(common).draft_tp1_greedy_argmax is False
    assert parser.parse_args([*common, "--draft-tp1-greedy-argmax"]).draft_tp1_greedy_argmax is True
