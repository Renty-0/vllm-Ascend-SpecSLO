# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preflight the Qwen2.5 PEARL vocabulary-prefix configuration.

This validates the Qwen2.5-0.5B-Instruct draft and Qwen2.5-14B-Instruct
target pair, then prints the vLLM speculative configuration that enables the
Ascend PEARL vocabulary projection. It intentionally does not start a server:
the caller owns NPU allocation and the normal ``vllm serve`` lifecycle.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from transformers import AutoConfig, AutoTokenizer

from vllm_ascend.spec_decode.pearl.vocab import PearlVocabProjection

DEFAULT_NUM_SPECULATIVE_TOKENS = 4
DEFAULT_TENSOR_PARALLEL_SIZE = 2


def build_speculative_config(
    draft_model: str,
    num_speculative_tokens: int = DEFAULT_NUM_SPECULATIVE_TOKENS,
    tensor_parallel_size: int = DEFAULT_TENSOR_PARALLEL_SIZE,
) -> dict[str, object]:
    """Return the vLLM configuration for a shared-prefix draft-model pair."""
    if num_speculative_tokens <= 0:
        raise ValueError("The number of PEARL speculative tokens must be positive.")
    if tensor_parallel_size <= 0:
        raise ValueError("The PEARL tensor-parallel size must be positive.")
    return {
        "method": "draft_model",
        "model": draft_model,
        "num_speculative_tokens": num_speculative_tokens,
        "draft_tensor_parallel_size": tensor_parallel_size,
        "use_heterogeneous_vocab": True,
        "draft_sample_method": "greedy",
    }


def validate_model_pair(draft_model: str, target_model: str) -> PearlVocabProjection:
    """Validate that target token IDs contain the complete draft prefix."""
    draft_config = AutoConfig.from_pretrained(draft_model)
    target_config = AutoConfig.from_pretrained(target_model)
    return PearlVocabProjection.from_tokenizers(
        draft_tokenizer=AutoTokenizer.from_pretrained(draft_model),
        target_tokenizer=AutoTokenizer.from_pretrained(target_model),
        draft_vocab_size=draft_config.vocab_size,
        target_vocab_size=target_config.vocab_size,
    )


def validate_tensor_parallel_size(draft_model: str, target_model: str, tensor_parallel_size: int) -> None:
    """Ensure the shared vLLM draft-model baseline can shard both models."""
    if tensor_parallel_size <= 0:
        raise ValueError("The PEARL tensor-parallel size must be positive.")
    for role, model_path in (("draft", draft_model), ("target", target_model)):
        num_attention_heads = AutoConfig.from_pretrained(model_path).num_attention_heads
        if num_attention_heads % tensor_parallel_size:
            raise ValueError(
                f"The {role} model has {num_attention_heads} attention heads, which "
                f"is not divisible by tensor_parallel_size={tensor_parallel_size}."
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--num-speculative-tokens", type=int, default=DEFAULT_NUM_SPECULATIVE_TOKENS)
    parser.add_argument("--tensor-parallel-size", type=int, default=DEFAULT_TENSOR_PARALLEL_SIZE)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    projection = validate_model_pair(args.draft_model, args.target_model)
    validate_tensor_parallel_size(args.draft_model, args.target_model, args.tensor_parallel_size)
    speculative_config = build_speculative_config(
        args.draft_model,
        args.num_speculative_tokens,
        args.tensor_parallel_size,
    )
    print(
        json.dumps(
            {
                "draft_vocab_size": projection.draft_vocab_size,
                "target_vocab_size": projection.target_vocab_size,
                "target_vocab_suffix_tokens_removed": projection.target_vocab_size - projection.draft_vocab_size,
                "speculative_config": speculative_config,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
