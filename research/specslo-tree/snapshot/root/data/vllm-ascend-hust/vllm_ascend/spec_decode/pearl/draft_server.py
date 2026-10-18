# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A local, token-ID based greedy draft service for PEARL speculative decoding."""

from __future__ import annotations

import argparse
import os
import socket
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from vllm_ascend.spec_decode.pearl.transport import (
    PearlTransportError,
    receive_message,
    send_message,
)

DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
LISTEN_BACKLOG = 64
MAX_DRAFT_TOKENS = 128

DraftGenerator = Callable[[list[list[int]], int], list[list[int]]]


@dataclass(frozen=True)
class DraftServerConfig:
    """Model configuration owned exclusively by the draft-service process."""

    model: str
    socket_path: str
    tensor_parallel_size: int = 1
    llm_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("A PEARL draft model is required.")
        if not self.socket_path:
            raise ValueError("A PEARL draft socket path is required.")
        if self.tensor_parallel_size <= 0:
            raise ValueError("PEARL draft tensor_parallel_size must be positive.")


def handle_request(request: dict[str, Any], generate: DraftGenerator) -> tuple[dict[str, Any], bool]:
    """Validate a draft-service request and return its response plus shutdown state."""
    operation = request.get("op")
    if operation == "health":
        return {"ok": True, "status": "ready"}, False
    if operation == "shutdown":
        return {"ok": True}, True
    if operation != "propose":
        return {"ok": False, "error": f"Unsupported PEARL operation: {operation!r}"}, False

    try:
        prompt_token_ids = _validate_prompt_rows(request.get("prompt_token_ids"))
        num_speculative_tokens = _validate_num_speculative_tokens(request.get("num_speculative_tokens"))
        candidate_token_ids = generate(prompt_token_ids, num_speculative_tokens)
        _validate_candidates(candidate_token_ids, len(prompt_token_ids), num_speculative_tokens)
    except (TypeError, ValueError) as error:
        return {"ok": False, "error": str(error)}, False

    return {"ok": True, "candidate_token_ids": candidate_token_ids}, False


def run_draft_server(config: DraftServerConfig) -> None:
    """Load the draft model and serve local proposal requests until shutdown."""
    generate = _create_greedy_generator(config)
    _serve(config.socket_path, generate)


def _create_greedy_generator(config: DraftServerConfig) -> DraftGenerator:
    # The process selects its cards before importing vLLM and torch.
    from vllm import LLM, SamplingParams

    llm_kwargs = dict(config.llm_kwargs or {})
    llm_kwargs.setdefault("enable_prefix_caching", True)
    llm = LLM(
        model=config.model,
        tensor_parallel_size=config.tensor_parallel_size,
        **llm_kwargs,
    )

    def generate(prompt_token_ids: list[list[int]], num_speculative_tokens: int) -> list[list[int]]:
        outputs = llm.generate(
            [{"prompt_token_ids": token_ids} for token_ids in prompt_token_ids],
            SamplingParams(temperature=0.0, max_tokens=num_speculative_tokens),
            use_tqdm=False,
        )
        return [list(output.outputs[0].token_ids) if output.outputs else [] for output in outputs]

    return generate


def _serve(socket_path: str, generate: DraftGenerator) -> None:
    if os.path.exists(socket_path):
        raise FileExistsError(f"Refusing to replace existing PEARL draft socket {socket_path!r}.")

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    owns_socket = False
    try:
        listener.bind(socket_path)
        owns_socket = True
        os.chmod(socket_path, 0o600)
        listener.listen(LISTEN_BACKLOG)
        should_shutdown = False
        while not should_shutdown:
            connection, _ = listener.accept()
            with connection:
                try:
                    request = receive_message(connection)
                    response, should_shutdown = handle_request(request, generate)
                except PearlTransportError as error:
                    response = {"ok": False, "error": str(error)}
                except Exception as error:  # Keep model failures visible to the target worker.
                    response = {"ok": False, "error": f"Draft model failed: {error}"}
                send_message(connection, response)
    finally:
        listener.close()
        if owns_socket and os.path.exists(socket_path):
            os.unlink(socket_path)


def _validate_prompt_rows(value: Any) -> list[list[int]]:
    if not isinstance(value, list) or not value:
        raise ValueError("PEARL proposal requests require a non-empty prompt_token_ids list.")
    prompt_rows: list[list[int]] = []
    for row in value:
        if not isinstance(row, list) or not row:
            raise ValueError("Every PEARL proposal prompt must be a non-empty token-id list.")
        if any(not isinstance(token_id, int) or token_id < 0 for token_id in row):
            raise ValueError("PEARL proposal prompt token IDs must be non-negative integers.")
        prompt_rows.append(row)
    return prompt_rows


def _validate_num_speculative_tokens(value: Any) -> int:
    if not isinstance(value, int) or not 0 < value <= MAX_DRAFT_TOKENS:
        raise ValueError(f"num_speculative_tokens must be an integer in [1, {MAX_DRAFT_TOKENS}].")
    return value


def _validate_candidates(
    candidates: Sequence[Sequence[int]],
    batch_size: int,
    num_speculative_tokens: int,
) -> None:
    if not isinstance(candidates, list):
        raise ValueError("Draft model must return a list of candidate-token rows.")
    if len(candidates) != batch_size:
        raise ValueError("Draft model returned a different number of proposal rows.")
    for row in candidates:
        if not isinstance(row, list) or len(row) > num_speculative_tokens:
            raise ValueError("Draft model returned an invalid candidate-token window.")
        if any(not isinstance(token_id, int) or token_id < 0 for token_id in row):
            raise ValueError("Draft model returned invalid token IDs.")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a local nano-PEARL draft service.")
    parser.add_argument("--model", required=True, help="Draft model path or Hugging Face identifier.")
    parser.add_argument("--socket-path", required=True, help="Private Unix socket path for the target worker.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--llm-kwargs",
        default="{}",
        help="JSON object forwarded to vLLM.LLM for the draft model.",
    )
    args = parser.parse_args(argv)

    import json

    try:
        llm_kwargs = json.loads(args.llm_kwargs)
    except json.JSONDecodeError as error:
        raise ValueError("--llm-kwargs must be a JSON object.") from error
    if not isinstance(llm_kwargs, dict):
        raise ValueError("--llm-kwargs must be a JSON object.")
    run_draft_server(
        DraftServerConfig(
            model=args.model,
            socket_path=args.socket_path,
            tensor_parallel_size=args.tensor_parallel_size,
            llm_kwargs=llm_kwargs,
        )
    )


if __name__ == "__main__":
    main()
