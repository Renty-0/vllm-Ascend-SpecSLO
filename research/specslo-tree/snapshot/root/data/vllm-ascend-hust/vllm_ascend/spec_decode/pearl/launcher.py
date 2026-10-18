# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch a separate PEARL draft service and a target ``vllm serve`` process."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

from vllm_ascend.spec_decode.pearl.qwen_pair import validate_model_pair
from vllm_ascend.spec_decode.pearl.transport import PearlTransportError, exchange_unix_message

CUSTOM_PROPOSER_PATH = "vllm_ascend.spec_decode.pearl.remote_proposer.PearlRemoteProposer"
DEFAULT_STARTUP_TIMEOUT_SECONDS = 180.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    target_args = list(args.target_args)
    if target_args[:1] == ["--"]:
        target_args = target_args[1:]
    if not target_args:
        parser.error("target vLLM arguments are required after '--'.")
    _reject_conflicting_target_args(target_args)

    with tempfile.TemporaryDirectory(prefix="pearl-") as temporary_directory:
        socket_path = str(Path(temporary_directory) / "draft.sock")
        draft_process = _start_draft_process(args, socket_path)
        try:
            _wait_for_draft_service(socket_path, draft_process, args.startup_timeout_seconds)
            target_process = _start_target_process(args, socket_path, target_args)
            try:
                target_process.wait()
            except KeyboardInterrupt:
                target_process.terminate()
                target_process.wait()
                raise
        finally:
            _stop_draft_process(draft_process, socket_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch a nano-PEARL draft engine and vLLM-Ascend target server.")
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--draft-devices", required=True, help="Comma-separated physical NPU IDs.")
    parser.add_argument("--draft-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--target-devices", required=True, help="Comma-separated physical NPU IDs.")
    parser.add_argument("--num-speculative-tokens", type=int, default=5)
    parser.add_argument("--request-timeout-seconds", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS)
    parser.add_argument("--startup-timeout-seconds", type=float, default=DEFAULT_STARTUP_TIMEOUT_SECONDS)
    parser.add_argument(
        "--draft-llm-kwargs",
        default="{}",
        help="JSON object forwarded to the draft vLLM.LLM constructor.",
    )
    parser.add_argument("target_args", nargs=argparse.REMAINDER)
    return parser


def _start_draft_process(args: argparse.Namespace, socket_path: str) -> subprocess.Popen[bytes]:
    try:
        draft_llm_kwargs = json.loads(args.draft_llm_kwargs)
    except json.JSONDecodeError as error:
        raise ValueError("--draft-llm-kwargs must be a JSON object.") from error
    if not isinstance(draft_llm_kwargs, dict):
        raise ValueError("--draft-llm-kwargs must be a JSON object.")
    draft_environment = _environment_with_devices(args.draft_devices)
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm_ascend.spec_decode.pearl.draft_server",
            "--model",
            args.draft_model,
            "--socket-path",
            socket_path,
            "--tensor-parallel-size",
            str(args.draft_tensor_parallel_size),
            "--llm-kwargs",
            json.dumps(draft_llm_kwargs),
        ],
        env=draft_environment,
        start_new_session=True,
    )


def _start_target_process(
    args: argparse.Namespace,
    socket_path: str,
    target_args: list[str],
) -> subprocess.Popen[bytes]:
    if args.num_speculative_tokens <= 0:
        raise ValueError("PEARL num_speculative_tokens must be positive.")
    projection = validate_model_pair(args.draft_model, _target_model_from_args(target_args))
    speculative_config = {
        "method": "custom_class",
        "model": CUSTOM_PROPOSER_PATH,
        "num_speculative_tokens": args.num_speculative_tokens,
    }
    additional_config = {
        "pearl": {
            "draft_socket_path": socket_path,
            "request_timeout_seconds": args.request_timeout_seconds,
            "draft_vocab_size": projection.draft_vocab_size,
            "target_vocab_size": projection.target_vocab_size,
        }
    }
    target_environment = _environment_with_devices(args.target_devices)
    vllm_command = Path(sys.executable).with_name("vllm")
    if not vllm_command.is_file():
        raise RuntimeError(f"PEARL could not find the vLLM console script beside {sys.executable!r}.")
    return subprocess.Popen(
        [
            str(vllm_command),
            "serve",
            *target_args,
            "--speculative-config",
            json.dumps(speculative_config),
            "--additional-config",
            json.dumps(additional_config),
        ],
        env=target_environment,
        start_new_session=True,
    )


def _wait_for_draft_service(
    socket_path: str,
    draft_process: subprocess.Popen[bytes],
    timeout_seconds: float,
) -> None:
    if timeout_seconds <= 0:
        raise ValueError("PEARL startup timeout must be positive.")
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if draft_process.poll() is not None:
            raise RuntimeError(f"PEARL draft service exited during startup with code {draft_process.returncode}.")
        try:
            response = exchange_unix_message(socket_path, {"op": "health"}, 1.0)
            if response.get("ok") is True:
                return
            last_error = RuntimeError(str(response.get("error", "draft service is not ready")))
        except PearlTransportError as error:
            last_error = error
        time.sleep(0.2)
    raise RuntimeError(f"PEARL draft service did not become ready: {last_error}")


def _stop_draft_process(draft_process: subprocess.Popen[bytes], socket_path: str) -> None:
    if draft_process.poll() is None:
        try:
            exchange_unix_message(socket_path, {"op": "shutdown"}, 1.0)
            draft_process.wait(timeout=10)
        except (PearlTransportError, subprocess.TimeoutExpired):
            draft_process.terminate()
            draft_process.wait(timeout=10)


def _environment_with_devices(device_ids: str) -> dict[str, str]:
    if not device_ids or any(not value.strip() for value in device_ids.split(",")):
        raise ValueError("PEARL NPU device lists must contain comma-separated device IDs.")
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = device_ids
    # The PEARL draft and target engines form independent HCCL worlds. Let
    # Ascend choose disjoint rendezvous ports so a concurrently running job
    # cannot make either service fail during process-group initialization.
    environment.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    return environment


def _reject_conflicting_target_args(target_args: list[str]) -> None:
    conflicting_options = {"--speculative-config", "-sc", "--additional-config"}
    for argument in target_args:
        if argument in conflicting_options or any(argument.startswith(f"{option}=") for option in conflicting_options):
            raise ValueError(f"{argument!r} is managed by the PEARL launcher and must not appear in target arguments.")


def _target_model_from_args(target_args: list[str]) -> str:
    """Resolve the target model positional/option before starting vLLM."""
    for index, argument in enumerate(target_args):
        if argument == "--model" and index + 1 < len(target_args):
            return target_args[index + 1]
        if argument.startswith("--model="):
            return argument.split("=", 1)[1]
    if target_args and not target_args[0].startswith("-"):
        return target_args[0]
    raise ValueError("PEARL target arguments must include a model path.")


if __name__ == "__main__":
    main()
