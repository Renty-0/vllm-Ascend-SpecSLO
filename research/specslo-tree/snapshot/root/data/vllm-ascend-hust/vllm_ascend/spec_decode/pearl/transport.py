# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small, authenticated-by-filesystem transport for the PEARL draft service."""

from __future__ import annotations

import json
import socket
import struct
from typing import Any

MAX_MESSAGE_BYTES = 16 * 1024 * 1024
_MESSAGE_SIZE_FORMAT = "!I"
_MESSAGE_SIZE_WIDTH = struct.calcsize(_MESSAGE_SIZE_FORMAT)


class PearlTransportError(RuntimeError):
    """Raised when a PEARL draft-service exchange is malformed or incomplete."""


def send_message(connection: socket.socket, message: dict[str, Any]) -> None:
    """Serialize and send one length-prefixed JSON object."""
    try:
        encoded = json.dumps(message, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PearlTransportError("PEARL messages must be JSON serializable.") from error
    if len(encoded) > MAX_MESSAGE_BYTES:
        raise PearlTransportError(f"PEARL message exceeds the {MAX_MESSAGE_BYTES}-byte transport limit.")
    connection.sendall(struct.pack(_MESSAGE_SIZE_FORMAT, len(encoded)) + encoded)


def receive_message(connection: socket.socket) -> dict[str, Any]:
    """Receive one length-prefixed JSON object."""
    header = _receive_exact(connection, _MESSAGE_SIZE_WIDTH)
    (message_size,) = struct.unpack(_MESSAGE_SIZE_FORMAT, header)
    if message_size > MAX_MESSAGE_BYTES:
        raise PearlTransportError(f"PEARL message exceeds the {MAX_MESSAGE_BYTES}-byte transport limit.")
    try:
        message = json.loads(_receive_exact(connection, message_size).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PearlTransportError("PEARL received malformed JSON.") from error
    if not isinstance(message, dict):
        raise PearlTransportError("PEARL messages must be JSON objects.")
    return message


def exchange_unix_message(
    socket_path: str,
    message: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    """Send one request to the local draft service and return its response."""
    if timeout_seconds <= 0:
        raise ValueError("PEARL transport timeout must be positive.")
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(timeout_seconds)
            connection.connect(socket_path)
            send_message(connection, message)
            return receive_message(connection)
    except (OSError, TimeoutError) as error:
        raise PearlTransportError(
            f"Unable to exchange a PEARL draft-service message through {socket_path!r}: {error}"
        ) from error


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        try:
            chunk = connection.recv(remaining)
        except (OSError, TimeoutError) as error:
            raise PearlTransportError("PEARL message receive failed.") from error
        if not chunk:
            raise PearlTransportError("PEARL peer closed the connection mid-message.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
