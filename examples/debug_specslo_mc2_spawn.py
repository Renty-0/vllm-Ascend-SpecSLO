# SPDX-License-Identifier: Apache-2.0
"""Launch the TP3 MC2 diagnostic with PEARL's multiprocessing-spawn model."""

from __future__ import annotations

import argparse
import os
import socket
import sys

import torch.multiprocessing as mp


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _worker(rank: int, world_size: int, port: int, forwarded: list[str]) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(world_size),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
    )
    from examples.debug_specslo_mc2_eager import main

    sys.argv = ["debug_specslo_mc2_eager.py", *forwarded]
    main()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=4)
    args, forwarded = parser.parse_known_args()
    if args.world_size <= 0:
        raise ValueError("--world-size must be positive")
    mp.spawn(
        _worker,
        args=(args.world_size, _free_port(), forwarded),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
