# SPDX-License-Identifier: Apache-2.0
"""Derive smaller real-input MC2 captures from a qualified larger row set.

The projection weight and RMSNorm gamma are row-invariant.  Activation and
residual rows are independent batch rows, so taking a contiguous prefix from
an actual model-runner capture produces a real-input smaller-M qualification
case without loading Qwen3-32B again.  The emitted manifest binds every output
to the source payload hash and records the exact slice used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--kind", choices=("attention", "down"), required=True)
    parser.add_argument("--source-rows", type=int, required=True)
    parser.add_argument("--rows", type=int, nargs="+", required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--ranks", type=int, default=3)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate(args: argparse.Namespace) -> None:
    if args.source_rows <= 0 or args.ranks <= 0:
        raise ValueError("source rows and rank count must be positive")
    if not args.rows or len(set(args.rows)) != len(args.rows):
        raise ValueError("rows must be a non-empty unique list")
    if min(args.rows) <= 0 or max(args.rows) >= args.source_rows:
        raise ValueError("derived rows must be positive and smaller than source rows")
    if not args.layers or min(args.layers) < 0 or len(set(args.layers)) != len(args.layers):
        raise ValueError("layers must be a non-empty unique non-negative list")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate(args)
    destination_kind = args.output_root / args.kind
    if destination_kind.exists():
        raise FileExistsError(f"refusing to overwrite existing capture kind: {destination_kind}")

    records = []
    for layer in args.layers:
        for rank in range(args.ranks):
            source = (
                args.source_root
                / args.kind
                / f"layer-{layer}"
                / f"m{args.source_rows}"
                / f"rank-{rank}.pt"
            )
            if not source.is_file():
                raise FileNotFoundError(f"missing source capture: {source}")
            payload = torch.load(source, map_location="cpu", weights_only=True)
            expected_keys = {"activation", "weight", "residual", "gamma"}
            if set(payload) != expected_keys or any(
                not torch.is_tensor(payload[key]) for key in expected_keys
            ):
                raise ValueError(f"invalid capture payload: {source}")
            activation = payload["activation"]
            residual = payload["residual"]
            if activation.ndim != 2 or residual.ndim != 2:
                raise ValueError(f"capture rows must be two-dimensional: {source}")
            if activation.shape[0] != args.source_rows or residual.shape[0] != args.source_rows:
                raise ValueError(f"capture source row count does not match: {source}")
            source_hash = _sha256(source)
            for rows in args.rows:
                destination = (
                    destination_kind
                    / f"layer-{layer}"
                    / f"m{rows}"
                    / f"rank-{rank}.pt"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                derived = {
                    "activation": activation[:rows].contiguous(),
                    "weight": payload["weight"].contiguous(),
                    "residual": residual[:rows].contiguous(),
                    "gamma": payload["gamma"].contiguous(),
                }
                temporary = destination.with_suffix(".pt.tmp")
                torch.save(derived, temporary)
                temporary.replace(destination)
                records.append(
                    {
                        "kind": args.kind,
                        "layer": layer,
                        "rank": rank,
                        "rows": rows,
                        "source": str(source.resolve()),
                        "source_sha256": source_hash,
                        "slice": f"[0:{rows}]",
                        "output": str(destination.resolve()),
                        "output_sha256": _sha256(destination),
                    }
                )

    manifest = {
        "schema_version": 1,
        "derivation": "contiguous-prefix-of-real-model-runner-capture",
        "kind": args.kind,
        "source_rows": args.source_rows,
        "rows": args.rows,
        "layers": args.layers,
        "ranks": args.ranks,
        "records": records,
    }
    manifest_path = args.output_root / f"{args.kind}-derivation-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
