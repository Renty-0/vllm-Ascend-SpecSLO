# SPDX-License-Identifier: Apache-2.0
"""Qualify many real-input MC2 jobs in one persistent TP3 HCCL group.

This driver deliberately delegates every numerical, changed-input and latency
gate to :mod:`measure_specslo_rank3_reduce_add_rmsnorm`.  It only owns the job
schedule, per-job status ledger and lifetime of the shared process group.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch.distributed as dist

from examples import measure_specslo_rank3_reduce_add_rmsnorm as measurement

_PROJECTION_K = {"attention": 3072, "down": 8576}
_QUALIFICATION_LAYERS = (0, 63)
_SCREENING_LAYER = 31
_PRODUCTION_LAYERS = (0, 31, 63)
_DEVICE_FAILURE_PATTERNS = re.compile(
    "|".join(
        (
            "Execution_Error_SDMA",
            "vector core exception",
            "DDR address of the MTE",
            "error code is 507035",
            "aicpu execute failed",
            "HcclAllocComResourceByTiling",
            "runtime result = 507018",
            "HCCL",
        )
    ),
    flags=re.IGNORECASE,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-root", type=Path, required=True)
    parser.add_argument("--down-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rows", type=str, required=True)
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--warmup-replays", type=int, default=40)
    parser.add_argument("--replays-per-sample", type=int, default=50)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--changed-input-replays", type=int, default=1024)
    parser.add_argument("--changed-input-delta", type=float, default=0.015625)
    parser.add_argument("--latency-percentile", type=float, default=95.0)
    parser.add_argument("--minimum-speedup", type=float, default=1.02)
    return parser


def _parse_rows(value: str) -> tuple[int, ...]:
    pieces = tuple(part.strip() for part in value.split(","))
    if not pieces or any(re.fullmatch(r"[1-9][0-9]*", part) is None for part in pieces):
        raise ValueError("rows must be a comma-separated list of positive integers")
    rows = tuple(int(part) for part in pieces)
    if len(set(rows)) != len(rows):
        raise ValueError("rows must be unique")
    return rows


def _job_args(
    driver_args: argparse.Namespace,
    *,
    kind: str,
    layer: int,
    row: int,
) -> argparse.Namespace:
    output_dir = driver_args.run_dir / "measurements" / kind / f"layer-{layer}" / f"m{row}"
    input_root = driver_args.attention_root if kind == "attention" else driver_args.down_root
    return argparse.Namespace(
        m=row,
        k=_PROJECTION_K[kind],
        n=5120,
        input_dir=input_root / kind / f"layer-{layer}" / f"m{row}",
        warmup_replays=driver_args.warmup_replays,
        replays_per_sample=driver_args.replays_per_sample,
        samples=driver_args.samples,
        changed_input_replays=driver_args.changed_input_replays,
        changed_input_delta=driver_args.changed_input_delta,
        execution_mode="graph",
        operations_per_graph=1,
        chained_flush=True,
        output=output_dir / "raw.json",
        production_profile_output=output_dir / "profile.json",
        latency_percentile=driver_args.latency_percentile,
        minimum_speedup=driver_args.minimum_speedup,
    )


def _status_text(value: str) -> str:
    return " ".join(value.replace("\t", " ").splitlines()).strip()


def _rank_zero_action(action) -> None:
    outcome = None

    def finalize() -> None:
        action()

    if dist.get_rank() == 0:
        outcome = measurement._capture_rank_zero_finalization(finalize)
    outcome = measurement._broadcast_rank_zero_finalization(outcome)
    failure = outcome.fatal_failure or outcome.admission_failure
    if failure is not None:
        raise RuntimeError(failure)


def _write_status(status_file: Path, kind: str, layer: int, row: int, status: str) -> None:
    def write() -> None:
        with status_file.open("a", encoding="utf-8") as stream:
            stream.write(f"{kind}\t{layer}\t{row}\t{_status_text(status)}\n")

    _rank_zero_action(write)


def _record_current_job(run_dir: Path, kind: str, layer: int, row: int) -> None:
    _rank_zero_action(
        lambda: (run_dir / "current-job.tsv").write_text(
            f"{kind}\t{layer}\t{row}\n",
            encoding="utf-8",
        )
    )


def _run_job(
    driver_args: argparse.Namespace,
    runtime: measurement.MeasurementRuntime,
    status_file: Path,
    *,
    kind: str,
    layer: int,
    row: int,
) -> bool:
    args = _job_args(driver_args, kind=kind, layer=layer, row=row)
    _record_current_job(driver_args.run_dir, kind, layer, row)
    try:
        failure = measurement.run_measurement_job(args, runtime)
        # Cleanup is safe only after the measured graph returned normally.  A
        # device/HCCL exception may poison the runtime, so that path skips all
        # further NPU API calls and lets torchrun terminate the peers.
        measurement.release_measurement_job_resources()
    except Exception as error:
        category = "device-fail" if _DEVICE_FAILURE_PATTERNS.search(str(error)) else "fatal"
        # Avoid another collective after a device error.  Each rank that
        # catches the exception writes an independent crash breadcrumb; the
        # launcher reconciles it with current-job.tsv after torchrun exits.
        try:
            fatal_file = driver_args.run_dir / f"fatal-rank-{runtime.rank}.tsv"
            with fatal_file.open("a", encoding="utf-8") as stream:
                stream.write(f"{kind}\t{layer}\t{row}\t{category}:{type(error).__name__}:{_status_text(str(error))}\n")
        except OSError:
            pass
        # Do not try another collective here: a device/HCCL exception may have
        # invalidated this rank's communicator.  Raising lets torchrun stop all
        # peers instead of attempting another shape with a poisoned group.
        raise

    # Do not commit a pass/reject ledger row until every rank has also survived
    # cleanup.  If this synchronization fails, the launcher's campaign-fatal
    # record must not be masked by an earlier healthy-looking job status.
    dist.barrier()
    if failure is None:
        _write_status(status_file, kind, layer, row, "pass")
        passed = True
    else:
        _write_status(status_file, kind, layer, row, f"fail:{failure}")
        passed = False
    return passed


def _write_rows(path: Path, rows: list[int]) -> None:
    path.write_text(" ".join(str(row) for row in rows) + "\n", encoding="utf-8")


def _projection_row_intersection(
    projection_rows: Mapping[str, Sequence[int]],
) -> list[int]:
    """Return rows qualified by every projection, for legacy text artifacts."""

    row_sets = [set(projection_rows[kind]) for kind in _PROJECTION_K]
    if not row_sets:
        return []
    return sorted(set.intersection(*row_sets))


def _write_projection_rows(
    path: Path,
    *,
    stage: str,
    qualification_layers: Sequence[int],
    projection_rows: Mapping[str, Sequence[int]],
) -> None:
    normalized = {
        kind: sorted(int(row) for row in projection_rows[kind])
        for kind in _PROJECTION_K
    }
    document = {
        "schema_version": 1,
        "kind": "specslo_mc2_projection_rows",
        "stage": stage,
        "execution_mode": "graph",
        "operations_per_graph": 1,
        "qualification_layers": [int(layer) for layer in qualification_layers],
        "projection_k": dict(_PROJECTION_K),
        "projection_rows": normalized,
        "intersection_rows": _projection_row_intersection(normalized),
    }
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _screen_projection_rows(
    args: argparse.Namespace,
    runtime: measurement.MeasurementRuntime,
    status_file: Path,
    rows: Sequence[int],
) -> dict[str, list[int]]:
    screened = {kind: [] for kind in _PROJECTION_K}
    for row in rows:
        for kind in _PROJECTION_K:
            if _run_job(
                args,
                runtime,
                status_file,
                kind=kind,
                layer=_SCREENING_LAYER,
                row=row,
            ):
                screened[kind].append(row)
    return screened


def _qualify_projection_rows(
    args: argparse.Namespace,
    runtime: measurement.MeasurementRuntime,
    status_file: Path,
    screened: Mapping[str, Sequence[int]],
    rows: Sequence[int],
) -> dict[str, list[int]]:
    """Qualify each projection/row independently on the remaining layers."""

    qualified = {kind: [] for kind in _PROJECTION_K}
    screened_sets = {kind: set(values) for kind, values in screened.items()}
    for row in rows:
        for kind in _PROJECTION_K:
            if row not in screened_sets[kind]:
                continue
            complete = True
            for layer in _QUALIFICATION_LAYERS:
                # Keep running every real layer after a normal admission
                # rejection so the ledger contains the complete independent
                # projection envelope.  Device failures still raise from
                # ``_run_job`` and terminate the poisoned campaign.
                complete = (
                    _run_job(
                        args,
                        runtime,
                        status_file,
                        kind=kind,
                        layer=layer,
                        row=row,
                    )
                    and complete
                )
            if complete:
                qualified[kind].append(row)
    return qualified


def main() -> None:
    args = _parser().parse_args()
    rows = _parse_rows(args.rows)
    status_file = args.run_dir / "qualification-status.tsv"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "measurements").mkdir(parents=True, exist_ok=True)

    runtime = measurement.initialize_measurement_runtime(
        chained_flush=True,
        require_production_operator=True,
    )
    _rank_zero_action(
        lambda: status_file.write_text(
            "kind\tlayer\tm\tstatus\n",
            encoding="utf-8",
        )
    )

    screened_rows = _screen_projection_rows(args, runtime, status_file, rows)

    def write_screened_rows() -> None:
        _write_projection_rows(
            args.run_dir / "screened-projection-rows.json",
            stage="layer-31-screen",
            qualification_layers=(_SCREENING_LAYER,),
            projection_rows=screened_rows,
        )
        _write_rows(
            args.run_dir / "screened-rows.txt",
            _projection_row_intersection(screened_rows),
        )

    _rank_zero_action(write_screened_rows)
    if args.screen_only:
        measurement.finalize_measurement_runtime(runtime)
        return

    qualified_rows = _qualify_projection_rows(
        args,
        runtime,
        status_file,
        screened_rows,
        rows,
    )

    def write_qualified_rows() -> None:
        _write_projection_rows(
            args.run_dir / "qualified-projection-rows.json",
            stage="per-layer-qualification",
            qualification_layers=_PRODUCTION_LAYERS,
            projection_rows=qualified_rows,
        )
        _write_rows(
            args.run_dir / "qualified-rows.txt",
            _projection_row_intersection(qualified_rows),
        )

    _rank_zero_action(write_qualified_rows)
    measurement.finalize_measurement_runtime(runtime)


if __name__ == "__main__":
    main()
