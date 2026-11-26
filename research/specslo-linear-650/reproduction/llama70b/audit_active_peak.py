#!/usr/bin/env python3
"""Rebuild the concurrent decode-request peak from a SpecSLO result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def audit(result_path: Path) -> dict:
    source = result_path.read_bytes()
    result = json.loads(source)
    if len(result.get("results", [])) != 1:
        raise ValueError("expected exactly one measured batch")
    batch = result["results"][0]
    request_metrics = batch["slo"]["request_metrics"]
    events: list[tuple[float, int]] = []
    for metric in request_metrics:
        start = float(metric["arrival_offset_sec"]) + float(metric["arrival_wait_ms"]) / 1000.0
        end = start + float(metric["finished_decode_elapsed_ms"]) / 1000.0
        if end < start:
            raise ValueError(f"negative service interval for {metric['request_id']}")
        events.extend(((start, 1), (end, -1)))
    # End events precede starts at equal timestamps: requests occupy [start,end).
    events.sort(key=lambda item: (item[0], item[1]))
    active = peak = 0
    peak_at = 0.0
    for at, change in events:
        active += change
        if active > peak:
            peak, peak_at = active, at
    if active:
        raise AssertionError("unbalanced request intervals")
    configured = int(batch["batch_size"])
    return {
        "status": "pass" if peak <= configured else "fail",
        "source_result_sha256": hashlib.sha256(source).hexdigest(),
        "method": "active start = arrival_offset_sec + arrival_wait_ms/1000; active end = start + finished_decode_elapsed_ms/1000; sweep half-open intervals",
        "configured_global_batch_size": configured,
        "inferred_simultaneous_active_peak": peak,
        "peak_at_seconds_after_first_arrival": peak_at,
        "requests": len(request_metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.result)
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(serialized)
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
