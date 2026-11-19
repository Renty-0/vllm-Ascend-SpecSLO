#!/usr/bin/env python3
"""Reconstruct simultaneous admitted requests from a completed result."""

import hashlib
import json
import sys
from pathlib import Path


def main() -> int:
    result_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    document_bytes = result_path.read_bytes()
    document = json.loads(document_bytes)
    result = document["results"][0]
    metrics = result["slo"]["request_metrics"]
    events = []
    for row in metrics:
        start = float(row["arrival_offset_sec"]) + float(row["arrival_wait_ms"]) / 1000
        end = start + float(row["finished_decode_elapsed_ms"]) / 1000
        if end < start:
            raise ValueError(f"Negative active interval for {row['request_id']}")
        events.append((start, 1))
        events.append((end, -1))
    # Half-open intervals: end before start when timestamps coincide.
    events.sort(key=lambda event: (event[0], event[1]))
    active = peak = 0
    peak_at = None
    for timestamp, delta in events:
        active += delta
        if active < 0:
            raise ValueError("Negative active count")
        if active > peak:
            peak = active
            peak_at = timestamp
    if active != 0:
        raise ValueError("Unclosed active intervals")
    cap = int(result["batch_size"])
    audit = {
        "status": "pass" if peak <= cap else "fail",
        "source_result_sha256": hashlib.sha256(document_bytes).hexdigest(),
        "method": "active start = arrival_offset_sec + arrival_wait_ms/1000; active end = start + finished_decode_elapsed_ms/1000; sweep half-open intervals",
        "configured_global_batch_size": cap,
        "inferred_simultaneous_active_peak": peak,
        "peak_at_seconds_after_first_arrival": peak_at,
        "requests": len(metrics),
    }
    output_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))
    return 0 if audit["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
