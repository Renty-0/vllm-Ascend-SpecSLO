# SPDX-License-Identifier: Apache-2.0
"""Shared, CPU-only SLO accounting; never infer missing timing as a pass."""

from __future__ import annotations

import math


def summarize_slo_rows(rows, *, tpot_field, definition, elapsed_seconds=None):
    constrained = [row for row in rows if row.get("slo_tpot_ms") is not None]
    by_class = {}
    attained = goodput = missing = 0
    for row in constrained:
        bound = float(row["slo_tpot_ms"])
        if not math.isfinite(bound) or bound <= 0:
            raise ValueError("A constrained request needs a finite positive TPOT SLO")
        value = row.get(tpot_field)
        valid = value is not None and math.isfinite(float(value)) and float(value) >= 0
        valid = valid and int(row["output_tokens"]) > 0
        passed = valid and float(value) <= bound
        group = by_class.setdefault(
            row.get("slo_class") or "unspecified",
            {"requests": 0, "attained_requests": 0, "goodput_tokens": 0, "missing_timing_requests": 0, "values": []},
        )
        group["requests"] += 1
        if valid:
            group["values"].append(float(value))
        else:
            missing += 1
            group["missing_timing_requests"] += 1
        if passed:
            attained += 1
            goodput += int(row["output_tokens"])
            group["attained_requests"] += 1
            group["goodput_tokens"] += int(row["output_tokens"])
    for group in by_class.values():
        values = group.pop("values")
        group["attainment"] = group["attained_requests"] / group["requests"]
        group["mean_tpot_ms"] = sum(values) / len(values) if values else None
        group["mean_tpot_scope"] = "valid timing rows only; missing rows never attain SLO"
    return {
        "tpot_definition": definition,
        "goodput_definition": (
            f"sum(output_tokens where {tpot_field} <= slo_tpot_ms) / "
            "measured_e2e_seconds"
        ),
        "constrained_requests": len(constrained),
        "attained_requests": attained,
        "attainment": attained / len(constrained) if constrained else None,
        "goodput_tokens": goodput,
        "goodput_tokens_per_e2e_second": (
            goodput / elapsed_seconds if elapsed_seconds is not None and elapsed_seconds > 0 else None
        ),
        "missing_timing_requests": missing,
        "by_class": by_class,
    }
