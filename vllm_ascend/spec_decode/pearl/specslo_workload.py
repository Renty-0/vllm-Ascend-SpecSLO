# SPDX-License-Identifier: Apache-2.0
"""SpecRhythm workload loading and deterministic multi-SLO generation.

The ATC'26 workload uses coding, chat, and summarization requests in a 6:2:2
mix.  This module deliberately keeps workload preparation independent from
vLLM so that the exact manifest can be reused by the native PEARL path and by
the target-only baseline.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

DEFAULT_MIX = (0.6, 0.2, 0.2)
DEFAULT_SLOS_MS = {"tight": 40.0, "normal": 50.0, "loose": 150.0}
DEFAULT_GAMMAS = {"tight": 4, "normal": 3, "loose": 2}
CATEGORIES = ("coding", "chat", "summarization")


def parse_mix(value: str | Iterable[float]) -> tuple[float, float, float]:
    """Parse and validate coding/chat/summarization proportions."""
    if isinstance(value, str):
        parts = tuple(float(item.strip()) for item in value.split(","))
    else:
        parts = tuple(float(item) for item in value)
    if len(parts) != 3 or any(item < 0 for item in parts):
        raise ValueError("mix must contain three non-negative values")
    total = sum(parts)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"mix must sum to 1.0, got {total}")
    return parts  # type: ignore[return-value]


def allocate_category_counts(num_requests: int, mix: Iterable[float] = DEFAULT_MIX) -> dict[str, int]:
    """Return exact integer quotas using the largest-remainder method."""
    if num_requests < 0:
        raise ValueError("num_requests must be non-negative")
    proportions = parse_mix(mix)
    raw = [num_requests * value for value in proportions]
    counts = [math.floor(value) for value in raw]
    remainder = num_requests - sum(counts)
    order = sorted(range(3), key=lambda index: (raw[index] - counts[index], -index), reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    return dict(zip(CATEGORIES, counts))


def _iter_json_records(path: str | os.PathLike[str]) -> Iterable[Mapping[str, Any]]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if dataset_path.suffix.lower() == ".json":
        value = json.loads(dataset_path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"Expected a JSON list in {dataset_path}")
        for record in value:
            if isinstance(record, Mapping):
                yield record
        return
    with dataset_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {dataset_path}:{line_number}") from exc
            if isinstance(record, Mapping):
                yield record


def extract_prompt(record: Mapping[str, Any], role: str) -> str | None:
    """Normalize one HumanEval, Alpaca, or CNN/DailyMail record."""
    if role == "summarization":
        for key in ("article", "document", "text"):
            if record.get(key):
                return "Summarize the following article:\n\n" + str(record[key]).strip()
    if role == "coding" and record.get("turns"):
        turns = record["turns"]
        if isinstance(turns, list) and turns:
            return str(turns[0]).strip()
    if record.get("turns"):
        turns = record["turns"]
        if isinstance(turns, list) and turns:
            return str(turns[0]).strip()
    if record.get("instruction"):
        instruction = str(record["instruction"]).strip()
        extra = str(record.get("input", "")).strip()
        return instruction + ("\n\nInput:\n" + extra if extra else "")
    for key in ("prompt", "question", "text"):
        if record.get(key):
            return str(record[key]).strip()
    return None


def load_prompts(path: str | os.PathLike[str], role: str) -> list[str]:
    prompts = [prompt for record in _iter_json_records(path) if (prompt := extract_prompt(record, role))]
    if not prompts:
        raise ValueError(f"No usable {role} prompts found in {path}")
    return prompts


def generate_poisson_arrivals(
    rps: float,
    *,
    duration_sec: float | None = None,
    num_requests: int | None = None,
    rng: random.Random | None = None,
) -> list[float]:
    """Generate monotonically increasing offsets for a synthetic Poisson trace."""
    if rps <= 0:
        raise ValueError("rps must be positive")
    if (duration_sec is None) == (num_requests is None):
        raise ValueError("provide exactly one of duration_sec or num_requests")
    if num_requests is not None and num_requests < 0:
        raise ValueError("num_requests must be non-negative")
    if duration_sec is not None and duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    random_source = rng or random.Random(0)
    offsets: list[float] = []
    current = 0.0
    if num_requests is not None:
        for _ in range(num_requests):
            current += random_source.expovariate(rps)
            offsets.append(current)
        return offsets
    assert duration_sec is not None
    while True:
        current += random_source.expovariate(rps)
        if current >= duration_sec:
            break
        offsets.append(current)
    return offsets


def load_arrival_offsets(path: str | os.PathLike[str], max_requests: int | None = None) -> list[float]:
    """Load offsets from JSONL, plain text, or CSV and normalize to zero."""
    values: list[float] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if max_requests is not None and len(values) >= max_requests:
                break
            line = line.strip()
            if not line:
                continue
            try:
                if line.startswith("{"):
                    record = json.loads(line)
                    value = next(
                        float(record[key])
                        for key in ("arrival_offset_sec", "timestamp", "arrival_ts", "time")
                        if key in record
                    )
                else:
                    value = float(line.split(",", 1)[0].strip())
            except (ValueError, TypeError, StopIteration, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid arrival trace at {path}:{line_number}") from exc
            values.append(value)
    if not values:
        raise ValueError(f"No arrival offsets found in {path}")
    values.sort()
    first = values[0]
    return [value - first for value in values[:max_requests]]


def _request_id(prefix: str, index: int) -> str:
    safe = prefix.replace(".", "p").replace(",", "_").replace("/", "_").replace(" ", "")
    return f"{safe}-{index:06d}"


def build_workload(
    *,
    coding_path: str | os.PathLike[str],
    chat_path: str | os.PathLike[str],
    summarization_path: str | os.PathLike[str],
    rps: float,
    num_requests: int | None = None,
    duration_sec: float | None = 120.0,
    mix: str | Iterable[float] = DEFAULT_MIX,
    tight_slo_tpot_ms: float = DEFAULT_SLOS_MS["tight"],
    normal_slo_tpot_ms: float = DEFAULT_SLOS_MS["normal"],
    loose_slo_tpot_ms: float = DEFAULT_SLOS_MS["loose"],
    tight_gamma: int = DEFAULT_GAMMAS["tight"],
    normal_gamma: int = DEFAULT_GAMMAS["normal"],
    loose_gamma: int = DEFAULT_GAMMAS["loose"],
    max_tokens: int = 256,
    seed: int = 0,
    arrival_trace: str | os.PathLike[str] | None = None,
    request_id_prefix: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build requests and metadata for the three SpecRhythm SLO classes."""
    if num_requests is not None and num_requests <= 0:
        raise ValueError("num_requests must be positive")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    proportions = parse_mix(mix)
    coding = load_prompts(coding_path, "coding")
    chat = load_prompts(chat_path, "chat")
    summarization = load_prompts(summarization_path, "summarization")
    arrival_rng = random.Random(seed)
    sample_rng = random.Random(seed + 1)
    if arrival_trace is not None:
        arrivals = load_arrival_offsets(arrival_trace, max_requests=num_requests)
        arrival_source = "external_trace"
    else:
        arrivals = generate_poisson_arrivals(
            rps, duration_sec=duration_sec if num_requests is None else None, num_requests=num_requests, rng=arrival_rng
        )
        arrival_source = "synthetic_poisson"
    if num_requests is None:
        num_requests = len(arrivals)
    if len(arrivals) != num_requests:
        raise ValueError(f"arrival trace produced {len(arrivals)} requests, expected {num_requests}")

    quotas = allocate_category_counts(num_requests, proportions)
    categories = [category for category in CATEGORIES for _ in range(quotas[category])]
    sample_rng.shuffle(categories)
    datasets = {"coding": coding, "chat": chat, "summarization": summarization}
    slo_classes = {"coding": "tight", "chat": "normal", "summarization": "loose"}
    slo_values = {
        "tight": float(tight_slo_tpot_ms),
        "normal": float(normal_slo_tpot_ms),
        "loose": float(loose_slo_tpot_ms),
    }
    gammas = {"tight": int(tight_gamma), "normal": int(normal_gamma), "loose": int(loose_gamma)}
    prefix = request_id_prefix or f"rps{rps}-seed{seed}"
    requests: list[dict[str, Any]] = []
    for index, (arrival, category) in enumerate(zip(arrivals, categories)):
        source_index = sample_rng.randrange(len(datasets[category]))
        slo_class = slo_classes[category]
        requests.append(
            {
                "request_id": _request_id(prefix, index),
                "arrival_offset_sec": float(arrival),
                "category": category,
                "slo_class": slo_class,
                "slo_tpot_ms": slo_values[slo_class],
                "per_request_gamma": gammas[slo_class],
                "max_tokens": int(max_tokens),
                "prompt": datasets[category][source_index],
                "source_dataset": str(
                    {"coding": coding_path, "chat": chat_path, "summarization": summarization_path}[category]
                ),
                "source_index": source_index,
            }
        )
    counts = Counter(item["category"] for item in requests)
    metadata = {
        "format": "specslo-workload-v1",
        "rps": float(rps),
        "duration_sec": duration_sec,
        "num_requests": len(requests),
        "mix_requested": dict(zip(CATEGORIES, proportions)),
        "category_counts": dict(counts),
        "mix_realized": {category: counts[category] / len(requests) if requests else 0.0 for category in CATEGORIES},
        "slo_tpot_ms": slo_values,
        "gamma_by_slo_class": gammas,
        "max_tokens": int(max_tokens),
        "seed": int(seed),
        "arrival_source": arrival_source,
        "dataset_paths": {"coding": str(coding_path), "chat": str(chat_path), "summarization": str(summarization_path)},
    }
    return requests, metadata


def write_workload(
    requests: list[Mapping[str, Any]], metadata: Mapping[str, Any], output_path: str | os.PathLike[str]
) -> None:
    """Write JSONL manifest and a same-name ``.meta.json`` sidecar."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for request in requests:
            stream.write(json.dumps(dict(request), ensure_ascii=False, sort_keys=True) + "\n")
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(dict(metadata), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
