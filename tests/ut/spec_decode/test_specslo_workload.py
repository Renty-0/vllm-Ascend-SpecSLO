# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

from vllm_ascend.spec_decode.pearl.specslo_workload import (
    allocate_category_counts,
    build_workload,
    load_prompts,
)


def test_allocate_category_counts_is_exact_for_paper_mix():
    assert allocate_category_counts(100, "0.6,0.2,0.2") == {
        "coding": 60,
        "chat": 20,
        "summarization": 20,
    }
    assert sum(allocate_category_counts(7).values()) == 7


def test_load_alpaca_json_array_and_build_manifest(tmp_path):
    coding = tmp_path / "coding.jsonl"
    coding.write_text(json.dumps({"turns": ["write code"]}) + "\n", encoding="utf-8")
    alpaca = tmp_path / "alpaca.json"
    alpaca.write_text(json.dumps([{"instruction": "chat", "input": "context"}]), encoding="utf-8")
    cnndm = tmp_path / "cnndm.jsonl"
    cnndm.write_text(json.dumps({"article": "news"}) + "\n", encoding="utf-8")
    assert load_prompts(alpaca, "chat") == ["chat\n\nInput:\ncontext"]
    requests, metadata = build_workload(
        coding_path=coding,
        chat_path=alpaca,
        summarization_path=cnndm,
        rps=4,
        num_requests=10,
        seed=4,
    )
    assert len(requests) == 10
    assert metadata["category_counts"] == {"coding": 6, "chat": 2, "summarization": 2}
    assert {row["slo_tpot_ms"] for row in requests} == {40.0, 50.0, 150.0}
    assert all("arrival_offset_sec" in row and row["max_tokens"] == 256 for row in requests)


def test_prepare_cli_is_importable():
    path = Path(__file__).parents[3] / "examples" / "prepare_specslo_workload.py"
    spec = importlib.util.spec_from_file_location("prepare_specslo_workload", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert callable(module.main)

