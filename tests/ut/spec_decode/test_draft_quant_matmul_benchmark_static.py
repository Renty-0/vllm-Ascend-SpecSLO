# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = REPO_ROOT / "examples/benchmark_nano_pearl_draft_quant_matmul.py"


def _literal_assignment(name: str):
    tree = ast.parse(BENCHMARK.read_text(encoding="utf-8"), filename=str(BENCHMARK))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing literal assignment {name}")


def test_qwen3_06b_projection_shapes_and_default_token_counts() -> None:
    assert _literal_assignment("DRAFT_PROJECTIONS") == (
        ("qkv", 1024, 4096),
        ("o_proj", 2048, 1024),
        ("gate_up", 1024, 6144),
        ("down", 3072, 1024),
        ("lm_head", 1024, 151936),
    )
    assert _literal_assignment("DEFAULT_TOKEN_COUNTS") == (8, 16, 20, 24, 32)


def test_benchmark_is_guarded_and_does_not_import_the_service_engine() -> None:
    source = BENCHMARK.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(BENCHMARK))

    assert any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        for node in tree.body
    )
    assert "native_engine" not in source
    assert "serve_specslo" not in source
    assert "AscendW8A16LinearMethod" in source
    assert "AscendW8A8DynamicLinearMethod" in source
