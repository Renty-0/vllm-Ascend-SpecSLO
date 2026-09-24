# SPDX-License-Identifier: Apache-2.0
"""CPU tests for combining v126 graph qualification documents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from examples import build_specslo_rank3_epilogue_profile as builder
from vllm_ascend.spec_decode.pearl import mc2


def _measurement(
    path: Path,
    shape: tuple[int, int, int],
    *,
    exact: bool = True,
    input_source: Path | None = None,
    chained_flush: bool = False,
) -> None:
    payload = {
        "schema_version": 1,
        "status": "measured",
        "metadata": {
            "execution_mode": "graph",
            "operations_per_graph": 1,
            "operator": (
                "torch.ops._C_ascend.allreduce_add_rmsnorm_chained"
                if chained_flush
                else "torch.ops._C_ascend.allreduce_add_rmsnorm"
            ),
            "chained_flush": chained_flush,
            "tensor_parallel_size": 3,
            "tp_rank_device_mapping": ["1", "2", "4"],
            "input_source": str(input_source or path.parent / f"{path.stem}-inputs"),
            "source_sha256": builder.mc2_source_sha256(),
        },
        "shape": dict(zip(("m", "k", "n"), shape)),
        "correctness": {
            "exact_all_ranks": exact,
            "max_abs_norm": 0.0,
            "max_abs_added": 0.0,
            "changed_input_replays": 8,
            "changed_input_delta": 0.015625,
            "changed_input_exact_all_ranks": exact,
            "changed_input_max_abs_norm": 0.0,
            "changed_input_max_abs_added": 0.0,
        },
        "samples": {
            "baseline_latency_ms": [0.14, 0.15, 0.14, 0.15, 0.14],
            "fused_latency_ms": [0.11, 0.12, 0.11, 0.12, 0.11],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _deferred_chain(path: Path, *, attention_source: Path, down_source: Path) -> None:
    zero_errors = {
        "attention_norm": 0.0,
        "attention_added": 0.0,
        "down_norm": 0.0,
        "down_added": 0.0,
        "chain_state": 0.0,
    }
    payload = {
        "schema_version": 1,
        "status": "measured",
        "execution_mode": "graph",
        "tensor_parallel_size": 3,
        "chain_scope": "whole_model",
        "layer_count": 64,
        "operations_per_chain": 128,
        "tp_rank_device_mapping": ["1", "2", "4"],
        "attention_input_source": str(attention_source),
        "down_input_source": str(down_source),
        "correctness": {
            "initial": {
                "fully_flushed": {"exact_all_ranks": True, "max_abs": zero_errors},
                "deferred_chain": {"exact_all_ranks": True, "max_abs": zero_errors},
            },
            "changed_input_exact_all_ranks": {
                "fully_flushed": True,
                "deferred_chain": True,
            },
            "changed_input_max_abs": {
                "fully_flushed": zero_errors,
                "deferred_chain": zero_errors,
            },
            "changed_input_replays": 5,
            "final_chain_state_zero_all_ranks": True,
        },
        "samples": {
            "fully_flushed_latency_ms": [0.3, 0.31, 0.3, 0.31, 0.3],
            "deferred_chain_latency_ms": [0.28, 0.29, 0.28, 0.29, 0.28],
        },
        "summary": {
            "deferred_vs_fully_flushed_p50": 0.3 / 0.28,
            "deferred_vs_fully_flushed_p95": 0.31 / 0.29,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _binding() -> dict[str, object]:
    symbols = mc2._mc2_opapi_symbols(mc2.MC2_TP3_NATIVE_EPILOGUE_OPERATOR)
    return {
        "hccl_deterministic": "true",
        "hccl_op_expansion_mode": "AIV",
        "reduction_mode": "global_01",
        "cann_version": "test-cann",
        "hccl_version": "test-hccl",
        "vendor_payload_sha256": "a" * 64,
        "opapi_symbol_provider_sha256": {
            symbol: (None if symbol in mc2._MC2_OPTIONAL_OPAPI_SYMBOLS else "b" * 64)
            for symbol in symbols
        },
        "adapter_binary_sha256": "c" * 64,
        "tp_rank_device_mapping": ("1", "2", "4"),
    }


def _args(attention: Path, down: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        attention=attention,
        down=down,
        hardware="Ascend910B2",
        latency_percentile=95.0,
        minimum_samples=5,
        minimum_speedup=1.02,
        minimum_chain_speedup=1.0,
        deferred_chain=None,
        output=output,
    )


def test_builder_emits_operator_specific_runtime_bound_profile(tmp_path: Path, monkeypatch):
    attention = tmp_path / "attention.json"
    down = tmp_path / "down.json"
    _measurement(attention, (100, 3072, 5120))
    _measurement(down, (100, 8576, 5120))
    calls: list[tuple[tuple[str, ...], str]] = []

    def binding(*, tp_rank_device_mapping, operator):
        calls.append((tuple(tp_rank_device_mapping), operator))
        return _binding()

    monkeypatch.setattr(builder, "current_mc2_runtime_binding", binding)
    document = builder.build_profile(_args(attention, down, tmp_path / "profile.json"))

    assert document["metadata"]["operator"] == mc2.MC2_TP3_NATIVE_EPILOGUE_OPERATOR
    assert calls == [(("1", "2", "4"), mc2.MC2_TP3_NATIVE_EPILOGUE_OPERATOR)]
    assert {(row["k"], row["n"]) for row in document["measurements"]} == {
        (3072, 5120),
        (8576, 5120),
    }
    assert all(row["changed_input_replays"] == 8 for row in document["measurements"])
    assert document["metadata"]["standalone_chained_flush"] is False


def test_builder_records_allocation_free_chained_flush_qualification(
    tmp_path: Path,
    monkeypatch,
):
    attention = tmp_path / "attention.json"
    down = tmp_path / "down.json"
    _measurement(attention, (100, 3072, 5120), chained_flush=True)
    _measurement(down, (100, 8576, 5120), chained_flush=True)
    monkeypatch.setattr(builder, "current_mc2_runtime_binding", lambda **_: _binding())

    document = builder.build_profile(_args(attention, down, tmp_path / "profile.json"))

    assert document["metadata"]["standalone_chained_flush"] is True


def test_builder_rejects_non_exact_graph_measurement(tmp_path: Path):
    attention = tmp_path / "attention.json"
    down = tmp_path / "down.json"
    _measurement(attention, (100, 3072, 5120), exact=False)
    _measurement(down, (100, 8576, 5120))

    with pytest.raises(ValueError, match="not exact"):
        builder.build_profile(_args(attention, down, tmp_path / "profile.json"))


def test_builder_rejects_measurement_from_stale_source(tmp_path: Path):
    attention = tmp_path / "attention.json"
    down = tmp_path / "down.json"
    _measurement(attention, (100, 3072, 5120))
    _measurement(down, (100, 8576, 5120))
    payload = json.loads(attention.read_text(encoding="utf-8"))
    payload["metadata"]["source_sha256"] = "0" * 64
    attention.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="source SHA"):
        builder.build_profile(_args(attention, down, tmp_path / "profile.json"))


def test_builder_admits_only_explicit_bit_exact_deferred_chain(tmp_path: Path, monkeypatch):
    attention = tmp_path / "attention.json"
    down = tmp_path / "down.json"
    attention_source = tmp_path / "attention-input"
    down_source = tmp_path / "down-input"
    _measurement(attention, (100, 3072, 5120), input_source=attention_source)
    _measurement(down, (100, 8576, 5120), input_source=down_source)
    chain = tmp_path / "chain.json"
    _deferred_chain(chain, attention_source=attention_source, down_source=down_source)
    monkeypatch.setattr(builder, "current_mc2_runtime_binding", lambda **_: _binding())
    args = _args(attention, down, tmp_path / "profile.json")
    args.deferred_chain = chain

    document = builder.build_profile(args)

    evidence = document["metadata"]["deferred_read_done_chain"]
    assert evidence["qualified"] is True
    assert evidence["execution_mode"] == "graph"
    assert evidence["chain_scope"] == "whole_model"
    assert evidence["layer_count"] == 64
    assert evidence["operations_per_chain"] == 128
    assert evidence["p95_speedup"] > 1.0
    assert len(evidence["report_sha256"]) == 64

    broken = json.loads(chain.read_text(encoding="utf-8"))
    broken["correctness"]["final_chain_state_zero_all_ranks"] = False
    chain.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="did not drain"):
        builder.build_profile(args)

    broken["correctness"]["final_chain_state_zero_all_ranks"] = True
    broken["samples"]["deferred_chain_latency_ms"] = [0.4] * 5
    chain.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="below the required"):
        builder.build_profile(args)
