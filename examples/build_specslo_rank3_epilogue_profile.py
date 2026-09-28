# SPDX-License-Identifier: Apache-2.0
"""Build a production-bound TP3 native-MatMul/epilogue MC2 profile.

The two inputs are raw graph qualification documents emitted by
``measure_specslo_rank3_reduce_add_rmsnorm.py`` for Qwen3-32B's attention
output and FFN down projections.  This command revalidates their exact
changed-input results and binds the combined profile to the currently loaded
Torch adapter, OPAPI provider, CANN/HCCL versions, and physical TP mapping.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from examples.specslo_mc2_deferred_chain_evidence import (
    load_deferred_chain_evidence,
)
from vllm_ascend.spec_decode.pearl.mc2 import (
    MC2_TP3_NATIVE_EPILOGUE_OPERATOR,
    current_mc2_runtime_binding,
    mc2_source_sha256,
    normalize_mc2_profile,
)

EXPECTED_SHAPES = frozenset(((100, 3072, 5120), (100, 8576, 5120)))
RMS_NORM_EPSILON = 1.0e-6


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention", type=Path, required=True)
    parser.add_argument("--down", type=Path, required=True)
    parser.add_argument(
        "--deferred-chain",
        type=Path,
        help=(
            "Graph qualification emitted by check_specslo_mc2_deferred_chain.py. "
            "Without it the profile keeps the fully-flushed epilogue path."
        ),
    )
    parser.add_argument("--hardware", default="Ascend910B2")
    parser.add_argument("--latency-percentile", type=float, default=95.0)
    parser.add_argument("--minimum-samples", type=int, default=5)
    parser.add_argument("--minimum-speedup", type=float, default=1.02)
    parser.add_argument(
        "--minimum-chain-speedup",
        type=float,
        default=1.0,
        help="Minimum graph p95 speedup of deferred versus fully-flushed epilogues.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_measurement(path: Path) -> dict[str, Any]:
    payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("status") != "measured":
        raise ValueError(f"{path}: expected a measured schema_version=1 document")
    metadata = payload.get("metadata")
    shape = payload.get("shape")
    correctness = payload.get("correctness")
    samples = payload.get("samples")
    if not all(isinstance(value, dict) for value in (metadata, shape, correctness, samples)):
        raise ValueError(f"{path}: incomplete qualification document")
    if metadata.get("execution_mode") != "graph" or metadata.get("operations_per_graph") != 1:
        raise ValueError(f"{path}: production profile requires one operation per ACLGraph")
    if metadata.get("tensor_parallel_size") != 3:
        raise ValueError(f"{path}: production profile requires TP3")
    if metadata.get("operator") not in {
        "torch.ops._C_ascend.allreduce_add_rmsnorm",
        "torch.ops.vllm_ascend.allreduce_add_rmsnorm",
        "torch.ops._C_ascend.allreduce_add_rmsnorm_chained",
        "torch.ops.vllm_ascend.allreduce_add_rmsnorm_chained",
    }:
        raise ValueError(f"{path}: measurement used the wrong custom operator")
    if metadata.get("operator", "").endswith("_chained") and metadata.get("chained_flush") is not True:
        raise ValueError(f"{path}: chained measurement did not flush every standalone invocation")
    if correctness.get("exact_all_ranks") is not True:
        raise ValueError(f"{path}: original-input result is not exact on every rank")
    if correctness.get("changed_input_exact_all_ranks") is not True:
        raise ValueError(f"{path}: changed-input result is not exact on every rank")
    exact_error_fields = (
        "max_abs_norm",
        "max_abs_added",
        "changed_input_max_abs_norm",
        "changed_input_max_abs_added",
    )
    if any(float(correctness.get(name, math.inf)) != 0.0 for name in exact_error_fields):
        raise ValueError(f"{path}: qualification is not bit-exact")
    if int(correctness.get("changed_input_replays", 0)) < 2:
        raise ValueError(f"{path}: insufficient changed-input graph replays")
    if not math.isfinite(float(correctness.get("changed_input_delta", math.nan))) or float(
        correctness["changed_input_delta"]
    ) == 0.0:
        raise ValueError(f"{path}: invalid changed-input delta")
    return payload


def _load_deferred_chain(
    path: Path,
    *,
    attention_source: Path,
    down_source: Path,
    expected_mapping: tuple[str, ...],
    minimum_samples: int,
    minimum_speedup: float,
) -> dict[str, Any]:
    return load_deferred_chain_evidence(
        (path,),
        expected_source_sha256=mc2_source_sha256(),
        expected_mapping=expected_mapping,
        minimum_samples=minimum_samples,
        minimum_speedup=minimum_speedup,
        allowed_input_sources=(attention_source, down_source),
        allowed_row_counts=(100,),
    )


def _profile_row(payload: dict[str, Any], *, minimum_samples: int) -> dict[str, Any]:
    shape = payload["shape"]
    correctness = payload["correctness"]
    samples = payload["samples"]
    key = tuple(int(shape[name]) for name in ("m", "k", "n"))
    if key not in EXPECTED_SHAPES:
        raise ValueError(f"unexpected Qwen3-32B TP3 qualification shape: {key!r}")
    baseline = [float(value) for value in samples.get("baseline_latency_ms", ())]
    fused = [float(value) for value in samples.get("fused_latency_ms", ())]
    if len(baseline) < minimum_samples or len(fused) < minimum_samples:
        raise ValueError(f"shape {key!r} has fewer than {minimum_samples} latency samples")
    if any(not math.isfinite(value) or value <= 0.0 for value in (*baseline, *fused)):
        raise ValueError(f"shape {key!r} contains invalid latency samples")
    return {
        "m": key[0],
        "k": key[1],
        "n": key[2],
        "dtype": "bfloat16",
        "weight_format": "ND",
        "is_trans_b": True,
        "activation_layout": "contiguous",
        "weight_layout": "contiguous",
        "residual_layout": "contiguous",
        "gamma_layout": "contiguous",
        "baseline_latency_ms": baseline,
        "fused_latency_ms": fused,
        "max_abs_norm": float(correctness["max_abs_norm"]),
        "norm_atol": 0.0,
        "max_abs_added": float(correctness["max_abs_added"]),
        "added_atol": 0.0,
        "changed_input_replays": int(correctness["changed_input_replays"]),
        "changed_input_delta": float(correctness["changed_input_delta"]),
        "changed_input_max_abs_norm": float(correctness["changed_input_max_abs_norm"]),
        "changed_input_max_abs_added": float(correctness["changed_input_max_abs_added"]),
    }


def build_profile(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 <= args.latency_percentile <= 100.0:
        raise ValueError("latency percentile must be in [0, 100]")
    if args.minimum_samples < 3:
        raise ValueError("minimum samples must be at least three")
    if args.minimum_speedup <= 1.0:
        raise ValueError("minimum speedup must be greater than one")
    if args.minimum_chain_speedup <= 0.0:
        raise ValueError("minimum chain speedup must be positive")
    measurements = [_load_measurement(args.attention), _load_measurement(args.down)]
    current_source_sha256 = mc2_source_sha256()
    measurement_source_sha256 = [
        item["metadata"].get("source_sha256") for item in measurements
    ]
    if any(value != current_source_sha256 for value in measurement_source_sha256):
        raise ValueError(
            "attention/down qualification source SHA does not match the current "
            "MC2 adapter, model wiring, and kernel sources"
        )
    mappings = [tuple(item["metadata"]["tp_rank_device_mapping"]) for item in measurements]
    if mappings[0] != mappings[1] or len(mappings[0]) != 3:
        raise ValueError("attention/down TP rank mappings differ")
    rows = [_profile_row(item, minimum_samples=args.minimum_samples) for item in measurements]
    if {tuple(row[name] for name in ("m", "k", "n")) for row in rows} != EXPECTED_SHAPES:
        raise ValueError("profile must contain exactly the attention and down M100 shapes")
    metadata: dict[str, Any] = {
        "operator": MC2_TP3_NATIVE_EPILOGUE_OPERATOR,
        "standalone_chained_flush": all(
            item["metadata"].get("operator", "").endswith("_chained")
            and item["metadata"].get("chained_flush") is True
            for item in measurements
        ),
        "hardware": str(args.hardware),
        "tensor_parallel_size": 3,
        "source_sha256": current_source_sha256,
        "measurement_scope": "graph_native_matmul_plus_rank3_epilogue",
        "input_source": [
            str(args.attention.resolve(strict=True)),
            str(args.down.resolve(strict=True)),
        ],
        "rms_norm_epsilon": RMS_NORM_EPSILON,
        "runtime_binding": current_mc2_runtime_binding(
            tp_rank_device_mapping=mappings[0],
            operator=MC2_TP3_NATIVE_EPILOGUE_OPERATOR,
        ),
    }
    deferred_chain_path = getattr(args, "deferred_chain", None)
    if deferred_chain_path is not None:
        metadata["deferred_read_done_chain"] = _load_deferred_chain(
            deferred_chain_path,
            attention_source=Path(measurements[0]["metadata"]["input_source"]),
            down_source=Path(measurements[1]["metadata"]["input_source"]),
            expected_mapping=mappings[0],
            minimum_samples=args.minimum_samples,
            minimum_speedup=args.minimum_chain_speedup,
        )
    document: dict[str, Any] = {
        "schema_version": 1,
        "status": "measured",
        "metadata": metadata,
        "latency_percentile": float(args.latency_percentile),
        "minimum_samples": int(args.minimum_samples),
        "minimum_speedup": float(args.minimum_speedup),
        "measurements": rows,
    }
    normalize_mc2_profile(document)
    return document


def main() -> None:
    args = _parser().parse_args()
    document = build_profile(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(args.output.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
