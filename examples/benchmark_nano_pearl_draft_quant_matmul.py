# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark Qwen3-0.6B TP1 draft matrix multiplication paths.

This is an operator-only benchmark.  It does not construct a PEARL engine,
load a model, allocate a KV cache, or mutate the serving path.  Each strategy
is captured in its own ACLGraph and only replay latency after warmup is timed.

Example::

    ASCEND_RT_VISIBLE_DEVICES=1 \
      python examples/benchmark_nano_pearl_draft_quant_matmul.py \
      --output-json /tmp/qwen3-06b-draft-matmul.json

The synthetic weights use the real Qwen3-0.6B projection shapes.  They are
intended to qualify kernels before a model integration, not to establish
end-to-end acceptance or generation accuracy.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.quantization.methods.w8a16 import AscendW8A16LinearMethod

# (name, input features, output features).  These are the fused native-model
# shapes, not the individual Hugging Face checkpoint shard shapes.
DRAFT_PROJECTIONS = (
    ("qkv", 1024, 4096),
    ("o_proj", 2048, 1024),
    ("gate_up", 1024, 6144),
    ("down", 3072, 1024),
    ("lm_head", 1024, 151936),
)
DEFAULT_TOKEN_COUNTS = (8, 16, 20, 24, 32)
DEFAULT_STRATEGIES = ("bf16", "fp16", "w8a16", "w8a8_dynamic")
ACL_FORMAT_FRACTAL_NZ = 29


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _parse_args() -> argparse.Namespace:
    projection_names = tuple(projection[0] for projection in DRAFT_PROJECTIONS)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-counts", type=_positive_int, nargs="+", default=list(DEFAULT_TOKEN_COUNTS))
    parser.add_argument("--projections", choices=projection_names, nargs="+", default=list(projection_names))
    parser.add_argument("--strategies", choices=DEFAULT_STRATEGIES, nargs="+", default=list(DEFAULT_STRATEGIES))
    parser.add_argument("--warmup-steps", type=_nonnegative_int, default=20)
    parser.add_argument("--profile-steps", type=_positive_int, default=200)
    parser.add_argument("--device", type=_nonnegative_int, default=0, help="Logical NPU after visibility filtering")
    parser.add_argument(
        "--quant-weight-layout",
        choices=("nd", "nz"),
        default="nz",
        help="Layout supplied to the repository W8A16/W8A8 matmul methods",
    )
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--output-json")
    return parser.parse_args()


def _capture(operation: Callable[[], torch.Tensor]) -> tuple[torch.npu.NPUGraph, torch.Tensor]:
    # Qualify eager execution first.  In particular, an unavailable quantized
    # kernel must fail before entering an ACLGraph capture context.
    operation()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = operation()
    torch.npu.synchronize()
    return graph, output


def _measure(graph: torch.npu.NPUGraph, warmup_steps: int, profile_steps: int) -> float:
    for _ in range(warmup_steps):
        graph.replay()
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(profile_steps):
        graph.replay()
    torch.npu.synchronize()
    return (time.perf_counter() - started) * 1_000_000 / profile_steps


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_float = actual.float()
    expected_float = expected.float()
    difference = actual_float - expected_float
    expected_rms = expected_float.square().mean().sqrt().clamp_min(1e-12)
    actual_norm = torch.linalg.vector_norm(actual_float)
    expected_norm = torch.linalg.vector_norm(expected_float)
    denominator = (actual_norm * expected_norm).clamp_min(1e-12)
    cosine = torch.dot(actual_float.reshape(-1), expected_float.reshape(-1)) / denominator
    return {
        "max_abs_error": float(difference.abs().max().cpu()),
        "mean_abs_error": float(difference.abs().mean().cpu()),
        "relative_rms_error": float((difference.square().mean().sqrt() / expected_rms).cpu()),
        "cosine_similarity": float(cosine.cpu()),
    }


def _make_weight(output_size: int, input_size: int) -> torch.Tensor:
    # Qwen dense weights are small while post-RMSNorm activations are O(1).
    return (torch.randn(output_size, input_size, dtype=torch.float32, device="npu") * 0.02).to(torch.bfloat16)


def _make_source(token_count: int, input_size: int) -> torch.Tensor:
    return torch.randn(token_count, input_size, dtype=torch.float32, device="npu").to(torch.bfloat16)


def _quantize_weight_per_output_channel(
    weight: torch.Tensor,
    *,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create the tensors consumed by the repository's W8A16/W8A8 methods.

    The method classes normally receive these tensors from a ModelSlim
    checkpoint.  This benchmark uses deterministic symmetric RTN so it can
    qualify operators without installing a quantization tool.  Weight layout
    mirrors ``process_weights_after_loading``: [N, K] -> [K, N] -> optional NZ.
    """

    weight_float = weight.float()
    scale_float = weight_float.abs().amax(dim=1).div(127.0).clamp_min(1e-8)
    quantized = weight_float.div(scale_float.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
    del weight_float
    quantized = quantized.transpose(0, 1).contiguous()
    if layout == "nz":
        quantized = torch_npu.npu_format_cast(quantized, ACL_FORMAT_FRACTAL_NZ)
    scale = scale_float.to(dtype=weight.dtype).contiguous()
    offset = torch.zeros_like(scale)
    return quantized, scale, offset


def _unsupported_reason(strategy: str) -> str | None:
    if strategy == "w8a16" and not hasattr(torch_npu, "npu_weight_quant_batchmatmul"):
        return "torch_npu.npu_weight_quant_batchmatmul is unavailable"
    if strategy == "w8a8_dynamic":
        missing = [name for name in ("npu_dynamic_quant", "npu_quant_matmul") if not hasattr(torch_npu, name)]
        if missing:
            return "missing torch_npu API(s): " + ", ".join(missing)
    return None


def _make_operations(
    source_bf16: torch.Tensor,
    weight_bf16: torch.Tensor,
    quant_layer: SimpleNamespace,
) -> dict[str, Callable[[], torch.Tensor]]:
    source_fp16 = source_bf16.to(torch.float16)
    weight_fp16 = weight_bf16.to(torch.float16)
    w8a16_method = AscendW8A16LinearMethod()
    dynamic_w8a8_method = AscendW8A8DynamicLinearMethod()
    return {
        "bf16": lambda: F.linear(source_bf16, weight_bf16),
        "fp16": lambda: F.linear(source_fp16, weight_fp16),
        "w8a16": lambda: w8a16_method.apply(quant_layer, source_bf16),
        "w8a8_dynamic": lambda: dynamic_w8a8_method.apply(quant_layer, source_bf16),
    }


def _benchmark_shape(
    *,
    projection: str,
    input_size: int,
    output_size: int,
    token_count: int,
    weight_bf16: torch.Tensor,
    quant_layer: SimpleNamespace,
    strategies: list[str],
    warmup_steps: int,
    profile_steps: int,
    quant_weight_layout: str,
) -> list[dict[str, Any]]:
    source_bf16 = _make_source(token_count, input_size)
    operations = _make_operations(source_bf16, weight_bf16, quant_layer)
    expected = operations["bf16"]().detach().clone()
    torch.npu.synchronize()

    results: list[dict[str, Any]] = []
    for strategy in strategies:
        result: dict[str, Any] = {
            "projection": projection,
            "token_count": token_count,
            "input_size": input_size,
            "output_size": output_size,
            "strategy": strategy,
            "quant_weight_layout": quant_weight_layout if strategy.startswith("w8") else None,
            "status": "ok",
            "latency_microseconds": None,
            "speedup_vs_bf16": None,
        }
        unsupported = _unsupported_reason(strategy)
        if unsupported is not None:
            result["status"] = "unsupported"
            result["error"] = unsupported
            results.append(result)
            continue

        graph = None
        try:
            graph, output = _capture(operations[strategy])
            result["latency_microseconds"] = _measure(graph, warmup_steps, profile_steps)
            # One explicit replay makes it unambiguous that errors describe the
            # captured path rather than the eager qualification invocation.
            graph.replay()
            torch.npu.synchronize()
            result["output_dtype"] = str(output.dtype).removeprefix("torch.")
            result.update(_errors(output, expected))
        except Exception as error:  # keep a partial matrix useful for qualification
            result["status"] = "error"
            result["error"] = f"{type(error).__name__}: {error}"
        finally:
            del graph
        results.append(result)

    baseline = next(
        (
            result["latency_microseconds"]
            for result in results
            if result["strategy"] == "bf16" and result["status"] == "ok"
        ),
        None,
    )
    if baseline is not None:
        for result in results:
            latency = result["latency_microseconds"]
            if result["status"] == "ok" and latency is not None and math.isfinite(latency) and latency > 0:
                result["speedup_vs_bf16"] = baseline / latency
    return results


def main() -> None:
    args = _parse_args()
    if not torch.npu.is_available():
        raise RuntimeError("This microbenchmark requires an Ascend NPU runtime.")
    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    torch.manual_seed(args.seed)

    selected_projections = [projection for projection in DRAFT_PROJECTIONS if projection[0] in args.projections]
    results: list[dict[str, Any]] = []
    for projection, input_size, output_size in selected_projections:
        weight_bf16 = _make_weight(output_size, input_size)
        quant_weight, weight_scale, weight_offset = _quantize_weight_per_output_channel(
            weight_bf16,
            layout=args.quant_weight_layout,
        )
        # Both repository methods consume the same symmetric int8 weight and
        # per-output-channel scale.  Dynamic W8A8 additionally quantizes x in
        # its existing apply() implementation.
        quant_layer = SimpleNamespace(
            weight=quant_weight,
            weight_scale=weight_scale,
            weight_offset=weight_offset,
            prefix=f"draft.{projection}",
            params_dtype=torch.bfloat16,
        )
        for token_count in args.token_counts:
            results.extend(
                _benchmark_shape(
                    projection=projection,
                    input_size=input_size,
                    output_size=output_size,
                    token_count=token_count,
                    weight_bf16=weight_bf16,
                    quant_layer=quant_layer,
                    strategies=args.strategies,
                    warmup_steps=args.warmup_steps,
                    profile_steps=args.profile_steps,
                    quant_weight_layout=args.quant_weight_layout,
                )
            )
            gc.collect()
        del quant_layer, quant_weight, weight_scale, weight_offset, weight_bf16
        gc.collect()
        torch.npu.empty_cache()

    payload = {
        "benchmark": "Qwen3-0.6B TP1 draft projection ACLGraph replay",
        "scope": "operator-only; no model, KV cache, scheduler, or service path",
        "visible_devices": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
        "logical_device": args.device,
        "seed": args.seed,
        "token_counts": args.token_counts,
        "warmup_steps": args.warmup_steps,
        "profile_steps": args.profile_steps,
        "strategies": args.strategies,
        "quantization": {
            "weight_algorithm": "symmetric RTN per output channel",
            "weight_layout": args.quant_weight_layout,
            "reference": "BF16 ND F.linear",
            "dynamic_w8a8": "repository AscendW8A8DynamicLinearMethod; capability-probed",
            "w8a16": "repository AscendW8A16LinearMethod",
        },
        "results": results,
    }
    rendered = json.dumps(payload, indent=2, allow_nan=False)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
