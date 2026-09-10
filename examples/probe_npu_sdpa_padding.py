# SPDX-License-Identifier: Apache-2.0
"""Isolate masked attention padding numerics on ONE explicitly assigned NPU.

This is not a speculative decoder, model benchmark or graph-tolerance change.
Identical BF16-quantized Q/K/V are evaluated with and without zero masked tails;
an independent CPU float64 score/softmax/value computation is the reference.

    ASCEND_RT_VISIBLE_DEVICES=<one-free-device> python examples/probe_npu_sdpa_padding.py \
      --device npu:0 --lengths 256 257 258 259 260 --pad-to 512 \
      --hf32 default off on --internal-formats on off --output /path/sdpa-probe.json

Optional --precision-mode must_keep_origin_dtype and --cube-math keep_dtype
run separate process-local precision controls. SDP backend selector flags are
not used: this installed torch_npu documents them as global flag setters, not
proof that its device operator selected a different kernel implementation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--lengths", type=int, nargs="+", default=[256, 257, 258, 259, 260])
    parser.add_argument("--pad-to", type=int, default=512)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dims", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0, 4.0])
    parser.add_argument("--masks", choices=("tail", "holes"), nargs="+", default=["tail", "holes"])
    parser.add_argument("--hf32", choices=("default", "off", "on"), nargs="+", default=["default", "off", "on"])
    parser.add_argument("--internal-formats", choices=("default", "off", "on"), nargs="+", default=["on", "off"])
    parser.add_argument("--precision-mode", choices=("default", "must_keep_origin_dtype"), default="default")
    parser.add_argument("--cube-math", choices=("default", "keep_dtype", "fp32_add"), default="default")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--atol", type=float, default=0.001)
    parser.add_argument("--rtol", type=float, default=0.001)
    parser.add_argument("--output", required=True)
    return parser


def _validate(args):
    if min(args.heads, args.kv_heads, args.pad_to, *args.lengths, *args.head_dims) <= 0:
        raise ValueError("Attention dimensions must be positive")
    if args.heads % args.kv_heads or max(args.lengths) > args.pad_to:
        raise ValueError("Query heads must divide KV expansion and pad_to must cover every length")
    if any(not math.isfinite(value) or value <= 0 for value in args.scales):
        raise ValueError("Input scales must be finite and positive")
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("Comparison tolerances must be non-negative")
    if not (args.device == "cpu" or args.device.startswith("npu:")):
        raise ValueError("This probe accepts cpu or one explicit npu:<index> device")


def _inputs(length, padded_length, heads, kv_heads, head_dim, mask_kind, seed, input_scale):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    query = (torch.randn(heads, head_dim, generator=generator) * input_scale).to(torch.bfloat16)
    keys = (torch.randn(length, kv_heads, head_dim, generator=generator) * input_scale).to(torch.bfloat16)
    values = torch.randn(length, kv_heads, head_dim, generator=generator).to(torch.bfloat16)
    visible = torch.zeros(padded_length, dtype=torch.bool)
    visible[:length] = True
    if mask_kind == "holes":
        # Keep the complete prefix except selected sibling/tree slots near
        # the frontier. Always leave the root/frontier and at least one key.
        for offset in (2, 4, 6):
            if length - offset > 0:
                visible[length - offset] = False
    padded_keys = torch.zeros(padded_length, kv_heads, head_dim, dtype=torch.bfloat16)
    padded_values = torch.zeros_like(padded_keys)
    padded_keys[:length] = keys
    padded_values[:length] = values
    keys = torch.where(visible.view(-1, 1, 1), padded_keys, 0)
    values = torch.where(visible.view(-1, 1, 1), padded_values, 0)
    # Match native dense attention's repeat/transpose and singleton query.
    keys = keys.transpose(0, 1).repeat_interleave(heads // kv_heads, dim=0).unsqueeze(0)
    values = values.transpose(0, 1).repeat_interleave(heads // kv_heads, dim=0).unsqueeze(0)
    query = query.unsqueeze(0).unsqueeze(2)
    return query, keys, values, visible.view(1, 1, 1, -1)


def _reference(query, keys, values, mask):
    """Independent CPU double arithmetic; does not call PyTorch SDPA."""
    import torch

    scores = torch.matmul(query.double(), keys.double().transpose(-1, -2)) / math.sqrt(query.shape[-1])
    scores = scores.masked_fill(~mask, -float("inf"))
    probabilities = scores.softmax(-1)
    return torch.matmul(probabilities, values.double())


def _execute(method, query, keys, values, mask):
    import torch
    import torch.nn.functional as functional

    if method != "sdpa_bf16":
        query, keys, values = query.float(), keys.float(), values.float()
    scale = 1.0 / math.sqrt(query.shape[-1])
    if method == "manual_fp32":
        scores = torch.matmul(query, keys.transpose(-1, -2)) * scale
        probabilities = scores.masked_fill(~mask, -float("inf")).softmax(-1)
        return torch.matmul(probabilities, values)
    if method == "sdpa_fp32_additive":
        mask = torch.where(mask, 0.0, -float("inf"))
    return functional.scaled_dot_product_attention(
        query,
        keys,
        values,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )


def _comparison(actual, reference, *, atol, rtol):
    import torch

    actual, reference = actual.double().cpu(), reference.double().cpu()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    if not finite:
        return {
            "finite": False,
            "within_tolerance": False,
            "nonfinite_actual": int((~torch.isfinite(actual)).sum()),
            "nonfinite_reference": int((~torch.isfinite(reference)).sum()),
        }
    error = (actual - reference).abs()
    return {
        "finite": True,
        "exact_equal": bool(torch.equal(actual, reference)),
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
        "rmse": float(error.square().mean().sqrt()),
        "outside_tolerance_count": int((error > atol + rtol * reference.abs()).sum()),
        "within_tolerance": bool((error <= atol + rtol * reference.abs()).all()),
    }


def _option(torch_npu, name):
    value = torch_npu._C._npu_getOption(name)
    return value.decode() if isinstance(value, bytes) else value


def main(argv=None):
    args = _build_parser().parse_args(argv)
    _validate(args)
    import torch

    npu = args.device.startswith("npu:")
    torch_npu = None
    if npu:
        import torch_npu

        torch.npu.set_device(args.device)
        if args.precision_mode != "default":
            torch.npu.set_option({"ACL_PRECISION_MODE": args.precision_mode})
        if args.cube_math != "default":
            torch.npu.matmul.cube_math_type = (
                torch.npu.CubeMathType.KEEP_DTYPE
                if args.cube_math == "keep_dtype"
                else torch.npu.CubeMathType.USE_FP32_ADD
            )
        initial_hf32 = torch.npu.matmul.allow_hf32
        initial_internal = _option(torch_npu, "ALLOW_INTERNAL_FORMAT")
        controls = list(itertools.product(args.hf32, args.internal_formats))
    else:
        controls = [("not_applicable", "not_applicable")]
    report = {
        "purpose": "Isolated masked attention numerics, not a speculative decoder or throughput benchmark",
        "config": vars(args),
        "torch_version": torch.__version__,
        "torch_npu_version": None if torch_npu is None else torch_npu.__version__,
        "device_name": torch.npu.get_device_name(args.device) if npu else "CPU float32 test",
        "operator_registration": {
            name: [
                line
                for line in torch._C._dispatch_dump_table(name).splitlines()
                if line.startswith(("PrivateUse1:", "AutogradPrivateUse1:", "AutocastPrivateUse1:"))
            ]
            for name in ("aten::scaled_dot_product_attention", "aten::_scaled_dot_product_attention_math")
        },
        "reference": "Identical BF16-quantized QKV; independent CPU float64 matmul-softmax-matmul",
        "status": "running",
        "cases": [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    methods = ("sdpa_bf16", "sdpa_fp32", "sdpa_fp32_additive", "manual_fp32")
    try:
        for hf32, internal in controls:
            if npu:
                torch.npu.matmul.allow_hf32 = initial_hf32 if hf32 == "default" else hf32 == "on"
                if internal != "default":
                    torch.npu.config.allow_internal_format = internal == "on"
                elif initial_internal is not None:
                    torch.npu.set_option({"ALLOW_INTERNAL_FORMAT": initial_internal})
            options = (
                {
                    name: _option(torch_npu, name)
                    for name in ("ALLOW_MATMUL_HF32", "ALLOW_INTERNAL_FORMAT", "ACL_PRECISION_MODE", "CUBE_MATH_TYPE")
                }
                if npu
                else {}
            )
            for length, head_dim, mask_kind, scale in itertools.product(
                args.lengths,
                args.head_dims,
                args.masks,
                args.scales,
            ):
                full = _inputs(length, args.pad_to, args.heads, args.kv_heads, head_dim, mask_kind, args.seed, scale)
                original = (
                    full[0],
                    full[1][..., :length, :].contiguous(),
                    full[2][..., :length, :].contiguous(),
                    full[3][..., :length],
                )
                reference = _reference(*original)
                padded_reference = _reference(*full)
                original_device = tuple(value.to(args.device) for value in original)
                padded_device = tuple(value.to(args.device) for value in full)
                for method in methods:
                    with torch.inference_mode():
                        unpadded = _execute(method, *original_device).cpu().clone()
                        padded = _execute(method, *padded_device).cpu().clone()
                    comparison = {"atol": args.atol, "rtol": args.rtol}
                    case = {
                        "length": length,
                        "pad_to": args.pad_to,
                        "head_dim": head_dim,
                        "mask": mask_kind,
                        "input_scale": scale,
                        "method": method,
                        "requested_hf32": hf32,
                        "requested_internal_format": internal,
                        "actual_options": options,
                        "input_strides": {
                            "query": list(original_device[0].stride()),
                            "original_key": list(original_device[1].stride()),
                            "padded_key": list(padded_device[1].stride()),
                        },
                        "cpu_reference_padding_invariance": _comparison(padded_reference, reference, **comparison),
                        "original_vs_cpu64": _comparison(unpadded, reference, **comparison),
                        "padded_vs_cpu64": _comparison(padded, reference, **comparison),
                        "padded_vs_original": _comparison(padded, unpadded, **comparison),
                        "rounded_bf16_padding_difference": _comparison(
                            padded.bfloat16(), unpadded.bfloat16(), **comparison
                        ),
                    }
                    report["cases"].append(case)
                save()
                print(
                    json.dumps(
                        {
                            "event": "case_group",
                            "length": length,
                            "head_dim": head_dim,
                            "mask": mask_kind,
                            "hf32": hf32,
                            "internal": internal,
                            "padding_max_errors": {
                                row["method"]: row["padded_vs_original"].get("max_abs_error")
                                for row in report["cases"][-len(methods) :]
                            },
                        }
                    ),
                    flush=True,
                )
        report["status"] = "complete"
        save()
    except Exception as error:
        report.update(status="failed", error=str(error))
        save()
        raise
    finally:
        if npu:
            torch.npu.matmul.allow_hf32 = initial_hf32
            if initial_internal is not None:
                torch.npu.set_option({"ALLOW_INTERNAL_FORMAT": initial_internal})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
