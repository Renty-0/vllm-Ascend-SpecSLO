# SPDX-License-Identifier: Apache-2.0
"""Diagnose exact-logit tie rules without changing production greedy behavior.

Single-device operator check (no model weights needed)::

    python examples/check_specslo_greedy_ties.py --device npu --operators-only --output ties.json

NativeLMHead distributed check with tiny synthetic weights, not the 32B model::

    torchrun --standalone --nproc_per_node=3 examples/check_specslo_greedy_ties.py \
      --device npu --nz-weight --output head-ties-tp3.json

``compare_native_head(head, hidden, valid_vocab_size)`` can also be called by
an existing multi-rank model probe. Every TP rank must call it in identical
order. It reuses the SAME hidden tensor for greedy and full-head projection,
reports projection differences, and never reruns the transformer or KV cache.
Exact equality is intentional: this diagnoses ties, not approximate matches.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "npu"), default="cpu")
    parser.add_argument("--dtypes", nargs="+", choices=("bfloat16", "float32"), default=["bfloat16", "float32"])
    parser.add_argument("--vocab-sizes", nargs="+", type=int, default=[16, 257, 50646, 151936])
    parser.add_argument("--head-vocab-size", type=int, default=151936)
    parser.add_argument("--operators-only", action="store_true")
    parser.add_argument("--nz-weight", action="store_true", help="NPU-only: store NativeLMHead weights in FRACTAL_NZ.")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _validate_args(args):
    if any(value < 2 for value in args.vocab_sizes) or args.head_vocab_size < 2:
        raise ValueError("Vocabulary sizes must be at least 2")
    if args.nz_weight and args.device != "npu":
        raise ValueError("NZ weight conversion requires --device npu")


def compare_logit_ties(logits, *, token_offset=0, max_ties_reported=32):
    """Compare device max/argmax against an explicit minimum-ID exact-tie rule."""
    import torch

    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError("Expected non-empty [rows, vocabulary] logits")
    logits = logits.detach()
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("The tie probe requires finite logits; NaN/Inf semantics are a separate test")
    max_values, max_ids = logits.max(dim=-1)
    argmax_ids = logits.argmax(dim=-1)
    float32_argmax_ids = logits.float().argmax(dim=-1)
    exact_ties = logits == logits.amax(dim=-1, keepdim=True)
    ids = torch.arange(logits.shape[1], dtype=torch.long, device=logits.device).expand(logits.shape[0], -1)
    explicit_ids = torch.where(exact_ties, ids, logits.shape[1]).amin(dim=-1)
    cpu_logits = logits.detach().float().cpu()
    cpu_argmax_ids = cpu_logits.argmax(dim=-1)
    rows = []
    for row in range(logits.shape[0]):
        tied = exact_ties[row].nonzero().flatten().cpu().tolist()
        max_id = int(max_ids[row]) + token_offset
        argmax_id = int(argmax_ids[row]) + token_offset
        explicit_id = int(explicit_ids[row]) + token_offset
        cpu_id = int(cpu_argmax_ids[row]) + token_offset
        rows.append(
            {
                "row": row,
                "max_value": float(max_values[row]),
                "max_index": max_id,
                "argmax_index": argmax_id,
                "float32_argmax_index": int(float32_argmax_ids[row]) + token_offset,
                "explicit_min_exact_tie_id": explicit_id,
                "cpu_argmax_index": cpu_id,
                "exact_tie_count": len(tied),
                "exact_tie_ids_first32": [index + token_offset for index in tied[:max_ties_reported]],
                "max_matches_explicit": max_id == explicit_id,
                "argmax_matches_explicit": argmax_id == explicit_id,
                "explicit_matches_cpu": explicit_id == cpu_id,
            }
        )
    return {
        "dtype": str(logits.dtype),
        "device": str(logits.device),
        "shape": list(logits.shape),
        "stride": list(logits.stride()),
        "contiguous": logits.is_contiguous(),
        "token_offset": token_offset,
        "rows": rows,
    }


def compare_native_head(head, hidden_states, vocabulary_size):
    """Same-hidden local reduction / full head / production greedy comparison.

    This helper includes TP collectives. Never call it on only the leader of a
    distributed head. Returned local details differ by rank; global greedy and
    full-head outputs should not. Both original BF16 logits and their FP32 CPU
    values remain distinguishable in the report through explicit dtype fields.
    """
    import torch.nn.functional as F

    local_logits = F.linear(hidden_states, head.weight)
    full_logits = head(hidden_states)[:, :vocabulary_size]
    greedy = head.greedy(hidden_states, vocabulary_size)
    valid_local = max(0, min(head.vocab_end, vocabulary_size) - head.vocab_start)
    local_report = None
    projection_error = None
    if valid_local:
        valid_logits = local_logits[:, :valid_local]
        local_report = compare_logit_ties(valid_logits, token_offset=head.vocab_start)
        full_slice = full_logits[:, head.vocab_start : head.vocab_start + valid_local]
        projection_error = float((valid_logits.detach().float() - full_slice.detach().float()).abs().max())
    full_report = compare_logit_ties(full_logits)
    greedy_ids = greedy.detach().cpu().tolist()
    full_argmax = [row["argmax_index"] for row in full_report["rows"]]
    explicit = [row["explicit_min_exact_tie_id"] for row in full_report["rows"]]
    return {
        "tp_rank": head.context.rank,
        "tp_size": head.context.size,
        "valid_vocabulary_size": vocabulary_size,
        "local_vocab_start": head.vocab_start,
        "local_vocab_end": head.vocab_end,
        "valid_local_vocabulary_size": valid_local,
        "hidden_shape": list(hidden_states.shape),
        "hidden_dtype": str(hidden_states.dtype),
        "local": local_report,
        "full": full_report,
        "same_hidden_local_vs_full_projection_max_abs_error": projection_error,
        "production_greedy_ids": greedy_ids,
        "full_head_argmax_ids": full_argmax,
        "explicit_min_tie_ids": explicit,
        "production_greedy_matches_full_argmax": greedy_ids == full_argmax,
        "production_greedy_matches_explicit_min_ties": greedy_ids == explicit,
        "interpretation": (
            "An exact tie does not by itself identify prefix rounding. "
            "This test holds hidden fixed; differing projections are reported separately."
        ),
    }


def _synthetic_logits(vocab_size, device, dtype):
    import torch

    rows = torch.full((3, vocab_size), -10.0, device=device, dtype=dtype)
    pairs = [(1052, 1449, 38.0), (594, 752, 34.5), (0, vocab_size - 1, 1.0)]
    for row, (first, second, value) in enumerate(pairs):
        if second >= vocab_size:
            first, second = 0, vocab_size - 1
        rows[row, first] = rows[row, second] = value
    return rows


def _operator_probes(args, device):
    import torch

    probes = []
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        for vocab_size in args.vocab_sizes:
            logits = _synthetic_logits(vocab_size, device, dtype)
            for layout in ("contiguous", "truncated", "stride2"):
                if layout == "contiguous":
                    view = logits
                elif layout == "truncated":
                    storage = torch.zeros((3, vocab_size + 17), dtype=dtype, device=device)
                    storage[:, :vocab_size].copy_(logits)
                    view = storage[:, :vocab_size]
                else:
                    storage = torch.zeros((3, vocab_size * 2), dtype=dtype, device=device)
                    storage[:, ::2].copy_(logits)
                    view = storage[:, ::2]
                probes.append({"layout": layout, **compare_logit_ties(view)})
        # Inputs distinct in FP32 become EXACT ties after BF16 quantization.
        values = torch.tensor([[38.01, 38.02], [34.501, 34.502]], dtype=torch.float32, device=device)
        probes.append(
            {
                "layout": "quantized_near_tie",
                "pre_quantization": values.cpu().tolist(),
                **compare_logit_ties(values.to(dtype)),
            }
        )
    return probes


def _native_head_probe(args, device, rank, world_size, dtype):
    import torch
    import torch.distributed as dist

    from vllm_ascend.spec_decode.pearl.native_model import NativeLMHead, NativeTPContext

    padded_vocab = math.ceil(args.head_vocab_size / world_size) * world_size
    context = NativeTPContext(
        group=dist.group.WORLD if world_size > 1 else None, rank=rank, size=world_size, leader_rank=0
    )
    head = NativeLMHead(padded_vocab, 16, context).to(device=device, dtype=dtype)
    shard = padded_vocab // world_size
    logits = _synthetic_logits(args.head_vocab_size, device, dtype)
    # Add a tie straddling the rank-0/rank-1 boundary; TP1 still tests tie ends.
    cross_rank = torch.full((1, args.head_vocab_size), -10.0, device=device, dtype=dtype)
    left, right = (shard - 1, shard) if world_size > 1 else (0, args.head_vocab_size - 1)
    cross_rank[0, left] = cross_rank[0, right] = 42.0
    logits = torch.cat((logits, cross_rank), dim=0)
    weight = torch.zeros((padded_vocab, 16), device=device, dtype=dtype)
    weight[: args.head_vocab_size, :4].copy_(logits.t())
    # Padded suffix must never win even though its scores are deliberately high.
    weight[args.head_vocab_size :, :4] = 100.0
    head.weight.data.copy_(weight[head.vocab_start : head.vocab_end])
    if args.nz_weight:
        import torch_npu

        head.weight.data = torch_npu.npu_format_cast(head.weight.data, 29)
    hidden = torch.eye(16, dtype=dtype, device=device)[:4]
    result = compare_native_head(head, hidden, args.head_vocab_size)
    result["weight_format"] = "FRACTAL_NZ" if args.nz_weight else "ND"
    result["cross_rank_tie_ids"] = [left, right]
    result["synthetic_weight_construction"] = (
        "Identity hidden selects exact score columns, plus masked padded suffix=100."
    )
    return result


def run(args):
    import torch
    import torch.distributed as dist

    _validate_args(args)
    rank, world_size = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.device == "npu":
        import torch_npu  # noqa: F401 -- registers device and HCCL backend

        torch.npu.set_device(local_rank)
        device = torch.device(f"npu:{local_rank}")
    else:
        device = torch.device("cpu")
    if world_size > 1:
        dist.init_process_group(backend="hccl" if args.device == "npu" else "gloo")
    try:
        with torch.inference_mode():
            result = {"rank": rank, "device": str(device), "operators": _operator_probes(args, device)}
            if not args.operators_only:
                result["native_heads"] = [
                    _native_head_probe(args, device, rank, world_size, getattr(torch, name)) for name in args.dtypes
                ]
        ranks = [None] * world_size
        if world_size > 1:
            dist.all_gather_object(ranks, result)
        else:
            ranks = [result]
        report = {
            "config": {**vars(args), "output": str(args.output)},
            "world_size": world_size,
            "torch_version": torch.__version__,
            "ranks": ranks,
            "production_semantics_changed": False,
            "scope": (
                "Exact ties and same-hidden synthetic NativeLMHead. "
                "Not proof of the cause of a historical output divergence."
            ),
        }
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            print(f"Saved tie diagnostics: {args.output}", flush=True)
        return report
    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    run(_build_parser().parse_args())
