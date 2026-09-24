# SPDX-License-Identifier: Apache-2.0
"""Probe production FULL-mask tree FIA without loading a model.

The contract is copied from attention_v1.py's tree_attention branch: paged
KV, TND, sparse_mode=1, inner_precise=1, [requests, 1, maxQ, max_context]
boolean mask (True=blocked), and actual query/KV lengths. Query tensors have
exactly sum(query_counts) rows; only the mask envelope is rectangular.

Q=1 scratch/holes are mandatory cases: sparse_mode handling may differ for
single-query kernels, so blocked K/V perturbations must preserve that row.
BNSD is additionally probed for uniform Q=1 without padding any query.

    ASCEND_RT_VISIBLE_DEVICES=<assigned-card> python examples/check_specslo_tree_fia.py \
      --device npu:0 --graph --output /path/tree-fia.json

CPU mode validates the independent float64 oracle and input accounting only;
it is never reported as NPU FIA/graph evidence. This changes no model default.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--kv-heads", type=int, default=3)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--contexts", type=int, nargs="+", default=[127, 128, 255, 256])
    parser.add_argument("--seed", type=int, default=867)
    parser.add_argument("--atol", type=float, default=0.001)
    parser.add_argument("--rtol", type=float, default=0.005)
    parser.add_argument("--inner-precise", type=int, choices=(1, 2), default=1)
    parser.add_argument("--benchmark-batch", type=int, default=0)
    parser.add_argument(
        "--benchmark-fixed-kv-capacity",
        type=int,
        default=0,
        help=(
            "Also benchmark the same FULL mask with every request's host KV "
            "length fixed to this capacity. Masked tail pages remain valid, "
            "which models a graph signature that can skip per-layer task updates."
        ),
    )
    parser.add_argument("--benchmark-warmup", type=int, default=10)
    parser.add_argument("--benchmark-repeats", type=int, default=30)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--output", required=True)
    return parser


def _validate(args):
    if min(args.heads, args.kv_heads, args.head_dim, args.block_size, args.max_context, *args.contexts) <= 0:
        raise ValueError("FIA dimensions and contexts must be positive")
    if args.heads % args.kv_heads:
        raise ValueError("FIA query heads must be divisible by KV heads")
    if args.max_context % args.block_size or max(args.contexts) + 16 > args.max_context:
        raise ValueError("Cache capacity must be block aligned and cover tree scratch positions")
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("Oracle tolerances must be non-negative")
    if args.benchmark_batch < 0 or args.benchmark_warmup < 0 or args.benchmark_repeats <= 0:
        raise ValueError("Benchmark sizes must be non-negative and repeats must be positive")
    if args.benchmark_fixed_kv_capacity < 0 or args.benchmark_fixed_kv_capacity > args.max_context:
        raise ValueError("Fixed benchmark KV capacity must fit max_context")
    if args.device != "cpu" and not args.device.startswith("npu:"):
        raise ValueError("Choose cpu or one explicitly assigned npu:<index>")
    if args.device == "cpu" and args.graph:
        raise ValueError("CPU mode cannot claim an NPU graph probe")


def _parents(query_count):
    # Four nodes correspond to ancestor-closed selection [0, 1, 3, 4]
    # from exploratory parents [-1, 0, 1, -1, 3, 4], not a linear prefix.
    return {1: [], 2: [-1], 5: [-1, 0, -1, 2]}[query_count]


def _make_case(contexts, counts, *, scratch, args, seed):
    import torch

    batch = len(counts)
    if len(contexts) != batch or any(count not in (1, 2, 5) for count in counts):
        raise ValueError("Probe needs one supported query count per context")
    generator = torch.Generator().manual_seed(seed)
    blocks_per_request = args.max_context // args.block_size
    block_tables = torch.randperm(batch * blocks_per_request, generator=generator).reshape(batch, -1).int()
    cache_shape = (batch * blocks_per_request, args.block_size, args.kv_heads, args.head_dim)
    keys = torch.randn(cache_shape, generator=generator).bfloat16()
    values = torch.randn(cache_shape, generator=generator).bfloat16()
    query = torch.randn(sum(counts), args.heads, args.head_dim, generator=generator).bfloat16()
    mask = torch.ones(batch, 1, max(counts), args.max_context, dtype=torch.bool)
    visible_rows, positions, parents_by_request, sequence_lengths = [], [], [], []
    for request, (prefix, count) in enumerate(zip(contexts, counts)):
        parents = _parents(count)
        parents_by_request.append(parents)
        if scratch and count == 1:
            # One frontier following a sparse dependency path. The physical
            # gaps before it are siblings, not new query/candidate rows.
            physical = [prefix + 5]
            dependencies = [prefix + 1, prefix + 3]
        elif scratch:
            physical = [prefix, *[prefix + 3 + 2 * node for node in range(count - 1)]]
            dependencies = []
        else:
            physical = list(range(prefix, prefix + count))
            dependencies = []
        positions.append(physical)
        sequence_lengths.append(max(physical) + 1)
        for row in range(count):
            visible = list(range(prefix)) + dependencies + [physical[0]]
            node = row - 1
            while node >= 0:
                visible.append(physical[node + 1])
                node = parents[node]
            visible = sorted(set(visible))
            mask[request, 0, row, visible] = False
            visible_rows.append(visible)
    cumulative, total = [], 0
    for count in counts:
        total += count
        cumulative.append(total)
    return {
        "query": query, "keys": keys, "values": values, "mask": mask,
        "block_tables": block_tables, "query_counts": list(counts),
        "actual_seq_lengths": cumulative, "sequence_lengths": sequence_lengths,
        "visible_rows": visible_rows, "positions": positions, "parents": parents_by_request,
        "contexts": list(contexts), "scratch": scratch, "heads": args.heads,
        "kv_heads": args.kv_heads, "head_dim": args.head_dim, "block_size": args.block_size,
        "candidate_count": sum(count - 1 for count in counts),
    }


def _slot(case, request, position):
    return int(case["block_tables"][request, position // case["block_size"]]) * case["block_size"] \
        + position % case["block_size"]


def _oracle(case, *, dtype=None):
    """Gather explicit ancestor positions and compute CPU scores in float64.

    Deliberately does not consume the FIA mask or invoke SDPA: a mask-layout
    mistake in the implementation cannot make its own oracle pass.
    """
    import torch

    dtype = torch.float64 if dtype is None else dtype
    outputs, row = [], 0
    keys, values = case["keys"].flatten(0, 1), case["values"].flatten(0, 1)
    repeat = case["heads"] // case["kv_heads"]
    for request, count in enumerate(case["query_counts"]):
        for _ in range(count):
            slots = torch.tensor([_slot(case, request, value) for value in case["visible_rows"][row]])
            k = keys.index_select(0, slots).to(dtype).transpose(0, 1).repeat_interleave(repeat, 0)
            v = values.index_select(0, slots).to(dtype).transpose(0, 1).repeat_interleave(repeat, 0)
            q = case["query"][row].to(dtype).unsqueeze(1)
            probabilities = ((q @ k.transpose(-1, -2)) / math.sqrt(case["head_dim"])).softmax(-1)
            outputs.append((probabilities @ v).squeeze(1))
            row += 1
    return torch.stack(outputs)


def _poison_blocked(case, *, nan=False):
    import torch

    result = dict(case)
    result["keys"], result["values"] = case["keys"].clone(), case["values"].clone()
    request = len(case["query_counts"]) - 1
    query_row = min(2, case["query_counts"][request] - 1)
    packed_row = sum(case["query_counts"][:request]) + query_row
    visible = set(case["visible_rows"][packed_row])
    blocked = [position for position in range(case["sequence_lengths"][request]) if position not in visible]
    if not blocked:
        return None
    slots = torch.tensor([_slot(case, request, position) for position in blocked])
    result["keys"].flatten(0, 1)[slots] = float("nan") if nan else 1000.0
    result["values"].flatten(0, 1)[slots] = float("nan") if nan else -1000.0
    result["checked_row"] = packed_row
    result["poisoned_logical_positions"] = blocked
    result["poison_kind"] = "nan" if nan else "large_finite"
    return result


def _device_tensors(case, device, layout):
    query = case["query"]
    if layout == "BNSD":
        if any(count != 1 for count in case["query_counts"]):
            raise ValueError("BNSD probe allows only Q=1; heterogeneous queries must stay unpadded TND")
        query = query.unsqueeze(2)
    return {name: value.to(device) for name, value in {
        "query": query, "key": case["keys"].flatten(2), "value": case["values"].flatten(2),
        "atten_mask": case["mask"], "block_table": case["block_tables"],
    }.items()}


def _fia_kwargs(case, tensors, layout, args=None):
    inner_precise = 1 if args is None else args.inner_precise
    return {
        **tensors, "input_layout": layout, "block_size": case["block_size"],
        "actual_seq_lengths": case["actual_seq_lengths"] if layout == "TND" else case["query_counts"],
        "actual_seq_lengths_kv": case["sequence_lengths"], "num_heads": case["heads"],
        "num_key_value_heads": case["kv_heads"], "scale": 1.0 / math.sqrt(case["head_dim"]),
        "sparse_mode": 1, "inner_precise": inner_precise,
    }


def _comparison(actual, reference, args):
    import torch

    actual, reference = actual.double().cpu(), reference.double().cpu()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    if not finite:
        return {"finite": False, "within_tolerance": False}
    error = (actual - reference).abs()
    return {
        "finite": True, "exact_equal": bool(torch.equal(actual, reference)),
        "max_abs_error": float(error.max()), "mean_abs_error": float(error.mean()),
        "within_tolerance": bool((error <= args.atol + args.rtol * reference.abs()).all()),
    }


def _describe(case, layout):
    return {
        "contexts": case["contexts"], "query_counts": case["query_counts"],
        "actual_seq_lengths": case["actual_seq_lengths"], "sequence_lengths": case["sequence_lengths"],
        "query_tensor_rows": int(case["query"].shape[0]), "candidate_count": case["candidate_count"],
        "mask_shape": list(case["mask"].shape), "cache_positions": case["positions"],
        "parents": case["parents"], "layout": layout, "scratch": case["scratch"],
    }


def _flatten_output(output, layout):
    return output.squeeze(2) if layout == "BNSD" else output


def _graph_replays(cases, args, layout):
    """Reuse production graph-task-update mechanics with FULL-mask parameters.

    Keep this prototype separate: NativeFusedInferAttentionGraphTask currently
    hardcodes sparse_mode=3 during update and cannot safely represent trees.
    """
    import torch
    import torch_npu

    tensors = _device_tensors(cases[0], args.device, layout)
    initial = _fia_kwargs(cases[0], tensors, layout, args)
    output = torch.empty_like(tensors["query"])
    lse = torch.empty(1, dtype=output.dtype, device=args.device)
    workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(**initial)
    for _ in range(3):
        torch_npu.npu_fused_infer_attention_score.out(**initial, workspace=workspace, out=[output, lse])
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    event = torch.npu.ExternalEvent()
    with torch.npu.graph(graph):
        capture_stream = torch.npu.current_stream()
        event.wait(capture_stream)
        event.reset(capture_stream)
        torch.npu.graph_task_group_begin(capture_stream)
        torch_npu.npu_fused_infer_attention_score.out(**initial, workspace=workspace, out=[output, lse])
        handle = torch.npu.graph_task_group_end(capture_stream)
    update_stream = torch.npu.Stream()
    rows = []
    for case in cases:
        current = _device_tensors(case, args.device, layout)
        for name, tensor in tensors.items():
            if tensor.shape != current[name].shape:
                raise ValueError("A replay family must keep all graph buffer shapes stable")
            tensor.copy_(current[name])
        kwargs = _fia_kwargs(case, tensors, layout, args)
        main_stream = torch.npu.current_stream()
        update_stream.wait_stream(main_stream)
        with torch.npu.stream(update_stream):
            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu.npu_fused_infer_attention_score.out(**kwargs, workspace=workspace, out=[output, lse])
            torch.npu.graph_task_update_end(update_stream)
            event.record(update_stream)
        main_stream.wait_stream(update_stream)
        graph.replay()
        torch.npu.synchronize()
        actual = _flatten_output(output.clone(), layout)
        eager, _ = torch_npu.npu_fused_infer_attention_score(**kwargs)
        eager = _flatten_output(eager, layout)
        row = _describe(case, layout)
        row["graph_vs_eager"] = _comparison(actual, eager, args)
        row["graph_vs_cpu64"] = _comparison(actual, _oracle(case), args)
        row["query_count_updated"] = case["query_counts"] != cases[0]["query_counts"]
        rows.append(row)
    return rows


def _benchmark(case, args):
    """Measure one representative FULL-tree FIA op and its graph update path."""
    import torch
    import torch_npu

    tensors = _device_tensors(case, args.device, "TND")
    kwargs = _fia_kwargs(case, tensors, "TND", args)
    for _ in range(args.benchmark_warmup):
        torch_npu.npu_fused_infer_attention_score(**kwargs)
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(args.benchmark_repeats):
        torch_npu.npu_fused_infer_attention_score(**kwargs)
    torch.npu.synchronize()
    eager_ms = (time.perf_counter() - started) * 1000.0 / args.benchmark_repeats

    output = torch.empty_like(tensors["query"])
    lse = torch.empty(1, dtype=output.dtype, device=args.device)
    workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(**kwargs)
    for _ in range(3):
        torch_npu.npu_fused_infer_attention_score.out(
            **kwargs, workspace=workspace, out=[output, lse]
        )
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    event = torch.npu.ExternalEvent()
    with torch.npu.graph(graph):
        capture_stream = torch.npu.current_stream()
        event.wait(capture_stream)
        event.reset(capture_stream)
        torch.npu.graph_task_group_begin(capture_stream)
        torch_npu.npu_fused_infer_attention_score.out(
            **kwargs, workspace=workspace, out=[output, lse]
        )
        handle = torch.npu.graph_task_group_end(capture_stream)
    update_stream = torch.npu.Stream()
    for _ in range(args.benchmark_warmup):
        main_stream = torch.npu.current_stream()
        update_stream.wait_stream(main_stream)
        with torch.npu.stream(update_stream):
            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu.npu_fused_infer_attention_score.out(
                **kwargs, workspace=workspace, out=[output, lse]
            )
            torch.npu.graph_task_update_end(update_stream)
            event.record(update_stream)
        main_stream.wait_stream(update_stream)
        graph.replay()
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(args.benchmark_repeats):
        main_stream = torch.npu.current_stream()
        update_stream.wait_stream(main_stream)
        with torch.npu.stream(update_stream):
            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu.npu_fused_infer_attention_score.out(
                **kwargs, workspace=workspace, out=[output, lse]
            )
            torch.npu.graph_task_update_end(update_stream)
            event.record(update_stream)
        main_stream.wait_stream(update_stream)
        graph.replay()
    torch.npu.synchronize()
    graph_update_ms = (time.perf_counter() - started) * 1000.0 / args.benchmark_repeats

    # A fixed host length signature needs no graph-task rebuild.  The graph
    # still waits on its captured ExternalEvent, so release that dependency
    # once per replay exactly as the production task-update skip path does.
    for _ in range(args.benchmark_warmup):
        event.record(torch.npu.current_stream())
        graph.replay()
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(args.benchmark_repeats):
        event.record(torch.npu.current_stream())
        graph.replay()
    torch.npu.synchronize()
    graph_stable_ms = (time.perf_counter() - started) * 1000.0 / args.benchmark_repeats
    return {
        **_describe(case, "TND"),
        "inner_precise": args.inner_precise,
        "eager_ms_per_op": eager_ms,
        "graph_update_replay_ms_per_op": graph_update_ms,
        "graph_stable_replay_ms_per_op": graph_stable_ms,
        "graph_task_update_saved_ms_per_op": graph_update_ms - graph_stable_ms,
    }


def _fixed_kv_capacity_case(case, capacity):
    """Keep exact visibility while extending FIA's host KV length literal."""

    if capacity <= 0 or any(length > capacity for length in case["sequence_lengths"]):
        raise ValueError("Fixed KV capacity must cover every exact request length")
    result = dict(case)
    result["sequence_lengths"] = [capacity] * len(case["sequence_lengths"])
    result["fixed_kv_capacity"] = capacity
    return result


def main(argv=None):
    args = _parser().parse_args(argv)
    _validate(args)
    import torch

    on_npu = args.device != "cpu"
    if on_npu:
        import torch_npu

        torch.npu.set_device(args.device)
    report = {
        "status": "running", "config": vars(args), "npu_executed": on_npu,
        "contract": (
            "paged FULL mask sparse_mode=1 "
            f"inner_precise={args.inner_precise}; no query padding"
        ),
        "oracle": "independent CPU float64 explicit ancestor gather and score-softmax-value",
        "torch_version": torch.__version__, "cases": [], "graphs": [],
        "benchmarks": [], "errors": [],
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def save():
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    family_counts = ([1, 1], [1, 5], [2, 5])
    try:
        for counts in family_counts:
            family = []
            for number, context in enumerate(args.contexts):
                contexts = [context, args.contexts[(number + 1) % len(args.contexts)]]
                for scratch in (False, True):
                    case = _make_case(contexts, counts, scratch=scratch, args=args, seed=args.seed + number)
                    family.append(case)
                    for layout in (["TND", "BNSD"] if max(counts) == 1 else ["TND"]):
                        description = _describe(case, layout)
                        try:
                            oracle = _oracle(case)
                            dense = _oracle(case, dtype=torch.float32)
                            description["dense_fp32_vs_cpu64"] = _comparison(dense, oracle, args)
                            if on_npu:
                                tensors = _device_tensors(case, args.device, layout)
                                actual, _ = torch_npu.npu_fused_infer_attention_score(
                                    **_fia_kwargs(case, tensors, layout, args)
                                )
                                actual = _flatten_output(actual, layout)
                                description["fia_vs_cpu64"] = _comparison(actual, oracle, args)
                                description["fia_vs_dense_fp32"] = _comparison(actual, dense, args)
                            description["blocked_kv_perturbations"] = []
                            for nan in (False, True):
                                poisoned = _poison_blocked(case, nan=nan)
                                if poisoned is None:
                                    continue
                                selected = poisoned["checked_row"]
                                perturbation = {
                                    "kind": poisoned["poison_kind"], "checked_row": selected,
                                    "positions": poisoned["poisoned_logical_positions"],
                                    "oracle_invariance": _comparison(
                                        _oracle(poisoned)[selected], oracle[selected], args,
                                    ),
                                }
                                if on_npu:
                                    tensors = _device_tensors(poisoned, args.device, layout)
                                    changed, _ = torch_npu.npu_fused_infer_attention_score(
                                        **_fia_kwargs(poisoned, tensors, layout, args)
                                    )
                                    changed = _flatten_output(changed, layout)
                                    perturbation["fia_invariance"] = _comparison(
                                        changed[selected], actual[selected], args,
                                    )
                                description["blocked_kv_perturbations"].append(perturbation)
                        except Exception as error:
                            description["error"] = repr(error)
                            report["errors"].append({**_describe(case, layout), "error": repr(error)})
                        report["cases"].append(description)
                        save()
            if on_npu and args.graph:
                # Same total query rows/maxQ, but a changed heterogeneous
                # cumulative vector must update task arguments on replay.
                if len(set(counts)) > 1:
                    family.append(_make_case(
                        list(reversed(family[-1]["contexts"])), list(reversed(counts)),
                        scratch=True, args=args, seed=args.seed + 91,
                    ))
                for layout in (["TND", "BNSD"] if max(counts) == 1 else ["TND"]):
                    try:
                        report["graphs"].extend(_graph_replays(family, args, layout))
                    except Exception as error:
                        report["errors"].append({"graph_counts": counts, "layout": layout, "error": repr(error)})
                    save()
        if on_npu and args.benchmark_batch:
            benchmark_context = min(args.contexts[0], args.max_context - 16)
            case = _make_case(
                [benchmark_context] * args.benchmark_batch,
                [5] * args.benchmark_batch,
                scratch=False,
                args=args,
                seed=args.seed + 197,
            )
            exact_tensors = _device_tensors(case, args.device, "TND")
            exact_output, _ = torch_npu.npu_fused_infer_attention_score(
                **_fia_kwargs(case, exact_tensors, "TND", args)
            )
            exact_benchmark = _benchmark(case, args)
            exact_benchmark["signature"] = "exact-kv-lengths"
            report["benchmarks"].append(exact_benchmark)
            if args.benchmark_fixed_kv_capacity:
                fixed = _fixed_kv_capacity_case(case, args.benchmark_fixed_kv_capacity)
                fixed_tensors = _device_tensors(fixed, args.device, "TND")
                fixed_output, _ = torch_npu.npu_fused_infer_attention_score(
                    **_fia_kwargs(fixed, fixed_tensors, "TND", args)
                )
                fixed_benchmark = _benchmark(fixed, args)
                fixed_benchmark["signature"] = "fixed-kv-capacity"
                fixed_benchmark["fixed_vs_exact"] = _comparison(
                    fixed_output,
                    exact_output,
                    args,
                )
                report["benchmarks"].append(fixed_benchmark)
            save()
        checks = []
        for row in report["cases"]:
            comparison = row.get("fia_vs_cpu64" if on_npu else "dense_fp32_vs_cpu64", {})
            checks.append(comparison.get("within_tolerance", False))
            checks.extend(
                item.get("fia_invariance" if on_npu else "oracle_invariance", {}).get("within_tolerance", False)
                for item in row.get("blocked_kv_perturbations", [])
            )
        for row in report["graphs"]:
            checks.extend([row["graph_vs_eager"]["exact_equal"], row["graph_vs_cpu64"]["within_tolerance"]])
        report["passed"] = bool(checks) and all(checks) and not report["errors"]
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["fatal_error"] = repr(error)
        raise
    finally:
        save()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
