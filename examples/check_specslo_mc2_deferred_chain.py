# SPDX-License-Identifier: Apache-2.0
"""Validate a deferred TP3 MC2 model-length epilogue chain.

This diagnostic compares three executions over real Qwen3-32B layer payloads:

* ordinary HCCL all-reduce followed by AddRMSNorm;
* the custom epilogue with both calls fully flushed; and
* one model-length attention/down sequence that defers ``READ_DONE`` until the
  following epilogue and flushes only the final down call.

The last path is the production dependency shape.  ``--layer-count`` repeats
the real attention/down payloads to stress the exact number of epilogues used
by the decoder while still making every iteration and ACLGraph replay end with
a drained mailbox.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch_npu

from vllm_ascend.spec_decode.pearl.mc2 import resolve_hccl_comm_name

PairOutput = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-input-dir", type=Path, required=True)
    parser.add_argument("--down-input-dir", type=Path, required=True)
    parser.add_argument("--execution-mode", choices=("eager", "graph"), required=True)
    parser.add_argument("--layer-count", type=int, default=1)
    parser.add_argument("--warmup-replays", type=int, default=20)
    parser.add_argument("--replays-per-sample", type=int, default=20)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--changed-input-replays", type=int, default=5)
    parser.add_argument("--changed-input-delta", type=float, default=0.015625)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.layer_count <= 0:
        raise ValueError("layer-count must be positive")
    if args.warmup_replays < 0 or args.replays_per_sample <= 0 or args.samples < 3:
        raise ValueError("need non-negative warmup, positive replay count, and >=3 samples")
    if args.changed_input_replays < 2:
        raise ValueError("changed-input qualification needs at least two replays")
    if not math.isfinite(args.changed_input_delta) or args.changed_input_delta == 0:
        raise ValueError("changed-input delta must be finite and non-zero")


def _load_payload(path: Path, rank: int) -> dict[str, torch.Tensor]:
    payload = torch.load(path / f"rank-{rank}.pt", map_location="cpu", weights_only=True)
    required = {"activation", "weight", "residual", "gamma"}
    if set(payload) < required:
        raise ValueError(f"{path} is missing tensors: {sorted(required - set(payload))}")
    return {name: payload[name].to(device="npu") for name in required}


def _capture(operation: Callable[[], PairOutput]) -> tuple[torch.npu.NPUGraph, PairOutput]:
    operation()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = operation()
    graph.replay()
    torch.npu.synchronize()
    return graph, output


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _max_rank_value(value: float) -> float:
    tensor = torch.tensor([value], dtype=torch.float32, device="npu")
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.cpu().item())


def _all_ranks_true(value: bool) -> bool:
    tensor = torch.tensor([int(value)], dtype=torch.int32, device="npu")
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return bool(tensor.cpu().item())


def _max_rank_samples(local: list[float]) -> list[float]:
    gathered: list[list[float] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    return [
        max(float(row[index]) for row in gathered if row is not None)
        for index in range(len(local))
    ]


def _sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _device_mapping(local_rank: int) -> tuple[str, ...]:
    visible = tuple(
        value.strip()
        for value in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    )
    local_device = visible[local_rank] if len(visible) > 1 else (visible[0] if visible else str(local_rank))
    gathered: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_device)
    if any(value is None for value in gathered):
        raise RuntimeError("could not collect the TP rank-to-device mapping")
    mapping = tuple(str(value) for value in gathered)
    if len(set(mapping)) != dist.get_world_size():
        raise ValueError("TP ranks must map to distinct physical NPUs")
    return mapping


def _compare(actual: PairOutput, expected: PairOutput) -> dict[str, Any]:
    names = ("attention_norm", "attention_added", "down_norm", "down_added", "chain_state")
    max_abs = {
        name: _max_rank_value(float((lhs.float() - rhs.float()).abs().max().cpu()))
        for name, lhs, rhs in zip(names, actual, expected)
    }
    exact = _all_ranks_true(all(torch.equal(lhs, rhs) for lhs, rhs in zip(actual, expected)))
    return {"exact_all_ranks": exact, "max_abs": max_abs}


def _measure(
    runners: dict[str, Callable[[], PairOutput]],
    *,
    warmup_replays: int,
    replays_per_sample: int,
    samples: int,
) -> dict[str, list[float]]:
    for _ in range(warmup_replays):
        for runner in runners.values():
            runner()
    torch.npu.synchronize()
    values = {name: [] for name in runners}
    names = tuple(runners)
    for sample in range(samples):
        order = names[sample % len(names) :] + names[: sample % len(names)]
        for name in order:
            dist.barrier()
            torch.npu.synchronize()
            started = time.perf_counter()
            for _ in range(replays_per_sample):
                runners[name]()
            torch.npu.synchronize()
            values[name].append(
                (time.perf_counter() - started) * 1000.0 / replays_per_sample
            )
    return {name: _max_rank_samples(samples_) for name, samples_ in values.items()}


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    if dist.get_world_size() != 3:
        raise ValueError("this diagnostic requires exactly three TP ranks")
    rank = dist.get_rank()
    torch.npu.config.allow_internal_format = True

    from vllm_ascend.utils import enable_custom_op

    if not enable_custom_op():
        raise RuntimeError("vLLM-Ascend custom operators could not be loaded")
    try:
        chained_op = torch.ops._C_ascend.allreduce_add_rmsnorm_chained
    except AttributeError as exc:
        raise RuntimeError("chained allreduce_add_rmsnorm operator is not registered") from exc

    comm_name = resolve_hccl_comm_name(dist.group.WORLD, device="npu", rank=rank)
    if not comm_name:
        raise RuntimeError("could not resolve TP3 HCCL communicator")
    device_mapping = _device_mapping(local_rank)
    attention = _load_payload(args.attention_input_dir, rank)
    down = _load_payload(args.down_input_dir, rank)
    if attention["residual"].shape != down["residual"].shape:
        raise ValueError("attention and down routes must have the same [M, N] output shape")
    chain_zero = torch.zeros((64, 4), dtype=torch.int64, device="npu")

    def baseline() -> PairOutput:
        for _ in range(args.layer_count):
            attention_projection = torch.nn.functional.linear(
                attention["activation"], attention["weight"]
            )
            dist.all_reduce(attention_projection)
            attention_norm, _, attention_added = torch_npu.npu_add_rms_norm(
                attention_projection,
                attention["residual"],
                attention["gamma"],
                1e-6,
            )
            down_projection = torch.nn.functional.linear(
                down["activation"], down["weight"]
            )
            dist.all_reduce(down_projection)
            down_norm, _, down_added = torch_npu.npu_add_rms_norm(
                down_projection,
                down["residual"],
                down["gamma"],
                1e-6,
            )
        return attention_norm, attention_added, down_norm, down_added, chain_zero

    def fully_flushed() -> PairOutput:
        for _ in range(args.layer_count):
            attention_projection = torch.nn.functional.linear(
                attention["activation"], attention["weight"]
            )
            attention_norm, attention_added, _ = chained_op(
                attention_projection,
                attention["residual"],
                attention["gamma"],
                chain_zero,
                comm_name,
                3,
                rank,
                1e-6,
                True,
                True,
            )
            down_projection = torch.nn.functional.linear(
                down["activation"], down["weight"]
            )
            down_norm, down_added, final_state = chained_op(
                down_projection,
                down["residual"],
                down["gamma"],
                chain_zero,
                comm_name,
                3,
                rank,
                1e-6,
                True,
                True,
            )
        return attention_norm, attention_added, down_norm, down_added, final_state

    def deferred_chain() -> PairOutput:
        pending_state = chain_zero
        for layer_index in range(args.layer_count):
            attention_projection = torch.nn.functional.linear(
                attention["activation"], attention["weight"]
            )
            attention_norm, attention_added, pending_state = chained_op(
                attention_projection,
                attention["residual"],
                attention["gamma"],
                pending_state,
                comm_name,
                3,
                rank,
                1e-6,
                True,
                False,
            )
            # Every local projection is submitted before its epilogue consumes
            # the prior state, so it can hide the previous READ_DONE tail.
            down_projection = torch.nn.functional.linear(
                down["activation"], down["weight"]
            )
            down_norm, down_added, pending_state = chained_op(
                down_projection,
                down["residual"],
                down["gamma"],
                pending_state,
                comm_name,
                3,
                rank,
                1e-6,
                True,
                layer_index + 1 == args.layer_count,
            )
        final_state = pending_state
        return attention_norm, attention_added, down_norm, down_added, final_state

    operations = {
        "baseline": baseline,
        "fully_flushed": fully_flushed,
        "deferred_chain": deferred_chain,
    }
    runners: dict[str, Callable[[], PairOutput]] = {}
    outputs: dict[str, PairOutput] = {}
    for name, operation in operations.items():
        dist.barrier()
        if args.execution_mode == "graph":
            graph, output = _capture(operation)

            def replay(
                graph: torch.npu.NPUGraph = graph,
                output: PairOutput = output,
            ) -> PairOutput:
                graph.replay()
                return output

            runners[name] = replay
            outputs[name] = output
        else:
            outputs[name] = operation()
            torch.npu.synchronize()
            runners[name] = operation

    initial_comparisons = {
        name: _compare(outputs[name], outputs["baseline"])
        for name in ("fully_flushed", "deferred_chain")
    }
    original_attention = attention["activation"].clone()
    original_down = down["activation"].clone()
    changed_exact = {"fully_flushed": True, "deferred_chain": True}
    changed_max = {
        "fully_flushed": {name: 0.0 for name in initial_comparisons["fully_flushed"]["max_abs"]},
        "deferred_chain": {name: 0.0 for name in initial_comparisons["deferred_chain"]["max_abs"]},
    }
    for replay_index in range(args.changed_input_replays):
        units = (1, -1, 2, -2, 3)[replay_index % 5]
        attention["activation"].copy_(original_attention)
        attention["activation"].add_(args.changed_input_delta * (rank + 1) * units)
        down["activation"].copy_(original_down)
        down["activation"].sub_(args.changed_input_delta * (rank + 1) * units)
        expected = runners["baseline"]()
        torch.npu.synchronize()
        expected_copy = tuple(value.clone() for value in expected)
        for name in ("fully_flushed", "deferred_chain"):
            actual = runners[name]()
            torch.npu.synchronize()
            local_exact = all(torch.equal(lhs, rhs) for lhs, rhs in zip(actual, expected_copy))
            changed_exact[name] = changed_exact[name] and _all_ranks_true(local_exact)
            for field, lhs, rhs in zip(changed_max[name], actual, expected_copy):
                error = _max_rank_value(float((lhs.float() - rhs.float()).abs().max().cpu()))
                changed_max[name][field] = max(changed_max[name][field], error)

    attention["activation"].copy_(original_attention)
    down["activation"].copy_(original_down)
    outputs = {name: runner() for name, runner in runners.items()}
    torch.npu.synchronize()
    measured = _measure(
        runners,
        warmup_replays=args.warmup_replays,
        replays_per_sample=args.replays_per_sample,
        samples=args.samples,
    )
    final_chain_state_zero_all_ranks = _all_ranks_true(
        bool(torch.count_nonzero(outputs["deferred_chain"][4]).cpu().item() == 0)
    )

    if rank == 0:
        summaries = {
            name: {
                "p50_ms": _percentile(values, 50.0),
                "p95_ms": _percentile(values, 95.0),
            }
            for name, values in measured.items()
        }
        document: dict[str, Any] = {
            "schema_version": 1,
            "status": "measured",
            "execution_mode": args.execution_mode,
            "tensor_parallel_size": 3,
            "chain_scope": "whole_model",
            "layer_count": args.layer_count,
            "operations_per_chain": 2 * args.layer_count,
            "tp_rank_device_mapping": list(device_mapping),
            "attention_input_source": str(args.attention_input_dir.resolve()),
            "down_input_source": str(args.down_input_dir.resolve()),
            "custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH", ""),
            "correctness": {
                "initial": initial_comparisons,
                "changed_input_exact_all_ranks": changed_exact,
                "changed_input_max_abs": changed_max,
                "changed_input_replays": args.changed_input_replays,
                "final_chain_state_zero_all_ranks": final_chain_state_zero_all_ranks,
                "deferred_down_norm_sha256": _sha256(outputs["deferred_chain"][2]),
            },
            "samples": {f"{name}_latency_ms": values for name, values in measured.items()},
            "summary": summaries,
        }
        document["summary"]["deferred_vs_fully_flushed_p50"] = (
            summaries["fully_flushed"]["p50_ms"]
            / summaries["deferred_chain"]["p50_ms"]
        )
        document["summary"]["deferred_vs_fully_flushed_p95"] = (
            summaries["fully_flushed"]["p95_ms"]
            / summaries["deferred_chain"]["p95_ms"]
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(document["correctness"] | document["summary"], indent=2))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
