# SPDX-License-Identifier: Apache-2.0
"""Benchmark parallel ACLGraph FIA task refresh without loading a model.

SpecSLO target verification captures one causal FIA operator per decoder
layer.  KV lengths change every decode cycle, so every captured operator must
be refreshed before graph replay.  This probe compares the production serial
refresh with disjoint update streams driven by a persistent thread pool.  It
is deliberately an opt-in diagnostic: production code must not use parallel
refresh until this probe establishes both exact output equality and a wall
time benefit on the active CANN/torch-npu build.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--tasks", type=int, default=64)
    parser.add_argument("--workers", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument(
        "--event-group-size",
        type=int,
        choices=[1, 2, 4, 8, 16, 32, 64],
        default=1,
    )
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--queries-per-request", type=int, default=5)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--kv-heads", type=int, default=3)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", required=True)
    return parser


def _validate(args: argparse.Namespace) -> None:
    positive = (
        args.tasks,
        args.batch,
        args.queries_per_request,
        args.context,
        args.heads,
        args.kv_heads,
        args.head_dim,
        args.block_size,
        args.repeats,
    )
    if min(positive) <= 0 or args.warmup < 0:
        raise ValueError("FIA dimensions, tasks and repeats must be positive")
    if args.heads % args.kv_heads:
        raise ValueError("Query heads must be divisible by KV heads")
    if not args.workers or min(args.workers) <= 1:
        raise ValueError("Parallel worker counts must all exceed one")
    if args.tasks % args.event_group_size:
        raise ValueError("Task count must be divisible by the event group size")
    if args.device == "cpu" or not args.device.startswith("npu:"):
        raise ValueError("This diagnostic requires one explicit npu:<index>")


def _make_inputs(args: argparse.Namespace):
    import torch

    query_tokens = args.batch * args.queries_per_request
    sequence_length = args.context + args.queries_per_request
    blocks_per_request = math.ceil(sequence_length / args.block_size)
    block_count = args.batch * blocks_per_request
    query = torch.randn(
        query_tokens,
        args.heads,
        args.head_dim,
        device=args.device,
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        block_count,
        args.block_size,
        args.kv_heads * args.head_dim,
        device=args.device,
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)
    block_table = torch.arange(
        block_count,
        device=args.device,
        dtype=torch.int32,
    ).reshape(args.batch, blocks_per_request)
    attention_mask = torch.triu(
        torch.ones(2048, 2048, dtype=torch.int8, device=args.device),
        diagonal=1,
    )
    query_lengths = [
        (index + 1) * args.queries_per_request for index in range(args.batch)
    ]
    kv_lengths = [sequence_length] * args.batch
    kwargs = {
        "query": query,
        "key": key,
        "value": value,
        "atten_mask": attention_mask,
        "block_table": block_table,
        "input_layout": "TND",
        "block_size": args.block_size,
        "actual_seq_lengths": query_lengths,
        "actual_seq_lengths_kv": kv_lengths,
        "num_key_value_heads": args.kv_heads,
        "num_heads": args.heads,
        "scale": 1.0 / math.sqrt(args.head_dim),
        "sparse_mode": 3,
        "next_tokens": 0,
    }
    return kwargs


def _capture(args: argparse.Namespace, kwargs):
    import torch
    import torch_npu

    outputs = [torch.empty_like(kwargs["query"]) for _ in range(args.tasks)]
    lse_outputs = [
        torch.empty(1, dtype=kwargs["query"].dtype, device=args.device)
        for _ in range(args.tasks)
    ]
    workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
        **kwargs
    )
    for _ in range(3):
        torch_npu.npu_fused_infer_attention_score.out(
            **kwargs,
            workspace=workspace,
            out=[outputs[0], lse_outputs[0]],
        )
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    events = []
    handles = []
    with torch.npu.graph(graph):
        stream = torch.npu.current_stream()
        event = None
        for index in range(args.tasks):
            if index % args.event_group_size == 0:
                event = torch.npu.ExternalEvent()
                event.wait(stream)
                event.reset(stream)
            assert event is not None
            torch.npu.graph_task_group_begin(stream)
            torch_npu.npu_fused_infer_attention_score.out(
                **kwargs,
                workspace=workspace,
                out=[outputs[index], lse_outputs[index]],
            )
            handles.append(torch.npu.graph_task_group_end(stream))
            events.append(event)
    return graph, events, handles, outputs, lse_outputs, workspace


def _update_chunk(
    device: str,
    stream,
    indices,
    handles,
    events,
    outputs,
    lse_outputs,
    workspace,
    kwargs,
    event_group_size,
    record_events=True,
) -> None:
    import torch
    import torch_npu

    torch.npu.set_device(device)
    with torch.npu.stream(stream):
        for index in indices:
            torch.npu.graph_task_update_begin(stream, handles[index])
            torch_npu.npu_fused_infer_attention_score.out(
                **kwargs,
                workspace=workspace,
                out=[outputs[index], lse_outputs[index]],
            )
            torch.npu.graph_task_update_end(stream)
            if record_events and (
                (index + 1) % event_group_size == 0
                or index + 1 == len(handles)
            ):
                events[index].record(stream)


def _chunks(total: int, workers: int, event_group_size: int):
    """Partition complete consecutive event groups across workers."""

    groups = [
        range(start, min(start + event_group_size, total))
        for start in range(0, total, event_group_size)
    ]
    return [
        tuple(index for group in groups[worker::workers] for index in group)
        for worker in range(workers)
    ]


def _measure(args, graph, events, handles, outputs, lse_outputs, workspace, kwargs):
    import torch

    results = []
    main_stream = torch.npu.current_stream()

    def run_serial(stream):
        stream.wait_stream(main_stream)
        _update_chunk(
            args.device,
            stream,
            range(args.tasks),
            handles,
            events,
            outputs,
            lse_outputs,
            workspace,
            kwargs,
            args.event_group_size,
            True,
        )
        main_stream.wait_stream(stream)
        graph.replay()

    serial_stream = torch.npu.Stream()
    for _ in range(args.warmup):
        run_serial(serial_stream)
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(args.repeats):
        run_serial(serial_stream)
    torch.npu.synchronize()
    serial_ms = (time.perf_counter() - started) * 1000.0 / args.repeats
    reference = outputs[-1].clone()
    results.append(
        {
            "workers": 1,
            "milliseconds_per_update_and_replay": serial_ms,
            "exact_output": True,
        }
    )

    for worker_count in args.workers:
        streams = [torch.npu.Stream() for _ in range(worker_count)]
        partitions = _chunks(
            args.tasks,
            worker_count,
            args.event_group_size,
        )

        def run_parallel(executor):
            for stream in streams:
                stream.wait_stream(main_stream)
            futures = [
                executor.submit(
                    _update_chunk,
                    args.device,
                    stream,
                    indices,
                    handles,
                    events,
                    outputs,
                    lse_outputs,
                    workspace,
                    kwargs,
                    args.event_group_size,
                    True,
                )
                for stream, indices in zip(streams, partitions)
            ]
            for future in futures:
                future.result()
            for stream in streams:
                main_stream.wait_stream(stream)
            graph.replay()

        def run_caller_primary(executor):
            """Let the caller own stream zero and delegate the remaining streams.

            The production target worker already spends the replay boundary in
            this Python thread.  Submitting stream zero back to the executor
            adds a second future dispatch without creating any additional
            concurrency, so measure that overhead independently before changing
            the hot path.
            """

            for stream in streams:
                stream.wait_stream(main_stream)
            futures = [
                executor.submit(
                    _update_chunk,
                    args.device,
                    stream,
                    indices,
                    handles,
                    events,
                    outputs,
                    lse_outputs,
                    workspace,
                    kwargs,
                    args.event_group_size,
                    True,
                )
                for stream, indices in zip(streams[1:], partitions[1:])
            ]
            _update_chunk(
                args.device,
                streams[0],
                partitions[0],
                handles,
                events,
                outputs,
                lse_outputs,
                workspace,
                kwargs,
                args.event_group_size,
                True,
            )
            for future in futures:
                future.result()
            for stream in streams[1:]:
                main_stream.wait_stream(stream)
            graph.replay()

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for _ in range(args.warmup):
                run_parallel(executor)
            torch.npu.synchronize()
            started = time.perf_counter()
            for _ in range(args.repeats):
                run_parallel(executor)
            torch.npu.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0 / args.repeats
        results.append(
            {
                "workers": worker_count,
                "strategy": "threadpool_all",
                "milliseconds_per_update_and_replay": elapsed_ms,
                "speedup_vs_serial": serial_ms / elapsed_ms,
                "exact_output": bool(torch.equal(outputs[-1], reference)),
                "max_abs_error": float(
                    (outputs[-1].float() - reference.float()).abs().max().cpu()
                ),
            }
        )
        if args.event_group_size == args.tasks:
            global_partitions = _chunks(args.tasks, worker_count, 1)

            def run_global_barrier(executor):
                for stream in streams:
                    stream.wait_stream(main_stream)
                futures = [
                    executor.submit(
                        _update_chunk,
                        args.device,
                        stream,
                        indices,
                        handles,
                        events,
                        outputs,
                        lse_outputs,
                        workspace,
                        kwargs,
                        1,
                        False,
                    )
                    for stream, indices in zip(
                        streams[1:],
                        global_partitions[1:],
                    )
                ]
                _update_chunk(
                    args.device,
                    streams[0],
                    global_partitions[0],
                    handles,
                    events,
                    outputs,
                    lse_outputs,
                    workspace,
                    kwargs,
                    1,
                    False,
                )
                for future in futures:
                    future.result()
                for stream in streams[1:]:
                    streams[0].wait_stream(stream)
                events[0].record(streams[0])
                graph.replay()

            with ThreadPoolExecutor(max_workers=worker_count - 1) as executor:
                for _ in range(args.warmup):
                    run_global_barrier(executor)
                torch.npu.synchronize()
                started = time.perf_counter()
                for _ in range(args.repeats):
                    run_global_barrier(executor)
                torch.npu.synchronize()
                global_elapsed_ms = (
                    time.perf_counter() - started
                ) * 1000.0 / args.repeats
            results.append(
                {
                    "workers": worker_count,
                    "strategy": "parallel_global_barrier",
                    "milliseconds_per_update_and_replay": global_elapsed_ms,
                    "speedup_vs_serial": serial_ms / global_elapsed_ms,
                    "exact_output": bool(torch.equal(outputs[-1], reference)),
                    "max_abs_error": float(
                        (outputs[-1].float() - reference.float()).abs().max().cpu()
                    ),
                }
            )
        with ThreadPoolExecutor(max_workers=worker_count - 1) as executor:
            for _ in range(args.warmup):
                run_caller_primary(executor)
            torch.npu.synchronize()
            started = time.perf_counter()
            for _ in range(args.repeats):
                run_caller_primary(executor)
            torch.npu.synchronize()
            caller_elapsed_ms = (
                time.perf_counter() - started
            ) * 1000.0 / args.repeats
        results.append(
            {
                "workers": worker_count,
                "strategy": "caller_primary",
                "milliseconds_per_update_and_replay": caller_elapsed_ms,
                "speedup_vs_serial": serial_ms / caller_elapsed_ms,
                "speedup_vs_threadpool_all": elapsed_ms / caller_elapsed_ms,
                "exact_output": bool(torch.equal(outputs[-1], reference)),
                "max_abs_error": float(
                    (outputs[-1].float() - reference.float()).abs().max().cpu()
                ),
            }
        )
    return results


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    _validate(args)
    import torch

    torch.npu.set_device(args.device)
    report = {"status": "running", "config": vars(args), "results": []}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        kwargs = _make_inputs(args)
        captured = _capture(args, kwargs)
        report["results"] = _measure(args, *captured, kwargs)
        report["passed"] = all(row["exact_output"] for row in report["results"])
        report["best_speedup_vs_serial"] = max(
            [1.0]
            + [
                row["speedup_vs_serial"]
                for row in report["results"]
                if "speedup_vs_serial" in row
            ]
        )
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        save()
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
