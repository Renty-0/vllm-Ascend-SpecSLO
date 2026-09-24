# SPDX-License-Identifier: Apache-2.0
"""Benchmark complete Qwen3-32B TP3 FFN graph blocks in ND and NZ.

The optional prefetch strategy mirrors vLLM-Ascend's dense MLP path: while
SwiGLU consumes the gate/up result, a side stream prefetches the first slice
of the down-projection weight and the compute stream waits before the down
matmul.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-counts", type=int, nargs="+", default=[32, 64, 96, 128])
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--profile-steps", type=int, default=100)
    parser.add_argument("--include-prefetch", action="store_true")
    parser.add_argument(
        "--pad-row-reduce-to",
        type=int,
        default=0,
        help=(
            "Also measure an ND block whose down-projection/all-reduce is "
            "zero-padded to this row count before slicing back to the real rows."
        ),
    )
    parser.add_argument("--prefetch-bytes", type=int, default=18 * 1024 * 1024)
    parser.add_argument("--output-json")
    return parser.parse_args()


def _capture(operation):
    operation()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = operation()
    torch.npu.synchronize()
    return graph, output


def _measure(graph, warmup_steps: int, profile_steps: int) -> float:
    for _ in range(warmup_steps):
        graph.replay()
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(profile_steps):
        graph.replay()
    torch.npu.synchronize()
    return (time.perf_counter() - started) * 1_000_000 / profile_steps


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "AIV")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    if dist.get_world_size() != 3:
        raise ValueError("The Qwen3 FFN benchmark requires exactly three ranks.")
    torch.npu.config.allow_internal_format = True
    if args.include_prefetch:
        # Importing the package registers the graph-visible prefetch custom
        # ops; loading the compiled extension alone is not sufficient.
        import vllm_ascend.ops  # noqa: F401
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            raise RuntimeError("vLLM-Ascend custom operators could not be loaded")

    hidden_size = 5120
    intermediate_per_rank = 8544
    results: list[dict[str, object]] = []
    for token_count in args.token_counts:
        torch.manual_seed(20260920 + rank)
        source = (
            torch.randn(token_count, hidden_size, dtype=torch.float32, device="npu")
            * 0.02
        ).to(torch.bfloat16)
        changed_source = (
            torch.randn_like(source, dtype=torch.float32) * 0.02
        ).to(torch.bfloat16)
        gate_up_nd = (
            torch.randn(
                2 * intermediate_per_rank,
                hidden_size,
                dtype=torch.float32,
                device="npu",
            )
            * 0.02
        ).to(torch.bfloat16)
        down_nd = (
            torch.randn(
                hidden_size,
                intermediate_per_rank,
                dtype=torch.float32,
                device="npu",
            )
            * 0.02
        ).to(torch.bfloat16)
        gate_up_nz = torch_npu.npu_format_cast(gate_up_nd, 29)
        down_nz = torch_npu.npu_format_cast(down_nd, 29)

        outputs: dict[str, torch.Tensor] = {}
        graphs: dict[str, torch.npu.NPUGraph] = {}
        strategies = [
            ("nd", gate_up_nd, down_nd, False, 0),
            ("gate_nz", gate_up_nz, down_nd, False, 0),
            ("down_nz", gate_up_nd, down_nz, False, 0),
            ("nz", gate_up_nz, down_nz, False, 0),
        ]
        if args.include_prefetch:
            strategies.insert(1, ("nd_prefetch", gate_up_nd, down_nd, True, 0))
        if args.pad_row_reduce_to > token_count:
            strategies.insert(
                1,
                (
                    f"nd_pad_reduce_{args.pad_row_reduce_to}",
                    gate_up_nd,
                    down_nd,
                    False,
                    args.pad_row_reduce_to,
                ),
            )
        for layout, gate_up_weight, down_weight, prefetch, padded_rows in strategies:

            def ffn_block() -> torch.Tensor:
                gate_up = F.linear(source, gate_up_weight)
                if prefetch:
                    torch.ops.vllm.prefetch_preprocess(
                        weight=down_weight,
                        start_flag=gate_up,
                        max_weight_size=min(
                            args.prefetch_bytes,
                            down_weight.numel() * down_weight.element_size(),
                        ),
                    )
                activated = torch_npu.npu_swiglu(gate_up)
                if prefetch:
                    torch.ops.vllm.prefetch_postprocess(activated)
                if padded_rows:
                    activated = F.pad(
                        activated,
                        (0, 0, 0, padded_rows - token_count),
                    )
                output = F.linear(activated, down_weight)
                dist.all_reduce(output)
                return output[:token_count]

            dist.barrier()
            try:
                graph, output = _capture(ffn_block)
                latency = _measure(graph, args.warmup_steps, args.profile_steps)
                source.copy_(changed_source)
                graph.replay()
                torch.npu.synchronize()
                changed_output = output.detach().clone()
                status = "ok"
            except Exception as error:
                latency = float("nan")
                changed_output = torch.empty(0, device="npu")
                status = f"{type(error).__name__}: {error}"
                graph = None
            local_result: dict[str, object] = {
                "rank": rank,
                "token_count": token_count,
                "layout": layout,
                "latency_microseconds": latency,
                "status": status,
            }
            gathered: list[dict[str, object] | None] = [None, None, None]
            dist.all_gather_object(gathered, local_result)
            if rank == 0:
                results.extend(value for value in gathered if value is not None)
            if graph is not None:
                graphs[layout] = graph
                outputs[layout] = changed_output

        if rank == 0 and "nd" in outputs:
            for comparison in (
                "nd_prefetch",
                f"nd_pad_reduce_{args.pad_row_reduce_to}",
                "gate_nz",
                "down_nz",
                "nz",
            ):
                if comparison not in outputs:
                    continue
                max_error = float(
                    (outputs["nd"].float() - outputs[comparison].float())
                    .abs()
                    .max()
                    .cpu()
                )
                field = f"nd_{comparison}_changed_input_max_abs_error"
                for result in results:
                    if result["token_count"] == token_count:
                        result[field] = max_error

        del source, changed_source, gate_up_nd, down_nd, gate_up_nz, down_nz

    if rank == 0:
        payload = {"world_size": 3, "results": results}
        rendered = json.dumps(payload, indent=2)
        print(rendered)
        if args.output_json:
            path = Path(args.output_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered + "\n", encoding="utf-8")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
