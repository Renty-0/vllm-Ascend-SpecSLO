# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate live SpecSLO admission on real persistent draft/target workers.

This is a functional smoke, not a roofline or throughput benchmark.  Its
explicit verification budget exists only to exercise the scheduler before the
paper Section 5.3 offline B experiment has been run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-model", default="/data/shared-models/Qwen3-0.6B")
    parser.add_argument("--target-model", default="/data/shared-models/Qwen3-32B")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--draft-temperature", type=float, default=0.0)
    parser.add_argument("--admission-delay-seconds", type=float, default=0.05)
    parser.add_argument("--verification-budget", type=int, default=4)
    parser.add_argument(
        "--worker-startup-timeout-seconds",
        type=float,
        default=900.0,
        help="Cold dual-model initialization deadline; excluded from the functional request interval.",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


async def _run(args) -> dict:
    from vllm_ascend.spec_decode.pearl.api import PEARLConfig
    from vllm_ascend.spec_decode.pearl.http_server import SpecSLOHTTPService
    from vllm_ascend.spec_decode.pearl.native_engine import SamplingParams

    config = PEARLConfig(
        draft_model_path=args.draft_model,
        target_model_path=args.target_model,
        draft_tensor_parallel_size=1,
        target_tensor_parallel_size=3,
        gamma=4,
        max_model_len=512,
        max_num_seqs=2,
        max_num_queued_seqs=4,
        prefill_chunk_size=2,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.85,
        max_aclgraph_entries=32,
        enable_prefix_caching=False,
        enable_continuous_batching=True,
        enable_preemptive_scheduling=True,
        enable_spec_rhythm=True,
        spec_rhythm_online_prefill=True,
        spec_rhythm_priority_mode=True,
        spec_rhythm_max_eager_tokens=4,
        spec_rhythm_urgency_threshold=0.0,
        spec_rhythm_acceptance_floor=0.0,
        spec_rhythm_verification_budget=args.verification_budget,
        spec_rhythm_tree_width=2,
        spec_rhythm_tree_depth=2,
        enforce_eager=args.enforce_eager,
        seed=0,
        worker_timeout_seconds=args.worker_startup_timeout_seconds,
    )
    service = SpecSLOHTTPService(config, batch_wait_ms=0, max_batch_size=2)
    startup_started = time.perf_counter()
    await service.start()
    startup_seconds = time.perf_counter() - startup_started
    args.output.write_text(
        json.dumps(
            {
                "status": "running",
                "phase": "online_requests",
                "startup_seconds": startup_seconds,
                "scope": "functional_smoke_not_a_B_or_throughput_measurement",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    started = time.perf_counter()
    try:
        params = SamplingParams(
            temperature=args.temperature,
            draft_temperature=args.draft_temperature,
            max_tokens=args.max_tokens,
            ignore_eos=True,
            slo_tpot_ms=150.0,
        )
        first_tokens = service.encode_completion("Solve 18 + 24. Give only the number.")
        second_tokens = service.encode_completion("Solve 35 - 17. Give only the number.")
        first = await service.submit(first_tokens, params, request_id="live-first", stream=True)
        deadline = time.monotonic() + 30.0
        while service.health()["inflight_requests"] == 0:
            if time.monotonic() >= deadline:
                raise TimeoutError("SpecSLO service did not start the initial live request")
            await asyncio.sleep(0.01)
        await asyncio.sleep(args.admission_delay_seconds)
        second = await service.submit(second_tokens, params, request_id="live-second", stream=True)
        first_result, second_result = await asyncio.gather(first.future, second.future)
        first_events, second_events = await asyncio.gather(
            _collect_events(first),
            _collect_events(second),
        )
        rows = [first_result, second_result]
        metrics = [dict(row.metrics) for row in rows]
        live_counts = [int(row.get("spec_rhythm", {}).get("spec_rhythm_live_admitted_requests", 0)) for row in metrics]
        if max(live_counts, default=0) < 1:
            raise RuntimeError("The second request was not admitted into an active worker epoch")
        return {
            "status": "passed",
            "scope": "functional_smoke_not_a_B_or_throughput_measurement",
            "mode": "eager" if args.enforce_eager else "graph",
            "temperature": args.temperature,
            "draft_temperature": args.draft_temperature,
            "startup_seconds": startup_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "request_ids": [row.request_id for row in rows],
            "completion_token_ids": [list(row.token_ids) for row in rows],
            "stream_event_counts": [len(first_events), len(second_events)],
            "live_admitted_requests": live_counts,
            "metrics": metrics,
            "health": service.health(),
        }
    finally:
        await service.stop()


async def _collect_events(handle) -> list[dict]:
    return [event async for event in handle.events()]


def main() -> None:
    args = _parser().parse_args()
    if args.max_tokens <= 0 or args.verification_budget <= 0 or args.worker_startup_timeout_seconds <= 0:
        raise ValueError("Token and functional-smoke budget values must be positive")
    if args.temperature < 0 or args.draft_temperature < 0 or args.admission_delay_seconds < 0:
        raise ValueError("Temperatures and admission delay must be non-negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "running",
                "phase": "worker_startup",
                "scope": "functional_smoke_not_a_B_or_throughput_measurement",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    try:
        result = asyncio.run(_run(args))
    except BaseException as error:
        args.output.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "phase": "worker_startup_or_online_requests",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
