# SPDX-License-Identifier: Apache-2.0
"""Actual four-NPU SpecSLO fail-stop acceptance, not a throughput benchmark.

Run with torchrun --standalone --nproc_per_node=4. One TP1+TP3 engine loads
Qwen3-0.6B and Qwen3-32B once. Two synthetic prompts generate one token in
each stage: healthy, a real weight NaN on one rank, restored weight with the
sticky fault retained across ordinary cache release. No health flag is
manually set or cleared. Every rank must report the expected error before
any callback in the two negative stages.

Default injection is a valid target-rank-2 LM-head weight. --fault-rank 0
uses the draft's final MLP weight: draft prefill does execute for max_tokens=1,
but its LM head does not, so a draft-head-only injection would be vacuous.
The script is eager-only: this verifies the prefill/first-token boundary,
not ACLGraph execution. Operator/model graph tests are separate evidence.
Use an external job timeout as well as --collective-timeout-seconds when
testing hardware failures; a process cannot roll back a failed HCCL kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-model", default="/data/shared-models/Qwen3-0.6B")
    parser.add_argument("--target-model", default="/data/shared-models/Qwen3-32B")
    parser.add_argument("--fault-rank", type=int, choices=(0, 1, 2, 3), default=2)
    parser.add_argument("--fault-site", choices=("auto", "head", "final-mlp"), default="auto")
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--collective-timeout-seconds", type=float, default=180)
    parser.add_argument("--seed", type=int, default=413)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="Write config only, without importing/loading NPU models"
    )
    return parser


def _validate_args(args):
    if args.max_model_len < 64:
        raise ValueError("max-model-len must leave room for prompt, one output and the tree suffix")
    if not math.isfinite(args.gpu_memory_utilization) or not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("gpu-memory-utilization must be finite and strictly between zero and one")
    if not math.isfinite(args.collective_timeout_seconds) or args.collective_timeout_seconds <= 0:
        raise ValueError("collective timeout must be finite and positive")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    site = ("final-mlp" if args.fault_rank == 0 else "head") if args.fault_site == "auto" else args.fault_site
    if args.fault_rank == 0 and site == "head":
        raise ValueError("Draft max_tokens=1 prefill does not execute its head; choose final-mlp for a real fault")
    return site


def _engine_config_values(args):
    return {
        "draft_model": args.draft_model,
        "target_model": args.target_model,
        "draft_tp_size": 1,
        "target_tp_size": 3,
        "gamma": 4,
        "max_model_len": args.max_model_len,
        "max_tokens": 1,
        "max_num_seqs": 2,
        "max_num_queued_seqs": 2,
        "max_num_batched_tokens": 512,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_prefix_caching": False,
        "enable_continuous_batching": True,
        "enable_preemptive_scheduling": True,
        "enable_spec_rhythm": True,
        "spec_rhythm_online_prefill": True,
        "spec_rhythm_tree_width": 2,
        "spec_rhythm_tree_depth": 2,
        "spec_rhythm_verification_budget": 4,
        "enforce_eager": True,
        "enable_cpu_binding": False,
        "seed": args.seed,
    }


def _weight_target(engine, site):
    if site == "head":
        head = engine.model.lm_head
        if not 0 <= int(head.vocab_start) < engine.draft_vocab_size:
            raise ValueError("Injection must belong to an active, non-truncated local vocabulary shard")
        return head.weight, "lm_head.weight", int(head.vocab_start)
    index = len(engine.model.layers) - 1
    return engine.model.layers[index].mlp.down_proj.weight, f"layers.{index}.mlp.down_proj.weight", None


def _dense_weight(parameter):
    if parameter.device.type == "npu":
        import torch_npu

        return torch_npu.npu_format_cast(parameter.detach(), 2)
    return parameter.detach()


def _weight_scalar(parameter):
    return _dense_weight(parameter)[0, 0].detach().cpu().clone()


def _replace_weight_scalar(parameter, value):
    """Copy through ND staging, retaining the Parameter and original NZ storage.

    This diagnostic can temporarily allocate one full parameter. It avoids
    assuming an indexed NZ tensor view is writable. The saved/restored scalar
    retains its original dtype; every other weight value is copied unchanged.
    """
    import torch

    with torch.no_grad():
        staging = _dense_weight(parameter).clone()
        staging[0, 0] = torch.as_tensor(value, dtype=parameter.dtype, device=parameter.device)
        parameter.copy_(staging)


def _summarize_stage(name, records, *, fault_rank, leader_rank=1):
    errors = []
    if sorted(record["rank"] for record in records) != [0, 1, 2, 3]:
        errors.append("missing or duplicate rank evidence")
    healthy = name == "healthy"
    for record in records:
        rank = record["rank"]
        if record.get("model_forward_calls", 0) <= 0:
            errors.append(f"rank {rank} did not execute a real model prefill")
        if not record.get("cache_allocation_released", False):
            errors.append(f"rank {rank} retained a live cache allocation")
        if healthy:
            if record.get("error") is not None or record.get("nonfinite_flag"):
                errors.append(f"rank {rank} failed the healthy control")
            if rank == leader_rank:
                outputs = record.get("outputs")
                chunks = record.get("chunks", [])
                if outputs is None or len(outputs) != 2 or any(len(row) != 1 for row in outputs):
                    errors.append("healthy target leader did not produce exactly two one-token outputs")
                streamed = {index: [] for index in range(2)}
                for chunk in chunks:
                    if chunk.get("request_index") not in streamed:
                        errors.append("healthy stream has an unknown request")
                        continue
                    streamed[chunk["request_index"]].extend(chunk["token_ids"])
                if outputs != list(streamed.values()) or len(chunks) != 2:
                    errors.append("healthy first-token callbacks differ from committed outputs")
                if outputs is not None and any(
                    not 0 <= token < record["vocabulary_size"] for row in outputs for token in row
                ):
                    errors.append("healthy output contains an invalid vocabulary ID")
            elif record.get("outputs") is not None or record.get("chunks"):
                errors.append(f"nonleader rank {rank} emitted user-visible output")
        else:
            message = record.get("error") or ""
            if "SpecRhythm prefill produced nonfinite" not in message or "no first token was committed" not in message:
                errors.append(f"rank {rank} did not report the expected collective prefill fail-stop")
            if record.get("outputs") is not None or record.get("chunks"):
                errors.append(f"rank {rank} emitted output after the fault")
            if rank == fault_rank and not record.get("nonfinite_flag"):
                errors.append("injected rank did not detect its real weight fault")
            if name == "restored_sticky" and rank == fault_rank and not record.get("nonfinite_flag_before_generate"):
                errors.append("restored stage lost the sticky fault before its model forward")
    return {"stage": name, "passed": not errors, "errors": errors, "ranks": records}


def _write_report(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _run_stage(engine, prompts, name):
    import torch

    from vllm_ascend.spec_decode.pearl.native_engine import NativeSamplingParams

    calls = []
    chunks = []
    initial_nonfinite = bool(engine._spec_rhythm_nonfinite_flag().cpu().item())
    handle = engine.model.register_forward_pre_hook(lambda module, args: calls.append(int(args[0].numel())))
    error = None
    error_type = None
    result = None
    started = time.perf_counter()
    try:
        result = engine.generate_batch(
            prompts,
            NativeSamplingParams(temperature=0, draft_temperature=0, max_tokens=1, ignore_eos=True),
            on_token_commit=lambda event: chunks.append(dict(event)),
        )
    except Exception as caught:
        error, error_type = str(caught), type(caught).__name__
    finally:
        handle.remove()
        # Failed generate_batch does not pretend to transactionally roll back
        # physical KV; ordinary release only frees its allocation for the next
        # negative case. It MUST NOT clear the device-resident sticky flags.
        if engine.cache_allocation is not None:
            engine._release_cache()
    torch.npu.synchronize()
    return {
        "rank": engine.rank,
        "role": "draft" if engine.is_draft else "target",
        "stage": name,
        "error": error,
        "error_type": error_type,
        "outputs": None if result is None else [row["completion_token_ids"] for row in result],
        "chunks": chunks,
        "model_forward_calls": len(calls),
        "model_query_tokens": calls,
        "nonfinite_flag": bool(engine._spec_rhythm_nonfinite_flag().cpu().item()),
        "nonfinite_flag_before_generate": initial_nonfinite,
        "cache_allocation_released": engine.cache_allocation is None,
        "vocabulary_size": engine.draft_vocab_size,
        "elapsed_seconds_diagnostic_only": time.perf_counter() - started,
    }


def main(argv=None):
    args = _build_parser().parse_args(argv)
    site = _validate_args(args)
    config_values = _engine_config_values(args)
    document = {
        "schema_version": 1,
        "status": "dry_run" if args.dry_run else "running",
        "passed": None,
        "npu_executed": False,
        "engine_loads_per_rank": 0,
        "tp": {"draft": 1, "target": 3, "world": 4},
        "prompt_count": 2,
        "max_tokens": 1,
        "fault_rank": args.fault_rank,
        "fault_site": site,
        "engine_config": config_values,
        "scope": "Real weight fault / collective first-token fail-stop; no throughput or ACLGraph claim.",
        "stages": [],
        "collective_close_barrier_completed": False,
        "destroy_process_group_returned": False,
    }
    if args.dry_run:
        _write_report(args.output, document)
        return 0
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    from vllm_ascend.spec_decode.pearl import native_model
    from vllm_ascend.spec_decode.pearl.native_engine import (
        NativePearlConfig,
        NativePearlEngine,
        _set_default_npu_environment,
    )

    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if world_size != 4:
        raise ValueError("Use torchrun --nproc_per_node=4 for exactly TP1 draft + TP3 target")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    leader_rank = 1
    torch.npu.set_device(local_rank)
    _set_default_npu_environment(target_tp_size=3)
    dist.init_process_group("hccl", timeout=timedelta(seconds=args.collective_timeout_seconds))
    engine = None
    parameter = saved = None
    restore_needed = False
    final_error = None
    try:
        if rank == leader_rank:
            document["native_model_source_sha256"] = hashlib.sha256(
                Path(native_model.__file__).read_bytes()
            ).hexdigest()
            _write_report(args.output, document)
        engine = NativePearlEngine(NativePearlConfig(**config_values))
        document["engine_loads_per_rank"] = 1
        document["npu_executed"] = True
        # Two identical-sized synthetic sanity questions avoid dataset/model
        # downloads and make no GSM8K or ShareGPT benchmark claim.
        prompts = [
            engine.tokenizer.encode(
                engine.tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
                )
            )
            for text in ("What is 2 + 3?", "What is 4 + 1?")
        ]
        document["prompt_token_ids"] = prompts
        if any(len(prompt) + 6 > args.max_model_len for prompt in prompts):
            raise ValueError("Encoded synthetic prompts do not fit max-model-len")
        for name in ("healthy", "poisoned", "restored_sticky"):
            local_injection = {"rank": rank, "changed": False, "error": None}
            if name != "healthy" and rank == args.fault_rank:
                try:
                    if name == "poisoned":
                        parameter, parameter_name, global_vocab_id = _weight_target(engine, site)
                        saved = _weight_scalar(parameter)
                        if not bool(torch.isfinite(saved)):
                            raise RuntimeError("Selected original weight was not finite")
                        restore_needed = True
                        _replace_weight_scalar(parameter, float("nan"))
                        observed = _weight_scalar(parameter)
                        if not bool(torch.isnan(observed)):
                            raise RuntimeError("Injected weight did not actually become NaN")
                        local_injection.update(
                            parameter=parameter_name,
                            index=[0, 0],
                            global_vocab_id=global_vocab_id,
                            original_value=float(saved),
                            injected_is_nan=True,
                            changed=True,
                        )
                    else:
                        _replace_weight_scalar(parameter, saved)
                        observed = _weight_scalar(parameter)
                        if not torch.equal(observed, saved):
                            raise RuntimeError("Original weight was not restored exactly")
                        restore_needed = False
                        local_injection.update(restored_exactly=True, changed=True)
                except Exception as caught:
                    local_injection["error"] = repr(caught)
            injections = [None] * 4
            dist.all_gather_object(injections, local_injection)
            if any(item["error"] is not None for item in injections):
                raise RuntimeError(f"Real weight injection/restoration failed: {injections}")
            record = _run_stage(engine, prompts, name)
            records = [None] * 4
            dist.all_gather_object(records, record)
            summary = _summarize_stage(name, records, fault_rank=args.fault_rank)
            summary["weight_changes"] = injections
            document["stages"].append(summary)
            if rank == leader_rank:
                _write_report(args.output, document)
                print(json.dumps({"stage": name, "passed": summary["passed"], "errors": summary["errors"]}), flush=True)
            if not summary["passed"]:
                raise RuntimeError(f"{name} acceptance failed: {summary['errors']}")
        document["passed"] = True
    except Exception as caught:
        document["passed"] = False
        final_error = repr(caught)
        document["error"] = final_error
    finally:
        if restore_needed and parameter is not None and saved is not None:
            _replace_weight_scalar(parameter, saved)
        if engine is not None and engine.cache_allocation is not None:
            engine._release_cache()
        # The expected negative cases reached common votes/gathers above.
        # A process that cannot reach this barrier leaves a non-complete JSON
        # and nonzero/timeout job outcome, never a fabricated close success.
        dist.barrier()
        document["collective_close_barrier_completed"] = True
        dist.destroy_process_group()
        document["destroy_process_group_returned"] = True
        document["status"] = "complete" if document["passed"] else "failed"
        if rank == leader_rank:
            _write_report(args.output, document)
    if final_error is not None:
        print(f"rank {rank}: {final_error}", flush=True)
    return 0 if document["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
