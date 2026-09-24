# SPDX-License-Identifier: Apache-2.0
"""Qualify SpecSLO's linear-tree causal FIA fast path on real TP3 hardware.

Run with four ranks (draft TP1 + target TP3).  For identical contiguous chain
plans this compares the production causal FIA metadata against the forced FULL
tree-FIA reference, checks query KV writes, and requires causal ACLGraph replay
to match its eager result.  It is a numerical/graph gate, not a throughput or
B-roof measurement.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-model", default="/data/shared-models/Qwen3-0.6B")
    parser.add_argument("--target-model", default="/data/shared-models/Qwen3-32B")
    parser.add_argument("--contexts", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.001)
    parser.add_argument("--max-probability-error", type=float, default=0.002)
    parser.add_argument("--max-total-variation", type=float, default=0.01)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--fixed-kv-capacity",
        type=int,
        default=0,
        help=(
            "Qualify a FULL-mask target graph whose host KV lengths remain "
            "fixed at this capacity so every layer can skip task refresh."
        ),
    )
    parser.add_argument("--output", required=True)
    return parser


def _compare(actual, reference, *, atol, rtol):
    import torch

    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    close = finite and bool(torch.allclose(actual, reference, atol=atol, rtol=rtol))
    probabilities = actual.float().softmax(-1)
    reference_probabilities = reference.float().softmax(-1)
    probability_error = (probabilities - reference_probabilities).abs()
    return {
        "finite": finite,
        "allclose": close,
        "argmax_exact": finite and torch.equal(actual.argmax(-1), reference.argmax(-1)),
        "max_abs_error": float((actual.float() - reference.float()).abs().max()) if finite else None,
        "max_probability_error": float(probability_error.max()) if finite else None,
        "max_total_variation": float((probability_error.sum(-1) * 0.5).max()) if finite else None,
    }


def _forward(engine, prepared, *, graph):
    inputs, positions, metadata = prepared
    if graph:
        hidden = engine.graph_runner.run_tree_hidden(inputs, positions, metadata)
        used_graph = engine.graph_runner.last_target_execution.used_aclgraph
    else:
        hidden = engine.model(inputs, positions, metadata)
        used_graph = False
    logits = engine.model.compute_logits(hidden)[:, : engine.target_vocab_size]
    return hidden.clone(), logits.clone(), bool(used_graph)


def _config(args):
    from vllm_ascend.spec_decode.pearl.native_engine import NativePearlConfig

    return NativePearlConfig(
        args.draft_model,
        args.target_model,
        1,
        3,
        4,
        args.max_model_len,
        1,
        max_num_seqs=len(args.contexts),
        enforce_eager=False,
        enable_prefix_caching=False,
        enable_spec_rhythm=True,
        enable_continuous_batching=True,
        enable_preemptive_scheduling=True,
        spec_rhythm_tree_width=2,
        spec_rhythm_tree_depth=2,
        max_aclgraph_entries=16,
        max_num_batched_tokens=max(16384, len(args.contexts) * args.max_model_len),
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=0,
    )


def _target_case(engine, args, filler):
    import torch

    from examples.check_specslo_fia_ancestor_reference import _restore, _snapshot
    from vllm_ascend.spec_decode.pearl.roofline import attention_backend_identity
    from vllm_ascend.spec_decode.pearl.tree import build_tree_speculation_plan, pack_selected_tree_plan

    # Keep one additional prefetched token so a second same-shape request can
    # advance every KV length by one.  Replaying that second request exercises
    # the dynamic FIA task-update path rather than only the unchanged-length
    # graph fast path.
    prompts = [
        (filler * ((length + 1 + len(filler) - 1) // len(filler)))[: length + 1]
        for length in args.contexts
    ]
    engine._allocate_cache(prompts, enable_prefix_caching=False)
    for sequence, prompt in enumerate(prompts):
        engine._run_packed_hidden(
            prompt[:-1],
            [sequence] * (len(prompt) - 1),
            list(range(len(prompt) - 1)),
            use_aclgraph=False,
            use_fused_infer_attention=True,
        )
    selections = ([0], [0, 1])
    plans = [
        pack_selected_tree_plan(
            build_tree_speculation_plan(2, 2, len(prompt) - 2, args.max_model_len, device=engine.device),
            selected,
        )
        for prompt, selected in zip(prompts, selections)
    ]
    roots = [prompt[-2] for prompt in prompts]
    candidates = [
        [(root + 17 + index) % engine.target_vocab_size for index in range(len(selected))]
        for root, selected in zip(roots, selections)
    ]
    changed_plans = [
        pack_selected_tree_plan(
            build_tree_speculation_plan(
                2,
                2,
                len(prompt) - 1,
                args.max_model_len,
                device=engine.device,
            ),
            selected,
        )
        for prompt, selected in zip(prompts, selections)
    ]
    changed_roots = [prompt[-1] for prompt in prompts]
    changed_candidates = [
        [
            (root + 29 + index) % engine.target_vocab_size
            for index in range(len(selected))
        ]
        for root, selected in zip(changed_roots, selections)
    ]
    branched_selections = ([0], [0, 1, 2])
    branched_plans = [
        pack_selected_tree_plan(
            build_tree_speculation_plan(2, 2, len(prompt) - 2, args.max_model_len, device=engine.device),
            selected,
        )
        for prompt, selected in zip(prompts, branched_selections)
    ]
    branched_candidates = [
        [(root + 41 + index) % engine.target_vocab_size for index in range(len(selected))]
        for root, selected in zip(roots, branched_selections)
    ]
    engine._ensure_cache_capacity(
        [
            sequence
            for rows in (plans, changed_plans, branched_plans)
            for sequence, plan in enumerate(rows)
            for _ in plan.cache_positions
        ],
        [
            int(position)
            for rows in (plans, changed_plans, branched_plans)
            for plan in rows
            for position in plan.cache_positions.cpu().tolist()
        ],
    )
    fast = engine.model.make_tree_attention_metadata(
        plans, roots, candidates, engine.cache_block_tables, sequence_ids=list(range(len(plans)))
    )
    full = engine.model.make_tree_attention_metadata(
        plans,
        roots,
        candidates,
        engine.cache_block_tables,
        sequence_ids=list(range(len(plans))),
        allow_causal_fast_path=False,
    )
    changed = engine.model.make_tree_attention_metadata(
        changed_plans,
        changed_roots,
        changed_candidates,
        engine.cache_block_tables,
        sequence_ids=list(range(len(changed_plans))),
    )
    fixed_full = None
    if args.fixed_kv_capacity:
        if max(full[2].sequence_lens) > args.fixed_kv_capacity:
            raise ValueError("Fixed KV capacity does not cover the exact linear-tree context")
        fixed_tables = full[2].request_block_tables.clone()
        for row in range(fixed_tables.shape[0]):
            valid = fixed_tables[row] >= 0
            if not bool(valid.any()):
                raise RuntimeError("Fixed-KV qualification found an empty request page table")
            first_page = fixed_tables[row, valid][0]
            fixed_tables[row].masked_fill_(~valid, first_page)
        fixed_full = (
            full[0],
            full[1],
            replace(
                full[2],
                block_tables=fixed_tables,
                request_block_tables=fixed_tables,
                sequence_lens=(args.fixed_kv_capacity,) * len(full[2].sequence_lens),
            ),
        )
    branched = engine.model.make_tree_attention_metadata(
        branched_plans,
        roots,
        branched_candidates,
        engine.cache_block_tables,
        sequence_ids=list(range(len(branched_plans))),
    )
    if attention_backend_identity(fast[2]) != "fused_infer_attention_causal_v1":
        raise RuntimeError("Contiguous selected trees did not enter causal FIA")
    if attention_backend_identity(full[2]) != "fused_infer_attention_tree_v1":
        raise RuntimeError("Forced reference did not enter FULL tree FIA")
    if attention_backend_identity(branched[2]) != "fused_infer_attention_tree_v1":
        raise RuntimeError("Branched selected tree did not enter FULL tree FIA")
    slots = torch.cat(
        (
            fast[2].slot_mapping.long(),
            changed[2].slot_mapping.long(),
            branched[2].slot_mapping.long(),
        )
    ).unique()
    before = _snapshot(engine.model, slots)
    fast_hidden, fast_logits, _ = _forward(engine, fast, graph=False)
    fast_kv = _snapshot(engine.model, slots)
    _restore(engine.model, slots, before)
    full_hidden, full_logits, _ = _forward(engine, full, graph=False)
    full_kv = _snapshot(engine.model, slots)
    _restore(engine.model, slots, before)
    changed_hidden, changed_logits, _ = _forward(engine, changed, graph=False)
    _restore(engine.model, slots, before)
    fixed_hidden = fixed_logits = None
    if fixed_full is not None:
        fixed_hidden, fixed_logits, _ = _forward(engine, fixed_full, graph=False)
        _restore(engine.model, slots, before)
    branched_hidden, branched_logits, _ = _forward(engine, branched, graph=False)
    _restore(engine.model, slots, before)
    graph_hidden, graph_logits, used_graph = _forward(engine, fast, graph=True)
    # A second call proves an actual resident replay rather than capture-only
    # setup. The runner also performs its own eager validation on first reuse.
    graph_hidden, graph_logits, used_graph_second = _forward(engine, fast, graph=True)
    _restore(engine.model, slots, before)
    parallel_update_before = engine.graph_runner.target_fia_parallel_update_replays
    changed_graph_hidden, changed_graph_logits, used_changed_graph = _forward(
        engine,
        changed,
        graph=True,
    )
    _restore(engine.model, slots, before)
    full_capture_before = engine.graph_runner.capture_count
    full_replay_before = engine.graph_runner.replay_count
    full_graph_hidden, full_graph_logits, used_full_graph = _forward(engine, branched, graph=True)
    # Reuse the resident FULL-mask graph as well.  This is the path that must
    # remain live once the selected tree contains a sibling branch.
    full_graph_hidden, full_graph_logits, used_full_graph_second = _forward(engine, branched, graph=True)
    fixed_graph_reference = None
    if fixed_full is not None:
        fixed_capture_before = engine.graph_runner.capture_count
        fixed_replay_before = engine.graph_runner.replay_count
        fixed_skip_before = engine.graph_runner.task_update_skip_replay_count
        fixed_graph_hidden, fixed_graph_logits, fixed_used_graph = _forward(engine, fixed_full, graph=True)
        fixed_graph_hidden, fixed_graph_logits, fixed_used_graph_second = _forward(engine, fixed_full, graph=True)
        fixed_graph_reference = {
            "kv_capacity": args.fixed_kv_capacity,
            "eager_hidden_vs_exact_full": _compare(fixed_hidden, full_hidden, atol=args.atol, rtol=args.rtol),
            "eager_logits_vs_exact_full": _compare(fixed_logits, full_logits, atol=args.atol, rtol=args.rtol),
            "graph_hidden_vs_fixed_eager": _compare(
                fixed_graph_hidden,
                fixed_hidden,
                atol=0.001,
                rtol=0.001,
            ),
            "graph_logits_vs_fixed_eager": _compare(
                fixed_graph_logits,
                fixed_logits,
                atol=0.001,
                rtol=0.001,
            ),
            "used_aclgraph": fixed_used_graph and fixed_used_graph_second,
            "capture_count_delta": engine.graph_runner.capture_count - fixed_capture_before,
            "replay_count_delta": engine.graph_runner.replay_count - fixed_replay_before,
            "task_update_skip_replay_count_delta": (
                engine.graph_runner.task_update_skip_replay_count - fixed_skip_before
            ),
        }
    kv_rows = []
    for layer, (fast_pair, full_pair) in enumerate(zip(fast_kv, full_kv)):
        for cache, actual, reference in zip(("key", "value"), fast_pair, full_pair):
            kv_rows.append({"layer": layer, "cache": cache, "exact": torch.equal(actual, reference)})
    tree_reference = {
        "hidden": _compare(fast_hidden, full_hidden, atol=args.atol, rtol=args.rtol),
        "logits": _compare(fast_logits, full_logits, atol=args.atol, rtol=args.rtol),
        "query_kv_exact": all(row["exact"] for row in kv_rows),
        "query_kv": kv_rows,
        "cross_backend_probability_within_configured_diagnostic_limit": False,
    }
    tree_reference["cross_backend_probability_within_configured_diagnostic_limit"] = bool(
        tree_reference["logits"]["max_probability_error"] <= args.max_probability_error
        and tree_reference["logits"]["max_total_variation"] <= args.max_total_variation
    )
    graph_reference = {
        "hidden": _compare(graph_hidden, fast_hidden, atol=0.001, rtol=0.001),
        "logits": _compare(graph_logits, fast_logits, atol=0.001, rtol=0.001),
        "used_aclgraph": used_graph and used_graph_second,
        "capture_count": engine.graph_runner.capture_count,
        "replay_count": engine.graph_runner.replay_count,
    }
    changed_length_graph_reference = {
        "sequence_lens_before": list(fast[2].sequence_lens),
        "sequence_lens_after": list(changed[2].sequence_lens),
        "hidden": _compare(
            changed_graph_hidden,
            changed_hidden,
            atol=0.001,
            rtol=0.001,
        ),
        "logits": _compare(
            changed_graph_logits,
            changed_logits,
            atol=0.001,
            rtol=0.001,
        ),
        "used_aclgraph": used_changed_graph,
        "parallel_update_replays_delta": (
            engine.graph_runner.target_fia_parallel_update_replays
            - parallel_update_before
        ),
        "configured_parallel_update_workers": (
            engine.graph_runner.target_fia_task_update_workers
        ),
    }
    full_graph_reference = {
        "candidate_counts": [len(row) for row in branched_candidates],
        "hidden": _compare(full_graph_hidden, branched_hidden, atol=0.001, rtol=0.001),
        "logits": _compare(full_graph_logits, branched_logits, atol=0.001, rtol=0.001),
        "used_aclgraph": used_full_graph and used_full_graph_second,
        "capture_count_delta": engine.graph_runner.capture_count - full_capture_before,
        "replay_count_delta": engine.graph_runner.replay_count - full_replay_before,
    }
    passed = (
        tree_reference["hidden"]["finite"]
        and tree_reference["logits"]["finite"]
        and tree_reference["logits"]["argmax_exact"]
        and graph_reference["hidden"]["allclose"]
        and graph_reference["logits"]["allclose"]
        and graph_reference["logits"]["argmax_exact"]
        and graph_reference["used_aclgraph"]
        and changed_length_graph_reference["hidden"]["allclose"]
        and changed_length_graph_reference["logits"]["allclose"]
        and changed_length_graph_reference["logits"]["argmax_exact"]
        and changed_length_graph_reference["used_aclgraph"]
        and (
            changed_length_graph_reference["configured_parallel_update_workers"] == 1
            or changed_length_graph_reference["parallel_update_replays_delta"] >= 1
        )
        and full_graph_reference["hidden"]["allclose"]
        and full_graph_reference["logits"]["allclose"]
        and full_graph_reference["logits"]["argmax_exact"]
        and full_graph_reference["used_aclgraph"]
        and full_graph_reference["capture_count_delta"] == 1
        and full_graph_reference["replay_count_delta"] >= 2
        and (
            fixed_graph_reference is None
            or (
                fixed_graph_reference["eager_hidden_vs_exact_full"]["finite"]
                and fixed_graph_reference["eager_logits_vs_exact_full"]["argmax_exact"]
                and fixed_graph_reference["graph_hidden_vs_fixed_eager"]["allclose"]
                and fixed_graph_reference["graph_logits_vs_fixed_eager"]["allclose"]
                and fixed_graph_reference["graph_logits_vs_fixed_eager"]["argmax_exact"]
                and fixed_graph_reference["used_aclgraph"]
                and fixed_graph_reference["capture_count_delta"] == 1
                and fixed_graph_reference["replay_count_delta"] >= 2
                and fixed_graph_reference["task_update_skip_replay_count_delta"] >= 1
            )
        )
    )
    engine.graph_runner.release_target_graph_entries()
    engine._release_cache()
    return {
        "contexts": args.contexts,
        "candidate_counts": [len(row) for row in candidates],
        "tree_fia_reference": tree_reference,
        "tree_fia_reference_is_cross_backend_diagnostic_not_acceptance_oracle": True,
        "causal_aclgraph_reference": graph_reference,
        "changed_length_causal_aclgraph_reference": changed_length_graph_reference,
        "full_tree_aclgraph_reference": full_graph_reference,
        "fixed_kv_full_tree_aclgraph_reference": fixed_graph_reference,
        "passed": passed,
    }


def main(argv=None):
    args = _parser().parse_args(argv)
    if min(args.contexts) < 2 or max(args.contexts) + 4 > args.max_model_len:
        raise ValueError("Contexts and tree slots must fit max_model_len")
    if args.fixed_kv_capacity < 0 or args.fixed_kv_capacity > args.max_model_len:
        raise ValueError("Fixed KV capacity must fit max_model_len")
    if min(args.atol, args.rtol, args.max_probability_error, args.max_total_variation) < 0:
        raise ValueError("Tolerances must be non-negative")
    import torch
    import torch.distributed as dist

    from vllm_ascend.spec_decode.pearl.native_engine import NativePearlEngine

    engine = NativePearlEngine(_config(args))
    leader = engine.rank == engine.topology.target_leader_rank
    report = {"purpose": __doc__, "args": vars(args), "target_ranks": [], "status": "running"}
    result = None
    error = None
    try:
        filler = engine.tokenizer.encode(" A deterministic causal tree verification prefix.", add_special_tokens=False)
        if not engine.is_draft:
            try:
                with torch.inference_mode():
                    result = _target_case(engine, args, filler)
            except (RuntimeError, ValueError) as failure:
                error = str(failure)
        failed = torch.zeros(4, dtype=torch.int64, device=engine.device)
        if not engine.is_draft:
            failed[engine.rank] = int(error is not None or not result["passed"])
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if leader:
            report["target_ranks"].append(result if result is not None else {"error": error})
            report["rank_failure_flags"] = failed.cpu().tolist()
            report["status"] = "passed" if not bool(failed.any().cpu()) else "failed"
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return int(bool(failed.any().cpu()))
    finally:
        if engine.cache_allocation is not None:
            engine._release_cache()
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
