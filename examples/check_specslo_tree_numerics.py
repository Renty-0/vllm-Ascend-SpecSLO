# SPDX-License-Identifier: Apache-2.0
"""Compare tree logits to ancestor-only teacher forcing on the SAME TP3 model.

Run under torchrun --nproc_per_node=4. Draft TP1 is loaded by the existing
engine but does not draft; only the three target ranks execute the probes.
There is no TP4 numerical-equivalence assumption and no throughput claim.

Example for the observed GSM8K row-5/row-6 divergences::

    torchrun --standalone --nproc_per_node=4 examples/check_specslo_tree_numerics.py \
      --baseline-json /path/baseline-tp4-eager-p8-t64.json \
      --candidate-json /path/tree-tp1tp3-eager-p8-t64-v3.json \
      --request-indices 5 6 --prefix-output-tokens 27 61 --output /path/numerics.json

Default thresholds are deliberately explicit: atol=0.01, rtol=0.001 over
every vocabulary logit, plus exact argmax by default. A small margin is a
diagnostic, NOT evidence that an observed difference is harmless or caused
by TP arithmetic. --no-require-argmax-match only changes the process verdict;
the JSON still records every changed argmax and its margin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--gsm8k", default="/data/datasets/gsm8k/test.parquet")
    source.add_argument("--manifest", help="JSONL prompt/prompt_token_ids input, as in regression_specslo_tree.py")
    parser.add_argument("--draft-model", default="/data/shared-models/Qwen3-0.6B")
    parser.add_argument("--target-model", default="/data/shared-models/Qwen3-32B")
    parser.add_argument(
        "--baseline-json", help="Only supplies fixed teacher-forced token prefixes, NOT TP3 reference logits."
    )
    parser.add_argument("--candidate-json", help="Optional prior tree outputs for the first-divergence context check.")
    parser.add_argument("--request-indices", type=int, nargs="+", default=[5, 6])
    parser.add_argument("--prefix-output-tokens", type=int, nargs="+", default=[27, 61])
    parser.add_argument("--width", type=int, default=2)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--layouts", choices=("normal", "scratch"), nargs="+", default=["normal", "scratch"])
    parser.add_argument(
        "--reference-backends",
        choices=("dense", "paged", "fia"),
        nargs="+",
        default=["dense", "paged", "fia"],
        help=(
            "Independent teacher-forced reference. 'fia' uses the production "
            "one-path FIA backend; dense/paged additionally quantify expected "
            "cross-kernel BF16 drift."
        ),
    )
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument(
        "--graph", action="store_true", help="Capture/replay the tree logits route; references remain eager."
    )
    parser.add_argument(
        "--prefix-mode",
        choices=("prefill", "incremental"),
        default="prefill",
        help="Recompute all fixed prefix in FIA, or prefill prompt then teacher-force dense decode.",
    )
    parser.add_argument(
        "--head-diagnostics",
        action="store_true",
        help="Eager-only: compare NativeLMHead greedy/full projection on the initial tree's SAME hidden tensor.",
    )
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--require-argmax-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--output", required=True)
    return parser


def _validate_args(args):
    if args.graph and args.head_diagnostics:
        raise ValueError("--head-diagnostics requires eager mode; graph logits do not expose their hidden tensor")
    if not args.request_indices or any(index < 0 for index in args.request_indices):
        raise ValueError("request indices must be non-negative")
    if len(args.prefix_output_tokens) not in (1, len(args.request_indices)):
        raise ValueError("prefix-output-tokens must have one value or one per request index")
    if any(value < 0 for value in args.prefix_output_tokens):
        raise ValueError("prefix output lengths must be non-negative")
    if min(args.width, args.depth, args.max_model_len, args.top_k) <= 0 or args.width * args.depth < 2:
        raise ValueError("tree dimensions and limits must be positive, with at least two nodes")
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("numerical tolerances must be non-negative")
    if not args.baseline_json and any(args.prefix_output_tokens):
        raise ValueError("nonzero prefix-output-tokens requires baseline-json")


def _output_rows(path):
    if path is None:
        return None
    payload = json.loads(Path(path).read_text())
    if "output_token_ids" in payload:
        return payload["output_token_ids"]
    for row in payload.get("results", []):
        if "output_token_ids" in row:
            return row["output_token_ids"]
    raise ValueError(f"No output_token_ids in {path}")


def _ancestor_path(parents, node):
    """Return root-excluding ancestor chain through the requested node."""
    if node == -1:
        return []
    if node < 0 or node >= len(parents):
        raise ValueError("tree node is outside the parent vector")
    path = []
    while node >= 0:
        path.append(node)
        parent = parents[node]
        if parent < -1 or parent >= node:
            raise ValueError("tree parents must be topologically ordered")
        node = parent
    return list(reversed(path))


def _candidate_rows(continuation, width, depth, vocabulary_size):
    if not continuation:
        raise ValueError("fixed candidate continuation cannot be empty")
    spine = [int(continuation[index % len(continuation)]) for index in range(depth)]
    siblings = [
        (spine[level] + alternative + 1) % vocabulary_size for level in range(depth) for alternative in range(1, width)
    ]
    return spine + siblings


def _logit_report(actual, reference, *, atol, rtol, top_k):
    import torch

    actual = torch.as_tensor(actual, dtype=torch.float32).reshape(-1)
    reference = torch.as_tensor(reference, dtype=torch.float32).reshape(-1)
    if actual.shape != reference.shape:
        raise ValueError("logit vectors have different vocabulary sizes")
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    if not finite:
        return {"finite": False, "within_tolerance": False, "argmax_exact": False}
    absolute = (actual - reference).abs()
    error = float(absolute.max())
    threshold = atol + rtol * reference.abs()
    actual_token = int(actual.argmax())
    reference_token = int(reference.argmax())
    count = min(max(top_k, 2), actual.numel())
    actual_values, actual_ids = actual.topk(count)
    reference_values, reference_ids = reference.topk(count)
    actual_probabilities = torch.softmax(actual, dim=-1)
    reference_probabilities = torch.softmax(reference, dim=-1)
    probability_error = (actual_probabilities - reference_probabilities).abs()
    mixture = 0.5 * (actual_probabilities + reference_probabilities)
    tiny = torch.finfo(torch.float32).tiny
    js_divergence = 0.5 * (
        (actual_probabilities * (actual_probabilities.clamp_min(tiny) / mixture.clamp_min(tiny)).log()).sum()
        + (reference_probabilities * (reference_probabilities.clamp_min(tiny) / mixture.clamp_min(tiny)).log()).sum()
    )
    margin = float(reference_values[0] - reference_values[1]) if count > 1 else None
    return {
        "finite": True,
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": error,
        "mean_abs_error": float(absolute.mean()),
        "rmse": float((absolute.square().mean()).sqrt()),
        "max_error_token_id": int(absolute.argmax()),
        "outside_tolerance_count": int((absolute > threshold).sum()),
        "within_tolerance": bool((absolute <= threshold).all()),
        # These metrics cover the complete probability vector. They quantify
        # BF16 FIA-versus-dense drift without relabeling a failed full-logit
        # allclose as success.
        "max_abs_probability_error": float(probability_error.max()),
        "mean_abs_probability_error": float(probability_error.mean()),
        "probability_l1": float(probability_error.sum()),
        "total_variation": float(0.5 * probability_error.sum()),
        "jensen_shannon_divergence": float(js_divergence),
        "argmax_exact": actual_token == reference_token,
        "actual_argmax": actual_token,
        "reference_argmax": reference_token,
        "actual_top1_top2_margin": float(actual_values[0] - actual_values[1]) if count > 1 else None,
        "reference_top1_top2_margin": margin,
        "reference_margin_within_2_max_error": margin is not None and margin <= 2 * error,
        "actual_logit_at_reference_argmax": float(actual[reference_token]),
        "reference_logit_at_actual_argmax": float(reference[actual_token]),
        "actual_top_k": list(zip(actual_ids[:top_k].tolist(), actual_values[:top_k].tolist())),
        "reference_top_k": list(zip(reference_ids[:top_k].tolist(), reference_values[:top_k].tolist())),
        "top_k_id_overlap": len(set(actual_ids[:top_k].tolist()).intersection(reference_ids[:top_k].tolist())),
        "interpretation": "Margin/error-bound overlap is only a possible sensitivity indicator, not a proven cause.",
    }


def _passed(comparison, require_argmax):
    return comparison["within_tolerance"] and (comparison["argmax_exact"] or not require_argmax)


def _prefix_prefill_length(base_tokens, prompt_token_count, prefix_mode):
    if prefix_mode == "prefill":
        return max(0, len(base_tokens) - 1)
    if prompt_token_count is None or not 0 < prompt_token_count <= len(base_tokens):
        raise ValueError("Incremental prefix preparation requires the original prompt length")
    return min(prompt_token_count, len(base_tokens) - 1)


def _prefill_base(engine, base_tokens, *, prompt_token_count=None, prefix_mode="prefill"):
    import torch

    engine._allocate_cache([base_tokens, base_tokens], enable_prefix_caching=False)
    prefix = base_tokens[:-1]
    prefill_length = _prefix_prefill_length(base_tokens, prompt_token_count, prefix_mode)
    if prefix:
        # Identical independent prefix forwards avoid giving the two sequences
        # different prefix GEMM shapes before the actual tree/path comparison.
        for sequence in (0, 1):
            engine._run_packed_hidden(
                prefix[:prefill_length],
                [sequence] * prefill_length,
                list(range(prefill_length)),
                use_aclgraph=False,
                use_fused_infer_attention=True,
            )
            for position in range(prefill_length, len(prefix)):
                positions, metadata = engine._prepare_attention_metadata([sequence], [position], False)
                mask = (torch.arange(engine.config.max_model_len, device=engine.device) > position).view(1, -1)
                metadata = replace(metadata, attention_mask=mask)
                engine.model(
                    torch.tensor([prefix[position]], dtype=torch.long, device=engine.device), positions, metadata
                )


def _tree_logits(
    engine,
    plan,
    root,
    candidates,
    use_graph,
    *,
    head_diagnostics=False,
    validate_graph_eager=False,
    atol=0.01,
    rtol=0.001,
    top_k=5,
):
    import torch

    if use_graph and head_diagnostics:
        raise ValueError("Head diagnostics require the original eager hidden tensor, not a second model forward")
    slots = plan.cache_positions.detach().cpu().tolist()
    engine._ensure_cache_capacity([0] * len(slots), slots)
    input_ids, positions, metadata = engine.model.make_tree_attention_metadata(
        [plan], [root], [candidates], engine.cache_block_tables, sequence_ids=[0]
    )
    if use_graph:
        logits = engine.graph_runner.run_tree_logits(input_ids, positions, metadata, engine.target_vocab_size)
        outcome = asdict(engine.graph_runner.last_target_execution)
        if validate_graph_eager:
            # Replay and eager consume the exact same packed tree, cache and
            # attention metadata.  This isolates graph state/input update bugs
            # from the separate tree-versus-one-path kernel comparison below.
            eager_hidden = engine.model(input_ids, positions, metadata)
            eager_logits = engine.model.compute_logits(eager_hidden)[:, : engine.target_vocab_size]
            torch.npu.synchronize()
            outcome["same_packed_eager"] = [
                _logit_report(actual, reference, atol=atol, rtol=rtol, top_k=top_k)
                for actual, reference in zip(logits.detach().float().cpu(), eager_logits.detach().float().cpu())
            ]
    else:
        hidden = engine.model(input_ids, positions, metadata)
        logits = engine.model.compute_logits(hidden)[:, : engine.target_vocab_size]
        outcome = {"mode": "eager", "capture_attempted": False, "replay_executed": False}
        if head_diagnostics:
            import torch.distributed as dist

            from examples.check_specslo_greedy_ties import compare_native_head

            # Every target TP rank executes this branch in the same order.
            # Reuse hidden rather than rerunning the transformer/KV writes.
            head = engine.model.lm_head
            local_report = compare_native_head(head, hidden, engine.target_vocab_size)
            reports = [local_report]
            if head.context.size > 1:
                reports = [None] * head.context.size
                dist.all_gather_object(reports, local_report, group=head.context.group)
            outcome["head_diagnostics"] = {
                "same_original_hidden_tensor": True,
                "additional_transformer_forwards": 0,
                "prefix_cache_recomputed_for_head_test": False,
                "ranks": reports,
            }
    torch.npu.synchronize()
    return logits.detach().float().cpu().clone(), outcome


def _teacher_path_logits(engine, root_position, tokens, backend):
    import torch

    hidden = None
    for offset, token in enumerate(tokens):
        position = root_position + offset
        positions, metadata = engine._prepare_attention_metadata([1], [position], backend == "fia")
        if backend == "dense":
            # Ordinary one-path causal mask, constructed independently of the
            # tree mask and parent-index packing code under test.
            mask = (torch.arange(engine.config.max_model_len, device=engine.device) > position).view(1, -1)
            metadata = replace(metadata, attention_mask=mask)
        hidden = engine.model(torch.tensor([token], dtype=torch.long, device=engine.device), positions, metadata)
    assert hidden is not None
    logits = engine.model.compute_logits(hidden)[0, : engine.target_vocab_size]
    return logits.detach().float().cpu()


def _run_probe(engine, base_tokens, continuation, args, layout, *, prompt_token_count=None):
    from vllm_ascend.spec_decode.pearl.tree import build_tree_speculation_plan, tree_primary_path

    _prefill_base(engine, base_tokens, prompt_token_count=prompt_token_count, prefix_mode=args.prefix_mode)
    root_position = len(base_tokens) - 1
    base_plan = build_tree_speculation_plan(
        args.width, args.depth, root_position, args.max_model_len, device=engine.device
    )
    parent_candidates = _candidate_rows(continuation, args.width, args.depth, engine.target_vocab_size)
    dependency_tokens = [base_tokens[-1]]
    parent = None
    if layout == "scratch":
        parent = base_plan
        _tree_logits(engine, parent, base_tokens[-1], parent_candidates, False)
        path = tree_primary_path(parent)
        dependency_tokens += [parent_candidates[node] for node in path]
        root = continuation[len(path) % len(continuation)]
        logical_plan = build_tree_speculation_plan(
            args.width, args.depth, root_position + len(path) + 1, args.max_model_len, device=engine.device
        )
        plan = engine._tree_eager_scratch_plan(logical_plan, parent)
        tail = continuation[len(path) + 1 :] or continuation
        candidates = _candidate_rows(tail, args.width, args.depth, engine.target_vocab_size)
        dependency_tokens.append(root)
    else:
        plan, root, candidates = base_plan, base_tokens[-1], parent_candidates
    tree_logits, graph_outcome = _tree_logits(
        engine,
        plan,
        root,
        candidates,
        args.graph,
        head_diagnostics=args.head_diagnostics,
        validate_graph_eager=args.graph,
        atol=args.atol,
        rtol=args.rtol,
        top_k=args.top_k,
    )
    parents = plan.parent_indices.detach().cpu().tolist()
    comparisons = []
    for node in range(-1, len(parents)):
        path = _ancestor_path(parents, node)
        path_tokens = dependency_tokens + [candidates[index] for index in path]
        for backend in args.reference_backends:
            reference = _teacher_path_logits(engine, root_position, path_tokens, backend)
            comparison = _logit_report(
                tree_logits[node + 1], reference, atol=args.atol, rtol=args.rtol, top_k=args.top_k
            )
            comparisons.append(
                {
                    "node": node,
                    "path_indices": path,
                    "path_tokens": path_tokens,
                    "reference_backend": backend,
                    **comparison,
                }
            )
    primary = tree_primary_path(plan)
    perturbed = [
        token if index in primary else (token + 113) % engine.target_vocab_size
        for index, token in enumerate(candidates)
    ]
    changed_logits, perturb_graph_outcome = _tree_logits(engine, plan, root, perturbed, args.graph)
    invariance = [
        {
            "node": node,
            **_logit_report(
                changed_logits[node + 1], tree_logits[node + 1], atol=args.atol, rtol=args.rtol, top_k=args.top_k
            ),
        }
        for node in [-1, *primary]
    ]
    parent_invariance = []
    if parent is not None:
        dependency = set(tree_primary_path(parent))
        changed_parent = [
            token if index in dependency else (token + 227) % engine.target_vocab_size
            for index, token in enumerate(parent_candidates)
        ]
        _tree_logits(engine, parent, base_tokens[-1], changed_parent, False)
        changed_logits, _ = _tree_logits(engine, plan, root, candidates, args.graph)
        parent_invariance = [
            {
                "node": node,
                **_logit_report(
                    changed_logits[node + 1], tree_logits[node + 1], atol=args.atol, rtol=args.rtol, top_k=args.top_k
                ),
            }
            for node in range(-1, len(parents))
        ]
    result = {
        "layout": layout,
        "prefix_token_ids": base_tokens,
        "root_token_id": root,
        "candidate_token_ids": candidates,
        "parent_indices": parents,
        "logical_positions": plan.positions.detach().cpu().tolist(),
        "physical_cache_positions": plan.cache_positions.detach().cpu().tolist(),
        "teacher_forced_dependency_tokens": dependency_tokens,
        "tree_graph_execution": graph_outcome,
        "perturbed_tree_graph_execution": perturb_graph_outcome,
        "path_comparisons": comparisons,
        "own_sibling_invariance": invariance,
        "parent_sibling_scratch_invariance": parent_invariance,
        "target_worker_graph_metrics": engine.graph_metrics(),
        "passed": all(
            _passed(value, args.require_argmax_match) for value in comparisons + invariance + parent_invariance
        )
        and all(
            _passed(value, args.require_argmax_match)
            for value in graph_outcome.get("same_packed_eager", ())
        ),
    }
    engine._release_cache()
    return result


def main(argv=None):
    args = _build_parser().parse_args(argv)
    _validate_args(args)
    import torch
    import torch.distributed as dist

    from examples.regression_specslo_tree import _encode_rows, _load_rows
    from vllm_ascend.spec_decode.pearl.native_engine import NativePearlConfig, NativePearlEngine

    config = NativePearlConfig(
        draft_model=args.draft_model,
        target_model=args.target_model,
        draft_tp_size=1,
        target_tp_size=3,
        gamma=args.width * args.depth,
        max_model_len=args.max_model_len,
        max_tokens=1,
        max_num_seqs=2,
        max_num_queued_seqs=2,
        enforce_eager=not args.graph,
        enable_prefix_caching=False,
        max_aclgraph_entries=64,
        max_num_batched_tokens=max(16384, args.max_model_len),
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=0,
    )
    engine = NativePearlEngine(config)
    report = {
        "purpose": "Same-TP3 tree/path logit regression; not a TP4 equivalence or performance claim",
        "args": vars(args),
        "engine_config": asdict(config),
        "cases": [],
        "status": "running",
        "prefix_cache_provenance": (
            "Independent prompt FIA prefill plus one-token dense teacher forcing; not prior-run KV snapshots."
            if args.prefix_mode == "incremental"
            else "Identical independent full-prefix FIA prefills; not prior-run KV snapshots."
        ),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
    }
    output = Path(args.output)
    leader = engine.rank == engine.topology.target_leader_rank
    overall_passed = True
    try:
        rows = _load_rows(args.manifest or args.gsm8k, max(args.request_indices) + 1)
        prompts = _encode_rows(rows, engine.tokenizer)
        baseline = _output_rows(args.baseline_json)
        observed = _output_rows(args.candidate_json)
        if baseline is not None:
            baseline_payload = json.loads(Path(args.baseline_json).read_text())
            report["baseline_first_prompt_matches"] = baseline_payload.get("first_prompt_token_ids") == prompts[0]
            report["baseline_full_input_identity_verified"] = False
            if (
                baseline_payload.get("first_prompt_token_ids") is not None
                and not report["baseline_first_prompt_matches"]
            ):
                raise ValueError("baseline first prompt differs from the selected dataset/template")
        offsets = (
            args.prefix_output_tokens * len(args.request_indices)
            if len(args.prefix_output_tokens) == 1
            else args.prefix_output_tokens
        )
        filler = engine.tokenizer.encode(" The answer is 42. Let us reason carefully.", add_special_tokens=False)
        for request_index, offset in zip(args.request_indices, offsets):
            reference_row = baseline[request_index] if baseline is not None else []
            if offset > len(reference_row):
                raise ValueError("teacher-forced prefix exceeds stored baseline output")
            base_tokens = prompts[request_index] + reference_row[:offset]
            continuation = reference_row[offset:] or filler
            if len(base_tokens) + 2 * args.width * args.depth + 2 > args.max_model_len:
                raise ValueError("prefix and scratch tree exceed max_model_len")
            for layout in args.layouts:
                if leader:
                    print(
                        json.dumps(
                            {
                                "event": "probe_start",
                                "request": request_index,
                                "prefix_output_tokens": offset,
                                "layout": layout,
                            }
                        ),
                        flush=True,
                    )
                if not engine.is_draft:
                    with torch.inference_mode():
                        result = _run_probe(
                            engine,
                            base_tokens,
                            continuation,
                            args,
                            layout,
                            prompt_token_count=len(prompts[request_index]),
                        )
                    if leader:
                        result.update(
                            {
                                "request_index": request_index,
                                "prefix_output_tokens": offset,
                                "input_sha256": hashlib.sha256(json.dumps(base_tokens).encode()).hexdigest(),
                                "stored_tp4_next_token": reference_row[offset] if offset < len(reference_row) else None,
                                "stored_tree_next_token": observed[request_index][offset]
                                if observed is not None and offset < len(observed[request_index])
                                else None,
                                "stored_prefixes_equal_before_probe": observed[request_index][:offset]
                                == reference_row[:offset]
                                if observed is not None and baseline is not None
                                else None,
                            }
                        )
                        report["cases"].append(result)
                        overall_passed &= result["passed"]
                        output.parent.mkdir(parents=True, exist_ok=True)
                        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                        print(
                            json.dumps(
                                {
                                    "event": "probe_end",
                                    "request": request_index,
                                    "layout": layout,
                                    "passed": result["passed"],
                                }
                            ),
                            flush=True,
                        )
                status = torch.tensor([int(overall_passed) if leader else 0], dtype=torch.int32, device=engine.device)
                dist.broadcast(status, src=engine.topology.target_leader_rank)
                overall_passed = bool(status.cpu().item())
        if leader:
            report["status"] = "passed" if overall_passed else "failed"
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    finally:
        if engine.cache_allocation is not None:
            engine._release_cache()
        dist.destroy_process_group()
    return int(not overall_passed)


if __name__ == "__main__":
    raise SystemExit(main())
