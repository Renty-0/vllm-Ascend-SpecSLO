# SPDX-License-Identifier: Apache-2.0
"""Independent same-FIA ancestor oracle, not a throughput benchmark.

Run once with four torchrun ranks: Qwen3-0.6B TP1 + Qwen3-32B TP3.
A preserves physical query shape and independently walks CPU parent lists to
construct FULL masks, positions and page addresses. It requires EXACT hidden,
logits, query KV and unchanged real committed-prefix/dependency KV. B retains
only ancestors, still using the same FIA FULL-mask ABI, and reports any
different-shape errors separately with unchanged explicit tolerances.
No production tree mask builder is used by the oracle. Production builders
are called only on the implementation-under-test side.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-model", default="/data/shared-models/Qwen3-0.6B")
    parser.add_argument("--target-model", default="/data/shared-models/Qwen3-32B")
    parser.add_argument("--contexts", type=int, nargs=2, default=[128, 256])
    parser.add_argument(
        "--prefix-cases-json",
        help="Use exactly two normal cases' real prefix/candidate IDs from an earlier diagnostic.",
    )
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--layouts", nargs="+", choices=("normal", "scratch"), default=["normal", "scratch"])
    parser.add_argument("--ancestor-reference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.001)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--output", required=True)
    return parser


def ancestor_nodes(parents, node):
    """Independent CPU traversal; reject cycles, forward edges and bad IDs."""
    if not -1 <= node < len(parents):
        raise ValueError("Oracle node is outside the declared tree")
    if any(not isinstance(parent, int) or not -1 <= parent < index for index, parent in enumerate(parents)):
        raise ValueError("Oracle parents must be acyclic and topologically ordered")
    path = []
    while node != -1:
        path.append(node)
        node = parents[node]
    return path[::-1]


def load_prefix_cases(path, max_model_len):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = [case for case in document["cases"] if case["layout"] == "normal"]
    if len(cases) != 2:
        raise ValueError("Select a diagnostic with exactly two normal prefix cases")
    for case in cases:
        prefix, candidates = case["prefix_token_ids"], case["candidate_token_ids"]
        if (
            len(prefix) < 2
            or len(prefix) + 16 > max_model_len
            or len(candidates) != 6
            or prefix[-1] != case["root_token_id"]
        ):
            raise ValueError("Saved prefix/candidate IDs do not fit the explicit 2x3 experiment")
    return cases


def selected_parents(parents, selected):
    if selected != sorted(set(selected)) or any(not 0 <= node < len(parents) for node in selected):
        raise ValueError("Oracle selection must be ordered and unique")
    remap = {node: index for index, node in enumerate(selected)}
    for node in selected:
        if any(ancestor not in remap for ancestor in ancestor_nodes(parents, node)):
            raise ValueError("Oracle selection is not ancestor closed")
    return [-1 if parents[node] == -1 else remap[parents[node]] for node in selected]


def oracle_metadata(specs, block_tables, block_size, max_model_len, *, device="cpu"):
    """Build addresses and both mask layouts from plain CPU facts only."""
    import torch

    from vllm_ascend.spec_decode.pearl.native_model import NativeAttentionMetadata

    tables = block_tables.detach().cpu().tolist() if isinstance(block_tables, torch.Tensor) else block_tables
    masks, token_tables, request_tables, slots, contexts, tokens, positions = [], [], [], [], [], [], []
    cumulative, sequence_lengths = [], []
    for spec in specs:
        parents, physical = spec["parents"], spec["physical"]
        count = len(parents) + 1
        if len(physical) != count or len(set(physical)) != count or len(spec["tokens"]) != count:
            raise ValueError("Oracle root/candidate counts or physical addresses are invalid")
        if len(spec["positions"]) != count or not 0 <= spec["prefix_count"] <= min(physical):
            raise ValueError("Oracle positions/prefix are inconsistent")
        if any(not 0 <= value < max_model_len for value in [*physical, *spec["dependencies"]]):
            raise ValueError("Oracle visible position exceeds cache capacity")
        if set(physical) & set(spec["dependencies"]):
            raise ValueError("Scratch query overlaps immutable dependency KV")
        table = tables[spec["sequence_id"]]
        request_tables.append(table)
        mask = torch.ones((count, max_model_len), dtype=torch.bool)
        for row, position in enumerate(physical):
            visible = [
                *range(spec["prefix_count"]),
                *spec["dependencies"],
                physical[0],
                *[physical[node + 1] for node in ancestor_nodes(parents, row - 1)],
            ]
            mask[row, visible] = False
            page = int(table[position // block_size])
            if page < 0:
                raise ValueError("Oracle query references an unallocated physical page")
            slots.append(page * block_size + position % block_size)
            token_tables.append(table)
            contexts.append(position + 1)
        masks.append(mask)
        tokens.extend(spec["tokens"])
        positions.extend(spec["positions"])
        cumulative.append((cumulative[-1] if cumulative else 0) + count)
        sequence_lengths.append(max(physical) + 1)
    envelope = torch.ones((len(specs), 1, max(mask.shape[0] for mask in masks), max_model_len), dtype=torch.bool)
    for request, mask in enumerate(masks):
        envelope[request, 0, : mask.shape[0]] = mask
    metadata = NativeAttentionMetadata(
        slot_mapping=torch.tensor(slots, dtype=torch.int32, device=device),
        context_lens=torch.tensor(contexts, dtype=torch.int32),
        block_tables=torch.tensor(token_tables, dtype=torch.int32, device=device),
        actual_seq_lengths_q=tuple(cumulative),
        sequence_lens=tuple(sequence_lengths),
        request_block_tables=torch.tensor(request_tables, dtype=torch.int32, device=device),
        attention_mask=torch.cat(masks).to(device),
        tree_attention_mask=envelope.to(device),
        use_fused_infer_attention=True,
        tree_attention=True,
    )
    return torch.tensor(tokens, dtype=torch.long, device=device), torch.tensor(positions, device=device), metadata


def ancestor_spec(spec, row):
    """Compact only query rows, retaining their original physical KV slots."""
    path = [0, *[node + 1 for node in ancestor_nodes(spec["parents"], row - 1)]]
    return {
        **spec,
        "parents": list(range(-1, len(path) - 2)),
        "tokens": [spec["tokens"][index] for index in path],
        "positions": [spec["positions"][index] for index in path],
        "physical": [spec["physical"][index] for index in path],
    }


def _storage(attention, name):
    value = getattr(attention, name)
    return value.flatten(0, 1) if value.ndim == 4 else value


def _snapshot(model, slots):
    return [
        tuple(_storage(layer.self_attn, name).index_select(0, slots).clone() for name in ("key_cache", "value_cache"))
        for layer in model.layers
    ]


def _restore(model, slots, snapshots):
    for layer, pair in zip(model.layers, snapshots):
        for name, saved in zip(("key_cache", "value_cache"), pair):
            _storage(layer.self_attn, name).index_copy_(0, slots, saved)


def _exact(actual, expected):
    import torch

    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    return {
        "finite": finite,
        "exact": finite and torch.equal(actual, expected),
        "max_abs_error": float((actual.float() - expected.float()).abs().max()) if finite else None,
    }


def _cache_compare(model, slots, snapshot):
    reports = []
    for layer_index, (layer, pair) in enumerate(zip(model.layers, snapshot)):
        for name, saved in zip(("key_cache", "value_cache"), pair):
            reports.append(
                {
                    "layer": layer_index,
                    "cache": name,
                    **_exact(_storage(layer.self_attn, name).index_select(0, slots), saved),
                }
            )
    return {"exact": all(row["exact"] for row in reports), "layers": reports}


def _persistent_slots(specs, tables, block_size, device):
    import torch

    tables = tables.cpu().tolist()
    slots = []
    for spec in specs:
        table = tables[spec["sequence_id"]]
        for position in [*range(spec["prefix_count"]), *spec["dependencies"]]:
            slots.append(table[position // block_size] * block_size + position % block_size)
    return torch.tensor(sorted(set(slots)), device=device, dtype=torch.long)


def _forward(engine, prepared, *, graph=False):
    inputs, positions, metadata = prepared
    if graph:
        hidden = engine.graph_runner.run_tree_hidden(inputs, positions, metadata)
        used_graph = bool(engine.graph_runner.last_target_execution.used_aclgraph)
    else:
        hidden, used_graph = engine.model(inputs, positions, metadata), False
    logits = engine.model.compute_logits(hidden)[:, : engine.target_vocab_size]
    return hidden.detach().clone(), logits.detach().clone(), used_graph


def _scenario(engine, args, layout, filler, prefix_cases=None):
    """Production side builds its plans; oracle facts use a separate formula."""
    from vllm_ascend.spec_decode.pearl.tree import build_tree_speculation_plan, pack_selected_tree_plan

    prompts = (
        [list(case["prefix_token_ids"]) for case in prefix_cases]
        if prefix_cases
        else [(filler * ((length + len(filler) - 1) // len(filler)))[:length] for length in args.contexts]
    )
    engine._allocate_cache(prompts, enable_prefix_caching=False)
    for sequence, row in enumerate(prompts):
        engine._run_packed_hidden(
            row[:-1],
            [sequence] * (len(row) - 1),
            list(range(len(row) - 1)),
            use_aclgraph=False,
            use_fused_infer_attention=True,
        )
    # Spine-first 2x3 topology: primary chain 0 -> 1 -> 2, and
    # alternatives 3 (root child), 4 (child of 0), 5 (child of 1).
    # This is an independent declaration, checked against the REAL builder
    # below; it must not be confused with two disjoint length-three chains.
    original_parents = [-1, 0, 1, -1, 0, 1]
    selections = [[0], [0, 1, 3, 5]]  # Heterogeneous q=[2,5], noncontiguous selection.
    plans, specs, roots, candidates = [], [], [], []
    for sequence, (prompt, selected) in enumerate(zip(prompts, selections)):
        base = len(prompt) - 1
        parent = build_tree_speculation_plan(2, 3, base, args.max_model_len, device=engine.device)
        if parent.parent_indices.cpu().tolist() != original_parents:
            raise ValueError("Production 2x3 topology differs from the independent spine-first declaration")
        row_candidates = (
            list(prefix_cases[sequence]["candidate_token_ids"])
            if prefix_cases
            else [(prompt[-1] + node + 17) % engine.target_vocab_size for node in range(6)]
        )
        root_position, physical_start, dependencies = base, base, []
        root = prompt[-1]
        if layout == "scratch":
            slots = parent.cache_positions.cpu().tolist()
            engine._ensure_cache_capacity([sequence] * len(slots), slots)
            prepared = engine.model.make_tree_attention_metadata(
                [parent], [root], [row_candidates], engine.cache_block_tables, sequence_ids=[sequence]
            )
            _forward(engine, prepared)
            # First branch is root -> 0 -> 1 -> 2. These facts do not read
            # tree_primary_path or the production continuation mask.
            dependencies = list(range(base, base + 4))
            root_position, physical_start = base + 4, base + 7
            root = (root + 83) % engine.target_vocab_size
        exploration = build_tree_speculation_plan(2, 3, root_position, args.max_model_len, device=engine.device)
        plan = pack_selected_tree_plan(exploration, selected)
        if layout == "scratch":
            plan = engine._tree_eager_scratch_plan(plan, parent)
        parents = selected_parents(original_parents, selected)
        spec = {
            "sequence_id": sequence,
            "parents": parents,
            "prefix_count": base,
            "dependencies": dependencies,
            "physical": list(range(physical_start, physical_start + len(selected) + 1)),
            "positions": [
                root_position,
                *[root_position + len(ancestor_nodes(original_parents, node)) for node in selected],
            ],
            "tokens": [root, *[row_candidates[node] for node in selected]],
            "original_selected_ids": selected,
            "prefix_token_ids": prompt,
        }
        if plan.parent_indices.cpu().tolist() != parents or plan.positions.cpu().tolist() != spec["positions"]:
            raise ValueError("Production selected plan differs from independent CPU topology/depth")
        plans.append(plan)
        specs.append(spec)
        roots.append(root)
        candidates.append(spec["tokens"][1:])
    engine._ensure_cache_capacity(
        [index for index, spec in enumerate(specs) for _ in spec["physical"]],
        [position for spec in specs for position in spec["physical"]],
    )
    production = engine.model.make_tree_attention_metadata(
        plans, roots, candidates, engine.cache_block_tables, sequence_ids=[0, 1]
    )
    oracle = oracle_metadata(
        specs, engine.cache_block_tables, engine.model.block_size, args.max_model_len, device=engine.device
    )
    return specs, production, oracle


def _run_case(engine, args, layout, filler, prefix_cases=None):
    import torch

    from examples.check_specslo_tree_numerics import _logit_report

    specs, production, oracle = _scenario(engine, args, layout, filler, prefix_cases)
    metadata_equal = all(
        torch.equal(getattr(production[2], key), getattr(oracle[2], key))
        for key in (
            "slot_mapping",
            "context_lens",
            "block_tables",
            "request_block_tables",
            "attention_mask",
            "tree_attention_mask",
        )
    )
    metadata_equal &= production[2].actual_seq_lengths_q == oracle[2].actual_seq_lengths_q
    metadata_equal &= production[2].sequence_lens == oracle[2].sequence_lens
    metadata_equal &= torch.equal(production[0], oracle[0]) and torch.equal(production[1], oracle[1])
    query_slots = oracle[2].slot_mapping.long()
    prefix_slots = _persistent_slots(specs, engine.cache_block_tables, engine.model.block_size, engine.device)
    if query_slots.unique().numel() != query_slots.numel() or bool(torch.isin(query_slots, prefix_slots).any()):
        raise ValueError("Query KV aliases another query or an immutable prefix/dependency")
    all_slots = torch.cat((query_slots, prefix_slots)).unique()
    before = _snapshot(engine.model, all_slots)
    prefix_before = _snapshot(engine.model, prefix_slots)
    baseline_hidden, baseline_logits, _ = _forward(engine, production)
    query_reference = _snapshot(engine.model, query_slots)
    baseline_prefix = _cache_compare(engine.model, prefix_slots, prefix_before)
    comparisons = []
    for label, prepared, graph in (
        ("independent_eager", oracle, False),
        ("production_graph", production, True),
        ("independent_graph", oracle, True),
    ):
        _restore(engine.model, all_slots, before)
        hidden, logits, used_graph = _forward(engine, prepared, graph=graph)
        comparisons.append(
            {
                "variant": label,
                "used_aclgraph": used_graph,
                "hidden": _exact(hidden, baseline_hidden),
                "logits": _exact(logits, baseline_logits),
                "query_kv": _cache_compare(engine.model, query_slots, query_reference),
                "persistent_prefix_kv": _cache_compare(engine.model, prefix_slots, prefix_before),
            }
        )
    # Root must not observe ANY candidate from its own tree. Keep full M/ABI
    # shape and mutate all candidates, including the primary branch.
    changed = production[0].clone()
    roots = [0, len(specs[0]["tokens"])]
    for row in range(changed.numel()):
        if row not in roots:
            changed[row] = (changed[row] + 113) % engine.target_vocab_size
    _restore(engine.model, all_slots, before)
    changed_hidden, changed_logits, _ = _forward(engine, (changed, production[1], production[2]))
    sibling = {
        "hidden": _exact(changed_hidden[roots], baseline_hidden[roots]),
        "logits": _exact(changed_logits[roots], baseline_logits[roots]),
        "persistent_prefix_kv": _cache_compare(engine.model, prefix_slots, prefix_before),
    }
    b_rows = []
    if args.ancestor_reference:
        offset = 0
        for spec in specs:
            for row in range(len(spec["tokens"])):
                _restore(engine.model, all_slots, before)
                subset = ancestor_spec(spec, row)
                prepared = oracle_metadata(
                    [subset],
                    engine.cache_block_tables,
                    engine.model.block_size,
                    args.max_model_len,
                    device=engine.device,
                )
                hidden, logits, _ = _forward(engine, prepared)
                b_rows.append(
                    {
                        "request": spec["sequence_id"],
                        "row": row,
                        "physical_query_count": len(subset["tokens"]),
                        "hidden": _exact(hidden[-1], baseline_hidden[offset + row]),
                        "logits": _logit_report(
                            baseline_logits[offset + row].cpu(),
                            logits[-1].cpu(),
                            atol=args.atol,
                            rtol=args.rtol,
                            top_k=5,
                        ),
                        "persistent_prefix_kv": _cache_compare(engine.model, prefix_slots, prefix_before),
                    }
                )
            offset += len(spec["tokens"])
    a_pass = (
        metadata_equal
        and baseline_prefix["exact"]
        and all(
            row["hidden"]["exact"]
            and row["logits"]["exact"]
            and row["query_kv"]["exact"]
            and row["persistent_prefix_kv"]["exact"]
            and (row["used_aclgraph"] or row["variant"] == "independent_eager")
            for row in comparisons
        )
        and all(row["exact"] for row in sibling.values())
    )
    b_pass = all(
        row["logits"]["within_tolerance"] and row["logits"]["argmax_exact"] and row["persistent_prefix_kv"]["exact"]
        for row in b_rows
    )
    _restore(engine.model, all_slots, before)
    engine._release_cache()
    return {
        "layout": layout,
        "specifications": specs,
        "query_counts": [2, 5],
        "independent_metadata_exact": metadata_equal,
        "baseline_prefix_kv": baseline_prefix,
        "A_same_shape": comparisons,
        "A_all_candidates_hidden_from_roots": sibling,
        "A_exact_passed": a_pass,
        "B_ancestor_only": b_rows,
        "B_requested": args.ancestor_reference,
        "B_strict_passed": b_pass if args.ancestor_reference else None,
        "graph_scope": "Transformer hidden graph; identical-shape logits head evaluated outside graph",
    }


def _engine_config(args):
    """Construct the REAL validated configuration before any device loading."""
    from vllm_ascend.spec_decode.pearl.native_engine import NativePearlConfig

    return NativePearlConfig(
        args.draft_model,
        args.target_model,
        1,
        3,
        6,
        args.max_model_len,
        1,
        max_num_seqs=2,
        enforce_eager=False,
        enable_prefix_caching=False,
        enable_spec_rhythm=True,
        enable_continuous_batching=True,
        enable_preemptive_scheduling=True,
        spec_rhythm_tree_width=2,
        spec_rhythm_tree_depth=3,
        max_aclgraph_entries=64,
        max_num_batched_tokens=max(16384, args.max_model_len),
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=0,
    )


def main(argv=None):
    args = _parser().parse_args(argv)
    if min(args.contexts) < 2 or max(args.contexts) + 16 > args.max_model_len:
        raise ValueError("Two context lengths must fit prefix and scratch KV")
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("Tolerances must be nonnegative")
    prefix_cases = load_prefix_cases(args.prefix_cases_json, args.max_model_len) if args.prefix_cases_json else None
    config = _engine_config(args)
    import torch
    import torch.distributed as dist

    from vllm_ascend.spec_decode.pearl.native_engine import NativePearlEngine

    engine = NativePearlEngine(config)
    leader = engine.rank == engine.topology.target_leader_rank
    source = Path(inspect.getfile(type(engine.model))).resolve()
    report = {
        "purpose": __doc__,
        "args": vars(args),
        "cases": [],
        "status": "running",
        "native_model_source": str(source),
        "native_model_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "attention_backend": "fused_infer_attention_tree_v1",
        "numerical_mode_modified": False,
    }
    overall = True
    try:
        filler = engine.tokenizer.encode(
            " Add 17 and 25 and explain the calculation carefully.", add_special_tokens=False
        )
        for layout in args.layouts:
            result = None
            if not engine.is_draft:
                try:
                    with torch.inference_mode():
                        result = _run_case(engine, args, layout, filler, prefix_cases)
                except (ValueError, RuntimeError) as error:
                    result = {
                        "layout": layout,
                        "A_exact_passed": False,
                        "B_strict_passed": False if args.ancestor_reference else None,
                        "execution_error": str(error),
                    }
                    if engine.cache_allocation is not None:
                        engine._release_cache()
            flags = torch.zeros((4, 2), dtype=torch.int64, device=engine.device)
            if result is not None:
                flags[engine.rank, 0] = int(not result["A_exact_passed"])
                flags[engine.rank, 1] = int(args.ancestor_reference and not result["B_strict_passed"])
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
            overall &= not bool(flags.any().cpu())
            if leader:
                result["rank_failure_flags_A_B"] = flags.cpu().tolist()
                report["cases"].append(result)
                output = Path(args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                print(json.dumps({"layout": layout, "rank_failure_flags_A_B": flags.cpu().tolist()}), flush=True)
        if leader:
            report["status"] = "passed" if overall else "failed"
            Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    finally:
        if engine.cache_allocation is not None:
            engine._release_cache()
        dist.destroy_process_group()
    return int(not overall)


if __name__ == "__main__":
    raise SystemExit(main())
