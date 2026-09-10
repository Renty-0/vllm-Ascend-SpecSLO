# SPDX-License-Identifier: Apache-2.0
"""Inspect actual online SpecSLO queries without rebuilding their prefix KV.

Run with torchrun --nproc_per_node=4. Diagnostic only: head projections, KV
snapshots and singleton teacher paths add substantial overhead. Never report
its elapsed time as a throughput result. This script is eager-only and does
not modify production greedy semantics. --watch entries are zero-based
request-index:completion-offset, e.g. 5:27 6:61.

The observed tree may predict a watched offset on several sibling branches.
All matching query rows are recorded; the final output is used to identify
which ancestor token paths actually match the generated completion prefix.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--gsm8k", default="/data/datasets/gsm8k/test.parquet")
    source.add_argument("--manifest")
    parser.add_argument("--draft-model", default="/data/shared-models/Qwen3-0.6B")
    parser.add_argument("--target-model", default="/data/shared-models/Qwen3-32B")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--batch-size", "--online-capacity", dest="batch_size", type=int, default=4)
    parser.add_argument("--queue-capacity", type=int)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--tree-width", type=int, default=2)
    parser.add_argument("--tree-depth", type=int, default=2)
    parser.add_argument("--gamma", type=int, help="Legacy draft gamma; independent of physical tree node count.")
    parser.add_argument("--verification-budget", type=int, default=8)
    parser.add_argument("--max-eager-tokens", type=int)
    parser.add_argument("--urgency-threshold", type=float, default=0.0)
    parser.add_argument("--acceptance-floor", type=float, default=0.0)
    parser.add_argument("--online-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--priority-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--slo-tpot-ms", type=float, nargs="+", help="Repeating TPOT limits; manifest values take priority."
    )
    parser.add_argument("--arrival-interval", type=float, default=0.0)
    parser.add_argument("--arrival-lead", type=float, default=0.0)
    parser.add_argument("--prefill-chunk-size", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-aclgraph-entries", type=int, default=64)
    parser.add_argument("--profile-decode-steps", type=int, default=0)
    parser.add_argument(
        "--preceding-manifest", help="Replay one earlier case on the SAME engine before installing hooks."
    )
    parser.add_argument("--watch", nargs="+", default=["5:27", "6:61"])
    parser.add_argument("--paged-attention", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--teacher-paths", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _engine_config_values(args):
    if (
        min(
            args.num_prompts,
            args.batch_size,
            args.max_tokens,
            args.max_model_len,
            args.tree_width,
            args.tree_depth,
            args.verification_budget,
        )
        <= 0
    ):
        raise ValueError("Batch, context and tree limits must be positive")
    queue_capacity = args.num_prompts if args.queue_capacity is None else args.queue_capacity
    if queue_capacity < args.num_prompts:
        raise ValueError("Queue capacity must hold every requested input")
    if min(args.arrival_interval, args.arrival_lead) < 0 or not all(
        math.isfinite(value) for value in (args.arrival_interval, args.arrival_lead)
    ):
        raise ValueError("Arrival offsets must be finite and nonnegative")
    return dict(
        draft_model=args.draft_model,
        target_model=args.target_model,
        draft_tp_size=1,
        target_tp_size=3,
        gamma=args.gamma if args.gamma is not None else args.tree_width * args.tree_depth,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        max_num_seqs=args.batch_size,
        max_num_queued_seqs=queue_capacity,
        max_num_batched_tokens=args.max_num_batched_tokens,
        prefill_chunk_size=args.prefill_chunk_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_aclgraph_entries=args.max_aclgraph_entries,
        profile_decode_steps=args.profile_decode_steps,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_continuous_batching=True,
        enable_preemptive_scheduling=True,
        enable_spec_rhythm=True,
        spec_rhythm_online_prefill=args.online_prefill,
        spec_rhythm_priority_mode=args.priority_mode,
        spec_rhythm_verification_budget=args.verification_budget,
        spec_rhythm_tree_width=args.tree_width,
        spec_rhythm_tree_depth=args.tree_depth,
        spec_rhythm_max_eager_tokens=(
            args.max_eager_tokens if args.max_eager_tokens is not None else args.tree_width * args.tree_depth
        ),
        spec_rhythm_urgency_threshold=args.urgency_threshold,
        spec_rhythm_acceptance_floor=args.acceptance_floor,
        draft_use_paged_attention=args.paged_attention,
        target_use_paged_attention=args.paged_attention,
        seed=0,
    )


def _request_parameter_rows(args, rows, epoch):
    """Preserve SLOs/relative arrivals, never reuse historical absolute times."""
    has_arrivals = (
        args.arrival_interval > 0
        or args.arrival_lead > 0
        or any(row.get("arrival_offset_sec") is not None for row in rows)
    )
    parameters = []
    for index, row in enumerate(rows):
        offset_value = row.get("arrival_offset_sec")
        offset = index * args.arrival_interval if offset_value is None else float(offset_value)
        slo = row.get("slo_tpot_ms")
        if slo is None and args.slo_tpot_ms:
            slo = args.slo_tpot_ms[index % len(args.slo_tpot_ms)]
        if not math.isfinite(offset) or offset < 0:
            raise ValueError("Manifest arrival offsets must be finite and nonnegative")
        if slo is not None and (not math.isfinite(float(slo)) or float(slo) <= 0):
            raise ValueError("TPOT SLOs must be finite and positive")
        parameters.append(
            dict(
                temperature=0.0,
                draft_temperature=0.0,
                max_tokens=args.max_tokens,
                ignore_eos=True,
                request_id=row.get("request_id", f"online-diagnostic:{index}"),
                arrival_ts=epoch + offset if has_arrivals else None,
                slo_tpot_ms=None if slo is None else float(slo),
                slo_class=row.get("slo_class"),
            )
        )
    return parameters


def _parse_watch(values, num_prompts, max_tokens):
    watches = set()
    for value in values:
        fields = value.split(":")
        if len(fields) != 2:
            raise ValueError("Watch entries must have request_index:output_offset form")
        request, offset = map(int, fields)
        if not 0 <= request < num_prompts or not 0 <= offset < max_tokens:
            raise ValueError("Watch entry lies outside the requested input/output range")
        watches.add((request, offset))
    return watches


def _selected_queries(plans, sequence_ids, prompt_lengths, watches):
    from examples.check_specslo_tree_numerics import _ancestor_path

    selected, cursor = [], 0
    for plan_index, (plan, request) in enumerate(zip(plans, sequence_ids)):
        parents = plan.parent_indices.detach().cpu().tolist()
        positions = plan.positions.detach().cpu().tolist()
        if len(positions) != len(parents) + 1:
            raise ValueError("Packed tree positions and physical candidate count disagree")
        for row, position in enumerate(positions):
            output_offset = int(position) + 1 - prompt_lengths[request]
            if (request, output_offset) in watches:
                selected.append(
                    {
                        "plan_index": plan_index,
                        "request_index": request,
                        "output_offset": output_offset,
                        "query_row": cursor + row,
                        "local_query_row": row,
                        "ancestor_nodes": _ancestor_path(parents, row - 1),
                        "tree_row_start": cursor,
                        "root_output_offset": int(positions[0]) + 1 - prompt_lengths[request],
                    }
                )
        cursor += len(positions)
    return selected


def _independent_mask(plan, node, *, device):
    import torch

    from examples.check_specslo_tree_numerics import _ancestor_path

    parents = plan.parent_indices.detach().cpu().tolist()
    cache_positions = plan.cache_positions.detach().cpu().tolist()
    mask = torch.ones(plan.max_model_len, dtype=torch.bool, device=device)
    mask[: plan.prefix_len] = False
    mask[cache_positions[0]] = False
    for ancestor in _ancestor_path(parents, node):
        mask[cache_positions[ancestor + 1]] = False
    return mask


def _storage(attention, name):
    value = getattr(attention, name)
    return value.flatten(0, 1) if attention.uses_paged_attention else value


def _snapshot_kv(model, slots):
    return [
        (
            _storage(layer.self_attn, "key_cache").index_select(0, slots).clone(),
            _storage(layer.self_attn, "value_cache").index_select(0, slots).clone(),
        )
        for layer in model.layers
    ]


def _restore_kv(model, slots, snapshot):
    for layer, (keys, values) in zip(model.layers, snapshot):
        _storage(layer.self_attn, "key_cache").index_copy_(0, slots, keys)
        _storage(layer.self_attn, "value_cache").index_copy_(0, slots, values)


def _kv_equal(model, slots, snapshot):
    import torch

    return all(
        torch.equal(_storage(layer.self_attn, name).index_select(0, slots), expected)
        for layer, pair in zip(model.layers, snapshot)
        for name, expected in zip(("key_cache", "value_cache"), pair)
    )


def _physical_prefix_slots(plan, block_table, block_size):
    import torch

    positions = torch.arange(plan.prefix_len, device=block_table.device, dtype=torch.long)
    blocks = block_table.index_select(0, positions // block_size).long()
    if bool((blocks < 0).any()):
        raise ValueError("The actual online committed prefix references an unallocated page")
    return blocks * block_size + positions.remainder(block_size)


def _singleton_metadata(metadata, query_row, mask):
    return replace(
        metadata,
        slot_mapping=metadata.slot_mapping[query_row : query_row + 1],
        context_lens=metadata.context_lens[query_row : query_row + 1],
        block_tables=metadata.block_tables[query_row : query_row + 1],
        attention_mask=mask.unsqueeze(0),
        actual_seq_lengths_q=(1,),
        sequence_lens=(int(metadata.context_lens[query_row]),),
        request_block_tables=None,
        use_fused_infer_attention=False,
    )


def _teacher_comparisons(engine, plans, selected, captured, args, tree_logits):
    import torch

    from examples.check_specslo_tree_numerics import _logit_report

    inputs, positions, metadata, hidden = captured
    query_slots = metadata.slot_mapping.long()
    if query_slots.unique().numel() != query_slots.numel():
        raise ValueError("Diagnostic refuses overlapping actual tree query slots")
    prefix_parts = [
        _physical_prefix_slots(
            plans[row["plan_index"]],
            metadata.block_tables[row["query_row"]],
            engine.model.layers[0].self_attn.block_size,
        )
        for row in selected
    ]
    prefix_slots = torch.cat(prefix_parts).long().unique()
    if bool(torch.isin(query_slots, prefix_slots).any()):
        raise ValueError("Diagnostic query slots overlap a watched committed prefix")
    query_snapshot = _snapshot_kv(engine.model, query_slots)
    prefix_snapshot = _snapshot_kv(engine.model, prefix_slots)
    reports = []
    for selection in selected:
        plan = plans[selection["plan_index"]]
        query_row = selection["query_row"]
        path_rows = [
            selection["tree_row_start"],
            *[selection["tree_row_start"] + node + 1 for node in selection["ancestor_nodes"]],
        ]
        row_report = {
            **selection,
            "teacher_query_rows": path_rows,
            "teacher_token_ids": inputs[path_rows].cpu().tolist(),
            "actual_prefix_recomputed": False,
        }
        masks = [
            _independent_mask(plan, row - selection["tree_row_start"] - 1, device=metadata.attention_mask.device)
            for row in path_rows
        ]
        if any(not torch.equal(mask, metadata.attention_mask[row]) for mask, row in zip(masks, path_rows)):
            row_report["status"] = "independent_ancestor_mask_mismatch_not_executed"
            reports.append(row_report)
            continue
        try:
            for row, mask in zip(path_rows, masks):
                teacher_hidden = engine.model(
                    inputs[row : row + 1], positions[row : row + 1], _singleton_metadata(metadata, row, mask)
                )
            reference = engine.model.compute_logits(teacher_hidden)[0, : engine.target_vocab_size]
            row_report["logits"] = _logit_report(
                tree_logits[query_row].cpu(),
                reference.detach().float().cpu(),
                atol=args.atol,
                rtol=args.rtol,
                top_k=args.top_k,
            )
            row_report["hidden_max_abs_error"] = float(
                (hidden[query_row].float() - teacher_hidden[0].float()).abs().max()
            )
            layers = []
            for layer_index, (layer, originals) in enumerate(zip(engine.model.layers, query_snapshot)):
                errors = {}
                for name, original in zip(("key_cache", "value_cache"), originals):
                    actual = _storage(layer.self_attn, name).index_select(0, query_slots[query_row : query_row + 1])[0]
                    errors[name + "_max_abs_error"] = float((actual.float() - original[query_row].float()).abs().max())
                layers.append({"layer": layer_index, **errors})
            row_report["query_kv_by_layer"] = layers
            row_report["committed_prefix_kv_unchanged"] = _kv_equal(engine.model, prefix_slots, prefix_snapshot)
            row_report["status"] = "measured"
        finally:
            # Restore BOTH queried and committed-prefix slots even on a probe
            # failure. The production result and its cache must remain intact.
            _restore_kv(engine.model, query_slots, query_snapshot)
            _restore_kv(engine.model, prefix_slots, prefix_snapshot)
            row_report["query_kv_restored"] = _kv_equal(engine.model, query_slots, query_snapshot)
            row_report["prefix_kv_restored"] = _kv_equal(engine.model, prefix_slots, prefix_snapshot)
        reports.append(row_report)
    return reports


def _install_observer(engine, prompts, watches, args, records):
    """Install an instance-local wrapper; return a function that removes it."""
    from vllm_ascend.spec_decode.pearl.tree import pack_selected_tree_plan

    original_target = engine.target_tree_forward

    def observed_target(plans, root_token_ids, draft_token_ids, sequence_ids=None, return_logits=True):
        if engine.is_draft:
            return original_target(plans, root_token_ids, draft_token_ids, sequence_ids, return_logits)
        ids = list(range(len(plans))) if sequence_ids is None else list(sequence_ids)
        packed = [pack_selected_tree_plan(plan, range(int(plan.candidate_budget))) for plan in plans]
        selected = _selected_queries(packed, ids, list(map(len, prompts)), watches)
        if not selected:
            return original_target(plans, root_token_ids, draft_token_ids, sequence_ids, return_logits)
        if engine.rank == engine.topology.target_leader_rank:
            print(json.dumps({"event": "online_probe_start", "probe": len(records), "selected": selected}), flush=True)
        forwards = []

        def capture_forward(_module, inputs, output):
            forwards.append((*inputs, output))

        handle = engine.model.register_forward_hook(capture_forward)
        try:
            production = original_target(plans, root_token_ids, draft_token_ids, sequence_ids, return_logits)
        finally:
            handle.remove()
        if len(forwards) != 1 or len(forwards[0]) != 4:
            raise RuntimeError("Online eager diagnostic requires exactly one original model forward")
        captured = forwards[0]
        inputs, positions, metadata, hidden = captured
        from examples.check_specslo_greedy_ties import compare_native_head

        head_report = compare_native_head(engine.model.lm_head, hidden, engine.target_vocab_size)
        record = {
            "probe_index": len(records),
            "rank": engine.rank,
            "actual_packed_query_count": int(inputs.numel()),
            "selected_queries": selected,
            "sequence_ids": ids,
            "root_token_ids": list(root_token_ids),
            "candidate_token_ids": [list(row) for row in draft_token_ids],
            "input_token_ids": inputs.cpu().tolist(),
            "logical_positions": positions.cpu().tolist(),
            "physical_slot_mapping": metadata.slot_mapping.cpu().tolist(),
            "physical_block_tables": metadata.block_tables.cpu().tolist(),
            "context_lens": metadata.context_lens.cpu().tolist(),
            "visible_mask_indices": [
                (~row.bool()).nonzero().flatten().cpu().tolist() for row in metadata.attention_mask
            ],
            "plans": [
                {
                    "parents": plan.parent_indices.cpu().tolist(),
                    "prefix_len": plan.prefix_len,
                    "cache_positions": plan.cache_positions.cpu().tolist(),
                    "logical_positions": plan.positions.cpu().tolist(),
                }
                for plan in packed
            ],
            "original_model_forward_calls": 1,
            "same_hidden_head": head_report,
            "original_production_query_tokens": production["target_query_token_ids"].cpu().tolist(),
        }
        record["original_production_matches_same_hidden_greedy"] = (
            record["original_production_query_tokens"] == head_report["production_greedy_ids"]
        )
        if args.teacher_paths:
            tree_logits = engine.model.compute_logits(hidden)[:, : engine.target_vocab_size].detach().float()
            record["teacher_paths"] = _teacher_comparisons(engine, packed, selected, captured, args, tree_logits)
        records.append(record)
        if engine.rank == engine.topology.target_leader_rank:
            print(json.dumps({"event": "online_probe_end", "probe": record["probe_index"]}), flush=True)
        return production

    engine.target_tree_forward = observed_target
    return lambda: setattr(engine, "target_tree_forward", original_target)


def _annotate_final_outputs(records, outputs):
    for record in records:
        for selection in record["selected_queries"]:
            output = outputs[selection["request_index"]]
            offset = selection["output_offset"]
            row = selection["query_row"]
            start = selection["tree_row_start"]
            path_tokens = [record["input_token_ids"][start + node + 1] for node in selection["ancestor_nodes"]]
            selection["ancestor_tokens_match_final_prefix"] = (
                path_tokens == output[selection["root_output_offset"] : offset]
            )
            selection["final_output_token"] = output[offset] if offset < len(output) else None
            selection["production_prediction_matches_final_output"] = (
                record["original_production_query_tokens"][row] == selection["final_output_token"]
            )


def main(argv=None):
    args = _build_parser().parse_args(argv)
    watches = _parse_watch(args.watch, args.num_prompts, args.max_tokens)
    config_values = _engine_config_values(args)
    import torch.distributed as dist

    from examples.regression_specslo_tree import _encode_rows, _load_rows
    from vllm_ascend.spec_decode.pearl.native_engine import NativePearlConfig, NativePearlEngine, NativeSamplingParams

    config = NativePearlConfig(**config_values)
    engine = NativePearlEngine(config)
    records = []
    restore = None
    try:
        rows = _load_rows(args.manifest or args.gsm8k, args.num_prompts)
        prompts = _encode_rows(rows, engine.tokenizer)

        def parameters_for(case_rows):
            epoch = [time.time() + args.arrival_lead if engine.rank == engine.topology.target_leader_rank else None]
            dist.broadcast_object_list(epoch, src=engine.topology.target_leader_rank)
            parameter_rows = _request_parameter_rows(args, case_rows, epoch[0])
            return [NativeSamplingParams(**values) for values in parameter_rows], parameter_rows

        preceding_outputs = None
        if args.preceding_manifest:
            preceding_rows = _load_rows(args.preceding_manifest, args.num_prompts)
            preceding_prompts = _encode_rows(preceding_rows, engine.tokenizer)
            preceding_parameters, _ = parameters_for(preceding_rows)
            preceding_result = engine.generate_batch(preceding_prompts, preceding_parameters)
            if preceding_result is not None:
                preceding_outputs = [row["completion_token_ids"] for row in preceding_result]
        restore = _install_observer(engine, prompts, watches, args, records)
        sampling_params, parameter_rows = parameters_for(rows)
        result = engine.generate_batch(prompts, sampling_params)
        all_records = [None] * engine.topology.world_size
        dist.all_gather_object(all_records, records)
        if engine.rank == engine.topology.target_leader_rank:
            outputs = [row["completion_token_ids"] for row in result]
            for worker_records in all_records:
                _annotate_final_outputs(worker_records, outputs)
            found = {
                (selection["request_index"], selection["output_offset"])
                for worker in all_records
                for record in worker
                for selection in record["selected_queries"]
            }
            report = {
                "args": {**vars(args), "output": str(args.output)},
                "engine_config": asdict(config),
                "request_parameters": parameter_rows,
                "preceding_case_output_token_ids": preceding_outputs,
                "prompt_token_ids": prompts,
                "output_token_ids": outputs,
                "records_by_rank": all_records,
                "unobserved_watch_entries": [list(pair) for pair in sorted(watches - found)],
                "scope": (
                    "Actual online hidden/prefix KV. Diagnostic overhead changes timing; not a performance result."
                ),
                "status": "measured_not_a_numerical_pass_verdict",
                "prefix_recomputed_for_teacher": False,
                "production_greedy_changed": False,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(f"Saved actual-online diagnostics: {args.output}", flush=True)
    finally:
        if restore is not None:
            restore()
        if engine.cache_allocation is not None:
            engine._release_cache()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
