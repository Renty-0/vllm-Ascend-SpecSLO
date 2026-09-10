# SPDX-License-Identifier: Apache-2.0
"""Trace original-vs-bucket attention through ONE Qwen3-0.6B TP1 model.

No target model, distributed workers or decoder is loaded. A deterministic
256-token input plus root/tree physical slots 255..259 reproduces the metadata
boundary [256,257,258,259,260] -> 512. Actual historical input IDs can instead
be taken from --regression-json --request-index 3. Candidate IDs remain explicit
diagnostic inputs, not a claim to replay an earlier speculative service round.

Every forward restores an exact saved prefix KV snapshot. SDPA is intercepted
only to copy its actual inputs/outputs; the original implementation executes.
Those host copies add synchronization, so this is NOT a performance measurement
or proof that a timing-sensitive race cannot exist without instrumentation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path
from types import MethodType, SimpleNamespace


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/shared-models/Qwen3-0.6B")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--prompt-length", type=int, default=256)
    parser.add_argument("--prompt-text", default="What is 17 plus 25? Explain step by step.")
    parser.add_argument("--regression-json", help="Reuse exact prompt_suites[synthetic_kv_boundary] token IDs.")
    parser.add_argument(
        "--replay-snapshot", help="Replay an actual pre-forward snapshot instead of synthetic candidates."
    )
    parser.add_argument("--request-index", type=int, default=3)
    parser.add_argument("--candidate-token-ids", type=int, nargs="+")
    parser.add_argument("--width", type=int, default=2)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--tail-init", choices=("zero", "nan", "untouched"), default="untouched")
    parser.add_argument("--hf32", choices=("default", "off", "on"), default="off")
    parser.add_argument("--production-rope", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--atol", type=float, default=0.001)
    parser.add_argument("--rtol", type=float, default=0.001)
    parser.add_argument("--dump-first-inputs", help="Optional .pt file containing first differing actual SDPA inputs.")
    parser.add_argument("--output", required=True)
    return parser


def _validate(args):
    if min(args.prompt_length, args.width, args.depth, args.max_model_len) <= 0:
        raise ValueError("Prompt/model/tree dimensions must be positive")
    if args.prompt_length + args.width * args.depth > args.max_model_len:
        raise ValueError("Prompt and tree slots exceed max_model_len")
    if args.request_index < 0 or not args.device.startswith("npu:"):
        raise ValueError("Use a non-negative request index and one explicit npu:<index>")
    if args.candidate_token_ids is not None and len(args.candidate_token_ids) != args.width * args.depth:
        raise ValueError("candidate-token-ids must include exactly width*depth IDs")
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("Diagnostic tolerances must be non-negative")


def _snapshot_cache(model):
    # Device clones retain all initialized prefix/cache contents, not only
    # visible lengths. No re-prefill or random tail allocation occurs between
    # the two measured model evaluations.
    return [
        (layer.self_attn.key_cache.detach().clone(), layer.self_attn.value_cache.detach().clone())
        for layer in model.layers
    ]


def _restore_cache(model, snapshots):
    if len(model.layers) != len(snapshots):
        raise ValueError("KV snapshot layer count does not match model")
    for layer, (keys, values) in zip(model.layers, snapshots):
        attention = layer.self_attn
        if attention.key_cache.shape != keys.shape or attention.value_cache.shape != values.shape:
            raise ValueError("KV snapshot shape does not match live cache")
    for layer, (keys, values) in zip(model.layers, snapshots):
        layer.self_attn.key_cache.copy_(keys)
        layer.self_attn.value_cache.copy_(values)


def _cpu(value):
    return value.detach().cpu().clone()


def capture_draft_replay_snapshot(model, input_ids, positions, metadata):
    """CPU-only payload for torch.save; call BEFORE the observed model forward.

    Copies only referenced physical pages, including dummy page zero used by
    the bucket helper. Page addresses are remapped consistently, so replay
    does not require the service's entire multi-gigabyte cache allocation.
    It adds synchronization and is exclusively a diagnostic hook.
    """
    import torch

    block_size = model.layers[0].self_attn.block_size
    tables = _cpu(metadata.block_tables)
    pages = sorted(
        {
            0,
            *[int(page) for page in tables.reshape(-1).tolist() if page >= 0],
            *[int(slot) // block_size for slot in metadata.slot_mapping.cpu().reshape(-1).tolist()],
        }
    )
    mapping = {page: index for index, page in enumerate(pages)}
    attributes, devices = {}, {}
    for field in fields(metadata):
        value = getattr(metadata, field.name)
        if isinstance(value, torch.Tensor):
            devices[field.name] = value.device.type
            value = _cpu(value)
            if field.name in ("block_tables", "request_block_tables"):
                value = torch.tensor(
                    [mapping[int(page)] if page >= 0 else -1 for page in value.reshape(-1).tolist()], dtype=value.dtype
                ).reshape_as(value)
            elif field.name == "slot_mapping":
                value = torch.tensor(
                    [
                        mapping[int(slot) // block_size] * block_size + int(slot) % block_size
                        for slot in value.reshape(-1).tolist()
                    ],
                    dtype=value.dtype,
                ).reshape_as(value)
        attributes[field.name] = value
    caches = []
    for layer in model.layers:
        pair = []
        for cache in (layer.self_attn.key_cache, layer.self_attn.value_cache):
            storage = cache if cache.ndim == 4 else cache.reshape(-1, block_size, *cache.shape[-2:])
            if max(pages) >= storage.shape[0]:
                raise ValueError("Replay metadata references a page outside the live cache")
            pair.append(_cpu(storage.index_select(0, torch.tensor(pages, device=storage.device))))
        caches.append(pair)
    return {
        "schema_version": 1,
        "kind": "native_draft_pre_forward",
        "input_ids": _cpu(input_ids),
        "positions": _cpu(positions),
        "metadata": attributes,
        "metadata_devices": devices,
        "block_size": block_size,
        "original_physical_pages": pages,
        "cache": caches,
    }


def restore_draft_replay_snapshot(model, payload, *, device):
    """Restore compact actual KV and metadata, validating before any mutation."""
    import torch

    from vllm_ascend.spec_decode.pearl.native_model import NativeAttentionMetadata

    if payload.get("schema_version") != 1 or payload.get("kind") != "native_draft_pre_forward":
        raise ValueError("Unsupported draft replay snapshot schema")
    caches = payload["cache"]
    if len(caches) != len(model.layers):
        raise ValueError("Replay snapshot model layer count differs")
    validated = []
    for layer, pair in zip(model.layers, caches):
        if layer.self_attn.block_size != payload["block_size"] or len(pair) != 2:
            raise ValueError("Replay snapshot cache block layout differs")
        for cache, saved in zip((layer.self_attn.key_cache, layer.self_attn.value_cache), pair):
            storage = cache if cache.ndim == 4 else cache.reshape(-1, payload["block_size"], *cache.shape[-2:])
            if (
                saved.ndim != 4
                or storage.shape[1:] != saved.shape[1:]
                or storage.shape[0] < saved.shape[0]
                or saved.dtype != storage.dtype
            ):
                raise ValueError("Replay snapshot cache shape/dtype differs from loaded model")
            validated.append((storage, saved))
    values = {
        name: value.to(device if payload["metadata_devices"][name] != "cpu" else "cpu")
        if isinstance(value, torch.Tensor)
        else value
        for name, value in payload["metadata"].items()
    }
    metadata = NativeAttentionMetadata(**values)
    for storage, saved in validated:
        storage[: saved.shape[0]].copy_(saved)
    return payload["input_ids"].to(device), payload["positions"].to(device), metadata


def install_failed_draft_capture_hook(runner, output_path, *, minimum_context=257):
    """Install a bounded diagnostic observer; return a callable restoring it.

    Intended only for the TP1 draft rank. The payload is copied BEFORE graph
    validation mutates KV, but written only on the first matching failure.
    The original capture implementation, validation thresholds and outputs are
    not modified. Host copies can change timing, so this is not a benchmark.
    """
    import torch

    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite an earlier graph failure snapshot: {destination}")
    if runner.model.context.size != 1:
        raise ValueError("Draft replay capture is supported only for a TP1 model")
    original = runner._capture_target
    previous_override = runner.__dict__.get("_capture_target")
    saved = False

    def capture(
        entry_key,
        input_ids,
        positions,
        attention_metadatas,
        vocabulary_size,
        *,
        reference_metadatas=None,
        output_kind="greedy",
    ):
        nonlocal saved
        reference = reference_metadatas if reference_metadatas is not None else attention_metadatas
        observed = (
            not saved
            and output_kind == "hidden"
            and len(reference) == len(input_ids) == 1
            and int(reference[0].context_lens.max()) >= minimum_context
        )
        snapshot = (
            capture_draft_replay_snapshot(runner.model, input_ids[0], positions[0], reference[0]) if observed else None
        )
        failures = runner.failed_capture_count
        result = original(
            entry_key,
            input_ids,
            positions,
            attention_metadatas,
            vocabulary_size,
            reference_metadatas=reference_metadatas,
            output_kind=output_kind,
        )
        if snapshot is not None and runner.failed_capture_count > failures:
            snapshot["failure"] = {
                "entry_key": repr(entry_key),
                "validation": getattr(runner, "last_target_validation_error", None),
            }
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(snapshot, destination)
            saved = True
            print(json.dumps({"actual_draft_failure_snapshot": str(destination)}), flush=True)
        return result

    runner._capture_target = capture

    def remove():
        if previous_override is None:
            del runner._capture_target
        else:
            runner._capture_target = previous_override

    return remove


def _capture_forward(model, input_ids, positions, metadata):
    """Trace the tensors actually passed to SDPA, not a reconstructed gather."""
    import torch
    import torch.nn.functional as functional

    records = {"layers": {}, "attention": {}, "sdpa": {}}
    current = {}
    handles = []
    methods = []
    original_sdpa = functional.scaled_dot_product_attention

    def sdpa(query, keys, values, *args, **kwargs):
        if not current:
            return original_sdpa(query, keys, values, *args, **kwargs)
        layer, row = current["layer"], current["row"]
        current["row"] += 1
        mask = kwargs.get("attn_mask", args[0] if args else None)
        result = original_sdpa(query, keys, values, *args, **kwargs)
        length = int(keys.shape[-2])
        table = current["tables"][row]
        block_size = current["block_size"]
        slots = [int(table[position // block_size]) * block_size + position % block_size for position in range(length)]
        records["sdpa"][(layer, row)] = {
            "query": _cpu(query),
            "keys": _cpu(keys),
            "values": _cpu(values),
            "mask": None if mask is None else _cpu(mask),
            "output": _cpu(result),
            "physical_slots": slots,
            "context_len": current["context_lens"][row],
            "operator_options": {
                "scale": kwargs.get("scale"),
                "dropout_p": kwargs.get("dropout_p", args[1] if len(args) > 1 else 0.0),
                "is_causal": kwargs.get("is_causal", args[2] if len(args) > 2 else False),
            },
            "strides": {"query": list(query.stride()), "keys": list(keys.stride()), "values": list(values.stride())},
        }
        return result

    functional.scaled_dot_product_attention = sdpa
    try:
        for index, layer in enumerate(model.layers):
            attention = layer.self_attn
            original_dense = attention._dense_attention
            prior = attention.__dict__.get("_dense_attention")
            methods.append((attention, prior))

            def dense(module, query, attn_metadata, *, layer_index=index, original=original_dense):
                current.update(
                    layer=layer_index,
                    row=0,
                    block_size=module.block_size,
                    tables=attn_metadata.block_tables.detach().cpu().tolist(),
                    context_lens=attn_metadata.context_lens.detach().cpu().tolist(),
                )
                try:
                    output = original(query, attn_metadata)
                    if current["row"] != query.shape[0]:
                        raise RuntimeError("Expected one SDPA call per actual packed query row")
                    return output
                finally:
                    current.clear()

            attention._dense_attention = MethodType(dense, attention)

            def layer_hook(module, args, output, *, layer_index=index):
                records["layers"][layer_index] = tuple(_cpu(tensor) for tensor in output)

            def attention_hook(module, args, output, *, layer_index=index):
                records["attention"][layer_index] = _cpu(output)

            handles.append(layer.register_forward_hook(layer_hook))
            handles.append(attention.register_forward_hook(attention_hook))
        with torch.inference_mode():
            result = model(input_ids, positions, metadata)
        records["output"] = _cpu(result)
        return records
    finally:
        functional.scaled_dot_product_attention = original_sdpa
        for handle in handles:
            handle.remove()
        for attention, previous in methods:
            if previous is None:
                del attention._dense_attention
            else:
                attention._dense_attention = previous


def _visible(record):
    import torch

    if record["mask"] is None:
        mask = torch.ones(record["keys"].shape[-2], dtype=torch.bool)
    elif record["mask"].dtype == torch.bool:
        mask = record["mask"].reshape(-1)
    else:
        mask = torch.isfinite(record["mask"].reshape(-1))
    indices = mask.nonzero().reshape(-1)
    return indices, record["keys"].index_select(-2, indices), record["values"].index_select(-2, indices)


def _compare_sdpa(original, bucket, *, atol, rtol):
    from examples.probe_npu_sdpa_padding import _comparison

    left_indices, left_keys, left_values = _visible(original)
    right_indices, right_keys, right_values = _visible(bucket)
    positions_equal = left_indices.tolist() == right_indices.tolist()
    left_slots = [original["physical_slots"][index] for index in left_indices.tolist()]
    right_slots = [bucket["physical_slots"][index] for index in right_indices.tolist()]
    result = {
        "original_context": original["context_len"],
        "bucket_context": bucket["context_len"],
        "visible_logical_positions_equal": positions_equal,
        "visible_physical_slots_equal": left_slots == right_slots,
        "original_visible_positions": left_indices.tolist(),
        "bucket_visible_positions": right_indices.tolist(),
        "query": _comparison(bucket["query"], original["query"], atol=atol, rtol=rtol),
        "sdpa_output": _comparison(bucket["output"], original["output"], atol=atol, rtol=rtol),
        "bf16_output": _comparison(bucket["output"].bfloat16(), original["output"].bfloat16(), atol=atol, rtol=rtol),
    }
    if left_keys.shape == right_keys.shape:
        result["visible_keys"] = _comparison(right_keys, left_keys, atol=atol, rtol=rtol)
        result["visible_values"] = _comparison(right_values, left_values, atol=atol, rtol=rtol)
    else:
        result["visible_keys"] = result["visible_values"] = {"exact_equal": False, "shape_mismatch": True}
    result["identical_visible_inputs"] = (
        positions_equal
        and left_slots == right_slots
        and all(result[key].get("exact_equal", False) for key in ("query", "visible_keys", "visible_values"))
    )
    return result


def _summarize(original, bucket, *, atol, rtol):
    from examples.probe_npu_sdpa_padding import _comparison

    layers = []
    first_different_sdpa = None
    first_different_bf16 = None
    first_different_inputs = None
    for index in sorted(original["layers"]):
        hidden, residual = original["layers"][index]
        bucket_hidden, bucket_residual = bucket["layers"][index]
        nodes = []
        for key in sorted(key for key in original["sdpa"] if key[0] == index):
            pair = _compare_sdpa(original["sdpa"][key], bucket["sdpa"][key], atol=atol, rtol=rtol)
            pair["row"] = key[1]
            nodes.append(pair)
            if first_different_sdpa is None and (
                not pair["identical_visible_inputs"] or not pair["sdpa_output"].get("exact_equal", False)
            ):
                first_different_sdpa = key
            if first_different_inputs is None and not pair["identical_visible_inputs"]:
                first_different_inputs = key
            if first_different_bf16 is None and not pair["bf16_output"].get("exact_equal", False):
                first_different_bf16 = key
        layers.append(
            {
                "layer": index,
                "hidden": _comparison(bucket_hidden, hidden, atol=atol, rtol=rtol),
                "residual": _comparison(bucket_residual, residual, atol=atol, rtol=rtol),
                "projected_attention": _comparison(
                    bucket["attention"][index], original["attention"][index], atol=atol, rtol=rtol
                ),
                "nodes": nodes,
            }
        )
    return {
        "output": _comparison(bucket["output"], original["output"], atol=atol, rtol=rtol),
        "layers": layers,
        "first_different_layer_output": next(
            (
                row["layer"]
                for row in layers
                if not row["hidden"].get("exact_equal", False) or not row["residual"].get("exact_equal", False)
            ),
            None,
        ),
        "first_different_sdpa": first_different_sdpa,
        "first_different_bf16_sdpa": first_different_bf16,
        "first_different_visible_inputs": first_different_inputs,
    }


def _load_single_model(args, replay=None):
    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoConfig, AutoTokenizer

    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
    from vllm_ascend.spec_decode.pearl.native_cache import NativePrefixCache
    from vllm_ascend.spec_decode.pearl.native_engine import NativePearlEngine
    from vllm_ascend.spec_decode.pearl.native_model import (
        NativeTPContext,
        build_native_model,
        load_native_model_weights,
    )

    torch.npu.set_device(args.device)
    torch.npu.config.allow_internal_format = True
    if args.hf32 != "default":
        torch.npu.matmul.allow_hf32 = args.hf32 == "on"
    init_device_properties_triton()
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config.pearl_use_production_rope = args.production_rope
    config.pearl_enable_mc2 = False
    model = build_native_model(
        config,
        NativeTPContext(group=None, rank=0, size=1, leader_rank=0),
        args.max_model_len,
        1,
        num_cache_blocks=max(16, len(replay["original_physical_pages"]) if replay else 16),
    )
    load_native_model_weights(model, args.model)
    model.eval()
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.model = model
    engine.device = torch.device(args.device)
    engine.config = SimpleNamespace(kvcache_block_size=model.block_size)
    engine.cache_allocation = None
    engine.cache_block_tables = None
    attention = model.layers[0].self_attn
    engine.prefix_cache = NativePrefixCache(
        attention.key_cache.shape[0], attention.blocks_per_sequence, attention.block_size
    )
    return engine, AutoTokenizer.from_pretrained(args.model, local_files_only=True)


def _prompt(args, tokenizer):
    if args.regression_json:
        saved = json.loads(Path(args.regression_json).read_text())
        tokens = list(saved["prompt_suites"]["synthetic_kv_boundary"]["prompt_token_ids"][args.request_index])
        if len(tokens) != args.prompt_length:
            raise ValueError("Saved regression input length differs from prompt-length; no silent reshaping")
        return tokens
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt_text}], tokenize=False, add_generation_prompt=True
    )
    tokens = tokenizer.encode(formatted)
    filler = tokenizer.encode(" context", add_special_tokens=False)[0]
    return ([filler] * max(0, args.prompt_length - len(tokens)) + tokens)[-args.prompt_length :]


def main(argv=None):
    args = _build_parser().parse_args(argv)
    _validate(args)
    import torch

    from vllm_ascend.spec_decode.pearl.native_graph import bucket_tree_attention_metadata
    from vllm_ascend.spec_decode.pearl.tree import build_tree_speculation_plan

    replay = torch.load(args.replay_snapshot, map_location="cpu", weights_only=True) if args.replay_snapshot else None
    engine, tokenizer = _load_single_model(args, replay)
    with torch.inference_mode():
        if replay is not None:
            input_ids, positions, metadata = restore_draft_replay_snapshot(engine.model, replay, device=engine.device)
            tokens, candidates = None, None
        else:
            tokens = _prompt(args, tokenizer)
            candidates = args.candidate_token_ids or tokenizer.encode(
                " The answer is 42. Let us reason carefully.", add_special_tokens=False
            )
            candidates = (candidates * (args.width * args.depth))[: args.width * args.depth]
            if any(not 0 <= token < engine.model.config.valid_vocab_size for token in [*tokens, *candidates]):
                raise ValueError("Input IDs exceed model vocabulary")
            engine._allocate_cache([tokens], enable_prefix_caching=False)
            for layer in engine.model.layers:
                if args.tail_init != "untouched":
                    fill = 0.0 if args.tail_init == "zero" else float("nan")
                    layer.self_attn.key_cache.fill_(fill)
                    layer.self_attn.value_cache.fill_(fill)
            if len(tokens) > 1:
                engine._run_packed_hidden(
                    tokens[:-1],
                    [0] * (len(tokens) - 1),
                    list(range(len(tokens) - 1)),
                    use_aclgraph=False,
                    use_fused_infer_attention=True,
                )
            plan = build_tree_speculation_plan(
                args.width, args.depth, len(tokens) - 1, args.max_model_len, device=engine.device
            )
            slots = plan.cache_positions.cpu().tolist()
            engine._ensure_cache_capacity([0] * len(slots), slots)
            input_ids, positions, metadata = engine.model.make_tree_attention_metadata(
                [plan],
                [tokens[-1]],
                [candidates],
                engine.cache_block_tables,
                sequence_ids=[0],
            )
        # Pin the diagnostic to the dense reference even if newer production
        # metadata automatically routes trees to FIA. A frozen PYTHONPATH is
        # still required to reproduce the historical SDPA implementation.
        dense_mask = metadata.attention_mask
        if dense_mask is None or dense_mask.ndim != 2:
            raise ValueError("Dense bucket diagnostic requires the independent packed 2-D mask")
        metadata = replace(
            metadata,
            use_fused_infer_attention=False,
            attention_mask=dense_mask,
        )
        bucket = bucket_tree_attention_metadata(metadata, block_size=engine.model.block_size)
        snapshot = _snapshot_cache(engine.model)
        _restore_cache(engine.model, snapshot)
        original = _capture_forward(engine.model, input_ids, positions, metadata)
        _restore_cache(engine.model, snapshot)
        padded = _capture_forward(engine.model, input_ids, positions, bucket)
        comparison = _summarize(original, padded, atol=args.atol, rtol=args.rtol)
        _restore_cache(engine.model, snapshot)
        repeated = _capture_forward(engine.model, input_ids, positions, metadata)
        repeat_comparison = _summarize(original, repeated, atol=args.atol, rtol=args.rtol)
    import inspect

    sources = {}
    for name, implementation in (
        ("native_model", type(engine.model)),
        ("native_graph", bucket_tree_attention_metadata),
    ):
        source = Path(inspect.getfile(implementation)).resolve()
        sources[name] = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    report = {
        "purpose": __doc__,
        "config": vars(args),
        "source_files": sources,
        "device_name": torch.npu.get_device_name(args.device),
        "model_layers": len(engine.model.layers),
        "prompt_token_ids": tokens,
        "candidate_token_ids": candidates,
        "input_ids": input_ids.cpu().tolist(),
        "positions": positions.cpu().tolist(),
        "input_sha256": hashlib.sha256(json.dumps(input_ids.cpu().tolist()).encode()).hexdigest(),
        "original_context_lens": metadata.context_lens.cpu().tolist(),
        "bucket_context_lens": bucket.context_lens.cpu().tolist(),
        "comparison": comparison,
        "repeat_original": repeat_comparison,
        "status": "complete",
        "note": "Complete means diagnostic execution finished, NOT numerical equivalence.",
    }
    key = comparison["first_different_sdpa"]
    if args.dump_first_inputs and key is not None:
        dump = Path(args.dump_first_inputs)
        dump.parent.mkdir(parents=True, exist_ok=True)
        cases = {}
        for reason in ("first_different_sdpa", "first_different_bf16_sdpa", "first_different_visible_inputs"):
            case_key = comparison[reason]
            if case_key is not None:
                cases[reason] = {
                    "layer": case_key[0],
                    "row": case_key[1],
                    "original": original["sdpa"][case_key],
                    "bucket": padded["sdpa"][case_key],
                }
        torch.save({"cases": cases, "config": vars(args)}, dump)
        report["first_input_dump"] = str(dump)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "first_different_layer": comparison["first_different_layer_output"],
                "first_different_sdpa": key,
                "output_error": comparison["output"],
                "restored_original_error": repeat_comparison["output"],
            }
        ),
        flush=True,
    )
    engine._release_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
