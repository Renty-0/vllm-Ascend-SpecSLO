"""Real 0.6B-only raw-tree primary-chain qualification; no serving metrics.

The ordinary Native tree method and its graph qualification bodies are retained.
Three independently prefilled cache banks compare generic four-step, chain4,
and chain4 plus an explicitly charged fifth KV-only step.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from types import MethodType, SimpleNamespace


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SOURCE = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust')
MODEL = Path('/root/data/shared-models/Qwen3-0.6B')
FIXTURE = BASE / 'tree-M160-S32-N32-target-math-cost-v326/FIXTURE326.json'
STEPS = 4
BANK_SIZE = 32
TIMED_PAIRS = 20


def require(value, message):
    if value is not True:
        raise RuntimeError('ROOT365: ' + message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records():
    # The first two prefixes span the original 1505/2555-token boundaries.
    return tuple((n, origin) for n in (1, 4, 12, 24, 32)
                 for origin in ((0,) if n > 16 else (0, 8, 16)))


def make_engine(torch, native, model):
    from vllm_ascend.spec_decode.pearl.native_cache import NativePrefixCache
    from vllm_ascend.spec_decode.pearl.native_graph import NativeACLGraphRunner
    engine = SimpleNamespace(
        device=torch.device('npu:0'), model=model,
        config=SimpleNamespace(kvcache_block_size=128, max_model_len=4096,
            max_num_seqs=64, enforce_eager=False, spec_rhythm_tree_width=2,
            spec_rhythm_tree_depth=4, spec_rhythm_stable_graphs=True),
        prefix_cache=NativePrefixCache(544, 32, 128),
        _cache_storage_sequence_capacity=527, _tree_cache_capacity=544 * 128,
        _tree_graph_scratch_sequence_ids=tuple(range(463, 527)),
        cache_allocation=None, cache_block_tables=None, is_draft=True, gamma=4,
        draft_vocab_size=151936, target_vocab_size=151936,
        topology=SimpleNamespace(draft_leader_rank=0),
        groups=SimpleNamespace(draft_group=None), _defer_tree_materialization=True)
    for name in ('_allocate_cache', '_activate_cache_sequence', '_ensure_cache_capacity',
                 '_release_cache', '_cache_slot_mapping', '_prepare_attention_metadata',
                 '_pack_tree_draft_level', '_pad_tree_draft_graph_requests', 'draft_tree_forward'):
        setattr(engine, name, MethodType(getattr(native.NativePearlEngine, name), engine))
    engine.graph_runner = NativeACLGraphRunner(model, enabled=True,
        max_graph_tokens=512, max_graph_entries=128, is_target_worker=False)
    return engine


def packed_frame(engine, plans, owners, roots, steps):
    positions, metadatas = [], []
    initial = None
    for step in range(steps):
        requests = [(p, owner, step - 1, root if step == 0 else 0)
                    for p, owner, root in zip(plans, owners, roots, strict=True)]
        padded, _ = engine._pad_tree_draft_graph_requests(requests)
        inputs, pos, meta = engine._pack_tree_draft_level(padded)
        if initial is None:
            initial = inputs
        require(inputs.numel() == initial.numel(), 'fixed original physical query extent')
        positions.append(pos)
        metadatas.append(meta)
    return initial, positions, metadatas


def run_candidate(engine, state, plans, owners, roots, final_kv_only):
    from spine364 import decode
    inputs, positions, metadata = packed_frame(
        engine, plans, owners, roots, 5 if final_kv_only else 4)
    result = state.run(inputs, positions, metadata, engine.draft_vocab_size,
        valid_row_count=inputs.numel(), logical_row_count=len(owners),
        final_kv_only=final_kv_only)
    execution = engine.graph_runner.last_draft_execution
    require(execution.used_aclgraph is True, 'actual original Native chain graph used')
    tokens, probabilities = decode(result[:len(owners)].detach().cpu(), engine.draft_vocab_size)
    return tokens, probabilities, dict(physical_queries_per_step=inputs.numel(),
        physical_model_queries=inputs.numel() * len(positions),
        real_model_queries=len(owners) * len(positions),
        original_execution=vars(execution), entry_key=list(engine.graph_runner.last_draft_entry_key))


def slots(torch, engine, owners, prefixes, count):
    values = []
    for owner, prefix in zip(owners, prefixes, strict=True):
        values.extend(engine._cache_slot_mapping([owner] * count, list(range(len(prefix), len(prefix) + count))))
    return torch.tensor(values, dtype=torch.long, device=engine.device)


def compare_kv(torch, engine, left, right):
    compared = 0
    for layer in engine.model.layers:
        for name in ('key_cache', 'value_cache'):
            view = getattr(layer.self_attn, name).flatten(0, 1)
            a = view.index_select(0, left)
            b = view.index_select(0, right)
            require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), 'finite all-layer KV')
            require(torch.equal(a.view(torch.int16), b.view(torch.int16)), 'all-layer raw BF16 KV equality')
            compared += 1
    return compared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    require(not output.exists(), 'fresh result path')
    sys.path[:0] = [str(BASE / 'tree-primary4-top2-chain-host-v364'),
                   str(BASE / 'tree-M160-S32-N32-target-math-cost-v326'),
                   str(BASE / 'tree-high-normal-cap4-native-families-v148')]
    import torch
    import torch_npu
    import torch.distributed as dist
    from transformers import AutoConfig
    from vllm_ascend.spec_decode.pearl import native_engine as native
    from vllm_ascend.spec_decode.pearl.native_model import (
        NativeTPContext, build_native_model, load_native_model_weights)
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan
    from runner364 import install
    from bank326 import allocate
    from dual_bank_v146 import prefix_evidence
    require(not dist.is_initialized(), 'no distributed or target-model constructor')
    require(os.environ.get('ASCEND_RT_VISIBLE_DEVICES') and torch.npu.device_count() == 4,
            'original owned four-card runtime mapping')
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    config = AutoConfig.from_pretrained(str(MODEL), local_files_only=True)
    require(getattr(config, 'quantization_config', None) is None, 'original unquantized draft model')
    config.torch_dtype = torch.bfloat16
    model = build_native_model(config, NativeTPContext(None, 0, 1, 0),
        max_model_len=4096, max_num_seqs=527, block_size=128, num_cache_blocks=544)
    load_native_model_weights(model, str(MODEL))
    model.eval()
    require(all(p.dtype == torch.bfloat16 for p in model.parameters() if p.is_floating_point()),
            'all real draft model parameters BF16')
    engine = make_engine(torch, native, model)
    state = install(engine.graph_runner)
    fixture = json.loads(FIXTURE.read_text())
    prefixes = fixture['prefixes']
    prompts, allocation = allocate(engine, prefixes)
    roots = []
    with torch.inference_mode():
        for owner, prefix in enumerate(prompts):
            positions, metadata = engine._prepare_attention_metadata(
                [owner] * len(prefix), list(range(len(prefix))), True)
            hidden = model(torch.tensor(prefix, dtype=torch.long, device=engine.device), positions, metadata)
            require(bool(torch.isfinite(hidden).all()), 'independently prefilled hidden finite')
            roots.append(int(model.compute_greedy_tokens(hidden[-1:], engine.draft_vocab_size).item()))
        require(roots[:32] == roots[32:64] == roots[64:96], 'independently prefilled triplet roots equal')
        initial_prefix = [prefix_evidence(engine, {i: prompts[i]}) for i in range(96)]
        for i in range(32):
            require(initial_prefix[i]['KV'] == initial_prefix[i + 32]['KV'] == initial_prefix[i + 64]['KV'],
                    'independent all-layer original prefix triplets equal')
        rows = []
        for n, origin in records():
            selected = list(range(origin, origin + n))
            chosen_prefixes = [prefixes[i] for i in selected]
            plans = [cached_cpu_tree_speculation_plan(2, 4, len(prefixes[i]), 4096,
                       candidate_budget=(4 if i % 2 else 8)) for i in selected]
            for phase in ('A1', 'B', 'A2'):
                chosen_roots = [roots[i] if phase != 'B' else (roots[i] + 1) % 151936 for i in selected]
                reference = engine.draft_tree_forward(plans, chosen_roots, [i + 32 for i in selected])
                require(reference['model_calls'] == reference['graph_calls'] == 4,
                        'original four actual per-depth model graphs')
                require(reference['materialization_deferred'] is True, 'same original deferred policy')
                expected_tokens = torch.tensor(reference['draft_token_ids'], dtype=torch.int32)
                expected_probabilities = torch.stack(reference['node_confidences'])
                details = []
                for final_kv, offset in ((False, 0), (True, 64)):
                    owners = [i + offset for i in selected]
                    tokens, probabilities, detail = run_candidate(
                        engine, state, plans, owners, chosen_roots, final_kv)
                    require(torch.equal(tokens, expected_tokens), 'all8 original raw tree tokens exact')
                    require(torch.equal(probabilities.view(torch.int32), expected_probabilities.view(torch.int32)),
                            'all8 original FP32 probabilities bitwise exact')
                    checked = compare_kv(torch, engine,
                        slots(torch, engine, owners, chosen_prefixes, 4),
                        slots(torch, engine, [i + 32 for i in selected], chosen_prefixes, 4))
                    if final_kv:
                        # Ordinary generic reference has not computed d4 KV.
                        # Independently materialize exactly that token, with no head.
                        leaf = expected_tokens[:, 3].tolist()
                        requests = [(p, i + 32, 3, token) for p, i, token in zip(plans, selected, leaf, strict=True)]
                        padded, _ = engine._pad_tree_draft_graph_requests(requests)
                        input_ids, pos, meta = engine._pack_tree_draft_level(padded)
                        model(input_ids, pos, meta)
                        checked = compare_kv(torch, engine,
                            slots(torch, engine, owners, chosen_prefixes, 5),
                            slots(torch, engine, [i + 32 for i in selected], chosen_prefixes, 5))
                    details.append(dict(final_kv_only=final_kv, all8_tokens_exact=True,
                        all8_FP32_probabilities_bitwise=True, all_KV_tensors=checked, **detail))
                row = dict(N=n, origin=origin, phase=phase, candidates=details)
                rows.append(row)
                with (output.parent / 'math365.jsonl').open('a') as saved:
                    saved.write(json.dumps(row) + '\n')
                print(json.dumps(dict(completed_math=len(rows), N=n, origin=origin, phase=phase)), flush=True)
        # Balanced same-instance timing, after all mathematical qualification.
        costs = []
        for n in (4, 12, 24, 32):
            selected = list(range(n))
            plans = [cached_cpu_tree_speculation_plan(2, 4, len(prefixes[i]), 4096, candidate_budget=4)
                     for i in selected]
            chosen_roots = roots[:n]
            def operation(mode):
                if mode == 'generic4':
                    r = engine.draft_tree_forward(plans, chosen_roots, [i + 32 for i in selected])
                    require(r['model_calls'] == r['graph_calls'] == 4, 'all generic timing calls actual graph')
                else:
                    final = mode == 'chain5'
                    run_candidate(engine, state, plans,
                        [i + (64 if final else 0) for i in selected], chosen_roots, final)
            for mode in ('generic4', 'chain4', 'chain5'):
                for _ in range(3):
                    operation(mode)
            for pair in range(TIMED_PAIRS):
                modes = ('generic4', 'chain4', 'chain5') if pair % 2 == 0 else ('chain5', 'chain4', 'generic4')
                for mode in modes:
                    torch.npu.synchronize()
                    began = time.perf_counter()
                    operation(mode)
                    torch.npu.synchronize()
                    elapsed = time.perf_counter() - began
                    require(math.isfinite(elapsed) and elapsed > 0, 'positive measured component wall time')
                    costs.append(dict(N=n, pair=pair, mode=mode, seconds=elapsed))
            print(json.dumps(dict(cost_N=n, completed_calls=len(costs))), flush=True)
        final_prefix = [prefix_evidence(engine, {i: prompts[i]}) for i in range(96)]
        require(final_prefix == initial_prefix, 'all96 committed prefixes protected after math and timings')
        summary = {f'N{n}': {mode: statistics.median([r['seconds'] for r in costs if r['N'] == n and r['mode'] == mode])
                            for mode in ('generic4', 'chain4', 'chain5')} for n in (4, 12, 24, 32)}
        value = dict(schema='ROOT365.real-draft-primary4-top2-chain-component.v1',
            status='complete', source_sha256=digest(__file__), model_path=str(MODEL.resolve()),
            models_loaded=['Qwen3-0.6B'], target_model_loaded=False, TP=1, dtype='BF16',
            quantized=False, independent_prefills=96, layers=len(model.layers), allocation=allocation,
            math_rows=rows, costs=costs, median_seconds=summary,
            all_committed_prefix_KV_protected=True, native_qualification_bodies_unchanged=True,
            graph_stats=engine.graph_runner.graph_qualification_status(),
            scope='component only; original per-depth generic path versus private chain4/chain5; no serving installation',
            production_runtime_equivalence_claimed=False, Raw=None, Goodput=None,
            adopted=False, goal_achieved=False)
        engine._release_cache()
        with output.open('x') as saved:
            json.dump(value, saved, indent=2)
            saved.write('\n')
        print(json.dumps(dict(status='complete', median_seconds=summary)), flush=True)


if __name__ == '__main__':
    main()
