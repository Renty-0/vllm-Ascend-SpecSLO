"""Real draft-only frontier fusion compared with the installed optimized baseline."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
MODEL = Path('/root/data/shared-models/Qwen3-0.6B')
FIXTURE = BASE / 'tree-M160-S32-N32-target-math-cost-v326/FIXTURE326.json'
BUDGETS = (1, 2, 3, 4, 5)

def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT374: ' + message)


def configurations():
    return tuple([(n, 0, budget) for n in range(1, 33) for budget in BUDGETS]
                 + [(1, origin, budget) for origin in (1, 2) for budget in BUDGETS])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    need(not output.exists(), 'fresh independent output')
    sys.path[:0] = [str(BASE / name) for name in (
        'tree-primary4-top2-chain-device-v365', 'tree-fastv5-eager-frontier-chain-host-v373',
        'tree-M160-S32-N32-target-math-cost-v326', 'tree-high-normal-cap4-native-families-v148')]
    import torch
    import torch_npu
    import torch.distributed as dist
    from transformers import AutoConfig
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
    from vllm_ascend.spec_decode.pearl import native_engine as native
    from vllm_ascend.spec_decode.pearl.native_model import NativeTPContext, build_native_model, load_native_model_weights
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan
    from probe365 import make_engine, compare_kv
    from forward373 import build
    from bank326 import allocate
    from dual_bank_v146 import prefix_evidence
    need(not dist.is_initialized(), 'independent TP1 draft; no target or HCCL constructor')
    need(torch.npu.device_count() == 4, 'original owned four-card visible mapping')
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    init_device_properties_triton()
    config = AutoConfig.from_pretrained(str(MODEL), local_files_only=True)
    need(getattr(config, 'quantization_config', None) is None, 'original nonquantized model')
    config.torch_dtype = torch.bfloat16
    model = build_native_model(config, NativeTPContext(None, 0, 1, 0),
        max_model_len=4096, max_num_seqs=527, block_size=128, num_cache_blocks=544)
    load_native_model_weights(model, str(MODEL))
    model.eval()
    need(all(p.dtype == torch.bfloat16 for p in model.parameters() if p.is_floating_point()), 'all BF16 parameters')
    import root_seed, parallel_steps, tree_target_replay
    from vllm_ascend.spec_decode.pearl import native_graph
    tree_target_replay.install(native_graph)
    root_seed.install(native)
    parallel_steps.install(native_graph)
    engine = make_engine(torch, native, model)
    engine.config.enable_prefix_caching = False
    engine.config.profile_host_decode_steps = 0
    engine.graph_runner._dynamic_tree_target_replay_first = False
    engine._tree_eager_scratch_plan = native.NativePearlEngine._tree_eager_scratch_plan
    original = native.NativePearlEngine.draft_tree_forward
    proposed, state = build(original, engine, native)
    prefixes = json.loads(FIXTURE.read_text())['prefixes']
    prompts, allocation = allocate(engine, prefixes)
    roots = []
    rows, costs = [], []
    with torch.inference_mode():
        for owner, prefix in enumerate(prompts):
            pos, meta = engine._prepare_attention_metadata([owner] * len(prefix), list(range(len(prefix))), True)
            hidden = model(torch.tensor(prefix, dtype=torch.long, device=engine.device), pos, meta)
            need(bool(torch.isfinite(hidden).all()), 'all independently prefilled hidden finite')
            roots.append(int(model.compute_greedy_tokens(hidden[-1:], 151936).item()))
        need(roots[:32] == roots[32:64] == roots[64:96], 'three independent bank roots exact')
        before = [prefix_evidence(engine, {i: prompts[i]}) for i in range(96)]
        for i in range(32):
            need(before[i]['KV'] == before[i + 32]['KV'] == before[i + 64]['KV'], 'all56 prefix tensors exact')

        def frame(n, origin, budget, phase):
            indices = list(range(origin, origin + n))
            roots_a = [roots[i] for i in indices]
            old_plans = [cached_cpu_tree_speculation_plan(2, 4, len(prefixes[i]), 4096,
                         candidate_budget=4) for i in indices]
            if phase == 'B':
                for offset in (0, 32):
                    original(engine, old_plans, roots_a, [i + offset for i in indices])
            extra = int(phase == 'B')
            base_roots = [(root + 1) % 151936 if extra else root for root in roots_a]
            parents = [cached_cpu_tree_speculation_plan(2, 4, len(prefixes[i]) + extra,
                       4096, candidate_budget=4) for i in indices]
            parent_outputs = {offset: original(engine, parents, base_roots, [i + offset for i in indices])
                              for offset in (0, 32)}
            need(parent_outputs[0]['draft_token_ids'] == parent_outputs[32]['draft_token_ids'],
                 'independent actual FastV5 parent paths exact')
            plans = [cached_cpu_tree_speculation_plan(2, 4, int(parent.prefix_len) + 5,
                     4096, candidate_budget=budget) for parent in parents]
            sources = {offset: {i + offset: (parent, parent_outputs[offset]['draft_token_ids'][j])
                                 for j, (i, parent) in enumerate(zip(indices, parents, strict=True))}
                       for offset in (0, 32)}
            return indices, plans, base_roots, parents, sources

        def invoke(function, prepared, offset):
            indices, plans, base_roots, parents, sources = prepared
            return function(engine, plans, base_roots, [i + offset for i in indices],
                            eager_parent_sources=sources[offset])

        def cache_indices(prepared, offset, budget):
            indices, plans, base_roots, parents, sources = prepared
            values = []
            for i, parent in zip(indices, parents, strict=True):
                start = int(parent.prefix_len)
                positions = list(range(start, start + 5 + min(budget, 4)))
                values.extend(engine._cache_slot_mapping([i + offset] * len(positions), positions))
            return torch.tensor(values, dtype=torch.long, device=engine.device)

        for n, origin, budget in configurations():
            restored = None
            for phase in ('A1', 'B', 'A2'):
                prepared = frame(n, origin, budget, phase)
                reference = invoke(original, prepared, 32)
                before_calls = len(state.calls)
                candidate = invoke(proposed, prepared, 0)
                need(len(state.calls) == before_calls + 1, 'one actual fused frontier/chain call')
                for key in ('draft_token_ids', 'root_token_ids', 'num_draft_tokens', 'query_count',
                            'graph_padding_query_count', 'materialization_deferred', 'cache_slot_mapping'):
                    need(candidate[key] == reference[key], 'original optimized field exact: ' + key)
                need(candidate['eager_frontier_tokens'] == {key - 32: value for key, value in reference['eager_frontier_tokens'].items()},
                     'original optimized frontier roots exact')
                need(torch.equal(candidate['parent_indices'], reference['parent_indices']), 'original parent IDs exact')
                need(torch.equal(candidate['draft_confidence'].view(torch.int32), reference['draft_confidence'].view(torch.int32)),
                     'original encoded mean confidence bits')
                for a, b in zip(candidate['node_confidences'], reference['node_confidences'], strict=True):
                    need(torch.equal(a.view(torch.int32), b.view(torch.int32)), 'original optimized node confidence bits')
                need(reference['model_calls'] == reference['graph_calls'] == 2, 'actual optimized two submissions')
                need(candidate['model_calls'] == candidate['graph_calls'] == 1, 'one actual fused submission')
                indices = prepared[0]
                ledger = engine._dynamic_tree_known_KV
                need(all(ledger[i] == ledger[i + 32] for i in indices), 'same original canonical KV ledger')
                kv = compare_kv(torch, engine, cache_indices(prepared, 0, budget), cache_indices(prepared, 32, budget))
                identity = (candidate['draft_token_ids'], candidate['root_token_ids'], [v.tolist() for v in candidate['node_confidences']])
                if phase == 'A1': restored = identity
                if phase == 'A2': need(identity == restored, 'original optimized token/root/confidence ABA')
                row = dict(N=n, origin=origin, budget=budget, phase=phase,
                    chain_steps=min(budget, 4), all_tokens_and_roots_exact=True,
                    all_original_confidence_bits_exact=True, canonical_ledger_equal=True,
                    all_KV_tensors=kv, original_graph_submissions=2, fused_graph_submissions=1,
                    physical_queries_equal=True, actual=state.calls[-1],
                    A2_restored=(True if phase == 'A2' else None))
                rows.append(row)
                with (output.parent / 'math374.jsonl').open('a') as saved:
                    saved.write(json.dumps(row) + '\n')
                print(json.dumps(dict(completed_math=len(rows), N=n, origin=origin, budget=budget, phase=phase)), flush=True)
        print(json.dumps(dict(stage='all510_actual_optimized_stack_math_passed_start_balanced_cost')), flush=True)
        for n in (4, 12, 24, 32):
            for budget in (4, 5):
                prepared = frame(n, 0, budget, 'A1')
                operations = dict(optimized_existing=lambda: invoke(original, prepared, 32),
                                  fused_frontier_chain=lambda: invoke(proposed, prepared, 0))
                for operation in operations.values():
                    for _ in range(3): operation()
                for pair in range(20):
                    names = ('optimized_existing', 'fused_frontier_chain') if pair % 2 == 0 else ('fused_frontier_chain', 'optimized_existing')
                    for name in names:
                        torch.npu.synchronize()
                        began = time.perf_counter()
                        actual = operations[name]()
                        torch.npu.synchronize()
                        elapsed = time.perf_counter() - began
                        need(actual['graph_calls'] == (2 if name == 'optimized_existing' else 1), 'actual optimized timing graph count')
                        need(math.isfinite(elapsed) and elapsed > 0, 'finite measured duration')
                        costs.append(dict(N=n, budget=budget, pair=pair, operation=name, seconds=elapsed))
                print(json.dumps(dict(cost_N=n, budget=budget, completed_calls=len(costs))), flush=True)
        after = [prefix_evidence(engine, {i: prompts[i]}) for i in range(96)]
        need(after == before, 'all96 independent original committed prefix KV protected')
        graph = engine.graph_runner.graph_qualification_status()
        need(graph['unvalidated_entries'] == graph['disabled_entries'] == 0, 'all original Native graph checks intact')
        summary = {f'N{n}_B{budget}': {name: statistics.median([r['seconds'] for r in costs
                    if r['N'] == n and r['budget'] == budget and r['operation'] == name])
                    for name in ('optimized_existing', 'fused_frontier_chain')}
                    for n in (4, 12, 24, 32) for budget in (4, 5)}
        value = dict(schema='ROOT374.actual-FastV5-all-eager-frontier-chain-component.v1', status='complete',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            models_loaded=['Qwen3-0.6B'], target_model_loaded=False, TP=1, dtype='BF16', quantized=False,
            comparison_baseline='actual installed RootSeed over FastV5, not bare Native',
            independent_prefills=96, layers=len(model.layers), allocation=allocation,
            math_rows=rows, costs=costs, median_seconds=summary, graph_stats=graph,
            all_committed_prefix_KV_protected=True, native_qualification_bodies_unchanged=True,
            original_confidence_encoding_retained=True, model_weights_or_precision_changed=False,
            serving_installed=False, Raw=None, Goodput=None, adopted=False, goal_achieved=False)
        engine._release_cache()
        with output.open('x') as saved:
            json.dump(value, saved, indent=2)
            saved.write('\n')
        print(json.dumps(dict(status='complete', median_seconds=summary)), flush=True)


if __name__ == '__main__':
    main()
