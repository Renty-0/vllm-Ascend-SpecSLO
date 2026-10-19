"""Real draft-only mixed rolling-eager and all-N1..32 tail qualification."""
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
PROFILES = ('normal', 'mixed_eager', 'all_eager')


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT371: ' + message)


def configurations():
    return tuple([(n, 0, mode) for n in range(1, 33) for mode in PROFILES]
                 + [(1, origin, mode) for origin in (1, 2) for mode in PROFILES])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    need(not output.exists(), 'fresh independent output')
    sys.path[:0] = [str(BASE / name) for name in (
        'tree-primary4-top2-chain-device-v365', 'tree-primary4-tail3-top2-chain-host-v369',
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
    from forward369 import build
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
    engine = make_engine(torch, native, model)
    engine._tree_eager_scratch_plan = native.NativePearlEngine._tree_eager_scratch_plan
    original = native.NativePearlEngine.draft_tree_forward
    proposed, state = build(original, engine)
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

        def frame(n, origin, mode, phase):
            indices = list(range(origin, origin + n))
            roots_a = [roots[i] for i in indices]
            old_plans = [cached_cpu_tree_speculation_plan(2, 4, len(prefixes[i]), 4096,
                         candidate_budget=4) for i in indices]
            # B extends each independently materialized neural history by its
            # genuine first root; its changed positions force Native's own
            # changed-input qualification without assigning any graph flags.
            if phase == 'B':
                for offset in (0, 32):
                    original(engine, old_plans, roots_a, [i + offset for i in indices])
            extra = int(phase == 'B')
            base_roots = [(root + 1) % 151936 if extra else root for root in roots_a]
            parents = [cached_cpu_tree_speculation_plan(2, 4, len(prefixes[i]) + extra, 4096,
                       candidate_budget=4) for i in indices]
            parent_outputs = {offset: original(engine, parents, base_roots, [i + offset for i in indices])
                              for offset in (0, 32)}
            need(parent_outputs[0]['draft_token_ids'] == parent_outputs[32]['draft_token_ids'],
                 'independently generated real parent primary paths exact')
            selected = [j for j in range(n) if mode == 'all_eager' or (mode == 'mixed_eager' and j % 2 == 0)]
            plans = [cached_cpu_tree_speculation_plan(2, 4, int(parent.prefix_len) + (5 if j in selected else 0),
                      4096, candidate_budget=1 + indices[j] % 8) for j, parent in enumerate(parents)]
            work = [engine._tree_eager_scratch_plan(plan, parents[j]) if j in selected else plan
                    for j, plan in enumerate(plans)]
            sources = {offset: {indices[j] + offset: (parents[j], parent_outputs[offset]['draft_token_ids'][j])
                                 for j in selected} for offset in (0, 32)}
            return indices, plans, base_roots, parents, work, sources, selected

        def invoke(function, prepared, offset):
            indices, plans, base_roots, parents, work, sources, selected = prepared
            return function(engine, plans, base_roots, [i + offset for i in indices],
                            eager_parent_sources=sources[offset])

        def cache_indices(prepared, offset):
            indices, plans, base_roots, parents, work, sources, selected = prepared
            values = []
            for j, (i, parent, current) in enumerate(zip(indices, parents, work, strict=True)):
                positions = current.cache_positions.tolist()[:4]
                if j in selected:
                    positions = parent.cache_positions.tolist()[:5] + positions
                values.extend(engine._cache_slot_mapping([i + offset] * len(positions), positions))
            return torch.tensor(values, dtype=torch.long, device=engine.device)

        def check_mapping(result, prepared, offset):
            indices, plans, base_roots, parents, work, sources, selected = prepared
            if not selected:
                need(result['cache_slot_mapping'] is None, 'original normal deferred slot-map policy')
                return
            values = []
            for i, current in zip(indices, work, strict=True):
                positions = current.cache_positions.tolist()
                values.extend(engine._cache_slot_mapping([i + offset] * len(positions), positions))
            expected = torch.tensor(values, dtype=torch.int32, device=engine.device)
            need(torch.equal(result['cache_slot_mapping'], expected), 'original exact private eager promotion map')

        for n, origin, mode in configurations():
            restored = None
            for phase in ('A1', 'B', 'A2'):
                prepared = frame(n, origin, mode, phase)
                indices, plans, base_roots, parents, work, sources, selected = prepared
                reference = invoke(original, prepared, 32)
                start_calls = len(state.calls)
                candidate = invoke(proposed, prepared, 0)
                need(len(state.calls) == start_calls + 1, 'one actual three-step tail')
                detail = state.calls[-1]
                for key in ('draft_token_ids', 'root_token_ids', 'num_draft_tokens', 'query_count',
                            'graph_padding_query_count', 'materialization_deferred'):
                    need(candidate[key] == reference[key], 'original field exact: ' + key)
                need(candidate['eager_frontier_tokens'] == {key - 32: value for key, value in reference['eager_frontier_tokens'].items()},
                     'actual same independent eager frontier roots')
                need(torch.equal(candidate['parent_indices'], reference['parent_indices']), 'all original parent IDs')
                need(torch.equal(candidate['draft_confidence'].view(torch.int32), reference['draft_confidence'].view(torch.int32)),
                     'original FP32 mean confidence bits')
                for a, b in zip(candidate['node_confidences'], reference['node_confidences'], strict=True):
                    need(torch.equal(a.view(torch.int32), b.view(torch.int32)), 'all8 FP32 node probabilities')
                extra = bool(selected)
                need(reference['model_calls'] == reference['graph_calls'] == 4 + extra, 'all original actual graphs')
                need(candidate['model_calls'] == candidate['graph_calls'] == 2 + extra, 'original frontier/first plus actual tail graph')
                check_mapping(candidate, prepared, 0)
                check_mapping(reference, prepared, 32)
                kv = compare_kv(torch, engine, cache_indices(prepared, 0), cache_indices(prepared, 32))
                identity = (candidate['draft_token_ids'], [v.tolist() for v in candidate['node_confidences']])
                if phase == 'A1': restored = identity
                if phase == 'A2': need(identity == restored, 'original raw8/probabilities ABA restored')
                row = dict(N=n, origin=origin, mode=mode, phase=phase, eager_owners=len(selected),
                    all8_tokens_exact=True, all8_FP32_probabilities_bitwise=True,
                    mean_confidence_bitwise=True, original_eager_frontier_roots_exact=True,
                    eager_promotion_map_exact=True, all_KV_tensors=kv,
                    original_graph_calls=4 + extra, candidate_graph_calls=2 + extra,
                    original_actual_model_steps=4 + extra, candidate_actual_model_steps=4 + extra,
                    physical_queries_equal=True, query_count=candidate['query_count'], tail=detail,
                    A2_restored=(True if phase == 'A2' else None))
                rows.append(row)
                with (output.parent / 'math371.jsonl').open('a') as saved:
                    saved.write(json.dumps(row) + '\n')
                print(json.dumps(dict(completed_math=len(rows), N=n, origin=origin, mode=mode, phase=phase)), flush=True)
        print(json.dumps(dict(stage='all306_math_passed_start_balanced_cost')), flush=True)
        for n in (4, 12, 24, 32):
            for mode in ('mixed_eager', 'all_eager'):
                prepared = frame(n, 0, mode, 'A1')
                operations = dict(generic=lambda: invoke(original, prepared, 32),
                                  first_plus_tail3=lambda: invoke(proposed, prepared, 0))
                for operation in operations.values():
                    for _ in range(3): operation()
                for pair in range(20):
                    names = ('generic', 'first_plus_tail3') if pair % 2 == 0 else ('first_plus_tail3', 'generic')
                    for name in names:
                        torch.npu.synchronize()
                        began = time.perf_counter()
                        actual = operations[name]()
                        torch.npu.synchronize()
                        elapsed = time.perf_counter() - began
                        need(actual['graph_calls'] == (5 if name == 'generic' else 3), 'actual eager timing graph count')
                        need(math.isfinite(elapsed) and elapsed > 0, 'finite actual duration')
                        costs.append(dict(N=n, mode=mode, pair=pair, operation=name, seconds=elapsed))
                print(json.dumps(dict(cost_N=n, mode=mode, completed_calls=len(costs))), flush=True)
        after = [prefix_evidence(engine, {i: prompts[i]}) for i in range(96)]
        need(after == before, 'all96 original committed prefix KV protected')
        graph = engine.graph_runner.graph_qualification_status()
        need(graph['unvalidated_entries'] == graph['disabled_entries'] == 0, 'all resident Native qualification intact')
        summary = {f'N{n}_{mode}': {name: statistics.median([r['seconds'] for r in costs
                    if r['N'] == n and r['mode'] == mode and r['operation'] == name])
                    for name in ('generic', 'first_plus_tail3')}
                    for n in (4, 12, 24, 32) for mode in ('mixed_eager', 'all_eager')}
        value = dict(schema='ROOT371.real-draft-eager-allN-tail3-component.v1', status='complete',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            models_loaded=['Qwen3-0.6B'], target_model_loaded=False, TP=1, dtype='BF16', quantized=False,
            independent_prefills=96, layers=len(model.layers), allocation=allocation,
            math_rows=rows, costs=costs, median_seconds=summary, graph_stats=graph,
            all_committed_prefix_KV_protected=True, native_qualification_bodies_unchanged=True,
            all_logical_N_1_to32_exercised=True, normal_mixed_and_all_eager_exercised=True,
            serving_installed=False, Raw=None, Goodput=None, adopted=False, goal_achieved=False)
        engine._release_cache()
        with output.open('x') as saved:
            json.dump(value, saved, indent=2)
            saved.write('\n')
        print(json.dumps(dict(status='complete', median_seconds=summary)), flush=True)


if __name__ == '__main__':
    main()
