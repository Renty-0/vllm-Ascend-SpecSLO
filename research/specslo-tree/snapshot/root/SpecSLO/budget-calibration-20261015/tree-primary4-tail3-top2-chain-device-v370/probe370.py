"""Real draft-only first/catch-up step plus three-step tail qualification."""
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
PROFILES = ('none', 'staggered', 'all4')


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT370: ' + message)


def configurations():
    return tuple((n, origin, profile) for n in (1, 4, 12, 24, 32)
                 for origin in ((0,) if n > 16 else (0, 8, 16))
                 for profile in PROFILES)


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

        def frame(n, origin, profile, phase):
            indices = list(range(origin, origin + n))
            bootstrap_plans = [cached_cpu_tree_speculation_plan(2, 4, len(prefixes[i]), 4096,
                                candidate_budget=4) for i in indices]
            base_roots = [roots[i] if phase != 'B' else (roots[i] + 1) % 151936 for i in indices]
            bootstrap = original(engine, bootstrap_plans, base_roots, [i + 64 for i in indices])
            need(bootstrap['model_calls'] == bootstrap['graph_calls'] == 4, 'genuine original bootstrap primary paths')
            count = [0 if profile == 'none' else (4 if profile == 'all4' else j % 5) for j in range(n)]
            paths = [[root] + bootstrap['draft_token_ids'][j][:count[j]] for j, root in enumerate(base_roots)]
            plans = [cached_cpu_tree_speculation_plan(2, 4, len(prefixes[i]) + k, 4096,
                        candidate_budget=1 + i % 8) for i, k in zip(indices, count, strict=True)]
            return indices, plans, paths, count

        def invoke(function, indices, plans, paths, offset):
            return function(engine, plans, [p[-1] for p in paths], [i + offset for i in indices],
                            committed_catchup_token_ids=paths)

        def query_slots(indices, count, offset):
            result = []
            for i, k in zip(indices, count, strict=True):
                positions = list(range(len(prefixes[i]), len(prefixes[i]) + k + 4))
                result.extend(engine._cache_slot_mapping([i + offset] * len(positions), positions))
            return torch.tensor(result, dtype=torch.long, device=engine.device)

        for n, origin, profile in configurations():
            restored = None
            for phase in ('A1', 'B', 'A2'):
                indices, plans, paths, count = frame(n, origin, profile, phase)
                reference = invoke(original, indices, plans, paths, 32)
                start_calls = len(state.calls)
                candidate = invoke(proposed, indices, plans, paths, 0)
                need(len(state.calls) == start_calls + 1, 'one actual tail3 submission per original round')
                detail = state.calls[-1]
                for key in ('draft_token_ids', 'root_token_ids', 'eager_frontier_tokens', 'num_draft_tokens',
                            'query_count', 'graph_padding_query_count', 'materialization_deferred', 'cache_slot_mapping'):
                    need(candidate[key] == reference[key], 'original field exact: ' + key)
                need(torch.equal(candidate['parent_indices'], reference['parent_indices']), 'all original parent IDs')
                need(torch.equal(candidate['draft_confidence'].view(torch.int32), reference['draft_confidence'].view(torch.int32)),
                     'original FP32 mean confidence bits')
                for a, b in zip(candidate['node_confidences'], reference['node_confidences'], strict=True):
                    need(torch.equal(a.view(torch.int32), b.view(torch.int32)), 'all8 FP32 node probability bits')
                need(reference['model_calls'] == reference['graph_calls'] == 4, 'four actual original graph submissions')
                need(candidate['model_calls'] == candidate['graph_calls'] == 2, 'first graph plus one actual tail graph')
                need(detail['actual_model_steps'] == 3 and detail['final_leaf_KV_materialized'] is False,
                     'same four actual steps with original deferred leaf policy')
                kv = compare_kv(torch, engine, query_slots(indices, count, 0), query_slots(indices, count, 32))
                identity = (candidate['draft_token_ids'], [v.tolist() for v in candidate['node_confidences']])
                if phase == 'A1': restored = identity
                if phase == 'A2': need(identity == restored, 'raw8 tokens and probabilities ABA restored')
                row = dict(N=n, origin=origin, profile=profile, phase=phase,
                    catchup_lengths=[k + 1 for k in count], all8_tokens_exact=True,
                    all8_FP32_probabilities_bitwise=True, mean_confidence_bitwise=True,
                    all_KV_tensors=kv, original_graph_calls=4, candidate_graph_calls=2,
                    original_actual_model_steps=4, candidate_actual_model_steps=4,
                    physical_queries_equal=True, query_count=candidate['query_count'],
                    prefix_preserved=True, tail=detail)
                rows.append(row)
                with (output.parent / 'math370.jsonl').open('a') as saved:
                    saved.write(json.dumps(row) + '\n')
                print(json.dumps(dict(completed_math=len(rows), N=n, origin=origin, profile=profile, phase=phase)), flush=True)
        print(json.dumps(dict(stage='all99_math_passed_start_balanced_cost', no_concurrent_HOST_load_requested=True)), flush=True)
        for n in (4, 12, 24, 32):
            for profile in ('none', 'staggered'):
                indices, plans, paths, count = frame(n, 0, profile, 'A1')
                operations = dict(generic4=lambda: invoke(original, indices, plans, paths, 32),
                                  first_plus_tail3=lambda: invoke(proposed, indices, plans, paths, 0))
                for mode in operations:
                    for _ in range(3): operations[mode]()
                for pair in range(20):
                    modes = ('generic4', 'first_plus_tail3') if pair % 2 == 0 else ('first_plus_tail3', 'generic4')
                    for mode in modes:
                        torch.npu.synchronize()
                        began = time.perf_counter()
                        actual = operations[mode]()
                        torch.npu.synchronize()
                        elapsed = time.perf_counter() - began
                        need(actual['graph_calls'] == (4 if mode == 'generic4' else 2), 'actual timing graph counts')
                        need(math.isfinite(elapsed) and elapsed > 0, 'finite measured component wall time')
                        costs.append(dict(N=n, profile=profile, pair=pair, mode=mode, seconds=elapsed))
                print(json.dumps(dict(cost_N=n, profile=profile, completed_calls=len(costs))), flush=True)
        after = [prefix_evidence(engine, {i: prompts[i]}) for i in range(96)]
        need(after == before, 'all96 original committed prefix KV protected after math/cost')
        graph = engine.graph_runner.graph_qualification_status()
        need(graph['unvalidated_entries'] == graph['disabled_entries'] == 0, 'all resident original graph checks pass')
        summary = {f'N{n}_{profile}': {mode: statistics.median([r['seconds'] for r in costs
                    if r['N'] == n and r['profile'] == profile and r['mode'] == mode])
                    for mode in ('generic4', 'first_plus_tail3')}
                    for n in (4, 12, 24, 32) for profile in ('none', 'staggered')}
        value = dict(schema='ROOT370.real-draft-first-catchup-plus-tail3-component.v1', status='complete',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            models_loaded=['Qwen3-0.6B'], target_model_loaded=False, TP=1, dtype='BF16', quantized=False,
            independent_prefills=96, layers=len(model.layers), allocation=allocation,
            math_rows=rows, costs=costs, median_seconds=summary, graph_stats=graph,
            all_committed_prefix_KV_protected=True, native_qualification_bodies_unchanged=True,
            serving_installed=False, production_runtime_equivalence_claimed=False,
            Raw=None, Goodput=None, adopted=False, goal_achieved=False)
        engine._release_cache()
        with output.open('x') as saved:
            json.dump(value, saved, indent=2)
            saved.write('\n')
        print(json.dumps(dict(status='complete', median_seconds=summary)), flush=True)


if __name__ == '__main__':
    main()
