"""Exercise private metadata/observer callbacks on real CPU metadata helpers."""
import json
from pathlib import Path
import sys
import types
import runtime336

P = Path(__file__).resolve().parent


def main():
    from fixture326_HOST import build
    env = build()
    engine, torch = env['engine'], env['torch']
    for name in ('_allocate_cache', '_activate_cache_sequence', '_ensure_cache_capacity',
                 '_release_cache', '_cache_slot_mapping', '_prepare_attention_metadata'):
        method = getattr(engine, name)
        setattr(engine, name, types.MethodType(method.__func__, engine))
    from host_target_metadata import make_host_tree_attention_metadata
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan
    from metadata336 import install
    from geometry336 import LAYOUTS
    engine._release_cache()
    engine._allocate_cache([[31]*2555], enable_prefix_caching=False, reserve_sequence_capacity=True)
    for owner in range(463, 527):
        engine._activate_cache_sequence(owner, [0], enable_prefix_caching=False)
    engine.is_draft = False
    model = engine.model
    model._mc2_chained_row_counts = {96, 120, 128}
    model._mc2_deferred_row_counts = set()
    calls = []

    def original_make(this, plans, roots, draft, block_tables, *args, **kwargs):
        assert this is model and kwargs['allow_causal_fast_path'] is False
        calls.append(len(plans))
        return make_host_tree_attention_metadata(engine, plans, roots, draft, *args, **kwargs)

    model.make_tree_attention_metadata = types.MethodType(original_make, model)
    engine.graph_runner.last_target_execution = types.SimpleNamespace(
        used_aclgraph=True, mode='HOST-fixture-only', fallback_reason=None)
    engine.graph_runner.run_target_greedy = lambda xs, ps, ms, vocabulary_size: (torch.zeros_like(xs[0]),)
    activation = install(engine)
    assert activation['Native_flags_assigned'] is False
    for name, (width, groups) in LAYOUTS.items():
        engine._q4_layout336 = name
        ids = [0]+[463+j for j in range(groups-1)]
        for owner in ids:
            engine._ensure_cache_capacity([owner], [2555+width-1 if owner == 0 else width-1])
        plans = [cached_cpu_tree_speculation_plan(1, width-1, 2555 if owner == 0 else 0,
                 4096, candidate_budget=width-1) for owner in ids]
        x, pos, meta = model.make_tree_attention_metadata(plans, [41]*groups,
            [[43]*(width-1) for _ in ids], engine.cache_allocation.block_tables,
            sequence_ids=ids, allow_causal_fast_path=True)
        assert x.numel() == width*groups and tuple(meta.tree_attention_mask.shape) == (groups, 1, 10, 4096)
        output = engine.graph_runner.run_target_greedy([x], [pos], [meta], 100)
        assert output[0].numel() == x.numel()
        assert engine._q4_last_inputs336[2] is meta
        assert engine._q4_native_calls336[-1]['layout'] == name
    assert calls == [24, 32, 24] and 'torch_npu' not in sys.modules
    result = dict(status='passed_CPU_callback_protocol', actual_CPU_metadata_layouts=3,
        force_FULL_argument_checked=True, actual_input_identity_preserved=True,
        observer_fixture_is_not_graph_evidence=True, real_model_constructed=False,
        NPU_used=False, real_model_math_qualified=False)
    with (P/'HOST336_CALLBACK_RESULT.json').open('x') as saved:
        saved.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
