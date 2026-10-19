"""Independent full-mask/table/input checks using the retained CPU protocol fixture."""
import hashlib
import json
from pathlib import Path
import sys
import types

P = Path(__file__).resolve().parent
import runtime336
from geometry336 import LAYOUTS, cases, cost_arms


def main():
    from fixture326_HOST import build
    env = build()
    engine, torch = env['engine'], env['torch']
    for name in ('_allocate_cache', '_activate_cache_sequence', '_ensure_cache_capacity',
                 '_release_cache', '_cache_slot_mapping', '_prepare_attention_metadata'):
        method = getattr(engine, name)
        setattr(engine, name, types.MethodType(method.__func__, engine))
    from metadata336 import reference_frame, canonical
    from host_target_metadata import make_host_tree_attention_metadata
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan
    descriptions = cases()
    assert len(descriptions) == 36 and len(cost_arms()) == 4
    assert len({row['name'] for row in descriptions}) == 36
    covered = set()
    for row in descriptions:
        assert len(row['owners']) == len(set(row['owners'])) == row['N']
        assert row['query_count'] == row['N']*row['width'] <= row['M']
        covered.update(row['owners'])
    assert covered == set(range(32))
    checks = []
    for layout, (width, groups) in LAYOUTS.items():
        for real in (1, groups):
            for prefix in (97, 127, 128, 1505, 2555):
                for depth in (0, width-1):
                    engine._release_cache()
                    prompts = [[31+i]*(prefix if i == 0 else 97) for i in range(real)]
                    engine._allocate_cache(prompts, enable_prefix_caching=False, reserve_sequence_capacity=True)
                    for owner in range(463, 527):
                        engine._activate_cache_sequence(owner, [0], enable_prefix_caching=False)
                    owners = list(range(real))
                    frontiers = [len(v)+depth for v in prompts]
                    expected_positions = []
                    reference_digests = []
                    for future in (0, 17):
                        x, pos, meta = reference_frame(engine, layout=layout, owners=owners,
                            positions=frontiers, tokens=[41+i for i in owners], future=future)
                        roots = frontiers+[0]*(groups-real)
                        expected = torch.ones((groups, 1, 10, 4096), dtype=torch.bool)
                        for group, root in enumerate(roots):
                            expected[group, 0, 0, :root+1] = False
                            expected[group, 0, 1:width, 0] = False
                        assert torch.equal(meta.tree_attention_mask, expected)
                        assert meta.actual_seq_lengths_q == tuple(range(width, width*groups+1, width))
                        assert meta.sequence_lens == (2560,)*groups
                        expected_positions = [p+j for p in roots for j in range(width)]
                        assert pos.tolist() == expected_positions
                        expected_tokens = [token for i in range(groups) for token in
                            ([41+i]+[future]*(width-1) if i < real else [0]+[future]*(width-1))]
                        assert x.tolist() == expected_tokens
                        ids = owners+[463+j for j in range(groups-real)]
                        zero = engine._q4_zero_page336
                        original = engine.cache_allocation.block_tables
                        expected_tables = [[page if page >= 0 else zero for page in original[owner]] for owner in ids]
                        assert meta.request_block_tables.tolist() == expected_tables
                        slots = [original[owner][p//128]*128+p % 128
                                 for owner, root in zip(ids, roots, strict=True) for p in range(root, root+width)]
                        assert meta.slot_mapping.tolist() == slots
                        reference_digests.append(hashlib.sha256(expected.view(torch.uint8).numpy().tobytes()).hexdigest())
                    assert reference_digests[0] == reference_digests[1]
                    # Independently check candidate causal visibility under the
                    # same explicit FULL execution extent; no real key is removed.
                    ids = owners+[463+j for j in range(groups-real)]
                    roots = frontiers+[0]*(groups-real)
                    plans = [cached_cpu_tree_speculation_plan(1, width-1, p, 4096,
                             candidate_budget=width-1) for p in roots]
                    _, _, raw = make_host_tree_attention_metadata(engine, plans, [41]*groups,
                        [[43]*(width-1) for _ in range(groups)], sequence_ids=ids,
                        allow_causal_fast_path=False)
                    if max(roots)+width <= 2560:
                        meta = canonical(engine, raw, width=width, groups=groups)
                        expected = torch.ones((groups, 1, 10, 4096), dtype=torch.bool)
                        for group, root in enumerate(roots):
                            for query in range(width):
                                expected[group, 0, query, :root+query+1] = False
                        assert torch.equal(meta.tree_attention_mask, expected)
                    else:
                        try:
                            canonical(engine, raw, width=width, groups=groups)
                        except RuntimeError:
                            pass
                        else:
                            raise AssertionError('candidate context truncation was accepted')
                    checks.append(dict(layout=layout, real=real, prefix=prefix, depth=depth,
                                       future_mask_independent=True, exact_slots=True))
    assert len(checks) == 60 and 'torch_npu' not in sys.modules
    result = dict(status='passed_CPU_metadata_only', finite_cases=36,
        boundary_cases=len(checks), reference_metadata_variants=2*len(checks),
        all_masks_independently_rebuilt=True, all_input_positions_and_KV_slots_exact=True,
        all32_prefix_origins_covered=True, real_model_constructed=False,
        NPU_used=False, Torch_NPU_imported=False, real_model_math_qualified=False,
        Goodput=None, cases=checks)
    with (P/'HOST336_METADATA_RESULT.json').open('x') as saved:
        saved.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'cases'}))


if __name__ == '__main__':
    main()
