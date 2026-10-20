"""CPU structural checks for actual M120/S32 projector, including mask ownership."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

import projection397 as candidate


def partition(budgets):
    sizes = [b + 1 for b in budgets]
    remaining = candidate.MATRIX_ROWS - sum(sizes)
    assert remaining >= 0
    while remaining:
        size = min(5, remaining)
        sizes.append(size)
        remaining -= size
    total = 0
    ends = []
    for size in sizes:
        total += size
        ends.append(total)
    return tuple(ends), sum(budgets) + len(budgets)


def require_rejected(ends, real_queries):
    try:
        candidate.plan(ends, real_queries)
    except ValueError:
        return
    raise AssertionError('invalid physical or candidate capacity admitted')


@dataclass
class Metadata:
    slot_mapping: object
    context_lens: object
    block_tables: object
    request_block_tables: object
    tree_attention_mask: object
    actual_seq_lengths_q: tuple
    sequence_lens: tuple
    tree_attention: bool = True
    use_fused_infer_attention: bool = True


def main():
    import torch
    torch.set_num_threads(1)
    rng = random.Random(0)
    vectors = []
    for n in range(1, 33):
        vectors.append([1] * n)
        vectors.append([3] * min(n, 24) + [2] * max(0, n - 24))
        for _ in range(40):
            budgets = [rng.choice((1, 2, 3)) for _ in range(n)]
            while budgets.count(3) > 24:
                budgets[budgets.index(3)] = 2
            assert sum(budgets) <= candidate.CANDIDATE_LIMIT
            rng.shuffle(budgets)
            vectors.append(budgets)
    fixed = None
    for budgets in vectors:
        ends, q = partition(budgets)
        p = candidate.plan(ends, q)
        assert p['real_N'] == len(budgets)
        assert p['selected_candidates'] == sum(budgets)
        assert sorted(p['physical_to_original']) == list(range(120))
        assert [p['physical_to_original'][v] for v in p['original_to_physical']] == list(range(120))
        assert len(p['real_moves']) == q
        assert len(p['phantom_rows']) == 120 - q
        assert p['production_permission'] is False
        if fixed is None:
            fixed = p['actual_seq_lengths_q']
        assert p['actual_seq_lengths_q'] == fixed
        for move in p['real_moves']:
            assert p['physical_group_sources'][move['new_group']] == move['old_group']
            assert move['new_query'] < candidate.SEGMENT_CAPACITIES[move['new_group']]
    # C88 by itself is insufficient:25 B3 owners exceed the24 q4 slots.
    ends, q = partition([3] * 25 + [1] * 7)
    assert q - 32 == 82
    require_rejected(ends, q)
    require_rejected((120,), 120)
    require_rejected((4, 4, 120), 4)
    require_rejected((5, 120), 5)
    require_rejected((2, 120), True)

    tensor_cases = 0
    for budgets in vectors[::21]:
        ends, q = partition(budgets)
        groups = len(ends)
        inputs = torch.arange(120, dtype=torch.long, device='cpu')
        positions = inputs + 512
        slots = inputs + 2048
        slots[q:] = -1
        context_lens = (inputs % 128 + 1).to(torch.int32)
        tables = torch.arange(groups * 32, dtype=torch.int32, device='cpu').reshape(groups, 32) % 500
        tables[:, -1] = -1
        mask = torch.ones((groups, 1, 10, 4096), dtype=torch.bool, device='cpu')
        for group in range(groups):
            for query in range(10):
                mask[group, 0, query, (group * 11 + query) % 128] = False
        metadata = Metadata(slots, context_lens, tables, tables, mask, ends, (128,) * groups)
        x, pos, transformed, inverse, p = candidate.transform(
            inputs, positions, metadata, q, zero_page=543, extent=512)
        assert x.device.type == pos.device.type == 'cpu'
        assert torch.equal(x.index_select(0, inverse), inputs)
        assert torch.equal(pos.index_select(0, inverse), positions)
        assert torch.equal(transformed.slot_mapping.index_select(0, inverse), slots)
        assert torch.equal(transformed.context_lens.index_select(0, inverse), context_lens)
        assert transformed.sequence_lens == (512,) * 32
        assert list(transformed.actual_seq_lengths_q) == fixed
        assert tuple(transformed.tree_attention_mask.shape) == (32, 1, 10, 4096)
        for move in p['real_moves']:
            g, j, old_g, old_j = (move[k] for k in ('new_group', 'new_query', 'old_group', 'old_query'))
            assert torch.equal(transformed.tree_attention_mask[g, 0, j], mask[old_g, 0, old_j])
        for row in p['phantom_rows']:
            v = transformed.tree_attention_mask[row['new_group'], 0, row['new_query']]
            assert not bool(v[0]) and bool(v[1:].all())
            assert int(transformed.slot_mapping[row['new_row']]) == -1
        for physical, original in enumerate(p['physical_group_sources']):
            wanted = (torch.full((32,), 543, dtype=torch.int32) if original is None
                      else torch.where(tables[original] >= 0, tables[original], 543))
            assert torch.equal(transformed.request_block_tables[physical], wanted)
        tensor_cases += 1
    # Saturated24 B3 +8 B2 needs no row permutation at all.
    ends, q = partition([3] * 24 + [2] * 8)
    assert q == 120
    assert candidate.plan(ends, q)['physical_to_original'] == list(range(120))
    result = dict(status='PASS_HOST397_FIXED_PARTITION_C88_INPUT_POSITION_KV_MASK_OWNERSHIP',
                  source_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
                  structural_cases=len(vectors), actual_CPU_tensor_cases=tensor_cases,
                  fixed_query_partition_ends=fixed, M=120, S=32, C=88,
                  global_active_limit_required=64, saturated_layout_identity=True,
                  invalid_C88_but_25_large_owners_rejected=True,
                  all120_rows_bijective=True, all_real_masks_and_page_owners_preserved=True,
                  original_padding_write_slots_preserved=True,
                  model_executed=False, NPU_used=False, runtime_implemented=False,
                  production_permission=False, Raw=None, Goodput=None)
    path = Path(__file__).with_name('HOST397_RESULT.json')
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'structural_cases',
                                          'actual_CPU_tensor_cases', 'NPU_used')}))


if __name__ == '__main__':
    main()
