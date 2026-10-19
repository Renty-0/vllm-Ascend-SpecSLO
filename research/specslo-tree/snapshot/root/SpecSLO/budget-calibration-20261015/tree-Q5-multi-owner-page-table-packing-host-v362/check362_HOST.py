"""Original tree masks/KV addresses through multi-owner page-table packing."""
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types


P = Path(__file__).resolve().parent
B = P.parent
NATIVE = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')


def main():
    import torch
    torch.set_num_threads(1)
    output = P / 'HOST362_RESULT.json'
    assert not output.exists()
    for name in ('vllm_ascend', 'vllm_ascend.spec_decode', 'vllm_ascend.spec_decode.pearl'):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    name = 'vllm_ascend.spec_decode.pearl.tree'
    spec = importlib.util.spec_from_file_location(name, NATIVE / 'tree.py')
    tree = importlib.util.module_from_spec(spec)
    sys.modules[name] = tree
    spec.loader.exec_module(tree)
    for directory in ('tree-alternate-chain-target-host-v269',
                      'tree-static-q5-M120-node-math-v255',
                      'tree-canonical-M80-FULL9-full463-own-AR-v231'):
        sys.path.append(str(B / directory))
    from plan269 import make
    from padding231 import root_only
    from layout269 import layout
    from packing362 import Segment, pack
    count = dict(frames=0, original_real_queries=0, ordered_KV_address_rows=0,
                 multi_owner_groups=0, original_branch_copy_frames=0, rejected_controls=0)
    examples = []
    saved = None

    def fixture(budgets, prefixes, *, shared_prefix=False):
        assert len(budgets) == len(prefixes)
        plans = [make(budget, prefix, 0) for budget, prefix in zip(budgets, prefixes, strict=True)]
        ids = list(range(len(plans)))
        real = sum(budget + 1 for budget in budgets)
        remaining = 120 - real
        assert remaining >= 0
        while remaining:
            q = min(5, remaining)
            ids.append(64 + len(ids) - len(budgets))
            plans.append(root_only(4096) if q == 1 else make(q - 1, 0))
            remaining -= q
        tables = [[-1] * 32 for _ in range(160)]
        cursor = prefixes[0] // 128 if shared_prefix else 0
        for owner, item in zip(ids, plans, strict=True):
            pages = (max(item.cache_positions.tolist()) + 128) // 128
            shared = prefixes[owner] // 128 if shared_prefix and owner < len(budgets) else 0
            tables[owner][:shared] = list(range(shared))
            tables[owner][shared:pages] = list(range(cursor, cursor + pages - shared))
            cursor += pages - shared
        scratch = list(range(96, 160))
        for owner in scratch:
            tables[owner][0] = cursor
            cursor += 1
        assert cursor < 544 and not set(ids) & set(scratch)
        zero = cursor
        desc = layout(plans, ids, tables, scratch)
        segments = []
        if desc['enabled']:
            for group in desc['groups']:
                if group['original_rows'][0] >= real:
                    break
                segments.append(Segment(group['owner'], tuple(group['original_rows']), group['KV'],
                                        tuple(group['tables']), group['masks']))
        else:
            start = 0
            for owner, item in zip(ids[:len(budgets)], plans[:len(budgets)], strict=True):
                q = item.candidate_budget + 1
                segments.append(Segment(owner, tuple(range(start, start + q)),
                                        max(item.cache_positions.tolist()) + 1,
                                        tuple(tables[owner]), item.attention_mask))
                start += q
        return plans, ids, tables, zero, real, segments, desc

    def one(budgets, prefixes, extent):
        nonlocal saved
        plans, ids, tables, zero, real, segments, desc = fixture(budgets, prefixes)
        result = pack(segments, real_queries=real, zero_page=zero, extent=extent,
                      mask_capacity=11 if desc['enabled'] else 10)
        labels = torch.arange(544 * 128)
        if desc['enabled']:
            labels[torch.tensor(desc['dest'])] = labels[torch.tensor(desc['source'])].clone()
        original_reads, old_positions, old_slots = {}, [], []
        old = 0
        for owner, item in zip(ids, plans, strict=True):
            for local in range(item.candidate_budget + 1):
                if owner < len(budgets):
                    visible = torch.nonzero(~item.attention_mask[local], as_tuple=False).flatten().tolist()
                    original_reads[old] = [tables[owner][pos // 128] * 128 + pos % 128 for pos in visible]
                old_positions.append(item.positions[local].item())
                pos = item.cache_positions[local].item()
                old_slots.append(tables[owner][pos // 128] * 128 + pos % 128)
                old += 1
        assert old == 120 and len(set(old_slots)) == 120
        for move in result['real_moves']:
            group, query = move['physical_group'], move['physical_query']
            visible = torch.nonzero(~result['mask'][group, 0, query], as_tuple=False).flatten().tolist()
            table = result['tables'][group].tolist()
            actual = [table[pos // 128] * 128 + pos % 128 for pos in visible]
            assert labels[actual].tolist() == original_reads[move['old_row']]
            count['ordered_KV_address_rows'] += len(actual)
        index = torch.tensor(result['physical_to_original'])
        inverse = torch.tensor(result['original_to_physical'])
        tokens = torch.arange(120) + 1000
        positions = torch.tensor(old_positions)
        slots = torch.tensor(old_slots)
        for value in (tokens, positions, slots):
            assert torch.equal(value.index_select(0, index).index_select(0, inverse), value)
        assert result['actual_seq_lengths_q'] == tuple(range(5, 121, 5))
        assert result['sequence_lens'] == (extent,) * 24
        assert result['used_groups'] <= 24 and result['actual_real_owners'] == len(budgets)
        assert sorted(result['physical_to_original']) == list(range(120))
        count['frames'] += 1
        count['original_real_queries'] += real
        count['multi_owner_groups'] += result['multi_owner_groups']
        count['original_branch_copy_frames'] += desc['enabled']
        if len(budgets) >= 31:
            examples.append(dict(N=len(budgets), budgets=budgets, candidates=sum(budgets),
                                 real_queries=real, extent=extent, source_segments=len(segments),
                                 physical_groups=result['used_groups'], multi_owner_groups=result['multi_owner_groups']))
        saved = segments, real, zero

    patterns = ([4] * 16 + [1] * 8 + [2] * 8,
                [8] + [4] * 14 + [1] * 8 + [2] * 8,
                [5] * 12 + [2] * 8 + [1] * 12)
    for prefix in (0, 1, 126, 127, 128, 129):
        for budgets in patterns:
            for extent in (2560, 4096):
                one(budgets, [prefix] * len(budgets), extent)
    for n in (1, 4, 12, 24):
        for budget in (1, 2, 3, 4):
            for extent in (2560, 4096):
                one([budget] * n, [127] * n, extent)
    for prefix in (511, 512, 1505, 2555):
        for budget in (5, 6, 7, 8):
            one([budget], [prefix], 4096)
    for prefixes in ([1505, 127], [2555, 0], [0, 2555], [2555, 2555]):
        for extent in (2560, 4096):
            one([1, 2], prefixes, extent)

    def reject(segments, real, zero, **kwargs):
        try:
            pack(segments, real_queries=real, zero_page=zero, **kwargs)
        except (ValueError, RuntimeError):
            count['rejected_controls'] += 1
        else:
            raise AssertionError('invalid real grouping or visible-key truncation accepted')

    segments, real, zero = saved
    reject(segments, True, zero)
    reject(segments, real, zero, extent=128)
    reject(segments, real, zero, mask_capacity=5)
    reject(segments, real, segments[0].table[0])
    bad = list(segments)
    bad[0] = replace(bad[0], rows=bad[1].rows)
    reject(bad, real, zero)
    bad = list(segments)
    table = list(bad[0].table)
    table[0] = -1
    bad[0] = replace(bad[0], table=tuple(table))
    reject(bad, real, zero)
    bad = list(segments)
    bad[0] = replace(bad[0], KV=1)
    reject(bad, real, zero)
    _, _, _, zero, real, segments, _ = fixture([1] * 32, [2555] * 32, shared_prefix=True)
    reject(segments, real, zero, extent=4096)
    assert 'torch_npu' not in sys.modules
    result = dict(status='PASS_HOST_multi_owner_Q5_page_table_views_and_original_tree_KV_address_order',
                  counts=count, largest_original_N31_N32_examples=examples,
                  all_original120_input_RoPE_and_canonical_write_rows_bijective=True,
                  original_side_branch_copies_included=True, no_additional_prefix_copy_for_packing=True,
                  model_or_FIA_arithmetic_executed=False, NPU_used=False, serving_installed=False,
                  numerical_offset_invariance_qualified=False, full463_qualified=False,
                  Raw=None, Goodput=None,
                  source_sha256=hashlib.sha256((P / 'packing362.py').read_bytes()).hexdigest())
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'largest_original_N31_N32_examples'}))


if __name__ == '__main__':
    main()
