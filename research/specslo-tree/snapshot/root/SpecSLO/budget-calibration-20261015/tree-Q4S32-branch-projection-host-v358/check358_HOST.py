"""Independent address/mask/bijection checks; no model or NPU arithmetic."""
from dataclasses import dataclass, replace
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types


P = Path(__file__).resolve().parent
B = P.parent
NATIVE = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')


@dataclass
class Metadata:
    tree_attention: bool
    use_fused_infer_attention: bool
    tree_attention_mask: object
    request_block_tables: object
    block_tables: object
    sequence_lens: tuple
    actual_seq_lengths_q: tuple
    slot_mapping: object
    context_lens: object


def main():
    import torch
    torch.set_num_threads(1)
    output = P / 'HOST358_RESULT.json'
    assert not output.exists()
    for name in ('vllm_ascend', 'vllm_ascend.spec_decode', 'vllm_ascend.spec_decode.pearl'):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    name = 'vllm_ascend.spec_decode.pearl.tree'
    spec = importlib.util.spec_from_file_location(name, NATIVE / 'tree.py')
    tree = importlib.util.module_from_spec(spec)
    sys.modules[name] = tree
    spec.loader.exec_module(tree)
    sys.path[:0] = [str(B / 'tree-depth3-branch6-protocol-host-v357'),
                    str(B / 'tree-alternate-chain-target-host-v269'),
                    str(B / 'tree-canonical-M80-FULL9-full463-own-AR-v231')]
    from plan357 import make, resource_use, spare_side_owners
    from padding231 import root_only
    from layout358 import layout
    from projection358 import transform, plan, CACHE
    counts = dict(frames=0, branches=0, real_queries=0, independently_checked_visibility_rows=0,
                  negative_controls=0, identity_projection_frames=0)
    kept = None

    def one(budgets, prefix):
        nonlocal kept
        use = resource_use(budgets)
        plans = [make(budget, prefix).plan for budget in budgets]
        real_queries = sum(budget + 1 for budget in budgets)
        ids = list(range(len(plans)))
        remaining = 128 - real_queries
        while remaining:
            q = min(4, remaining)
            ids.append(64 + len(ids) - len(budgets))
            plans.append(root_only(4096) if q == 1 else make(q - 1, 0).plan)
            remaining -= q
        tables = [[-1] * 32 for _ in range(160)]
        cursor = 0
        for owner, item in zip(ids, plans, strict=True):
            pages = (max(item.cache_positions.tolist()) + 128) // 128
            tables[owner][:pages] = list(range(cursor, cursor + pages))
            cursor += pages
        scratch = list(range(96, 160))
        for owner in scratch:
            tables[owner][0] = cursor
            cursor += 1
        zero = cursor
        assert zero < 544 and len(set(ids)) == len(ids) and not set(ids) & set(scratch)
        desc = layout(plans, ids, tables, scratch)
        slots, positions, contexts, group_tables, masks, lengths, ends = [], [], [], [], [], [], []
        for owner, item in zip(ids, plans, strict=True):
            slots.extend(tables[owner][pos // 128] * 128 + pos % 128 for pos in item.cache_positions.tolist())
            positions.extend(item.positions.tolist())
            contexts.extend(pos + 1 for pos in item.positions.tolist())
            group_tables.append(tables[owner])
            q = item.candidate_budget + 1
            mask = torch.ones((1, 10, 4096), dtype=torch.bool)
            mask[0, :q] = item.attention_mask
            masks.append(mask)
            lengths.append(max(item.cache_positions.tolist()) + 1)
            ends.append(len(slots))
        if desc['enabled']:
            group_tables = desc['tables']
            mask = desc['mask']
            lengths = desc['sequence_lens']
            ends = desc['actual_seq_lengths_q']
            cache = torch.arange(544 * 128)
            cache[torch.tensor(desc['dest'])] = cache[torch.tensor(desc['source'])].clone()
            original_cache = torch.arange(544 * 128)
            old_group, old_begin = 0, 0
            for owner, item in zip(ids[:len(budgets)], plans[:len(budgets)], strict=True):
                branch = item.candidate_budget > 3
                segments = (list(range(4)), list(range(4, item.candidate_budget + 1))) if branch else (
                    list(range(item.candidate_budget + 1)),)
                for queries in segments:
                    for local, query in enumerate(queries):
                        visible = torch.nonzero(~item.attention_mask[query], as_tuple=False).flatten().tolist()
                        expected = [tables[owner][pos // 128] * 128 + pos % 128 for pos in visible]
                        seen = torch.nonzero(~mask[old_group, 0, local], as_tuple=False).flatten().tolist()
                        actual_slots = [group_tables[old_group][pos // 128] * 128 + pos % 128 for pos in seen]
                        assert cache[actual_slots].tolist() == original_cache[expected].tolist()
                        counts['independently_checked_visibility_rows'] += 1
                    old_group += 1
                    old_begin += len(queries)
            assert old_begin == real_queries
            assert len(set(desc['dest'])) == 1792
            copied = desc['real_copy_slots']
            assert not set(desc['dest'][:copied]) & set(desc['source'])
            if kept is None:
                kept = copy.deepcopy((plans, ids, tables, scratch))
        else:
            mask = torch.stack(masks)
        table_tensor = torch.tensor(group_tables, dtype=torch.int32)
        meta = Metadata(True, True, mask, table_tensor, table_tensor, tuple(lengths), tuple(ends),
                        torch.tensor(slots, dtype=torch.long), torch.tensor(contexts, dtype=torch.long))
        tokens = torch.arange(128, dtype=torch.long) + 1000
        pos = torch.tensor(positions, dtype=torch.long)
        out, out_pos, out_meta, inverse, mapping = transform(tokens, pos, meta, real_queries, zero)
        assert out.numel() == 128 and tuple(out_meta.tree_attention_mask.shape) == (32, 1, 10, 4096)
        assert out_meta.actual_seq_lengths_q == tuple(range(4, 129, 4))
        assert out_meta.sequence_lens == (2560,) * 32
        assert mapping['supported_primary_side_groups'] == use['primary_and_side_groups']
        assert torch.equal(out.index_select(0, inverse), tokens)
        assert torch.equal(out_pos.index_select(0, inverse), pos)
        assert torch.equal(out_meta.slot_mapping.index_select(0, inverse), meta.slot_mapping)
        assert torch.equal(out_meta.context_lens.index_select(0, inverse), meta.context_lens)
        for move in mapping['real_moves']:
            physical = move['new_row']
            assert torch.equal(out_meta.tree_attention_mask[physical // 4, 0, physical % 4],
                               meta.tree_attention_mask[move['old_group'], 0, move['old_query']])
            expected = meta.request_block_tables[move['old_group']]
            assert torch.equal(out_meta.request_block_tables[physical // 4],
                               torch.where(expected >= 0, expected, zero))
        for physical in mapping['phantom_rows']:
            row_mask = out_meta.tree_attention_mask[physical // 4, 0, physical % 4]
            assert torch.nonzero(~row_mask, as_tuple=False).flatten().tolist() == [0]
        assert sorted(mapping['physical_to_original']) == list(range(128))
        counts['frames'] += 1
        counts['branches'] += use['branches']
        counts['real_queries'] += real_queries
        counts['identity_projection_frames'] += mapping['physical_to_original'] == list(range(128))

    for n in range(1, 33):
        for side_count in range(spare_side_owners(n) + 1):
            one([6] * side_count + [3] * (n - side_count), 127)
    for prefix in (0, 1, 126, 128, 129, 1505, 2556):
        for budget in range(1, 7):
            for n in (1, 4, 12):
                one([budget] * n, prefix)

    def reject(function, *args, **kwargs):
        try:
            function(*args, **kwargs)
        except (ValueError, RuntimeError):
            counts['negative_controls'] += 1
        else:
            raise AssertionError('invalid geometry or alias accepted')

    args = kept
    reject(layout, *args, copy_slots=1)
    bad = copy.deepcopy(args)
    bad[2][bad[3][0]][0] = bad[2][bad[3][1]][0]
    reject(layout, *bad)
    bad = copy.deepcopy(args)
    bad[2][bad[3][0]][0] = bad[2][bad[1][0]][0]
    reject(layout, *bad)
    bad = copy.deepcopy(args)
    bad[0][0].attention_mask[-1, 0] = True
    reject(layout, *bad)
    reject(plan, [4] * 32, 128)
    reject(plan, list(range(1, 129)), 33)
    reject(plan, [5, 128], 5)
    reject(plan, tuple(range(4, 129, 4)), True)
    assert len(CACHE.entries) <= CACHE.maximum and 'torch_npu' not in sys.modules
    result = dict(status='PASS_HOST_Q4S32_primary3_alternate3_address_and_physical_projection',
                  counts=counts, every_original_physical_writer_preserved=True,
                  all_real_tree_visibility_and_private_alias_addresses_independently_checked=True,
                  NPU_used=False, serving_installed=False, neural_math_qualified=False,
                  eager_promotion_qualified=False, full463_qualified=False, Goodput=None,
                  source_hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (P / 'layout358.py', P / 'projection358.py',
                                              B / 'tree-depth3-branch6-protocol-host-v357/plan357.py',
                                              NATIVE / 'tree.py')})
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
