"""Physical M165 projector:33 q5 groups.

Every real input, RoPE position and physical KV write is retained. The entire
165-row permutation includes original padding rows. No real budget is changed.
"""
from pathlib import Path
import hashlib

def plan(ends, real_queries):
    if type(real_queries) is not int or not 1 <= real_queries <= 165:
        raise ValueError('original real prefix within165, under N33+C96')
    ends = tuple(ends)
    if not ends or ends[-1] != 165 or any((type(x) is not int for x in ends)):
        raise ValueError('complete original165 physical rows')
    groups = []
    begin = 0
    for group, end in enumerate(ends):
        if not begin < end <= 165:
            raise ValueError('strict original query partition')
        if begin < real_queries:
            if end > real_queries or end - begin > 5:
                raise ValueError('one real primary/side segment fits q5')
            groups.append((group, list(range(begin, end))))
        begin = end
    if real_queries not in ends or len(groups) > 33:
        raise ValueError('at most33 real primary/side groups and exact real boundary')
    unused = iter(range(real_queries, 165))
    order = []
    moves = []
    sources = []
    phantoms = []
    for physical, size in enumerate([5] * 33):
        if physical < len(groups):
            source, rows = groups[physical]
            sources.append(source)
            for q, old in enumerate(rows):
                moves.append(dict(old_row=old, new_row=len(order), old_group=source, old_query=q))
                order.append(old)
        else:
            rows = []
            sources.append(None)
        for q in range(len(rows), size):
            old = next(unused, None)
            if old is None:
                raise RuntimeError('all real groups fit without inventing physical rows')
            phantoms.append(len(order))
            order.append(old)
    if next(unused, None) is not None or sorted(order) != list(range(165)):
        raise RuntimeError('all165 original input/write rows form a bijection')
    inverse = [None] * 165
    for new, old in enumerate(order):
        inverse[old] = new
    if sorted((m['old_row'] for m in moves)) != list(range(real_queries)):
        raise RuntimeError('every real node maps exactly once')
    return dict(physical_to_original=order, original_to_physical=inverse, real_moves=moves, physical_group_sources=sources, phantom_rows=phantoms, actual_seq_lengths_q=list(range(5, 166, 5)), physical_queries=165, supported_primary_side_groups=len(groups), real_queries=real_queries)

def transform(inputs, positions, metadata, real_queries, zero_page, *, extent=2560, mask_capacity=10):
    import torch
    from dataclasses import replace
    from projection247 import CACHE
    if type(zero_page) is not int or not 0 <= zero_page < 544:
        raise ValueError('private zero page')
    if type(extent) is not int or not 1 <= extent <= 4096 or mask_capacity < 5:
        raise ValueError('bounded explicit extent/mask capacity')
    p = plan(metadata.actual_seq_lengths_q, real_queries)
    mask_in = metadata.tree_attention_mask
    if not metadata.tree_attention or not metadata.use_fused_infer_attention or mask_in.shape[-1] != 4096:
        raise ValueError('original complete FULL visibility masks')
    sources = [x for x in p['physical_group_sources'] if x is not None]
    if any((not 0 < metadata.sequence_lens[g] <= extent for g in sources)):
        raise ValueError('never truncate visible real keys')
    device = inputs.device
    geometry = CACHE.get(p, device, int(mask_in.shape[2]), mask_capacity)
    identity = p['physical_to_original'] == list(range(165))
    simple = identity and real_queries % 5 == 0 and (sources == list(range(real_queries // 5))) and (tuple(metadata.actual_seq_lengths_q) == tuple(range(5, 166, 5)))
    tables = torch.full((33, 32), zero_page, dtype=torch.int32, device=device)
    if sources:
        live = metadata.request_block_tables[:len(sources)] if simple else metadata.request_block_tables.index_select(0, geometry['selected'])
        tables[:len(sources)].copy_(torch.where(live >= 0, live, zero_page))
    mask = torch.ones((33, 1, mask_capacity, 4096), dtype=torch.bool, device=device)
    if simple:
        n = real_queries // 5
        mask[:n, :, :5].copy_(mask_in[:n, :, :5])
        mask[n:33, :, :5, 0] = False
    else:
        mask.reshape(-1, 4096).index_copy_(0, geometry['destination'], mask_in.reshape(-1, 4096).index_select(0, geometry['source']))
        mask.reshape(-1, 4096)[geometry['phantom'], 0] = False
    index = geometry['index']
    lengths = metadata.context_lens if identity else metadata.context_lens.index_select(0, geometry['cpu_index'] if metadata.context_lens.device.type == 'cpu' else index)
    out = replace(metadata, slot_mapping=metadata.slot_mapping if identity else metadata.slot_mapping.index_select(0, index), context_lens=lengths, block_tables=tables, request_block_tables=tables, tree_attention_mask=mask, actual_seq_lengths_q=tuple(range(5, 166, 5)), sequence_lens=(extent,) * 33)
    return (inputs if identity else inputs.index_select(0, index), positions if identity else positions.index_select(0, index), out, geometry['inverse'], p)
