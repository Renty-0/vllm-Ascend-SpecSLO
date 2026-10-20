"""HOST prototype: preserve M120 while packing 32 FULL-attention segments.

Fixed segment capacities are 24*4+8*3=120. They permit up to24 linear B3
proposals and8 linear B2 proposals. Every original query, RoPE position and KV
write slot is preserved by a complete120-row permutation. No model, Native
graph, scheduler, or production capability is installed by this module.
"""
from dataclasses import replace

MATRIX_ROWS = 120
SEGMENT_CAPACITIES = (4,) * 24 + (3,) * 8
SEGMENTS = len(SEGMENT_CAPACITIES)
CANDIDATE_LIMIT = MATRIX_ROWS - SEGMENTS


def plan(ends, real_queries):
    if type(real_queries) is not int or not 1 <= real_queries <= MATRIX_ROWS:
        raise ValueError('complete positive real-query prefix within M120')
    ends = tuple(ends)
    if not ends or ends[-1] != MATRIX_ROWS or any(type(v) is not int for v in ends):
        raise ValueError('complete original physical120 query partition')
    groups = []
    begin = 0
    for owner, end in enumerate(ends):
        if not begin < end <= MATRIX_ROWS:
            raise ValueError('strict original query partition')
        if begin < real_queries:
            if end > real_queries or not 2 <= end - begin <= 4:
                raise ValueError('finite linear B1..3 groups only; do not split a real group')
            groups.append((owner, tuple(range(begin, end))))
        begin = end
    if real_queries not in ends or not 1 <= len(groups) <= SEGMENTS:
        raise ValueError('exact real-query boundary and at most32 real owners')
    if real_queries - len(groups) > CANDIDATE_LIMIT:
        raise ValueError('selected candidates must fit the actual C88 target envelope')
    # A descending stable assignment is complete for these nested capacities.
    # Greedy metadata order is restored with the independently explicit inverse.
    groups.sort(key=lambda item: (-len(item[1]), item[0]))
    if any(len(rows) > capacity for (_, rows), capacity in zip(groups, SEGMENT_CAPACITIES)):
        raise ValueError('at most24 real B3 owners fit the fixed 4/3 partition')
    unused = iter(range(real_queries, MATRIX_ROWS))
    order, moves, sources, phantom = [], [], [], []
    query_ends = []
    for physical_group, capacity in enumerate(SEGMENT_CAPACITIES):
        if physical_group < len(groups):
            owner, rows = groups[physical_group]
        else:
            owner, rows = None, ()
        sources.append(owner)
        for query, old in enumerate(rows):
            moves.append(dict(old_row=old, new_row=len(order), old_group=owner,
                              old_query=query, new_group=physical_group, new_query=query))
            order.append(old)
        for query in range(len(rows), capacity):
            old = next(unused, None)
            if old is None:
                raise ValueError('actual original padding does not cover the physical capacity')
            phantom.append(dict(new_row=len(order), new_group=physical_group, new_query=query))
            order.append(old)
        query_ends.append(len(order))
    if next(unused, None) is not None or sorted(order) != list(range(MATRIX_ROWS)):
        raise ValueError('all120 original rows must form a complete bijection')
    inverse = [None] * MATRIX_ROWS
    for new, old in enumerate(order):
        inverse[old] = new
    assert sorted(v['old_row'] for v in moves) == list(range(real_queries))
    assert len(sources) == SEGMENTS and query_ends[-1] == MATRIX_ROWS
    return dict(physical_to_original=order, original_to_physical=inverse,
                real_moves=moves, physical_group_sources=sources, phantom_rows=phantom,
                actual_seq_lengths_q=query_ends, physical_queries=MATRIX_ROWS,
                supported_primary_side_groups=len(groups), real_queries=real_queries,
                selected_candidates=real_queries - len(groups), real_N=len(groups),
                production_permission=False)


def transform(inputs, positions, metadata, real_queries, zero_page, *, extent=2560,
              mask_capacity=10):
    import torch
    if type(zero_page) is not int or not 0 <= zero_page < 544:
        raise ValueError('independently allocated finite private zero page required')
    if type(extent) is not int or not 1 <= extent <= 4096 or mask_capacity < 4:
        raise ValueError('complete visible KV extent and maximum physical q4')
    if not (metadata.tree_attention and metadata.use_fused_infer_attention
            and tuple(inputs.shape) == tuple(positions.shape) == (MATRIX_ROWS,)
            and metadata.tree_attention_mask.shape[-1] == 4096):
        raise ValueError('complete original FULL120 input and metadata required')
    mapping = plan(metadata.actual_seq_lengths_q, real_queries)
    sources = [v for v in mapping['physical_group_sources'] if v is not None]
    if any(not 0 < metadata.sequence_lens[g] <= extent for g in sources):
        raise ValueError('never truncate visible keys')
    device = inputs.device
    index = torch.tensor(mapping['physical_to_original'], dtype=torch.long, device=device)
    inverse = torch.tensor(mapping['original_to_physical'], dtype=torch.long, device=device)
    tables = torch.full((SEGMENTS, 32), zero_page, dtype=torch.int32, device=device)
    selected = torch.tensor(sources, dtype=torch.long, device=device)
    live = metadata.request_block_tables.index_select(0, selected)
    tables[:len(sources)].copy_(torch.where(live >= 0, live, zero_page))
    mask_in = metadata.tree_attention_mask
    old_capacity = int(mask_in.shape[2])
    mask = torch.ones((SEGMENTS, 1, mask_capacity, 4096), dtype=torch.bool, device=device)
    source = torch.tensor([v['old_group'] * old_capacity + v['old_query']
                           for v in mapping['real_moves']], dtype=torch.long, device=device)
    destination = torch.tensor([v['new_group'] * mask_capacity + v['new_query']
                                for v in mapping['real_moves']], dtype=torch.long, device=device)
    mask.reshape(-1, 4096).index_copy_(
        0, destination, mask_in.reshape(-1, 4096).index_select(0, source))
    phantom = torch.tensor([v['new_group'] * mask_capacity + v['new_query']
                            for v in mapping['phantom_rows']], dtype=torch.long, device=device)
    mask.reshape(-1, 4096)[phantom, 0] = False
    context_index = index if metadata.context_lens.device == device else index.to(metadata.context_lens.device)
    out = replace(metadata, slot_mapping=metadata.slot_mapping.index_select(0, index),
                  context_lens=metadata.context_lens.index_select(0, context_index),
                  block_tables=tables, request_block_tables=tables,
                  tree_attention_mask=mask,
                  actual_seq_lengths_q=tuple(mapping['actual_seq_lengths_q']),
                  sequence_lens=(extent,) * SEGMENTS)
    return inputs.index_select(0, index), positions.index_select(0, index), out, inverse, mapping
