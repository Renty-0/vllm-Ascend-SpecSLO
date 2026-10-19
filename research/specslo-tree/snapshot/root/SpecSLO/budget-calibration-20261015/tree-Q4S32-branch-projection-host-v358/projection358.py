"""HOST candidate for a bijective Q4/S32 physical tree frame.

This module is not installed in serving and does not qualify model arithmetic.
It retains all 128 original physical rows, including original padding writers.
"""
from collections import OrderedDict


ROWS = 128
WIDTH = 4
GROUPS = 32


def plan(ends, real_queries):
    if type(real_queries) is not int or not 1 <= real_queries <= ROWS:
        raise ValueError('integer real query prefix within128')
    ends = tuple(ends)
    if not ends or ends[-1] != ROWS or any(type(x) is not int for x in ends):
        raise ValueError('complete original128 physical partition')
    groups = []
    begin = 0
    for group, end in enumerate(ends):
        if not begin < end <= ROWS:
            raise ValueError('strict increasing physical query partition')
        if begin < real_queries:
            if end > real_queries or end - begin > WIDTH:
                raise ValueError('every real primary or side segment fits Q4')
            groups.append((group, list(range(begin, end))))
        begin = end
    if real_queries not in ends or len(groups) > GROUPS:
        raise ValueError('exact real boundary and at most32 primary plus side groups')
    unused = iter(range(real_queries, ROWS))
    order, moves, sources, phantoms = [], [], [], []
    for physical in range(GROUPS):
        if physical < len(groups):
            source, rows = groups[physical]
            sources.append(source)
            for query, old in enumerate(rows):
                moves.append(dict(old_row=old, new_row=len(order), old_group=source, old_query=query))
                order.append(old)
        else:
            rows = []
            sources.append(None)
        for _ in range(len(rows), WIDTH):
            old = next(unused, None)
            if old is None:
                raise RuntimeError('padding cannot invent or duplicate original writers')
            phantoms.append(len(order))
            order.append(old)
    if next(unused, None) is not None or sorted(order) != list(range(ROWS)):
        raise RuntimeError('all128 input and KV write rows must form a bijection')
    inverse = [None] * ROWS
    for new, old in enumerate(order):
        inverse[old] = new
    if sorted(move['old_row'] for move in moves) != list(range(real_queries)):
        raise RuntimeError('each real query appears exactly once')
    return dict(physical_to_original=order, original_to_physical=inverse, real_moves=moves,
                physical_group_sources=sources, phantom_rows=phantoms,
                actual_seq_lengths_q=list(range(WIDTH, ROWS + 1, WIDTH)), physical_queries=ROWS,
                supported_primary_side_groups=len(groups), real_queries=real_queries)


class IndexCache:
    """Bounded cache of immutable index geometry only; no live metadata cached."""

    def __init__(self, maximum=256):
        if type(maximum) is not int or maximum < 1:
            raise ValueError('positive integer index cache bound')
        self.maximum = maximum
        self.entries = OrderedDict()

    def get(self, projection, device, old_extent, mask_capacity):
        import torch
        moves = projection['real_moves']
        sources = [v for v in projection['physical_group_sources'] if v is not None]
        order, inverse = projection['physical_to_original'], projection['original_to_physical']
        phantoms = projection['phantom_rows']
        key = (ROWS, WIDTH, str(device), tuple(order), tuple(inverse), tuple(sources),
               old_extent, mask_capacity,
               tuple((m['old_group'], m['old_query'], m['new_row']) for m in moves), tuple(phantoms))
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key]
        value = dict(index=torch.tensor(order, dtype=torch.long, device=device),
                     cpu_index=torch.tensor(order, dtype=torch.long, device='cpu'),
                     inverse=torch.tensor(inverse, dtype=torch.long, device=device),
                     selected=torch.tensor(sources, dtype=torch.long, device=device),
                     source=torch.tensor([m['old_group'] * old_extent + m['old_query'] for m in moves],
                                         dtype=torch.long, device=device),
                     destination=torch.tensor([(m['new_row'] // WIDTH) * mask_capacity
                                               + m['new_row'] % WIDTH for m in moves],
                                              dtype=torch.long, device=device),
                     phantom=torch.tensor([(r // WIDTH) * mask_capacity + r % WIDTH for r in phantoms],
                                          dtype=torch.long, device=device))
        self.entries[key] = value
        if len(self.entries) > self.maximum:
            self.entries.popitem(last=False)
        return value


CACHE = IndexCache()


def transform(inputs, positions, metadata, real_queries, zero_page, *, extent=2560, mask_capacity=10):
    import torch
    from dataclasses import replace
    if type(zero_page) is not int or not 0 <= zero_page < 544:
        raise ValueError('original544 pool private zero page')
    if type(extent) is not int or not 1 <= extent <= 4096:
        raise ValueError('bounded explicit visible KV extent')
    if type(mask_capacity) is not int or mask_capacity < WIDTH:
        raise ValueError('explicit mask capacity must fit Q4')
    p = plan(metadata.actual_seq_lengths_q, real_queries)
    mask_in = metadata.tree_attention_mask
    if not metadata.tree_attention or not metadata.use_fused_infer_attention or mask_in.shape[-1] != 4096:
        raise ValueError('original full tree visibility masks')
    if inputs.numel() != ROWS or positions.numel() != ROWS or metadata.slot_mapping.numel() != ROWS:
        raise ValueError('complete physical128 inputs, RoPE positions and real KV writers')
    sources = [v for v in p['physical_group_sources'] if v is not None]
    if any(not 0 < metadata.sequence_lens[g] <= extent for g in sources):
        raise ValueError('visible real keys cannot be truncated')
    device = inputs.device
    geometry = CACHE.get(p, device, int(mask_in.shape[2]), mask_capacity)
    identity = p['physical_to_original'] == list(range(ROWS))
    simple = (identity and real_queries % WIDTH == 0
              and sources == list(range(real_queries // WIDTH))
              and tuple(metadata.actual_seq_lengths_q) == tuple(range(WIDTH, ROWS + 1, WIDTH)))
    tables = torch.full((GROUPS, 32), zero_page, dtype=torch.int32, device=device)
    if sources:
        live = (metadata.request_block_tables[:len(sources)] if simple
                else metadata.request_block_tables.index_select(0, geometry['selected']))
        tables[:len(sources)].copy_(torch.where(live >= 0, live, zero_page))
    mask = torch.ones((GROUPS, 1, mask_capacity, 4096), dtype=torch.bool, device=device)
    if simple:
        n = real_queries // WIDTH
        mask[:n, :, :WIDTH].copy_(mask_in[:n, :, :WIDTH])
        mask[n:GROUPS, :, :WIDTH, 0] = False
    else:
        mask.reshape(-1, 4096).index_copy_(
            0, geometry['destination'], mask_in.reshape(-1, 4096).index_select(0, geometry['source']))
        mask.reshape(-1, 4096)[geometry['phantom'], 0] = False
    index = geometry['index']
    lengths = (metadata.context_lens if identity else metadata.context_lens.index_select(
        0, geometry['cpu_index'] if metadata.context_lens.device.type == 'cpu' else index))
    output = replace(metadata,
                     slot_mapping=metadata.slot_mapping if identity else metadata.slot_mapping.index_select(0, index),
                     context_lens=lengths, block_tables=tables, request_block_tables=tables,
                     tree_attention_mask=mask,
                     actual_seq_lengths_q=tuple(range(WIDTH, ROWS + 1, WIDTH)),
                     sequence_lens=(extent,) * GROUPS)
    return (inputs if identity else inputs.index_select(0, index),
            positions if identity else positions.index_select(0, index), output, geometry['inverse'], p)
