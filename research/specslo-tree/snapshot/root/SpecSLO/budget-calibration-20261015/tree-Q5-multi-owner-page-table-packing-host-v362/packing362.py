"""HOST-only packing of independent small query segments into Q5/S24.

Each segment retains its original KV table and full query visibility. Joined
tables start at whole pages, and only the view's mask columns move. Neither
query RoPE positions nor canonical KV writers move to these virtual offsets.
There is no serving installation or device qualification in this module.
"""
from dataclasses import dataclass


ROWS, WIDTH, GROUPS, PAGE, CONTEXT = 120, 5, 24, 128, 4096


@dataclass(frozen=True)
class Segment:
    owner: int
    rows: tuple[int, ...]
    KV: int
    table: tuple[int, ...]
    mask: object


def pack(segments, *, real_queries, zero_page, extent=4096, mask_capacity=10):
    import torch
    if type(real_queries) is not int or not 1 <= real_queries <= ROWS:
        raise ValueError('positive complete real prefix within120')
    if type(zero_page) is not int or not 0 <= zero_page < 544:
        raise ValueError('separately reserved finite private zero page')
    if type(extent) is not int or extent not in (2560, 4096):
        raise ValueError('explicit uniform FULL2560 or FULL4096 candidate')
    if type(mask_capacity) is not int or mask_capacity not in (10, 11):
        raise ValueError('original complete fixed Q5 mask capacities')
    segments = tuple(segments)
    if not segments or any(type(segment) is not Segment for segment in segments):
        raise ValueError('positive typed original segment sequence')
    owners = {segment.owner for segment in segments}
    if len(owners) > 32:
        raise ValueError('positive original real owner set within32')
    if not 2 * len(owners) <= real_queries <= min(9 * len(owners), len(owners) + 96, ROWS):
        raise ValueError('original B1..8, normal C96 and physical Q120 resource envelope')
    if len(segments) - len(owners) > 12:
        raise ValueError('at most12 real alternate owner segments')
    old_rows = []
    for segment in segments:
        if type(segment) is not Segment or type(segment.owner) is not int or not 0 <= segment.owner < 463:
            raise ValueError('typed original real owner segment')
        if (type(segment.rows) is not tuple or not 1 <= len(segment.rows) <= WIDTH
                or any(type(row) is not int or not 0 <= row < real_queries for row in segment.rows)):
            raise ValueError('every real query segment fits one Q5 group')
        if (type(segment.KV) is not int or not 0 < segment.KV <= extent
                or type(segment.table) is not tuple or len(segment.table) != 32
                or any(type(page) is not int for page in segment.table)):
            raise ValueError('bounded source KV extent and original32-column page table')
        pages = (segment.KV + PAGE - 1) // PAGE
        if any(not 0 <= page < 544 for page in segment.table[:pages]):
            raise ValueError('all referenced source pages must already be allocated')
        if zero_page in segment.table[:pages]:
            raise ValueError('the reserved zero page cannot alias a real source view')
        if (segment.mask.device.type != 'cpu' or segment.mask.dtype != torch.bool
                or tuple(segment.mask.shape) != (len(segment.rows), CONTEXT)):
            raise ValueError('complete original CPU bool visibility mask')
        if bool((~segment.mask[:, segment.KV:]).any()) or bool(segment.mask.all(dim=1).any()):
            raise ValueError('all actual visible keys fit each source view; every query is finite')
        old_rows.extend(segment.rows)
    if sorted(old_rows) != list(range(real_queries)):
        raise ValueError('every original real query must appear exactly once')
    for owner in owners:
        parts = sorted((segment for segment in segments if segment.owner == owner), key=lambda item: min(item.rows))
        if (len(parts) > 2 or len(parts) == 2 and (len(parts[0].rows) != 5 or len(parts[1].rows) > 4)
                or len(parts) == 1 and len(parts[0].rows) < 2):
            raise ValueError('each owner is one positive primary, optionally followed by one alternate')
        combined = [row for part in parts for row in part.rows]
        if combined != list(range(combined[0], combined[0] + len(combined))):
            raise ValueError('all queries retain the original per-owner primary/alternate order')
    ordered = sorted(enumerate(segments), key=lambda item: (-len(item[1].rows), -item[1].KV, item[0]))
    bins = []
    max_pages = extent // PAGE
    for source, segment in ordered:
        pages = (segment.KV + PAGE - 1) // PAGE
        eligible = [i for i, group in enumerate(bins)
                    if group['queries'] + len(segment.rows) <= WIDTH
                    and group['pages'] + pages <= max_pages
                    and segment.owner not in group['owners']]
        if eligible:
            destination = min(eligible, key=lambda i: (WIDTH - bins[i]['queries'] - len(segment.rows),
                                                       max_pages - bins[i]['pages'] - pages, i))
        else:
            if len(bins) >= GROUPS:
                raise ValueError('no feasible conservative packing within24 Q5 groups')
            destination = len(bins)
            bins.append(dict(queries=0, pages=0, owners=set(), segments=[]))
        group = bins[destination]
        offset = group['pages'] * PAGE
        group['segments'].append((source, offset))
        group['queries'] += len(segment.rows)
        group['pages'] += pages
        group['owners'].add(segment.owner)
    tables = torch.full((GROUPS, 32), zero_page, dtype=torch.int32, device='cpu')
    masks = torch.ones((GROUPS, 1, mask_capacity, CONTEXT), dtype=torch.bool, device='cpu')
    order, moves, group_rows = [], [], []
    unused = iter(range(real_queries, ROWS))
    for group_index in range(GROUPS):
        members = []
        if group_index < len(bins):
            for source, offset in bins[group_index]['segments']:
                segment = segments[source]
                pages = (segment.KV + PAGE - 1) // PAGE
                first = offset // PAGE
                tables[group_index, first:first + pages] = torch.tensor(segment.table[:pages], dtype=torch.int32)
                for local, old in enumerate(segment.rows):
                    query = len(members)
                    if offset + segment.KV > extent:
                        raise RuntimeError('packing cannot truncate visible source keys')
                    masks[group_index, 0, query, offset:offset + segment.KV] = segment.mask[local, :segment.KV]
                    moves.append(dict(old_row=old, new_row=len(order), source_segment=source,
                                      source_query=local, physical_group=group_index,
                                      physical_query=query, virtual_offset=offset, owner=segment.owner))
                    order.append(old)
                    members.append(old)
        for query in range(len(members), WIDTH):
            old = next(unused, None)
            if old is None:
                raise RuntimeError('all original physical rows must fit without invented padding')
            masks[group_index, 0, query, 0] = False
            order.append(old)
        group_rows.append(members)
    if next(unused, None) is not None or sorted(order) != list(range(ROWS)):
        raise RuntimeError('all120 original physical input and write rows form one bijection')
    inverse = [None] * ROWS
    for new, old in enumerate(order):
        inverse[old] = new
    return dict(physical_to_original=order, original_to_physical=inverse, real_moves=moves,
                group_original_real_rows=group_rows, mask=masks, tables=tables,
                actual_seq_lengths_q=tuple(range(WIDTH, ROWS + 1, WIDTH)), sequence_lens=(extent,) * GROUPS,
                physical_queries=ROWS, actual_real_owners=len({segment.owner for segment in segments}),
                source_segments=len(segments), used_groups=len(bins), real_queries=real_queries,
                multi_owner_groups=sum(len(group['owners']) > 1 for group in bins),
                referenced_pages_per_group=[group['pages'] for group in bins],
                no_additional_KV_copy_for_packing=True,
                query_positions_and_canonical_write_slots_must_only_follow_row_permutation=True)
