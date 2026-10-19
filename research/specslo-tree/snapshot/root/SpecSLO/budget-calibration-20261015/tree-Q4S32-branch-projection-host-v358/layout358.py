"""HOST-only root-alternate3 private KV views for a Q4/S32 candidate."""
BLOCK = 128
CONTEXT = 4096
COPY_SLOTS = 1792


def is_fork(plan):
    parents = plan.parent_indices.tolist()
    return any(parents.count(parent) > 1 for parent in set(parents))


def layout(plans, ids, tables, scratch_ids, *, copy_slots=COPY_SLOTS):
    import torch
    if len(plans) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError('independent physical plan owners')
    if sum(p.parent_indices.numel() + 1 for p in plans) != 128:
        raise ValueError('complete original128 physical query rows')
    branches = sum(is_fork(p) for p in plans)
    if not branches:
        return dict(enabled=False, branches=0, physical_queries=128)
    if branches > 12:
        raise ValueError('at most12 alternate owner groups')
    used = set(ids)
    pages = [tables[owner][0] for owner in scratch_ids if owner not in used]
    if len(pages) != len(set(pages)) or any(not 0 <= page < 544 for page in pages):
        raise ValueError('independent original pool scratch pages')
    if set(pages) & {page for owner in ids for page in tables[owner] if page >= 0}:
        raise ValueError('scratch pages cannot alias a physical writer or committed prefix')
    consumed, row = 0, 0
    groups, source, dest = [], [], []
    for plan, owner in zip(plans, ids, strict=True):
        parents = plan.parent_indices.tolist()
        physical, logical = plan.cache_positions.tolist(), plan.positions.tolist()
        mask, table = plan.attention_mask, list(tables[owner])
        q = len(parents) + 1
        if tuple(mask.shape) != (q, CONTEXT) or mask.device.type != 'cpu':
            raise ValueError('complete original CPU tree mask')
        if not is_fork(plan):
            if not 0 <= len(parents) <= 3 or parents != list(range(-1, len(parents) - 1)):
                raise ValueError('linear Q4 plan must be a depth0..3 chain; zero depth is padding only')
            groups.append(dict(owner=owner, original_rows=list(range(row, row + q)),
                               tables=table, KV=max(physical) + 1, masks=mask))
            row += q
            continue
        side = len(parents) - 3
        if not 1 <= side <= 3 or parents != [-1, 0, 1, -1, 3, 4][:len(parents)]:
            raise ValueError('exact primary3 plus root-alternate1..3 ancestry')
        prefix = physical[0]
        if (physical != list(range(prefix, prefix + q))
                or logical != list(range(prefix, prefix + 4)) + list(range(prefix + 1, prefix + side + 1))):
            raise ValueError('complete physical and RoPE ancestry correspondence')
        groups.append(dict(owner=owner, original_rows=list(range(row, row + 4)),
                           tables=table, KV=prefix + 4, masks=mask[:4]))
        visible = (~mask[q - 1]).nonzero().flatten().tolist()
        if (visible[:prefix] != list(range(prefix)) or len(visible) != logical[-1] + 1
                or visible[-side:] != physical[4:]):
            raise ValueError('complete ordered alternate ancestry and prefix visibility')
        complete = prefix // BLOCK
        tail = visible[complete * BLOCK:]
        needed = (len(tail) + BLOCK - 1) // BLOCK
        own = pages[consumed:consumed + needed]
        if len(own) != needed:
            raise ValueError('private branch tail page capacity')
        consumed += needed
        virtual = list(table)
        if complete + needed > len(virtual):
            raise ValueError('canonical alternate KV must fit model context')
        for j, page in enumerate(own):
            virtual[complete + j] = page
        for page in {table[pos // BLOCK] for pos in tail}:
            if not 0 <= page < 544:
                raise ValueError('alternate copies read only allocated original KV')
        source.extend(table[pos // BLOCK] * BLOCK + pos % BLOCK for pos in tail)
        for start in range(0, len(tail), BLOCK):
            first = own[start // BLOCK] * BLOCK
            dest.extend(range(first, first + min(BLOCK, len(tail) - start)))
        canonical_mask = torch.arange(CONTEXT, device='cpu')[None, :] >= torch.tensor(
            logical[4:], device='cpu')[:, None] + 1
        groups.append(dict(owner=owner, original_rows=list(range(row + 4, row + q)),
                           tables=virtual, KV=logical[-1] + 1, masks=canonical_mask))
        row += q
    real_copies = len(source)
    if type(copy_slots) is not int or real_copies > copy_slots:
        raise ValueError('fixed original branch copy slot bound')
    padding = copy_slots - real_copies
    extra = (padding + BLOCK - 1) // BLOCK
    identity = pages[consumed:consumed + extra]
    if len(identity) != extra:
        raise ValueError('private identity copy padding pages')
    for start in range(0, padding, BLOCK):
        first = identity[start // BLOCK] * BLOCK
        run = range(first, first + min(BLOCK, padding - start))
        source.extend(run)
        dest.extend(run)
    if len(dest) != copy_slots or len(set(dest)) != copy_slots:
        raise RuntimeError('unique complete private copy destinations')
    if any(not 0 <= slot < 544 * BLOCK for slot in source + dest):
        raise ValueError('original544 cache pool capacity')
    masks = torch.ones((len(groups), 1, 10, CONTEXT), dtype=torch.bool, device='cpu')
    ends, cursor = [], 0
    for i, group in enumerate(groups):
        q = len(group['original_rows'])
        if not 1 <= q <= 4:
            raise ValueError('every projected primary or alternate segment fits Q4')
        masks[i, 0, :q] = group['masks']
        cursor += q
        ends.append(cursor)
    if cursor != 128 or [r for g in groups for r in g['original_rows']] != list(range(128)):
        raise RuntimeError('original128 physical query order retained')
    return dict(enabled=True, branches=branches, physical_queries=128, groups=groups, mask=masks,
                actual_seq_lengths_q=tuple(ends), sequence_lens=tuple(g['KV'] for g in groups),
                tables=[g['tables'] for g in groups], source=source, dest=dest,
                real_copy_slots=real_copies, fixed_copy_slots=copy_slots,
                branch_tail_pages=consumed, identity_padding_pages=extra,
                attention_segments=len(groups))
