"""Explicit retained B1..6 real proposals and private physical128 padding."""
def family(plans):
    if not plans:
        raise ValueError('positive real plan batch')
    side = False
    for plan in plans:
        budget = plan.candidate_budget
        if type(budget) is not int or not 1 <= budget <= 6:
            raise ValueError('real retained B1..6')
        if plan.parent_indices.tolist() != [-1, 0, 1, -1, 3, 4][:budget]:
            raise ValueError('exact primary3 root-alternate3 family')
        side |= budget > 3
    return 'primary3-root-alternate3' if side else 'primary-depth-at-most3'


def pad(engine, plans, roots, rows, ids, active=(), states=()):
    from plan357 import resource_use, make
    from padding231 import root_only
    n = len(plans)
    if not n == len(roots) == len(rows) == len(ids) or len(set(ids)) != n:
        raise ValueError('real plan, root, token and owner alignment')
    family(plans)
    resource_use([plan.candidate_budget for plan in plans])
    if any(len(row) != plan.candidate_budget for row, plan in zip(rows, plans, strict=True)):
        raise ValueError('preserve every real selected token')
    if any(type(owner) is not int or not 0 <= owner < 463 for owner in ids):
        raise ValueError('original real workload owner domain')
    if engine.config.max_model_len != 4096:
        raise ValueError('original4096 model domain')
    remaining = 128 - sum(plan.candidate_budget + 1 for plan in plans)
    widths = []
    while remaining:
        width = min(4, remaining)
        widths.append(width)
        remaining -= width
    scratch = [owner for owner in engine._tree_graph_scratch_sequence_ids if owner != 526][:len(widths)]
    if (len(scratch) != len(widths) or len(set(scratch)) != len(scratch)
            or set(scratch) & set(ids)):
        raise ValueError('independent private padding owners and reserved zero owner')
    pp, rr, dd, ss = list(plans), list(roots), [list(row) for row in rows], list(ids)
    for owner, width in zip(scratch, widths, strict=True):
        pp.append(root_only(4096) if width == 1 else make(width - 1, 0).plan)
        rr.append(0)
        dd.append([0] * (width - 1))
        ss.append(owner)
    if sum(plan.candidate_budget + 1 for plan in pp) != 128:
        raise RuntimeError('all original128 physical rows accounted')
    return pp, rr, dd, ss, n
