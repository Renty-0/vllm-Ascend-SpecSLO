"""Actual fixed M160 frames, preserving every real query and KV slot."""

def pack(engine, owners, positions, tokens, *, rows, reference=False):
    import torch
    from dataclasses import replace
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan
    from host_target_metadata import make_host_tree_attention_metadata
    from fixed_mask_extent import padded
    if rows != 160:
        raise ValueError('explicit fixed165 shape')
    q5groups = 32
    if not 0 < len(owners) <= q5groups or not len(owners) == len(positions) == len(tokens):
        raise ValueError('exact real owner/frontier/input correspondence')
    if len(set(owners)) != len(owners) or any((len(v) != 5 for v in tokens)):
        raise ValueError('independent real q5 frames')
    plans = []
    ids = list(owners)
    roots = []
    draft = []
    for owner, p, values in zip(owners, positions, tokens, strict=True):
        if not 0 <= p or p + (1 if reference else 5) > 2560:
            raise ValueError('all real five-query keys fit declared2560 extent')
        plans.append(cached_cpu_tree_speculation_plan(1, 4, p, 4096, candidate_budget=4))
        roots.append(values[0])
        draft.append(values[1:])
    for i in range(q5groups - len(owners)):
        ids.append(463 + i)
        plans.append(cached_cpu_tree_speculation_plan(1, 4, 0, 4096, candidate_budget=4))
        roots.append(0)
        draft.append([0] * 4)
    x, pos, meta = make_host_tree_attention_metadata(engine, plans, roots, draft, sequence_ids=ids, allow_causal_fast_path=False)
    if x.numel() != rows:
        raise RuntimeError('all actual physical model rows')
    meta = padded(meta, 10)
    mask = meta.tree_attention_mask.clone()
    if reference:
        mask[:len(owners), :, 1:5, :] = True
        mask[:len(owners), :, 1:5, 0] = False
    mask[len(owners):, :, :, :] = True
    for group, plan in enumerate(plans[len(owners):], len(owners)):
        mask[group, 0, :plan.candidate_budget + 1, 0] = False
    zero = engine.cache_allocation.block_tables[526][0]
    if zero < 0:
        raise RuntimeError('actual separately allocated private zero page')
    tables = torch.where(meta.request_block_tables >= 0, meta.request_block_tables, zero)
    ends = tuple(range(5, 161, 5))
    if tuple(meta.actual_seq_lengths_q) != ends:
        raise RuntimeError('actual33 q5 partition')
    return (x, pos, replace(meta, block_tables=tables, request_block_tables=tables, tree_attention_mask=mask, actual_seq_lengths_q=ends, sequence_lens=(2560,) * len(plans)))
