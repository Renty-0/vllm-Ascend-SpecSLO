"""Identity-row FULL frames with exact visibility and bounded zero-filled tables."""
from dataclasses import replace
from types import MethodType
from geometry336 import LAYOUTS


def ensure_zero(engine):
    if getattr(engine, '_q4_zero_allocation336', None) is engine.cache_allocation:
        return engine._q4_zero_page336
    if engine.cache_allocation.block_tables[526][0] < 0:
        engine._activate_cache_sequence(526, [0], enable_prefix_caching=False)
    page = engine.cache_allocation.block_tables[526][0]
    if not 0 <= page < 544:
        raise RuntimeError('actual independent zero page in original544 pool')
    for layer in engine.model.layers:
        layer.self_attn.key_cache[page].zero_()
        layer.self_attn.value_cache[page].zero_()
    engine._q4_zero_allocation336 = engine.cache_allocation
    engine._q4_zero_page336 = page
    return page


def canonical(engine, metadata, *, width, groups, reference=False):
    import torch
    from fixed_mask_extent import padded
    if not metadata.tree_attention or not metadata.use_fused_infer_attention:
        raise RuntimeError('explicit original FULL FIA metadata required')
    if tuple(metadata.actual_seq_lengths_q) != tuple(range(width, width*groups+1, width)):
        raise RuntimeError('exact uniform query groups required')
    if any(not 0 < length <= 2560 for length in metadata.sequence_lens):
        # Reference future queries can extend beyond the consumed root.
        if not reference:
            raise RuntimeError('no real candidate context truncated')
    metadata = padded(metadata, 10)
    mask = metadata.tree_attention_mask.clone()
    if reference:
        mask[:, :, 1:width, :] = True
        mask[:, :, 1:width, 0] = False
    zero = ensure_zero(engine)
    tables = torch.where(metadata.request_block_tables >= 0, metadata.request_block_tables, zero)
    return replace(metadata, block_tables=tables, request_block_tables=tables,
        tree_attention_mask=mask, actual_seq_lengths_q=tuple(range(width, width*groups+1, width)),
        sequence_lens=(2560,)*groups)


def install(engine):
    if engine.is_draft:
        return dict(target_only=False)
    model = engine.model
    original = model.make_tree_attention_metadata
    engine._q4_layout336 = None
    engine._q4_native_calls336 = []
    engine._q4_last_inputs336 = None
    for rows in (96, 120, 128):
        if rows not in model._mc2_chained_row_counts or rows in model._mc2_deferred_row_counts:
            raise RuntimeError('all actual matrix shapes must use original qualified flushed MC2')

    def make(this, plans, roots, draft, block_tables, *args, **kwargs):
        if this is not model or engine._q4_layout336 not in LAYOUTS:
            raise RuntimeError('same model and explicit diagnostic layout')
        width, groups = LAYOUTS[engine._q4_layout336]
        kwargs['allow_causal_fast_path'] = False
        x, positions, meta = original(plans, roots, draft, block_tables, *args, **kwargs)
        if x.numel() != width*groups or len(plans) != groups:
            raise RuntimeError('complete actual physical input frame')
        return x, positions, canonical(engine, meta, width=width, groups=groups)

    model.make_tree_attention_metadata = MethodType(make, model)
    original_greedy = engine.graph_runner.run_target_greedy

    def greedy(xs, ps, ms, vocabulary_size):
        if len(xs) != 1 or len(ps) != 1 or len(ms) != 1:
            raise RuntimeError('one original Native target step')
        width, groups = LAYOUTS[engine._q4_layout336]
        if xs[0].numel() != width*groups or tuple(ms[0].tree_attention_mask.shape) != (groups, 1, 10, 4096):
            raise RuntimeError('actual Native graph metadata differs from declared frame')
        result = original_greedy(xs, ps, ms, vocabulary_size)
        execution = engine.graph_runner.last_target_execution
        engine._q4_last_inputs336 = (xs[0], ps[0], ms[0])
        engine._q4_native_calls336.append(dict(layout=engine._q4_layout336, M=width*groups,
            groups=groups, width=width, mask_shape=list(ms[0].tree_attention_mask.shape),
            actual_seq_lengths_q=list(ms[0].actual_seq_lengths_q),
            sequence_lens=list(ms[0].sequence_lens), used_aclgraph=execution.used_aclgraph,
            mode=execution.mode, fallback_reason=execution.fallback_reason))
        return result

    engine.graph_runner.run_target_greedy = greedy
    return dict(target_only=True, physical_rows=[96, 120, 128], original_model_forward_unchanged=True,
        original_capture_validation_replay_unchanged=True, Native_flags_assigned=False,
        actual_qualified_MC2_rows=True, deferred_read_done=False)


def reference_frame(engine, *, layout, owners, positions, tokens, future):
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan
    from host_target_metadata import make_host_tree_attention_metadata
    width, groups = LAYOUTS[layout]
    if len(owners) > groups or len(set(owners)) != len(owners):
        raise ValueError('independent unique reference owners')
    if any(not 0 <= p < 2560 for p in positions):
        raise ValueError('consumed reference root never truncated')
    selected = list(owners)
    frontiers = list(positions)
    roots = list(tokens)
    for j in range(groups-len(owners)):
        selected.append(463+j)
        frontiers.append(0)
        roots.append(0)
    plans = [cached_cpu_tree_speculation_plan(1, width-1, p, 4096, candidate_budget=width-1)
             for p in frontiers]
    for owner, p in zip(selected, frontiers, strict=True):
        engine._ensure_cache_capacity([owner], [p+width-1])
    x, pos, meta = make_host_tree_attention_metadata(engine, plans, roots,
        [[future]*(width-1) for _ in selected], sequence_ids=selected, allow_causal_fast_path=False)
    if x.numel() != width*groups:
        raise RuntimeError('independent reference physical rows')
    return x, pos, canonical(engine, meta, width=width, groups=groups, reference=True)
