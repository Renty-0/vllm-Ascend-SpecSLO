"""Bounded static q5 adapter around original real-query Native protocol."""

def ensure_zero(engine):
    state = engine._branch_tail_state
    if getattr(state, 'static_zero_allocation', None) is engine.cache_allocation:
        return state.static_zero_page
    if engine.cache_allocation.block_tables[526][0] < 0:
        engine._activate_cache_sequence(526, [0], enable_prefix_caching=False)
    page = engine.cache_allocation.block_tables[526][0]
    if not 0 <= page < 544:
        raise RuntimeError('actual reserved zero page')
    for layer in engine.model.layers:
        layer.self_attn.key_cache[page].zero_()
        layer.self_attn.value_cache[page].zero_()
    state.static_zero_allocation = engine.cache_allocation
    state.static_zero_page = page
    return page

def install(engine, native):
    import branch_tail285
    from projection285 import transform
    activation = branch_tail285.install(engine, native)
    state = engine._branch_tail_state
    state.static_current = None
    state.static_real_count = None
    state.static_graph_calls = []
    prepare = state.prepare
    eager = state.eager_last

    def prepared(plans, ids, x, pos, meta):
        page = ensure_zero(engine)
        result = prepare(plans, ids, x, pos, meta)
        if 526 in ids:
            raise RuntimeError('zero owner cannot be an original real or padding writer')
        count = state.static_real_count
        if type(count) is not int or not 1 <= count <= len(plans):
            raise RuntimeError('explicit original real plan count')
        q = sum((p.candidate_budget + 1 for p in plans[:count]))
        physical = transform(*result, q, page, extent=2560, mask_capacity=11 if state.enabled else 10)
        state.static_current = dict(original=result, physical=physical, copy_enabled=state.enabled, real_queries=q, real_N=count)
        return result
    state.prepare = prepared
    original_forward = native.NativePearlEngine.target_tree_forward

    def forwarded(e, *args, **kwargs):
        if kwargs.get('return_logits', False):
            raise RuntimeError('bounded static epoch qualifies original greedy only')
        state.static_current = None
        prior = state.static_real_count
        state.static_real_count = kwargs.get('real_tree_count')
        try:
            return original_forward(e, *args, **kwargs)
        finally:
            state.static_real_count = prior
    native.NativePearlEngine.target_tree_forward = forwarded
    original_greedy = engine.graph_runner.run_target_greedy

    def greedy(xs, ps, ms, vocabulary_size):
        current = state.static_current
        if current is None or not (xs[0] is current['original'][0] and ps[0] is current['original'][1] and (ms[0] is current['original'][2])):
            return original_greedy(xs, ps, ms, vocabulary_size)
        if len(xs) != 1 or len(ps) != 1 or len(ms) != 1:
            raise RuntimeError('one actual original tree verification step')
        x, p, meta, inverse, mapping = current['physical']
        values = original_greedy([x], [p], [meta], vocabulary_size)
        execution = engine.graph_runner.last_target_execution
        state.static_graph_calls.append(dict(N=current['real_N'], real_queries=current['real_queries'], M=165, attention_segments=33, actual_seq_lengths_q=list(meta.actual_seq_lengths_q), sequence_lens=list(meta.sequence_lens), mask_shape=list(meta.tree_attention_mask.shape), copy_enabled=current['copy_enabled'], physical_to_original=mapping['physical_to_original'], original_to_physical=mapping['original_to_physical'], real_moves=mapping['real_moves'], primary_side_groups=mapping['supported_primary_side_groups'], execution_mode=execution.mode, used_aclgraph=execution.used_aclgraph, fallback_reason=execution.fallback_reason))
        return tuple((v.index_select(0, inverse) for v in values))
    engine.graph_runner.run_target_greedy = greedy

    def eager_projected():
        current = state.static_current
        if current is None:
            return eager()
        x, p, meta, inverse, _ = current['physical']
        with state.active(current['copy_enabled']):
            return engine.model(x, p, meta).index_select(0, inverse).clone()
    state.eager_last = eager_projected
    return dict(original_branch_activation=activation, fixed_M=165, fixed_q5_groups=33, fixed_extent=2560, linear_mask_capacity=10, branch_copy_mask_capacity=11, real_input_position_and_KV_slot_bijection=True, greedy_only=True, production_qualified=False, whole_workload_AR_qualified=False)
