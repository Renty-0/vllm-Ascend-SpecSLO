"""One original target copy state, replacing only its per-layer copy kernel."""


def install(engine, native):
    import torch
    import install377
    import target_geometry
    from projection425 import project_function
    from bucket460 import bind
    from spans import pack

    def checked_layout(*args, **kwargs):
        desc = target_geometry.layout(*args, **kwargs)
        if desc['enabled']:
            compressed = pack(desc['source'], desc['dest'])
            state = engine._branch_tail_state
            state.span_copy_metadata.copy_(torch.tensor(compressed['spans'], dtype=torch.long))
            state.span_copy_layout_checks += 1
            state.span_copy_last_map = compressed
        return desc

    adapted = project_function(install377.install, (
        ('from geometry377 import pad_function', 'from target_geometry import pad_function'),
    ), dict(layout=checked_layout if not engine.is_draft else target_geometry.layout))
    activation = adapted(engine, native)
    extent = bind(engine)
    state = engine._branch_tail_state
    state.adaptive377_enabled = True
    if engine.is_draft:
        return dict(original=activation, extent=extent, draft_unchanged=True)
    from span_kernel import copy_spans
    from span_fixture import run
    state.uniform_KV_bucket425_enabled = True
    state.span_copy_metadata = torch.zeros((64, 3), dtype=torch.long, device=engine.device)
    state.span_copy_layout_checks = 0
    state.span_copy_last_map = None
    state.span_copy_bindings = []
    # Actual raw bits, changing descriptors, full-storage comparison and graph
    # replay are checked before any real model graph captures this new kernel.
    raw_proof = run(engine)
    for layer in engine.model.layers:
        attn = layer.self_attn
        prior = attn._fused_infer_attention
        if prior.__kwdefaults__.get('_attention') is not attn:
            raise RuntimeError('Exactly one original branch-copy wrapper required')
        old = prior.__kwdefaults__['_old']
        key, value = attn.key_cache.flatten(0, 1), attn.value_cache.flatten(0, 1)
        if tuple(key.shape) not in ((69632, 2, 128), (69632, 3, 128)):
            raise RuntimeError('Actual Native TP3 cache geometry')
        addresses = key.data_ptr(), value.data_ptr()

        def execute(query, metadata, *, _attention=attn, _old=old,
                    _key=key, _value=value, _addresses=addresses):
            if state.enabled:
                if (_attention.key_cache.data_ptr(), _attention.value_cache.data_ptr()) != _addresses:
                    raise RuntimeError('Retained canonical KV storage changed')
                copy_spans(_key, _value, state.span_copy_metadata)
            return _old(query, metadata)

        attn._fused_infer_attention = execute
        state.span_copy_bindings.append((attn, execute, addresses))
    state.span_copy_original_index_pointers = state.source.data_ptr(), state.dest.data_ptr()
    state.span_copy_descriptor_pointer = state.span_copy_metadata.data_ptr()
    state.span_copy_raw_proof = raw_proof
    return dict(original=activation, extent=extent, raw_copy_proof=raw_proof,
                actual_layers=len(state.span_copy_bindings), fixed_copy_programs=64,
                full_original1792_pairs_preserved=True, mask_shape_unchanged=True,
                Neural_qualification_inherited=False, serving_adopted=False)


def snapshot(engine):
    if engine.is_draft:
        return dict(draft_unchanged=True)
    state = engine._branch_tail_state
    if ((state.source.data_ptr(), state.dest.data_ptr()) != state.span_copy_original_index_pointers
            or state.span_copy_metadata.data_ptr() != state.span_copy_descriptor_pointer):
        raise RuntimeError('Persistent graph index or descriptor storage changed')
    if len(state.span_copy_bindings) != 64:
        raise RuntimeError('All64 target attention layers must use the measured copy path')
    for attn, function, addresses in state.span_copy_bindings:
        if attn._fused_infer_attention is not function or (attn.key_cache.data_ptr(), attn.value_cache.data_ptr()) != addresses:
            raise RuntimeError('Actual copied model layer binding changed')
    return dict(actual_layers=64, raw_proof=state.span_copy_raw_proof,
                checked_real_layouts=state.span_copy_layout_checks,
                last_copy_map=state.span_copy_last_map, bindings_unchanged=True,
                original_gather_scatter_bypassed=True, BF16_arithmetic=False)
