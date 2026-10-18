"""Original M120 projection with checked, raw-byte K/V branch copies."""
import types

from static_state255 import ensure_zero


def bind_checked_prepare(state, branch_tail255):
    from contract262 import checked
    from layout269 import layout as original_layout
    state.copy270_layout_checks = 0
    state.copy280_bucket = 0
    state.copy280_last_dispatch_key = None

    def checked_layout(*args, **kwargs):
        desc = original_layout(*args, **kwargs)
        if desc['enabled']:
            from prefix278 import select_grid
            state.copy280_bucket=select_grid(desc['source'],desc['dest'],rows=544*128)
            checked(desc['source'], desc['dest'], rows=544*128)
            state.copy270_layout_checks += 1
        else:
            state.copy280_bucket=0
        return desc

    # The private prepared wrapper already owns a bound State.prepare cell.
    # Replace only that cell with the same original implementation plus the
    # checked HOST layout; validation runs before either index buffer is copied.
    prepared = state.prepare
    cells = dict(zip(prepared.__code__.co_freevars, prepared.__closure__))
    original_prepare = cells['prepare'].cell_contents
    if original_prepare.__func__ is not branch_tail255.State.prepare:
        raise RuntimeError('ROOT280 original branch State.prepare required')
    private_prepare = types.FunctionType(original_prepare.__func__.__code__,
        dict(original_prepare.__func__.__globals__, layout=checked_layout),
        original_prepare.__func__.__name__, original_prepare.__func__.__defaults__,
        original_prepare.__func__.__closure__)
    bound_prepare = types.MethodType(private_prepare, state)
    def cell(value):
        return (lambda: value).__closure__[0]
    replacement = tuple(cell(bound_prepare) if name == 'prepare' else cells[name]
                        for name in prepared.__code__.co_freevars)
    state.prepare = types.FunctionType(prepared.__code__, prepared.__globals__,
        prepared.__name__, prepared.__defaults__, replacement)
    return state.prepare


def install(engine, native):
    import torch
    import branch_tail255
    import static_state255
    from contract262 import checked
    from copy278 import copy_rows

    activation = static_state255.install(engine, native)
    from padding269 import pad
    native.NativePearlEngine._pad_spec_rhythm_target_tree_graph=pad
    if engine.is_draft:
        return dict(activation, ROOT280_draft_uses_unchanged_original_path=True)
    state = engine._branch_tail_state
    bind_checked_prepare(state, branch_tail255)
    state.copy270_bindings = []
    widths = set()
    for layer in engine.model.layers:
        attn = layer.self_attn
        widths.add(attn.key_cache.flatten(0,1).numel()//(544*128))
    # Compile every real rank's row width outside any model graph capture.
    # Identity maps leave these private toy tensors unchanged.
    ids = torch.arange(1792, dtype=torch.long, device=engine.device)
    for width in sorted(widths):
        key = torch.zeros((1792, width), dtype=torch.bfloat16, device=engine.device)
        value = torch.zeros_like(key)
        copy_rows(key, value, ids, ids)
    torch.npu.synchronize()

    for layer in engine.model.layers:
        attn = layer.self_attn
        prior = attn._fused_infer_attention
        if prior.__kwdefaults__.get('_attention') is not attn:
            raise RuntimeError('ROOT280 same original branch attention binding')
        old = prior.__kwdefaults__['_old']
        key, value = attn.key_cache.flatten(0,1), attn.value_cache.flatten(0,1)
        if key.shape[0] != 544*128 or tuple(key.shape[1:]) not in ((2,128),(3,128)):
            raise RuntimeError('ROOT280 actual Native rank3 KV geometry')
        addresses = (key.data_ptr(), value.data_ptr())
        def execute(query, metadata, *, _attention=attn, _old=old,
                    _key=key, _value=value, _addresses=addresses):
            if state.enabled:
                if (_attention.key_cache.data_ptr(), _attention.value_cache.data_ptr()) != _addresses:
                    raise RuntimeError('ROOT280 persistent captured K/V storage changed')
                copy_rows(_key, _value, state.source, state.dest,bucket=state.copy280_bucket)
            return _old(query, metadata)
        attn._fused_infer_attention = execute
        state.copy270_bindings.append((attn, execute, addresses))
    state.copy270_source_pointers = (state.source.data_ptr(), state.dest.data_ptr())
    state.copy270_warm_widths = sorted(widths)
    from capture_key280 import install as install_key
    key_proof=install_key(engine)
    previous_greedy=engine.graph_runner.run_target_greedy
    def observed_greedy(*args,**kwargs):
        before=len(state.static_graph_calls)
        result=previous_greedy(*args,**kwargs)
        for call in state.static_graph_calls[before:]:
            call['copy_grid']=state.copy280_bucket if call['copy_enabled'] else 0
            call['actual_capture_key']=state.copy280_last_dispatch_key
        return result
    engine.graph_runner.run_target_greedy=observed_greedy
    return dict(activation, ROOT280_raw_byte_copy=True,ROOT280_copy_grid_key=key_proof,
                ROOT280_current_HOST_maps_validated_before_index_H2D=True,
                ROOT280_private_original_State_prepare_code_preserved=True,
                ROOT280_actual_rank3_cache=True, ROOT280_Triton_compiled_outside_graph=True)


def require_copy_bindings(engine):
    if engine.is_draft:
        return True
    state = engine._branch_tail_state
    if (state.source.data_ptr(), state.dest.data_ptr()) != state.copy270_source_pointers:
        raise RuntimeError('ROOT280 original captured copy index buffers changed')
    if len(state.copy270_bindings) != len(engine.model.layers):
        raise RuntimeError('ROOT280 every target layer must bind checked raw copy')
    for attention, function, addresses in state.copy270_bindings:
        if (attention._fused_infer_attention is not function
                or (attention.key_cache.data_ptr(), attention.value_cache.data_ptr()) != addresses):
            raise RuntimeError('ROOT280 captured attention or persistent K/V binding changed')
    return True
