"""Original M160 projection with checked, raw-byte K/V branch copies."""
import types
from static_base326 import ensure_zero

def bind_checked_prepare(state, branch_tail326):
    from contract262 import checked
    from layout326 import layout as original_layout
    state.copy270_layout_checks = 0
    state.copy280_bucket = 0
    state.copy280_last_dispatch_key = None
    state.copy326_layout_HOST = None
    state.copy326_contract = None

    def checked_layout(*args, **kwargs):
        desc = original_layout(*args, **kwargs)
        state.copy326_layout_HOST = desc
        if desc['enabled']:
            from descriptors304 import CheckedRows
            validated = CheckedRows(desc['source'], desc['dest'], rows=544 * 128)
            desc['source'] = validated.source
            desc['dest'] = validated.destination
            state.copy326_contract = validated
            state.copy280_bucket = validated.grid
            state.copy270_layout_checks += 1
        else:
            state.copy280_bucket = 0
            state.copy326_contract = None
        return desc
    prepared = state.prepare
    cells = dict(zip(prepared.__code__.co_freevars, prepared.__closure__))
    original_prepare = cells['prepare'].cell_contents
    if original_prepare.__func__ is not branch_tail326.State.prepare:
        raise RuntimeError('ROOT280 original branch State.prepare required')
    private_prepare = types.FunctionType(original_prepare.__func__.__code__, dict(original_prepare.__func__.__globals__, layout=checked_layout), original_prepare.__func__.__name__, original_prepare.__func__.__defaults__, original_prepare.__func__.__closure__)
    bound_prepare = types.MethodType(private_prepare, state)

    def cell(value):
        return (lambda: value).__closure__[0]
    replacement = tuple((cell(bound_prepare) if name == 'prepare' else cells[name] for name in prepared.__code__.co_freevars))
    state.prepare = types.FunctionType(prepared.__code__, prepared.__globals__, prepared.__name__, prepared.__defaults__, replacement)
    return bind_split_prepare(state)

def bind_split_prepare(state):
    """Derive current writers from HOST plans, without reading device metadata."""
    import torch
    from descriptors326 import separate
    engine = state.engine
    state.alias326_destination = torch.empty(160, dtype=torch.long, device=engine.device)
    state.prefix326_spans = torch.empty((12, 3), dtype=torch.long, device=engine.device)
    original = state.prepare

    def prepared(plans, ids, *args, **kwargs):
        result = original(plans, ids, *args, **kwargs)
        if state.enabled:
            canonical = []
            for plan, owner in zip(plans, ids, strict=True):
                if plan.cache_positions.device.type != 'cpu':
                    raise RuntimeError('ROOT326 original scheduling cache positions must remain HOST')
                table = engine.cache_allocation.block_tables[owner]
                for pos in plan.cache_positions.tolist():
                    page = table[pos // 128]
                    if not 0 <= page < 544:
                        raise RuntimeError('ROOT326 actual allocated canonical current writer')
                    canonical.append(page * 128 + pos % 128)
            physical = state.static_current['physical']
            order = physical[4]['physical_to_original']
            if len(canonical) != 160 or sorted(order) != list(range(160)):
                raise RuntimeError('ROOT326 original165 query slots and physical bijection')
            query_slots = [canonical[i] for i in order]
            desc = state.copy326_layout_HOST
            if desc['source'] is not state.copy326_contract.source or desc['dest'] is not state.copy326_contract.destination:
                raise RuntimeError('ROOT326 original H2D maps must be the exact immutable checked maps')
            proposal = separate(state.copy326_contract, query_slots)
            state.alias326_destination.copy_(torch.tensor(proposal['query_alias_destination'], dtype=torch.long, device='cpu'))
            state.prefix326_spans.copy_(torch.tensor(proposal['spans'], dtype=torch.long, device='cpu'))
            state.split326_last_HOST = dict(prefix_copies=proposal['prefix_copies'], current_aliases=proposal['current_aliases'], original_copies=proposal['original_copies'], query_slots=query_slots)
        else:
            state.split326_last_HOST = None
        return result
    state.prepare = prepared
    return prepared

def install(engine, native):
    import torch
    import branch_tail326, static_base326
    from padding326 import pad
    activation = static_base326.install(engine, native)
    native.NativePearlEngine._pad_spec_rhythm_target_tree_graph = pad
    if engine.is_draft:
        return dict(activation, ROOT305_draft_original_path=True)
    from write326 import dual_write
    from kernels292 import SpanCopies
    state = engine._branch_tail_state
    bind_checked_prepare(state, branch_tail326)
    caches = tuple(((layer.self_attn.key_cache, layer.self_attn.value_cache) for layer in engine.model.layers))
    from device296 import resolve_cache_device
    actual_device = resolve_cache_device(engine.device, caches[0][0].device, torch.npu.current_device())
    state.prefix326_copies = SpanCopies(caches, actual_device)
    empty_spans = torch.zeros((12, 3), dtype=torch.long, device=engine.device)
    state.prefix326_copies.run_spans(empty_spans)
    attn = engine.model.layers[0].self_attn
    width = attn.num_kv_heads * 128
    qkv_width = attn.q_size + 2 * attn.kv_size
    toy_key = torch.zeros_like(attn.key_cache)
    toy_value = torch.zeros_like(attn.value_cache)
    contiguous = torch.zeros((160, attn.num_kv_heads, 128), dtype=torch.bfloat16, device=engine.device)
    qkv = torch.zeros((160, qkv_width), dtype=torch.bfloat16, device=engine.device)
    strided = qkv[:, attn.q_size:attn.q_size + width].view(160, attn.num_kv_heads, 128)
    slots = torch.arange(160, dtype=torch.int32, device=engine.device)
    aliases = torch.full((160,), -1, dtype=torch.long, device=engine.device)
    for key in (contiguous, strided):
        for value in (contiguous, strided):
            dual_write(key, value, toy_key, toy_value, slots, aliases)
    torch.npu.synchronize()
    state.copy326_bindings = []
    for layer in engine.model.layers:
        attention = layer.self_attn
        previous_attention = attention._fused_infer_attention
        if previous_attention.__kwdefaults__.get('_attention') is not attention:
            raise RuntimeError('ROOT326 retained original branch attention binding')
        original_attention = previous_attention.__kwdefaults__['_old']
        original_write = attention._write_to_cache
        addresses = (attention.key_cache.data_ptr(), attention.value_cache.data_ptr())
        attention._fused_infer_attention = original_attention

        def write(slot_mapping, key, value, *, _attention=attention, _original=original_write, _addresses=addresses):
            if not state.enabled:
                return _original(slot_mapping, key, value)
            if (_attention.key_cache.data_ptr(), _attention.value_cache.data_ptr()) != _addresses:
                raise RuntimeError('ROOT326 retained main cache addresses changed')
            return dual_write(key, value, _attention.key_cache, _attention.value_cache, slot_mapping, state.alias326_destination)
        attention._write_to_cache = write
        state.copy326_bindings.append((attention, write, original_attention, addresses))

    def before_model(module, args):
        if module is not engine.model:
            raise RuntimeError('ROOT326 exact model prefix hook')
        if state.enabled:
            state.prefix326_copies.run_spans(state.prefix326_spans)
    handle = engine.model.register_forward_pre_hook(before_model)
    state.prefix326_hook = (handle, before_model)
    state.copy326_source_pointers = (state.source.data_ptr(), state.dest.data_ptr(), state.alias326_destination.data_ptr(), state.prefix326_spans.data_ptr())
    from capture_key296 import install as install_key
    key_proof = install_key(engine)
    previous_greedy = engine.graph_runner.run_target_greedy

    def observed_greedy(*args, **kwargs):
        before = len(state.static_graph_calls)
        result = previous_greedy(*args, **kwargs)
        for call in state.static_graph_calls[before:]:
            call['copy_grid'] = state.copy280_bucket if call['copy_enabled'] else 0
            call['actual_capture_key'] = state.copy280_last_dispatch_key
            call['ROOT326_prefix_dual_cache'] = bool(call['copy_enabled'])
            call['ROOT326_split'] = state.split326_last_HOST
        return result
    engine.graph_runner.run_target_greedy = observed_greedy
    return dict(activation, ROOT305_all64_prefix_copy=True, ROOT305_prefix_strategy='contiguous_spans_coreDim768', ROOT305_current_dual_cache_writes=True, ROOT305_original_attention_and_Q5_geometry_preserved=True, ROOT305_new_capture_key=key_proof, ROOT305_no_device_metadata_reads=True, ROOT305_NPU_qualified=False)

def require_copy_bindings(engine):
    if engine.is_draft:
        return True
    state = engine._branch_tail_state
    pointers = (state.source.data_ptr(), state.dest.data_ptr(), state.alias326_destination.data_ptr(), state.prefix326_spans.data_ptr())
    if pointers != state.copy326_source_pointers:
        raise RuntimeError('ROOT326 persistent source and alias buffers changed')
    if len(state.copy326_bindings) != 64:
        raise RuntimeError('ROOT326 all64 cache writers required')
    for attention, write, original, addresses in state.copy326_bindings:
        if attention._write_to_cache is not write or attention._fused_infer_attention is not original:
            raise RuntimeError('ROOT326 qualified writer or unchanged Native attention binding changed')
        if (attention.key_cache.data_ptr(), attention.value_cache.data_ptr()) != addresses:
            raise RuntimeError('ROOT326 persistent model KV storage changed')
    handle, hook = state.prefix326_hook
    if engine.model._forward_pre_hooks.get(handle.id) is not hook:
        raise RuntimeError('ROOT326 all-layer prefix hook removed or changed')
    return True
