"""Original M120 projection with checked, raw-byte K/V branch copies."""
import types

from static_state255 import ensure_zero


def bind_checked_prepare(state, branch_tail255):
    from contract262 import checked
    from layout269 import layout as original_layout
    state.copy270_layout_checks = 0
    state.copy280_bucket = 0
    state.copy280_last_dispatch_key = None
    state.copy291_layout_HOST = None

    def checked_layout(*args, **kwargs):
        desc = original_layout(*args, **kwargs)
        state.copy291_layout_HOST = desc
        if desc['enabled']:
            from prefix278 import select_grid
            state.copy280_bucket=select_grid(desc['source'],desc['dest'],rows=544*128)
            # select_grid already executes the identical complete contract278.checked.
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
    return bind_split_prepare(state)


def bind_split_prepare(state):
    """Derive current writers from HOST plans, without reading device metadata."""
    import torch
    from split288 import split
    engine=state.engine
    state.prefix291_source=torch.empty(1792,dtype=torch.long,device=engine.device)
    state.alias291_destination=torch.empty(120,dtype=torch.long,device=engine.device)
    state.prefix291_spans=torch.empty((12,3),dtype=torch.long,device=engine.device)
    original=state.prepare
    def prepared(plans,ids,*args,**kwargs):
        result=original(plans,ids,*args,**kwargs)
        if state.enabled:
            canonical=[]
            for plan,owner in zip(plans,ids,strict=True):
                if plan.cache_positions.device.type!='cpu':
                    raise RuntimeError('ROOT291 original scheduling cache positions must remain HOST')
                table=engine.cache_allocation.block_tables[owner]
                for pos in plan.cache_positions.tolist():
                    page=table[pos//128]
                    if not 0<=page<544:raise RuntimeError('ROOT291 actual allocated canonical current writer')
                    canonical.append(page*128+pos%128)
            physical=state.static_current['physical']
            order=physical[4]['physical_to_original']
            if len(canonical)!=120 or sorted(order)!=list(range(120)):
                raise RuntimeError('ROOT291 original120 query slots and physical bijection')
            query_slots=[canonical[i]for i in order]
            desc=state.copy291_layout_HOST
            proposal=split(desc['source'],desc['dest'],query_slots)
            state.prefix291_source.copy_(torch.tensor(proposal['prefix_source'],dtype=torch.long,device='cpu'))
            state.alias291_destination.copy_(torch.tensor(proposal['query_alias_destination'],dtype=torch.long,device='cpu'))
            from spans292 import pack as pack_spans
            spans=pack_spans(proposal['prefix_source'],proposal['prefix_destination'])
            state.prefix291_spans.copy_(torch.tensor(spans['spans'],dtype=torch.long,device='cpu'))
            state.split291_last_HOST=dict(prefix_copies=proposal['prefix_copies'],current_aliases=proposal['current_aliases'],
                original_copies=proposal['original_copies'],query_slots=query_slots)
        else:state.split291_last_HOST=None
        return result
    state.prepare=prepared
    return prepared


def install(engine,native):
    import torch
    import branch_tail255,static_state255
    from padding269 import pad
    activation=static_state255.install(engine,native)
    native.NativePearlEngine._pad_spec_rhythm_target_tree_graph=pad
    if engine.is_draft:return dict(activation,ROOT291_draft_original_path=True)
    from kernels288 import dual_write
    from kernels292 import SpanCopies
    state=engine._branch_tail_state
    bind_checked_prepare(state,branch_tail255)
    caches=tuple((layer.self_attn.key_cache,layer.self_attn.value_cache)for layer in engine.model.layers)
    state.prefix291_copies=SpanCopies(caches,engine.device)
    empty_spans=torch.zeros((12,3),dtype=torch.long,device=engine.device)
    state.prefix291_copies.run_spans(empty_spans)
    # Compile the actual projection stride combinations using toy cache storage.
    # Main committed/model caches receive only identity copies during setup.
    attn=engine.model.layers[0].self_attn
    width=attn.num_kv_heads*128
    qkv_width=attn.q_size+2*attn.kv_size
    toy_key=torch.zeros_like(attn.key_cache);toy_value=torch.zeros_like(attn.value_cache)
    contiguous=torch.zeros((120,attn.num_kv_heads,128),dtype=torch.bfloat16,device=engine.device)
    qkv=torch.zeros((120,qkv_width),dtype=torch.bfloat16,device=engine.device)
    strided=qkv[:,attn.q_size:attn.q_size+width].view(120,attn.num_kv_heads,128)
    slots=torch.arange(120,dtype=torch.int32,device=engine.device)
    aliases=torch.full((120,),-1,dtype=torch.long,device=engine.device)
    for key in(contiguous,strided):
        for value in(contiguous,strided):dual_write(key,value,toy_key,toy_value,slots,aliases)
    torch.npu.synchronize()
    state.copy291_bindings=[]
    for layer in engine.model.layers:
        attention=layer.self_attn
        previous_attention=attention._fused_infer_attention
        if previous_attention.__kwdefaults__.get('_attention')is not attention:
            raise RuntimeError('ROOT291 retained original branch attention binding')
        original_attention=previous_attention.__kwdefaults__['_old']
        original_write=attention._write_to_cache
        addresses=(attention.key_cache.data_ptr(),attention.value_cache.data_ptr())
        # Native attention and its masks remain unchanged. Copies now precede
        # the model or are included in the cache writer itself.
        attention._fused_infer_attention=original_attention
        def write(slot_mapping,key,value,*,_attention=attention,_original=original_write,_addresses=addresses):
            if not state.enabled:return _original(slot_mapping,key,value)
            if(_attention.key_cache.data_ptr(),_attention.value_cache.data_ptr())!=_addresses:
                raise RuntimeError('ROOT291 retained main cache addresses changed')
            return dual_write(key,value,_attention.key_cache,_attention.value_cache,slot_mapping,state.alias291_destination)
        attention._write_to_cache=write
        state.copy291_bindings.append((attention,write,original_attention,addresses))
    def before_model(module,args):
        if module is not engine.model:raise RuntimeError('ROOT291 exact model prefix hook')
        if state.enabled:state.prefix291_copies.run_spans(state.prefix291_spans)
    handle=engine.model.register_forward_pre_hook(before_model)
    state.prefix291_hook=(handle,before_model)
    state.copy291_source_pointers=(state.source.data_ptr(),state.dest.data_ptr(),
        state.prefix291_source.data_ptr(),state.alias291_destination.data_ptr(),state.prefix291_spans.data_ptr())
    from capture_key291 import install as install_key
    key_proof=install_key(engine)
    previous_greedy=engine.graph_runner.run_target_greedy
    def observed_greedy(*args,**kwargs):
        before=len(state.static_graph_calls)
        result=previous_greedy(*args,**kwargs)
        for call in state.static_graph_calls[before:]:
            call['copy_grid']=state.copy280_bucket if call['copy_enabled']else 0
            call['actual_capture_key']=state.copy280_last_dispatch_key
            call['ROOT291_prefix_dual_cache']=bool(call['copy_enabled'])
            call['ROOT291_split']=state.split291_last_HOST
        return result
    engine.graph_runner.run_target_greedy=observed_greedy
    return dict(activation,ROOT291_all64_prefix_copy=True,ROOT291_prefix_strategy="contiguous_spans_coreDim768",ROOT291_current_dual_cache_writes=True,
        ROOT291_original_attention_and_Q5_geometry_preserved=True,ROOT291_new_capture_key=key_proof,
        ROOT291_no_device_metadata_reads=True,ROOT291_NPU_qualified=False)


def require_copy_bindings(engine):
    if engine.is_draft:return True
    state=engine._branch_tail_state
    pointers=(state.source.data_ptr(),state.dest.data_ptr(),state.prefix291_source.data_ptr(),state.alias291_destination.data_ptr(),state.prefix291_spans.data_ptr())
    if pointers!=state.copy291_source_pointers:raise RuntimeError('ROOT291 persistent source and alias buffers changed')
    if len(state.copy291_bindings)!=64:raise RuntimeError('ROOT291 all64 cache writers required')
    for attention,write,original,addresses in state.copy291_bindings:
        if attention._write_to_cache is not write or attention._fused_infer_attention is not original:
            raise RuntimeError('ROOT291 qualified writer or unchanged Native attention binding changed')
        if(attention.key_cache.data_ptr(),attention.value_cache.data_ptr())!=addresses:
            raise RuntimeError('ROOT291 persistent model KV storage changed')
    handle,hook=state.prefix291_hook
    if engine.model._forward_pre_hooks.get(handle.id)is not hook:
        raise RuntimeError('ROOT291 all-layer prefix hook removed or changed')
    return True
