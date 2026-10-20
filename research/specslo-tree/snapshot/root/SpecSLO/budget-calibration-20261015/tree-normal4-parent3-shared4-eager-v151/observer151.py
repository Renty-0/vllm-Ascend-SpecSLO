"""Diagnostic-only immediate snapshots at each ORIGINAL public graph return.

No captured operator, native output contract or qualification flag changes.
Captured hidden references are retained only to observe the current graph,
then cloned before another graph can reuse the shared pool.
"""
from types import SimpleNamespace


def install(native, graph):
    cls=native.NativeACLGraphRunner
    execute=cls._execute_target
    def scoped(runner,inputs,positions,metadata,vocabulary_size,*,output_kind='greedy'):
        prior=getattr(runner,'_v151_target_scope',None)
        scope=None
        if (getattr(runner,'_v151_is_draft',False) and len(inputs)==1
                and not metadata[0].use_fused_infer_attention and metadata[0].attention_mask is None):
            scope=(f'target-{output_kind}:{vocabulary_size}|steps:1|paged',(len(inputs[0]),))
        runner._v151_target_scope=scope
        try:return execute(runner,inputs,positions,metadata,vocabulary_size,output_kind=output_kind)
        finally:runner._v151_target_scope=prior
    cls._execute_target=scoped
    original_init=native.NativePearlEngine.__init__
    def initialized(engine,*args,**kwargs):
        original_init(engine,*args,**kwargs)
        if not engine.is_draft:return
        runner=engine.graph_runner;runner._v151_is_draft=True
        runner._v151_target_hidden={};runner._v151_eager_target_hidden={}
        def hidden_hook(model,inputs,hidden):
            key=getattr(runner,'_v151_target_scope',None)
            if key is None:return
            store=(runner._v151_target_hidden if graph._CAPTURED_TASKS is not None
                   else runner._v151_eager_target_hidden)
            store[key]=hidden
        engine._v151_hidden_hook=engine.model.register_forward_hook(hidden_hook)
    native.NativePearlEngine.__init__=initialized


def _metadata(inputs,positions,meta):
    return dict(inputs=inputs.detach().clone(),positions=positions.detach().clone(),
        context_lens=meta.context_lens.detach().clone(),slot_mapping=meta.slot_mapping.detach().clone(),
        block_tables=meta.block_tables.detach().clone(),host_Q=tuple(meta.actual_seq_lengths_q),
        host_KV=tuple(meta.sequence_lens),use_fused_infer_attention=meta.use_fused_infer_attention,
        has_attention_mask=meta.attention_mask is not None,
        has_request_tables=meta.request_block_tables is not None)


def run(engine,method):
    import torch
    import tail_hidden_observer
    runner=engine.graph_runner
    records=dict(hidden=[],greedy=[],draft=[],root_head=[])
    originals={name:getattr(runner,name) for name in ('run_tree_hidden','run_target_greedy','run_draft_greedy')}
    missing=object();instance={name:vars(runner).get(name,missing) for name in originals}
    def hidden(inputs,positions,meta):
        result=originals['run_tree_hidden'](inputs,positions,meta)
        # First device operation after return, before frontier/head/tail.
        copied=result.detach().clone()
        key=('target-hidden:0|steps:1|paged',(len(inputs),))
        real_rows=int((meta.slot_mapping>=0).sum().item())
        padding=None
        if real_rows<len(inputs):
            import padding_reference
            padding=padding_reference.inspect(engine,inputs,positions,meta,real_rows)
        records['hidden'].append(dict(hidden=copied,key=key,
            metadata=_metadata(inputs,positions,meta),real_rows=real_rows,padding=padding))
        return result
    def greedy(inputs,positions,metadata,vocabulary_size):
        result=originals['run_target_greedy'](inputs,positions,metadata,vocabulary_size)
        if len(inputs)!=1:raise RuntimeError('V62 frontier requires one original native step')
        key=(f'target-greedy:{vocabulary_size}|steps:1|paged',(len(inputs[0]),))
        captured=runner._v151_target_hidden.get(key)
        eager=runner._v151_eager_target_hidden.get(key)
        if captured is None or eager is None:
            raise RuntimeError('Original frontier captured/eager hidden references unavailable')
        copied=captured.detach().clone()  # Before chain graph replay.
        interval=lambda h:(h.data_ptr(),h.data_ptr()+h.numel()*h.element_size())
        ca,ea=interval(captured),interval(eager)
        if ca[0]<ea[1] and ea[0]<ca[1]:
            raise RuntimeError('Frontier capture/eager hidden storage aliases')
        records['greedy'].append(dict(hidden=copied,tokens=result[0].detach().clone(),key=key,
            metadata=_metadata(inputs[0],positions[0],metadata[0]),
            capture_interval=ca,eager_interval=ea))
        return result
    def draft(inputs,positions,metadata,vocabulary_size,**kwargs):
        result=originals['run_draft_greedy'](inputs,positions,metadata,vocabulary_size,**kwargs)
        key=runner.last_draft_entry_key
        tail,intervals,eager_intervals=tail_hidden_observer.snapshot(runner,key,len(positions))
        records['draft'].append(dict(key=key,hidden=tail,
            encoded=runner.draft_entries[key].output.detach().clone(),
            metadata=[_metadata(inputs,pos,meta) for pos,meta in zip(positions,metadata)],
            metadata_input_scope='initial_input_only; subsequent graph inputs are previous full-physical top1 successors',
            capture_intervals=intervals,eager_intervals=eager_intervals))
        return result
    runner.run_tree_hidden=hidden;runner.run_target_greedy=greedy;runner.run_draft_greedy=draft
    try:return method(),records
    finally:
        for name,value in instance.items():
            if value is missing:delattr(runner,name)
            else:setattr(runner,name,value)
