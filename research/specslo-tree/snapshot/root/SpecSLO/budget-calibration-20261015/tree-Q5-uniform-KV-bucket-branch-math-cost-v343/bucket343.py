"""Uniform execution extent after the original checked M120 tree projection."""
from dataclasses import replace


def extent(lengths):
    if not lengths or any(type(x)is not int or not 0<x<=2560 for x in lengths):
        raise ValueError('complete original visible KV contexts within2560')
    return ((max(lengths)+511)//512)*512


def case_extent(case,prefixes):
    return extent([len(prefixes[i])+min(b,4)+1 for i,b in enumerate(case['budgets'])])


def bind(engine):
    if engine.is_draft:raise RuntimeError("target-only bucket binding")
    state=engine._branch_tail_state
    state.uniform_bucket343_enabled=True
    previous=state.prepare

    def prepare(*args,**kwargs):
        result=previous(*args,**kwargs)
        current=state.static_current
        if current is None:
            raise RuntimeError('actual complete checked projection required')
        original=current['original'][2]
        selected=extent(tuple(original.sequence_lens)) if state.uniform_bucket343_enabled else 2560
        x,p,metadata,inverse,mapping=current['physical']
        if x.numel()!=120 or tuple(metadata.actual_seq_lengths_q)!=tuple(range(5,121,5)):
            raise RuntimeError('original fixedM120/Q5/S24 geometry required')
        if selected<max(original.sequence_lens):
            raise RuntimeError('no visible real or padding KV may be truncated')
        if tuple(metadata.sequence_lens)!=(2560,)*24:
            raise RuntimeError('original unmodified projection extent required')
        current['physical']=(x,p,replace(metadata,sequence_lens=(selected,)*24),inverse,mapping)
        current['uniform_KV_bucket343']=dict(extent=selected,
            original_max_context=max(original.sequence_lens),enabled=state.uniform_bucket343_enabled)
        return result

    state.prepare=prepare
    greedy=engine.graph_runner.run_target_greedy
    def observed(*args,**kwargs):
        before=len(state.static_graph_calls)
        result=greedy(*args,**kwargs)
        details=state.static_current['uniform_KV_bucket343']
        for call in state.static_graph_calls[before:]:
            call['uniform_KV_bucket343']=dict(details)
        return result
    engine.graph_runner.run_target_greedy=observed
    return dict(target_only=True,M=120,groups=24,
        all_original_copy_validation_and_Native_attention_retained=True,
        no_visible_context_truncation=True,independent_reference_stays_FULL2560=True,
        model_qualified=False,whole_workload_AR_qualified=False)


def install(engine,native):
    from static_state311 import install as original_install
    if not engine.is_draft:
        if 120 not in engine.model._mc2_chained_row_counts or 120 in engine.model._mc2_deferred_row_counts:
            raise RuntimeError('actual M120 original qualified flushed MC2 required')
    activation=original_install(engine,native)
    return dict(original=activation,bucket=bind(engine)if not engine.is_draft else None)
