"""Install one original copy state; adaptive geometry stays off for old gates."""
from dataclasses import replace
from bucket425 import extent

def valid_record(record):
    rows=record['M'];groups=rows//5;detail=record['uniform_KV_bucket425']
    if rows not in(80,120):return False
    selected=extent(tuple(detail['original_sequence_lens']))
    return(detail['enabled']is True and detail['extent']==selected
           and record['fixed_KV_extent']==[selected]*groups
           and record['attention_segments']==groups)

def bind(engine):
    if engine.is_draft:return None
    state=engine._branch_tail_state
    if hasattr(state,'uniform_KV_bucket425_enabled'):
        raise RuntimeError('ROOT460 one actual uniform bucket adapter')
    state.uniform_KV_bucket425_enabled=False
    previous=state.prepare;previous_greedy=engine.graph_runner.run_target_greedy
    def prepare(*args,**kwargs):
        result=previous(*args,**kwargs);current=state.static_current
        if current is None:raise RuntimeError('ROOT460 original checked physical frame')
        original_lengths=tuple(current['original'][2].sequence_lens)
        selected=extent(original_lengths)if state.uniform_KV_bucket425_enabled else 2560
        x,p,meta,inverse,mapping=current['physical'];rows=int(x.numel());groups=rows//5
        if(rows not in(80,120)or tuple(meta.sequence_lens)!=(2560,)*groups
           or tuple(meta.actual_seq_lengths_q)!=tuple(range(5,rows+1,5))):
            raise RuntimeError('ROOT460 complete selected q5 family')
        current['physical']=(x,p,replace(meta,sequence_lens=(selected,)*groups),inverse,mapping)
        detail=dict(enabled=state.uniform_KV_bucket425_enabled,extent=selected,original_sequence_lens=list(original_lengths))
        current['uniform_KV_bucket425']=detail
        current['adaptive460_enabled']=state.adaptive377_enabled
        return result
    def greedy(*args,**kwargs):
        before=len(state.static_graph_calls);result=previous_greedy(*args,**kwargs)
        if len(state.static_graph_calls)>before:
            current=state.static_current
            if current is None or 'uniform_KV_bucket425'not in current:
                raise RuntimeError('ROOT460 actual dispatch has checked visible extent')
            for call in state.static_graph_calls[before:]:
                detail=dict(current['uniform_KV_bucket425'])
                call['uniform_KV_bucket425']=detail
                call['uniform_KV_bucket377']=dict(detail)
                call['adaptive377_enabled']=state.adaptive377_enabled
                call['adaptive460_enabled']=state.adaptive377_enabled
        return result
    state.prepare=prepare;engine.graph_runner.run_target_greedy=greedy
    return dict(physical_families=[80,120],original425_uniform_bucket_fields_preserved=True,
                maximum_extent=2560,original_full_M120_controls_required=True,NN_permission=False)

def install(engine,native):
    from install377 import install as original
    activation=original(engine,native)
    engine._branch_tail_state.adaptive377_enabled=False
    return dict(original=activation,bucket=bind(engine),adaptive_enabled=False,
                original_copy_state_installed_once=True,Native_flags_assigned=False)
