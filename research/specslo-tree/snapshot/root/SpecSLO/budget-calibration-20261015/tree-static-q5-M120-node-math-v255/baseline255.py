"""Read-only difference measurement; old-layout negative control is not a gate pass."""
import json
from pathlib import Path

def retain(output,name,value):
    if output is None:return
    encoded=json.dumps(value,allow_nan=False)
    with(Path(output)/name).open('x')as f:f.write(encoded+'\n')

def measurement(engine,case,actual,own):
    import torch
    match=engine.graph_runner._target_outputs_match
    a=actual['tokens'].cpu().tolist();b=own['tokens'].cpu().tolist()
    bad_hidden=[i for i in range(len(a))if not match((actual['hidden'][i:i+1],),(own['hidden'][i:i+1],))]
    bad_KV=[i for i,(x,y)in enumerate(zip(actual['KV'],own['KV'],strict=True))if not match((x,),(y,))]
    return dict(rank=engine.rank,case=case,candidate_tokens=a,reference_tokens=b,tokens_equal=a==b,
        hidden_original_matcher_passed=not bad_hidden,hidden_different_real_query_rows=bad_hidden,
        hidden_max_abs_by_query=(actual['hidden'].float()-own['hidden'].float()).abs().amax(-1).cpu().tolist(),
        KV_original_matcher_passed=not bad_KV,KV_different_tensor_indices=bad_KV,first_different_KV_layer=bad_KV[0]//2 if bad_KV else None,
        actual_branch_layout=actual['branch_layout'],attention_segments=actual['attention_segments'],mask_query_capacity=actual['mask_query_capacity'],
        formal_Goodput=False,whole_workload_AR_qualified=False,adopted=False)

def run(engine,case,bank,plans,rows,stage,output):
    from oracle255 import candidate
    from oracle255 import legacy_reference as reference
    from reference_schedule255 import schedule
    target=not engine.is_draft;state=engine._branch_tail_state;old=state.prepare
    def original_layout(plans,ids,inputs,pos,meta):
        from fixed_mask_extent import padded
        meta=padded(meta,9)
        state.enabled=False;state.last=(inputs,pos,meta,False)
        state.last_record=dict(enabled=False,branches=0,physical_queries=120,baseline_original_layout=True)
        return inputs,pos,meta
    state.prepare=original_layout
    try:
        actual=stage('old-layout-negative/candidate',lambda:candidate(engine,case,bank,plans,rows)if target else None)
        if target:own=reference(engine,case,bank,plans,rows,lambda label,fn:stage('old-layout-negative/'+label,fn))
        else:
            own=None
            for i,frame in enumerate(schedule(plans,[0]*case['N'],rows)):stage('old-layout-negative/independent-reference/frame'+str(i),lambda:None)
        result=stage('old-layout-negative/measure-with-unchanged-Native-matcher',lambda:measurement(engine,case,actual,own)if target else None)
        if target:result.update(diagnostic_only=True,used_as_positive_qualification=False)
        stage('old-layout-negative/retain',lambda:retain(output,f'rank{engine.rank}-old-layout-negative-control.json',result))
        return result
    finally:
        state.prepare=old;state.enabled=False
