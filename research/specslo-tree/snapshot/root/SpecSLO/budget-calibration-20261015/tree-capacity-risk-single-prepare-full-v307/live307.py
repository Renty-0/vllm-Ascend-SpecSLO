"""LIVE-only same-prefix learning and genuine alternate generation."""
def begin(engine):
    from risk307 import begin_generation
    if engine._canonical307_phase!='qualified':raise RuntimeError('ROOT307 qualified same-instance LIVE only')
    return begin_generation(engine._ROOT307_policy,engine._dynamic_tree_generation)

def observe(engine,request,plan,accepted,outputs):
    from risk307 import observe as paired,observe_primary
    remaining=(request.output_token_budget or 256)-request.delivered_tokens
    args=dict(parents=plan.parent_indices.tolist(),accepted=accepted,outputs=outputs,remaining_before=remaining)
    result=None
    if plan.candidate_budget>=1:result=paired(engine._ROOT307_policy,request,**args)
    if remaining>=5:
        b=plan.candidate_budget;delivered=sum(v>=0 for v in outputs);side=sum(v>=4 for v in accepted)
        generation=str(engine._dynamic_tree_generation)
        owners=engine._ROOT307_request_gain_observations.setdefault(generation,{})
        record=owners.setdefault(str(request.request_index),dict(slo_class=request.slo_class,by_budget={}))
        row=record['by_budget'].setdefault(str(b),dict(samples=0,delivered_sum=0,
            shadow_B4_sum=0,shadow_B4_available=(b>=4),accepted_side_nodes=0,delivered_histogram=[0]*6,side_histogram=[0]*5))
        row['samples']+=1;row['delivered_sum']+=delivered
        if b>=4:row['shadow_B4_sum']+=delivered-side
        row['accepted_side_nodes']+=side;row['delivered_histogram'][delivered]+=1;row['side_histogram'][side]+=1
    from debug_snapshot307 import maybe_save
    maybe_save(engine,request,plan,accepted,outputs)
    return result

def subset(output,indices):
    import torch
    confidence=output['draft_confidence']
    return dict(output,draft_token_ids=[output['draft_token_ids'][i]for i in indices],
        root_token_ids=[output['root_token_ids'][i]for i in indices],
        num_draft_tokens=[output['num_draft_tokens'][i]for i in indices],
        node_confidences=[output['node_confidences'][i]for i in indices],
        draft_confidence=confidence.index_select(0,torch.tensor(indices,dtype=torch.long,device=confidence.device)))

def merged(output,indices,alternate,physical):
    rows=list(output['draft_token_ids']);scores=list(output['node_confidences']);confidence=output['draft_confidence'].clone()
    for slot,index in enumerate(indices):
        if alternate['draft_token_ids'][slot][:5]!=rows[index][:5]:raise RuntimeError('ROOT307 original primary4/rootsecond changed')
        rows[index]=alternate['draft_token_ids'][slot];scores[index]=alternate['node_confidences'][slot]
        confidence[index]=alternate['draft_confidence'][slot]
    result=dict(output,draft_token_ids=rows,node_confidences=scores,draft_confidence=confidence,
        num_draft_tokens=list(output['num_draft_tokens']),query_count=output['query_count']+3*physical,
        graph_padding_query_count=output['graph_padding_query_count']+3*(physical-len(indices)),
        model_calls=output['model_calls']+1,graph_calls=output['graph_calls']+2,
        ROOT307_private_copy_graph_calls=1,ROOT307_alternate_model_graph_calls=1,
        ROOT307_alternate_model_steps=3,ROOT307_alternate_raw_decisions=6*len(indices),
        execution_mode='original-primary-plus-real-private-root-alternate3')
    if sum(result['num_draft_tokens'])!=sum(output['num_draft_tokens']):raise RuntimeError('ROOT307 selected candidate charges changed')
    return result

def install(engine,native,policy):
    if hasattr(engine,'_ROOT307_draft_forward'):raise RuntimeError('ROOT307 install once')
    engine._ROOT307_policy=policy;engine._ROOT307_draft_counts={}
    engine._ROOT307_request_gain_observations={}
    original=native.NativePearlEngine.draft_tree_forward
    def forwarded(e,plans,roots,ids,*args,**kwargs):
        result=original(e,plans,roots,ids,*args,**kwargs)
        if e is not engine:raise RuntimeError('ROOT307 same model instance')
        if not e.is_draft or e._canonical307_phase!='qualified':return result
        indices=[i for i,p in enumerate(plans)if p.candidate_budget>=6]
        if not indices:return result
        sources=kwargs.get('eager_parent_sources')or{}
        if len(indices)>12 or any(ids[i]in sources for i in indices):raise RuntimeError('ROOT307 atmost12 normal alternate owners')
        state=getattr(e,'_ROOT281_qualified_private_copy',None)
        if state is None:raise RuntimeError('ROOT307 same-instance qualified private copy required')
        for owner in range(500,524):
            if e.cache_allocation.block_tables[owner][0]<0:e._activate_cache_sequence(owner,[0],enable_prefix_caching=False)
        e._ensure_cache_capacity([ids[i]for i in indices],[plans[i].prefix_len+8 for i in indices])
        from producer277 import generate
        partial=subset(result,indices)
        actual=generate(e,[ids[i]for i in indices],[int(plans[i].prefix_len)for i in indices],
            [roots[i]for i in indices],partial,copy_state=state,steps=3)
        result=merged(result,indices,actual['output'],actual['metadata'][0].numel())
        generation=str(e._dynamic_tree_generation)
        row=e._ROOT307_draft_counts.setdefault(generation,dict(calls=0,owners=0,
            alternate_model_steps=0,extra_raw_decisions=0,private_copy_graph_replays=0))
        row['calls']+=1;row['owners']+=len(indices);row['alternate_model_steps']+=3
        row['extra_raw_decisions']+=6*len(indices);row['private_copy_graph_replays']+=1
        return result
    native.NativePearlEngine.draft_tree_forward=forwarded;engine._ROOT307_draft_forward=forwarded
    return dict(LIVE_only=True,original_qualification_draft_preserved=True,genuine_alternate_steps=3,candidate_charges_preserved=True)
