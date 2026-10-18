"""Bounded fixed-shape FIA prototype; no complete workload qualification."""
def fixed_query_metadata(meta, zero_page):
    """Input visibility and all live prefix pages were allocator-checked upstream."""
    import torch
    from dataclasses import replace
    from query1_reference231 import canonical
    if type(zero_page) is not int or not 0 <= zero_page < 544:
        raise ValueError('actual dedicated zero page in original pool')
    out=canonical(meta)
    tables=torch.where(out.request_block_tables>=0,out.request_block_tables,zero_page)
    if tuple(tables.shape)!=(80,32) or tuple(out.tree_attention_mask.shape)!=(80,1,1,4096):
        raise RuntimeError('fixed80 independent query segments and exact original masks')
    return replace(out,block_tables=tables,request_block_tables=tables,
                   actual_seq_lengths_q=tuple(range(1,81)),sequence_lens=(4096,)*80)

def run(engine, stage, fixture):
    import torch
    import time
    import statistics
    from dataclasses import replace
    from metadata231 import pack_q5
    from query1_reference231 import canonical
    from dual_bank_v146 import query_KV
    target=not engine.is_draft
    prompts=fixture['prompts'];n=len(prompts);selected=fixture['selected_local_owners']
    samples=fixture['teacher_forced_common_inputs']
    versions={};executions=[];zero_page=None

    def initialize():
        nonlocal zero_page
        if not target:return None
        zero_page=engine.cache_allocation.block_tables[526][0]
        if zero_page<0 or any(zero_page in engine.cache_allocation.block_tables[i] for i in range(4*n)):
            raise RuntimeError('dedicated allocated non-real zero page')
        for layer in engine.model.layers:
            layer.self_attn.key_cache[zero_page].zero_()
            layer.self_attn.value_cache[zero_page].zero_()
        return dict(owner=526,page=zero_page,all128_KV_pages_zeroed=True)
    zero=stage('static240/dedicated-zero-KV-page',initialize)

    def transform(meta):
        return fixed_query_metadata(meta,zero_page)

    def execute(x,pos,meta,rows):
        with engine._branch_tail_state.active(False):
            graph=engine.graph_runner.run_target_greedy([x],[pos],[meta],engine.draft_vocab_size)[0].clone()
            execution=engine.graph_runner.last_target_execution
            executions.append(dict(mode=execution.mode,fallback_reason=execution.fallback_reason,
                                   used_aclgraph=execution.used_aclgraph))
            hidden=engine.model(x,pos,meta).clone()
            eager=engine.model.compute_greedy_tokens(hidden,engine.draft_vocab_size)
        if not torch.equal(graph,eager):raise RuntimeError('static240 original graph/eager greedy comparator')
        index=torch.tensor(rows,dtype=torch.long,device=engine.device)
        return dict(heads=graph.index_select(0,index),hidden=hidden.index_select(0,index).clone(),
                    KV=[v.clone() for v in query_KV(engine,meta.slot_mapping.index_select(0,index).long())])

    candidate_data=None
    def candidate():
        nonlocal candidate_data
        if not target:return None
        x,pos,meta,enabled=engine._branch_tail_state.last
        if enabled:raise RuntimeError('initial real prefix prototype is linear only')
        candidate_data=(x,pos,meta,transform(meta))
        return execute(x,pos,candidate_data[3],list(range(len(selected)*5)))
    actual=stage('static240/canonical-candidate',candidate)
    if target:versions['candidate']=actual

    first_reference=None
    for name,cohort,owners,teacher in [
        ('same11',1,selected,samples),
        ('original6',2,fixture['original_AR_local_owners'],fixture['original_AR_common_inputs'])]:
        snapshots=[]
        for depth in range(5):
            def one(depth=depth,cohort=cohort,owners=owners,teacher=teacher):
                nonlocal first_reference
                if not target:return None
                ids=[];positions=[];tokens=[]
                for j,i in enumerate(owners):
                    p=len(prompts[i])+depth
                    ids.extend([cohort*n+i]*5);positions.extend(range(p,p+5));tokens.extend([teacher[j][depth]]+[0]*4)
                for i in range(16-len(owners)):
                    ids.extend([463+i]*5);positions.extend(range(5));tokens.extend([0]*5)
                engine._prepare_attention_metadata(ids,positions,True)
                x,pos,meta=pack_q5(engine,dict(ids=ids,positions=positions,tokens=tokens,selected=ids[::5]),full=True)
                transformed=transform(meta)
                if first_reference is None:first_reference=(x,pos,meta,transformed)
                return execute(x,pos,transformed,list(range(0,len(owners)*5,5)))
            snapshots.append(stage('static240/'+name+'-own-reference-depth'+str(depth),one))
        if target:
            versions[name]={key:(torch.stack([snapshots[d][key][j] for j in range(len(owners)) for d in range(5)])
                                if key!='KV' else [torch.stack([snapshots[d]['KV'][a][j] for j in range(len(owners)) for d in range(5)]) for a in range(128)])
                            for key in ('heads','hidden','KV')}

    def compare():
        if not target:return None
        a,b=versions['candidate'],versions['same11']
        q=fixture['selected_original_owners'].index(19)*5
        r=fixture['original_AR_request19_subset_index']*5
        c=versions['original6']
        return dict(all55_heads_equal=torch.equal(a['heads'],b['heads']),
                    all55_hidden_bitwise_equal=torch.equal(a['hidden'],b['hidden']),
                    all55_KV_bitwise_equal=all(torch.equal(x,y)for x,y in zip(a['KV'],b['KV'],strict=True)),
                    hidden_original_matcher=engine.graph_runner._target_outputs_match((a['hidden'],),(b['hidden'],)),
                    KV_original_matcher=engine.graph_runner._target_outputs_match(tuple(a['KV']),tuple(b['KV'])),
                    original6_request19_heads_equal=torch.equal(a['heads'][q:q+5],c['heads'][r:r+5]),
                    original6_request19_hidden_bitwise_equal=torch.equal(a['hidden'][q:q+5],c['hidden'][r:r+5]),
                    original6_request19_KV_bitwise_equal=all(torch.equal(x[q:q+5],y[r:r+5])for x,y in zip(a['KV'],c['KV'],strict=True)),
                    request19_heads=dict(candidate=a['heads'][q:q+5].cpu().tolist(),same11=b['heads'][q:q+5].cpu().tolist(),original6=c['heads'][r:r+5].cpu().tolist()))
    comparison=stage('static240/independent-real-prefix-comparison',compare)

    def benchmark():
        if not target:return None
        runner=engine.graph_runner
        # Both original and proposed graphs must naturally validate with two
        # distinct actual inputs before any component timing is reported.
        for data in (candidate_data,first_reference):
            for k in (2,3):
                with engine._branch_tail_state.active(False):
                    out=runner.run_target_greedy([data[0]],[data[1]],[data[k]],engine.draft_vocab_size)[0]
                out.cpu();torch.npu.synchronize()
        entries=[e for e in runner.target_entries.values() if e.tree_attention_masks and e.tree_attention_masks[0] is not None
                 and tuple(e.tree_attention_masks[0].shape) in ((16,1,9,4096),(80,1,1,4096))]
        validated=bool(entries)and all(e.runtime_validated is True for e in entries)
        if not validated:return dict(status='not_timed_unvalidated_graph',entries=len(entries),validated=False)
        values=dict(original=[],static=[])
        before=runner.graph_metrics() if hasattr(runner,'graph_metrics') else engine.graph_metrics()
        for iteration in range(20):
            for name,k in (('original',2),('static',3))if iteration%2==0 else(('static',3),('original',2)):
                data=candidate_data if iteration%2==0 else first_reference
                started=time.perf_counter()
                with engine._branch_tail_state.active(False):
                    out=runner.run_target_greedy([data[0]],[data[1]],[data[k]],engine.draft_vocab_size)[0]
                out.cpu();torch.npu.synchronize()
                elapsed=(time.perf_counter()-started)*1000
                if not runner.last_target_execution.used_aclgraph:raise RuntimeError('component timing requires actual graph output')
                values[name].append(elapsed)
        after=engine.graph_metrics()
        return dict(status='bounded_component_timed',validated=True,entries=len(entries),samples_ms=values,
                    median_ms={k:statistics.median(v)for k,v in values.items()},
                    scope='prebuilt inputs/metadata through original model+greedy graph and CPU outputs; no prefill/scheduler/transport; no Goodput inference',
                    counter_delta={k:after[k]-v for k,v in before.items()if type(v)is int and type(after.get(k))is int})
    timing=stage('static240/interleaved-existing-graph-component-timing',benchmark)
    if not target:return None
    return dict(schema='ROOT240.fixed-query-fixed-KV-shape-real-prefix-prototype.v1',zero_page=zero,
                constant_M=80,constant_attention_segments=80,constant_query_lengths=[1]*80,constant_KV_extent=4096,
                exact_original_visibility_masks_preserved=True,unallocated_tail_pages_mapped_to_private_zero_page=True,
                comparison=comparison,heads={k:v['heads'].cpu().tolist()for k,v in versions.items()},
                executions=executions,timing=timing,weights_and_original_greedy_unchanged=True,
                attention_execution_geometry_changed=True,whole_workload_AR_qualified=False,formal_Goodput=False,adopted=False)
