"""Independent same-M128 node math, then paired M80/M128 component cost."""
def run(engine,native,stage,output,fixture):
    import statistics,time,torch
    from install234 import activate
    from dual_bank_v146 import prefix_evidence,query_KV
    from frames248 import pack
    target=not engine.is_draft
    activation=stage('M128/activate-original-model-hooks',lambda:activate(engine,native))
    prompts=[list(p)for p in fixture['prompts']]
    shortest=sorted(range(len(prompts)),key=lambda i:len(prompts[i]))[:2]
    prompts.extend(list(prompts[i])for i in shortest)
    n=len(prompts)
    if n!=24:raise RuntimeError('22 original real prompts plus2 explicitly duplicated short diagnostic prefixes')
    roots=[]
    def allocate():
        engine._allocate_cache(prompts*4,enable_prefix_caching=False,reserve_sequence_capacity=True)
        for seq,p in enumerate(prompts*4):engine._ensure_cache_capacity([seq],[len(p)+16])
        if target:
            for seq in range(463,527):engine._activate_cache_sequence(seq,[0],enable_prefix_caching=False)
        pages=[v for row in engine.cache_allocation.block_tables for v in row if v>=0]
        if len(pages)!=len(set(pages))or len(pages)>544:raise RuntimeError('96 disjoint real prefixes and private scratch in original544pages')
        return dict(independent_prefixes=96,pages=len(pages),duplicate_fixture_prompt_indices=shortest,
                    original_full_workload_changed=False,diagnostic_only=True)
    allocation=stage('M128/four-independent-prefill-banks',allocate)
    for cohort in range(4):
        ids=list(range(cohort*n,(cohort+1)*n))
        states=[native.PearlPipelineState(list(p),len(p),max_tokens=256,ignore_eos=True)for p in prompts]
        roots.append(stage('M128/own-prefill'+str(cohort),lambda ids=ids,states=states:engine._prefill_and_sample_target_batch(prompts,states,ids)))
    evidence=None
    def initialize():
        nonlocal evidence
        if not target:return None
        evidence=[prefix_evidence(engine,{cohort*n+i:prompts[i]for i in range(n)})for cohort in range(4)]
        if not all(roots[0]==r for r in roots[1:])or not all(evidence[0]['KV']==v['KV']for v in evidence[1:]):
            raise RuntimeError('four genuinely independent prefilled banks match')
        page=engine.cache_allocation.block_tables[526][0]
        for layer in engine.model.layers:layer.self_attn.key_cache[page].zero_();layer.self_attn.value_cache[page].zero_()
        return dict(all96_independent_prefixes_equal=True,zero_page=page)
    independent=stage('M128/independent-prefix-and-zero-page-check',initialize)
    if target:
        samples=[[roots[0][i],0,0,0,0]for i in range(n)]
        for local,values in zip(fixture['selected_local_owners'],fixture['teacher_forced_common_inputs'],strict=True):samples[local]=list(values)
    else:samples=None
    executions=[];candidate_data={}
    def execute(data,rows):
        x,pos,meta=data;runner=engine.graph_runner
        with engine._branch_tail_state.active(False):
            graph=runner.run_target_greedy([x],[pos],[meta],engine.draft_vocab_size)[0].clone()
            execution=runner.last_target_execution;hidden=engine.model(x,pos,meta).clone()
            eager=engine.model.compute_greedy_tokens(hidden,engine.draft_vocab_size)
        if not torch.equal(graph,eager):raise RuntimeError('original exact graph/eager greedy comparator')
        executions.append(dict(M=x.numel(),mode=execution.mode,used_aclgraph=execution.used_aclgraph,fallback_reason=execution.fallback_reason))
        index=torch.tensor(rows,dtype=torch.long,device=engine.device)
        return dict(heads=graph.index_select(0,index).clone(),hidden=hidden.index_select(0,index).clone(),
                    KV=[v.clone()for v in query_KV(engine,meta.slot_mapping.index_select(0,index).long())])
    def candidate():
        if not target:return None
        candidate_data[128]=pack(engine,list(range(n)),[len(p)for p in prompts],samples,rows=128)
        return execute(candidate_data[128],list(range(n*5)))
    actual=stage('M128/actual120-candidate-nodes',candidate)
    snapshots=[]
    for depth in range(5):
        def one(depth=depth):
            if not target:return None
            data=pack(engine,list(range(n,2*n)),[len(p)+depth for p in prompts],
                      [[samples[i][depth]]+[0]*4 for i in range(n)],rows=128,reference=True)
            return execute(data,list(range(0,n*5,5)))
        snapshots.append(stage('M128/independent-own-prefix-depth'+str(depth),one))
    def compare():
        if not target:return None
        reference={key:(torch.stack([snapshots[d][key][i]for i in range(n)for d in range(5)])if key!='KV'
                       else[torch.stack([snapshots[d]['KV'][a][i]for i in range(n)for d in range(5)])for a in range(128)])
                   for key in('heads','hidden','KV')}
        result=dict(real_nodes=120,heads_equal=torch.equal(actual['heads'],reference['heads']),
            hidden_bitwise_equal=torch.equal(actual['hidden'],reference['hidden']),
            all128_KV_bitwise_equal=all(torch.equal(a,b)for a,b in zip(actual['KV'],reference['KV'],strict=True)),
            hidden_original_matcher=engine.graph_runner._target_outputs_match((actual['hidden'],),(reference['hidden'],)),
            KV_original_matcher=engine.graph_runner._target_outputs_match(tuple(actual['KV']),tuple(reference['KV'])),
            candidate_heads=actual['heads'].cpu().tolist(),reference_heads=reference['heads'].cpu().tolist())
        if not all(result[k]for k in('heads_equal','hidden_bitwise_equal','all128_KV_bitwise_equal','hidden_original_matcher','KV_original_matcher')):
            raise RuntimeError('fixed128 actual120 independent nodes mismatch '+str({k:v for k,v in result.items()if not k.endswith('heads')}))
        after=[prefix_evidence(engine,{cohort*n+i:prompts[i]for i in range(n)})for cohort in range(4)]
        if evidence!=after:raise RuntimeError('all96 independent committed prefixes must remain unchanged')
        result['all96_independent_prefixes_unchanged']=True
        return result
    math=stage('M128/complete-independent120-node-and-prefix-check',compare)
    def benchmark():
        if not target:return None
        candidate_data[80]=pack(engine,list(range(2*n,2*n+16)),[len(p)for p in prompts[:16]],samples[:16],rows=80)
        # The actual Native runner performs its existing changed-input/restore
        # validation. No entry flag is assigned by this prototype.
        execute(candidate_data[80],list(range(80)))
        for rows in(80,128):
            data=candidate_data[rows]
            with engine._branch_tail_state.active(False):engine.graph_runner.run_target_greedy([data[0]],[data[1]],[data[2]],engine.draft_vocab_size)[0].cpu()
        entries=[e for e in engine.graph_runner.target_entries.values()if e.tree_attention_masks and e.tree_attention_masks[0]is not None and tuple(e.tree_attention_masks[0].shape)in((16,1,10,4096),(26,1,10,4096))]
        if len(entries)!=2 or not all(e.runtime_validated for e in entries):raise RuntimeError('both actual naturally validated model graph entries')
        timings={80:[],128:[]};pairs=[];before=engine.graph_metrics()
        for i in range(30):
            order=(80,128)if i%2==0 else(128,80);pair={}
            for rows in order:
                x,pos,meta=candidate_data[rows];torch.npu.synchronize();begin=time.perf_counter()
                with engine._branch_tail_state.active(False):out=engine.graph_runner.run_target_greedy([x],[pos],[meta],engine.draft_vocab_size)[0]
                out.cpu();torch.npu.synchronize();elapsed=(time.perf_counter()-begin)*1000
                if not engine.graph_runner.last_target_execution.used_aclgraph:raise RuntimeError('actual model graph required')
                timings[rows].append(elapsed);pair[str(rows)]=elapsed
            pairs.append(dict(order=order,milliseconds=pair))
        after=engine.graph_metrics()
        delta={k:after[k]-v for k,v in before.items()if type(v)is int and type(after.get(k))is int}
        if any(v for k,v in delta.items()if any(word in k for word in('captures','fallback','failure','disabled'))):raise RuntimeError('no model graph discovery or failures in paired component timing')
        return dict(median_ms={str(k):statistics.median(v)for k,v in timings.items()},pairs=pairs,counter_delta=delta,
            actual_real_candidate_queries={'80':80,'128':120},actual_owners={'80':16,'128':24},
            scope='prebuilt exact metadata, model+greedy graph and CPU output; no scheduler/transport/prefill; no Goodput inference')
    timing=stage('M128/paired-existing-model-graph-cost',benchmark)
    stage('M128/release-independent-banks',engine._release_cache)
    return dict(rank=engine.rank,activation=activation,allocation=allocation,independent=independent,
        comparison=math,timing=timing,executions=executions,physical_M=128,query_partition=[5]*25+[3],
        performance=False,whole_workload_AR_qualified=False,adopted=False)
