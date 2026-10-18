"""Real common-prefix diagnostic only. No accuracy or performance promotion."""
def run(engine, native, stage, output, fixture):
    import torch
    from pathlib import Path
    import json
    from install234 import activate
    from dual_bank_v146 import prefix_evidence, query_KV
    from metadata231 import pack_q5
    from query1_reference231 import canonical
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan
    from dataclasses import replace

    target = not engine.is_draft
    activate_report = stage('real-prefix/activate', lambda: activate(engine, native))
    prompts = fixture['prompts']; n = len(prompts)
    selected = fixture['selected_local_owners']; inputs = fixture['teacher_forced_common_inputs']
    roots = []
    capture = dict(mode=None, row=None)
    attention_records = dict(candidate=[], original_AR=[])
    if target:
        for index, layer in enumerate(engine.model.layers):
            attention = layer.self_attn
            prior = attention._fused_infer_attention
            def recorded(query, meta, *, _prior=prior, _attention=attention, _index=index):
                out = _prior(query, meta)
                if capture['mode'] is not None:
                    row = capture['row']
                    owner = next(i for i, end in enumerate(meta.actual_seq_lengths_q) if row < end)
                    table = meta.request_block_tables[owner]
                    positions = torch.arange(1509, dtype=torch.long, device=engine.device)
                    slots = table.long().index_select(0, positions//128)*128 + positions%128
                    attention_records[capture['mode']].append(dict(layer=_index,
                        query=query[row].clone(), output=out[row].clone(),
                        key=_attention.key_cache.flatten(0,1).index_select(0, slots).clone(),
                        value=_attention.value_cache.flatten(0,1).index_select(0, slots).clone(),
                        query_shape=list(query.shape), sequence_lens=list(meta.sequence_lens)))
                return out
            attention._fused_infer_attention = recorded

    def allocate():
        engine._allocate_cache(prompts * 4, enable_prefix_caching=False, reserve_sequence_capacity=True)
        for seq, p in enumerate(prompts * 4):
            engine._ensure_cache_capacity([seq], [len(p) + 16])
        if target:
            for seq in range(463, 527):
                engine._activate_cache_sequence(seq, [0], enable_prefix_caching=False)
        pages = [v for row in engine.cache_allocation.block_tables for v in row if v >= 0]
        if len(set(pages)) != len(pages) or len(pages) > 544:
            raise RuntimeError('real-prefix independent disjoint cache lease')
        return dict(pages=len(pages), independent_cohorts=4, candidate_KV_seeded=False)

    allocation = stage('real-prefix/allocate', allocate)
    for cohort in range(4):
        ids = list(range(cohort*n, (cohort+1)*n))
        states = [native.PearlPipelineState(list(p), len(p), max_tokens=256, ignore_eos=True) for p in prompts]
        roots.append(stage('real-prefix/prefill-cohort'+str(cohort),
                           lambda ids=ids, states=states: engine._prefill_and_sample_target_batch(prompts, states, ids)))

    def prefix_check():
        if not target: return None
        evidence = [prefix_evidence(engine, {cohort*n+i: prompts[i]}) for cohort in range(4) for i in selected]
        m = len(selected)
        pairs = [dict(owner=fixture['selected_original_owners'][j],
                      roots_equal=roots[0][i] == roots[1][i] == roots[2][i] == roots[3][i] == inputs[j][0],
                      KV_bitwise_equal=evidence[j]['KV'] == evidence[m+j]['KV'] == evidence[2*m+j]['KV'] == evidence[3*m+j]['KV'])
                 for j, i in enumerate(selected)]
        if not all(row['roots_equal'] and row['KV_bitwise_equal'] for row in pairs):
            raise RuntimeError('independently prefilled real-prefix mismatch: '+str(pairs))
        return pairs

    prefix = stage('real-prefix/independent-prefix-check', prefix_check)

    def candidate():
        if not target: return None
        plans = [cached_cpu_tree_speculation_plan(1, 4, len(prompts[i]), 4096, candidate_budget=4) for i in selected]
        rr = [v[0] for v in inputs]; rows = [v[1:] for v in inputs]
        pp, roots_p, dd, ids, real = engine._pad_spec_rhythm_target_tree_graph(plans, rr, rows, selected, [], [])
        out = engine.target_tree_forward(pp, roots_p, dd, sequence_ids=ids, return_logits=False, real_tree_count=real)
        state = engine._branch_tail_state
        x, pos, meta, enabled = state.last
        capture.update(mode='candidate', row=selected.index(fixture['cohort_owners'].index(19))*5+3)
        try: hidden = state.eager_last()
        finally: capture.update(mode=None, row=None)
        heads = engine.model.compute_greedy_tokens(hidden, engine.draft_vocab_size)
        q = len(selected)*5
        if not torch.equal(heads[:q], out['target_query_token_ids']):
            raise RuntimeError('real grouped graph/eager head mismatch')
        grouped = dict(hidden=hidden[:q].clone(), heads=heads[:q].clone(), KV=[v.clone() for v in query_KV(engine, meta.slot_mapping[:q].long())])
        split_meta = canonical(meta)
        with state.active(False):
            split_hidden = engine.model(x, pos, split_meta).clone()
        split = dict(hidden=split_hidden[:q].clone(), heads=engine.model.compute_greedy_tokens(split_hidden, engine.draft_vocab_size)[:q].clone(),
                     KV=[v.clone() for v in query_KV(engine, meta.slot_mapping[:q].long())])
        return grouped, split

    actual = stage('real-prefix/grouped-and-split-candidate', candidate)
    references = []
    for mode in ('grouped', 'split'):
        cohort = 1 if mode == 'grouped' else 2
        snapshots = []
        for depth in range(5):
            def step(depth=depth, mode=mode, cohort=cohort):
                if not target: return None
                ss = [cohort*n+i for i in selected]
                ids=[]; positions=[]; tokens=[]
                for j, i in enumerate(selected):
                    p = len(prompts[i])+depth
                    ids += [ss[j]]*5; positions += list(range(p,p+5)); tokens += [inputs[j][depth]]+[0]*4
                for k in range(16-len(selected)):
                    ids += [463+k]*5; positions += list(range(5)); tokens += [0]*5
                engine._prepare_attention_metadata(ids, positions, True)
                frame = dict(ids=ids, positions=positions, tokens=tokens, selected=ids[::5])
                x, pos, meta = pack_q5(engine, frame, full=True)
                if mode == 'split': meta = canonical(meta)
                with engine._branch_tail_state.active(False):
                    hidden=engine.model(x, pos, meta).clone()
                heads=engine.model.compute_greedy_tokens(hidden, engine.draft_vocab_size)
                idx=torch.tensor(list(range(0,len(selected)*5,5)),dtype=torch.long,device=engine.device)
                return dict(hidden=hidden.index_select(0,idx).clone(), heads=heads.index_select(0,idx).clone(),
                            KV=[v.clone() for v in query_KV(engine,meta.slot_mapping.index_select(0,idx).long())])
            snapshots.append(stage('real-prefix/'+mode+'-reference-depth'+str(depth), step))
        if target:
            # Depth-major independent results into original owner-major node order.
            m=len(selected)
            references.append({key: (torch.stack([snapshots[d][key][j] for j in range(m) for d in range(5)])
                                    if key != 'KV' else [torch.stack([snapshots[d]['KV'][a][j] for j in range(m) for d in range(5)]) for a in range(128)])
                               for key in ('hidden','heads','KV')})

    original_AR_heads = []
    for depth in range(5):
        def original_step(depth=depth):
            if not target: return None
            owners = fixture['original_AR_local_owners']
            ids=[]; positions=[]; tokens=[]
            for j, i in enumerate(owners):
                p=len(prompts[i])+depth
                ids += [3*n+i]*5; positions += list(range(p,p+5))
                tokens += [fixture['original_AR_common_inputs'][j][depth]]+[0]*4
            for j in range(16-len(owners)):
                ids += [463+j]*5; positions += list(range(5)); tokens += [0]*5
            engine._prepare_attention_metadata(ids, positions, True)
            x,pos,meta=pack_q5(engine,dict(ids=ids,positions=positions,tokens=tokens,selected=ids[::5]),full=True)
            if depth == 3: capture.update(mode='original_AR', row=fixture['original_AR_request19_subset_index']*5)
            try:
                with engine._branch_tail_state.active(False): hidden=engine.model(x,pos,meta).clone()
            finally: capture.update(mode=None,row=None)
            heads=engine.model.compute_greedy_tokens(hidden,engine.draft_vocab_size)
            return int(heads[fixture['original_AR_request19_subset_index']*5].item())
        original_AR_heads.append(stage('real-prefix/exact-original-AR-partition-depth'+str(depth), original_step))

    def compare():
        if not target: return None
        versions=dict(candidate_grouped=actual[0], candidate_split=actual[1], reference_grouped=references[0], reference_split=references[1])
        pairs=[]
        for a,b in [('candidate_grouped','reference_grouped'),('candidate_grouped','reference_split'),('candidate_split','reference_split'),('reference_grouped','reference_split')]:
            x,y=versions[a],versions[b]
            pairs.append(dict(a=a,b=b,heads_equal=torch.equal(x['heads'],y['heads']),
                              hidden_equal=torch.equal(x['hidden'],y['hidden']),hidden_max_abs=float((x['hidden'].float()-y['hidden'].float()).abs().max().item()),
                              KV_equal=all(torch.equal(u,v) for u,v in zip(x['KV'],y['KV'],strict=True)),
                              first_KV_difference=next((k for k,(u,v) in enumerate(zip(x['KV'],y['KV'],strict=True)) if not torch.equal(u,v)),None)))
        origin=[]
        a=attention_records['candidate'];b=attention_records['original_AR']
        if len(a)!=len(b) or len(a)!=64: raise RuntimeError('all64 actual attention layer snapshots required')
        for x,y in zip(a,b,strict=True):
            if x['layer']!=y['layer']:raise RuntimeError('same layer identity')
            item=dict(layer=x['layer'],candidate_query_shape=x['query_shape'],reference_query_shape=y['query_shape'],
                      candidate_sequence_lens=x['sequence_lens'],reference_sequence_lens=y['sequence_lens'])
            for key in ('query','output','key','value'):
                item[key+'_equal']=torch.equal(x[key],y[key])
                item[key+'_max_abs']=float((x[key].float()-y[key].float()).abs().max().item())
            origin.append(item)
        owner_row=fixture['selected_original_owners'].index(19)*5
        candidate_heads=actual[0]['heads'][owner_row:owner_row+5].cpu().tolist()
        return dict(pairs=pairs,heads={name:v['heads'].cpu().tolist() for name,v in versions.items()},
                    original_AR_request19_heads=original_AR_heads,candidate_request19_heads=candidate_heads,
                    original_observed_disagreement_reproduced=candidate_heads[3]==358 and original_AR_heads[3]==773,
                    all64_attention_query_output_and_visible_prefix_KV=origin,
                    original_owner_order=fixture['selected_original_owners'],original_sampler_unchanged=True,teacher_forced_shared_prefix_only=True)

    result=stage('real-prefix/compare-all-four-variants',compare)
    stage('real-prefix/release',engine._release_cache)
    return dict(rank=engine.rank,activation=activate_report,allocation=allocation,prefix=prefix,comparison=result,
                model_dtype=None if not target else str(engine.model.embed_tokens.weight.dtype),
                performance=False,whole_workload_AR_qualified=False,adopted=False)
