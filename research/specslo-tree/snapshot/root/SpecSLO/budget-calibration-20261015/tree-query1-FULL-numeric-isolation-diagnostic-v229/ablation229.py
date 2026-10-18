"""Fresh-prefix, eager-only factorial M/Q observations; no graph timing claim."""
def run(engine,native,stage,output,fixture):
    import torch
    from fresh229 import prepare_frame
    from dual_bank_v146 import prefix_evidence,query_KV,_slots,tensor_hash
    target=not engine.is_draft;results=[]
    for case in fixture['cases']:
        lane=lambda name,fn:stage('ROOT229ablation/'+case['name']+'/'+name,fn)
        def allocate():
            if not target:return None
            if engine.cache_allocation is not None:raise RuntimeError('independent fresh ablation lease')
            engine._allocate_cache([r['prefix']for r in case['rows']],enable_prefix_caching=False,reserve_sequence_capacity=True)
            for i,r in enumerate(case['rows']):engine._ensure_cache_capacity([i],[len(r['prefix'])+9])
            for i in range(463,527):engine._activate_cache_sequence(i,[0],enable_prefix_caching=False)
        lane('fresh-independent-prompt-lease',allocate)
        for owner,row in enumerate(case['rows']):
            lane('original-eager-prefix'+str(owner),lambda:engine._run_packed_hidden(row['prefix'],[owner]*len(row['prefix']),list(range(len(row['prefix']))),use_aclgraph=False,use_fused_infer_attention=True)if target else None)
        history={case['affected_cache_owner']:case['rows'][case['affected_cache_owner']]['prefix']}
        before=lane('original-committed-prefix',lambda:prefix_evidence(engine,history)if target else None)
        variants=[];private=[]
        controls=[('q1_M40_B5',0,False,True),('q1_M80_B5',0,True,True),('q1_M80_B4',1,True,True),('q5_M80_B5',0,True,False),('q5_M80_B4',1,True,False),('q1_M40_B5_REPEAT',2,False,True)]
        for name,index,fixed,split in controls:
            one=lambda label,fn:lane(name+'/'+label,fn)
            prepared=one('original-metadata-with-explicit-M-Q',lambda:prepare_frame(engine,case,case['frames'][index],fixed_M=fixed,split_queries=split)if target else None)
            h=one('original-model-eager-only',lambda:engine.model(prepared['inputs'],prepared['positions'],prepared['metadata']).clone()if target else None)
            tokens=one('original-global-greedy',lambda:engine.model.compute_greedy_tokens(h,engine.draft_vocab_size).cpu().tolist()if target else None)
            def observe():
                if not target:return None
                if not bool(torch.isfinite(h).all())or prefix_evidence(engine,history)!=before:raise RuntimeError('finite eager math and original prefix unchanged')
                q=prepared['query_row'];vector=h[q].clone();kv=query_KV(engine,_slots(engine,prepared['affected_owner'],[int(prepared['positions'][q])]))
                private.append((vector,kv));return dict(name=name,token=tokens[q],shape=prepared['actual_shape'],query_hidden=tensor_hash(vector),query_KV=[tensor_hash(t)for t in kv],eager_only=True)
            variants.append(one('retain-observation',observe))
        def compare():
            if not target:return None
            pairs=[]
            for i in range(len(private)):
                for j in range(i+1,len(private)):
                    a,ak=private[i];b,bk=private[j]
                    bad=[n for n,(x,y)in enumerate(zip(ak,bk,strict=True))if not torch.equal(x,y)]
                    pairs.append(dict(a=variants[i]['name'],b=variants[j]['name'],hidden_equal=torch.equal(a,b),hidden_max_abs=float((a.float()-b.float()).abs().max()),KV_equal=not bad,first_different_KV_layer=bad[0]//2 if bad else None))
            return pairs
        pairs=lane('independent-M-Q-comparisons',compare)
        result=dict(case=case['name'],request_id=case['request_id'],variants=variants if target else[],pairs=pairs,eager_only=True,original_sampler_unchanged=True,no_performance_claim=True)
        results.append(result)
        from runtime229 import save
        save(output,f'rank{engine.rank}-ablation-case{len(results)}.json',result)
        lane('release-independent-lease',lambda:engine._release_cache()if target else None)
    return results
