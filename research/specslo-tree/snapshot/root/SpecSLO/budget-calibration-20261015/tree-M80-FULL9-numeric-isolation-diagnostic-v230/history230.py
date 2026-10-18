"""Replay known common histories from original prompt cohorts; no GP claim."""
from dataclasses import asdict

def need(value,message):
    if value is not True:raise RuntimeError('ROOT230history: '+message)

def query(engine,frame):
    import torch
    for seq,p in zip(frame['ids'],frame['positions'],strict=True):engine._ensure_cache_capacity([seq],[p])
    from metadata230 import pack
    inputs,positions,metadata=pack(engine,frame,full=True)
    need(metadata.tree_attention and metadata.use_fused_infer_attention,'explicit canonical FULL9 path')
    result=engine.graph_runner.run_tree_hidden(inputs,positions,metadata).clone()
    execution=asdict(engine.graph_runner.last_target_execution)
    need(execution['mode']in('capture_replay','replay')and execution['fallback_reason']is None,'original naturally validated ACLGraph')
    need(bool(torch.isfinite(result).all()),'all incremental hidden finite')
    return result,execution

def run(engine,native,stage,output,fixture):
    import torch
    from dual_bank_v146 import prefix_evidence,query_KV,_slots,tensor_hash
    target=not engine.is_draft;results=[]
    for case in fixture['cases']:
        lane=lambda label,fn:stage('ROOT230history/'+case['name']+'/'+label,fn)
        observations=[];private=[]
        for mode in('candidate','reference'):
            frames=case[mode+'_frames'];query_row=case[mode+'_query_row'];owner=case['affected_owner']
            step=lambda label,fn:lane(mode+'/'+label,fn)
            def allocate():
                need(engine.cache_allocation is None,'fresh independent original prompt allocation')
                engine._allocate_cache(case['prompts'],enable_prefix_caching=False,reserve_sequence_capacity=True)
                for seq,n in enumerate(case['reserve_lengths']):engine._ensure_cache_capacity([seq],[n-1])
                for seq in range(463,527):engine._activate_cache_sequence(seq,[0],enable_prefix_caching=False)
                pages=[v for row in engine.cache_allocation.block_tables for v in row if v>=0]
                need(len(pages)==len(set(pages))<=case['page_upper']<=544,'unique bounded replay pages')
                return dict(pages=len(pages),rows=len(engine.cache_allocation.block_tables))
            allocation=step('allocate-original32-prompt-cohort',allocate)
            states=[native.PearlPipelineState(list(p),len(p),max_tokens=256,ignore_eos=True)for p in case['prompts']]
            roots=step('original-target-batch-prefill',lambda:engine._prefill_and_sample_target_batch(case['prompts'],states,list(range(32))))
            step('same-original-first-roots',lambda:need(list(roots)==case['original_first_roots'],'exact recorded original first roots'))
            final=None
            for ordinal,frame in enumerate(frames):
                hidden=step('causal80-step'+str(ordinal),lambda:query(engine,frame)if target else None)
                if frame['final']:final=hidden
                else:del hidden
            def exact_eager_reference():
                if not target:return None
                from metadata230 import pack
                inputs,pos,meta=pack(engine,frames[-1],full=True)
                index=_slots(engine,owner,[frames[-1]['positions'][query_row]])
                graph_KV=query_KV(engine,index)
                hidden=engine.model(inputs,pos,meta).clone()
                matched=engine.graph_runner._target_outputs_match((hidden,),(final[0],))
                need(matched,'all80 current exact-length FULL hidden pass original Native matcher')
                eager_KV=query_KV(engine,index)
                return dict(original_matcher_passed=matched,all80_hidden_bitwise_equal=torch.equal(hidden,final[0]),
                    all80_hidden_max_abs=float((hidden.float()-final[0].float()).abs().max()),
                    query_KV_bitwise_equal=all(torch.equal(a,b)for a,b in zip(graph_KV,eager_KV,strict=True)),
                    identical_exact_metadata_used=True,bucket_helper_used=False)
            eager_match=step('exact-current-real16-same-metadata-eager-model-reference',exact_eager_reference)
            def retain():
                if not target:return None
                h,execution=final;position=frames[-1]['positions'][query_row]
                common=case['common_context'];need(position==len(common)-1 and frames[-1]['tokens'][query_row]==common[-1],'same actual common last input')
                evidence=prefix_evidence(engine,{owner:common})
                kv=query_KV(engine,_slots(engine,owner,[position]));vector=h[query_row].clone()
                private.append(dict(hidden=vector,KV=kv,prefix=evidence))
                return dict(mode=mode,query_position=position,query_token=common[-1],
                    model_hidden=tensor_hash(vector),common_prefix=evidence,graph_execution=execution,
                    frames=len(frames),allocation=allocation,query_row=query_row,original_first_roots_equal=True,
                    current_real16_exact_metadata_original_matcher_passed=eager_match['original_matcher_passed'],same_metadata_eager_graph_observation=eager_match)
            observation=step('retain-complete-common-prefix-and-query',retain)
            heads=step('original-global-greedy',lambda:engine.model.compute_greedy_tokens(final[0],engine.draft_vocab_size).cpu().tolist()if target else None)
            logits=step('original-global-BF16-logits',lambda:engine.model.compute_logits(final[0])[:,:engine.draft_vocab_size].clone()if target else None)
            from precision_ties230 import greedy as refined_greedy
            refined=step('uninstalled-FP32-tie-refinement',lambda:refined_greedy(engine.model.lm_head,final[0],engine.draft_vocab_size)if target else None)
            def scores():
                if not target:return None
                need(heads[query_row]==int(logits[query_row].argmax()),'original distributed and full-logit argmax agree')
                need(refined[1]['original'].cpu().tolist()==heads,'observer recomputes unchanged complete original head')
                observation.update(diagnostic_refined_token=int(refined[0][query_row]),diagnostic_tie_count=int(refined[1]['global_tie_count'][query_row]),diagnostic_tie_overflow=bool(refined[1]['overflow'][query_row]))
                values,ids=logits[query_row].topk(8)
                observation.update(token=heads[query_row],BF16_top8=dict(tokens=ids.cpu().tolist(),values=values.float().cpu().tolist()))
            step('retain-original-head',scores)
            last=frames[-1];altered=dict(last,tokens=list(last['tokens']))
            for i in range(query_row+1,5*(query_row//5+1)):altered['tokens'][i]=151643
            alternate=step('same-query-with-future-canary',lambda:query(engine,altered)if target else None)
            def canary():
                if not target:return None
                need(torch.equal(final[0][query_row],alternate[0][query_row]),'future tokens do not change same-query hidden')
                need(prefix_evidence(engine,{owner:case['common_context']})==private[-1]['prefix'],'future canary does not change complete common-prefix KV')
                observation['future_canary_query_hidden_and_common_KV_exact']=True
            step('future-canary-exact-common-prefix',canary)
            observations.append(observation)
            if not(mode=='reference'and case['original_owner']==34):step('release-independent-lease',engine._release_cache)
            del final,hidden,alternate,logits
        def compare():
            if not target:return None
            a,b=private
            different=[i for i,(x,y)in enumerate(zip(a['KV'],b['KV'],strict=True))if not torch.equal(x,y)]
            pd=[i for i,(x,y)in enumerate(zip(a['prefix']['KV'],b['prefix']['KV'],strict=True))if x!=y]
            tokens=sorted({case['actual_candidate_token'],case['actual_reference_token'],
                *(t for r in observations for t in r['BF16_top8']['tokens'])})
            head=engine.model.lm_head;local=[i for i in tokens if head.vocab_start<=i<head.vocab_end]
            for row,values in zip(observations,private,strict=True):
                if local:
                    index=torch.tensor([i-head.vocab_start for i in local],dtype=torch.long,device=engine.device)
                    dots=(head.weight.index_select(0,index).float()*values['hidden'].float()[None,:]).sum(-1,dtype=torch.float32).cpu().tolist()
                    row['diagnostic_FP32_scalar_dots_local_shard']=dict(zip(map(str,local),dots,strict=True))
                else:row['diagnostic_FP32_scalar_dots_local_shard']={}
            return dict(routes=observations,hidden_equal=torch.equal(a['hidden'],b['hidden']),
                hidden_max_abs=float((a['hidden'].float()-b['hidden'].float()).abs().max()),
                query_KV_different_indices=different,query_first_different_layer=different[0]//2 if different else None,
                complete_common_prefix_KV_different_indices=pd,prefix_first_different_layer=pd[0]//2 if pd else None,
                actual_recorded_candidate_token=case['actual_candidate_token'],actual_recorded_reference_token=case['actual_reference_token'])
        detail=lane('same-prompt-common-history-two-routes',compare)
        results.append(dict(case=case['name'],request_id=case['request_id'],neural_math_on_this_rank=target,detail=detail,
            teacher_forced_common_history_diagnostic=True,not_exact_production_unaccepted_future_inputs=True,all_decode_attention_metadata_M80_FULL9=True,
            independent_AR_qualification=False,performance_qualification=False))
        from runtime230 import save
        save(output,f'rank{engine.rank}-history-complete-case{len(results)}.json',results[-1])
        if case['original_owner']==34:
            from body_cost230 import measure
            cost=measure(engine,case['reference_frames'][-1],lambda label,fn:lane('separate-post-numeric-cost/'+label,fn))
            if target:results[-1]['detail']['routes'][-1]['dynamic_attention_body_cost']=cost
            lane('release-last-reference-after-separate-cost',engine._release_cache)
    return dict(schema='ROOT230history.incremental-common-history-numeric.v1',rank=engine.rank,cases=results,
        no_performance_claim=True,no_sampler_changed=True,source_config_unchanged=True,
        prefill201_and_production_qualification_not_run=True,formal_Goodput=False,adopted=False)
