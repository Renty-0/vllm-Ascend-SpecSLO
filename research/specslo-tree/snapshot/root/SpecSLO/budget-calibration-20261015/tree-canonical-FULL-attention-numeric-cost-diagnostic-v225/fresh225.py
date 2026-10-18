"""Bounded shared-prefix numerical observations, never a performance trial."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import types


def need(value,message):
    if value is not True:raise RuntimeError('ROOT225fresh probes: '+message)


def prepare_frame(engine,case,frame):
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan,pack_selected_tree_plan
    from target_layout import pad_target,_lengths
    from host_target_metadata import make_host_tree_attention_metadata
    from fixed_mask_extent import padded
    plans=[]
    for owner,budget,shift in zip(frame['ids'],frame['budgets'],frame['shifts'],strict=True):
        p=len(case['rows'][owner]['prefix'])+shift
        raw=cached_cpu_tree_speculation_plan(2,4,p,4096,candidate_budget=budget)
        plans.append(pack_selected_tree_plan(raw,list(range(budget))))
    n=len(plans);q=sum(p.candidate_budget+1 for p in plans)
    def envelope(count,queries,family):
        need((count,queries)==(n,q),'exact fixed real layout')
        lengths=_lengths(n,q,frame['M'],frame['S'])
        need(lengths is not None,'legal original dummy partition')
        return frame['M'],frame['S'],lengths
    pad=types.FunctionType(pad_target.__code__,dict(pad_target.__globals__,envelope=envelope))
    pp,roots,rows,ids,real=pad(engine,plans,frame['roots'],frame['futures'],frame['ids'],[],[])
    for owner,plan in zip(ids,pp,strict=True):
        positions=plan.cache_positions.tolist() if getattr(plan,'cache_positions',None)is not None else plan.positions.tolist()
        engine._ensure_cache_capacity([owner]*len(positions),positions)
    inputs,positions,meta=make_host_tree_attention_metadata(engine,pp,roots,rows,sequence_ids=ids,allow_causal_fast_path=False)
    meta=padded(meta,9)
    need(real==n and inputs.numel()==frame['M']and len(meta.actual_seq_lengths_q)==frame['S'],'exact source physical shape')
    row=sum(p.candidate_budget+1 for p in plans[:frame['affected_row']])+frame['query_offset']
    return dict(inputs=inputs,positions=positions,metadata=meta,query_row=row,
        affected_owner=frame['ids'][frame['affected_row']],
        root_position=len(case['rows'][frame['ids'][frame['affected_row']]]['prefix'])+frame['shifts'][frame['affected_row']],
        actual_shape=dict(M=inputs.numel(),S=len(meta.actual_seq_lengths_q),tree=bool(meta.tree_attention),
            q=list(meta.actual_seq_lengths_q),KV=list(meta.sequence_lens),
            block_tables_shape=list(meta.block_tables.shape),
            mask_shape=list((meta.tree_attention_mask if meta.tree_attention else meta.attention_mask).shape)))


def run(engine,native,stage,output,fixture):
    import torch
    from dual_bank_v146 import prefix_evidence,query_KV,_slots,tensor_hash
    target=not engine.is_draft
    results=[]
    for case in fixture['cases']:
        lane=lambda label,fn:stage('ROOT225fresh/'+case['name']+'/'+label,fn)
        def allocate():
            if not target:return None
            need(engine.cache_allocation is None,'fresh diagnostic lease')
            engine._allocate_cache([r['prefix'] for r in case['rows']],enable_prefix_caching=False,reserve_sequence_capacity=True)
            for i,row in enumerate(case['rows']):engine._ensure_cache_capacity([i],[len(row['prefix'])+9])
            for i in range(463,527):engine._activate_cache_sequence(i,[0],enable_prefix_caching=False)
            pages=[p for t in engine.cache_allocation.block_tables for p in t if p>=0]
            need(len(pages)==len(set(pages))<=544,'unique bounded shared-prefix allocation')
            return dict(pages=len(pages),rows=len(engine.cache_allocation.block_tables))
        allocation=lane('allocate-shared-prefix-lease',allocate)
        for owner,row in enumerate(case['rows']):
            hidden=lane('prefill'+str(owner)+'/original-eager-FIA',lambda:engine._run_packed_hidden(
                row['prefix'],[owner]*len(row['prefix']),list(range(len(row['prefix']))),
                use_aclgraph=False,use_fused_infer_attention=True).clone()if target else None)
            lane('prefill'+str(owner)+'/finite',lambda:need(bool(torch.isfinite(hidden).all()),'complete prefix hidden finite')if target else None)
        history={case['affected_cache_owner']:case['rows'][case['affected_cache_owner']]['prefix']}
        before=lane('shared-affected-prefix-fingerprint',lambda:prefix_evidence(engine,history)if target else None)
        native_before=lane('Native-before',engine.graph_metrics)
        observed=[];private=[]
        for frame in case['frames']:
            one=lambda name,fn:lane(frame['name']+'/'+name,fn)
            prepared=one('original-host-metadata',lambda:prepare_frame(engine,case,frame)if target else None)
            eager=one('original-eager-model',lambda:engine.model(prepared['inputs'],prepared['positions'],prepared['metadata']).clone()if target else None)
            def drain_eager():
                if not target:return None
                need(bool(torch.isfinite(eager).all()),'all eager hidden finite')
                need(prefix_evidence(engine,history)==before,'original affected prefix unchanged by eager queries')
                root=query_KV(engine,_slots(engine,prepared['affected_owner'],[int(prepared['positions'][prepared['query_row']])]))
                return root
            eager_root=one('eager-prefix-and-root-KV',drain_eager)
            eager_tokens=one('original-eager-greedy-head',lambda:engine.model.compute_greedy_tokens(eager,engine.draft_vocab_size).clone()if target else None)
            eager_logits=one('original-eager-global-logits',lambda:engine.model.compute_logits(eager)[:,:engine.draft_vocab_size].clone()if target else None)
            graph=one('original-hidden-ACLGraph',lambda:engine.graph_runner.run_tree_hidden(prepared['inputs'],prepared['positions'],prepared['metadata']).clone()if target else None)
            execution=one('original-graph-execution',lambda:asdict(engine.graph_runner.last_target_execution)if target else None)
            def drain_graph():
                if not target:return None
                need(bool(torch.isfinite(graph).all()),'all graph hidden finite')
                need(prefix_evidence(engine,history)==before,'original affected prefix unchanged by graph queries')
                return query_KV(engine,_slots(engine,prepared['affected_owner'],[int(prepared['positions'][prepared['query_row']])]))
            graph_root=one('graph-prefix-and-root-KV',drain_graph)
            graph_tokens=one('original-graph-greedy-head',lambda:engine.model.compute_greedy_tokens(graph,engine.draft_vocab_size).clone()if target else None)
            from precision_ties225 import greedy as refined_greedy
            refined=one('uninstalled-FP32-tie-refinement',lambda:refined_greedy(engine.model.lm_head,graph,engine.draft_vocab_size)if target else None)
            graph_logits=one('original-graph-global-logits',lambda:engine.model.compute_logits(graph)[:,:engine.draft_vocab_size].clone()if target else None)
            def retain():
                if not target:return None
                q=prepared['query_row'];need(torch.equal(refined[1]['original'],graph_tokens),'same original head re-evaluated by observer')
                need(torch.equal(eager_tokens,eager_logits.argmax(-1))and torch.equal(graph_tokens,graph_logits.argmax(-1)),
                    'original distributed greedy equals original complete BF16 logits argmax')
                def top(logits):
                    values,ids=logits[q].topk(8)
                    return dict(tokens=ids.cpu().tolist(),BF16_values=values.float().cpu().tolist())
                e=eager[q].clone();g=graph[q].clone()
                private.append(dict(eager=e,graph=g,eager_root=eager_root,graph_root=graph_root))
                row=dict(name=frame['name'],shape=prepared['actual_shape'],query_row=q,
                    query_position=int(prepared['positions'][q]),query_input_token=int(prepared['inputs'][q]),
                    expected_actual_candidate=case['actual_candidate_token'],expected_actual_reference=case['actual_reference_token'],
                    eager_token=int(eager_tokens[q]),graph_token=int(graph_tokens[q]),
                    diagnostic_refined_token=int(refined[0][q]),diagnostic_tie_count=int(refined[1]['global_tie_count'][q]),diagnostic_tie_overflow=bool(refined[1]['overflow'][q]),
                    eager_top8=top(eager_logits),graph_top8=top(graph_logits),
                    graph_execution=execution,eager_graph_hidden_bitwise_equal=torch.equal(e,g),
                    eager_graph_hidden_max_abs=float((e.float()-g.float()).abs().max()),
                    eager_hidden=tensor_hash(e),graph_hidden=tensor_hash(g),
                    eager_graph_query_KV_equal=all(torch.equal(a,b)for a,b in zip(eager_root,graph_root,strict=True)),
                    query_KV_tensors=len(eager_root),original_affected_prefix_unchanged=True)
                observed.append(row)
                return row
            one('retain-bounded-observation',retain)
            if frame['name']=='FULL40_REPEAT':
                from head_cost225 import measure
                cost=measure(engine,graph,lambda label,fn:one(label,fn))
                if target:observed[-1]['isolated_head_cost']=cost
            del eager,eager_logits,graph,graph_logits
        def comparisons():
            if not target:return None
            tokens=sorted({case['actual_candidate_token'],case['actual_reference_token'],
                *(i for r in observed for key in('eager_top8','graph_top8')for i in r[key]['tokens'])})
            head=engine.model.lm_head
            local=[i for i in tokens if head.vocab_start<=i<head.vocab_end]
            for row,values in zip(observed,private,strict=True):
                dots={}
                for mode in('eager','graph'):
                    if local:
                        index=torch.tensor([i-head.vocab_start for i in local],dtype=torch.long,device=engine.device)
                        weights=head.weight.index_select(0,index).float()
                        dot=(weights*values[mode].float()[None,:]).sum(dim=-1,dtype=torch.float32)
                        dots[mode]=dict(zip(map(str,local),dot.cpu().tolist(),strict=True))
                    else:dots[mode]={}
                row['diagnostic_FP32_scalar_dots_local_shard']=dots
            def kv_difference(a,b):
                different=[i for i,(x,y)in enumerate(zip(a,b,strict=True))if not torch.equal(x,y)]
                return dict(different_tensor_indices=different,first_different_layer=(different[0]//2 if different else None),
                    first_different_kind=('key'if different[0]%2==0 else 'value')if different else None)
            pairs=[]
            for i in range(len(private)):
                for j in range(i+1,len(private)):
                    pairs.append(dict(a=observed[i]['name'],b=observed[j]['name'],
                        eager_hidden_equal=torch.equal(private[i]['eager'],private[j]['eager']),
                        graph_hidden_equal=torch.equal(private[i]['graph'],private[j]['graph']),
                        eager_hidden_max_abs=float((private[i]['eager'].float()-private[j]['eager'].float()).abs().max()),
                        graph_hidden_max_abs=float((private[i]['graph'].float()-private[j]['graph'].float()).abs().max()),
                        same_query_token_and_position=(observed[i]['query_position'],observed[i]['query_input_token'])==(observed[j]['query_position'],observed[j]['query_input_token']),
                        eager_query_KV_equal=all(torch.equal(a,b)for a,b in zip(private[i]['eager_root'],private[j]['eager_root'],strict=True)),
                        graph_query_KV_equal=all(torch.equal(a,b)for a,b in zip(private[i]['graph_root'],private[j]['graph_root'],strict=True)),
                        eager_query_KV_difference=kv_difference(private[i]['eager_root'],private[j]['eager_root']),
                        graph_query_KV_difference=kv_difference(private[i]['graph_root'],private[j]['graph_root'])))
            return dict(frames=observed,pairs=pairs,local_vocab=[head.vocab_start,head.vocab_end],
                scalar_dots_are_diagnostic_only_not_new_sampler=True,all_decode_attention_metadata_forced_FULL9=True,original_affected_prefix_evidence=before)
        detail=lane('cross-variant-precision-and-determinism',comparisons)
        after=lane('Native-after',engine.graph_metrics)
        lane('release-shared-prefix-lease',lambda:engine._release_cache()if target else None)
        results.append(dict(case=case['name'],request_id=case['request_id'],neural_math_on_this_rank=target,
            allocation=allocation,detail=detail,Native_before=native_before,Native_after=after,
            performance_qualification=False,independent_AR_qualification=False))
    return dict(schema='ROOT225fresh.shared-prefix-numeric-diagnostics.v1',rank=engine.rank,cases=results,
        source_config_unchanged=True,prefill201_and_production_qualification_not_run=True,
        no_performance_claim=True,no_sampler_changed=True,formal_Goodput=False,adopted=False)
