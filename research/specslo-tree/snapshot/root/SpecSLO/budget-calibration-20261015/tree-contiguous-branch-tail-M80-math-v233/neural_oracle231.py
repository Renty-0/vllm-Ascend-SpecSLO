"""Additional same-engine target math oracle; not a serving performance run.

Requires the actual canonical production pad/metadata hooks to be active.
Reference paths are independently prefilled and teacher-force only the
explicit synthetic tree inputs, never candidate KV or hidden states.
"""
def need(v,m):
    if v is not True:raise RuntimeError('ROOT231oracle: '+m)

def cases():
    from mapper231 import envelope,repair_Q79
    result=[]
    for n in range(1,17):
        patterns=[('all1',[1]*n),('all4',[4]*n),('fork5',[5]*n),('mixed',[(i%5)+1 for i in range(n)])]
        for label,raw in patterns:
            budgets=list(raw)
            while n+sum(budgets)>80:
                j=next(i for i in reversed(range(n))if budgets[i]>1);budgets[j]-=1
            delta=None
            family='full9'if 5 in budgets else'causal';m,s,pad=envelope(n,n+sum(budgets),family)
            for anchor in range(4)if 5 in budgets else(0,):
                result.append(dict(name=f'N{n}-{label}-anchor{anchor}',N=n,budgets=budgets,anchor=anchor,M=m,S=s,padding=pad,plan_family=family,query_count=n+sum(budgets),fixture_budget_reduction=delta))
    for label,budgets in [('tail67',[3]*10+[4]*4+[1,4]),('tail78',[4]*14+[3]*2),('padding79',[4]*15+[3])]:
        delta=None
        m,s,pad=envelope(16,16+sum(budgets),'causal')
        result.append(dict(name='N16-'+label,N=16,budgets=budgets,anchor=0,M=m,S=s,padding=pad,plan_family='causal',query_count=16+sum(budgets),fixture_budget_reduction=delta))
    for anchor in range(4):
        budgets=[5]*12+[2,1,1];m,s,pad=envelope(15,79,'full9')
        result.append(dict(name=f'N15-fork-padding79-anchor{anchor}',N=15,budgets=budgets,anchor=anchor,M=m,S=s,padding=pad,plan_family='full9',query_count=79,fixture_budget_reduction=None))
    return result

def prepare(case):
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan,pack_selected_tree_plan
    raw=cached_cpu_tree_speculation_plan(2,4,127,4096,candidate_budget=8)
    plans=[];rows=[]
    for i,b in enumerate(case['budgets']):
        selected=list(range(min(b,4)))+([4+case['anchor']]if b==5 else[])
        plans.append(pack_selected_tree_plan(raw,selected))
        rows.append([103+17*i+29*node for node in selected])
    return plans,rows

def candidate(engine,case,bank,plans,rows):
    import torch
    from dual_bank_v146 import query_KV
    from host_target_metadata import make_host_tree_attention_metadata
    from fixed_mask_extent import padded
    roots=bank['roots'][:case['N']];ids=list(range(case['N']))
    packed=engine._pad_spec_rhythm_target_tree_graph(plans,roots,rows,ids,[],[])
    pp,rr,dd,ss,real=packed
    need(real==case['N']and len(pp)==case['S']and sum(p.candidate_budget+1 for p in pp)==80,'actual canonical production padding callback')
    need(all(a is b for a,b in zip(pp[:real],plans,strict=True))and rr[:real]==roots and dd[:real]==rows and ss[:real]==ids,'original real plan, input and owner identity retained')
    out=engine.target_tree_forward(pp,rr,dd,sequence_ids=ss,return_logits=False,real_tree_count=real)
    need(out['used_aclgraph']is True and out['used_tree_attention']is True and out['query_count']==80 and out['logical_query_count']==case['query_count'],'actual original Native target forward under FULL9 canonical scope')
    need(out['num_draft_tokens']==case['budgets'],'logical retained budgets are not padded candidates')
    graph_tokens=out['target_query_token_ids'].clone();graph_KV=query_KV(engine,out['cache_slot_mapping'].long())
    state=engine._branch_tail_state
    inputs,pos,meta,enabled=state.last
    baseline=state.last_record.get('baseline_original_layout',False)
    from branch_tail231 import is_fork
    S=case['S']+(sum(is_fork(p)for p in pp)if enabled else 0);Q=10 if enabled else 9
    need(tuple(meta.tree_attention_mask.shape)==(S,1,Q,4096)and meta.actual_seq_lengths_q[-1]==80,'actual original-order canonical FULL metadata ABI')
    hidden=state.eager_last()
    eager_tokens=engine.model.compute_greedy_tokens(hidden,engine.draft_vocab_size)[:case['query_count']].clone()
    eager_KV=query_KV(engine,meta.slot_mapping[:case['query_count']].long())
    need(bool(torch.isfinite(hidden).all())and torch.equal(graph_tokens,eager_tokens),'complete real graph/eager original greedy agreement')
    need(engine.graph_runner._target_outputs_match(tuple(graph_KV),tuple(eager_KV)),'all128 graph/eager actual query-KV tensors pass original matcher')
    return dict(branch_layout=state.last_record,attention_segments=S,mask_query_capacity=Q,tokens=graph_tokens,KV=graph_KV,hidden=hidden[:case['query_count']].clone(),
        graph_eager_KV_bitwise_equal=all(torch.equal(a,b)for a,b in zip(graph_KV,eager_KV,strict=True)))

def reference(engine,case,bank,plans,rows,stage):
    import torch
    from dual_bank_v146 import query_KV
    from metadata231 import pack_q5
    from query1_reference231 import canonical
    from reference_schedule231 import schedule
    roots=bank['roots'][:case['N']];rounds=schedule(plans,roots,rows)
    offsets=[];cursor=0
    for plan in plans:offsets.append(cursor);cursor+=plan.candidate_budget+1
    tokens=[None]*cursor;hidden_rows={};KV_rows={};frame_reports=[]
    for number,frame in enumerate(rounds):
        def execute():
            for seq,p in zip(frame['ids'],frame['positions'],strict=True):engine._ensure_cache_capacity([seq],[p])
            inputs,pos,meta=pack_q5(engine,frame,full=True);meta=canonical(meta)
            hidden=engine.model(inputs,pos,meta).clone()
            need(bool(torch.isfinite(hidden).all()),'finite independently computed stepwise reference hidden')
            heads=engine.model.compute_greedy_tokens(hidden,engine.draft_vocab_size)
            labels=[r for r in frame['real_queries']if r['candidate_query']is not None]
            head_rows=[r['head_row']for r in labels]
            index=torch.tensor(head_rows,dtype=torch.long,device=engine.device)
            kv=query_KV(engine,meta.slot_mapping.index_select(0,index).long())
            own_heads=heads.index_select(0,index).cpu().tolist()
            for k,(entry,token)in enumerate(zip(labels,own_heads,strict=True)):
                owner,q=entry['candidate_query'];destination=offsets[owner]+q
                need(tokens[destination]is None,'each actual tree node independently evaluated exactly once')
                tokens[destination]=token;hidden_rows[destination]=hidden[entry['head_row']].clone()
                KV_rows[destination]=[v[k].clone()for v in kv]
            return dict(depth=frame['depth'],chunk=frame['chunk'],active_reference_owners=[r['reference_owner']for r in frame['real_queries']],mapped_real_nodes=len(labels),physical_queries=80,reference_eager_only=True,candidate_KV_copied=False)
        frame_reports.append(stage('independent-reference/frame'+str(number),execute))
    need(all(type(t)is int for t in tokens)and sorted(hidden_rows)==list(range(cursor)),'every real candidate node independently covered')
    hidden=torch.stack([hidden_rows[i]for i in range(cursor)])
    kv=[torch.stack([KV_rows[i][a]for i in range(cursor)])for a in range(128)]
    return dict(tokens=torch.tensor(tokens,dtype=torch.long,device=engine.device),hidden=hidden,KV=kv,frames=frame_reports)

def run(engine,native,stage,bank,output=None):
    import torch
    from dual_bank_v146 import prefix_evidence
    target=not engine.is_draft;records=[];negative_controls=[]
    for case in cases():
        lane=lambda label,fn:stage('canonical-math/'+case['name']+'/'+label,fn)
        plans,rows=prepare(case)
        if case['name']=='N1-fork5-anchor0':
            from baseline233 import run as negative
            negative_controls.append(negative(engine,case,bank,plans,rows,lane,output))
        actual=lane('actual-production-target',lambda:candidate(engine,case,bank,plans,rows)if target else None)
        own=reference(engine,case,bank,plans,rows,lane)if target else None
        # All roles must enter the same stage sequence. The reference frames
        # are scheduled from host plans on draft as well, with no model call.
        if not target:
            from reference_schedule231 import schedule
            for number,frame in enumerate(schedule(plans,[0]*case['N'],rows)):
                lane('independent-reference/frame'+str(number),lambda:None)
        def compare():
            if not target:return None
            matcher=engine.graph_runner._target_outputs_match
            from baseline233 import measurement,retain
            metrics=measurement(engine,case,actual,own)
            if not(metrics['tokens_equal']and metrics['hidden_original_matcher_passed']and metrics['KV_original_matcher_passed']):retain(output,f'rank{engine.rank}-FAILED-'+case['name']+'.json',metrics)
            need(torch.equal(actual['tokens'],own['tokens']),'all real original greedy predictions match independent spine/side paths')
            need(matcher((actual['hidden'],),(own['hidden'],)),'all real hidden vectors pass unchanged Native tolerance')
            need(matcher(tuple(actual['KV']),tuple(own['KV'])),'all128 real query-KV tensors pass independent oracle')
            return dict(branch_layout=actual['branch_layout'],attention_segments=actual['attention_segments'],mask_query_capacity=actual['mask_query_capacity'],case=case,real_predictions=case['query_count'],exact_greedy=True,candidate_tokens=actual['tokens'].cpu().tolist(),reference_tokens=own['tokens'].cpu().tolist(),hidden_bitwise_equal=torch.equal(actual['hidden'],own['hidden']),
                KV_bitwise_equal=all(torch.equal(a,b)for a,b in zip(actual['KV'],own['KV'],strict=True)),all128_KV_original_matcher_passed=True,
                graph_eager_KV_bitwise_equal=actual['graph_eager_KV_bitwise_equal'],independent_reference_frames=own['frames'],candidate_KV_copied=False)
        records.append(lane('complete-node-hidden-KV-independent-gate',compare))
        if output is not None:
            def retain():
                import json
                from pathlib import Path
                value=dict(rank=engine.rank,case=case,observation=records[-1],neural_math_on_rank=target,performance_qualified=False)
                encoded=json.dumps(value,allow_nan=False)
                with(Path(output)/f'rank{engine.rank}-canonical-case{len(records):03d}.json').open('x')as stream:stream.write(encoded+'\n')
            lane('retain-completed-case',retain)
    def protected():
        if not target:return None
        for owner,expected in enumerate(bank['evidence']['all_prefix_evidence']):
            now=prefix_evidence(engine,{owner:bank['prefixes'][owner%16]})
            need(now==expected,'all48 own committed prefixes protected across all cases')
        return True
    prefixes=stage('canonical-math/all48-independent-prefixes-still-exact',protected)
    return dict(rank=engine.rank,old_layout_negative_controls=negative_controls,cases=records,all_independent_prefixes_unchanged=prefixes,target_math_on_rank=target,teacher_forced_fixture_only=True,whole_workload_AR_qualified=False,performance_qualified=False,adopted=False)
