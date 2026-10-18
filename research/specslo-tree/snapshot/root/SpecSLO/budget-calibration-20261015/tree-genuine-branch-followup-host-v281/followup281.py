"""Future full-candidate gate: actual next draft after accepted alternate path.

This has not run a model. It must execute in the same live model instance
before production scope, using original Native changed-input gates and a
fresh independent reference bank. All simulated commits are labeled controls.
"""
import copy
from types import SimpleNamespace

def one(engine,native,prefixes,roots,n,accepted,state):
    import torch
    from neural277 import case as initial,need,top2
    from history281 import advance
    from draft_observer_v146 import actual_draft_observer,_used_KV
    from next4_209 import plans_for
    from dual_bank_v146 import _slots,query_KV,prefix_evidence,tensor_hash
    from chain_metadata import build
    from producer277 import generate
    from tail_hidden_observer import snapshot
    first=initial(engine,native,prefixes,roots,n,3,0,state)
    raw=first['genuine_output_rows'];owners=list(range(n));refs=list(range(12,12+n))
    controls=[advance(prefixes[i],roots[i],raw[i],accepted,
        (int(raw[i][4+max(0,accepted-1)])+97+i)%engine.draft_vocab_size)for i in owners]
    histories=[c['history']for c in controls];ps=[c['new_prefix']for c in controls]
    starts=[c['first_missing']for c in controls];current=[h[-1]for h in histories]
    need(all(p<=2555 for p in ps),'explicit entire followup stays inside private prefix domain')
    known_before=copy.deepcopy(engine._dynamic_tree_known_KV)
    protected=prefix_evidence(engine,{i:histories[i][:starts[i]]for i in owners})
    source=dict(ids=owners,contexts=[dict(owner=i,token_ids=h)for i,h in enumerate(histories)])
    plans=plans_for(histories,[4]*n)
    with actual_draft_observer(engine,source)as observed:
        output=engine.draft_tree_forward(plans,current,owners,
            committed_catchup_token_ids=[h[-5:]for h in histories])
    repaired=observed['first']is not None
    need(repaired is(accepted>0),'actual original packed root_seed iff missing accepted branch suffix')
    if repaired:
        root_hidden=observed['first']['hidden'];primary_tail=observed['tail']['hidden']
        encoded=torch.cat((observed['seed'][:,None,:],observed['tail']['encoded']),dim=1)
    else:
        root_hidden=observed['tail']['hidden'][0];primary_tail=observed['tail']['hidden'][1:]
        encoded=observed['tail']['encoded']
    candidate_KV=_used_KV(engine,owners,starts,[p+4 for p in ps])
    candidate_ledger=copy.deepcopy(engine._dynamic_tree_known_KV)
    need(prefix_evidence(engine,{i:histories[i][:starts[i]]for i in owners})==protected,
        'all committed original prompt plus oldroot protected by deferred repair')
    # Independently reconstruct the SAME physical packed repair on reference
    # owners. No candidate KV, hidden tensor or candidate metadata is reused.
    ids=[];positions=[];tokens=[];root_indices=[]
    for owner,c in zip(refs,controls,strict=True):
        ids.extend([owner]*len(c['packed_inputs']));positions.extend(c['packed_positions'])
        tokens.extend(c['packed_inputs']);root_indices.append(len(tokens)-1)
    physical=native._next_linear_draft_graph_bucket(len(tokens),min(128,engine.graph_runner.max_graph_tokens))
    inputs=torch.tensor(tokens,dtype=torch.long,device=engine.device)
    pp,meta=engine._prepare_attention_metadata(ids,positions,False)
    inputs,pp,meta=engine.graph_runner._pad_inputs(inputs,pp,meta,physical)
    reference_root=engine.model(inputs,pp,meta).clone()
    head_bucket=native._next_linear_draft_graph_bucket(n,engine.config.max_num_seqs)
    if repaired:
        selected=reference_root.new_zeros((head_bucket,reference_root.shape[-1]))
        selected[:n].copy_(reference_root.index_select(0,torch.tensor(root_indices,dtype=torch.long,device=engine.device)))
    else:selected=reference_root
    first_encoded,next_tokens=top2(engine,selected);reference_encoded=[first_encoded];reference_hidden=[]
    _,tail_pos,tail_meta=build(engine,[SimpleNamespace(prefix_len=p+1)for p in ps],refs,[0]*n,3,head_bucket)
    for pos,attn in zip(tail_pos,tail_meta,strict=True):
        hidden=engine.model(next_tokens,pos,attn).clone();reference_hidden.append(hidden)
        pair,next_tokens=top2(engine,hidden);reference_encoded.append(pair)
    reference_encoded=torch.stack(reference_encoded,dim=1)
    reference_KV=_used_KV(engine,refs,starts,[p+4 for p in ps])
    matcher=engine.graph_runner._target_outputs_match
    need(torch.equal(encoded,reference_encoded)and matcher((root_hidden,),(reference_root,))
        and matcher(tuple(primary_tail),tuple(reference_hidden))and matcher(candidate_KV,reference_KV),
        'actual deferred k-token repair allphysical hidden/exacttop2/all56KV to independent sameM reference')
    for owner,p,row,root in zip(owners,ps,output['draft_token_ids'],current,strict=True):
        expected=dict(safe_before=p,tokens={p+j:token for j,token in enumerate([root]+list(row[:3]))})
        need(candidate_ledger[owner]==expected,'truthful actual next primary ledger after repair')
    # Also regenerate a genuine alternate path AFTER this actual repair.
    tables=copy.deepcopy(engine.cache_allocation.block_tables)
    alternate=generate(engine,owners,ps,current,output,copy_state=state,steps=3)
    actual_hidden,_,_=snapshot(engine.graph_runner,alternate['actual_draft_entry_key'],3)
    private_slots=torch.tensor([s for r in alternate['description']['rows']for s in r['slots']],dtype=torch.long,device=engine.device)
    private_KV=query_KV(engine,private_slots)
    alternate_input=(reference_encoded[:,0,1]//1_000_001).clone();alternate_input[n:]=0
    pairs=[];hidden_rows=[]
    for pos,attn in zip(tail_pos,tail_meta,strict=True):
        hidden=engine.model(alternate_input,pos,attn).clone();hidden_rows.append(hidden)
        pair,alternate_input=top2(engine,hidden);pairs.append(pair)
    reference_alternate=torch.stack(pairs,dim=1)
    ref_slots=torch.cat([_slots(engine,i,range(p+1,p+4))for i,p in zip(refs,ps,strict=True)])
    ref_KV=query_KV(engine,ref_slots)
    need(torch.equal(alternate['encoded'],reference_alternate)
        and all(torch.equal(a,z)for a,z in zip(actual_hidden,hidden_rows,strict=True))
        and all(torch.equal(a,z)for a,z in zip(private_KV,ref_KV,strict=True)),
        'postrepair real alternate exacttop2/bitwise hidden/all56KV')
    need(engine._dynamic_tree_known_KV==candidate_ledger and engine.cache_allocation.block_tables==tables
        and matcher(candidate_KV,_used_KV(engine,owners,starts,[p+4 for p in ps])),
        'postrepair private alternate protects canonical KV, table and ledger')
    return dict(N=n,accepted_alternate_tokens=accepted,
        teacher_forced_commit_control=True,actual_target_verdict_sample=False,
        old_prefixes=[len(p)for p in prefixes[:n]],new_prefixes=ps,
        actual_root_seed_used=repaired,root_repair_physical_M=physical,root_head_M=head_bucket,
        repair_start_positions=starts,original_packed_root_hidden_checked=True,
        original_next_primary_exacttop2_and_all56KV=True,
        next_genuine_alternate_bitwise_hidden_and56KV=True,
        canonical_prefix_KV_ledger_tables_protected=True,
        actual_known_before=known_before,actual_known_after=candidate_ledger,
        next_encoded_SHA=tensor_hash(encoded),neural_model_executed=True,
        empirical_g_observation=False,Goodput=None)

def run(engine,native,stage,output,fixture,*,save):
    from neural277 import allocate,prefill,need
    from dual_bank_v146 import prefix_evidence
    from copy277 import PrivateCopy
    selected=[list(v[:2550])for v in fixture['prefixes'][:12]];active=engine.is_draft
    lease=stage('ROOT281/allocate-independent-followup-bank',lambda:allocate(engine,selected)if active else None)
    roots=[]
    for owner in range(24):
        roots.append(stage(f'ROOT281/own-prefill{owner}',lambda:prefill(engine,owner,selected[owner%12])if active else None))
    protected=stage('ROOT281/protect-prefixes',lambda:prefix_evidence(engine,{i:selected[i%12]for i in range(24)})if active else None)
    state=stage('ROOT281/bind-private-copy',lambda:PrivateCopy(engine)if active else None)
    if active:engine._ROOT281_qualified_private_copy=state
    rows=[]
    for n in(1,4,8,12):
        for k in range(5):
            row=stage(f'ROOT281/N{n}-accepted{k}',lambda:one(engine,native,selected,roots,n,k,state)if active else None)
            def retain():
                if not active:return
                need(prefix_evidence(engine,{i:selected[i%12]for i in range(24)})==protected,
                    'all24 actual independent committed prefixes protected after followup')
                save(output,f'rank0-branch-followup-{len(rows):03d}.json',row)
            stage('ROOT281/retain-current',retain);rows.append(row)
    stage('ROOT281/release-followup-bank',lambda:engine._release_cache()if active else None)
    return dict(rank=engine.rank,lease=lease,cases=rows,case_count=20,
        teacher_forced_controls_not_actual_target_verdicts=True,
        performance_qualified=False,whole463_AR=False,Goodput=None,adopted=False)
