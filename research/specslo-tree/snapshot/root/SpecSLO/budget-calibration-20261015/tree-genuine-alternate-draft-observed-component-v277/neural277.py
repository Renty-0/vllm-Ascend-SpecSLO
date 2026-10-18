"""Independent genuine0.6B primary and alternate-path numerical oracle."""
import copy
from types import SimpleNamespace

def need(ok,message):
    if not ok:raise RuntimeError('ROOT277 '+message)

def top2(engine,hidden):
    import torch
    logits=engine.model.compute_logits(hidden)[:,:engine.draft_vocab_size]
    confidence,tokens=torch.softmax(logits.float(),dim=-1).topk(2,dim=-1)
    return tokens.long()*1_000_001+torch.round(confidence.clamp(0,1)*1_000_000).long(),tokens[:,0]

def allocate(engine,prefixes):
    need(engine.cache_allocation is None,'fresh independent draft cache lease')
    prompts=[list(p)for _ in range(2)for p in prefixes]
    engine._allocate_cache(prompts,enable_prefix_caching=False,reserve_sequence_capacity=True)
    for owner,p in enumerate(prompts):engine._ensure_cache_capacity([owner],[len(p)+8])
    for owner in range(500,524):engine._activate_cache_sequence(owner,[0],enable_prefix_caching=False)
    need(len(engine.cache_allocation.block_tables)==527,'original527 storage rows')
    pages=[p for row in engine.cache_allocation.block_tables for p in row if p>=0]
    need(len(pages)==len(set(pages))<=544,'unique original physical allocations')
    return dict(candidate_owners=list(range(12)),reference_owners=list(range(12,24)),
        scratch_owners=list(range(500,524)),pages=len(pages),candidate_prefix_copied=False)

def prefill(engine,owner,prompt):
    import torch
    h=engine._run_packed_hidden(prompt,[owner]*len(prompt),list(range(len(prompt))),
        use_aclgraph=False,use_fused_infer_attention=True).clone()
    need(bool(torch.isfinite(h).all()),'finite independently evaluated prefix')
    return int(engine.model.compute_greedy_tokens(h[-1:],engine.draft_vocab_size).item())

def case(engine,native,prefixes,roots,n,steps,variant,state):
    import torch
    from draft_observer_v146 import actual_draft_observer,_used_KV
    from dual_bank_v146 import _arrays,_slots,query_KV,prefix_evidence,tensor_hash
    from next4_209 import plans_for
    from producer277 import generate
    from chain_metadata import build
    from tail_hidden_observer import snapshot
    owners=list(range(n));references=list(range(12,12+n));ps=[len(p)for p in prefixes[:n]]
    current=[(roots[i]+variant)%engine.draft_vocab_size for i in owners]
    histories=[list(p)+[t]for p,t in zip(prefixes[:n],current,strict=True)]
    source=dict(ids=owners,contexts=[dict(owner=i,token_ids=h)for i,h in zip(owners,histories,strict=True)])
    engine._dynamic_tree_known_KV={i:dict(safe_before=p,tokens={})for i,p in zip(owners,ps,strict=True)}
    engine._defer_tree_materialization=engine._defer_normal_tree_materialization=True
    plans=plans_for(histories,[4]*n)
    with actual_draft_observer(engine,source)as observed:
        original=engine.draft_tree_forward(plans,current,owners,
            committed_catchup_token_ids=[h[-5:]for h in histories])
    main_before=_used_KV(engine,owners,ps,[p+4 for p in ps])
    primary_encoded=(observed['tail']['encoded']if observed['seed']is None else
        torch.cat((observed['seed'][:,None,:],observed['tail']['encoded']),dim=1))
    primary_hidden=((observed['tail']['hidden'])if observed['first']is None else
                    (observed['first']['hidden'],*observed['tail']['hidden']))
    tables=copy.deepcopy(engine.cache_allocation.block_tables);ledger=copy.deepcopy(engine._dynamic_tree_known_KV)
    candidate=generate(engine,owners,ps,current,original,copy_state=state,steps=steps)
    runner=engine.graph_runner;key=candidate['actual_draft_entry_key'];args=candidate['metadata']
    h,_,_=snapshot(runner,key,steps)
    slots=torch.tensor([slot for row in candidate['description']['rows']for slot in row['slots'][:steps]],dtype=torch.long,device=engine.device)
    actual_KV=query_KV(engine,slots);encoded=candidate['encoded'].clone()
    changed=args[0].clone();changed[0]=(changed[0]+1)%engine.draft_vocab_size
    runner.run_draft_greedy(changed,*args[1:],engine.draft_vocab_size,valid_row_count=n,return_confidence=True)
    restored=runner.run_draft_greedy(*args,engine.draft_vocab_size,valid_row_count=n,return_confidence=True).clone()
    rh,_,_=snapshot(runner,key,steps)
    matcher=runner._target_outputs_match
    need(torch.equal(encoded,restored)and matcher(h,rh)and matcher(actual_KV,query_KV(engine,slots)),
        'alternate changed/restored exact encoded, allhidden and all56 KV')
    entry=runner.draft_entries[key]
    need(entry.runtime_validated is True and entry.validated_real_row_count>=n and key not in runner.disabled_entry_keys,
        'original Native changed-input and fullrealrow gate')
    need(engine.cache_allocation.block_tables==tables and engine._dynamic_tree_known_KV==ledger,
        'alternate never changes canonical tables or known-KV ledger')
    need(matcher(main_before,_used_KV(engine,owners,ps,[p+4 for p in ps])),
        'all56 canonical primary KV preserved by alternate and controls')
    # Evaluate the same physical bucket on independently prefilled owners.
    bucket=args[0].numel();inp=torch.tensor(current,dtype=torch.long,device=engine.device)
    positions,meta=engine._prepare_attention_metadata(references,ps,False)
    inp,positions,meta=runner._pad_inputs(inp,positions,meta,bucket)
    hh=engine.model(inp,positions,meta).clone();ee,tokens=top2(engine,hh)
    ph=[hh];pe=[ee]
    _,pp,mm=build(engine,[SimpleNamespace(prefix_len=p+1)for p in ps],references,[0]*n,3,bucket)
    for pos,meta in zip(pp,mm,strict=True):
        hh=engine.model(tokens,pos,meta).clone();ph.append(hh);ee,tokens=top2(engine,hh);pe.append(ee)
    primary_reference=torch.stack(pe,dim=1)
    need(torch.equal(primary_encoded,primary_reference)and matcher(tuple(primary_hidden),tuple(ph))
        and matcher(main_before,_used_KV(engine,references,ps,[p+4 for p in ps])),
        'independently evaluated samebucket primary allhidden/56KV/exacttop2')
    # Start from the independent root's second token. Its subsequent states
    # are evaluated on its own canonical pages; no private or candidate KV.
    alternate=(primary_reference[:,0,1]//1_000_001).clone()
    alternate[n:]=0  # Original PA padding supplies zero initial dummy tokens.
    _,pp,mm=build(engine,[SimpleNamespace(prefix_len=p+1)for p in ps],references,[0]*n,steps,bucket)
    ah=[];ae=[];tokens=alternate
    for pos,meta in zip(pp,mm,strict=True):
        hh=engine.model(tokens,pos,meta).clone();ah.append(hh);ee,tokens=top2(engine,hh);ae.append(ee)
    reference=torch.stack(ae,dim=1)
    reference_slots=torch.cat([_slots(engine,i,range(p+1,p+1+steps))for i,p in zip(references,ps,strict=True)])
    ref_KV=query_KV(engine,reference_slots)
    need(torch.equal(encoded,reference)and matcher(h,tuple(ah))and matcher(actual_KV,ref_KV),
        'independent alternate encoded/allphysical hidden/all56 queryKV')
    need(all(torch.equal(a,b)for a,b in zip(actual_KV,ref_KV,strict=True))
        and all(torch.equal(a,b)for a,b in zip(h,ah,strict=True)),
        'bitwise alternate hidden and all56 layer KV')
    rows=candidate['output']['draft_token_ids']
    need(all(row[:5]==old[:5]and row[5:5+steps]==(reference[i,:,0]//1_000_001).cpu().tolist()
        for i,(row,old)in enumerate(zip(rows,original['draft_token_ids'],strict=True))),
        'actual original primary and rootsecond retained, genuine continuation tokens')
    return dict(N=n,steps=steps,variant=variant,prefixes=ps,bucket=bucket,
        exact_primary_top2=True,primary_allhidden_and56KV=True,
        alternate_allhidden_bitwise=True,alternate_all56KV_bitwise=True,
        exact_alternate_top2=True,changed_restored_exact=True,
        canonical_mainKV_ledger_and_tables_unchanged=True,
        original_Native_runtime_validated=entry.runtime_validated,
        actual_entry=repr(key),encoded_SHA=tensor_hash(encoded),
        genuine_output_rows=rows,all_prefixes_checked_by_caller=True)


def run(engine,native,stage,output,fixture):
    from pathlib import Path
    from runtime277 import save
    from dual_bank_v146 import prefix_evidence
    selected=[list(fixture['prefixes'][i])for i in range(12)]
    active=engine.is_draft
    lease=stage('ROOT277/allocate-independent-draft',lambda:allocate(engine,selected)if active else None)
    roots=[]
    for owner in range(24):
        roots.append(stage(f'ROOT277/independent-prefill-{owner}',
            lambda:prefill(engine,owner,selected[owner%12])if active else None))
    def before():
        if not active:return None
        need(roots[:12]==roots[12:],'independent prefill roots agree')
        histories={i:selected[i%12]for i in range(24)}
        return prefix_evidence(engine,histories)
    protected=stage('ROOT277/protect-all24-prefixes',before)
    from raw_probe277 import run as raw_probe
    raw=stage('ROOT277/draft1024-raw-bit-copy',lambda:raw_probe(engine)if active else None)
    from copy277 import PrivateCopy
    state=stage('ROOT277/bind-persistent-copy-graph',lambda:PrivateCopy(engine)if active else None)
    stage('ROOT277/retain-raw-before-neural',lambda:save(output,f'rank{engine.rank}-pre-neural.json',
        dict(rank=engine.rank,raw_copy=raw,lease=lease,prefixes_protected=True,neural_cases_completed=0)))
    results=[]
    for n in(1,2,4,8,12):
        for steps in(1,2,3):
            for variant in(0,1,0):
                row=stage(f'ROOT277/N{n}-steps{steps}-case{len(results)}',
                    lambda:case(engine,native,selected,roots,n,steps,variant,state)if active else None)
                def retain():
                    if not active:return
                    need(prefix_evidence(engine,{i:selected[i%12]for i in range(24)})==protected,
                        'all24 independently committed prefixes remain exact')
                    save(output,f'draft-case-{len(results):03d}.json',row)
                stage('ROOT277/retain-current-case',retain);results.append(row)
    stage('ROOT277/release-bank',lambda:engine._release_cache()if active else None)
    return dict(rank=engine.rank,cases=results,lease=lease,raw_copy=raw,
        case_count=45,NN_draft_executed=active,whole_workload_AR_qualified=False,
        original201_startup_inventory_run=False,Goodput=None,adopted=False)
