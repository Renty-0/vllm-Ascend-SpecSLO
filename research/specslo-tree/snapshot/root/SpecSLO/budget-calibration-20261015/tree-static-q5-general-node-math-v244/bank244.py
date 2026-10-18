"""Bounded target-only real16/spine16/side16 neural cache bank.

This is a math qualification fixture; the production workload remains463.
No KV is copied between any candidate/reference owners.
"""
def need(v,m):
    if v is not True:raise RuntimeError('ROOT231bank: '+m)

def allocate(engine,prefixes):
    need(len(prefixes)==16 and all(0<len(p)<=2555 for p in prefixes),'sixteen explicit bounded prefixes')
    need(engine.cache_allocation is None,'fresh original native cache lease')
    prompts=[list(p)for _ in range(3)for p in prefixes]
    engine._allocate_cache(prompts,enable_prefix_caching=False,reserve_sequence_capacity=True)
    for seq,prompt in enumerate(prompts):engine._ensure_cache_capacity([seq],[len(prompt)+8])
    for seq in range(463,527):engine._activate_cache_sequence(seq,[0],enable_prefix_caching=False)
    pages=[page for row in engine.cache_allocation.block_tables for page in row if page>=0]
    need(len(pages)==len(set(pages))==sum((len(v)+8)//128+1 for v in prompts)+64,'48 neural owners times2pages plus64 independent scratch pages')
    need(len(engine.cache_allocation.block_tables)==527,'original storage527 and pool544')
    return prompts,dict(active_owners=48,candidate_owners=list(range(16)),spine_reference_owners=list(range(16,32)),side_reference_owners=list(range(32,48)),scratch_owners=list(range(463,527)),unique_pages=len(pages),pool_pages=544)

def build(engine,native,stage,prefixes):
    import torch
    from dual_bank_v146 import prefix_evidence
    target=not engine.is_draft
    lease=stage('canonical-bank/allocate-real16-spine16-side16',lambda:allocate(engine,prefixes)if target else None)
    roots=[]
    for owner in range(48):
        prefix=list(prefixes[owner%16])
        hidden=stage(f'canonical-bank/prefill{owner}',lambda:engine._run_packed_hidden(prefix,[owner]*len(prefix),list(range(len(prefix))),use_aclgraph=False,use_fused_infer_attention=True).clone()if target else None)
        def head():
            if not target:return None
            need(bool(torch.isfinite(hidden).all()),'own neural prefix hidden finite')
            return int(engine.model.compute_greedy_tokens(hidden[-1:],engine.draft_vocab_size).item())
        roots.append(stage(f'canonical-bank/first-root{owner}',head))
    def check():
        if not target:return None
        evidence=[]
        for owner in range(48):evidence.append(prefix_evidence(engine,{owner:list(prefixes[owner%16])}))
        for i in range(16):
            need(roots[i]==roots[i+16]==roots[i+32],'same original sampler roots on independently prefilled triplets')
            need(evidence[i]['KV']==evidence[i+16]['KV']==evidence[i+32]['KV'],'all128 independent committed-prefix KV tensors bitwise equal')
        return dict(lease=lease[1],independent_neural_prefills=48,all_triplet_roots_equal=True,all_triplet_prefix_KV_equal=True,candidate_KV_copied=False,all_prefix_evidence=evidence)
    evidence=stage('canonical-bank/independent-triplets-all-KV',check)
    return dict(prefixes=[list(p)for p in prefixes],roots=roots[:16]if target else None,evidence=evidence)
