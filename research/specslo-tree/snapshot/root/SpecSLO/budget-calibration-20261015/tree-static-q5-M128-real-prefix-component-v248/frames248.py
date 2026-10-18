"""Actual fixed M80 or M128 frames, preserving every real query and KV slot."""
def pack(engine,owners,positions,tokens,*,rows,reference=False):
    import torch
    from dataclasses import replace
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan
    from host_target_metadata import make_host_tree_attention_metadata
    from fixed_mask_extent import padded
    if rows not in(80,128):raise ValueError('two explicitly compared physical shapes')
    q5groups=16 if rows==80 else 25
    if not 0<len(owners)<=q5groups or not len(owners)==len(positions)==len(tokens):
        raise ValueError('exact real owner/frontier/input correspondence')
    if len(set(owners))!=len(owners)or any(len(v)!=5 for v in tokens):raise ValueError('independent real q5 frames')
    plans=[];ids=list(owners);roots=[];draft=[]
    for owner,p,values in zip(owners,positions,tokens,strict=True):
        if not 0<=p or p+5>2560:raise ValueError('all real five-query keys fit declared2560 extent')
        plans.append(cached_cpu_tree_speculation_plan(1,4,p,4096,candidate_budget=4))
        roots.append(values[0]);draft.append(values[1:])
    for i in range(q5groups-len(owners)):
        ids.append(463+i);plans.append(cached_cpu_tree_speculation_plan(1,4,0,4096,candidate_budget=4))
        roots.append(0);draft.append([0]*4)
    if rows==128:
        ids.append(488);plans.append(cached_cpu_tree_speculation_plan(1,2,0,4096,candidate_budget=2))
        roots.append(0);draft.append([0]*2)
    x,pos,meta=make_host_tree_attention_metadata(engine,plans,roots,draft,sequence_ids=ids,allow_causal_fast_path=False)
    if x.numel()!=rows:raise RuntimeError('all actual physical model rows')
    meta=padded(meta,10);mask=meta.tree_attention_mask.clone()
    # Only real q5 rows are mathematical candidate nodes. All other rows are
    # computed, but attend one finite private dummy prefix key.
    if reference:
        mask[:len(owners),:,1:5,:]=True;mask[:len(owners),:,1:5,0]=False
    mask[len(owners):,:,:,:]=True
    for group,plan in enumerate(plans[len(owners):],len(owners)):
        mask[group,0,:plan.candidate_budget+1,0]=False
    zero=engine.cache_allocation.block_tables[526][0]
    if zero<0:raise RuntimeError('actual separately allocated private zero page')
    tables=torch.where(meta.request_block_tables>=0,meta.request_block_tables,zero)
    ends=tuple(range(5,q5groups*5+1,5))+((128,)if rows==128 else())
    if tuple(meta.actual_seq_lengths_q)!=ends:raise RuntimeError('actual25 q5 plus nonreal3 tail partition')
    return x,pos,replace(meta,block_tables=tables,request_block_tables=tables,
        tree_attention_mask=mask,actual_seq_lengths_q=ends,sequence_lens=(2560,)*len(plans))
