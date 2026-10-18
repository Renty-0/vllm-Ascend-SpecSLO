"""Original linear query plans packed as FULL9, with exact causal visibility."""
def pack(engine,frame,*,full):
    import torch
    if not full:raise ValueError('ROOT228 diagnostic is explicitly FULL only')
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan
    from host_target_metadata import make_host_tree_attention_metadata
    from fixed_mask_extent import padded
    plans=[];roots=[];rows=[]
    for j,seq in enumerate(frame['selected']):
        p=frame['positions'][5*j]
        if frame['ids'][5*j:5*j+5]!=[seq]*5 or frame['positions'][5*j:5*j+5]!=list(range(p,p+5)):raise RuntimeError('original real16 query5 partition')
        plans.append(cached_cpu_tree_speculation_plan(1,4,p,4096,candidate_budget=4))
        roots.append(frame['tokens'][5*j]);rows.append(frame['tokens'][5*j+1:5*j+5])
    inputs,pos,meta=make_host_tree_attention_metadata(engine,plans,roots,rows,sequence_ids=frame['selected'],allow_causal_fast_path=False)
    meta=padded(meta,9)
    if not meta.tree_attention or not meta.use_fused_infer_attention or tuple(meta.tree_attention_mask.shape)!=(16,1,9,4096):raise RuntimeError('exact original FULL9 family')
    from canonical228 import canonical
    return inputs,pos,canonical(meta)

def q5_full_reference(engine,frame,*,full):
    import torch
    if not full:raise ValueError('ROOT228 original FULL9 reference only')
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan
    from host_target_metadata import make_host_tree_attention_metadata
    from fixed_mask_extent import padded
    plans=[];roots=[];rows=[]
    for j,seq in enumerate(frame['selected']):
        p=frame['positions'][5*j]
        if frame['ids'][5*j:5*j+5]!=[seq]*5 or frame['positions'][5*j:5*j+5]!=list(range(p,p+5)):raise RuntimeError('original real16 query5 partition')
        plans.append(cached_cpu_tree_speculation_plan(1,4,p,4096,candidate_budget=4))
        roots.append(frame['tokens'][5*j]);rows.append(frame['tokens'][5*j+1:5*j+5])
    inputs,pos,meta=make_host_tree_attention_metadata(engine,plans,roots,rows,sequence_ids=frame['selected'],allow_causal_fast_path=False)
    meta=padded(meta,9)
    if not meta.tree_attention or not meta.use_fused_infer_attention or tuple(meta.tree_attention_mask.shape)!=(16,1,9,4096):raise RuntimeError('exact original FULL9 family')
    return inputs,pos,meta
