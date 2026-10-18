"""Own prefix reference: one real first query per uniform q5 group."""
def pack_q5(engine,frame,*,full):
    import torch
    from dataclasses import replace
    from metadata231 import pack_q5 as original
    from fixed_mask_extent import padded
    from static_state244 import ensure_zero
    x,pos,meta=original(engine,frame,full=full)
    page=ensure_zero(engine)
    if tuple(meta.actual_seq_lengths_q)!=tuple(range(5,81,5)):
        raise RuntimeError('actual independent original16x5 frame')
    real=meta.context_lens.tolist()[::5]
    if not all(0<v<=2560 for v in real):raise RuntimeError('independent real prefix outside declared extent')
    meta=padded(meta,10)
    mask=meta.tree_attention_mask.clone()
    # Future canaries are non-real. They write their own original future
    # slots, attend a finite single prefix key, and cannot influence q0.
    mask[:,:,1:5,:]=True;mask[:,:,1:5,0]=False
    tables=torch.where(meta.request_block_tables>=0,meta.request_block_tables,page)
    return x,pos,replace(meta,block_tables=tables,request_block_tables=tables,tree_attention_mask=mask,
                         actual_seq_lengths_q=tuple(range(5,81,5)),sequence_lens=(2560,)*16)

def checked(meta):
    if tuple(meta.tree_attention_mask.shape)!=(16,1,10,4096)or tuple(meta.sequence_lens)!=(2560,)*16:
        raise RuntimeError('actual fixedq5 own reference metadata')
    return meta
