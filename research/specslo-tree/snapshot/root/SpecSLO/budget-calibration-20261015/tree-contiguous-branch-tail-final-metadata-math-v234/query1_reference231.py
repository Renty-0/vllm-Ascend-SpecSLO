"""Preserve each query's original FULL visibility while splitting FIA segments."""
from dataclasses import replace
def canonical(meta):
    import torch
    if not meta.tree_attention or not meta.use_fused_infer_attention:raise ValueError('original FULL metadata required')
    if meta.context_lens.device.type!='cpu':raise ValueError('original host context lengths required')
    ends=meta.actual_seq_lengths_q;start=0;owners=[];rows=[]
    mask=meta.tree_attention_mask;extent=int(mask.shape[2])
    if len(ends)!=int(mask.shape[0])or meta.request_block_tables.shape[0]!=len(ends):raise ValueError('original request alignment')
    for owner,end in enumerate(ends):
        count=end-start
        if not 0<count<=extent:raise ValueError('original query partition fits mask')
        owners.extend([owner]*count);rows.extend(owner*extent+k for k in range(count));start=end
    M=start;lengths=tuple(int(n)for n in meta.context_lens.tolist())
    if len(lengths)!=M or any(not 0<n<=int(mask.shape[3])for n in lengths):raise ValueError('exact per-query host KV lengths')
    device=mask.device
    oi=torch.tensor(owners,dtype=torch.long,device=device);ri=torch.tensor(rows,dtype=torch.long,device=device)
    tables=meta.request_block_tables.index_select(0,oi)
    qmask=mask.reshape(-1,int(mask.shape[3])).index_select(0,ri).reshape(M,1,1,int(mask.shape[3]))
    return replace(meta,block_tables=tables,request_block_tables=tables,
        actual_seq_lengths_q=tuple(range(1,M+1)),sequence_lens=lengths,tree_attention_mask=qmask)
