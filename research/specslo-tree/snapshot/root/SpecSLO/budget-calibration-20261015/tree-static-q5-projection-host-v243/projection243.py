"""Explicit physical packing; original real queries and KV writes are preserved.

Only a bounded prototype. No production integration or NPU qualification.
"""
def plan(ends, real_queries):
    if type(real_queries) is not int or not 1 <= real_queries <= 80:
        raise ValueError('positive original real query prefix within80')
    ends=tuple(ends)
    if not ends or ends[-1]!=80 or any(type(x)is not int for x in ends):
        raise ValueError('original physical query partition consumes80')
    start=0;groups=[]
    for group,end in enumerate(ends):
        if not start < end <= 80:raise ValueError('strict original query partition')
        if start < real_queries:
            if end > real_queries or end-start > 5:raise ValueError('real primary/side segments must fit one q5 group')
            groups.append((group,list(range(start,end))))
        start=end
    if real_queries not in ends or len(groups)>16:
        raise ValueError('supported real-query boundary and at most16 primary/side groups')
    unused=iter(range(real_queries,80));order=[];moves=[];sources=[];phantom=[]
    for physical in range(16):
        if physical<len(groups):
            source,rows=groups[physical];sources.append(source)
            for q,old in enumerate(rows):
                moves.append(dict(old_row=old,new_row=len(order),old_group=source,old_query=q))
                order.append(old)
        else:rows=[];sources.append(None)
        for q in range(len(rows),5):
            old=next(unused,None)
            if old is None:raise RuntimeError('original padding rows must cover exactly remaining physical rows')
            phantom.append(len(order));order.append(old)
    if next(unused,None)is not None or sorted(order)!=list(range(80)):
        raise RuntimeError('complete bijection of80 original physical input and KV write rows')
    inverse=[None]*80
    for new,old in enumerate(order):inverse[old]=new
    if sorted(x['old_row']for x in moves)!=list(range(real_queries)):
        raise RuntimeError('all real logical queries occur exactly once')
    return dict(physical_to_original=order,original_to_physical=inverse,real_moves=moves,
                physical_group_sources=sources,phantom_rows=phantom,
                actual_seq_lengths_q=list(range(5,81,5)),physical_queries=80,
                supported_primary_side_groups=len(groups),real_queries=real_queries)


def transform(inputs, positions, metadata, real_queries, zero_page, *, extent=2560, mask_capacity=10):
    import torch
    from dataclasses import replace
    if type(zero_page)is not int or not 0<=zero_page<544:
        raise ValueError('dedicated actual zero page in original pool')
    if type(extent)is not int or not 1<=extent<=4096 or mask_capacity<5:
        raise ValueError('explicit context and query mask capacity')
    p=plan(metadata.actual_seq_lengths_q,real_queries)
    original_mask=metadata.tree_attention_mask
    if not metadata.tree_attention or not metadata.use_fused_infer_attention or original_mask.shape[-1]!=4096:
        raise ValueError('original FULL mask is required')
    real_sources=[x for x in p['physical_group_sources']if x is not None]
    if any(not 0<metadata.sequence_lens[g]<=extent for g in real_sources):
        raise ValueError('real visible KV extent is unsupported; never truncate keys')
    device=inputs.device
    index=torch.tensor(p['physical_to_original'],dtype=torch.long,device=device)
    cpu_index=torch.tensor(p['physical_to_original'],dtype=torch.long,device='cpu')
    inverse=torch.tensor(p['original_to_physical'],dtype=torch.long,device=device)
    tables=torch.full((16,32),zero_page,dtype=torch.int32,device=device)
    if real_sources:
        selected=torch.tensor(real_sources,dtype=torch.long,device=device)
        live=metadata.request_block_tables.index_select(0,selected)
        tables[:len(real_sources)].copy_(torch.where(live>=0,live,zero_page))
    mask=torch.ones((16,1,mask_capacity,4096),dtype=torch.bool,device=device)
    old_extent=int(original_mask.shape[2]);old_flat=original_mask.reshape(-1,4096)
    source=torch.tensor([m['old_group']*old_extent+m['old_query']for m in p['real_moves']],dtype=torch.long,device=device)
    destination=torch.tensor([(m['new_row']//5)*mask_capacity+m['new_row']%5 for m in p['real_moves']],dtype=torch.long,device=device)
    mask.reshape(-1,4096).index_copy_(0,destination,old_flat.index_select(0,source))
    # Non-real rows write only the original padding slots. Give them a finite
    # one-key view; their results are computed and remain non-real outputs.
    phantom=torch.tensor([(r//5)*mask_capacity+r%5 for r in p['phantom_rows']],dtype=torch.long,device=device)
    mask.reshape(-1,4096)[phantom,0]=False
    lengths=metadata.context_lens.index_select(0,cpu_index if metadata.context_lens.device.type=='cpu'else index)
    out=replace(metadata,slot_mapping=metadata.slot_mapping.index_select(0,index),context_lens=lengths,
                block_tables=tables,request_block_tables=tables,tree_attention_mask=mask,
                actual_seq_lengths_q=tuple(range(5,81,5)),sequence_lens=(extent,)*16)
    return inputs.index_select(0,index),positions.index_select(0,index),out,inverse,p
