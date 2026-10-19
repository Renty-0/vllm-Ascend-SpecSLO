"""Batch PA-chain positions/slots and share one unchanged block-table gather."""
from dataclasses import replace


def build(engine, plans, ids, roots, steps, bucket):
    import torch
    if not plans or len(plans)!=len(ids) or len(ids)!=len(roots) or not 1<=steps<=8:
        raise ValueError('PA chain inputs must be non-empty, aligned and bounded')
    if len(set(ids))!=len(ids) or bucket<len(ids):
        raise ValueError('PA chain requests must be unique and fit the bucket')
    # The caller reserves ALL query pages before this first gather. No cache
    # allocation or publication can occur between the chain's metadata steps.
    inp=torch.tensor(roots,dtype=torch.long,device=engine.device)
    prefixes=[int(p.prefix_len) for p in plans]
    pos,meta=engine._prepare_attention_metadata(ids,prefixes,False)
    inputs,pos,base=engine.graph_runner._pad_inputs(inp,pos,meta,bucket)
    tables=engine.cache_allocation.block_tables
    block_size=int(engine.config.kvcache_block_size)
    capacity=engine.prefix_cache.num_blocks*block_size
    max_model_len=int(engine.config.max_model_len)
    position_rows=[];slot_rows=[];context_rows=[];sequence_lengths=[]
    for step in range(steps):
        row_positions=[prefix+step for prefix in prefixes]
        row_slots=[]
        for seq,position in zip(ids,row_positions):
            if not 0<=seq<len(tables) or not 0<=position<max_model_len:
                raise ValueError('PA chain query is outside its request/context range')
            page=tables[seq][position//block_size]
            slot=page*block_size+position%block_size
            if page<0 or not 0<=slot<capacity:
                raise RuntimeError('PA chain referenced an unallocated/out-of-range KV page')
            row_slots.append(slot)
        if len(set(row_slots))!=len(row_slots):
            raise RuntimeError('Distinct PA chain requests alias the same KV slot')
        dummy=bucket-len(ids)
        position_rows.append(row_positions+[0]*dummy)
        slot_rows.append(row_slots+[-1]*dummy)
        context_rows.append([position+1 for position in row_positions]+[1]*dummy)
        sequence_lengths.append(tuple(context_rows[-1]))
    positions=torch.tensor(position_rows,dtype=pos.dtype,device=engine.device)
    slots=torch.tensor(slot_rows,dtype=base.slot_mapping.dtype,device=engine.device)
    contexts=torch.tensor(context_rows,dtype=base.context_lens.dtype,device=base.context_lens.device)
    metadatas=[replace(base,slot_mapping=slots[step],context_lens=contexts[step],
                       sequence_lens=sequence_lengths[step]) for step in range(steps)]
    return inputs,list(positions.unbind(0)),metadatas
