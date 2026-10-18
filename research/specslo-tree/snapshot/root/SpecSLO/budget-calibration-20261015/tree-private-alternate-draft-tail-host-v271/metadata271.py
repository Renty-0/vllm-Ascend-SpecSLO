"""Build ordinary PA-chain inputs over private alternate tables; HOST only.

The real draft producer must have materialized the primary root first. This
does not execute the KV copy or model, nor update a committed-token ledger.
"""
from dataclasses import replace

def build(engine,desc,alternate_roots,primary_roots,*,steps,bucket,pad):
    import torch
    owners=desc['owners'];count=len(owners)
    if not 1<=steps<=3 or bucket<count or len(alternate_roots)!=count or len(primary_roots)!=count:
        raise ValueError('HOST271 aligned actual roots and one-to3 additional PA steps')
    tables=engine.cache_allocation.block_tables
    if id(tables)!=desc['source_tables_identity']:
        raise RuntimeError('HOST271 original allocation table object changed')
    if [tables[i][0]for i in desc['private_owners']]!=desc['private_page_pool']:
        raise RuntimeError('HOST271 private scratch page lease changed')
    known=getattr(engine,'_dynamic_tree_known_KV',{})
    for ordinal,(owner,row,root)in enumerate(zip(owners,desc['rows'],primary_roots,strict=True)):
        prefix=row['prefix'];state=known.get(owner)
        if (state is None or state['safe_before']>prefix or state['tokens'].get(prefix)!=int(root)
                or tables[owner][:prefix//128+1]!=desc['source_prefix_pages'][ordinal]):
            raise RuntimeError('HOST271 actual materialized primary root or prefix lease is stale')
    overlay=[list(row)for row in tables]
    for owner,row in zip(owners,desc['virtual_tables'],strict=True):overlay[owner]=row
    inp=torch.tensor(alternate_roots,dtype=torch.long,device=engine.device)
    positions,base=engine.model.make_attention_metadata(owners,
        [r['positions'][0]for r in desc['rows']],block_tables=overlay,
        slot_mapping=[r['slots'][0]for r in desc['rows']],use_fused_infer_attention=False)
    inp,positions,base=pad(inp,positions,base,bucket)
    if base.request_block_tables is not None or base.use_fused_infer_attention:
        raise RuntimeError('HOST271 same ordinary per-token PA metadata mode required')
    dummy=bucket-count
    position_rows=[];slot_rows=[];contexts=[]
    for step in range(steps):
        position_rows.append([r['positions'][step]for r in desc['rows']]+[0]*dummy)
        slot_rows.append([r['slots'][step]for r in desc['rows']]+[-1]*dummy)
        contexts.append([r['contexts'][step]for r in desc['rows']]+[1]*dummy)
    ps=torch.tensor(position_rows,dtype=positions.dtype,device=engine.device)
    slots=torch.tensor(slot_rows,dtype=base.slot_mapping.dtype,device=engine.device)
    ctx=torch.tensor(contexts,dtype=base.context_lens.dtype,device=base.context_lens.device)
    metadata=[replace(base,slot_mapping=slots[step],context_lens=ctx[step],
                      sequence_lens=tuple(contexts[step]))for step in range(steps)]
    return inp,list(ps.unbind(0)),metadata
