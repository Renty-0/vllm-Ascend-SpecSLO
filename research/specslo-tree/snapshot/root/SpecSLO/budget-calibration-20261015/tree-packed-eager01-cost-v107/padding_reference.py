"""Oracle-only, independent same-M PA padding; no candidate path changes."""
from dataclasses import replace


def inspect(engine,inputs,positions,metadata,first_padding_row):
    import torch
    import source_epoch as epoch
    count=int(inputs.numel());start=int(first_padding_row)
    if not 0<=start<=count:raise RuntimeError('Invalid full-physical PA padding boundary')
    if (metadata.use_fused_infer_attention or metadata.request_block_tables is not None
            or metadata.attention_mask is not None or metadata.block_tables.shape[0]!=count):
        raise RuntimeError('Padding witness left original per-row PA metadata')
    def take(value):return value[start:count].detach().cpu().tolist()
    fields=dict(rows=list(range(start,count)),tokens=take(inputs),positions=take(positions),
        slot_mapping=take(metadata.slot_mapping),context_lens=take(metadata.context_lens),
        block_tables=take(metadata.block_tables),host_Q=list(metadata.actual_seq_lengths_q),
        host_KV=list(metadata.sequence_lens),host_tuples_ignored_by_per_row_PA=True)
    # Native _pad_inputs copies the first live metadata row, which can belong
    # to any repair owner. Bind that row to the authoritative allocation;
    # comparing only candidate padding rows with each other is insufficient.
    source_table=metadata.block_tables[0].detach().cpu().tolist()
    owners=[owner for owner,table in enumerate(engine.cache_allocation.block_tables)
            if list(map(int,table))==list(map(int,source_table))]
    if len(owners)!=1:
        raise RuntimeError('PA first live table must identify exactly one authoritative cache owner')
    source_owner=owners[0]
    live=[i for i,v in enumerate(metadata.slot_mapping[:start].detach().cpu().tolist()) if v>=0]
    if live:
        if live[0]!=0:
            raise RuntimeError('PA live rows must precede the padding boundary')
        position=int(positions[0].item());slot=int(metadata.slot_mapping[0].item())
        block=engine.config.kvcache_block_size
        if (position<0 or position//block>=len(source_table)
                or source_table[position//block]*block+position%block!=slot
                or int(metadata.context_lens[0].item())!=position+1):
            raise RuntimeError('PA first live slot/position/context must match its authoritative owner')
    elif (any(v!=-1 for v in metadata.slot_mapping.detach().cpu().tolist())
            or any(v!=0 for v in inputs.detach().cpu().tolist()+positions.detach().cpu().tolist())
            or any(v!=1 for v in metadata.context_lens.detach().cpu().tolist())):
        raise RuntimeError('Only the independent readonly PAD reference may have no live slot')
    if (any(v!=0 for v in fields['tokens']+fields['positions'])
            or any(v!=-1 for v in fields['slot_mapping'])
            or any(v!=1 for v in fields['context_lens'])
            or any(row!=source_table for row in fields['block_tables'])):
        raise RuntimeError('PA padding must preserve token0/pos0/slot-1/context1 and authoritative first-live-owner table')
    # Original _pad_inputs copies the first live row's table, not a zero table.
    # Context1 therefore reads that real source page's first token. Do not
    # assume a physical page number or any cache value, and do not copy KV.
    slots=torch.tensor([row[0]*engine.config.kvcache_block_size for row in fields['block_tables']],
        dtype=torch.long,device=engine.device)
    caches=epoch.read(engine,slots)
    if not all(bool(torch.isfinite(v).all().item()) for v in caches):
        raise RuntimeError('Actual PA padding source prefix0 KV is nonfinite')
    return dict(metadata=fields,source_owner=source_owner,source_block_table=source_table,
        prefix0_slots=slots.cpu().tolist(),prefix0_KV=caches)


def unchanged(before,after):
    import torch
    return (before['source_owner']==after['source_owner'] and before['metadata']==after['metadata'] and before['prefix0_slots']==after['prefix0_slots']
        and len(before['prefix0_KV'])==len(after['prefix0_KV'])
        and all(torch.equal(a,b) for a,b in zip(before['prefix0_KV'],after['prefix0_KV'])))


def independent(engine,physical,real_rows,*,source_owner=0):
    """Fresh explicit source owner, all-readonly token0 query rows at the same M.

    Original native metadata/padding build the real page-table/context ABI;
    the single sentinel slot is changed to-1 too, so every query is read-only.
    Host Q tuples are recorded truthfully and are not a PA kernel input.
    """
    import torch
    from vllm_ascend.spec_decode.pearl.native_graph import NativeACLGraphRunner
    inputs=torch.zeros(1,dtype=torch.long,device=engine.device)
    positions,meta=engine._prepare_attention_metadata([source_owner],[0],False)
    inputs,positions,meta=NativeACLGraphRunner._pad_inputs(inputs,positions,meta,physical)
    meta=replace(meta,slot_mapping=torch.full_like(meta.slot_mapping,-1))
    if (not bool((inputs==0).all().item()) or not bool((positions==0).all().item())
            or not bool((meta.slot_mapping==-1).all().item())
            or not bool((meta.context_lens==1).all().item())):
        raise RuntimeError('Independent PAD-only oracle would write KV or change the PAD inputs')
    before=inspect(engine,inputs,positions,meta,real_rows)
    if before['source_owner']!=source_owner:raise RuntimeError('Independent PAD source owner differs from its declared allocation')
    hidden=engine.model(inputs,positions,meta).clone()
    after=inspect(engine,inputs,positions,meta,real_rows)
    if not unchanged(before,after) or not bool(torch.isfinite(hidden).all().item()):
        raise RuntimeError('Independent readonly PAD oracle changed source prefix0 or became nonfinite')
    return hidden[real_rows:].clone(),dict(before=before,after=after,
        full_M=physical,real_rows=real_rows,all_slots_readonly=True,
        oracle_only_eager_forward=True,source_reconstructed_independently=True)


def compare(actual,reference):
    import torch
    before=reference['before']
    fields=('rows','tokens','positions','slot_mapping','context_lens')
    if (actual['source_owner']!=before['source_owner'] or not unchanged(before,reference['after'])
            or any(actual['metadata'][k]!=before['metadata'][k] for k in fields)
            or len(actual['prefix0_KV'])!=len(before['prefix0_KV'])
            or not all(torch.equal(a,b) for a,b in zip(actual['prefix0_KV'],before['prefix0_KV']))):
        raise RuntimeError('Independent PA padding metadata/source-prefix0 reconstruction or protection failed')
    return dict(full_physical_padding_rows=actual['metadata']['rows'],original_PAD_perrow_ABI=True,
        independent_prefix0_rebuilt_bitwise=True,oracle_prefix0_finite_and_protected=True,
        all_oracle_slots_readonly=True,actual_host_Q=actual['metadata']['host_Q'],
        actual_host_KV=actual['metadata']['host_KV'],reference_host_Q=before['metadata']['host_Q'],
        reference_host_KV=before['metadata']['host_KV'],
        actual_source_owner=actual['source_owner'],reference_source_owner=before['source_owner'],
        actual_source_table=actual['source_block_table'],reference_source_table=before['source_block_table'],
        actual_prefix0_slots=actual['prefix0_slots'],reference_prefix0_slots=before['prefix0_slots'],
        cross_epoch_physical_page_identity_required=False,
        host_tuple_equality_claimed=False,host_tuples_ignored_by_perrow_PA=True)
