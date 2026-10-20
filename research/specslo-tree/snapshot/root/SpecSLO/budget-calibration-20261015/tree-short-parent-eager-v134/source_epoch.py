"""Fresh actual token prefixes; no candidate KV is copied into any oracle."""
import copy


def rebuild(engine,contexts,repairs):
    if len(contexts)!=len(repairs):raise ValueError('Source repairs are per real owner')
    prefixes=[len(c['token_ids'])-1 for c in contexts]
    known=[list(c['token_ids'][:p-(r if engine.is_draft else 0)])
           for c,p,r in zip(contexts,prefixes,repairs)]
    engine._release_cache()
    engine._allocate_cache(known,enable_prefix_caching=False,reserve_sequence_capacity=True)
    engine._dynamic_tree_known_KV={}
    for seq,tokens in enumerate(known):
        for start in range(0,len(tokens),512):
            chunk=tokens[start:start+512]
            engine._run_packed_hidden(chunk,[seq]*len(chunk),list(range(start,start+len(chunk))),
                use_aclgraph=False,use_fused_infer_attention=True)
        engine._dynamic_tree_known_KV[seq]=dict(safe_before=len(tokens),tokens={})
    private=tuple(getattr(engine,'_v126_private_scratch_owners',()))
    if set(private)&set(range(len(contexts))) or any(not 64<=seq<128 for seq in private):
        raise RuntimeError('V126 private scratch owners overlap actual source or exceed128 component storage rows')
    for seq in private:
        engine._activate_cache_sequence(seq,[0],enable_prefix_caching=False)
        engine._ensure_cache_capacity([seq],[8])
    engine._v126_source_epoch=getattr(engine,'_v126_source_epoch',0)+1
    engine._ensure_cache_capacity(list(range(len(contexts))),[p+8 for p in prefixes])
    engine._v126_last_source_lease=lease(engine,len(contexts),private)
    from short_scope_v126 import digest
    engine._v126_last_source_lease["actual_known_prefix_tokens_sha256"]=[digest(t) for t in known]
    engine._v126_last_source_lease["actual_known_prefix_lengths"]=[len(t) for t in known]
    return dict(prefixes=prefixes,roots=[int(c['token_ids'][-1]) for c in contexts],
        catchups=[list(c['token_ids'][p-r:p+1]) for c,p,r in zip(contexts,prefixes,repairs)],
        ids=list(range(len(contexts))))


def arrays(engine):
    rows=[]
    for layer in engine.model.layers:
        for name in ('key_cache','value_cache'):
            raw=getattr(layer.self_attn,name);value=raw.flatten(0,1)
            if value.data_ptr()!=raw.data_ptr():raise RuntimeError('KV view must preserve write-through storage')
            rows.append(value)
    return rows


def slots(engine,seq,positions):
    import torch
    size=engine.config.kvcache_block_size;table=engine.cache_allocation.block_tables[seq]
    values=[table[p//size]*size+p%size for p in positions]
    if len(set(values))!=len(values) or any(v<0 for v in values):raise RuntimeError('Invalid source KV geometry')
    return torch.tensor(values,dtype=torch.long,device=engine.device)


def read(engine,index):return tuple(v.index_select(0,index).clone() for v in arrays(engine))


def query_KV(engine,meta):return read(engine,meta.slot_mapping.long())


def logical_KV(engine,starts,lengths):
    import torch
    ids=[slots(engine,seq,range(p,p+n)) for seq,(p,n) in enumerate(zip(starts,lengths))]
    return read(engine,torch.cat(ids))


def protect(engine,prefixes):
    """RAM-only exact hashes and explicit finite checks, outside cost samples."""
    import hashlib
    import torch
    ids=torch.cat([slots(engine,seq,range(p)) for seq,p in enumerate(prefixes)])
    result=[];failed=False
    for value in arrays(engine):
        used=value.index_select(0,ids)
        failed|=not bool(torch.isfinite(used).all().item())
        result.append(hashlib.sha256(used.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
    # Finish every local span/layer before any rank rejects the next TP call.
    if not engine.is_draft:
        import torch.distributed as dist
        flag=torch.tensor([int(failed)],dtype=torch.int64,device=engine.device)
        dist.all_reduce(flag,op=dist.ReduceOp.MAX,group=engine.groups.target_group)
        failed=bool(flag.item())
    if failed:raise RuntimeError('Used committed source KV is nonfinite (target-group consensus)')
    return result


def clone_ledger(engine):return copy.deepcopy(engine._dynamic_tree_known_KV)


def lease(engine,count,private):
    """Observe actual host page tables and require unique live page leases."""
    rows=tuple(range(count))+tuple(private);tables=engine.cache_allocation.block_tables
    if engine._cache_storage_sequence_capacity!=128 or len(tables)!=128:
        raise RuntimeError('V126 actual standalone component storage must be128')
    pages=[int(v) for i in rows for v in tables[i] if int(v)!=-1]
    if any(v<0 or v>=544 for v in pages) or len(pages)!=len(set(pages)):
        raise RuntimeError('V126 real owner/scratch page lease aliases or exceeds544')
    return dict(source_epoch=engine._v126_source_epoch,real_work_owners=list(range(count)),
        scratch_owners=list(private),actual_storage_rows=len(tables),actual_page_roof=544,
        actual_block_tables={str(i):list(map(int,tables[i])) for i in rows},
        allocated_pages=len(pages),work_and_dummy_disjoint=True,actual_page_leases_unique=True,online_storage527_qualified=False)
