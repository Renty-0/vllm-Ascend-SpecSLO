"""Actual target prefill chunks in q5 padding, independent serial references.

Only diagnostic owners16..23 and independently prefilled40..47 are used.
The original16 decode owners and all72 committed prefixes remain protected.
No production request, model parameter, Native flag, or graph ticket changes.
"""
from dataclasses import replace
from pathlib import Path
import hashlib,json,math,statistics
from layout445 import plan

def measure(engine,unused_payloads,output):
    import torch
    import torch.distributed as dist
    state=engine._branch_tail_state
    if engine.is_draft or state.enabled:raise RuntimeError('ROOT445 original nonbranching target frame only')
    current=state.static_current
    x,p,meta,inverse,mapping=current['physical']
    if (x.shape[0]!=120 or current['real_N']!=16 or current['real_queries']!=80
        or mapping['physical_group_sources']!=list(range(16))+[None]*8):
        raise RuntimeError('ROOT445 exactly16 original full primary groups and8 wholly unused groups')
    group=engine.model.layers[0].context.group
    if dist.get_world_size(group=group)!=3:raise RuntimeError('ROOT445 actual TP3')
    prefixes=engine._root445_fixture_prefixes
    tables=engine.cache_allocation.block_tables
    params=[(n,t.data_ptr())for n,t in engine.model.named_parameters()]
    manifest=engine.mc2_static_route_manifest
    manifest_digest=manifest.digest if manifest is not None else None
    original_slots=meta.slot_mapping[:80].clone().long()
    chunks=[list(prefixes[i][:5])for i in range(16,24)]
    if any(len(v)!=5 for v in chunks):raise RuntimeError('ROOT445 five real known fixture tokens per continuation')
    def digest(t):
        c=t.detach().cpu().contiguous()
        return hashlib.sha256(c.view(torch.uint8).numpy().tobytes()).hexdigest()
    def kv(slots):
        return [c.reshape(c.shape[0]*c.shape[1],-1).index_select(0,slots).detach().cpu().contiguous()
            for layer in engine.model.layers
            for c in (layer.self_attn.key_cache,layer.self_attn.value_cache)]
    def frame(owners,length,step=None):
        # A serial reference places one next known token in each independent
        # reference owner's first physical row; the other four rows stay padding.
        tokens=[v[:length]if step is None else [v[step]]for v in chunks]
        if step is None:
            rows=plan(prefixes,tables,owners,list(range(16,24)),tokens,real_groups=16)
        else:
            # The independent prefix has grown by the previously processed
            # known tokens; planning uses those exact logical positions.
            grown=[list(v)for v in prefixes]
            for i in range(16,24):grown[i].extend(chunks[i-16][:step])
            rows=plan(grown,tables,owners,list(range(16,24)),tokens,real_groups=16)
        xx=x.clone();pp=p.clone();slots=meta.slot_mapping.clone()
        context=meta.context_lens.clone();bt=meta.request_block_tables.clone()
        mask=meta.tree_attention_mask.clone()
        zero=state.static_zero_page
        for owner,g in zip(owners,range(16,24),strict=True):
            host=[page if page>=0 else zero for page in tables[owner]]
            if len(host)!=bt.shape[1]:raise RuntimeError('ROOT445 complete original32-column table')
            bt[g].copy_(torch.tensor(host,dtype=bt.dtype,device=bt.device))
        for row in rows:
            q=row['row'];xx[q]=row['token'];pp[q]=row['position'];slots[q]=row['slot']
            context[q]=row['visible_key_count']
            mask[row['group'],0,q%5].fill_(True)
            mask[row['group'],0,q%5,:row['visible_key_count']]=False
        out=replace(meta,slot_mapping=slots,context_lens=context,block_tables=bt,
            request_block_tables=bt,tree_attention_mask=mask)
        if not torch.equal(xx[:80],x[:80])or not torch.equal(pp[:80],p[:80])or not torch.equal(slots[:80],meta.slot_mapping[:80]):
            raise RuntimeError('ROOT445 original80 decode inputs/positions/writes untouched')
        return xx,pp,out,rows
    def eager(f):
        with state.active(False):return engine.model(*f[:3]).clone()
    def capture(f):
        for _ in range(3):eager(f)
        torch.npu.synchronize();g=torch.npu.NPUGraph()
        with torch.npu.graph(g):h=eager(f)
        for _ in range(3):g.replay()
        torch.npu.synchronize()
        return g,h
    base=(x,p,meta,[])
    graph0,hidden0=capture(base)
    decode_hidden=hidden0[:80].detach().cpu().contiguous()
    decode_KV=kv(original_slots)
    decode_tokens=engine.model.compute_greedy_tokens(hidden0,engine.draft_vocab_size)[:80].detach().cpu()
    # Independent serial known-token reference at the identical physical
    # M120/S24 shape, on owners whose prefixes were separately prefetched.
    ref_hidden={};ref_tokens={};ref_rows=[]
    for step in range(5):
        f=frame(list(range(40,48)),1,step=step)
        h=eager(f);torch.npu.synchronize()
        tokens=engine.model.compute_greedy_tokens(h,engine.draft_vocab_size).detach().cpu()
        cpu=h.detach().cpu()
        for k,row in enumerate(f[3]):
            ref_hidden[k,step]=cpu[row['row']].clone()
            ref_tokens[k,step]=int(tokens[row['row']])
        ref_rows.extend((k,step,row['slot'])for k,row in enumerate(f[3]))
    ref_slots=torch.tensor([slot for k,step,slot in sorted(ref_rows)],dtype=torch.long,device=engine.device)
    full_reference_KV=kv(ref_slots)
    references_sha256=[digest(v)for v in full_reference_KV]
    graphs={'baseline':graph0};hidden_outputs={'baseline':hidden0};frames={'baseline':base};cases=[]
    for length in range(1,6):
        f=frame(list(range(16,24)),length)
        g,h=capture(f);name='chunk'+str(length);graphs[name]=g;hidden_outputs[name]=h;frames[name]=f
        rows=f[3];indices=torch.tensor([v['row']for v in rows],dtype=torch.long,device=engine.device)
        slots=torch.tensor([v['slot']for v in rows],dtype=torch.long,device=engine.device)
        got=h.index_select(0,indices).detach().cpu().contiguous()
        expected=torch.stack([ref_hidden[k,j]for k in range(8)for j in range(length)])
        tokens=engine.model.compute_greedy_tokens(h,engine.draft_vocab_size).detach().cpu()
        actual_tokens=[int(tokens[row['row']])for row in rows]
        expected_tokens=[ref_tokens[k,j]for k in range(8)for j in range(length)]
        chosen=torch.tensor([5*k+j for k in range(8)for j in range(length)],dtype=torch.long,device='cpu')
        actual_KV=kv(slots)
        expected_KV=[v.index_select(0,chosen)for v in full_reference_KV]
        current_decode_KV=kv(original_slots)
        finite=bool(torch.isfinite(got).all())
        case=dict(chunk_length_per_owner=length,real_prefill_owners=list(range(16,24)),
            independent_serial_reference_owners=list(range(40,48)),prefill_queries=8*length,
            actual_rows=rows,decode80_hidden_exact=bool(torch.equal(h[:80].detach().cpu(),decode_hidden)),
            decode80_tokens_exact=bool(torch.equal(tokens[:80],decode_tokens)),
            decode128_query_KV_exact=all(torch.equal(a,b)for a,b in zip(current_decode_KV,decode_KV,strict=True)),
            prefill_hidden_exact=bool(torch.equal(got,expected)),prefill_hidden_max_abs=float((got.float()-expected.float()).abs().max()),
            prefill_tokens_exact=actual_tokens==expected_tokens,actual_prefill_tokens=actual_tokens,
            serial_reference_tokens=expected_tokens,prefill_all128_query_KV_exact=all(torch.equal(a,b)for a,b in zip(actual_KV,expected_KV,strict=True)),
            query_KV=[dict(index=i,equal=bool(torch.equal(a,b)),actual_sha256=digest(a),reference_sha256=digest(b),max_abs=float((a.float()-b.float()).abs().max()))for i,(a,b)in enumerate(zip(actual_KV,expected_KV,strict=True))],
            all_finite=finite)
        if not finite:raise RuntimeError('ROOT445 nonfinite mixed prefill hidden')
        cases.append(case)
    if set(frames)!=set(graphs)or len(frames)!=6:raise RuntimeError('ROOT445 every exact graph frame remains owned')
    with (Path(output)/f'rank{engine.rank}-padding-prefill445-before-timing.json').open('x')as stream:
        json.dump(dict(rank=engine.rank,cases=cases,actual_owned_graph_frames=list(frames),Native_flags_assigned=False,production_permission=False,Raw=None,Goodput=None),stream,indent=2);stream.write('\n')
    times={name:[]for name in graphs};names=list(graphs)
    for repeat in range(9):
        order=names if repeat%2==0 else list(reversed(names))
        for name in order:
            dist.barrier(group=group)
            start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(10):graphs[name].replay()
            end.record();end.synchronize();value=float(start.elapsed_time(end))/10
            if not math.isfinite(value)or value<=0:raise RuntimeError('ROOT445 actual positive event timing')
            times[name].append(value)
    if params!=[(n,t.data_ptr())for n,t in engine.model.named_parameters()]or (manifest.digest if manifest is not None else None)!=manifest_digest:
        raise RuntimeError('ROOT445 parameters and MC2 static manifest unchanged')
    local=dict(rank=engine.rank,cases=cases,whole_target_device_event_ms=times,
        independent_serial_reference_query_KV_sha256=references_sha256,
        original_decode_owners_unchanged=True,all_parameter_storages_unchanged=True,actual_owned_graph_frames=list(frames),
        MC2_manifest_unchanged=True,physical_M=120,physical_S=24,
        all_math_exact=all(all(c[k]is True for k in ('decode80_hidden_exact','decode80_tokens_exact','decode128_query_KV_exact','prefill_hidden_exact','prefill_tokens_exact','prefill_all128_query_KV_exact'))for c in cases))
    gathered=[None]*3;dist.all_gather_object(gathered,local,group=group)
    cost={}
    for name in names:
        values=[max(v['whole_target_device_event_ms'][name][i]for v in gathered)for i in range(9)]
        cost[name]=dict(max_rank_samples_ms=values,p50_ms=statistics.median(values))
    report=dict(schema='ROOT445.real-target-q5-padding-prefill-math-cost.v1',rank=engine.rank,
        by_target_rank=gathered,all3_all5_lengths_math_exact=all(v['all_math_exact']for v in gathered),
        max_rank_costs=cost,decode_owners=16,maximum_real_prefill_owners=8,
        maximum_prefill_queries=40,physical_M=120,physical_S=24,
        scope='diagnostic known prompt-continuation chunks only; no admission, draft prefill, production tickets or full463',
        Native_flags_assigned=False,production_permission=False,Raw=None,Goodput=None,adopted=False)
    path=Path(output)/f'rank{engine.rank}-padding-prefill445.json'
    with path.open('x')as f:json.dump(report,f,indent=2);f.write('\n')
    return dict(report=str(path),all5_lengths_math_exact=local['all_math_exact'],
        all_parameter_storages_unchanged=True,full463=False,
        Native_flags_assigned=False,production_permission=False,Raw=None,Goodput=None)
