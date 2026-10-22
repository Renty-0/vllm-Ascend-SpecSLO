"""Fresh445 checks plus initial/same-owner bulk known-prefill references."""
from dataclasses import replace
from pathlib import Path
import hashlib,json,math,statistics

def rows_for_span(owner,start,tokens,table):
    if type(owner)is not int or owner<0 or type(start)is not int or start<0:
        raise ValueError('ROOT446 explicit owner and nonnegative known-prefix length')
    if not 1<=len(tokens)<=40 or start+len(tokens)>2560 or len(table)!=32:
        raise ValueError('ROOT446 bounded known span and complete original table')
    out=[]
    for j,token in enumerate(tokens):
        position=start+j;page=table[position//128]
        if type(token)is not int or token<0 or type(page)is not int or not 0<=page<544:
            raise ValueError('ROOT446 valid known tokens and allocated real KV')
        out.append(dict(owner=owner,row=80+j,group=16+j//5,token=token,
            position=position,slot=page*128+position%128,visible_key_count=position+1))
    if len({v['slot']for v in out})!=len(out):raise ValueError('ROOT446 unique writes across same-owner groups')
    return out

def measure(engine,payloads,output):
    # Reexecute the complete successful445 scope on this fresh engine first.
    from forward445 import measure as original
    parent=original(engine,payloads,output)
    import torch
    import torch.distributed as dist
    state=engine._branch_tail_state;current=state.static_current
    x,p,meta,_,mapping=current['physical']
    if engine.is_draft or state.enabled or current['real_N']!=16 or current['real_queries']!=80:
        raise RuntimeError('ROOT446 exact original N16 nonbranching decode frame')
    group=engine.model.layers[0].context.group
    if dist.get_world_size(group=group)!=3:raise RuntimeError('ROOT446 real TP3')
    params=[(n,t.data_ptr())for n,t in engine.model.named_parameters()]
    manifest=engine.mc2_static_route_manifest
    manifest_digest=manifest.digest if manifest is not None else None
    tokens=list(engine._root445_fixture_prefixes[16][:40])
    if len(tokens)!=40:raise RuntimeError('ROOT446 forty actual known fixture tokens')
    for owner in(72,73):
        if any(v>=0 for v in engine.cache_allocation.block_tables[owner]):
            raise RuntimeError('ROOT446 fresh independent initial-prompt owner')
        engine._activate_cache_sequence(owner,tokens,enable_prefix_caching=False)
        engine._ensure_cache_capacity([owner],[40])
        if engine.cache_allocation.num_cached_tokens[owner]!=0:
            raise RuntimeError('ROOT446 no prefilled prefix on initial owner')
    tables=engine.cache_allocation.block_tables
    def digest(t):
        cpu=t.detach().cpu().contiguous()
        return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
    def kv(slots):
        return [c.reshape(c.shape[0]*c.shape[1],-1).index_select(0,slots).detach().cpu().contiguous()
            for layer in engine.model.layers for c in(layer.self_attn.key_cache,layer.self_attn.value_cache)]
    def frame(owner,start,known):
        rows=rows_for_span(owner,start,known,tables[owner])
        xx=x.clone();pp=p.clone();slots=meta.slot_mapping.clone();ctx=meta.context_lens.clone()
        bt=meta.request_block_tables.clone();mask=meta.tree_attention_mask.clone()
        table=torch.tensor([v if v>=0 else state.static_zero_page for v in tables[owner]],dtype=bt.dtype,device=bt.device)
        for g in sorted({v['group']for v in rows}):bt[g].copy_(table)
        for r in rows:
            q=r['row'];xx[q]=r['token'];pp[q]=r['position'];slots[q]=r['slot'];ctx[q]=r['visible_key_count']
            mask[r['group'],0,q%5].fill_(True);mask[r['group'],0,q%5,:r['visible_key_count']]=False
        if not torch.equal(xx[:80],x[:80])or not torch.equal(pp[:80],p[:80])or not torch.equal(slots[:80],meta.slot_mapping[:80]):
            raise RuntimeError('ROOT446 original80 decode rows preserved')
        return xx,pp,replace(meta,slot_mapping=slots,context_lens=ctx,block_tables=bt,
            request_block_tables=bt,tree_attention_mask=mask),rows
    def eager(f):
        with state.active(False):return engine.model(*f[:3]).clone()
    def capture(f):
        for _ in range(3):eager(f)
        torch.npu.synchronize();g=torch.npu.NPUGraph()
        with torch.npu.graph(g):h=eager(f)
        for _ in range(3):g.replay()
        torch.npu.synchronize();return g,h
    base=(x,p,meta,[]);g0,h0=capture(base)
    decode=h0[:80].detach().cpu().contiguous()
    decode_slots=meta.slot_mapping[:80].long().clone();decode_KV=kv(decode_slots)
    decode_tokens=engine.model.compute_greedy_tokens(h0,engine.draft_vocab_size)[:80].detach().cpu()
    graphs={'baseline':g0};frames={'baseline':base};outputs={'baseline':h0};cases=[]
    for family,owner,ref,start,lengths in(
        ('initial',72,73,0,(1,2,3,4,5,17,40)),
        ('continuation',16,40,len(engine._root445_fixture_prefixes[16]),(17,40))):
        engine._ensure_cache_capacity([owner,ref],[start+40,start+40])
        expected_hidden=[];expected_tokens=[];reference_slots=[]
        for step,token in enumerate(tokens):
            f=frame(ref,start+step,[token]);h=eager(f);torch.npu.synchronize()
            expected_hidden.append(h[80].detach().cpu().clone())
            sampled=engine.model.compute_greedy_tokens(h,engine.draft_vocab_size)
            expected_tokens.append(int(sampled[80].item()));reference_slots.append(f[3][0]['slot'])
        reference_KV=kv(torch.tensor(reference_slots,dtype=torch.long,device=engine.device))
        for length in lengths:
            name=family+'-L'+str(length);f=frame(owner,start,tokens[:length])
            g,h=capture(f);graphs[name]=g;frames[name]=f;outputs[name]=h
            actual=h[80:80+length].detach().cpu().contiguous()
            expected=torch.stack(expected_hidden[:length])
            sampled=engine.model.compute_greedy_tokens(h,engine.draft_vocab_size).detach().cpu()
            actual_KV=kv(torch.tensor([r['slot']for r in f[3]],dtype=torch.long,device=engine.device))
            wanted_KV=[v[:length]for v in reference_KV]
            case=dict(name=name,family=family,prefix_length=start,known_prefill_queries=length,
                candidate_owner=owner,independent_serial_reference_owner=ref,
                physical_groups=sorted({r['group']for r in f[3]}),actual_rows=f[3],
                prefill_hidden_exact=bool(torch.equal(actual,expected)),
                prefill_hidden_max_abs=float((actual.float()-expected.float()).abs().max()),
                actual_tokens=[int(v)for v in sampled[80:80+length]],reference_tokens=expected_tokens[:length],
                prefill_tokens_exact=[int(v)for v in sampled[80:80+length]]==expected_tokens[:length],
                prefill_all128_query_KV_exact=all(torch.equal(a,b)for a,b in zip(actual_KV,wanted_KV,strict=True)),
                decode_hidden_exact=bool(torch.equal(h[:80].detach().cpu(),decode)),
                decode_tokens_exact=bool(torch.equal(sampled[:80],decode_tokens)),
                decode_all128_query_KV_exact=all(torch.equal(a,b)for a,b in zip(kv(decode_slots),decode_KV,strict=True)),
                query_KV=[dict(index=i,equal=bool(torch.equal(a,b)),actual_sha256=digest(a),reference_sha256=digest(b),max_abs=float((a.float()-b.float()).abs().max()))for i,(a,b)in enumerate(zip(actual_KV,wanted_KV,strict=True))])
            if not bool(torch.isfinite(actual).all()):raise RuntimeError('ROOT446 nonfinite actual prefill hidden')
            cases.append(case)
    names=list(graphs)
    if len(frames)!=10 or set(frames)!=set(graphs):raise RuntimeError('ROOT446 all ten external frames owned')
    with(Path(output)/f'rank{engine.rank}-bulk-prefill446-before-timing.json').open('x')as f:
        json.dump(dict(rank=engine.rank,cases=cases,actual_owned_graph_frames=names),f,indent=2);f.write('\n')
    times={n:[]for n in names}
    for repeat in range(9):
        for name in(names if repeat%2==0 else list(reversed(names))):
            dist.barrier(group=group);begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
            begin.record()
            for _ in range(10):graphs[name].replay()
            end.record();end.synchronize();v=float(begin.elapsed_time(end))/10
            if not math.isfinite(v)or v<=0:raise RuntimeError('ROOT446 genuine positive device event')
            times[name].append(v)
    if params!=[(n,t.data_ptr())for n,t in engine.model.named_parameters()]or(manifest.digest if manifest is not None else None)!=manifest_digest:
        raise RuntimeError('ROOT446 unchanged actual weights and MC2')
    keys=('prefill_hidden_exact','prefill_tokens_exact','prefill_all128_query_KV_exact',
        'decode_hidden_exact','decode_tokens_exact','decode_all128_query_KV_exact')
    local=dict(rank=engine.rank,cases=cases,whole_target_event_ms=times,actual_owned_graph_frames=names,
        all9_cases_math_exact=all(all(c[k]is True for k in keys)for c in cases),
        all_parameter_storages_unchanged=True,MC2_manifest_unchanged=True)
    gathered=[None]*3;dist.all_gather_object(gathered,local,group=group)
    costs={}
    for name in names:
        samples=[max(v['whole_target_event_ms'][name][j]for v in gathered)for j in range(9)]
        costs[name]=dict(max_rank_samples_ms=samples,p50_ms=statistics.median(samples))
    report=dict(schema='ROOT446.actual-initial-and-bulk-q5-prefill.v1',rank=engine.rank,
        by_target_rank=gathered,all3_all9_cases_math_exact=all(v['all9_cases_math_exact']for v in gathered),
        max_rank_costs=costs,physical_M=120,physical_S=24,
        Native_flags_assigned=False,production_permission=False,
        scope='initial empty prefix and same-owner multigroup known spans; no branch-copy/draft/admission/full463 qualification',
        Raw=None,Goodput=None,adopted=False)
    path=Path(output)/f'rank{engine.rank}-bulk-prefill446.json'
    with path.open('x')as f:json.dump(report,f,indent=2);f.write('\n')
    return dict(fresh_single_group445=parent,bulk446_report=str(path),
        all_parameter_storages_unchanged=True,full463=False,
        Native_flags_assigned=False,production_permission=False,Raw=None,Goodput=None)
