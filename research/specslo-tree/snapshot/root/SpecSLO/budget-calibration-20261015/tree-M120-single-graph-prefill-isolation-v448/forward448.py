"""Independent eager references first; exactly one persistent graph afterward."""
from dataclasses import replace
from pathlib import Path
import hashlib,json,math,statistics
from forward446 import rows_for_span

KEYS=('graph_baseline_matches_eager','baseline_before_case_stable','eager_reference_decode_stable',
    'prefill_hidden_exact','prefill_tokens_exact','prefill_all128_query_KV_exact',
    'decode_hidden_exact','decode_tokens_exact','decode_all128_query_KV_exact')

def measure(engine,payloads,output):
    import torch
    import torch.distributed as dist
    state=engine._branch_tail_state;current=state.static_current
    x,p,meta,_,mapping=current['physical']
    if engine.is_draft or state.enabled or current['real_N']!=16 or current['real_queries']!=80:
        raise RuntimeError('ROOT448 actual N16 B4 nonbranching model frame')
    group=engine.model.layers[0].context.group
    if dist.get_world_size(group=group)!=3:raise RuntimeError('ROOT448 actual TP3')
    params=[(n,t.data_ptr())for n,t in engine.model.named_parameters()]
    manifest=engine.mc2_static_route_manifest;manifest_digest=manifest.digest if manifest else None
    def save(name,value):
        with(Path(output)/f'rank{engine.rank}-{name}.json').open('w')as stream:
            json.dump(value,stream,indent=2);stream.write('\n')
    def digest(t):
        cpu=t.detach().cpu().contiguous()
        return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
    # Hash one parameter at a time on CPU; retain only hashes, never weight dumps.
    weight_hashes_before={name:digest(value)for name,value in engine.model.named_parameters()}
    save('single-graph448-weight-hashes-before',weight_hashes_before)
    def inventory():
        tables=[list(v)for v in engine.cache_allocation.block_tables]
        pages=[v for row in tables for v in row if v>=0]
        if len(set(pages))!=len(pages):raise RuntimeError('ROOT448 independent allocated pages must never alias')
        return tables
    before=inventory();decode_slots=meta.slot_mapping[:80].long().clone()
    decode_host_slots=decode_slots.detach().cpu().tolist()
    tokens=list(engine._root445_fixture_prefixes[16][:40])
    if len(tokens)!=40:raise RuntimeError('ROOT448 forty known fixture tokens')
    for owner in(72,73):
        if any(v>=0 for v in before[owner]):raise RuntimeError('ROOT448 fresh initial candidate/reference')
        engine._activate_cache_sequence(owner,tokens,enable_prefix_caching=False)
        engine._ensure_cache_capacity([owner],[40])
        if engine.cache_allocation.num_cached_tokens[owner]!=0:raise RuntimeError('ROOT448 actual initially empty prefix')
    engine._ensure_cache_capacity([16,40],[167,167])
    tables=inventory()
    if any(before[i]!=tables[i]for i in range(72)):raise RuntimeError('ROOT448 old72 page bindings preserved')
    save('single-graph448-page-inventory',dict(before=before,after=tables,
        decode_query_slots=decode_host_slots,zero_page=state.static_zero_page,
        all_allocated_pages_disjoint=True,original72_tables_unchanged=True))
    def kv(slots):
        return [cache.reshape(cache.shape[0]*cache.shape[1],-1).index_select(0,slots).detach().cpu().contiguous()
            for layer in engine.model.layers for cache in(layer.self_attn.key_cache,layer.self_attn.value_cache)]
    def frame(owner,start,known):
        rows=rows_for_span(owner,start,known,tables[owner])
        if set(v['slot']for v in rows)&set(decode_host_slots):raise RuntimeError('ROOT448 new known queries cannot overwrite old decode slots')
        xx=x.clone();pp=p.clone();slots=meta.slot_mapping.clone();ctx=meta.context_lens.clone()
        bt=meta.request_block_tables.clone();mask=meta.tree_attention_mask.clone()
        table=torch.tensor([v if v>=0 else state.static_zero_page for v in tables[owner]],dtype=bt.dtype,device=bt.device)
        for g in {v['group']for v in rows}:bt[g].copy_(table)
        for r in rows:
            q=r['row'];xx[q]=r['token'];pp[q]=r['position'];slots[q]=r['slot'];ctx[q]=r['visible_key_count']
            mask[r['group'],0,q%5].fill_(True);mask[r['group'],0,q%5,:r['visible_key_count']]=False
        if not torch.equal(xx[:80],x[:80])or not torch.equal(pp[:80],p[:80])or not torch.equal(slots[:80],meta.slot_mapping[:80]):
            raise RuntimeError('ROOT448 complete original decode input/position/slot alignment')
        return xx,pp,replace(meta,slot_mapping=slots,context_lens=ctx,block_tables=bt,
            request_block_tables=bt,tree_attention_mask=mask),rows
    model_body_closed=False
    def eager(f):
        if model_body_closed:raise RuntimeError('ROOT448 no ordinary model-body execution after capture')
        with state.active(False):return engine.model(*f[:3]).clone()
    def sampled(h):return engine.model.compute_greedy_tokens(h,engine.draft_vocab_size).detach().cpu()
    base=(x,p,meta,[])
    eager_base=eager(base);torch.npu.synchronize()
    decode=eager_base[:80].detach().cpu().clone();decode_KV=kv(decode_slots)
    decode_tokens=sampled(eager_base)[:80].clone()
    references={};reference_checks={}
    for family,owner,start in(('initial',73,0),('continuation',40,127)):
        hidden=[];ids=[];slots=[];decode_checks=[]
        for step,token in enumerate(tokens):
            f=frame(owner,start+step,[token]);h=eager(f);torch.npu.synchronize()
            cpu=h.detach().cpu();hidden.append(cpu[80].clone());ids.append(int(sampled(h)[80]));slots.append(f[3][0]['slot'])
            decode_checks.append(bool(torch.equal(cpu[:80],decode)))
        reference_KV=kv(torch.tensor(slots,dtype=torch.long,device=engine.device))
        references[family]=(torch.stack(hidden),ids,reference_KV)
        reference_checks[family]=decode_checks
        save('single-graph448-reference-progress',dict(completed_families=list(references),
            decode_reference_checks=reference_checks,all_references_run_before_capture=True))
    # All model-body eager reference calls end before the single new capture.
    sx=x.clone();sp=p.clone();bt=meta.request_block_tables.clone()
    sm=replace(meta,slot_mapping=meta.slot_mapping.clone(),context_lens=meta.context_lens.clone(),
        block_tables=bt,request_block_tables=bt,tree_attention_mask=meta.tree_attention_mask.clone())
    static=(sx,sp,sm,[])
    def load(f):
        fm=f[2]
        if fm.actual_seq_lengths_q!=sm.actual_seq_lengths_q or fm.sequence_lens!=sm.sequence_lens:
            raise RuntimeError('ROOT448 original constant q5/S24/extent length metadata')
        sx.copy_(f[0]);sp.copy_(f[1]);sm.slot_mapping.copy_(fm.slot_mapping)
        sm.context_lens.copy_(fm.context_lens);bt.copy_(fm.request_block_tables)
        sm.tree_attention_mask.copy_(fm.tree_attention_mask)
    load(base)
    for _ in range(3):eager(static)
    torch.npu.synchronize();graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph):output_hidden=eager(static)
    model_body_closed=True
    for _ in range(3):graph.replay()
    torch.npu.synchronize()
    static_ptrs=[v.data_ptr()for v in(sx,sp,sm.slot_mapping,bt,sm.tree_attention_mask)]
    first_graph_decode=output_hidden[:80].detach().cpu().clone()
    baseline_exact=bool(torch.equal(first_graph_decode,decode))and all(torch.equal(a,b)for a,b in zip(kv(decode_slots),decode_KV,strict=True))
    cases=[]
    for family,owner,start,lengths in(('initial',72,0,(1,2,3,4,5,17,40)),('continuation',16,127,(17,40))):
        rh,rt,rkv=references[family]
        for length in lengths:
            name=family+'-L'+str(length);f=frame(owner,start,tokens[:length])
            load(base);graph.replay();torch.npu.synchronize()
            stable_base=bool(torch.equal(output_hidden[:80].detach().cpu(),first_graph_decode))
            load(f);graph.replay();torch.npu.synchronize()
            cpu=output_hidden.detach().cpu().clone();actual=cpu[80:80+length];got_tokens=sampled(output_hidden)
            qslots=torch.tensor([r['slot']for r in f[3]],dtype=torch.long,device=engine.device)
            actual_KV=kv(qslots);wanted_KV=[v[:length]for v in rkv]
            case=dict(name=name,family=family,prefix_length=start,known_prefill_queries=length,
                candidate_owner=owner,independent_serial_reference_owner=73 if family=='initial'else 40,
                actual_rows=f[3],physical_groups=sorted({r['group']for r in f[3]}),
                graph_baseline_matches_eager=baseline_exact,baseline_before_case_stable=stable_base,
                eager_reference_decode_stable=all(reference_checks[family]),
                prefill_hidden_exact=bool(torch.equal(actual,rh[:length])),
                prefill_hidden_max_abs=float((actual.float()-rh[:length].float()).abs().max()),
                actual_tokens=[int(v)for v in got_tokens[80:80+length]],reference_tokens=rt[:length],
                prefill_tokens_exact=[int(v)for v in got_tokens[80:80+length]]==rt[:length],
                prefill_all128_query_KV_exact=all(torch.equal(a,b)for a,b in zip(actual_KV,wanted_KV,strict=True)),
                decode_hidden_exact=bool(torch.equal(cpu[:80],decode)),decode_tokens_exact=bool(torch.equal(got_tokens[:80],decode_tokens)),
                decode_all128_query_KV_exact=all(torch.equal(a,b)for a,b in zip(kv(decode_slots),decode_KV,strict=True)),
                query_KV=[dict(index=i,equal=bool(torch.equal(a,b)),actual_sha256=digest(a),reference_sha256=digest(b),max_abs=float((a.float()-b.float()).abs().max()))for i,(a,b)in enumerate(zip(actual_KV,wanted_KV,strict=True))])
            local_gate=dict(rank=engine.rank,name=name,exact_before=all(case[k]is True for k in KEYS))
            all_gates=[None]*3;dist.all_gather_object(all_gates,local_gate,group=group)
            case['actual_all3_pre_timing_gate']=all_gates
            save('single-graph448-'+name+'-before-timing',case)
            pair={'baseline':[],name:[]}
            if all(v['exact_before']is True for v in all_gates):
                for repeat in range(9):
                    for kind in(('baseline',name)if repeat%2==0 else(name,'baseline')):
                        dist.barrier(group=group);load(base if kind=='baseline'else f)
                        begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True);begin.record()
                        for _ in range(10):graph.replay()
                        end.record();end.synchronize();value=float(begin.elapsed_time(end))/10
                        if not math.isfinite(value)or value<=0:raise RuntimeError('ROOT448 actual positive model-body device event')
                        pair[kind].append(value)
                case['timing_skipped']=None
            else:case['timing_skipped']='actual independent numerical mismatch; no timing claim'
            load(f);graph.replay();torch.npu.synchronize()
            case['post_replay_hidden_stable']=bool(torch.equal(output_hidden[80:80+length].detach().cpu(),actual))
            case['post_replay_prefill128_KV_exact']=all(torch.equal(a,b)for a,b in zip(kv(qslots),wanted_KV,strict=True))
            case['device_event_ms']=pair
            cases.append(case);save('single-graph448-case-progress',dict(rank=engine.rank,cases=cases))
    if static_ptrs!=[v.data_ptr()for v in(sx,sp,sm.slot_mapping,bt,sm.tree_attention_mask)]:raise RuntimeError('ROOT448 exact static input buffers held for every replay')
    if params!=[(n,t.data_ptr())for n,t in engine.model.named_parameters()]or(manifest.digest if manifest else None)!=manifest_digest:
        raise RuntimeError('ROOT448 original model weights and MC2 bindings')
    weight_hashes_after={name:digest(value)for name,value in engine.model.named_parameters()}
    save('single-graph448-weight-hashes-after',weight_hashes_after)
    if weight_hashes_before!=weight_hashes_after:raise RuntimeError('ROOT448 actual model parameter bytes changed')
    local=dict(rank=engine.rank,cases=cases,all9_cases_math_exact=all(all(c[k]is True for k in(*KEYS,'post_replay_hidden_stable','post_replay_prefill128_KV_exact'))for c in cases),
        graph_baseline_matches_eager=baseline_exact,eager_reference_decode_checks=reference_checks,
        exactly_one_persistent_diagnostic_graph=True,all_reference_model_calls_before_capture=True,
        ordinary_model_body_calls_after_capture=0,
        no_graph_reset_or_empty_cache_during_epoch=True,static_input_pointers_unchanged=True,
        page_inventory_before=before,page_inventory_after=tables,decode_query_slots=decode_host_slots,
        all_parameter_storages_unchanged=True,all_parameter_bytes_SHA256_unchanged=True,MC2_manifest_unchanged=True)
    gathered=[None]*3;dist.all_gather_object(gathered,local,group=group)
    costs={}
    for i,case in enumerate(cases):
        name=case['name']
        if any(v['cases'][i]['timing_skipped']is not None for v in gathered):costs[name]=None;continue
        values={k:[max(v['cases'][i]['device_event_ms'][k][j]for v in gathered)for j in range(9)]for k in('baseline',name)}
        costs[name]=dict(baseline_samples_ms=values['baseline'],variant_samples_ms=values[name],baseline_p50_ms=statistics.median(values['baseline']),variant_p50_ms=statistics.median(values[name]))
    report=dict(schema='ROOT448.one-persistent-graph-known-prefill-isolation.v1',rank=engine.rank,by_target_rank=gathered,
        all3_all9_cases_math_exact=all(v['all9_cases_math_exact']for v in gathered),paired_max_rank_costs=costs,
        physical_M=120,physical_S=24,diagnostic_graphs=1,Native_flags_assigned=False,
        production_permission=False,whole_workload_AR_qualified=False,Raw=None,Goodput=None,adopted=False)
    path=Path(output)/f'rank{engine.rank}-single-graph-prefill448.json'
    with path.open('x')as stream:json.dump(report,stream,indent=2);stream.write('\n')
    return dict(report=str(path),all_parameter_storages_unchanged=True,full463=False,
        Native_flags_assigned=False,production_permission=False,Raw=None,Goodput=None)
