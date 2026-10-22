"""Whole32B unfused reference, actual qualified160 and current120 cost."""
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import math
import statistics

from frame454 import extra_slots


def unfused_forward(model,x,positions,metadata):
    """Ordinary Native attention/MLP reductions with explicit residual norms.

    No model attribute or Native validation flag is changed. The same real
    modules and weights supply the independent ordinary computation.
    """
    import torch
    hidden=model.embed_tokens(x);residual=None
    for layer in model.layers:
        if residual is None:
            residual=hidden;hidden=layer.input_layernorm(hidden)
        else:hidden,residual=layer.input_layernorm(hidden,residual)
        hidden=layer.self_attn(positions,hidden,metadata)
        hidden,residual=layer.post_attention_layernorm(hidden,residual)
        hidden=layer.mlp(hidden)
    if model.track_cache_finiteness:
        model.output_nonfinite.logical_or_(~(torch.isfinite(hidden).all()&torch.isfinite(residual).all()))
    hidden,_=model.norm(hidden,residual)
    if model.track_cache_finiteness:model.output_nonfinite.logical_or_(~torch.isfinite(hidden).all())
    return hidden


def measure(engine,payloads,output):
    import torch
    import torch.distributed as dist
    state=engine._branch_tail_state;current=state.static_current
    if engine.is_draft or state.enabled or current['real_N']!=24 or current['real_queries']!=120:
        raise RuntimeError('ROOT455 original canonical real24 B4 frame')
    if 160 not in engine.model._mc2_chained_row_counts or 160 in engine.model._mc2_deferred_row_counts:
        raise RuntimeError('ROOT455 actual constructor must naturally bind fully-flushed160')
    group=engine.model.layers[0].context.group
    if dist.get_world_size(group=group)!=3:raise RuntimeError('ROOT455 actual target TP3')
    output=Path(output)
    def save(name,value):
        with(output/f'rank{engine.rank}-{name}.json').open('w')as f:json.dump(value,f,indent=2);f.write('\n')
    def digest(value):
        cpu=value.detach().cpu().contiguous()
        return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
    parameters=[(n,v.data_ptr())for n,v in engine.model.named_parameters()]
    before={n:digest(v)for n,v in engine.model.named_parameters()};save('whole455-weights-before',before)
    manifest=engine.mc2_static_route_manifest;manifest_digest=manifest.digest
    x,p,m,_,_=current['physical'];tables=[list(row)for row in engine.cache_allocation.block_tables]
    prefixes=engine._root445_fixture_prefixes
    owners,slots=extra_slots(p.detach().cpu().tolist(),tables,[len(v)for v in prefixes])
    base_slots=m.slot_mapping.detach().cpu().tolist()
    if len(set(base_slots+slots))!=160:raise RuntimeError('ROOT455 all160 writes private and real')
    xx=torch.cat((x,x[:40]));pp=torch.cat((p,p[:40]))
    extra=torch.tensor([[v if v>=0 else state.static_zero_page for v in tables[owner]]for owner in owners],dtype=m.request_block_tables.dtype,device=m.request_block_tables.device)
    bt=torch.cat((m.request_block_tables,extra))
    mm=replace(m,slot_mapping=torch.cat((m.slot_mapping,torch.tensor(slots,dtype=m.slot_mapping.dtype,device=m.slot_mapping.device))),
        context_lens=torch.cat((m.context_lens,m.context_lens[:40])),block_tables=bt,request_block_tables=bt,
        actual_seq_lengths_q=tuple(5*i for i in range(1,33)),sequence_lens=(2560,)*32,
        tree_attention_mask=torch.cat((m.tree_attention_mask,m.tree_attention_mask[:8])))
    if tuple(mm.tree_attention_mask.shape)!=(32,1,10,4096):raise RuntimeError('ROOT455 current fixed FULL160 frame')
    qslots={120:base_slots,160:base_slots+slots}
    def kv(count):
        index=torch.tensor(qslots[count],dtype=torch.long,device=engine.device)
        return [cache.flatten(0,1).index_select(0,index).detach().cpu().contiguous()
            for layer in engine.model.layers for cache in(layer.self_attn.key_cache,layer.self_attn.value_cache)]
    def ids(h):return engine.model.compute_greedy_tokens(h,engine.draft_vocab_size).detach().cpu().clone()
    def run_control():
        with state.active(False):return unfused_forward(engine.model,xx,pp,mm).clone()
    def run160():
        with state.active(False):return engine.model(xx,pp,mm).clone()
    def run120():
        with state.active(False):return engine.model(x,p,m).clone()
    # CPU copies precede any following graph or HCCL/LM-head operation.
    reference=run_control();torch.npu.synchronize()
    ref_cpu=reference.detach().cpu().clone();ref_kv=kv(160);ref_ids=ids(reference)
    actual=run160();torch.npu.synchronize()
    actual_cpu=actual.detach().cpu().clone();actual_kv=kv(160);actual_ids=ids(actual)
    current120=run120();torch.npu.synchronize()
    gold120=(current120.detach().cpu().clone(),kv(120),ids(current120))
    eager=dict(hidden_exact=bool(torch.equal(actual_cpu,ref_cpu)),tokens_exact=bool(torch.equal(actual_ids,ref_ids)),
        all128_query_KV_exact=all(torch.equal(a,b)for a,b in zip(actual_kv,ref_kv,strict=True)),
        hidden_max_abs=float((actual_cpu.float()-ref_cpu.float()).abs().max()),
        query_KV=[dict(index=i,equal=bool(torch.equal(a,b)),actual_sha256=digest(a),reference_sha256=digest(b),max_abs=float((a.float()-b.float()).abs().max()))for i,(a,b)in enumerate(zip(actual_kv,ref_kv,strict=True))])
    save('whole455-eager-before-capture',eager)
    operations={'control160':run_control,'qualified160':run160,'current120':run120}
    graphs={};outputs={}
    # Finish ordinary model warmups before any new capture, and retain all
    # graph-owned inputs/outputs without reset throughout both shape families.
    for operation in operations.values():
        for _ in range(3):operation()
    torch.npu.synchronize()
    for name,operation in operations.items():
        dist.barrier(group=group)
        graph=torch.npu.NPUGraph()
        with torch.npu.graph(graph):h=operation()
        graphs[name]=graph;outputs[name]=h
        for _ in range(3):graph.replay()
        torch.npu.synchronize()
    # Retain every graph, original frame, static input and parameter owner.
    engine._root455_whole_resources=(graphs,outputs,(x,p,m,xx,pp,mm),reference,actual,current120)
    def check(name,*,replay=True):
        if replay:graphs[name].replay()
        torch.npu.synchronize()
        cpu=outputs[name].detach().cpu().clone()
        count=120 if name=='current120' else 160
        wanted,wkv,wids=gold120 if count==120 else(ref_cpu,ref_kv,ref_ids)
        actualkv=kv(count);actualids=ids(outputs[name])
        return dict(name=name,hidden_exact=bool(torch.equal(cpu,wanted)),tokens_exact=bool(torch.equal(actualids,wids)),
            all128_query_KV_exact=all(torch.equal(a,b)for a,b in zip(actualkv,wkv,strict=True)),
            hidden_max_abs=float((cpu.float()-wanted.float()).abs().max()),actual_hidden_sha256=digest(cpu),reference_hidden_sha256=digest(wanted))
    initial={name:check(name)for name in operations}
    local_gate=dict(rank=engine.rank,exact=eager['hidden_exact']and eager['tokens_exact']and eager['all128_query_KV_exact']and all(all(v[k]is True for k in('hidden_exact','tokens_exact','all128_query_KV_exact'))for v in initial.values()))
    all_gates=[None]*3;dist.all_gather_object(all_gates,local_gate,group=group)
    timings={name:[]for name in operations};post=[]
    if all(v['exact']is True for v in all_gates):
        for repeat in range(9):
            names=list(operations);rotation=repeat%3;names=names[rotation:]+names[:rotation]
            if repeat%2:names.reverse()
            for name in names:
                dist.barrier(group=group)
                begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True);begin.record()
                for _ in range(20):graphs[name].replay()
                end.record();end.synchronize();elapsed=float(begin.elapsed_time(end))/20
                if not math.isfinite(elapsed)or elapsed<=0:raise RuntimeError('ROOT455 actual positive whole-model event')
                timings[name].append(elapsed)
                # Read all outputs before the next collective can reuse views.
                post.append(dict(repeat=repeat,**check(name,replay=False)))
    final={name:check(name)for name in operations}
    after={n:digest(v)for n,v in engine.model.named_parameters()};save('whole455-weights-after',after)
    if before!=after or parameters!=[(n,v.data_ptr())for n,v in engine.model.named_parameters()]or manifest.digest!=manifest_digest:
        raise RuntimeError('ROOT455 unchanged model parameters and constructor route bindings')
    local=dict(rank=engine.rank,eager=eager,initial_graph_checks=initial,actual_all3_pre_timing_gate=all_gates,
        timings_ms=timings,timing_skipped=None if all(v['exact']for v in all_gates)else'actual unfused/full-model mismatch',
        post_timing_checks=post,final_graph_checks=final,
        all_graph_checks_exact=all(all(v[k]is True for k in('hidden_exact','tokens_exact','all128_query_KV_exact'))for v in[*initial.values(),*post,*final.values()]),
        physical_M160=160,physical_S160=32,real160_queries=160,real160_owners=list(range(32)),
        actual_chained_rows=sorted(engine.model._mc2_chained_row_counts),actual_deferred_rows=sorted(engine.model._mc2_deferred_row_counts),
        original_unfused_reference_uses_real_modules_and_weights=True,model_configuration_flags_modified=False,
        model_parameter_SHA_unchanged=True,mc2_manifest_unchanged_during_epoch=True,ordinary_prompt_prefill_unchanged=True,
        Native_flags_assigned=False,production_permission=False,full463=False,Raw=None,Goodput=None)
    gathered=[None]*3;dist.all_gather_object(gathered,local,group=group)
    costs=None
    if all(v['timing_skipped']is None for v in gathered):
        samples={name:[max(v['timings_ms'][name][i]for v in gathered)for i in range(9)]for name in operations}
        median={name:statistics.median(values)for name,values in samples.items()}
        costs=dict(paired_max_rank_samples_ms=samples,p50_ms=median,
            qualified160_over_unfused160_speedup=median['control160']/median['qualified160'],
            fully_occupied_request_rate160_over120=(32/median['qualified160'])/(24/median['current120']))
    report=dict(schema='ROOT455.actual_qualified160_full_model_reference_and_cost.v1',rank=engine.rank,by_target_rank=gathered,
        all3_eager_and_graph_math_exact=all(v['eager']['hidden_exact']and v['eager']['tokens_exact']and v['eager']['all128_query_KV_exact']and v['all_graph_checks_exact']for v in gathered),
        actual_whole_model_costs=costs,production_permission=False,whole_workload_AR_qualified=False,
        Native_flags_assigned=False,Raw=None,Goodput=None,adopted=False)
    path=output/f'rank{engine.rank}-whole455-final.json'
    with path.open('x')as f:json.dump(report,f,indent=2);f.write('\n')
    return dict(report=str(path),all_parameter_storages_unchanged=True,full463=False,Native_flags_assigned=False,production_permission=False,Raw=None,Goodput=None)
