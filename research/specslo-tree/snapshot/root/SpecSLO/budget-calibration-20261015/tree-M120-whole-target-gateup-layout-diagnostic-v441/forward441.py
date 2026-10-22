"""One genuine full target frame, transient gate-up layouts, exact outputs.

This grants no production scope. Old parameter storages are always restored;
no Native validation flag or MC2 down-projection route is modified.
"""
import hashlib,json,statistics
from pathlib import Path


def measure(engine,unused_payloads,output):
    import torch,torch_npu
    import torch.distributed as dist
    from vllm_ascend.spec_decode.pearl.native_model import ACL_FORMAT_FRACTAL_NZ
    if engine.is_draft:raise RuntimeError('ROOT441 target roles only')
    state=engine._branch_tail_state
    x,p,meta,inverse,_=state.static_current['physical']
    if x.shape[0]!=120 or len(engine.model.layers)!=64:raise RuntimeError('ROOT441 exact real M120/64-layer target')
    group=engine.model.layers[0].context.group
    if dist.get_world_size(group=group)!=3:raise RuntimeError('ROOT441 actual borrowed TP3 group')
    slots=meta.slot_mapping.reshape(-1).long()
    slots=slots[slots>=0]
    if slots.numel()!=120:raise RuntimeError('ROOT441 complete120 real query slots')
    row_bindings=[(layer.self_attn.o_proj.weight.data_ptr(),layer.mlp.down_proj.weight.data_ptr())for layer in engine.model.layers]
    manifest=engine.mc2_static_route_manifest
    manifest_digest=manifest.digest if manifest is not None else None
    originals=[];graphs={};outputs={};numerical={};kv={};layout_documents=[]
    def digest(t):
        cpu=t.detach().cpu().contiguous()
        return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
    def query_KV():
        tensors=[]
        for layer in engine.model.layers:
            for cache in(layer.self_attn.key_cache,layer.self_attn.value_cache):
                flat=cache.reshape(cache.shape[0]*cache.shape[1],-1)
                tensors.append(flat.index_select(0,slots).detach().cpu().contiguous())
        return tensors
    def capture(name):
        for _ in range(3):state.eager_last()
        torch.npu.synchronize()
        g=torch.npu.NPUGraph()
        with torch.npu.graph(g):h=state.eager_last()
        graphs[name]=g;outputs[name]=h
        for _ in range(3):g.replay()
        torch.npu.synchronize()
        return h.detach().cpu().contiguous(),query_KV()
    try:
        nd_hidden,nd_KV=capture('ND')
        for i,layer in enumerate(engine.model.layers):
            module=layer.mlp.gate_up_proj;original=module.weight.data
            if module.bias is not None or original.dtype!=torch.bfloat16:raise RuntimeError('ROOT441 actual BF16 bias-free gate-up')
            alternate=module.large_m_nz_weight
            nz=alternate if alternate is not None else torch_npu.npu_format_cast(original,ACL_FORMAT_FRACTAL_NZ)
            if nz.shape!=original.shape or nz.dtype!=original.dtype:raise RuntimeError('ROOT441 layout-only same shape/dtype')
            originals.append((module,original,nz,alternate))
            layout_documents.append(dict(layer=i,shape=list(original.shape),dtype=str(original.dtype),reused_original_mode14_NZ_copy=alternate is not None))
            module.weight.data=nz
        nz_hidden,nz_KV=capture('NZ')
        diff=(nd_hidden.float()-nz_hidden.float()).abs()
        numerical=dict(all_hidden_bitwise_equal=bool(torch.equal(nd_hidden,nz_hidden)),hidden_max_abs_difference=float(diff.max()),hidden_mean_abs_difference=float(diff.mean()),ND_hidden_sha256=digest(nd_hidden),NZ_hidden_sha256=digest(nz_hidden),all128_query_KV_bitwise_equal=all(torch.equal(a,b)for a,b in zip(nd_KV,nz_KV,strict=True)),query_KV=[dict(index=i,equal=bool(torch.equal(a,b)),ND_sha256=digest(a),NZ_sha256=digest(b),max_abs_difference=float((a.float()-b.float()).abs().max()))for i,(a,b)in enumerate(zip(nd_KV,nz_KV,strict=True))])
        if not torch.isfinite(nd_hidden).all()or not torch.isfinite(nz_hidden).all():raise RuntimeError('ROOT441 finite genuine full-frame outputs')
        # Both captured graphs retain their own weight storage; restore the
        # model parameters before timing either graph and before any exit.
        for module,old,_,alternate in originals:
            module.weight.data=old
            if module.large_m_nz_weight is not alternate:raise RuntimeError('ROOT441 original alternate storage retained')
        torch.npu.synchronize()
        for name,g in graphs.items():
            for _ in range(10):g.replay()
            torch.npu.synchronize()
        timings={'ND':[],'NZ':[]};orders=[];replays=10
        for sample in range(9):
            order=('ND','NZ')if sample%2==0 else('NZ','ND');orders.append(list(order))
            for name in order:
                dist.barrier(group=group)
                begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
                begin.record()
                for _ in range(replays):graphs[name].replay()
                end.record();end.synchronize()
                value=float(begin.elapsed_time(end))/replays
                if value<=0:raise RuntimeError('ROOT441 positive actual full-forward event timing')
                timings[name].append(value)
        # Recheck graph outputs after interleaved replays and restored params.
        nd_graph_hidden=outputs['ND'].detach().cpu().contiguous()
        nz_graph_hidden=outputs['NZ'].detach().cpu().contiguous()
        numerical['ND_graph_output_stable_after_replays']=bool(torch.equal(nd_graph_hidden,nd_hidden))
        numerical['NZ_graph_output_stable_after_replays']=bool(torch.equal(nz_graph_hidden,nz_hidden))
    finally:
        for module,old,_,_ in originals:module.weight.data=old
        torch.npu.synchronize()
    restored=all(module.weight.data_ptr()==old.data_ptr()for module,old,_,_ in originals)
    after=[(layer.self_attn.o_proj.weight.data_ptr(),layer.mlp.down_proj.weight.data_ptr())for layer in engine.model.layers]
    if len(originals)!=64 or not restored or after!=row_bindings or (manifest.digest if manifest is not None else None)!=manifest_digest:raise RuntimeError('ROOT441 all original parameters and static MC2 rows restored')
    local=dict(rank=engine.rank,numerical=numerical,whole_target_forward_event_ms=timings,sample_orders=orders,replays_per_sample=replays,changed_layer_count=64,layout_documents=layout_documents,all_original_parameter_storages_restored=True,all_MC2_row_weight_bindings_unchanged=True)
    all_ranks=[None]*3;dist.all_gather_object(all_ranks,local,group=group)
    summaries={}
    for name in('ND','NZ'):
        values=[max(r['whole_target_forward_event_ms'][name][j]for r in all_ranks)for j in range(9)]
        summaries[name]=dict(actual_max_rank_samples_ms=values,p50_ms=statistics.median(values))
    report=dict(schema='ROOT441.actual-whole-target-gateup-layout-cost.v1',rank=engine.rank,actual_input_M=120,physical_S=24,TP_world_size=3,by_target_rank=all_ranks,max_rank_event_costs=summaries,p50_speedup=summaries['ND']['p50_ms']/summaries['NZ']['p50_ms'],all3_hidden_exact=all(r['numerical']['all_hidden_bitwise_equal']for r in all_ranks),all3_query_KV_exact=all(r['numerical']['all128_query_KV_bitwise_equal']for r in all_ranks),actual_model_frame_scope='one real N24 all4 M120/S24 frame; only target body, excludes LM head/driver/arrivals/full workload',transient_gateup_layout_swap_only=True,all_parameter_storages_restored=True,MC2_profile_replaced=False,Native_flags_assigned=False,whole_workload_AR_qualified=False,production_permission=False,Raw=None,Goodput=None,adopted=False)
    path=Path(output)/f'rank{engine.rank}-whole-target-layout441.json'
    with path.open('x')as f:json.dump(report,f,indent=2);f.write('\n')
    return dict(report=str(path),changed_gateup_layers=64,all_original_parameter_storages_restored=True,whole_target_frame=True,full463=False,Native_flags_assigned=False,production_permission=False,Raw=None,Goodput=None)
