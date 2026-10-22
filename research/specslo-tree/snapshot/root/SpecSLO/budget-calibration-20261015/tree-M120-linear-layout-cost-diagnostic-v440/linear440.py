"""Actual layer0 inputs: local BF16 matmul layout/padding cost, no service."""
import json,statistics
from pathlib import Path

VARIANTS=('ND120','NZ120','NDpad128','NZpad128')

def measure(engine,payloads,output):
    import torch,torch_npu
    import torch.nn.functional as F
    import torch.distributed as dist
    from vllm_ascend.spec_decode.pearl.native_model import ACL_FORMAT_FRACTAL_NZ
    if engine.is_draft or set(payloads)!={'attention','down','gate_up'}:
        raise RuntimeError('ROOT440 genuine all3 target projection payloads')
    group=engine.model.layers[0].context.group
    if dist.get_world_size(group=group)!=3:raise RuntimeError('ROOT440 borrowed current TP3 group')
    results=[];ops_per_graph=16;replays=10;samples=9
    for role in('attention','gate_up','down'):
        x=payloads[role]['activation'];weight=payloads[role]['weight']
        if x.ndim!=2 or x.shape[0]!=120 or weight.ndim!=2 or x.shape[1]!=weight.shape[1]or x.dtype!=torch.bfloat16 or weight.dtype!=torch.bfloat16:
            raise RuntimeError('ROOT440 actual two-dimensional BF16 M120 tensors')
        xpad=torch.cat((x,torch.zeros((8,x.shape[1]),dtype=x.dtype,device=x.device)))
        nz=torch_npu.npu_format_cast(weight.detach(),ACL_FORMAT_FRACTAL_NZ)
        reference=F.linear(x,weight).detach().clone()
        inputs={'ND120':(x,weight),'NZ120':(x,nz),'NDpad128':(xpad,weight),'NZpad128':(xpad,nz)}
        graphs={};outputs={};comparison={}
        for name,(a,w)in inputs.items():
            for _ in range(10):F.linear(a,w)
            torch.npu.synchronize()
            graph=torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for _ in range(ops_per_graph):last=F.linear(a,w)
            graphs[name]=graph;outputs[name]=last
            graph.replay();torch.npu.synchronize()
            actual=last[:120];difference=(actual.float()-reference.float()).abs()
            comparison[name]=dict(bitwise_equal_to_ND120=bool(torch.equal(actual,reference)),
                maximum_abs_difference=float(difference.max().cpu()),
                mean_abs_difference=float(difference.mean().cpu()),finite=bool(torch.isfinite(actual).all().cpu()))
            if not comparison[name]['finite']:raise RuntimeError('ROOT440 finite actual operator outputs')
            for _ in range(20):graph.replay()
            torch.npu.synchronize()
        timings={name:[]for name in VARIANTS};orders=[]
        for sample in range(samples):
            rotated=VARIANTS[sample%4:]+VARIANTS[:sample%4]
            order=rotated if sample%2==0 else tuple(reversed(rotated));orders.append(list(order))
            for name in order:
                dist.barrier(group=group)
                start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
                start.record()
                for _ in range(replays):graphs[name].replay()
                end.record();end.synchronize()
                ms=float(start.elapsed_time(end))/(ops_per_graph*replays)
                if not ms>0:raise RuntimeError('ROOT440 positive actual device-event time')
                timings[name].append(ms)
        rank_record=dict(rank=engine.rank,projection=role,actual_input_rows=120,
            input_shape=list(x.shape),weight_shape=list(weight.shape),dtype=str(weight.dtype),
            comparison=comparison,device_event_ms_per_local_matmul=timings,
            sample_orders=orders,ops_per_graph=ops_per_graph,replays_per_sample=replays,
            input_scope='genuine current layer0 target M120 activations and immutable current model weights; pad128 adds8 zero rows',
            attention_or_communication_included=False,tensors_serialized=False)
        gathered=[None]*3;dist.all_gather_object(gathered,rank_record,group=group)
        summaries={}
        for name in VARIANTS:
            maximum=[max(r['device_event_ms_per_local_matmul'][name][j]for r in gathered)for j in range(samples)]
            summaries[name]=dict(max_TP_rank_samples_ms=maximum,p50_ms=statistics.median(maximum))
        results.append(dict(projection=role,by_target_rank=gathered,balanced_max_rank_costs=summaries))
        del graphs,outputs,inputs,nz,xpad,reference
    report=dict(schema='ROOT440.genuine-M120-local-matmul-layout-pad-diagnostic.v1',rank=engine.rank,
        TP_world_size=3,roles=results,physical_cards=[2,5,6,7],
        model_or_communication_profile_installed=False,Native_flags_assigned=False,
        performance_scope='local matrix operations only; excludes attention/allreduce/host scheduling and complete generation',
        Raw=None,Goodput=None,whole_workload_AR_qualified=False,production_permission=False,adopted=False)
    p=Path(output)/f'rank{engine.rank}-linear-layout-cost440.json'
    with p.open('x')as f:json.dump(report,f,indent=2);f.write('\n')
    return dict(report=str(p),actual_projections=3,local_matrix_samples=samples,full463=False,
        production_permission=False,Native_flags_assigned=False,Raw=None,Goodput=None)
