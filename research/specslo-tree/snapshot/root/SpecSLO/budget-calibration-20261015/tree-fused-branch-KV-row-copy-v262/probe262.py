"""Owned synthetic copy test: real Native maps, raw bits and ACLGraph replay.

No model weights, attention, decoding or serving is executed. Not qualified
until the separate frozen guard executes this after the active260 job ends.
"""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import gc,hashlib,json,statistics,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent

def main():
    output=Path(sys.argv[1]);assert not output.exists()
    os.environ['TRITON_CACHE_DIR']=str(output.parent/'triton-cache')
    sys.path[:0]=[str(P),str(B/'tree-deferred-FULL-mask-host-projection-v259')]
    from fixture259 import build
    from contract262 import checked
    env=build();engine=env['engine'];pad=env['pad'];torch=env['torch']
    from oracle255 import cases,prepare
    from branch_tail255 import layout
    maps=[]
    for case in cases():
        prefixes=[[11]*1505,[12]*2555]+[[13+i]*127 for i in range(22)]
        plans,values=prepare(case,prefixes);n=case['N']
        pp,rr,dd,ss,real=pad(engine,plans,list(range(41,41+n)),values,list(range(n)),[],[])
        physical=[p for plan in pp for p in plan.cache_positions.tolist()]
        owners=[owner for owner,plan in zip(ss,pp,strict=True)for _ in range(plan.candidate_budget+1)]
        engine._prepare_attention_metadata(owners,physical,True)
        desc=layout(pp,ss,engine.cache_allocation.block_tables,engine._tree_graph_scratch_sequence_ids)
        if desc['enabled']:
            proof=checked(desc['source'],desc['dest'],rows=69632)
            maps.append(dict(case=case['name'],source=desc['source'],destination=desc['dest'],proof=proof))
    assert len(maps)==172 and 'torch_npu'not in sys.modules
    # The CPU fixture created only these private stand-in modules. Remove them
    # before loading the real installed Triton helper in this isolated child.
    removed=[name for name in sys.modules if name=='vllm_ascend'or name.startswith('vllm_ascend.')]
    for name in removed:sys.modules.pop(name)
    import torch_npu
    torch.npu.set_device(1);torch.npu.config.allow_internal_format=True
    device=torch.device('npu:1')
    from copy262 import copy_rows
    rows=[];timings=[];bitwise=[];eager_controls=[]
    for width in(256,384):
        generator=torch.Generator(device='cpu');generator.manual_seed(262+width)
        basekey=torch.randn((69632,width),generator=generator,dtype=torch.bfloat16,device='cpu').to(device)
        basevalue=torch.randn((69632,width),generator=generator,dtype=torch.bfloat16,device='cpu').to(device)
        oldkey=basekey.clone();oldvalue=basevalue.clone();newkey=basekey.clone();newvalue=basevalue.clone()
        source=torch.arange(1792,dtype=torch.long,device='cpu').to(device);destination=source.clone()
        def original():
            oldkey.index_copy_(0,destination,oldkey.index_select(0,source))
            oldvalue.index_copy_(0,destination,oldvalue.index_select(0,source))
        def candidate():copy_rows(newkey,newvalue,source,destination)
        for _ in range(3):original();candidate()
        torch.npu.synchronize()
        first=maps[0];checked(first['source'],first['destination'],rows=69632)
        source.copy_(torch.tensor(first['source'],dtype=torch.long,device='cpu'))
        destination.copy_(torch.tensor(first['destination'],dtype=torch.long,device='cpu'))
        original();candidate();torch.npu.synchronize()
        assert bool((oldkey==newkey).all().item())and bool((oldvalue==newvalue).all().item())
        eager_controls.append(dict(width=width,case=first['case'],uncaptured_nonidentity_KV_equal=True))
        oldgraph=torch.npu.NPUGraph();newgraph=torch.npu.NPUGraph()
        with torch.npu.graph(oldgraph):original()
        with torch.npu.graph(newgraph):candidate()
        torch.npu.synchronize()
        def install(mapping):
            checked(mapping['source'],mapping['destination'],rows=69632)
            source.copy_(torch.tensor(mapping['source'],dtype=torch.long,device='cpu'))
            destination.copy_(torch.tensor(mapping['destination'],dtype=torch.long,device='cpu'))
        def reset():
            oldkey.copy_(basekey);oldvalue.copy_(basevalue);newkey.copy_(basekey);newvalue.copy_(basevalue)
        def same():
            assert bool((oldkey==newkey).all().item())and bool((oldvalue==newvalue).all().item())
        for mapping in maps:
            install(mapping);reset();oldgraph.replay();newgraph.replay();torch.npu.synchronize();same()
            rows.append(dict(width=width,case=mapping['case'],actual_original_native_map=True,
                HOST_contract=mapping['proof'],all_full_cache_values_equal=True,actual_ACLGraph_replayed=True))
        for name in('N1-fork5-anchor0','N8-fork5-anchor0','N12-fork5-anchor0','N16-mixed-anchor0','N23-fork5-anchor0'):
            mapping=next(m for m in maps if m['case']==name);install(mapping)
            paired=[];values=dict(original=[],fused=[])
            for pair in range(80):
                order=('original','fused')if pair%2==0 else('fused','original');measured={}
                for key in order:
                    reset();torch.npu.synchronize();begin=time.perf_counter()
                    (oldgraph if key=='original'else newgraph).replay()
                    torch.npu.synchronize();elapsed=(time.perf_counter()-begin)*1000
                    measured[key]=elapsed;values[key].append(elapsed)
                paired.append(dict(order=order,milliseconds=measured))
            reset();oldgraph.replay();newgraph.replay();torch.npu.synchronize();same()
            timings.append(dict(width=width,case=name,nonidentity=mapping['proof']['nonidentity'],
                paired_calls=80,median_ms={k:statistics.median(v)for k,v in values.items()},pairs=paired))
        del oldgraph,newgraph,oldkey,oldvalue,newkey,newvalue,basekey,basevalue
        torch.npu.synchronize();gc.collect();torch.npu.empty_cache()
        # All65536 possible raw16-bit payloads, including signedzero and NaN
        # payloads, are checked on CPU as int16 after actual device execution.
        count=4096*width;bits=(torch.arange(count,dtype=torch.long,device='cpu')%65536).to(torch.int16).reshape(4096,width)
        valbits=((torch.arange(count,dtype=torch.long,device='cpu')*73+17)%65536).to(torch.int16).reshape(4096,width)
        key=bits.view(torch.bfloat16).to(device);value=valbits.view(torch.bfloat16).to(device)
        source=torch.arange(1792,dtype=torch.long,device='cpu').to(device)
        destination=(torch.arange(1792,dtype=torch.long,device='cpu')+2048).to(device)
        copy_rows(key,value,source,destination);torch.npu.synchronize()
        graph=torch.npu.NPUGraph()
        with torch.npu.graph(graph):copy_rows(key,value,source,destination)
        torch.npu.synchronize()
        for variant in(0,1,0):
            src=list(range(variant*64,variant*64+1792));dst=list(range(2048+variant*64,2048+variant*64+1792))
            checked(src,dst,rows=4096)
            inputkey=(bits.to(torch.int32)+variant*311).to(torch.int16)
            inputvalue=(valbits.to(torch.int32)+variant*97).to(torch.int16)
            expectedkey=inputkey.clone();expectedvalue=inputvalue.clone()
            cpu_src=torch.tensor(src,dtype=torch.long,device='cpu');cpu_dst=torch.tensor(dst,dtype=torch.long,device='cpu')
            expectedkey.index_copy_(0,cpu_dst,inputkey.index_select(0,cpu_src))
            expectedvalue.index_copy_(0,cpu_dst,inputvalue.index_select(0,cpu_src))
            key.copy_(inputkey.view(torch.bfloat16));value.copy_(inputvalue.view(torch.bfloat16))
            source.copy_(cpu_src);destination.copy_(cpu_dst);graph.replay();torch.npu.synchronize()
            assert torch.equal(key.cpu().view(torch.int16),expectedkey)and torch.equal(value.cpu().view(torch.int16),expectedvalue)
            bitwise.append(dict(width=width,variant=variant,all65536_raw16_bit_patterns=True,
                complete_K_and_V_bitwise_equal=True,changed_and_restored_same_buffers=True,actual_ACLGraph_replayed=True))
        del graph,key,value,source,destination;torch.npu.synchronize();gc.collect();torch.npu.empty_cache()
    payload=dict(schema='ROOT262.actual-synthetic-fused-KV-copy-ACLGraph.v1',status='complete',
        source_sha256=hashlib.sha256((P/'copy262.py').read_bytes()).hexdigest(),
        candidate_source_sha256=hashlib.sha256((P/'copy262.py').read_bytes()).hexdigest(),
        physical_device=5,logical_device=1,torch_version=torch.__version__,torch_npu_version=torch_npu.__version__,
        all172_actual_Native_maps_each_of_two_widths=True,map_cases=rows,raw_bit_replays=bitwise,timings=timings,eager_controls=eager_controls,
        fake_CPU_modules_removed=removed,Triton_compiled_before_any_graph_capture=True,
        model_attention_or_KV_executed=True,synthetic_KV_copy_executed=True,attention_executed=False,
        models_loaded=False,real_model_weights_loaded=False,neural_qualified=False,Goodput=None,
        production_installed=False,adopted=False,
        timing_scope='paired isolated ACLGraph replay+synchronize; original4copyops versus fused1; no model or end-to-end speedup inferred')
    with output.open('x')as f:json.dump(payload,f,indent=2);f.write('\n')
    print(json.dumps(dict(status='complete',map_cases=len(rows),raw_bit_replays=len(bitwise),
        timings=[dict(width=r['width'],case=r['case'],median_ms=r['median_ms'])for r in timings])))
if __name__=='__main__':main()
