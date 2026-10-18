"""Isolated all64-layer synthetic cache operators; no model or attention."""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import gc,hashlib,json,statistics,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
K=B/'tree-prefix-precopy-dual-cachewrite-host-v288'
SPAN=B/'tree-contiguous-prefix-span-copy-host-v292'

def maps_HOST():
    sys.path[:0]=[str(K),str(SPAN),str(B/'tree-dynamic-root-alternate-eager-compat-full-v287'),str(B/'tree-deferred-FULL-mask-host-projection-v259')]
    import check287
    from fixture259 import build
    env=build();e=env['engine'];native=env['native']
    from padding269 import pad
    from plan269 import make
    from layout269 import layout
    from static_state287 import bind_checked_prepare
    from split288 import split
    import branch_tail255
    bind_checked_prepare(e._branch_tail_state,branch_tail255)
    native.NativePearlEngine._pad_spec_rhythm_target_tree_graph=pad
    results=[]
    specs=[(0,[5]),(0,[8]),(127,[8]),(128,[8]),(1505,[8]),(2555,[8]),
        (127,[8]*12),(128,[8]*8+[4]*8),(127,[5]*4+[4]*16),(2555,[5]+[4]*22)]
    for prefix,budgets in specs:
        n=len(budgets);prefixes=[prefix]+[127]*(n-1)
        plans=[make(b,p,0)for b,p in zip(budgets,prefixes,strict=True)]
        roots=list(range(31,31+n));draft=[[101+17*i+29*j for j in range(b)]for i,b in enumerate(budgets)]
        pp,rr,dd,ss,real=pad(e,plans,roots,draft,list(range(n)),[],[])
        for owner,plan in zip(range(n),plans,strict=True):
            length=plan.prefix_len+plan.candidate_budget+1
            e._prepare_attention_metadata([owner]*length,list(range(length)),True)
        owners=[owner for owner,plan in zip(ss,pp,strict=True)for _ in range(plan.candidate_budget+1)]
        positions=[v for plan in pp for v in plan.cache_positions.tolist()]
        e._prepare_attention_metadata(owners,positions,True)
        native.NativePearlEngine.target_tree_forward(e,pp,rr,dd,sequence_ids=ss,return_logits=False,real_tree_count=real)
        state=e._branch_tail_state
        desc=layout(pp,ss,e.cache_allocation.block_tables,e._tree_graph_scratch_sequence_ids)
        slots=state.static_current['physical'][2].slot_mapping.tolist()
        new=split(desc['source'],desc['dest'],slots)
        results.append(dict(name=f'p{prefix}-N{n}-B'+','.join(map(str,budgets)),source=desc['source'],
            destination=desc['dest'],slots=slots,split=new,grid=state.copy280_bucket))
    assert 'torch_npu'not in sys.modules
    return results

def main():
    output=Path(sys.argv[1]);assert not output.exists()
    os.environ['TRITON_CACHE_DIR']=str(output.parent/'triton-cache')
    maps=maps_HOST()
    removed=[n for n in sys.modules if n=='vllm_ascend'or n.startswith('vllm_ascend.')]
    for name in removed:sys.modules.pop(name)
    import torch,torch_npu
    torch.npu.set_device(1);torch.npu.config.allow_internal_format=True
    device=torch.device('npu:1')
    from kernels288 import dual_write
    from kernels292 import SpanCopies
    from spans292 import pack as pack_spans
    from copy278 import copy_rows
    checks=[];timings=[];coverage={}
    for width in(256,384):
        shape=(544,128,width//128,128)
        old=[(torch.zeros(shape,dtype=torch.bfloat16,device=device),torch.zeros(shape,dtype=torch.bfloat16,device=device))for _ in range(64)]
        new=[(torch.zeros(shape,dtype=torch.bfloat16,device=device),torch.zeros(shape,dtype=torch.bfloat16,device=device))for _ in range(64)]
        prefix=SpanCopies(new,device)
        spans=torch.zeros((12,3),dtype=torch.long,device=device)
        source=torch.arange(1792,dtype=torch.long,device=device);destination=source.clone()
        prefix_source=source.clone();prefix_dest=source.clone()
        slots=torch.arange(120,dtype=torch.int32,device=device)
        aliases=torch.full((120,),-1,dtype=torch.long,device=device)
        seen=torch.zeros(65536,dtype=torch.bool,device='cpu')
        for style in('contiguous','strided-QKV-slices'):
            if style=='contiguous':
                inputs=[(torch.zeros((120,width//128,128),dtype=torch.bfloat16,device=device),
                         torch.zeros((120,width//128,128),dtype=torch.bfloat16,device=device))for _ in range(64)]
                bases=[]
            else:
                bases=[torch.zeros((120,3*width),dtype=torch.bfloat16,device=device)for _ in range(64)]
                inputs=[(base[:,width:2*width].view(120,width//128,128),base[:,2*width:].view(120,width//128,128))for base in bases]
            assert all((k.stride(0)==width if style=='contiguous'else k.stride(0)==3*width)for k,v in inputs)
            def baseline():
                for (key,value),(kc,vc)in zip(inputs,old,strict=True):
                    torch_npu._npu_reshape_and_cache(key=key,value=value,key_cache=kc,value_cache=vc,slot_indices=slots)
                    copy_rows(kc.flatten(0,1),vc.flatten(0,1),source,destination,bucket=current_grid)
            def candidate():
                prefix.run_spans(spans)
                for (key,value),(kc,vc)in zip(inputs,new,strict=True):dual_write(key,value,kc,vc,slots,aliases)
            current_grid=1792
            baseline();candidate();torch.npu.synchronize()
            old_graph=torch.npu.NPUGraph();new_graph=torch.npu.NPUGraph()
            with torch.npu.graph(old_graph):baseline()
            with torch.npu.graph(new_graph):candidate()
            torch.npu.synchronize()
            cost_graphs={1792:(old_graph,new_graph)}
            # These graphs deliberately retain1792 programs. Smaller grids
            # require their own captures and will be measured separately.
            for mapping in maps:
                source.copy_(torch.tensor(mapping['source'],dtype=torch.long))
                destination.copy_(torch.tensor(mapping['destination'],dtype=torch.long))
                prefix_source.copy_(torch.tensor(mapping['split']['prefix_source'],dtype=torch.long))
                prefix_dest.copy_(torch.tensor(mapping['split']['prefix_destination'],dtype=torch.long))
                slots.copy_(torch.tensor(mapping['slots'],dtype=torch.int32))
                aliases.copy_(torch.tensor(mapping['split']['query_alias_destination'],dtype=torch.long))
                packed=pack_spans(mapping['split']['prefix_source'],mapping['split']['prefix_destination'])
                spans.copy_(torch.tensor(packed['spans'],dtype=torch.long))
                touched=sorted(set(mapping['source']+mapping['destination']+mapping['slots']))
                index=torch.tensor(touched,dtype=torch.long,device=device)
                # Duplicate padding writers receive identical raw input words.
                unique={s:i for i,s in enumerate(dict.fromkeys(mapping['slots']))}
                writer_ids=torch.tensor([unique[s]for s in mapping['slots']],dtype=torch.long)
                for variant in(0,1,0):
                    for layer,((ok,ov),(nk,nv),(key,value))in enumerate(zip(old,new,inputs,strict=True)):
                        for cache in(ok,ov,nk,nv):cache.zero_()
                        for kv,(oc,nc)in enumerate(((ok,nk),(ov,nv))):
                            bits=((torch.arange(len(touched)*width,dtype=torch.long).reshape(len(touched),width)*73+layer*137+kv*8191+variant*311)%65536).to(torch.int16)
                            payload=bits.view(torch.bfloat16).view(len(touched),width//128,128).to(device)
                            oc.flatten(0,1).index_copy_(0,index,payload);nc.flatten(0,1).index_copy_(0,index,payload)
                        for kv,dst in enumerate((key,value)):
                            bits=((writer_ids[:,None]*width+torch.arange(width)[None,:]+layer*120*width+kv*179+variant*311)%65536).to(torch.int16)
                            seen[bits.to(torch.long)&65535]=True
                            dst.copy_(bits.view(torch.bfloat16).view(120,width//128,128))
                    torch.npu.synchronize();old_graph.replay();new_graph.replay();torch.npu.synchronize()
                    for (ok,ov),(nk,nv)in zip(old,new,strict=True):
                        assert torch.equal(ok.view(torch.int16),nk.view(torch.int16))
                        assert torch.equal(ov.view(torch.int16),nv.view(torch.int16))
                    checks.append(dict(width=width,style=style,case=mapping['name'],variant=variant,
                        all128_full_cache_storages_raw16_equal=True,same_buffers_changed_and_restored=True,
                        actual_ACLGraph_replayed=True,copy_programs=1792))
                if mapping['name'].startswith(('p0-N1-B8','p127-N12','p128-N16','p2555-N23')):
                    current_grid=mapping['grid']
                    if current_grid not in cost_graphs:
                        baseline();candidate();torch.npu.synchronize()
                        old_cost=torch.npu.NPUGraph();new_cost=torch.npu.NPUGraph()
                        with torch.npu.graph(old_cost):baseline()
                        with torch.npu.graph(new_cost):candidate()
                        torch.npu.synchronize();cost_graphs[current_grid]=(old_cost,new_cost)
                    old_cost,new_cost=cost_graphs[current_grid]
                    old_cost.replay();new_cost.replay();torch.npu.synchronize()
                    for(ok,ov),(nk,nv)in zip(old,new,strict=True):
                        assert torch.equal(ok.view(torch.int16),nk.view(torch.int16))
                        assert torch.equal(ov.view(torch.int16),nv.view(torch.int16))
                    measured={'baseline':[],'candidate':[]};pairs=[]
                    for repeat in range(40):
                        order=('baseline','candidate')if repeat%2==0 else('candidate','baseline');row={}
                        for name in order:
                            torch.npu.synchronize();begin=time.perf_counter()
                            {'baseline':old_cost,'candidate':new_cost}[name].replay();torch.npu.synchronize()
                            value_ms=(time.perf_counter()-begin)*1000;measured[name].append(value_ms);row[name]=value_ms
                        pairs.append(dict(order=order,ms=row))
                    timings.append(dict(width=width,style=style,case=mapping['name'],copy_programs=current_grid,
                        median_ms={k:statistics.median(v)for k,v in measured.items()},pairs=pairs,
                        scope='64 native cache writers plus64 per-layer copies versus one all-layer contiguous span copy plus64 dual cache writers; no model'))
                    current_grid=1792
            del old_graph,new_graph,cost_graphs,old_cost,new_cost,inputs,bases
            torch.npu.synchronize();gc.collect();torch.npu.empty_cache()
        assert bool(seen.all().item());coverage[str(width)]=65536
        del prefix,old,new,source,destination,prefix_source,prefix_dest,slots,aliases,spans
        torch.npu.synchronize();gc.collect();torch.npu.empty_cache()
    result=dict(schema='ROOT295.raw-prefix-and-dual-cache-write-isolated.v1',status='complete',
        candidate_source_sha256=hashlib.sha256((SPAN/'kernels292.py').read_bytes()).hexdigest(),
        dual_source_sha256=hashlib.sha256((K/'kernels288.py').read_bytes()).hexdigest(),
        prefix_copy_strategy='12contiguous_spans_x64_layers',prefix_programs=768,hardware_coreDim_max=65535,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),physical_device=5,logical_device=1,
        torch_version=torch.__version__,torch_npu_version=torch_npu.__version__,map_count=len(maps),
        cache_layers=64,actual_rank3_cache_shapes=True,map_cases=checks,raw_bit_coverage=coverage,
        raw_bit_replays=checks,timings=timings,fake_CPU_modules_removed=removed,
        Triton_compiled_before_any_graph_capture=True,models_loaded=False,real_model_weights_loaded=False,
        model_attention_or_KV_executed=True,synthetic_KV_copy_executed=True,attention_executed=False,
        neural_qualified=False,Goodput=None,production_installed=False,adopted=False)
    with output.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status='complete',checks=len(checks),coverage=coverage,timings=[{k:v for k,v in t.items()if k!='pairs'}for t in timings])))

if __name__=='__main__':main()
