"""Fresh165-program raw storage proof; synthetic caches, no model output claim."""
def run(engine):
    if engine.is_draft:
        return dict(rank=engine.rank,synthetic_operator_on_rank=False)
    import gc
    import torch
    import torch_npu
    from write313 import dual_write
    checks=[]
    coverages={}
    for width in (256,384):
        shape=(544,128,width//128,128)
        old=[torch.zeros(shape,dtype=torch.bfloat16,device=engine.device)for _ in range(2)]
        new=[torch.zeros_like(v)for v in old]
        slots=torch.arange(512,677,dtype=torch.int32,device=engine.device)
        aliases=torch.full((165,),-1,dtype=torch.long,device=engine.device)
        alias_dest=slots.long().clone()
        seen=torch.zeros(65536,dtype=torch.bool,device='cpu')
        for key_style,value_style in ((False,False),(True,False),(False,True),(True,True)):
            query_width=engine.model.layers[0].self_attn.q_size
            qkv_width=query_width+2*width
            bases=[torch.zeros((165,qkv_width if style else width),dtype=torch.bfloat16,device=engine.device)
                   for style in (key_style,value_style)]
            if key_style and value_style:
                bases[1]=bases[0]
            key,value=[(base[:,query_width+kv*width:query_width+(kv+1)*width]if style else base).view(165,width//128,128)
                       for kv,(base,style)in enumerate(zip(bases,(key_style,value_style),strict=True))]
            for slot_dtype in (torch.int32,torch.long):
                active_slots=slots.to(slot_dtype)
                def baseline():
                    torch_npu._npu_reshape_and_cache(key=key,value=value,key_cache=old[0],value_cache=old[1],slot_indices=active_slots)
                    for cache in old:
                        flat=cache.flatten(0,1)
                        flat.index_copy_(0,alias_dest,flat.index_select(0,active_slots.long()))
                def candidate():
                    dual_write(key,value,new[0],new[1],active_slots,aliases)
                baseline();candidate();torch.npu.synchronize()
                old_graph=torch.npu.NPUGraph();new_graph=torch.npu.NPUGraph()
                with torch.npu.graph(old_graph):baseline()
                with torch.npu.graph(new_graph):candidate()
                torch.npu.synchronize()
                for mode in ('none','mixed','all'):
                    host_alias=[-1 if mode=='none' or mode=='mixed' and row%3==0 else 4096+row for row in range(165)]
                    aliases.copy_(torch.tensor(host_alias,dtype=torch.long,device='cpu'))
                    alias_dest.copy_(torch.tensor([512+row if dest<0 else dest for row,dest in enumerate(host_alias)],dtype=torch.long,device='cpu'))
                    for variant in (0,1,0):
                        for cache in old+new:cache.zero_()
                        for kv,destination in enumerate((key,value)):
                            bits=((torch.arange(165*width,dtype=torch.long)+kv*32768+variant*311)%65536).to(torch.int16)
                            seen[bits.long()&65535]=True
                            destination.copy_(bits.view(torch.bfloat16).reshape(165,width//128,128))
                        torch.npu.synchronize();old_graph.replay();new_graph.replay();torch.npu.synchronize()
                        if not all(torch.equal(a.view(torch.int16),b.view(torch.int16))for a,b in zip(old,new,strict=True)):
                            raise RuntimeError('ROOT317 full synthetic raw16 cache mismatch')
                        checks.append(dict(width=width,key_strided=key_style,value_strided=value_style,actual_key_row_stride=key.stride(0),actual_value_row_stride=value.stride(0),actual_QKV_width=qkv_width,shared_QKV_storage=key_style and value_style,
                            slots_dtype=str(slot_dtype),alias_mode=mode,variant=variant,
                            actual_165_program_graph_replay=True,full_key_and_value_cache_raw16_equal=True))
                del old_graph,new_graph,active_slots
            del key,value,bases
        if not bool(seen.all()):raise RuntimeError('ROOT317 incomplete BF16 raw16 input bit coverage')
        coverages[str(width)]=int(seen.sum())
        del old,new,slots,aliases,alias_dest
        torch.npu.synchronize();gc.collect()
    return dict(rank=engine.rank,synthetic_operator_on_rank=True,device=str(engine.device),
                checks=checks,raw16_bit_coverage=coverages,model_output_qualified=False,
                performance_measured=False,Goodput=None)
