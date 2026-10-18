"""All raw16-bit values, changed/restored buffers, new draft KV width."""
def run(engine):
    import torch
    from copy277 import copy_rows
    from contract262 import checked
    device=engine.device;width=1024;count=4096*width
    bits=(torch.arange(count,dtype=torch.long,device='cpu')%65536).to(torch.int16).reshape(4096,width)
    vals=((torch.arange(count,dtype=torch.long,device='cpu')*73+17)%65536).to(torch.int16).reshape(4096,width)
    key=bits.view(torch.bfloat16).to(device);value=vals.view(torch.bfloat16).to(device)
    source=torch.arange(1792,dtype=torch.long,device=device)
    dest=(torch.arange(1792,dtype=torch.long,device=device)+2048)
    copy_rows(key,value,source,dest);torch.npu.synchronize()
    graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph):copy_rows(key,value,source,dest)
    torch.npu.synchronize();rows=[]
    for variant in(0,1,0):
        for identity in(False,True):
            src=list(range(variant*64,variant*64+1792))
            dst=src if identity else list(range(2048+variant*64,2048+variant*64+1792))
            checked(src,dst,rows=4096)
            kb=(bits.to(torch.int32)+variant*311).to(torch.int16)
            vb=(vals.to(torch.int32)+variant*97).to(torch.int16)
            si=torch.tensor(src,dtype=torch.long,device='cpu');di=torch.tensor(dst,dtype=torch.long,device='cpu')
            ek=kb.clone();ev=vb.clone()
            ek.index_copy_(0,di,kb.index_select(0,si));ev.index_copy_(0,di,vb.index_select(0,si))
            key.copy_(kb.view(torch.bfloat16));value.copy_(vb.view(torch.bfloat16))
            source.copy_(si);dest.copy_(di);graph.replay();torch.npu.synchronize()
            if not(torch.equal(key.cpu().view(torch.int16),ek)and torch.equal(value.cpu().view(torch.int16),ev)):
                raise RuntimeError('ROOT277 raw16-bit whole-cache copy mismatch')
            rows.append(dict(variant=variant,identity=identity,whole_KV_raw16_equal=True))
    return dict(width=1024,rows=rows,all65536_payloads=True,actual_graph_replays=6,
        compilation_outside_graph=True,model_math_changed=False)
