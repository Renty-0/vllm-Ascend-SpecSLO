"""Read-only first-row module taps, installed only around eager ablations."""
def observe(layer,fn,*,rows,query_row):
    import torch
    taps={};handles=[]
    def hook(name):
        def collect(module,args,output):
            def walk(v,suffix):
                if isinstance(v,torch.Tensor):
                    if v.is_floating_point()and v.ndim>=2 and v.shape[0]==rows:
                        taps[name+suffix]=v.detach()[query_row].clone()
                elif isinstance(v,(tuple,list)):
                    for j,item in enumerate(v):walk(item,suffix+'/'+str(j))
            walk(output,'')
        return collect
    try:
        for name,module in layer.named_modules():handles.append(module.register_forward_hook(hook(name or '<layer0>')))
        result=fn()
        if not taps:raise RuntimeError('actual eager module callbacks required')
        return result,taps
    finally:
        for handle in handles:handle.remove()
