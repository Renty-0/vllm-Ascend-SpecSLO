"""Draft-width specialization of the retained raw-byte KV copy kernel.

This is an unqualified prototype until the separate device probe passes.
It neither reads BF16 numerically nor changes any model calculation.
"""
def copy_rows(key,value,source,destination):
    import torch
    import torch_npu
    from copy263 import _copy_kv_rows263
    if key.device.type!='npu' or not(key.device==value.device==source.device==destination.device):
        raise ValueError('ROOT277 one NPU device')
    if key.dtype!=torch.bfloat16 or value.dtype!=key.dtype or key.shape!=value.shape:
        raise ValueError('ROOT277 same-shape original BF16 K/V')
    if key.ndim<2 or key.shape[0]<=0 or not(key.is_contiguous()and value.is_contiguous()):
        raise ValueError('ROOT277 contiguous token-row-major K/V')
    if any(int(torch_npu.get_npu_format(v))not in(0,2)for v in(key,value)):
        raise ValueError('ROOT277 plain physical layout only')
    width=key.numel()//key.shape[0];nbytes=key.numel()*2
    if width!=1024 or nbytes>=2**31 or source.shape!=(1792,)or destination.shape!=source.shape:
        raise ValueError('ROOT277 fixed1792 map and actual0.6B KV width1024')
    if any(v.dtype!=torch.long or not v.is_contiguous()for v in(source,destination)):
        raise ValueError('ROOT277 contiguous int64 indices')
    kb,vb=key.data_ptr(),value.data_ptr()
    if max(kb,vb)<min(kb+nbytes,vb+nbytes):raise ValueError('ROOT277 K/V storage overlap')
    return _copy_kv_rows263[(1792,)](kb,vb,source,destination,
        ROW_BYTES=2048,PAIRS=1792,COL_TILE=2048)


class PrivateCopy:
    """Bind persistent model KV once; update validated indices before replay."""
    def __init__(self,engine):
        import torch
        self.engine=engine
        self.cache=tuple((layer.self_attn.key_cache.flatten(0,1),
                          layer.self_attn.value_cache.flatten(0,1))
                         for layer in engine.model.layers)
        if len(self.cache)!=28 or any(tuple(k.shape)!=(69632,8,128)or v.shape!=k.shape for k,v in self.cache):
            raise RuntimeError('ROOT277 original28-layer draft cache shape')
        self.addresses=tuple((k.data_ptr(),v.data_ptr())for k,v in self.cache)
        self.source=torch.arange(1792,dtype=torch.long,device=engine.device)
        self.destination=self.source.clone()
        # Compile this new width outside capture using identity rows only.
        for k,v in self.cache:copy_rows(k,v,self.source,self.destination)
        torch.npu.synchronize()
        self.graph=torch.npu.NPUGraph()
        with torch.npu.graph(self.graph):
            for k,v in self.cache:copy_rows(k,v,self.source,self.destination)
        torch.npu.synchronize()
        self.replays=0

    def run(self,desc):
        import torch
        from contract262 import checked
        proof=checked(desc['source'],desc['destination'],rows=69632)
        current=tuple((layer.self_attn.key_cache.data_ptr(),layer.self_attn.value_cache.data_ptr())
                      for layer in self.engine.model.layers)
        if current!=self.addresses:raise RuntimeError('ROOT277 persistent KV addresses changed')
        self.source.copy_(torch.tensor(desc['source'],dtype=torch.long,device=self.engine.device))
        self.destination.copy_(torch.tensor(desc['destination'],dtype=torch.long,device=self.engine.device))
        self.graph.replay();self.replays+=1
        return proof
