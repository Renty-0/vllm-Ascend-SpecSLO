"""Unqualified raw-byte operators for prefix pre-copy and dual cache writes.

These definitions are not imported or installed by any serving runtime.
Current HOST proofs cover descriptors, not these Triton implementations.
"""
import torch
import torch_npu
from vllm.triton_utils import tl,triton

@triton.jit
def _prefix_all_layers288(KEY_POINTERS,VALUE_POINTERS,SOURCE,DESTINATION,
                          ROW_BYTES:tl.constexpr,COL_TILE:tl.constexpr):
    row=tl.program_id(0)
    layer=tl.program_id(1)
    key=tl.load(KEY_POINTERS+layer).to(tl.pointer_type(tl.uint8))
    value=tl.load(VALUE_POINTERS+layer).to(tl.pointer_type(tl.uint8))
    source=tl.load(SOURCE+row).to(tl.int32)
    destination=tl.load(DESTINATION+row).to(tl.int32)
    columns=tl.arange(0,COL_TILE)
    mask=(source!=destination)&(columns<ROW_BYTES)
    keys=tl.load(key+source*ROW_BYTES+columns,mask=mask,other=0)
    values=tl.load(value+source*ROW_BYTES+columns,mask=mask,other=0)
    tl.store(key+destination*ROW_BYTES+columns,keys,mask=mask)
    tl.store(value+destination*ROW_BYTES+columns,values,mask=mask)

@triton.jit
def _dual_cache_write288(KEY_ADDRESS,VALUE_ADDRESS,CACHE_KEY_ADDRESS,CACHE_VALUE_ADDRESS,
                         SLOT_MAPPING,ALIAS_DESTINATION,ROW_BYTES:tl.constexpr,
                         KEY_STRIDE_BYTES:tl.constexpr,VALUE_STRIDE_BYTES:tl.constexpr,
                         COL_TILE:tl.constexpr):
    row=tl.program_id(0)
    key=KEY_ADDRESS.to(tl.pointer_type(tl.uint8))
    value=VALUE_ADDRESS.to(tl.pointer_type(tl.uint8))
    key_cache=CACHE_KEY_ADDRESS.to(tl.pointer_type(tl.uint8))
    value_cache=CACHE_VALUE_ADDRESS.to(tl.pointer_type(tl.uint8))
    slot=tl.load(SLOT_MAPPING+row).to(tl.int32)
    alias=tl.load(ALIAS_DESTINATION+row).to(tl.int32)
    columns=tl.arange(0,COL_TILE)
    mask=columns<ROW_BYTES
    keys=tl.load(key+row*KEY_STRIDE_BYTES+columns,mask=mask,other=0)
    values=tl.load(value+row*VALUE_STRIDE_BYTES+columns,mask=mask,other=0)
    tl.store(key_cache+slot*ROW_BYTES+columns,keys,mask=mask)
    tl.store(value_cache+slot*ROW_BYTES+columns,values,mask=mask)
    safe_alias=tl.maximum(alias,0)
    tl.store(key_cache+safe_alias*ROW_BYTES+columns,keys,mask=mask&(alias>=0))
    tl.store(value_cache+safe_alias*ROW_BYTES+columns,values,mask=mask&(alias>=0))

def dual_write(key,value,key_cache,value_cache,slots,aliases):
    if key.device.type!='npu'or any(v.device!=key.device for v in(value,key_cache,value_cache,slots,aliases)):
        raise ValueError('ROOT288 one actual NPU device')
    if key.dtype!=torch.bfloat16 or any(v.dtype!=key.dtype for v in(value,key_cache,value_cache)):
        raise ValueError('ROOT288 unchanged BF16 raw storage')
    if key.shape!=value.shape or key.ndim!=3 or key.shape[0]!=120 or key.shape[1:]not in((2,128),(3,128)):
        raise ValueError('ROOT288 actual fixed120 rank3 K/V shape')
    if any(v.stride(2)!=1 or v.stride(1)!=128 for v in(key,value)):
        raise ValueError('ROOT288 contiguous heads within each potentially strided QKV row')
    if key_cache.shape!=value_cache.shape or not key_cache.is_contiguous()or not value_cache.is_contiguous():
        raise ValueError('ROOT288 matching contiguous canonical caches')
    if tuple(key_cache.shape)!=(544,128,*key.shape[1:]):
        raise ValueError('ROOT288 exact retained canonical cache geometry')
    if slots.shape!=(120,)or slots.dtype not in(torch.int32,torch.long)or aliases.shape!=(120,)or aliases.dtype!=torch.long:
        raise ValueError('ROOT288 validated physical120 canonical and optional alias slots')
    if not slots.is_contiguous()or not aliases.is_contiguous():
        raise ValueError('ROOT288 contiguous persistent row metadata')
    if any(int(torch_npu.get_npu_format(v))not in(0,2)for v in(key,value,key_cache,value_cache)):
        raise ValueError('ROOT288 plain unblocked storage only')
    width=key.shape[1]*128
    return _dual_cache_write288[(120,)](key.data_ptr(),value.data_ptr(),key_cache.data_ptr(),value_cache.data_ptr(),
        slots,aliases,ROW_BYTES=width*2,KEY_STRIDE_BYTES=key.stride(0)*2,VALUE_STRIDE_BYTES=value.stride(0)*2,
        COL_TILE=triton.next_power_of_2(width*2))

class PrefixCopies:
    """Bind original cache pointers outside graphs; callers validate HOST maps."""
    def __init__(self,caches,device):
        self.caches=tuple(caches)
        if len(self.caches)!=64:raise ValueError('ROOT288 actual64 target layers')
        shapes={tuple(t.shape)for pair in self.caches for t in pair}
        if len(shapes)!=1 or next(iter(shapes))not in((544,128,2,128),(544,128,3,128)):
            raise ValueError('ROOT288 same-width retained per-layer canonical caches')
        tensors=[t for pair in self.caches for t in pair]
        if any(t.device!=device or t.dtype!=torch.bfloat16 or not t.is_contiguous()or int(torch_npu.get_npu_format(t))not in(0,2)for t in tensors):
            raise ValueError('ROOT288 matching plain contiguous BF16 caches on one NPU')
        self.width=tensors[0].shape[-2]*128
        self.addresses=tuple((k.data_ptr(),v.data_ptr())for k,v in self.caches)
        nbytes=tensors[0].numel()*2
        starts=sorted(a for pair in self.addresses for a in pair)
        if any(a+nbytes>b for a,b in zip(starts,starts[1:])):
            raise ValueError('ROOT288 all128 layer K/V storage ranges must be disjoint')
        self.keys=torch.tensor([a for a,b in self.addresses],dtype=torch.long,device=device)
        self.values=torch.tensor([b for a,b in self.addresses],dtype=torch.long,device=device)

    def run(self,source,destination,bucket):
        if bucket not in(128,256,512,1024,1792):raise ValueError('ROOT288 checked prefix grid')
        if any(v.shape!=(1792,)or v.dtype!=torch.long or not v.is_contiguous()or v.device!=self.keys.device for v in(source,destination)):
            raise ValueError('ROOT288 original1792 persistent int64 maps')
        if tuple((k.data_ptr(),v.data_ptr())for k,v in self.caches)!=self.addresses:
            raise ValueError('ROOT288 retained cache pointers changed')
        return _prefix_all_layers288[(bucket,64)](self.keys,self.values,source,destination,
            ROW_BYTES=self.width*2,COL_TILE=triton.next_power_of_2(self.width*2))
