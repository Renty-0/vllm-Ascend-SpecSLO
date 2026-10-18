"""Unqualified all-layer contiguous prefix memcpy; raw bytes only."""
import torch
from vllm.triton_utils import tl,triton
from kernels288 import PrefixCopies

@triton.jit
def _all_layer_spans292(KEY_POINTERS,VALUE_POINTERS,SPANS,
                        ROW_BYTES:tl.constexpr,TILE:tl.constexpr):
    span=tl.program_id(0)
    layer=tl.program_id(1)
    key=tl.load(KEY_POINTERS+layer).to(tl.pointer_type(tl.uint8))
    value=tl.load(VALUE_POINTERS+layer).to(tl.pointer_type(tl.uint8))
    source=tl.load(SPANS+span*3).to(tl.int32)*ROW_BYTES
    destination=tl.load(SPANS+span*3+1).to(tl.int32)*ROW_BYTES
    size=tl.load(SPANS+span*3+2).to(tl.int32)*ROW_BYTES
    columns=tl.arange(0,TILE)
    for start in range(0,size,TILE):
        mask=start+columns<size
        keys=tl.load(key+source+start+columns,mask=mask,other=0)
        values=tl.load(value+source+start+columns,mask=mask,other=0)
        tl.store(key+destination+start+columns,keys,mask=mask)
        tl.store(value+destination+start+columns,values,mask=mask)

class SpanCopies(PrefixCopies):
    """Reuse all actual64-layer pointer and storage checks from288."""
    def run_spans(self,spans):
        if spans.shape!=(12,3)or spans.dtype!=torch.long or not spans.is_contiguous()or spans.device!=self.keys.device:
            raise ValueError('ROOT292 checked persistent12x3 int64 span descriptors')
        if tuple((k.data_ptr(),v.data_ptr())for k,v in self.caches)!=self.addresses:
            raise ValueError('ROOT292 retained canonical cache pointers changed')
        return _all_layer_spans292[(12,64)](self.keys,self.values,spans,ROW_BYTES=self.width*2,TILE=8192)
