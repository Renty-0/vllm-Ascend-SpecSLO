"""Conditional one-launch K/V row copy; not installed or device-qualified.

Caller must validate the current HOST maps with contract262.checked before
updating the two persistent device index buffers. Nonidentity destinations
cannot overlap any source, and all destinations are unique. Each identity
padding row is skipped. BF16 values are only loaded/stored, never computed.
"""
import torch
import torch_npu
from vllm.triton_utils import tl,triton

@triton.jit
def _copy_kv_rows262(KEY,VALUE,SOURCE,DESTINATION,
                     WIDTH:tl.constexpr,PAIRS:tl.constexpr,
                     ROW_TILE:tl.constexpr,COL_TILE:tl.constexpr):
    # Raw16-bit pointer casts follow the existing Ascend batch_memcpy kernel's
    # byte-pointer idiom. No BF16 arithmetic/conversion or Torch dtype view.
    KEY=KEY.to(tl.pointer_type(tl.uint16))
    VALUE=VALUE.to(tl.pointer_type(tl.uint16))
    rows=tl.program_id(0)*ROW_TILE+tl.arange(0,ROW_TILE)
    source=tl.load(SOURCE+rows,mask=rows<PAIRS,other=0).to(tl.int32)
    destination=tl.load(DESTINATION+rows,mask=rows<PAIRS,other=0).to(tl.int32)
    columns=tl.arange(0,COL_TILE)
    mask=((rows<PAIRS)&(source!=destination))[:,None]&(columns<WIDTH)[None,:]
    offsets_in=source[:,None]*WIDTH+columns[None,:]
    offsets_out=destination[:,None]*WIDTH+columns[None,:]
    keys=tl.load(KEY+offsets_in,mask=mask,other=0)
    values=tl.load(VALUE+offsets_in,mask=mask,other=0)
    tl.store(KEY+offsets_out,keys,mask=mask)
    tl.store(VALUE+offsets_out,values,mask=mask)

def copy_rows(key,value,source,destination):
    if key.device.type!='npu' or not(key.device==value.device==source.device==destination.device):
        raise ValueError('one actual NPU device for K/V and indices')
    if key.dtype!=torch.bfloat16 or value.dtype!=key.dtype or key.shape!=value.shape:
        raise ValueError('same-shape original BF16 K/V cache')
    if key.ndim<2 or key.shape[0]<=0 or not(key.is_contiguous()and value.is_contiguous()):
        raise ValueError('contiguous token-row-major K/V cache')
    if int(torch_npu.get_npu_format(key))not in(0,2)or int(torch_npu.get_npu_format(value))not in(0,2):
        raise ValueError('plain unblocked ND/NCHW storage only; blocked physical layouts are unsupported')
    key_begin=key.data_ptr();value_begin=value.data_ptr();nbytes=key.numel()*key.element_size()
    if max(key_begin,value_begin)<min(key_begin+nbytes,value_begin+nbytes):
        raise ValueError('K and V storage ranges must not overlap')
    if source.ndim!=1 or source.dtype!=torch.long or destination.shape!=source.shape or destination.dtype!=torch.long:
        raise ValueError('same-length original int64 source/destination indices')
    if not(source.is_contiguous()and destination.is_contiguous()):
        raise ValueError('contiguous original index buffers')
    width=key.numel()//key.shape[0];pairs=source.numel()
    if width not in(256,384)or key.numel()>=2**31 or pairs!=1792:
        raise ValueError('explicit fixed1792 copy map and supported Native K/V row widths')
    return _copy_kv_rows262[(triton.cdiv(pairs,16),)](
        key,value,source,destination,WIDTH=width,PAIRS=pairs,ROW_TILE=16,COL_TILE=triton.next_power_of_2(width))
