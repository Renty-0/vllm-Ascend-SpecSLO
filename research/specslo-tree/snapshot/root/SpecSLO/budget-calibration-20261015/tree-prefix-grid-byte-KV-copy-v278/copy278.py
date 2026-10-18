"""Reuse exact263 one-row rawbyte binary; skip only proved identity suffix.

Caller must validate the current HOST maps with contract263.checked before
updating the two persistent device index buffers. Nonidentity destinations
cannot overlap any source, and all destinations are unique. Each identity
padding row is skipped. The two bytes of each BF16 value are copied without arithmetic or conversion.
"""
import torch
import torch_npu
from vllm.triton_utils import tl,triton

from copy263 import _copy_kv_rows263

def copy_rows(key,value,source,destination,*,bucket=1792):
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
    if width not in(256,384)or key.numel()*key.element_size()>=2**31 or pairs!=1792:
        raise ValueError('explicit fixed1792 copy map and supported Native K/V row widths')
    if type(bucket)is not int or bucket not in(128,256,512,1024,1792):
        raise ValueError('ROOT278 explicit captured copy-grid bucket')
    return _copy_kv_rows263[(bucket,)](
        key.data_ptr(),value.data_ptr(),source,destination,
        ROW_BYTES=width*2,PAIRS=pairs,COL_TILE=triton.next_power_of_2(width*2))
