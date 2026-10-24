"""Raw-byte contiguous span copies for the already validated original maps.

One fixed64-program launch copies both K and V. No BF16 arithmetic occurs.
HOST code proves unique destinations and that moved destinations cannot alias
any source, preserving the original index_select snapshot semantics.
"""
import torch
import torch_npu
from vllm.triton_utils import tl, triton


@triton.jit
def _copy_spans(KEY_ADDRESS, VALUE_ADDRESS, SPANS,
                ROW_BYTES: tl.constexpr, TILE: tl.constexpr):
    span = tl.program_id(0)
    key = KEY_ADDRESS.to(tl.pointer_type(tl.uint8))
    value = VALUE_ADDRESS.to(tl.pointer_type(tl.uint8))
    source = tl.load(SPANS + span * 3).to(tl.int32) * ROW_BYTES
    destination = tl.load(SPANS + span * 3 + 1).to(tl.int32) * ROW_BYTES
    size = tl.load(SPANS + span * 3 + 2).to(tl.int32) * ROW_BYTES
    columns = tl.arange(0, TILE)
    for start in range(0, size, TILE):
        mask = start + columns < size
        keys = tl.load(key + source + start + columns, mask=mask, other=0)
        values = tl.load(value + source + start + columns, mask=mask, other=0)
        tl.store(key + destination + start + columns, keys, mask=mask)
        tl.store(value + destination + start + columns, values, mask=mask)


def copy_spans(key, value, spans):
    if key.device.type != 'npu' or not key.device == value.device == spans.device:
        raise ValueError('Same actual NPU device')
    if (key.dtype != torch.bfloat16 or value.dtype != key.dtype or key.shape != value.shape
            or key.ndim < 2 or key.shape[0] <= 0
            or not key.is_contiguous() or not value.is_contiguous()):
        raise ValueError('Original contiguous BF16 token-row K/V storage')
    if any(int(torch_npu.get_npu_format(t)) not in (0, 2) for t in (key, value)):
        raise ValueError('Plain ND/NCHW storage required')
    width = key.numel() // key.shape[0]
    if width not in (256, 384) or key.numel() * 2 >= 2**31:
        raise ValueError('Exact Native TP3 widths and int32 byte offsets')
    size = key.numel() * 2
    if max(key.data_ptr(), value.data_ptr()) < min(key.data_ptr() + size, value.data_ptr() + size):
        raise ValueError('Disjoint K and V storage')
    if spans.shape != (64, 3) or spans.dtype != torch.long or not spans.is_contiguous():
        raise ValueError('Checked persistent fixed64x3 span descriptors')
    return _copy_spans[(64,)](key.data_ptr(), value.data_ptr(), spans, ROW_BYTES=width * 2, TILE=8192)
