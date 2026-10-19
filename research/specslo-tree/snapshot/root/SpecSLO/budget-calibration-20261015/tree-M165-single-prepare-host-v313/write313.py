"""Same raw288 per-row device body,165 independent row programs; unqualified."""
from kernels288 import torch,torch_npu,triton,_dual_cache_write288
def dual_write(key, value, key_cache, value_cache, slots, aliases):
    if key.device.type != 'npu' or any((v.device != key.device for v in (value, key_cache, value_cache, slots, aliases))):
        raise ValueError('ROOT288 one actual NPU device')
    if key.dtype != torch.bfloat16 or any((v.dtype != key.dtype for v in (value, key_cache, value_cache))):
        raise ValueError('ROOT288 unchanged BF16 raw storage')
    if key.shape != value.shape or key.ndim != 3 or key.shape[0] != 165 or (key.shape[1:] not in ((2, 128), (3, 128))):
        raise ValueError('ROOT288 actual fixed120 rank3 K/V shape')
    if any((v.stride(2) != 1 or v.stride(1) != 128 for v in (key, value))):
        raise ValueError('ROOT288 contiguous heads within each potentially strided QKV row')
    if key_cache.shape != value_cache.shape or not key_cache.is_contiguous() or (not value_cache.is_contiguous()):
        raise ValueError('ROOT288 matching contiguous canonical caches')
    if tuple(key_cache.shape) != (544, 128, *key.shape[1:]):
        raise ValueError('ROOT288 exact retained canonical cache geometry')
    if slots.shape != (165,) or slots.dtype not in (torch.int32, torch.long) or aliases.shape != (165,) or (aliases.dtype != torch.long):
        raise ValueError('ROOT288 validated physical120 canonical and optional alias slots')
    if not slots.is_contiguous() or not aliases.is_contiguous():
        raise ValueError('ROOT288 contiguous persistent row metadata')
    if any((int(torch_npu.get_npu_format(v)) not in (0, 2) for v in (key, value, key_cache, value_cache))):
        raise ValueError('ROOT288 plain unblocked storage only')
    width = key.shape[1] * 128
    return _dual_cache_write288[165,](key.data_ptr(), value.data_ptr(), key_cache.data_ptr(), value_cache.data_ptr(), slots, aliases, ROW_BYTES=width * 2, KEY_STRIDE_BYTES=key.stride(0) * 2, VALUE_STRIDE_BYTES=value.stride(0) * 2, COL_TILE=triton.next_power_of_2(width * 2))
