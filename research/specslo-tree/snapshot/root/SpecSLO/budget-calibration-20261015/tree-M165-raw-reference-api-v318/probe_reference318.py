"""Model-free diagnostic of native ReshapeAndCache slot dtype support."""
import json
import os
import sys

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import torch
import torch_npu

torch.npu.set_device(1)
torch.npu.config.allow_internal_format = True
device = torch.device('npu:1')
dtype = {'int32': torch.int32, 'int64': torch.int64}[sys.argv[1]]
width = 256
key = torch.arange(165 * width, dtype=torch.float32, device=device).to(torch.bfloat16).reshape(165, 2, 128)
value = key.clone()
caches = [torch.zeros((544, 128, 2, 128), dtype=torch.bfloat16, device=device) for _ in range(2)]
slots = torch.arange(512, 677, dtype=dtype, device=device)
print(json.dumps(dict(stage='before_native_op',slot_dtype=str(dtype),key_format=int(torch_npu.get_npu_format(key)),cache_format=int(torch_npu.get_npu_format(caches[0])),models_loaded=False)), flush=True)
torch_npu._npu_reshape_and_cache(key=key, value=value, key_cache=caches[0], value_cache=caches[1], slot_indices=slots)
torch.npu.synchronize()
assert torch.equal(caches[0].flatten(0, 1).index_select(0, slots.long()), key)
assert torch.equal(caches[1].flatten(0, 1).index_select(0, slots.long()), value)
print(json.dumps(dict(stage='native_op_complete',slot_dtype=str(dtype),exact_values=True,models_loaded=False)), flush=True)
