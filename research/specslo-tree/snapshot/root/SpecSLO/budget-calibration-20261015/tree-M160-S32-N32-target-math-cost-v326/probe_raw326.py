"""Fresh144 synthetic raw16 checks without loading model weights."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
P = Path(__file__).resolve().parent
B = P.parent
sys.path[:0] = [str(B/'tree-M165-single-prepare-host-v313'),str(B/'tree-prefix-precopy-dual-cachewrite-host-v288')]
import torch
import torch_npu
torch.npu.set_device(1)
torch.npu.config.allow_internal_format = True
from raw_controls326 import run
# Largest real TP3 Q partition is22 heads. Smaller actual ranks are checked
# again in the original four-role qualification with their real model q_size.
engine = SimpleNamespace(is_draft=False,rank=1,device=torch.device('npu:1'),model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(q_size=24*128))]))
with torch.inference_mode():
    result = run(engine)
assert len(result['checks']) == 144
assert result['raw16_bit_coverage'] == {'256':65536,'384':65536}
assert {c['slots_dtype']for c in result['checks']} == {'torch.int32','torch.int64'}
assert all(c['reference_slots_dtype']=='torch.int32'for c in result['checks'])
out = P/'RAW326_MODEL_FREE_RESULT.json'
with out.open('x')as f:
    json.dump(dict(result=result,models_loaded=False,model_qualified=False),f,indent=2)
print(json.dumps(dict(status='pass',checks=144,coverage=result['raw16_bit_coverage'],models_loaded=False)),flush=True)
