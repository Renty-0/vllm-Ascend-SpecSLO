"""Catch the previous fixed127 overwrite, including reference schedule writes."""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
from pathlib import Path
import sys,json,types,importlib.util,hashlib
import torch
torch.set_num_threads(1)
P=Path(__file__).resolve().parent;B=P.parent
N=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
sys.path[:0]=[str(P),str(B/'tree-static-q5-M128-projection-host-v249'),str(B/'tree-canonical-contiguous-full463-own-AR-v235'),str(B/'tree-high-normal-cap4-native-families-v148')]
for name in ('vllm_ascend','vllm_ascend.spec_decode','vllm_ascend.spec_decode.pearl'):
    m=types.ModuleType(name);m.__path__=[];sys.modules[name]=m
spec=importlib.util.spec_from_file_location('vllm_ascend.spec_decode.pearl.tree',N/'tree.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
import oracle251
from reference_schedule251 import schedule
prefixes=json.loads((P/'FIXTURE251.json').read_text())['prefixes']
checked=reference_positions=phantoms_beyond_extent=0
for case in oracle251.cases():
    plans,rows=oracle251.prepare(case,prefixes)
    for owner,plan in enumerate(plans):
        assert int(plan.cache_positions[0])==len(prefixes[owner])and int(plan.positions[0])==len(prefixes[owner])
        assert all(v>=len(prefixes[owner])for v in plan.cache_positions.tolist())
        checked+=len(plan.cache_positions)
    for frame in schedule(plans,[41]*case['N'],rows):
        for owner,pos in zip(frame['ids'],frame['positions'],strict=True):
            if owner<72:assert pos>=len(prefixes[owner%24]);reference_positions+=1
            assert 0<=pos<4096
            phantoms_beyond_extent+=pos>=2560
        for real in frame['real_queries']:
            assert frame['positions'][real['head_row']]<2560
bad=oracle251.original.prepare(oracle251.cases()[0])[0]
assert int(bad[0].cache_positions[0])<len(prefixes[0])
assert 'torch_npu'not in sys.modules
r=dict(status='HOST_actual_candidate_and_reference_prefix_origins_passed',cases=233,
       checked_real_cache_writes=checked,checked_reference_positions=reference_positions,
       actual_first_two_prefixes=[len(v)for v in prefixes[:2]],old_fixed127_overwrite_detected=True,
       all_real_reference_visible_positions_below2560=True,nonreal_future_positions_past_extent=phantoms_beyond_extent,
       NPU_initialized=False,source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in [P/'oracle251.py',P/'reference_schedule251.py']})
(P/'HOST_ACTUAL_PREFIX_ORIGINS251_RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
