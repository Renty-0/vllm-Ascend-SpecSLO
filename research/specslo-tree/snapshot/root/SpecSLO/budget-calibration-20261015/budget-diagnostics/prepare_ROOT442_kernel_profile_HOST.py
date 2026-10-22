"""Build a bounded unchanged-model target profiler; no NPU or launch."""
from pathlib import Path
import json,re
B=Path('/root/SpecSLO/budget-calibration-20261015')
OLD=B/'tree-M120-whole-target-gateup-layout-diagnostic-v441'
P=B/'tree-M120-target-kernel-profile-diagnostic-v442'
assert not P.exists();P.mkdir()
def rename(t):
    return re.sub(r'[a-f0-9]{64}|(?<!\d)441(?!\d)',
        lambda m:'442' if m.group()=='441' else m.group(),t).replace(
        'tree-M120-whole-target-gateup-layout-diagnostic-v442',P.name)
t=rename((OLD/'host_env441.py').read_text())
t=t.replace("str(Path(__file__).resolve().parent)+':'+", "str(Path(__file__).resolve().parent)+':'+str(B/'tree-M120-whole-target-gateup-layout-diagnostic-v441')+':'+")
(P/'host_env442.py').write_text(t)
t=rename((OLD/'check441.py').read_text()).replace('check385,check440','check385,check441').replace("check440.ready(result['current440_freeze'])","check441.ready(result['current441_freeze'])")
(P/'check442.py').write_text(t)
t=rename((OLD/'guard441.py').read_text()).replace('import guard440 as current','import guard441 as current').replace("'SOURCE440_RECEIPT.json'","'SOURCE441_RECEIPT.json'")
t=t.replace('M120-all64-gateup-NZ-whole-target-cost','M120-unchanged-target-device-kernel-profile')
t=t.replace('same actual M120 full target frame; all64 gate-up ND/NZ layouts, exact hidden/query-KV checks and balanced full-forward event costs; no production qualification','same unchanged actual M120 full target frame; bounded single graph replay profiler and unprofiled device costs; no serving qualification')
(P/'guard442.py').write_text(t)
t=rename((OLD/'startup441.sh').read_text()).replace('M120-all64-gateup-NZ-whole-target-cost','M120-unchanged-target-device-kernel-profile')
t=t.replace('export PYTHONPATH='+str(P)+':','export PYTHONPATH='+str(P)+':'+str(OLD)+':')
(P/'startup442.sh').write_text(t)
t=(OLD/'runtime441.py').read_text()
t=rename(t).replace('from check440 import ready','from check441 import ready').replace('from probe440 import run','from probe441 import run')
t=t.replace('import runtime440 as original','import runtime441 as original')
t=t.replace('M120-ND-NZ-layout-padding-local-linear-cost','M120-all64-gateup-NZ-whole-target-cost')
t=t.replace("ACTION='M120-all64-gateup-NZ-whole-target-cost'", "ACTION='M120-unchanged-target-device-kernel-profile'")
(P/'runtime442.py').write_text(t)
t=rename((OLD/'probe441.py').read_text()).replace('whole-target-all64-gateup-ND-NZ-frame-and-cost','unchanged-whole-target-single-replay-profiler')
(P/'probe442.py').write_text(t)
(P/'PLAN442.json').write_text(json.dumps(dict(schema='ROOT442.unchanged-target-profile.v1',
    scope='one genuine M120/S24 target frame, unchanged weights and MC2, single NPU-only profile replay per target rank',
    model_parameter_swaps=False,Native_flags_assigned=False,
    profiler=dict(activities=['NPU'],level=0,replays_per_target_rank=1,record_shapes=False,with_stack=False,profile_memory=False),
    caps=dict(seconds=3600,tag_MiB=192,aggregate_MiB=3584),
    actual_protected_prefixes=72,full463=False,production_permission=False,Raw=None,Goodput=None,Git_pushes=0),indent=2)+'\n')
print(P)
