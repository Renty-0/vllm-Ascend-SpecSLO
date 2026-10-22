"""Build isolated fixed-M120 padding-prefill math diagnostic; no NPU or launch."""
from pathlib import Path
import json,re
B=Path('/root/SpecSLO/budget-calibration-20261015')
OLD=B/'tree-M120-whole-target-gateup-layout-diagnostic-v441'
P=B/'tree-M120-padding-prefill-math-diagnostic-v444'
assert not P.exists();P.mkdir()
def rename(t):
    return re.sub(r'[a-f0-9]{64}|(?<!\d)441(?!\d)',
        lambda m:'444' if m.group()=='441' else m.group(),t).replace(
        'tree-M120-whole-target-gateup-layout-diagnostic-v444',P.name)
t=rename((OLD/'host_env441.py').read_text())
t=t.replace("str(Path(__file__).resolve().parent)+':'+", "str(Path(__file__).resolve().parent)+':'+str(B/'tree-M120-whole-target-gateup-layout-diagnostic-v441')+':'+")
(P/'host_env444.py').write_text(t)
t=rename((OLD/'check441.py').read_text()).replace('check385,check440','check385,check441').replace("check440.ready(result['current440_freeze'])","check441.ready(result['current441_freeze'])")
(P/'check444.py').write_text(t)
t=rename((OLD/'guard441.py').read_text()).replace('import guard440 as current','import guard441 as current').replace("'SOURCE440_RECEIPT.json'","'SOURCE441_RECEIPT.json'")
t=t.replace('M120-all64-gateup-NZ-whole-target-cost','M120-empty-q5-groups-prefill-math-and-cost')
t=t.replace('same actual M120 full target frame; all64 gate-up ND/NZ layouts, exact hidden/query-KV checks and balanced full-forward event costs; no production qualification','actual N16 B4 decode with eight known prompt continuations in unused q5 groups; lengths1..5, independent serial references, exact hidden/queryKV and full-forward costs; no serving qualification')
(P/'guard444.py').write_text(t)
t=rename((OLD/'startup441.sh').read_text()).replace('M120-all64-gateup-NZ-whole-target-cost','M120-empty-q5-groups-prefill-math-and-cost')
t=t.replace('export PYTHONPATH='+str(P)+':','export PYTHONPATH='+str(P)+':'+str(OLD)+':')
(P/'startup444.sh').write_text(t)
t=(OLD/'runtime441.py').read_text()
t=rename(t).replace('from check440 import ready','from check441 import ready').replace('from probe440 import run','from probe441 import run')
t=t.replace('import runtime440 as original','import runtime441 as original')
t=t.replace('M120-ND-NZ-layout-padding-local-linear-cost','M120-all64-gateup-NZ-whole-target-cost')
t=t.replace("ACTION='M120-all64-gateup-NZ-whole-target-cost'", "ACTION='M120-empty-q5-groups-prefill-math-and-cost'")
(P/'runtime444.py').write_text(t)
print(P)
