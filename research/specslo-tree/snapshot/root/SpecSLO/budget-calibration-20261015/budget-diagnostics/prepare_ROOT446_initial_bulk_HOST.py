"""Build a bounded initial/bulk known-prefill diagnostic; no NPU or launch."""
from pathlib import Path
import re,json
B=Path('/root/SpecSLO/budget-calibration-20261015')
OLD=B/'tree-M120-owned-padding-prefill-math-diagnostic-v445'
P=B/'tree-M120-initial-and-bulk-prefill-math-diagnostic-v446'
assert not P.exists();P.mkdir()
def rename(t):
    return re.sub(r'[a-f0-9]{64}|(?<!\d)445(?!\d)',lambda m:'446'if m.group()=='445'else m.group(),t).replace('tree-M120-owned-padding-prefill-math-diagnostic-v446',P.name)
t=rename((OLD/'host_env445.py').read_text())
t=t.replace("str(Path(__file__).resolve().parent)+':'+", "str(Path(__file__).resolve().parent)+':'+str(B/'tree-M120-owned-padding-prefill-math-diagnostic-v445')+':'+")
(P/'host_env446.py').write_text(t)
t=rename((OLD/'check445.py').read_text()).replace('check385,check444','check385,check445').replace("check444.ready(result['current444_freeze'])","check445.ready(result['current445_freeze'])")
(P/'check446.py').write_text(t)
t=rename((OLD/'guard445.py').read_text()).replace('import guard444 as current','import guard445 as current').replace("'SOURCE444_RECEIPT.json'","'SOURCE445_RECEIPT.json'")
t=t.replace('M120-owned-q5-prefill-frames-math-and-cost','M120-initial-bulk-q5-prefill-math-and-cost')
(P/'guard446.py').write_text(t)
t=rename((OLD/'startup445.sh').read_text()).replace('M120-owned-q5-prefill-frames-math-and-cost','M120-initial-bulk-q5-prefill-math-and-cost')
t=t.replace('export PYTHONPATH='+str(P)+':','export PYTHONPATH='+str(P)+':'+str(OLD)+':')
(P/'startup446.sh').write_text(t)
t=rename((OLD/'runtime445.py').read_text()).replace('import runtime444 as original','import runtime445 as original')
t=t.replace('from check444 import ready','from check445 import ready').replace('from probe444 import run','from probe445 import run')
t=t.replace('M120-empty-q5-groups-prefill-math-and-cost','M120-owned-q5-prefill-frames-math-and-cost')
t=t.replace("ACTION='M120-owned-q5-prefill-frames-math-and-cost'","ACTION='M120-initial-bulk-q5-prefill-math-and-cost'")
(P/'runtime446.py').write_text(t)
t=rename((OLD/'probe445.py').read_text()).replace('_root446_fixture_prefixes','_root445_fixture_prefixes')
(P/'probe446.py').write_text(t)
print(P)
