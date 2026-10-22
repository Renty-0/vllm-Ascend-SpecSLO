"""Build a bounded initial/bulk known-prefill diagnostic; no NPU or launch."""
from pathlib import Path
import re,json
B=Path('/root/SpecSLO/budget-calibration-20261015')
OLD=B/'tree-M120-initial-and-bulk-prefill-math-diagnostic-v446'
P=B/'tree-M120-paired-memory-prefill-math-diagnostic-v447'
assert not P.exists();P.mkdir()
def rename(t):
    return re.sub(r'[a-f0-9]{64}|(?<!\d)446(?!\d)',lambda m:'447'if m.group()=='446'else m.group(),t).replace('tree-M120-initial-and-bulk-prefill-math-diagnostic-v447',P.name)
t=rename((OLD/'host_env446.py').read_text())
t=t.replace("str(Path(__file__).resolve().parent)+':'+", "str(Path(__file__).resolve().parent)+':'+str(B/'tree-M120-initial-and-bulk-prefill-math-diagnostic-v446')+':'+")
(P/'host_env447.py').write_text(t)
t=rename((OLD/'check446.py').read_text()).replace('check385,check445','check385,check446').replace("check444.ready(result['current444_freeze'])","check446.ready(result['current446_freeze'])")
(P/'check447.py').write_text(t)
t=rename((OLD/'guard446.py').read_text()).replace('import guard445 as current','import guard446 as current').replace("'SOURCE445_RECEIPT.json'","'SOURCE446_RECEIPT.json'")
t=t.replace('M120-initial-bulk-q5-prefill-math-and-cost','M120-initial-bulk-q5-prefill-math-and-cost')
(P/'guard447.py').write_text(t)
t=rename((OLD/'startup446.sh').read_text()).replace('M120-initial-bulk-q5-prefill-math-and-cost','M120-initial-bulk-q5-prefill-math-and-cost')
t=t.replace('export PYTHONPATH='+str(P)+':','export PYTHONPATH='+str(P)+':'+str(OLD)+':')
(P/'startup447.sh').write_text(t)
t=rename((OLD/'runtime446.py').read_text()).replace('import runtime445 as original','import runtime446 as original')
t=t.replace('from check445 import ready','from check446 import ready').replace('from probe445 import run','from probe446 import run')
t=t.replace('M120-initial-bulk-q5-prefill-math-and-cost','M120-initial-bulk-q5-prefill-math-and-cost')
t=t.replace("ACTION='M120-initial-bulk-q5-prefill-math-and-cost'","ACTION='M120-paired-memory-q5-prefill-math-cost'")
(P/'runtime447.py').write_text(t)
t=rename((OLD/'probe446.py').read_text()).replace('_root447_fixture_prefixes','_root445_fixture_prefixes')
(P/'probe447.py').write_text(t)
print(P)
