"""Resolve the actual launch's task directories for HOST construction checks."""
from pathlib import Path
import sys

P=Path(__file__).parent;B=P.parent
line=next(s for s in (P/'startup383.sh').read_text().splitlines()if s.startswith('export PYTHONPATH='))
sys.path[:0]=[str(P)]+[x for x in line.split('=',1)[1].split(':')if x.startswith('/root')and Path(x).is_dir()]
sys.path[:0]=[str(B/n)for n in ['tree-causal80-small-cohort-role-aware-audit-full-v209',
    'tree-high-normal-cap4-native-families-v148','tree-packed-KV-preflight-rank-aware-own-AR-full-v218']]
