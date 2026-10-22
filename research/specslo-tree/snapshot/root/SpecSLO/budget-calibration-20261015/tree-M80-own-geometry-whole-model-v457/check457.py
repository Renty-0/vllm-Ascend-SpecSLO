"""Exact current freeze plus the full original parent closure, no NN license."""
from pathlib import Path
import hashlib,json,os
from host_env457 import setup
setup()
P=Path(__file__).resolve().parent;B=P.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ready(expected=None,*,include_parent=False):
 expected=expected or os.environ.get('SPECSLO_V457_ROOT_FREEZE_SHA')
 if not expected or sha(P/'FREEZE457.json')!=expected:raise RuntimeError('ROOT457 explicit own freeze')
 value=json.loads((P/'FREEZE457.json').read_text())
 for path,h in value['files'].items():
  if sha(path)!=h:raise RuntimeError('ROOT457 own immutable source '+path)
 from flat457 import verified
 parent_sources=verified()
 return(value,parent_sources)if include_parent else value
if __name__=='__main__':ready()
