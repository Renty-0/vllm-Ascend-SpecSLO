"""Additive441 source gate; diagnostics confer no production capability."""
import hashlib,json
from pathlib import Path
from host_env441 import setup
setup()
P=Path(__file__).resolve().parent
B=P.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ready(expected=None):
    expected=expected or json.loads((P/'SOURCE441_RECEIPT.json').read_text())['freeze_sha256']
    if sha(P/'FREEZE441.json')!=expected:raise RuntimeError('ROOT441 explicit source freeze')
    result=json.loads((P/'FREEZE441.json').read_text())
    for path,h in result['files'].items():
        if sha(path)!=h:raise RuntimeError('ROOT441 unchanged own source '+path)
    import check385,check440
    check385.ready(result['parent385_freeze']);check440.ready(result['current440_freeze'])
    host=json.loads((P/'HOST441_CONSTRUCTION_RESULT.json').read_text())
    if not host['status'].startswith('PASS_HOST441')or host['NPU_used']is not False:raise RuntimeError('ROOT441 actual source construction prerequisite')
    return result

if __name__=='__main__':
    import os
    ready(os.environ.get('SPECSLO_V441_ROOT_FREEZE_SHA'))
