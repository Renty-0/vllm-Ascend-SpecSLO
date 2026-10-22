"""Additive449 source gate; diagnostics confer no production capability."""
import hashlib,json
from pathlib import Path
from host_env449 import setup
setup()
P=Path(__file__).resolve().parent
B=P.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ready(expected=None):
    expected=expected or json.loads((P/'SOURCE449_RECEIPT.json').read_text())['freeze_sha256']
    if sha(P/'FREEZE449.json')!=expected:raise RuntimeError('ROOT449 explicit source freeze')
    result=json.loads((P/'FREEZE449.json').read_text())
    for path,h in result['files'].items():
        if sha(path)!=h:raise RuntimeError('ROOT449 unchanged own source '+path)
    import check385,check448
    check385.ready(result['parent385_freeze']);check448.ready(result['current448_freeze'])
    host=json.loads((P/'HOST449_CONSTRUCTION_RESULT.json').read_text())
    if not host['status'].startswith('PASS_HOST449')or host['NPU_used']is not False:raise RuntimeError('ROOT449 actual source construction prerequisite')
    return result

if __name__=='__main__':
    import os
    ready(os.environ.get('SPECSLO_V449_ROOT_FREEZE_SHA'))
