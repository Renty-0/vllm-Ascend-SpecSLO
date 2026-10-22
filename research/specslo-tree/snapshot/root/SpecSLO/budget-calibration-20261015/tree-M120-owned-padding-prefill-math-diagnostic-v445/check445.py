"""Additive445 source gate; diagnostics confer no production capability."""
import hashlib,json
from pathlib import Path
from host_env445 import setup
setup()
P=Path(__file__).resolve().parent
B=P.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ready(expected=None):
    expected=expected or json.loads((P/'SOURCE445_RECEIPT.json').read_text())['freeze_sha256']
    if sha(P/'FREEZE445.json')!=expected:raise RuntimeError('ROOT445 explicit source freeze')
    result=json.loads((P/'FREEZE445.json').read_text())
    for path,h in result['files'].items():
        if sha(path)!=h:raise RuntimeError('ROOT445 unchanged own source '+path)
    import check385,check444
    check385.ready(result['parent385_freeze']);check444.ready(result['current444_freeze'])
    host=json.loads((P/'HOST445_CONSTRUCTION_RESULT.json').read_text())
    if not host['status'].startswith('PASS_HOST445')or host['NPU_used']is not False:raise RuntimeError('ROOT445 actual source construction prerequisite')
    return result

if __name__=='__main__':
    import os
    ready(os.environ.get('SPECSLO_V445_ROOT_FREEZE_SHA'))
