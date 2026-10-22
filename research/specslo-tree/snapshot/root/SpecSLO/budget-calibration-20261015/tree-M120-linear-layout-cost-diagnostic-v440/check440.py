"""Additive440 source gate; diagnostics confer no production capability."""
import hashlib,json
from pathlib import Path
from host_env440 import setup
setup()
P=Path(__file__).resolve().parent
B=P.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ready(expected=None):
    expected=expected or json.loads((P/'SOURCE440_RECEIPT.json').read_text())['freeze_sha256']
    if sha(P/'FREEZE440.json')!=expected:raise RuntimeError('ROOT440 explicit source freeze')
    result=json.loads((P/'FREEZE440.json').read_text())
    for path,h in result['files'].items():
        if sha(path)!=h:raise RuntimeError('ROOT440 unchanged own source '+path)
    import check385,check438
    check385.ready(result['parent385_freeze']);check438.ready(result['current438_freeze'])
    host=json.loads((P/'HOST440_CONSTRUCTION_RESULT.json').read_text())
    if not host['status'].startswith('PASS_HOST440')or host['NPU_used']is not False:raise RuntimeError('ROOT440 actual source construction prerequisite')
    return result

if __name__=='__main__':
    import os
    ready(os.environ.get('SPECSLO_V440_ROOT_FREEZE_SHA'))
