"""Additive447 source gate; diagnostics confer no production capability."""
import hashlib,json
from pathlib import Path
from host_env447 import setup
setup()
P=Path(__file__).resolve().parent
B=P.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ready(expected=None):
    expected=expected or json.loads((P/'SOURCE447_RECEIPT.json').read_text())['freeze_sha256']
    if sha(P/'FREEZE447.json')!=expected:raise RuntimeError('ROOT447 explicit source freeze')
    result=json.loads((P/'FREEZE447.json').read_text())
    for path,h in result['files'].items():
        if sha(path)!=h:raise RuntimeError('ROOT447 unchanged own source '+path)
    import check385,check446
    check385.ready(result['parent385_freeze']);check446.ready(result['current446_freeze'])
    host=json.loads((P/'HOST447_CONSTRUCTION_RESULT.json').read_text())
    if not host['status'].startswith('PASS_HOST447')or host['NPU_used']is not False:raise RuntimeError('ROOT447 actual source construction prerequisite')
    return result

if __name__=='__main__':
    import os
    ready(os.environ.get('SPECSLO_V447_ROOT_FREEZE_SHA'))
