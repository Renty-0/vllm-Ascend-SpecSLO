"""Additive451 source gate; diagnostics confer no production capability."""
import hashlib,json
from pathlib import Path
from host_env451 import setup
setup()
P=Path(__file__).resolve().parent
B=P.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ready(expected=None):
    expected=expected or json.loads((P/'SOURCE451_RECEIPT.json').read_text())['freeze_sha256']
    if sha(P/'FREEZE451.json')!=expected:raise RuntimeError('ROOT451 explicit source freeze')
    result=json.loads((P/'FREEZE451.json').read_text())
    for path,h in result['files'].items():
        if sha(path)!=h:raise RuntimeError('ROOT451 unchanged own source '+path)
    import check385,check450
    check385.ready(result['parent385_freeze']);check450.ready(result['current450_freeze'])
    host=json.loads((P/'HOST451_CONSTRUCTION_RESULT.json').read_text())
    if not host['status'].startswith('PASS_HOST451')or host['NPU_used']is not False:raise RuntimeError('ROOT451 actual source construction prerequisite')
    settings=json.loads((P/'HOST451_SETTINGS_RESULT.json').read_text())
    if settings['status']!='PASS_HOST451_EXACT_VALID_NUMERICAL_EXPORTS_AND_ACTUAL_NATIVE_PARSER' or settings['NPU_used']is not False:raise RuntimeError('ROOT451 actual startup numerical settings prerequisite')
    return result

if __name__=='__main__':
    import os
    ready(os.environ.get('SPECSLO_V451_ROOT_FREEZE_SHA'))
