"""Additive450 source gate; diagnostics confer no production capability."""
import hashlib,json
from pathlib import Path
from host_env450 import setup
setup()
P=Path(__file__).resolve().parent
B=P.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ready(expected=None):
    expected=expected or json.loads((P/'SOURCE450_RECEIPT.json').read_text())['freeze_sha256']
    if sha(P/'FREEZE450.json')!=expected:raise RuntimeError('ROOT450 explicit source freeze')
    result=json.loads((P/'FREEZE450.json').read_text())
    for path,h in result['files'].items():
        if sha(path)!=h:raise RuntimeError('ROOT450 unchanged own source '+path)
    import check385,check449
    check385.ready(result['parent385_freeze']);check449.ready(result['current449_freeze'])
    host=json.loads((P/'HOST450_CONSTRUCTION_RESULT.json').read_text())
    if not host['status'].startswith('PASS_HOST450')or host['NPU_used']is not False:raise RuntimeError('ROOT450 actual source construction prerequisite')
    settings=json.loads((P/'HOST450_SETTINGS_RESULT.json').read_text())
    if settings['status']!='PASS_HOST450_EXACT_VALID_NUMERICAL_EXPORTS_AND_ACTUAL_NATIVE_PARSER' or settings['NPU_used']is not False:raise RuntimeError('ROOT450 actual startup numerical settings prerequisite')
    return result

if __name__=='__main__':
    import os
    ready(os.environ.get('SPECSLO_V450_ROOT_FREEZE_SHA'))
