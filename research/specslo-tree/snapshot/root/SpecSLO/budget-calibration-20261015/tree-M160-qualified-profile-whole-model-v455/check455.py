"""Additive455 source gate; diagnostics confer no production capability."""
import hashlib,json
from pathlib import Path
from host_env455 import setup
setup()
P=Path(__file__).resolve().parent
B=P.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ready(expected=None):
    expected=expected or json.loads((P/'SOURCE455_RECEIPT.json').read_text())['freeze_sha256']
    if sha(P/'FREEZE455.json')!=expected:raise RuntimeError('ROOT455 explicit source freeze')
    result=json.loads((P/'FREEZE455.json').read_text())
    for path,h in result['files'].items():
        if sha(path)!=h:raise RuntimeError('ROOT455 unchanged own source '+path)
    import check385,check454
    check385.ready(result['parent385_freeze']);check454.ready(result['current454_freeze'])
    host=json.loads((P/'HOST455_CONSTRUCTION_RESULT.json').read_text())
    if not host['status'].startswith('PASS_HOST455')or host['NPU_used']is not False:raise RuntimeError('ROOT455 actual source construction prerequisite')
    settings=json.loads((P/'HOST455_SETTINGS_RESULT.json').read_text())
    if settings['status']!='PASS_HOST455_EXACT_VALID_NUMERICAL_EXPORTS_AND_ACTUAL_NATIVE_PARSER' or settings['NPU_used']is not False:raise RuntimeError('ROOT455 actual startup numerical settings prerequisite')
    native=json.loads((P/'HOST455_NATIVE_PROFILE_RESULT.json').read_text())
    if native['status']!='PASS_HOST455_ACTUAL_NATIVE_NORMALIZER_AND_PROFILE_ONLY_CONFIG_CHANGE'or native['NPU_initialized']is not False or native['changed_config_fields']!=['mc2_profile']or native['max_num_seqs']!=64:raise RuntimeError('ROOT455 actual Native profile and constructor configuration prerequisite')
    return result

if __name__=='__main__':
    import os
    ready(os.environ.get('SPECSLO_V455_ROOT_FREEZE_SHA'))
