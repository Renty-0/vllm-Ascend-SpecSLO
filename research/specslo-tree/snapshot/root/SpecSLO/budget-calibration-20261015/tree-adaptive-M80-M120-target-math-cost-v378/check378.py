"""Explicit additive source gate; component evidence never grants LIVE scope."""
import hashlib
import json
import os
from pathlib import Path
import sys

P=Path(__file__).parent
B=P.parent
PARENT=B/'tree-Q5-bucket-completion-and-paired-mode-cost-v361'
PARENT_FREEZE='a1dbe700b6fa857631be31878df58fd7b129de38491ae027a8823799ce81bf3f'


def ready(expected=None):
    expected=expected or os.environ.get('SPECSLO_V378_ROOT_FREEZE_SHA')
    path=P/'FREEZE378.json'
    if expected is None or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise RuntimeError('ROOT378 explicit frozen source pin required')
    value=json.loads(path.read_text())
    for name,digest in value['files'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('ROOT378 source changed: '+name)
    sys.path.insert(0,str(PARENT))
    from check361 import ready as parent_ready
    parent_ready(PARENT_FREEZE)
    for name in ('HOST378_CONSTRUCTION_RESULT.json','HOST378_COST_AND_AUDIT_RESULT.json'):
        result=json.loads((P/name).read_text())
        if not result['status'].startswith('PASS_HOST378') or result['NPU_used'] is not False:
            raise RuntimeError('ROOT378 current HOST prerequisites required: '+name)
    for name in ('HOST377_RESULT.json','HOST377_DISPATCH_RESULT.json'):
        result=json.loads((B/'tree-adaptive-M80-M120-target-host-v377'/name).read_text())
        if not result['status'].startswith('PASS_HOST377') or result['NPU_used'] is not False:
            raise RuntimeError('ROOT378 checked source377 prerequisites required: '+name)
    return value


if __name__=='__main__':
    ready()
