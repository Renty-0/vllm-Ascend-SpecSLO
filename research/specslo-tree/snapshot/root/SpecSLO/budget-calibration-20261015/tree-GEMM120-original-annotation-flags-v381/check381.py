"""Explicit additive source gate; component evidence never grants LIVE scope."""
import hashlib
import json
import os
from pathlib import Path
import sys

P=Path(__file__).parent
B=P.parent
PARENT=B/'tree-M80-original-GEMM120-controls-v380'
PARENT_FREEZE='a1a52c1d2b34e5ded30ad1a2a5e52baea39369ffdf26e6f6ffc0dd4e2920934e'


def ready(expected=None):
    expected=expected or os.environ.get('SPECSLO_V381_ROOT_FREEZE_SHA')
    path=P/'FREEZE381.json'
    if expected is None or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise RuntimeError('ROOT381 explicit frozen source pin required')
    value=json.loads(path.read_text())
    for name,digest in value['files'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('ROOT381 source changed: '+name)
    sys.path.insert(0,str(PARENT))
    from check380 import ready as parent_ready
    parent_ready(PARENT_FREEZE)
    for name in ('HOST381_CONSTRUCTION_RESULT.json',):
        result=json.loads((P/name).read_text())
        if not result['status'].startswith('PASS_HOST381') or result['NPU_used'] is not False:
            raise RuntimeError('ROOT381 current HOST prerequisites required: '+name)
    for name in ('HOST377_RESULT.json','HOST377_DISPATCH_RESULT.json'):
        result=json.loads((B/'tree-adaptive-M80-M120-target-host-v377'/name).read_text())
        if not result['status'].startswith('PASS_HOST377') or result['NPU_used'] is not False:
            raise RuntimeError('ROOT381 checked source377 prerequisites required: '+name)
    return value


if __name__=='__main__':
    ready()
