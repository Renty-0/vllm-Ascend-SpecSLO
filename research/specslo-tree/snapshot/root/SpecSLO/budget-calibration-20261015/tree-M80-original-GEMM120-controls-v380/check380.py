"""Explicit additive source gate; component evidence never grants LIVE scope."""
import hashlib
import json
import os
from pathlib import Path
import sys

P=Path(__file__).parent
B=P.parent
PARENT=B/'tree-adaptive80-first-layer-divergence-v379'
PARENT_FREEZE='318f60b9d142a4c67c6c6f50e0b91e5f38aec40b365a0d4dea2a32bc1e28449c'


def ready(expected=None):
    expected=expected or os.environ.get('SPECSLO_V380_ROOT_FREEZE_SHA')
    path=P/'FREEZE380.json'
    if expected is None or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise RuntimeError('ROOT380 explicit frozen source pin required')
    value=json.loads(path.read_text())
    for name,digest in value['files'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('ROOT380 source changed: '+name)
    sys.path.insert(0,str(PARENT))
    from check379 import ready as parent_ready
    parent_ready(PARENT_FREEZE)
    for name in ('HOST380_CONSTRUCTION_RESULT.json',):
        result=json.loads((P/name).read_text())
        if not result['status'].startswith('PASS_HOST380') or result['NPU_used'] is not False:
            raise RuntimeError('ROOT380 current HOST prerequisites required: '+name)
    for name in ('HOST377_RESULT.json','HOST377_DISPATCH_RESULT.json'):
        result=json.loads((B/'tree-adaptive-M80-M120-target-host-v377'/name).read_text())
        if not result['status'].startswith('PASS_HOST377') or result['NPU_used'] is not False:
            raise RuntimeError('ROOT380 checked source377 prerequisites required: '+name)
    return value


if __name__=='__main__':
    ready()
