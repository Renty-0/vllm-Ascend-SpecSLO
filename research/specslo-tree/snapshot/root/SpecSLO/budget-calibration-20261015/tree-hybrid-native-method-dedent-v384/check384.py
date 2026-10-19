"""Explicit additive source gate; component evidence never grants LIVE scope."""
import hashlib
import json
import os
from pathlib import Path
import sys

P=Path(__file__).parent
B=P.parent
PARENT=B/'tree-M80-attention-GEMM120-target-v382'
PARENT_FREEZE='f8b65aa81b78b9f33d3e0e97e30f62841f310160e52c321fd0b8a0243023d6c4'


def ready(expected=None):
    expected=expected or os.environ.get('SPECSLO_V384_ROOT_FREEZE_SHA')
    path=P/'FREEZE384.json'
    if expected is None or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise RuntimeError('ROOT384 explicit frozen source pin required')
    value=json.loads(path.read_text())
    for name,digest in value['files'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('ROOT384 source changed: '+name)
    sys.path.insert(0,str(PARENT))
    from check382 import ready as parent_ready
    parent_ready(PARENT_FREEZE)
    for name in ('HOST384_CONSTRUCTION_RESULT.json',):
        result=json.loads((P/name).read_text())
        if not result['status'].startswith('PASS_HOST384') or result['NPU_used'] is not False:
            raise RuntimeError('ROOT384 current HOST prerequisites required: '+name)
    for name in ('HOST377_RESULT.json','HOST377_DISPATCH_RESULT.json'):
        result=json.loads((B/'tree-adaptive-M80-M120-target-host-v377'/name).read_text())
        if not result['status'].startswith('PASS_HOST377') or result['NPU_used'] is not False:
            raise RuntimeError('ROOT384 checked source377 prerequisites required: '+name)
    return value


if __name__=='__main__':
    ready()
