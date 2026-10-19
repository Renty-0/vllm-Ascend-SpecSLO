"""Explicit additive source gate; component evidence never grants LIVE scope."""
import hashlib
import json
import os
from pathlib import Path
import sys

P=Path(__file__).parent
B=P.parent
PARENT=B/'tree-adaptive-M80-M120-target-math-cost-v378'
PARENT_FREEZE='e7beb04cadd68479440ec26a1e8b3a39949d395e7f6485af658bc219c5853bc3'


def ready(expected=None):
    expected=expected or os.environ.get('SPECSLO_V379_ROOT_FREEZE_SHA')
    path=P/'FREEZE379.json'
    if expected is None or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise RuntimeError('ROOT379 explicit frozen source pin required')
    value=json.loads(path.read_text())
    for name,digest in value['files'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('ROOT379 source changed: '+name)
    sys.path.insert(0,str(PARENT))
    from check378 import ready as parent_ready
    parent_ready(PARENT_FREEZE)
    for name in ('HOST379_CONSTRUCTION_RESULT.json',):
        result=json.loads((P/name).read_text())
        if not result['status'].startswith('PASS_HOST379') or result['NPU_used'] is not False:
            raise RuntimeError('ROOT379 current HOST prerequisites required: '+name)
    for name in ('HOST377_RESULT.json','HOST377_DISPATCH_RESULT.json'):
        result=json.loads((B/'tree-adaptive-M80-M120-target-host-v377'/name).read_text())
        if not result['status'].startswith('PASS_HOST377') or result['NPU_used'] is not False:
            raise RuntimeError('ROOT379 checked source377 prerequisites required: '+name)
    return value


if __name__=='__main__':
    ready()
