"""Additive source gate; no failed result confers model qualification."""
import hashlib
import json
import os
from pathlib import Path

P = Path(__file__).resolve().parent


def ready(expected=None):
    path = P/'FREEZE344.json'
    expected = expected or os.environ.get('SPECSLO_V344_ROOT_FREEZE_SHA')
    if expected is None or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError('ROOT344 explicit source pin mismatch')
    value = json.loads(path.read_text())
    for name, digest in value['files'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('ROOT344 frozen source changed: '+name)
    for name in ('HOST344_PREPARE_RESULT.json','HOST344_RUNTIME_RESULT.json','HOST344_DISPATCH_RESULT.json','HOST344_OBSERVER_RESULT.json'):
        result = json.loads((P/name).read_text())
        if not result['status'].startswith('passed') or result['NPU_used'] is not False:
            raise RuntimeError('ROOT344 actual HOST prerequisite required: '+name)
    return value


if __name__ == '__main__':
    value = ready()
    print(json.dumps(dict(status='source_only', files=len(value['files']), NPU_used=False)))
