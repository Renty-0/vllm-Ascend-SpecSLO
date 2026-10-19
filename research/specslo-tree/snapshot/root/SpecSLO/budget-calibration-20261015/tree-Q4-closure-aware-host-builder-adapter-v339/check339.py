"""Additive source gate; no failed result confers model qualification."""
import hashlib
import json
import os
from pathlib import Path

P = Path(__file__).resolve().parent


def ready(expected=None):
    path = P/'FREEZE339.json'
    expected = expected or os.environ.get('SPECSLO_V339_ROOT_FREEZE_SHA')
    if expected is None or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError('ROOT339 explicit source pin mismatch')
    value = json.loads(path.read_text())
    for name, digest in value['files'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('ROOT339 frozen source changed: '+name)
    for name in ('HOST339_ACTUAL_FRONTEND_RESULT.json', 'HOST339_RUNTIME_RESULT.json', 'HOST339_ACTUAL_INSTALL_RESULT.json'):
        result = json.loads((P/name).read_text())
        if not result['status'].startswith('passed') or result['NPU_used'] is not False:
            raise RuntimeError('ROOT339 actual HOST prerequisite required: '+name)
    return value


if __name__ == '__main__':
    value = ready()
    print(json.dumps(dict(status='source_only', files=len(value['files']), NPU_used=False)))
