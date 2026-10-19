"""Own source checks; the supervisor separately checks all parent archives."""
import hashlib
import json
import os
from pathlib import Path

P = Path(__file__).resolve().parent


def ready(expected=None):
    path = P/'FREEZE336.json'
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = expected or os.environ.get('SPECSLO_V336_ROOT_FREEZE_SHA')
    if expected is None or expected != actual:
        raise RuntimeError('ROOT336 explicit source freeze mismatch')
    value = json.loads(path.read_text())
    for name, digest in value['files'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('ROOT336 frozen source changed: '+name)
    for name in ('HOST336_METADATA_RESULT.json', 'HOST336_CALLBACK_RESULT.json',
                 'HOST336_GUARD_RESULT.json', 'HOST336_RUNTIME_PROJECTION_RESULT.json'):
        result = json.loads((P/name).read_text())
        if not result['status'].startswith('passed') or result['NPU_used'] is not False:
            raise RuntimeError('ROOT336 actual HOST prerequisite missing: '+name)
    return value


if __name__ == '__main__':
    value = ready()
    print(json.dumps(dict(status='source_only', files=len(value['files']), NPU_used=False)))
