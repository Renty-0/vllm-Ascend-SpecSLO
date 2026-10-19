"""New source pin plus all unchanged original344 HOST prerequisites."""
import hashlib
import json
import os
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
PARENT = P.parent / 'tree-Q5-uniform-KV-bucket-branch-math-cost-v344'


def ready(expected=None):
    expected = expected or os.environ.get('SPECSLO_V361_ROOT_FREEZE_SHA')
    raw = (P / 'FREEZE361.json').read_bytes()
    if expected is None or hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError('ROOT361 explicit source pin mismatch')
    value = json.loads(raw)
    for name, digest in value['files'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('ROOT361 immutable source changed: ' + name)
    sys.path.insert(0, str(PARENT))
    from check344 import ready as previous_ready
    previous_ready(value['parent_freeze_sha256'])
    for name in ('HOST361_COST_PROTOCOL_RESULT.json', 'HOST361_RUNTIME_RESULT.json',
                 'HOST361_MODE_AUDIT_RESULT.json', 'HOST361_GUARD_SOURCE_RESULT.json'):
        result = json.loads((P / name).read_text())
        if not result['status'].startswith('PASS_HOST') or result['NPU_used'] is not False:
            raise RuntimeError('ROOT361 HOST prerequisite missing: ' + name)
    return value


if __name__ == '__main__':
    value = ready()
    print(json.dumps(dict(status='source_only', files=len(value['files']), NPU_used=False)))
