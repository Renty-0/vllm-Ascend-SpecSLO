"""Own additive source check; numerical source336 remains byte-for-byte frozen."""
import hashlib
import json
import os
from pathlib import Path

P = Path(__file__).resolve().parent


def ready(expected=None):
    freeze = P/'FREEZE337.json'
    expected = expected or os.environ.get('SPECSLO_V337_ROOT_FREEZE_SHA')
    if expected is None or hashlib.sha256(freeze.read_bytes()).hexdigest() != expected:
        raise RuntimeError('ROOT337 explicit frozen source pin required')
    value = json.loads(freeze.read_text())
    for name, digest in value['files'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('ROOT337 immutable source changed: '+name)
    evidence = json.loads((P/'HOST337_OUTPUT_ROOT_AND_LIFECYCLE_RESULT.json').read_text())
    if evidence['status'] != 'passed_HOST_only' or evidence['NPU_used'] is not False:
        raise RuntimeError('ROOT337 actual output/lifecycle HOST regression required')
    return value


if __name__ == '__main__':
    result = ready()
    print(json.dumps(dict(status='source_only', files=len(result['files']), NPU_used=False)))
