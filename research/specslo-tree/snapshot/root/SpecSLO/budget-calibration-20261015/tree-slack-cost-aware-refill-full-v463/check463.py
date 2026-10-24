"""Pin the qualified numerical parent and the new host admission source."""

from pathlib import Path
import hashlib
import json
import os
import sys

P = Path(__file__).resolve().parent
B = P.parent
BASE = B / 'tree-adaptive-qualified-wait-margin-full-v462'
sys.path.insert(0, str(BASE))
import check462

PARENT_FREEZE = '9f79697914d74f79cc94fe4605c8f20665c44ee2f6b33fe99b59b73d7b4e2280'
PARENT_ARCHIVE = 'cebb1af09f459d8e0a64b9a6b7561721f07e787cca5fb809fd09552e483bfbd9'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready(expected=None):
    expected = expected or os.environ.get('SPECSLO_V463_ROOT_FREEZE_SHA')
    if not expected or sha(P / 'FREEZE463.json') != expected:
        raise RuntimeError('ROOT463 explicit source freeze pin required')
    own = json.loads((P / 'FREEZE463.json').read_text())
    for path, wanted in own['files'].items():
        if sha(path) != wanted:
            raise RuntimeError('ROOT463 changed source: ' + path)
    check462.ready(PARENT_FREEZE)
    if sha(BASE / 'SOURCE462.tar.gz') != PARENT_ARCHIVE:
        raise RuntimeError('ROOT463 exact original numerical parent archive required')
    from admission463 import validate_model
    validate_model()
    return own


if __name__ == '__main__':
    ready()
