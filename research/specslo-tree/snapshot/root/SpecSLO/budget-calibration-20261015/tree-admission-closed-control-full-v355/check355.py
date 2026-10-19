"""Own additive source gate plus the exact immutable full348 source gate."""
import os
import sys
from pathlib import Path


P = Path(__file__).resolve().parent
B = P.parent
PARENT = B / 'tree-coalesced-live-refill-M120-singleton-capture-full-v348'
PARENT_FREEZE = '09119c918ad6d28d137a731cf041e69f7a0135dc302169455fa944df6caab5c8'
sys.path[:0] = [str(P), str(PARENT), str(B / 'tree-priority-protected-live-refill-host-v354')]
import check348 as parent

DONOR, OLD, OLD_SHA = parent.DONOR, parent.OLD, parent.OLD_SHA
sha, read = parent.sha, parent.read


def require(value, message):
    if value is not True:
        raise RuntimeError('ROOT355: ' + message)


def ready(expected=None):
    expected = expected or os.environ.get('SPECSLO_V355_ROOT_FREEZE_SHA')
    require(isinstance(expected, str) and len(expected) == 64
            and sha(P / 'FREEZE355.json') == expected, 'exact own frozen source')
    value = read(P / 'FREEZE355.json')
    require(value['admission_mode'] == 'closed' and not value['device_qualification_inherited'],
            'predeclared closed control; fresh model proof required')
    for path, digest in value['files'].items():
        require(sha(path) == digest, 'unchanged frozen dependency ' + path)
    parent.ready(PARENT_FREEZE)
    return value


if __name__ == '__main__':
    ready()
