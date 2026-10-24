"""Pin the current component correction and all frozen parent sources."""
from pathlib import Path
import hashlib
import json
import os
import sys

P = Path(__file__).resolve().parent
B = P.parent
BASE = B / 'tree-current-card-rebind-full-v464'
sys.path.insert(0, str(BASE))
import check464

PARENT_FREEZE = 'aadc2090b2cb5ffc22243ab19fdbc007481b79a08c7b5c1b806867e3835c12b1'
PARENT_ARCHIVE = 'c1e891ef13827d0439f0f56728612b25e0ea00ff194bed87d386026b6a004ae3'
PHYSICAL = (0, 1, 2, 7)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready(expected=None):
    expected = expected or os.environ.get('SPECSLO_V465_ROOT_FREEZE_SHA')
    if not expected or sha(P / 'FREEZE465.json') != expected:
        raise RuntimeError('Explicit465 source freeze required')
    own = json.loads((P / 'FREEZE465.json').read_text())
    for path, wanted in own['files'].items():
        if sha(path) != wanted:
            raise RuntimeError('Changed465 source: ' + path)
    check464.ready(PARENT_FREEZE)
    if sha(BASE / 'SOURCE464.tar.gz') != PARENT_ARCHIVE:
        raise RuntimeError('Exact464 parent archive required')
    if own['physical_cards'] != list(PHYSICAL):
        raise RuntimeError('Explicit current four-card assignment required')
    return own


if __name__ == '__main__':
    ready()
