"""Preserve the frozen463 algorithm while pinning the current device epoch."""
from pathlib import Path
import hashlib
import json
import os
import sys

P = Path(__file__).resolve().parent
B = P.parent
BASE = B / 'tree-slack-cost-aware-refill-full-v463'
sys.path.insert(0, str(BASE))
import check463

PARENT_FREEZE = 'd27c7eb676ad7268ad7911c455e86177b3691b7e8b21cc4c28799c5f04de19a5'
PARENT_ARCHIVE = 'e312cdeaf16066b101b565e67e0a1258eb4cfa6683632af6d279527f741a630c'
PHYSICAL = (0, 1, 2, 7)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready(expected=None):
    expected = expected or os.environ.get('SPECSLO_V464_ROOT_FREEZE_SHA')
    if not expected or sha(P / 'FREEZE464.json') != expected:
        raise RuntimeError('Explicit current464 source freeze required')
    own = json.loads((P / 'FREEZE464.json').read_text())
    for path, wanted in own['files'].items():
        if sha(path) != wanted:
            raise RuntimeError('Changed current464 source: ' + path)
    check463.ready(PARENT_FREEZE)
    if sha(BASE / 'SOURCE463.tar.gz') != PARENT_ARCHIVE:
        raise RuntimeError('Exact original463 source archive required')
    if own['physical_cards'] != list(PHYSICAL):
        raise RuntimeError('Explicit current four-card assignment required')
    return own


if __name__ == '__main__':
    ready()
