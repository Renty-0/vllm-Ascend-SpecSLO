"""Require the next source freeze without modifying any original source."""
from pathlib import Path
import hashlib
import json
import os
import sys

P = Path(__file__).resolve().parent
B = P.parent
BASE = B/'tree-current-card-component-binding-full-v465'
sys.path.insert(0, str(BASE))
import check465

PARENT_FREEZE = '34ca11632de0bb62e061c56d41ae066f4b4a4ea5d27c346020f36ebef15d85e5'
PARENT_ARCHIVE = '8533fbb9d2e348765daaec0a0e9328277834dbf0a128638032b1dae016ff48d4'
PHYSICAL = (0, 1, 2, 7)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready(expected=None):
    expected = expected or os.environ.get('SPECSLO_V466_ROOT_FREEZE_SHA')
    if not expected or sha(P/'FREEZE466.json') != expected:
        raise RuntimeError('Explicit466 source freeze required')
    own = json.loads((P/'FREEZE466.json').read_text())
    for path, wanted in own['files'].items():
        if sha(path) != wanted:
            raise RuntimeError('Changed466 source: '+path)
    check465.ready(PARENT_FREEZE)
    if sha(BASE/'SOURCE465.tar.gz') != PARENT_ARCHIVE:
        raise RuntimeError('Exact465 archive required')
    if own['physical_cards'] != list(PHYSICAL) or own['maximum_new_cohort'] != 48:
        raise RuntimeError('Declared current cards and cohort48 required')
    return own


if __name__ == '__main__':
    ready()
