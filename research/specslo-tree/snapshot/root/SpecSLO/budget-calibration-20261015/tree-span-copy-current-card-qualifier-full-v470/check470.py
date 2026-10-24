"""Explicit new source freeze above the unchanged complete466 lifecycle."""
from pathlib import Path
import hashlib
import json
import os
import sys

P=Path(__file__).resolve().parent
B=P.parent
BASE=B/'tree-credit-cost-cohort48-full-v466'
sys.path.insert(0,str(BASE))
import check466

PARENT_FREEZE='69c5d2eb35f248fc47c37db5d5d61786986af8ea3c52eb629b69077b54f55f34'
PARENT_ARCHIVE='55b111f5703faa04dd189f6c2b307c9574446371debf42ab218e9c4959eb678a'
PHYSICAL=(0,1,2,7)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready(expected=None):
    expected=expected or os.environ.get('SPECSLO_V470_ROOT_FREEZE_SHA')
    if not expected or sha(P/'FREEZE470.json')!=expected:
        raise RuntimeError('Explicit470 source freeze required')
    own=json.loads((P/'FREEZE470.json').read_text())
    for path,wanted in own['files'].items():
        if sha(path)!=wanted:raise RuntimeError('Changed470 source: '+path)
    check466.ready(PARENT_FREEZE)
    if sha(BASE/'SOURCE466.tar.gz')!=PARENT_ARCHIVE:
        raise RuntimeError('Exact466 archive required')
    if (own['physical_cards']!=list(PHYSICAL) or own['global_active_limit']!=64
            or own['maximum_new_cohort']!=48 or own['target_copy_programs']!=64):
        raise RuntimeError('Exact declared current-card cohort48/global64/two-worker scope required')
    return own


if __name__=='__main__':ready()
