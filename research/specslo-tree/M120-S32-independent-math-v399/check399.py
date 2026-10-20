"""Source readiness for finite new math; no inherited model qualification."""
import hashlib
import json
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
PREVIOUS = P.parent/'tree-active64-whole-proposal-segment-limit-v398'
PARENT_FREEZE = 'dfb6cfd44d2f524d7ad28892ec055d50d1035910fb71dbbc5486dc060cb5e73d'
PARENT_ARCHIVE = '5c79ae661393d9bb347dd6f53ff447d34e09fa121bc90d4d43844cdb63357f6a'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready(expected=None):
    expected = expected or json.loads((P/'SOURCE399_RECEIPT.json').read_text())['freeze_sha256']
    if sha(P/'FREEZE399.json') != expected:
        raise RuntimeError('ROOT399 exact own source freeze')
    value = json.loads((P/'FREEZE399.json').read_text())
    for name, digest in value['files'].items():
        if sha(name) != digest:
            raise RuntimeError('ROOT399 immutable source changed '+name)
    if not (value['parent398_freeze_sha256'] == sha(PREVIOUS/'FREEZE398.json') == PARENT_FREEZE
            and value['parent398_archive_sha256'] == sha(PREVIOUS/'SOURCE398.tar.gz') == PARENT_ARCHIVE):
        raise RuntimeError('ROOT399 exact reviewed parent398 source')
    for name in ('HOST399_RESULT.json', 'HOST399_PRECONSTRUCTOR_RESULT.json'):
        host = json.loads((P/name).read_text())
        if not host['status'].startswith('PASS_HOST399') or host['NPU_used'] is not False:
            raise RuntimeError('ROOT399 actual HOST prerequisite '+name)
    sys.path.insert(0, str(PREVIOUS))
    from check398 import ready as original
    original(PARENT_FREEZE)
    return value


if __name__ == '__main__':
    ready()
