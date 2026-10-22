"""Additive source gate. All original428 NN obligations remain fresh."""
import hashlib
import json
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
B = P.parent
BASE = B / 'tree-parent4-current-state-metadata-binding-v428'
sys.path.insert(0, str(BASE))
PARENT_FREEZE = 'd484d2cdb3387abb487c2a44ce865571ebb8d3047aeef2c77965fb866b7d0a4a'
PARENT_ARCHIVE = '8e145c79e496d096faa598bc59dd30bf70521a10c34a73c51f3e0cc186041fd1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready(expected=None):
    expected = expected or json.loads((P / 'SOURCE430_RECEIPT.json').read_text())['freeze_sha256']
    if sha(P / 'FREEZE430.json') != expected:
        raise RuntimeError('ROOT430 exact own source freeze')
    value = json.loads((P / 'FREEZE430.json').read_text())
    for path, digest in value['files'].items():
        if sha(path) != digest:
            raise RuntimeError('ROOT430 frozen own file ' + path)
    if sha(BASE / 'FREEZE428.json') != PARENT_FREEZE or sha(BASE / 'SOURCE428.tar.gz') != PARENT_ARCHIVE:
        raise RuntimeError('ROOT430 exact428 source/archive; no NN permission inherited')
    import check428
    check428.ready(PARENT_FREEZE)
    for name in ('HOST430_BINDING_RESULT.json', 'HOST430_CONSTRUCTION_RESULT.json', 'HOST430_CLEAN_STARTUP_RUN.json'):
        row = json.loads((P / name).read_text())
        if not row['status'].startswith('PASS_HOST430') or row['NPU_used'] is not False:
            raise RuntimeError('ROOT430 HOST prerequisite ' + name)
    return value
