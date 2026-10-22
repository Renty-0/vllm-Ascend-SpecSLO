"""Additive source gate. All original434 NN obligations remain fresh."""
import hashlib
import json
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
B = P.parent
BASE = B / 'tree-refill-witness-boolean-retention-full-v437'
sys.path.insert(0, str(BASE))
PARENT_FREEZE = '8ecd9f97279d1ae6023e4ca14188954b34575b26940ec64d1848c8b93a99f8d8'
PARENT_ARCHIVE = '1553265a38fa19caf8bcb32067e9e921627f97f9c892429d68bd891e97aa366e'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready(expected=None):
    expected = expected or json.loads((P / 'SOURCE438_RECEIPT.json').read_text())['freeze_sha256']
    if sha(P / 'FREEZE438.json') != expected:
        raise RuntimeError('ROOT438 exact own source freeze')
    value = json.loads((P / 'FREEZE438.json').read_text())
    for path, digest in value['files'].items():
        if sha(path) != digest:
            raise RuntimeError('ROOT438 frozen own file ' + path)
    if sha(BASE / 'FREEZE437.json') != PARENT_FREEZE or sha(BASE / 'SOURCE437.tar.gz') != PARENT_ARCHIVE:
        raise RuntimeError('ROOT438 exact434 source/archive; no NN permission inherited')
    import check437
    check437.ready(PARENT_FREEZE)
    for name in ('HOST438_POLICY_RESULT.json', 'HOST438_COMPLETE_INTERFACE_RESULT.json'):
        row = json.loads((P / name).read_text())
        if not row['status'].startswith('PASS_HOST438') or row['NPU_used'] is not False:
            raise RuntimeError('ROOT438 HOST prerequisite ' + name)
    return value
