"""Additive source gate. All original434 NN obligations remain fresh."""
import hashlib
import json
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
B = P.parent
BASE = B / 'tree-nonempty-qualified-RPC-full-v436'
sys.path.insert(0, str(BASE))
PARENT_FREEZE = '4390b843830383905243d8dfb9aa37328a2c4a0076184cb171605b6cca39e5a3'
PARENT_ARCHIVE = 'e9d88c023243b1eb67cc410979b126eb9acf047f3bcad0ea1aa4f6ef2144ddc9'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready(expected=None):
    expected = expected or json.loads((P / 'SOURCE437_RECEIPT.json').read_text())['freeze_sha256']
    if sha(P / 'FREEZE437.json') != expected:
        raise RuntimeError('ROOT437 exact own source freeze')
    value = json.loads((P / 'FREEZE437.json').read_text())
    for path, digest in value['files'].items():
        if sha(path) != digest:
            raise RuntimeError('ROOT437 frozen own file ' + path)
    if sha(BASE / 'FREEZE436.json') != PARENT_FREEZE or sha(BASE / 'SOURCE436.tar.gz') != PARENT_ARCHIVE:
        raise RuntimeError('ROOT437 exact434 source/archive; no NN permission inherited')
    import check436
    check436.ready(PARENT_FREEZE)
    for name in ('HOST437_RETENTION_RESULT.json', 'HOST437_CONSTRUCTION_RESULT.json'):
        row = json.loads((P / name).read_text())
        if not row['status'].startswith('PASS_HOST437') or row['NPU_used'] is not False:
            raise RuntimeError('ROOT437 HOST prerequisite ' + name)
    return value
