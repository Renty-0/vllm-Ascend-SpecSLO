"""Additive source gate. All original430 NN obligations remain fresh."""
import hashlib
import json
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
B = P.parent
BASE = B / 'tree-parent4-prestartup-commit-binding-v430'
sys.path.insert(0, str(BASE))
PARENT_FREEZE = '7f62e8f127ea33a70bde2068399e7d4469fb5312149fd9e73ff2e206bf619424'
PARENT_ARCHIVE = 'd4450d7a20b5eb3ad710901b3246c5b56eae4365377b55f81b5b1dce5d7d1d4a'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready(expected=None):
    expected = expected or json.loads((P / 'SOURCE433_RECEIPT.json').read_text())['freeze_sha256']
    if sha(P / 'FREEZE433.json') != expected:
        raise RuntimeError('ROOT433 exact own source freeze')
    value = json.loads((P / 'FREEZE433.json').read_text())
    for path, digest in value['files'].items():
        if sha(path) != digest:
            raise RuntimeError('ROOT433 frozen own file ' + path)
    if sha(BASE / 'FREEZE430.json') != PARENT_FREEZE or sha(BASE / 'SOURCE430.tar.gz') != PARENT_ARCHIVE:
        raise RuntimeError('ROOT433 exact430 source/archive; no NN permission inherited')
    import check430
    check430.ready(PARENT_FREEZE)
    for name in ('HOST433_LIFETIME_RESULT.json', 'HOST433_CONSTRUCTION_RESULT.json', 'HOST433_CLEAN_STARTUP_RUN.json'):
        row = json.loads((P / name).read_text())
        if not row['status'].startswith('PASS_HOST433') or row['NPU_used'] is not False:
            raise RuntimeError('ROOT433 HOST prerequisite ' + name)
    return value
