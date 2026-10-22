"""Additive source gate. All original434 NN obligations remain fresh."""
import hashlib
import json
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
B = P.parent
BASE = B / 'tree-nonempty-wholeprompt-current-model-full-v434'
sys.path.insert(0, str(BASE))
PARENT_FREEZE = 'a79c6bd86cc38230d4e3ed13e4ee35499da05ebcb6d82338317f110ef452a042'
PARENT_ARCHIVE = '6ff2dd9c0dc0d6f51919627f83bd6e0b06ae575b060e2f8d4cc6edb4588e73d0'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready(expected=None):
    expected = expected or json.loads((P / 'SOURCE436_RECEIPT.json').read_text())['freeze_sha256']
    if sha(P / 'FREEZE436.json') != expected:
        raise RuntimeError('ROOT436 exact own source freeze')
    value = json.loads((P / 'FREEZE436.json').read_text())
    for path, digest in value['files'].items():
        if sha(path) != digest:
            raise RuntimeError('ROOT436 frozen own file ' + path)
    if sha(BASE / 'FREEZE434.json') != PARENT_FREEZE or sha(BASE / 'SOURCE434.tar.gz') != PARENT_ARCHIVE:
        raise RuntimeError('ROOT436 exact434 source/archive; no NN permission inherited')
    import check434
    check434.ready(PARENT_FREEZE)
    for name in ('HOST436_RPC_RESULT.json', 'HOST436_CONSTRUCTION_RESULT.json', 'HOST436_CLEAN_STARTUP_RUN.json'):
        row = json.loads((P / name).read_text())
        if not row['status'].startswith('PASS_HOST436') or row['NPU_used'] is not False:
            raise RuntimeError('ROOT436 HOST prerequisite ' + name)
    return value
