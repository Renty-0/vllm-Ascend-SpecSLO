"""HOST source gate only; no device/model import or inherited neural permit."""
import hashlib
import json
import os
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
B = P.parent
PARENT = B / 'tree-static-q5-M120-SLO-first-full463-own-AR-v257'
PARENT_FREEZE = 'b352ac6bb9f3ae182d33d2f11dc278f7e49c2b8dff9d9d514c0b32d7a14a01f2'
PARENT_ARCHIVE = 'f268e73e73b334f29dba4da8413a9a9efc6f4bbcabdbd975554c801f6e34e8b0'
DONOR = B / 'tree-causal80-small-cohort-role-aware-audit-full-v209'
OLD = B / 'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'
OLD_SHA = 'fa58f2866e8fed383e2d1ed1103b2fcb72a731aab3d65060e3a7c3e350a4d260'


def read(path):
    return json.loads(Path(path).read_text())



def require(value, message):
    if value is not True:
        raise RuntimeError('ROOT394: ' + message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 ** 2), b''):
            value.update(data)
    return value.hexdigest()


def ready(expected=None):
    expected = expected or read(P / 'SOURCE394_RECEIPT.json')['freeze_sha256']
    require(isinstance(expected, str) and len(expected) == 64 and sha(P / 'FREEZE394.json') == expected,
            'explicit394 source freeze')
    value = json.loads((P / 'FREEZE394.json').read_text())
    for filename, digest in value['files'].items():
        require(sha(filename) == digest, 'immutable394 input ' + filename)
    require(sha(PARENT / 'FREEZE257.json') == PARENT_FREEZE
            and sha(PARENT / 'SOURCE257.tar.gz') == PARENT_ARCHIVE, 'exact original best257 source/archive')
    if str(PARENT) not in sys.path:
        sys.path.append(str(PARENT))
    from check257 import ready as original_ready
    original_ready(PARENT_FREEZE)
    for filename in ('HOST394_RESULT.json', 'HOST394_SCOPE_RESULT.json', 'HOST394_CONSTRUCTION_RESULT.json', 'HOST394_CAPACITY_RESULT.json'):
        prerequisites = read(P / filename)
        require(prerequisites['status'].startswith('PASS_HOST394')
                and prerequisites['NPU_used'] is False,
                'current394 boundary/callback/runtime/dispatch HOST prerequisites: ' + filename)
    return value


if __name__ == '__main__':
    ready()
