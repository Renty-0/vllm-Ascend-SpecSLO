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
        raise RuntimeError('ROOT383: ' + message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 ** 2), b''):
            value.update(data)
    return value.hexdigest()


def ready(expected=None):
    expected = expected or os.environ.get('SPECSLO_V383_ROOT_FREEZE_SHA')
    require(isinstance(expected, str) and len(expected) == 64 and sha(P / 'FREEZE383.json') == expected,
            'explicit383 source freeze')
    value = json.loads((P / 'FREEZE383.json').read_text())
    for filename, digest in value['files'].items():
        require(sha(filename) == digest, 'immutable383 input ' + filename)
    require(sha(PARENT / 'FREEZE257.json') == PARENT_FREEZE
            and sha(PARENT / 'SOURCE257.tar.gz') == PARENT_ARCHIVE, 'exact original best257 source/archive')
    if str(PARENT) not in sys.path:
        sys.path.append(str(PARENT))
    from check257 import ready as original_ready
    original_ready(PARENT_FREEZE)
    component=B/'tree-hybrid-native-method-dedent-v384'
    require(sha(component/'FREEZE384.json')=='a647343b81fac81459d6829644a59ab12425955647b11a78c15a4978871a1e5b'
        and sha(component/'SOURCE384.tar.gz')=='2309d02384a0f76202cfc3c9de26e8d041990fc224d03e5a192cb53e2126767f',
        'exact additive hybrid implementation sources; no model proof inherited')
    for filename in ('HOST383_RESULT.json', 'HOST383_SCOPE_RESULT.json', 'HOST383_CONSTRUCTION_RESULT.json'):
        prerequisites = read(P / filename)
        require(prerequisites['status'].startswith('PASS_HOST383')
                and prerequisites['NPU_used'] is False,
                'current383 boundary/callback/runtime/dispatch HOST prerequisites: ' + filename)
    return value


if __name__ == '__main__':
    ready()
