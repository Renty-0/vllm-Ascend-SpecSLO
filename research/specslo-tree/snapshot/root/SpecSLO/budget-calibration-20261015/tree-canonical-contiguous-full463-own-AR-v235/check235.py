"""Explicit frozen local source gate; no model or device action on import."""
import hashlib
import json
import os
from pathlib import Path

P = Path(__file__).resolve().parent
B = P.parent
DONOR = B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
OLD = B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'
OLD_SHA = 'fa58f2866e8fed383e2d1ed1103b2fcb72a731aab3d65060e3a7c3e350a4d260'


def require(value, message):
    if value is not True:
        raise RuntimeError('ROOT235: '+message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def ready(expected=None):
    expected = expected or os.environ.get('SPECSLO_V235_ROOT_FREEZE_SHA')
    require(isinstance(expected, str) and len(expected) == 64 and sha(P/'FREEZE235.json') == expected,
            'explicit own235 source freeze')
    value = read(P/'FREEZE235.json')
    for path, digest in value['files'].items():
        require(sha(path) == digest, 'immutable source '+path)
    require(sha(OLD/'FREEZE218.json') == OLD_SHA, 'original218 freeze retained')
    return value


if __name__ == '__main__':
    ready()
