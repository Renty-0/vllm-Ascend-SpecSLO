"""Explicit frozen local source gate; no model or device action on import."""
import hashlib
import json
import os
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
B = P.parent
DONOR = B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
OLD = B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'
sys.path.extend(['/root/SpecSLO/budget-calibration-20261015/tree-prefix-grid-target-replay-node-cost-v280', '/root/SpecSLO/budget-calibration-20261015/tree-prefix-grid-byte-KV-copy-v278', '/root/SpecSLO/budget-calibration-20261015/tree-genuine-alternate-draft-observed-component-v277', '/root/SpecSLO/budget-calibration-20261015/tree-private-alternate-draft-tail-host-v271', '/root/SpecSLO/budget-calibration-20261015/tree-alternate-chain-target-host-v269', '/root/SpecSLO/budget-calibration-20261015/tree-fused-byte-branch-KV-copy-v263', '/root/SpecSLO/budget-calibration-20261015/tree-fused-branch-KV-row-copy-v262', '/root/SpecSLO/budget-calibration-20261015/tree-paired-alternate-gain-host-v274', '/root/SpecSLO/budget-calibration-20261015/tree-genuine-branch-followup-host-v281', '/root/SpecSLO/budget-calibration-20261015/tree-root-alternate-live-policy-host-v276'])
OLD_SHA = 'fa58f2866e8fed383e2d1ed1103b2fcb72a731aab3d65060e3a7c3e350a4d260'


def require(value, message):
    if value is not True:
        raise RuntimeError('ROOT287: '+message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def ready(expected=None):
    expected = expected or os.environ.get('SPECSLO_V287_ROOT_FREEZE_SHA')
    require(isinstance(expected, str) and len(expected) == 64 and sha(P/'FREEZE287.json') == expected,
            'explicit own256 source freeze')
    value = read(P/'FREEZE287.json')
    for path, digest in value['files'].items():
        require(sha(path) == digest, 'immutable source '+path)
    require(sha(OLD/'FREEZE218.json') == OLD_SHA, 'original218 freeze retained')
    return value


if __name__ == '__main__':
    ready()
