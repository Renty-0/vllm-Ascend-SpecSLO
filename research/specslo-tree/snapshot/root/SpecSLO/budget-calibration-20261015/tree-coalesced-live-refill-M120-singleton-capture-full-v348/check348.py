"""Explicit frozen local source gate; no model or device action on import."""
import hashlib
import json
import os
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
B = P.parent
sys.path.extend([str(B/'tree-small-refill-full-next-subset-domain-v350'),str(B/'tree-coalesced-live-refill-host-v345'),str(B/'tree-current-prepare-cost-eligibility-host-v312')])
sys.path.extend(['/root/SpecSLO/budget-calibration-20261015/tree-single-validated-prefix-prepare-host-v305', '/root/SpecSLO/budget-calibration-20261015/tree-single-validated-prefix-descriptors-host-v304'])
sys.path.extend(['/root/SpecSLO/budget-calibration-20261015/tree-concrete-device-span-dual-cache-target-cost-v296', '/root/SpecSLO/budget-calibration-20261015/tree-spare-capacity-budget-host-v290', '/root/SpecSLO/budget-calibration-20261015/tree-request-conditional-risk-gain-host-v294', '/root/SpecSLO/budget-calibration-20261015/tree-prefix-precopy-dual-cachewrite-host-v288', '/root/SpecSLO/budget-calibration-20261015/tree-contiguous-prefix-span-copy-host-v292'])
DONOR = B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
OLD = B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'
sys.path.extend(['/root/SpecSLO/budget-calibration-20261015/tree-prefix-grid-target-replay-node-cost-v280', '/root/SpecSLO/budget-calibration-20261015/tree-prefix-grid-byte-KV-copy-v278', '/root/SpecSLO/budget-calibration-20261015/tree-genuine-alternate-draft-observed-component-v277', '/root/SpecSLO/budget-calibration-20261015/tree-private-alternate-draft-tail-host-v271', '/root/SpecSLO/budget-calibration-20261015/tree-alternate-chain-target-host-v269', '/root/SpecSLO/budget-calibration-20261015/tree-fused-byte-branch-KV-copy-v263', '/root/SpecSLO/budget-calibration-20261015/tree-fused-branch-KV-row-copy-v262', '/root/SpecSLO/budget-calibration-20261015/tree-paired-alternate-gain-host-v274', '/root/SpecSLO/budget-calibration-20261015/tree-genuine-branch-followup-host-v281', '/root/SpecSLO/budget-calibration-20261015/tree-root-alternate-live-policy-host-v276'])
OLD_SHA = 'fa58f2866e8fed383e2d1ed1103b2fcb72a731aab3d65060e3a7c3e350a4d260'


def require(value, message):
    if value is not True:
        raise RuntimeError('ROOT348: '+message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def ready(expected=None):
    expected = expected or os.environ.get('SPECSLO_V348_ROOT_FREEZE_SHA')
    require(isinstance(expected, str) and len(expected) == 64 and sha(P/'FREEZE348.json') == expected,
            'explicit own256 source freeze')
    value = read(P/'FREEZE348.json')
    for path, digest in value['files'].items():
        require(sha(path) == digest, 'immutable source '+path)
    require(sha(OLD/'FREEZE218.json') == OLD_SHA, 'original218 freeze retained')
    return value


if __name__ == '__main__':
    ready()
