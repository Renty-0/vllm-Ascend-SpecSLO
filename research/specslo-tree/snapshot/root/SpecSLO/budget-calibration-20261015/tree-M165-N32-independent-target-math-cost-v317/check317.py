"""Explicit bounded diagnostic source binding; no Torch import."""
import hashlib,json,os,sys
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
sys.path.extend(['/root/SpecSLO/budget-calibration-20261015/tree-M165-N32-independent-target-math-cost-v317', '/root/SpecSLO/budget-calibration-20261015/tree-M165-single-prepare-host-v313', '/root/SpecSLO/budget-calibration-20261015/tree-uniform-q5-M165-S33-C128-host-v285', '/root/SpecSLO/budget-calibration-20261015/tree-single-prepare-target-math-cost-v311'])
sys.path.extend(['/root/SpecSLO/budget-calibration-20261015/tree-single-validated-prefix-prepare-host-v305', '/root/SpecSLO/budget-calibration-20261015/tree-single-validated-prefix-descriptors-host-v304', '/root/SpecSLO/budget-calibration-20261015/tree-concrete-device-span-dual-cache-target-cost-v296'])
sys.path.extend([str(B/'tree-prefix-precopy-dual-cachewrite-host-v288'),str(B/'tree-contiguous-prefix-span-copy-host-v292'),str(B/'tree-fused-branch-KV-row-copy-v262')])
OLD=B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'
DONOR=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
sys.path.append(str(B/'tree-high-normal-cap4-native-families-v148'))
sys.path.extend([str(B/'tree-prefix-grid-byte-KV-copy-v278'),str(B/'tree-alternate-chain-target-host-v269'),str(B/'tree-static-q5-M165-node-math-v255')])
sys.path.insert(0,str(OLD))
sys.path.append(str(B/'tree-contiguous-branch-tail-final-metadata-math-v234'))
from check218 import require,sha,read
OLD_SHA='fa58f2866e8fed383e2d1ed1103b2fcb72a731aab3d65060e3a7c3e350a4d260'
def ready(expected=None):
    expected=expected or os.environ.get('SPECSLO_V317_ROOT_FREEZE_SHA')
    require(isinstance(expected,str)and len(expected)==64 and sha(P/'FREEZE317.json')==expected,'explicit ROOT317 freeze')
    f=read(P/'FREEZE317.json')
    for path,digest in f['files'].items():require(sha(path)==digest,'ROOT317 exact input '+path)
    require(sha(OLD/'FREEZE218.json')==OLD_SHA,'immutable original218 closure')
    return f
if __name__=='__main__':ready()
