"""Explicit bounded diagnostic source binding; no Torch import."""
import hashlib,json,os,sys
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
OLD=B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'
DONOR=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
sys.path.insert(0,str(OLD))
from check218 import require,sha,read
OLD_SHA='fa58f2866e8fed383e2d1ed1103b2fcb72a731aab3d65060e3a7c3e350a4d260'
def ready(expected=None):
    expected=expected or os.environ.get('SPECSLO_V227_ROOT_FREEZE_SHA')
    require(isinstance(expected,str)and len(expected)==64 and sha(P/'FREEZE227.json')==expected,'explicit ROOT227 freeze')
    f=read(P/'FREEZE227.json')
    for path,digest in f['files'].items():require(sha(path)==digest,'ROOT227 exact input '+path)
    require(sha(OLD/'FREEZE218.json')==OLD_SHA,'immutable original218 closure')
    return f
if __name__=='__main__':ready()
