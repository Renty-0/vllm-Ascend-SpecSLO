from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');P=B/'tree-same-prefix-logits-KV-numeric-forensics-v220';OLD=B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'
check='''"""Explicit bounded diagnostic source binding; no Torch import."""
import hashlib,json,os,sys
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
OLD=B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'
DONOR=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
sys.path.insert(0,str(OLD))
from check218 import require,sha,read
OLD_SHA='fa58f2866e8fed383e2d1ed1103b2fcb72a731aab3d650a4d260'
# Correct complete SHA is read only after the immutable known binding below.
OLD_SHA='fa58f2866e8fed383e2d1ed1103b2fcb72a731aab3d650e3a7c3e350a4d260'
def ready(expected=None):
    expected=expected or os.environ.get('SPECSLO_V220_ROOT_FREEZE_SHA')
    require(isinstance(expected,str)and len(expected)==64 and sha(P/'FREEZE220.json')==expected,'explicit ROOT220 freeze')
    f=read(P/'FREEZE220.json')
    for path,digest in f['files'].items():require(sha(path)==digest,'ROOT220 exact input '+path)
    require(sha(OLD/'FREEZE218.json')==OLD_SHA,'immutable original218 closure')
    return f
if __name__=='__main__':ready()
'''
# Use actual already bound SHA, never a hand-transcribed shortened constant.
import hashlib
oldsha=hashlib.sha256((OLD/'FREEZE218.json').read_bytes()).hexdigest()
start=check.index("OLD_SHA='");end=check.index('def ready',start)
check=check[:start]+f'OLD_SHA={oldsha!r}\n'+check[end:]
(P/'check220.py').write_text(check)
for name in ('startup','launch'):
    text=(OLD/f'{name}218.sh').read_text()
    text=text.replace('causal80-N9-to15-allB4-empty-active32-full463','same-prefix-logits-KV')
    text=text.replace(str(OLD),str(P)).replace('check218.py','check220.py').replace('runtime218.py','runtime220.py').replace('startup218.sh','startup220.sh')
    if name=='startup':text=text.replace('export PYTHONPATH='+str(P)+':','export PYTHONPATH='+str(P)+':'+str(OLD)+':')
    else:
        text=text.replace('SPECSLO_V218_ROOT_FREEZE_SHA="${SPECSLO_V218_ROOT_FREEZE_SHA:?explicit ROOT218 freeze required}"',
            'SPECSLO_V220_ROOT_FREEZE_SHA="${SPECSLO_V220_ROOT_FREEZE_SHA:?explicit ROOT220 freeze required}" \\\n  SPECSLO_V218_ROOT_FREEZE_SHA='+oldsha)
    (P/f'{name}220.sh').write_text(text)
print({'old218_sha':oldsha,'prepared':str(P)})
