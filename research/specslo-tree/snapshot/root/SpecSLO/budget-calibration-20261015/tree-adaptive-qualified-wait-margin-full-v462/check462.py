"""Preserve all numerical461/460 sources; pin only the declared wait delta."""
from pathlib import Path
import hashlib,json,os,sys
P=Path(__file__).resolve().parent;B=P.parent;BASE=B/'tree-adaptive-retained-CANN-path-full-v461'
sys.path.insert(0,str(BASE))
import check461
PARENT_FREEZE='ec113f2b8231fb1ec0dcf310ae2bf6b9ecccab0624dfcf25608319f9299ea568'
PARENT_ARCHIVE='fd7a531429b415d28446f53d6a6a12d7c294473ea5136bde0786c02cff89a4b7'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ready(expected=None):
    expected=expected or os.environ.get('SPECSLO_V462_ROOT_FREEZE_SHA')
    if not expected or sha(P/'FREEZE462.json')!=expected:raise RuntimeError('ROOT462 explicit source freeze pin')
    own=json.loads((P/'FREEZE462.json').read_text())
    for path,wanted in own['files'].items():
        if sha(path)!=wanted:raise RuntimeError('ROOT462 changed source '+path)
    check461.ready(PARENT_FREEZE)
    if sha(BASE/'SOURCE461.tar.gz')!=PARENT_ARCHIVE:raise RuntimeError('ROOT462 unchanged parent archive')
    return own
if __name__=='__main__':ready()
