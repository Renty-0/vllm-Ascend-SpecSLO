"""Only restore inherited CANN search paths; all460 full neural gates remain."""
from pathlib import Path
import hashlib,json,os,sys
P=Path(__file__).resolve().parent;B=P.parent;BASE=B/'tree-adaptive-own-history-full-v460'
sys.path.insert(0,str(BASE))
from host_env460 import setup
setup()
PARENT_FREEZE='89475ed7a4beebdb9ac7029d3b847e38d9327415852bc63782a3372dbc48bcb8'
PARENT_ARCHIVE='44ed9bdbe55fcf839b05d98ad81299aa8236050f46a74a53fcf69cc4be553a6a'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ready(expected=None):
    expected=expected or os.environ.get('SPECSLO_V461_ROOT_FREEZE_SHA')
    if not expected or sha(P/'FREEZE461.json')!=expected:raise RuntimeError('ROOT461 explicit exact source pin')
    value=json.loads((P/'FREEZE461.json').read_text())
    for path,wanted in value['files'].items():
        if sha(path)!=wanted:raise RuntimeError('ROOT461 changed source '+path)
    import check460
    check460.ready(PARENT_FREEZE)
    if sha(BASE/'SOURCE460.tar.gz')!=PARENT_ARCHIVE:raise RuntimeError('ROOT461 unchanged numerical parent archive')
    return value
if __name__=='__main__':ready()
