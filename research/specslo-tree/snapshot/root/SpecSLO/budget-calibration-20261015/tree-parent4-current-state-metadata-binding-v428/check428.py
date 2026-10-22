"""Additive reference-binding source gate; no prior device permission."""
import hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent
B=P.parent
BASE=B/'tree-parent4-canonical-tail-observer-binding-v425'
if str(BASE)not in sys.path:sys.path.insert(0,str(BASE))
PARENT_FREEZE='3f504d53ba5501e3cb20e2a72afff6415ad8a119f61502524bf76c3e8568d4b9'
PARENT_ARCHIVE='0ac3ee9080323b61ee4e8064b7ae0a65ad685defff013b3ae21fccd1e04dcc3d'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready(expected=None):
    expected=expected or json.loads((P/'SOURCE428_RECEIPT.json').read_text())['freeze_sha256']
    if sha(P/'FREEZE428.json')!=expected:raise RuntimeError('ROOT428 exact own source freeze')
    value=json.loads((P/'FREEZE428.json').read_text())
    for path,h in value['files'].items():
        if sha(path)!=h:raise RuntimeError('ROOT428 frozen own file '+path)
    if sha(BASE/'FREEZE425.json')!=PARENT_FREEZE or sha(BASE/'SOURCE425.tar.gz')!=PARENT_ARCHIVE:
        raise RuntimeError('ROOT428 exact425 source/archive, no model proof inherited')
    import check425
    check425.ready(PARENT_FREEZE)
    for name in ('HOST428_METADATA_RESULT.json','HOST428_CONSTRUCTION_RESULT.json','HOST428_CLEAN_STARTUP_RUN.json'):
        result=json.loads((P/name).read_text())
        if not result['status'].startswith('PASS_HOST428')or result['NPU_used']is not False:
            raise RuntimeError('ROOT428 actual source/metadata HOST prerequisite '+name)
    return value
