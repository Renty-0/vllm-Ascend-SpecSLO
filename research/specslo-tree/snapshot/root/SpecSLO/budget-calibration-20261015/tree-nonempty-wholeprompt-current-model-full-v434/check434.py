"""Conditional future source gate; all current-model NN checks remain required."""
import hashlib,json
from pathlib import Path
import sys
P=Path(__file__).resolve().parent;B=P.parent
COMPOSITION=B/'tree-nonempty-current-incumbent-full-proposal-v432'
sys.path.insert(0,str(COMPOSITION))
COMPOSITION_FREEZE='22d68cf40570cbe37df0980401274258fc14c833cf4b49a0c8cf13cd01f1c838'
COMPOSITION_ARCHIVE='47eb4439875b7dec410be917478f58d44dcdf6d009046de5c48d2fba8442e36d'
PARENT=B/'tree-parent4-preflight-component-lifetime-v433'
PARENT_FREEZE='67560bc431ed7fd44aad19c49e060e4049227f128570a43d433dc5e491330a27'
PARENT_ARCHIVE='a75a6cb9553036bbfd56d15a4019baf8bb17fd7f95f73b300f2b70eece7ac730'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready(expected=None):
    expected=expected or json.loads((P/'SOURCE434_RECEIPT.json').read_text())['freeze_sha256']
    if sha(P/'FREEZE434.json')!=expected:raise RuntimeError('ROOT434 exact own source freeze')
    value=json.loads((P/'FREEZE434.json').read_text())
    for path,digest in value['files'].items():
        if sha(path)!=digest:raise RuntimeError('ROOT434 frozen own source '+path)
    import check432
    check432.checked_archive(COMPOSITION,432,COMPOSITION_FREEZE,COMPOSITION_ARCHIVE)
    check432.ready(COMPOSITION_FREEZE)
    for name in('HOST434_CONSTRUCTION_RESULT.json','HOST434_CLEAN_STARTUP_RUN.json'):
        result=json.loads((P/name).read_text())
        if not(result['status'].startswith('PASS_HOST434')and result['NPU_used']is False):
            raise RuntimeError('ROOT434 actual HOST prerequisite '+name)
    return value
