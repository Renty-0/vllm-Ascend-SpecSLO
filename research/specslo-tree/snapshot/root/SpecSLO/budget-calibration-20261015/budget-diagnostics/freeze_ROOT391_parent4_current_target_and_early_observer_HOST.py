"""Freeze finite current M120 target and pre-capture observer repair, no NN."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tarfile
B=Path('/root/SpecSLO/budget-calibration-20261015')
P=B/'tree-parent4-current-M120-target-protocol-v391'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    outputs=[P/name for name in ('FREEZE391.json','SOURCE391.tar.gz','SOURCE391_RECEIPT.json')]
    assert not any(path.exists()for path in outputs)
    sys.path.insert(0,str(P));import guard391
    guard391.parent.EXPECTED_FREEZE=guard391.PARENT_FREEZE
    guard391.parent.EXPECTED_ARCHIVE=guard391.PARENT_ARCHIVE
    _,previous,_=guard391.parent.prepared()
    receipt=guard391.parent.DONOR/'receipt-parent4-child4-shared4-root-v389-bounded-v3.json'
    terminal=json.loads(receipt.read_text())
    assert terminal['status']=='failed'and terminal['sources_unchanged']and terminal['owned_remaining']==[]and terminal['devices_after']['all_idle']
    actual=guard391.parent.DONOR/'actual'/terminal['tag']
    for rank in range(4):
        error=json.loads((actual/f'rank{rank}-last-error.json').read_text())
        assert 'Original frontier captured/eager hidden references unavailable'in error['error']
    tests=[P/'HOST391_RESULT.json',P/'HOST391_CONSTRUCTION_RESULT.json',P/'HOST391_PHYSICAL_RESULT.json',
        B/'tree-parent4-best376-target-binding-host-v390/HOST390_RESULT.json',
        B/'tree-parent4-frontier-observer-before-capture-host-v392/HOST392_RESULT.json']
    for path in tests:
        host=json.loads(path.read_text());assert host['status'].startswith('PASS_HOST')and host['NPU_used']is False
    selected=[Path(__file__).resolve()]
    for folder in (P,B/'tree-parent4-best376-target-binding-host-v390',B/'tree-parent4-frontier-observer-before-capture-host-v392'):
        selected.extend(path for path in folder.iterdir()if path.suffix in ('.py','.sh','.json'))
    sources={}
    for path in selected:
        assert str(path)not in previous
        if path.suffix=='.py':ast.parse(path.read_text())
        if path.suffix in ('.py','.sh'):assert all(line.rstrip()==line for line in path.read_text().splitlines())
        sources[str(path)]=sha(path)
    value=dict(schema='ROOT391.finite-current-M120-parent4-with-early-observer.v1',
        parent389_freeze_sha256=guard391.PARENT_FREEZE,parent389_archive_sha256=guard391.PARENT_ARCHIVE,
        files=sources,preceding389_receipt_sha256=sha(receipt),preceding389_expected_failure='missing constructor-time frontier observer references',
        fresh_original173_baseline_required=True,finite_normal_N=16,parent_B=4,child_B=4,E=[1,2,3,4],
        target='actual376 M120 Q5 S24 ceil-visible-KV/512',draft='original199 fused prelude, not whole best376 draft',
        observer='same read-only observer registered before constructor captures; no tensor ops or Native flags changed',
        original227_and681_claimed=False,production_permission=False,full463=False,Raw=None,Goodput=None)
    freeze,archive,receipt=outputs
    with freeze.open('x')as out:json.dump(value,out,indent=2);out.write('\n')
    inventory=dict(sources,**{str(freeze):sha(freeze)})
    with archive.open('xb')as out:
        with gzip.GzipFile(fileobj=out,filename='',mode='wb',mtime=0)as compressed:
            with tarfile.open(fileobj=compressed,mode='w|')as saved:
                for name in sorted(inventory):
                    path=Path(name);info=saved.gettarinfo(str(path),arcname=name.lstrip('/'))
                    info.uid=info.gid=0;info.uname=info.gname='root';info.mtime=0
                    with path.open('rb')as content:saved.addfile(info,content)
    result=dict(freeze_sha256=sha(freeze),archive_sha256=sha(archive),archive_members=len(inventory),
        archive_bytes=archive.stat().st_size,NPU_used=False,models_qualified=False,Raw=None,Goodput=None)
    with receipt.open('x')as out:json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
