"""Freeze the supervisor for the complete386 source archive."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tarfile

B=Path('/root/SpecSLO/budget-calibration-20261015')
P=B/'tree-parent4-full-archive-guard-v387'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    outputs=[P/name for name in ['FREEZE387.json','SOURCE387.tar.gz','SOURCE387_RECEIPT.json']]
    assert not any(p.exists()for p in outputs)
    sys.path.insert(0,str(P));import guard387
    _,parent=guard387.checked_archive(guard387.BASE,386,guard387.BASE_FREEZE,guard387.BASE_ARCHIVE)
    host=json.loads((P/'HOST387_CONSTRUCTION_RESULT.json').read_text())
    assert host['status'].startswith('PASS_HOST387')and host['NPU_used'] is False
    selected=[p for p in P.iterdir()if p.suffix in ['.py','.json']]+[Path(__file__).resolve()]
    files={}
    for path in selected:
        assert str(path)not in parent
        if path.suffix=='.py':ast.parse(path.read_text());assert all(s.rstrip()==s for s in path.read_text().splitlines())
        files[str(path)]=sha(path)
    value=dict(schema='ROOT387.complete386-source-supervisor.v1',files=files,
        complete386_freeze_sha256=guard387.BASE_FREEZE,complete386_archive_sha256=guard387.BASE_ARCHIVE,
        original199_runtime_and_actors_unchanged=True,child_runtime_source_gate_and_owned_exit_only=True,
        NPU_used=False,model_qualification_inherited=False,production_permission=False,Raw=None,Goodput=None)
    freeze,archive,receipt=outputs
    with freeze.open('x')as saved:json.dump(value,saved,indent=2);saved.write('\n')
    inventory=dict(files,**{str(freeze):sha(freeze)})
    with archive.open('xb')as out:
        with gzip.GzipFile(fileobj=out,filename='',mode='wb',mtime=0)as compressed:
            with tarfile.open(fileobj=compressed,mode='w|')as saved:
                for name in sorted(inventory):
                    path=Path(name);info=saved.gettarinfo(str(path),arcname=name.lstrip('/'))
                    info.uid=info.gid=0;info.uname=info.gname='root';info.mtime=0
                    with path.open('rb')as source:saved.addfile(info,source)
    result=dict(freeze_sha256=sha(freeze),archive_sha256=sha(archive),archive_members=len(inventory),
        archive_bytes=archive.stat().st_size,NPU_used=False,models_qualified=False,Raw=None,Goodput=None)
    with receipt.open('x')as out:json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
