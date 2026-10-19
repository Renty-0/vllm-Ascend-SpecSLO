"""Freeze a bounded diagnostic, without inheriting neural qualification."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

B=Path('/root/SpecSLO/budget-calibration-20261015')
P=B/'tree-hybrid-native-method-dedent-v384'
PARENT=B/'tree-M80-attention-GEMM120-target-v382'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    outputs=[P/n for n in ('FREEZE384.json','SOURCE384.tar.gz','SOURCE384_RECEIPT.json')]
    assert not any(p.exists()for p in outputs)
    sys.path.insert(0,str(PARENT))
    import guard382
    guard382.EXPECTED_FREEZE='f8b65aa81b78b9f33d3e0e97e30f62841f310160e52c321fd0b8a0243023d6c4'
    guard382.EXPECTED_ARCHIVE='872dde76b7c77e884b304205b27f0fc55706c9b75d3160378a25d62342ff7cea'
    _,parents,_=guard382.prepared()
    host=json.loads((P/'HOST384_CONSTRUCTION_RESULT.json').read_text())
    assert host['status'].startswith('PASS_HOST384')and host['NPU_used'] is False
    selected=[f for f in P.iterdir()if f.suffix in ('.py','.sh','.json')]
    selected.append(Path(__file__).resolve())
    files={}
    for f in selected:
        if f.suffix=='.py':
            ast.parse(f.read_text())
            assert all(line.rstrip()==line for line in f.read_text().splitlines()),f
        elif f.suffix=='.sh':subprocess.run(['/bin/bash','-n',str(f)],check=True)
        assert str(f)not in parents
        files[str(f)]=sha(f)
    modelenv=lambda p:[s for s in p.read_text().splitlines()if s.startswith(('export VLLM_','unset VLLM_'))]
    assert modelenv(P/'startup384.sh')==modelenv(PARENT/'startup382.sh')
    value=dict(schema='ROOT384.additive-native-method-source-dedent.v1',files=files,
        parent_freeze_sha256=guard382.EXPECTED_FREEZE,parent_archive_sha256=guard382.EXPECTED_ARCHIVE,
        actual_target_cases_each_target=693,independent_prefills_each_target=216,
        fresh_cases='231 each mixed/mid/short banks, original227 plus4 max80 branch cases',
        physical_families=[80,120],Q=5,S=[16,24],
        preserves_original_mc2_function=True,original_untimed_numerical_stage_fences_retained=True,
        diagnostic_does_not_grant_neural_or_performance_qualification=True,
        NPU_used=False,Raw=None,Goodput=None)
    freeze,archive,receipt=outputs
    with freeze.open('x')as out:json.dump(value,out,indent=2);out.write('\n')
    inventory=dict(files,**{str(freeze):sha(freeze)})
    with archive.open('xb')as output:
        with gzip.GzipFile(fileobj=output,filename='',mode='wb',mtime=0)as compressed:
            with tarfile.open(fileobj=compressed,mode='w|')as saved:
                for name in sorted(inventory):
                    path=Path(name);info=saved.gettarinfo(str(path),arcname=name.lstrip('/'))
                    info.uid=info.gid=0;info.uname=info.gname='root';info.mtime=0
                    with path.open('rb')as source:saved.addfile(info,source)
    result=dict(freeze_sha256=sha(freeze),archive_sha256=sha(archive),own_archive_members=len(inventory),
        own_archive_bytes=archive.stat().st_size,parent_source_members=len(parents),NPU_used=False,
        model_or_performance_qualification_inherited=False,Raw=None,Goodput=None)
    with receipt.open('x')as out:json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
