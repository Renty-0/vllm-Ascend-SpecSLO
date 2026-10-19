"""Additive source freeze; no model launch or inherited neural qualification."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

B=Path('/root/SpecSLO/budget-calibration-20261015')
P=B/'tree-adaptive-M80-M120-target-math-cost-v378'
CORE=B/'tree-adaptive-M80-M120-target-host-v377'
PARENT=B/'tree-Q5-bucket-completion-and-paired-mode-cost-v361'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    outputs=[P/n for n in ('FREEZE378.json','SOURCE378.tar.gz','SOURCE378_RECEIPT.json')]
    assert not any(p.exists()for p in outputs)
    sys.path.insert(0,str(PARENT))
    import guard361
    guard361.EXPECTED_FREEZE='a1dbe700b6fa857631be31878df58fd7b129de38491ae027a8823799ce81bf3f'
    guard361.EXPECTED_ARCHIVE='3e96cc84744c376ef5855bc57cda8627323c4ba841bc3843420c906d0618c20a'
    _,parents,_=guard361.prepared()
    for directory,names in ((P,('HOST378_CONSTRUCTION_RESULT.json','HOST378_COST_AND_AUDIT_RESULT.json')),
                            (CORE,('HOST377_RESULT.json','HOST377_DISPATCH_RESULT.json'))):
        for name in names:
            v=json.loads((directory/name).read_text())
            assert v['status'].startswith('PASS_HOST')and v['NPU_used'] is False
    selected=[f for directory in (P,CORE)for f in directory.iterdir()if f.suffix in ('.py','.sh','.json')]
    selected.extend([Path(__file__).resolve(),
        B/'tree-best257-uniform-KV-buckets-full-v376/prefixes376.py',
        B/'tree-static-q5-M120-SLO-first-full463-own-AR-v257/FREEZE257.json',
        B/'tree-static-q5-M120-SLO-first-full463-own-AR-v257/audit_static_calls257.py'])
    files={}
    for f in selected:
        if f.suffix=='.py':
            ast.parse(f.read_text())
            assert all(line.rstrip()==line for line in f.read_text().splitlines()),f
        elif f.suffix=='.sh':subprocess.run(['/bin/bash','-n',str(f)],check=True)
        if str(f)in parents:assert sha(f)==parents[str(f)]
        else:files[str(f)]=sha(f)
    modelenv=lambda p:[s for s in p.read_text().splitlines()if s.startswith(('export VLLM_','unset VLLM_'))]
    assert modelenv(P/'startup378.sh')==modelenv(PARENT/'startup361.sh')
    value=dict(schema='ROOT378.additive-adaptive-physical-target-source.v1',files=files,
        parent_freeze_sha256=guard361.EXPECTED_FREEZE,parent_archive_sha256=guard361.EXPECTED_ARCHIVE,
        actual_target_cases_each=693,independent_prefills_each_target=216,
        original_real_budget_domain=[1,2,3,4,5],original_target_KV_bucket_policy_retained=True,
        independent_FULL120_reference_required=True,physical_families=[80,120],Q=5,S=[16,24],
        paired_modes=['fixed120','adaptive80_120'],six_arms20_pairs=True,
        model_or_performance_qualification_inherited=False,NPU_used=False,Raw=None,Goodput=None)
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
