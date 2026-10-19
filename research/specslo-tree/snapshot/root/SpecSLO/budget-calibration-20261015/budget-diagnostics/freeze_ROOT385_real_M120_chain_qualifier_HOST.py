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
P=B/'tree-real-M120-deferred-chain-profile-v385'
PARENT=B/'tree-hybrid-native-method-dedent-v384'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    outputs=[P/n for n in ('FREEZE385.json','SOURCE385.tar.gz','SOURCE385_RECEIPT.json')]
    assert not any(p.exists()for p in outputs)
    sys.path.insert(0,str(PARENT))
    import guard384
    guard384.EXPECTED_FREEZE='a647343b81fac81459d6829644a59ab12425955647b11a78c15a4978871a1e5b'
    guard384.EXPECTED_ARCHIVE='2309d02384a0f76202cfc3c9de26e8d041990fc224d03e5a192cb53e2126767f'
    _,parents,_=guard384.prepared()
    host=json.loads((P/'HOST385_CONSTRUCTION_RESULT.json').read_text())
    assert host['status'].startswith('PASS_HOST385')and host['NPU_used'] is False
    selected=[f for f in P.iterdir()if f.suffix in ('.py','.sh','.json')]
    selected.extend([Path(__file__).resolve(),
        Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/examples/check_specslo_mc2_deferred_chain.py'),
        Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/examples/specslo_mc2_deferred_chain_evidence.py')])
    files={}
    for f in selected:
        if f.suffix=='.py':
            ast.parse(f.read_text())
            assert all(line.rstrip()==line for line in f.read_text().splitlines()),f
        elif f.suffix=='.sh':subprocess.run(['/bin/bash','-n',str(f)],check=True)
        if str(f)in parents:assert sha(f)==parents[str(f)]
        else:files[str(f)]=sha(f)
    modelenv=lambda p:[s for s in p.read_text().splitlines()if s.startswith(('export VLLM_','unset VLLM_'))]
    assert modelenv(P/'startup385.sh')==modelenv(PARENT/'startup384.sh')
    value=dict(schema='ROOT385.additive-real120-deferred-chain-original-qualifier.v1',files=files,
        parent_freeze_sha256=guard384.EXPECTED_FREEZE,parent_archive_sha256=guard384.EXPECTED_ARCHIVE,
        actual_operator_cases_each_target=1,independent_prefills_each_target=72,
        current_real_payload_case='N24-all4-anchor0',
        physical_families=[120],Q=5,S=[24],epilogues_per_chain=128,
        original_measurement_and_numerical_functions_unchanged=True,new_profile_installed=False,
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
