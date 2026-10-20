"""Freeze exact original199/173 source dependencies and own guarded runtime."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

B=Path('/root/SpecSLO/budget-calibration-20261015')
P=B/'tree-parent4-child4-original-bounded-v386'
OLD=B/'tree-bounded-parent4-shared4-eager-diagnostic-v199'
PARENT=B/'tree-real-M120-deferred-chain-profile-v385'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    outputs=[P/n for n in ['FREEZE386.json','SOURCE386.tar.gz','SOURCE386_RECEIPT.json']]
    assert not any(p.exists()for p in outputs)
    host=json.loads((P/'HOST386_CONSTRUCTION_RESULT.json').read_text())
    assert host['status'].startswith('PASS_HOST386')and host['NPU_used'] is False
    assert host['actual_original199_require_ready_passed']and host['actual_original_prepare_install_and_origins_passed']
    sys.path.insert(0,str(PARENT));import guard385
    guard385.EXPECTED_FREEZE='e62fc5f7740fa493906a7889a8af1baf2fd118696e8a271b5b5c493a7f892a1f'
    guard385.EXPECTED_ARCHIVE='0f310f5d96ab4efabba1529dc70c40b3e33206c4e0aab8b204d113e2f3dfb64d'
    _,files,_=guard385.prepared()
    files=dict(files)
    def add(path,expected=None):
        actual=sha(path)
        if expected is not None:assert actual==expected,str(path)
        if str(path)in files:assert files[str(path)]==actual,str(path)
        else:files[str(path)]=actual
    for folder,number in [(OLD,199),(B/'tree-causal80-real16-bankfix-full-v173',173)]:
        manifest=folder/f'FROZEN_SOURCE{number}.json';value=json.loads(manifest.read_text())
        for key in ['files','external_files','retained_HOST_evidence','execution_guard_inputs','peer_reviews']:
            for name,digest in value[key].items():add(Path(name),digest)
        review=value['root_review'];add(Path(review['path']),review['sha256'])
        add(manifest);add(folder/f'CPU_RESULT{number}.json')
    for filename,digest in json.loads((OLD/'INPUTS199.json').read_text())['files'].items():add(Path(filename),digest)
    selected=[f for f in P.iterdir()if f.suffix in ['.py','.sh','.json']]+[Path(__file__).resolve()]
    for path in selected:
        if path.suffix=='.py':
            ast.parse(path.read_text());assert all(s.rstrip()==s for s in path.read_text().splitlines()),path
        if path.suffix=='.sh':subprocess.run(['/bin/bash','-n',str(path)],check=True)
        add(path)
    env=lambda path:[s for s in path.read_text().splitlines()if s.startswith(('export VLLM_','unset VLLM_'))]
    assert env(P/'startup386.sh')==env(PARENT/'startup385.sh')
    for name,digest in files.items():assert sha(name)==digest,name
    value=dict(schema='ROOT386.original199-parent4-child4-source.v1',files=files,
        original199_freeze_sha256=sha(OLD/'FROZEN_SOURCE199.json'),
        original199_CPU_sha256=sha(OLD/'CPU_RESULT199.json'),
        original173_baseline_fresh_required=True,original_parent_and_child_budget=4,
        fresh_bounded_cases=5,actual_passes=3,original_two_zero_traces_required=True,
        original_numerical_and_protocol_actors_unchanged=True,own_worker_delta='source gate and owned exit only',
        original_arrivals_or_performance_not_measured=True,production_permission=False,
        Native_flags_assigned=False,NPU_used=False,model_qualification_inherited=False,Raw=None,Goodput=None)
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
    result=dict(freeze_sha256=sha(freeze),archive_sha256=sha(archive),archive_members=len(inventory),
        archive_bytes=archive.stat().st_size,NPU_used=False,model_qualified=False,Raw=None,Goodput=None)
    with receipt.open('x')as out:json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
