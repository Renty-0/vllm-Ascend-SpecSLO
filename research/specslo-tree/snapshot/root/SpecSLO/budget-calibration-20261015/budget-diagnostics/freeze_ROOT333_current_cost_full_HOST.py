"""Freeze current333 only after explicit HOST and prior clean geometry proof."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

B=Path('/root/SpecSLO/budget-calibration-20261015')
D=B/'budget-diagnostics'
P=B/'tree-M160-C128-F64-global192-full-v333'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def add(files,path,wanted=None):
    path=Path(path);actual=sha(path)
    if wanted is not None and actual!=wanted:raise RuntimeError('Changed source '+str(path))
    if str(path)in files and files[str(path)]!=actual:raise RuntimeError('Conflicting source '+str(path))
    files[str(path)]=actual


def main():
    if(P/'FREEZE333.json').exists():raise RuntimeError('Fresh source freeze only')
    review=json.loads((P/'SOURCE_ONLY333_REVIEW.json').read_text())
    assert review['status']=='SOURCE_ONLY_HOST_PASSED_NOT_FROZEN_NOT_NPU_LAUNCHED'
    for name in review['required_HOST']:
        result=json.loads((P/name).read_text())
        assert result.get('passed')is True or result.get('status','').startswith('HOST_'),name
    audit=json.loads((D/'ROOT329_INDEPENDENT_TARGET_MATH_AND_COST_AUDIT.json').read_text())
    assert audit['status'].startswith('PASS_')and audit['target_cases_per_rank']==532
    assert audit['owned_cleanup_complete']is audit['all4_idle_at_terminal']is True
    sys.path.insert(0,str(B/'tree-FIA160-projection165-MC2-audit-adapter-v332'))
    import guard332
    guard332.PIN='b14c31542a98d6cffb742a1632210ea235a61bf86028ea70d4ca710157fc6eea'
    _,files,_=guard332.prepared()
    files=dict(files)
    for folder,version in [('tree-measured-cost-risk-single-prepare-full-v316',316),
                           ('tree-FIA160-projection165-chained-flush-host-v334',334)]:
        manifest=B/folder/f'FREEZE{version}.json'
        for path,wanted in json.loads(manifest.read_text())['files'].items():add(files,path,wanted)
        add(files,manifest)
    for path in P.iterdir():
        if path.is_file()and path.suffix in('.py','.sh'):
            if path.suffix=='.py':ast.parse(path.read_text())
            else:subprocess.run(['bash','-n',str(path)],check=True,capture_output=True)
            add(files,path)
    for name in [*review['required_HOST'],'SOURCE_ONLY333_REVIEW.json','INTEGRATION_REVIEW333.json',
                 'RETIRED_OBSOLETE333_TEST_COPIES.json','FIXTURE333.json']:
        add(files,P/name)
    for name in ['build_ROOT333_M160_full_HOST.py','integrate_ROOT333_geometry_HOST.py',
                 'freeze_ROOT333_current_cost_full_HOST.py','ROOT329_INDEPENDENT_TARGET_MATH_AND_COST_AUDIT.json',
                 'ROOT331_334_ABI_SOURCE_REVIEW_20261019.json']:
        add(files,D/name)
    original=B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218/FREEZE218.json'
    assert all(files.get(path)==wanted for path,wanted in json.loads(original.read_text())['files'].items())
    assert not any('triton-cache'in Path(path).parts for path in files)
    frozen=dict(schema='ROOT333.original-controls-phased-C128-current-cost-full-qualification.v1',
        frozen_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        files=dict(sorted(files.items())),scope=review,
        current_model_qualification_inherited=False,NPU333_launched=False,Goodput=None,adopted=False)
    manifest=P/'FREEZE333.json'
    manifest.write_text(json.dumps(frozen,indent=2)+'\n');freeze_sha=sha(manifest)
    inventory=dict(files);inventory[str(manifest)]=freeze_sha
    archive=P/'SOURCE333.tar.gz'
    with tarfile.open(archive,'w:gz')as output:
        for path in sorted(inventory):output.add(path,arcname=path.lstrip('/'),recursive=False)
    archive_sha=sha(archive)
    receipt=dict(freeze_sha256=freeze_sha,archive_sha256=archive_sha,members=len(inventory),
                 archive_bytes=archive.stat().st_size,source_freeze=str(manifest),archive=str(archive),
                 HOST_only=True,NPU_qualified=False,Goodput=None)
    (P/'SOURCE333_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
