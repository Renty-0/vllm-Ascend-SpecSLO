"""Review stopped196 source and retained HOST evidence, then prepare source only."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

B=Path('/root/SpecSLO/budget-calibration-20261015')
D=B/'budget-diagnostics'
P=B/'tree-causal80-fused-bonus-devicefix-full-v196'
G=D/'v196-root-execution'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pin(path):return dict(path=str(path),sha256=sha(path))
def save(path,value):
    with path.open('x')as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')

ready=D/'V196_DEVICE_DESCRIPTOR_FIX_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json'
peer=D/'V196_INDEPENDENT_SOURCE_PEER_REVIEW.json'
assert sha(ready)=='b780cb60f44afe809968cc52fdb4254432c980ba4c3e1d891cc226a9fc0520be'
r=json.loads(ready.read_text());pr=json.loads(peer.read_text())
assert pr['status']=='passed_source_review'and pr['candidate']==str(P)
assert r['source_stopped_writing']is True and r['source_frozen']is False
assert len(r['flat_review_inputs_before_peer'])==599
for path,digest in r['flat_review_inputs_before_peer'].items():assert sha(path)==digest,path
projection=json.loads((P/'PROJECTION196.json').read_text())
assert len(projection['files'])==24
for name,row in projection['files'].items():
    original=Path(row['parent']).read_text()
    assert sha(row['parent'])==row['parent_sha256']
    value=original
    for change in row['changes']:
        assert value.count(change['old'])==change['count'],(name,change['old'])
        value=value.replace(change['old'],change['new'])
    path=G/name.split('/',1)[1]if name.startswith('guard/')else P/name
    assert path.read_text()==value and sha(path)==row['sha256'],name
    for change in reversed(row['changes']):
        assert value.count(change['new'])==change['count'],(name,change['new'])
        value=value.replace(change['new'],change['old'])
    assert value==original,name
for name,digest in projection['new_sources'].items():assert sha(P/name)==digest
for path in r['files']:
    if Path(path).suffix=='.py':ast.parse(Path(path).read_text(),filename=path)
ast.parse((G/'guard196.py').read_text(),filename=str(G/'guard196.py'))
for name in ('launch196.sh','startup196.sh'):
    subprocess.run(['bash','-n',str(G/name)],check=True)
spec=importlib.util.spec_from_file_location('_ROOT196_stopped_source_guard',G/'guard196.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
assert guard.P==P and guard.CPU==P/'CPU_RESULT196.json'
state=guard.source_state(archive=True);guards=guard.guard_inputs()
assert len(state['local'])==25 and len(state['external'])==379 and len(state['evidence'])==190 and len(guards)==4
assert sha(guard.CPU)=='279c818ceeac982c6b0623458af3989960cde0de1f9bf059ca522e021bb59a06'
inputs=guard.merge(state['local'],state['external'],state['evidence'],guards,
    {str(guard.CPU):sha(guard.CPU),str(peer):sha(peer)})
assert len(inputs)==600 and 'torch'not in sys.modules
review=D/'ROOT_V196_DEVICE_DESCRIPTOR_FIX_SOURCE_AND_GUARD_REVIEW.json'
save(review,dict(schema='ROOT.V196.device-descriptor-source-review.v1',
    UTC=datetime.now(timezone.utc).isoformat(),status='passed_source_review',candidate=str(P),
    reviewed_inputs=inputs,ready=pin(ready),peer=pin(peer),
    parent_archives_verified_in_place=state['parent_archive'],whole_text_projections_verified=24,
    ROOT_source_reviewed=True,HOST_tests_rerun=False,NPU_used=False,Torch_imported=False,model_run=False,
    reviewed_semantics=[
        'Only production algorithm changes are actual indexed tensor/model/local-rank/current-device binding and all16 BONUS mask allocation/check on that actual device.',
        'Original Native engine.device is an indexless descriptor; equality is checked between indexed packed/target/model tensors, with explicit descriptor index validated when present.',
        'Original long dtype, physical M80, sixteen complete B4 windows, current proposal identity, staging lifetime, operator source identity and all-rank error packet remain mandatory.',
        'Actual own queue and consume source plus original BONUS kernel body ran only on explicit integer carriers;15 negative device/dtype/shape/mask cases and old193 failure regression are retained.',
        'Sole source parent193 and production ancestor173 are explicit; prior193 component failure gives no device qualification.',
        'Current original B4 and M80 full hidden/head/KV/verdict/commit/ownnext qualification, trueB5/heterogeneous fallback, warm fixedpoint/twozero/seal/full463 remain unchanged.',
        'Original active32, normal96/eager96/global192, all463 arrivals/output/clocks, page pool544/storage527 and SLO/coalescing settings are preserved.',
        'Original self-owned lock, baseline pre/post, owned cleanup and frozen177 scoped mutable-tmp handling remain; all generated CPU/freeze/archive/attempt paths bind196.',
        'Every actual failed and successful HOST attempt, source snapshot, superseded successful CPU, metadata and cold-guard failure is retained; no old HOST suite or model was rerun.'
    ],limitations=[
        'Source/HOST preparation does not prove current NPU operator/model execution or whole463 own-AR equality.',
        'This repairs a pre-performance ABI failure; no Goodput benefit is established.',
        'Known canonical output differences are not resolved by the device descriptor fix.',
        'Old binary archives are verified in place; selected-source archive is not a complete environment import closure.'
    ],formal_Goodput=False,adopted=False))
command=[sys.executable,'-B',str(G/'guard196.py'),'--prepare-only','--root-review',str(review),
    '--expected-root-review-sha256',sha(review),'--peer-review',str(peer),sha(peer)]
result=subprocess.run(command,capture_output=True,text=True)
save(D/'ROOT_V196_GUARD_PREPARE_STDOUT_AND_STDERR.json',dict(command=command,
    returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
assert result.returncode==0,result.stderr
f,sources,prepared=guard.prepared()
assert len(sources)==prepared['archive']['members']==602 and 'torch'not in sys.modules
output=D/'ROOT_V196_PREPARED_SOURCE_HOST_AUDIT.json'
save(output,dict(schema='ROOT.V196.prepared-source-HOST-audit.v1',
    status='passed_source_frozen_unrun',candidate=str(P),review=pin(review),peer=pin(peer),
    freeze=pin(guard.FREEZE),archive=pin(guard.ARCHIVE),actual_source_members=len(sources),
    actual_archive_members=prepared['archive']['members'],prepared=prepared,
    NPU_used=False,Torch_imported=False,models_launched=False,formal_Goodput=False,adopted=False))
print(json.dumps(dict(status='passed_source_frozen_unrun',record=pin(output),
    freeze=pin(guard.FREEZE),archive=pin(guard.ARCHIVE),members=len(sources))))
