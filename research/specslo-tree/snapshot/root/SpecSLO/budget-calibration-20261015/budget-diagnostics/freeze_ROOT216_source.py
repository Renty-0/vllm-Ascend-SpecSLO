"""Actual final HOST checks, source closure and regular archive; no device launch."""
import ast,datetime,difflib,hashlib,json,os,shutil,subprocess,tarfile,time
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-cost-aware-N9to16-with-own-FIA-AR-full-v216'
DONOR=B/'tree-cost-aware-mixed16-with-own-FIA-AR-full-v215'
assert not (P/'FREEZE216.json').exists()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
passed_attempts=[q.parent for q in (P/'HOST-attempts').glob('*/RESULT.json')if json.loads(q.read_text()).get('exit_code')==0 and json.loads(q.read_text()).get('tests_run')==15 and all((q.parent/f.name).exists()and sha(q.parent/f.name)==sha(f)for f in P.iterdir()if f.suffix in('.py','.sh'))]
assert passed_attempts,'actual current-source final15 HOST attempt required; do not invent a pass'
attempt=max(passed_attempts,key=lambda p:p.stat().st_mtime)
result=json.loads((attempt/'RESULT.json').read_text())
assert result['status']=='passed'and result['errors']==result['failures']==0 and result['NPU_used']is result['Torch_imported']is False
projection=json.loads((P/'PROJECTION216.json').read_text())
for r in projection['projection']:
    old=Path(r['source']).read_text().replace(str(DONOR),str(P)).replace('215','216');new=Path(r['new']).read_text()
    r['new_sha256']=sha(r['new']);r['patch_after_namespace_projection']=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True)))
for p in P.glob('*.py'):ast.parse(p.read_text())
projection.update(new_reference_module_sha256=sha(P/'reference216.py'),independent_HOST_auditor_sha256=sha(P/'audit_reference216.py'),final_HOST_attempt=str(attempt))
save(P/'PROJECTION216.json',projection)
donor=json.loads((DONOR/'FREEZE215.json').read_text());files=dict(donor['files']);files[str(DONOR/'FREEZE215.json')]=sha(DONOR/'FREEZE215.json')
for p in P.rglob('*'):
    if p.is_file()and '__pycache__'not in p.parts:files[str(p)]=sha(p)
for p in (D/'prepare_ROOT216_source.py',Path(__file__)):files[str(p)]=sha(p)
review=dict(schema='ROOT216.postfull-independent-FIA-AR-source-review.v1',UTC=now,status='passed_source_review',
    source_peer_review=False,HOST_tests=result,actual_HOST_attempt=str(attempt),reviewed_inputs=files.copy(),
    performance_source_delta_from215='mixed3/4 budget cost repair extends16 to current-qualified9..16; original mapper cost table prepared once outside model work',
    reference='fresh same original prefill cohorts, independent prompt-only cache; own target-only255steps per owner; fixed80/S16 causal segments consume first query only; future zeros/151643 canary at0/127/254; all3 target output hashes checked; original nonfinite vote; original21 counters all zero',
    independent_audit='stdlib reconstructs every reference output from80heads, metadata samples/page tables, whole463 exact comparison including failures',
    same_backend_AR_must_be_measured=True,cross_PA_or_official_AR_equivalence_claimed=False,
    full_candidate_native_qualification_repeated=True,NPU_started=False,Goodput=None,adopted=False)
rp=P/'ROOT_SOURCE_REVIEW216.json';save(rp,review);files[str(rp)]=sha(rp)
freeze=dict(schema='ROOT216.cost-aware-N9to16-plus-postfull-own-AR-source-freeze.v1',UTC=now,files=files,
    source_review=dict(path=str(rp),sha256=sha(rp)),original209_source_only_dependency=donor['original209_source_only_dependency'],
    source_donor215=dict(path=str(DONOR),freeze_sha256=sha(DONOR/'FREEZE215.json'),device_qualification_inherited=False),
    HOST_tests=15,source_peer_review=False,SLO_AWARE_ADMISSION=1,budget_grants_may_change=True,
    performance_source_same214=False,extends_current_qualified_N9to16=True,reference_only_after_performance=True,full463_required=True,
    own_AR_device_qualified=False,Goodput=None,adopted=False)
fp=P/'FREEZE216.json';save(fp,freeze);archived=dict(files);archived[str(fp)]=sha(fp)
ar=P/'SOURCE216.tar.gz'
with tarfile.open(ar,'w:gz')as saved:
    for p in sorted(archived):saved.add(p,arcname=p.lstrip('/'),recursive=False)
with tarfile.open(ar,'r:gz')as saved:
    members=saved.getmembers();assert len(members)==len(archived)
    for m in members:assert m.isfile()and hashlib.sha256(saved.extractfile(m).read()).hexdigest()==archived['/'+m.name]
row=dict(schema='ROOT216.frozen-unrun-source.v1',UTC=now,directory=str(P),freeze_sha256=sha(fp),archive_sha256=sha(ar),
    archive_members=len(archived),archive_bytes=ar.stat().st_size,HOST_tests=15,source_peer_review=False,
    NPU_started=False,performance_source_same214=False,extends_current_qualified_N9to16=True,own_AR_device_qualified=False,Goodput=None,adopted=False)
save(D/'ROOT_V216_SOURCE_FROZEN_UNRUN.json',row);print(json.dumps(row))
