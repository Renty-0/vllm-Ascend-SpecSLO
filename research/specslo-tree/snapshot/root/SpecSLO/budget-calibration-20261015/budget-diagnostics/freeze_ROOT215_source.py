"""Actual final HOST checks, source closure and regular archive; no device launch."""
import ast,datetime,difflib,hashlib,json,os,shutil,subprocess,tarfile,time
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-cost-aware-mixed16-with-own-FIA-AR-full-v215'
DONOR=B/'tree-cost-aware-mixed16-budget-full-v214'
assert not (P/'FREEZE215.json').exists()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
attempt=P/'HOST-attempts'/str(time.time_ns());attempt.mkdir()
for p in P.iterdir():
    if p.suffix in('.py','.sh'):shutil.copyfile(p,attempt/p.name)
env=dict(os.environ,TORCH_DEVICE_BACKEND_AUTOLOAD='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
with (attempt/'stdout.log').open('w')as out,(attempt/'stderr.log').open('w')as err:
    child=subprocess.run(['/root/miniconda3/envs/vllm-hust-dev/bin/python3.11','-B',str(P/'host215.py')],env=env,stdout=out,stderr=err,timeout=120)
line=(attempt/'stdout.log').read_text().splitlines()[-1];result=json.loads(line)
save(attempt/'RESULT.json',dict(result,exit_code=child.returncode,actual_stdout_sha256=sha(attempt/'stdout.log'),actual_stderr_sha256=sha(attempt/'stderr.log')))
assert child.returncode==0 and result==dict(status='passed',tests_run=14,failures=0,errors=0,Torch_imported=False,NPU_used=False),result
projection=json.loads((P/'PROJECTION215.json').read_text())
for r in projection['projection']:
    old=Path(r['source']).read_text().replace(str(DONOR),str(P)).replace('214','215');new=Path(r['new']).read_text()
    r['new_sha256']=sha(r['new']);r['patch_after_namespace_projection']=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True)))
for p in P.glob('*.py'):ast.parse(p.read_text())
projection.update(new_reference_module_sha256=sha(P/'reference215.py'),independent_HOST_auditor_sha256=sha(P/'audit_reference215.py'),final_HOST_attempt=str(attempt))
save(P/'PROJECTION215.json',projection)
donor=json.loads((DONOR/'FREEZE214.json').read_text());files=dict(donor['files']);files[str(DONOR/'FREEZE214.json')]=sha(DONOR/'FREEZE214.json')
for p in P.rglob('*'):
    if p.is_file()and '__pycache__'not in p.parts:files[str(p)]=sha(p)
for p in (D/'prepare_ROOT215_AR_source.py',Path(__file__)):files[str(p)]=sha(p)
review=dict(schema='ROOT215.postfull-independent-FIA-AR-source-review.v1',UTC=now,status='passed_source_review',
    source_peer_review=False,HOST_tests=result,actual_HOST_attempt=str(attempt),reviewed_inputs=files.copy(),
    performance_source_delta_from214='only namespace/provenance; independent reference called after run_windows has returned',
    reference='fresh same original prefill cohorts, independent prompt-only cache; own target-only255steps per owner; fixed80/S16 causal segments consume first query only; future zeros/151643 canary at0/127/254; all3 target output hashes checked; original nonfinite vote; original21 counters all zero',
    independent_audit='stdlib reconstructs every reference output from80heads, metadata samples/page tables, whole463 exact comparison including failures',
    same_backend_AR_must_be_measured=True,cross_PA_or_official_AR_equivalence_claimed=False,
    full_candidate_native_qualification_repeated=True,NPU_started=False,Goodput=None,adopted=False)
rp=P/'ROOT_SOURCE_REVIEW215.json';save(rp,review);files[str(rp)]=sha(rp)
freeze=dict(schema='ROOT215.cost-aware-plus-postfull-own-AR-source-freeze.v1',UTC=now,files=files,
    source_review=dict(path=str(rp),sha256=sha(rp)),original209_source_only_dependency=donor['original209_source_only_dependency'],
    source_donor214=dict(path=str(DONOR),freeze_sha256=sha(DONOR/'FREEZE214.json'),device_qualification_inherited=False),
    HOST_tests=14,source_peer_review=False,SLO_AWARE_ADMISSION=1,budget_grants_may_change=True,
    performance_source_same214=True,reference_only_after_performance=True,full463_required=True,
    own_AR_device_qualified=False,Goodput=None,adopted=False)
fp=P/'FREEZE215.json';save(fp,freeze);archived=dict(files);archived[str(fp)]=sha(fp)
ar=P/'SOURCE215.tar.gz'
with tarfile.open(ar,'w:gz')as saved:
    for p in sorted(archived):saved.add(p,arcname=p.lstrip('/'),recursive=False)
with tarfile.open(ar,'r:gz')as saved:
    members=saved.getmembers();assert len(members)==len(archived)
    for m in members:assert m.isfile()and hashlib.sha256(saved.extractfile(m).read()).hexdigest()==archived['/'+m.name]
row=dict(schema='ROOT215.frozen-unrun-source.v1',UTC=now,directory=str(P),freeze_sha256=sha(fp),archive_sha256=sha(ar),
    archive_members=len(archived),archive_bytes=ar.stat().st_size,HOST_tests=14,source_peer_review=False,
    NPU_started=False,performance_source_same214=True,own_AR_device_qualified=False,Goodput=None,adopted=False)
save(D/'ROOT_V215_SOURCE_FROZEN_UNRUN.json',row);print(json.dumps(row))
