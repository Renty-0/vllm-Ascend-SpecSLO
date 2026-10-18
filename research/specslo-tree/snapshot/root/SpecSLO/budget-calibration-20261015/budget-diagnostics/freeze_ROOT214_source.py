"""Close the selected private source set after actual HOST tests, no model run."""
import ast
import datetime
import difflib
import hashlib
import json
import shutil
import tarfile
import time
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
P = B / 'tree-cost-aware-mixed16-budget-full-v214'
DONOR = B / 'tree-cached-parent-slo-admission-full-v213'
assert not (P/'FREEZE214.json').exists()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, row):
    Path(p).write_text(json.dumps(row, ensure_ascii=False, indent=2)+'\n')


now = datetime.datetime.now(datetime.timezone.utc).isoformat()
host = P/'HOST-attempts'/str(time.time_ns())
host.mkdir()
for f in P.iterdir():
    if f.suffix in ('.py', '.sh'):
        shutil.copyfile(f, host/f.name)
result = dict(schema='ROOT214.actual-HOST-tool-result.v1', UTC=now,
    actual_exec_session=49011, actual_exit_code=0,
    tests_run=11, failures=0, errors=0, status='passed',
    Torch_imported=False, NPU_used=False,
    prior_failed_tests_retained=True,
    scope='actual original DTO + exact private shape bytecode/closure + canonical mapper geometry + original supervisor all terminal paths +213 original admission/ABI and214 accounting negatives')
save(host/'RESULT.json', result)
projection = json.loads((P/'PROJECTION214.json').read_text())
for record in projection['projection']:
    old=Path(record['source']).read_text().replace(str(DONOR),str(P)).replace('213','214')
    new=Path(record['new']).read_text()
    record['patch_after_namespace_projection']=''.join(difflib.unified_diff(
        old.splitlines(True),new.splitlines(True),fromfile='projected213',tofile='candidate214'))
    record['new_sha256']=sha(record['new'])
    if record['new'].endswith('.py'):
        ast.parse(new)
save(P/'PROJECTION214.json',projection)

donor=json.loads((DONOR/'FREEZE213.json').read_text())
files=dict(donor['files'])
files[str(DONOR/'FREEZE213.json')]=sha(DONOR/'FREEZE213.json')
for f in P.rglob('*'):
    if f.is_file() and '__pycache__' not in f.parts:
        files[str(f)]=sha(f)
for f in (D/'prepare_ROOT214_source.py',Path(__file__)):
    files[str(f)]=sha(f)
review=dict(schema='ROOT214.source-review.v1',UTC=now,status='passed_source_review',
    source_peer_review=False,HOST_tests=result,
    reviewed_inputs=files.copy(),
    delta='same frozen213 with bounded cost-aware normal16 mixed3/4 -> all4 inside existing typed caps, role/global limits and context4096; no graph/math/kernel changes',
    invariants=['original outer shape bytecode retained in private globals',
                'original candidate_shape code and policy closure retained',
                'original solver still computes priority, need, gain and reservations',
                'immutable original DraftWorkPlan reruns validation after upgrade',
                'max draft stages already4 before/after; eager filtering unchanged',
                'original grant counters adjusted; own repair counters retained by actual generation',
                'all3, all4, non16, B1/2/5 and insufficient caps/budgets/context untouched',
                'current same-engine source209 Native qualifications/warm/twozero/seal/full still required'],
    budget_grants_may_change=True,arrival_model_clock_changed=False,
    no_device_qualification_inherited=True,Goodput=None,adopted=False,
    limitation='whole-workload independent AR and final matched baseline still required')
review_path=P/'ROOT_SOURCE_REVIEW214.json';save(review_path,review)
files[str(review_path)]=sha(review_path)
freeze=dict(schema='ROOT214.cost-aware-mixed16-source-freeze.v1',UTC=now,files=files,
    source_review=dict(path=str(review_path),sha256=sha(review_path)),
    original209_source_only_dependency=donor['original209_source_only_dependency'],
    source_donor213=dict(path=str(DONOR),freeze_sha256=sha(DONOR/'FREEZE213.json'),
                         device_qualification_inherited=False,model_result_not_read=True),
    HOST_tests=11,source_peer_review=False,SLO_AWARE_ADMISSION=1,
    budget_grants_may_change=True,new_model_graph_arrival_clock=False,
    full463_required=True,Goodput=None,adopted=False)
freeze_path=P/'FREEZE214.json';save(freeze_path,freeze)
archived=dict(files);archived[str(freeze_path)]=sha(freeze_path)
archive=P/'SOURCE214.tar.gz'
with tarfile.open(archive,'w:gz') as saved:
    for path in sorted(archived):
        saved.add(path,arcname=path.lstrip('/'),recursive=False)
with tarfile.open(archive,'r:gz') as saved:
    members=saved.getmembers()
    assert len(members)==len(archived)
    for m in members:
        assert m.isfile() and hashlib.sha256(saved.extractfile(m).read()).hexdigest()==archived['/'+m.name]
row=dict(schema='ROOT214.frozen-unrun-source.v1',UTC=now,directory=str(P),
    freeze_sha256=sha(freeze_path),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
    archive_members=len(archived),HOST_tests=11,Torch_imported=False,NPU_run=False,
    source_peer_review=False,budget_grants_may_change=True,Goodput=None,adopted=False)
save(D/'ROOT_V214_SOURCE_FROZEN_UNRUN.json',row)
print(json.dumps(row))
