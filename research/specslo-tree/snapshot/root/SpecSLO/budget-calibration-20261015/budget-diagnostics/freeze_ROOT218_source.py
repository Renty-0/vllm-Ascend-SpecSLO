"""Freeze actual HOST/CPU and synthetic-flow evidence, never device results."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B/'budget-diagnostics'
P = B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'
DONOR = B/'tree-packed-KV-preflight-with-own-FIA-AR-full-v217'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def desc(path):
    return dict(path=str(path), sha256=sha(path))


assert not (P/'FREEZE218.json').exists()
host_path = next(p for p in (P/'HOST-attempts').glob('*/RESULT.json') if read(p).get('total_tests') == 28)
host = read(host_path)
assert host['status'] == 'passed' and host['source_unchanged'] is True
assert host['Torch_imported'] is host['NPU_used'] is host['model_launched'] is False
for path, digest in host['source_sha256'].items():
    assert sha(path) == digest, path
cpu_path = P/'Torch-CPU-attempts/control-v1/RESULT.json'
cpu = read(cpu_path)
assert cpu['status'] == 'passed' and cpu['cases'] == 28 and len(cpu['negative_cases']) == 3
assert cpu['Torch_imported'] is True and cpu['Torch_npu_imported'] is cpu['NPU_used'] is cpu['model_constructed'] is False
for path, digest in cpu['source_sha256'].items():
    assert sha(path) == digest, path
loop_path = next((P/'HOST-loop-attempts').glob('*/RESULT.json'))
loop = read(loop_path)
assert loop['status'] == 'passed' and loop['synthetic_fixture_only'] is True
assert loop['real_Torch_imported'] is loop['NPU_used'] is loop['model_constructed'] is loop['neural_output_qualified'] is False
assert loop['actual_reference_runner_roles'] == 4 and loop['simulated_target_steps_per_rank'] == 7905
for path, digest in loop['source_sha256'].items():
    assert sha(path) == digest, path
for path, digest in loop['original_actual_input_sha256'].items():
    assert sha(path) == digest, path
for path in P.glob('*.py'):
    ast.parse(path.read_text())

evidence = dict(schema='ROOT218.actual-source-HOST-CPU-ready.v1', UTC=datetime.now(timezone.utc).isoformat(),
    status='passed', HOST=desc(host_path), HOST_tests=28, Torch_CPU=desc(cpu_path), Torch_CPU_cases=28,
    synthetic_HOST_loop=desc(loop_path), synthetic_loop_does_not_qualify_NPU=True,
    real_failed216_role_inputs=loop['original_actual_input_sha256'],
    known216_failure=desc(D/'ROOT_V216_AR_ROLE_FAILURE_AND_NEXT_SOURCE_PLAN.json'),
    actual216_terminal_failure_audit=desc(D/'ROOT_V216_INDEPENDENT_TERMINAL_FAILURE_AUDIT_V3_PY311.json'),
    no_existing_frozen_source_or_raw_modified=True, NPU_used=False, Goodput=None, adopted=False,
    source_peer_review=False)
evidence_path = D/'ROOT_V218_ROLE_AWARE_SOURCE_AND_HOST_PREPARATION.json'
save(evidence_path, evidence)
donor = read(DONOR/'FREEZE217.json')
files = dict(donor['files'])
files[str(DONOR/'FREEZE217.json')] = sha(DONOR/'FREEZE217.json')
for path in P.rglob('*'):
    if path.is_file() and '__pycache__' not in path.parts:
        files[str(path)] = sha(path)
for path in (D/'prepare_ROOT218_source.py', D/'candidate_comparison218_SOURCE.py', D/'run_ROOT218_HOST.py',
             Path(__file__), evidence_path, D/'ROOT_V216_AR_ROLE_FAILURE_AND_NEXT_SOURCE_PLAN.json',
             D/'ROOT_V216_INDEPENDENT_TERMINAL_FAILURE_AUDIT_V3_PY311.json'):
    files[str(path)] = sha(path)
files.update(loop['original_actual_input_sha256'])
review = dict(schema='ROOT218.rank-aware-own-AR-source-review.v1', UTC=datetime.now(timezone.utc).isoformat(),
    status='passed_source_review', source_peer_review=False, reviewed_inputs=files.copy(), actual_HOST=evidence,
    source_delta='same217 production packed preflight; own-AR candidate comparison loads original leader1 artifact inside all4 original stage; independent audit uses that same explicitly pinned artifact',
    preserved=['Only rank1 materializes Native generate_results; other ranks retain None.',
        'Every rank retains own input/cohort artifact and verifies manifest, tokenized inputs, request identities and original prefill cohorts against leader1.',
        'Each target rank independently generates its reference from prompts plus its own previous AR outputs. Candidate outputs enter comparison only.',
        'The candidate-loading helper is inside the original collective error-propagation stage before independent allocation.',
        'All original model/graph, packed-preflight shadow gate, 96/96/global192 budget, warm/twozero/seal/full463 remain required.',
        'Independent reference remains after measured full463 clock. No cold-first-cohort admission policy was added.'],
    device_shadow_qualified=False, own_AR_device_qualified=False, NPU_used=False, Goodput=None, adopted=False)
review_path = P/'ROOT_SOURCE_REVIEW218.json'
save(review_path, review)
files[str(review_path)] = sha(review_path)
freeze = dict(schema='ROOT218.rank-aware-own-AR-source-freeze.v1', UTC=datetime.now(timezone.utc).isoformat(),
    files=files, source_review=desc(review_path), original209_source_only_dependency=donor['original209_source_only_dependency'],
    source_donor217=dict(path=str(DONOR), freeze_sha256=sha(DONOR/'FREEZE217.json'), device_qualification_inherited=False),
    HOST_tests=28, Torch_CPU_cases=28, synthetic_HOST_four_role_loop=True, synthetic_loop_is_not_model_qualification=True,
    source_peer_review=False, SLO_AWARE_ADMISSION=1, packed_target_KV_preflight=True,
    device_shadow_qualified=False, full463_required=True, own_AR_device_qualified=False, Goodput=None, adopted=False)
freeze_path = P/'FREEZE218.json'
save(freeze_path, freeze)
archived = dict(files)
archived[str(freeze_path)] = sha(freeze_path)
archive = P/'SOURCE218.tar.gz'
with tarfile.open(archive, 'w:gz') as stream:
    for path in sorted(archived):
        stream.add(path, arcname=path.lstrip('/'), recursive=False)
with tarfile.open(archive, 'r:gz') as stream:
    members = stream.getmembers()
    assert len(members) == len(archived)
    for member in members:
        assert member.isfile() and hashlib.sha256(stream.extractfile(member).read()).hexdigest() == archived['/'+member.name]
result = dict(schema='ROOT218.frozen-unrun-source.v1', UTC=datetime.now(timezone.utc).isoformat(),
    directory=str(P), freeze_sha256=sha(freeze_path), archive_sha256=sha(archive), archive_members=len(archived),
    archive_bytes=archive.stat().st_size, HOST_tests=28, Torch_CPU_cases=28,
    actual_failed216_role_regression=True, synthetic_four_role_control_flow=True,
    device_shadow_qualified=False, own_AR_device_qualified=False, source_peer_review=False,
    NPU_started=False, Goodput=None, adopted=False)
save(D/'ROOT_V218_SOURCE_FROZEN_UNRUN.json', result)
print(json.dumps(result))
