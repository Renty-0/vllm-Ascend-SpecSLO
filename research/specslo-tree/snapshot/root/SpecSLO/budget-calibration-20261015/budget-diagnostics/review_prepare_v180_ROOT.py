"""ROOT source-only review/freeze; never enters a supervisor/model action."""
import ast
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
P = B / 'tree-uniform-b3-three-bucket-full-v180'
G = D / 'v180-root-execution'
READY = D / 'V180_UNIFORM_B3_THREE_BUCKET_SOURCE_AND_HOST_READY_FOR_ROOT_REVIEW.json'
PEER = D / 'V180_INDEPENDENT_SOURCE_PEER_REVIEW.json'
REVIEW = D / 'ROOT_V180_UNIFORM_B3_SOURCE_AND_GUARD_REVIEW.json'
PREPARED = D / 'ROOT_V180_PREPARED_SOURCE_HOST_AUDIT.json'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def pin(path):
    return {'path': str(path), 'sha256': sha(path)}

def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')

assert sys.version_info[:2] == (3, 11)
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
assert sha(READY) == 'fb906f75b9fd708998ce59010421783dacfffa1a08b23e30e6037c71b7951b6c'
ready = read(READY)
peer = read(PEER)
assert peer['status'].replace('-', '_') == 'passed_source_review'
assert peer['candidate'] == str(P)
sys.path.insert(0, str(P))
spec = importlib.util.spec_from_file_location('_ROOT_V180_source_guard', G / 'guard180.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
state = guard.source_state(archive=True)
guard_inputs = guard.guard_inputs()
flat = guard.merge(state['local'], state['external'], state['evidence'], guard_inputs,
                   {str(P / 'CPU_RESULT180.json'): sha(P / 'CPU_RESULT180.json')})
assert flat == ready['reviewed_inputs']
for path, expected in flat.items():
    path = Path(path)
    assert path.is_file() and not path.is_symlink() and sha(path) == expected, path
for path in state['local']:
    if path.endswith('.py'):
        ast.parse(Path(path).read_text(), filename=path)
assert len(state['local']) == 25 and len(state['external']) == 208
assert len(flat) == 243
cpu = read(P / 'CPU_RESULT180.json')
assert cpu['status'] == 'passed' and cpu['tests_run'] == 5
assert cpu['failures'] == cpu['errors'] == 0
assert cpu['Torch_imported'] is cpu['NPU_used'] is False
assert cpu['CPU_Torch_metadata_child'] is True
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
review_inputs = guard.merge(flat, {str(PEER): sha(PEER)})
save(REVIEW, dict(
    schema='ROOT.V180.source-and-guard-review.v1', status='passed_source_review',
    candidate=str(P), UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    ready=pin(READY), peer=pin(PEER), reviewed_inputs=review_inputs,
    local_source_files=25, selected_readonly_inputs=208,
    CPU_tests=5, CPU_elapsed_seconds=cpu['elapsed_seconds'],
    original_parent_HOST_not_rerun=True, NPU_used=False, Torch_imported=False,
    source_findings=[
        'Production gate reads actual uniform B3 plans and the original contiguous predicate, greedy state, exact N band; mixed B2+B4 same N/Q retains original mapper.',
        'Three layouts M64/S16, M80/S20, M96/S24 use original disjoint scratch ownership. All six actual Native entry objects and base/math proof are bound to this engine.',
        'True separate repair plus independent three-step PA reference checks six encoded predictions, raw8 zero slots3/7, all28 draft KV pairs and original ledger; no four-step output truncation.',
        'All16 Ns have three actual successive contexts; six extrema have three complete fresh traces, four natural protocol cycles plus next and two 21-counter zero traces.',
        'Target same-M independent model metadata, all physical hidden and heads, all64 target KV pairs, genuine prefix, original verifier/commit/next checks; old/new cross-M differences retained and not declared exact.',
        'Exactly six target keys may be added to actual post-base dictionaries; original mappings and natural validation flags retained. No new MC2 rows outside frozen profile.',
        'Original114 draft and201+44 target startup, own full527 B4 component, bounded prewarm65, two zero65, one seal and original full463 remain hard requirements.',
        'Full metrics use original elapsed fields and exact463 Poisson/output/SLO source. Local11 AB/BA target timings are separate and do not predict Goodput.',
        'Thin runner projection and original selflocked guard retain original pre/post verifier, owned cleanup, private size/time limits; only frozen scoped ENOENT tmp snapshot helper is used.',
        'HOST carriers test storage/metadata only. Actual model mathematics, whole-workload AR, performance, adoption and goal completion are not established by this source review.',
    ],
    limitations=[
        'Selected source and original archives are verified; complete environment import closure is not asserted.',
        'Historical failed HOST artifacts are preserved; their superseded parse failures are not current execution sources.',
        'HOST metadata uses an equivalent packed contiguous B3 plan; actual width2/depth4 selection is validated by current model component before full.',
    ], formal_Goodput=False, adopted=False,
))
args = [sys.executable, '-B', str(G / 'guard180.py'), '--prepare-only',
        '--root-review', str(REVIEW), '--expected-root-review-sha256', sha(REVIEW),
        '--peer-review', str(PEER), sha(PEER)]
completed = subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
frozen, sources, prepared = guard.prepared()
archive = P / 'PREPARED_SOURCE_ARCHIVE180.tar.gz'
with tarfile.open(archive, 'r:gz') as stream:
    members = stream.getmembers()
    assert len(members) == len(sources) == 246
    for member in members:
        assert member.isfile() and hashlib.sha256(stream.extractfile(member).read()).hexdigest() == sources['/' + member.name]
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
save(PREPARED, dict(schema='ROOT.V180.prepared-source-HOST-audit.v1', status='passed',
                   candidate=str(P), review=pin(REVIEW), peer=pin(PEER),
                   freeze=pin(P / 'FROZEN_SOURCE180.json'), archive=pin(archive),
                   actual_archive_members=len(members), actual_source_members=len(sources),
                   prepared=prepared, prepare_stdout=completed.stdout,
                   NPU_used=False, Torch_imported=False, formal_Goodput=False, adopted=False))
print(json.dumps({'review': pin(REVIEW), 'prepared': pin(PREPARED),
                  'freeze': pin(P / 'FROZEN_SOURCE180.json'), 'archive': pin(archive),
                  'members': len(members), 'NPU_used': False}))
