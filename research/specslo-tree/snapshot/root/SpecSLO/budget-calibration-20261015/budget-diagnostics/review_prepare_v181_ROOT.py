"""ROOT source-only review and immutable freeze; no supervisor/model action."""
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
P = B / 'tree-parent3-repair-frontier-union3-active32-full-v181'
G = D / 'v181-root-execution'
READY = D / 'V181_PARENT3_UNION3_ACTIVE32_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW_REVISION2.json'
PEER = D / 'V181_INDEPENDENT_SOURCE_PEER_REVIEW.json'
REVIEW = D / 'ROOT_V181_PARENT3_UNION3_SOURCE_AND_GUARD_REVIEW.json'
PREPARED = D / 'ROOT_V181_PREPARED_SOURCE_HOST_AUDIT.json'

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
assert sha(READY) == 'dd9091b148d2cafe44fed451c749ff2ede3a8feba680b5aa63e2fa43bdcc34b2'
assert sha(PEER) == '5eefd7b038c90e5ef3b90b15fb401d995a199ffa1b419e81765a782c8d1dcfc2'
ready, peer = read(READY), read(PEER)
assert peer['status'].replace('-', '_') == 'passed_source_review' and peer['candidate'] == str(P)
sys.path.insert(0, str(P))
spec = importlib.util.spec_from_file_location('_ROOT_V181_source_guard', G / 'guard181.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
state = guard.source_state(archive=True)
guard_inputs = guard.guard_inputs()
flat = guard.merge(state['local'], state['external'], state['evidence'], guard_inputs,
                   {str(P / 'CPU_RESULT181.json'): sha(P / 'CPU_RESULT181.json')})
assert state['local'] == ready['files'] and state['external'] == ready['external_files']
assert state['evidence'] == ready['retained_HOST_evidence']
assert guard_inputs == ready['execution_guard_inputs']
assert (len(state['local']), len(state['external']), len(state['evidence']), len(flat)) == (40, 329, 272, 646)
for path, expected in flat.items():
    path = Path(path)
    assert path.is_file() and not path.is_symlink() and sha(path) == expected, path
for path in state['local']:
    if path.endswith('.py'):
        ast.parse(Path(path).read_text(), filename=path)
cpu = read(P / 'CPU_RESULT181.json')
assert cpu['status'] == 'passed' and cpu['tests_run'] == 4
assert cpu['failures'] == cpu['errors'] == 0
assert cpu['Torch_imported'] is cpu['NPU_used'] is cpu['model_constructed'] is False
assert ready['whole_source_projections'] == 26
assert ready['receipt_label_revision']['selected_total'] == 96
assert ready['receipt_label_revision']['global_selected_candidate_ceiling'] == 192
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
review_inputs = guard.merge(flat, {str(PEER): sha(PEER)})
save(REVIEW, dict(
    schema='ROOT.V181.source-and-guard-review.v1', status='passed_source_review',
    candidate=str(P), UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    ready=pin(READY), peer=pin(PEER), reviewed_inputs=review_inputs,
    local_source_files=40, selected_readonly_inputs=329, retained_HOST_evidence=272,
    CPU_tests=4, CPU_elapsed_seconds=cpu['elapsed_seconds'],
    original_parent_HOST_not_rerun=True, NPU_used=False, Torch_imported=False,
    source_findings=[
        'Original P176 active32/C96/E96/global192 policy, high-N B3 and low-N B5, unchanged original arrival/clocks/output/SLO; no target geometry or budget priority changes.',
        'Only parent3/child3/shared-three-step, normal/eager each at most16, total at most32, one-token repair and legal original source ticket may use the union. R0/E0/multitoken/parent4/out-of-scope requests use the original path before writes.',
        'One PA hidden union combines actual repair and eager frontier work, retaining the original three-step chain and original Eprime head; no four-step truncation or altered acceptance algorithm.',
        'Current engine/base/new-use proof/private issuer and exact real Native objects are bound; no Native runtime_validated flags are assigned and no production PA or chain family is added.',
        'Three new representatives cover minimal/interior-padding/max32 union; independent original PA and three-step model reference compare all physical hidden, full encoded predictions and all28 draft K/V pairs, canonical8 zeros and ledger.',
        'Natural current parent3 source must produce both true promotion/child consume and true invalidation with independent target same-M hidden/all64 K/V, original commit/compaction/next. No positive or negative means no full production capability.',
        'Explicit original P102 native_protocol module supplies commit_states only; original protocol is captured before decoder replacement and its exact prestartup object is required at every controlled commit. No late AST recompilation.',
        'Actual allocation/owner/page lease is checked around model writes; shared fatal status occurs before publication, no fallback after partial writes, and the all-four expected lease error is exercised independently.',
        'Receipt labels now report102 actually unique fixture pages and96 actual selected candidates separately from192 global ceiling/raw pool nodes. Strict execution gates are unchanged and prelabel artifacts/failed HOST attempts are preserved.',
        'Original201+44 target and114 draft startup, own full527 B4 component, bounded warm65 fixedpoint, two full Native21 zero65 traces, one seal and original full463 remain required.',
        'Original selflocked supervisor, exact pre/post baseline verifier, owned cleanup,7200s/private256MiB tag and512MiB aggregate limits retained. ENOENT handling is restricted to enumerated temporary files under this private actual directory.',
        'HOST and source review do not establish current NPU mathematics, whole-workload AR equivalence, performance, adoption or goal completion.',
    ],
    limitations=[
        'Selected sources and parent archives are verified; full environment import closure is not claimed.',
        'Standalone source does not include V180 compact geometry or V183 persistent metadata; combining them requires a new current joint qualification.',
        'Natural component is not a substitute for the two complete post-warm zero traces required by the full runner.',
    ], formal_Goodput=False, adopted=False,
))
args = [sys.executable, '-B', str(G / 'guard181.py'), '--prepare-only',
        '--root-review', str(REVIEW), '--expected-root-review-sha256', sha(REVIEW),
        '--peer-review', str(PEER), sha(PEER)]
completed = subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
frozen, sources, prepared = guard.prepared()
archive = P / 'PREPARED_SOURCE_ARCHIVE181.tar.gz'
with tarfile.open(archive, 'r:gz') as stream:
    members = stream.getmembers()
    assert len(members) == len(sources) == 649
    for member in members:
        assert member.isfile() and hashlib.sha256(stream.extractfile(member).read()).hexdigest() == sources['/' + member.name]
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
save(PREPARED, dict(schema='ROOT.V181.prepared-source-HOST-audit.v1', status='passed',
                   candidate=str(P), review=pin(REVIEW), peer=pin(PEER),
                   freeze=pin(P / 'FROZEN_SOURCE181.json'), archive=pin(archive),
                   actual_archive_members=len(members), actual_source_members=len(sources),
                   prepared=prepared, prepare_stdout=completed.stdout,
                   NPU_used=False, Torch_imported=False, formal_Goodput=False, adopted=False))
print(json.dumps({'review': pin(REVIEW), 'prepared': pin(PREPARED),
                  'freeze': pin(P / 'FROZEN_SOURCE181.json'), 'archive': pin(archive),
                  'members': len(members), 'NPU_used': False}))
