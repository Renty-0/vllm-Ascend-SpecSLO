"""ROOT source review and immutable preparation, without model execution."""
import argparse
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
P = B / 'tree-parent3-union3-oracle-device-indexfix-full-v187'
G = D / 'v187-root-execution'
READY = D / 'V187_ORACLE_DEVICE_INDEX_FIX_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json'
PEER = D / 'V187_INDEPENDENT_SOURCE_PEER_REVIEW.json'
REVIEW = D / 'ROOT_V187_ORACLE_DEVICE_INDEX_FIX_SOURCE_AND_GUARD_REVIEW.json'
PREPARED = D / 'ROOT_V187_PREPARED_SOURCE_HOST_AUDIT.json'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def pin(path):
    return dict(path=str(path), sha256=sha(path))

def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')

parser = argparse.ArgumentParser()
parser.add_argument('--expected-peer-sha256', required=True)
args = parser.parse_args()
assert sys.version_info[:2] == (3, 11)
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
assert sha(READY) == '176b8a8701abc68f79c37b0bd739746c526070881b9000bb8c5f6cc0a0118d84'
assert sha(PEER) == args.expected_peer_sha256
ready, peer = read(READY), read(PEER)
assert peer['status'].replace('-', '_') == 'passed_source_review'
assert peer['candidate'] == str(P)
assert ready['source_writes_stopped'] is True
sys.path.insert(0, str(P))
spec = importlib.util.spec_from_file_location('_ROOT_V187_source_guard', G / 'guard187.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
state = guard.source_state(archive=True)
guard_inputs = guard.guard_inputs()
flat = guard.merge(state['local'], state['external'], state['evidence'], guard_inputs,
                   {str(P / 'CPU_RESULT187.json'): sha(P / 'CPU_RESULT187.json')})
assert state['local'] == ready['local_files']
assert state['external'] == ready['external_files']
assert state['evidence'] == ready['retained_HOST_evidence']
assert guard_inputs == ready['guard_inputs']
assert flat == ready['reviewed_inputs_before_peer']
assert (len(state['local']), len(state['external']), len(state['evidence']), len(flat)) == (41, 658, 93, 797)
for path, expected in flat.items():
    path = Path(path)
    assert path.is_file() and not path.is_symlink() and sha(path) == expected, path
for path in state['local']:
    if path.endswith('.py'):
        ast.parse(Path(path).read_text(), filename=path)
cpu = read(P / 'CPU_RESULT187.json')
assert cpu['status'] == 'passed' and cpu['tests_run'] == 3
assert cpu['failures'] == cpu['errors'] == 0
assert cpu['Torch_imported'] is cpu['NPU_used'] is cpu['model_constructed'] is False
assert ready['counts']['whole_parent181_projection'] == 39
assert ready['counts']['whole_parent176_projection'] == 26
from projection187 import verify_derivations, verify_oracle_projection
assert verify_derivations()['projected_sources'] == 26
assert verify_oracle_projection()['projected_sources'] == 39
descriptor = read(P / 'ORACLE_PROJECTION187.json')
namespace_only = []
for name, row in descriptor['files'].items():
    if name in ('DERIVATIONS187.json', 'check187.py', 'host187.py', 'projection187.py', 'pure187.py'):
        continue
    original = Path(row['parent']).read_text().replace('181', '187')
    original = original.replace('tree-parent3-repair-frontier-union3-active32-full-v187', P.name)
    assert (P / name).read_text() == original, name
    namespace_only.append(name)
assert len(namespace_only) == 34
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
review_inputs = guard.merge(flat, {str(PEER): sha(PEER)})
save(REVIEW, dict(
    schema='ROOT.V187.oracle-device-index-source-and-guard-review.v1',
    status='passed_source_review', candidate=str(P),
    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    ready=pin(READY), peer=pin(PEER), reviewed_inputs=review_inputs,
    namespace_only_whole_sources=namespace_only,
    local_source_files=41, selected_readonly_inputs=658, retained_HOST_evidence=93,
    CPU_tests=3, CPU_elapsed_seconds=cpu['elapsed_seconds'],
    original_parent_HOST_not_rerun=True, NPU_used=False, Torch_imported=False,
    source_findings=[
        'Frozen failed181 remains unchanged. Its failure is an independent oracle index-device exception, not a demonstrated model arithmetic mismatch; original B4 completed but new device qualification did not.',
        'After namespace changes34 of39 selected source files are byte-identical. The only numerical-path edit is independent_frame: each of input/position/slots/CPU context/tables is permuted with an index allocated on that tensor own device.',
        'The complete physical permutation, original PA input values, host Q/KV schema, frozen NativeAttentionMetadata dataclass and interior-padding semantics are unchanged; there is no cast of CPU metadata to the NPU.',
        'Three new HOST checks execute actual old/corrected functions on36 explicit mixed-device carriers with the real frozen NativeMetadata dataclass, whole39/26 inverses, real runtime protocol capture and actual new freeze/receipt constructors. HOST carriers are not device numerical evidence.',
        'Production fast source, policy, budget, original arrival/output/SLO and clocks remain unchanged: active32, normal/eager96/global192, high-N cap3/low-N true B5 and one-token parent3 frontier union scope.',
        'All original startup201+44 target/114 draft, current B4/full527 component, three actual union representatives, same-M independent hidden/head/56KV/encoded6/canonical8/ledger, natural positive and invalidation, real child consume/commit/next remain compulsory on the current engine.',
        'Original protocol is captured before decoder replacement and original native_protocol commit_states is pinned. No late recompilation, runtime qualification flag assignment or new production PA/chain family is introduced.',
        'Bounded warm fixedpoint, two complete Native21-zero traces, seal and original full463 still follow current mathematics. No previous failed run or successful parent startup supplies a current device capability.',
        'New private selflocked supervisor retains exact pre/post baseline verification, owned worker cleanup,7200s,256MiB tag/512MiB aggregate bounds and narrowly scoped temporary ENOENT handling. Guard changes outside namespace are only the current HOST count4 to3.',
        'Cold actual guardian and all parent archives are verified in place; preserved failed HOST attempts and failed181 evidence remain pinned. No model/dataset/artifact relocation is performed.',
    ],
    limitations=[
        'This standalone candidate omits V180 compact target geometry, V183 metadata storage and V185 prefill admission. Combining them requires a new source epoch and current joint qualification.',
        'No current NPU arithmetic, whole-workload autoregressive equivalence, Goodput, adoption or goal completion follows from this source review.',
    ], formal_Goodput=False, adopted=False,
))
completed = subprocess.run(
    [sys.executable, '-B', str(G / 'guard187.py'), '--prepare-only',
     '--root-review', str(REVIEW), '--expected-root-review-sha256', sha(REVIEW),
     '--peer-review', str(PEER), sha(PEER)],
    check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
frozen, sources, prepared = guard.prepared()
archive = P / 'PREPARED_SOURCE_ARCHIVE187.tar.gz'
with tarfile.open(archive, 'r:gz') as stream:
    members = stream.getmembers()
    assert len(members) == len(sources) == 800
    for member in members:
        assert member.isfile()
        assert hashlib.sha256(stream.extractfile(member).read()).hexdigest() == sources['/' + member.name]
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
save(PREPARED, dict(
    schema='ROOT.V187.prepared-source-HOST-audit.v1', status='passed',
    candidate=str(P), review=pin(REVIEW), peer=pin(PEER),
    freeze=pin(P / 'FROZEN_SOURCE187.json'), archive=pin(archive),
    actual_archive_members=len(members), actual_source_members=len(sources),
    prepared=prepared, prepare_stdout=completed.stdout,
    NPU_used=False, Torch_imported=False, formal_Goodput=False, adopted=False))
print(json.dumps(dict(review=pin(REVIEW), prepared=pin(PREPARED),
                     freeze=pin(P / 'FROZEN_SOURCE187.json'), archive=pin(archive),
                     members=len(members), NPU_used=False)))
