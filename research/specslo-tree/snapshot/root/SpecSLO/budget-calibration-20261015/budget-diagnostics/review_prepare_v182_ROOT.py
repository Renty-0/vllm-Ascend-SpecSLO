"""ROOT combined-source review/freeze only. Never starts a model or guard job."""
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
P = B / 'tree-compact-b3-target-union3-active32-full-v182'
G = D / 'v182-root-execution'
READY = D / 'V182_COMPACT_B3_TARGET_UNION3_ACTIVE32_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json'
PEER = D / 'V182_INDEPENDENT_SOURCE_PEER_REVIEW.json'
REVIEW = D / 'ROOT_V182_COMPACT_TARGET_UNION3_SOURCE_AND_GUARD_REVIEW.json'
PREPARED = D / 'ROOT_V182_PREPARED_SOURCE_HOST_AUDIT.json'

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
assert len(sys.argv) == 2 and len(sys.argv[1]) == 64, 'supply actual final peer SHA'
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
assert sha(READY) == '492d90f68eba5000b4b4c693ad121c3c92b40f04fa98952833c2660dcf4bae96'
assert sha(PEER) == sys.argv[1]
ready, peer = read(READY), read(PEER)
assert peer['status'].replace('-', '_') == 'passed_source_review' and peer['candidate'] == str(P)
sys.path.insert(0, str(P))
spec = importlib.util.spec_from_file_location('_ROOT_V182_source_guard', G / 'guard182.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
state = guard.source_state(archive=True)
guard_inputs = guard.guard_inputs()
flat = guard.merge(state['local'], state['external'], state['evidence'], guard_inputs,
                   {str(P / 'CPU_RESULT182.json'): sha(P / 'CPU_RESULT182.json')})
assert state['local'] == ready['files'] and state['external'] == ready['external_files']
assert state['evidence'] == ready['retained_HOST_evidence']
assert guard_inputs == ready['execution_guard_inputs']
assert (len(state['local']), len(state['external']), len(state['evidence']), len(flat)) == (26, 462, 97, 590)
for path, expected in flat.items():
    path = Path(path)
    assert path.is_file() and not path.is_symlink() and sha(path) == expected, path
for path in state['local']:
    if path.endswith('.py'):
        ast.parse(Path(path).read_text(), filename=path)
cpu = read(P / 'CPU_RESULT182.json')
assert cpu['status'] == 'passed' and cpu['tests_run'] == 5
assert cpu['failures'] == cpu['errors'] == 0
assert cpu['Torch_imported'] is cpu['NPU_used'] is cpu['model_constructed'] is False
assert cpu['CPU_Torch_metadata_child'] is True
assert ready['whole_source_projections'] == 9
assert ready['constructor_per_worker'] == ready['original_B4_component_per_worker'] == 1
assert ready['prestartup_original_protocol_capture_per_worker'] == 1
assert ready['original_B4_same_object_aliases'] == ['_v180_base_component', '_v181_baseline_component']
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
review_inputs = guard.merge(flat, {str(PEER): sha(PEER)})
save(REVIEW, dict(
    schema='ROOT.V182.source-and-guard-review.v1', status='passed_source_review',
    candidate=str(P), UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    ready=pin(READY), peer=pin(PEER), reviewed_inputs=review_inputs,
    local_source_files=26, selected_readonly_inputs=462, retained_HOST_evidence=97,
    CPU_tests=5, CPU_elapsed_seconds=cpu['elapsed_seconds'],
    original_parent_HOST_not_rerun=True, NPU_used=False, Torch_imported=False,
    source_findings=[
        'One constructor, one prestartup protocol capture, one actual original full527 B4 component; the same current proof object is explicitly aliased to both immutable source issuer markers.',
        'Unchanged original active32/cap3-highN/trueB5-lowN/C96/E96/global192 and coalesce1, Poisson463/output256/SLO/source elapsed fields; no mixed prompt, new budget, parent4 or persistent metadata.',
        'Frozen180 actual16N geometry actor executes before the new union qualification; E0 calls retain the original FastV5 fallback. Six actual current Native objects are naturally qualified and bound on this engine.',
        'Frozen181 three numerical representatives and actual same-U/encoded/56draftKV/ledger gates remain exact; production union capability still requires a fresh real positive/invalid/consume/next proof.',
        'New joint actual target calls the actually installed production pad. Independent target metadata builds its own declarative partition from original CPU plans on a fresh independent target allocation and model history.',
        'Joint parent/normal N16 must really use64/S16, promoted child9..16 uses64/S16, smaller children retain the original mapper. All physical hidden, heads and128 target KV tensors are checked, restored and used in original commit.',
        'Target-only epoch activates actual dedicated463..526 scratch with128 independent pages; draft role delegates unchanged181 physical source with102pages. No all-role ambiguous rebuild or candidate KV seeding.',
        'Actual joint raw audit rederives all owners/queries/positions/slots/page tables/host lengths and reexecutes the original HOST verifier against all three target ranks. Previously qualified Native entries are recorded honestly without artificial cold flags.',
        'Both current issuer objects, actual compact proof, actual production union proof, engine and the single baseline object are required by a new combined scope. No old final scope or model receipt is reused.',
        'Original P102 commit_states module and exact prestartup protocol object are retained; no late Native decoder reflection. Original lease/shared failure handling remains before publication.',
        'Original201+44/114 startup, current base and both changed-path actors, bounded prewarm, warm65/two complete Native21 zero65, one seal, original full463, raw-before-metrics and worker cleanup remain mandatory.',
        'Guard retains its own original selflock, exact pre/post baseline verifier, cleanup and7200s/private256MiB tag512MiB aggregate limits; only scoped temporary-file ENOENT is tolerated.',
    ],
    limitations=[
        'CPU metadata uses an explicit HOST-only route carrier for the original NPU-causal branch, with complete inverse. It does not establish current model/NPU mathematics.',
        'All parent sources and selected archives are verified; full environment import closure is not asserted.',
        'Joint natural source can honestly have no positive under new-M arithmetic; that must retain subset evidence and prevent full production use.',
        'No current full463 performance, whole-workload AR, adoption or goal completion follows from this review.',
    ], formal_Goodput=False, adopted=False,
))
args = [sys.executable, '-B', str(G / 'guard182.py'), '--prepare-only',
        '--root-review', str(REVIEW), '--expected-root-review-sha256', sha(REVIEW),
        '--peer-review', str(PEER), sha(PEER)]
completed = subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
frozen, sources, prepared = guard.prepared()
archive = P / 'PREPARED_SOURCE_ARCHIVE182.tar.gz'
with tarfile.open(archive, 'r:gz') as stream:
    members = stream.getmembers()
    assert len(members) == len(sources) == 593
    for member in members:
        assert member.isfile() and hashlib.sha256(stream.extractfile(member).read()).hexdigest() == sources['/' + member.name]
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
save(PREPARED, dict(schema='ROOT.V182.prepared-source-HOST-audit.v1', status='passed',
                   candidate=str(P), review=pin(REVIEW), peer=pin(PEER),
                   freeze=pin(P / 'FROZEN_SOURCE182.json'), archive=pin(archive),
                   actual_archive_members=len(members), actual_source_members=len(sources),
                   prepared=prepared, prepare_stdout=completed.stdout,
                   NPU_used=False, Torch_imported=False, formal_Goodput=False, adopted=False))
print(json.dumps({'review': pin(REVIEW), 'prepared': pin(PREPARED),
                  'freeze': pin(P / 'FROZEN_SOURCE182.json'), 'archive': pin(archive),
                  'members': len(members), 'NPU_used': False}))
