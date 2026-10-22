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
P = B / 'tree-uniform-b3-persistent-metadata-full-v183'
G = D / 'v183-root-execution'
READY = D / 'V183_PERSISTENT_METADATA_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json'
PEER = D / 'V183_INDEPENDENT_SOURCE_PEER_REVIEW.json'
REVIEW = D / 'ROOT_V183_PERSISTENT_METADATA_SOURCE_AND_GUARD_REVIEW.json'
PREPARED = D / 'ROOT_V183_PREPARED_SOURCE_HOST_AUDIT.json'

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
assert sha(READY) == 'e24070f06b57e4419e806254894040947fab64334b00a8aafbf99ebf4cb7e397'
assert sha(PEER) == sys.argv[1]
ready, peer = read(READY), read(PEER)
assert peer['status'].replace('-', '_') == 'passed_source_review' and peer['candidate'] == str(P)
sys.path.insert(0, str(P))
spec = importlib.util.spec_from_file_location('_ROOT_V183_source_guard', G / 'guard183.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
state = guard.source_state(archive=True)
guard_inputs = guard.guard_inputs()
flat = guard.merge(state['local'], state['external'], state['evidence'], guard_inputs,
                   {str(P / 'CPU_RESULT183.json'): sha(P / 'CPU_RESULT183.json')})
assert state['local'] == ready['files'] and state['external'] == ready['external_files']
assert state['evidence'] == ready['retained_HOST_evidence']
assert guard_inputs == ready['execution_guard_inputs']
assert (len(state['local']), len(state['external']), len(state['evidence']), len(flat)) == (28, 247, 202, 482)
for path, expected in flat.items():
    path = Path(path)
    assert path.is_file() and not path.is_symlink() and sha(path) == expected, path
for path in state['local']:
    if path.endswith('.py'):
        ast.parse(Path(path).read_text(), filename=path)
cpu = read(P / 'CPU_RESULT183.json')
assert cpu['status'] == 'passed' and cpu['tests_run'] == 3
assert cpu['failures'] == cpu['errors'] == 0
assert cpu['Torch_imported'] is cpu['NPU_used'] is cpu['model_constructed'] is False
assert ready['whole_source_projections'] == 20
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
review_inputs = guard.merge(flat, {str(PEER): sha(PEER)})
save(REVIEW, dict(
    schema='ROOT.V183.source-and-guard-review.v1', status='passed_source_review',
    candidate=str(P), UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    ready=pin(READY), peer=pin(PEER), reviewed_inputs=review_inputs,
    local_source_files=28, selected_readonly_inputs=247, retained_HOST_evidence=202,
    CPU_tests=3, CPU_elapsed_seconds=cpu['elapsed_seconds'],
    original_parent_HOST_not_rerun=True, NPU_used=False, Torch_imported=False,
    source_findings=['Only uniform B3 contiguous greedy target metadata uses external persistent staging after the unchanged original HOST slot/owner/page checks; original P180 high-N cap3/low-N trueB5/active32/C96/E96/global192 and coalesce1 remain unchanged.', 'Original graph entry buffers, layouts, flags, input-copy implementation, two attention task workers and replay-first order are unchanged. No new family or key is introduced beyond the original six geometry keys.', 'Five pinned HOST and five device payload tensors persist. The immutable original NativeAttentionMetadata record is rebuilt with dataclasses.replace, rather than illegally assigning its frozen fields.', 'Each use fills actual current values, gathers the current maintained device tables into persistent output, binds the allocation generation, and retains actual table/allocation/context references through an original-current-stream completion Event.', 'The completion event follows the original copy and replay. The next reuse or release drains that event before CPU buffer overwrite or source reference retirement. Partial copies/model errors poison the pool and propagate; no fallback after writes.', 'Current engine, original base, actual new geometry/staging audit, six naturally validated Native objects and causal mask identity/version are bound to a private issuer before production. Unsupported/no-permission requests retain the original helper.', 'Original graph_metrics function identity and closure are retained; staging uses an independent explicit snapshot and per-trace deltas, avoiding corruption of original split/gain state discovery.', 'Every target component cycle executes actual staged greedy and hidden calls, comparing exact metadata, full physical hidden/head and all128 target KV tensors with the independent original same-M history. The staged API output feeds the actual original commit.', 'Six extrema cycle1 probes run staged calls before ordinary current F/T/F updates. They must observe genuine changed previous signatures, exactly two task-update replays with real task inventory counts and no skips, then match the fresh independent oracle. No signature or flag is edited.', 'Original201+44/114 startup, own full527 B4, all16 geometry cases/102 protocol cycles/28 next, current staged use and original warm/twozero/seal/full463 remain hard gates. No prior device receipt grants current qualification.', 'Original Poisson463/256 output/SLO/Native elapsed and actual stream/TTFT accounting remain unchanged. Local geometry AB/BA timings precede staging and cannot be reported as staging cost.', 'Original selflocked guard, baseline pre/post verification, owned cleanup and7200s/private256MiB tag512MiB aggregate limits remain. Only frozen177 scoped temporary-file ENOENT tolerance is imported.', 'P183 does not import or reuse the known failing pure181 independent_frame path. This source review does not establish current asynchronous device behavior, model mathematics, performance, adoption or the goal.'],
    limitations=['HOST metadata and event carriers establish source/ABI contracts only; actual pinning, asynchronous lifetime and changed-literal device task behavior require current NPU gates.', 'Original cold graph capture is performed by the original path. New staging qualifies reuse of naturally closed current graph objects and does not claim a new cold-capture path.', 'Selected sources and parent archives are verified; full environment import closure and whole-workload AR are not claimed.', 'P180 target-only full463 had no overall Goodput improvement; this source-only staging candidate remains unrun and no speedup is predicted.'], formal_Goodput=False, adopted=False,
))
args = [sys.executable, '-B', str(G / 'guard183.py'), '--prepare-only',
        '--root-review', str(REVIEW), '--expected-root-review-sha256', sha(REVIEW),
        '--peer-review', str(PEER), sha(PEER)]
completed = subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
frozen, sources, prepared = guard.prepared()
archive = P / 'PREPARED_SOURCE_ARCHIVE183.tar.gz'
with tarfile.open(archive, 'r:gz') as stream:
    members = stream.getmembers()
    assert len(members) == len(sources) == 485
    for member in members:
        assert member.isfile() and hashlib.sha256(stream.extractfile(member).read()).hexdigest() == sources['/' + member.name]
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
save(PREPARED, dict(schema='ROOT.V183.prepared-source-HOST-audit.v1', status='passed',
                   candidate=str(P), review=pin(REVIEW), peer=pin(PEER),
                   freeze=pin(P / 'FROZEN_SOURCE183.json'), archive=pin(archive),
                   actual_archive_members=len(members), actual_source_members=len(sources),
                   prepared=prepared, prepare_stdout=completed.stdout,
                   NPU_used=False, Torch_imported=False, formal_Goodput=False, adopted=False))
print(json.dumps({'review': pin(REVIEW), 'prepared': pin(PREPARED),
                  'freeze': pin(P / 'FROZEN_SOURCE183.json'), 'archive': pin(archive),
                  'members': len(members), 'NPU_used': False}))
