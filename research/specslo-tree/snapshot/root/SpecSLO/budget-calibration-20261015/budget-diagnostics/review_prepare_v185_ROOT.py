"""ROOT source-only review and freeze of the vacancy-cohort experiment."""
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
P = B / 'tree-vacancy-cohort-prefill-full-v185'
G = D / 'v185-root-execution'
READY = D / 'V185_VACANCY_COHORT_PREFILL_FULL_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json'
PEER = D / 'V185_INDEPENDENT_SOURCE_PEER_REVIEW.json'
REVIEW = D / 'ROOT_V185_VACANCY_COHORT_PREFILL_FULL_SOURCE_AND_GUARD_REVIEW.json'
PREPARED = D / 'ROOT_V185_PREPARED_SOURCE_HOST_AUDIT.json'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def pin(p):
    return dict(path=str(p), sha256=sha(p))

def save(p, value):
    with p.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')

parser = argparse.ArgumentParser()
parser.add_argument('--expected-peer-sha256', required=True)
args = parser.parse_args()
assert sys.version_info[:2] == (3, 11)
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
assert sha(READY) == '70344b4cbf4399fe4e42c3ba424c50865d446bd4eebfd9020dc38647ca7aefc4'
assert sha(PEER) == args.expected_peer_sha256
ready, peer = read(READY), read(PEER)
assert peer['status'].replace('-', '_') == 'passed_source_review'
assert peer['candidate'] == str(P) and ready['source_stopped'] is True
sys.path.insert(0, str(P))
spec = importlib.util.spec_from_file_location('_ROOT_V185_source_guard', G / 'guard185.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
state = guard.source_state(archive=True)
guard_inputs = guard.guard_inputs()
flat = guard.merge(state['local'], state['external'], state['evidence'], guard_inputs,
                   {str(P / 'CPU_RESULT185.json'): sha(P / 'CPU_RESULT185.json')})
assert state['local'] == ready['source_files']
assert state['external'] == ready['external_files']
assert state['evidence'] == ready['retained_HOST_evidence']
assert guard_inputs == ready['execution_guard_inputs']
assert (len(state['local']), len(state['external']), len(state['evidence']), len(flat)) == (25, 250, 129, 409)
for path, expected in flat.items():
    p = Path(path)
    assert p.is_file() and not p.is_symlink() and sha(p) == expected, p
for path in state['local']:
    if path.endswith('.py'):
        ast.parse(Path(path).read_text(), filename=path)
cpu = read(P / 'CPU_RESULT185.json')
assert cpu['status'] == 'passed' and cpu['tests_run'] == 4
assert cpu['failures'] == cpu['errors'] == 0
assert cpu['Torch_imported'] is cpu['NPU_used'] is False
assert ready['counts']['whole_source_projections'] == 7
assert len(list(P.glob('HOST_ATTEMPT185_*.json'))) == 4
assert sha(P / 'source_patch185.py') == 'c61f0844fcbfd2bbe86628e297927566038de01e796d43db411c3126b38ca030'
assert sha(P / 'lifecycle185.py') == 'a3cf057b179abbc39862f3c35a981fc521e76e9fca40b4ef1a681dab7c3fad97'
assert sha(P / 'host_source_patch185.py') == '2160d25985cdb56a4784b2ab69f86b64c07b5e68f7d36e6ee741f4e4de50bd69'
assert sha(P / 'CORE_HOST_RESULT185.json') == 'a27225355daa209366505aefe458b93ba6a1ea8bdec6d3d94ae17c224231b96b'
assert sha(P / 'SOURCE_PATCH_HOST_RESULT185.json') == '0b0aaea3f4fe5765d66b7fce7fe3b185b4fd7b7f83edf88e984ff2b127e239b1'
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
save(REVIEW, dict(
    schema='ROOT.V185.vacancy-cohort-prefill-full-source-and-guard-review.v1',
    status='passed_source_review', candidate=str(P),
    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    ready=pin(READY), peer=pin(PEER),
    reviewed_inputs=guard.merge(flat, {str(PEER): sha(PEER)}),
    prior_narrow_review=pin(D / 'ROOT_V185_UNINSTALLED_PATCH_HOST_REVIEW.json'),
    local_source_files=25, selected_readonly_inputs=250, retained_HOST_evidence=129,
    CPU_tests=4, CPU_elapsed_seconds=cpu['elapsed_seconds'],
    old_suites_not_rerun=True, NPU_used=False, Torch_imported=False,
    source_findings=[
        'Frozen P180 remains unchanged. Only fresh configuration coalesce1/0 becomes4/150 and the reversible actual live decoder admission decision/lifetime changes. Active32, normal/eager96/global192, high-N cap3/low-N true B5 and original target geometry remain.',
        'Vacancy episode begins when the actual oldest ready owner and capacity coexist, using the existing broadcast leader timestamp. It does not use old arrival age. Original arrival timestamps, output length, elapsed fields and TTFT/TPOT definitions are unchanged.',
        'Empty active set, sufficient ready owners, token-budget saturation, impossible finite tail and missing arrival have explicit immediate rules. Four is a batching threshold rather than a hard maximum. The150ms bound is checked at polls, not claimed as an end-to-end latency bound.',
        'Already reviewed cohort/source_patch/lifecycle and both earlier HOST results remain byte-identical. Whole decoder AST inverse preserves original all-admission, prefill, root publication, decode_start and accounting statements. Original reserved-slot full predicate is retained and current standalone capability excludes reservations.',
        'Fresh per-generation state has unconditional finally cleanup, retained exact original exceptions and separate complete/partial history. Retained production decisions are independently scalar-replayed from actual inputs across all four ranks after original measurement retention.',
        'Fresh current math uses complete distinct-length original prompts at genuine2/4-owner eager FIA shapes, independent allocated reference owners, all physical hidden/56 draft or128 target query KV, protected incumbent prefix and the original model metadata producer. Candidate KV is never copied into the reference.',
        'Target independent greedy head has the same cohort N as the original last-token helper. Original root append and first-token publication AST execute with real Native pipeline/runtime state and callback; actual root then feeds the original independent three-step draft180 producer and full encoded/KV/ledger proof.',
        'The genuine model forward returns a tensor; forward capture retains actual three positional inputs and restores the original attribute. Original cache_slot_mapping returns a list passed directly to the original make_attention_metadata. The actor Bank implements every actual frozen draft180 read.',
        'All six current cases retain actual metadata/page/owner invariants and two complete21-counter zero sweeps per cohort,336 scalar subtractions. All actual Native entry dictionaries and objects must remain unchanged and runtime-qualified; no singleton201 receipt authorizes new multiowner numerical math.',
        'Current same-engine issuer requires both original P180 component capability and current multiowner audit identity. Original201+44 target/114 draft, full527 B4, complete current16-N compact geometry, bounded warm fixedpoint, two full zero traces, seal and full463 remain mandatory.',
        'Original selflocked guardian and exact pre/post baseline verification/owned cleanup are retained, with scoped frozen177 temporary-file storage handling and7200s/256MiB/512MiB limits. All four HOST attempts and preparation source snapshots remain pinned.',
    ],
    limitations=[
        'Representative2/4-owner math does not establish exhaustive32-owner packed tensor or transient workspace qualification. Actual reached cohort/query sizes and allocation evidence must be reported.',
        'The standalone candidate omits union187 and persistent-metadata183. No source or HOST result supplies current device arithmetic, whole-workload AR equivalence, Goodput, adoption or goal completion.',
    ], formal_Goodput=False, adopted=False,
))
completed = subprocess.run(
    [sys.executable, '-B', str(G / 'guard185.py'), '--prepare-only',
     '--root-review', str(REVIEW), '--expected-root-review-sha256', sha(REVIEW),
     '--peer-review', str(PEER), sha(PEER)],
    check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
frozen, sources, prepared = guard.prepared()
archive = P / 'PREPARED_SOURCE_ARCHIVE185.tar.gz'
with tarfile.open(archive, 'r:gz') as stream:
    members = stream.getmembers()
    assert len(members) == len(sources) == 412
    for member in members:
        assert member.isfile()
        assert hashlib.sha256(stream.extractfile(member).read()).hexdigest() == sources['/' + member.name]
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
save(PREPARED, dict(
    schema='ROOT.V185.prepared-source-HOST-audit.v1', status='passed',
    candidate=str(P), review=pin(REVIEW), peer=pin(PEER),
    freeze=pin(P / 'FROZEN_SOURCE185.json'), archive=pin(archive),
    actual_archive_members=len(members), actual_source_members=len(sources),
    prepared=prepared, prepare_stdout=completed.stdout,
    NPU_used=False, Torch_imported=False, formal_Goodput=False, adopted=False))
print(json.dumps(dict(review=pin(REVIEW), prepared=pin(PREPARED),
                     freeze=pin(P / 'FROZEN_SOURCE185.json'), archive=pin(archive),
                     members=len(members), NPU_used=False)))
