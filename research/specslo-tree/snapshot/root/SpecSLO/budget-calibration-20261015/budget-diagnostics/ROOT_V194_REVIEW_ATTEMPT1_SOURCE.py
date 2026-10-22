"""ROOT source-only review and freeze for the new causal84 variable-Q family."""
import ast, datetime, hashlib, importlib.util, json, os, subprocess, sys, tarfile
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
P = B / 'tree-causal84-variable-b4-cohort4-active32-full-v194'
G = D / 'v194-root-execution'
READY = D / 'V194_CAUSAL84_VARIABLE_Q_COHORT4_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW_REVISION2.json'
PEER = D / 'V194_INDEPENDENT_SOURCE_PEER_REVIEW.json'
REVIEW = D / 'ROOT_V194_CAUSAL84_VARIABLE_Q_SOURCE_AND_GUARD_REVIEW.json'
PREPARED = D / 'ROOT_V194_PREPARED_SOURCE_HOST_AUDIT.json'

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

assert sys.version_info[:2] == (3, 11)
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
assert sha(READY) == '9c2315cd18a51d485473a413c92e65974791b8b38670be2821b4a48b70872e47'
assert sha(PEER) == 'bd246e4d0516b03c466664699be9a703337d956b6440e472a0a7978fd7518063'
ready, peer = read(READY), read(PEER)
assert ready['status'] == 'ready_for_independent_source_review'
assert ready['freeze_created'] is False
assert peer['status'] == 'passed_source_review' and peer['candidate'] == str(P)
assert peer['remaining_blockers'] == []
sys.path.insert(0, str(P))
spec = importlib.util.spec_from_file_location('_ROOT_V194_actual_source_guard', G / 'guard194.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
state = guard.source_state(archive=True)
guards = guard.guard_inputs()
flat = guard.merge(state['local'], state['external'], state['evidence'], guards,
                   {str(guard.CPU): sha(guard.CPU)})
assert state['local'] == ready['local_sources']
assert state['external'] == ready['selected_readonly_sources']
assert state['evidence'] == ready['retained_HOST_evidence']
assert guards == ready['execution_guard_inputs']
assert flat == ready['reviewed_inputs_before_peer']
assert (len(state['local']), len(state['external']), len(state['evidence']), len(flat)) == (23, 272, 213, 513)
for path, expected in flat.items():
    path = Path(path)
    assert path.is_file() and not path.is_symlink() and sha(path) == expected, path
for path in state['local']:
    if path.endswith('.py'):
        ast.parse(Path(path).read_text(), filename=path)
cpu = read(guard.CPU)
assert cpu['status'] == 'passed' and cpu['tests_run'] == 4 and cpu['errors'] == cpu['failures'] == 0
assert cpu['Torch_imported'] is cpu['NPU_used'] is cpu['model_constructed'] is False
assert cpu['old_suites_rerun'] is cpu['model_math_qualified'] is False
assert sha(guard.CPU) == ready['CPU']['sha256']
assert len(list(P.glob('HOST_ATTEMPT194_*.json'))) == 5
cold = ready['cold_guard']
assert sha(cold['path']) == cold['sha256']
startup = (G / 'startup194.sh').read_text()
assert startup.count('export VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION=0') == 1
assert startup.count('export VLLM_ASCEND_SPECRHYTHM_TARGET_SLO_FIRST=0') == 1
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules

save(REVIEW, dict(
    schema='ROOT.V194.current-causal84-variable-Q-source-and-guard-review.v1',
    status='passed_source_review', candidate=str(P),
    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), ready=pin(READY), peer=pin(PEER),
    reviewed_inputs=guard.merge(flat, {str(PEER): sha(PEER)}),
    complete_parent_archives_verified_in_place=state['parent_archive'],
    cold_source_only=cold, CPU_tests=4, CPU_elapsed_seconds=cpu['elapsed_seconds'],
    Torch_imported=False, NPU_used=False, original_parent_suites_not_rerun=True,
    source_findings=[
        'Sole production parent is completed and frozen190. Its cap4/highN, trueB5/lowN, active32, normal96/eager96/global192, whole-prompt vacancy cohort4/150 and SLO0/TARGET0 remain. No191 SLO,192 real-chunk cursor,193 fused-verdict,171 active64 or187 union execution is combined.',
        'The new mapper accepts only actual greedy causal contiguous linear B1..4 work with sixteen unique real owners and real Q70..79. It selects M84/S18 with two disjoint scratch chains of lengths2..9. All other cases delegate the original parent pad, including Q80 retaining exact original M80/S16. Candidate budgets, priorities and encoded proposals are unchanged.',
        'Bank194 retains the original independent candidate16/reference16 and44 disjoint dummy rows. The original128 reference remains independent. New84 uses two candidate and two separate reference scratch owners; actual108/544 pages and full527 storage are retained. No candidate KV is copied into the reference.',
        'The bounded current actor exercises Q70/73/77/79, three fresh transactions each and four original protocol cycles per transaction. Actual allphysical hidden, original model head at M84, every query KV, finite preserved prefixes, original verdict/compaction/commit and own repaired nextdraft are required. Separate original128 model work records cross-M token/verdict differences; it is not the same-M oracle.',
        'Both new Native greedy/hidden M84/S18 entries require natural first/changed/restored model calls and original automatic qualification. The cold greedy adapter already exists in production173 and checks all three real forwards. No Native validation flag or sealed entry is assigned. Twelve full traces include eight complete zero traces and672 actual21-counter zero subtractions across four ranks.',
        'The private permission requires this exact current engine, its current190 component and current84 proof, plus current naturally validated new Native objects. Bounded qualification tokens are distinct from production permission; the new mapper is unavailable before the current qualification. Exact inherited entry identity and disabled sets remain unchanged.',
        'Peer correction is in the final source: overall old-padding compatibility is strict parent80 AND new84 compatibility. Aggregate computation and permission issue occur inside the same existing shared stage, adding no collective. Component, scope, audit and supervisor retain the same classification. A cross-M numerical variant cannot be mislabeled compatible.',
        'New HOST negatives exercise parent-compatible/new-incompatible and invalid classification types before the issuer. HOST geometry, whole-source inverses, actual component/scope/audit bindings and guard source checks import no Torch and prove no NPU model mathematics. All five attempts, one failure, earlier successful proofs and source/log snapshots are retained.',
        'Before the new84 actor, current workers must complete original201/44/114 startup, current original full527 B4, current causal80 and original190 two/four-owner eager-prefill full-prompt proofs with their own model/normal publication/nextdraft evidence. Frozen parent receipts cannot substitute for these current checks.',
        'Production preserves original Native code, proposal and all4 protocol, finite votes, normal TreeCommitFence and per-cycle output clocks. Original max8 convergence, warm65, two zero65, seal and full original463 Poisson requests/256 outputs follow the new component. Raw mapper geometry and every full metric are recomputed by the original arithmetic.',
        'ROOT verifies the original BASE/P160 and completed190 archive member bytes in place, including all271 parent190 members. This private archive includes only explicitly selected current source, guard, HOST evidence and reviews; old binary archives are not recopied.',
        'Original ROOT selflock, baseline pre/post evidence, owned-process cleanup,7200second limit and256MiB per-tag/512MiB aggregate limits remain. Frozen177 temporary-race handling is restricted to this private actual directory. Models/datasets may remain public; code and all experiment artifacts remain private.',
    ],
    limitations=[
        'The completed190 regression motivates this shape test; it does not isolate causal gain. Smaller physical work may still lose to submission, graph, draft or prefill costs. No performance estimate is a measured result.',
        'This is source/HOST readiness only. Current actual NPU component, natural keys, all4 lifecycle and full463 may still fail. Cross-M behavior can remain a numerical variant despite passing independent same-M checks.',
        'Finite component checks do not establish whole-workload independent autoregressive equality or adoption. Historical fixed and TP4 Goodput clocks differ; final improvement claims need fresh common-clock same-card measurements.',
    ], formal_Goodput=False, adopted=False,
))

completed = subprocess.run([
    sys.executable, '-B', str(G / 'guard194.py'), '--prepare-only',
    '--root-review', str(REVIEW), '--expected-root-review-sha256', sha(REVIEW),
    '--peer-review', str(PEER), sha(PEER),
], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
frozen, sources, prepared = guard.prepared()
archive = P / 'PREPARED_SOURCE_ARCHIVE194.tar.gz'
with tarfile.open(archive, 'r:gz') as stream:
    members = stream.getmembers()
    assert len(members) == len(sources) == 516
    assert len({member.name for member in members}) == len(members)
    for member in members:
        assert member.isfile()
        assert hashlib.sha256(stream.extractfile(member).read()).hexdigest() == sources['/' + member.name]
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
save(PREPARED, dict(
    schema='ROOT.V194.prepared-source-HOST-audit.v1', status='passed', candidate=str(P),
    review=pin(REVIEW), peer=pin(PEER), freeze=pin(P / 'FROZEN_SOURCE194.json'), archive=pin(archive),
    actual_archive_members=len(members), actual_source_members=len(sources), prepared=prepared,
    prepare_stdout=completed.stdout, NPU_used=False, Torch_imported=False,
    formal_Goodput=False, adopted=False,
))
print(json.dumps(dict(review=pin(REVIEW), prepared=pin(PREPARED),
                      freeze=pin(P / 'FROZEN_SOURCE194.json'), archive=pin(archive),
                      members=len(members), NPU_used=False)))
