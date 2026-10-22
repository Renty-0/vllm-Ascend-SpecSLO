"""ROOT review/freeze of real-cursor M197/cap64; no model or old-suite work."""
import argparse, ast, datetime, hashlib, importlib.util, json, os, subprocess, sys, tarfile
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
P = B / 'tree-live-small-mixed197-cap64-active32-full-v192'
G = D / 'v192-root-execution'
READY = D / 'V192_M197_CAP64_REAL_CURSOR_ACTIVE32_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json'
PEER = D / 'V192_INDEPENDENT_SOURCE_PEER_REVIEW.json'
REVIEW = D / 'ROOT_V192_REAL_CURSOR197_CAP64_SOURCE_AND_GUARD_REVIEW.json'
PREPARED = D / 'ROOT_V192_PREPARED_SOURCE_HOST_AUDIT.json'
ARCHIVE_AUDIT = D / 'ROOT_V192_PARENT163_172_ARCHIVES_IN_PLACE_AUDIT_REVISION2.json'
SUPPLEMENT = D / 'V192_PARENT_ARCHIVE_REVISION2_INDEPENDENT_REVIEW.json'
REJECTION = D / 'ROOT_V192_PARENT_ARCHIVE_AUDIT_REV1_REJECTION.json'

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
parser.add_argument('--expected-supplement-sha256', required=True)
args = parser.parse_args()
assert sys.version_info[:2] == (3, 11)
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
assert sha(READY) == '118f19d51ebfd019e4a2da8715143696e4f1c19645a735f89797175db1e4528e'
assert sha(PEER) == args.expected_peer_sha256
assert sha(ARCHIVE_AUDIT) == 'bebb8bb1bd94ff92ef858b5d11cfd62778601e9abe7801775a7e2ede3cbf0f01'
assert sha(SUPPLEMENT) == args.expected_supplement_sha256
assert sha(REJECTION) == '5adb109fb1fdc50003a395c16e7d453d8b186c403e8ade650fca130eb2f0c5d3'
supplement = read(SUPPLEMENT)
assert supplement['status'].replace('-', '_') == 'passed_source_review' and supplement['candidate'] == str(P)
ready, peer, archive_audit = read(READY), read(PEER), read(ARCHIVE_AUDIT)
assert ready['source_writes_stopped'] is True and peer['candidate'] == str(P)
assert peer['status'].replace('-', '_') == 'passed_source_review'
assert archive_audit['status'] == 'passed' and archive_audit['candidate'] == str(P)
for report in [archive_audit['base_archive'], *archive_audit['parent_archives']]:
    assert report['all_member_SHA256_checked'] is True and sha(report['path']) == report['sha256']
    wanted = report['expected_member_map']
    with tarfile.open(report['path'], 'r:gz') as stream:
        members = stream.getmembers()
        assert len(members) == report['regular_members'] == len(wanted) == len({m.name for m in members})
        for m in members:
            assert m.isfile() and hashlib.sha256(stream.extractfile(m).read()).hexdigest() == wanted['/' + m.name.lstrip('/')]

sys.path.insert(0, str(P))
spec = importlib.util.spec_from_file_location('_ROOT_V192_source_guard', G / 'guard192.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
state = guard.source_state(archive=True)
guards = guard.guard_inputs()
flat = guard.merge(state['local'], state['external'], state['evidence'], guards, {str(guard.CPU): sha(guard.CPU)})
assert state['local'] == ready['local_files'] and state['external'] == ready['external_files']
assert state['evidence'] == ready['retained_HOST_evidence'] and guards == ready['guard_inputs']
assert (len(state['local']), len(state['external']), len(state['evidence']), len(flat)) == (39, 558, 171, 773)
for path, digest in flat.items():
    assert Path(path).is_file() and not Path(path).is_symlink() and sha(path) == digest, path
for path in state['local']:
    if path.endswith('.py'):
        ast.parse(Path(path).read_text(), filename=path)
cpu = read(guard.CPU)
assert cpu['status'] == 'passed' and cpu['actual_check_count'] == 4
assert cpu['Torch_imported'] is cpu['NPU_initialized'] is cpu['model_initialized'] is False
assert all(v['status'] == 'passed' for v in cpu['checks'])
assert ready['counts']['P172_whole_source_projections'] == 30
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules

save(REVIEW, dict(
    schema='ROOT.V192.real-cursor197-cap64-source-and-guard-review.v1', status='passed_source_review',
    candidate=str(P), UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    ready=pin(READY), peer=pin(PEER), archive_revision_peer=pin(SUPPLEMENT),
    reviewed_inputs=guard.merge(flat, {str(PEER): sha(PEER), str(SUPPLEMENT): sha(SUPPLEMENT)}),
    rejected_archive_audit_revision1=dict(**pin(REJECTION), complete_evidence=read(REJECTION)),
    independent_parent_archive_audit=dict(**pin(ARCHIVE_AUDIT), complete_evidence=archive_audit),
    corrections_to_READY=[
        'guard192.parent_archive invokes check192.parent_projection with its default archive=False. Its nested P163 label actually identifies frozen172, and does not independently verify the172 tar members. ROOT explicitly closes this gap by rechecking the complete BASE/P163/P172 archives in place and embedding their complete SHA/member audit here. All old bytes excluded from the new archive are covered; no source or device capability is changed.',
        'There is no max8 prewarm fixedpoint loop in phases192. The actual inherited172 schedule is one warm65, two complete zero65 traces, one seal, then full463. ROOT accepts this original fail-closed schedule and does not claim max8 or a fixedpoint loop. Either zero trace can reject the run.',
    ],
    source_findings=[
        'Production parent is frozen172 with its existing real typed chunk transaction. Thirty complete source projections/inverses are checked; fresh namespaces retain original finite guards, immutable full prompt lease, real prepared/submitted/commit/publication and original Native methods. No173 M80,185 cohort,189 SLO,187 union or183 metadata implementation is combined.',
        'Physical target M is197, prompt chunk cap64 is separate from FULL mask128. Every call computes min(remaining,64,197-Q-2*scratch_count); all1702 admissible aggregate N/C points stay positive,87 clip below64 and minimum is53. Physical S is19 or35. Original causal predicate, cached2048 int8 mask, full-tree bool mask, owner-slot maps and original buffer key producers are retained.',
        'Final generated decoder applies actual service window32 to both admission and publication. Reserved unfinished prompts occupy service slots; active plus reserved remains at most32. Configured64, queued463, storage527,64 scratch slots and independent diagnostic owners are distinct concepts. Pending463 is not a claim to463 active rows.',
        'Four current changed-path mathematical transactions cover causal S19 with a P1 final chunk, true-fork S19 with P1, S35 true-fork capacity53 with P1, and genuine whole2301 variable32/C96/Q128. Candidate and reference use disjoint original allocations and independently evaluated prefix histories, fullphysical197 hidden, all128 target queryKV, actual same-query head shapes, own draft prompt all56queryKV and real normal nextdraft.',
        'Each transaction keeps the genuine full immutable prompt and continuous committed cursor. Submitted and prepared states cannot publish an intermediate first token. The original TreeCommitFence and real normal finish/compaction commit move cursor exactly once. Only final valid target token plus current lease/vacancy can enter original publication after the cycle clock, exactly once.',
        'Current numerical actor runs original changed-input/restored FTF with actual64 task handles and counter subtraction, three complete own-history transactions, two full21zero sweeps, exact token/path/publication/nextdraft fingerprint stability, current Native entry identity and natural runtime validation. No flag is assigned and no old receipt issues the new private same-engine capability.',
        'The strict NativeGraphExecution serialized schema is exactly mode/fallback_reason/capture_attempted/replay_executed. The removed used_aclgraph lookup was a derived property absent from asdict, not a failed or missing execution verdict. No permissive default substitutes for actual replay and no fallback.',
        'The actual original constructor runs once with target buffer capacity1022 and draft512, same class/model and empty initial inventories, exact method restoration, original527 storage/544pages. Original MC2 22-shape profile already contains197; its current numeric/resource qualification remains required.',
        'Before new mixed actors, original201/44/114 startup and full527 normal-B4 actor must pass in this current engine. After actor permit/cache cleanup, private scope checks current engine/report/actual four Native families on every use. Required current warm65/twozero65/seal/full463 and original Poisson arrivals,256 outputs, clocks and accounting remain.',
        'Four new HOST checks execute real producer/transaction/TreeCommitFence/cursor/publication source contracts, actual patched active-plus-reserved admission and full original finite463 admission with genuine Native parameters, exchange ABI and real NativeGraphExecution dataclass. They use explicit HOST completion carriers and never claim model256 correctness. All initial failures, earlier passed proofs and source snapshots are retained.',
        'Original ROOT sole selflock, baseline pre/post verifier, owned worker cleanup and7200s/256MiB/512MiB limits stay. Exact frozen177 temporary-race helper is scoped only to this private live path. No non-model artifact enters public storage.',
    ],
    limitations=[
        'Finite numerical representatives and per-call checked structure do not numerically enumerate every budget tuple or token, nor establish full-workload independent AR equality.',
        'More64-token chunks can increase joins and normal rounds. Neither a latency gain nor a Goodput gain is predicted by source readiness.',
        'Actual model HBM/workspaces, current numerical gates, twozero traces and complete full463 can still fail. No performance result, adoption or goal completion follows from this source freeze.',
    ],
    CPU_tests=4, CPU_elapsed_seconds=cpu['elapsed_seconds'], old_suites_not_rerun=True,
    Torch_imported=False, NPU_used=False, formal_Goodput=False, adopted=False,
))

completed = subprocess.run([
    sys.executable, '-B', str(G / 'guard192.py'), '--prepare-only', '--root-review', str(REVIEW),
    '--expected-root-review-sha256', sha(REVIEW), '--peer-review', str(PEER), sha(PEER),
    '--peer-review', str(SUPPLEMENT), sha(SUPPLEMENT),
], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
frozen, sources, prepared = guard.prepared()
archive = P / 'PREPARED_SOURCE_ARCHIVE192.tar.gz'
with tarfile.open(archive, 'r:gz') as stream:
    members = stream.getmembers()
    assert len(members) == len(sources) == len({m.name for m in members})
    for member in members:
        assert member.isfile() and hashlib.sha256(stream.extractfile(member).read()).hexdigest() == sources['/' + member.name]
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
save(PREPARED, dict(schema='ROOT.V192.prepared-source-HOST-audit.v1', status='passed', candidate=str(P),
    review=pin(REVIEW), peer=pin(PEER), archive_revision_peer=pin(SUPPLEMENT),
    freeze=pin(P / 'FROZEN_SOURCE192.json'), archive=pin(archive),
    independent_parent_archive_audit=pin(ARCHIVE_AUDIT), parent_archive_audit_embedded_in_frozen_ROOT_review=True,
    actual_archive_members=len(members), actual_source_members=len(sources), prepared=prepared,
    actual_schedule='warm65/twozero65/seal/full463; no max8 loop', prepare_stdout=completed.stdout,
    NPU_used=False, Torch_imported=False, formal_Goodput=False, adopted=False))
print(json.dumps(dict(review=pin(REVIEW), prepared=pin(PREPARED), freeze=pin(P / 'FROZEN_SOURCE192.json'),
    archive=pin(archive), members=len(members), NPU_used=False)))
