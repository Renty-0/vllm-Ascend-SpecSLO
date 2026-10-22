"""ROOT source review/freeze: current N16 four-token bonus verdict adapter."""
import ast, datetime, hashlib, importlib.util, json, os, subprocess, sys, tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-causal80-fused-bonus-verdict-full-v193';G=D/'v193-root-execution'
READY=D/'V193_FUSED_BONUS_VERDICT_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json'
PEER=D/'V193_INDEPENDENT_SOURCE_PEER_REVIEW.json'
REVIEW=D/'ROOT_V193_FUSED_BONUS_SOURCE_AND_GUARD_REVIEW.json'
PREPARED=D/'ROOT_V193_PREPARED_SOURCE_HOST_AUDIT.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def pin(p):return dict(path=str(p),sha256=sha(p))
def save(p,r):
 with p.open('x')as f:json.dump(r,f,indent=2,sort_keys=True);f.write('\n')
assert sys.version_info[:2]==(3,11) and 'torch' not in sys.modules and 'torch_npu' not in sys.modules
assert sha(READY)=='f8d6591145acc566cc8cc63d4c44fee9524db2021ab1f9704ec70c2b579402bc'
assert sha(PEER)=='d10140d7166955ad3430341917615016866dd9f73bbda6e3127e824c626e08f5'
ready,peer=read(READY),read(PEER)
assert ready['status']=='source_HOST_ready_for_ROOT_review' and ready['frozen'] is False
assert peer['status'].replace('-','_')=='passed_source_review' and peer['candidate']==str(P)
sys.path.insert(0,str(P));spec=importlib.util.spec_from_file_location('_ROOT_V193_source_guard',G/'guard193.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
state=guard.source_state(archive=True);guards=guard.guard_inputs()
flat=guard.merge(state['local'],state['external'],state['evidence'],guards,{str(guard.CPU):sha(guard.CPU)})
assert state['local']==ready['files'] and state['external']==ready['external_files']
assert state['evidence']==ready['retained_HOST_evidence'] and guards==ready['execution_guard_inputs']
assert (len(state['local']),len(state['external']),len(state['evidence']),len(flat))==(25,167,177,374)
cold=ready['cold_actual_guard'];assert sha(cold['path'])==cold['sha256']
assert ready['reviewed_inputs']==guard.merge(flat,{cold['path']:cold['sha256']})
for path,h in flat.items():
 p=Path(path);assert p.is_file() and not p.is_symlink() and sha(p)==h,p
for path in state['local']:
 if path.endswith('.py'):ast.parse(Path(path).read_text(),filename=path)
cpu=read(guard.CPU)
assert cpu['status']=='passed' and cpu['tests_run']==4 and cpu['errors']==cpu['failures']==0
assert cpu['Torch_imported'] is cpu['NPU_used'] is cpu['model_constructed'] is False
assert cpu['HOST_interpreter_not_operator_qualification'] is True and cpu['original_parent_suite_rerun'] is False
assert len(list(P.glob('HOST_ATTEMPT193_*.json')))==5 and ready['counts']['projected_original173']==16
startup=(G/'startup193.sh').read_text()
assert startup.count('export VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION=0')==1
assert startup.count('export VLLM_ASCEND_SPECRHYTHM_TARGET_SLO_FIRST=0')==1
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
save(REVIEW,dict(schema='ROOT.V193.current-fused-bonus-source-and-guard-review.v1',status='passed_source_review',
 candidate=str(P),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),ready=pin(READY),peer=pin(PEER),
 reviewed_inputs=guard.merge(flat,{str(PEER):sha(PEER)}),cold_source_only=ready['cold_actual_guard'],
 cold_archiveFalse_explicitly_closed_by_ROOT_full_parent_archive_checks=True,
 complete_parent_archives_verified_in_place=state['parent_archive'],CPU_tests=4,CPU_elapsed_seconds=cpu['elapsed_seconds'],
 original_parent_suites_not_rerun=True,Torch_imported=False,NPU_used=False,
 source_findings=[
  'Sole production parent is frozen173. Sixteen complete source projections/inverses retain cap4/highN, trueB5/lowN, active32, normal/eager96/global192, causal80 mapper and original no-reservation whole-prompt admission. No190 cohort,191 SLO,192 mixed chunk,194 shape extension or187 union is combined.',
  'Fast scope is only current normal N16/B4/Q80 causal M80/S16 greedy work. The observer retains actual original runner input references across the natural cold first/changed/restored sequence and restores the exact old instance hook in finally. The returned restored input and current target successors form the operator input; no HOST candidate upload or next-proposal row substitution is used.',
  'The existing original fixed_greedy_full_window_bonus_verdict receives real target successor[16,5], a contiguous D2D current packed[:,1:5] draft[16,4], and allTrue bonus mask. Its exact integer accept0..4 plus correction/frontier maps to original five token/four path payload. The nonbonus kernel loses the fullaccept frontier and is explicitly rejected.',
  'Generated source has only two AST replacements: queue immediately after original target_window_observation assignment, then delegate original leader-only verdict. Original target_ended/target phase/window predictor, model, proposal exchange, finite/structural all4 vote, preflight/compaction/ledger, per-cycle clock, publication/callbacks and output truncation stay in order.',
  'Original persistent pinned allocator is reused separately. Pending identity binds generation/cycle/ordered owner/ticket/proposal/home/prefix/actual current row and plan object, plus original capture binding. Staging cannot be overwritten. Read occurs after actual copy event synchronization and exactly-once consumption; failed pending events keep ownership and reject the job.',
  'All4 always broadcast one identical status/token/path/time packet, fast or generic. The common status word adds one int64 and no collective. Leader queue/finish/packet exceptions publish rejection at the same collective and all ranks fail before original commit; no rank-dependent packet length or silent numerical fallback is introduced.',
  'LowN trueB5 and heterogeneous plans take the unchanged original HOST verifier with the new universal status framing. New HOST checks execute actual Native verifier through actual generic finish and packet with accepted nontrivial B5 node4 plus mixedB3/B4; classification is separately counted, not inferred from source retention.',
  'Before production permission, the new engine runs original201/44/114 startup, full527 B4 math/protocol and current M80 greedy/hidden FTF, allphysical hidden/head/all128 queryKV, original commit, own nextdraft and two complete zero traces. Twelve current-model N16 comparisons use independent sameM eager model/verifier before actual original commit. Original NPU BONUS semantics also cover mismatch0..3/fullaccept, zero/duplicates/EOS; fabricated integer operator fixtures are labelled separately from natural model outputs.',
  'Private issuer binds actual current engine/model component and original operator identity only after real cache release and before seal. Original M80 permission, naturally validated greedy/hidden entry objects and existing standalone scope remain necessary. No Native qualification flag, global gamma/linear mode or bonus config is spoofed; no new Native model key is introduced.',
  'Every actual trace retains before/after/delta and pending-empty proof. Each rank successful packet count equals actual normal target calls; leader queued+fallback equals calls and queued equals consumed. Actual B5 and heterogeneous fallback counts equal categories from this same retained raw trace, and full463 requires consumed/B5/heterogeneous allpositive with zero queue/finish failures. Prior173 counts only motivated the test and grant no current coverage.',
  'Original max8 prewarm convergence, warm65, two full21zero65 traces, one seal and original463 Poisson arrivals/256 outputs/Native and outer clocks remain. Full output metrics are recomputed using original code and true sha binding; source readiness does not imply Goodput or independent fullAR equality.',
  'All five HOST attempts, initial failed sources/logs and previous passed CPU/PREPARED/cold proof are retained. HOST executes the original Triton integer body on explicit HOST carriers; it imports no Torch and grants no NPU qualification. ROOT now fully verifies BASE/P160/P166 and production173 archives in place, unlike the intentionally archiveFalse cold result.',
  'Original ROOT selflock, baseline prepost verifier and owned cleanup retain7200s/256MiB/512MiB limits. Exact frozen177 temporary-race helper remains scoped to this private actual path. Models and datasets stay public; other artifacts stay private.',
 ],limitations=[
  'Small integer D2H savings can be outweighed by D2D/kernel/event and HOST validation work. No gain is predicted before actual full463.',
  'Current NPU operator, natural model/commit/next, packet fallback and pinned lifetime tests can still reject. Native21zero alone does not qualify the new non-graph kernel, hence separate current proof and counters are mandatory.',
  'Inherited cross-M behavior is classified explicitly as compatible or numerical variant. Neither finite components nor this source freeze prove whole-workload independent AR equality, adoption or final goal completion.',
 ],formal_Goodput=False,adopted=False))
completed=subprocess.run([sys.executable,'-B',str(G/'guard193.py'),'--prepare-only','--root-review',str(REVIEW),
 '--expected-root-review-sha256',sha(REVIEW),'--peer-review',str(PEER),sha(PEER)],check=True,text=True,
 stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
frozen,sources,prepared=guard.prepared();archive=P/'PREPARED_SOURCE_ARCHIVE193.tar.gz'
with tarfile.open(archive,'r:gz')as stream:
 members=stream.getmembers();assert len(members)==len(sources)==377
 for m in members:assert m.isfile() and hashlib.sha256(stream.extractfile(m).read()).hexdigest()==sources['/'+m.name]
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
save(PREPARED,dict(schema='ROOT.V193.prepared-source-HOST-audit.v1',status='passed',candidate=str(P),
 review=pin(REVIEW),peer=pin(PEER),freeze=pin(P/'FROZEN_SOURCE193.json'),archive=pin(archive),
 actual_archive_members=len(members),actual_source_members=len(sources),prepared=prepared,
 prepare_stdout=completed.stdout,NPU_used=False,Torch_imported=False,formal_Goodput=False,adopted=False))
print(json.dumps(dict(review=pin(REVIEW),prepared=pin(PREPARED),freeze=pin(P/'FROZEN_SOURCE193.json'),
 archive=pin(archive),members=len(members),NPU_used=False)))
