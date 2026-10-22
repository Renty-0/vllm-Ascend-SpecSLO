"""ROOT source review and immutable preparation of cap4 vacancy-prefill."""
import argparse,ast,datetime,hashlib,importlib.util,json,os,subprocess,sys,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-vacancy-cohort-prefill-cap4-active32-full-v190';G=D/'v190-root-execution'
READY=D/'V190_VACANCY_COHORT_PREFILL_CAP4_ACTIVE32_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json'
PEER=D/'V190_INDEPENDENT_SOURCE_PEER_REVIEW.json'
REVIEW=D/'ROOT_V190_CAP4_VACANCY_PREFILL_SOURCE_AND_GUARD_REVIEW.json'
PREPARED=D/'ROOT_V190_PREPARED_SOURCE_HOST_AUDIT.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def pin(p):return dict(path=str(p),sha256=sha(p))
def save(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
parser=argparse.ArgumentParser();parser.add_argument('--expected-peer-sha256',required=True);args=parser.parse_args()
assert sys.version_info[:2]==(3,11) and 'torch' not in sys.modules and 'torch_npu' not in sys.modules
assert sha(READY)=='18e2af7cd53eb89c8ec9c9863ac28515c7bbfa124da1797d8d1ef0ca214aae2f'
assert sha(PEER)==args.expected_peer_sha256
ready,peer=read(READY),read(PEER)
assert ready['source_stopped_writing'] is True and peer['candidate']==str(P)
assert peer['status'].replace('-','_')=='passed_source_review'
sys.path.insert(0,str(P));spec=importlib.util.spec_from_file_location('_ROOT_V190_source_guard',G/'guard190.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
state=guard.source_state(archive=True);guards=guard.guard_inputs()
flat=guard.merge(state['local'],state['external'],state['evidence'],guards,{str(guard.CPU):sha(guard.CPU)})
assert state['local']==ready['files'] and state['external']==ready['external_files']
assert state['evidence']==ready['retained_HOST_evidence'] and guards==ready['execution_guard_inputs']
assert (len(state['local']),len(state['external']),len(state['evidence']),len(flat))==(21,185,57,268)
for path,h in flat.items():
 p=Path(path);assert p.is_file() and not p.is_symlink() and sha(p)==h,p
for path in state['local']:
 if path.endswith('.py'):ast.parse(Path(path).read_text(),filename=path)
cpu=read(guard.CPU)
assert cpu['status']=='passed' and cpu['tests_run']==4 and cpu['errors']==cpu['failures']==0
assert cpu['Torch_imported'] is cpu['NPU_used'] is cpu['model_constructed'] is cpu['model_math_qualified'] is False
assert cpu['child_Torch_CPU_control_only'] is True and cpu['old_suites_rerun'] is False
assert len(list(P.glob('HOST_ATTEMPT190_*.json')))==2
assert ready['counts']['whole_source_projections']==11
for path,h in ready['readonly_185_core_and_original_two_HOST_pins'].items():assert sha(path)==h
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
save(REVIEW,dict(schema='ROOT.V190.cap4-vacancy-prefill-source-and-guard-review.v1',status='passed_source_review',
 candidate=str(P),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),ready=pin(READY),peer=pin(PEER),
 reviewed_inputs=guard.merge(flat,{str(PEER):sha(PEER)}),CPU_tests=4,CPU_elapsed_seconds=cpu['elapsed_seconds'],
 original_parent_suites_not_rerun=True,Torch_imported=False,NPU_used=False,
 CPU_child_actual_Torch_control_with_synthetic_model_boundary=True,
 source_findings=[
  'Sole production parent is completed173: original high-N cap4/low-N trueB5, active32, normal/eager96/global192, causal80/M80/S16 optimization and original model math remain. This candidate omits cap3/union187/SLO189/metadata183.',
  'Only fresh constructor fields coalesce1/0 become4/150. The exact reviewed185 cohort/lifecycle/source_patch modules are readonly pinned imports, with their original five core tests and two source-HOST results retained rather than rerun. The complete live decoder AST inverse preserves original arrivals, output256, publication, accounting, decode_start, TPOT and TTFT clocks.',
  'Vacancy episode uses the existing broadcast leader now and actual already-ready owners; full/empty/no-ready/impossible-tail/token-budget behavior remains explicit. Four is not a hard maximum;150ms is poll checked and is not an end-to-end TTFT guarantee. Both SLO flags remain0.',
  'Current same-engine173 original B4 and causal80 actors run first. Fresh new multiowner2/4 eager FIA cases allocate independent full-prompt reference rows and a protected incumbent, compare all physical hidden, all56 draft or128 target queryKV and same-N last-token heads. No candidate KV copy or singleton201 qualification transfer occurs.',
  'Original root append/publication AST uses genuine NativeSamplingParams/PearlPipelineState/runtime state and original callback, exactly once on target leader. Original callback fields and model.forward attribute are restored after component observation.',
  'next4_190 executes real FastV5 normalB4 and an independent original PA first step plus3 autoregressive model steps on its own prefix. The complete encoded[N,4,2]/raw8 probabilities, mean8 confidence, root+3 ledger and56 queryKV tensors are checked. It does not relabel or trim a three-step output.',
  'Fresh prefilling has no missing committed suffix. Reference uses actual full prompt and published root; candidate no-repair four-step route and any actual root-seed route retain original source conditions and observer checks. Current graph objects/dictionaries/qualified flags must remain unchanged.',
  'New HOST runs actual CPU Torch only in an explicitly isolated child with original NativePrefixCache, NativeMetadata, tree and chain_metadata producers. Its model boundary is synthetic and tests ABI/control only; the outer process forbids backend imports. Actual four calls/tail3/raw8/own slots/ledger and an old3-stage negative were checked, never reported as model numerical qualification.',
  'New current six cases require3 passes and336 Native zero scalar subtractions, then a private issuer binds current original173 plus current multiowner/next4 audit identity. Original startup201/44/114, full527 B4, actual causal80 qualification/protocol/next, warm fixedpoint, two complete original21zero traces, seal and full463 remain compulsory.',
  'Actual generation admission events are retained after original metric retention and independently scalar replayed across all4 ranks. Cold parent archives are verified in place; all2 HOST attempts,54 source/log paths and exact11 source projections are bound.',
  'Original selflocked guardian, baseline prepost verification and owned cleanup retain7200s/256MiB/512MiB limits. The exact frozen177 storage helper handles only temporary publication races under this private actual path; no public artifact relocation occurs.',
 ],limitations=[
  'New numerical representatives cover2/4 owners, not all32-owner prompt tuples or all transient workspace peaks. Actual reached cohort/query sizes must be retained and reported.',
  'Source/HOST evidence grants no current NPU capability, Goodput, whole-workload AR equivalence, adoption or goal completion. Required current device gates can still fail.',
 ],formal_Goodput=False,adopted=False))
completed=subprocess.run([sys.executable,'-B',str(G/'guard190.py'),'--prepare-only','--root-review',str(REVIEW),
 '--expected-root-review-sha256',sha(REVIEW),'--peer-review',str(PEER),sha(PEER)],check=True,text=True,
 stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
frozen,sources,prepared=guard.prepared();archive=P/'PREPARED_SOURCE_ARCHIVE190.tar.gz'
with tarfile.open(archive,'r:gz')as stream:
 members=stream.getmembers();assert len(members)==len(sources)==271
 for member in members:
  assert member.isfile() and hashlib.sha256(stream.extractfile(member).read()).hexdigest()==sources['/'+member.name]
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
save(PREPARED,dict(schema='ROOT.V190.prepared-source-HOST-audit.v1',status='passed',candidate=str(P),
 review=pin(REVIEW),peer=pin(PEER),freeze=pin(P/'FROZEN_SOURCE190.json'),archive=pin(archive),
 actual_archive_members=len(members),actual_source_members=len(sources),prepared=prepared,
 prepare_stdout=completed.stdout,NPU_used=False,Torch_imported=False,formal_Goodput=False,adopted=False))
print(json.dumps(dict(review=pin(REVIEW),prepared=pin(PREPARED),freeze=pin(P/'FROZEN_SOURCE190.json'),
 archive=pin(archive),members=len(members),NPU_used=False)))
