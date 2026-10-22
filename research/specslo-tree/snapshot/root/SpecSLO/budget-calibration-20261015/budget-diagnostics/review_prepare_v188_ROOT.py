"""ROOT diagnostic-observer fix review and source preparation, no model run."""
import argparse,ast,datetime,hashlib,importlib.util,json,os,subprocess,sys,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-parent3-union3-padding-observer-devicefix-full-v188';G=D/'v188-root-execution'
READY=D/'V188_PADDING_OBSERVER_DEVICE_INDEX_FIX_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json'
PEER=D/'V188_INDEPENDENT_SOURCE_PEER_REVIEW.json'
REVIEW=D/'ROOT_V188_PADDING_OBSERVER_SOURCE_AND_GUARD_REVIEW.json'
PREPARED=D/'ROOT_V188_PREPARED_SOURCE_HOST_AUDIT.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def pin(p):return dict(path=str(p),sha256=sha(p))
def save(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
parser=argparse.ArgumentParser();parser.add_argument('--expected-peer-sha256',required=True);args=parser.parse_args()
assert sys.version_info[:2]==(3,11) and 'torch' not in sys.modules and 'torch_npu' not in sys.modules
assert sha(READY)=='95e68df8c5186a7bd4b10c0335f490ed62d5796259e5908607c2cc98f5a83c11'
assert sha(PEER)==args.expected_peer_sha256
ready,peer=read(READY),read(PEER)
assert ready['source_stopped_writing'] is True and peer['candidate']==str(P)
assert peer['status'].replace('-','_')=='passed_source_review'
sys.path.insert(0,str(P));spec=importlib.util.spec_from_file_location('_ROOT_V188_source_guard',G/'guard188.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
state=guard.source_state(archive=True);guards=guard.guard_inputs()
flat=guard.merge(state['local'],state['external'],state['evidence'],guards,{str(guard.CPU):sha(guard.CPU)})
assert state['local']==ready['files'] and state['external']==ready['external_files']
assert state['evidence']==ready['retained_HOST_evidence'] and guards==ready['execution_guard_inputs']
assert (len(state['local']),len(state['external']),len(state['evidence']),len(flat))==(39,810,137,991)
for path,h in flat.items():
 p=Path(path);assert p.is_file() and not p.is_symlink() and sha(p)==h,p
for path in state['local']:
 if path.endswith('.py'):ast.parse(Path(path).read_text(),filename=path)
cpu=read(guard.CPU)
assert cpu['status']=='passed' and cpu['tests_run']==2 and cpu['errors']==cpu['failures']==0
assert cpu['Torch_imported'] is cpu['NPU_used'] is cpu['model_constructed'] is False
assert len(list(P.glob('HOST_ATTEMPT188_*.json')))==3
from projection188 import verify_observer_projection
assert verify_observer_projection()['projected_sources']==34
assert ready['HOST_boundary']['corrected_cases']==ready['HOST_boundary']['old_observer_negative_cases']==12
assert ready['HOST_boundary']['old36_oracle_cases_rerun'] is False
binary=B/'tree-natural-positive-union-discovery-local-v154/PREPARED_SOURCE_ARCHIVE_V154.tar.gz'
assert str(binary) in state['external'] and sha(binary)==state['external'][str(binary)]
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
save(REVIEW,dict(schema='ROOT.V188.padding-observer-device-source-and-guard-review.v1',status='passed_source_review',
 candidate=str(P),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),ready=pin(READY),peer=pin(PEER),
 reviewed_inputs=guard.merge(flat,{str(PEER):sha(PEER)}),CPU_tests=2,CPU_elapsed_seconds=cpu['elapsed_seconds'],
 original_parent_suites_not_rerun=True,Torch_imported=False,NPU_used=False,external_binary_not_nested=pin(binary),
 source_findings=[
  'The actual187 failure was a component-only diagnostic after the original union hidden returned and was cloned. It used an NPU index on CPU context_lens in observer.inspect_indices; it was not a model arithmetic mismatch and not a7200second timeout. Frozen failed187/181 evidence is retained.',
  'Only model-adjacent edit is the diagnostic observer five-tensor permutation: each tensor now creates its own long index on value.device. Original inputs/positions/metadata supplied to model execution, real+dummy order, slot/owner/context guards, ledger, production policy/budget/clocks are unchanged. The prior pure187 oracle fix is retained.',
  'Both new HOST checks execute the actual corrected observer and original padding_reference.inspect with the actual frozen NativeMetadata dataclass. Twelve explicit mixed-device/cache carrier cases reject the old code and preserve corrected values/shape/dtype/device, authoritative owners/slots/padding and the unmodified original metadata. These are control/ABI checks, not model mathematics.',
  'The original36 oracle mixed-device cases and parent source contracts remain immutable proof inputs; no old suite is rerun and no old device receipt supplies current capability. All34 whole-source parent187 inverses and current guard/receipt constructors are source-bound; all3 new HOST attempts and133 preparation source/log paths are retained.',
  'Current original startup201/44/114, full527 B4, all3 real parent3/union3 representatives including interior/max, independent sameM allphysical hidden/heads/56KV/raw6-canonical8/ledger, natural positive/invalidations/real consume/commit/next remain required.',
  'Current private engine/base/issuer and exact Native graph objects must qualify naturally. Warm fixedpoint, two complete original21-zero traces, seal and original full463 follow current math; no first-representative partial record or failed187 control flow authorizes the new component.',
  'The original selflock/baseline prepost/owned cleanup/7200s/256MiB/512MiB bounds remain. Storage now delegates to the exact frozen177 temporary-file helper scoped only to this private actual root.',
  'Archive omission is one exact pinned inherited P154 binary tar only. It remains an external SHA-checked input and a verified parent187 regular archive member in place. The archive function rejects every other tar.gz omission and retains every selected source/evidence file. No public artifact transfer occurs.',
 ],limitations=[
  'The standalone source retains cap3 high-N and does not combine target180/metadata183/prefill190/SLO189. It is a diagnostic-fix backup and has not been selected for immediate NPU testing.',
  'No current numerical capability, whole-workload AR equivalence, Goodput, adoption or goal achievement follows from this review.',
 ],formal_Goodput=False,adopted=False))
completed=subprocess.run([sys.executable,'-B',str(G/'guard188.py'),'--prepare-only','--root-review',str(REVIEW),
 '--expected-root-review-sha256',sha(REVIEW),'--peer-review',str(PEER),sha(PEER)],check=True,text=True,
 stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
frozen,sources,prepared=guard.prepared();archive=P/'PREPARED_SOURCE_ARCHIVE188.tar.gz'
assert str(binary) not in sources
with tarfile.open(archive,'r:gz')as stream:
 members=stream.getmembers();assert len(members)==len(sources)==993
 for member in members:
  assert member.isfile() and hashlib.sha256(stream.extractfile(member).read()).hexdigest()==sources['/'+member.name]
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
save(PREPARED,dict(schema='ROOT.V188.prepared-source-HOST-audit.v1',status='passed',candidate=str(P),
 review=pin(REVIEW),peer=pin(PEER),freeze=pin(P/'FROZEN_SOURCE188.json'),archive=pin(archive),
 actual_archive_members=len(members),actual_source_members=len(sources),prepared=prepared,
 external_binary_not_nested=pin(binary),prepare_stdout=completed.stdout,NPU_used=False,Torch_imported=False,
 formal_Goodput=False,adopted=False))
print(json.dumps(dict(review=pin(REVIEW),prepared=pin(PREPARED),freeze=pin(P/'FROZEN_SOURCE188.json'),
 archive=pin(archive),members=len(members),NPU_used=False)))
