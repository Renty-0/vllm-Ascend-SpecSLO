"""ROOT freeze of original SLO admission priority with retained cap4/cohort4."""
import argparse, ast, datetime, hashlib, importlib.util, json, os, subprocess, sys, tarfile
from pathlib import Path

B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-slo-admission1-vacancy4-cap4-active32-full-v191';G=D/'v191-root-execution'
READY=D/'V191_SLO_ADMISSION1_VACANCY4_CAP4_ACTIVE32_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json'
PEER=D/'V191_INDEPENDENT_SOURCE_PEER_REVIEW.json'
REVIEW=D/'ROOT_V191_SLO_ADMISSION_COHORT4_SOURCE_AND_GUARD_REVIEW.json'
PREPARED=D/'ROOT_V191_PREPARED_SOURCE_HOST_AUDIT.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def pin(p):return dict(path=str(p),sha256=sha(p))
def save(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
parser=argparse.ArgumentParser();parser.add_argument('--expected-peer-sha256',required=True);args=parser.parse_args()
assert sys.version_info[:2]==(3,11) and 'torch' not in sys.modules and 'torch_npu' not in sys.modules
assert sha(READY)=='eaa72b4a64a37493c216baa85aab8f164b9e1609f85c489a519024beda02ea46'
assert sha(PEER)==args.expected_peer_sha256
ready,peer=read(READY),read(PEER)
assert ready['source_stopped_writing'] is True and peer['candidate']==str(P)
assert peer['status'].replace('-','_')=='passed_source_review'
sys.path.insert(0,str(P));spec=importlib.util.spec_from_file_location('_ROOT_V191_source_guard',G/'guard191.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
state=guard.source_state(archive=True);guards=guard.guard_inputs()
flat=guard.merge(state['local'],state['external'],state['evidence'],guards,{str(guard.CPU):sha(guard.CPU)})
assert state['local']==ready['files'] and state['external']==ready['external_files']
assert state['evidence']==ready['retained_HOST_evidence'] and guards==ready['execution_guard_inputs']
assert (len(state['local']),len(state['external']),len(state['evidence']),len(flat))==(20,277,82,384)
for path,h in flat.items():
 p=Path(path);assert p.is_file() and not p.is_symlink() and sha(p)==h,p
for path in state['local']:
 if path.endswith('.py'):ast.parse(Path(path).read_text(),filename=path)
cpu=read(guard.CPU)
assert cpu['status']=='passed' and cpu['tests_run']==3 and cpu['errors']==cpu['failures']==0
assert cpu['Torch_imported'] is cpu['NPU_used'] is cpu['model_constructed'] is cpu['model_math_qualified'] is False
assert len(list(P.glob('HOST_ATTEMPT191_*.json')))==3
assert ready['counts']['whole190_source_projections']==15
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
save(REVIEW,dict(schema='ROOT.V191.SLO-admission-cohort4-source-and-guard-review.v1',status='passed_source_review',
 candidate=str(P),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),ready=pin(READY),peer=pin(PEER),
 reviewed_inputs=guard.merge(flat,{str(PEER):sha(PEER)}),CPU_tests=3,CPU_elapsed_seconds=cpu['elapsed_seconds'],
 old190_4HOST_and185_suites_not_rerun=True,Torch_imported=False,NPU_used=False,
 source_findings=[
  'Sole production parent is frozen190. Fifteen complete source projections preserve its original173 cap4/highN and trueB5/lowN, active32, C96/E96/global192, current causal80 mapping and vacancy4/150. No192/183/187 model or schedule changes are combined.',
  'Only the original SLO_AWARE_ADMISSION flag becomes1. TARGET_SLO_FIRST stays0. The private env-i launcher explicitly exports both before source/runtime checks. Parent and all4 worker processes verify actual Native env getters/config outside timed generation; no model state flag is manufactured.',
  'Original admission filters by already-arrived timestamp then applies original priority order. Future tight requests remain excluded. Every original463 request, ID, arrival, SLO and max256 stays; there is no drop or abort path. Original TPOT/TTFT/outer clocks and metric arithmetic are unchanged.',
  'Actual original patched admit_available HOST executes with genuine NativeSamplingParams and PearlPipelineState, ON/OFF order comparison, a future tighter request negative and complete463 exactly-once admission in15 explicit HOST completion groups. This is admission control evidence, not256-token model completion.',
  'Priority arrival can change the oldest-ready anchor and restart a vacancy episode. A current HOST case executes real episode reset and later timeout.150ms is a poll-based per-episode threshold, not a strict TTFT/total-wait bound. Finite463 must all finish; no infinite-stream fairness is claimed.',
  'Raw current policy audit explicitly requires admission1/target0, enabled4/150 and any nonnegative deferred polls. It rejects old coalesce1 policy and accepts genuinely nonzero deferred counts. Parent generation event audit still independently replays all4 ranks and closes active cohort state.',
  'All mathematical gates run anew in the current engine: original201/44/114, full527 B4 and causal80, then multiowner2/4 eager-prefill fullhidden/all role-KV/same-N heads/original root publication and genuine nextB4 raw8. No current190 receipt or model flag is inherited.',
  'Current bounded warm65 fixedpoint, two complete Native21zero65 traces, one seal and full463 remain mandatory. Current issuer binds actual same-engine173 and new multiowner evidence. Original no-reservation mixed authorization stays empty.',
  'All three HOST attempts including missing-source-pin and insufficient HOST carrier-accounting failures are retained with exact source/log bytes. Frozen190 and its4 HOST plus185 core/2 HOST remain SHA-pinned and are not rerun.',
  'Original ROOT selflock, baseline verifier/prepost and owned cleanup retain7200s/256MiB/512MiB limits. Parent190 archive and original archive are verified in place. Exact frozen177 temporary-file handling remains private and restricted.',
 ],limitations=[
  'The finite admission HOST proof assumes explicit completion carriers. Only current actual full463 establishes real256-token completion and observed latency.',
  'Numerical representatives cover2/4 prefill owners, not all32-owner tuples. Source readiness grants no current device qualification, Goodput, adoption or full-workload independent AR equality.',
  'Priority and coalescing can improve or worsen service. No performance increase is inferred before current full463.',
 ],formal_Goodput=False,adopted=False))
completed=subprocess.run([sys.executable,'-B',str(G/'guard191.py'),'--prepare-only','--root-review',str(REVIEW),
 '--expected-root-review-sha256',sha(REVIEW),'--peer-review',str(PEER),sha(PEER)],check=True,text=True,
 stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
frozen,sources,prepared=guard.prepared();archive=P/'PREPARED_SOURCE_ARCHIVE191.tar.gz'
with tarfile.open(archive,'r:gz')as stream:
 members=stream.getmembers();assert len(members)==len(sources)==387
 for member in members:
  assert member.isfile() and hashlib.sha256(stream.extractfile(member).read()).hexdigest()==sources['/'+member.name]
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
save(PREPARED,dict(schema='ROOT.V191.prepared-source-HOST-audit.v1',status='passed',candidate=str(P),
 review=pin(REVIEW),peer=pin(PEER),freeze=pin(P/'FROZEN_SOURCE191.json'),archive=pin(archive),
 actual_archive_members=len(members),actual_source_members=len(sources),prepared=prepared,
 prepare_stdout=completed.stdout,NPU_used=False,Torch_imported=False,formal_Goodput=False,adopted=False))
print(json.dumps(dict(review=pin(REVIEW),prepared=pin(PREPARED),freeze=pin(P/'FROZEN_SOURCE191.json'),
 archive=pin(archive),members=len(members),NPU_used=False)))
