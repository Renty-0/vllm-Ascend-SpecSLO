"""ROOT review and immutable source preparation; does not execute a model."""
import argparse, ast, datetime, hashlib, importlib.util, json, os, subprocess, sys, tarfile
from pathlib import Path

B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-slo-admission1-causal80-active32-full-v189';G=D/'v189-root-execution'
READY=D/'V189_SLO_ADMISSION1_CAUSAL80_ACTIVE32_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json'
PEER=D/'V189_INDEPENDENT_SOURCE_PEER_REVIEW.json'
REVIEW=D/'ROOT_V189_SLO_ADMISSION1_SOURCE_AND_GUARD_REVIEW.json'
PREPARED=D/'ROOT_V189_PREPARED_SOURCE_HOST_AUDIT.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def pin(p):return dict(path=str(p),sha256=sha(p))
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
parser=argparse.ArgumentParser();parser.add_argument('--expected-peer-sha256',required=True);args=parser.parse_args()
assert sys.version_info[:2]==(3,11) and 'torch' not in sys.modules and 'torch_npu' not in sys.modules
assert sha(READY)=='5f0dde5389726068ec8db9460f9109674a1bf4b9865373f65d59d04b21150463'
assert sha(PEER)==args.expected_peer_sha256
ready,peer=read(READY),read(PEER)
assert peer['status'].replace('-','_')=='passed_source_review' and peer['candidate']==str(P)
assert ready['source_writes_stopped'] is True
sys.path.insert(0,str(P));spec=importlib.util.spec_from_file_location('_ROOT_V189_source_guard',G/'guard189.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
state=guard.source_state(archive=True);guards=guard.guard_inputs()
flat=guard.merge(state['local'],state['external'],state['evidence'],guards,{str(guard.CPU):sha(guard.CPU)})
assert state['local']==ready['local_files'] and state['external']==ready['external_files']
assert state['evidence']==ready['retained_HOST_evidence'] and guards==ready['guard_inputs']
assert flat==ready['reviewed_inputs_before_peer']
assert (len(state['local']),len(state['external']),len(state['evidence']),len(flat))==(23,165,57,250)
for path,h in flat.items():
 p=Path(path);assert p.is_file() and not p.is_symlink() and sha(p)==h,p
for path in state['local']:
 if path.endswith('.py'):ast.parse(Path(path).read_text(),filename=path)
cpu=read(guard.CPU)
assert cpu['status']=='passed' and cpu['tests_run']==3 and cpu['failures']==cpu['errors']==0
assert cpu['Torch_imported'] is cpu['NPU_used'] is cpu['model_constructed'] is False
from check189 import verify_projection
assert verify_projection()['projected_sources']==20
descriptor=read(P/'ADMISSION_PROJECTION189.json');identical=[]
for name,row in descriptor['files'].items():
 if name in ('audit189.py','base189.py','check189.py','host189.py','runtime189.py'):continue
 text=Path(row['parent']).read_text().replace('173','189').replace('tree-causal80-real16-bankfix-full-v189',P.name)
 assert text==(P/name).read_text(),name
 identical.append(name)
assert len(identical)==15
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
save(REVIEW,dict(schema='ROOT.V189.SLO-admission1-source-and-guard-review.v1',status='passed_source_review',
 candidate=str(P),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),ready=pin(READY),peer=pin(PEER),
 reviewed_inputs=guard.merge(flat,{str(PEER):sha(PEER)}),namespace_only_whole_sources=identical,
 CPU_tests=3,CPU_elapsed_seconds=cpu['elapsed_seconds'],old_suites_not_rerun=True,NPU_used=False,Torch_imported=False,
 source_findings=[
  'Sole production parent is completed frozen173. Fifteen of20 selected parent files are identical after namespace; all20 explicit whole-source inverses hold. No actor/mapper/M80/hidden/KV/commit/next model mathematics changes.',
  'Only production scheduling change is original SLO_AWARE_ADMISSION branch0 to1. Clean private startup exports1 and explicitly TARGET_SLO_FIRST0 before the actual checker/launcher; external shell exports are not the authority.',
  'Original Native sort orders only candidates passed by the existing arrival filter. Genuine NativeSamplingParams and actual complete admit_available AST HOST checks demonstrate ON/OFF order, exclusion of future arrivals and finite463 admission without deletion or duplication. HOST root/completion carriers do not prove model outputs.',
  'Configuration remains coalesce1/0, cap4 at high-N/trueB5 at low-N, active32, normal/eager96/global192,544KV blocks and64/463/527 capacities. Original arrivals, output256, TPOT, first-token callback TTFT and full wall remain.',
  'New checks and report occur outside measured generation. Each retained actual trace must independently carry SLO1/TARGET0/coalesce1 counters and actual prefill request count; no per-call scheduling wrapper or result-clock rewrite is introduced.',
  'Current startup201/44/114, own full527 B4 component, actual N16 causal80/S16 original Native greedy/hidden entries, independent sameM allphysical hidden/heads/all64layer KV and original verdict/compaction/next remain compulsory.',
  'Current changed-policy warm fixedpoint, original warm65/two complete21-zero traces, seal and full463 must all execute. Previous173 numerical receipts confer no new-engine runtime capability. Old128/M80 crossM comparisons are explicitly classified as finite numerical behavior, never whole-workload AR equivalence.',
  'Original selflock/baseline prepost/owned cleanup and7200s/256MiB/512MiB bounds remain. Exact frozen177 temporary-file race handling is scoped only to this private actual root; original173 and its158 regular archive members remain immutable and verified in place.',
 ],limitations=[
  'Strict class priority has no bounded-age guarantee for an infinite arrival stream. This experiment must actually finish every request in the original finite463 workload.',
  'Different order may change cohorts/contexts/graph reachability and TTFT. No performance gain, current NPU qualification, whole-workload AR equivalence, adoption or goal achievement follows from this source review.',
  'No185 prefilling,187 union or183 storage optimization is combined in this candidate.',
 ],formal_Goodput=False,adopted=False))
completed=subprocess.run([sys.executable,'-B',str(G/'guard189.py'),'--prepare-only','--root-review',str(REVIEW),
 '--expected-root-review-sha256',sha(REVIEW),'--peer-review',str(PEER),sha(PEER)],check=True,text=True,
 stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
frozen,sources,prepared=guard.prepared();archive=P/'PREPARED_SOURCE_ARCHIVE189.tar.gz'
with tarfile.open(archive,'r:gz') as stream:
 members=stream.getmembers();assert len(members)==len(sources)==253
 for member in members:
  assert member.isfile() and hashlib.sha256(stream.extractfile(member).read()).hexdigest()==sources['/'+member.name]
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
save(PREPARED,dict(schema='ROOT.V189.prepared-source-HOST-audit.v1',status='passed',candidate=str(P),
 review=pin(REVIEW),peer=pin(PEER),freeze=pin(P/'FROZEN_SOURCE189.json'),archive=pin(archive),
 actual_archive_members=len(members),actual_source_members=len(sources),prepared=prepared,
 prepare_stdout=completed.stdout,NPU_used=False,Torch_imported=False,formal_Goodput=False,adopted=False))
print(json.dumps(dict(review=pin(REVIEW),prepared=pin(PREPARED),freeze=pin(P/'FROZEN_SOURCE189.json'),
 archive=pin(archive),members=len(members),NPU_used=False)))
