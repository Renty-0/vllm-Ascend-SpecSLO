"""ROOT review of stopped198 source, retained HOST proof, and bounded guard."""
import ast, hashlib, importlib.util, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-b4-joint-subtree-m80-gain-diagnostic-v198';G=D/'v198-root-execution'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p))
def save(p,x):
 with p.open('x')as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

ready=D/'V198_B4_JOINT_SUBTREE_BOUNDED_GAIN_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json'
peer=D/'V198_INDEPENDENT_SOURCE_PEER_REVIEW.json'
assert sha(ready)=='838781e6c0a9f5bb83ddb66d0cab95bb12e823ae8e08dc94efd3a03404ead79d'
r=json.loads(ready.read_bytes());pr=json.loads(peer.read_bytes())
assert pr['status']=='passed_source_review'and pr['candidate']==str(P)
assert r['source_stopped_writing']is True and r['freeze_created']is False
for key in('local_files','selected_readonly_files','execution_guard_inputs','retained_HOST_evidence'):
 for path,h in r[key].items():assert sha(path)==h,path
derivation=json.loads((P/'DERIVATION198.json').read_bytes())
for path,row in derivation['whole_text_namespace_projections'].items():
 original=Path(row['source']).read_text();a,b=row['replacement'];value=Path(path).read_text()
 assert sha(row['source'])==row['sha256']and value==original.replace(a,b)
 assert value.replace(b,a)==original
for path in r['local_files']:
 if Path(path).suffix=='.py':ast.parse(Path(path).read_text(),filename=path)
ast.parse((G/'guard198.py').read_text(),filename=str(G/'guard198.py'))
for name in('launch198.sh','startup198.sh'):subprocess.run(['bash','-n',str(G/name)],check=True)
spec=importlib.util.spec_from_file_location('_ROOT198_source_review_guard',G/'guard198.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
assert guard.P==P and guard.CPU==P/'CPU_RESULT198.json'
state=guard.source_state(archive=True);guards=guard.guard_inputs()
assert state['local']==r['local_files']and state['external']==r['selected_readonly_files']
assert state['evidence']==r['retained_HOST_evidence']and guards==r['execution_guard_inputs']
assert [len(state[k])for k in('local','external','evidence')]==[19,165,128]and len(guards)==4
assert sha(guard.CPU)=='500d8c98818372ae931e30525ecc2a6320e45661c6240a505334919d6ed7af82'
inputs=guard.merge(state['local'],state['external'],state['evidence'],guards,
 {str(guard.CPU):sha(guard.CPU),str(peer):sha(peer)})
assert len(inputs)==318 and'torch'not in sys.modules
review=D/'ROOT_V198_BOUNDED_JOINT_B4_SOURCE_AND_GUARD_REVIEW.json'
save(review,dict(schema='ROOT.V198.bounded-joint-B4-source-review.v1',
 UTC=datetime.now(timezone.utc).isoformat(),status='passed_source_review',candidate=str(P),
 reviewed_inputs=inputs,ready=pin(ready),peer=pin(peer),parent_archives_verified_in_place=state['parent_archive'],
 whole_text_projections_verified=2,ROOT_source_reviewed=True,HOST_tests_rerun=False,NPU_used=False,Torch_imported=False,
 reviewed_semantics=[
  'Bounded diagnostic only: no production/full463 switch, posterior installation, or new issuer.',
  'Unchanged frozen173 constructor and complete same-engine original B4 plus M80 components precede the new diagnostic.',
  'Exactly four ancestor-closed raw8 nodes selected by current draft joint probability; six controlled fork shapes and two linear controls, each three complete traces.',
  'Real same-M80/S16 full9 or causal target, full80 hidden/heads, every128 queryKV and compacted path KV versus independent owners; original Native verdict, typed tickets, commit and own next remain.',
  'Three classes each use12 correlated old V15 source prefixes and four excluded filler owners; every policy recomputes fresh actual draft and target history.',
  'Only the necessary full9 hidden key is naturally added; original authoritative key objects remain runtime validated and no flag is assigned.',
  'Both complete new-topology zero traces and three fresh whole-protocol-plus-next cost observations per policy/class retain actual Native field differences.',
  'HOST carriers check exact source, slots, all mask bits, original integer verifier and cache move bookkeeping; they do not claim hidden/KV numerical proof.',
  'Original ROOT self-lock, original baseline pre/post, owned cleanup, storage177 race handling, all five HOST attempts and source/guard snapshots are retained.'
 ],limitations=[
  'Source/HOST only; no current198 device math, branch coverage, timing, or gain is established.',
  'The diagnostic cost timer includes actual original_call greedy-output cloning and shape/key/flag checks, explicit stage synchronization, and drains; observer=False excludes the full hidden/KV observer and independent reference, not every diagnostic operation.',
  'Three old-policy observations precede three joint-policy observations rather than alternating ABBA; per-rank times are not additive and are not pure production costs or Goodput.',
  'Natural confidence may choose no useful fork or no side acceptance; that is retained as subset, not converted into production qualification.',
  'Twelve class samples are correlated source prefixes and do not establish a complete g function; initial roots are teacher-forced and whole463 own-AR remains false.',
  'Known canonical own-AR differences and formal final-nine common-clock1.5x qualification remain unresolved.'
 ],formal_Goodput=False,adopted=False))
command=[sys.executable,'-B',str(G/'guard198.py'),'--prepare-only','--root-review',str(review),
 '--expected-root-review-sha256',sha(review),'--peer-review',str(peer),sha(peer)]
result=subprocess.run(command,capture_output=True,text=True)
save(D/'ROOT_V198_GUARD_PREPARE_STDOUT_AND_STDERR.json',dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
assert result.returncode==0,result.stderr
f,sources,prepared=guard.prepared();assert len(sources)==prepared['archive']['members']==320 and'torch'not in sys.modules
output=D/'ROOT_V198_PREPARED_SOURCE_HOST_AUDIT.json'
save(output,dict(schema='ROOT.V198.prepared-source-HOST-audit.v1',status='passed_source_frozen_unrun',candidate=str(P),
 review=pin(review),peer=pin(peer),freeze=pin(guard.FREEZE),archive=pin(guard.ARCHIVE),
 actual_source_members=len(sources),actual_archive_members=prepared['archive']['members'],prepared=prepared,
 NPU_used=False,Torch_imported=False,models_launched=False,formal_Goodput=False,adopted=False))
print(json.dumps(dict(status='passed_source_frozen_unrun',record=pin(output),freeze=pin(guard.FREEZE),archive=pin(guard.ARCHIVE),members=len(sources))))
