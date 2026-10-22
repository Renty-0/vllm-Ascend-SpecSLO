"""Review stopped199 REV2, its strict rollback fix, and bounded execution guard."""
import ast, hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-bounded-parent4-shared4-eager-diagnostic-v199';G=D/'v199-root-execution'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p))
def save(p,x):
 with p.open('x')as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
ready=D/'V199_BOUNDED_PARENT4_CHILD4_SHARED4_DIAGNOSTIC_SOURCE_AND_HOST_READY_FOR_ROOT_REVIEW_REVISION2.json'
peer=D/'V199_INDEPENDENT_SOURCE_PEER_REVIEW.json'
assert sha(ready)=='939bc2d968be8feaede93ede4175358b031e047ad29aea77009b7009328dbc72'
assert sha(peer)=='aa9aebcc201060bcdbfacb156ec0c00a13dba8a489b3e41b8211fe6514989cbc'
r=json.loads(ready.read_bytes());pr=json.loads(peer.read_bytes());closure=r['source_closure']
assert r['source_stopped']is True and r['freeze_created']is False
assert pr['status']=='passed_source_review'and pr['candidate']==str(P)and pr['blocking_findings']==[]
for key in('local_files','readonly_inputs','execution_guard_inputs','retained_HOST_evidence'):
 for p,h in closure[key].items():assert sha(p)==h,p
projection=json.loads((P/'PROJECTIONS199.json').read_bytes())['projection'];assert len(projection)==9
for row in projection:
 parent=Path(row['parent']);assert sha(parent)==row['parent_sha256']
 original=parent.read_text();normalized=original.replace(*row['namespace']);parts=[];position=shift=0;inverse=[]
 for op in row['deltas']:
  assert op['start']>=position and normalized[op['start']:op['end']]==op['old']
  parts.extend((normalized[position:op['start']],op['new']));position=op['end']
  start=op['start']+shift;inverse.append((start,start+len(op['new']),op['new'],op['old']))
  shift+=len(op['new'])-len(op['old'])
 append=row.get('append_text','');value=''.join(parts)+normalized[position:]+append
 assert (P/row['file']).read_text()==value,row['file']
 if append:assert value.endswith(append);value=value[:-len(append)]
 for start,end,new,old in reversed(inverse):
  assert value[start:end]==new;value=value[:start]+old+value[end:]
 assert value==normalized and value.replace(row['namespace'][1],row['namespace'][0])==original
for path in closure['local_files']:
 if Path(path).suffix=='.py':ast.parse(Path(path).read_text(),filename=path)
ast.parse((G/'guard199.py').read_text(),filename=str(G/'guard199.py'))
for name in('launch199.sh','startup199.sh'):subprocess.run(['bash','-n',str(G/name)],check=True)
# The fixed rollback really restores model-generated parent/shared4 before the
# original next proof snapshots its ledger. Its strict equality source is intact.
actor=(P/'actor199.py').read_text();tree=ast.parse(actor)
run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run_case')
body=next(n for n in run.body if isinstance(n,ast.Try)).body
i=next(i for i,n in enumerate(body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='rollback_next'for t in n.targets))
assert isinstance(body[i-1],ast.Expr)and ast.unparse(body[i-1].value.func)=='restore_source'
oldready=json.loads(Path(r['supersedes_ready']['path']).read_bytes())
assert sha(r['supersedes_ready']['path'])==r['supersedes_ready']['sha256']
assert sha(P/'next199.py')==oldready['source_closure']['local_files'][str(P/'next199.py')]
spec=importlib.util.spec_from_file_location('_ROOT199_source_review_guard',G/'guard199.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
assert guard.P==P and guard.CPU==P/'CPU_RESULT199.json'
state=guard.source_state(archive=True);guards=guard.guard_inputs()
assert state['local']==closure['local_files']and state['external']==closure['readonly_inputs']and guards==closure['execution_guard_inputs']
assert [len(state[k])for k in('local','external','evidence')]==[28,465,251]and len(guards)==4
assert sha(guard.CPU)=='266e42c1b3611e4611d30ef983f7f11c6b02c0accefc95e837f7dad7eead1d06'
supplement={p:h for p,h in closure['retained_HOST_evidence'].items()if p not in state['evidence']}
assert len(supplement)==5 and all(sha(p)==h for p,h in supplement.items())
assert all(closure['retained_HOST_evidence'][p]==h for p,h in state['evidence'].items())
inputs=guard.merge(state['local'],state['external'],state['evidence'],guards,
 {str(guard.CPU):sha(guard.CPU),str(peer):sha(peer)})
assert len(inputs)==750 and'torch'not in sys.modules
nested=[p for p in state['external']if p.endswith('.tar.gz')]
assert nested==[str(B/'tree-natural-positive-union-discovery-local-v154/PREPARED_SOURCE_ARCHIVE_V154.tar.gz')]
assert Path(nested[0]).stat().st_size==4163298
review=D/'ROOT_V199_BOUNDED_PARENT4_CHILD4_REV2_SOURCE_AND_GUARD_REVIEW.json'
save(review,dict(schema='ROOT.V199.bounded-parent4-child4-REV2-source-review.v1',
 UTC=datetime.now(timezone.utc).isoformat(),status='passed_source_review',candidate=str(P),reviewed_inputs=inputs,
 ready=pin(ready),peer=pin(peer),supplemental_ready_evidence=supplement,
 supplemental_ready_evidence_is_outside_guard_archive=True,
 parent_archives_verified_in_place=state['parent_archive'],whole_text_projections_verified=9,
 ROOT_source_reviewed=True,HOST_tests_rerun=False,NPU_used=False,Torch_imported=False,
 selected_old_binary_archive_exception=[dict(**pin(nested[0]),bytes=Path(nested[0]).stat().st_size)],
 reviewed_semantics=[
  'Bounded diagnostic only: no production issuer, online activation, full463 or performance score.',
  'One original constructor and original same-engine full527 B4 plus M80 qualification; pre-install original protocol capture and original commit_states identity remain.',
  'Fresh component173 namespace changes only the exact FastV5 origin binding; every other canonical module path/hash remains original. Empty eager sources preserve the complete original startup/normal path.',
  'Only private finally-revoked typed E4 classes permit P4/child4 shared4 with normal16, E1..4, genuine repair0/1, M20, original F/U frontier and device-correct metadata permutation.',
  'Allphysical hidden/encoded, raw8/score8/mean8, every56 draft queryKV, target128 query/committed KV, original integer consensus, ticket cleanup and own-next are real future device gates.',
  'Original controller consumes real normal16 before child tails in the default fixture. Missing child-containing N16/Q80 or natural promotion/reject/final e4 repair is retained as bounded subset.',
  'REV2 adds actual common source restoration before rollback nextproof snapshots its ledger; original next proof source and strict ledger equality are unchanged. The old failing order and new next boundary are both actual HOST checked.',
  'Three complete five-case traces require two real complete21-counter zeros; original entry identities/keys remain observed, not assigned.',
  'Seven HOST attempts, the earlier249-member source/HOST preservation archive, original baseline pre/post, self-lock, storage177 and owned cleanup are retained.'
 ],limitations=[
  'No current199 model/hidden/KV qualification, actual promotion/side coverage, timing, full463 or Goodput exists.',
  'The private E4 producer does not change production cap3/default planner and does not authorize arbitrary tuples or joint childM80 production.',
  'Retained hashes and HOST carriers cannot recompute absent numerical model/KV tensors.',
  'One selected old P154 archive of4,163,298 bytes is embedded as an explicit frozen dependency. P148/P160/P173 archives remain verified in place; do not claim every old binary archive is uncopied.',
  'The ready record binds five extra D artifacts beyond the actual749 pre-peer guard inputs. They are independently pinned here, not silently added to the guarded archive set.',
  'Whole463 canonical own-AR, final-nine same-clock1.5x and adoption remain false.'
 ],formal_Goodput=False,adopted=False))
command=[sys.executable,'-B',str(G/'guard199.py'),'--prepare-only','--root-review',str(review),
 '--expected-root-review-sha256',sha(review),'--peer-review',str(peer),sha(peer)]
result=subprocess.run(command,capture_output=True,text=True)
save(D/'ROOT_V199_GUARD_PREPARE_STDOUT_AND_STDERR.json',dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
assert result.returncode==0,result.stderr
f,sources,prepared=guard.prepared();assert len(sources)==prepared['archive']['members']==752 and'torch'not in sys.modules
assert all(sha(p)==h for p,h in supplement.items())
out=D/'ROOT_V199_PREPARED_SOURCE_HOST_AUDIT.json'
save(out,dict(schema='ROOT.V199.prepared-source-HOST-audit.v1',status='passed_source_frozen_unrun',candidate=str(P),
 review=pin(review),peer=pin(peer),ready=pin(ready),freeze=pin(guard.FREEZE),archive=pin(guard.ARCHIVE),
 actual_source_members=len(sources),actual_archive_members=prepared['archive']['members'],prepared=prepared,
 supplemental_ready_evidence=supplement,selected_old_binary_archive_exception=[pin(nested[0])],
 NPU_used=False,Torch_imported=False,models_launched=False,formal_Goodput=False,adopted=False))
print(json.dumps(dict(status='passed_source_frozen_unrun',record=pin(out),freeze=pin(guard.FREEZE),archive=pin(guard.ARCHIVE),members=len(sources))))
