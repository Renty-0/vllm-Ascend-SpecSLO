"""Stopped197 source/retained-HOST review and source-only preparation."""
import ast,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-empty-active-cohort-causal80-active32-full-v197';G=D/'v197-root-execution'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p))
def save(p,x):
 with p.open('x')as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
ready=D/'V197_EMPTY_ACTIVE_COHORT_PREFILL_CAP4_ACTIVE32_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json'
peer=D/'V197_INDEPENDENT_SOURCE_PEER_REVIEW.json'
assert sha(ready)=='9caa1791f023bdf9e7d905a4fb9329c56c6111e1c910b84652f57165f65245b1'
r=json.loads(ready.read_text());pr=json.loads(peer.read_text())
assert pr['status']=='passed_source_review'and pr['candidate']==str(P)
assert r['source_stopped_writing']is True and r['ROOT_freeze_created']is False
for key in('files','external_files','execution_guard_inputs','retained_HOST_evidence'):
 for path,digest in r[key].items():assert sha(path)==digest,path
projection=json.loads((P/'PROJECTIONS197.json').read_text())['projection'];assert len(projection)==16
for row in projection:
 parent=Path(row['parent']);assert sha(parent)==row['parent_sha256']
 original=parent.read_text();normalized=original.replace(*row['namespace'])
 chunks=[];position=0;shift=0;inverse=[]
 for op in row['deltas']:
  assert op['start']>=position and normalized[op['start']:op['end']]==op['old']
  chunks.extend((normalized[position:op['start']],op['new']));position=op['end']
  start=op['start']+shift;inverse.append((start,start+len(op['new']),op['new'],op['old']))
  shift+=len(op['new'])-len(op['old'])
 append=row.get('append_text','');value=''.join(chunks)+normalized[position:]+append
 assert (P/row['file']).read_text()==value,row['file']
 if append:assert value.endswith(append);value=value[:-len(append)]
 for start,end,new,old in reversed(inverse):
  assert value[start:end]==new;value=value[:start]+old+value[end:]
 assert value==normalized,row['file']
 assert value.replace(row['namespace'][1],row['namespace'][0])==original,row['file']
for path in r['files']:
 if Path(path).suffix=='.py':ast.parse(Path(path).read_text(),filename=path)
ast.parse((G/'guard197.py').read_text(),filename=str(G/'guard197.py'))
for name in('launch197.sh','startup197.sh'):subprocess.run(['bash','-n',str(G/name)],check=True)
spec=importlib.util.spec_from_file_location('_ROOT197_source_review_guard',G/'guard197.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
assert guard.P==P and guard.CPU==P/'CPU_RESULT197.json'
state=guard.source_state(archive=True);guards=guard.guard_inputs()
assert len(state['local'])==21 and len(state['external'])==179 and len(state['evidence'])==86 and len(guards)==4
assert sha(guard.CPU)=='3add299b8ea2d9c7b5dbde0c959b4942c65df83a19b5eeb112451d157f4e991c'
inputs=guard.merge(state['local'],state['external'],state['evidence'],guards,
 {str(guard.CPU):sha(guard.CPU),str(peer):sha(peer)})
assert len(inputs)==292 and 'torch'not in sys.modules
review=D/'ROOT_V197_EMPTY_ACTIVE_COHORT_SOURCE_AND_GUARD_REVIEW.json'
save(review,dict(schema='ROOT.V197.empty-active-cohort-source-review.v1',
 UTC=datetime.now(timezone.utc).isoformat(),status='passed_source_review',candidate=str(P),
 reviewed_inputs=inputs,ready=pin(ready),peer=pin(peer),parent_archives_verified_in_place=state['parent_archive'],
 whole_text_projections_verified=16,ROOT_source_reviewed=True,HOST_tests_rerun=False,NPU_used=False,Torch_imported=False,
 reviewed_semantics=[
  'Sole production parent173; the only production decoder AST delta is common active-nonempty return immediately after original receive_live_admissions.',
  'Original arrival filter, pending order, immediate per-ready-token publication, root callback, clocks,256 outputs,while/final drain, active32/target16/C96E96/global192 and lowN trueB5 remain exact.',
  'One unchanged constructor and current original full527 B4 plus M80 numerical/verdict/commit/ownnext proofs precede a fresh extra multiowner eager component.',
  'Real original Native wholeprompt helper N2/32 compares independent disjoint same-M full hidden, every56draft/128target queryKV, original metadata and same-N root head; incumbent prefixes remain protected.',
  'Original root append/publication AST executes exactly once; original planner naturally chooses N1/16 next, with true four-step draft/encoded confidence/ledger/reference/transport and ready-target admission at most16/C96.',
  'The additional component does not submit target; it does not claim new N32 target scope. Same-engine original B4/M80 target commit-next proof remains mandatory.',
  'All real Native dictionaries and runtime-validated entry objects must remain identical; no singleton201 inference or assigned qualification flag is accepted.',
  'Three passes include two complete zero sweeps of the extra component; warm fixedpoint,two original zero65,sealed full463,finite drain and no incumbent refill are still fresh device gates.',
  'Sixteen whole-text namespace-plus-declared-delta projections were reversed; production decoder insertion has its own entire AST inverse.',
  'Original ROOT self-lock,baseline pre/post,owned group cleanup and frozen177 scoped mutable-tmp race handling remain; no hot-path collective is added.',
  'All three HOST attempts and complete source/guard/log snapshots remain; no old suite or model was rerun.'
 ],limitations=[
  'Source/HOST preparation only: no current197 device numerical/cost/full463 qualification or performance result exists.',
  'Inherited guard cost_measured/new_target_keys labels do not establish a measured197 cost or extra prefill graph keys; the two M80 keys belong to the unchanged parent173 qualification.',
  'N2/32 are representative eager-prefill cases, not an exhaustive proof of all prompt/cohort tuples or arbitrary peak HBM fit.',
  'Cohort drain can worsen TTFT, shrink tail batches and change wall/Goodput; no benefit follows merely from moving prefill waits outside active decode.',
  'Known canonical own-AR differences remain unresolved; whole463 AR and formal same-clock1.5x acceptance remain false.'
 ],formal_Goodput=False,adopted=False))
command=[sys.executable,'-B',str(G/'guard197.py'),'--prepare-only','--root-review',str(review),
 '--expected-root-review-sha256',sha(review),'--peer-review',str(peer),sha(peer)]
result=subprocess.run(command,capture_output=True,text=True)
save(D/'ROOT_V197_GUARD_PREPARE_STDOUT_AND_STDERR.json',dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
assert result.returncode==0,result.stderr
f,sources,prepared=guard.prepared();assert len(sources)==prepared['archive']['members']==294 and 'torch'not in sys.modules
output=D/'ROOT_V197_PREPARED_SOURCE_HOST_AUDIT.json'
save(output,dict(schema='ROOT.V197.prepared-source-HOST-audit.v1',status='passed_source_frozen_unrun',candidate=str(P),
 review=pin(review),peer=pin(peer),freeze=pin(guard.FREEZE),archive=pin(guard.ARCHIVE),
 actual_source_members=len(sources),actual_archive_members=prepared['archive']['members'],prepared=prepared,
 NPU_used=False,Torch_imported=False,models_launched=False,formal_Goodput=False,adopted=False))
print(json.dumps(dict(status='passed_source_frozen_unrun',record=pin(output),freeze=pin(guard.FREEZE),archive=pin(guard.ARCHIVE),members=len(sources))))
