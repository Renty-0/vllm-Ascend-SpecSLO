"""Review stopped202 REV3 and freeze only; ROOT remains sole separate NPU owner."""
import ast,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-causal80-small-cohort-active32-full-v202';G=D/'v202-root-execution'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p))
def save(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
ready=D/'V202_SEVEN_REAL_N_EXISTING80_EMPTY_ACTIVE32_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW_REVISION3.json'
peer=D/'V202_INDEPENDENT_SOURCE_PEER_REVIEW.json'
assert sha(ready)=='6efd24f8cf26870054079999039b60e60d5de4c0f326c193cd2bd4d86e3218a7'
r=json.loads(ready.read_bytes());pr=json.loads(peer.read_bytes());closure=r['source_closure']
assert r['source_stopped']is True and r['freeze_created']is False and r['revision']==3
assert pr['status'].replace('-','_')=='passed_source_review'and pr['candidate']==str(P)
assert pr.get('blocking_findings',[])==pr.get('unresolved_source_blockers',[])==[]
for group in('local_files','readonly_inputs','execution_guard_inputs','retained_HOST_evidence','auxiliary_review_evidence'):
 for path,h in closure[group].items():assert sha(path)==h,path
for k,wanted in(('REVISION2_changes',42),('REVISION3_changes',44)):
 m=r[k]['preservation_manifest'];assert sha(m['path'])==m['sha256']
 values=json.loads(Path(m['path']).read_bytes());assert len(values['files'])==wanted
 for old,row in values['files'].items():assert sha(row['snapshot'])==row['sha256'],old
 for path,h in values['original_full_HOST_bytes_retained_in_place'].items():assert sha(path)==h,path
projection=json.loads((P/'PROJECTIONS202.json').read_bytes())
assert len(projection['projection'])==17 and len(projection['guard_projection'])==4
for group,root in(('projection',P),('guard_projection',G)):
 for row in projection[group]:
  assert sha(row['parent'])==row['parent_sha256'];original=Path(row['parent']).read_text();old,new=row['namespace'];text=original.replace(old,new)
  parts=[];offset=0;shift=0;inverse=[]
  for op in row['deltas']:
   assert op['start']>=offset and text[op['start']:op['end']]==op['old']
   parts.extend((text[offset:op['start']],op['new']));offset=op['end']
   start=op['start']+shift;inverse.append((start,start+len(op['new']),op['new'],op['old']));shift+=len(op['new'])-len(op['old'])
  value=''.join(parts)+text[offset:];assert value==(root/row['file']).read_text(),row['file']
  for start,end,new,old in reversed(inverse):
   assert value[start:end]==new;value=value[:start]+old+value[end:]
  assert value==text
for path in closure['local_files']:
 if Path(path).suffix=='.py':ast.parse(Path(path).read_text(),filename=path)
ast.parse((G/'guard202.py').read_text())
for name in('launch202.sh','startup202.sh'):subprocess.run(['bash','-n',str(G/name)],check=True)
spec=importlib.util.spec_from_file_location('_ROOT202_source_review_guard',G/'guard202.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
state=guard.source_state(archive=True);guards=guard.guard_inputs()
assert state['local']==closure['local_files']and state['external']==closure['readonly_inputs']and state['evidence']==closure['retained_HOST_evidence']and guards==closure['execution_guard_inputs']
assert [len(state[k])for k in('local','external','evidence')]==[26,202,523]and len(guards)==4
assert sha(guard.CPU)=='771734febadf966906afa18067557a3c591f86e52280508b5cc28762bc353f64'
assert not[p for p in state['external']if p.endswith('.tar.gz')]
base_inputs=guard.merge(state['local'],state['external'],state['evidence'],guards,{str(guard.CPU):sha(guard.CPU)})
assert base_inputs==closure['reviewed_inputs']and len(base_inputs)==756
aux=closure['auxiliary_review_evidence'];assert len(aux)==2 and not(set(aux)&set(base_inputs))
assert guard.merge(base_inputs,aux)==closure['all_pins']and len(closure['all_pins'])==758
inputs=guard.merge(base_inputs,{str(peer):sha(peer)});assert len(inputs)==757 and'torch'not in sys.modules
review=D/'ROOT_V202_SEVEN_REAL_N_REV3_SOURCE_AND_GUARD_REVIEW.json'
save(review,dict(schema='ROOT.V202.seven-real-N-existing80-REV3-source-review.v1',UTC=datetime.now(timezone.utc).isoformat(),
 status='passed_source_review',candidate=str(P),reviewed_inputs=inputs,ready=pin(ready),peer=pin(peer),
 auxiliary_ready_evidence=aux,auxiliary_ready_evidence_is_outside_guard_archive=True,
 preserved_revision_members={'REVISION1':42,'REVISION2':44},parent_archives_verified_in_place=state['parent_archive'],
 whole_private_projections=17,whole_guard_projections=4,ROOT_source_reviewed=True,HOST_tests_rerun=False,NPU_used=False,Torch_imported=False,
 reviewed_semantics=[
 'Sole production parent197. Original current201/44/114, fullB4/M80 and197 independent N2/N32 prefill/next1&16 gates execute anew before usage issuer; no old device receipt authorizes a new instance.',
 'Only realN9..15, allB4, causal Q5N maps through the original pad function to existingM80/S16 with original q5 dummy owners. All other mapper branches delegate to original173; no new Native key or MC2 profile is introduced.',
 'Fresh original bank keeps16 candidate and16 independent reference histories; all64 production scratch owners463..526 are genuinely allocated128/544 pages. Candidate old128 scratch463..491 and independent reference scratch492..520 remain disjoint.',
 'Seven actual same-engine usages require full80 physical hidden/head/all128 queryKV including dummy, protected unselected owners, original typed publication/consensus/commit and own next16. Old128 greedy/head/queryKV is independently checked; crossM differences stay numerical variants.',
 'Three fresh four-cycle traces perN plus actual own-next require1176 zero counter subtractions; both preexisting G80/H80 object identities/flags/keys remain unchanged. Actual changed host-length task update replay is required.',
 'REV2 causal predicate does not widen parent production B4fork scope: frozen target_layout already rejects it. Separate generic Native full9 ABI carrier passthrough is a wrapper boundary check, not observed parent production regression or a new tree permission.',
 'REV3 wrapper only checks frames while the exact mapper delegate, its owning control, same engine and candidate mode are installed. Original-mode/uninstalled delegate returns the original same frame with permission retained; candidate corrupt frame remains rejected.',
 'Scope report uses the component combined old/new numerical-variant classification. Source domain/math actor/check_frame hard gates, original g/BONUS/global192/C96/E96/target16/arrivals/O256/publication/clocks and197 empty-active admission remain unchanged.',
 'Six current HOST checks execute in one exact before/after revision, including actual original Controls candidate-original-candidate transitions for all7N, original P148 permit positive/wrong197 negative, full scope/guard metadata and strict rejection cases. Nine attempts and513 source/log files plus both42/44 pre-revision snapshots are retained.'
 ],limitations=[
 'No202 current model, numerical qualification, new-domain production hit or GP has run. HOST carriers cannot regenerate hidden/logit/KV tensors.',
 'Static saved27504 physical queries on completed197 trajectory is not a latency or Goodput forecast; changed shapes may change numerical outputs/trajectory.',
 'ExistingM80 runtime validation does not qualify these seven new real/dummy usages by itself. All new actual sameM/protocol/zero/issuer gates remain mandatory.',
 'The two ready auxiliary cold artifacts are independently pinned here, outside the exact guarded756 pre-peer map and future759-member archive.',
 'Whole463 canonical own-AR, production bitwise losslessness, adoption and formal final-nine1.5x remain false.'
 ],formal_Goodput=False,adopted=False))
command=[sys.executable,'-B',str(G/'guard202.py'),'--prepare-only','--root-review',str(review),
 '--expected-root-review-sha256',sha(review),'--peer-review',str(peer),sha(peer)]
result=subprocess.run(command,capture_output=True,text=True)
save(D/'ROOT_V202_GUARD_PREPARE_STDOUT_AND_STDERR.json',dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
assert result.returncode==0,result.stderr
f,sources,prepared=guard.prepared();assert len(sources)==prepared['archive']['members']==759 and'torch'not in sys.modules
out=D/'ROOT_V202_PREPARED_SOURCE_HOST_AUDIT.json'
save(out,dict(schema='ROOT.V202.prepared-source-HOST-audit.v1',status='passed_source_frozen_unrun',candidate=str(P),
 review=pin(review),peer=pin(peer),ready=pin(ready),freeze=pin(guard.FREEZE),archive=pin(guard.ARCHIVE),
 actual_source_members=len(sources),actual_archive_members=prepared['archive']['members'],prepared=prepared,auxiliary_ready_evidence=aux,
 NPU_used=False,Torch_imported=False,models_launched=False,formal_Goodput=False,adopted=False))
print(json.dumps(dict(status='passed_source_frozen_unrun',record=pin(out),freeze=pin(guard.FREEZE),archive=pin(guard.ARCHIVE),members=len(sources))))
