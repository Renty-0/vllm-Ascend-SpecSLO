"""ROOT source review/freeze: leader-only directory, unchanged final-KV math."""
import ast,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-normal-b4-final-kv5-single-writer-full-v204';G=D/'v204-root-execution'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p))
def save(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
ready=D/'V204_SINGLE_WRITER_FINAL_KV5_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json'
peer=D/'V204_INDEPENDENT_SOURCE_PEER_REVIEW.json'
assert sha(ready)=='7661a8fcc3ca6e164464655f80d32f76c7087db32b0fa90bf1fabb6819e084a4'
assert sha(peer)=='8c4c38efd76e6006a843ae2a48cc08821508661014e9b168f6fbfca6ed68ecef'
r=json.loads(ready.read_bytes());pr=json.loads(peer.read_bytes())
assert r['source_stopped']is True and r['freeze_created']is False
assert pr['status'].replace('-','_')=='passed_source_review'and pr['candidate']==str(P)
assert pr.get('blocking_findings',[])==pr.get('unresolved_source_blockers',[])==[]
for group in('files','external_files','guards','retained_HOST_evidence'):
 for path,h in r[group].items():assert sha(path)==h,path
projection=json.loads((P/'PROJECTIONS204.json').read_bytes())['projection'];assert len(projection)==21
for row in projection:
 assert sha(row['parent'])==row['parent_sha256'];text=Path(row['parent']).read_text()
 for old,new,count in row['namespace']:
  assert text.count(old)==count;text=text.replace(old,new)
 parts=[];offset=0
 for op in row['deltas']:
  assert op['start']>=offset and text[op['start']:op['end']]==op['old']
  parts.extend((text[offset:op['start']],op['new']));offset=op['end']
 assert ''.join(parts)+text[offset:]==(P/row['file']).read_text(),row['file']
source_parent=B/'tree-normal-b4-final-kv5-cohort16-full-v201'
old=(source_parent/'actor201.py').read_text().replace('201','204')
anchor="stage('finalKV204/output',lambda:folder.mkdir(exist_ok=False))"
fixed="stage('finalKV204/output',lambda:folder.mkdir(exist_ok=False)if engine.rank==engine.topology.draft_leader_rank else None)"
assert old.count(anchor)==1 and old.replace(anchor,fixed)==(P/'actor204.py').read_text()
for name in('audit','audit_component','component','config','draft','empty','extension','host','host_tensor','lane','math','observer','progress','runtime','scope','source','trace'):
 assert (source_parent/(name+'201.py')).read_text().replace('201','204')==(P/(name+'204.py')).read_text(),name
for path in r['files']:
 if Path(path).suffix=='.py':ast.parse(Path(path).read_text(),filename=path)
ast.parse((G/'guard204.py').read_text())
for name in('startup204.sh','launch204.sh'):subprocess.run(['bash','-n',str(G/name)],check=True)
spec=importlib.util.spec_from_file_location('_ROOT204_source_review_guard',G/'guard204.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
state=guard.source_state(archive=True);guards=guard.guard_inputs()
assert state['local']==r['files']and state['external']==r['external_files']and state['evidence']==r['retained_HOST_evidence']and guards==r['guards']
assert [len(state[k])for k in('local','external','evidence')]==[24,251,32]and len(guards)==4
assert sha(guard.CPU)=='e4f53a661bc2ed8bdd0b54e8891549a2183a64c855e70861982b4363f75fe7ef'
cpu=json.loads(guard.CPU.read_bytes());assert cpu['tests_run']==6
directory=cpu['details']['single_writer_directory']
assert directory['actual_actor_lambda']and directory['actual_original_stage4']and directory['real_four_process_shared_FS']
assert [(x['case'],x['local_FileExists_errors'],x['all4_passed'],x['common_all4_result_identical'])for x in directory['cases']]==[
 ('old-race',3,False,True),('new-single-writer',0,True,True),('new-preexisting-refused',1,False,True)]
assert not[p for p in state['external']if p.endswith('.tar.gz')]
inputs=guard.merge(state['local'],state['external'],state['evidence'],guards,{str(guard.CPU):sha(guard.CPU),str(peer):sha(peer)})
assert len(inputs)==313 and'torch'not in sys.modules
review=D/'ROOT_V204_SINGLE_WRITER_FINAL_KV5_SOURCE_AND_GUARD_REVIEW.json'
save(review,dict(schema='ROOT.V204.single-writer-final-KV5-source-review.v1',UTC=datetime.now(timezone.utc).isoformat(),
 status='passed_source_review',candidate=str(P),ready=pin(ready),peer=pin(peer),reviewed_inputs=inputs,
 ROOT_source_reviewed=True,HOST_tests_rerun=False,NPU_used=False,Torch_imported=False,
 parent_archives_verified_in_place=state['parent_archive'],whole_text_projections_verified=21,
 reviewed_semantics=[
 'Sole production parent remains197. Frozen201 is a source derivation only; its failed model receipt supplies no mathematics permission. All531 original source/archive members are independently verified in place.',
 'Only component execution delta after namespace is leader-only folder.mkdir(exist_ok=False). All four ranks still call the original common stage at the original position, and preexisting folders still fail collectively. Original exception propagation is unchanged.',
 'Six actual current HOST checks ran together. The added check uses four real stdlib processes and shared filesystem with the exact old/new actor lambdas and unchanged original stage4 body: old three local failures/common failure; new all pass; preexisting one local failure/common failure. Named synchronization/collective carriers are not device execution.',
 'All17 non-administrative derivative modules and the full actor inverse are checked byte for byte; draft, target, reference, commit, graph keys, guards on actual tensor state and performance clocks are not relaxed.',
 'Current original201/44/114, B4/M80 and197 N2/N32 own-history multiowner/eager/next mathematics all run anew before installation. Only actual N16/allB4/noE/noMissing dispatches five model steps with four original top2 heads; all other cases retain original root_seed.',
 'The fifth real d4 KV write, all five hidden states, all56 query KV, first-four head/score, reservation and truthful ledger require current independent tensor comparisons; four-output Native validation alone does not qualify the fifth step.',
 'Three fresh original-protocol four-cycle histories and own next, two complete all4 Native21-zero repeats, original targetM80 and128queryKV/compaction/commit, actual original identity/defer and original P148 permit remain mandatory before issuer.',
 'One naturally validated draft steps5 final-kv key may be added. No new target key, MC2 profile, manually assigned Native flag, g table, B allocation or target BONUS change is introduced.',
 'Original197 empty-active32/target16/C96/E96/global192 scheduling, original463 Poisson arrivals/256 tokens/immediate publication, warm65/twozero/seal/full and private selflocking guardian/10496-15-212 prepost/owned cleanup are retained.',
 'Historical previous_HOST deliberately delegates the exact original check201.previous_HOST four-check source. New check204 separately binds own current six-check CPU, original201 source and sole197 production lineage. Historical checks are not model qualification.'
 ],limitations=[
 'No204 numerical model gate or performance measurement has occurred at source freeze. No speedup follows from the HOST directory repair.',
 'The fifth model step can cost more than avoided next repair; actual new usage and full results must be measured. The bounded forced old transition is qualification coverage, not natural production frequency.',
 'Nested old_four_HOST_results_retained_not_rerun and guardian own4/new_target_keys2/cost_measured template strings are inherited labels. Actual current HOST count is six; no new204 cost or target keys are thereby claimed.',
 'Native-charged/256, worker publication/255, original-arrival TTFT and future common-origin final-nine clocks remain separately named. Whole463 canonical own-AR, adoption and formal1.5x are false.'
 ],formal_Goodput=False,adopted=False))
command=[sys.executable,'-B',str(G/'guard204.py'),'--prepare-only','--root-review',str(review),
 '--expected-root-review-sha256',sha(review),'--peer-review',str(peer),sha(peer)]
result=subprocess.run(command,capture_output=True,text=True)
save(D/'ROOT_V204_GUARD_PREPARE_STDOUT_AND_STDERR.json',dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
assert result.returncode==0,result.stderr
f,sources,prepared=guard.prepared();assert len(sources)==prepared['archive']['members']==315 and'torch'not in sys.modules
out=D/'ROOT_V204_PREPARED_SOURCE_HOST_AUDIT.json'
save(out,dict(schema='ROOT.V204.prepared-source-HOST-audit.v1',status='passed_source_frozen_unrun',candidate=str(P),
 review=pin(review),peer=pin(peer),ready=pin(ready),freeze=pin(guard.FREEZE),archive=pin(guard.ARCHIVE),
 actual_source_members=len(sources),actual_archive_members=prepared['archive']['members'],prepared=prepared,
 NPU_used=False,Torch_imported=False,models_launched=False,formal_Goodput=False,adopted=False))
print(json.dumps(dict(status='passed_source_frozen_unrun',record=pin(out),freeze=pin(guard.FREEZE),archive=pin(guard.ARCHIVE),members=len(sources))))
