"""ROOT source review/freeze: single-writer seven-tail usage, original mathematics."""
import ast,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-causal80-small-cohort-single-writer-full-v205';G=D/'v205-root-execution'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p))
def save(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
ready=D/'V205_SINGLE_WRITER_SEVEN_N_FROM_FROZEN202_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json'
peer=D/'V205_INDEPENDENT_SOURCE_PEER_REVIEW.json'
assert sha(ready)=='737be8d457bab4228acd3d1dfe473f34b8c8fa6632967712d9789a9a7d16d1b0'
assert sha(peer)=='67794d6c4240cad58d74074314cf901ad2003a93b1ee3ffe9db54e5a978ff2dd'
r=json.loads(ready.read_bytes());pr=json.loads(peer.read_bytes());closure=r['source_closure']
assert r['source_stopped']is True and r['freeze_created']is False
assert pr['status'].replace('-','_')=='passed_source_review'and pr['candidate']==str(P)
assert pr.get('blocking_findings_remaining',[])==[]
for group in('local_files','readonly_inputs','execution_guard_inputs','retained_HOST_evidence','auxiliary_review_evidence'):
 for path,h in closure[group].items():assert sha(path)==h,path
projection=json.loads((P/'PROJECTIONS205.json').read_bytes())
assert len(projection['projection'])==23 and len(projection['guard_projection'])==4
for group,root in(('projection',P),('guard_projection',G)):
 for row in projection[group]:
  assert sha(row['parent'])==row['parent_sha256'];text=Path(row['parent']).read_text().replace(*row['namespace'])
  parts=[];offset=0;shift=0;inverse=[]
  for op in row['deltas']:
   assert op['start']>=offset and text[op['start']:op['end']]==op['old']
   parts.extend((text[offset:op['start']],op['new']));offset=op['end']
   start=op['start']+shift;inverse.append((start,start+len(op['new']),op['new'],op['old']));shift+=len(op['new'])-len(op['old'])
  projected=''.join(parts)+text[offset:];value=projected+row.get('append_text','')
  assert value==(root/row['file']).read_text(),row['file']
  for start,end,new,old in reversed(inverse):
   assert projected[start:end]==new;projected=projected[:start]+old+projected[end:]
  assert projected==text
source_parent=B/'tree-causal80-small-cohort-active32-full-v202'
old=(source_parent/'actor202.py').read_text().replace('202','205')
anchor='lambda:folder.mkdir(exist_ok=False)';assert old.count(anchor)==1
assert old.replace(anchor,anchor+'if engine.rank==engine.topology.draft_leader_rank else None')==(P/'actor205.py').read_text()
for name in('audit','audit_usage','bank','component','config','empty','extension','mapper','prefill','progress','publication','runtime','scope','source','subset','target','trace'):
 assert (source_parent/(name+'202.py')).read_text().replace('202','205')==(P/(name+'205.py')).read_text(),name
assert (source_parent/'next4_202.py').read_text().replace('202','205')==(P/'next4_205.py').read_text()
for path in closure['local_files']:
 if Path(path).suffix=='.py':ast.parse(Path(path).read_text(),filename=path)
ast.parse((G/'guard205.py').read_text())
for name in('launch205.sh','startup205.sh'):
 text=(G/name).read_text();assert '20561015'not in text and '20261015'in text
 subprocess.run(['bash','-n',str(G/name)],check=True)
spec=importlib.util.spec_from_file_location('_ROOT205_source_review_guard',G/'guard205.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
state=guard.source_state(archive=True);guards=guard.guard_inputs()
assert state['local']==closure['local_files']and state['external']==closure['readonly_inputs']and state['evidence']==closure['retained_HOST_evidence']and guards==closure['execution_guard_inputs']
assert [len(state[k])for k in('local','external','evidence')]==[26,760,174]and len(guards)==4
assert sha(guard.CPU)=='a560a56c807c3600e231e230b80869768731c92772e47f707f4b7fb2520b3c78'
cpu=json.loads(guard.CPU.read_bytes());assert cpu['tests_run']==7
directory=cpu['details']['actual_single_writer']
assert directory['real_four_process_shared_temporary_FS']and directory['actual_old_and_new_actor_mkdir_lambdas_AST_executed']
assert directory['old_three_FileExistsErrors']and directory['new_all4_pass']and directory['preexisting_leader_FileExistsError_and_all4_common_reject']
assert directory['NPU_stage_fence_not_executed_or_claimed']is True
assert cpu['details']['actual_launch_paths']['original_launch_argv_AST_executed']and cpu['details']['actual_launch_paths']['launch_startup_runtime_map_paths_exist']
assert not[p for p in state['external']if p.endswith('.tar.gz')]
base_inputs=guard.merge(state['local'],state['external'],state['evidence'],guards,{str(guard.CPU):sha(guard.CPU)})
assert base_inputs==closure['reviewed_inputs']and len(base_inputs)==965
aux=closure['auxiliary_review_evidence'];assert len(aux)==3 and not(set(aux)&set(base_inputs))
assert guard.merge(base_inputs,aux)==closure['all_pins']and len(closure['all_pins'])==968
inputs=guard.merge(base_inputs,{str(peer):sha(peer)});assert len(inputs)==966 and'torch'not in sys.modules
review=D/'ROOT_V205_SINGLE_WRITER_SEVEN_TAIL_SOURCE_AND_GUARD_REVIEW.json'
save(review,dict(schema='ROOT.V205.single-writer-seven-tail-source-review.v1',UTC=datetime.now(timezone.utc).isoformat(),
 status='passed_source_review',candidate=str(P),reviewed_inputs=inputs,ready=pin(ready),peer=pin(peer),
 auxiliary_ready_evidence=aux,auxiliary_ready_evidence_is_outside_guard_archive=True,
 parent_archives_verified_in_place=state['parent_archive'],whole_private_projections=23,whole_guard_projections=4,
 ROOT_source_reviewed=True,HOST_tests_rerun=False,NPU_used=False,Torch_imported=False,
 reviewed_semantics=[
 'Sole production parent197; frozen202 REV3 is a source donor only. All759 donor source/archive regular members are verified in place; old202 was never run and supplies no numerical qualification.',
 'Only mathematical actor control delta after namespace is leader-only mkdir(exist_ok=False). Original four-rank common stage call, position and implementation are unchanged. A preexisting directory still causes collective rejection.',
 'Current seven HOST checks run together against exact before/after source. The new check executes actual old/new actor mkdir lambdas with four real processes, shared temporary filesystem and explicit Barrier/errors common-outcome carriers. It does not execute original stage4 body or NPU fences. Those calls remain byte-identical in the actor and must run in actual model qualification.',
 'All18 non-administrative donor mathematical/runtime modules and whole actor inverse are exact namespace. Original mapper REV3 candidate/original identity and mode gates, causal filter and corrupt-frame rejection are retained.',
 'Only realN9..15 allB4 causal Q5N uses the original pad function with existingM80/S16 and original q5 dummy rows. Existing80 graph key/MC2 profile is not alone permission for seven new usages; actual current same-engine hidden/head/all128 queryKV including dummy and changed task updates remain mandatory.',
 'All64 production scratch owners463..526 are genuinely allocated128/544 pages. Sixteen candidate and sixteen independent own-history reference banks use disjoint old128 scratch ranges. Original typed work/publication/commit, protected unselected owners and own next16 remain checked.',
 'Seven real usages each require three fresh four-cycle traces plus actual own next,1176 zero counter subtractions and all inherited G80/H80 object/flag identity. Old128 controls independently check their own head/KV; crossM output differences remain numerical variants.',
 'Original current201/44/114, fullB4/M80 and197 N2/N32 prefill/ownnext qualifications execute anew before usage issuer; no historical receipt grants a current engine permission. Original P148 actual permit identity and foreign197 rejection remain required.',
 'Original full32 prewarm/warm65/two complete zero sweeps/seal/full463, exact Poisson arrivals/O256/immediate publication and charged/publication/TTFT clocks remain. C96/E96/global192/target16, original g/BONUS/cap4-lowN5 and197 empty-active scheduling do not change.',
 'All three seven-check HOST attempts, source snapshots and initial cold date-path failure are preserved. Actual original launch_argv AST executes as HOST control without launching, and validates all four launch/startup/runtime/map paths. Original20261015 date and inherited SHA literals stay intact.',
 'Three extra readiness/cold/inverse audit artifacts are independently pinned outside exact965 guarded pre-peer inputs. The peer final67794d6c supersedes its briefly visible WIP metadata; source/CPU/guard are unchanged.'
 ],limitations=[
 'No205 model, new numerical usage or full performance result has run at source freeze. Static saved27504 physical queries in completed197 are not a time or GP forecast.',
 'HOST filesystem/barrier carriers qualify only directory control. They are not device collectives and cannot recompute unretained hidden/logit/KV tensors.',
 'One inherited limitation says no202 model/performance has run and guardian cost_measured labels refer to old metadata; neither claims new205 measured cost or numerical qualification.',
 'SLO ordering change203, finalKV204, fused verdict196, B3/newg and other candidates are not combined here. Whole463 own-AR, production bitwise losslessness, adoption and formal final-nine1.5x remain false.'
 ],formal_Goodput=False,adopted=False))
command=[sys.executable,'-B',str(G/'guard205.py'),'--prepare-only','--root-review',str(review),
 '--expected-root-review-sha256',sha(review),'--peer-review',str(peer),sha(peer)]
result=subprocess.run(command,capture_output=True,text=True)
save(D/'ROOT_V205_GUARD_PREPARE_STDOUT_AND_STDERR.json',dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
assert result.returncode==0,result.stderr
f,sources,prepared=guard.prepared();assert len(sources)==prepared['archive']['members']==968 and'torch'not in sys.modules
out=D/'ROOT_V205_PREPARED_SOURCE_HOST_AUDIT.json'
save(out,dict(schema='ROOT.V205.prepared-source-HOST-audit.v1',status='passed_source_frozen_unrun',candidate=str(P),
 review=pin(review),peer=pin(peer),ready=pin(ready),freeze=pin(guard.FREEZE),archive=pin(guard.ARCHIVE),
 actual_source_members=len(sources),actual_archive_members=prepared['archive']['members'],prepared=prepared,auxiliary_ready_evidence=aux,
 NPU_used=False,Torch_imported=False,models_launched=False,formal_Goodput=False,adopted=False))
print(json.dumps(dict(status='passed_source_frozen_unrun',record=pin(out),freeze=pin(guard.FREEZE),archive=pin(guard.ARCHIVE),members=len(sources))))
