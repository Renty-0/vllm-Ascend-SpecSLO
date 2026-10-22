"""ROOT source review/freeze only: original SLO admission plus empty-active197."""
import ast,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-empty-active-slo-admission1-causal80-active32-full-v203';G=D/'v203-root-execution'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p))
def save(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
ready=D/'V203_EMPTY_ACTIVE_SLO_ADMISSION1_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json'
peer=D/'V203_INDEPENDENT_SOURCE_PEER_REVIEW.json'
assert sha(ready)=='9b82e156079bad7969a361cf428c249910709bf7f828495b8b1c9f7ee331931c'
r=json.loads(ready.read_bytes());pr=json.loads(peer.read_bytes())
assert r['source_stopped']is True and r['freeze_created']is False
assert pr['status'].replace('-','_')=='passed_source_review'and pr['candidate']==str(P)
assert pr.get('blocking_findings',[])==pr.get('unresolved_source_blockers',[])==[]
for group in('files','external_files','guards','retained_HOST_evidence'):
 for path,h in r[group].items():assert sha(path)==h,path
projection=json.loads((P/'PROJECTIONS203.json').read_bytes())['projection'];assert len(projection)==14
for row in projection:
 assert sha(row['parent'])==row['parent_sha256'];text=Path(row['parent']).read_text()
 for old,new,count in row['namespace']:
  assert text.count(old)==count;text=text.replace(old,new)
 parts=[];offset=0
 for op in row['deltas']:
  assert op['start']>=offset and text[op['start']:op['end']]==op['old']
  parts.extend((text[offset:op['start']],op['new']));offset=op['end']
 assert ''.join(parts)+text[offset:]==(P/row['file']).read_text(),row['file']
for path in r['files']:
 if Path(path).suffix=='.py':ast.parse(Path(path).read_text(),filename=path)
ast.parse((G/'guard203.py').read_text())
for name in('startup203.sh','launch203.sh'):subprocess.run(['bash','-n',str(G/name)],check=True)
old=(D/'v197-root-execution/startup197.sh').read_text().replace('197','203')
old=old.replace('empty-active-cohort-causal80-active32-full463','empty-active-slo-admission1-causal80-active32-full463').replace('tree-empty-active-cohort-causal80-active32-full-v203','tree-empty-active-slo-admission1-causal80-active32-full-v203')
expected=old.replace('VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION=0','VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION=1')
policy_line='export VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION=1 VLLM_ASCEND_SPECRHYTHM_TARGET_SLO_FIRST=0\n'
assert expected.count(policy_line)==1
expected=expected.replace(policy_line,'')
anchor='"$python" -B "$task_root/check203.py"\n';assert expected.count(anchor)==1
expected=expected.replace(anchor,policy_line+anchor)
assert (G/'startup203.sh').read_text()==expected,'only startup policy delta after exact namespace'
spec=importlib.util.spec_from_file_location('_ROOT203_source_review_guard',G/'guard203.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
state=guard.source_state(archive=True);guards=guard.guard_inputs()
assert state['local']==r['files']and state['external']==r['external_files']and state['evidence']==r['retained_HOST_evidence']and guards==r['guards']
assert [len(state[k])for k in('local','external','evidence')]==[22,214,88]and len(guards)==4
assert sha(guard.CPU)=='692a5c4938b32c9750036c2f488681127ac3bf6896d4864b63c8399e9fcaf049'
assert not[p for p in state['external']if p.endswith('.tar.gz')]
inputs=guard.merge(state['local'],state['external'],state['evidence'],guards,{str(guard.CPU):sha(guard.CPU),str(peer):sha(peer)})
assert len(inputs)==330 and'torch'not in sys.modules
review=D/'ROOT_V203_EMPTY_ACTIVE_SLO_ADMISSION1_SOURCE_AND_GUARD_REVIEW.json'
save(review,dict(schema='ROOT.V203.empty-active-original-admission1-source-review.v1',UTC=datetime.now(timezone.utc).isoformat(),
 status='passed_source_review',candidate=str(P),ready=pin(ready),peer=pin(peer),reviewed_inputs=inputs,
 ROOT_source_reviewed=True,HOST_tests_rerun=False,NPU_used=False,Torch_imported=False,
 parent_archives_verified_in_place=state['parent_archive'],whole_text_projections_verified=14,
 reviewed_semantics=[
  'Sole production parent197, unchanged original empty-active guard after receive_live. Only actual original SLO_AWARE_ADMISSION changes0 to1 in clean startup; TARGET_SLO_FIRST remains0.',
  'The explicit policy export moves before check203 so its strict clean-environment check can run; all other startup commands are exact parent namespace.',
 'Original priority is finite SLO then arrival then request index. Original per-candidate arrival filter excludes future requests without blocking already-arrived lower-priority ones. No new priority function, g table, budget, class quota, coalesce wait or hot-path wrapper is introduced.',
 'Private policy report validates actual Native env resolution/64capacity/463queue/527storage/coalesce1-0; actual trace and leader-result counters must be integer1/0/coalesce0 and all requests genuinely prefilled.',
 'Original current engine201/44/114, fullB4/M80, independent N2/N32 multiowner prefill and ownnext1/16 mathematics, original protocol and retained entry objects all execute anew before issuer; no old receipt permits a new engine.',
 'Original prewarm, warm65, two complete65 Native21-zero traces, seal and full463 remain. Original model weights, MC2/NZ/graph source, global192/C96/E96/target16, gamma/budget estimator and256-token publication/charging/clocks are unchanged.',
 'Three actual new HOST checks execute original generated admit_available plus Native params/order with declared control carriers: active early return, future exclusion, token budget, stable ordering, immediate roots and finite463 unique drain. They do not prove model tensors.',
 'Actual original P148 permit and runtime-spec identity positive/wrong197 negative, strict source/freeze/receipt177 storage/selflock/owned cleanup are tested. Three attempts and all84 source/log files are retained; parent suites are not rerun.'
 ],limitations=[
 'No203 model or performance measurement has occurred. The source-only admission carrier active.clear is a declared completion boundary, not model execution.',
 'SLO ordering is not g-aware or guaranteed homogeneous grouping. Loose/normal can wait longer; report original-arrival TTFT by class beside Native-charged Goodput.',
 'Arrival-to-first and worker-publication/255 clocks must stay separately named from charged/256 and common-origin final-nine measurement. No new performance prediction follows from source control.',
 'Existing guardian new_target_keys2/cost_measured template labels are inherited173 scope metadata, not two newly introduced203 keys or measured203 cost.',
 'Whole463 own-AR, canonical numerical losslessness, adoption and formal final-nine1.5x remain false.'
 ],formal_Goodput=False,adopted=False))
command=[sys.executable,'-B',str(G/'guard203.py'),'--prepare-only','--root-review',str(review),
 '--expected-root-review-sha256',sha(review),'--peer-review',str(peer),sha(peer)]
result=subprocess.run(command,capture_output=True,text=True)
save(D/'ROOT_V203_GUARD_PREPARE_STDOUT_AND_STDERR.json',dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
assert result.returncode==0,result.stderr
f,sources,prepared=guard.prepared();assert len(sources)==prepared['archive']['members']==332 and'torch'not in sys.modules
out=D/'ROOT_V203_PREPARED_SOURCE_HOST_AUDIT.json'
save(out,dict(schema='ROOT.V203.prepared-source-HOST-audit.v1',status='passed_source_frozen_unrun',candidate=str(P),
 review=pin(review),peer=pin(peer),ready=pin(ready),freeze=pin(guard.FREEZE),archive=pin(guard.ARCHIVE),
 actual_source_members=len(sources),actual_archive_members=prepared['archive']['members'],prepared=prepared,
 NPU_used=False,Torch_imported=False,models_launched=False,formal_Goodput=False,adopted=False))
print(json.dumps(dict(status='passed_source_frozen_unrun',record=pin(out),freeze=pin(guard.FREEZE),archive=pin(guard.ARCHIVE),members=len(sources))))
