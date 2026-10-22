"""ROOT review and freeze of stopped final-KV5 source; never device execution."""
import ast,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-normal-b4-final-kv5-cohort16-full-v201';G=D/'v201-root-execution'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p))
def save(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
ready=D/'V201_FINAL_KV5_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW_REVISION2.json'
peer=D/'V201_INDEPENDENT_SOURCE_PEER_REVIEW.json'
assert sha(ready)=='2325adc368d892d9da39e22ee62a6f166b62e1c0155b6b9475509979614b1b24'
r=json.loads(ready.read_bytes());pr=json.loads(peer.read_bytes())
assert r['source_stopped']is True and r['freeze_created']is False
assert pr['status'].replace('-','_')=='passed_source_review'and pr['candidate']==str(P)
assert pr.get('blocking_findings',[])==[]
for key in('files','external_files','guards','retained_HOST_evidence'):
 for path,h in r[key].items():assert sha(path)==h,path
preserved=r['preserved_revision']['manifest'];assert sha(preserved['path'])==preserved['sha256']
old_manifest=json.loads(Path(preserved['path']).read_bytes())['files'];assert len(old_manifest)==218
for original,row in old_manifest.items():assert sha(row['snapshot'])==row['sha256'],original
for path,change in r['revision_change']['changed_inputs'].items():
 assert sha(path)==change['after_sha256']and sha(change['preserved'])==change['before_sha256']
assert len(r['revision_change']['changed_inputs'])==5 and r['revision_change']['mathematical_and_runner_source_unchanged']is True
old_ready=r['preserved_revision']['ready'];assert sha(old_ready['path'])==old_ready['sha256']
old=json.loads(Path(old_ready['path']).read_bytes())
old_source={**old['files'],**old['guards']};new_source={**r['files'],**r['guards']}
assert old_source.keys()==new_source.keys()
assert {p for p in old_source if old_source[p]!=new_source[p]}==set(r['revision_change']['changed_inputs'])
for row in json.loads((P/'PROJECTIONS201.json').read_bytes())['projection']:
 original=Path(row['parent']).read_text();assert sha(row['parent'])==row['parent_sha256']
 text=original
 for old,new,count in row['namespace']:
  assert text.count(old)==count; text=text.replace(old,new)
 parts=[];offset=0
 for op in row['deltas']:
  assert op['start']>=offset and text[op['start']:op['end']]==op['old']
  parts.extend((text[offset:op['start']],op['new']));offset=op['end']
 assert ''.join(parts)+text[offset:]==(P/row['file']).read_text(),row['file']
for path in r['files']:
 if Path(path).suffix=='.py':ast.parse(Path(path).read_text(),filename=path)
ast.parse((G/'guard201.py').read_text(),filename=str(G/'guard201.py'))
for name in('launch201.sh','startup201.sh'):subprocess.run(['bash','-n',str(G/name)],check=True)
spec=importlib.util.spec_from_file_location('_ROOT201_source_review_guard',G/'guard201.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
state=guard.source_state(archive=True);guards=guard.guard_inputs()
assert state['local']==r['files']and state['external']==r['external_files']and guards==r['guards']
assert state['evidence']==r['retained_HOST_evidence']
assert [len(state[k])for k in('local','external','evidence')]==[24,219,280]and len(guards)==4
assert sha(guard.CPU)=='38f406d674370d1de61658a08c2508e01b943a78871d87834214b1b654b92c15'
assert not [p for p in state['external']if p.endswith('.tar.gz')]
inputs=guard.merge(state['local'],state['external'],state['evidence'],guards,{str(guard.CPU):sha(guard.CPU),str(peer):sha(peer)})
assert len(inputs)==529 and'torch'not in sys.modules
review=D/'ROOT_V201_FINAL_KV5_SOURCE_AND_GUARD_REVIEW.json'
save(review,dict(schema='ROOT.V201.final-KV5-source-review.v1',UTC=datetime.now(timezone.utc).isoformat(),
 status='passed_source_review',candidate=str(P),ready=pin(ready),peer=pin(peer),reviewed_inputs=inputs,
 ROOT_source_reviewed=True,HOST_tests_rerun=False,NPU_used=False,Torch_imported=False,
 parent_archives_verified_in_place=state['parent_archive'],whole_text_projections_verified=7,
 retained_pre_revision_manifest=preserved,verified_preserved_pre_revision_members=218,
 reviewed_semantics=[
 'Sole production parent is197. Original startup201/44/114, full B4/M80 and actual N2/N32 multiowner eager prefill/next qualifications execute before installation in each new engine.',
 'One new actual class method is installed after parent components. Real __func__ identity satisfies original deferred-KV checks without spoofing. The original bound method and unbound fallback are retained; other engine instances use their original function.',
 'Only actual draft N16/allB4/width2/depth4/greedy/noE/no missing committed suffix may use five model steps with four original top2 heads. Other shapes, eager work and actual missing suffix use the captured original root_seed route before model execution.',
 'Private exact Fast derivative reserves all five real positions, writes root+d1+d2+d3+d4 ledger with safe_before unchanged, emits original16x4x2 encoded eight candidates and mean-over-eight confidence; target BONUS max5/g/protocol remain unchanged.',
 'Native output validation omits final metadata. Therefore separate actual independent own-history PA d4 model execution, fifth hidden and all56 queryKV, all physical first-four hidden/head/score, full metadata and page-boundary checks are mandatory before issuer.',
 'Three fresh original-protocol four-cycle histories include explicit original-method transition, real new/fallback hits, actual own next and two complete all4 Native21-zero repeats. Actual class identity/defer and Controls.select are new HOST exercised.',
 'Only one naturally validated five-metadata/four-output draft entry may be added; inherited objects/disabled keys remain identical. No new target key, MC2 profile or fabricated Native qualification flag.',
 'Original197 empty-active cohort scheduling,463 Poisson arrival IDs and256 outputs, current target geometry, immediate publication, original clocks, prewarm/warm/twozero/seal/full and private selflocked guardian/10496-15-212 prepost/owned cleanup are retained.',
  'REV2 restores actual original P148 permit identity while separate require_parent preserves sole197 production lineage. Actual original activate_local_diagnostic accepts correct runtime-spec identity, rejects wrong197 identity, and restores context on exit.',
  'Nine HOST attempts/source snapshots and old revision are retained. Five necessary current-source HOST checks execute in one final run with exact current source before/after plus cold original guard/source/archive checks; parent suites are not rerun and old archives remain verified in place.'
 ],limitations=[
 'No current201 model or fifth hidden/KV numerical gate has run at source review. HOST carriers and hashes cannot establish absent numerical tensor equality.',
 'The extra fifth model step may cost more than the avoided future repair, especially after rejection. No speedup or Goodput prediction is made.',
 'The bounded original-method transition is explicitly forced qualification coverage, not natural runtime policy. Actual new production usage must be positive and separately counted.',
  'Native validation of four output columns alone cannot qualify the fifth KV write. All current hard gates must pass before any full performance measurement.',
  'The nested permit test detail old_four_HOST_results_retained_not_rerun is a stale label from the intermediate single-test run. Current final CPU actually executes five tests on one exact before/after source revision; old four results are not merged into its count.',
 'Whole463 canonical own-AR, common-clock final-nine MC2 baseline1.5x, adoption and formal Goodput remain false.'
 ],formal_Goodput=False,adopted=False))
command=[sys.executable,'-B',str(G/'guard201.py'),'--prepare-only','--root-review',str(review),
 '--expected-root-review-sha256',sha(review),'--peer-review',str(peer),sha(peer)]
result=subprocess.run(command,capture_output=True,text=True)
save(D/'ROOT_V201_GUARD_PREPARE_STDOUT_AND_STDERR.json',dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
assert result.returncode==0,result.stderr
f,sources,prepared=guard.prepared();assert len(sources)==prepared['archive']['members']==531 and'torch'not in sys.modules
out=D/'ROOT_V201_PREPARED_SOURCE_HOST_AUDIT.json'
save(out,dict(schema='ROOT.V201.prepared-source-HOST-audit.v1',status='passed_source_frozen_unrun',candidate=str(P),
 review=pin(review),peer=pin(peer),ready=pin(ready),freeze=pin(guard.FREEZE),archive=pin(guard.ARCHIVE),
 actual_source_members=len(sources),actual_archive_members=prepared['archive']['members'],prepared=prepared,
 NPU_used=False,Torch_imported=False,models_launched=False,formal_Goodput=False,adopted=False))
print(json.dumps(dict(status='passed_source_frozen_unrun',record=pin(out),freeze=pin(guard.FREEZE),archive=pin(guard.ARCHIVE),members=len(sources))))
