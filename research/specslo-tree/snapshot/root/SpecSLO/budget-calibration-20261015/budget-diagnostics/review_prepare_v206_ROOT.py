"""ROOT review/freeze of bounded joint-tree diagnostic directory repair."""
import ast,hashlib,importlib.util,io,json,re,subprocess,sys,tokenize
from pathlib import Path
from datetime import datetime,timezone
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-b4-joint-subtree-single-writer-diagnostic-v206';G=D/'v206-root-execution'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p))
def save(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def namespace(text):
 out=[]
 for token in tokenize.generate_tokens(io.StringIO(text).readline):
  value=token.string;literal=None
  if token.type in(tokenize.NAME,tokenize.COMMENT,tokenize.STRING):
   if token.type==tokenize.STRING:
    try:literal=ast.literal_eval(value)
    except Exception:pass
   if not(isinstance(literal,str)and re.fullmatch('[0-9a-f]{64}',literal)):value=value.replace('198','206')
  out.append(token._replace(string=value))
 return tokenize.untokenize(out)
ready=D/'V206_SINGLE_WRITER_BOUNDED_JOINT_SUBTREE_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json'
peer=D/'V206_INDEPENDENT_SOURCE_PEER_REVIEW.json'
assert sha(ready)=='0376b94503ec97a39950c990e29d0979c9fa52542189fd64db687f12b952128d'
assert sha(peer)=='b0335d46323c1f0ecb3c5d308daa044eb406c99212408e8e935190c527a06ece'
r=json.loads(ready.read_bytes());pr=json.loads(peer.read_bytes())
assert r['source_stopped']is True and r['freeze_created']is False
assert pr['status'].replace('-','_')=='passed_source_review'and pr['candidate']==str(P)
assert not pr.get('blocking_findings_remaining',[])and not pr.get('blocking_findings',[])
for group in('files','external_files','guards','retained_HOST_evidence'):
 for path,h in r[group].items():assert sha(path)==h,path
projection=json.loads((P/'DERIVATION206.json').read_bytes())['projection'];assert len(projection)==16
for row in projection:
 assert sha(row['parent'])==row['parent_sha256'];text=namespace(Path(row['parent']).read_text());parts=[];offset=0
 for op in row['deltas']:
  assert op['start']>=offset and text[op['start']:op['end']]==op['old']
  parts.extend((text[offset:op['start']],op['new']));offset=op['end']
 assert ''.join(parts)+text[offset:]==(P/row['file']).read_text(),row['file']
old=namespace((B/'tree-b4-joint-subtree-m80-gain-diagnostic-v198/actor198.py').read_text())
anchor='lambda:folder.mkdir(exist_ok=False)';assert old.count(anchor)==1
assert old.replace(anchor,anchor+'if engine.rank==engine.topology.draft_leader_rank else None')==(P/'actor206.py').read_text()
for name in('audit','bank','draft','extension','host_tensor','progress','runtime','selector','source','target'):
 assert namespace((B/'tree-b4-joint-subtree-m80-gain-diagnostic-v198'/(name+'198.py')).read_text())==(P/(name+'206.py')).read_text(),name
for path in r['files']:
 if Path(path).suffix=='.py':ast.parse(Path(path).read_text(),filename=path)
ast.parse((G/'guard206.py').read_text())
for name in('launch206.sh','startup206.sh'):
 assert '20261015'in(G/name).read_text()
 subprocess.run(['bash','-n',str(G/name)],check=True)
spec=importlib.util.spec_from_file_location('_ROOT206_source_review_guard',G/'guard206.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
state=guard.source_state(archive=True);guards=guard.guard_inputs()
assert state['local']==r['files']and state['external']==r['external_files']and state['evidence']==r['retained_HOST_evidence']and guards==r['guards']
assert [len(state[k])for k in('local','external','evidence')]==[19,193,27]and len(guards)==4
assert sha(guard.CPU)=='2138fc214f2f557ae35cbe10b34e505184cba4c6ec7025b9a895f74617b925a3'
cpu=json.loads(guard.CPU.read_bytes());assert cpu['tests_run']==5
writer=cpu['details']['single_writer_directory']
assert writer['actual_actor_lambda']and writer['actual_original_stage4']and writer['real_four_process_shared_FS']and writer['HOST_collective_and_sync_carriers']
assert [(x['case'],x['local_FileExists_errors'],x['all4_passed'],x['common_all4_result_identical'])for x in writer['cases']]==[
 ('old-race',3,False,True),('new-single-writer',0,True,True),('new-preexisting-refused',1,False,True)]
assert not[p for p in state['external']if p.endswith('.tar.gz')]
inputs=guard.merge(state['local'],state['external'],state['evidence'],guards,{str(guard.CPU):sha(guard.CPU),str(peer):sha(peer)})
assert len(inputs)==245 and'torch'not in sys.modules
review=D/'ROOT_V206_SINGLE_WRITER_BOUNDED_JOINT_TREE_SOURCE_AND_GUARD_REVIEW.json'
save(review,dict(schema='ROOT.V206.single-writer-bounded-joint-tree-source-review.v1',UTC=datetime.now(timezone.utc).isoformat(),
 status='passed_source_review',candidate=str(P),ready=pin(ready),peer=pin(peer),reviewed_inputs=inputs,
 ROOT_source_reviewed=True,HOST_tests_rerun=False,NPU_used=False,Torch_imported=False,
 parent_archives_verified_in_place=state['parent_archive'],whole_text_projections_verified=16,
 reviewed_semantics=[
 'Frozen198 is a source-only donor; its320 source/archive regular members are verified in place. Original173 remains the actual mathematical basis. Frozen198 known directory race stays unchanged and never runs.',
 'Only actor execution delta after token namespace is draft-leader-only mkdir(exist_ok=False) in the original common startup stage. All4 stage position/implementation and preexisting-directory rejection remain.',
 'Token namespace preserves numeric literals including HOST random seed198, all64hex string pins and dates. Complete whole source projections and actor inverse are checked; selector, bank, draft, target, audit, runtime and mathematical guards are unchanged.',
 'Five current HOST checks ran together against exact source pre/post, with all25 snapshots/log/results retained. Added real four-process sharedFS check executes actual actor lambdas and original actor_v146.stage4 body through named HOST synchronization/collective carriers.',
 'The directory worker is AST-reused from an immutable saved204 HOST snapshot, not204 actual/model proof; actual model startup still uses original startup_receipts stage. No device collective is established by HOST carriers.',
 'Bounded diagnostic only: no production issuer, full463 switch or g/posterior installation. Current original constructor/B4/M80 mathematics execute anew before joint-tree tests.',
 'Real raw8 top2 probabilities choose four ancestor-closed nodes: six controlled true-fork shapes and two linear controls, each three fresh full-protocol traces. Current sameM80 fullphysical hidden/head/all128queryKV and compacted-path own reference/original verdict/commit/next are mandatory.',
 'Only necessary diagnostic full9 hidden key naturally adds; inherited actual entry objects, flags and keys remain. Both complete new-topology zero sweeps and actual changed task updates remain strict.',
 'Three classes each use twelve correlated oldV15 prefixes plus four excluded filler owners. Old and joint policies use fresh independent histories,36 each; no-fork or no-side-accept natural outcomes stay subset.',
 'Original selflocking guardian, saved original baseline10496/15/212 pre/post, storage177 handling, owned cleanup and four idle cards remain mandatory.'
 ],limitations=[
 'No206 device mathematics, observed gain or cost has run at source freeze. Existing198 source/HOST proof does not issue current engine permission.',
 'Cost samples run old three then joint three, not ABBA. Timer includes greedy output clone/shape-key-flag checks, explicit all4 stages/drains and own next. Observer/reference are excluded but this is neither pure production latency nor Goodput; per-rank costs cannot be added.',
 'Twelve source-correlated prefixes per class do not determine a complete g function; initial roots are teacher forced. Natural useful-fork coverage can fail or remain subset.',
 'HOST cannot regenerate unretained floating hidden/logit/KV. New production budget policy, whole463 own-AR, adoption and formal final-nine1.5x remain false.'
 ],formal_Goodput=False,adopted=False))
command=[sys.executable,'-B',str(G/'guard206.py'),'--prepare-only','--root-review',str(review),
 '--expected-root-review-sha256',sha(review),'--peer-review',str(peer),sha(peer)]
result=subprocess.run(command,capture_output=True,text=True)
save(D/'ROOT_V206_GUARD_PREPARE_STDOUT_AND_STDERR.json',dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
assert result.returncode==0,result.stderr
f,sources,prepared=guard.prepared();assert len(sources)==prepared['archive']['members']==247 and'torch'not in sys.modules
out=D/'ROOT_V206_PREPARED_SOURCE_HOST_AUDIT.json'
save(out,dict(schema='ROOT.V206.prepared-source-HOST-audit.v1',status='passed_source_frozen_unrun',candidate=str(P),
 review=pin(review),peer=pin(peer),ready=pin(ready),freeze=pin(guard.FREEZE),archive=pin(guard.ARCHIVE),
 actual_source_members=len(sources),actual_archive_members=prepared['archive']['members'],prepared=prepared,
 NPU_used=False,Torch_imported=False,models_launched=False,formal_Goodput=False,adopted=False))
print(json.dumps(dict(status='passed_source_frozen_unrun',record=pin(out),freeze=pin(guard.FREEZE),archive=pin(guard.ARCHIVE),members=len(sources))))
