"""ROOT-only source/retained-HOST review and source archive preparation; no NPU."""
import ast, hashlib, importlib.util, json, pathlib, subprocess, sys
from datetime import datetime, timezone
B=pathlib.Path('/root/SpecSLO/budget-calibration-20261015')
D=B/'budget-diagnostics';P=B/'tree-causal80-normal128-active64-full-v195';G=D/'v195-root-execution'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p))
def save(p,x):
 with p.open('x')as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
ready=D/'V195_CAUSAL80_NORMAL128_ACTIVE64_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json'
peer=D/'V195_INDEPENDENT_SOURCE_PEER_REVIEW.json'
assert sha(ready)=='96f8528e4980c69bb885d7210273a596881c055220013cf19fc95b77b0736575'
assert sha(peer)=='6f8936c5ef6ee6511fbcb534c61288d3761efc4382710f7d1147a08a0b188ced'
r=json.loads(ready.read_text());pr=json.loads(peer.read_text())
assert pr['status']=='passed_source_review'
for name,h in r['flat_review_inputs'].items():assert sha(name)==h,name
projection=json.loads((P/'PROJECTION195.json').read_text());assert len(projection['files'])==26
for name,row in projection['files'].items():
 original=pathlib.Path(row['parent']).read_text();assert sha(row['parent'])==row['parent_sha256']
 value=original.replace(row['namespace_from'],row['namespace_to'])
 assert hashlib.sha256(value.encode()).hexdigest()==row['namespace_projection_sha256'],name
 inverse=value
 for delta in row['changes']:
  assert value.count(delta['old'])==delta['count'],(name,delta['old'])
  value=value.replace(delta['old'],delta['new'])
 assert value==(P/name).read_text() and sha(P/name)==row['sha256'],name
 for delta in reversed(row['changes']):
  assert value.count(delta['new'])==delta['count'],(name,delta['new'])
  value=value.replace(delta['new'],delta['old'])
 assert value==inverse,name
for name,h in projection['new_sources'].items():assert sha(P/name)==h,name
for name in r['local_files']:
 if name.endswith('.py'):ast.parse(pathlib.Path(name).read_text(),filename=name)
for name in ('launch195.sh','startup195.sh'):subprocess.run(['bash','-n',str(G/name)],check=True)
spec=importlib.util.spec_from_file_location('_ROOT_review_guard195',G/'guard195.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
state=guard.source_state(archive=True);guards=guard.guard_inputs()
assert len(state['local'])==35 and len(state['external'])==151 and len(guards)==4
inputs=guard.merge(state['local'],state['external'],state['evidence'],guards,
 {str(P/'CPU_RESULT195.json'):sha(P/'CPU_RESULT195.json'),str(peer):sha(peer)})
assert 'torch'not in sys.modules
review=D/'ROOT_V195_CAUSAL80_NORMAL128_ACTIVE64_SOURCE_AND_GUARD_REVIEW.json'
save(review,dict(schema='ROOT.V195.source-and-guard-review.v1',UTC=datetime.now(timezone.utc).isoformat(),
 status='passed_source_review',candidate=str(P),reviewed_inputs=inputs,ready=pin(ready),peer=pin(peer),
 parent_archives_verified_in_place=state['parent_archive'],whole_text_projections_verified=26,
 ROOT_source_reviewed=True,HOST_tests_rerun=False,NPU_used=False,Torch_imported=False,model_run=False,
 reviewed_semantics=[
  'Sole production parent171: active64 C128/E64/global192, original arrivals/sampling/controls/prefill retained.',
  'Only additional production geometry is exact N16/Q80 causal M80/S16; high C97..128 retains171 M165/S34 or S36.',
  'One original constructor/capture, startup245, originalB4 proof247, independent current165 proof251, independent current80 proof253.',
  'Current165 qualification uses explicit fixture frame -> original target_tree_forward -> metadata -> Native graph runner, not scheduler padding dispatch.',
  'Original controlled verdict/commit/preflight/compact/finish and ownnext reuse existing output; no premature joint permission is needed by fixture qualification.',
  'Production mapper requires current three proof identities and six authoritative naturally validated Native entry objects; no metadata-only scope.',
  'Normal128 dual bank144 pages and causal80 dual bank108 pages release sequentially; original544 pool/527storage retained.',
  'Every component freshly computes independent histories/full physical hidden/head and original KV/verdict/commit/next; changed/restored and two zero passes required.',
  'Same-M qualification and cross-M behavior equality are reported separately; no forced flags or whole-AR promise.',
  'Fresh fixedpoint/warm65/twozero/seal/full463 and raw Native counter/proposal/arrival/output audits remain mandatory.',
  'Original guardian self-lock, full baseline pre/post, owned group cleanup and frozen177 scoped mutable-tmp storage handling preserved.',
  'All selected source/CPU/guard/evidence bytes checked;26 exact namespace-plus-declared text projections reversed; no old suites rerun.'
 ],limitations=[
  'Source/HOST freeze only; no model/device/math/performance qualification is inherited from171 or173.',
  'In completed171, exact80 eligible geometry occurred only14 of1402calls; this addition alone need not repair large-N SLO regression.',
  'Active64 arbitrary-longest-input fit is not proved by keeping544 pages; new actual full page peaks and release must be checked.',
  'Whole463 own-AR equivalence remains unqualified; earlier canonical counterexamples and diagnosis are not resolved by this mapper.'
 ],formal_Goodput=False,adopted=False))
cmd=[sys.executable,'-B',str(G/'guard195.py'),'--prepare-only','--root-review',str(review),
 '--expected-root-review-sha256',sha(review),'--peer-review',str(peer),sha(peer)]
run=subprocess.run(cmd,capture_output=True,text=True)
save(D/'ROOT_V195_GUARD_PREPARE_STDOUT_AND_STDERR.json',dict(command=cmd,returncode=run.returncode,stdout=run.stdout,stderr=run.stderr))
assert run.returncode==0,run.stderr
f,s,prepared=guard.prepared();assert len(s)==prepared['archive']['members']
assert 'torch'not in sys.modules
result=dict(schema='ROOT.V195.prepared-source-HOST-audit.v1',status='passed_source_frozen_unrun',candidate=str(P),
 review=pin(review),peer=pin(peer),freeze=pin(P/'FROZEN_SOURCE195.json'),archive=pin(P/'PREPARED_SOURCE_ARCHIVE195.tar.gz'),
 actual_source_members=len(s),actual_archive_members=prepared['archive']['members'],prepared=prepared,
 NPU_used=False,Torch_imported=False,models_launched=False,formal_Goodput=False,adopted=False)
out=D/'ROOT_V195_PREPARED_SOURCE_HOST_AUDIT.json';save(out,result)
print(json.dumps(dict(status=result['status'],record=pin(out),freeze=result['freeze'],archive=result['archive'],members=len(s))))
