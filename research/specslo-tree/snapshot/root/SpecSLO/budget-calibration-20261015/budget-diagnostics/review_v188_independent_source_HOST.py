"""Independent read-only mechanical peer; no candidate import/test/backend work."""
import ast,datetime,difflib,hashlib,json,subprocess
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-parent3-union3-padding-observer-devicefix-full-v188';G=D/'v188-root-execution'
R=D/'V188_PADDING_OBSERVER_DEVICE_INDEX_FIX_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json';O=D/'V188_INDEPENDENT_SOURCE_PEER_REVIEW.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
assert sha(R)=='95e68df8c5186a7bd4b10c0335f490ed62d5796259e5908607c2cc98f5a83c11'
r=read(R);c=read(P/'CPU_RESULT188.json');pr=read(P/'PREPARED_HOST188.json');projection=read(P/'OBSERVER_PROJECTION188.json');coldp=Path(r['cold_actual_guard_source_state']['path']);cold=read(coldp)
assert sha(P/'CPU_RESULT188.json')==r['CPU']['sha256']=='397c57ad26e9aad61ae1a9e2b91a19c7c6c80a2ae9cd992f965a9a1b7464c0d9'
assert c['status']=='passed'and c['tests_run']==2 and c['failures']==c['errors']==0 and c['Torch_imported']is c['NPU_used']is c['model_constructed']is c['model_math_qualified']is False
assert c['old36_oracle_carriers_rerun']is c['original_parent_suite_rerun']is False
assert c['local_before']==c['local_after']==r['files']==pr['files'] and c['external_before']==c['external_after']==r['external_files']==pr['external_files']
assert (len(r['files']),len(r['external_files']),len(r['execution_guard_inputs']),len(r['retained_HOST_evidence']))==(39,810,4,137)
flat={}
for name in ('files','external_files','execution_guard_inputs','retained_HOST_evidence'):
 for path,h in r[name].items():
  assert path not in flat or flat[path]==h
  assert sha(path)==h;flat[path]=h
flat[str(P/'CPU_RESULT188.json')]=sha(P/'CPU_RESULT188.json')
assert pr['CPU_sha256']==sha(P/'CPU_RESULT188.json') and set(pr['actual_HOST_attempts'])=={str(p)for p in P.glob('HOST_ATTEMPT188_*.json')}
assert len(pr['actual_HOST_attempts'])==3 and len(pr['preserved_preparation_sources'])==133
expected={str(P/'PREPARED_HOST188.json'):sha(P/'PREPARED_HOST188.json'),**pr['actual_HOST_attempts'],**pr['preserved_preparation_sources']}
assert expected==r['retained_HOST_evidence']
for p,h in c['preserved_preparation_sources'].items():assert pr['preserved_preparation_sources'][p]==h
attempts=[]
for p,h in pr['actual_HOST_attempts'].items():
 a=read(p);assert sha(p)==h and a['tests_run']==2
 for q,w in a['preserved_preparation_sources'].items():assert pr['preserved_preparation_sources'][q]==w==sha(q)
 logs=[q for q in a['preserved_preparation_sources']if q.endswith('unittest.log')];assert len(logs)==1
 attempts.append(dict(path=p,sha256=h,status=a['status'],failures=a['failures'],errors=a['errors'],log_sha256=sha(logs[0])))
assert [x['status']for x in attempts]==['failed','passed','passed']
assert len(c['details']['actual_inspector_control']['cases'])==12 and all(v['actual_inspector']is v['original_rejected']is True and v['queryKV_arrays']==56 for v in c['details']['actual_inspector_control']['cases'])
# Full source inverse, implemented independently rather than calling a checker.
assert len(projection['files'])==34
for name,row in projection['files'].items():
 source=Path(row['parent']);assert sha(source)==row['parent_sha256']
 t=source.read_text()
 for key in ('namespace','basename_namespace','epoch_namespace'):t=t.replace(*row[key])
 for change in row['changes']:
  assert t.count(change['old'])==change['count'];t=t.replace(change['old'],change['new'])
 assert t==(P/name).read_text()and sha(P/name)==row['sha256']
assert [n for n,x in projection['files'].items()if x['changes'] and n.endswith('.py')]==['observer188.py']
assert [n for n,x in projection['files'].items()if x['changes'] and not n.endswith('.py')]==['SCOPE188.txt']
obs=(P/'observer188.py').read_text();parent=Path(projection['files']['observer188.py']['parent']).read_text();diff=''.join(difflib.unified_diff(parent.splitlines(True),obs.replace('188','187').splitlines(True),fromfile='parent187',tofile='188-normalized'))
assert 'device=value.device' in obs and 'device=engine.device'not in ast.unparse(next(n for n in ast.parse(obs).body if isinstance(n,ast.FunctionDef)and n.name=='inspect_indices'))
# Prior independent_frame fix is exactly preserved by the whole-file inverse.
assert not projection['files']['pure188.py']['changes']
assert 'device=value.device'in (P/'pure188.py').read_text()
asts=0
for q in [*[Path(p)for p in r['files']if p.endswith('.py')],G/'guard188.py']:
 compile(ast.parse(q.read_text()),str(q),'exec');asts+=1
for q in (G/'launch188.sh',G/'startup188.sh'):subprocess.run(['/bin/bash','-n',str(q)],check=True,capture_output=True)
# Only the one exact old binary is omitted, while its external identity stays.
dep=str(B/'tree-natural-positive-union-discovery-local-v154/PREPARED_SOURCE_ARCHIVE_V154.tar.gz')
assert {p for p in r['external_files']if p.endswith('.tar.gz')}=={dep}
assert sha(dep)==r['external_files'][dep]
gsource=(G/'guard188.py').read_text();an=next(n for n in ast.parse(gsource).body if isinstance(n,ast.FunctionDef)and n.name=='archive_sources')
body=ast.unparse(an)
assert "if p != dependency"in body and "{p for p in sources if p.endswith('.tar.gz')} == {dependency}"in body
assert "dependency in frozen['external_files']"in body and 'sources[dependency] == sha(dependency)'in body
assert 'PREPARED_SOURCE_ARCHIVE_V154.tar.gz'in body and 'source_state' in gsource
assert "LOCK=B/'.ROOT_NPU_EXECUTION.lock'"in gsource and "live_roots=(P/'actual',)"in gsource
assert 'MAX_SECONDS=7200;MAX_TAG_BYTES=256*1024**2;MAX_AGGREGATE_BYTES=512*1024**2'in gsource
assert sha(coldp)==r['cold_actual_guard_source_state']['sha256']=='df86490cc9c719d7a2623083ac90dc4606ae3d361557ecf4a5e3571fe2241c24'
assert cold['status']=='passed'and cold['Torch_imported']is cold['NPU_used']is False
s=cold['source_state'];assert s['local']==r['files']and s['external']==r['external_files']and s['evidence']==r['retained_HOST_evidence']and cold['guard_inputs']==r['execution_guard_inputs']
assert s['parent_archive']['P187']['archive_verified_in_place']is True and s['parent_archive']['P187']['old36_oracle_carriers_rerun']is False
assert s['parent_archive']['original']['P148']['members']==6390 and s['parent_archive']['original']['P160']['members']==42
storage=str(D/'v177-root-safe-storage-overlay/safe_storage177.py');assert storage in r['external_files']
assert not (P/'FROZEN_SOURCE188.json').exists() and not (P/'PREPARED_SOURCE_ARCHIVE188.tar.gz').exists()
sourcepaths=[R,P/'observer188.py',P/'projection188.py',P/'check188.py',P/'host188.py',P/'base188.py',P/'OBSERVER_PROJECTION188.json',P/'INPUTS188.json',P/'pure188.py',G/'guard188.py',Path(storage),B/'tree-packed-eager01-cost-v107/padding_reference.py',P/'epoch188.py',Path(__file__)]
out=dict(schema='V188.independent-source-and-guard-peer.v1',status='passed_source_review',candidate=str(P),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),ready=dict(path=str(R),sha256=sha(R)),CPU=r['CPU'],cold_actual_guard=r['cold_actual_guard_source_state'],reviewed_counts=dict(unique_flat_inputs=len(flat),local=39,selected_external=810,guards=4,retained_evidence=137,whole_source_projections=34,AST_compiled=asts,bash_n=2),reviewed_inputs=flat,reviewed_source_pins={str(p):sha(p)for p in sourcepaths},actual_attempts_retained=attempts,
 conclusion=dict(no_reviewed_source_blocker=True,only_model_adjacent_delta='Component padding observer indexes all five tensor fields on each value.device; original metadata/model input/order and original inspector checks remain unchanged.',prior_oracle187_fix_exactly_preserved=True,full_parent_projection_exact=True,archive_omission=dict(path=dep,sha256=sha(dep),only_exact_binary_excluded=True,remains_in_external_identity=True,verified_current_SHA_and_original_parent_archive_in_place=True,no_source_evidence_omitted=True),original_lock_verifier_timeout_cleanup_scope_unchanged=True),exact_observer_normalized_diff=diff,
 scope_limits=['Reviewed retained 12 actual-function HOST mixed-device carrier cases and old-source negative logs; did not rerun suites or instantiate real tensors/models.','The inspector cache carrier checks control/device/slot invariants, not model arithmetic.','Prior187 three HOST/36 oracle results are source provenance only; failed187 provides no union or full463 model qualification.','New constructor/originalB4/newunion representatives/natural protocol/twozero/seal/full463 are still required; no performance or current-device qualification is granted.'],tests_rerun=False,backend_imported=False,Torch_imported=False,NPU_used=False,device_queried=False,model_constructed=False,freeze_created=False,frozen_source_modified=False,adopted=False)
assert not O.exists();O.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(dict(path=str(O),sha256=sha(O),bytes=O.stat().st_size,counts=out['reviewed_counts'])))
