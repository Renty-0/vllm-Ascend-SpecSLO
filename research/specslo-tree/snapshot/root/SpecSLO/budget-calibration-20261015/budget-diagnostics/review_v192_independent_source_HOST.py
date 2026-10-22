"""Read-only independent source peer: no producer imports, suites or backend."""
import ast,datetime,hashlib,json,subprocess
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-live-small-mixed197-cap64-active32-full-v192';G=D/'v192-root-execution'
READY=D/'V192_M197_CAP64_REAL_CURSOR_ACTIVE32_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json';OUT=D/'V192_INDEPENDENT_SOURCE_PEER_REVIEW.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
assert sha(READY)=='118f19d51ebfd019e4a2da8715143696e4f1c19645a735f89797175db1e4528e'
r=read(READY);c=read(P/'CPU_RESULT192.json');prep=read(P/'PREPARED_HOST192.json');projection=read(P/'PROJECTION192.json')
assert c['status']=='passed'and c['tests_run']==c['actual_check_count']==4 and c['errors']==c['failures']==0
assert c['Torch_imported']is c['NPU_initialized']is c['model_initialized']is c['original_suites_rerun']is False
assert c['local_before']==c['local_after']==r['local_files']==prep['files'] and c['external_before']==c['external_after']==r['external_files']==prep['external_files']
assert (len(r['local_files']),len(r['external_files']),len(r['guard_inputs']),len(r['retained_HOST_evidence']))==(39,558,4,171)
flat={}
for name in ('local_files','external_files','guard_inputs','retained_HOST_evidence'):
 for path,h in r[name].items():
  assert path not in flat or flat[path]==h
  assert sha(path)==h;flat[path]=h
flat[str(P/'CPU_RESULT192.json')]=sha(P/'CPU_RESULT192.json')
assert flat==r['reviewed_inputs_before_peer'] and len(flat)==773
assert sha(P/'CPU_RESULT192.json')==r['CPU']['sha256']==prep['CPU_sha256']=='a92d7ceea9531ad79020707814dc8e17cbf1e99524a6e78fd5dd20a7138db39b'
all_evidence={str(q):sha(q)for q in (P/'host-attempts').rglob('*')if q.is_file()}
assert all_evidence==prep['all_HOST_evidence']
assert r['retained_HOST_evidence']=={str(P/'PREPARED_HOST192.json'):sha(P/'PREPARED_HOST192.json'),**all_evidence}
attempts=[]
for q in sorted((P/'host-attempts').glob('*/result.json')):
 a=read(q);attempts.append(dict(path=str(q),sha256=sha(q),status=a['status'],tests_run=a['tests_run'],failures=a['failures'],errors=a['errors']))
assert len(attempts)==3 and attempts[-1]['status']=='passed'
assert all(v['status']=='passed'for v in c['checks'])and len(c['checks'])==4
# Independently reconstruct and reverse every full frozen172 projection.
assert projection['changes_not_finalized']is False and len(projection['projection'])==30
for row in projection['projection']:
 parent=Path(row['parent']);assert sha(parent)==row['parent_sha256']
 original=parent.read_text().replace('172','192');assert hashlib.sha256(original.encode()).hexdigest()==row['namespace_projection_sha256']
 result=original
 for delta in reversed(row['edits']):
  assert result[delta['start']:delta['end']]==delta['old']
  result=result[:delta['start']]+delta['new']+result[delta['end']:]
 assert result==(P/row['output']).read_text() and sha(P/row['output'])==row['current_sha256']
for name,h in projection['new_files'].items():assert sha(P/name)==h
asts=0
for q in [*[Path(v)for v in r['local_files']if v.endswith('.py')],G/'guard192.py']:
 compile(ast.parse(q.read_text()),str(q),'exec');asts+=1
for q in (G/'startup192.sh',G/'launch192.sh'):subprocess.run(['/bin/bash','-n',str(q)],check=True,capture_output=True)
# Explicit route, scope and current-instance gates reviewed in actual sources.
recipe=(P/'recipe192.py').read_text();scope=(P/'scope192.py').read_text();decode=(P/'decode192.py').read_text();component=(P/'component192.py').read_text();audit=(P/'audit192.py').read_text();guard=(G/'guard192.py').read_text();host=(P/'host192.py').read_text()
assert "n = min(remaining,64,cap)"in recipe and "pad_full_query_extent(metadata, 128)"in recipe
assert 'min(64, self.config.max_num_batched_tokens)'in recipe and 'min(64, self.engine.config.max_num_batched_tokens)'in recipe
assert 'return adapt_source(adapt_live_source' in decode and ',32)'in decode
assert "capacity=64,local_states=local"in component and 'capacity=64,local_states=local,context=context' in component
assert "valid_execution(execution)"in audit and "execution['used_aclgraph']"not in audit
assert "capture_original_protocol_before_startup(native)"in (P/'runtime192.py').read_text()
assert "from domain192 import envelope"in (P/'contracts192.py').read_text() and "from domain192 import envelope"in (P/'capability192.py').read_text()
assert "owner not in"not in '' # no executable producer called by this reviewer
coldpath=Path(r['cold_guard']['path']);cold=read(coldpath);assert sha(coldpath)==r['cold_guard']['sha256']=='06f9245f96046bf969a59eff7260cd417fb78227df961f6ef6c05c9a75c74631'
s=cold['source_state'];assert cold['status']=='passed'and s['local']==r['local_files']and s['external']==r['external_files']and s['evidence']==r['retained_HOST_evidence']and cold['guard_inputs']==r['guard_inputs']
# ROOT closes the prior-archive omission explicitly. No old suite is rerun here.
ap=D/'ROOT_V192_PARENT163_172_ARCHIVES_IN_PLACE_AUDIT.json';assert sha(ap)=='9d458931c99a9e20b28e6f64627e94c30b9a925c0102ef9c0b4efff654a93bcc'
a=read(ap);assert a['status']=='passed'and a['NPU_used']is a['Torch_imported']is a['source_changed']is False
assert a['READY_default_archiveFalse_claim_corrected']is True and a['old_suites_not_rerun']is True
assert [(x['epoch'],x['regular_members'])for x in a['parent_archives']]==[(163,94),(172,45)]
assert all(x['all_member_SHA256_checked']is True and x['archive_recopied']is False for x in a['parent_archives'])
assert sha(a['script']['path'])==a['script']['sha256'];compile(ast.parse(Path(a['script']['path']).read_text()),a['script']['path'],'exec')
# The real schedule is inherited fail-closed one warm/two zero, without max8.
phase_source=(P/'phases192.py').read_text();original_phases=B/'tree-high-normal-cap4-native-families-v148/candidate_phases_v132.py';ot=original_phases.read_text()
assert "(('warmup65',65),('zero65-1',65),('zero65-2',65),('seal',0),('full463',463))"in ot
assert 'prewarm'not in phase_source and 'max8'not in phase_source
pinpaths=[READY,P/'scope192.py',P/'recipe192.py',P/'contracts192.py',P/'layout192.py',P/'target192.py',P/'component192.py',P/'runtime192.py',P/'capability192.py',P/'inventory192.py',P/'decode192.py',P/'factory192.py',P/'probe192.py',P/'observation192.py',P/'execution_schema192.py',P/'host192.py',P/'abi_host192.py',P/'check192.py',G/'guard192.py',original_phases,ap,Path(a['script']['path']),Path(__file__)]
record=dict(schema='V192.independent-current-cursor-source-and-guard-peer.v1',status='passed_source_review',candidate=str(P),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),ready=dict(path=str(READY),sha256=sha(READY)),CPU=r['CPU'],cold_guard=r['cold_guard'],reviewed_inputs=flat,reviewed_counts=dict(flat=773,local=39,selected_external=558,guard=4,retained_HOST_evidence=171,full_source_projections=30,AST_compiled=asts,bash_n=2),source_pins={str(q):sha(q)for q in pinpaths},retained_actual_HOST_attempts=attempts,
 no_remaining_reviewed_source_blocker=True,archive_evidence_closure=dict(ROOT_in_place_audit=dict(path=str(ap),sha256=sha(ap)),ROOT_script=a['script'],required_ROOT_freeze_binding='ROOT review must explicitly bind this exact complete archive audit artifact and SHA; frozen guard binds the ROOT review SHA.',default_guard_parent_archiveFalse_disclosed=True,no_parent_source_changed=True),
 lifecycle_statement_correction=dict(READY_max8_claim_is_descriptive_error=True,actual_schedule=['warm65','zero65-1','zero65-2','seal','full463'],max8_prewarm_exists=False,two_complete_Native_zero_gates_retained=True,ROOT_explicitly_accepts_inherited_fail_closed172_schedule=True),
 key_findings=dict(active_window='Final actual generated decode caps initial_batch_size32; admission guard/vacancy subtract reserved_slots; reservation and postclock publication both forward32. Diagnostic N32+prompt actor intentionally uses constructor64 and is not production window evidence.',geometry='One checked domain yields P=min(remaining,64,197-Q-2d); FULL128 distinct from prompt cap64; causal cached2048 int8 identity and slotslong vs FULL slotsint32 retained.',transactions='Original reservation/submit/native finite-WORLD/PREPARED/TreeCommitFence/cursor/original old-active clock/final-once-root callback plus true fourstep next are retained; no partial root publication.',current_math='Four real-prefix transactions 129/65/54/2301 with 3/2/2/36 chunks each need own current fullphysical197/128targetKV/56draftKV/heads/FTF/protocol/next and two complete zeros before same-instance four Native entry issuer.',schema='Serialized NativeGraphExecution exact four fields checked; absent used_aclgraph property not queried.',HOST='Reviewed actual complete generated admit_available with original NativeSamplingParams/PearlPipelineState and all463 untouched records drained once in15 explicit HOST completion groups; no model output256 or timing equivalence inferred.',constructor='Original target1022/draft512 runner constructor patch restores on success and failure; original201/44/114/full527 currentB4 remain mandatory.',guard='ROOT self lock, original baseline pre/post and owned cleanup retained; frozen177 storage code via private managed-link walk, mutable tmp exceptions only registered current output/progress roots.'),
 limitations=['Numerical evidence is finite representatives plus per-call structural/lease checks, not all token/budget tuples or full AR equality.','Same-model/head/KV and performance gates remain future actual process obligations.','Active32 is active+reserved service occupancy; queued histories, independent diagnostic banks and64 scratch owners are not counted as service requests.','No benefit follows from M197<645; more chunks can increase normal work and synchronization.','No candidate import, old/new suite execution, Torch/backend/model/device invocation or current190 raw access occurred in this peer.'],tests_rerun=False,Torch_imported=False,NPU_used=False,device_queried=False,model_initialized=False,source_modified=False,freeze_created=False,adopted=False)
assert not OUT.exists();OUT.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n');print(json.dumps(dict(path=str(OUT),sha256=sha(OUT),bytes=OUT.stat().st_size,counts=record['reviewed_counts'])))
