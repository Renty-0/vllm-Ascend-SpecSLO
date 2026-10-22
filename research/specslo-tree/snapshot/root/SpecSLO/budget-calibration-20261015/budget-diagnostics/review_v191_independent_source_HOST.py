"""Independent static/inventory peer; no candidate import, suite or device."""
import ast,datetime,hashlib,json,subprocess
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-slo-admission1-vacancy4-cap4-active32-full-v191';G=D/'v191-root-execution'
R=D/'V191_SLO_ADMISSION1_VACANCY4_CAP4_ACTIVE32_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json';O=D/'V191_INDEPENDENT_SOURCE_PEER_REVIEW.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
assert sha(R)=='eaa72b4a64a37493c216baa85aab8f164b9e1609f85c489a519024beda02ea46'
r=read(R);c=read(P/'CPU_RESULT191.json');prep=read(P/'PREPARED_HOST191.json');projection=read(P/'PROJECTIONS191.json')
assert c['status']=='passed'and c['tests_run']==3 and c['errors']==c['failures']==0
assert c['Torch_imported']is c['NPU_used']is c['model_constructed']is c['model_math_qualified']is c['old190_4HOST_and185_suites_rerun']is False
assert c['local_before']==c['local_after']==r['files']==prep['files']and c['external_before']==c['external_after']==r['external_files']==prep['external_files']
assert (len(r['files']),len(r['external_files']),len(r['execution_guard_inputs']),len(r['retained_HOST_evidence']))==(20,277,4,82)
flat={}
for name in ('files','external_files','execution_guard_inputs','retained_HOST_evidence'):
 for p,h in r[name].items():
  assert p not in flat or flat[p]==h
  assert sha(p)==h;flat[p]=h
flat[str(P/'CPU_RESULT191.json')]=sha(P/'CPU_RESULT191.json')
assert sha(P/'CPU_RESULT191.json')==r['CPU']['sha256']==prep['CPU_sha256']=='f7cb0d38178ae372d2b01619cf7aff35bcde932ead2a9cbd9962eeaf81a111ba'
assert len(projection['files'])==15
for name,row in projection['files'].items():
 parent=Path(row['parent']);assert sha(parent)==row['parent_sha256']
 source=parent.read_text()
 for key in ('namespace','basename_namespace','epoch_namespace'):source=source.replace(*row[key])
 for delta in row['changes']:
  assert source.count(delta['old'])==delta['count'];source=source.replace(delta['old'],delta['new'])
 assert source==(P/name).read_text()and sha(P/name)==row['sha256']
assert {n for n,x in projection['files'].items()if x['changes']}=={'SCOPE191.txt','audit191.py','base191.py','runtime191.py'}
assert not any(projection['files'][n]['changes'] for n in ('next4_191.py','prefill191.py','component191.py','scope191.py','publication191.py','trace_admission191.py','config191.py','extension191.py','trace191.py'))
retained={str(p):sha(p)for p in (P/'full-host-attempts').rglob('*')if p.is_file()};assert retained==prep['retained_HOST_files']
attempt_paths={str(p):sha(p)for p in P.glob('HOST_ATTEMPT191_*.json')};assert attempt_paths==prep['actual_HOST_attempts']and len(attempt_paths)==3
assert r['retained_HOST_evidence']=={str(P/'PREPARED_HOST191.json'):sha(P/'PREPARED_HOST191.json'),**attempt_paths,**retained}
attempts=[]
for p,h in sorted(attempt_paths.items()):
 a=read(p);log=Path(a['attempt'])/'unittest.log';assert sha(log)==a['log_sha256']
 attempts.append(dict(path=p,sha256=h,status=a['status'],tests_run=a['tests_run'],errors=a['errors'],failures=a['failures'],log_sha256=sha(log)))
assert [a['status']for a in attempts]==['failed','failed','passed']
assert c['details']['actual_original_admission']['finite463_admitted_exact_once']is True
assert c['details']['actual_original_admission']['finite_groups']==15
assert c['details']['actual_original_admission']['completion_is_explicit_HOST_boundary']is True
assert c['details']['config_policy_source']['nonzero_deferred_polls_accepted']is True
anchors=c['details']['actual_original_admission']['priority_anchor_restart_events'];assert any(e['kind']=='reset'and e['reason']=='anchor_changed'for e in anchors)
asts=0
for p in [*[Path(x)for x in r['files']if x.endswith('.py')],G/'guard191.py']:
 compile(ast.parse(p.read_text()),str(p),'exec');asts+=1
for p in (G/'startup191.sh',G/'launch191.sh'):subprocess.run(['/bin/bash','-n',str(p)],check=True,capture_output=True)
admission=(P/'admission191.py').read_text();host=(P/'host191.py').read_text();startup=(G/'startup191.sh').read_text();runtime=(P/'runtime191.py').read_text();guard=(G/'guard191.py').read_text()
assert "'VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION':'1'"in admission and "'VLLM_ASCEND_SPECRHYTHM_TARGET_SLO_FIRST':'0'"in admission
assert startup.count('export VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION=1')==1 and startup.count('export VLLM_ASCEND_SPECRHYTHM_TARGET_SLO_FIRST=0')==1
assert startup.index('export VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION=1')<startup.index('"$task_root/check191.py"')
assert 'capture_original_protocol_before_startup(native)'in runtime and runtime.index('protocol=capture_original_protocol_before_startup(native)')<runtime.index("policy=entry['install']")
assert "counters[key]>=0"in admission and "counters[key]==0"not in admission
assert "P190=require_parent(archive=True)"in guard and "live_roots=(P/'actual',)"in guard and "LOCK=B/'.ROOT_NPU_EXECUTION.lock'"in guard
coldp=Path(r['cold_actual_guard_source_state']['path']);cold=read(coldp)
assert sha(coldp)==r['cold_actual_guard_source_state']['sha256']=='da95cbe9022eb6dbeba08fdbd84b5bb2893e645a13da2c17847d56e8c56c45eb'
assert cold['status']=='passed'and cold['Torch_imported']is cold['NPU_used']is False
s=cold['source_state'];assert s['local']==r['files']and s['external']==r['external_files']and s['evidence']==r['retained_HOST_evidence']and cold['guard_inputs']==r['execution_guard_inputs']
assert s['parent_archive']['P190']['archive_verified_in_place']is True
assert s['parent_archive']['original']['P148']['members']==6390 and s['parent_archive']['original']['P160']['members']==42
assert not any(p.endswith('.tar.gz')for p in r['external_files'])
# Source-only parent190/freeze/archive inputs never include its mutable actual data.
parent=Path(r['production_parent']['path']);assert sha(parent/'FROZEN_SOURCE190.json')==r['production_parent']['freeze_sha256']
assert not any('/tree-vacancy-cohort-prefill-cap4-active32-full-v190/actual/'in p or '/receipt-'in p for p in flat)
assert not (P/'FROZEN_SOURCE191.json').exists()and not (P/'PREPARED_SOURCE_ARCHIVE191.tar.gz').exists()
pins=[R,P/'admission191.py',P/'runtime191.py',P/'audit191.py',P/'host191.py',P/'check191.py',P/'PROJECTIONS191.json',P/'trace_admission191.py',P/'scope191.py',P/'prefill191.py',P/'next4_191.py',G/'guard191.py',G/'startup191.sh',coldp,Path(__file__)]
record=dict(schema='V191.independent-SLO-admission-vacancy-source-and-guard-peer.v1',status='passed_source_review',candidate=str(P),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),ready=dict(path=str(R),sha256=sha(R)),CPU=r['CPU'],cold_actual_guard=r['cold_actual_guard_source_state'],reviewed_inputs=flat,reviewed_counts=dict(flat=len(flat),local=20,selected_external=277,guard=4,retained_HOST_evidence=82,whole190_projections=15,AST_compiled=asts,bash_n=2),reviewed_source_pins={str(p):sha(p)for p in pins},actual_HOST_attempts=attempts,no_reviewed_source_blocker=True,
 findings=dict(production_delta='Only original SLO_AWARE_ADMISSION=1/TARGET_SLO_FIRST=0 with explicit clean environment and current Native resolution/report/audit; parent190 vacancy4/150/cap4+lowN trueB5/M80/active32/96+96=192 and all model math gates unchanged.',arrival_and_finite_workload='Original Native sort and original arrival filter retained. HOST executes patched actual nested admission over real463 parameter records with unchanged IDs, arrival/SLO/max256; exactlyonce drain is an explicit HOST completion boundary, not model256 output qualification.',priority_anchor='New earlier-ranked ready request changes anchor and starts a fresh vacancy episode. 150ms is per observed stable episode, not a whole-request TTFT/admission bound; finite trace completion still required.',deferred='Nonnegative deferred polls accepted, including nonzero; enabled4/150/SLO1/target0 counters checked in real raw leader results.',current_math='Same new-engine original201/44/114/full527 B4/currentM80 plus finite N2/4 multiowner eager ownmetadata/hidden/KV/Nheads/oncepub/truefourstepnext and twozero are all still required before warm/seal/full.',guard='Original selfROOTlock, baseline prepost/ownedcleanup/allidle, 7200s/256MiB/512MiB and frozen177 currentprivateactual-only tmp safety retained; exact parent190/P148/P160 archive closure verified in place.'),
 limitations=['No model or current190 actual/receipt was read; frozen190 source/archive identity is provenance only.','HOST finite completion proof frees service through an explicit carrier boundary; actual finite463x256 completion and original clocks/SLO must still pass.','Priorities can repeatedly restart a vacancy anchor under changing load; no infinite-flow starvation guarantee or strict150ms wall bound.','New multiowner numerical representatives are2/4, not exhaustive all32 prompt tuples; no full-AR equality or performance improvement is asserted.','SCOPE first inherited paragraph preserves historical parent SLO0 text; its appended V191 declaration and actual runtime/launch/audit unambiguously require1/0.'],tests_rerun=False,Torch_imported=False,NPU_used=False,device_queried=False,model_constructed=False,current190_actual_read=False,source_modified=False,freeze_created=False,adopted=False)
assert not O.exists();O.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n');print(json.dumps(dict(path=str(O),sha256=sha(O),bytes=O.stat().st_size,counts=record['reviewed_counts'])))
