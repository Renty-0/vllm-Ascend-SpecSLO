"""Future bounded206 terminal HOST audit, inert until exact ROOT authorization.

No actual is read during source preparation. No model, backend or device work is
permitted. A failed/partial job never acquires a successful qualification label.
"""
import argparse,ast,collections,copy,datetime,gzip,hashlib,importlib.abc,itertools
import json,math,re,statistics,struct,sys,tarfile,types
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-b4-joint-subtree-single-writer-diagnostic-v206'
ACTION='b4-joint-subtree-single-writer-bounded-gain'
FREEZE_SHA='4b9019fe70d88f6ba0dc03442bbdfb6b0ecb6748b405ddf08d5c65f5fa15e2dd'
ARCHIVE_SHA='48a97c9706ca92028bcdcc4933f464edadc801519b13e67c80f1a74781ffad1b'
ROOT_PREPARED=D/'ROOT_V206_PREPARED_SOURCE_HOST_AUDIT.json'
ROOT_PREPARED_SHA='55f11c6882875deec2592a1787048313263a12d55da26c38148d406bb6a0e208'
HELPERS=D/'audit_v173_terminal_full463_HOST.py'
HELPER_SHA='ed8aef51646ccf60c4812782d080d119fa406d7e53138ce24d6b470bce74c6f5'
PRIOR=D/'audit_v197_terminal_HOST.py'
PRIOR_SHA='5118f3e4f2da7ad3de169dfce9e48f310df8097c41c7e0b350df8852eedc81b7'
DERIVED_FROM=D/'audit_v198_terminal_HOST.py'
DERIVED_FROM_SHA='71c813a003564c60b41e16b6cf0c580e4cc907370c76960d8e658f5d8ec6fbb9'
NATIVE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
BACKENDS=('torch','torch_npu','vllm','vllm_ascend','transformers','safetensors','acl')
if hashlib.sha256(HELPERS.read_bytes()).hexdigest()!=HELPER_SHA:raise RuntimeError('Immutable stdlib helper source')
_names={'need','sha','desc','read','archive','Firewall','extracted','Value'}
_nodes=[copy.deepcopy(n)for n in ast.parse(HELPERS.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in _names]
if {n.name for n in _nodes}!=_names:raise RuntimeError('Complete exact stdlib helper AST')
exec(compile(ast.fix_missing_locations(ast.Module(body=_nodes,type_ignores=[])),str(HELPERS)+'::V206-stdlib-only','exec'),globals())
if sha(PRIOR)!=PRIOR_SHA:raise RuntimeError('Immutable successful197 original component integer helper')
_node=next(n for n in ast.parse(PRIOR.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='component_integer_verdicts')
exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(_node)],type_ignores=[])),str(PRIOR)+'::V206-parent-integer-only','exec'),globals())

def normalize(v):return json.loads(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False))
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def f32(v):return struct.unpack('<f',struct.pack('<f',v))[0]

def provenance(receipt):
 fp=P/'FROZEN_SOURCE206.json';need(sha(fp)==FREEZE_SHA and sha(ROOT_PREPARED)==ROOT_PREPARED_SHA,'exact206 freeze and ROOT prepared proof')
 need(sha(DERIVED_FROM)==DERIVED_FROM_SHA,'unchanged original198 independent auditor source')
 f=read(fp);pins={}
 for k in ('files','external_files','retained_HOST_evidence','execution_guard_inputs','peer_reviews'):
  for p,h in f[k].items():need(p not in pins or pins[p]==h,'no conflicting selected source hash');pins[p]=h
 pins.update({str(P/'CPU_RESULT206.json'):f['CPU_sha256'],str(fp):FREEZE_SHA,f['root_review']['path']:f['root_review']['sha256']})
 need(len(pins)==247 and all(sha(p)==h for p,h in pins.items()),'all247 selected current source/CPU/freeze/guard/peer/ROOT bytes')
 cpu=read(P/'CPU_RESULT206.json');prep=read(P/'PREPARED_HOST206.json')
 need(cpu['schema']=='V206.actual-bounded-selector-metadata-bank-HOST.v1' and cpu['status']=='passed'and cpu['tests_run']==5 and cpu['errors']==cpu['failures']==0,'actual frozen206 CPU five-check producer')
 need(all(cpu[k]is False for k in ('Torch_imported','NPU_used','model_constructed','model_KV_qualified','old_suites_rerun','freeze_created','formal_Goodput')),'HOST flags are not model qualification')
 need(cpu['local_before']==cpu['local_after']==f['files']==prep['files'] and cpu['external_before']==cpu['external_after']==f['external_files']==prep['external_files'] and f['CPU_sha256']==prep['CPU_sha256']==sha(P/'CPU_RESULT206.json'),'current CPU/prepared/freeze/source exact')
 ev={str(P/'PREPARED_HOST206.json'):sha(P/'PREPARED_HOST206.json'),**prep['actual_HOST_attempts'],**prep['retained_failed_attempt_inputs']}
 need(ev==f['retained_HOST_evidence']and len(ev)==27 and len(prep['actual_HOST_attempts'])==1 and len(prep['retained_failed_attempt_inputs'])==25 and set(prep['actual_HOST_attempts'])=={str(p)for p in P.glob('HOST_ATTEMPT206_*.json')},'all one actual attempt plus25 source/log/result snapshots and27 evidence inputs')
 need(set(prep['retained_failed_attempt_inputs'])=={str(p)for p in(P/'full-host-attempts').rglob('*')if p.is_file()}and prep['current_actual_tests_run']==5 and prep['parent_suites_rerun']is False,'all current five-check retained bytes; no old suite transfer')
 single=cpu['details']['single_writer_directory']
 need(all(single[k]is True for k in('actual_actor_lambda','actual_original_stage4','real_four_process_shared_FS','HOST_collective_and_sync_carriers'))and all(single[k]is False for k in('Torch_library_imported','model_constructed','NPU_used')),'actual HOST singlewriter/fs/stage-carrier evidence only')
 need([(v['case'],v['all4_passed'],v['local_FileExists_errors'],v['common_all4_result_identical'])for v in single['cases']]==[('old-race',False,3,True),('new-single-writer',True,0,True),('new-preexisting-refused',False,1,True)],'old race/new singlewriter/preexisting hard-refusal HOST cases')
 root=read(f['root_review']['path']);reviewed={p:h for p,h in pins.items()if p not in(str(fp),f['root_review']['path'])}
 need(root['status']=='passed_source_review'and root['reviewed_inputs']==reviewed and len(reviewed)==245,'exact ROOT245 reviewed source inputs')
 for k,v in dict(configured_capacity=64,queued=463,storage=527,bounded_diagnostic_only=True,whole463_run=False,new_production_issuer=False,diagnostic_cost_only=True,active_windows_requested=[],jointB4_diagnostic_domain=[16,80,'full9'],causal80_domain=[16,80,'causal'],whole_workload_AR_qualified=False,formal_Goodput=False,adopted=False).items():need(f[k]==v and receipt[k]==v,'unchanged bounded scope '+k)
 ready=receipt['readiness'];need(ready['freeze_sha256']==FREEZE_SHA and ready['CPU_sha256']==f['CPU_sha256'] and ready['root_review']==f['root_review']and ready['peers']==f['peer_reviews']and ready['device_qualification_inherited']is False,'actual frozen206 readiness')
 ar=ready['archive'];need(Path(ar['path'])==P/'PREPARED_SOURCE_ARCHIVE206.tar.gz'and ar['members']==247 and sha(ar['path'])==ar['sha256']==ARCHIVE_SHA,'actual own247 archive binary')
 own=archive(ar['path'],pins)
 from base206 import parent_archive
 deps=parent_archive();need(deps==f['dependency_archives_verified_in_place']==ready['dependency_archives'],'original173/P148/P160/P166 regular archives and source198320 donor in place')
 donor=deps['source_derivation198'];need(donor['path']==str(B/'tree-b4-joint-subtree-m80-gain-diagnostic-v198')and donor['members']==320 and donor['archive_verified_in_place']is True and donor['device_qualification_inherited']is False,'source198320 is immutable derivation, not current device qualification')
 carrier=read(P/'INPUTS206.json')['directory_HOST_carrier'];need(f['external_files'][carrier['path']]==carrier['sha256']==sha(carrier['path']),'exact immutable204 HOST-only process-worker source')
 for k in ('pre_verifier','post_verifier'):
  rr=receipt[k];value=read(rr['path']);need(sha(rr['path'])==rr['sha256']and rr['exit_code']==0 and value==rr['actual_result']and value['status']=='pass'and value['errors']==[]and(value['checked_frozen_files'],value['checked_external_files'],value['checked_packages'])==(10496,15,212),'saved original pre/post verifier '+k)
 for k in ('devices_before','devices_after'):
  v=receipt[k];need(v['all_idle']is v['four_910B2']is True and set(v['processes'])=={'2','5','6','7'},'saved all4 original idle '+k)
  need(all(row['exit_code']==0 and row['idle']is True and not row.get('timed_out',False)and re.search(r'No process in device\.|No running processes found',row['output'],re.I)for row in v['processes'].values()),'saved device witness only; no new query '+k)
 need(receipt['physical_cards']==[2,5,6,7]and receipt['sources_unchanged']is True and receipt['owned_remaining']==[]and receipt['execution_lock']['path']==str(B/'.ROOT_NPU_EXECUTION.lock')and receipt['execution_lock']['nonblocking_exclusive']is True,'owned cleanup, original self lock and unchanged sources')
 limits=dict(child_execution_seconds=7200,actual_tag_bytes=256*1024**2,aggregate_private_bytes=512*1024**2);need(receipt['limits']==limits and math.isfinite(receipt['elapsed_seconds'])and receipt['elapsed_seconds']>0,'original finite elapsed/limits')
 if receipt['status']=='complete':need(receipt['elapsed_seconds']<7200,'bounded completed child, not generic failure text')
 st=receipt['private_storage_final'];need(0<=st['actual_tag_bytes']<=limits['actual_tag_bytes']and 0<=st['aggregate_private_bytes']<=limits['aggregate_private_bytes'],'saved private final storage')
 return dict(selected_current_hashes=247,freeze=desc(fp),CPU=desc(P/'CPU_RESULT206.json'),ROOT_prepared=desc(ROOT_PREPARED),own_archive=own,parent_archives=deps,current_HOST_tests=5,HOST_process_worker_source=desc(carrier['path']),prepost='10496/15/212 PASS, saved logs reread; not rerun',saved_all4_idle=True,owned_remaining=[],cost_measured_template_flag_not_actual_cost_evidence=True,complete_environment_import_closure_claimed=False)


def raw8(work,case):
 from selector206 import choose,validate,POLICIES,PARENTS
 p=work['draft_proof'];raw=p['raw_pool'];scores=p['node_confidences'];means=p['mean_confidence'];encoded=[]
 need(len(raw)==len(scores)==len(means)==16 and p.get('raw_encoded_shape',[16,4,2])==[16,4,2],'four model steps and all eight real candidates/scores')
 for i,(tokens,ss)in enumerate(zip(raw,scores,strict=True)):
  need(len(tokens)==len(ss)==8 and all(type(t)is int and 0<=t<151936 for t in tokens)and all(type(x)in(int,float)and math.isfinite(x)and 0<=x<=1 for x in ss),'all real raw8 token/score ABI')
  quant=[int(round(x*1_000_000))for x in ss];need(all(ss[j]==f32(quant[j]/1_000_000)for j in range(8)),'each original quantized confidence exact float32 carrier')
  need(all(ss[j]>=ss[j+4]for j in range(4))and means[i]==f32(sum(q/1_000_000 for q in quant)/8),'original top2 order and mean8 denominator/rounding')
  encoded.append([[tokens[j]*1_000_001+quant[j],tokens[j+4]*1_000_001+quant[j+4]]for j in range(4)])
  selected=choose(ss)if case['policy']==POLICIES[1]else validate(case.get('selection',(0,1,2,3)))
  need(tuple(work['selected'][i])==selected and work['rows'][i]==[tokens[j]for j in selected]and work['confidences'][i]==sum(ss[j]for j in selected)/4,'real selected four and exact selected confidence')
  if case['policy']==POLICIES[1]:
   def score(nodes):
    total=0.
    for node in nodes:
     value=1.;n=node
     while n>=0:value*=ss[n];n=PARENTS[n]
     total+=value
    return total
   legal=[v for v in itertools.combinations(range(8),4)if all(PARENTS[j]<0 or PARENTS[j]in v for j in v)]
   need(len(legal)==8 and math.isclose(score(selected),max(map(score,legal)),rel_tol=0,abs_tol=1e-14),'independent exhaustive ancestor-closed joint objective')
 if 'encoded_sha256'in p:need(digest(encoded)==p['encoded_sha256'],'all real four-step encoded integer bytes reconstructed')
 return dict(owners=16,model_steps=4,raw_nodes=128,selected_C=64,encoded_sha256=digest(encoded))

def g_rows(row):
 from selector206 import PARENTS,is_fork
 w=row['work'];case=row['case'];commit=row['commit'];records=[]
 for i,(selection,path,emitted,owner)in enumerate(zip(w['selected'],commit['accepted_paths'],commit['committed_tokens'],case['owners'],strict=True)):
  nodes=[v for v in path if v>=0];original=[selection[j]for j in nodes]
  need(0<=len(nodes)<=4 and len(emitted)==len(nodes)+1 and 1<=len(emitted)<=5,'bounded original path+frontier, no output truncation/relabelled B')
  records.append(dict(owner=i,measured=bool(owner.get('measured_owner',False)),policy=case['policy'],category=case.get('category'),selected_original_nodes=selection,true_fork=is_fork(selection),accepted_packed_nodes=nodes,accepted_original_nodes=original,accepted_side_original_nodes=[j for j in original if j>=4],delivered_tokens=emitted,g=len(emitted),accepted_count=len(nodes),correction_or_bonus_count=1,original_request_id=owner.get('request_id'),source_context_sha256=digest(owner['token_ids']),source_context_tokens=len(owner['token_ids']),filler_of=owner.get('filler_of')))
 need(records==row['rows'],'all g/fork/side/filler/source rows independently reconstructed')
 return records

def bounded(directory):
 import audit206,bank206
 from scope_v146 import NATIVE_COUNTERS,FAILURES
 geometry=normalize(bank206.geometry_cases());gcases=normalize(bank206.gain_cases());need(len(geometry)==8 and len(gcases)==6,'exact pinned original input producers')
 reports=[read(directory/f'rank{r}-diagnostic.json.gz')for r in range(4)];records=[];costs=[];zero=0;raw_checks=collections.Counter();summaries=[]
 for rank,report in enumerate(reports):
  outcomes=[]
  for phase in range(3):
   for i,case in enumerate(geometry):
    path=directory/f'rank{rank}-geometry-p{phase}-case{i}.json.gz';row=read(path);need(row['case']==case,'actual six fork/two linear input/topology producer '+str(path))
    need(row['work']['selected']==[case['selection']]*16,'controlled four-node shape exactly retained')
    for w in (row['work'],row['next_work']):
     if rank==0:raw8(w,case);raw_checks['geometry_full_raw8_work_and_next']+=1
    audit206.check_row(row,rank);rr=g_rows(row);outcomes.append((row['commit']['committed_tokens'],row['commit']['accepted_paths']))
    native=report['geometry'][phase*8+i]['Native'];need(set(native['before'])==set(native['after'])==set(native['delta'])==set(NATIVE_COUNTERS)and all(type(native[k][n])is int for k in ('before','after','delta')for n in NATIVE_COUNTERS),'all21 exact integer Native fields')
    need(all(native['after'][n]-native['before'][n]==native['delta'][n]>=0 for n in NATIVE_COUNTERS)and not any(native['delta'][n]for n in FAILURES),'independent actual Native subtraction')
    if phase:need(not any(native['delta'].values()),'two entire eight-shape zeros');zero+=21
  need(outcomes[:8]==outcomes[8:16]==outcomes[16:],'all eight original geometry histories repeat on both zeros')
  per=[]
  for i,case in enumerate(gcases):
   q=read(directory/f'rank{rank}-gain-case{i}.json.gz');need(q['case']==case,'fresh policy/category actual input source')
   need([v['token_ids']for v in q['work']['source']['contexts']]==[v['token_ids']for v in case['owners']],'only initial source prefix, no observed future target token injected')
   audit206.check_row(q,rank);gg=g_rows(q);need(sum(v['measured']for v in gg)==12,'twelve measured; four fillers excluded')
   if rank==0:
    for w in (q['work'],q['next_work']):raw8(w,case);raw_checks['gain_full_raw8_work_and_next']+=1
   if rank==1:records.extend(gg)
   per.append((q['commit']['committed_tokens'],q['commit']['accepted_paths']))
   for repeat in range(3):
    c=read(directory/f'rank{rank}-cost-case{i}-repeat{repeat}.json.gz')
    for key in ('source','rows','selected','confidences'):
     need(c['work'][key]==q['work'][key]and c['next_work'][key]==q['next_work'][key],'complete fresh same-input timed raw fields '+key)
    for key in ('committed_tokens','accepted_paths','actual_logical_tokens'):need(c['commit'][key]==q['commit'][key],'timed actual original protocol matches qualified history '+key)
    need(c['heads']==q['heads'],'all80 actual timed head output matches qualified call')
    if rank==0:
     for w in(c['work'],c['next_work']):raw8(w,case);raw_checks['cost_full_raw8_work_and_next']+=1
    n=c['Native'];need(set(n['before'])==set(n['after'])==set(n['delta'])==set(NATIVE_COUNTERS)and all(type(n[k][v])is int for k in ('before','after','delta')for v in NATIVE_COUNTERS)and all(n['after'][v]-n['before'][v]==n['delta'][v]==0 for v in NATIVE_COUNTERS),'timed call actual21 current Native zero')
    elapsed=c['diagnostic_whole_protocol_and_next_HOST_seconds'];need(type(elapsed)in(int,float)and math.isfinite(elapsed)and elapsed>0,'finite actual role diagnostic elapsed')
    if rank==1:costs.append(dict(category=case['category'],policy=case['policy'],case_name=case['name'],repeat=repeat,seconds=elapsed,family=c['family']))
  summaries.append(dict(geometry=outcomes,gain=per))
 need(summaries[0]==summaries[1]==summaries[2]==summaries[3]and zero==1344,'all4 integer original histories and1344 geometry zero scalars')
 policies=('old-primary-spine-B4-v1','joint-confidence-closed-B4-v206');gs=[];cs=[]
 for cat in ('coding','chat','summarization'):
  hashes=[]
  for policy in policies:
   rows=[v for v in records if v['category']==cat and v['policy']==policy and v['measured']];filler=[v for v in records if v['category']==cat and v['policy']==policy and not v['measured']]
   need(len(rows)==12 and len(filler)==4 and len({v['original_request_id']for v in rows})==4,'twelve correlated prefixes of four actual request IDs; fillers excluded')
   hashes.append([v['source_context_sha256']for v in rows]);samples=[v['seconds']for v in costs if(v['category'],v['policy'])==(cat,policy)];need(len(samples)==3,'three whole-call target-leader diagnostic samples')
   gs.append(dict(category=cat,policy=policy,measured_samples=12,filler_excluded=4,source_hashes=hashes[-1],g=[v['g']for v in rows],g_mean=statistics.mean(v['g']for v in rows),accepted_mean=statistics.mean(v['accepted_count']for v in rows),fork_rows=sum(v['true_fork']for v in rows),side_accept_rows=sum(bool(v['accepted_side_original_nodes'])for v in rows)))
   cs.append(dict(category=cat,policy=policy,samples_seconds=samples,median_seconds=statistics.median(samples)))
  need(hashes[0]==hashes[1],'paired initial input prefixes, independent model histories')
 need(len([v for v in records if v['measured']and v['policy']==policies[0]])==len([v for v in records if v['measured']and v['policy']==policies[1]])==36,'36 old and36 joint fresh measured rows')
 joint=[v for v in records if v['measured']and v['policy']==policies[1]];forks=sum(v['true_fork']for v in joint);sides=sum(bool(v['accepted_side_original_nodes'])for v in joint)
 return dict(geometry_layouts=8,controlled_forks=6,linear_controls=2,complete_geometry_traces=3,raw_geometry_files=96,raw_gain_files=24,raw_cost_files=72,Native_geometry_zero_subtractions=zero,old_measured_rows=36,joint_measured_rows=36,filler_rows_excluded=24,g_summary=gs,natural_joint_fork_rows=forks,natural_joint_side_accept_rows=sides,expected_status='passed_bounded_diagnostic'if forks and sides else'passed_subset',all4_original_history_agreement=True,raw8_checks=dict(raw_checks),target_leader_diagnostic_samples=costs,target_leader_descriptive_medians=cs,cost_order='For each category: old qualification then old repeat0/1/2, joint qualification then joint repeat0/1/2. Source-defined order; not ABBA or randomized A/B.',cost_scope='Includes immediate clone of all80 greedy outputs, shape/key/flag checks, all4 stages/drains, actual repair/draft/transport/target/original verdict/compaction/commit/ownnext. Full hidden/KV observer and independent model reference are outside; not pure production critical-path/device time. Cross-role intervals are not additive.',GP_computed=False,hidden_KV_recomputed_by_HOST=False,new_g_posterior_installed=False)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--terminal-authorized',action='store_true',required=True);parser.add_argument('--tag',required=True);parser.add_argument('--expected-receipt-sha256',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 need(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',args.tag)is not None and re.fullmatch(r'[0-9a-f]{64}',args.expected_receipt_sha256)is not None,'explicit safe tag and exact ROOT receipt SHA')
 need(args.output.parent.resolve()==D and not args.output.exists(),'fresh private D output')
 receiptpath=P/('receipt-'+args.tag+'.json');need(sha(receiptpath)==args.expected_receipt_sha256,'ROOT explicitly authorized terminal receipt bytes')
 receipt=read(receiptpath);need(receipt['status']in('complete','failed')and receipt['action']==ACTION,'terminal bounded206 only; running refused BEFORE any actual access')
 actual=P/'actual'/args.tag;need(actual.resolve()==actual and not actual.is_symlink(),'private authorized output path even for failed pre-model job');actual_exists=actual.exists()
 if actual_exists:need(actual.is_dir()and actual.resolve()==actual and not actual.is_symlink(),'private authorized terminal actual')
 else:need(receipt['status']=='failed','complete job requires an actual directory; failed pre-model job may have none')
 sys.meta_path.insert(0,Firewall());sys.path.insert(0,str(P));errors=[]
 result=dict(schema='V206.independent-terminal-bounded-HOST-audit.v1',candidate=str(P),tag=args.tag,job_status=receipt['status'],receipt=desc(receiptpath),script=desc(__file__),errors=errors,Torch_imported=False,NPU_used=False,full463=False,formal_Goodput=False,GP_computed=False,whole_workload_AR_qualified=False,production_issuer=False,adopted=False)
 try:
  result['provenance']=provenance(receipt)
  if receipt['status']=='complete':need(read(actual/'worker-exit.json')==dict(closed=True,all_exited=True),'actual all4 worker pool closed')
  elif(actual/'worker-exit.json').is_file():result['failed_job_worker_exit']=read(actual/'worker-exit.json')
 except Exception as error:errors.append(dict(stage='terminal_source_and_cleanup_provenance',error=repr(error)))
 paths=sorted(actual.rglob('*'));need(not any(p.is_symlink()for p in paths),'no raw aliases');manifest={str(p):desc(p)for p in paths if p.is_file()};mp=args.output.with_name(args.output.stem+'_RAW_MANIFEST.json');need(not mp.exists(),'fresh raw manifest');mp.write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n');result['raw_manifest']=dict(**desc(mp),files=len(manifest))
 result['worker_errors']=[read(p)for p in sorted(actual.glob('rank*-last-error.json'))]
 if receipt['status']=='complete'and'provenance'in result:
  try:
   import audit206
   original=audit206.audit_all(actual);complete=read(actual/'completed.json')
   need(original['passed']is True and original==complete['audit']==receipt['actual_bounded_audit']and original['status']==receipt['actual_bounded_status'],'original bounded audit==completed==guard')
   need(complete['action']==ACTION and complete['replies']==[['calibrated',i,dict(bounded_diagnostic=True,whole463=False)]for i in range(4)],'all four bounded workers, never full463')
   need(receipt['errors']==[]and receipt['child']['exit_code']==0 and receipt['actual_execution_complete']is True and not result['worker_errors'],'actual successful terminal child')
   need(not(actual/'windows').exists()and all(v['whole463_run']is False for v in [read(actual/f'rank{r}-bounded-worker.json.gz')for r in range(4)]),'no full workload substituted for bounded diagnostic')
   result['original_unmodified_bounded_audit']=dict(source=desc(P/'audit206.py'),result=original)
   result['original_B4_M80_integer_component']=component_integer_verdicts(actual)
   result['independent_bounded']=bounded(actual/'joint-B4-current-diagnostic');need(original['status']==result['independent_bounded']['expected_status'],'natural fork/accepted-side absent remains subset')
  except Exception as error:errors.append(dict(stage='bounded_original_and_independent_integer_audit',error=repr(error)))
 elif receipt['status']=='failed':
  result['failure_only']=dict(qualification_success=False,actual_directory_exists=actual_exists,child=receipt.get('child'),outer_errors=receipt.get('errors'),completed_file_present=(actual/'completed.json').is_file(),partial_error_files=[str(p)for p in sorted(actual.rglob('*partial-error*'))],worker_error_ranks=[v.get('rank')for v in result['worker_errors']],no_GP0_assigned=True,within7200_text_is_not_timeout_evidence=True)
 need(not Firewall.attempts and not any(n.split('.')[0]in BACKENDS for n in sys.modules),'no actual backend imports')
 result['status']=('passed_terminal_bounded_diagnostic_audit'if receipt['status']=='complete'else'terminal_failed_job_evidence_only')if not errors else'terminal_evidence_with_audit_errors'
 result['limitations']=['No full463/GP/production issuer/adoption/wholeAR result exists in this bounded experiment.','HOST reopens original integer verifier, masks, metadata/slots, current selection, raw4step/score8 encoding/mean8 and g arithmetic; unretained hidden/logit/KV tensors and device finite/model matcher gates cannot be CPU-recomputed.','Saved prefixes are teacher-forced input fixtures with12 correlated offsets of4 request identities per category;36 old/36 joint samples are not72 independent workload requests.','Diagnostic cost includes greedy clone/checks and all4 synchronization/drains, uses fixed old3 then joint3 rather than ABBA, and excludes full hidden/KV observer/reference. It is not pure production cost, device critical-path time or an end-to-end GP prediction.','A missing natural fork or accepted-side observation must remain passed_subset; failed or partial jobs stay failed and receive no GP0 or successful qualification.','Saved verifier/idle/process-cleanup evidence is reread; no device or process query, old suite or model is executed.']
 result['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with args.output.open('x')as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
 print(json.dumps(dict(**desc(args.output),status=result['status'],job_status=result['job_status'],errors=errors)));return int(bool(errors))
if __name__=='__main__':raise SystemExit(main())
