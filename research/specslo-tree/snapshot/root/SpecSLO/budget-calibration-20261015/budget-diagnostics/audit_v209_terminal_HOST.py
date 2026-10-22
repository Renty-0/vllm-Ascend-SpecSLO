"""Future V209 terminal evidence audit; standard library only, never a launcher.

No actual/receipt access before explicit terminal authorization and exact SHA.
Freeze/archive/ROOT-prepared hashes are supplied after ROOT has frozen V209.
"""
import argparse,ast,collections,copy,datetime,gzip,hashlib,importlib.abc,json
import math,re,statistics,sys,tarfile,types
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
ACTION='causal80-N9-to15-allB4-empty-active32-full463'
NATIVE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
BACKENDS=('torch','torch_npu','vllm','vllm_ascend','transformers','safetensors','acl')
HELPERS=D/'audit_v173_terminal_full463_HOST.py'
PARENT_AUDITOR=D/'audit_v197_terminal_HOST.py'
CLOCKS=D/'analyze_completed_tree_full463_cost.py'
READY=D/'V209_ROLE_AWARE_RETAINED_AUDIT_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW.json'
CPU_SHA='8c1c085cb831ff26a28fda0ba92e908cd730bfdf067b09c25304a5708f5e60d0'
SOURCE_PINS={str(HELPERS):'ed8aef51646ccf60c4812782d080d119fa406d7e53138ce24d6b470bce74c6f5',
 str(PARENT_AUDITOR):'5118f3e4f2da7ad3de169dfce9e48f310df8097c41c7e0b350df8852eedc81b7',
 str(CLOCKS):'d3266647799fc0cf00c49fe98e195ecb85dc446e51ab66c643a5b1d5c3ca0b74',
 str(READY):'6b4a8e617cc128cae7ae98a72730ecefc19e3c2af03e2fe71cf1ffd5656a52d6',
 str(D/'audit_v205_terminal_HOST.py'):'2ab9b0e074fb8b35f739c5aaeeee0921ef92ee7655fe52ed881d8de67b1607c8'}
FINAL_FREEZE_SHA='428fcad26eabc586c9e7991213fd0ebfcf6dbf1a46c4bf58e7bc2cf379ef03ef'
FINAL_ARCHIVE_SHA='2664fcaf763b0de8b99fb3975a639c718a572cf72df4c30a45c53113e53a4f8b'
FINAL_ROOT_PREPARED=D/'ROOT_V209_PREPARED_SOURCE_HOST_AUDIT.json'
FINAL_ROOT_PREPARED_SHA='4f48a31e6dc4e17b13b7b8575569c0542e416c9c82d2658590b57ff0eb8b73c6'
FINAL_PEER_SHA='e415d39acd3c9cfe65c624771b9d1b126f0c95d06adaf01616fb8e45c6829ba2'
SOURCE_DONOR_202_FREEZE_SHA='1b4912eb5abc53ca6ca9ffbabb8993fd11d780827dd2bf31baff4e4aa9e3ee82'
for _p,_h in SOURCE_PINS.items():
 if hashlib.sha256(Path(_p).read_bytes()).hexdigest()!=_h:raise RuntimeError('Pinned auditor/source-ready changed: '+_p)

def definitions(path,names,changes=()):
 text=Path(path).read_text()
 for old,new in changes:
  if text.count(old)!=1:raise RuntimeError('Exact independent audit source delta: '+old)
  text=text.replace(old,new)
 nodes=[copy.deepcopy(n)for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names]
 if len(nodes)!=len(names)or {n.name for n in nodes}!=set(names):raise RuntimeError('Complete independent helper AST')
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path)+'::isolated209-HOST','exec'),globals())

definitions(HELPERS,{'need','sha','desc','read','archive','Firewall','extracted','Value','pct'})
definitions(PARENT_AUDITOR,{'component_integer_verdicts','role_counter_fields','independent_multiowner'})
definitions(PARENT_AUDITOR,{'common_full_statistics','admission_and_full_statistics'},(
 ("if (c['N'], c['real_query_count'], c['family']) == (16, 80, 'causal')", "if (c['N'], c['real_query_count'], c['family']) in {(n,5*n,'causal') for n in range(9,17)}"),
 ("(80, 16, [v + 1 for v in c['budgets']])", "(80, 16, [v + 1 for v in c['budgets']]+[5]*(16-c['N']))"),
 ('from empty197 import audit_trace','from empty209 import audit_trace')))

def provenance(receipt,args):
 freeze=P/'FROZEN_SOURCE209.json';cpu=P/'CPU_RESULT209.json';prepared=P/'PREPARED_HOST209.json'
 need(args.expected_freeze_sha256==FINAL_FREEZE_SHA and args.expected_archive_sha256==FINAL_ARCHIVE_SHA
  and args.root_prepared.resolve()==FINAL_ROOT_PREPARED and args.expected_root_prepared_sha256==FINAL_ROOT_PREPARED_SHA,'actual ROOT final209 source pins')
 need(sha(freeze)==args.expected_freeze_sha256 and sha(cpu)==CPU_SHA,'explicit final209 freeze and stopped CPU')
 need(sha(args.root_prepared)==args.expected_root_prepared_sha256,'explicit ROOT prepared record bytes')
 f=read(freeze);c=read(cpu);pr=read(prepared);pins={}
 for field in('files','external_files','retained_HOST_evidence','execution_guard_inputs','peer_reviews'):
  for path,h in f[field].items():
   need(path not in pins or pins[path]==h,'nonconflicting exact final source map');pins[path]=h
 pins.update({str(cpu):f['CPU_sha256'],str(freeze):args.expected_freeze_sha256,f['root_review']['path']:f['root_review']['sha256']})
 need(all(sha(p)==h for p,h in pins.items()),'every final selected source/evidence byte')
 need(c['schema']=='V209.full-source-HOST.v1'and c['status']=='passed'and c['tests_run']==3 and c['failures']==c['errors']==0,
  'actual current209 CPU3 producer schema; donor205 seven not counted')
 need(all(c[k]is False for k in('Torch_imported','NPU_used','child_Torch_CPU_control_only','model_constructed','model_math_qualified','old_suites_rerun','full463_executed','freeze_created')),'actual HOST-only source proof flags')
 need(c['local_before']==c['local_after']==f['files']==pr['files']and c['external_before']==c['external_after']==f['external_files']==pr['external_files']
  and pr['CPU_sha256']==f['CPU_sha256']==CPU_SHA,'current CPU/preparation/freeze exact pre/post source')
 need(set(pr['actual_HOST_attempts'])=={str(p)for p in P.glob('HOST_ATTEMPT209_*.json')}
  and len(pr['actual_HOST_attempts'])==3 and len(pr['retained_HOST_files'])==96,'three own209 HOST attempts/all96 source-log-result files')
 evidence={str(prepared):sha(prepared),**pr['actual_HOST_attempts'],**pr['retained_HOST_files']}
 need(evidence==f['retained_HOST_evidence'],'actual retained HOST map, not guessed archive count')
 need(len(evidence)==100 and len(f['files'])==26 and len(f['external_files'])==1395
  and len(f['execution_guard_inputs'])==4 and list(f['peer_reviews'].values())==[FINAL_PEER_SHA],'final actual209 member closure distinct from auxiliary cold evidence')
 root=read(f['root_review']['path']);reviewed={p:h for p,h in pins.items()if p not in(str(freeze),f['root_review']['path'])}
 need(len(reviewed)==len(pins)-2 and len(reviewed)-len(f['peer_reviews'])==1526,'actual current209 flat source closure; archive count derived only from future frozen map')
 source_ready=read(READY);auxiliary={}
 for entry in(source_ready['HOST']['actual_cold_guard'],source_ready['HOST']['cold_guard_source'],source_ready['exact_source_projection']['mechanical']):
  need(entry['path']not in auxiliary and sha(entry['path'])==entry['sha256'],'exact source-ready auxiliary evidence bound through ROOT ready SHA');auxiliary[entry['path']]=entry['sha256']
 need(root['ready']==dict(path=str(READY),sha256=sha(READY))and len(auxiliary)==3 and not(set(auxiliary)&set(pins)),
  'actual209 ROOT schema binds ready SHA; three source-ready cold/mechanical records remain outside archive')
 rp=read(args.root_prepared);need(rp['schema']=='ROOT.V209.prepared-source-HOST-audit.v1'and rp['status']=='passed_source_frozen_unrun'
  and rp['candidate']==str(P)and rp['actual_source_members']==rp['actual_archive_members']==len(pins)
  and rp['freeze']==dict(path=str(freeze),sha256=FINAL_FREEZE_SHA)and rp['archive']==dict(path=str(P/'PREPARED_SOURCE_ARCHIVE209.tar.gz'),sha256=FINAL_ARCHIVE_SHA)
  and rp['ready']==dict(path=str(READY),sha256=sha(READY))and rp['review']==f['root_review'],'actual ROOT prepared schema and209 source/archive count')
 from base209 import require_source_parent,require_derivation205
 donor205=require_derivation205(archive=True)
 need(donor205['members']==968 and donor205['archive_verified_in_place']is True and donor205['production_parent_remains197']is True and donor205['device_qualification_inherited']is False,'exact frozen205968 source donor only; failed old model run is not current qualification')
 donor=require_source_parent(archive=True)
 need(donor['members']==759 and donor['archive_verified_in_place']is True and donor['recopied']is donor['device_qualification_inherited']is False
  and donor['production_parent_remains197']is True,'original frozen202 REV3 source-only donor759 verified in place')
 donor_root=Path(donor['source_parent'])
 need(sha(donor_root/'FROZEN_SOURCE202.json')==SOURCE_DONOR_202_FREEZE_SHA,'exact source donor202 REV3 freeze')
 need(root['status']=='passed_source_review'and root['reviewed_inputs']==reviewed and f['ROOT_source_reviewed']is True,'ROOT exact final flat source review')
 for path in f['peer_reviews']:
  peer=read(path);need(peer['status'].replace('-','_')=='passed_source_review'and peer['candidate']==str(P),'current exact source peer')
 scope=dict(active_windows_requested=[32],configured_capacity=64,queued=463,storage=527,empty_active_only=True,current_multiowner_eager_math=True,
  new_target_keys=0,existing_target_keys=2,causal80_domain=[[n,5*n,'causal']for n in range(9,17)],original_parent173_cap4=True)
 need(all(f[k]==v and receipt[k]==v for k,v in scope.items()),'current seven-use scope and unchanged parent budgets')
 ready=receipt['readiness'];need(ready['freeze_sha256']==args.expected_freeze_sha256 and ready['CPU_sha256']==CPU_SHA
  and ready['root_review']==f['root_review']and ready['peers']==f['peer_reviews']and ready['device_qualification_inherited']is False,'current own source readiness, never parent result transfer')
 ar=ready['archive'];need(Path(ar['path'])==P/'PREPARED_SOURCE_ARCHIVE209.tar.gz'and sha(ar['path'])==ar['sha256']==args.expected_archive_sha256
  and ar['members']==len(pins),'exact actual archive count derived from final source closure')
 own=archive(ar['path'],pins)
 spec=__import__('importlib.util',fromlist=['spec_from_file_location']);gp=D/'v209-root-execution/guard209.py'
 need(str(gp)in f['execution_guard_inputs'],'actual execution guardian source pin')
 module_spec=spec.spec_from_file_location('_V209_terminal_actual_guard',gp);guard=spec.module_from_spec(module_spec);module_spec.loader.exec_module(guard)
 deps=guard.parent_archives();need(deps==f['dependency_archives_verified_in_place']==ready['dependency_archives'],'all old source archives verified in place')
 safe=D/'v177-root-safe-storage-overlay/safe_storage177.py';need(f['external_files'][str(safe)]==sha(safe),'frozen177 scoped mutable.tmp storage')
 for field in('pre_verifier','post_verifier'):
  row=receipt[field];value=read(row['path']);need(sha(row['path'])==row['sha256']and row['exit_code']==0 and value==row['actual_result']
   and value['status']=='pass'and value['errors']==[]and(value['checked_frozen_files'],value['checked_external_files'],value['checked_packages'])==(10496,15,212),'saved original pre/post10496/15/212 '+field)
 for field in('devices_before','devices_after'):
  row=receipt[field];need(row['all_idle']is row['four_910B2']is True and set(row['processes'])=={'2','5','6','7'},'saved same four idle cards '+field)
  need(all(v['exit_code']==0 and v['idle']is True and not v.get('timed_out',False)and re.search(r'No process in device\.|No running processes found',v['output'],re.I)for v in row['processes'].values()),'saved device command receipts only, not a new query')
 need(receipt['physical_cards']==[2,5,6,7]and receipt['sources_unchanged']is True and receipt['owned_remaining']==[]
  and receipt['execution_lock']['path']==str(B/'.ROOT_NPU_EXECUTION.lock')and receipt['execution_lock']['nonblocking_exclusive']is True,'original selfROOTlock/owned cleanup/source continuity')
 limits=dict(child_execution_seconds=7200,actual_tag_bytes=256*1024**2,aggregate_private_bytes=512*1024**2)
 need(receipt['limits']==limits and type(receipt['elapsed_seconds'])in(int,float)and math.isfinite(receipt['elapsed_seconds'])and receipt['elapsed_seconds']>0,'actual deadline and storage limits')
 if receipt['status']=='complete':need(receipt['elapsed_seconds']<7200,'complete bounded execution')
 storage=receipt['private_storage_final'];need(0<=storage['actual_tag_bytes']<=limits['actual_tag_bytes']and 0<=storage['aggregate_private_bytes']<=limits['aggregate_private_bytes'],'actual saved private storage')
 return dict(freeze=desc(freeze),CPU=desc(cpu),ROOT_prepared=desc(args.root_prepared),source_count=len(pins),archive=own,parent_archives=deps,
  saved_baseline_prepost='10496/15/212 PASS; never rerun',saved_idle=True,owned_remaining=[],source_ready=desc(READY),source_donor202=donor,auxiliary_cold_evidence_outside_archive=auxiliary,current_HOST_test3_scope='three new source/retained-role HOST checks; old205 seven verified donor-only and no newengine math',source_donor205=donor205,complete_environment_closure_claimed=False)

def independent_seven_usage(directory):
 folder=directory/'causal80-seven-real-N-current-process';reports=[read(folder/f'rank{r}-component.json.gz')for r in range(4)]
 from mapper209 import GREEDY,HIDDEN
 names,failures=role_counter_fields();counts=collections.Counter();cross_by_rank=[];raw_pins={};logical=[]
 ht=types.SimpleNamespace(Tensor=Value,tensor=lambda v,**kw:Value(v),long='long')
 verify=extracted(NATIVE/'native_engine.py','_verify_tree_outputs_host',dict(torch=ht,TreeVerificationOutput=lambda a,b:types.SimpleNamespace(token_ids=a,accepted_node_indices=b)))
 for rank,report in enumerate(reports):
  need(report['schema']=='V209.seven-real-N-current-usage.v1'and report['rank']==rank and report['role']==('draft'if rank==0 else'target')
   and report['completed']is report['cache_released']is report['math_independent_qualified']is True and report['errors']==[],'actual current role-specific seven-N report')
  need([(t['pass_index'],t['N'])for t in report['traces']]==[(p,n)for p in range(3)for n in range(9,16)],'three genuine traces for every real N9..15')
  need(all(report[k]is False for k in('formal_Goodput','adopted','Native_flags_assigned','whole_AR')),'no assigned flags/adoption/wholeAR')
  need(all(report['inventory'][k]is True for k in('zero_new_keys','objects_retained','flags_unchanged')),'retained actual existing entry object gate')
  role_cross=[];role_history=[]
  for trace in report['traces']:
   phase,n=trace['pass_index'],trace['N'];change=trace['Native']
   need(set(change['before'])==set(change['after'])==set(change['delta'])==set(names)and all(type(change[s][k])is int for s in('before','after','delta')for k in names)
    and all(change['after'][k]-change['before'][k]==change['delta'][k]>=0 for k in names)and not any(change['delta'][k]for k in failures),'independent entire21 monotone counter subtraction')
   if phase:need(not any(change['delta'].values()),'all current protocol-next zero sweep');counts['new_zero_scalars']+=21
   lease=trace['lease']['lease'];need(lease['pages']==128 and lease['candidate_owners']==list(range(16))and lease['reference_owners']==list(range(16,32))
    and lease['dummy_owners']==list(range(463,527)),'actual full16+16 and64scratch lease; not physical rows relabeled realN')
   last=None;cycle_cross=[]
   for cycle in range(4):
    path=folder/f'rank{rank}-pass{phase}-N{n}-cycle{cycle}.json.gz';row=read(path);raw_pins[str(path)]=sha(path);commit=row['commit']
    need(row['N']==n and row['cycle']==cycle and row['source']['ids']==list(range(n))and len(row['rows'])==n
     and row['selected_normal_C']==4*n and row['selected']==[list(range(4))]*n and row['full16_draft']is True,'actual original trueN B4 spine proposals')
    need(commit['actual_structural_failed']==commit['actual_numerical_failed']==0 and type(commit['actual_structural_failed'])is type(commit['actual_numerical_failed'])is int
     and commit['original_finish_cycle_executed']is True,'original typed preflight and actual original finish')
    need(len(commit['committed_tokens'])==len(commit['accepted_paths'])==n and len(row['source']['contexts'])==n,'retained real owner count')
    for source,out,tokens in zip(row['source']['contexts'],commit['actual_logical_tokens'],commit['committed_tokens'],strict=True):
     need(source['token_ids']+tokens==out,'logical history advances from actual original verdict only')
    need(row['independent_commit']['histories_equal']is row['independent_commit']['unselected_owners_preserved']is True
     and row['independent_commit']['candidate_KV_copied']is False,'fresh own histories and protected unselected owners')
    if last is not None:need([v['token_ids']for v in row['source']['contexts']]==last,'actual continuous committed histories, no candidate KV copied as reference')
    last=commit['actual_logical_tokens']
    if rank==0:
     need(row['target']is row['reference']is row['old_control']is row['crossM']is None,'draft has no invented target evidence')
     dp=row['draft'];need(type(dp)is dict and dp['variable_num_draft_tokens']==[4]*16 and dp['raw_encoded_shape']==[16,4,2]and dp['all_physical_hidden_original_matcher']is dp['all28_query_KV_original_matcher']is dp['exact_full8_encoded']is True
      and dp['source_KV_copied']is False,'retained independent allphysical draft/raw8/all56KV gates only')
    else:
     need(row['draft']is None and all(type(row[k])is dict for k in('target','reference','old_control','crossM')),'target retains every own proof and no fabricated draft proof')
     target,reference,old,x=(row[k]for k in('target','reference','old_control','crossM'))
     need(target['production_greedy_M']==target['separate_hidden_counterpart_M']==reference['same_M']==80 and reference['same_S']==16
      and reference['independent_owners']==list(range(16,16+n))and reference['candidate_KV_copied']is False,'current sameM80/S16 disjoint reference owners')
     need(all(target[k]is True for k in('all_physical_hidden','all64_query_KV'))and all(reference[k]is True for k in('all_physical_hidden_original_matcher','all64_query_KV_original_matcher','all_physical_greedy_exact')),
      'all80 hidden/head and128queryKV actual matcher gates retained, not recomputed tensors')
     need(old['sameM_head_KV']is True and old['new_hidden_key']is False,'actual independent old128 sameM control')
     for arm,key in(('greedy',GREEDY),('hidden',HIDDEN)):
      phases=target[arm];need([v['phase']for v in phases]==['first','changed','restored'],'actual F/T/F order')
      for item in phases:
       execution=item['execution'];need(item['key']==repr(key)and item['runtime_validated_before']is item['runtime_validated_after']is True
        and set(execution)=={'mode','fallback_reason','capture_attempted','replay_executed'}and execution['mode']=='replay'
        and execution['fallback_reason']is None and execution['capture_attempted']is False and execution['replay_executed']is True,'actual four-field execution on already-natural original object')
      if arm=='greedy':need(phases[0]['full_physical_tokens']==phases[2]['full_physical_tokens']==x['new_all80']and len(phases[1]['full_physical_tokens'])==80,'restored complete80 head, changed head retained')
     need(len(x['new_all80'])==80 and len(x['old_all128'])==128 and x['new_all80'][:5*n]==x['new_real']==reference['real_query_tokens']
      and x['old_all128'][:5*n]==x['old_real'],'all realQ and complete physical80/128 heads retained')
     positions=[i for i,(a,b)in enumerate(zip(x['old_real'],x['new_real'],strict=True))if a!=b]
     need(positions==x['different_head_positions']and x['head_equal']==(not positions),'old128/new80 real head differences exactly reported')
     plans=[types.SimpleNamespace(candidate_budget=4,parent_indices=Value([-1,0,1,2]))for _ in range(n)]
     for arm in('old','new'):
      outcome=verify(row['rows'],plans,Value(x[arm+'_real']),4,'explicit-HOST-values');tokens=[[v for v in r if v>=0]for r in outcome.token_ids.values]
      need(tokens==[[v for v in r if v>=0]for r in x[arm+'_verdict_tokens']]and outcome.accepted_node_indices.values==x[arm+'_accepted'],'actual original Native integer verifier '+arm)
      if arm=='new':need(tokens==commit['committed_tokens']and outcome.accepted_node_indices.values==commit['accepted_paths'],'actual committed output uses new current head')
      counts['new_integer_verifier_calls']+=1;counts['new_owner_verdicts']+=n
     need(x['verdict_equal']==(x['old_verdict_tokens']==x['new_verdict_tokens'])and x['accepted_equal']==(x['old_accepted']==x['new_accepted'])and x['cause_established']is False,'honest crossM classification; no BF16 causal assumption')
     cycle_cross.append(dict(phase=phase,N=n,cycle=cycle,different_head_positions=positions,verdict_equal=x['verdict_equal'],accepted_equal=x['accepted_equal']))
    counts['role_current_commits']+=1
   nxtpath=folder/f'rank{rank}-pass{phase}-N{n}-next.json.gz';nxt=read(nxtpath);raw_pins[str(nxtpath)]=sha(nxtpath)
   need(nxt['source']['ids']==list(range(16))and [v['token_ids']for v in nxt['source']['contexts'][:n]]==last,'actual independent full16 ownnext consumes committed selected histories')
   need({k:nxt[k]for k in('source','rows','selected','confidences')}==trace['logical']['next'],'raw ownnext agrees complete trace history')
   if rank==0:
    dp=nxt['draft_proof'];need(type(dp)is dict and dp['variable_num_draft_tokens']==[4]*16 and dp['raw_encoded_shape']==[16,4,2]and all(dp[k]is True for k in('all_physical_hidden_original_matcher','all28_query_KV_original_matcher','exact_full8_encoded'))and dp['source_KV_copied']is False,'ownnext original all56KV/full8 proof retained')
   else:need(nxt['draft_proof']is None,'target ownnext retains no fabricated draft proof')
   role_cross.extend(cycle_cross);role_history.append((phase,n,trace['logical']))
  need(report['MC2']['delta']['fallback']==report['MC2']['delta']['exception']==0 and(rank==0 or report['MC2']['delta']['native_epilogue_success']>0),'actual existing MC2 dispatch success, no fallback')
  cross_by_rank.append(role_cross);logical.append(role_history)
 need(counts['new_zero_scalars']==1176 and counts['role_current_commits']==336 and counts['new_integer_verifier_calls']==504 and counts['new_owner_verdicts']==6048,'all seven real scopes/1176zero/504 original integer verifiers')
 need(logical[0]==logical[1]==logical[2]==logical[3]and cross_by_rank[1]==cross_by_rank[2]==cross_by_rank[3],'all4 logical transcripts and three target verdict/head classifications')
 same=all(not v['different_head_positions']and v['verdict_equal']and v['accepted_equal']for v in cross_by_rank[1])
 return dict(counts=dict(counts),crossM_behavior_equal=same,numerical_variant_only=not same,crossM_records=cross_by_rank[1],raw_checkpoint_sha256=raw_pins,
  new_usage_real_N=list(range(9,16)),existing80_entries_retained=True,hidden_or_KV_tensors_recomputed_by_HOST=False)

def full_statistics(traces,reports):
 value=admission_and_full_statistics(traces,reports);leader=traces[1];calls=leader['normal_target_calls'];hits=[]
 from mapper209 import expected_geometry
 for call in calls:
  n,q,f=call['N'],call['real_query_count'],call['family'];m,s,pad=expected_geometry(n,q,f)
  need((call['physical_M'],call['physical_S'],call['physical_query_partition'])==(m,s,[v+1 for v in call['budgets']]+list(pad)),'every actual new-or-original production mapper geometry')
  if 9<=n<=15 and(q,f)==(5*n,'causal'):
   need(call['budgets']==[4]*n and(m,s,pad)==(80,16,[5]*(16-n)),'true allB4 new current usage with original dummy pad');hits.append(call)
 value.update(new_usage_calls=len(hits),new_usage_observed=bool(hits),zero_hits_do_not_establish_optimization=True,new_usage_N_histogram=dict(sorted(collections.Counter(c['N']for c in hits).items())),
  new_usage_real_queries=sum(c['real_query_count']for c in hits),new_usage_physical_queries=80*len(hits),
  complete_all463_clock_scope='Native charged decode/256 and worker publication first-last/255; original outer generate wall, not external client clocks')
 return value

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--terminal-authorized',action='store_true',required=True);parser.add_argument('--tag',required=True)
 parser.add_argument('--expected-receipt-sha256',required=True);parser.add_argument('--expected-freeze-sha256',required=True);parser.add_argument('--expected-archive-sha256',required=True)
 parser.add_argument('--root-prepared',type=Path,required=True);parser.add_argument('--expected-root-prepared-sha256',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 need(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',args.tag)is not None,'exact safe tag')
 need(all(re.fullmatch(r'[0-9a-f]{64}',getattr(args,k))is not None for k in('expected_receipt_sha256','expected_freeze_sha256','expected_archive_sha256','expected_root_prepared_sha256')),'explicit exact SHA256 values, no readiness guesses')
 need(args.output.parent.resolve()==D and not args.output.exists(),'fresh D output')
 need(all(type(v)is str and re.fullmatch(r'[0-9a-f]{64}',v)is not None for v in(FINAL_FREEZE_SHA,FINAL_ARCHIVE_SHA,FINAL_ROOT_PREPARED_SHA)),'Future209 terminal CLI remains disabled until exact ROOT final freeze/archive/prepared pins are bound in a reviewed revision')
 receipt_path=P/('receipt-'+args.tag+'.json')
 # FIRST actual-related read; ROOT must have explicitly authorized this terminal SHA.
 need(sha(receipt_path)==args.expected_receipt_sha256,'ROOT exact terminal receipt bytes');receipt=read(receipt_path)
 need(receipt['status']in('complete','failed')and receipt['action']==ACTION and receipt['tag']==args.tag,'terminal matching job only; running rejected before actual access')
 actual=P/'actual'/args.tag;need(actual.is_dir()and actual.resolve()==actual and not actual.is_symlink(),'own terminal actual directory')
 sys.meta_path.insert(0,Firewall());sys.path.insert(0,str(P));errors=[]
 result=dict(schema='V209.independent-terminal-HOST-audit.v1',candidate=str(P),tag=args.tag,job_status=receipt['status'],receipt=desc(receipt_path),script=desc(__file__),errors=errors,
  source_dependencies=SOURCE_PINS,Torch_imported=False,NPU_used=False,whole_workload_AR_qualified=False,formal_Goodput=False,adopted=False)
 try:
  result['provenance']=provenance(receipt,args);need(read(actual/'worker-exit.json')==dict(closed=True,all_exited=True),'all four actual workers closed')
 except Exception as e:errors.append(dict(stage='source_or_terminal_provenance',error=repr(e)))
 paths=sorted(actual.rglob('*'));need(not any(p.is_symlink()for p in paths),'no actual raw aliases')
 manifest={str(p):desc(p)for p in paths if p.is_file()};mp=args.output.with_name(args.output.stem+'_RAW_MANIFEST.json');need(not mp.exists(),'fresh manifest');mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 result['raw_manifest']=dict(**desc(mp),files=len(manifest));result['worker_errors']=[read(p)for p in sorted(actual.glob('rank*-last-error.json'))]
 if receipt['status']=='complete'and 'provenance'in result:
  try:
   import audit209
   original=audit209.audit(actual,(32,));complete=read(actual/'completed.json')
   need(original['passed']is True and original==complete['audit']==receipt['actual_full_audit']and complete['action']==ACTION
    and complete['replies']==[['calibrated',r,dict(windows=[32],full463=True)]for r in range(4)],'unmodified209 audit == completed == actual guardian')
   need(receipt['errors']==[]and receipt['child']['exit_code']==0 and receipt['actual_execution_complete']is True and not result['worker_errors']
    and receipt['actual_full_status']==('passed_numerical_variant_full'if original['numerical_variant_only']else'passed_full'),'actual original job complete remains distinct from compatibility')
   need(original['native_zero_subtractions']==252 and original['current_multiowner_eager_component']['Native_zero_scalar_subtractions']==336
    and original['seven_usage']['Native_zero_subtractions']==1176,'base/full252 plus current extra336 and new1176 zero scalar gates')
   result['original_unmodified_audit']=dict(source=desc(P/'audit209.py'),result=original)
   result['original_components_integer_audit']=component_integer_verdicts(actual)
   result['independent_seven_usage']=independent_seven_usage(actual)
   need(result['independent_seven_usage']['crossM_behavior_equal']==original['seven_usage']['crossM_behavior_equal'],'independent compatibility arithmetic agrees current producer')
   result['actual_compatibility_classification']=dict(old_padding_behavior_equal=original['old_padding_behavior_equal'],numerical_variant_only=original['numerical_variant_only'],whole_AR_qualified=False,adopted=False)
  except Exception as e:errors.append(dict(stage='original_and_independent_current_components',error=repr(e)))
  try:
   traces=[read(actual/'windows/window32/full463'/f'rank{r}-full463-actual-trace.json.gz')for r in range(4)];reports=[read(actual/f'rank{r}-local-diagnostic.json')for r in range(4)]
   result['independent_multiowner']=independent_multiowner(actual,traces,reports);result['retained_full463']=full_statistics(traces,reports)
  except Exception as e:errors.append(dict(stage='independent_multiowner_or_full_clocks',error=repr(e)))
 elif receipt['status']=='failed':
  result['failure_only']=dict(child=receipt['child'],outer_errors=receipt['errors'],completed_file_present=(actual/'completed.json').is_file(),full_qualification_issued=False,GP_assigned=False,timeout_inferred_from_generic_text=False)
 need(not Firewall.attempts and not any(n.split('.')[0]in BACKENDS for n in sys.modules),'stdlib HOST only, no forbidden import attempts')
 result['status']=('passed_terminal_component_and_full463_audit'if receipt['status']=='complete'else'terminal_failed_job_evidence_only')if not errors else'terminal_evidence_with_audit_errors'
 result['limitations']=['HOST independently recomputes source/archive, actual Native integer verdicts, retained history/counter/heads/clock/bonus arithmetic. Full hidden/KV/finite tensors are unretained and are never claimed CPU-recomputed.',
  'New seven real-N uses keep actual existing80G/H objects and original dummy pad; no new key alone grants no mathematical usage permission. Original current B4/M80/N2&32 and all new current gates must independently pass.',
  'Three complete fresh histories per real N, four commits and ownnext each, two full zero repeats;1176 counts raw scalar differences across four ranks, not device speed or timing evidence.',
  'All463/256 and original arrival/Native charged/publication255 clocks remain separate; role HOST phase seconds cannot be added as device critical path. No external-client or whole463 independent AR equivalence claim.',
  'Only actual own3 HOST is current source evidence; failed old205 actual/raw fixture is never current209 numerical qualification. Failure stays failure, no full GP or adoption is assigned. Final archive member count is computed from actual freeze+peer+ROOT source closure, never predicted from unfrozen readiness.']
 result['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with args.output.open('x')as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
 print(json.dumps(dict(**desc(args.output),status=result['status'],errors=errors)));return int(bool(errors))
if __name__=='__main__':raise SystemExit(main())
