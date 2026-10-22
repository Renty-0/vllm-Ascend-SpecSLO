"""SOURCE ONLY: future independent201 terminal HOST evidence audit.

No launch/device/backend entry. ROOT must explicitly authorize an exact terminal
receipt. Saved tensor matcher/finite results are evidence, not HOST tensor math.
"""
import argparse, ast, collections, copy, datetime, gzip, hashlib, importlib.abc
import json, math, re, statistics, sys, tarfile, types
from pathlib import Path

B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-normal-b4-final-kv5-cohort16-full-v201'
ACTION='normal-b4-final-kv5-cohort16-full463'
FREEZE_SHA='035cf69b61bcf6871baa472d5f9eb12001454e2ba076de07baf025d703ee460b'
CPU_SHA='38f406d674370d1de61658a08c2508e01b943a78871d87834214b1b654b92c15'
ARCHIVE_SHA='edbb8cb19e3a851331fdbbba10ed54e37f99250e838584429ce5d5eeda7e36c8'
ROOT_PREPARED=D/'ROOT_V201_PREPARED_SOURCE_HOST_AUDIT.json'
ROOT_PREPARED_SHA='12d0cb3db68009f60230349dce10980b8b505c141b40a4fb18ded8d1423d37e6'
READY=D/'V201_FINAL_KV5_SOURCE_AND_GUARD_READY_FOR_ROOT_REVIEW_REVISION2.json'
READY_SHA='2325adc368d892d9da39e22ee62a6f166b62e1c0155b6b9475509979614b1b24'
HELPERS=D/'audit_v173_terminal_full463_HOST.py'
HELPER_SHA='ed8aef51646ccf60c4812782d080d119fa406d7e53138ce24d6b470bce74c6f5'
REUSE=D/'audit_v197_terminal_HOST.py'
REUSE_SHA='5118f3e4f2da7ad3de169dfce9e48f310df8097c41c7e0b350df8852eedc81b7'
CLOCKS=D/'analyze_completed_tree_full463_cost.py'
CLOCK_SHA='d3266647799fc0cf00c49fe98e195ecb85dc446e51ab66c643a5b1d5c3ca0b74'
STAT_SOURCE=D/'audit_v189_terminal_HOST.py'
STAT_SHA='3935ae12f3b7983bed7fc0349ced8befd2c59f31fe103c2e263de174f1631920'
NATIVE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
BACKENDS=('torch','torch_npu','vllm','vllm_ascend','transformers','safetensors','acl')
NEW_KEY=('draft-greedy:151936|steps:5|paged|final-kv|confidence',16)

# Read only pinned helper SOURCE at import. No old auditor main/provenance runs.
for path,want in ((HELPERS,HELPER_SHA),(REUSE,REUSE_SHA)):
 if hashlib.sha256(path.read_bytes()).hexdigest()!=want:raise RuntimeError('changed exact HOST helper source '+str(path))
for path,names in ((HELPERS,{'need','sha','desc','read','archive','Firewall','extracted','Value','pct'}),
 (REUSE,{'component_integer_verdicts','role_counter_fields','independent_multiowner','common_full_statistics','admission_and_full_statistics'})):
 nodes=[copy.deepcopy(n)for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names]
 if {n.name for n in nodes}!=names:raise RuntimeError('incomplete selected pure HOST AST '+str(path))
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path)+'::isolated201-pure-HOST','exec'),globals())

def strict_zero(value):return type(value)in(int,bool)and value==0

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def merge_pins(*maps):
 result={}
 for values in maps:
  for p,h in values.items():
   need(p not in result or result[p]==h,'nonconflicting source pin '+p);result[p]=h
 return result

def provenance(receipt):
 """Independent byte/archive closure; original pure source checks supplement it."""
 freeze_path=P/'FROZEN_SOURCE201.json';cpu_path=P/'CPU_RESULT201.json'
 for path,h in ((freeze_path,FREEZE_SHA),(cpu_path,CPU_SHA),(ROOT_PREPARED,ROOT_PREPARED_SHA),
                (READY,READY_SHA),(CLOCKS,CLOCK_SHA),(STAT_SOURCE,STAT_SHA)):
  need(sha(path)==h,'exact201 source/preparation/helper SHA '+str(path))
 f=read(freeze_path);c=read(cpu_path);prepared=read(P/'PREPARED_HOST201.json')
 need(f['schema']=='V201.ROOT-source-freeze.v1'and f['ROOT_source_reviewed']is True,'explicit own ROOT freeze')
 maps=[f[k]for k in('files','external_files','retained_HOST_evidence','execution_guard_inputs','peer_reviews')]
 pins=merge_pins(*maps,{str(cpu_path):f['CPU_sha256'],str(freeze_path):FREEZE_SHA,f['root_review']['path']:f['root_review']['sha256']})
 need(len(pins)==531 and all(sha(p)==h for p,h in pins.items()),'all531 selected current bytes')
 need(c['schema']=='V201.full-source-HOST.v1'and c['status']=='passed'and c['tests_run']==5 and c['failures']==c['errors']==0,
      'actual current REV2 five HOST checks, not four plus one historical sum')
 need(all(c[k]is False for k in('Torch_imported','NPU_used','model_constructed','model_math_qualified','old_suites_rerun','full463_executed','freeze_created')),
      'current CPU source-only actual backend fields')
 need(c['local_before']==c['local_after']==f['files']==prepared['files']and
      c['external_before']==c['external_after']==f['external_files']==prepared['external_files']and
      f['CPU_sha256']==prepared['CPU_sha256']==CPU_SHA,'current source pre/post/CPU/prepared/freeze identity')
 permit=c['details']['permit_identity']
 need(all(permit[k]is True for k in('actual_P148_checker','actual_activate_and_bounded_scope','actual_runtime_spec_assignment','wrong197identity_rejected'))and
      permit['model_backend_imported']is False,'actual REV2 permit positive/foreign197 negative HOST evidence')
 attempts=prepared['actual_HOST_attempts'];retained=prepared['retained_HOST_files']
 need(set(attempts)=={str(p)for p in P.glob('HOST_ATTEMPT201_*.json')}and len(attempts)==9 and len(retained)==270,
      'all nine attempts and270 retained HOST files')
 need(prepared['current_revision_actual_tests_run']==5 and prepared['prior_revision_qualification_transferred']is False,
      'current five-check revision did not borrow earlier qualification')
 evidence=merge_pins({str(P/'PREPARED_HOST201.json'):sha(P/'PREPARED_HOST201.json')},attempts,retained)
 need(evidence==f['retained_HOST_evidence']and len(evidence)==280,'exact retained HOST archive map')
 previous=read(P/'INPUTS201.json')['previous_revision'];old=read(previous['manifest']['path'])
 need(previous['manifest']['sha256']=='befb26b67a452a10fea94e49df5fe0924e38cfedfae33ad5b8c28990b3334229'and
      sha(previous['manifest']['path'])==previous['manifest']['sha256']and len(old['files'])==218,'all218 pre-permit-revision preservation manifest')
 for original,row in old['files'].items():need(sha(row['snapshot'])==row['sha256'],'preserved previous byte '+original)
 prior=read(previous['CPU']['path'])
 need(sha(previous['CPU']['path'])==previous['CPU']['sha256']=='b03c3cbb61c0a5e954d49016a8dac7a815db663e6a935bb89a55545495ce9046'and
      prior['status']=='passed'and prior['tests_run']==4 and prior['failures']==prior['errors']==0 and
      prior['local_before']==prior['local_after']and prior['external_before']==prior['external_after'],
      'retained earlier four-check source evidence distinct from current five')
 root=read(f['root_review']['path'])
 need(root['status']=='passed_source_review'and root['reviewed_inputs']=={p:h for p,h in pins.items()if p not in(str(freeze_path),f['root_review']['path'])},
      'ROOT exact529 selected input/CPU/peer review map')
 for path in f['peer_reviews']:
  peer=read(path);need(peer['status'].replace('-','_')=='passed_source_review'and peer['candidate']==str(P),'explicit source peer current candidate')
 expected=dict(active_windows_requested=[32],configured_capacity=64,queued=463,storage=527,
  empty_active_only=True,current_multiowner_eager_math=True,new_target_keys=2,causal80_domain=[16,80,'causal'],
  original_parent173_cap4=True,new_final_KV_draft_keys=1,original_BONUS_g_target_protocol_unchanged=True,
  whole_workload_AR_qualified=False,formal_Goodput=False,adopted=False)
 need(all(f[k]==v and receipt[k]==v for k,v in expected.items()),'actual fixed production scope')
 readiness=receipt['readiness']
 need(readiness['freeze_sha256']==FREEZE_SHA and readiness['CPU_sha256']==CPU_SHA and readiness['root_review']==f['root_review']and
      readiness['peers']==f['peer_reviews']and readiness['device_qualification_inherited']is False,'current201 receipt readiness')
 ar=readiness['archive'];need(Path(ar['path'])==P/'PREPARED_SOURCE_ARCHIVE201.tar.gz'and ar['members']==531 and
      sha(ar['path'])==ar['sha256']==ARCHIVE_SHA,'exact531 own regular source archive')
 own_archive=archive(ar['path'],pins)
 import base201,check201
 deps=check201.require_parent(archive=True);parent=base201.parent_ready();old_proof=check201.previous_HOST()
 need(deps==f['dependency_archives_verified_in_place']==readiness['dependency_archives']and
      parent==readiness['parent_ready'],'actual P197 lineage/in-place archives separate from original P148 permit')
 need(old_proof['tests_run']==4 and old_proof['mathematical_source_unchanged']is True,'original retained permit repair mathematical equality')
 safe=D/'v177-root-safe-storage-overlay/safe_storage177.py';need(f['external_files'][str(safe)]==sha(safe),'scoped177 storage race fix pinned')
 cleanup=terminal_cleanup(receipt)
 return dict(freeze=desc(freeze_path),CPU=desc(cpu_path),ROOT_prepared=desc(ROOT_PREPARED),source_ready=desc(READY),
  selected_current_SHA_checked=531,archive=own_archive,parent_archives=deps,current_HOST_tests=5,
  previous_revision_preserved_files=218,previous_HOST_tests=4,previous_tests_not_added_to_current=True,
  old_revision=desc(previous['manifest']['path']),original_P148_permit_identity=parent,
  saved_baseline_and_cleanup=cleanup,complete_environment_import_closure_claimed=False)

def terminal_cleanup(receipt):
 for key in('pre_verifier','post_verifier'):
  row=receipt[key];value=read(row['path'])
  need(sha(row['path'])==row['sha256']and row['exit_code']==0 and value==row['actual_result']and value['status']=='pass'and
       value['errors']==[]and(value['checked_frozen_files'],value['checked_external_files'],value['checked_packages'])==(10496,15,212),
       'saved original baseline10496/15/212 '+key)
 for key in('devices_before','devices_after'):
  value=receipt[key];need(value['all_idle']is value['four_910B2']is True and set(value['processes'])=={'2','5','6','7'},'saved four idle cards '+key)
  need(all(v['exit_code']==0 and v['idle']is True and not v.get('timed_out',False)and
       re.search(r'No process in device\.|No running processes found',v['output'],re.I)for v in value['processes'].values()),'saved idle output '+key)
 need(receipt['physical_cards']==[2,5,6,7]and receipt['sources_unchanged']is True and receipt['owned_remaining']==[]and
      receipt['execution_lock']['path']==str(B/'.ROOT_NPU_EXECUTION.lock')and receipt['execution_lock']['nonblocking_exclusive']is True,
      'original ROOT self-lock/owned cleanup/source continuity')
 limits=dict(child_execution_seconds=7200,actual_tag_bytes=256*1024**2,aggregate_private_bytes=512*1024**2)
 need(receipt['limits']==limits and type(receipt['elapsed_seconds'])in(int,float)and math.isfinite(receipt['elapsed_seconds'])and
      receipt['elapsed_seconds']>0,'saved bounded elapsed and exact limits')
 if receipt['status']=='complete':need(receipt['elapsed_seconds']<7200,'complete within actual bound')
 storage=receipt['private_storage_final'];need(0<=storage['actual_tag_bytes']<=limits['actual_tag_bytes']and
      0<=storage['aggregate_private_bytes']<=limits['aggregate_private_bytes'],'private tag/aggregate bounds')
 return dict(saved_baseline='10496/15/212 PASS pre/post; not rerun',saved_four_idle=True,owned_remaining=[],storage=storage)

def verify_integer(rows,heads):
 # Full80 target BONUS head, including h4 after all four accepted candidates.
 ht=types.SimpleNamespace(Tensor=Value,tensor=lambda v,**kw:Value(v),long='long')
 ns=dict(torch=ht,TreeVerificationOutput=lambda a,b:types.SimpleNamespace(token_ids=a,accepted_node_indices=b))
 verify=extracted(NATIVE/'native_engine.py','_verify_tree_outputs_host',ns)
 plans=[types.SimpleNamespace(candidate_budget=4,parent_indices=Value([-1,0,1,2]))for _ in rows]
 need(len(rows)==16 and all(len(r)==4 for r in rows)and len(heads)==80,'real16 root+four BONUS head carrier')
 out=verify(rows,plans,Value(heads),4,'independent-HOST-integers')
 return [[v for v in row if v>=0]for row in out.token_ids.values],out.accepted_node_indices.values

def native_delta(n,*,zero):
 names,failures=role_counter_fields()
 need(set(n['before'])==set(n['after'])==set(n['delta'])==set(names),'actual21 counter names')
 need(all(type(n[s][k])is int for s in('before','after','delta')for k in names)and
      all(0<=n['before'][k]<=n['after'][k]and n['after'][k]-n['before'][k]==n['delta'][k]for k in names),
      'strict actual monotone21 arithmetic')
 need(not any(n['delta'][k]for k in failures),'no Native failure discovery')
 if zero:need(not any(n['delta'].values()),'complete protocol zero replay')
 return 21 if zero else 0

def usage(u,*,require_both=False):
 """Per-producer actual method calls, not inference from target geometry."""
 before,after,d,rows=(u[k]for k in('before','after','delta','usage'))
 need(set(d)==set(before['counts'])|set(after['counts']),'complete counter delta key union')
 need(all(type(v)is int and v>=0 for part in(before['counts'],after['counts'])for v in part.values())and
      all(type(d[k])is int and d[k]==after['counts'].get(k,0)-before['counts'].get(k,0)>=0 for k in d),'strict actual usage subtraction')
 need(type(before['usage_length'])is type(after['usage_length'])is int and
      after['usage_length']-before['usage_length']==len(rows)==d.get('calls',0),'actual usage slice and call count')
 for row in rows:
  need(type(row['new'])is bool and type(row['generation'])is int and len(row['owners'])==len(row['prefixes'])==len(row['budgets'])and
       all(type(v)is int and v>=0 for v in row['owners']+row['prefixes'])and len(set(row['owners']))==len(row['owners']),'actual call identity and owner/prefix ABI')
  if row['new']:need(len(row['owners'])==16 and row['budgets']==[4]*16 and row['reason']=='qualified-no-missing','only current qualified noMissing real16B4 route')
  else:need(type(row['reason'])is str and row['reason']!='qualified-no-missing','honest fallback reason')
 new=sum(r['new']for r in rows);fallback=len(rows)-new
 need(d.get('new_calls',0)==new and d.get('fallback_calls',0)==fallback and
      sum(v for k,v in d.items()if k.startswith('fallback_')and k!='fallback_calls')==fallback,'actual new/fallback/reason counts')
 for key,m in(('new_model_stages',5),('new_head_stages',4),('new_query_rows',80),('new_generated_nodes',128)):
  need(d.get(key,0)==m*new,'actual five models/four heads/80 rows/128 nodes '+key)
 if require_both:need(new>0 and fallback>0,'actual new and original transition both exercised')
 return dict(counts=d,new_calls=new,fallback_calls=fallback,reasons=dict(collections.Counter(r['reason']for r in rows if not r['new'])))

def graph_execution(e):
 need(set(e)=={'mode','fallback_reason','capture_attempted','replay_executed'}and type(e['capture_attempted'])is type(e['replay_executed'])is bool,
      'actual NativeGraphExecution four serialized dataclass fields; no used_aclgraph property')
 need(e['mode']in('capture_replay','replay')and e['fallback_reason']is None and e['replay_executed']is True,'actual returned graph route, not fabricated property')


def draft_proof(d,source):
 need(d['raw_encoded_shape']==[16,4,2]and d['raw_pool_nodes']==128 and d['selected_C']==64 and d['variable_num_draft_tokens']==[4]*16 and
      all(d[k]is True for k in('exact_full8_encoded','all_physical_hidden_original_matcher','all28_query_KV_original_matcher','independent_history'))and
      d['source_KV_copied']is False and d['independent_owners']==list(range(16,32)),
      'retained actual independent full8/56KV/ledger hard gates; not HOST tensor recomputation')
 new=d['final_KV_only'];need(type(new)is bool and d['actual_draft_model_stages']==(5 if new else 4)and d['final_d4_KV_original_matcher']is new,
      'explicit fifth d4 hidden/KV versus original transition')
 if not new:return dict(final_KV=False,metadata_records=0,census_records=0)
 need(d['natural_missing_suffix_observed']is False and len(d['final_metadata'])==5,'five real PA stage records and no missing suffix')
 for step,m in enumerate(d['final_metadata']):
  positions=[v['prefix']+step for v in source['contexts']];tables=m['block_tables']
  need(m['step']==step and m['owners']==list(range(16))and m['positions']==positions and m['contexts']==[p+1 for p in positions]and
       m['sequence_lens']==m['contexts']and m['actual_seq_lengths_q']==list(range(1,17))and len(tables)==16 and
       m['use_fused_infer_attention']is False and m['slot_dtype']=='torch.int32','actual original PA five-stage owner/position/context/table ABI')
  slots=[]
  for table,p in zip(tables,positions,strict=True):
   page=table[p//128];need(type(page)is int and 0<=page<544,'actual allocated page');slots.append(page*128+p%128)
  need(m['slots']==slots and len(set(slots))==16,'independent physical slot arithmetic from retained tables')
 census=[e for e in d['model_head_census']if e['final_kv_only']]
 need(all(e==dict(steps=5,final_kv_only=True,model=5,head=4)for e in census),'real eager/capture five model calls/four heads')
 # Warm replay may execute no Python model body; an empty per-call census is honest.
 return dict(final_KV=True,metadata_records=5,census_records=len(census))


def independent_final_component(directory,reports):
 folder=directory/'final-KV5-own-current-process';zeros=0;integer_calls=0;owner_verdicts=0;census=0;new_math=0
 roles=[];histories=[];labels=[];usage_rows=[];all_cycles={};executions=0
 expected_install=dict(real_class_method_installed=True,canonical_Fast_module_changed=False,decode_source_changed=False,
  original_bound_root_seed_retained=True,original_components_completed_before_install=True)
 for rank in range(4):
  r=read(folder/f'rank{rank}-component.json.gz')
  need(r['schema']=='V201.final-KV5-current-component.v1'and r['rank']==rank and r['role']==('draft'if rank==0 else'target')and r['errors']==[]and
       all(r[k]is True for k in('completed','math_independent_qualified','cache_released'))and
       all(r[k]is False for k in('Native_flags_assigned','adopted','formal_Goodput','whole_AR_qualified')),'actual role-specific retained component')
  need(reports[rank]['component']['final_KV5_install']==expected_install,'original components before honest outer method install')
  need(reports[rank]['component']['final_KV5']['audit']['passed']is True and len(r['traces'])==3,'same current report and three fresh traces')
  role_hist=[];role_counts=[]
  for p,t in enumerate(r['traces']):
   need(t['pass_index']==p and t['zero_replay']==(p>0)and t['complete_protocol_cycles']==4,'current three trace/four commit schedule')
   zeros+=native_delta(t['Native'],zero=p>0);fingerprints=[];previous=None
   for c in range(4):
    row=read(folder/f'rank{rank}-pass{p}-cycle{c}.json.gz');commit=row['commit']
    need(row['cycle']==c and row['phase']==('original-transition'if c==1 else'finalKV-eligible')and row['source']['ids']==list(range(16))and
         row['selected']==[list(range(4))]*16 and len(row['rows'])==len(row['source']['contexts'])==16 and
         all(len(v)==4 for v in row['rows']),'actual source/selection/forced-original cycle identity')
    need(all(strict_zero(commit[k])for k in('actual_structural_failed','actual_numerical_failed'))and
         commit['original_finish_cycle_executed']is True and row['independent_commit']['histories_equal']is True and
         row['independent_commit']['candidate_KV_copied']is False,'original verdict/commit/own independent compaction evidence')
    if previous is not None:need(row['source']['contexts']==previous,'source follows actual previous committed history')
    previous=[dict(v,token_ids=list(tokens),prefix=len(tokens)-1,output_offset=len(tokens)-v['initial_prefix']-1)
              for v,tokens in zip(row['source']['contexts'],commit['actual_logical_tokens'],strict=True)]
    for v,tokens,emitted in zip(row['source']['contexts'],commit['actual_logical_tokens'],commit['committed_tokens'],strict=True):
     need(v['token_ids']+emitted==tokens and emitted,'original nonempty logical update from own current prefix')
    if rank==0:
     need(row['heads']is row['target']is row['reference']is None,'draft does not own target model tensors')
     summary=draft_proof(row['draft'],row['source']);census+=summary['census_records'];new_math+=int(summary['final_KV'])
     if c in(0,1):need(summary['final_KV']==(c==0),'fresh new lane and real original transition required')
    else:
     need(row['draft']is None,'target does not own draft model proof')
     emitted,paths=verify_integer(row['rows'],row['heads'])
     need(emitted==commit['committed_tokens']and paths==commit['accepted_paths'],'independent unchanged Native full80 BONUS integer verifier')
     ref=row['reference'];need(ref['same_M']==80 and ref['same_S']==16 and ref['independent_owners']==list(range(16,32))and
          all(ref[k]is True for k in('all_physical_hidden_original_matcher','all64_query_KV_original_matcher','all_physical_greedy_exact'))and
          ref['candidate_KV_copied']is False and ref['real_query_tokens']==row['heads'],'actual own full80 head/hidden/all128 targetKV evidence')
     for family in('greedy','hidden'):
      records=row['target'][family];need([e['phase']for e in records]==['first','changed','restored'],'actual current target first/changed/restored')
      for e in records:
       need(type(e['runtime_validated_before'])is type(e['runtime_validated_after'])is bool and e['runtime_validated_after']is(e['runtime_validated_before']or e['phase']!='first'),
            'actual natural Native flag progression, not assignment');graph_execution(e['execution']);executions+=1
     integer_calls+=1;owner_verdicts+=16
    fp=digest(dict(source=row['source'],rows=row['rows'],selected=row['selected'],confidences=row['confidences'],tokens=commit['committed_tokens'],paths=commit['accepted_paths']))
    fingerprints.append(fp);key=(p,c)
    shared={k:row[k]for k in('source','rows','selected','confidences')};shared.update(tokens=commit['committed_tokens'],paths=commit['accepted_paths'])
    if key in all_cycles:need(all_cycles[key]==shared,'all4 actual proposal/verdict/commit agreement')
    else:all_cycles[key]=shared
   nxt=read(folder/f'rank{rank}-pass{p}-next.json.gz')
   need(nxt['source']['contexts']==previous and nxt['selected']==[list(range(4))]*16,'own next follows fourth actual commit')
   if rank==0:
    summary=draft_proof(nxt['draft_proof'],nxt['source']);census+=summary['census_records']
   else:need(nxt['draft_proof']is None,'target next draft proof is None')
   actual=dict(cycles=fingerprints,next={k:nxt[k]for k in('source','rows','selected','confidences')})
   need(actual==t['actual_history'],'independent source/history fingerprints and own next')
   role_hist.append(actual)
   u=usage(t['usage'],require_both=rank==0)
   if rank:need(u['counts']=={}and not t['usage']['usage'],'target has no draft usage counts')
   role_counts.append(u)
  need(role_hist[0]==role_hist[1]==role_hist[2],'both fresh complete zero histories reproduce')
  inv=r['inventory'];need(inv['inherited_entry_identity_preserved']is True and inv['Native_flags_assigned']is False,'actual object-identity/source flag gate evidence')
  for family in('entries','draft_entries','target_entries'):
   before,after=map(set,(inv['before'][family],inv['after'][family]));want={repr(NEW_KEY)}if rank==0 and family=='draft_entries'else set()
   need(before<=after and after-before==want and set(inv['added'][family])==want,'one exact new draft key; target keys remain inherited actual objects')
  ev=r['boundary_events'];seq=[(e['sequence'],e['label'])for e in ev]
  need([v[0]for v in seq]==list(range(len(seq)))and all(len(e['ranks'])==4 and all(v['error']is None for v in e['ranks'])for e in ev),'actual common all4 boundary sequence')
  roles.append(dict(rank=rank,role=r['role'],new_draft_keys=1 if rank==0 else 0,commits=12,next=3));histories.append(role_hist);labels.append(seq);usage_rows.append(role_counts)
 need(all(v==histories[0]for v in histories)and all(v==labels[0]for v in labels)and zeros==168 and integer_calls==36 and owner_verdicts==576 and new_math>=3 and census>0,
      'all4 source/protocol identity,168zero,36BONUS verdicts,current natural capture census')
 return dict(roles=roles,Native_zero_scalar_subtractions=zeros,independent_Native_integer_verifier_calls=integer_calls,
  independent_owner_verdicts=owner_verdicts,actual_new_math_calls=new_math,retained_model_head_census_records=census,
  target_execution_records_four_fields_checked=executions,usage_by_rank=usage_rows,same_history_all4=True,
  hidden_KV_tensor_recomputed_on_HOST=False,current_engine_issuer_is_source_and_retained_gate_not_serialized_object_identity=True)


def independent_full_usage(reports):
 for report in reports:
  scope=report['standalone_scope']
  need(scope['schema']=='V201.current-final-KV5-normal-only-scope.v1'and
       scope['current_final_KV5_usage']is scope['fifth_hidden_and_56KV_separately_qualified']is scope['target_protocol_and_g_unchanged']is True and
       scope['new_draft_keys']==1 and scope['model_stages']==5 and scope['head_stages']==4 and
       scope['final_KV_domain']=='normalN16/allB4/noE/noMissing','current engine issuer report is explicitly fifth-KV scope')
 rows=[]
 for index,env in enumerate(reports[0]['windows']['traces']):
  u=env['original_trace']['actual_final_KV201'];value=usage(u)
  need(env['window']==32 and env['role']in('warmup65','zero65-1','zero65-2','full463'),'actual usage tied to original trace role/window')
  if env['role']=='full463':need(value['new_calls']>0,'full actually used current fifth-KV route')
  for rank in range(1,4):
   other=reports[rank]['windows']['traces'][index]['original_trace']['actual_final_KV201'];x=usage(other)
   need(not other['usage']and not any(x['counts'].values()),'only draft owns actual execution usage')
  rows.append(dict(role=env['role'],window=32,**value))
 need([r['role']for r in rows]==['warmup65','zero65-1','zero65-2','full463'],'all original fresh trace roles')
 return dict(traces=rows,full=rows[-1],counts_are_model_method_calls_not_target_call_counts=True,
  no_GP_gain_or_saved_latency_inferred_from_stage_counts=True)


def source_import_origins(pins):
 # Original dependency manifests are themselves selected201 bytes and were
 # fully verified by provenance/require_parent(archive=True). Read their
 # source maps, rather than pretending the thin201 map lists every imported
 # P148 dependency individually.
 pins=dict(pins)
 for path in (B/'tree-empty-active-cohort-causal80-active32-full-v197/FROZEN_SOURCE197.json',
              B/'tree-causal80-real16-bankfix-full-v173/FROZEN_SOURCE173.json',
              B/'tree-high-normal-cap4-native-families-v148/PREPARED_SOURCE_HASHES_V132.json'):
  need(str(path)in pins and sha(path)==pins[str(path)],'selected exact dependency source manifest')
  f=read(path);pins=merge_pins(pins,f['files'],f['external_files'])
 checked=[]
 for name,module in list(sys.modules.items()):
  if module is None:continue
  path=vars(module).get('__file__')
  if not isinstance(path,str)or not Path(path).is_absolute():continue
  source=Path(path).resolve()
  # Only the private candidate and frozen source tree roots are claimed here.
  if source==Path(__file__).resolve()and name=='__main__':continue
  if source.is_relative_to(P)or source.is_relative_to(B):
   need(str(source)in pins and sha(source)==pins[str(source)],'actual imported selected module source '+name);checked.append(name)
 need(not Firewall.attempts and not any(n.split('.')[0]in BACKENDS for n in sys.modules),'no imported backend/Torch')
 return checked


def failure_evidence(actual,receipt):
 items=[]
 for path in sorted(actual.rglob('*')):
  if path.is_file()and('last-error'in path.name or'partial-error'in path.name):
   try:items.append(dict(file=desc(path),value=read(path)))
   except Exception as e:items.append(dict(file=desc(path),read_error=repr(e)))
 return dict(child=receipt.get('child'),outer_errors=receipt.get('errors'),worker_or_component_failures=items,
  completed_file_present=(actual/'completed.json').is_file(),job_remains_failed=True,full_qualification_issued=False,
  Goodput_assigned=False,actual_first_stage_must_come_from_retained_error_not_generic_timeout_text=True)


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--terminal-authorized',action='store_true',required=True)
 parser.add_argument('--tag',required=True);parser.add_argument('--expected-receipt-sha256',required=True)
 parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
 need(a.terminal_authorized is True and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',a.tag)is not None,'ROOT explicit terminal authorization and safe exact tag')
 need(re.fullmatch(r'[0-9a-f]{64}',a.expected_receipt_sha256)is not None,'ROOT exact receipt SHA argument')
 need(a.output.parent.resolve()==D and not a.output.exists(),'fresh private D result')
 rp=P/('receipt-'+a.tag+'.json')
 # No actual/receipt read before caller provides explicit terminal identity.
 need(rp.is_file()and not rp.is_symlink()and rp.resolve()==rp and sha(rp)==a.expected_receipt_sha256,'ROOT authorized exact private terminal receipt bytes')
 receipt=read(rp);need(receipt['status']in('complete','failed')and receipt['action']==ACTION,'terminal201 only; running rejected before actual read')
 actual=P/'actual'/a.tag;errors=[]
 result=dict(schema='V201.independent-terminal-HOST-audit.v1',candidate=str(P),tag=a.tag,job_status=receipt['status'],
  script=desc(__file__),receipt=desc(rp),errors=errors,Torch_imported=False,NPU_used=False,
  whole_workload_AR_qualified=False,formal_Goodput=False,adopted=False)
 sys.meta_path.insert(0,Firewall());sys.path.insert(0,str(P))
 try:
  result['provenance']=provenance(receipt)
 except Exception as e:errors.append(dict(stage='source_archive_CPU_oldrevision_and_saved_cleanup',error=repr(e)))
 try:
  need(actual.is_dir()and actual.resolve()==actual and not actual.is_symlink(),'private terminal actual directory')
  paths=sorted(actual.rglob('*'));need(not any(p.is_symlink()for p in paths),'no raw aliases')
  manifest={str(p):desc(p)for p in paths if p.is_file()};mp=a.output.with_name(a.output.stem+'_RAW_MANIFEST.json')
  need(not mp.exists(),'fresh raw manifest')
  with mp.open('x')as f:json.dump(manifest,f,indent=2,sort_keys=True);f.write('\n')
  result['raw_manifest']=dict(**desc(mp),files=len(manifest))
  need(read(actual/'worker-exit.json')==dict(closed=True,all_exited=True),'actual all4 workers closed')
 except Exception as e:errors.append(dict(stage='raw_manifest_and_worker_close',error=repr(e)))
 if receipt['status']=='complete'and'provenance'in result and'raw_manifest'in result:
  try:
   import audit201
   original=audit201.audit(actual,(32,));complete=read(actual/'completed.json')
   need(original['passed']is True and original==complete['audit']==receipt['actual_full_audit']and complete['action']==ACTION and
        complete['replies']==[['calibrated',r,dict(windows=[32],full463=True)]for r in range(4)],'original exact audit/completed/guard equality')
   need(receipt['errors']==[]and receipt['child']['exit_code']==0 and receipt['actual_execution_complete']is True and
        receipt['actual_full_status']=='passed_full'and not list(actual.glob('rank*-last-error.json')),'actual complete201 job; guard status is not crossM equivalence')
   need(original['native_zero_subtractions']==252 and original['math_independent_qualified']is True and
        original['current_multiowner_eager_component']['Native_zero_scalar_subtractions']==336 and
        original['current_final_KV5_component']['actual21_zero_subtractions']==168,'original full252, multiowner336 and new168 zero evidence')
   result['original_unmodified_audit']=dict(source=desc(P/'audit201.py'),result=original)
   result['actual_compatibility_classification']=dict(old_padding_behavior_equal=original['old_padding_behavior_equal'],
    numerical_variant_only=original['numerical_variant_only'],not_whole_AR_or_lossless_qualification=True)
   result['independent_original_B4_and_M80_integer_verdicts']=component_integer_verdicts(actual)
  except Exception as e:errors.append(dict(stage='original_all_current_components_and_full_audit',error=repr(e)))
  try:
   reports=[read(actual/f'rank{r}-local-diagnostic.json')for r in range(4)]
   traces=[read(actual/'windows/window32/full463'/f'rank{r}-full463-actual-trace.json.gz')for r in range(4)]
   result['independent_fifth_KV_component']=independent_final_component(actual,reports)
   result['independent_multiowner']=independent_multiowner(actual,traces,reports)
   result['actual_new_and_fallback_usage']=independent_full_usage(reports)
   result['retained_full463']=admission_and_full_statistics(traces,reports)
  except Exception as e:errors.append(dict(stage='independent_fifthKV_integer_metadata_usage_or_full463',error=repr(e)))
 elif receipt['status']=='failed':
  result['failure_only']=(failure_evidence(actual,receipt)if'raw_manifest'in result else
   dict(child=receipt.get('child'),outer_errors=receipt.get('errors'),raw_not_traversed_without_private_directory_verification=True,job_remains_failed=True,Goodput_assigned=False,full_qualification_issued=False))
 try:
  f=read(P/'FROZEN_SOURCE201.json');pins=merge_pins(*(f[k]for k in('files','external_files')))
  result['selected_module_origins_checked']=source_import_origins(pins)
 except Exception as e:errors.append(dict(stage='pure_HOST_source_origin_or_backend_firewall',error=repr(e)))
 result['status']=(('passed_terminal_current_component_and_full463_audit'if receipt['status']=='complete'else'terminal_failed_job_evidence_only')
                   if not errors else'terminal_failed_or_incomplete_evidence_with_audit_errors')
 result['limitations']=[
  'Independent HOST recomputes unchanged Native integer BONUS verdicts, full8/source history hashes, metadata owner/slot/table arithmetic, counter deltas, current usage and original charged/publication clocks. It cannot reexecute hidden/logit/KV comparisons or device finite predicates; retained actual source-gated booleans are explicitly not CPU tensor proof.',
  'The new route is only real16 normal allB4/noE/noMissing: five model calls, four heads, full8 candidate pairs, fifth d4 KV. Real root_seed/other-budget/repair/lowN/truefork fallback and original BONUS/g/clock/publication remain original. Call counts alone cannot predict latency or GP.',
  'Current issuer and inherited Native entry-object identities are runtime in-memory gates bound by frozen source and retained evidence; JSON cannot recreate or issue those objects. NativeGraphExecution serializes only four dataclass fields; used_aclgraph is a derived property.',
  'Original201/44/114 plus original current B4/M80 and N2/N32 eager prefill/next are prerequisites. N32 prefill is not a new targetN32 scope. Exactly one finalKV draft key is added; inherited new_target_keys2 refers to original173 M80 keys.',
  'Native charged TPOT/256 and worker-publication first-last/255 are distinct clocks; both keep original generate wall denominator. TTFT includes original arrivals. No external client-delivery or final9 common-clock claim is fabricated; HOST phase spans are not additive device timings.',
  'Finite sameM hidden/head/56draftKV/128targetKV and current protocol tests do not establish whole463 independent AR equivalence, losslessness, adoption or formal Goodput. CrossM numerical variants stay explicitly separate.',
  'A failed job remains failed; missing artifacts are localized failed/incomplete evidence, never GP0 or invented full qualification. Generic within7200 text alone cannot identify a timeout.'
 ]
 result['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with a.output.open('x')as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
 print(json.dumps(dict(**desc(a.output),status=result['status'],job_status=result['job_status'],errors=errors)))
 return int(bool(errors))

if __name__=='__main__':raise SystemExit(main())
