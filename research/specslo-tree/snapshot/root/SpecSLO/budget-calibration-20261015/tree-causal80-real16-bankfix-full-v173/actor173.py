"""Own N16/Q80 Native qualification and actual original commit/next history.

No cost clock. Cross-M equality is independent of the same-M numerical gate.
Both layouts run the same current draft/input; old128 uses its retained greedy
key and fresh own-bank eager oracle, never a new hidden128 graph.
"""
import copy,gzip,json,traceback
from pathlib import Path
from base173 import BASE,require
from scope_v146 import NATIVE_COUNTERS,FAILURES,inventory_record,validate_full_configuration,digest
from actor_v146 import native,delta,stage4,save,fingerprint
from full_process_v146 import capture_fixture_state,finish_fixture_state,reset_fixture_gain,track_actual_target_key
from mapper173 import GREEDY,HIDDEN,qualification_geometry


def old_control(engine,case,bank,work,stage):
 import torch
 from bank173 import prefix_evidence,query_KV
 from target173 import metadata,original_call
 target=not engine.is_draft
 def prepare():
  if not target:return None
  require(GREEDY not in engine.graph_runner.target_entries or engine.graph_runner.target_entries[GREEDY].runtime_validated,
   'new80 is either absent before discovery or naturally closed afterwards')
  return metadata(engine,case,work),prefix_evidence(engine,{i:v[:-1] for i,v in enumerate(bank.contexts)})
 before=stage('old128/prepare',prepare)
 value=stage('old128/original-greedy',lambda:original_call(engine,case,work) if target else None)
 def snapshot():
  if not target:return None
  out,physical=value
  require(physical.numel()==128 and bool(torch.isfinite(physical).all()),'old all128 real greedy snapshot')
  require(prefix_evidence(engine,{i:v[:-1] for i,v in enumerate(bank.contexts)})==before[1],'old candidate protected prefix')
  return query_KV(engine,before[0][2].slot_mapping.long())
 actual_kv=stage('old128/immediate-query-KV',snapshot)
 def ref_prepare():
  if not target:return None
  return metadata(engine,case,work,reference=True),prefix_evidence(engine,{i+16:v[:-1] for i,v in enumerate(bank.reference_contexts)})
 reference=stage('old128/fresh-reference-prepare',ref_prepare)
 hidden=stage('old128/independent-eager-model',lambda:engine.model(*reference[0]).clone() if target else None)
 def ref_finite():
  if not target:return None
  kv=query_KV(engine,reference[0][2].slot_mapping.long())
  require(len(kv)==128 and all(bool(torch.isfinite(v).all()) for v in (hidden,*kv))
   and prefix_evidence(engine,{i+16:v[:-1] for i,v in enumerate(bank.reference_contexts)})==reference[1],
   'old independent all64queryKV finite and prefix protected before TP head')
  return kv
 ref_kv=stage('old128/reference-query-finite',ref_finite)
 head=stage('old128/independent-full128-head',lambda:engine.model.compute_greedy_tokens(hidden,engine.draft_vocab_size).clone() if target else None)
 def drain():
  if not target:return None
  out,physical=value
  require(torch.equal(physical,head) and engine.graph_runner._target_outputs_match(actual_kv,ref_kv),
   'old same-M actual all128 head/all64queryKV independent eager reference')
  require(torch.equal(out['target_query_token_ids'],physical[:80]),'old real80 head is first actual128 outputs')
  verdict=engine._verify_tree_outputs_host(work['rows'],work['plans'],physical[:80],4,engine.device)
  return dict(real80=physical[:80].cpu().tolist(),all128=physical.cpu().tolist(),
   verdict_tokens=verdict.token_ids.cpu().tolist(),accepted=verdict.accepted_node_indices.cpu().tolist(),
   sameM_greedy_and_KV_checked=True,new_hidden_key_created=False,
   physical_M=128,physical_S=38,candidate_KV_copied=False)
 return stage('old128/head-KV-and-verdict',drain)


def inventory_before(engine):
 runner=engine.graph_runner
 counts=dict(entries=0,draft_entries=60,target_entries=54) if engine.is_draft else dict(entries=0,draft_entries=0,target_entries=247)
 require({f:len(getattr(runner,f)) for f in counts}==counts,'after actual originalB4 component, before new80')
 if not engine.is_draft:require(all(k not in runner.target_entries for k in (GREEDY,HIDDEN)),'new80 two keys genuinely cold')
 return dict(record=inventory_record(engine),objects={f:dict(getattr(runner,f)) for f in counts},
  flags={f:{k:(v.runtime_validated,getattr(v,'validated_real_row_count',None)) for k,v in getattr(runner,f).items()} for f in counts},
  disabled=set(runner.disabled_entry_keys))


def inventory_after(engine,before):
 runner=engine.graph_runner;after=inventory_record(engine)
 wanted={repr(GREEDY),repr(HIDDEN)} if not engine.is_draft else set()
 changes={}
 for f,old in before['objects'].items():
  require(set(after[f])-set(before['record'][f])==(wanted if f=='target_entries' else set())
   and set(before['record'][f])<=set(after[f]),'only new80 greedy+hidden keys; no inherited key removed')
  changes[f]={}
  for k,v in old.items():
   require(getattr(runner,f)[k] is v,'inherited authoritative Native object identity')
   was=before['flags'][f][k];now=(v.runtime_validated,getattr(v,'validated_real_row_count',None))
   require((not was[0] or now[0] is True) and (was[1] is None or (now[1] is not None and now[1]>=was[1])),
    'inherited validation cannot regress')
   if was!=now:changes[f][repr(k)]=dict(before=list(was),after=list(now))
 require(set(runner.disabled_entry_keys)==before['disabled'],'no disabled Native key changed')
 if not engine.is_draft:
  require(all(runner.target_entries[k].runtime_validated is True for k in (GREEDY,HIDDEN)),'both80 original natural changed gates closed')
 return dict(before=before['record'],after=after,new_keys=sorted(wanted),inherited_entry_identity_preserved=True,
  natural_validation_changes=changes,Native_flags_assigned=False,disabled_keys_unchanged=True)


def run(engine,native_module,stage,policy,output):
 import torch
 from bank173 import build_bank,cases
 from draft173 import generate
 from target173 import qualify,reference,commit_reference
 from original_tree_cycle_v75 import setup_normal_cycle,attach_normal_cycle_proposals,run_original_tree_cycle
 from vllm_ascend.spec_decode.pearl.mc2 import snapshot_mc2_dispatch_counters
 folder=Path(output)/'causal80-own-current-process'
 stage('causal80/fresh-output',lambda:folder.mkdir(exist_ok=True))
 events=[];s=lambda label,fn:stage4(engine,events,'V173/causal80/'+label,fn)
 report=dict(schema='V173.causal80-own-component.v1',rank=engine.rank,role='draft' if engine.is_draft else 'target',
  completed=False,traces=[],errors=[],adopted=False,formal_Goodput=False,cost_measured=False,
  source_root_teacher_forced_not_own_AR=True,subsequent_draft_and_original_verdict_actual=True,
  crossM_difference_is_not_sameM_KV_corruption=True,old_padding_behavior_equal=None,math_independent_qualified=False)
 engine._v173_causal80_partial=report
 fixture=s('fixture-state',lambda:capture_fixture_state(engine))
 gain=s('gain-before',lambda:reset_fixture_gain(policy))
 before=s('inventory-before',lambda:inventory_before(engine))
 report['configuration']=s('actual-full-configuration',lambda:validate_full_configuration(engine))
 dispatch=s('MC2-before',snapshot_mc2_dispatch_counters)
 expected=None
 try:
  with track_actual_target_key(engine):
   for pass_index in range(3):
    new,old=cases();prefix=f'pass{pass_index}'
    lane=lambda label,fn:s(prefix+'/'+label,fn)
    bank=build_bank(engine,new,stage=lambda label,fn:lane('fresh-dual-bank/'+label,fn))
    start=lane('Native-before',lambda:native(engine));rows=[];state=None
    for cycle in range(4):
     c=lambda label,fn:lane(f'cycle{cycle}/'+label,fn)
     work=generate(engine,native_module,bank,new,100000+pass_index*5+cycle,c)
     def publish():
      nonlocal state
      if state is None:state=setup_normal_cycle(native_module,work['source'],work['plans'],work['rows'],work['confidences'],compiled_protocol=engine._v75_original_protocol)
      attach_normal_cycle_proposals(state,work['source'],work['plans'],work['rows'],work['confidences'],cycle_id=cycle)
      work['work_plan'].require_published_tickets(state['tickets'])
     c('original-typed-publication',publish)
     old_result=old_control(engine,old,bank,work,c)
     def qualifier_counts():return copy.deepcopy(getattr(engine,'_dynamic_verification_qualification_counts',{}))
     qbefore=c('original-autoqualifier-before',qualifier_counts)
     geometry=qualification_geometry(engine)
     c('new80-geometry-open',lambda:geometry.__enter__())
     try:
      candidate=qualify(engine,new,bank,work,stage=c)
      qafter=c('original-autoqualifier-after',qualifier_counts)
      oracle=reference(engine,new,bank,work,candidate,stage=c)
     finally:
      c('new80-geometry-close',lambda:geometry.__exit__(None,None,None))
     def cross():
      if engine.is_draft:return None
      actual=candidate['physical_tokens'].cpu().tolist();previous=old_result['real80']
      require(len(actual)==len(previous)==80,'both complete actual real80 heads retained')
      positions=[i for i,(a,b) in enumerate(zip(previous,actual,strict=True)) if a!=b]
      return dict(old_real80=previous,new_real80=actual,different_head_positions=positions,
       head_equal=not positions,old_verdict_tokens=old_result['verdict_tokens'],new_verdict_tokens=oracle['verdict_tokens'],
       old_accepted=old_result['accepted'],new_accepted=oracle['accepted'],
       verdict_equal=old_result['verdict_tokens']==oracle['verdict_tokens'],accepted_equal=old_result['accepted']==oracle['accepted'],
       old_layout=dict(M=128,S=38),new_layout=dict(M=80,S=16),same_work_source=True,
       same_source_digest=digest(work['source']),cause_established=False)
     cross_result=c('actual-old-new-crossM-full80-head-verdict',cross)
     commit=run_original_tree_cycle(engine,native_module,work['source'],work['plans'],work['rows'],
      candidate['output'] if not engine.is_draft else None,state,cycle_id=cycle,target_home=cycle%2,stage_all4=c)
     own=c('independent-own-query-compaction-history',lambda:commit_reference(engine,bank,work,oracle,commit))
     row=dict(cycle=cycle,source=work['source'],selected=work['selected'],rows=work['rows'],confidences=work['confidences'],
      draft=work['draft_proof'],target=None if engine.is_draft else candidate['report'],
      reference=None if engine.is_draft else oracle['report'],old128_control=old_result,crossM=cross_result,
      original_autoqualification=dict(before=qbefore,after=qafter),commit=commit,independent_commit=own,
      work_plan_id=work['work_plan'].plan_id,selected_normal_candidates=work['work_plan'].allocated_selected_candidates,
      controlled_typed_fixture_grants_not_policy_allocation=True)
     rows.append(row)
     c('retain-completed-cycle',lambda:save(folder,f'rank{engine.rank}-pass{pass_index}-cycle{cycle}.json.gz',row))
     del candidate,oracle,work
    next_work=generate(engine,native_module,bank,new,100004+pass_index*5,lambda label,fn:lane('next-draft/'+label,fn))
    nxt=dict(source=next_work['source'],selected=next_work['selected'],rows=next_work['rows'],confidences=next_work['confidences'],draft=next_work['draft_proof'])
    lane('retain-next-draft',lambda:save(folder,f'rank{engine.rank}-pass{pass_index}-next.json.gz',nxt))
    del next_work
    def finish_trace():
     value=dict(pass_index=pass_index,zero_replay=pass_index>0,complete_protocol_cycles=4,
      Native=delta(start,native(engine),zero=pass_index>0),lease=bank.evidence(),fingerprints=[fingerprint(v) for v in rows],
      next_draft={k:nxt[k] for k in ('source','selected','rows','confidences')},
      natural_missing_suffix_observed=any(v['draft']['natural_missing_suffix_observed'] for v in rows) or nxt['draft']['natural_missing_suffix_observed'] if engine.is_draft else None,
      crossM_behavior_equal=all(v['crossM']['head_equal'] and v['crossM']['verdict_equal'] and v['crossM']['accepted_equal'] for v in rows) if not engine.is_draft else None)
     actual=dict(commits=value['fingerprints'],next_draft=value['next_draft'])
     nonlocal expected
     if pass_index:require(actual==expected,'both zero fullprotocol traces reproduce actual history/draft/commit/next')
     else:expected=actual
     return value
    report['traces'].append(lane('complete-protocol-trace',finish_trace))
    lane('release-original-cache',engine._release_cache)
  report['inventory']=s('only-two-new80-Native-keys',lambda:inventory_after(engine,before))
  def mc2():
   now=snapshot_mc2_dispatch_counters();difference={k:now[k]-dispatch[k] for k in dispatch}
   require(all(v>=0 for v in difference.values()) and difference['fallback']==difference['exception']==0
    and (engine.is_draft or difference['native_epilogue_success']>0),'original MC2 no fallback/exception and actual target dispatch')
   return dict(before=dispatch,after=now,delta=difference)
  report['MC2_dispatch']=s('actual-MC2',mc2)
  report['gain_reset']=dict(before=gain,after=s('gain-after',lambda:reset_fixture_gain(policy)))
  report['fresh_boundary']=s('release-fixture-state',lambda:finish_fixture_state(engine,fixture))
  def summary():
   require(engine.cache_allocation is None and engine.cache_block_tables is None and not engine.graph_runner.graph_cache_sealed,
    'component released; production unified seal pending')
   return dict(rank=engine.rank,passed=True,complete_zero_traces=2,
    crossM_behavior_equal=all(t['crossM_behavior_equal'] for t in report['traces']) if not engine.is_draft else None,
    natural_missing_suffix_observed=any(t['natural_missing_suffix_observed'] for t in report['traces']) if engine.is_draft else None)
  votes=stage4(engine,events,'V173/causal80/all4-complete',summary,share=True)
  evidence=[r['evidence'] for r in votes]
  require({r['rank'] for r in evidence}==set(range(4)) and all(r['passed'] and r['complete_zero_traces']==2 for r in evidence),
   'own actual all4 math/protocol/two-zero required')
  require(evidence[0]['rank']==0 and evidence[0]['natural_missing_suffix_observed'] is True,'own N16 natural missing suffix must exercise next repair')
  # Missing-suffix is a natural observation, never manufactured. Protocol
  # negative coverage is retained separately and cannot be borrowed from B4.
  report.update(completed=True,math_independent_qualified=True,old_padding_behavior_equal=all(r['crossM_behavior_equal'] for r in evidence if r['rank']!=0),
   all4_evidence=evidence,component_sealed=False,Native_flags_assigned=False,cache_released=True,
   numerical_variant_only=not all(r['crossM_behavior_equal'] for r in evidence if r['rank']!=0))
  report['boundary_events']=events
  s('retain-final-component',lambda:save(folder,f'rank{engine.rank}-component.json.gz',report))
  from audit80173 import audit
  result=s('audit-retained-component',lambda:audit(folder))
  proof=dict(directory=str(folder),audit=result,Native_flags_assigned=False,component_sealed=False)
  engine._v173_actual_causal80=proof
  return proof
 except BaseException:
  try:save(folder,f'rank{engine.rank}-partial-component-error.json.gz',dict(partial=report,error=traceback.format_exc(),adopted=False))
  except BaseException:pass
  raise
