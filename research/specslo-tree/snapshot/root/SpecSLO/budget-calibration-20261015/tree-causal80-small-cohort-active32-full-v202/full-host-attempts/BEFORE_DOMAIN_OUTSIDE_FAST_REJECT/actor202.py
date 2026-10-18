"""Seven true-N protocol histories on retained current80 entries; no new keys."""
import copy,gzip,json
from pathlib import Path
from base202 import require
from actor_v146 import native,delta,stage4,save,fingerprint
from full_process_v146 import capture_fixture_state,finish_fixture_state,reset_fixture_gain,track_actual_target_key
from mapper202 import GREEDY,HIDDEN,qualification_geometry

def subset(engine,work,n,ordinal):
 from split_budget_v132 import DraftWorkPlan,DraftWorkLimit,TargetAdmissionLimit
 ids=list(range(n));source=dict(ids=ids,contexts=work['source']['contexts'][:n]);budgets=dict.fromkeys(ids,4)
 plan=DraftWorkPlan(plan_id=ordinal,target_limit=TargetAdmissionLimit(),work_limit=DraftWorkLimit(),available_selected_candidates=192,
  requested_normal=tuple(ids),admitted_eager=(),normal_budgets=budgets,eager_budgets={},progress_gaps=dict.fromkeys(ids,0.),eager_priorities={},caps=budgets,needs=budgets,reserve_requested=0,reserve_used=0)
 result={k:work[k][:n]for k in('plans','rows','confidences','selected')}
 result.update(source=source,work_plan=plan,draft_proof=work['draft_proof'],full16_draft_before_subset=True,_engine=engine)
 plan.require_work(ids,[4]*n,[False]*n,result['plans'])
 return result

def inventory(e):
 r=e.graph_runner
 return dict(objects={f:dict(getattr(r,f))for f in('entries','draft_entries','target_entries')},flags={f:{k:(v.runtime_validated,getattr(v,'validated_real_row_count',None))for k,v in getattr(r,f).items()}for f in('entries','draft_entries','target_entries')},disabled=set(r.disabled_entry_keys))

def same_inventory(e,before):
 now=inventory(e)
 for f,old in before['objects'].items():
  require(set(old)==set(now['objects'][f])and all(now['objects'][f][k]is v for k,v in old.items()),'zero new entries; every actual Native object retained')
  require(now['flags'][f]==before['flags'][f],'all original flags/realrow values retained exactly')
 require(now['disabled']==before['disabled'],'no disabled keys')
 return dict(zero_new_keys=True,objects_retained=True,flags_unchanged=True,counts={f:len(v)for f,v in now['objects'].items()},key_objects={repr(k):id(e.graph_runner.target_entries[k])for k in(GREEDY,HIDDEN)}if not e.is_draft else{})

def run(e,native_module,stage,policy,output):
 from bank202 import build_bank,cases,prefix_evidence
 from draft173 import generate
 from target202 import qualify,reference,commit_reference,old_control
 from original_tree_cycle_v75 import setup_normal_cycle,attach_normal_cycle_proposals,run_original_tree_cycle
 from scope_v146 import digest
 from candidate_phases_v132 import update_snapshot
 from vllm_ascend.spec_decode.pearl.mc2 import snapshot_mc2_dispatch_counters
 folder=Path(output)/'causal80-seven-real-N-current-process';stage('usage202/output',lambda:folder.mkdir(exist_ok=False))
 events=[];s=lambda name,fn:stage4(e,events,'V202/usage/'+name,fn)
 fixture=s('fixture',lambda:capture_fixture_state(e));gain=s('gain-before',lambda:reset_fixture_gain(policy));before=s('entry-object-inventory',lambda:inventory(e));mc2=s('MC2-before',snapshot_mc2_dispatch_counters)
 report=dict(schema='V202.seven-real-N-current-usage.v1',rank=e.rank,role='draft'if e.is_draft else'target',completed=False,traces=[],errors=[],production_parent197=True,formal_Goodput=False,adopted=False,Native_flags_assigned=False,whole_AR=False)
 e._v202_usage_partial=report;expected={}
 try:
  with track_actual_target_key(e):
   for p in range(3):
    for case in cases():
     n=case['N'];lane=lambda name,fn:s('pass'+str(p)+'/N'+str(n)+'/'+name,fn)
     bank=build_bank(e,case,stage=lambda name,fn:lane('bank/'+name,fn));start=lane('Native-before',lambda:native(e));updates=lane('TTU-before',lambda:update_snapshot(e));rows=[];state=None
     for cycle in range(4):
      c=lambda name,fn:lane('cycle'+str(cycle)+'/'+name,fn)
      full=generate(e,native_module,bank,case['full_bank_case'],100000+p*100+n*5+cycle,c)
      work=c('real-firstN-work',lambda:subset(e,full,n,200000+p*100+n*5+cycle))
      def publish():
       nonlocal state
       if state is None:state=setup_normal_cycle(native_module,work['source'],work['plans'],work['rows'],work['confidences'],compiled_protocol=e._v75_original_protocol)
       attach_normal_cycle_proposals(state,work['source'],work['plans'],work['rows'],work['confidences'],cycle_id=cycle)
       work['work_plan'].require_published_tickets(state['tickets'])
       plan=state['controller'].build_plan(list(range(n)),active_indices_registered_unique=True,merge_ready_homes=True,max_target_requests=16,verification_budget=96,ready_candidate_counts=dict.fromkeys(range(n),4))
       require(list(plan.target_request_indices)==list(range(n))and len(plan.target_request_indices)==n,'actual original planner chooses exactly published trueN proposals')
      c('original-typed-publication-and-planner',publish)
      c('length-observer-reset',lambda:setattr(e,'_v202_length_observations',[]))
      old=old_control(e,case,bank,work,c)
      manager=qualification_geometry(e);c('bounded-geometry-open',lambda:manager.__enter__())
      try:
       candidate=qualify(e,case,bank,work,stage=c);own=reference(e,case,bank,work,candidate,stage=c)
      finally:c('bounded-geometry-finally-close',lambda:manager.__exit__(None,None,None))
      def cross():
       if e.is_draft:return None
       actual=candidate['physical_tokens'].cpu().tolist();q=case['Q'];a=actual[:q];b=old['real'];positions=[i for i,(x,y)in enumerate(zip(a,b,strict=True))if x!=y]
       return dict(old_real=b,new_real=a,old_all128=old['all128'],new_all80=actual,different_head_positions=positions,head_equal=not positions,
        old_verdict_tokens=old['verdict_tokens'],new_verdict_tokens=own['verdict_tokens'],old_accepted=old['accepted'],new_accepted=own['accepted'],
        verdict_equal=old['verdict_tokens']==own['verdict_tokens'],accepted_equal=old['accepted']==own['accepted'],cause_established=False)
      comparison=c('actual-crossM128-versus80-realQ',cross)
      committed=run_original_tree_cycle(e,native_module,work['source'],work['plans'],work['rows'],candidate['output']if not e.is_draft else None,state,cycle_id=cycle,target_home=cycle%2,stage_all4=c)
      compact=c('independent-selected-reference-compaction',lambda:commit_reference(e,bank,work,own,committed))
      def receipt():
       row=dict(N=n,cycle=cycle,source=work['source'],rows=work['rows'],selected=work['selected'],confidences=work['confidences'],draft=work['draft_proof'],full16_draft=True,selected_normal_C=4*n,
        target=None if e.is_draft else candidate['report'],reference=None if e.is_draft else own['report'],old_control=old,crossM=comparison,commit=committed,independent_commit=compact,
        actual_length_observations=list(e._v202_length_observations),lease=bank.guard())
       save(folder,f'rank{e.rank}-pass{p}-N{n}-cycle{cycle}.json.gz',row);return row
      rows.append(c('retain-current-real-cycle',receipt))
     next_work=generate(e,native_module,bank,case['full_bank_case'],100004+p*100+n*5,lambda name,fn:lane('own-next16/'+name,fn))
     nxt={k:next_work[k]for k in('source','rows','selected','confidences','draft_proof')}
     lane('retain-next',lambda:save(folder,f'rank{e.rank}-pass{p}-N{n}-next.json.gz',nxt))
     def finish_trace():
      change=delta(start,native(e),zero=p>0);after=update_snapshot(e);logical=dict(commits=[fingerprint(v)for v in rows],next={k:nxt[k]for k in('source','rows','selected','confidences')})
      if p:require(logical==expected[n],'two whole protocol/next zero traces reproduce original history')
      else:expected[n]=logical
      return dict(N=n,pass_index=p,Native=change,TTU=dict(before=updates,after=after,delta={k:after[k]-updates[k]for k in updates}),logical=logical,
       crossM_behavior_equal=None if e.is_draft else all(v['crossM']['head_equal']and v['crossM']['verdict_equal']and v['crossM']['accepted_equal']for v in rows),lease=bank.evidence())
     report['traces'].append(lane('complete-real-usage-trace',finish_trace));lane('cache-release',e._release_cache)
  report['inventory']=s('zero-new-keys-identities-flags',lambda:same_inventory(e,before))
  def dispatch():
   after=snapshot_mc2_dispatch_counters();d={k:after[k]-mc2[k]for k in mc2}
   require(all(v>=0 for v in d.values())and d['fallback']==d['exception']==0 and(e.is_draft or d['native_epilogue_success']>0),'existing80/profiled128 actual MC2 route unchanged')
   return dict(before=mc2,after=after,delta=d)
  report['MC2']=s('MC2-actual',dispatch);report['gain_reset']=dict(before=gain,after=s('gain-after',lambda:reset_fixture_gain(policy)))
  report['fresh_boundary']=s('fixture-release',lambda:finish_fixture_state(e,fixture))
  def complete():
   require(e.cache_allocation is None and e.cache_block_tables is None and not hasattr(e,'_v202_usage_geometry')and not e.graph_runner.graph_cache_sealed,'released current usage before issuer')
   report.update(completed=True,cache_released=True,math_independent_qualified=True,boundary_events=events)
   save(folder,f'rank{e.rank}-component.json.gz',report);return report
  s('save-complete-current-usage',complete)
  from audit_usage202 import audit
  checked=s('independent-retained-usage-audit',lambda:audit(folder))
  result=dict(report,audit=checked);e._v202_actual_usage=result;return result
 finally:
  s('fixed-final-release',e._release_cache)
