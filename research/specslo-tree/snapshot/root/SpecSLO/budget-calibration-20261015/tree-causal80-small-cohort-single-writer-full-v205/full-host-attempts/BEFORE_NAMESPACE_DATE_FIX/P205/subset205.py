"""Original controller chooses a legal post-root home; wide prefill is not N32 target."""
from base205 import require
from dual_bank_v146 import prefix_evidence

class View:
 def source(self):
  return dict(ids=list(self.ids),contexts=[dict(owner=i,reference_owner=r,request_id=self.bank.rows[i]['request_id'],token_ids=list(v),prefix=len(v)-1,initial_prefix=len(self.bank.prompts[i]),output_offset=0)
   for i,r,v in zip(self.ids,self.reference_ids,self.contexts,strict=True)])

def prepare(engine,native,bank):
 controller=native.SpecRhythmPipelineController(bank.runtime_states)
 active=bank.publication_active
 kwargs=dict(verification_budget=96,priority=True,projected_wait_ms=1e-6,priority_burst=2,target_slo_first=False,merge_ready_homes=True,max_target_requests=16)
 plan=controller.build_plan(active,**kwargs);ids=list(plan.normal_draft_request_indices)
 require(len(ids)==(1 if bank.n==2 else 16)and plan.target_request_indices==()and all(bank.runtime_states[i].home_batch_id==plan.draft_home_batch_id for i in ids),'original natural home selection after all roots')
 result=View();result.bank=bank;result.engine=engine;result.n=len(ids);result.ids=ids;result.reference_ids=[i+bank.n for i in ids]
 result.contexts=[bank.contexts[i]for i in ids];result.reference_contexts=[bank.reference_contexts[i]for i in ids]
 result.reference_ledger={i:bank.reference_ledger[i]for i in result.reference_ids}
 result.controller=controller;result.plan=plan;result.kwargs=kwargs;return result

def guard_rest(engine,bank,selected):
 selected_set=set(selected.ids);histories={i:bank.contexts[i][:-1] for i in range(bank.n)if i not in selected_set}
 histories.update({i+bank.n:bank.reference_contexts[i][:-1]for i in range(bank.n)if i not in selected_set})
 return prefix_evidence(engine,histories)

def finish(engine,native,bank,selected,work,protected):
 require(guard_rest(engine,bank,selected)==protected,'all untouched owners computed prompt KV protected; root not yet processed after actual own next')
 tickets=[selected.controller.new_ticket(i,gamma=4,eager=False)for i in selected.ids]
 selected.controller.publish(tickets)
 target=selected.controller.build_plan(bank.publication_active,**selected.kwargs)
 require(tuple(target.target_request_indices)==tuple(selected.ids)and len(target.target_request_indices)<=16
  and sum(target.target_candidate_budgets.values())<=96,'original typed ready ticket packing never enlarges target beyond16/C96')
 return dict(active_owners=bank.n,planner_normal=list(selected.plan.normal_draft_request_indices),planner_home=selected.plan.draft_home_batch_id,
  actual_next_owners=selected.ids,actual_reference_owners=selected.reference_ids,actual_next_B=[4]*selected.n,
  target_ready_admission=list(target.target_request_indices),target_candidates=sum(target.target_candidate_budgets.values()),
  target_max16=True,actual_target_submit_in_this_extra_component=False,unchanged_current_base_actual_target_commit_proof_required=True,
  untouched_owner_KV_protected=True,N32_target_scope_added=False)
