"""Typed issued work grants; they do not assert online eligibility/window gates."""
from scope179 import require

def work_plan(case):
 from split_budget_v123 import DraftWorkPlan,DraftWorkLimit,TargetAdmissionLimit
 normal=dict(enumerate(case.normal_budgets));eager={i:3 for i in range(16,32)};ids=tuple(range(32))
 return DraftWorkPlan(plan_id=1,target_limit=TargetAdmissionLimit(),work_limit=DraftWorkLimit(),available_selected_candidates=192,
  requested_normal=tuple(normal),admitted_eager=tuple(eager),normal_budgets=normal,eager_budgets=eager,
  progress_gaps={i:0. for i in ids},eager_priorities={i:0. for i in eager},caps={**normal,**eager},needs={i:None for i in ids},
  reserve_requested=0,reserve_used=0)
def refine(case,fixture,mixed):
 from vllm_ascend.spec_decode.pearl.tree_budget import TreeCandidateRequest
 from split_budget_v123 import refine_draft_groups
 from backbone_selection import select
 ids=list(range(32));roles=[False]*16+[True]*16;work=work_plan(case)
 work.require_work(ids,list(case.work_budgets),roles,fixture['plans'])
 requests={i:TreeCandidateRequest(parents=fixture['plans'][i].parent_indices.tolist(),token_ids=mixed['draft_token_ids'][i],
  conditional_confidences=mixed['node_confidences'][i],max_candidates=case.work_budgets[i],minimum_candidates=0 if roles[i] else 1) for i in ids}
 result=refine_draft_groups(requests,work,select);result.require_published(dict(enumerate(case.work_budgets)))
 require(all(result.selected_indices[i]==tuple(range(case.work_budgets[i])) for i in ids),'original backbone refinement actual variable normal4 and child3')
 require(result.normal_candidates==64 and result.future_eager_candidates==48,'two independent selected96 pools; no current target192')
 return dict(normal_candidates=64,future_eager_candidates=48,selected_total=112,actual_work_rows=32,
  budgets=list(case.work_budgets),original_group_calls=[list(v) for v in result.original_group_calls],
  controlled_issued_work_grants=True,online_floor_window_or_need_observed=False)
