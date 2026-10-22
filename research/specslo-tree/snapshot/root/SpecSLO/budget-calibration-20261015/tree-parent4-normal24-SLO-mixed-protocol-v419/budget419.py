"""Private E4 typed classes; original production cap3 objects stay untouched."""
import types,sys
from base199 import BASE,checked,require
_original=None
_namespace=None
def classes():
 global _namespace,_original
 if _namespace is None:
  import split_budget_v132 as original
  _original=original;before=original.DraftWorkLimit
  source=checked(BASE/'split_budget_v132.py').read_text()
  edits=(('eager_per_request: int = 3','eager_per_request: int = 4',1),('eager_rows=32,normal_per_request=5,eager_per_request=3','eager_rows=32,normal_per_request=5,eager_per_request=4',1),('Eager per-request cap3 was bypassed','Private diagnostic eager per-request cap4 was bypassed',1))
  for old,new,count in edits:
   require(source.count(old)==count,'exact private typed E4 anchor '+old);source=source.replace(old,new)
  ns=dict(vars(original),__name__='_v419_private_typed_budget',__file__=str(__import__('pathlib').Path(__file__)))
  module=types.ModuleType(ns['__name__']);module.__dict__.update(ns);sys.modules[module.__name__]=module
  try:exec(compile(source,ns['__file__']+'::private-E4','exec'),module.__dict__)
  except BaseException:sys.modules.pop(module.__name__,None);raise
  require(original.DraftWorkLimit is before and before().eager_per_request==3,'production cap3 class identity and fields unchanged')
  _namespace=module
 return _namespace
def work_plan(case):
 ns=classes();normal=dict.fromkeys(range(case.normal),4);eager=dict.fromkeys(case.eager_ids,4);ids=tuple(normal)+tuple(eager)
 return ns.DraftWorkPlan(plan_id=1,target_limit=ns.TargetAdmissionLimit(),work_limit=ns.DraftWorkLimit(eager_per_request=4),available_selected_candidates=192,requested_normal=tuple(normal),admitted_eager=tuple(eager),normal_budgets=normal,eager_budgets=eager,progress_gaps=dict.fromkeys(ids,0.),eager_priorities=dict.fromkeys(eager,0.),caps={**normal,**eager},needs=dict.fromkeys(ids,None),reserve_requested=0,reserve_used=0,raw_eager_budgets=eager)
def refine(case,fixture,mixed):
 from vllm_ascend.spec_decode.pearl.tree_budget import TreeCandidateRequest
 from backbone_selection import select
 ns=classes();work=work_plan(case);ids=fixture['ids'];roles=[i in fixture['sources']for i in ids]
 work.require_work(ids,[4]*len(ids),roles,fixture['plans'])
 requests={i:TreeCandidateRequest(parents=p.parent_indices.tolist(),token_ids=mixed['draft_token_ids'][j],conditional_confidences=mixed['node_confidences'][j],max_candidates=4,minimum_candidates=4)for j,(i,p)in enumerate(zip(ids,fixture['plans'],strict=True))}
 result=ns.refine_draft_groups(requests,work,select);result.require_published(dict.fromkeys(ids,4))
 require(all(result.selected_indices[i]==(0,1,2,3)for i in ids),'real bounded selected linear B4')
 return dict(selected_total=4*case.normal+4*case.eager,normal_candidates=4*case.normal,future_eager_candidates=4*case.eager,paid_controlled_grants=True,online_eligibility_claimed=False)
def clear():
 global _namespace,_original
 if _namespace is not None:
  require(_original.DraftWorkLimit().eager_per_request==3,'original typed cap3 remains before private namespace clear')
  require(sys.modules.get(_namespace.__name__)is _namespace,'private typed module identity');sys.modules.pop(_namespace.__name__);_namespace=None;_original=None
