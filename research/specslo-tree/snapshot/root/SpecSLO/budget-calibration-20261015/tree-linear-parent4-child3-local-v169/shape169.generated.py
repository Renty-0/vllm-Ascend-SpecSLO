"""One typed HOST draft policy, bounded current target and exactB3 future work."""
from dataclasses import replace
from split_budget_v132 import DraftWorkLimit,shape_draft_work,refine_draft_groups
from normal_admission_v140 import select as select_normal,finish as finish_normal


def filter_eager_parents(indices,controller,payloads,counters):
    selected=[]
    _v169_parent_budgets.clear()
    for budget in range(1,6):
        for suffix in ('seen','excluded_unqualified'):
            counters.setdefault(f'spec_rhythm_eager_parent_B{budget}_{suffix}',0)
    counters['spec_rhythm_eager_parent_filter_calls']=counters.get('spec_rhythm_eager_parent_filter_calls',0)+1
    for index in indices:
        ticket=controller.ready.get(index)
        if ticket is None:raise RuntimeError('V132 eligible eager lost original ready parent')
        parent=payloads.get(ticket.proposal_id)
        if parent is None or 'plan' not in parent:raise RuntimeError('V132 eligible eager lost original parent payload')
        budget=parent['plan'].candidate_budget
        if type(budget) is not int or budget not in range(1,6):raise RuntimeError('V132 actual parent B must be positive qualified tree metadata')
        key='spec_rhythm_eager_parent_B'+str(budget)+'_seen'
        counters[key]=counters.get(key,0)+1
        _v169_parent_budgets[index]=budget
        if budget==3:selected.append(index)
        elif budget==4 and _v169_parent4_enabled:
            from vllm_ascend.spec_decode.pearl.tree import tree_primary_path
            if tuple(tree_primary_path(parent['plan']))==tuple(range(4)):selected.append(index)
            else:
                counters['spec_rhythm_eager_parent_B4_excluded_unqualified']+=1
        else:
            key='spec_rhythm_eager_parent_B'+str(budget)+'_excluded_unqualified'
            counters[key]=counters.get(key,0)+1
            if budget==5:counters['spec_rhythm_eager_parent_B5_deferred_first_candidate']=counters.get('spec_rhythm_eager_parent_B5_deferred_first_candidate',0)+1
    return selected


def shape(policy,shaper,**kwargs):
    # Original gain/need/priority remain in the reused V123 allocator. Both
    # normal and future still have their own ceiling96, global selected192.
    raw_normal=tuple(kwargs['normal_request_indices']);eager=tuple(kwargs['eager_request_indices'])
    admission=select_normal(policy,raw_normal,eager,kwargs['states'],batch_size=kwargs.get('batch_size'))
    normal=admission.admitted
    if len(raw_normal)>32:
        kwargs=dict(kwargs,normal_request_indices=normal)
    limit=DraftWorkLimit(normal_per_request=4 if len(normal)>8 else 5)
    policy.counts[f'draft_normal_N{len(normal)}_shape_calls']+=1
    policy.counts['normal_cap4_shape_calls' if len(normal)>8 else 'normal_cap5_shape_calls']+=1
    policy.counts['normal_cap4_limited_owners']+=sum(min(8,kwargs['states'][i].max_gamma or 8,shaper.max_gamma,5)>4 for i in normal) if len(normal)>8 else 0
    plan=shape_draft_work(policy,shaper,admitted_eager_indices=eager,work_limit=limit,
        normal_gt16_cap4=False,**kwargs)
    for budget in (1,2):policy.counts.setdefault(f'partial_eager_grant_deferred_B{budget}',0)
    shared_steps4=bool(plan.normal_budgets) and max(plan.normal_budgets.values())>3
    shared4_ready=(shared_steps4 and max(plan.normal_budgets.values())==4
        and _v159_capability.allows_shared4(len(plan.normal_budgets),sum(b==3 for b in plan.eager_budgets.values())))
    policy.counts['eager_deferred_shared_steps4_rows']+=len(plan.eager_budgets) if shared_steps4 and not shared4_ready else 0
    policy.counts['eager_deferred_shared_steps4_candidates']+=sum(plan.eager_budgets.values()) if shared_steps4 and not shared4_ready else 0
    kept={i:b for i,b in plan.eager_budgets.items() if b==3 and (not shared_steps4 or shared4_ready) and (_v169_parent_budgets.get(i)==3 or (_v169_parent_budgets.get(i)==4 and shared4_ready and len(plan.normal_budgets)<=16 and len(plan.eager_budgets)<=16 and len(plan.normal_budgets)+len(plan.eager_budgets)<=32))}
    for i,b in plan.eager_budgets.items():
        if b!=3:policy.counts[f'partial_eager_grant_deferred_B{b}']+=1
    reserved_by_owner={i:value for i,value in plan.reserved_by_owner.items() if i in kept}
    reserved=sum(reserved_by_owner.values())
    policy.counts['split_draft_deferred_reserved_candidates']+=plan.reserve_used-reserved
    plan=replace(plan,eager_budgets=kept,eager_priorities={i:plan.eager_priorities[i] for i in kept},reserve_used=reserved,reserved_by_owner=reserved_by_owner)
    if plan.allocated_selected_candidates:
        policy.counts[f'actual_draft_normal_N{len(plan.normal_budgets)}_work_calls']+=1
        policy.counts['actual_draft_nonempty_work_calls']+=1
        policy.counts['actual_draft_work_rows']+=len(plan.normal_budgets)+len(plan.eager_budgets)
        policy.counts['actual_draft_normal_rows']+=len(plan.normal_budgets)
        policy.counts['actual_draft_eager_rows']+=len(plan.eager_budgets)
    policy.counts['split_draft_final_normal_candidates']+=sum(plan.normal_budgets.values())
    policy.counts['split_draft_final_eager_candidates']+=sum(plan.eager_budgets.values())
    policy.counts['split_draft_final_unused_candidates']+=plan.unused_draft_candidates
    finish_normal(policy,admission,plan.normal_budgets)
    return plan
