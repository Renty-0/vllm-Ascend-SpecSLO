"""HOST proposal for exactly N24/B4 normal plus E4/B4 future work.

Use the original gain/need/priority allocator with the private typed E4
classes. A read-only probe operates on a policy copy. Only a complete supported
shape gets proposed; all other inputs use the original normal24 callback and
its original B3-parent filter. This module installs no model/runtime callback.
Fresh same-engine protocol proof and full serving qualification are required.
"""
import copy
from dataclasses import dataclass

import admission405
import budget419


@dataclass(frozen=True)
class Proposal:
    plan: object
    mode: str
    future4_owners: tuple=()
    HOST_only: bool=True
    production_permission: bool=False


def propose(policy,shaper,original_shape,kwargs,*,parent_budgets):
    eager=tuple(kwargs['eager_request_indices'])
    normal=tuple(kwargs['normal_request_indices'])
    if (set(parent_budgets)!=set(eager)
            or any(type(b) is not int or not 1<=b<=5 for b in parent_budgets.values())):
        raise ValueError('HOST420 exact complete post-window parent metadata required')
    selected=admission405.select(policy,normal,eager,kwargs['states'],batch_size=kwargs.get('batch_size'))
    admitted=selected.admitted
    parents4=tuple(i for i in eager if parent_budgets[i]==4)

    def fallback():
        work=dict(kwargs,eager_request_indices=tuple(i for i in eager if parent_budgets[i]==3))
        return Proposal(original_shape(policy,shaper,**work),'original_normal24_and_B3_parent_filter')

    if (len(admitted)!=24 or len(parents4)<4 or kwargs.get('verification_roof')!=96
            or kwargs.get('draft_token_budget',192)<112
            or any(min(shaper.max_gamma,kwargs['states'][i].max_gamma or shaper.max_gamma)<4
                   for i in (*admitted,*parents4[:4]))):
        return fallback()
    future=parents4[:4] # The caller supplies the unchanged ranked post-window order.
    work=dict(kwargs,normal_request_indices=admitted,eager_request_indices=future,eager_token_cap=4)
    namespace=budget419.classes()
    limit=namespace.DraftWorkLimit(normal_rows=24,eager_rows=4,work_rows=28,
        normal_per_request=4,eager_per_request=4)
    def allocate(actual_policy):
        return namespace.shape_draft_work(actual_policy,shaper,
            admitted_eager_indices=future,work_limit=limit,normal_gt16_cap4=False,**work)
    probe_policy=copy.deepcopy(policy)
    probe=allocate(probe_policy)
    if (tuple(probe.normal_budgets)!=admitted or tuple(probe.eager_budgets)!=future
            or any(b!=4 for b in (*probe.normal_budgets.values(),*probe.eager_budgets.values()))):
        return fallback()
    actual=allocate(policy)
    if actual!=probe:raise RuntimeError('HOST420 actual original-g allocation differs from its read-only probe')
    admission405.finish(policy,selected,actual.normal_budgets)
    policy.counts['HOST420_proposed_exact_N24_E4_calls']+=1
    policy.counts['HOST420_proposed_exact_future_candidates']+=16
    return Proposal(actual,'proposed_exact_normal24_B4_future4_B4',future)
