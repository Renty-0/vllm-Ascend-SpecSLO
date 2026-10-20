"""Independent HOST algorithm candidate. Importing this module installs nothing.

Budget units are retained candidates. Actual generated tensor nodes/model
queries remain unobserved here; the existing8-unit row window is unchanged.
"""
from dataclasses import dataclass, field
from heapq import heappop, heappush
import math
from types import MappingProxyType

PRIORITY = {'tight':3, 'normal':2, 'loose':1}


def nonnegative(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name+' must be a nonnegative HOST integer')
    return value


def freeze_map(value):
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class TargetAdmissionLimit:
    candidates: int = 96
    queries: int = 128
    rows: int = 32

    def __post_init__(self):
        nonnegative(self.candidates,'target candidates')
        if self.candidates > 96 or self.queries != 128 or self.rows != 32:
            raise ValueError('V123 target remains C<=96/Q128/N32')

    def check_actual_call(self, budgets):
        values = tuple(budgets)
        if any(type(b) is not int or b <= 0 for b in values):
            raise ValueError('Actual target budgets must be positive HOST integers')
        if (len(values)>self.rows or sum(values)>self.candidates
                or sum(b+1 for b in values)>self.queries):
            raise ValueError('Actual current target exceeds unchanged admission')
        return sum(b+1 for b in values)


@dataclass(frozen=True)
class DraftWorkLimit:
    normal_candidates: int = 96
    future_eager_candidates: int = 96
    global_selected_candidates: int = 192
    work_rows: int = 64
    normal_rows: int = 32
    eager_rows: int = 32
    normal_per_request: int = 5
    eager_per_request: int = 3

    def __post_init__(self):
        limits = dict(normal_candidates=96,future_eager_candidates=96,
            global_selected_candidates=192,work_rows=64,normal_rows=32,
            eager_rows=32,normal_per_request=5,eager_per_request=3)
        for key, maximum in limits.items():
            value = nonnegative(getattr(self,key),key)
            if value>maximum:
                raise ValueError(key+' exceeds the explicit V123 candidate scope')


@dataclass(frozen=True)
class DraftWorkPlan:
    plan_id: int
    target_limit: TargetAdmissionLimit
    work_limit: DraftWorkLimit
    available_selected_candidates: int
    requested_normal: tuple
    admitted_eager: tuple
    normal_budgets: object
    eager_budgets: object
    progress_gaps: object
    eager_priorities: object
    caps: object
    needs: object
    reserve_requested: int
    reserve_used: int
    deferred_normal_request_indices: tuple = ()
    reserved_by_owner: object = field(default_factory=dict)
    raw_reserved_by_owner: object = field(default_factory=dict)
    raw_eager_budgets: object = field(default_factory=dict)

    def __post_init__(self):
        for key in ('plan_id','available_selected_candidates','reserve_requested','reserve_used'):
            nonnegative(getattr(self,key),key)
        if self.available_selected_candidates>self.work_limit.global_selected_candidates:
            raise ValueError('Available draft candidates exceed explicit global limit')
        normal,eager = dict(self.normal_budgets),dict(self.eager_budgets)
        requested = self.requested_normal+self.admitted_eager
        if len(set(requested))!=len(requested) or any(type(i) is not int or i<0 for i in requested):
            raise ValueError('Normal/eager owner sets must be unique and disjoint')
        if not set(normal)<=set(self.requested_normal) or not set(eager)<=set(self.admitted_eager):
            raise ValueError('Grant escaped its original role/window admission')
        if (len(self.requested_normal)>self.work_limit.normal_rows
            or len(self.admitted_eager)>self.work_limit.eager_rows
            or len(requested)>self.work_limit.work_rows):
            raise ValueError('Original per-role/combined row capacity exceeded')
        if any(type(b) is not int or b<=0 for b in (*normal.values(),*eager.values())):
            raise ValueError('Allocated candidates must be positive HOST integers')
        if (sum(normal.values())>min(self.work_limit.normal_candidates,self.target_limit.candidates)
            or sum(eager.values())>self.work_limit.future_eager_candidates
            or sum(normal.values())+sum(eager.values())>self.available_selected_candidates):
            raise ValueError('A role or global draft selected-candidate limit was exceeded')
        if set(self.caps)!=set(requested) or set(self.needs)!=set(requested):
            raise ValueError('Actual cap/need owner sets differ from original admission')
        if set(self.progress_gaps)!=set(requested) or set(self.eager_priorities)!=set(eager):
            raise ValueError('Progress/priority metadata is incomplete')
        if any(type(c) is not int or c<0 for c in self.caps.values()):
            raise ValueError('Actual per-owner caps must be nonnegative HOST integers')
        if any(n is not None and (type(n) is not int or not 1<=n<=self.caps[i])
               for i,n in self.needs.items()):
            raise ValueError('Actual per-owner need is inconsistent with its cap')
        if any(b>self.caps[i] for i,b in (*normal.items(),*eager.items())):
            raise ValueError('Grant exceeds its actual per-owner cap')
        if any(b>self.work_limit.eager_per_request for b in eager.values()):
            raise ValueError('Eager per-request cap3 was bypassed')
        if any(b>self.work_limit.normal_per_request for b in normal.values()):
            raise ValueError('Normal per-request candidate cap was bypassed')
        deferred = tuple(i for i in self.requested_normal if i not in normal)
        if deferred!=self.deferred_normal_request_indices:
            raise ValueError('Deferred normal accounting differs from actual grants')
        if self.reserve_used>self.reserve_requested or self.reserve_used>sum(eager.values()):
            raise ValueError('Eager reserve accounting is inconsistent')
        reserved=dict(self.reserved_by_owner);raw_reserved=dict(self.raw_reserved_by_owner);raw_eager=dict(self.raw_eager_budgets)
        if (not set(reserved)<=set(eager) or not set(raw_reserved)<=set(raw_eager) or not set(raw_eager)<=set(self.admitted_eager)
            or reserved!={i:v for i,v in raw_reserved.items() if i in eager}
            or sum(reserved.values())!=self.reserve_used or sum(raw_reserved.values())>self.reserve_requested
            or any(type(v) is not int or v<=0 for values in (reserved,raw_reserved,raw_eager) for v in values.values())
            or any(v>eager[i] or v!=raw_reserved.get(i) for i,v in reserved.items())
            or any(v>raw_eager[i] for i,v in raw_reserved.items())
            or any(eager[i]!=raw_eager.get(i) for i in eager)):
            raise ValueError('V132 actual reserved-owner/raw/final selected-budget conservation differs')
        for key in ('normal_budgets','eager_budgets','progress_gaps','eager_priorities','caps','needs',
                    'reserved_by_owner','raw_reserved_by_owner','raw_eager_budgets'):
            object.__setattr__(self,key,freeze_map(getattr(self,key)))

    @property
    def allocated_selected_candidates(self):
        return sum(self.normal_budgets.values())+sum(self.eager_budgets.values())

    @property
    def unused_draft_candidates(self):
        return self.available_selected_candidates-self.allocated_selected_candidates

    def require_work(self, indices, budgets, eager_roles, plans):
        """Grant -> actual plan metadata. No tensor/node-count assertion."""
        expected = [*self.normal_budgets,*self.eager_budgets]
        values = [*self.normal_budgets.values(),*self.eager_budgets.values()]
        roles = [False]*len(self.normal_budgets)+[True]*len(self.eager_budgets)
        if (any(type(role) is not bool for role in eager_roles)
            or any(type(b) is not int for b in budgets)
            or list(indices)!=expected or list(budgets)!=values or list(eager_roles)!=roles
            or len(plans)!=len(expected)
            or any(type(p.candidate_budget) is not int or p.candidate_budget!=b
                   for p,b in zip(plans,values))):
            raise RuntimeError('Actual work plans differ from typed shape grants')
        return True

    def require_published_tickets(self, tickets):
        """All ranks can check their actual HOST tickets after row selection."""
        pairs=[(ticket.request_index,ticket.gamma) for ticket in tickets]
        expected={**self.normal_budgets,**self.eager_budgets}
        if (any(type(i) is not int or type(b) is not int for i,b in pairs)
            or len(pairs)!=len(dict(pairs)) or dict(pairs)!=expected):
            raise RuntimeError('Actual published ticket budgets differ from shaped/selected work')
        return True


def shape_draft_work(policy, shaper, *, plan_id, normal_request_indices,
                     eager_request_indices, states, projected_wait_ms, context_len,
                     draft_token_budget=None, batch_size=None, eager_token_cap=None,
                     eager_reserve_tokens=0, verification_roof=None,
                     admitted_eager_indices, work_limit=DraftWorkLimit(),
                     normal_gt16_cap4=False):
    """Retain original gain/need/priority/whole-jump rules, split the ceilings.

    Eager reserve is an upper bound on complete minimum proposals, matching
    the original Native meaning; it is not a quota or a claim to close a_need.
    Every eager input must already have passed the unchanged runtime gates
    and actual8-unit row window. This function does not infer those gates.
    """
    normal,eager = tuple(normal_request_indices),tuple(eager_request_indices)
    if type(normal_gt16_cap4) is not bool:
        raise ValueError('Family cap selection must be an explicit boolean')
    admitted_eager_indices=tuple(admitted_eager_indices)
    if eager!=admitted_eager_indices:
        raise ValueError('Eager inputs differ from actual post-window admission')
    indices=normal+eager
    if len(set(indices))!=len(indices) or any(type(i) is not int or i<0 for i in indices):
        raise ValueError('Normal/eager requests overlap or contain invalid owners')
    if any(i not in states for i in indices):
        raise ValueError('A draft request has no actual runtime state')
    if (len(normal)>work_limit.normal_rows or len(eager)>work_limit.eager_rows
        or len(indices)>work_limit.work_rows):
        raise ValueError('Draft row bounds exceeded before allocation')
    nonnegative(context_len,'context_len')
    if batch_size is not None and (nonnegative(batch_size,'batch_size')>64 or len(indices)>batch_size):
        raise ValueError('Actual work exceeds the explicit active batch bound')
    if not math.isfinite(projected_wait_ms):
        raise ValueError('Wait must be finite')
    reserve=nonnegative(eager_reserve_tokens,'eager_reserve_tokens')
    explicit_eager_cap=work_limit.eager_per_request if eager_token_cap is None else nonnegative(eager_token_cap,'eager_token_cap')
    roof=policy.candidate_roof if verification_roof is None else nonnegative(verification_roof,'verification_roof')
    target=TargetAdmissionLimit(candidates=roof,queries=policy.target_queries,rows=policy.max_target_requests)
    available=work_limit.global_selected_candidates if draft_token_budget is None else min(
        work_limit.global_selected_candidates,nonnegative(draft_token_budget,'draft_token_budget'))
    left=available
    role_left={False:min(work_limit.normal_candidates,roof),True:work_limit.future_eager_candidates}
    grants=dict.fromkeys(indices,0);caps={};needs={};curves={};gaps={}
    eager_set=set(eager)
    for i in indices:
        state=states[i]
        cap=min(8,state.max_gamma or 8,shaper.max_gamma,roof)
        if i in eager_set:
            cap=min(cap,work_limit.eager_per_request,explicit_eager_cap)
        else:
            cap=min(cap,work_limit.normal_per_request,
                    4 if normal_gt16_cap4 and len(normal)>16 else 8)
        caps[i]=cap
        curves[i]=[0.]+[policy.gain(state,n) for n in range(1,cap+1)]
        elapsed=max(0.,state.decode_elapsed_ms-(state.decode_start_elapsed_ms or 0.))
        delivered=max(0,state.delivered_tokens-1)
        gap=max(0.,(elapsed+max(0.,projected_wait_ms))/state.slo_tpot_ms-delivered)
        gaps[i]=state.projected_progress_gap(projected_wait_ms)
        needs[i]=next((n for n in range(1,cap+1) if curves[i][n]>=gap),None)
        if needs[i] is None:
            policy.counts['current_forecast_unfit']+=1
    order=sorted(indices,key=lambda i:(-PRIORITY[states[i].slo_class],i in eager_set,i))
    def charge(i,cost):
        nonlocal left
        role=i in eager_set
        if cost<0 or cost>left or cost>role_left[role]:
            return False
        grants[i]+=cost;left-=cost;role_left[role]-=cost
        return True
    reserved=0
    reserved_by_owner={}
    # Inputs have already passed original positive-gap/acceptance/urgency gates;
    # do not recompute missing gate values or reinterpret a reserve as admission.
    for i in sorted(eager,key=lambda i:(gaps[i]*states[i].expected_acceptance_benefit,
                     states[i].urgency(projected_wait_ms),-i),reverse=True) if reserve else ():
        minimum=min(shaper.min_gamma,caps[i])
        if minimum>0 and gaps[i]>0 and reserved+minimum<=reserve and charge(i,minimum):
            reserved+=minimum
            reserved_by_owner[i]=minimum
    for i in order:
        need=needs[i]
        if need is not None and need>grants[i]:
            charge(i,need-grants[i])
    for i in order:
        if not grants[i] and i not in eager_set and caps[i]>0:
            charge(i,1)
    heap=[]
    def push_next(i):
        n=grants[i]
        if 0<n<caps[i]:
            next_n=next((m for m in range(n+1,caps[i]+1) if curves[i][m]>curves[i][n]),None)
            if next_n is not None:
                cost=next_n-n
                score=PRIORITY[states[i].slo_class]*(curves[i][next_n]-curves[i][n])/cost
                heappush(heap,(-score,i,next_n))
    for i in order:push_next(i)
    while left and heap:
        _,i,next_n=heappop(heap)
        if charge(i,next_n-grants[i]):push_next(i)
    normal_grants={i:grants[i] for i in normal if grants[i]}
    eager_grants={i:grants[i] for i in eager if grants[i]}
    plan=DraftWorkPlan(plan_id=plan_id,target_limit=target,work_limit=work_limit,
        available_selected_candidates=available,requested_normal=normal,admitted_eager=eager,
        normal_budgets=normal_grants,eager_budgets=eager_grants,progress_gaps=gaps,
        eager_priorities={i:float(gaps[i]) for i in eager_grants},caps=caps,needs=needs,
        reserve_requested=reserve,reserve_used=reserved,reserved_by_owner=reserved_by_owner,
        raw_reserved_by_owner=reserved_by_owner,raw_eager_budgets=eager_grants,
        deferred_normal_request_indices=tuple(i for i in normal if i not in normal_grants))
    policy.counts['allocation_calls']+=1
    for n in grants.values():policy.counts[f'grant_B{n}']+=1
    policy.counts['split_draft_reserved_eager_candidates']+=reserved
    return plan


@dataclass(frozen=True)
class DraftCandidateSelection:
    selected_indices: object
    expected_progress: object
    stage1_candidate_counts: object
    stage2_candidate_counts: object
    normal_candidates: int
    future_eager_candidates: int
    global_draft_candidate_limit: int
    original_group_calls: tuple = field(default=())

    def require_published(self, owner_budgets):
        expected={i:len(nodes) for i,nodes in self.selected_indices.items()}
        if dict(owner_budgets)!=expected:
            raise RuntimeError('Published owner budgets differ from actual refined nodes')
        return True


def refine_draft_groups(requests, plan, original_select):
    """Call the ORIGINAL backbone function twice, with separate role limits."""
    expected={**plan.normal_budgets,**plan.eager_budgets}
    if set(requests)!=set(expected):
        raise RuntimeError('Refinement owners differ from shaped work')
    if any(type(r.max_candidates) is not int or r.max_candidates!=expected[i] for i,r in requests.items()):
        raise RuntimeError('Actual candidate metadata differs from shape grants')
    fields={k:{} for k in ('selected_indices','expected_progress','stage1_candidate_counts','stage2_candidate_counts')}
    calls=[];counts=[]
    for name,owners,limit in (
        ('normal',plan.normal_budgets,min(plan.work_limit.normal_candidates,plan.target_limit.candidates)),
        ('future_eager',plan.eager_budgets,plan.work_limit.future_eager_candidates)):
        if owners:
            group={i:requests[i] for i in owners}
            result=original_select(group,limit)
            if (set(result.selected_indices)!=set(owners) or result.verification_budget!=limit
                or result.total_candidates!=sum(owners.values())
                or any(len(result.selected_indices[i])!=owners[i] for i in owners)):
                raise RuntimeError('Original group selector did not preserve every admitted B')
            for key in fields:fields[key].update(getattr(result,key))
            calls.append((name,tuple(owners),limit,result.total_candidates))
            counts.append(result.total_candidates)
        else:counts.append(0)
    if sum(counts)!=plan.allocated_selected_candidates:
        raise RuntimeError('Combined refinement candidate accounting differs')
    return DraftCandidateSelection(**{k:freeze_map(v) for k,v in fields.items()},
        normal_candidates=counts[0],future_eager_candidates=counts[1],
        global_draft_candidate_limit=plan.available_selected_candidates,
        original_group_calls=tuple(calls))
