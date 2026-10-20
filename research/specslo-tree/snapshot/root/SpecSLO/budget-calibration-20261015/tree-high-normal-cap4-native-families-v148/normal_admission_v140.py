"""Overflow-only normal row admission with class priority and defer age.

The Native owner/home/KV mapping is unchanged. Age counts eligible normal
attempts without a positive grant, not time or a promise of next-cycle service.
"""
from dataclasses import dataclass
import copy
from budget_policy import PRIORITY

STATE='_v140_normal_admission'
RULE='V140.normal-class-priority-defer-age-owner-order.v1'


def reset(policy, generation):
    if type(generation) is not int or generation <= 0:
        raise ValueError('V140 normal admission requires a positive generation')
    setattr(policy,STATE,dict(rule=RULE,generation=generation,ages={},call_count=0,calls=[],
        raw_N_histogram=[0]*65,admitted_N_histogram=[0]*33,
        positive_grant_N_histogram=[0]*33,overflow_calls=0))


def _state(policy):
    value=getattr(policy,STATE,None)
    if not isinstance(value,dict) or value.get('rule')!=RULE:
        raise RuntimeError('V140 reset normal admission at the phase boundary before use')
    return value


@dataclass(frozen=True)
class Selection:
    call: int
    raw: tuple
    eager: tuple
    admitted: tuple
    cap_deferred: tuple
    classes: tuple
    ages_before: tuple
    states: object
    batch_size: object


def select(policy,raw_normal,eager,states,*,batch_size):
    state=_state(policy)
    raw,eager=tuple(raw_normal),tuple(eager)
    combined=raw+eager
    if ((batch_size is not None and (type(batch_size) is not int or not 0<=batch_size<=64))
        or len(raw)>64 or len(eager)>32 or len(combined)>64
        or (batch_size is not None and len(combined)>batch_size)
        or len(combined)!=len(set(combined))
        or any(type(i) is not int or i<0 or i not in states for i in combined)):
        raise ValueError('V140 raw owners must fit contract64 and any explicit active bound')
    if any(states[i].slo_class not in PRIORITY for i in raw):
        raise ValueError('V140 normal admission received an unknown SLO class')
    if len(raw)<=32:
        admitted=raw;classes=ages=();cap_deferred=()
    else:
        classes=tuple(states[i].slo_class for i in raw)
        ages=tuple(state['ages'].get(i,0) for i in raw)
        if any(type(age) is not int or age<0 for age in ages):
            raise ValueError('V140 normal defer age is invalid')
        ranked=sorted(range(len(raw)),key=lambda j:(-PRIORITY[classes[j]],-ages[j],raw[j]))
        chosen={raw[j] for j in ranked[:32]}
        admitted=tuple(i for i in raw if i in chosen)
        cap_deferred=tuple(i for i in raw if i not in chosen)
    return Selection(state['call_count'],raw,eager,admitted,cap_deferred,classes,ages,states,batch_size)


def finish(policy,selected,positive_normal_grants):
    state=_state(policy)
    grants=dict(positive_normal_grants)
    if (selected.call!=state['call_count'] or not set(grants)<=set(selected.admitted)
        or any(type(v) is not int or v<=0 for v in grants.values())):
        raise RuntimeError('V140 actual normal grants escaped the admitted owner set')
    interesting=len(selected.raw)>32 or len(grants)!=len(selected.raw)
    if interesting:
        ages_before=selected.ages_before or tuple(state['ages'].get(i,0) for i in selected.raw)
        classes=selected.classes or tuple(selected.states[i].slo_class for i in selected.raw)
    for i in selected.raw:
        if i in grants:state['ages'].pop(i,None)
        else:state['ages'][i]=state['ages'].get(i,0)+1
    state['call_count']+=1
    state['raw_N_histogram'][len(selected.raw)]+=1
    state['admitted_N_histogram'][len(selected.admitted)]+=1
    state['positive_grant_N_histogram'][len(grants)]+=1
    state['overflow_calls']+=int(len(selected.raw)>32)
    if not interesting:return None
    actual=tuple(i for i in selected.raw if i in grants)
    deferred=tuple(i for i in selected.raw if i not in grants)
    row=dict(call=selected.call,raw_normal=list(selected.raw),post_window_eager=list(selected.eager),
        admitted_normal=list(selected.admitted),positive_normal_grants=[dict(owner=i,B=grants[i]) for i in actual],
        cap_deferred_normal=list(selected.cap_deferred),
        zero_grant_deferred_normal=[i for i in selected.admitted if i not in grants],
        all_deferred_normal=list(deferred),classes=list(classes),
        ages_before=list(ages_before),ages_after=[state['ages'].get(i,0) for i in selected.raw],
        raw_N=len(selected.raw),admitted_N=len(selected.admitted),positive_grant_N=len(actual),
        eager_N=len(selected.eager),overflow=len(selected.raw)>32,
        actual_work_normal_row_limit=32,actual_work_combined_row_limit=64,
        explicit_active_batch_size=selected.batch_size)
    state['calls'].append(row)
    return row


def snapshot(policy):
    state=getattr(policy,STATE,None)
    if state is None:return dict(available=False,rule=RULE)
    state=_state(policy)
    return dict(available=True,rule=RULE,generation=state['generation'],call_count=state['call_count'],
        ages=[dict(owner=i,age=age) for i,age in sorted(state['ages'].items())],
        calls=copy.deepcopy(state['calls']),raw_N_histogram=list(state['raw_N_histogram']),
        admitted_N_histogram=list(state['admitted_N_histogram']),
        positive_grant_N_histogram=list(state['positive_grant_N_histogram']),
        overflow_calls=state['overflow_calls'],age_unit='eligible normal attempts without a positive grant',
        age_owner_scope='historical deferred owners in this generation; not an active-owner snapshot',
        complete_owner_records_scope='overflow or actual deferred owners; successful at-most32 calls retain counters only',
        next_cycle_service_guaranteed=False,Native_home_or_KV_mapping_changed=False)
