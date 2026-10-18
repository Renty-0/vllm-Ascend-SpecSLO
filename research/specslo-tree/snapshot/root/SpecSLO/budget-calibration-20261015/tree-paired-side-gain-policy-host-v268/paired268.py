"""Same-prefix side-leaf gain observations; not installed in production.

Only the instance predictor changes. Model, verifier, original B<=4 prediction
and original actual-gain observations keep their implementations.
"""
import types

CLASSES=('tight','normal','loose')


def install(policy):
    if hasattr(policy, '_ROOT268_pair_state'):
        return policy._ROOT268_pair_state
    from budget_policy import TreeBudgetPolicy
    original=policy.gain
    if original.__func__ is not TreeBudgetPolicy.gain:
        raise RuntimeError('ROOT268 original bound per-instance gain required')
    state=dict(by_class={c:dict(samples=0,side_hits=0,sum_B4=0,sum_B5=0,
                              terminal_skipped=0) for c in CLASSES},
               by_class_anchor={},prior_samples=12,
               original_gain=original,generation=None,by_generation={})
    policy._ROOT268_pair_state=state
    def gain(this,request,n):
        if this is not policy:
            raise RuntimeError('ROOT268 predictor instance changed')
        if n!=5:
            return original(request,n)
        row=state['by_class'][request.slo_class]
        prior=max(0.,min(1.,float(this.base[request.slo_class][5]-this.base[request.slo_class][4])))
        bonus=(state['prior_samples']*prior+row['side_hits'])/(state['prior_samples']+row['samples'])
        remaining=max(0,(request.output_token_budget or 256)-request.delivered_tokens)
        return min(remaining,max(1.,original(request,4)+bonus))
    policy.gain=types.MethodType(gain,policy)
    policy.counts['ROOT268_paired_predictor_bindings']+=1
    return state


def begin_generation(policy,generation):
    """Call after original fresh_decode clears observations and advances gen.

    Use the actual control.engine generation, never a counter inferred from
    cached fixtures. Original observations and native fields are not edited.
    """
    state=install(policy)
    if type(generation)is not int or generation<=0:
        raise ValueError('ROOT268 actual positive generation required')
    if state['generation']==generation:
        return state
    if state['generation']is not None and generation<state['generation']:
        raise ValueError('ROOT268 stale generation cannot reuse paired state')
    row=dict(by_class={c:dict(samples=0,side_hits=0,sum_B4=0,sum_B5=0,
                             terminal_skipped=0)for c in CLASSES},by_class_anchor={})
    state['by_generation'][str(generation)]=row
    state['generation']=generation
    state['by_class']=row['by_class'];state['by_class_anchor']=row['by_class_anchor']
    return state


def observe(policy,request,*,parents,accepted,outputs,remaining_before):
    """Observe an actual B5 verdict before commit or terminal truncation.

    Caller must supply the actual same-proposal parent vector, verdict path,
    output row and original remaining output budget. No fabricated verdicts
    may enter real measurements; HOST fixtures are separately labeled.
    """
    state=install(policy)
    if type(state['generation'])is not int:
        raise RuntimeError('ROOT268 begin actual fresh generation before observing')
    if len(parents)!=5 or list(parents[:4])!=[-1,0,1,2]or parents[4]not in(-1,0,1,2):
        raise ValueError('ROOT268 exact qualified four-spine plus side leaf')
    if len(accepted)!=4 or len(outputs)!=5:
        raise ValueError('ROOT268 original depth4 padded verdict shape')
    path=[int(n) for n in accepted if n>=0]
    tokens=[int(n) for n in outputs if n>=0]
    if list(accepted)!=path+[-1]*(4-len(path)) or list(outputs)!=tokens+[-1]*(5-len(tokens)):
        raise ValueError('ROOT268 original trailing placeholder convention')
    prior=-1
    for node in path:
        if node>=5 or parents[node]!=prior:
            raise ValueError('ROOT268 actual ancestor-closed verdict required')
        prior=node
    if len(tokens)!=len(path)+1 or len(path)>4:
        raise ValueError('ROOT268 original one-frontier verdict convention')
    if type(remaining_before)is not int or remaining_before<=0:
        raise ValueError('ROOT268 actual positive remaining horizon')
    row=state['by_class'][request.slo_class]
    if remaining_before<5:
        row['terminal_skipped']+=1
        return None
    side=int(4 in path);b5=len(tokens);b4=b5-side
    key=f'{request.slo_class}:{parents[4]}'
    anchor=state['by_class_anchor'].setdefault(key,dict(samples=0,side_hits=0,sum_B4=0,sum_B5=0))
    for record in (row,anchor):
        record['samples']+=1;record['side_hits']+=side
        record['sum_B4']+=b4;record['sum_B5']+=b5
        if record['sum_B5']-record['sum_B4']!=record['side_hits']:
            raise RuntimeError('ROOT268 complete same-prefix observation accounting')
    rounds,total=policy.observed.get((request.slo_class,4),(0,0))
    policy.observed[request.slo_class,4]=(rounds+1,total+b4)
    policy.counts['ROOT268_actual_B5_paired_observations']+=1
    policy.counts['ROOT268_shadow_B4_gain_updates']+=1
    return dict(B5=b5,same_prefix_B4=b4,side_bonus=side)


def snapshot(policy):
    state=policy._ROOT268_pair_state
    result={k:v for k,v in state.items() if k!='original_gain'}
    import json
    return json.loads(json.dumps(result))
