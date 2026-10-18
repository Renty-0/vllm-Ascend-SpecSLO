"""Same-prefix gains for one primary-four and one alternate chain.

HOST prototype. Observations must come from the actual untruncated original
verdict. Short successful chains are right-censored at their computed depth.
"""
import types
CLASSES=('tight','normal','loose')

def blank():
    return dict(calls=0,primary_shadow_calls=0,terminal_skipped=0,
        layers=[dict(samples=0,hits=0)for _ in range(4)],
        by_budget={str(b):dict(samples=0,sum_actual=0,sum_shadow4=0,side_nodes=0)for b in range(5,9)})

def install(policy):
    if hasattr(policy,'_ROOT274_pairs'):return policy._ROOT274_pairs
    from budget_policy import TreeBudgetPolicy
    original=policy.gain
    if original.__func__ is not TreeBudgetPolicy.gain:
        raise RuntimeError('ROOT274 original per-instance predictor required')
    state=dict(generation=None,by_generation={},by_class={c:blank()for c in CLASSES},
        by_class_anchor={},prior_samples=12,original_gain=original)
    policy._ROOT274_pairs=state
    def gain(this,request,n):
        if this is not policy:raise RuntimeError('ROOT274 predictor instance changed')
        if n<=4:return original(request,n)
        if not 5<=n<=8:raise ValueError('ROOT274 B1..8 only')
        # This predictor is for the new root-alternate topology. Other anchor
        # observations remain diagnostics and cannot contaminate its curve.
        record=state['by_class_anchor'].get(f'{request.slo_class}:0',blank())
        prior_first=max(0.,min(1.,float(this.base[request.slo_class][5]-this.base[request.slo_class][4])))
        previous=1.;bonus=0.
        for j,layer in enumerate(record['layers'][:n-4]):
            prior=prior_first*(0.5**j)
            survival=(state['prior_samples']*prior+layer['hits'])/(state['prior_samples']+layer['samples'])
            survival=min(previous,max(0.,survival));previous=survival;bonus+=survival
        remaining=max(0,(request.output_token_budget or 256)-request.delivered_tokens)
        return min(remaining,5.,max(1.,original(request,4)+bonus))
    policy.gain=types.MethodType(gain,policy)
    policy.counts['ROOT274_paired_predictor_bindings']+=1
    return state

def begin_generation(policy,generation):
    state=install(policy)
    if type(generation)is not int or generation<=0:raise ValueError('ROOT274 actual positive generation')
    if generation==state['generation']:return state
    if state['generation']is not None and generation<state['generation']:
        raise ValueError('ROOT274 stale generation')
    if policy.observed:
        raise RuntimeError('ROOT274 call only after original fresh-generation observations reset')
    row=dict(by_class={c:blank()for c in CLASSES},by_class_anchor={})
    state['by_generation'][str(generation)]=row
    state['generation']=generation;state['by_class']=row['by_class'];state['by_class_anchor']=row['by_class_anchor']
    return state

def observe(policy,request,*,parents,accepted,outputs,remaining_before):
    state=install(policy);b=len(parents)
    if type(state['generation'])is not int:raise RuntimeError('ROOT274 actual generation not begun')
    if not 5<=b<=8 or list(parents[:4])!=[-1,0,1,2]:raise ValueError('ROOT274 primary4 plus alternate chain')
    anchor=parents[4]+1
    if not 0<=anchor<=8-b or list(parents[5:])!=list(range(4,b-1)):
        raise ValueError('ROOT274 one alternate chain, total path depth<=4')
    if len(accepted)!=4 or len(outputs)!=5:raise ValueError('ROOT274 original padded verdict shape')
    path=[int(v)for v in accepted if v>=0];tokens=[int(v)for v in outputs if v>=0]
    if list(accepted)!=path+[-1]*(4-len(path))or list(outputs)!=tokens+[-1]*(5-len(tokens)):
        raise ValueError('ROOT274 original trailing placeholders')
    last=-1
    for node in path:
        if node>=b or parents[node]!=last:raise ValueError('ROOT274 ancestor-closed actual accepted path')
        last=node
    if len(tokens)!=len(path)+1:raise ValueError('ROOT274 original accepted plus frontier convention')
    if type(remaining_before)is not int or remaining_before<=0:raise ValueError('ROOT274 actual remaining horizon')
    row=state['by_class'][request.slo_class]
    if remaining_before<5:row['terminal_skipped']+=1;return None
    side=sum(node>=4 for node in path);length=b-4;actual=len(tokens);shadow=actual-side
    group=state['by_class_anchor'].setdefault(f'{request.slo_class}:{anchor}',blank())
    for record in(row,group):
        record['calls']+=1
        for j,layer in enumerate(record['layers'],1):
            # A complete accepted short chain leaves deeper acceptance unknown.
            # An observed earlier rejection proves every longer survival false.
            if j<=length or side<length:
                layer['samples']+=1;layer['hits']+=int(side>=j)
        item=record['by_budget'][str(b)];item['samples']+=1
        item['sum_actual']+=actual;item['sum_shadow4']+=shadow;item['side_nodes']+=side
        if item['sum_actual']-item['sum_shadow4']!=item['side_nodes']:
            raise RuntimeError('ROOT274 same-prefix pair accounting')
    for n in range(1,5):
        rounds,total=policy.observed.get((request.slo_class,n),(0,0))
        policy.observed[request.slo_class,n]=(rounds+1,total+min(shadow,n+1))
    policy.counts['ROOT274_actual_paired_observations']+=1
    return dict(B=b,actual=actual,same_prefix_B4=shadow,accepted_side_nodes=side,
        counterfactuals={str(n):shadow+min(side,n-4)for n in range(5,b+1)})

def observe_primary(policy,request,*,parents,accepted,outputs,remaining_before):
    """Supply B1..3 shadows; the caller retains original B4 observe once."""
    state=install(policy)
    if type(state['generation'])is not int:raise RuntimeError('ROOT274 actual generation not begun')
    if list(parents)!=[-1,0,1,2]or len(accepted)!=4 or len(outputs)!=5:
        raise ValueError('ROOT274 original primary-four verdict geometry')
    path=[int(v)for v in accepted if v>=0];tokens=[int(v)for v in outputs if v>=0]
    if (path!=list(range(len(path)))or list(accepted)!=path+[-1]*(4-len(path))
            or list(outputs)!=tokens+[-1]*(5-len(tokens))or len(tokens)!=len(path)+1):
        raise ValueError('ROOT274 original primary accepted-prefix verdict')
    if type(remaining_before)is not int or remaining_before<=0:raise ValueError('ROOT274 actual remaining horizon')
    if remaining_before<5:return None
    for n in range(1,4):
        rounds,total=policy.observed.get((request.slo_class,n),(0,0))
        policy.observed[request.slo_class,n]=(rounds+1,total+min(len(tokens),n+1))
    state['by_class'][request.slo_class]['primary_shadow_calls']+=1
    return {str(n):min(len(tokens),n+1)for n in range(1,4)}

def snapshot(policy):
    import json
    return json.loads(json.dumps({k:v for k,v in policy._ROOT274_pairs.items()if k!='original_gain'}))
