"""Causal per-request conditional g(n), with other-request class priors.

HOST prototype. No serving path imports this module yet. Unexposed branch
continuations are never recorded as failed trials; original actual-budget
accounting remains the caller's responsibility.
"""
import json,types
CLASSES=('tight','normal','loose')
PARENTS=(-1,0,1,2,-1,4,5,6)

def blank():return dict(samples=[0.]*8,hits=[0.]*8)

def priors(curve):
    primary=[];previous_survival=1.;previous_g=1.
    for n in range(1,5):
        survival=min(previous_survival,max(0.,curve[n]-previous_g))
        primary.append(survival/previous_survival if previous_survival>0 else .5)
        previous_g=curve[n];previous_survival=survival
    miss=1-primary[0]
    first=min(1.,max(0.,curve[5]-curve[4])/miss)if miss>0 else .5
    return primary+[first,.5,.5,.5]

def curve_from(q):
    out=[0.];product=1.;g=1.
    for p in q[:4]:product*=p;g+=product;out.append(g)
    survival=1-q[0]
    for p in q[4:]:survival*=p;g+=survival;out.append(min(5.,g))
    assert all(0<=x<=1 for x in q)and all(out[i]<=out[i+1]+1e-12 for i in range(8))
    return tuple(out)

def install(policy):
    if hasattr(policy,'_ROOT294'):return policy._ROOT294
    from budget_policy import TreeBudgetPolicy
    original=policy.gain
    if original.__func__ is not TreeBudgetPolicy.gain:raise RuntimeError('ROOT294 original per-instance policy required')
    state=dict(generation=None,revision=0,prior_samples=12.,request_prior_samples=2.,decay=31./32.,
        priors={c:priors(policy.base[c])for c in CLASSES},classes={c:blank()for c in CLASSES},
        requests={},history={},original_gain=original,terminal_skipped=0)
    policy._ROOT294=state
    def gain(this,request,n):
        if this is not policy:raise RuntimeError('ROOT294 same policy instance')
        if type(n)is not int or not 0<=n<=8:raise ValueError('ROOT294 B0..8')
        if n==0:return 0.
        values=prediction(policy,request)['g']
        remaining=max(0,(request.output_token_budget or 256)-request.delivered_tokens)
        return min(remaining,values[n])
    policy.gain=types.MethodType(gain,policy)
    return state

def begin_generation(policy,generation):
    state=install(policy)
    if type(generation)is not int or generation<=0:raise ValueError('ROOT294 actual positive generation')
    if state['generation']==generation:return state
    if state['generation']is not None and generation<state['generation']:raise ValueError('ROOT294 stale generation')
    if policy.observed:raise RuntimeError('ROOT294 original fresh observed reset required')
    if state['generation']is not None:
        state['history'][str(state['generation'])]=dict(classes=state['classes'],requests=state['requests'],terminal_skipped=state['terminal_skipped'])
    state.update(generation=generation,revision=0,classes={c:blank()for c in CLASSES},requests={},terminal_skipped=0)
    return state

def owner(state,request):
    cls=request.slo_class
    if cls not in CLASSES:raise ValueError('ROOT294 original three SLO classes')
    key=str(request.request_index)
    row=state['requests'].setdefault(key,dict(slo_class=cls,raw=blank(),recent=blank(),observations=0,cache=None))
    if row['slo_class']!=cls:raise RuntimeError('ROOT294 request identity changed class')
    return row

def prediction(policy,request):
    state=install(policy);row=owner(state,request)
    cached=row['cache']
    if cached is not None and cached['revision']==state['revision']:return cached
    cls=request.slo_class;total=state['classes'][cls];raw=row['raw'];recent=row['recent']
    k=state['prior_samples'];r=state['request_prior_samples'];prob=[];other=[]
    for i,prior in enumerate(state['priors'][cls]):
        samples=total['samples'][i]-raw['samples'][i];hits=total['hits'][i]-raw['hits'][i]
        if not 0<=hits<=samples:raise RuntimeError('ROOT294 exact own-data subtraction from class counts')
        base=(k*prior+hits)/(k+samples);other.append(base)
        prob.append((r*base+recent['hits'][i])/(r+recent['samples'][i]))
    value=dict(revision=state['revision'],g=curve_from(prob),conditional_probabilities=prob,other_request_class_probabilities=other)
    row['cache']=value
    return value

def events(parents,accepted,outputs):
    b=len(parents)
    if not 1<=b<=8 or tuple(parents)!=PARENTS[:b]:raise ValueError('ROOT294 exact LIVE root-alternate or linear prefix')
    path=[int(v)for v in accepted if v>=0];tokens=[int(v)for v in outputs if v>=0]
    if len(accepted)!=4 or len(outputs)!=5 or list(accepted)!=path+[-1]*(4-len(path))or list(outputs)!=tokens+[-1]*(5-len(tokens))or len(tokens)!=len(path)+1:
        raise ValueError('ROOT294 original untruncated Native verdict')
    previous=-1
    for node in path:
        if not 0<=node<b or parents[node]!=previous:raise ValueError('ROOT294 ancestor-closed verified path')
        previous=node
    primary=len(path)if path and path[0]==0 else 0
    side=len(path)if path and path[0]>=4 else 0
    counts=blank()
    for depth in range(1,min(4,b)+1):
        if primary>=depth-1:counts['samples'][depth-1]=1.;counts['hits'][depth-1]=float(primary>=depth)
    if b>=5 and primary==0:
        for depth in range(1,b-3):
            if side>=depth-1:counts['samples'][depth+3]=1.;counts['hits'][depth+3]=float(side>=depth)
    return b,counts,len(tokens),side,(len(tokens)-side if b>=4 else None)

def observe(policy,request,*,parents,accepted,outputs,remaining_before):
    state=install(policy)
    if state['generation']is None:raise RuntimeError('ROOT294 actual generation must begin')
    b,counts,actual,side,shadow=events(parents,accepted,outputs)
    if type(remaining_before)is not int or remaining_before<=0:raise ValueError('ROOT294 positive horizon')
    if remaining_before<5:state['terminal_skipped']+=1;return None
    row=owner(state,request);cls=request.slo_class;decay=state['decay']
    for name in('samples','hits'):
        for i,value in enumerate(counts[name]):
            state['classes'][cls][name][i]+=value;row['raw'][name][i]+=value
            row['recent'][name][i]=row['recent'][name][i]*decay+value
    row['observations']+=1;state['revision']+=1
    # Preserve old observable shadow accounting, without inventing values
    # beyond an exposed short primary prefix or counting actual B twice.
    if b>=4:
        for n in range(1,5 if b>4 else 4):
            rounds,total=policy.observed.get((cls,n),(0,0))
            policy.observed[cls,n]=(rounds+1,total+min(shadow,n+1))
    else:
        for n in range(1,b):
            rounds,total=policy.observed.get((cls,n),(0,0));policy.observed[cls,n]=(rounds+1,total+min(actual,n+1))
    return dict(B=b,actual=actual,same_prefix_B4=shadow,accepted_side_nodes=side,
        actually_exposed=counts['samples'],actually_correct=counts['hits'])

def observe_primary(*args,**kwargs):return observe(*args,**kwargs)
def snapshot(policy):return json.loads(json.dumps({k:v for k,v in policy._ROOT294.items()if k!='original_gain'}))
