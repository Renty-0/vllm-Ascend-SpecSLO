"""HOST prototype: a raw primary4 plus root-alternate4 candidate pool."""
from dataclasses import replace
from functools import lru_cache
PARENTS=(-1,0,1,2,-1,4,5,6)
@lru_cache(maxsize=1024)
def raw(budget,prefix,context=4096):
    from plan269 import make
    if type(budget)is not int or not 1<=budget<=8:raise ValueError('HOST276 raw grant B1..8')
    return replace(make(8,prefix,0,context),candidate_budget=budget)

def for_state(engine,state,budget):
    return raw(int(budget),max(0,len(state.token_ids)-1),engine.config.max_model_len)

def select(requests,verification_budget):
    from vllm_ascend.spec_decode.pearl.tree_budget import GlobalTreeCandidateSelection
    if verification_budget<=0 or sum(r.max_candidates for r in requests.values())>verification_budget:
        raise ValueError('HOST276 actual positive complete selected-candidate budget')
    selected={};progress={};stage1={};stage2={}
    for owner,request in requests.items():
        n=request.max_candidates
        if not 1<=n<=8 or tuple(request.parents)!=PARENTS:
            raise ValueError('HOST276 exact primary4/root-alternate4 topology')
        if any(cost!=1. for cost in request.candidate_costs):
            raise ValueError('HOST276 unchanged original unit candidate charges')
        selected[owner]=tuple(range(n));joint=[]
        for node,parent in enumerate(request.parents):
            joint.append(request.conditional_confidences[node]*(joint[parent]if parent>=0 else 1.))
        progress[owner]=sum(joint[:n])*request.acceptance_rate
        stage1[owner]=request.minimum_candidates;stage2[owner]=n-request.minimum_candidates
    count=sum(map(len,selected.values()))
    return GlobalTreeCandidateSelection(selected,progress,stage1,stage2,count,float(count),verification_budget)
