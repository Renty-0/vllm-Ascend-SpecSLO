"""HOST-only plan construction: primary4 plus one alternate chain1..4.

This changes tree topology. It is not installed in any NPU or serving run.
"""
from functools import lru_cache

@lru_cache(maxsize=1024)
def make(budget,prefix,anchor=0,context=4096):
    import torch
    from vllm_ascend.spec_decode.pearl.tree import TreeSpeculationPlan,_relative_tree_attention_mask_view
    if type(budget)is not int or not 1<=budget<=8:
        raise ValueError('HOST269 selected candidate budget1..8')
    side=max(0,budget-4)
    if type(anchor)is not int or anchor<0 or (side and anchor+side>4):
        raise ValueError('HOST269 entire alternate path must fit depth4')
    parents=tuple([-1,0,1,2][:min(budget,4)]+([anchor-1]+list(range(4,budget-1))if side else[]))
    levels=[]
    for parent in parents:levels.append(1 if parent<0 else levels[parent]+1)
    return TreeSpeculationPlan(width=2,depth=4,candidate_budget=budget,prefix_len=prefix,max_model_len=context,
        parent_indices=torch.tensor(parents,dtype=torch.int32,device='cpu'),
        positions=torch.tensor([prefix]+[prefix+d for d in levels],dtype=torch.long,device='cpu'),
        attention_mask=_relative_tree_attention_mask_view(parents,prefix,context),
        cache_positions=torch.arange(prefix,prefix+budget+1,dtype=torch.long,device='cpu'))


def envelope(n,q,family):
    if type(n)is not int or type(q)is not int or family not in('causal','full9'):
        raise ValueError('HOST269 original integer geometry and Native plan families')
    if not 1<=n<=24 or not 2*n<=q<=min(9*n,120)or q-n>96:
        raise ValueError('HOST269 original N24/C96/Q120')
    if family=='causal'and q>5*n:
        raise ValueError('HOST269 a linear primary spine has at most4 candidates')
    remaining=120-q;padding=[]
    while remaining:padding.append(min(5,remaining));remaining-=min(5,remaining)
    return 120,n+len(padding),padding


def plan_family(plans):
    branched=False
    for plan in plans:
        parents=plan.parent_indices.tolist();b=plan.candidate_budget
        if b!=len(parents)or not 1<=b<=8:
            raise ValueError('HOST269 declared selected tree B1..8 only')
        if b<=4:
            if parents!=[-1,0,1,2][:b]:raise ValueError('HOST269 original primary spine')
        else:
            if (parents[:4]!=[-1,0,1,2]or parents[4]not in(-1,0,1,2)
                    or parents[5:]!=list(range(4,b-1))or parents[4]+2+b-5>4):
                raise ValueError('HOST269 exact alternate chain within depth4')
            branched=True
    return 'full9'if branched else'causal'
