"""Explicit prospective capacity; not a LIVE scheduling or NPU qualification."""
from plan269 import make

def envelope(n,q,family):
    if type(n)is not int or type(q)is not int or family not in('causal','full9'):
        raise ValueError('HOST285 original integer geometry and Native families')
    if not 1<=n<=32 or not 2*n<=q<=min(9*n,165)or q-n>128:
        raise ValueError('HOST285 declared N32/C128/M165')
    if family=='causal'and q>5*n:raise ValueError('primary spine at most4')
    left=165-q;padding=[]
    while left:padding.append(min(5,left));left-=min(5,left)
    return 165,n+len(padding),padding

def plan_family(plans):
    from plan269 import plan_family as old
    result=old(plans)
    if sum(p.candidate_budget>4 for p in plans)>12:
        raise ValueError('HOST285 retained private-copy and genuine-draft fork limit12')
    return result
