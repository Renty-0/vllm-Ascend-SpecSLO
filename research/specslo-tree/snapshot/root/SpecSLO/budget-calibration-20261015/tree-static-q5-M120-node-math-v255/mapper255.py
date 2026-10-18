"""Original real limits, with an explicit120-row physical envelope."""
def envelope(n,q,family):
    if type(n)is not int or type(q)is not int or family not in('causal','full9'):
        raise ValueError('original integer geometry and plan family')
    if not 1<=n<=24 or not 2*n<=q<=min(6*n,120)or q-n>96:
        raise ValueError('realN24/C96/Q120 within original32/96/128 limits')
    if family=='causal'and q>5*n or family=='full9'and q<2*n+4:raise ValueError('real plan family agrees with budgets')
    remaining=120-q;padding=[]
    while remaining:padding.append(min(5,remaining));remaining-=min(5,remaining)
    return 120,n+len(padding),padding
