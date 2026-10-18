"""Uninstalled canonical M80 physical envelopes; model/query budgets stay explicit."""
from target_layout import _lengths

M=80
SEGMENTS=(16,14,17,18,22,15)

def envelope(n,q,family):
    if type(n)is not int or type(q)is not int or family not in('causal','full9'):
        raise ValueError('integer real geometry and original plan family required')
    if not 1<=n<=16 or not 2*n<=q<=min(6*n,80):raise ValueError('canonical domain requires N<=16 and real Q<=80')
    if family=='causal'and q>5*n or family=='full9'and q<2*n+4:raise ValueError('real family and retained budgets disagree')
    if q==79:return M,n+1,[1]
    for s in SEGMENTS:
        lengths=_lengths(n,q,M,s)
        if lengths is not None:return M,s,lengths
    raise ValueError('no explicit canonical physical envelope')

def repair_Q79(priority_order,budgets,caps):
    """Reduce the last eligible priority entry by one; never invent candidates.

    Caller must supply the actual original priority order and account the
    returned grant delta. Not bound to any runtime DTO or policy yet.
    """
    order=tuple(priority_order)
    if not order or len(order)!=len(set(order))or set(order)!=set(budgets)or set(order)!=set(caps):raise ValueError('aligned unique actual priority IDs required')
    if not 1<=len(order)<=16 or any(type(budgets[i])is not int or not 1<=budgets[i]<=caps[i]<=5 for i in order):raise ValueError('original retained candidate bounds required')
    q=len(order)+sum(budgets.values())
    if q>80:raise ValueError('upstream target query limit80 required before allocation')
    if q!=79:return dict(budgets),None
    chosen=next((i for i in reversed(order)if budgets[i]>1),None)
    if chosen is None:raise RuntimeError('Q79 must contain a reducible positive grant')
    result=dict(budgets);result[chosen]-=1
    if len(result)+sum(result.values())!=78:raise RuntimeError('exact one-candidate reduction')
    return result,dict(request_id=chosen,before=budgets[chosen],after=result[chosen],candidate_delta=-1,reason='canonical M80 has no root-only padding plan')
