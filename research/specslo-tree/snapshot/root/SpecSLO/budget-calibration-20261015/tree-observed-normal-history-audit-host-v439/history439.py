"""Audit actual overflow when present; prove its absence from full histograms.

This module contains only HOST history checks. It cannot grant NN permission.
"""
def need(v,m):
    if v is not True:raise RuntimeError('ROOT439 actual normal history: '+m)

def verify(snapshots,audits):
    need(len(snapshots)==len(audits)==4,'all4 actual rank snapshots and audits')
    need(all(s==snapshots[0]for s in snapshots),'identical complete rank histories')
    for state,proof in zip(snapshots,audits,strict=True):
        raw=state['raw_N_histogram'];admitted=state['admitted_N_histogram'];positive=state['positive_grant_N_histogram']
        need([len(raw),len(admitted),len(positive)]==[65,25,25],'full global64 and normal24 histograms')
        need(all(type(n)is int and n>=0 for h in(raw,admitted,positive)for n in h),'actual nonnegative integer histogram counts')
        need(type(state['call_count'])is int and state['call_count']>0
            and all(sum(h)==state['call_count']for h in(raw,admitted,positive)), 'all actual calls represented')
        actual_overflow=sum(raw[25:])
        retained=sum(row['overflow']is True for row in state['calls'])
        need(type(state['overflow_calls'])is int and proof['overflow_calls']==state['overflow_calls']==actual_overflow==retained,
             'reported/retained overflow exactly equals actual raw-N tail')
        need(proof['calls']==state['call_count']and proof['exact_class_age_policy']is True
            and proof['every_deferred_owner_retained']is True,'unchanged original detailed class/age/defer audit')
        if actual_overflow==0:
            need(raw[:25]==admitted,'without raw overflow every narrow row is admitted unchanged')
    return True
