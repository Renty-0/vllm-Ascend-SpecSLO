"""Retain original physical/copy-grid checks, with explicit extent and reuse."""
import inspect
from audit_static_calls311 import audit_calls as original
from bucket343 import case_extent


def audit_calls(cases,calls,prefixes,*,enabled):
    source=inspect.getsource(original)
    changes=[('and call[\'sequence_lens\']==[2560]*24',
              "and call['sequence_lens']==[_ROOT343_extent(case)]*24"),
             ('cold=family not in seen;count=3 if cold else 1',
              "cold=family not in seen and cursor<len(calls) and calls[cursor]['execution_mode']=='capture_replay';count=3 if cold else 1")]
    projected=source
    for before,after in changes:
        if projected.count(before)!=1:raise RuntimeError('exact original343 dispatch anchor '+before)
        projected=projected.replace(before,after)
    inverse=projected
    for before,after in reversed(changes):inverse=inverse.replace(after,before)
    if inverse!=source:raise RuntimeError('whole original physical/copy dispatch inverse')
    namespace=dict(original.__globals__,_ROOT343_extent=lambda case:case_extent(case,prefixes)if enabled else 2560)
    exec(compile(projected,__file__+'::original-dispatch-with-explicit-bucket-and-warm-reuse','exec'),namespace)
    result=namespace['audit_calls'](cases,calls)
    for call in calls:
        detail=call['uniform_KV_bucket343']
        if detail['enabled']is not enabled or detail['extent']!=call['sequence_lens'][0] or not 0<detail['original_max_context']<=detail['extent']:
            raise RuntimeError('actual prepared extent/visible-context bound differs')
    return dict(result,uniform_extent_checked=True,original_reference_FULL2560=True)
