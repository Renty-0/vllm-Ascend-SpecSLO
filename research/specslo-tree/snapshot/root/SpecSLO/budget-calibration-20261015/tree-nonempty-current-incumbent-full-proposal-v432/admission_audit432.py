"""Require actual saved witnesses instead of the old empty-active predicate."""
from audit429 import audit_trace as audit_witnesses


def audit_trace(trace):
    rows = trace.get('ROOT432_refill_witnesses')
    if type(rows) is not list or not rows:
        raise RuntimeError('ROOT432 genuine current saved refill witnesses required')
    return audit_witnesses(trace, rows)
