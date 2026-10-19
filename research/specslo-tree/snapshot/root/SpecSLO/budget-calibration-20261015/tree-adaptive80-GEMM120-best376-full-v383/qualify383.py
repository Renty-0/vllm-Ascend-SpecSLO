"""Replace the old681 subset with fresh693 strict logical-superset cases."""
from projection383 import project_function
import qualify376 as original


def qualifier():
    changes=(
        ('from oracle376 import run','from oracle377 import run'),
        ('from oracle257 import cases','from oracle377 import cases'),
        ('from dispatch376 import audit_calls','from dispatch377 import audit_calls'),
        ('engine._branch_tail_state.uniform_KV_bucket376_enabled = True',
         'engine._branch_tail_state.uniform_KV_bucket377_enabled = True'),
        ('    reports = []','    engine._branch_tail_state.adaptive377_enabled = True\n    reports = []'),
        ('ROOT376','ROOT383'),('math376','math383'),('proof376','proof383'),
        ('cases=227','cases=231'),('cases=681','cases=693'),
        ('engine._uniform_KV_bucket376_proof','engine._hybrid383_actual_proof'),
    )
    # This factory allows repeated labels only through the exact global source
    # inverse below; the shared projection helper enforces single anchors.
    import inspect
    source=inspect.getsource(original.qualify);projected=source
    for before,after in changes:
        if before not in projected:raise RuntimeError('ROOT383 exact prior qualifier anchor '+before)
        projected=projected.replace(before,after)
    inverse=projected
    for before,after in reversed(changes):inverse=inverse.replace(after,before)
    if inverse!=source:raise RuntimeError('ROOT383 complete original qualifier source inverse')
    namespace=dict(vars(original))
    exec(compile(projected,__file__+'::original-qualifier-231-case-superset','exec',dont_inherit=True),namespace)
    return namespace['qualify']


def qualify(*args,**kwargs):return qualifier()(*args,**kwargs)
