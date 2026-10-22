"""Run every original full245 audit, using the separately corrected time index."""
import hashlib
from pathlib import Path

ORIGINAL=Path('/root/SpecSLO/budget-calibration-20261015/tree-static-q5-full463-own-AR-v245/audit245.py')
ORIGINAL_SHA='6964c2c0fb3ae746637ddd7205a4293f3d1e47911a0f8d2976c1fb9d453624e8'


def prepared():
    source=ORIGINAL.read_text()
    if hashlib.sha256(source.encode()).hexdigest()!=ORIGINAL_SHA:raise RuntimeError('exact full245 raw auditor')
    before='from audit_reference245 import audit as original_reference_audit'
    after='from audit245_temporal_reference_HOST import audit as original_reference_audit'
    if source.count(before)!=1:raise RuntimeError('one original reference dependency')
    changed=source.replace(before,after)
    if changed.replace(after,before)!=source:raise RuntimeError('entire full audit inverse')
    namespace=dict(__file__=str(ORIGINAL))
    exec(compile(changed,str(Path(__file__).resolve())+'::all-original-full-gates','exec'),namespace)
    return namespace['audit']


audit=prepared()
