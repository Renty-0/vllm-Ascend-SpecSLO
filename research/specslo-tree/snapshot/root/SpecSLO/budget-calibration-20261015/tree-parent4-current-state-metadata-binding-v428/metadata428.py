"""Same independent390 builder, bound to the actual retained425 state field."""
import hashlib
import inspect
import metadata390 as original
from projection425 import project_function


def sources():
    before=inspect.getsource(original.physical)
    old='uniform_KV_bucket376_enabled';new='uniform_KV_bucket425_enabled'
    if before.count(old)!=1:raise RuntimeError('ROOT428 exact independent selector anchor')
    after=before.replace(old,new)
    if after.replace(new,old)!=before:raise RuntimeError('ROOT428 complete independent builder inverse')
    return before,after


physical=project_function(original.physical,(
    ('uniform_KV_bucket376_enabled','uniform_KV_bucket425_enabled'),))


def descriptor():
    before,after=sources()
    return dict(original_source_sha256=hashlib.sha256(before.encode()).hexdigest(),
        projected_source_sha256=hashlib.sha256(after.encode()).hexdigest(),
        entire_source_inverse=True,actual_selector='uniform_KV_bucket425_enabled',
        independent_inputs_geometry_and_math_unchanged=True,
        backend_callback_reinstalled=False,Native_flags_assigned=False)
