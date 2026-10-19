"""Read failed333 computation with an independent160 scatter-map audit.

This does not rewrite its receipt, rerun inference, or authorize adoption.
"""
import hashlib
import inspect
from pathlib import Path
import types


def scatter_plan(ends, real_queries):
    ends = list(ends)
    assert ends and ends[-1] == 160 and 1 <= real_queries <= 160
    spans = list(zip([0]+ends[:-1], ends, strict=True))
    assert all(type(a) is int and type(b) is int and 0 <= a < b <= 160 for a, b in spans)
    assert real_queries in ends
    real = [(a, b) for a, b in spans if a < real_queries]
    assert len(real) <= 32 and all(b <= real_queries and b-a <= 5 for a, b in real)
    order = [None]*160
    for group, (begin, end) in enumerate(real):
        for index in range(end-begin):
            order[5*group+index] = begin+index
    vacant = [i for i, old in enumerate(order) if old is None]
    assert len(vacant) == 160-real_queries
    for destination, original in zip(vacant, range(real_queries, 160), strict=True):
        order[destination] = original
    assert sorted(order) == list(range(160))
    inverse = [0]*160
    for destination, original in enumerate(order):
        inverse[original] = destination
    return dict(physical_to_original=order, original_to_physical=inverse)


def corrected_module():
    import audit333
    root = Path('/root/SpecSLO/budget-calibration-20261015/tree-M160-C128-F64-global192-full-v333')
    assert Path(audit333.__file__).resolve() == root/'audit333.py'
    import json
    frozen = json.loads((root/'FREEZE333.json').read_text())
    assert hashlib.sha256((root/'audit333.py').read_bytes()).hexdigest() == frozen['files'][str(root/'audit333.py')]
    source = inspect.getsource(audit333.audit)
    before = 'from projection255 import plan as physical_plan'
    after = 'physical_plan = _posthoc_independent_scatter160'
    assert source.count(before) == 1
    projected = source.replace(before, after)
    assert projected.replace(after, before) == source
    namespace = dict(vars(audit333), _posthoc_independent_scatter160=scatter_plan)
    exec(compile(projected, __file__+'::unchanged333-audit-with-independent160-map', 'exec'), namespace)
    module = types.ModuleType('ROOT333_POSTHOC_READONLY_corrected_projection')
    module.audit = namespace['audit']
    return module


module = corrected_module()
