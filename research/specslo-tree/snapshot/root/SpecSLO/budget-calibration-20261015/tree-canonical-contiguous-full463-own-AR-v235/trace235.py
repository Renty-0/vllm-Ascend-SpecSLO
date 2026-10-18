"""Keep the original trace functions; project the actual LIVE builder only."""
import ast
import copy
import types

import trace209 as old
from generation231 import patch as canonical_patch
from scope235 import report as standalone_report

_NAMESPACE = None


def namespace():
    global _NAMESPACE
    if _NAMESPACE is not None:
        return _NAMESPACE
    source = (old.P/'trace209.py').read_text()
    before = "state['source']=patch(adapt_source(builder(),active_window))"
    after = "state['source']=canonical_patch(patch(adapt_source(builder(),active_window)))"
    if source.count(before) != 1:
        raise RuntimeError('ROOT235 exactly one post-original-builder projection')
    projected = source.replace(before, after)
    if projected.replace(after, before) != source:
        raise RuntimeError('ROOT235 entire trace209 source inverse')
    nodes = [copy.deepcopy(n) for n in ast.parse(projected).body
             if isinstance(n, ast.FunctionDef) and n.name == 'make_trace']
    result = dict(vars(old), canonical_patch=canonical_patch, standalone_report=standalone_report)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 __file__+'::original-make-trace-with-declared-generation-limits', 'exec'), result)
    for name in ('warm_to_fixedpoint', 'run_windows'):
        fn = getattr(old, name)
        result[name] = types.FunctionType(fn.__code__, result, fn.__name__, fn.__defaults__, fn.__closure__)
    _NAMESPACE = result
    return result


def warm_to_fixedpoint(*args, **kwargs):
    return namespace()['warm_to_fixedpoint'](*args, **kwargs)


def run_windows(*args, **kwargs):
    return namespace()['run_windows'](*args, **kwargs)


metrics = old.metrics
