"""Keep the original trace functions; project the actual LIVE builder only."""
import ast
import copy
import types

import trace209 as old
from generation333 import patch as canonical_patch
from scope333 import report as standalone_report

_NAMESPACE = None


def namespace():
    global _NAMESPACE
    if _NAMESPACE is not None:
        return _NAMESPACE
    source = (old.P/'trace209.py').read_text()
    before = "state['source']=patch(adapt_source(builder(),active_window))"
    after = "state['source']=canonical_patch(patch(adapt_source(builder(),active_window)))"
    if source.count(before) != 1:
        raise RuntimeError('ROOT333 exactly one post-original-builder projection')
    projected = source.replace(before, after)
    if projected.replace(after, before) != source:
        raise RuntimeError('ROOT333 entire trace209 source inverse')
    nodes = [copy.deepcopy(n) for n in ast.parse(projected).body
             if isinstance(n, ast.FunctionDef) and n.name == 'make_trace']
    from window333 import window,adapt_source,trace_source,windows,DEFAULT_WINDOWS
    result = dict(vars(old), canonical_patch=canonical_patch, standalone_report=standalone_report,window=window,adapt_source=adapt_source,trace_source=trace_source,windows=windows,DEFAULT_WINDOWS=DEFAULT_WINDOWS)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 __file__+'::original-make-trace-with-declared-generation-limits', 'exec'), result)
    for name in ('warm_to_fixedpoint', 'run_windows'):
        fn = getattr(old, name)
        if name=='run_windows':
            original_body=__import__('inspect').getsource(fn)
            changed_body=original_body.replace('from window160 import DEFAULT_WINDOWS,windows','from window333 import DEFAULT_WINDOWS,windows')
            if changed_body==original_body:raise RuntimeError('one explicit window pair dependency')
            exec(compile(changed_body,__file__+'::same-pair-new-window','exec'),result)
            fn=result[name]
        result[name] = types.FunctionType(fn.__code__, result, fn.__name__, fn.__defaults__, fn.__closure__)
        result[name].__kwdefaults__ = None if fn.__kwdefaults__ is None else dict(fn.__kwdefaults__)
        result[name].__annotations__ = dict(fn.__annotations__)
    _NAMESPACE = result
    return result


def warm_to_fixedpoint(*args, **kwargs):
    return namespace()['warm_to_fixedpoint'](*args, **kwargs)


def run_windows(*args, **kwargs):
    return namespace()['run_windows'](*args, **kwargs)


metrics = old.metrics
