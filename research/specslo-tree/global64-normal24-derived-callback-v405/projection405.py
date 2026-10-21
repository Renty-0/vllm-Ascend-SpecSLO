"""Explicit reversible source deltas around the frozen best257 protocol."""
import ast
import inspect


def project_function(function, changes, additions=None):
    source = inspect.getsource(function)
    result = source
    for before, after in changes:
        if result.count(before) != 1:
            raise RuntimeError('ROOT405 exact source anchor: ' + before)
        result = result.replace(before, after)
    inverse = result
    for before, after in reversed(changes):
        if inverse.count(after) != 1:
            raise RuntimeError('ROOT405 exact inverse anchor: ' + after)
        inverse = inverse.replace(after, before)
    if inverse != source:
        raise RuntimeError('ROOT405 complete original function inverse')
    namespace = dict(function.__globals__)
    namespace.update(additions or {})
    exec(compile(ast.fix_missing_locations(ast.parse(result)),
                 __file__ + '::' + function.__name__, 'exec'), namespace)
    return namespace[function.__name__]
