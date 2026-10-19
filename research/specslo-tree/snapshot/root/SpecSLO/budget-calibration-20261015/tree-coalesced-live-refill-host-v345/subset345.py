"""Original home selection with the actual size bound for small refills."""
import ast
import inspect

import subset209 as original


def namespace():
    source = inspect.getsource(original.prepare)
    before = '(1 if bank.n==2 else 16)'
    after = 'min(16,(bank.n+1)//2)'
    if source.count(before) != 1:
        raise RuntimeError('one original expected home occupancy expression')
    projected = source.replace(before, after)
    if projected.replace(after, before) != source:
        raise RuntimeError('whole original planner preparation inverse')
    value = dict(vars(original))
    exec(compile(ast.parse(projected), __file__+'::original-small-home-selection', 'exec'), value)
    return value


def prepare(*args, **kwargs): return namespace()['prepare'](*args, **kwargs)


guard_rest = original.guard_rest
finish = original.finish
