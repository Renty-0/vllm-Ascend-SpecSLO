"""Original463 full lifecycle; route only current component device checks."""
from pathlib import Path
import ast
import copy
from check465 import ready
import runtime463 as original

P = Path(__file__).resolve().parent
_RUNTIME = None


def source():
    before = original.source()
    changes = (
        ("extension_module='extension463'", "extension_module='extension465'"),
        ("require(extension_module=='extension463'", "require(extension_module=='extension465'"),
    )
    after = before
    for old, new in changes:
        if after.count(old) != 1:
            raise RuntimeError('ROOT465 exact source route: ' + old)
        after = after.replace(old, new)
    inverse = after
    for old, new in reversed(changes):
        inverse = inverse.replace(new, old)
    if inverse != before:
        raise RuntimeError('ROOT465 complete original463 worker/main inverse required')
    return after


def build_runtime():
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    ready()
    old = original.build_runtime()
    ns = dict(old['worker'].__globals__, __name__='ROOT465.original463-current-component-cards')
    nodes = [copy.deepcopy(n) for n in ast.parse(source()).body
             if isinstance(n, ast.FunctionDef) and n.name in ('save', 'worker', 'main')]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 __file__ + '::original-full-lifecycle-current-component', 'exec'), ns)
    actual = ns['worker']
    ns['worker'] = worker
    _RUNTIME = dict(worker=actual, main=ns['main'], save=ns['save'])
    return _RUNTIME


def worker(*args, **kwargs):
    return build_runtime()['worker'](*args, **kwargs)


if __name__ == '__main__':
    build_runtime()['main']()
