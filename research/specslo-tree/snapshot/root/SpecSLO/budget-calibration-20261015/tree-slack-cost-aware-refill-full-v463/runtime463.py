"""Original462 complete lifecycle; route only admission, witness and audit."""

from pathlib import Path
import ast
import copy

from check463 import ready
import runtime462 as original
import runtime460

P = Path(__file__).resolve().parent
_RUNTIME = None


def source():
    before = original.source()
    changes = (
        ("extension_module='extension460'", "extension_module='extension463'"),
        ("require(extension_module=='extension460'", "require(extension_module=='extension463'"),
        ('from trace460 import run_windows,warm_to_fixedpoint', 'from trace463 import run_windows,warm_to_fixedpoint'),
        ('   from audit460 import audit', '   from audit463 import audit'),
    )
    after = before
    for old, new in changes:
        if after.count(old) != 1:
            raise RuntimeError('ROOT463 exact original source route: ' + old)
        after = after.replace(old, new)
    inverse = after
    for old, new in reversed(changes):
        inverse = inverse.replace(new, old)
    if inverse != before:
        raise RuntimeError('ROOT463 complete original462 worker/main source inverse required')
    return after


def build_runtime():
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    ready()
    old = runtime460.build_runtime()
    ns = dict(old['worker'].__globals__, __name__='ROOT463.original462-with-credit-cost-admission')
    nodes = [copy.deepcopy(n) for n in ast.parse(source()).body
             if isinstance(n, ast.FunctionDef) and n.name in ('save', 'worker', 'main')]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 __file__ + '::original-full-lifecycle-admission-routing', 'exec'), ns)
    actual = ns['worker']
    ns['worker'] = worker
    _RUNTIME = dict(worker=actual, main=ns['main'], save=ns['save'])
    return _RUNTIME


def worker(*args, **kwargs):
    return build_runtime()['worker'](*args, **kwargs)


if __name__ == '__main__':
    build_runtime()['main']()
