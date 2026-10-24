"""Full465 lifecycle with cohort48 and corrected terminal-audit interface."""
import ast
import copy
from check466 import ready
import runtime465 as original

_RUNTIME = None
ROUTES = (
    ("extension_module='extension465'", "extension_module='extension466'"),
    ("require(extension_module=='extension465'", "require(extension_module=='extension466'"),
    ('from trace463 import run_windows,warm_to_fixedpoint', 'from trace466 import run_windows,warm_to_fixedpoint'),
    ('from audit463 import audit', 'from audit466 import audit'),
)


def source():
    before = original.source()
    after = before
    for old, new in ROUTES:
        if after.count(old) != 1:
            raise RuntimeError('Exact full465 route required: '+old)
        after = after.replace(old, new)
    inverse = after
    for old, new in reversed(ROUTES):
        inverse = inverse.replace(new, old)
    if inverse != before:
        raise RuntimeError('Complete full465 lifecycle inverse required')
    return after


def build_runtime():
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    ready()
    old = original.build_runtime()
    namespace = dict(old['worker'].__globals__, __name__='ROOT466.current-cards-cohort48')
    nodes = [copy.deepcopy(n) for n in ast.parse(source()).body
             if isinstance(n, ast.FunctionDef) and n.name in ('save', 'worker', 'main')]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 __file__+'::full-lifecycle', 'exec'), namespace)
    actual = namespace['worker']
    namespace['worker'] = worker
    _RUNTIME = dict(worker=actual, main=namespace['main'], save=namespace['save'])
    return _RUNTIME


def worker(*args, **kwargs):
    return build_runtime()['worker'](*args, **kwargs)


if __name__ == '__main__':
    build_runtime()['main']()
