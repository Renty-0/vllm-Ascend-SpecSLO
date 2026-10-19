"""Retained four-role diagnostic loader with an explicit new finite probe."""
import ast
import copy
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
B = P.parent
OLD = B/'tree-M160-S32-N32-target-math-cost-v326'
sys.path.insert(0, str(OLD))
import runtime326 as original
_RUNTIME = None
ACTION = 'linear-Q4-Q5-target-shape-node-cost-component'


def project_source(source):
    changes = (
        ('from check326 import ready', 'from check336 import ready', 2),
        ('alternate-chain-B1to8-M160-N32-node-component', ACTION, 2),
        ('from probes326 import run', 'from probe336 import run', 1),
        ('numeric_cases=532', 'numeric_cases=36', 2),
        ('ROOT326 bounded numeric diagnostics', 'ROOT336 bounded target-shape diagnostics', 1))
    projected = source
    for before, after, count in changes:
        if projected.count(before) != count:
            raise RuntimeError('ROOT336 exact retained loader anchor '+before)
        projected = projected.replace(before, after)
    inverse = projected
    for before, after, _ in reversed(changes):
        inverse = inverse.replace(after, before)
    if inverse != source:
        raise RuntimeError('ROOT336 entire retained loader source inverse')
    return projected, changes


def build_runtime():
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    from check336 import ready
    ready()
    source = Path(original.__file__).read_text()
    projected, changes = project_source(source)
    namespace = dict(vars(original), DONOR=P, __name__='ROOT336.projected-original-loader')
    nodes = [copy.deepcopy(n) for n in ast.parse(projected).body
             if isinstance(n, ast.FunctionDef) and n.name in {'worker', 'main'}]
    if len(nodes) != 2:
        raise RuntimeError('original worker and parent lifecycle required')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 __file__+'::retained-loader-new-finite-probe', 'exec'), namespace)
    actual_worker = namespace['worker']
    namespace['worker'] = worker
    _RUNTIME = dict(worker=actual_worker, main=namespace['main'], changes=changes,
                    original_fixture_directory=str(OLD), projected_source=projected)
    return _RUNTIME


def worker(*args, **kwargs):
    return build_runtime()['worker'](*args, **kwargs)


if __name__ == '__main__':
    build_runtime()['main']()
