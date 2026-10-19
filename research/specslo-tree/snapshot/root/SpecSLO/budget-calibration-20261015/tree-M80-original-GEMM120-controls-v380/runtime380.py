"""Original344 four-role lifecycle with new source gate and cost retention."""
import ast
import copy
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
B = P.parent
PARENT = B / 'tree-adaptive80-first-layer-divergence-v379'
sys.path.insert(0, str(PARENT))
import runtime379 as original
_RUNTIME = None
ACTION = 'M80-original-GEMM120-three-eager-controls'


def worker_source():
    source = original.worker_source()
    changes = (('from check379 import ready','from check380 import ready',2),
               ('from probe379 import run','from probe380 import run',1),
               ('numeric_cases=1','numeric_cases=3',2),
               ('adaptive80-first-layer-stage-observation','M80-original-GEMM120-three-eager-controls',2))
    projected = source
    for before, after, count in changes:
        if projected.count(before) != count:
            raise RuntimeError('ROOT380 exact original lifecycle import required: ' + before)
        projected = projected.replace(before, after)
    inverse = projected
    for before, after, _ in reversed(changes):
        inverse = inverse.replace(after, before)
    if inverse != source:
        raise RuntimeError('ROOT380 complete original worker lifecycle source inverse')
    return projected


def build_runtime():
    global _RUNTIME
    if _RUNTIME is None:
        from check380 import ready
        ready()
        nodes = [copy.deepcopy(node) for node in ast.parse(worker_source()).body
                 if isinstance(node, ast.FunctionDef) and node.name in ('worker', 'main')]
        if len(nodes) != 2:
            raise RuntimeError('both complete original344 lifecycle functions required')
        namespace = dict(vars(__import__('runtime311')), __name__='ROOT380.original344-lifecycle-retained')
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                     __file__ + '::original344-two-import-projection', 'exec'), namespace)
        actual_worker = namespace['worker']
        namespace['worker'] = worker
        _RUNTIME = dict(worker=actual_worker, main=namespace['main'])
    return _RUNTIME


def worker(*args, **kwargs):
    return build_runtime()['worker'](*args, **kwargs)


if __name__ == '__main__':
    build_runtime()['main']()
