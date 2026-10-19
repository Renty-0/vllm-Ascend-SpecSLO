"""Original344 four-role lifecycle with new source gate and cost retention."""
import ast
import copy
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
B = P.parent
PARENT = B / 'tree-GEMM120-original-annotation-flags-v381'
sys.path.insert(0, str(PARENT))
import runtime381 as original
_RUNTIME = None
ACTION = 'hybrid-M80-attention-GEMM120-693-and-paired-cost'


def worker_source():
    source = original.worker_source()
    changes = (('from check381 import ready','from check382 import ready',2),
               ('from repair381 import run','from probe382 import run',1),
               ('numeric_cases=3','numeric_cases=693',2),
               ('M80-GEMM120-controls-original-annotation-flags','hybrid-M80-attention-GEMM120-693-and-paired-cost',2))
    projected = source
    for before, after, count in changes:
        if projected.count(before) != count:
            raise RuntimeError('ROOT382 exact original lifecycle import required: ' + before)
        projected = projected.replace(before, after)
    inverse = projected
    for before, after, _ in reversed(changes):
        inverse = inverse.replace(after, before)
    if inverse != source:
        raise RuntimeError('ROOT382 complete original worker lifecycle source inverse')
    return projected


def build_runtime():
    global _RUNTIME
    if _RUNTIME is None:
        from check382 import ready
        ready()
        nodes = [copy.deepcopy(node) for node in ast.parse(worker_source()).body
                 if isinstance(node, ast.FunctionDef) and node.name in ('worker', 'main')]
        if len(nodes) != 2:
            raise RuntimeError('both complete original344 lifecycle functions required')
        namespace = dict(vars(__import__('runtime311')), __name__='ROOT382.original344-lifecycle-retained')
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
