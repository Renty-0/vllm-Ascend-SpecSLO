"""Original344 four-role lifecycle with new source gate and cost retention."""
import ast
import copy
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
B = P.parent
PARENT = B / 'tree-Q5-uniform-KV-bucket-branch-math-cost-v344'
sys.path.insert(0, str(PARENT))
import runtime344 as original
_RUNTIME = None
ACTION = original.ACTION


def worker_source():
    source = original.worker_source()
    changes = (('from check344 import ready', 'from check361 import ready', 2),
               ('from probe344 import run', 'from probe361 import run', 1))
    projected = source
    for before, after, count in changes:
        if projected.count(before) != count:
            raise RuntimeError('ROOT361 exact original lifecycle import required: ' + before)
        projected = projected.replace(before, after)
    inverse = projected
    for before, after, _ in reversed(changes):
        inverse = inverse.replace(after, before)
    if inverse != source:
        raise RuntimeError('ROOT361 complete original worker lifecycle source inverse')
    return projected


def build_runtime():
    global _RUNTIME
    if _RUNTIME is None:
        from check361 import ready
        ready()
        nodes = [copy.deepcopy(node) for node in ast.parse(worker_source()).body
                 if isinstance(node, ast.FunctionDef) and node.name in ('worker', 'main')]
        if len(nodes) != 2:
            raise RuntimeError('both complete original344 lifecycle functions required')
        namespace = dict(vars(original.original), __name__='ROOT361.original344-lifecycle-retained')
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
