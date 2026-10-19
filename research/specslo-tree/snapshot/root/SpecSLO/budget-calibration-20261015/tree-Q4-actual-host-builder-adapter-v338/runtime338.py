"""Original337 lifecycle; only the finite probe import changes to338."""
import ast
import copy
from pathlib import Path
import sys
from types import FunctionType

P = Path(__file__).resolve().parent
B = P.parent
PARENT = B/'tree-Q4-original-output-root-adapter-v337'
sys.path.insert(0, str(PARENT))
import runtime337 as parent
import runtime336 as numerical
_RUNTIME = None


def worker_source():
    source, _ = numerical.project_source(Path(numerical.original.__file__).read_text())
    before, after = 'from probe336 import run', 'from probe338 import run'
    if source.count(before) != 1:
        raise RuntimeError('ROOT338 one original finite probe import required')
    projected = source.replace(before, after)
    if projected.replace(after, before) != source:
        raise RuntimeError('ROOT338 entire worker import source inverse')
    return projected


def clone_main(function):
    namespace = dict(function.__globals__, worker=worker)
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    result.__annotations__ = dict(function.__annotations__)
    return result


def build_runtime():
    global _RUNTIME
    if _RUNTIME is None:
        from check338 import ready
        ready()
        previous = parent.build_runtime()
        nodes = [copy.deepcopy(n) for n in ast.parse(worker_source()).body
                 if isinstance(n, ast.FunctionDef) and n.name == 'worker']
        if len(nodes) != 1:
            raise RuntimeError('one complete original worker body required')
        namespace = dict(vars(numerical.original), __name__='ROOT338.original-worker-new-probe')
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                     __file__+'::original-worker-with-actual-builder-probe', 'exec'), namespace)
        _RUNTIME = dict(worker=namespace['worker'], main=clone_main(previous['main']))
    return _RUNTIME


def worker(*args, **kwargs):
    return build_runtime()['worker'](*args, **kwargs)


if __name__ == '__main__':
    build_runtime()['main']()
