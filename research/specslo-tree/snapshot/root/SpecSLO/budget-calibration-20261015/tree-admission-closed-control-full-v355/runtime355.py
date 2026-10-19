"""Additive original348 worker projection with explicit admission-only delegates."""
import ast
import copy
import sys
from pathlib import Path


P = Path(__file__).resolve().parent
B = P.parent
sys.path[:0] = [str(P), str(B / 'tree-coalesced-live-refill-M120-singleton-capture-full-v348'),
               str(B / 'tree-priority-protected-live-refill-host-v354')]
import runtime348 as parent


_RUNTIME = None


def project_source(source):
    changes = (
        ("extension_module='extension348'", "extension_module='extension355'"),
        ("require(extension_module=='extension348'", "require(extension_module=='extension355'"),
        ('     from trace348 import run_windows,warm_to_fixedpoint',
         '     from trace355 import run_windows,warm_to_fixedpoint'),
        ('   from audit348 import audit', '   from audit355 import audit'),
        ('\n  from check348 import ready as own_ready\n', '\n  from check355 import ready as own_ready\n'),
        ('\n from check348 import ready as own_ready\n', '\n from check355 import ready as own_ready\n'),
        ("     report['Native_final']=", "     report['ROOT355_admission_mode']='closed'\n     report['Native_final']="),
    )
    result = source
    for before, after in changes:
        if result.count(before) != 1:
            raise RuntimeError('exact original348 worker anchor ' + before)
        result = result.replace(before, after)
    inverse = result
    for before, after in reversed(changes):
        if inverse.count(after) != 1:
            raise RuntimeError('exact inverse348 worker anchor ' + after)
        inverse = inverse.replace(after, before)
    if inverse != source:
        raise RuntimeError('whole original348 projected worker inverse')
    return result, changes


def build_runtime():
    global _RUNTIME
    if _RUNTIME is None:
        from check355 import ready
        ready()
        original = parent.build_runtime()
        projected, changes = project_source(original['projected_source'])
        private = dict(original['worker'].__globals__, __name__='ROOT355.explicit-original348-worker')
        names = {'save', 'worker', 'main'}
        nodes = [copy.deepcopy(node) for node in ast.parse(projected).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
        if len(nodes) != len(names):
            raise RuntimeError('all original save/worker/main bodies required')
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                     __file__ + '::admission-only-worker-projection', 'exec'), private)
        actual_worker = private['worker']
        private['worker'] = worker
        _RUNTIME = dict(worker=actual_worker, main=private['main'], save=private['save'],
                        changes=changes, projected_source=projected)
    return _RUNTIME


def worker(*args, **kwargs):
    return build_runtime()['worker'](*args, **kwargs)


if __name__ == '__main__':
    build_runtime()['main']()
