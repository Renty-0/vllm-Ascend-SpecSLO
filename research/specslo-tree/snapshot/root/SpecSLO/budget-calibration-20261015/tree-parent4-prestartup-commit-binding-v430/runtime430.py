"""Exact428 worker with the omitted original199 pre-startup identity capture."""
import ast
import copy
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P.parent / 'tree-parent4-current-state-metadata-binding-v428'))
import runtime428 as original
_RUNTIME = None


def source():
    before = original.source()
    changes = (
        ("extension_module='extension428'", "extension_module='extension430'"),
        ("require(extension_module=='extension428'", "require(extension_module=='extension430'"),
        ('   from audit428 import audit', '   from audit430 import audit'),
        ('   protocol=capture_original_protocol_before_startup(native)',
         '   protocol=capture_original_protocol_before_startup(native)\n   from lifecycle430 import capture,bind\n   original_commit430=capture(protocol,native)'),
        ('   engine._v75_original_protocol=protocol',
         '   engine._v75_original_protocol=protocol\n   bind(engine,original_commit430)'),
        ("     report['Native_final']=", "     report['ROOT430_original_commit_binding']=__import__('lifecycle430').snapshot(engine)\n     report['Native_final']="),
    )
    after = before
    for old, new in changes:
        if after.count(old) != 1:
            raise RuntimeError('ROOT430 exact428 worker anchor ' + old)
        after = after.replace(old, new)
    reverse = after
    for old, new in reversed(changes):
        reverse = reverse.replace(new, old)
    if reverse != before:
        raise RuntimeError('ROOT430 entire428 worker inverse')
    return after


def build_runtime():
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    from check430 import ready
    ready()
    parent = original.build_runtime()
    ns = dict(parent['worker'].__globals__, __name__='ROOT430.exact428-with-original199-commit-binding')
    nodes = [copy.deepcopy(n) for n in ast.parse(source()).body
             if isinstance(n, ast.FunctionDef) and n.name in ('save', 'worker', 'main')]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 __file__ + '::explicit428-worker-commit-binding', 'exec'), ns)
    current = ns['worker']
    ns['worker'] = worker
    _RUNTIME = dict(worker=current, main=ns['main'], save=ns['save'])
    return _RUNTIME


def worker(*args, **kwargs):
    return build_runtime()['worker'](*args, **kwargs)


if __name__ == '__main__':
    build_runtime()['main']()
