"""Explicit original209/218 worker projection, with new declared math scope."""
import ast
import copy
import importlib.util
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
B = P.parent
OLD = B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'
DONOR = B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
sys.path[:0] = [str(P), str(OLD), str(DONOR)]
_RUNTIME = None


def project_source(source):
    changes = (
        ("extension_module='extension209'", "extension_module='extension235'"),
        ("require(extension_module=='extension209'", "require(extension_module=='extension235'"),
        ('  require_ready(own_identity)', '  from check235 import ready as own_ready\n  own_ready()\n  require_ready(own_identity)'),
        (' readiness=require_ready();spec=', ' from check235 import ready as own_ready\n own_ready()\n readiness=require_ready();spec='),
        ("   policy=entry['install'](fast=True,host=True,transport=True,replay=True,repair=True,prefill=True)",
         '   policy=extension.install_entry(entry,native)'),
        ('     from scope209 import issue,report as scope_report', '     from scope235 import issue,report as scope_report'),
        ('     from trace209 import run_windows,warm_to_fixedpoint', '     from trace235 import run_windows,warm_to_fixedpoint'),
        ("     report['Native_final']=", "     report['postfull_independent_FIA_AR']=extension.postfull_AR(engine,native,stage,report['windows'],output_directory)\n     report['ROOT218_slot_preflight']=__import__('slot_preflight218').snapshot()\n     report['ROOT218_cached_parent_dispatch']=extension.cache_snapshot()\n     report['ROOT235_current_math']=getattr(engine,'_canonical235_actual_proof')\n     report['ROOT235_actual_geometry']=extension.snapshot(engine)\n     report['Native_final']="),
        ('   from audit209 import audit', '   from audit235 import audit'),
        ('   if engine is not None:engine.exit()',
         "   from shutdown234 import finish\n   exited,exit_wait=finish(engine)\n   save(output,'worker-exit-wait.json',exit_wait)"),
        ("   save(output,'worker-exit.json',dict(closed=bool(engine and engine._closed),all_exited=bool(engine and all(not p.is_alive() for p in engine._processes))))",
         "   save(output,'worker-exit.json',exited)"),
    )
    projected = source
    for before, after in changes:
        if projected.count(before) != 1:
            raise RuntimeError('ROOT235 exact original runtime anchor '+before)
        projected = projected.replace(before, after)
    inverse = projected
    for before, after in reversed(changes):
        if inverse.count(after) != 1:
            raise RuntimeError('ROOT235 exact inverse runtime anchor '+after)
        inverse = inverse.replace(after, before)
    if inverse != source:
        raise RuntimeError('ROOT235 entire original runtime inverse')
    return projected, changes


def build_runtime():
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    from check235 import ready
    ready()
    path = DONOR/'runtime209.py'
    spec = importlib.util.spec_from_file_location('_ROOT235_original_runtime209', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    namespace = dict(vars(module), __name__='ROOT235.projected-original-runtime209')
    source = path.read_text()
    projected, changes = project_source(source)
    names = {'save', 'worker', 'main'}
    nodes = [copy.deepcopy(n) for n in ast.parse(projected).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 __file__+'::explicit-original-runtime-projection', 'exec'), namespace)
    actual_worker = namespace['worker']
    namespace['worker'] = worker
    _RUNTIME = dict(worker=actual_worker, main=namespace['main'], save=namespace['save'],
                    changes=changes, projected_source=projected)
    return _RUNTIME


def save(*args, **kwargs):
    return build_runtime()['save'](*args, **kwargs)


def worker(*args, **kwargs):
    return build_runtime()['worker'](*args, **kwargs)


if __name__ == '__main__':
    build_runtime()['main']()
