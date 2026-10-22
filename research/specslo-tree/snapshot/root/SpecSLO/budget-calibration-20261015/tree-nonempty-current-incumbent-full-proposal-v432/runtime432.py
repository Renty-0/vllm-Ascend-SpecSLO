"""Proposed original430 runtime with explicit new nonempty-refill scope.

The check432 gate verifies HOST source provenance; there is no launcher here.
The current433 NPU job is separate and unchanged.
"""
import ast
import copy
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'tree-parent4-preflight-component-lifetime-v433'))
import runtime433 as original
_RUNTIME = None


def source():
    before = original.source()
    changes = (
        ("extension_module='extension433'","extension_module='extension432'"),
        ("require(extension_module=='extension433'","require(extension_module=='extension432'"),
        ('from scope425 import issue,report as scope_report','from scope432 import issue,report as scope_report'),
        ('from trace425 import run_windows,warm_to_fixedpoint','from trace432 import run_windows,warm_to_fixedpoint'),
        ('   from audit433 import audit','   from audit432 import audit'),
        ("     report['Native_final']=","     report['ROOT432_current_incumbent_refill']=extension.refill_snapshot(engine)\n     report['Native_final']="),
    )
    after = before
    for old,new in changes:
        if after.count(old)!=1:
            raise RuntimeError('ROOT432 exact430 source anchor '+old)
        after=after.replace(old,new)
    inverse=after
    for old,new in reversed(changes):inverse=inverse.replace(new,old)
    if inverse!=before:raise RuntimeError('ROOT432 complete430 runtime inverse')
    return after


def build_runtime():
    global _RUNTIME
    if _RUNTIME is not None:return _RUNTIME
    from check432 import ready
    ready()
    parent=original.build_runtime()
    ns=dict(parent['worker'].__globals__,__name__='ROOT432.original430-current-nonempty-scope')
    nodes=[copy.deepcopy(n)for n in ast.parse(source()).body
           if isinstance(n,ast.FunctionDef)and n.name in('save','worker','main')]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),
                 __file__+'::explicit430-nonempty-scope','exec'),ns)
    current=ns['worker'];ns['worker']=worker
    _RUNTIME=dict(worker=current,main=ns['main'],save=ns['save'])
    return _RUNTIME


def worker(*args,**kwargs):return build_runtime()['worker'](*args,**kwargs)
