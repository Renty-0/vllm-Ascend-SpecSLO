"""Exact425 runtime, routing to the own explicit reference binding extension."""
import ast
import copy
from pathlib import Path
import sys

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'tree-parent4-canonical-tail-observer-binding-v425'))
import runtime425 as original
_RUNTIME=None


def source():
    before=original.build_runtime()['projected_source']
    changes=(("extension_module='extension425'","extension_module='extension428'"),
        ("require(extension_module=='extension425'","require(extension_module=='extension428'"),
        ('   from audit425 import audit','   from audit428 import audit'),
        ("     report['Native_final']=","     report['ROOT428_reference_metadata_binding']=extension.metadata_snapshot(engine)\n     report['Native_final']="))
    after=before
    for old,new in changes:
        if after.count(old)!=1:raise RuntimeError('ROOT428 exact425 runtime anchor '+old)
        after=after.replace(old,new)
    reverse=after
    for old,new in reversed(changes):reverse=reverse.replace(new,old)
    if reverse!=before:raise RuntimeError('ROOT428 entire425 runtime inverse')
    return after


def build_runtime():
    global _RUNTIME
    if _RUNTIME is not None:return _RUNTIME
    from check428 import ready
    ready()
    parent=original.build_runtime()
    namespace=dict(parent['worker'].__globals__,__name__='ROOT428.exact425-with-reference-selector')
    nodes=[copy.deepcopy(n)for n in ast.parse(source()).body
           if isinstance(n,ast.FunctionDef)and n.name in('save','worker','main')]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),
        __file__+'::explicit425-runtime-reference-binding','exec'),namespace)
    current=namespace['worker'];namespace['worker']=worker
    _RUNTIME=dict(worker=current,main=namespace['main'],save=namespace['save'])
    return _RUNTIME


def worker(*args,**kwargs):return build_runtime()['worker'](*args,**kwargs)


if __name__=='__main__':build_runtime()['main']()
