"""Original430 worker; corrected lifetime for the extra diagnostic component."""
import ast,copy
from pathlib import Path
import sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'tree-parent4-prestartup-commit-binding-v430'))
import runtime430 as original
_RUNTIME=None


def source():
    before=original.source();after=before
    changes=(("extension_module='extension430'","extension_module='extension433'"),
        ("require(extension_module=='extension430'","require(extension_module=='extension433'"),
        ('   from audit430 import audit','   from audit433 import audit'),
        ("     report['Native_final']=","     report['ROOT433_preflight_component_lifetime']=extension.preflight_snapshot(engine)\n     report['Native_final']="))
    for old,new in changes:
        if after.count(old)!=1:raise RuntimeError('ROOT433 exact430 source anchor '+old)
        after=after.replace(old,new)
    inverse=after
    for old,new in reversed(changes):inverse=inverse.replace(new,old)
    if inverse!=before:raise RuntimeError('ROOT433 entire430 runtime inverse')
    return after


def build_runtime():
    global _RUNTIME
    if _RUNTIME is not None:return _RUNTIME
    from check433 import ready
    ready();parent=original.build_runtime()
    ns=dict(parent['worker'].__globals__,__name__='ROOT433.original430-component-preflight-lifetime')
    nodes=[copy.deepcopy(n)for n in ast.parse(source()).body
           if isinstance(n,ast.FunctionDef)and n.name in('save','worker','main')]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),
                 __file__+'::explicit430-preflight-lifetime','exec'),ns)
    current=ns['worker'];ns['worker']=worker
    _RUNTIME=dict(worker=current,main=ns['main'],save=ns['save'])
    return _RUNTIME


def worker(*args,**kwargs):return build_runtime()['worker'](*args,**kwargs)


if __name__=='__main__':build_runtime()['main']()
