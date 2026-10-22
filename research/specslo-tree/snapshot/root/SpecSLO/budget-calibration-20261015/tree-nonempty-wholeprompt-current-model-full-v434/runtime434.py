"""Conditional future runner for frozen432; current433 experiment is separate."""
import ast,copy
from pathlib import Path
import sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'tree-nonempty-current-incumbent-full-proposal-v432'))
import runtime432 as original
_RUNTIME=None


def source():
    before=original.source();after=before
    changes=(("extension_module='extension432'","extension_module='extension434'"),
        ("require(extension_module=='extension432'","require(extension_module=='extension434'"),
        ('   from audit432 import audit','   from audit434 import audit'),
        ("     report['Native_final']=","     report['ROOT434_frozen_implementation']=extension.source_binding(engine)\n     report['Native_final']="))
    for old,new in changes:
        if after.count(old)!=1:raise RuntimeError('ROOT434 exact frozen432 anchor '+old)
        after=after.replace(old,new)
    inverse=after
    for old,new in reversed(changes):inverse=inverse.replace(new,old)
    if inverse!=before:raise RuntimeError('ROOT434 entire frozen432 worker inverse')
    return after


def build_runtime():
    global _RUNTIME
    if _RUNTIME is not None:return _RUNTIME
    from check434 import ready
    ready();parent=original.build_runtime()
    ns=dict(parent['worker'].__globals__,__name__='ROOT434.frozen432-owned-full-worker')
    nodes=[copy.deepcopy(n)for n in ast.parse(source()).body
           if isinstance(n,ast.FunctionDef)and n.name in('save','worker','main')]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),
                 __file__+'::explicit432-source-only-projection','exec'),ns)
    current=ns['worker'];ns['worker']=worker
    _RUNTIME=dict(worker=current,main=ns['main'],save=ns['save'])
    return _RUNTIME


def worker(*args,**kwargs):return build_runtime()['worker'](*args,**kwargs)


if __name__=='__main__':build_runtime()['main']()
