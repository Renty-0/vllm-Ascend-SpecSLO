"""Original438 full worker with explicit new extension, scope, trace and audit."""
from pathlib import Path
import ast,copy,sys
from host_env460 import setup
setup()
P=Path(__file__).resolve().parent
import runtime438 as original
_RUNTIME=None
def source():
    before=original.source();after=before
    changes=(("extension_module='extension438'","extension_module='extension460'"),
             ("require(extension_module=='extension438'","require(extension_module=='extension460'"),
             ('from trace438 import run_windows,warm_to_fixedpoint','from trace460 import run_windows,warm_to_fixedpoint'),
             ('from scope438 import issue,report as scope_report','from scope460 import issue,report as scope_report'),
             ('   from audit438 import audit','   from audit460 import audit'))
    for old,new in changes:
        if after.count(old)!=1:raise RuntimeError('ROOT460 exact original438 routing '+old)
        after=after.replace(old,new)
    inverse=after
    for old,new in reversed(changes):inverse=inverse.replace(new,old)
    if inverse!=before:raise RuntimeError('ROOT460 entire full438 worker/main source inverse')
    return after
def build_runtime():
    global _RUNTIME
    if _RUNTIME is not None:return _RUNTIME
    from check460 import ready
    ready();parent=original.build_runtime()
    ns=dict(parent['worker'].__globals__,__name__='ROOT460.original438-full-lifecycle-adaptive-target')
    nodes=[copy.deepcopy(n)for n in ast.parse(source()).body if isinstance(n,ast.FunctionDef)and n.name in('save','worker','main')]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),__file__+'::full438-explicit-adaptive-routing','exec'),ns)
    actual=ns['worker'];ns['worker']=worker;_RUNTIME=dict(worker=actual,main=ns['main'],save=ns['save']);return _RUNTIME
def worker(*args,**kwargs):
    from check460 import ready
    ready();return build_runtime()['worker'](*args,**kwargs)
if __name__=='__main__':build_runtime()['main']()
