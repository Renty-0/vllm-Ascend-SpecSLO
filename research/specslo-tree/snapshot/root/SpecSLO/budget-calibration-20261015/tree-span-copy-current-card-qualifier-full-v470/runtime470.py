"""Complete original466 lifecycle with the scoped parallel-update installer."""
import ast
import copy
from check470 import ready
import runtime466 as original

_RUNTIME=None
ROUTES=(("extension_module='extension466'","extension_module='extension470'"),
        ("require(extension_module=='extension466'","require(extension_module=='extension470'"),
        ('from audit466 import audit','from audit470 import audit'))


def source():
    before=original.source();after=before
    for old,new in ROUTES:
        if after.count(old)!=1:raise RuntimeError('One original466 lifecycle route required: '+old)
        after=after.replace(old,new)
    inverse=after
    for old,new in reversed(ROUTES):inverse=inverse.replace(new,old)
    if inverse!=before:raise RuntimeError('Complete original466 worker/main source inverse required')
    return after


def build_runtime():
    global _RUNTIME
    if _RUNTIME is not None:return _RUNTIME
    ready();old=original.build_runtime()
    ns=dict(old['worker'].__globals__,__name__='ROOT470.tree-contiguous-KV-span-copy-full')
    nodes=[copy.deepcopy(n)for n in ast.parse(source()).body
           if isinstance(n,ast.FunctionDef)and n.name in('save','worker','main')]
    if len(nodes)!=3:raise RuntimeError('Complete original save/worker/main required')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),__file__+'::full-lifecycle','exec'),ns)
    actual=ns['worker'];ns['worker']=worker
    _RUNTIME=dict(worker=actual,main=ns['main'],save=ns['save']);return _RUNTIME


def worker(*args,**kwargs):return build_runtime()['worker'](*args,**kwargs)
if __name__=='__main__':build_runtime()['main']()
