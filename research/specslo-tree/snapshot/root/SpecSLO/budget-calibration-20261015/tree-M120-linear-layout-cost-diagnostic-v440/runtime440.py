"""Exact385 four-role diagnostic lifecycle with an explicit local-cost action."""
import ast,copy
from pathlib import Path
from host_env440 import setup
setup()
import runtime385 as original
_RUNTIME=None
ACTION='M120-ND-NZ-layout-padding-local-linear-cost'

def worker_source():
    before=original.worker_source();after=before
    changes=(('from check385 import ready','from check440 import ready',2),
             ('from probe385 import run','from probe440 import run',1),
             ('real-M120-deferred-chain-original-graph-qualifier',ACTION,2))
    for old,new,n in changes:
        if after.count(old)!=n:raise RuntimeError('ROOT440 exact385 lifecycle anchor '+old)
        after=after.replace(old,new)
    inverse=after
    for old,new,_ in reversed(changes):inverse=inverse.replace(new,old)
    if inverse!=before:raise RuntimeError('ROOT440 complete385 worker/main source inverse')
    return after

def build_runtime():
    global _RUNTIME
    if _RUNTIME is None:
        from check440 import ready
        ready();parent=original.build_runtime()
        ns=dict(parent['worker'].__globals__,__name__='ROOT440.original385-diagnostic-lifecycle')
        nodes=[copy.deepcopy(n)for n in ast.parse(worker_source()).body if isinstance(n,ast.FunctionDef)and n.name in('worker','main')]
        if len(nodes)!=2:raise RuntimeError('ROOT440 both full original lifecycle functions')
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),__file__+'::explicit-local-linear-cost','exec'),ns)
        actual=ns['worker'];ns['worker']=worker
        _RUNTIME=dict(worker=actual,main=ns['main'])
    return _RUNTIME

def worker(*a,**k):return build_runtime()['worker'](*a,**k)
if __name__=='__main__':build_runtime()['main']()
