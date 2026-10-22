"""Complete actual440 diagnostic lifecycle, two source gates and new action."""
import ast,copy
from pathlib import Path
from host_env454 import setup
setup()
import runtime453 as original
_RUNTIME=None
ACTION='M160-real32-native-epilogue-qualification'

def worker_source():
    before=original.worker_source();after=before
    changes=(('from check453 import ready','from check454 import ready',2),
        ('from probe453 import run','from probe454 import run',1),
        ('M120-native-prefill-independent-AR-isolation',ACTION,2))
    for a,b,n in changes:
        if after.count(a)!=n:raise RuntimeError('ROOT454 exact440 lifecycle anchor '+a)
        after=after.replace(a,b)
    inverse=after
    for a,b,_ in reversed(changes):inverse=inverse.replace(b,a)
    if inverse!=before:raise RuntimeError('ROOT454 complete440 source inverse')
    return after

def build_runtime():
    global _RUNTIME
    if _RUNTIME is None:
        from check454 import ready
        ready();parent=original.build_runtime()
        ns=dict(parent['worker'].__globals__,__name__='ROOT454.same440-whole-target-variant')
        nodes=[copy.deepcopy(n)for n in ast.parse(worker_source()).body if isinstance(n,ast.FunctionDef)and n.name in('worker','main')]
        if len(nodes)!=2:raise RuntimeError('ROOT454 both original worker/main')
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),__file__+'::whole-target-layout-diagnostic','exec'),ns)
        actual=ns['worker'];ns['worker']=worker
        _RUNTIME=dict(worker=actual,main=ns['main'])
    return _RUNTIME

def worker(*a,**k):return build_runtime()['worker'](*a,**k)
if __name__=='__main__':build_runtime()['main']()
