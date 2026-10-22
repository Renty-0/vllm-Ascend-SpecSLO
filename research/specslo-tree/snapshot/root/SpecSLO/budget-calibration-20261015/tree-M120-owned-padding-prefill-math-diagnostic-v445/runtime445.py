"""Complete actual440 diagnostic lifecycle, two source gates and new action."""
import ast,copy
from pathlib import Path
from host_env445 import setup
setup()
import runtime444 as original
_RUNTIME=None
ACTION='M120-owned-q5-prefill-frames-math-and-cost'

def worker_source():
    before=original.worker_source();after=before
    changes=(('from check444 import ready','from check445 import ready',2),
        ('from probe444 import run','from probe445 import run',1),
        ('M120-empty-q5-groups-prefill-math-and-cost',ACTION,2))
    for a,b,n in changes:
        if after.count(a)!=n:raise RuntimeError('ROOT445 exact440 lifecycle anchor '+a)
        after=after.replace(a,b)
    inverse=after
    for a,b,_ in reversed(changes):inverse=inverse.replace(b,a)
    if inverse!=before:raise RuntimeError('ROOT445 complete440 source inverse')
    return after

def build_runtime():
    global _RUNTIME
    if _RUNTIME is None:
        from check445 import ready
        ready();parent=original.build_runtime()
        ns=dict(parent['worker'].__globals__,__name__='ROOT445.same440-whole-target-variant')
        nodes=[copy.deepcopy(n)for n in ast.parse(worker_source()).body if isinstance(n,ast.FunctionDef)and n.name in('worker','main')]
        if len(nodes)!=2:raise RuntimeError('ROOT445 both original worker/main')
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),__file__+'::whole-target-layout-diagnostic','exec'),ns)
        actual=ns['worker'];ns['worker']=worker
        _RUNTIME=dict(worker=actual,main=ns['main'])
    return _RUNTIME

def worker(*a,**k):return build_runtime()['worker'](*a,**k)
if __name__=='__main__':build_runtime()['main']()
