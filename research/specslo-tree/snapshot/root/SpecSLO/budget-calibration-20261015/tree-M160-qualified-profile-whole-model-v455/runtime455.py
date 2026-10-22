"""Complete actual440 diagnostic lifecycle, two source gates and new action."""
import ast,copy
from pathlib import Path
from host_env455 import setup
setup()
import runtime454 as original
_RUNTIME=None
ACTION='M160-qualified-profile-whole-model-reference-cost'

def worker_source():
    before=original.worker_source();after=before
    changes=(('from check454 import ready','from check455 import ready',2),
        ('from probe454 import run','from probe455 import run',1),
        ('M160-real32-native-epilogue-qualification',ACTION,2),
        ('from startup_config_v132 import make_config','from config455 import make_config',1))
    for a,b,n in changes:
        if after.count(a)!=n:raise RuntimeError('ROOT455 exact440 lifecycle anchor '+a)
        after=after.replace(a,b)
    inverse=after
    for a,b,_ in reversed(changes):inverse=inverse.replace(b,a)
    if inverse!=before:raise RuntimeError('ROOT455 complete440 source inverse')
    return after

def build_runtime():
    global _RUNTIME
    if _RUNTIME is None:
        from check455 import ready
        ready();parent=original.build_runtime()
        ns=dict(parent['worker'].__globals__,__name__='ROOT455.same440-whole-target-variant')
        nodes=[copy.deepcopy(n)for n in ast.parse(worker_source()).body if isinstance(n,ast.FunctionDef)and n.name in('worker','main')]
        if len(nodes)!=2:raise RuntimeError('ROOT455 both original worker/main')
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),__file__+'::whole-target-layout-diagnostic','exec'),ns)
        actual=ns['worker'];ns['worker']=worker
        _RUNTIME=dict(worker=actual,main=ns['main'])
    return _RUNTIME

def worker(*a,**k):return build_runtime()['worker'](*a,**k)
if __name__=='__main__':build_runtime()['main']()
