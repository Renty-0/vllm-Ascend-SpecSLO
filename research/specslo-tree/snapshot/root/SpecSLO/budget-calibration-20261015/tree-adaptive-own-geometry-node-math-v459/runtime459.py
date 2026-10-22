"""Original385 four-role lifecycle; fresh693 cases, no serving claim."""
import ast,copy
from host_env459 import setup
setup()
import runtime385 as original
ACTION='adaptive-own-physical80-120-693-node-math-and-cost'
_RUNTIME=None
def worker_source():
    before=original.worker_source();after=before
    changes=(('from check385 import ready','from check459 import ready',2),
             ('from probe385 import run','from probe459 import run',1),
             ('numeric_cases=1','numeric_cases=693',2),
             ('real-M120-deferred-chain-original-graph-qualifier',ACTION,2))
    for old,new,count in changes:
        if after.count(old)!=count:
            raise RuntimeError('ROOT459 exact original lifecycle source '+old)
        after=after.replace(old,new)
    inverse=after
    for old,new,count in reversed(changes):
        if inverse.count(new)!=count:
            raise RuntimeError('ROOT459 exact source inverse '+new)
        inverse=inverse.replace(new,old)
    if inverse!=before:
        raise RuntimeError('ROOT459 whole original worker/main inverse')
    return after
def build_runtime():
    global _RUNTIME
    if _RUNTIME is not None:return _RUNTIME
    from check459 import ready
    ready()
    import runtime311
    namespace=dict(vars(runtime311),__name__='ROOT459.original385-lifecycle-own-physical-node-protocol')
    nodes=[copy.deepcopy(n)for n in ast.parse(worker_source()).body if isinstance(n,ast.FunctionDef)and n.name in('worker','main')]
    if len(nodes)!=2:raise RuntimeError('ROOT459 both full original lifecycle functions')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),__file__+'::fresh693-own-physical-protocol','exec'),namespace)
    actual=namespace['worker'];namespace['worker']=worker
    _RUNTIME=dict(worker=actual,main=namespace['main']);return _RUNTIME
def worker(*args,**kwargs):return build_runtime()['worker'](*args,**kwargs)
if __name__=='__main__':build_runtime()['main']()
