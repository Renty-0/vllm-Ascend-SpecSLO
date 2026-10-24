"""Original385 target diagnostic lifecycle;135 new B8 cases."""
import ast
import copy
from host_env_span import setup
setup()
import runtime385 as original

ACTION='parallel-fork-branch-span-copy-concurrent-cost'
_RUNTIME=None


def worker_source():
    before=original.worker_source();after=before
    changes=(('from check385 import ready','from check_span import ready',2),
             ('from probe385 import run','from probe_span import run',1),
             ('numeric_cases=1','numeric_cases=21',2),
             ('dict(numeric_cases=21,performance=False)','dict(numeric_cases=21,performance=True)',2),
             ("dict(replies=replies,performance=False,adopted=False)","dict(replies=replies,performance=True,adopted=False)",1),
             ('real-M120-deferred-chain-original-graph-qualifier',ACTION,2))
    for old,new,count in changes:
        if after.count(old)!=count:raise RuntimeError('Exact retained worker source anchor: '+old)
        after=after.replace(old,new)
    inverse=after
    for old,new,count in reversed(changes):
        if inverse.count(new)!=count:raise RuntimeError('Exact worker inverse: '+new)
        inverse=inverse.replace(new,old)
    if inverse!=before:raise RuntimeError('Whole original385 lifecycle inverse required')
    return after


def build_runtime():
    global _RUNTIME
    if _RUNTIME is None:
        from check_span import ready
        ready()
        import runtime311
        nodes=[copy.deepcopy(n)for n in ast.parse(worker_source()).body
               if isinstance(n,ast.FunctionDef)and n.name in('worker','main')]
        if len(nodes)!=2:raise RuntimeError('Both complete original lifecycle functions required')
        ns=dict(vars(runtime311),__name__='parallel_fork_actual_target_lifecycle')
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),
                     __file__+'::original385-lifecycle','exec'),ns)
        actual=ns['worker'];ns['worker']=worker
        _RUNTIME=dict(worker=actual,main=ns['main'])
    return _RUNTIME


def worker(*args,**kwargs):return build_runtime()['worker'](*args,**kwargs)


if __name__=='__main__':build_runtime()['main']()
