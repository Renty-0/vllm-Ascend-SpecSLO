"""Exact434 numerical worker; only the parent's qualification receive changes."""
import ast,copy
from pathlib import Path
import sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'tree-nonempty-wholeprompt-current-model-full-v434'))
import runtime434 as original
_RUNTIME=None


def source():
    before=original.source();after=before
    changes=(("extension_module='extension434'","extension_module='extension436'"),
        ("require(extension_module=='extension434'","require(extension_module=='extension436'"),
        ('   from audit434 import audit','   from audit436 import audit'),
        ("   replies=engine._receive_all('V209 '+ACTION)",
         "   from wait436 import receive_qualification\n   replies=receive_qualification(engine,'V209 '+ACTION)"),
        ("   save(output,'completed.json',dict(action=ACTION,replies=replies,audit=result,formal_Goodput=False,adopted=False))",
         "   save(output,'completed.json',dict(action=ACTION,replies=replies,audit=result,qualification_rpc_wait=__import__('wait436').snapshot(),formal_Goodput=False,adopted=False))"))
    for old,new in changes:
        if after.count(old)!=1:raise RuntimeError('ROOT436 exact434 parent anchor '+old)
        after=after.replace(old,new)
    inverse=after
    for old,new in reversed(changes):inverse=inverse.replace(new,old)
    if inverse!=before:raise RuntimeError('ROOT436 entire434 runtime inverse')
    return after


def build_runtime():
    global _RUNTIME
    if _RUNTIME is not None:return _RUNTIME
    from check436 import ready
    ready();parent=original.build_runtime()
    ns=dict(parent['worker'].__globals__,__name__='ROOT436.original434-qualification-receive-view')
    nodes=[copy.deepcopy(n)for n in ast.parse(source()).body
           if isinstance(n,ast.FunctionDef)and n.name in('save','worker','main')]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),
                 __file__+'::explicit434-parent-receive-only','exec'),ns)
    current=ns['worker'];ns['worker']=worker
    _RUNTIME=dict(worker=current,main=ns['main'],save=ns['save'])
    return _RUNTIME


def worker(*args,**kwargs):return build_runtime()['worker'](*args,**kwargs)


if __name__=='__main__':build_runtime()['main']()
