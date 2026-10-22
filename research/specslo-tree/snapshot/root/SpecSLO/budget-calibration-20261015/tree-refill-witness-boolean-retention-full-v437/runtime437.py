"""Exact436 worker, with only declared own routing and trace-retention fix."""
import ast,copy
from pathlib import Path
import sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'tree-nonempty-qualified-RPC-full-v436'))
import runtime436 as original
_RUNTIME=None


def source():
    before=original.source();after=before
    changes=(("extension_module='extension436'","extension_module='extension437'"),
        ("require(extension_module=='extension436'","require(extension_module=='extension437'"),
        ('from trace432 import run_windows,warm_to_fixedpoint','from trace437 import run_windows,warm_to_fixedpoint'),
        ('   from audit436 import audit','   from audit437 import audit'))
    for old,new in changes:
        if after.count(old)!=1:raise RuntimeError('ROOT437 exact436 routing anchor '+old)
        after=after.replace(old,new)
    inverse=after
    for old,new in reversed(changes):inverse=inverse.replace(new,old)
    if inverse!=before:raise RuntimeError('ROOT437 entire436 runtime inverse')
    return after


def build_runtime():
    global _RUNTIME
    if _RUNTIME is not None:return _RUNTIME
    from check437 import ready
    ready();parent=original.build_runtime()
    ns=dict(parent['worker'].__globals__,__name__='ROOT437.original436-retention-fix')
    nodes=[copy.deepcopy(n)for n in ast.parse(source()).body if isinstance(n,ast.FunctionDef)and n.name in('save','worker','main')]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),__file__+'::trace-retention-only','exec'),ns)
    current=ns['worker'];ns['worker']=worker
    _RUNTIME=dict(worker=current,main=ns['main'],save=ns['save'])
    return _RUNTIME


def worker(*args,**kwargs):return build_runtime()['worker'](*args,**kwargs)
if __name__=='__main__':build_runtime()['main']()
