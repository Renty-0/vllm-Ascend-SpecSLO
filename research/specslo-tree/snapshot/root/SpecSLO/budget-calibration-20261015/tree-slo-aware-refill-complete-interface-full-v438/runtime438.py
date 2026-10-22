"""Same437 numerical worker; explicit scheduling/reporting source routing."""
import ast,copy
from pathlib import Path
import sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'tree-refill-witness-boolean-retention-full-v437'))
import runtime437 as original
_RUNTIME=None


def source():
    before=original.source();after=before
    changes=(("extension_module='extension437'","extension_module='extension438'"),
        ("require(extension_module=='extension437'","require(extension_module=='extension438'"),
        ('from trace437 import run_windows,warm_to_fixedpoint','from trace438 import run_windows,warm_to_fixedpoint'),
        ('from scope432 import issue,report as scope_report','from scope438 import issue,report as scope_report'),
        ('   from audit437 import audit','   from audit438 import audit'))
    for old,new in changes:
        if after.count(old)!=1:raise RuntimeError('ROOT438 exact437 routing '+old)
        after=after.replace(old,new)
    inverse=after
    for old,new in reversed(changes):inverse=inverse.replace(new,old)
    if inverse!=before:raise RuntimeError('ROOT438 entire437 worker inverse')
    return after


def build_runtime():
    global _RUNTIME
    if _RUNTIME is not None:return _RUNTIME
    from check438 import ready
    ready();parent=original.build_runtime()
    ns=dict(parent['worker'].__globals__,__name__='ROOT438.current-SLO-refill-and-complete-interface')
    nodes=[copy.deepcopy(n)for n in ast.parse(source()).body if isinstance(n,ast.FunctionDef)and n.name in('save','worker','main')]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),__file__+'::explicit-scheduling-source','exec'),ns)
    current=ns['worker'];ns['worker']=worker
    _RUNTIME=dict(worker=current,main=ns['main'],save=ns['save']);return _RUNTIME


def worker(*a,**k):return build_runtime()['worker'](*a,**k)
if __name__=='__main__':build_runtime()['main']()
