"""Retained original four-role lifecycle; finite new42-case Q5 diagnostic."""
import ast,copy
from pathlib import Path
import sys
P=Path(__file__).resolve().parent;B=P.parent
sys.path.insert(0,str(B/'tree-Q4-closure-aware-host-builder-adapter-v339'))
import runtime339 as retained_paths
import runtime336 as numerical
_RUNTIME=None

def worker_source():
    source,_=numerical.project_source(Path(numerical.original.__file__).read_text())
    changes=[('from check336 import ready','from check341 import ready',2),
             ('from probe336 import run','from probe341 import run',1),
             ('numeric_cases=36','numeric_cases=42',2)]
    out=source
    for before,after,count in changes:
        if out.count(before)!=count:raise RuntimeError('exact341 retained lifecycle anchor '+before)
        out=out.replace(before,after)
    reverse=out
    for before,after,_ in reversed(changes):reverse=reverse.replace(after,before)
    if reverse!=source:raise RuntimeError('entire341 lifecycle source inverse')
    return out

def build_runtime():
    global _RUNTIME
    if _RUNTIME is None:
        from check341 import ready
        ready()
        nodes=[copy.deepcopy(n) for n in ast.parse(worker_source()).body if isinstance(n,ast.FunctionDef)and n.name in ('worker','main')]
        if len(nodes)!=2:raise RuntimeError('both unchanged lifecycle functions required')
        ns=dict(vars(numerical.original),__name__='ROOT341.original-lifecycle-new-probe')
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),__file__+'::retained-lifecycle','exec'),ns)
        original_worker=ns['worker'];ns['worker']=worker
        _RUNTIME=dict(worker=original_worker,main=ns['main'])
    return _RUNTIME

def worker(*args,**kwargs):return build_runtime()['worker'](*args,**kwargs)
if __name__=='__main__':build_runtime()['main']()
