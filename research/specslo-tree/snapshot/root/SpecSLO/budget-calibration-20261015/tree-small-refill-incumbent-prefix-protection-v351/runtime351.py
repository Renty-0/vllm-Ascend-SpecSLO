"""Original311 four-role lifecycle;594finite branch/bucket target cases."""
import ast,copy
from pathlib import Path
import sys
P=Path(__file__).resolve().parent;B=P.parent
sys.path.insert(0,str(B/'tree-single-prepare-target-math-cost-v311'))
import runtime311 as original
_RUNTIME=None
ACTION='small-refill-singleton-graph-and-eager-component'

def worker_source():
    source=Path(original.__file__).read_text()
    changes=[('from check311 import ready','from check351 import ready',2),
             ('from probes311 import run','from probe351 import run',1),
             ('numeric_cases=408','numeric_cases=16',2),
             ('alternate-chain-B1to8-M120-node-component',ACTION,2)]
    result=source
    for before,after,count in changes:
        if result.count(before)!=count:raise RuntimeError('exact351 original loader anchor '+before)
        result=result.replace(before,after)
    reverse=result
    for before,after,_ in reversed(changes):reverse=reverse.replace(after,before)
    if reverse!=source:raise RuntimeError('whole351 loader source inverse')
    return result

def build_runtime():
    global _RUNTIME
    if _RUNTIME is None:
        from check351 import ready
        ready()
        nodes=[copy.deepcopy(n)for n in ast.parse(worker_source()).body if isinstance(n,ast.FunctionDef)and n.name in('worker','main')]
        if len(nodes)!=2:raise RuntimeError('both complete original lifecycle functions required')
        ns=dict(vars(original),__name__='ROOT351.retained-lifecycle-new-finite-probe')
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),__file__+'::original-lifecycle','exec'),ns)
        actual_worker=ns['worker'];ns['worker']=worker
        _RUNTIME=dict(worker=actual_worker,main=ns['main'])
    return _RUNTIME

def worker(*args,**kwargs):return build_runtime()['worker'](*args,**kwargs)
if __name__=='__main__':build_runtime()['main']()
