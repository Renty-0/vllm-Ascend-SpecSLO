"""Original199 worker with finite current M120 target component; owned exit retained."""
import ast
import copy
from pathlib import Path
import sys

HERE=Path(__file__).parent;B=HERE.parent
PARENT=B/'tree-bounded-parent4-shared4-eager-diagnostic-v199'
sys.path.insert(0,str(PARENT))
sys.path.insert(0,str(B/'tree-contiguous-branch-tail-final-metadata-math-v234'))
sys.path.insert(0,str(B/'tree-parent4-best376-target-binding-host-v390'))
sys.path.insert(0,str(B/'tree-parent4-frontier-observer-before-capture-host-v392'))
import runtime199 as original
_RUNTIME=None


def projected_source():
    source=(PARENT/'runtime199.py').read_text()
    changes=(('from check199 import require_ready','from check396 import require_ready',2),
        ("extension_module='extension199'","extension_module='extension396'",1),
        ("require(extension_module=='extension199'","require(extension_module=='extension396'",1),
        ('from audit199 import audit','from audit396 import audit',1),
        ('   if engine is not None:engine.exit()\n   save(output,\'worker-exit.json\',dict(closed=bool(engine and engine._closed),all_exited=bool(engine and all(not p.is_alive() for p in engine._processes))))',
         "   from shutdown234 import finish\n   exited,exit_wait=finish(engine)\n   save(output,'worker-exit-wait.json',exit_wait)\n   save(output,'worker-exit.json',exited)",1))
    result=source
    for before,after,count in changes:
        if result.count(before)!=count:raise RuntimeError('ROOT396 exact original runtime anchor '+before)
        result=result.replace(before,after)
    inverse=result
    for before,after,_ in reversed(changes):inverse=inverse.replace(after,before)
    if inverse!=source:raise RuntimeError('ROOT396 entire original worker/main source inverse')
    return result


def build_runtime():
    global _RUNTIME
    if _RUNTIME is None:
        from check396 import ready
        ready()
        nodes=[copy.deepcopy(n)for n in ast.parse(projected_source()).body
               if isinstance(n,ast.FunctionDef)and n.name in ('worker','main')]
        if len(nodes)!=2:raise RuntimeError('ROOT396 both original runtime bodies required')
        namespace=dict(vars(original),__name__='ROOT396.original199-owned-exit')
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),
                     __file__+'::original199-worker-main','exec'),namespace)
        actual=namespace['worker'];namespace['worker']=worker
        _RUNTIME=dict(worker=actual,main=namespace['main'])
    return _RUNTIME


def worker(*args,**kwargs):return build_runtime()['worker'](*args,**kwargs)


if __name__=='__main__':build_runtime()['main']()
