"""Register the existing read-only frontier observer before startup captures.

No tensor operation, Native entry, validation flag, model input or dispatch is
changed. Existing captured/eager hidden tensors retain their original lifetime.
"""
import ast
from pathlib import Path


def install(native,graph):
    from base199 import V151,checked
    path=checked(V151/'observer151.py');source=path.read_text()
    changes=(
        ('runner._v151_target_hidden={};runner._v151_eager_target_hidden={}',
         "if not(hasattr(runner,'_v151_target_hidden')and hasattr(runner,'_v151_eager_target_hidden')):raise RuntimeError('ROOT392 early frontier observer was not installed')"),
        ('engine._v151_hidden_hook=engine.model.register_forward_hook(hidden_hook)',
         'engine._v151_hidden_hook=runner._v151_hidden_hook'))
    current=source
    for before,after in changes:
        if current.count(before)!=1:raise RuntimeError('ROOT392 exact observer anchor '+before)
        current=current.replace(before,after)
    inverse=current
    for before,after in reversed(changes):inverse=inverse.replace(after,before)
    if inverse!=source:raise RuntimeError('ROOT392 complete observer source inverse')
    node=next(n for n in ast.parse(current).body if isinstance(n,ast.FunctionDef)and n.name=='install')
    namespace={}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
        __file__+'::original151-observer-preserve-early-references','exec'),namespace)
    namespace['install'](native,graph)
    cls=native.NativeACLGraphRunner;original=cls.__init__
    def initialized(runner,*args,**kwargs):
        original(runner,*args,**kwargs)
        role=kwargs.get('is_target_worker')
        if type(role)is not bool:raise RuntimeError('ROOT392 explicit original runner role required')
        if role:return
        if hasattr(runner,'_v151_hidden_hook'):raise RuntimeError('ROOT392 observer already installed')
        runner._v151_is_draft=True
        runner._v151_target_hidden={};runner._v151_eager_target_hidden={}
        def hook(model,inputs,hidden):
            key=getattr(runner,'_v151_target_scope',None)
            if key is None:return
            store=runner._v151_target_hidden if graph._CAPTURED_TASKS is not None else runner._v151_eager_target_hidden
            store[key]=hidden
        runner._v151_hidden_hook=runner.model.register_forward_hook(hook)
    cls.__init__=initialized
