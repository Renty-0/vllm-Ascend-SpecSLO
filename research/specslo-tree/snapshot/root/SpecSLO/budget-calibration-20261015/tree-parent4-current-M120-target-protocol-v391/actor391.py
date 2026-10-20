"""Original199 actor, changing only target delegate and diagnostic-state check."""
import ast
from pathlib import Path
from types import FunctionType
import actor199 as original
import target391


def build():
    namespace=dict(vars(original),target=target391)
    for name,function in vars(original).items():
        if callable(function)and getattr(function,'__module__',None)==original.__name__:
            namespace[name]=FunctionType(function.__code__,namespace,function.__name__,
                function.__defaults__,function.__closure__)
            namespace[name].__kwdefaults__=function.__kwdefaults__
    source=Path(original.__file__).read_text()
    before='from live199 import diagnostic';after='from binding391 import diagnostic'
    if source.count(before)!=1:raise RuntimeError('ROOT391 exact finite diagnostic import')
    changed=source.replace(before,after)
    if changed.replace(after,before)!=source:raise RuntimeError('ROOT391 exact actor source inverse')
    node=next(n for n in ast.parse(changed).body if isinstance(n,ast.FunctionDef)and n.name=='run')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
        __file__+'::original199-run-with-explicit-current-state','exec'),namespace)
    return namespace


_namespace=build()
run=_namespace['run']
