"""Exact original173/209/218 math bodies, with own explicit import origins."""
import ast
import copy
from pathlib import Path
from types import FunctionType,SimpleNamespace


def parent_component():
    import component199
    from base199 import P173,BASE,functions
    original=component199._PARENT
    def selected_functions(path,namespace,**kwargs):
        if Path(path)==BASE/'actor_v146.py':
            kwargs['changes']=tuple(kwargs.get('changes',()))+((
                'from import_origins_v132 import require_loaded_origins as assert_origins',
                'from extension423 import loaded_origins as assert_origins',1),)
        return functions(path,namespace,**kwargs)
    namespace=dict(vars(original),functions=selected_functions)
    functions(P173/'component173.py',namespace)
    return SimpleNamespace(**namespace)


def construct209():
    import component209 as original
    old_parent=original.parent_module
    def parent(name):
        return parent_component() if name=='component173' else old_parent(name)
    namespace=dict(vars(original),parent_module=parent)
    function=original.run
    return FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)


def qualify(engine,native,stage,policy,output):
    result=construct209()(engine,native,stage,policy,output)
    # Exactly the original218 tail after its original209 component.
    from slot_preflight218 import end_shadow
    stage('ROOT218/complete-actual-preflight-shadow',lambda:end_shadow(engine))
    return result
