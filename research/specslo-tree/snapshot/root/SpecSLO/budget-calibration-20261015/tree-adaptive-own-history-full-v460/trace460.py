"""Original438 trace and clocks; report the additional current adaptive scope."""
from types import FunctionType
import trace438 as original
from scope460 import report
_NAMESPACE=None
def namespace():
    global _NAMESPACE
    if _NAMESPACE is not None:return _NAMESPACE
    parent=original.namespace();ns=dict(parent,standalone_report=report)
    for name in('make_trace','warm_to_fixedpoint','run_windows'):
        fn=parent[name];actual=FunctionType(fn.__code__,ns,fn.__name__,fn.__defaults__,fn.__closure__)
        actual.__kwdefaults__=fn.__kwdefaults__;actual.__annotations__=dict(fn.__annotations__);ns[name]=actual
    _NAMESPACE=ns;return ns
def warm_to_fixedpoint(*args,**kwargs):return namespace()['warm_to_fixedpoint'](*args,**kwargs)
def run_windows(*args,**kwargs):return namespace()['run_windows'](*args,**kwargs)
