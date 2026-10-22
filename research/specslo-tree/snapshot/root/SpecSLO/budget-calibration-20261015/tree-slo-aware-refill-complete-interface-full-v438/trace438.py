"""Same original437 trace code and retention; declare scheduling explicitly."""
from types import FunctionType
import trace437 as original
from generation438 import patch
from scope438 import report
_NAMESPACE=None


def namespace():
    global _NAMESPACE
    if _NAMESPACE is not None:return _NAMESPACE
    parent=original.namespace();ns=dict(parent,canonical_patch=patch,standalone_report=report)
    for name in('make_trace','warm_to_fixedpoint','run_windows'):
        fn=parent[name];actual=FunctionType(fn.__code__,ns,fn.__name__,fn.__defaults__,fn.__closure__)
        actual.__kwdefaults__=fn.__kwdefaults__;ns[name]=actual
    _NAMESPACE=ns;return ns


def warm_to_fixedpoint(*a,**k):return namespace()['warm_to_fixedpoint'](*a,**k)
def run_windows(*a,**k):return namespace()['run_windows'](*a,**k)
