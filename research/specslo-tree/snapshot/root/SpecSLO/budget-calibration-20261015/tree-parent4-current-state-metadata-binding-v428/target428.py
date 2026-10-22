"""Exact391 finite target proof bodies, using the current-state selector."""
from types import FunctionType
import target391 as original
from metadata428 import physical


def cloned():
    namespace=dict(vars(original),physical=physical)
    fn=original.cloned
    factory=FunctionType(fn.__code__,namespace,fn.__name__,fn.__defaults__,fn.__closure__)
    return factory()


_namespace=cloned()
for _name in ('padded','clone_output','original_call','own_compact','rebuild_history','qualify'):
    globals()[_name]=_namespace[_name]
