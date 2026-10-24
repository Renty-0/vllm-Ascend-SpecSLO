"""Original460 clocks and records, with the declared admission projection."""

from types import FunctionType
import trace460 as original
from generation463 import patch


def namespace():
    parent = original.namespace()
    ns = dict(parent, canonical_patch=patch)
    for name in ('make_trace', 'warm_to_fixedpoint', 'run_windows'):
        fn = parent[name]
        actual = FunctionType(fn.__code__, ns, fn.__name__, fn.__defaults__, fn.__closure__)
        actual.__kwdefaults__ = fn.__kwdefaults__
        actual.__annotations__ = dict(fn.__annotations__)
        ns[name] = actual
    return ns


def warm_to_fixedpoint(*args, **kwargs):
    return namespace()['warm_to_fixedpoint'](*args, **kwargs)


def run_windows(*args, **kwargs):
    return namespace()['run_windows'](*args, **kwargs)
