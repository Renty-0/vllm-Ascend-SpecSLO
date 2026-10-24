"""Original full-workload clocks; independently declared 48-owner cohort cap."""
from types import FunctionType
import trace463 as original
from generation466 import patch


def namespace():
    parent = original.namespace()
    result = dict(parent, canonical_patch=patch)
    for name in ('make_trace', 'warm_to_fixedpoint', 'run_windows'):
        function = parent[name]
        selected = FunctionType(function.__code__, result, function.__name__,
                                function.__defaults__, function.__closure__)
        selected.__kwdefaults__ = function.__kwdefaults__
        selected.__annotations__ = dict(function.__annotations__)
        result[name] = selected
    return result


def warm_to_fixedpoint(*args, **kwargs):
    return namespace()['warm_to_fixedpoint'](*args, **kwargs)


def run_windows(*args, **kwargs):
    return namespace()['run_windows'](*args, **kwargs)
