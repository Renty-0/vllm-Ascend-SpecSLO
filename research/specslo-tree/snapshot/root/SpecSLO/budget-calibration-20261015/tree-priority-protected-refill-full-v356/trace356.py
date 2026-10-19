"""Explicit private trace globals; frozen348 numerical functions stay unchanged."""
import types

import trace348
from generation356 import patch


_NAMESPACE = None


def namespace():
    global _NAMESPACE
    if _NAMESPACE is None:
        parent = trace348.namespace()
        private = dict(parent, canonical_patch=patch)
        for name in ('make_trace', 'warm_to_fixedpoint', 'run_windows'):
            original = parent[name]
            function = types.FunctionType(original.__code__, private, original.__name__,
                                          original.__defaults__, original.__closure__)
            function.__kwdefaults__ = (None if original.__kwdefaults__ is None
                                       else dict(original.__kwdefaults__))
            function.__annotations__ = dict(original.__annotations__)
            private[name] = function
        _NAMESPACE = private
    return _NAMESPACE


def warm_to_fixedpoint(*args, **kwargs):
    return namespace()['warm_to_fixedpoint'](*args, **kwargs)


def run_windows(*args, **kwargs):
    return namespace()['run_windows'](*args, **kwargs)


metrics = trace348.metrics
