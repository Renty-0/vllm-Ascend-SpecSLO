"""Preserve the original constructor receipt root; reuse frozen336 math."""
from pathlib import Path
import sys
from types import FunctionType

P = Path(__file__).resolve().parent
B = P.parent
PARENT = B/'tree-Q4-linear-target-math-cost-v336'
DONOR = B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
sys.path.insert(0, str(PARENT))
import runtime336 as parent
_RUNTIME = None


def clone_main(function):
    namespace = dict(function.__globals__, DONOR=DONOR, worker=worker)
    cloned = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    cloned.__kwdefaults__ = function.__kwdefaults__
    cloned.__annotations__ = dict(function.__annotations__)
    return cloned


def build_runtime():
    global _RUNTIME
    if _RUNTIME is None:
        from check337 import ready
        ready()
        original = parent.build_runtime()
        _RUNTIME = dict(worker=original['worker'], main=clone_main(original['main']))
    return _RUNTIME


def worker(*args, **kwargs):
    return build_runtime()['worker'](*args, **kwargs)


if __name__ == '__main__':
    build_runtime()['main']()
