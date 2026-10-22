"""HOST-only404 body with exact real derived control ownership; installs nothing."""
from pathlib import Path
import sys
from types import FunctionType

sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       'tree-S32-actual-typed-allocation-host-v404'))
import shape404 as original
from callback405 import inspect_actual


def shape(engine, shaper, **kwargs):
    def bind(actual_shaper, generation):
        method, policy, callback = inspect_actual(engine, actual_shaper)
        return original.Binding(method, policy, callback, generation)
    function = FunctionType(original.shape.__code__, dict(vars(original), bind=bind),
                            original.shape.__name__, original.shape.__defaults__,
                            original.shape.__closure__)
    return function(engine, shaper, **kwargs)


snapshot = original.snapshot
