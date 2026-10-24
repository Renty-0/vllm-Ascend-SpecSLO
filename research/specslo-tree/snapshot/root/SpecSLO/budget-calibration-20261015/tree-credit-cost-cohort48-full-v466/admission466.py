"""Independent cohort48 policy; original admission function body is unchanged."""
from functools import lru_cache
from types import FunctionType
import admission463 as original

MAXIMUM_NEW=(32,48)


@lru_cache(maxsize=2)
def selected(maximum_new):
    if type(maximum_new) is not int or maximum_new not in MAXIMUM_NEW:
        raise ValueError('Only the prospectively declared32/48 admission arms exist')
    function=original.choose
    namespace=dict(function.__globals__,MAX_NEW=maximum_new)
    result=FunctionType(function.__code__,namespace,function.__name__,
                        function.__defaults__,function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__
    return result


def choose(engine, admitted, *, maximum_new, **kwargs):
    # Retain exact clock, priority-prefix selection, incumbent credit, cost
    # prediction and decision witness. Global active capacity remains64.
    return selected(maximum_new)(engine,admitted,**kwargs)
