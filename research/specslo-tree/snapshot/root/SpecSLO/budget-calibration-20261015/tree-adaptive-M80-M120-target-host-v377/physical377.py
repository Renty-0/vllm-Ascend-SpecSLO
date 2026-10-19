"""Same bijection, masks and native cache writes for complete80/120 families."""
import projection255 as original
from source377 import source_module

SMALL, SOURCE = source_module(original, 'projection255.py', {'plan', 'transform'},
                              {120: 80, 121: 81, 24: 16})


def plan(ends, real_queries, m):
    if m not in (80, 120):
        raise ValueError('ROOT377 exactly declared complete family')
    return (SMALL['plan'] if m == 80 else original.plan)(ends, real_queries)


def transform(inputs, *args, **kwargs):
    m = int(inputs.numel())
    if m not in (80, 120):
        raise ValueError('ROOT377 actual input rows equal selected geometry')
    return (SMALL['transform'] if m == 80 else original.transform)(inputs, *args, **kwargs)
