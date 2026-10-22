"""Costs only after independent own-physical math; retain cross-M differences."""
from source377 import function
import benchmark378 as original

arms=original.arms

def projected():
    return function(original.run,(
        ("            if heads[modes[0]] != heads[modes[1]]:\n                raise RuntimeError('ROOT378 FULL2560 and bucket greedy outputs differ')\n            expected[arm['name']] = heads[modes[0]]",
         "            expected[arm['name']] = heads"),
        ("expected[arm['name']]:", "expected[arm['name']][mode]:"),
        ('outputs_equal_between_modes=True,',
         "outputs_equal_between_modes=None,actual_warm_outputs_per_mode=expected,\n                    cross_mode_greedy_equal={name:(values[modes[0]]==values[modes[1]] if target else None)for name,values in expected.items()},"),
    ))

def run(*args,**kwargs):
    return projected()(*args,**kwargs)
