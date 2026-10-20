"""Read actual existing class/B gain state at the existing phase boundary."""
import math
from budget_policy import PRIORITY


def snapshot(policy,*,engine_generation):
    if type(engine_generation) is not int or engine_generation<0:
        raise ValueError('V140 actual gain-state generation is invalid')
    base=[]
    for cls,values in sorted(policy.base.items()):
        if cls not in PRIORITY or not isinstance(values,(tuple,list)) or len(values)!=9:
            raise ValueError('V140 actual base class/B domain changed')
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values):
            raise ValueError('V140 actual base gain is invalid')
        base.append(dict(slo_class=cls,values=[dict(B=b,g=values[b]) for b in range(1,len(values))]))
    observed=[]
    for key,values in sorted(policy.observed.items()):
        if (not isinstance(key,tuple) or len(key)!=2 or key[0] not in PRIORITY
            or type(key[1]) is not int or not 1<=key[1]<=8
            or not isinstance(values,(tuple,list)) or len(values)!=2
            or type(values[0]) is not int or values[0]<=0 or type(values[1]) is not int or values[1]<0):
            raise ValueError('V140 actual observed class/B sample domain changed')
        observed.append(dict(slo_class=key[0],B=key[1],rounds=values[0],delivered=values[1]))
    return dict(actual_engine_generation=engine_generation,base=base,observed=observed,
        prior_weight=12,unobserved_B_filled_with_zero=False,
        actual_candidate_topology_B=[1,2,3,4,5],unused_base_prior_B=[6,7,8],
        base_is_calibrated_prior_not_current_generation_measurements=True,
        before_fresh_decode_may_describe_previous_generation=True,
        counter_delta_or_cross_generation_subtraction=False)
