"""Independent spine/side frames at the candidate's physical M80 or M120.

Bank offsets stay24/48, regardless of the physical frame's capacity. No
candidate query/KV is used to construct any independent reference value.
"""
from functools import lru_cache
import reference_schedule255 as original
from source377 import function

@lru_cache(maxsize=2)
def framed(physical_M):
    if physical_M not in (80,120):
        raise ValueError('ROOT459 two explicitly bounded physical families')
    return function(original.schedule,(
        ('range(0,len(active),24)', 'range(0,len(active),_groups459)'),
        ('active[begin:begin+24]', 'active[begin:begin+_groups459]'),
        ('range(24-len(selected))', 'range(_groups459-len(selected))'),
        ('len(tokens)!=120', 'len(tokens)!=_physical459'),
        ('chunk=begin//24', 'chunk=begin//_groups459'),
        ('for j in range(24)]', 'for j in range(_groups459)]'),
        ('list(range(5,121,5))', 'list(range(5,_physical459+1,5))'),
    ),dict(_physical459=physical_M,_groups459=physical_M//5))

def schedule(plans,roots,rows,*,future=0,prefix=None,physical_M=120):
    if physical_M not in (80,120):
        raise ValueError('ROOT459 physical80 or120')
    if len(plans)+sum(p.candidate_budget==5 for p in plans)>physical_M//5:
        raise ValueError('ROOT459 complete primary and side reference groups')
    rounds=framed(physical_M)(plans,roots,rows,future=future,prefix=prefix)
    for frame in rounds:
        frame['physical_M']=physical_M
        if len(frame['tokens'])!=physical_M or len(frame['selected'])!=physical_M//5:
            raise RuntimeError('ROOT459 complete independently computed frame')
    return rounds
