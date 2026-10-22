"""Original independent q5 frame construction with an explicit row capacity."""
from functools import lru_cache
import frames255 as original
from source377 import function

@lru_cache(maxsize=2)
def framed(physical_M):
    if physical_M not in (80,120):
        raise ValueError('ROOT459 full80 or120 only')
    return function(original.pack,(
        ("if rows!=120:raise ValueError('explicit fixed120 shape')",
         "if rows!=_physical459:raise ValueError('ROOT459 exact declared physical shape')"),
        ('q5groups=24','q5groups=_groups459'),
        ('tuple(range(5,121,5))','tuple(range(5,_physical459+1,5))'),
    ),dict(_physical459=physical_M,_groups459=physical_M//5))

def pack(engine,owners,positions,tokens,*,rows,reference=False):
    return framed(rows)(engine,owners,positions,tokens,rows=rows,reference=reference)
