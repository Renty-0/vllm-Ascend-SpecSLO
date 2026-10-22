"""Original385 protected72-prefix lifecycle, actual N16 B4 frame."""
from projection425 import project_function
import probe385 as original
_run=project_function(original.run,(
    ('from original385 import run as original','from forward444 import measure as original'),
    ("c['name']=='N24-all4-anchor0'","c['name']=='N16-all4-anchor0'"),
    ("roots=bank['roots'][:24]","roots=bank['roots'][:16]"),
    ('rows,list(range(24)),[],[])','rows,list(range(16)),[],[])'),
    ('ROOT385/original128-epilogue-graph-qualifier-and-balanced-cost',
     'ROOT444/real-N16-q5-padding-prefill-math-and-full-target-cost'),
))

def run(engine,native,stage,output,fixture):
    if hasattr(engine,'_root444_fixture_prefixes'):raise RuntimeError('ROOT444 fresh fixture binding')
    engine._root444_fixture_prefixes=tuple(tuple(p)for p in fixture['prefixes'])
    try:return _run(engine,native,stage,output,fixture)
    finally:del engine._root444_fixture_prefixes
