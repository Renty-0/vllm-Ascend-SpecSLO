"""Original385 N24 bank, then true M80 ordinary-reference model diagnostic."""
from projection425 import project_function
import probe385 as original
_run=project_function(original.run,(
 ('from original385 import run as original','from whole457 import measure as original'),
 ('ROOT385/original128-epilogue-graph-qualifier-and-balanced-cost','ROOT457/actual-own-geometry80-whole-model-reference-and-cost'),
))
def run(engine,native,stage,output,fixture):
 if hasattr(engine,'_root445_fixture_prefixes'):raise RuntimeError('ROOT457 fresh fixture binding')
 engine._root445_fixture_prefixes=tuple(tuple(p)for p in fixture['prefixes'])
 try:return _run(engine,native,stage,output,fixture)
 finally:del engine._root445_fixture_prefixes
