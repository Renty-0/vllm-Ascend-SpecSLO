"""Only a finally-scoped diagnostic may invoke new P4/E4 frontier usage."""
import sys,types
from contextlib import contextmanager
from base199 import P,V159,functions,checked,require
from scope159 import Capability,native_ready
from lease159 import guarded
_capability=Capability(frozenset((16,e)for e in(1,2,3,4)),frozenset((e,1)for e in(1,2,3,4)),mode='qualifying')
functions(V159/'live159.py',globals(),names=('fused_prelude',),changes=(('int(p.candidate_budget)!=3','int(p.candidate_budget)!=4',2),))
_TOKEN=object()
@contextmanager
def diagnostic(engine,case):
 require(not hasattr(engine,'_v199_diagnostic')and getattr(engine,'_v173_qualified_component',None)is not None,'same current173 baseline before bounded context')
 engine._v199_diagnostic=(_TOKEN,case)
 try:yield
 finally:
  require(engine._v199_diagnostic==(_TOKEN,case),'diagnostic context identity until cleanup');del engine._v199_diagnostic

def validate(engine,native,plans,ids,sources,repairs,frontiers,steps):
 value=getattr(engine,'_v199_diagnostic',None)
 require(value is not None and value[0]is _TOKEN,'no production permission; explicit current diagnostic only')
 c=value[1];normal=[p for p,i in zip(plans,ids)if i not in sources]
 require(steps==4 and len(normal)==16 and all(p.candidate_budget==4 for p in plans)and set(sources)==set(c.eager_ids)and len(frontiers)==c.eager and len(repairs)==c.repair_count and all(len(v[2])==1 for v in repairs)and all(p.candidate_budget==4 for p,row in sources.values())and native._next_linear_draft_graph_bucket(len(ids),64)==20,'exact diagnostic parent4/child4/N16/E1..4/single repair/M20')

def fast_source():
 source=checked(V159/'fast159.generated.py').read_text();old='        from live159 import fused_prelude\n';new='        from live199 import fused_prelude,validate\n        validate(engine,native,plans,ids,sources,repairs,frontiers,steps)\n'
 require(source.count(old)==1,'exact new local prelude anchor');return source.replace(old,new)

def prepare_helpers():
 require('fast_tree_draft_v5'not in sys.modules,'prepare canonical fast before root_seed capture')
 from source159 import module,FUSION
 fusion=module('repair_frontier_fusion',checked(FUSION))
 path=P/'fast199.generated.py';source=fast_source();require(path.read_text()==source,'materialized exact new FastV5 source')
 fast=types.ModuleType('fast_tree_draft_v5');fast.__file__=str(path);fast.__package__='';exec(compile(source,str(path),'exec'),vars(fast));sys.modules['fast_tree_draft_v5']=fast
 return dict(path=str(path),production_scope_opened=False)
