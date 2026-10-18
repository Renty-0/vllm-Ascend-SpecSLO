"""Same current173 qualification plus this instance's multiowner+next4 proof."""
from dataclasses import dataclass,field
from base190 import require,parent_module
_ISSUER=object()
@dataclass(frozen=True)
class StandaloneScope:
 engine:object=field(repr=False)
 proof:object=field(repr=False)
 original:object=field(repr=False)
 issuer:object=field(repr=False)
 def require(self,engine):
  require(self.issuer is _ISSUER and engine is self.engine
   and self.proof is getattr(engine,'_v190_qualified_component',None)
   and self.proof['multiowner_prefill']is getattr(engine,'_v190_actual_prefill',None)
   and self.proof['multiowner_prefill']['audit']['passed_current_multiowner_math']is True,
   'same current independent multiowner+next4 proof')
  require(self.original.require(engine)==frozenset(),'original173 standalone domain')
  return frozenset()
def issue(engine,*,component_proof):
 require(component_proof is getattr(engine,'_v190_qualified_component',None),'own same proof')
 original=parent_module('scope173').issue(engine,component_proof=engine._v173_qualified_component)
 result=StandaloneScope(engine,component_proof,original,_ISSUER);result.require(engine)
 require(not hasattr(engine,'_v190_scope'),'issue once');engine._v190_scope=result;return result
def report(scope,engine):
 scope.require(engine);old=parent_module('scope173').report(scope.original,engine)
 return dict(old,schema='V190.current-cap4-vacancy-cohort-scope.v1',production_parent173=True,
  current_multiowner_eager_math=True,actual_next_draft_steps=4,representative_cohort_sizes=[2,4],
  production_cohort_limit=32,all_prompt_tuples_numerically_exhaustive=False,
  minimum_requests=4,maximum_wait_ms=150.,minimum_is_not_maximum=True)
