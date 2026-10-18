"""Engine-local standalone capability; it authorizes no mixed target count."""
from dataclasses import dataclass,field
from base173 import require
_ISSUER=object()
@dataclass(frozen=True)
class StandaloneScope:
 _engine:object=field(repr=False)
 _proof:object=field(repr=False)
 _issuer:object=field(repr=False)
 def require(self,engine):
  require(self._issuer is _ISSUER and engine is self._engine
   and getattr(engine,'_v173_qualified_component',None) is self._proof
   and self._proof['audit']['component_numerical_qualified'] is True,
   'same current actual standalone component identity')
  from mapper173 import current
  current(engine)
  require(self._proof['causal80'] is getattr(engine,'_v173_actual_causal80',None),'same own new80 current proof')
  return frozenset()
def issue(engine,*,component_proof=None):
 expected=dict(entries=0,draft_entries=60,target_entries=54) if engine.is_draft else dict(entries=0,draft_entries=0,target_entries=245)
 require(component_proof is not None and getattr(engine,'_v173_qualified_component',None) is component_proof
  and getattr(engine,'_v173_actual_startup',None) is not None
  and engine.device.type=='npu' and engine._v173_actual_startup['actual_native_entry_counts']==expected
  and engine.cache_allocation is None and engine.cache_block_tables is None
  and engine.graph_runner.graph_cache_sealed is False,'own actual startup/component released before scope')
 result=StandaloneScope(engine,component_proof,_ISSUER);result.require(engine)
 require(getattr(engine,'_v173_standalone_scope',None) is None,'issue current-engine scope once')
 engine._v173_standalone_scope=result;return result
def report(scope,engine):
 require(scope.require(engine)==frozenset(),'standalone never authorizes mixed N')
 return dict(schema='V173.actual-standalone-scope.v1',N=[],mixed_reservation_allowed=False,
  original_S35_scope_claimed=False,old_21scope26phase_executed=False,
  actual_current_startup=True,actual_current_B4_component=True,standalone_whole_prompt_only=True,
  arbitrary_token_AR_qualified=False,formal_Goodput=False,adopted=False,
  actual_current_causal80_component=True,math_independent_qualified=True,
  old_padding_behavior_equal=scope._proof['old_padding_behavior_equal'],numerical_variant_only=scope._proof['numerical_variant_only'])
