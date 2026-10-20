"""Finite new parent4 local domain, retaining typed C96/Q128 and original M128."""
from dataclasses import dataclass
from base169 import P,require,sha
from scope151 import bucket,target_shape,NATIVE_COUNTERS,FAILURES,digest
@dataclass(frozen=True)
class Case:
 prefix:int=128
 normal:int=16
 eager:int=16
 parent_B:int=4
 child_B:int=3
 repair_count:int=1
 normal_budgets:tuple=(4,)*16
 def __post_init__(self):
  require((self.prefix,self.normal,self.eager,self.parent_B,self.child_B)==(128,16,16,4,3) and self.repair_count in (0,1,16)
   and self.normal_budgets==(4,)*16,'only new finite N16/E16/parent4/child3/R1')
 @property
 def count(self):return 32
 @property
 def repair(self):return 1
 @property
 def repair_owners(self):return tuple(range(self.repair_count))
 @property
 def repair_vector(self):return (1,)*self.repair_count+(0,)*(64-self.repair_count)
 @property
 def union_M(self):return bucket(16+self.repair_count)
 @property
 def work_budgets(self):return (4,)*16+(3,)*16
 @property
 def name(self):return 'N16E16-parent4-child3-R1-U20-shared4-M32'
def case():return Case()
def contract():
 return dict(normal=16,eager=16,parent_B=4,child_B=3,parent_C=64,parent_Q=80,normal_C=64,normal_Q=80,
  child_C=48,child_Q=64,selected_draft=112,global_draft_roof=192,target_roof=dict(C=96,Q=128,N=32),
  prefix=128,parent_frontier_position=132,child_prefix=133,parent_chain_steps=4,parent_chain_M=16,
  shared_steps=4,shared_chain_M=32,frontier_head_M=16,union_M=20,R=1,original_target_M=128,original_target_S=38,
  actual_requests=32,logical_source_owners=64,real_candidate_prefixes=8,old_negative_prefixes=8,
  old_raw5_stream_is_only_fixture_selection=True,new_positive_assumed=False,force_acceptance=False,
  two_complete_zeros=False,full_qualification=False,production_gates_opened=False,formal_Goodput=False)

from dataclasses import field
_ISSUER=object()
@dataclass(frozen=True)
class StandaloneScope:
 _engine:object=field(repr=False)
 _proof:object=field(repr=False)
 _issuer:object=field(repr=False)
 def require(self,engine):
  require(self._issuer is _ISSUER and engine is self._engine
   and getattr(engine,'_v169_qualified_component',None) is self._proof
   and self._proof['audit']['component_numerical_qualified'] is True,
   'same current actual standalone component identity')
  return frozenset()
def issue(engine,*,component_proof=None):
 expected=dict(entries=0,draft_entries=60,target_entries=54) if engine.is_draft else dict(entries=0,draft_entries=0,target_entries=245)
 require(component_proof is not None and getattr(engine,'_v169_qualified_component',None) is component_proof
  and getattr(engine,'_v169_actual_startup',None) is not None
  and engine.device.type=='npu' and engine._v169_actual_startup['actual_native_entry_counts']==expected
  and engine.cache_allocation is None and engine.cache_block_tables is None
  and engine.graph_runner.graph_cache_sealed is False,'own actual startup/component released before scope')
 result=StandaloneScope(engine,component_proof,_ISSUER);result.require(engine)
 require(getattr(engine,'_v169_standalone_scope',None) is None,'issue current-engine scope once')
 engine._v169_standalone_scope=result;return result
def report(scope,engine):
 require(scope.require(engine)==frozenset(),'standalone never authorizes mixed N')
 return dict(schema='V169.actual-standalone-scope.v1',N=[],mixed_reservation_allowed=False,
  original_S35_scope_claimed=False,old_21scope26phase_executed=False,
  actual_current_startup=True,actual_current_shared4_union_component=True,standalone_whole_prompt_only=True,
  arbitrary_token_AR_qualified=False,formal_Goodput=False)
