"""Only N32/E32, true partial repair owners, parent3/shared4/child3."""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent;B=P.parent
ACTION='union-shared4-protocol-local'
NORMAL_B=(4,)*19+(3,)+(2,)*5+(1,)*7
MC2_ROWS=(40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,116,120,124,128,165,197)
NATIVE_COUNTERS=('aclgraph_captures','aclgraph_capture_attempts','aclgraph_runtime_validation_replays','aclgraph_failed_captures','aclgraph_capacity_fallbacks','aclgraph_shape_fallbacks',*(f'aclgraph_{r}_{f}' for r in ('generic','draft','target') for f in ('eager_fallback_calls','disabled_entry_calls','unclassified_calls','runtime_validation_calls','runtime_validation_failures')))
FAILURES=tuple(k for k in NATIVE_COUNTERS if any(v in k for v in ('failed','fallback','disabled','unclassified','validation_failures')))
def require(value,message):
 if value is not True:raise RuntimeError('V152: '+message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
@dataclass(frozen=True)
class Case:
 repair_count:int
 normal:int=32
 eager:int=32
 prefix:int=128
 parent_B:int=3
 child_B:int=3
 normal_budgets:tuple=NORMAL_B
 def __post_init__(self):
  require(type(self.repair_count)is int and self.repair_count in (0,1,16,32) and self.normal==self.eager==32 and self.prefix==128 and self.parent_B==self.child_B==3 and self.normal_budgets==NORMAL_B,'exact four controlled partial-repair scopes')
 @property
 def repair(self):return int(bool(self.repair_count)) # Only original provider length availability check.
 @property
 def repair_owners(self):return tuple(range(self.repair_count))
 @property
 def repair_vector(self):return tuple(int(i in self.repair_owners) for i in range(64))
 @property
 def name(self):return 'N32E32-parent3-normal4-child3-union-R'+str(self.repair_count)
 @property
 def count(self):return 64
 @property
 def selected(self):return 192
 @property
 def parent_ids(self):return tuple(range(32,64))
 @property
 def deferred_normal(self):return tuple(range(32))
 @property
 def work_budgets(self):return (*self.normal_budgets,*([3]*32))
 @property
 def union_M(self):return bucket(32+self.repair_count) if self.repair_count else None
def cases():return tuple(Case(n) for n in (0,1,16,32))
def bucket(rows):
 require(type(rows)is int and 1<=rows<=64,'real bounded physical row count')
 return next(n for n in (1,2,4,8,12,16,20,24,28,32,36,40,44,48,64) if n>=rows)
def target_shape(budgets):
 import reuse152
 from originals151 import original_module
 v=tuple(budgets);n=len(v);c=sum(v);q=n+c
 require(1<=n<=32 and all(type(x)is int and 1<=x<=4 for x in v) and c<=96 and q<=128,'independent target C96/Q128/N32')
 m,s,pad=original_module('target_layout').envelope(n,q,'causal')
 return dict(budgets=list(v),N=n,C=c,real_Q=q,M=m,S=s,scratch=pad,family='causal')
def source_contract():
 return dict(cases=[dict(repair_count=c.repair_count,repair_owners=list(c.repair_owners),repair_vector=list(c.repair_vector),E=32,F=32,U=c.union_M,normal_budgets=list(NORMAL_B),normal_C=96,child_C=96,parent_B=3,child_B=3,shared_steps=4,chain_M=64,raw_nodes=512) for c in cases()],teacher_forced_initial_normal_suffix=True,initial_suffix_is_natural_acceptance=False,actual_target_outcomes_only=True,original_production_gates_opened=False,P151_actual_dependency_pending=True,qualification_inherited=False,dynamic_prefix_cost_enabled=False,dynamic_TTU_proved=False,fixed_prefix_counters_only=True,diagnostic_guard_full_device_table_D2H_hash=True,production_cheap_guard_qualified=False,P149_cost_inherited=False,full_workload_qualified=False,adopted=False,formal_Goodput=False)
def make_config():
 import reuse152
 from originals151 import original_function
 return original_function('short_scope_v126','make_config')()
