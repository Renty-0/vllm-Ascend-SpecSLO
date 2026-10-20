"""Only normal variableB4 + parent3 + child3 at real shared4/N32/E32."""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent;B=P.parent
ACTION='normal4-parent3-child3-shared4-local';NORMAL_B=(4,)*19+(3,)+(2,)*5+(1,)*7
MC2_ROWS=(40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,116,120,124,128,165,197)
NATIVE_COUNTERS=('aclgraph_captures','aclgraph_capture_attempts','aclgraph_runtime_validation_replays','aclgraph_failed_captures','aclgraph_capacity_fallbacks','aclgraph_shape_fallbacks',*(f'aclgraph_{r}_{f}' for r in ('generic','draft','target') for f in ('eager_fallback_calls','disabled_entry_calls','unclassified_calls','runtime_validation_calls','runtime_validation_failures')))
FAILURES=tuple(k for k in NATIVE_COUNTERS if any(v in k for v in ('failed','fallback','disabled','unclassified','validation_failures')))
def require(value,message):
 if value is not True:raise RuntimeError('V151: '+message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
@dataclass(frozen=True)
class Case:
 repair:int
 normal:int=32
 eager:int=32
 prefix:int=128
 parent_B:int=3
 child_B:int=3
 normal_budgets:tuple=NORMAL_B
 def __post_init__(self):
  require(self.repair in (0,1) and self.normal==self.eager==32 and self.prefix==128
   and self.parent_B==self.child_B==3 and self.normal_budgets==NORMAL_B,
   'exact two issued scopes; parent4 and other configurations remain closed')
 @property
 def name(self):return 'N32E32-normal-variable4-parent3-child3-shared4-repair'+str(self.repair)
 @property
 def count(self):return 64
 @property
 def selected(self):return sum(self.normal_budgets)+3*self.eager
 @property
 def parent_ids(self):return tuple(range(32,64))
 @property
 def deferred_normal(self):return tuple(range(32))
 @property
 def work_budgets(self):return (*self.normal_budgets,*([3]*32))
def cases():return (Case(0),Case(1))
def bucket(rows):
 require(type(rows)is int and 1<=rows<=64,'actual64 row source bound')
 return next(n for n in (1,2,4,8,12,16,20,24,28,32,36,40,44,48,64) if n>=rows)
def target_shape(budgets):
 from originals151 import original_module
 values=tuple(budgets);n=len(values);c=sum(values);q=n+c
 require(1<=n<=32 and all(type(v)is int and 1<=v<=4 for v in values) and c<=96 and q<=128,'actual heterogeneous target N32/C96/Q128')
 m,s,padding=original_module('target_layout').envelope(n,q,'causal')
 return dict(budgets=list(values),N=n,C=c,real_Q=q,M=m,S=s,scratch=padding,family='causal')
def source_contract():
 return dict(cases=[dict(name=c.name,repair=c.repair,normal_budgets=list(c.normal_budgets),normal_C=96,parent_B=3,child_B=3,eager_C=96,selected=192,actual_work_rows=64,actual_shared_steps=4,generated_raw_nodes=512,returned_ABI_nodes=512) for c in cases()],
  parent4_allowed=False,forced_success_or_rejection_allowed=False,natural_positive_required_for_full_component_coverage=True,
  same_M_reference_required=True,two_complete_zero_traces_required=True,local_sequential_not_live_owner_alignment=True,
  original_window_or_floor_bypassed_in_live=False,full527_or_full463_qualified=False,adopted=False,Goodput_measured=False)
def make_config():
 from originals151 import original_function
 return original_function('short_scope_v126','make_config')()
