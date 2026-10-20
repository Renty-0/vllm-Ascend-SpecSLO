"""Exact finite diagnostic scopes; no production permission is ever issued."""
from dataclasses import dataclass
from base199 import require,sha
from scope151 import NATIVE_COUNTERS,FAILURES,digest,bucket
@dataclass(frozen=True)
class Case:
 eager:int
 repair_count:int
 prefix:int=128
 normal:int=16
 parent_B:int=4
 child_B:int=4
 def __post_init__(self):
  require((self.eager,self.repair_count)in((1,0),(1,1),(2,1),(3,1),(4,1))and(self.prefix,self.normal,self.parent_B,self.child_B)==(128,16,4,4),'exact five diagnostic cases')
 @property
 def eager_ids(self):return {1:(16,),2:(16,24),3:(16,17,24),4:(16,17,24,25)}[self.eager]
 @property
 def repair_owners(self):return tuple(range(self.repair_count))
 @property
 def repair_vector(self):return (1,)*self.repair_count+(0,)*(64-self.repair_count)
 @property
 def normal_budgets(self):return(4,)*16
 @property
 def count(self):return 16+self.eager
 @property
 def union_M(self):return bucket(bucket(self.eager)+self.repair_count)if self.repair_count else bucket(self.eager)
 @property
 def name(self):return f'N16-parent16-P4-E{self.eager}-B4-R{self.repair_count}-M{bucket(self.count)}'
def cases():return tuple(Case(*v)for v in((1,0),(1,1),(2,1),(3,1),(4,1)))
def target_shape(budgets):
 from target_layout import envelope
 values=tuple(budgets);n=len(values);c=sum(values);q=n+c
 require(1<=n<=16 and all(type(v)is int and v==4 for v in values)and c<=96 and q<=128,'original max_target_batch16/current C96 and only B4')
 m,s,pad=envelope(n,q,'causal')
 return dict(budgets=list(values),N=n,C=c,real_Q=q,M=m,S=s,scratch=pad,family='causal')
def contract():
 return dict(cases=[dict(name=c.name,E=c.eager,R=c.repair_count,owners=list(c.eager_ids),parent_M=16,shared_M=bucket(c.count),F=bucket(c.eager),U=c.union_M,normal_C=64,eager_C=4*c.eager)for c in cases()],parent_B=4,child_B=4,steps=4,normalC=96,eagerC=96,globalC=192,source_only=True,production_permission=False,full463=False,formal_Goodput=False,whole_AR=False,child3_source_contrast='same shared4 computes/writes e3 already; B4 leaves e4 to original missing_committed_suffix repair')
