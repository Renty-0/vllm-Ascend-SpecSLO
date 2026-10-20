"""Proposed finite normal24/parent24 E1..8 scope; no runtime permission."""
from dataclasses import dataclass
from scope199 import require,bucket,digest,NATIVE_COUNTERS,FAILURES


@dataclass(frozen=True)
class Case:
    eager:int
    repair_count:int
    prefix:int=128
    normal:int=24
    parent_B:int=4
    child_B:int=4
    def __post_init__(self):
        require(type(self.eager)is int and self.eager==4 and type(self.repair_count)is int
            and 0<=self.repair_count<=24 and (self.prefix,self.normal,self.parent_B,self.child_B)==(128,24,4,4),
            'proposed exact N24/P24 E4/R0..24 source domain')
    @property
    def eager_ids(self):
        positive=(self.eager+1)//2;negative=self.eager//2
        return tuple(range(24,24+positive))+tuple(range(32,32+negative))
    @property
    def repair_owners(self):return tuple(range(self.repair_count))
    @property
    def repair_vector(self):return (1,)*self.repair_count+(0,)*(64-self.repair_count)
    @property
    def normal_budgets(self):return (4,)*24
    @property
    def count(self):return 24+self.eager
    @property
    def union_M(self):return bucket(bucket(self.eager)+self.repair_count)if self.repair_count else bucket(self.eager)
    @property
    def name(self):return f'N24-parent24-P4-E{self.eager}-B4-R{self.repair_count}-M{bucket(self.count)}'


def cases():return tuple(Case(4,r)for r in range(25))


def contract():
    return dict(cases=[dict(name=c.name,N=c.normal,E=c.eager,R=c.repair_count,
        owners=list(c.eager_ids),parent_M=24,shared_M=bucket(c.count),F=bucket(c.eager),U=c.union_M,
        normal_C=96,eager_C=4*c.eager)for c in cases()],parent_B=4,child_B=4,
        steps=4,normalC=96,eagerC=96,globalC=192,source_only=False,finite_component_only=True,
        production_permission=False,full463=False,formal_Goodput=False,whole_AR=False)
