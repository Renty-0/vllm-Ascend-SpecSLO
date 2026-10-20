"""HOST admission of a new, separately qualified live execution domain.

Empty capabilities preserve V148. This descriptor cannot qualify Native math.
Actual component gates must populate it before the formal generation starts.
"""
from dataclasses import dataclass

BUCKETS=(1,2,4,8,12,16,20,24,28,32,36,40,44,48,64)

def require(ok,message):
    if ok is not True:raise RuntimeError('V159: '+message)

def bucket(rows):
    require(type(rows)is int and 1<=rows<=64,'bounded real row count')
    return next(n for n in BUCKETS if n>=rows)

@dataclass(frozen=True)
class Capability:
    # Pairs are real (normal,eager) counts, not merely physical M: each pair
    # needs actual shared4/fullphysical/canonical8/parent3+child3 lifecycle gates.
    shared4_pairs:frozenset=frozenset()
    # Pairs are real (frontier,one-token-repair) counts. This retains interior
    # padding/query-partition scope, rather than qualifying just U or a count.
    union_pairs:frozenset=frozenset()
    mode:str='disabled'
    def __post_init__(self):
        require(self.mode in ('disabled','qualifying','production'),'explicit mode')
        require(type(self.shared4_pairs)is frozenset and type(self.union_pairs)is frozenset,'immutable pairs')
        for pairs in (self.shared4_pairs,self.union_pairs):
            require(all(type(p)is tuple and len(p)==2 and all(type(v)is int and 1<=v<=32 for v in p) for p in pairs),'exact real-row pairs')
        require(self.mode!='disabled' or not self.shared4_pairs and not self.union_pairs,'disabled has no enabled pairs')
    def allows_shared4(self,normal,eager):
        return self.mode!='disabled' and (normal,eager) in self.shared4_pairs
    def allows_union(self,eager,repair):
        return self.mode!='disabled' and (eager,repair) in self.union_pairs

def native_ready(engine,capability,*,chain_M=None,real_rows=None,union_M=None):
    """Read authoritative original dictionaries; never assign Native flags.

    Qualification mode deliberately allows actual cold captures. Production
    requires actual closed entries; component same-M/protocol proof is separate.
    """
    require(type(capability)is Capability,'current capability descriptor')
    if capability.mode=='qualifying':return True
    if capability.mode!='production':return False
    runner=engine.graph_runner
    if chain_M is not None:
        key=(f'draft-greedy:{engine.draft_vocab_size}|steps:4|paged|confidence',chain_M)
        entry=runner.draft_entries.get(key)
        if not (entry is not None and entry.runtime_validated is True and entry.validated_real_row_count>=real_rows):return False
    if union_M is not None:
        key=('target-hidden:0|steps:1|paged',(union_M,))
        entry=runner.target_entries.get(key)
        if not (entry is not None and entry.runtime_validated is True):return False
    return True
