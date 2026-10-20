"""Single synchronous draft-call lease; no device reads or cross-round cache.

Native reserve/ensure must finish first. During the lease, other cache lifecycle
mutators are rejected before execution, and ensure may not change allocation or
tables. All guards are HOST identities/values; no GPU table hash or D2H.
"""
from contextlib import contextmanager
from scope159 import require

def _ledger(engine,owners):
    known=getattr(engine,'_dynamic_tree_known_KV',{})
    require(all(i in known for i in owners),'tracked source owners')
    return tuple((i,known[i]['safe_before'],tuple(sorted(known[i]['tokens'].items()))) for i in owners)

class DraftLease:
    def __init__(self,engine,owners,witness_type):
        self.e=engine;self.owners=tuple(owners)
        require(len(self.owners)==len(set(self.owners)) and bool(self.owners),'unique participating owners')
        self.allocation=engine.cache_allocation;self.device_table=engine.cache_block_tables
        require(self.allocation is not None and self.device_table is not None,'live allocation/table')
        self.generation=getattr(engine,'_dynamic_tree_generation',0)
        self.rows=tuple((i,tuple(self.allocation.block_tables[i])) for i in self.owners)
        self.ledger=_ledger(engine,self.owners)
        # Sparse HOST witness: only participating owners are copied; empty rows
        # preserve logical indexing. Never copy/hash the device block table.
        tables=[()]*(max(self.owners)+1)
        for i,row in self.rows:tables[i]=row
        self.witness=witness_type(id(self.allocation),engine.config.kvcache_block_size,tuple(tables))
    def check(self):
        e=self.e
        require(e.cache_allocation is self.allocation and e.cache_block_tables is self.device_table,'allocation/storage changed inside synchronous prelude')
        require(getattr(e,'_dynamic_tree_generation',0)==self.generation,'generation changed inside prelude')
        require(all(tuple(self.allocation.block_tables[i])==row for i,row in self.rows),'participating HOST pages changed')
        require(_ledger(e,self.owners)==self.ledger,'source ledger changed before successful frontier commit')
        return self.witness

@contextmanager
def guarded(engine,owners,witness_type):
    """Restore every per-instance method even on failure; no stale-KV fallback."""
    lease=DraftLease(engine,owners,witness_type);saved={}
    def bind(name,value):
        saved[name]=(name in engine.__dict__,engine.__dict__.get(name));setattr(engine,name,value)
    def forbidden(*args,**kwargs):raise RuntimeError('V159: cache lifecycle mutation inside synchronous prelude')
    original_ensure=engine._ensure_cache_capacity
    def ensure(*args,**kwargs):
        lease.check();out=original_ensure(*args,**kwargs);lease.check();return out
    try:
        for name in ('_allocate_cache','_release_cache','_activate_cache_sequence','_release_cache_sequence'):
            if hasattr(engine,name):bind(name,forbidden)
        bind('_ensure_cache_capacity',ensure)
        yield lease
        lease.check()
    finally:
        for name,(had,value) in saved.items():
            if had:setattr(engine,name,value)
            else:engine.__dict__.pop(name,None)
