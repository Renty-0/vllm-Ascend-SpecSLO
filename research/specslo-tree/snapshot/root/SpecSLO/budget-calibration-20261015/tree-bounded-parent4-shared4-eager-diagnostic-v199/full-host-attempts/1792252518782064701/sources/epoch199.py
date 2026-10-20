"""Original fresh-prefix/model ABI with real role scratch and derived pages."""
import copy
from base199 import V151,functions,require
from scope199 import digest
functions(V151/'source151.py',globals(),names=('arrays','slots','read','query_KV','logical_KV','protect','ledger','rebuild'))
# Replace allocator-boundary definitions, retaining actual per-owner prepare/model/
# finite common lanes of source151.rebuild by an exact small source transformation.
def scratch_ids(e):return list(range(64,102))if e.is_draft else list(e._tree_graph_scratch_sequence_ids)
def lease(e):
 tables=e.cache_allocation.block_tables;owners=list(range(64))+scratch_ids(e);pages=[p for i in owners for p in tables[i]if p>=0]
 require(e._cache_storage_sequence_capacity==527 and len(tables)==527 and len(pages)==len(set(pages))and all(0<=p<544 for p in pages),'actual full527 unique independently allocated pages/544 pool')
 return dict(source_epoch=e._v151_source_epoch,storage=527,private_scratch=scratch_ids(e),real_owners=list(range(64)),actual_block_tables={str(i):list(tables[i])for i in owners},actual_unique_pages=len(pages),whole_pool_claimed=False)
functions(V151/'source151.py',globals(),names=('rebuild',),changes=(
 ('for owner in range(64,102):e._activate_cache_sequence(owner,[0],enable_prefix_caching=False)','for owner in scratch_ids(e):e._activate_cache_sequence(owner,[0],enable_prefix_caching=False)',1),
 ('e._ensure_cache_capacity(list(range(64)),[len(v)+8 for v in known])','e._ensure_cache_capacity(list(range(64)),[len(v)+32 for v in known])',1),
 ('for owner in range(64,102):e._ensure_cache_capacity([owner],[8])','for owner in scratch_ids(e):e._ensure_cache_capacity([owner],[32])',1)))

def tensor_hash(value):
 import hashlib,torch
 return dict(dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
