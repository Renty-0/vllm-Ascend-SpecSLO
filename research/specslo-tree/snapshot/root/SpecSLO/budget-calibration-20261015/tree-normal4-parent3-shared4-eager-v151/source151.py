"""Fresh actual source histories; every TP model call has common lanes."""
import copy
from scope151 import require,digest

def arrays(e):
 return [getattr(layer.self_attn,name).flatten(0,1) for layer in e.model.layers for name in ('key_cache','value_cache')]
def slots(e,owner,positions):
 import torch
 table=e.cache_allocation.block_tables[owner];values=[table[p//128]*128+p%128 for p in positions]
 require(len(values)==len(set(values)) and all(v>=0 for v in values),'real distinct allocated source slots')
 return torch.tensor(values,dtype=torch.long,device=e.device)
def read(e,index):return tuple(v.index_select(0,index).clone() for v in arrays(e))
def query_KV(e,metadata):return read(e,metadata.slot_mapping.long())
def logical_KV(e,starts,lengths):
 import torch
 return read(e,torch.cat([slots(e,i,range(p,p+n)) for i,(p,n) in enumerate(zip(starts,lengths,strict=True))]))
def protect(e,lengths):
 import torch,hashlib
 index=torch.cat([slots(e,i,range(p)) for i,p in enumerate(lengths)])
 values=[v.index_select(0,index) for v in arrays(e)]
 require(all(bool(torch.isfinite(v).all()) for v in values),'all actual used source-prefix KV finite')
 return [hashlib.sha256(v.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest() for v in values]
def ledger(e):return copy.deepcopy(e._dynamic_tree_known_KV)
def lease(e):
 tables=e.cache_allocation.block_tables;owners=list(range(64))+list(range(64,102));pages=[p for i in owners for p in tables[i] if p>=0]
 require(e._cache_storage_sequence_capacity==128 and len(tables)==128 and len(pages)==len(set(pages))==166
  and all(p<544 for p in pages),'actual128 storage/166 independent real+scratch pages/544 pool')
 return dict(source_epoch=e._v151_source_epoch,storage=128,private_scratch=list(range(64,102)),real_owners=list(range(64)),
  actual_block_tables={str(i):list(tables[i]) for i in owners},actual_unique_pages=166,full527_qualified=False)
def rebuild(e,contexts,repairs,stage,*,role='all'):
 import torch
 active=role=='all' or (role=='draft')==e.is_draft
 require(role in ('all','draft','target') and len(contexts)==len(repairs)==64,'exact64 source initialization scope')
 known=[list(c['token_ids'][:-1-(r if e.is_draft else 0)]) for c,r in zip(contexts,repairs,strict=True)]
 def allocate():
  if not active:return None
  require(all(len(v)==128 or len(v)==129 for v in known),'only original source prefixes; no teacher-forced intermediate history')
  e._release_cache();e._allocate_cache(known,enable_prefix_caching=False,reserve_sequence_capacity=True)
  e._dynamic_tree_known_KV={};e._v151_source_epoch=getattr(e,'_v151_source_epoch',0)+1
  for owner in range(64,102):e._activate_cache_sequence(owner,[0],enable_prefix_caching=False)
  e._ensure_cache_capacity(list(range(64)),[len(v)+8 for v in known])
  for owner in range(64,102):e._ensure_cache_capacity([owner],[8])
  return lease(e)
 evidence=stage('fresh-'+role+'-actual-allocator-lease',allocate)
 for owner,tokens in enumerate(known):
  def prepare(owner=owner,tokens=tokens):
   if not active:return None
   x=torch.tensor(tokens,dtype=torch.long,device=e.device);pos,meta=e._prepare_attention_metadata([owner]*len(tokens),list(range(len(tokens))),True)
   return x,pos,meta
  args=stage('source-owner'+str(owner)+'/prepare',prepare)
  hidden=stage('source-owner'+str(owner)+'/original-model-immediate-clone',lambda:e.model(*args).clone() if active else None)
  stage('source-owner'+str(owner)+'/finite-drain',lambda:require(bool(torch.isfinite(hidden).all()),'actual independent initial hidden finite') if active else None)
 def finish():
  if not active:return None
  e._dynamic_tree_known_KV={i:dict(safe_before=len(v),tokens={}) for i,v in enumerate(known)}
  e._v151_source_lease=lease(e)
  return dict(lease=e._v151_source_lease,initial_prefix_lengths=list(map(len,known)),initial_prefix_token_sha256=[digest(v) for v in known])
 return stage('fresh-'+role+'-source-prefix-ledger',finish)
