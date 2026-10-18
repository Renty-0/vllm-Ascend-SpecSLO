"""Simultaneous independently prefetched owner banks; no candidate KV transfer."""
from scope_v146 import require,digest

def _arrays(engine):
 for layer in engine.model.layers:
  for name in ('key_cache','value_cache'):
   raw=getattr(layer.self_attn,name);view=raw.flatten(0,1)
   require(raw.data_ptr()==view.data_ptr(),'write-through KV storage')
   yield view

def slot_values(engine,owner,positions):
 table=engine.cache_allocation.block_tables[owner];size=engine.config.kvcache_block_size
 values=[]
 for position in positions:
  page=table[position//size];require(type(page) is int and 0<=page<544,'allocated actual page')
  values.append(page*size+position%size)
 return values

def _slots(engine,owner,positions):
 import torch
 return torch.tensor(slot_values(engine,owner,positions),dtype=torch.long,device=engine.device)

def tensor_hash(value):
 import hashlib,torch
 return hashlib.sha256(value.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

def prefix_evidence(engine,histories):
 import torch
 index=torch.cat([_slots(engine,i,range(len(v))) for i,v in histories.items()])
 return dict(tokens=digest({i:list(v) for i,v in histories.items()}),slots=tensor_hash(index),
  KV=[tensor_hash(a.index_select(0,index)) for a in _arrays(engine)])

def query_KV(engine,indices):return tuple(a.index_select(0,indices).clone() for a in _arrays(engine))

class DualBank:
 def __init__(self,engine,case):
  import torch,copy
  self.engine,self.case=engine,case
  require(engine.cache_allocation is None,'fresh dual-bank allocation')
  prompts=[list(o['token_ids'][:-1]) for o in case['owners']]
  engine._allocate_cache(prompts,enable_prefix_caching=False,reserve_sequence_capacity=True)
  for i,tokens in enumerate(prompts):engine._activate_cache_sequence(32+i,tokens,enable_prefix_caching=False)
  for i in range(463,475):engine._activate_cache_sequence(i,[0],enable_prefix_caching=False)
  self.prefill=[];self.prompts=prompts
  for owner in range(64):engine._ensure_cache_capacity([owner],[len(prompts[owner%32])+23])
  for owner in range(463,475):engine._ensure_cache_capacity([owner],[8])
  self.contexts=[list(o['token_ids']) for o in case['owners']]
  self.reference_contexts=copy.deepcopy(self.contexts)
  self.reference_ledger={i+32:dict(safe_before=len(v)-1,tokens={}) for i,v in enumerate(self.contexts)}
  if engine.is_draft:engine._dynamic_tree_known_KV={i:dict(safe_before=len(v)-1,tokens={}) for i,v in enumerate(self.contexts)}
  self.initial_tables=copy.deepcopy(engine.cache_allocation.block_tables)
  self.allocation_id=id(engine.cache_allocation);self.device_tables=tensor_hash(engine.cache_block_tables)
  pages=[p for table in self.initial_tables for p in table if p>=0]
  require(len(pages)==140 and len(pages)==len(set(pages)),'actual140/544 unique pages with bounded future reservation')
  self.page_count=len(pages)
  self.initial_prefix=None

 def prefill_call(self,owner):
  tokens=self.prompts[owner%32]
  returned=self.engine._run_packed_hidden(tokens,[owner]*len(tokens),list(range(len(tokens))),
   use_aclgraph=False,use_fused_infer_attention=False)
  return returned.clone()

 def prefill_finite(self,owner,hidden):
  import torch
  require(bool(torch.isfinite(hidden).all()),'independently evaluated actual prefix finite before next TP call')
  tokens=self.prompts[owner%32]
  self.prefill.append(dict(owner=owner,tokens=len(tokens),sha256=digest(tokens),actual_eager_PA=True,candidate_KV_copied=False))

 def prefix_ready(self):
  require([r['owner'] for r in self.prefill]==list(range(64)),'all64 independently prefetched owners before first draft')
  self.initial_prefix=prefix_evidence(self.engine,{i:self.prompts[i%32] for i in range(64)})
  return self.guard()

 def guard(self):
  e=self.engine
  require(id(e.cache_allocation)==self.allocation_id and e.cache_allocation.block_tables==self.initial_tables
   and tensor_hash(e.cache_block_tables)==self.device_tables,'stable actual lease, no release/reallocate')
  return dict(allocation_id=self.allocation_id,tables_sha256=digest(self.initial_tables),pages=self.page_count,
   storage_rows=len(self.initial_tables),candidate_owners=list(range(32)),reference_owners=list(range(32,64)),
   dummy_owners=list(range(463,475)),source_KV_copied=False)

 def source(self):
  require(self.contexts==self.reference_contexts,'independently committed histories match')
  return dict(ids=list(range(32)),contexts=[dict(o,token_ids=list(v),initial_prefix=o['prefix'],prefix=len(v)-1,output_offset=len(v)-len(o['token_ids']))
   for o,v in zip(self.case['owners'],self.contexts,strict=True)])

 def evidence(self):return dict(lease=self.guard(),prefill=self.prefill,initial_prefix=self.initial_prefix,
  current_context_sha256=[digest(v) for v in self.contexts],initial_root_teacher_forced_not_own_AR=True)

def build_bank(engine,case,*,stage):
 """No TP call follows a role-local prefill/finite failure without an all4 vote."""
 bank=stage('allocate-dual-bank',lambda:DualBank(engine,case))
 for owner in range(64):
  hidden=stage(f'prefill-owner{owner}/original-model',lambda:bank.prefill_call(owner))
  stage(f'prefill-owner{owner}/finite',lambda:bank.prefill_finite(owner,hidden))
 stage('independent-prefixes-ready',bank.prefix_ready)
 return bank
