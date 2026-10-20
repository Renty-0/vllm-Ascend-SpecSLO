"""Independent real prefix/parent banks; teacher-forced normal suffix fixture."""
import copy
import hashlib
from scope_v149 import require,digest,owner_plan
from reuse_v149 import contexts,kv_helpers,math_helpers

def tensor_hash(value):
 import torch
 return hashlib.sha256(value.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

def finite(values):
 import torch
 require(all(bool(torch.isfinite(v).all()) for v in values),'all actual snapshot elements finite')

class Bank:
 def __init__(self,e,case,changed):
  self.e,self.case,self.changed=e,case,changed
  self.contexts=contexts(case,changed);self.kv=kv_helpers();self.math=math_helpers()
  require(e.cache_allocation is None,'fresh simultaneous bank')
  prefixes=[list(c['token_ids'][:128]) for c in self.contexts]
  e._allocate_cache(prefixes,enable_prefix_caching=False,reserve_sequence_capacity=True)
  for owner,tokens in zip(case.reference_ids,prefixes,strict=True):
   e._activate_cache_sequence(owner,tokens,enable_prefix_caching=False)
  for owner in (*case.candidate_ids,*case.reference_ids):e._ensure_cache_capacity([owner],[136])
  require(e._cache_storage_sequence_capacity==128 and len(e.cache_allocation.block_tables)==128
   and tuple(e._tree_graph_scratch_sequence_ids)==() and tuple(e._mixed_target_scratch_sequence_ids)==(),
   'actual original V134 storage128/no implicit scratch reuse')
  self.allocation=e.cache_allocation;self.tables=copy.deepcopy(e.cache_allocation.block_tables)
  self.tables_sha256=digest(self.tables)
  self.pages=[x for table in self.tables for x in table if x>=0]
  require(len(self.pages)==len(set(self.pages))==4*case.count
   and all(0<=p<544 for p in self.pages),'actual distinct2pages per participating owner within544')
  self.device_table_hash=tensor_hash(e.cache_block_tables)
  self.prefixes={owner:prefixes[i] for i,owner in enumerate(case.candidate_ids)}
  self.prefixes.update({owner:prefixes[i] for i,owner in enumerate(case.reference_ids)})
  e._dynamic_tree_known_KV={owner:dict(safe_before=128,tokens={}) for owner in case.candidate_ids}
  self.parents=None;self.protected=None

 def guard_host(self):
  require(self.e.cache_allocation is self.allocation
   and self.e.cache_allocation.block_tables==self.tables,'current actual allocation lease unchanged')
  return dict(allocation_id=id(self.allocation),tables_sha256=self.tables_sha256,pages=len(self.pages))

 def prefix_hash(self,histories):
  import torch
  indices=torch.cat([self.kv['slots'](self.e,owner,range(len(tokens))) for owner,tokens in histories.items()])
  values=self.kv['read'](self.e,indices);finite(values)
  return dict(tokens=digest(histories),KV=[tensor_hash(v) for v in values])

 def guard_full(self):
  result=self.guard_host()
  require(tensor_hash(self.e.cache_block_tables)==self.device_table_hash,'actual device tables match reserved lease')
  if self.protected is not None:
   require(self.prefix_hash(self.protected_histories)==self.protected,'independent real prefixes and actual parent KV protected')
  return result

 def prefill(self,owner):
  tokens=self.prefixes[owner]
  value=self.e._run_packed_hidden(tokens,[owner]*len(tokens),list(range(len(tokens))),
   use_aclgraph=False,use_fused_infer_attention=True).clone()
  finite((value,))
  return dict(owner=owner,tokens=len(tokens),token_sha256=digest(tokens),actual_eager_forward=True,candidate_KV_copied=False)

 def indices(self,owners,positions):
  import torch
  return torch.cat([self.kv['slots'](self.e,owner,[position]) for owner,position in zip(owners,positions,strict=True)])

 def query_indices(self,offset=0):
  ids=[offset+i for i in range(self.case.count)]
  return self.indices(ids,[128]*self.case.normal+[131]*self.case.eager)

 def query(self,offset=0):return self.kv['read'](self.e,self.query_indices(offset))

 def capture_initial_queries(self):
  self.initial_queries={offset:self.query(offset) for offset in (0,64)}

 def restore_queries(self,offset=0):
  index=self.query_indices(offset)
  for target,value in zip(self.kv['arrays'](self.e),self.initial_queries[offset],strict=True):target.index_copy_(0,index,value)
  return dict(owner_offset=offset,real_write_rows=self.case.count,source='same-bank-own-prelude-initial-query-snapshot',candidate_to_reference=False)

def build_bank(e,case,changed,stage):
 bank=stage('allocate-independent-real-banks',lambda:Bank(e,case,changed) if e.is_draft else None)
 records=[]
 for owner in (*case.candidate_ids,*case.reference_ids):
  value=stage(f'independent-prefill-owner{owner}',lambda:bank.prefill(owner) if e.is_draft else None)
  if e.is_draft:records.append(value)
 def done():
  if not e.is_draft:return None
  bank.guard_full();bank.initial_prefix=bank.prefix_hash(bank.prefixes)
  return dict(plan=owner_plan(case),actual_lease=bank.guard_host(),prefix=bank.initial_prefix,
   actual_leased_block_tables={str(i):bank.tables[i] for i in (*case.candidate_ids,*case.reference_ids)},
   device_table_hash=bank.device_table_hash,actual_prefill=records,
   teacher_forced_normal_suffix=True,target_acceptance_observed=False,natural_commit_claim=False)
 proof=stage('complete-independent-prefixes-and-leases',done)
 return bank,proof

def build_parents(e,native,helpers,bank,stage):
 """Actual FastV5 B3 parent versus new independent same-M PA recurrence."""
 import torch
 from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan,pack_selected_tree_plan,tree_primary_path
 case=bank.case
 ids=list(range(case.normal,case.count));ref=[64+i for i in ids]
 roots=[bank.contexts[i]['token_ids'][-1] for i in ids]
 plans=[cached_cpu_tree_speculation_plan(2,4,128,4096,candidate_budget=3) for _ in ids]
 def snapshot():
  key=e.graph_runner.last_draft_entry_key
  hidden,_,_=helpers['tail'].snapshot(e.graph_runner,key,3)
  encoded=e.graph_runner.draft_entries[key].output.detach().clone()
  query=bank.kv['read'](e,torch.cat([bank.kv['slots'](e,i,range(128,131)) for i in ids]))
  return key,hidden,encoded,query
 cold=e.draft_tree_forward(plans,roots,ids);cold_key,cold_hidden,cold_encoded,cold_KV=snapshot()
 changed_roots=list(roots);changed_roots[0]=(changed_roots[0]+1)%e.draft_vocab_size
 e.draft_tree_forward(plans,changed_roots,ids) # unpublished original Native changed-input control
 output=e.draft_tree_forward(plans,roots,ids)
 key,hidden,encoded,parent_KV=snapshot();matcher=e.graph_runner._target_outputs_match
 require(key==cold_key and cold['draft_token_ids']==output['draft_token_ids']
  and torch.equal(cold_encoded,encoded) and matcher(cold_hidden,hidden) and matcher(cold_KV,parent_KV)
  and e.graph_runner.draft_entries[key].runtime_validated is True
  and e.graph_runner.draft_entries[key].validated_real_row_count>=case.eager,
  'parent original Native cold/changed/restored control and true same-real-row gate')
 finite((*cold_hidden,*cold_KV))
 reference_encoded,reference_hidden=bank.math['chain'](e,ref,[128]*case.eager,roots,3,
  native._next_linear_draft_graph_bucket(case.eager,64))
 reference_KV=bank.kv['read'](e,torch.cat([bank.kv['slots'](e,i,range(128,131)) for i in ref]))
 reference_rows,reference_scores=bank.math['decode'](reference_encoded,case.eager,3)
 finite((*hidden,*reference_hidden,*parent_KV,*reference_KV))
 require(matcher(hidden,reference_hidden) and torch.equal(encoded,reference_encoded)
  and len(parent_KV)==len(reference_KV)==56 and matcher(parent_KV,reference_KV),
  'true recursive parentB3 same-M allphysical hidden/encoded/56KV independent PA reference')
 require(output['draft_token_ids']==reference_rows,'original parent canonical8 rows vs own reference')
 for row,scores in enumerate(reference_scores):
  require(torch.equal(output['node_confidences'][row].cpu(),torch.tensor(scores,
   dtype=output['node_confidences'][row].dtype)),'exact parent scores from independent recurrence')
 packed=[pack_selected_tree_plan(p,[0,1,2]) for p in plans]
 children=[e._spec_rhythm_tree_eager_plan(p,type('LogicalPrefix',(),{'token_ids':bank.contexts[i]['token_ids']})(),3)
  for p,i in zip(packed,ids,strict=True)]
 require(all(tuple(tree_primary_path(p))==(0,1,2) for p in packed)
  and all(p.prefix_len==132 and p.candidate_budget==3 for p in children),'actual original parentB3 childB3 prefix132')
 repair=[];frontier=[];ref_repair=[];ref_frontier=[]
 for i in range(case.normal):
  tokens=bank.contexts[i]['token_ids'];start,missing=helpers['fast'].missing_committed_suffix(
   e._dynamic_tree_known_KV[i],129,tokens[128:])
  require(start==128 and missing==[tokens[128]],'teacher-forced suffix fixture really needs exactly one materialization query')
  repair.append((i,start,missing,tokens[129]));ref_repair.append((64+i,start,list(missing),tokens[129]))
 for row,i in enumerate(ids):
  leaf=output['draft_token_ids'][row][2];ref_leaf=reference_rows[row][2]
  frontier.append((i,131,leaf));ref_frontier.append((64+i,131,ref_leaf))
  # This records actual independent reference recurrence, never candidate KV.
  for owner in (i,64+i):bank.prefixes[owner]=bank.prefixes[owner]+[roots[row],*reference_rows[row][:2]]
 bank.protected_histories=copy.deepcopy(bank.prefixes);bank.protected=bank.prefix_hash(bank.protected_histories)
 bank.parents=dict(repairs=repair,frontiers=frontier,reference_repairs=ref_repair,reference_frontiers=ref_frontier,
  original_parent_rows=output['draft_token_ids'],parent_score8=[v.cpu().tolist() for v in output['node_confidences']],
  child_prefixes=[p.prefix_len for p in children],children_not_executed=True)
 bank.capture_initial_queries()
 return dict(actual_parent_B3_recursive=True,independent_reference_same_M=True,allphysical_hidden=True,
  exact_parent_encoded=True,all56_query_KV=True,parent_key=repr(key),
  parent_Native_runtime_validated=bool(e.graph_runner.draft_entries[key].runtime_validated),
  real_query_positions=[128,129,130],original_child_B3_prefixes=bank.parents['child_prefixes'],
  child_chain_executed=False,target_acceptance_observed=False,parent_rows=bank.parents['original_parent_rows'],
  parent_scores=bank.parents['parent_score8'],parent_encoded_sha256=digest(encoded.cpu().tolist()))
