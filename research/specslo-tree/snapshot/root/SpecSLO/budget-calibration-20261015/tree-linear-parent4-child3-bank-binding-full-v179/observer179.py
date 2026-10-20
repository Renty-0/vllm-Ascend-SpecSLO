"""Component-only observer: inspect actual noncontiguous PA padding safely.

Model input/metadata/output order is unchanged. A separate read-only view puts
all actual real rows before every dummy, for the unchanged original inspector.
Actual metadata snapshots retain the original physical order.
"""
from dataclasses import replace
from base179 import V151,B,functions,require,checked
import importlib.util,sys
# Explicit pinned diagnostic dependency; the new full worker does not inherit
# the local P151 runtime's sys.path setup. Never replace a foreign module.
_padding_path=checked(B/'tree-packed-eager01-cost-v107/padding_reference.py')
if 'padding_reference' in sys.modules:
 require(__import__('pathlib').Path(vars(sys.modules['padding_reference']).get('__file__','')).resolve()==_padding_path,
  'original padding inspector canonical source')
else:
 _spec=importlib.util.spec_from_file_location('padding_reference',_padding_path)
 _module=importlib.util.module_from_spec(_spec);sys.modules['padding_reference']=_module
 try:_spec.loader.exec_module(_module)
 except BaseException:sys.modules.pop('padding_reference',None);raise

def inspect_indices(engine,inputs,positions,metadata,real_rows):
 import torch,padding_reference
 slots=metadata.slot_mapping.detach().cpu().tolist();physical=len(slots)
 require(all(type(v)is int and v>=-1 for v in slots),'only original real slots and readonly -1 dummies')
 real=[i for i,s in enumerate(slots) if s>=0];dummy=[i for i,s in enumerate(slots) if s==-1]
 require(len(real)==real_rows and len(inputs)==physical and real and real[0]==0,'actual real-row and first source-owner geometry')
 tables=metadata.block_tables.detach().cpu().tolist();pos=positions.detach().cpu().tolist();contexts=metadata.context_lens.detach().cpu().tolist()
 witnesses=[];block=engine.config.kvcache_block_size
 for i in real:
  owners=[owner for owner,table in enumerate(engine.cache_allocation.block_tables) if list(table)==tables[i]]
  require(len(owners)==1 and pos[i]>=0 and pos[i]//block<len(tables[i])
   and tables[i][pos[i]//block]*block+pos[i]%block==slots[i] and contexts[i]==pos[i]+1,
   'every actual real row retains its own authoritative physical owner/slot/position/context')
  witnesses.append(dict(physical_row=i,owner=owners[0],position=pos[i],slot=slots[i]))
 order=real+dummy;index=torch.tensor(order,dtype=torch.long,device=engine.device)
 view=replace(metadata,slot_mapping=metadata.slot_mapping.index_select(0,index),
  context_lens=metadata.context_lens.index_select(0,index),block_tables=metadata.block_tables.index_select(0,index),
  sequence_lens=tuple(metadata.sequence_lens[i] for i in order))
 result=padding_reference.inspect(engine,inputs.index_select(0,index),positions.index_select(0,index),view,len(real))
 result.update(actual_physical_real_indices=real,actual_physical_dummy_indices=dummy,diagnostic_row_permutation=order,
  real_row_owner_witness=witnesses,model_metadata_reordered=False,diagnostic_view_only=True,
  actual_host_Q=list(metadata.actual_seq_lengths_q),actual_host_KV=list(metadata.sequence_lens))
 return result

functions(V151/'observer151.py',globals(),names=('_metadata','run'),changes=((
 'padding=padding_reference.inspect(engine,inputs,positions,meta,real_rows)',
 'padding=inspect_indices(engine,inputs,positions,meta,real_rows)',1),))

# Same actual model-head wrapper as P152, over the indexed padding observer.
_indexed_run=run

def run(e,method):
 model=e.model;original=model.compute_greedy_tokens;had='compute_greedy_tokens' in vars(model);prior=vars(model).get('compute_greedy_tokens');heads=[]
 def head(hidden,vocabulary_size):
  value=original(hidden,vocabulary_size);heads.append(dict(M=int(hidden.shape[0]),tokens=value.detach().clone()));return value
 model.compute_greedy_tokens=head
 try:
  out,seen=_indexed_run(e,method);seen['union_head']=heads;return out,seen
 finally:
  if had:model.compute_greedy_tokens=prior
  else:vars(model).pop('compute_greedy_tokens',None)
