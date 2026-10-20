"""Naturally qualify new PA repair-hidden keys before any output is consumed.

Only unpublished +1 input and exact actual-input restoration are added. Native
entry flags, validators, attention metadata and the original method stay intact.
"""
def install(native):
 import torch
 cls=native.NativeACLGraphRunner;original=cls.run_tree_hidden
 def qualified(runner,inputs,positions,metadata):
  result=original(runner,inputs,positions,metadata)
  if (runner.graph_cache_sealed or getattr(metadata,'use_fused_infer_attention',False)
      or not getattr(runner,'_dynamic_tree_top2',False)):return result
  key=getattr(runner,'_frontier_qualification_last_key',None);entry=runner.target_entries.get(key)
  if entry is None or entry.runtime_validated:return result
  if not key[0].startswith('target-hidden:0|steps:1|paged'):raise RuntimeError('V126 repair-hidden PA key required')
  actual=inputs.clone();expected=result.clone()
  original(runner,(actual+1)%int(runner.model.lm_head.vocab_size),positions,metadata)
  restored=original(runner,actual,positions,metadata)
  if not entry.runtime_validated or not runner._target_outputs_match((expected,),(restored,)):
   raise RuntimeError('V126 original hidden PA changed-input gate/actual restoration failed')
  rows=getattr(runner,'_v126_hidden_qualification',None)
  if rows is None:rows=runner._v126_hidden_qualification=[]
  rows.append(dict(key=repr(key),original_calls=3,unpublished_changed_calls=1,actual_restore_calls=1,
   runtime_validated=True,Native_flag_assigned=False))
  return restored
 cls.run_tree_hidden=qualified
