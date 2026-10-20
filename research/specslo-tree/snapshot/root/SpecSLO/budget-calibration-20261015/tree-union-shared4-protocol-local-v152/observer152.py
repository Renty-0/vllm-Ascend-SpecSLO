"""Immediate original hidden snapshots plus the actual graph-external F head."""
from scope152 import require
import observer151

def run(e,method):
 model=e.model;original=model.compute_greedy_tokens;had='compute_greedy_tokens' in vars(model);prior=vars(model).get('compute_greedy_tokens');heads=[]
 def head(hidden,vocabulary_size):
  value=original(hidden,vocabulary_size)
  heads.append(dict(M=int(hidden.shape[0]),tokens=value.detach().clone()))
  return value
 model.compute_greedy_tokens=head
 try:
  out,seen=observer151.run(e,method);seen['union_head']=heads
  return out,seen
 finally:
  if had:model.compute_greedy_tokens=prior
  else:vars(model).pop('compute_greedy_tokens',None)
