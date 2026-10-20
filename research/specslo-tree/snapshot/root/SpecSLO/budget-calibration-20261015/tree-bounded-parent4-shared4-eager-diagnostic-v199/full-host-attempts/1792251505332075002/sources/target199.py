"""Actual173 mapper/owners and independent fullphysical target history."""
from dataclasses import asdict
from base199 import V151,functions,require
from scope199 import target_shape
import epoch199 as epoch
functions(V151/'target151.py',globals(),names=('clone_output','metadata','own_compact','rebuild_history','qualify','original_call'),changes=(
 ("require(len(saved)==1 and out['used_aclgraph'] is True", "require(len(saved)in(1,3) and all(__import__('torch').equal(saved[0],v)for v in saved[2:]) and out['used_aclgraph'] is True",1),
 ("return clone_output(out),saved[0]","return clone_output(out),saved[-1]",1),
 ("require(bool(entry.runtime_validated)==(was or phase!='first'),'Native greedy original cold/changed/restored flags')","require(entry.runtime_validated is True,'original greedy wrapper naturally closes cold3 or warm1; no flag assignment')",1)))
def padded(e,source,ids):
 plans=[];roots=[];rows=[]
 for owner in ids:
  ticket=source['controller'].validate_verification(owner);payload=source['payloads'][ticket.proposal_id]
  require(payload['plan'].prefix_len==len(source['logical'][owner].token_ids)-1,'real planner owner current prefix/root')
  plans.append(payload['plan']);roots.append(source['logical'][owner].token_ids[-1]);rows.append(payload['row'])
 g=target_shape([p.candidate_budget for p in plans]);frame=e._pad_spec_rhythm_target_tree_graph(plans,roots,rows,list(ids),list(source['logical']),source['logical'])
 require(frame[4]==len(ids)and frame[3][:len(ids)]==list(ids)and len(frame[0])==g['S']and sum(p.candidate_budget+1 for p in frame[0])==g['M']and set(frame[3][len(ids):])<=set(e._tree_graph_scratch_sequence_ids),'actual current173 pad with truthful source owner IDs/production scratch')
 return frame
