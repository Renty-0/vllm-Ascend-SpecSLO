"""Original verifier/consensus/compact/finish/cleanup, with current object identity."""
from collections import Counter
from base199 import require
import epoch199 as epoch
import target199 as target

def setup(native,c,trace):
 from vllm_ascend.spec_decode.pearl.spec_rhythm import SpecRhythmRuntimeState,SpecRhythmPipelineController
 from vllm_ascend.spec_decode.pearl.tree import pack_selected_tree_plan,tree_primary_path
 f=trace['fixture'];mixed=trace['mixed'];cs=f['contexts']
 require(f['ids']==list(range(16))+list(c.eager_ids)and set(f['sources'])==set(c.eager_ids),'real bounded normal/child source IDs')
 controller=SpecRhythmPipelineController({i:SpecRhythmRuntimeState(i,0 if i<16 else 1,max_gamma=4)for i in range(64)})
 logical={i:native.PearlPipelineState(list(v['token_ids']),len(v['token_ids'])-1,max_tokens=64,ignore_eos=True)for i,v in enumerate(cs)}
 payloads={};parents={};staged={}
 for i in range(32):
  ticket=controller.new_ticket(i,gamma=4,eager=False)
  if i<16:
   p=pack_selected_tree_plan(f['plans'][i],range(4));row=mixed['draft_token_ids'][i][:4];confidence=sum(mixed['node_confidences'][i][:4])/4
  else:p,row=f['parent_sources'][i];confidence=f['parent_confidences'][i]
  payloads[ticket.proposal_id]=dict(ticket=ticket,plan=p,row=list(row),confidence=float(confidence),draft_kv_materialized=False)
  parents[i]=ticket;controller.publish([ticket])
  if i in c.eager_ids:
   j=f['ids'].index(i);child=controller.new_ticket(i,gamma=4,eager=True);cp=pack_selected_tree_plan(f['plans'][j],range(4));path=tuple(tree_primary_path(p));require(path==(0,1,2,3)and cp.prefix_len==p.prefix_len+5,'actual P4 dependency child4 plan')
   payloads[child.proposal_id]=dict(ticket=child,plan=cp,row=list(mixed['draft_token_ids'][j][:4]),confidence=float(sum(mixed['node_confidences'][j][:4])/4),draft_kv_materialized=False,draft_kv_deferred_canonical=True,eager_parent_proposal_id=ticket.proposal_id,eager_dependency_length=4,eager_dependency_tokens=tuple(row[v]for v in path),eager_frontier_token=int(mixed['eager_frontier_tokens'][i]))
   staged[i]=child;controller.publish([child])
 return dict(controller=controller,logical=logical,payloads=payloads,parent_tickets=parents,staged=staged,contexts=cs)

def exchange(e,source,trace,c,parent,stage):
 from budget199 import work_plan,refine
 def prepare():
  ids=list(range(16,32))if parent else trace['fixture']['ids'];ordered=[source['parent_tickets'][i]if i<16 or parent else source['staged'][i]for i in ids]
  payloads=[source['payloads'][t.proposal_id]for t in ordered];plans=[p['plan']for p in payloads];rows=[p['row']for p in payloads];frontiers=[trace['mixed']['eager_frontier_tokens'][t.request_index]if t.eager else None for t in ordered];selected=[list(range(4))for _ in ids]
  if not parent:work_plan(c).require_published_tickets(ordered)
  return ids,ordered,payloads,plans,rows,frontiers,selected
 ids,ordered,payloads,plans,rows,frontiers,selected=stage('prepare-WORLD',prepare)
 packet=stage('original-WORLD',lambda:e._exchange_spec_rhythm_tree_candidates(rows if e.is_draft else None,plans,frontier_tokens=frontiers if e.is_draft else None,confidences=[p['confidence']for p in payloads]if e.is_draft else None,selected_indices=selected if e.is_draft else None,draft_compute_ms=0.))
 def drain():
  rr,ff,conf,ix,_=e._split_tree_candidates(packet,plans);require(rr==rows and ff==frontiers and ix==selected,'actual variable-owner original WORLD/float64/canonical8/selected indices')
  for p,v in zip(payloads,conf,strict=True):p['confidence']=float(v)
  return dict(owner_ids=ids,proposal_ids=[t.proposal_id for t in ordered],selected_candidates=4*len(ids),typed=None if parent else refine(c,trace['fixture'],trace['mixed']),real_WORLD=True)
 return stage('original-WORLD-finite-drain',drain)

def check_commit(e,frame,accepted,reference_KV):
 import torch
 plans,_,_,ids,real=frame;source=[];destination=[];offset=0;records=[]
 require(len(accepted)==real,'every genuine verifier row')
 for p,owner,path in zip(plans[:real],ids[:real],accepted,strict=True):
  nodes=[int(v) for v in path if v>=0];source.extend(offset+v for v in [0,*[1+j for j in nodes]])
  dst=epoch.slots(e,owner,range(p.prefix_len,p.prefix_len+1+len(nodes)));destination.append(dst)
  records.append(dict(owner=owner,accepted_nodes=nodes,committed_count=1+len(nodes),destination_slots=dst.cpu().tolist()))
  offset+=p.candidate_budget+1
 src=torch.tensor(source,dtype=torch.long,device=e.device);actual=epoch.read(e,torch.cat(destination))
 expected=tuple(v.index_select(0,src) for v in reference_KV)
 require(all(bool(torch.isfinite(v).all()) for v in actual) and e.graph_runner._target_outputs_match(actual,expected),
  'actual original compacted every committed root/path across all128 KV tensors equals own independent queryKV')
 return dict(all128_committed_KV=True,per_owner=records,independent_query_KV_only=True,actual_hashes=[epoch.tensor_hash(v)for v in actual],reference_hashes=[epoch.tensor_hash(v)for v in expected])

def commit(e,source,ids,frame,output,reference_KV,original,stage,*,home=None):
 import original_native_protocol_v75 as native_protocol
 from original_tree_cycle_v75 import _require_compiled
 _require_compiled(original)
 require(original is e._v75_original_protocol and native_protocol.commit_states is e._v199_commit_states,'same pre-startup compiled object and explicitly pinned original commit_states identity')
 controller=source['controller'];payloads=source['payloads'];logical=source['logical'];plans=frame[0][:len(ids)];rows=frame[2][:len(ids)]
 def preflight():
  tickets=[controller.validate_verification(i) for i in ids];target_payloads=[payloads[t.proposal_id] for t in tickets]
  require(len(tickets)==len(rows)==len(plans) and all(p.candidate_budget==t.gamma for p,t in zip(plans,tickets,strict=True)),
   'all4 actual ticket/plan row identity before any verifier broadcast')
  return tickets,target_payloads
 tickets,target_payloads=stage('pre-verdict-ticket-identity',preflight)
 def leader():
  if e.rank!=e.topology.target_leader_rank:return None
  return e._verify_tree_outputs_host(rows,plans,output['target_query_token_ids'],4,e.device)
 observed=stage('actual-target-leader-verdict-before-broadcast',leader)
 outputs,accepted,_=stage('original-verdict-broadcast',lambda:e._broadcast_spec_rhythm_tree_verdict(observed,plans,0.))
 if source.get('_v154_before_parent_commit') is not None:source['_v154_before_parent_commit'](outputs,accepted)
 counters=Counter();mappings={};before_lengths=[len(logical[i].token_ids)-1 for i in range(64)]
 protected=stage('precommit-all64-owner-prefix',lambda:epoch.protect(e,before_lengths) if not e.is_draft else None)
 observed_before=stage('actual-consensus-count-before',lambda:len(original['actual_commit_consensus_observations']))
 stage('original-finite-preflight-vote-and-compaction',lambda:original['preflight_consensus_compact'](e,controller,payloads,mappings,ids,plans,
  target_payloads,rows,output,logical,outputs,accepted,counters))
 def actual_consensus():
  values=original['actual_commit_consensus_observations'];require(len(values)==observed_before+1 and len(values[-1])==2 and all(type(v)is int and v==0 for v in values[-1]),'actual original two-integer allrank structural/numerical consensus')
  return list(values[-1])
 consensus=stage('actual-original-commit-consensus-integers',actual_consensus)
 def compact_gate():
  if e.is_draft:return None
  proof=check_commit(e,frame,accepted,reference_KV)
  require(epoch.protect(e,before_lengths)==protected,'Native compaction preserves every old logical prefix')
  return proof
 kv=stage('own-all128-committed-KV-gate',compact_gate)
 def finish():
  promoted={};values_by_owner={};tickets_record=[]
  for row,i in enumerate(ids):
   values=[int(v) for v in outputs[row] if v>=0];values_by_owner[i]=values;eager=controller.staged_eager.get(i)
   child=original['finish'](e,controller,i,plans[row],target_payloads[row:row+1],payloads,outputs[row:row+1],accepted[row:row+1],values,False,counters)
   replica=logical[i].clone();native_protocol.commit_states(logical[i],replica,outputs[row],accepted[row],plans[row].candidate_budget)
   moves=original['cleanup'](e,i,payloads,mappings,tickets[row],eager,child,counters)
   require(not moves and logical[i].token_ids==replica.token_ids,'original deferred canonical finish and replicated logical state; no relocations')
   if child is not None:
    require(child is source['staged'].get(i) and controller.validate_verification(i) is child,'real naturally promoted exact staged child ticket')
    promoted[i]=child
   tickets_record.append(dict(owner=i,proposal_id=tickets[row].proposal_id,gamma=tickets[row].gamma,
    eager=bool(tickets[row].eager),promoted_proposal_id=None if child is None else child.proposal_id,
    accepted_nodes=[int(v) for v in accepted[row] if v>=0],emitted_tokens=values))
  controller.finish_cycle(home)
  return promoted,values_by_owner,tickets_record
 promoted,values,tickets_record=stage('original-finish-cleanup-and-promotion-identity',finish)
 return dict(actual_consensus=consensus,outputs=outputs,accepted=accepted,promoted=promoted,values_by_owner=values,commit_KV=kv,counters=dict(counters),tickets=tickets_record)
