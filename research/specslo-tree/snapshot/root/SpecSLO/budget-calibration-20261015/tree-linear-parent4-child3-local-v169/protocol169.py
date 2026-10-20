"""Original typed finish/compact with finite parent4 setup and ticket publication."""
from collections import Counter
from scope169 import require
import source169 as epoch
import target169 as target
def setup(native,case,trace):
 from vllm_ascend.spec_decode.pearl.spec_rhythm import SpecRhythmRuntimeState,SpecRhythmPipelineController
 from vllm_ascend.spec_decode.pearl.tree import pack_selected_tree_plan,tree_primary_path
 f=trace['fixture'];mixed=trace['mixed'];cs=f['contexts'];ids=list(range(64))
 require(f['ids']==list(range(32)) and set(f['sources'])==set(range(16,32)),'actual original parent3 source owners')
 controller=SpecRhythmPipelineController({i:SpecRhythmRuntimeState(i,i%2,max_gamma=4) for i in ids})
 logical={i:native.PearlPipelineState(list(c['token_ids']),len(c['token_ids'])-1,max_tokens=64,ignore_eos=True) for i,c in enumerate(cs)}
 payloads={};parents={};staged={}
 for i in range(32):
  B=case.normal_budgets[i] if i<16 else 4;ticket=controller.new_ticket(i,gamma=B,eager=False)
  if i<16:
   plan=pack_selected_tree_plan(f['plans'][i],range(B));row=mixed['draft_token_ids'][i][:B]
   confidence=sum(mixed['node_confidences'][i][:B])/B
  else:plan,row=f['sources'][i];confidence=f['parent_confidences'][i]
  payloads[ticket.proposal_id]=dict(ticket=ticket,plan=plan,row=list(row),confidence=float(confidence),draft_kv_materialized=False)
  parents[i]=ticket;controller.publish([ticket])
  if i>=16:
   child=controller.new_ticket(i,gamma=3,eager=True);child_plan=pack_selected_tree_plan(f['plans'][i],range(3));path=tuple(tree_primary_path(plan))
   require(path==(0,1,2,3) and child_plan.prefix_len==plan.prefix_len+5,'only parent3 dependency/child3; parent4 remains closed')
   payloads[child.proposal_id]=dict(ticket=child,plan=child_plan,row=list(mixed['draft_token_ids'][i][:3]),
    confidence=float(sum(mixed['node_confidences'][i][:3])/3),draft_kv_materialized=False,draft_kv_deferred_canonical=True,
    eager_parent_proposal_id=ticket.proposal_id,eager_dependency_length=4,eager_dependency_tokens=tuple(row[j] for j in path),
    eager_frontier_token=int(mixed['eager_frontier_tokens'][i]))
   staged[i]=child;controller.publish([child])
 return dict(controller=controller,logical=logical,payloads=payloads,parent_tickets=parents,staged=staged,contexts=cs)

def exchange(e,source,trace,case,previous_parent,stage):
 from budget169 import work_plan,refine
 def prepare():
  ordered=([source['parent_tickets'][i] for i in range(16,32)] if previous_parent else
   [source['parent_tickets'][i] for i in range(16)]+[source['staged'][i] for i in range(16,32)])
  payloads=[source['payloads'][t.proposal_id] for t in ordered];plans=[p['plan'] for p in payloads];rows=[p['row'] for p in payloads]
  frontiers=[int(trace['mixed']['eager_frontier_tokens'][t.request_index]) if t.eager else None for t in ordered]
  selected=[list(range(t.gamma)) for t in ordered]
  if not previous_parent:work_plan(case).require_published_tickets(ordered)
  return ordered,payloads,plans,rows,frontiers,selected
 ordered,payloads,plans,rows,frontiers,selected=stage('prepare-before-WORLD',prepare)
 packet=stage('original-WORLD-exchange',lambda:e._exchange_spec_rhythm_tree_candidates(rows if e.is_draft else None,plans,
  frontier_tokens=frontiers if e.is_draft else None,confidences=[p['confidence'] for p in payloads] if e.is_draft else None,
  selected_indices=selected if e.is_draft else None,draft_compute_ms=0.))
 def drain():
  result,ft,confidence,indices,_=e._split_tree_candidates(packet,plans)
  require(result==rows and ft==frontiers and indices==selected,'original real WORLD rows/frontiers/variable selected-index ABI')
  for payload,value in zip(payloads,confidence,strict=True):payload['confidence']=float(value)
  return dict(original_WORLD_exchange=True,owner_ids=[t.request_index for t in ordered],proposal_ids=[t.proposal_id for t in ordered],
   selected_candidates=sum(t.gamma for t in ordered),actual_rows=len(ordered),budgets=[t.gamma for t in ordered],
   typed_budget=None if previous_parent else refine(case,trace['fixture'],trace['mixed']),source_and_current_packets_separate=True)
 return stage('actual-WORLD-packet-finite-drain',drain)

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
 return dict(all128_committed_KV=True,per_owner=records,independent_query_KV_only=True)

def commit(e,source,ids,frame,output,reference_KV,original,stage):
 import native_protocol
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
 stage('original-finite-preflight-vote-and-compaction',lambda:original['preflight_consensus_compact'](e,controller,payloads,mappings,ids,plans,
  target_payloads,rows,output,logical,outputs,accepted,counters))
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
  controller.finish_cycle(None)
  return promoted,values_by_owner,tickets_record
 promoted,values,tickets_record=stage('original-finish-cleanup-and-promotion-identity',finish)
 return dict(outputs=outputs,accepted=accepted,promoted=promoted,values_by_owner=values,commit_KV=kv,counters=dict(counters),tickets=tickets_record)
