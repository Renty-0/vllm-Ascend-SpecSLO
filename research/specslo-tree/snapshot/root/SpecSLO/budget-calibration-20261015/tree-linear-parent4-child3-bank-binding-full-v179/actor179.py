"""Actual finite parent4 discovery; positive branch continues own numeric/protocol/next."""
from collections import Counter
from scope179 import require,case,contract,NATIVE_COUNTERS,FAILURES,digest
import draft179 as draft
import source179 as epoch
import protocol179 as protocol
import target179 as target
def native(e):
 values=e.graph_metrics();out={k:values[k] for k in NATIVE_COUNTERS}
 require(all(type(v)is int and v>=0 for v in out.values()) and not any(out[k] for k in FAILURES),
  'all21 original Native counters; no failures/fallbacks')
 return out

def ticket(value):
 return dict(proposal_id=value.proposal_id,owner=value.request_index,home=value.home_batch_id,
  gamma=value.gamma,required_prefix_epoch=value.required_prefix_epoch,eager=value.eager,
  lifecycle=value.lifecycle.value)

def gate_rows(source,outputs,accepted):
 """Report the unchanged original finish predicate from real verifier rows."""
 from vllm_ascend.spec_decode.pearl.tree import tree_primary_path
 rows=[]
 for row,owner in enumerate(range(16,32)):
  parent=source['controller'].validate_verification(owner);child=source['staged'][owner]
  payload=source['payloads'][parent.proposal_id];eager=source['payloads'][child.proposal_id]
  path=tuple(tree_primary_path(payload['plan']));length=int(eager['eager_dependency_length'])
  nodes=[int(v) for v in accepted[row] if v>=0];out=list(map(int,outputs[row]))
  dependency=tuple(map(int,eager['eager_dependency_tokens']))
  observed_path=tuple(out[:length]);frontier=out[length] if length<len(out) else -1
  tests=dict(fully_accepted=len(nodes)>=len(path),parent_proposal_matches=eager['eager_parent_proposal_id']==parent.proposal_id,
   dependency_length_positive=length>0,observed_path_equal=observed_path==dependency,
   frontier_present=eager.get('eager_frontier_token') is not None,
   frontier_equal=eager.get('eager_frontier_token') is not None and frontier==int(eager['eager_frontier_token']),
   not_will_finish=True)
  rows.append(dict(owner=owner,parent_ticket=ticket(parent),child_ticket=ticket(child),
   original_parent_primary_path=list(path),parent_selected_row=list(payload['row']),
   staged_dependency=list(dependency),staged_frontier=eager['eager_frontier_token'],staged_parent_proposal_id=eager['eager_parent_proposal_id'],staged_dependency_length=length,
   accepted_nodes=nodes,accepted_count=len(nodes),actual_output_row=out,
   actual_observed_dependency=list(observed_path),actual_observed_frontier=frontier,
   gates=tests,predicted_original_finish_eager_valid=all(tests.values())))
 return rows

def warm_source(e,c,stage):
 """Three fresh sources qualify the resident graphs; retain final real source."""
 import torch.distributed as dist
 values=[];latest=None;warm=[]
 for number,changed in enumerate((False,True,False)):
  lane=lambda name,fn:stage('source-FTF'+str(number)+'/'+name,fn)
  contexts=lane('contexts',lambda:draft.contexts(c,changed))
  epoch.rebuild(e,contexts,list(c.repair_vector),lane,role='draft')
  f=lane('actual-parent4',lambda:draft.fixture(e,c,changed) if e.is_draft else None)
  output=lane('actual-union20-shared4',lambda:draft.invoke(e,f) if e.is_draft else None)
  def snapshot():
   if not e.is_draft:return None
   return dict(parent=draft.cpu_output(f['parent_output']),mixed=draft.cpu_output(output))
  value=lane('raw-actual-outputs',snapshot);values.append(value)
  warm.append(dict(changed=changed,actual_source_epoch=None if not e.is_draft else f['source_epoch'],
   actual_mixed_TTU=None if not e.is_draft else e._v152_last_mixed_TTU,
   dynamic_prefix_cost_measured=False))
  if number==2:
   def final_trace():
    if not e.is_draft:return None
    require(values[0]==values[2],'entire real source F/T/F restores actual parent/mixed outputs')
    for family in (e.graph_runner.draft_entries,e.graph_runner.target_entries):
     require(all(v.runtime_validated for v in family.values()),'all initial draft/union Native gates naturally qualified')
    return dict(case=c.__dict__,fixture=f,mixed=draft.cpu_output(output),changed=False,
     numerical_proof=None,actual_source_lease=f['source_lease'],controlled_grants_not_live_policy=True,
     adopted=False,Goodput_measured=False)
   latest=lane('final-actual-source-and-natural-flags',final_trace)
 def share():
  packet=[latest if e.is_draft else None]
  dist.broadcast_object_list(packet,src=e.topology.draft_leader_rank,group=e.groups.verification_coordination_group)
  require(packet[0] is not None,'actual source published from draft rank')
  return packet[0]
 return stage('share-actual-discovery-source',share),warm

def abort_unconsumed(source):
 controller=source['controller']
 pending=dict(ready=[ticket(v) for v in controller.ready.values()],
  staged=[ticket(v) for v in controller.staged_eager.values()],payload_ids=sorted(source['payloads']))
 for owner in range(64):controller.invalidate_request(owner)
 source['payloads'].clear()
 require(not controller.ready and not controller.staged_eager and not source['payloads'],'explicit discovery abort releases unconsumed tickets')
 return dict(pending_before_abort=pending,cleanup='original controller.invalidate_request for discovery abort',
  normal_or_child_consume_claimed=False)

def run_actor(e,native_module,stage):
 import torch.distributed as dist
 import mc2_preflight151,native_protocol
 from provider179 import evidence
 c=case();report=dict(schema='V179.natural-positive-discovery.v1',contract=contract(),
  Native_flags_assigned=False,formal_Goodput=False,adopted=False,full_qualification=False,
  performance_measured=False,two_complete_zero_traces=False)
 e._v151_partial=report
 report['input_prefixes']=stage('exact-retained-input-prefixes',lambda:evidence(c))
 report['Native_before']=stage('Native-before',lambda:native(e))
 report['MC2_static']=stage('original-MC2-static',lambda:mc2_preflight151.check(e) if not e.is_draft else None)
 trace,warm=warm_source(e,c,stage);report['warm_source']=warm
 report['raw_parent_output']=trace['fixture']['parent_output']
 report['raw_mixed_output']=trace['mixed']
 source=stage('setup-original-typed-controller',lambda:protocol.setup(native_module,c,trace))
 original=stage('compile-original-protocol-AST',lambda:native_protocol.compile_protocol(native_module))
 report['previous_source_exchange']=protocol.exchange(e,source,trace,c,True,lambda n,f:stage('previous-parent/'+n,f))
 report['current_work_exchange']=protocol.exchange(e,source,trace,c,False,lambda n,f:stage('current-work/'+n,f))
 epoch.rebuild(e,source['contexts'],[0]*64,stage,role='target')
 history=[];phases=[];box=dict(trace=trace,gates=None,union_sameM=None)
 def before_parent(outputs,accepted):
  gates=stage('discovery/raw-original-finish-gates',lambda:gate_rows(source,outputs,accepted));box['gates']=gates
  report['actual_discovery_gate_preview']=gates
  def agreement():
   rows=[None]*4;value=digest(gates)
   dist.all_gather_object(rows,value,group=e.groups.verification_coordination_group)
   require(rows==[value]*4,'all4 exact actual gate evidence before any conditional model work')
  stage('discovery/all4-gate-consensus',agreement)
  if not any(row['predicted_original_finish_eager_valid'] for row in gates):return
  # No ticket or actual verifier row has been committed yet. The fresh own
  # union oracle restores the SAME exact real source before original preflight.
  saved=stage('own-union/pre-commit-actual-ledger',lambda:epoch.ledger(e) if e.is_draft else None)
  qualified=draft.prepare(e,c,lambda n,f:stage('own-union/'+n,f))
  def restore_gate():
   require(qualified['mixed']==trace['mixed'] and qualified['fixture']['parent_output']==trace['fixture']['parent_output']
    and qualified['fixture']['contexts']==trace['fixture']['contexts'],'own union reference restores exact actual input, parent and staged output')
   if e.is_draft:require(epoch.ledger(e)==saved,'own union history restores actual precommit known-KV ledger')
  stage('own-union/precommit-exact-source-restoration',restore_gate)
  box.update(trace=qualified,union_sameM=qualified['numerical_proof'])
 source['_v154_before_parent_commit']=before_parent
 def execute(ids,label):
  lane=lambda n,f:stage(label+'/'+n,f)
  frame=lane('prepare-original-target',lambda:target.padded(e,source,ids))
  qualified=target.qualify(e,frame,source,history,lane)
  output,KV,proof=qualified if not e.is_draft else (None,None,None)
  actual=protocol.commit(e,source,ids,frame,output,KV,original,lane)
  history.append(dict(frame=frame,output=output,KV=KV,accepted=actual['accepted']))
  phases.append(dict(label=label,owners=list(ids),target=proof,accepted=actual['accepted'],outputs=actual['outputs'],
   committed_KV=actual['commit_KV'],tickets=actual['tickets'],counters=actual['counters']))
  return actual
 parent=execute(list(range(16,32)),'discovery-parent4')
 source.pop('_v154_before_parent_commit')
 promoted=sorted(parent['promoted']);invalid=sorted(set(range(16,32))-set(promoted))
 def actual_promotion_gate():
  expected=[r['owner'] for r in box['gates'] if r['predicted_original_finish_eager_valid']]
  require(expected==promoted,'actual ORIGINAL finish returned exactly predicted raw verifier dependencies; never force promotion')
  for owner in promoted:require(source['controller'].validate_verification(owner) is source['staged'][owner],'actual promoted ticket identity')
  return dict(promoted_owners=promoted,invalidated_owners=invalid,actual_positive=bool(promoted),actual_negative=bool(invalid),
   gates=box['gates'],parent_commit_tickets=parent['tickets'],parent_outputs=parent['outputs'],parent_accepted=parent['accepted'])
 report['discovery']=stage('discovery/original-actual-promotion-gate',actual_promotion_gate)
 report['union_sameM']=box['union_sameM'];trace=box['trace']
 report['qualified_union_generation']=None if box['union_sameM'] is None else trace['generation_observation']
 if not promoted:
  report['early_stop']=stage('discovery/abort-unconsumed-normal-tickets',lambda:abort_unconsumed(source))
  report['continuation']=None
 else:
  normal=execute(list(range(16)),'true-normal-variable4')
  stage('normal-no-grandchildren',lambda:require(not normal['promoted'],'normal has no staged grandchild'))
  child=execute(promoted,'true-promoted-child3-consume')
  stage('child-no-grandchildren',lambda:require(not child['promoted'],'consumed child has no staged grandchild'))
  def closed():
   require(not source['controller'].ready and not source['controller'].staged_eager and not source['payloads'],
    'all real normal/promoted-child tickets consumed once; natural invalidations removed')
   for row in child['tickets']:
    owner=row['owner'];require(row['eager'] is True and row['proposal_id']==source['staged'][owner].proposal_id,
     'actual target consumed original naturally promoted child identity')
  stage('closed-original-publication-and-real-child-consume',closed)
  import next179 as next154
  nextproof=next154.proof(e,c,trace,source['logical'],lambda n,f:stage('actual-nextdraft/'+n,f))
  report['continuation']=dict(real_promoted_child_consumed=True,consumed_owners=promoted,
   child_commit_tickets=child['tickets'],nextdraft=nextproof,original_protocol_closed=True)
 report['phases']=phases
 report['Native_after']=stage('Native-after',lambda:native(e))
 report['Native_delta']={k:report['Native_after'][k]-v for k,v in report['Native_before'].items()}
 require(all(v>=0 for v in report['Native_delta'].values()),'actual Native discovery counters monotone')
 report['actual_graph_states']={family:[dict(key=repr(k),runtime_validated=bool(v.runtime_validated),
  validated_real_row_count=getattr(v,'validated_real_row_count',None)) for k,v in getattr(e.graph_runner,name).items()]
  for family,name in (('generic','entries'),('draft','draft_entries'),('target','target_entries'))}
 report['result']=dict(status='positive-consumed-and-next' if promoted else 'negative-only-subset',
  positive_count=len(promoted),negative_count=len(invalid),new_union_sameM_qualified=box['union_sameM'] is not None,
  natural_positive_and_negative_observed=bool(promoted and invalid),full_qualification=False,
  two_complete_zero_traces=False,production_gates_opened=False,formal_Goodput=False,adopted=False)
 report['lease_error_status']=lease_error_status(e,stage)
 stage('release-original-cache',e._release_cache)
 return report

from base179 import B,functions
functions(B/'tree-live-shared4-union-full-v165/natural165.py',globals(),names=('lease_error_status',))
