"""Bounded genuine promotion/reject/consume/rollback/next; no online issuer."""
import gzip,json
from pathlib import Path
from scope199 import cases,contract,require,NATIVE_COUNTERS,FAILURES,digest
import draft199 as draft
import epoch199 as epoch
import protocol199 as protocol
import target199 as target

def native(e):
 v=e.graph_metrics();out={k:v[k]for k in NATIVE_COUNTERS};require(all(type(x)is int and x>=0 for x in out.values())and not any(out[k]for k in FAILURES),'all21 actual original counters/failures');return out

def delta(before,after,zero):
 result={k:after[k]-v for k,v in before.items()};require(all(v>=0 for v in result.values())and(not zero or not any(result.values())),'complete current Native zero trace');return result

def save(folder,name,value):
 p=Path(folder)/name;raw=json.dumps(value,allow_nan=False,separators=(',',':')).encode();p.write_bytes(gzip.compress(raw,mtime=0))

def gate_rows(source,outputs,accepted):
 from vllm_ascend.spec_decode.pearl.tree import tree_primary_path
 result=[]
 for row,owner in enumerate(range(16,32)):
  if owner not in source['staged']:continue
  parent=source['controller'].validate_verification(owner);child=source['staged'][owner];p=source['payloads'][parent.proposal_id];eager=source['payloads'][child.proposal_id];path=tuple(tree_primary_path(p['plan']));nodes=[int(v)for v in accepted[row]if v>=0];out=list(map(int,outputs[row]));length=eager['eager_dependency_length']
  tests=dict(full_parent=len(nodes)>=len(path),parent_identity=eager['eager_parent_proposal_id']==parent.proposal_id,dependency_length=length==4,dependency=tuple(out[:length])==eager['eager_dependency_tokens'],frontier=len(out)>length and out[length]==eager['eager_frontier_token'])
  result.append(dict(owner=owner,accepted=nodes,output=out,tests=tests,promotable=all(tests.values()),parent_proposal_id=parent.proposal_id,child_proposal_id=child.proposal_id))
 return result

def warm(e,c,stage):
 out=[]
 for j,changed in enumerate((False,True,False)):
  lane=lambda n,f:stage('FTF'+str(j)+'/'+n,f);cs=lane('contexts',lambda:draft.contexts(c,changed));epoch.rebuild(e,cs,list(c.repair_vector),lane,role='draft');f=lane('actual-parent16',lambda:draft.fixture(e,c,changed)if e.is_draft else None)
  value=lane('actual-shared4',lambda:draft.cpu_output(draft.invoke(e,f))if e.is_draft else None);out.append(value)
 stage('FTF-complete-source-restored',lambda:require(out[0]==out[2],'entire real source changed/restored, never assign Native flags')if e.is_draft else None)
 return dict(first_restored_equal=True,changed_actual_source=True)

def run_case(e,native_module,c,stage,pass_index):
 from original_tree_cycle_v75 import _require_compiled
 from provider199 import evidence
 from next199 import proof as nextproof
 source=None;history=[];phases=[]
 try:
  original=stage('same-prestartup-protocol',lambda:(_require_compiled(e._v75_original_protocol),e._v75_original_protocol)[1]);inputs=stage('exact-prefix-only-inputs',lambda:evidence(c))
  before=stage('Native-before',lambda:native(e))
  warmed=warm(e,c,stage)if pass_index==0 else None
  trace=draft.prepare(e,c,stage)
  source=stage('actual-typed-controller',lambda:protocol.setup(native_module,c,trace))
  exchanges=[protocol.exchange(e,source,trace,c,True,lambda n,f:stage('previous-parent/'+n,f)),protocol.exchange(e,source,trace,c,False,lambda n,f:stage('current-work/'+n,f))]
  epoch.rebuild(e,source['contexts'],[0]*64,stage,role='target')
  box={}
  def preview(outputs,accepted):
   value=stage('actual-parent-promotion-gates',lambda:gate_rows(source,outputs,accepted));box['gates']=value
   def agree():
    import torch.distributed as dist
    values=[None]*4;local=digest(value);dist.all_gather_object(values,local,group=e.groups.verification_coordination_group);require(values==[local]*4,'all4 agree before any conditional original finish')
   stage('all4-parent-gate-agreement',agree)
  source['_v154_before_parent_commit']=preview
  def execute(ids,label,home):
   lane=lambda n,f:stage(label+'/'+n,f)
   frame=lane('actual-owner-current173-pad',lambda:target.padded(e,source,ids))
   qualified=target.qualify(e,frame,source,history,lane);out,KV,math=qualified if not e.is_draft else(None,None,None)
   current=protocol.commit(e,source,ids,frame,out,KV,original,lane,home=home)
   history.append(dict(frame=frame,output=out,KV=KV,accepted=current['accepted']))
   phases.append(dict(label=label,owners=list(ids),target=math,accepted=current['accepted'],outputs=current['outputs'],committed_KV=current['commit_KV'],tickets=current['tickets'],counters=current['counters'],actual_consensus=current['actual_consensus'],promoted=sorted(current['promoted'])))
   return current
  parent=execute(list(range(16,32)),'real-parent16-B4',1);source.pop('_v154_before_parent_commit')
  promoted=sorted(parent['promoted']);invalid=sorted(set(c.eager_ids)-set(promoted))
  def promotion_gate():
   require(promoted==[v['owner']for v in box['gates']if v['promotable']],'only original real target verdict/finish determines promotion')
   for owner in promoted:require(source['controller'].validate_verification(owner)is source['staged'][owner],'promoted actual child ticket identity')
  stage('original-natural-promotion-identity',promotion_gate)
  consumed=[];child_full=[];mixed80=False;plans=[]
  for ordinal in range(33):
   def build():
    ctl=source['controller'];p=ctl.build_plan(list(range(32)),active_indices_registered_unique=True,merge_ready_homes=True,max_target_requests=16,verification_budget=96,ready_candidate_counts={i:source['payloads'][t.proposal_id]['plan'].candidate_budget for i,t in ctl.ready.items()})
    return p,list(p.target_request_indices)
   p,ids=stage('planner'+str(ordinal)+'/original-whole-proposal-pack',build)
   if not ids:break
   def descriptor():
    eager=[i for i in ids if source['controller'].ready[i].eager];return dict(owners=ids,eager=eager,target_home=p.target_home_batch_id,deferred=list(p.deferred_target_request_indices),N=len(ids),Q=5*len(ids),C=4*len(ids))
   desc=stage('planner'+str(ordinal)+'/actual-ticket-shape',descriptor);plans.append(desc)
   result=execute(ids,'planner'+str(ordinal)+'/real-target',p.target_home_batch_id)
   eager=[r for r in result['tickets']if r['eager']];consumed.extend(r['owner']for r in eager);child_full.extend(r['owner']for r in eager if len(r['accepted_nodes'])==4)
   if len(ids)==16 and eager:mixed80=True
  def closed():
   require(not source['controller'].ready and not source['controller'].staged_eager and not source['payloads']and sorted(consumed)==promoted,'all real normal/child tickets consumed exactly once; genuine invalidations cleaned')
  stage('original-all-ticket-consumption-closed',closed)
  nxt=nextproof(e,c,trace,source['logical'],lambda n,f:stage('own-next3/'+n,f))
  def leaf_gate():
   if not e.is_draft:return None
   lengths=nxt['actual_missing_suffix_lengths'];observed=[i for i in child_full if lengths[i]==1]
   return dict(child_fullaccept_owners=child_full,actual_e4_missing_suffix_repaired=observed,fullaccept_leaf_repair_observed=bool(observed),next_requested_B=3,original_PA_repair_then_three_step_next=True,all56_next_queryKV=nxt['all28_written_KV'])
  leaf=stage('actual-child4-fullaccept-tail-repair-gate',leaf_gate)
  # A stale source is rejected before its first model/graph submission. The old
  # controller is already closed; a separate real ticket set exercises rollback.
  rollback=stage('new-real-rollback-controller',lambda:protocol.setup(native_module,c,trace))
  epoch.rebuild(e,trace['fixture']['contexts'],list(c.repair_vector),stage,role='draft')
  def stale():
   if not e.is_draft:return None
   snapshot=native(e)
   try:draft.invoke(e,trace['fixture'])
   except RuntimeError as error:
    require('source lease epoch'in str(error),'expected exact source identity rejection');require(native(e)==snapshot,'stale source rejected before Native work');return dict(rejected=True,error=str(error),Native_unchanged=True)
   raise RuntimeError('stale source unexpectedly executed')
  staleproof=stage('expected-stale-epoch-rejection',stale)
  def abort():
   ctl=rollback['controller'];pending=dict(ready=len(ctl.ready),staged=len(ctl.staged_eager),payloads=len(rollback['payloads']))
   for owner in range(64):ctl.invalidate_request(owner)
   rollback['payloads'].clear();require(not ctl.ready and not ctl.staged_eager and not rollback['payloads'],'original invalidate_request cleans all pending real tickets')
   return dict(before=pending,after=dict(ready=0,staged=0,payloads=0),no_fabricated_target_verdict=True)
  aborted=stage('actual-controller-epoch-rollback',abort)
  rollback_next=nextproof(e,c,trace,rollback['logical'],lambda n,f:stage('rollback-own-next3/'+n,f))
  difference=stage('complete-Native-delta',lambda:delta(before,native(e),pass_index>0))
  return dict(case=c.__dict__,name=c.name,input=inputs,warm=warmed,math=trace['numerical_proof'],exchanges=exchanges,phases=phases,promotion_gates=box['gates'],promoted=promoted,invalid=invalid,consumed=consumed,planner=plans,mixed16Q80_child_observed=mixed80,next=nxt,leaf=leaf,stale=staleproof,rollback=aborted,rollback_next=rollback_next,Native_delta=difference,production_permission=False,formal_Goodput=False,whole_AR=False)
 finally:
  def cleanup():
   if source is not None:
    for owner in range(64):source['controller'].invalidate_request(owner)
    source['payloads'].clear()
   e._release_cache()
  stage('fixed-final-controller-and-cache-cleanup',cleanup)

def inventory(e):
 r=e.graph_runner
 return dict(objects={name:{repr(k):id(v)for k,v in getattr(r,name).items()}for name in('entries','draft_entries','target_entries')},disabled=sorted(map(repr,r.disabled_entry_keys)))

def run(e,native_module,stage,output):
 from live199 import diagnostic
 folder=Path(output)/'bounded199';stage('private-output-dir',lambda:folder.mkdir(exist_ok=True))
 before=stage('actual-inventory-before',lambda:inventory(e));traces=[];fingerprints={}
 for pass_index in range(3):
  for c in cases():
   lane=lambda n,f:stage('pass'+str(pass_index)+'/'+c.name+'/'+n,f)
   manager=diagnostic(e,c);entered={'value':False}
   def enter():
    result=manager.__enter__();entered['value']=True;return result
   try:
    lane('common-enter-bounded-context',enter)
    value=run_case(e,native_module,c,lane,pass_index)
   finally:
    lane('common-finally-revoke-bounded-context',lambda:manager.__exit__(None,None,None)if entered['value']else None)
   def retain():
    key=c.name;fingerprint=digest(dict(parent=value['promotion_gates'],phases=[{k:v for k,v in p.items()if k in('owners','accepted','outputs','tickets','promoted')}for p in value['phases']],next=value['next'],rollback_next=value['rollback_next']))
    if pass_index:require(fingerprint==fingerprints[key],'two complete zero traces reproduce exact original commit/own-next history')
    else:fingerprints[key]=fingerprint
    save(folder,f'rank{e.rank}-pass{pass_index}-{c.name}.json.gz',value)
    return dict(pass_index=pass_index,case=key,fingerprint=fingerprint,promoted=value['promoted'],invalid=value['invalid'],consumed=value['consumed'],leaf=value['leaf'],mixed16Q80_child=value['mixed16Q80_child_observed'],Native_delta=value['Native_delta'],math=value['math'],stale=value['stale'])
   traces.append(lane('retain-real-trace',retain))
 after=stage('actual-inventory-after',lambda:inventory(e))
 def finish():
  require(before['disabled']==after['disabled'],'no Native disabled key added')
  for name,old in before['objects'].items():require(all(after['objects'][name].get(k)==v for k,v in old.items()),'all original Native entry objects retained '+name)
  require(not hasattr(e,'_v199_diagnostic')and e.cache_allocation is None and e.cache_block_tables is None,'no leaked diagnostic permission/cache')
  result=dict(contract=contract(),traces=traces,inventory_before=before,inventory_after=after,new_entries={name:sorted(set(now)-set(before['objects'][name]))for name,now in after['objects'].items()},two_complete21_zero_traces=True,Native_flags_assigned=False,production_permission=False,full463=False,formal_Goodput=False,whole_AR=False,arbitrary_tuple_scope=False)
  e._v199_bounded_actual=result;return result
 return stage('complete-bound-component-summary',finish)
