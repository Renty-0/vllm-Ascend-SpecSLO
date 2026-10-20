"""Heterogeneous causal same-M oracle; every actual TP call has all4 lanes."""
from dataclasses import asdict
from scope151 import require,target_shape
import source151 as epoch

def clone_output(out):
 import torch
 return {k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in out.items()}
def padded(e,source,ids):
 from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan
 plans=[];roots=[];rows=[]
 for owner in ids:
  ticket=source['controller'].validate_verification(owner);payload=source['payloads'][ticket.proposal_id]
  require(payload['plan'].prefix_len==len(source['logical'][owner].token_ids)-1,'current genuine root/prefix before target')
  plans.append(payload['plan']);roots.append(source['logical'][owner].token_ids[-1]);rows.append(payload['row'])
 g=target_shape([p.candidate_budget for p in plans]);owners=list(ids)
 for owner,q in zip(range(64,102),g['scratch'],strict=False):
  plans.append(cached_cpu_tree_speculation_plan(1,q-1,0,4096,candidate_budget=q-1));roots.append(0);rows.append([0]*(q-1));owners.append(owner)
 require(len(plans)==g['S'] and sum(p.candidate_budget+1 for p in plans)==g['M'] and not set(owners[len(ids):])&set(source['logical']),
  'exact heterogeneous current C96/Q128/N32 and independent scratch')
 return plans,roots,rows,owners,len(ids)
def metadata(e,frame,*,changed=False):
 from originals151 import original_function
 plans,roots,rows,ids,real=frame
 plans=[original_function('short_target_v126','independent_plan')(p) for p in plans]
 roots=[(v+1)%151936 for v in roots] if changed else roots
 call=e.model.make_tree_attention_metadata(plans,roots,rows,e.cache_allocation.block_tables,ids,allow_causal_fast_path=True)
 x,pos,meta=call;g=target_shape([p.candidate_budget for p in plans[:real]])
 require(not meta.tree_attention and x.numel()==g['M'] and len(meta.actual_seq_lengths_q)==g['S']
  and sum(p.candidate_budget+1 for p in plans[:real])==g['real_Q'],'same-M actual variable partition causal FIA')
 return call

def original_call(e,frame,*,changed=False):
 plans,roots,rows,ids,real=frame;roots=[(v+1)%151936 for v in roots] if changed else roots
 runner=e.graph_runner;old=runner.run_target_greedy;had='run_target_greedy' in vars(runner);prior=vars(runner).get('run_target_greedy');saved=[]
 def immediate(*args,**kwargs):
  out=old(*args,**kwargs);saved.append(out[0].clone());return out
 runner.run_target_greedy=immediate
 try:out=e.target_tree_forward(plans,roots,rows,ids,return_logits=False,real_tree_count=real)
 finally:
  if had:runner.run_target_greedy=prior
  else:vars(runner).pop('run_target_greedy',None)
 require(len(saved)==1 and out['used_aclgraph'] is True and out['num_draft_tokens']==[p.candidate_budget for p in plans[:real]],'one actual original full-M greedy API/variable budget')
 return clone_output(out),saved[0]

def own_compact(e,frame,accepted,kv):
 import torch
 plans,_,_,ids,real=frame;offset=0;src=[];dst=[]
 for row,(p,owner) in enumerate(zip(plans,ids,strict=True)):
  if row<real:
   nodes=[0]+[1+v for v in accepted[row] if v>=0];src.extend(offset+j for j in nodes)
   dst.extend(epoch.slots(e,owner,range(p.prefix_len,p.prefix_len+len(nodes))).cpu().tolist())
  offset+=p.candidate_budget+1
 require(len(dst)==len(set(dst)),'independent compact destination owners remain distinct')
 src=torch.tensor(src,dtype=torch.long,device=e.device);dst=torch.tensor(dst,dtype=torch.long,device=e.device)
 for cache,value in zip(epoch.arrays(e),kv,strict=True):cache.index_copy_(0,dst,value.index_select(0,src))
 return dict(independent_query_KV_only=True,candidate_KV_copied=False,selected_rows=len(dst))

def rebuild_history(e,source,history,stage,*,independent):
 import torch
 epoch.rebuild(e,source['contexts'],[0]*64,stage,role='target')
 for index,old in enumerate(history):
  frame=old['frame'];lane=lambda name,fn:stage('prior'+str(index)+'/'+name,fn)
  call=lane('prepare',lambda:metadata(e,frame) if not e.is_draft else None)
  if independent:
   hidden=lane('own-model-immediate-clone',lambda:e.model(*call).clone() if not e.is_draft else None)
   def finite_query():
    if e.is_draft:return None
    values=epoch.query_KV(e,call[2]);require(all(bool(torch.isfinite(v).all()) for v in (hidden,*values)),'prior own hidden/queryKV finite before head TP')
    return values
   kv=lane('own-KV-finite',finite_query)
   tokens=lane('own-same-M-head-immediate-clone',lambda:e.model.compute_greedy_tokens(hidden,e.draft_vocab_size).clone() if not e.is_draft else None)
   def compare():
    if e.is_draft:return None
    require(all(bool(torch.isfinite(v).all()) for v in (hidden,*kv)) and e.graph_runner._target_outputs_match(kv,old['KV'])
     and torch.equal(tokens[:old['output']['logical_query_count']],old['output']['target_query_token_ids']),
     'fresh own prior history actual full-M queryKV and exact tokens')
    return own_compact(e,frame,old['accepted'],kv)
   lane('own-compare-and-compact',compare)
  else:
   current=lane('original-API-immediate-clone',lambda:original_call(e,frame) if not e.is_draft else None)
   def restore():
    if e.is_draft:return None
    output,physical=current
    require(torch.equal(output['target_query_token_ids'],old['output']['target_query_token_ids'])
     and e.graph_runner._target_outputs_match(epoch.query_KV(e,call[2]),old['KV']),'original exact prior history restore')
    e.compact_tree_round(output['cache_slot_mapping'],old['accepted'],[p.candidate_budget for p in frame[0][:frame[4]]])
   lane('original-current-slot-compaction',restore)

def qualify(e,frame,source,history,stage):
 import torch
 from vllm_ascend.spec_decode.pearl.mc2 import snapshot_mc2_dispatch_counters
 target=not e.is_draft;runner=e.graph_runner;g=target_shape([p.candidate_budget for p in frame[0][:frame[4]]])
 lengths=[len(source['logical'][i].token_ids)-1 for i in range(64)]
 protected=stage('target-allowner-prefix-before',lambda:epoch.protect(e,lengths) if target else None)
 before=stage('target-MC2-before',snapshot_mc2_dispatch_counters)
 greedy=None;physical=None;first=None;g_records=[]
 for phase in ('first','changed','restored'):
  state=stage('greedy/'+phase+'/prepare',lambda:(metadata(e,frame,changed=phase=='changed'),{k:bool(v.runtime_validated) for k,v in runner.target_entries.items()}) if target else None)
  current=stage('greedy/'+phase+'/Native-immediate-clone',lambda:original_call(e,frame,changed=phase=='changed') if target else None)
  def drain():
   nonlocal first,greedy,physical
   if not target:return None
   out,full=current;execution=runner.last_target_execution
   require(execution.used_aclgraph and execution.mode in ('capture_replay','replay') and full.numel()==g['M'],'actual full-M Native greedy graph')
   key=runner._v151_last_target_key;was=state[1].get(key,False);entry=runner.target_entries[key]
   require(bool(entry.runtime_validated)==(was or phase!='first'),'Native greedy original cold/changed/restored flags')
   if phase=='first':first=full
   if phase=='restored':
    require(torch.equal(first,full),'actual greedy F/T/F exact restoration');greedy,physical=out,full
   g_records.append(dict(phase=phase,key=repr(key),before_validated=was,runtime_validated=bool(entry.runtime_validated),execution=asdict(execution)))
  stage('greedy/'+phase+'/drain',drain)
 hidden=None;actual_KV=None;h_records=[]
 for phase in ('first','changed','restored'):
  call=stage('hidden/'+phase+'/prepare',lambda:metadata(e,frame,changed=phase=='changed') if target else None)
  flags=stage('hidden/'+phase+'/Native-flags-before',lambda:{k:bool(v.runtime_validated) for k,v in runner.target_entries.items()} if target else None)
  current=stage('hidden/'+phase+'/Native-immediate-clone',lambda:runner.run_tree_hidden(*call).clone() if target else None)
  def drain():
   nonlocal hidden,actual_KV
   if not target:return None
   values=epoch.query_KV(e,call[2]);require(len(values)==128 and all(bool(torch.isfinite(v).all()) for v in (current,*values)),'full physical hidden/128target KV finite')
   if phase=='first':hidden,actual_KV=current,values
   if phase=='restored':require(runner._target_outputs_match((hidden,),(current,)) and runner._target_outputs_match(actual_KV,values),'allphysical hidden/all64KV F/T/F')
   entry=runner.target_entries[runner._v151_last_target_key]
   was=flags.get(runner._v151_last_target_key,False)
   require(bool(entry.runtime_validated)==(was or phase!='first'),'Native hidden original cold/changed/restored flags')
   h_records.append(dict(phase=phase,key=repr(runner._v151_last_target_key),before_validated=was,runtime_validated=bool(entry.runtime_validated)))
  stage('hidden/'+phase+'/finite-drain',drain)
 # The fresh oracle reconstructs every prior commit; no candidate KV initializes it.
 rebuild_history(e,source,history,stage,independent=True)
 ref_call=stage('own-current/prepare',lambda:metadata(e,frame) if target else None)
 ref_hidden=stage('own-current/model-immediate-clone',lambda:e.model(*ref_call).clone() if target else None)
 def finite_query():
  if not target:return None
  values=epoch.query_KV(e,ref_call[2]);require(all(bool(torch.isfinite(v).all()) for v in (ref_hidden,*values)),'own current hidden/queryKV finite before head TP')
  return values
 ref_KV=stage('own-current/KV-finite',finite_query)
 ref_tokens=stage('own-current/same-M-head-immediate-clone',lambda:e.model.compute_greedy_tokens(ref_hidden,e.draft_vocab_size).clone() if target else None)
 def compare():
  if not target:return None
  require(all(bool(torch.isfinite(v).all()) for v in (ref_hidden,*ref_KV)) and runner._target_outputs_match((hidden,),(ref_hidden,))
   and runner._target_outputs_match(actual_KV,ref_KV) and torch.equal(physical,ref_tokens)
   and epoch.protect(e,lengths)==protected,'fresh own complete same-M history/hidden/128KV/exact whole-head and prefix')
 stage('own-current/complete-numerical-gate',compare)
 rebuild_history(e,source,history,stage,independent=False)
 restore_call=stage('actual-current-restore/prepare',lambda:metadata(e,frame) if target else None)
 restored=stage('actual-current-restore/Native-immediate-clone',lambda:original_call(e,frame) if target else None)
 after=stage('target-MC2-after',snapshot_mc2_dispatch_counters)
 def finish():
  delta={k:after[k]-before[k] for k in before}
  require(all(v>=0 for v in delta.values()) and delta['fallback']==delta['exception']==0,'original MC2 counters no hidden fallback/exception')
  if not target:return None
  require(delta['native_epilogue_success']>0,'real original qualified-M MC2 dispatch')
  output,full=restored
  require(torch.equal(full,physical) and runner._target_outputs_match(epoch.query_KV(e,restore_call[2]),actual_KV)
   and epoch.protect(e,lengths)==protected,'entire original candidate history and all128query KV restored before true commit')
  return output,ref_KV,dict(shape=g,greedy=g_records,hidden=h_records,allphysical_hidden=True,all128query_KV=True,
   independent_complete_history=True,candidate_KV_copied=False,full_physical_tokens=full.cpu().tolist(),MC2_delta=delta)
 return stage('actual-target-final-restore-and-MC2-gate',finish)
