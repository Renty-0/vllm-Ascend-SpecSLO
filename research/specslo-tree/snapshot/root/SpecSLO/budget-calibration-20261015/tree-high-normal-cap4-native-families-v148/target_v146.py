"""Original causal target geometries and independent own-bank same-M oracle."""
from dataclasses import asdict
from scope_v146 import require,target_key
from dual_bank_v146 import _slots,_arrays,query_KV,prefix_evidence

def frame(case,work,*,reference=False,changed=False):
 from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan
 plans=list(work['plans']);roots=[v['token_ids'][-1] for v in work['source']['contexts']]
 rows=[list(v) for v in work['rows']];ids=list(range(32,64)) if reference else list(range(32))
 if changed:roots=[(r+1)%151936 for r in roots]
 dummy_start=469 if reference else 463
 for owner,q in zip(range(dummy_start,dummy_start+6),case['dummy_lengths'],strict=False):
  plans.append(cached_cpu_tree_speculation_plan(1,q-1,0,4096,candidate_budget=q-1));roots.append(0);rows.append([0]*(q-1));ids.append(owner)
 require(len(plans)==case['S'] and sum(p.candidate_budget+1 for p in plans)==case['M'],'actual physical M/S with legal dummy2..9')
 return plans,roots,rows,ids

def metadata(engine,case,work,*,reference=False,changed=False):
 from fixed_mask_extent import padded
 p,r,d,ids=frame(case,work,reference=reference,changed=changed)
 if reference:
  call=engine.model.make_tree_attention_metadata(p,r,d,engine.cache_allocation.block_tables,sequence_ids=ids,allow_causal_fast_path=True)
 else:
  from host_target_metadata import make_host_tree_attention_metadata
  call=make_host_tree_attention_metadata(engine,p,r,d,sequence_ids=ids,allow_causal_fast_path=True)
 x,pos,meta=call;meta=padded(meta,9)
 require(x.numel()==case['M'] and len(meta.actual_seq_lengths_q)==case['S'] and not meta.tree_attention,'actual same-M causal metadata')
 return x,pos,meta

def original_call(engine,case,work,*,changed=False):
 p,r,d,ids=frame(case,work,changed=changed);runner=engine.graph_runner
 old=runner.run_target_greedy;had='run_target_greedy' in runner.__dict__;prior=runner.__dict__.get('run_target_greedy');saved=[]
 def immediate(*args,**kwargs):
  returned=old(*args,**kwargs)
  full=tuple(v.clone() for v in returned)  # FIRST operations after full Native return.
  require(len(full)==1 and tuple(full[0].shape)==(case['M'],),'all physical production greedy outputs, no M1')
  saved.append(full[0]);return returned
 runner.run_target_greedy=immediate
 try:out=engine.target_tree_forward(p,r,d,ids,return_logits=False,real_tree_count=32)
 finally:
  if had:runner.run_target_greedy=prior
  else:runner.__dict__.pop('run_target_greedy',None)
 require(len(saved)==1 and out['used_aclgraph'] and not out['used_tree_attention']
  and out['query_count']==case['M'] and out['logical_query_count']==case['Q']
  and out['num_draft_tokens']==case['budgets'],'original target real-query/B/physical route')
 require(runner._v146_last_target_key==target_key(case),'actual Native greedy key')
 return out,saved[0]

def qualify(engine,case,bank,work,*,stage):
 """All four roles enter every lane; no outer/nested stage or private label."""
 import torch
 target=not engine.is_draft;runner=engine.graph_runner;matcher=runner._target_outputs_match
 def initial():
  return prefix_evidence(engine,{i:v[:-1] for i,v in enumerate(bank.contexts)}) if target else None
 before=stage('target-source-prefix-before',initial)
 records=[];greedy=None;first=None;greedyKV=None
 for phase in ('first','changed','restored'):
  def prepare():
   if not target:return None
   # Validate the complete intended metadata before any rank starts its TP lane.
   call=metadata(engine,case,work,changed=phase=='changed')
   key=target_key(case);entry=runner.target_entries.get(key)
   return call,bool(entry and entry.runtime_validated)
  state=stage('target-greedy/'+phase+'/prepare',prepare)
  returned=stage('target-greedy/'+phase+'/Native',lambda:original_call(engine,case,work,changed=phase=='changed') if target else None)
  def drain():
   nonlocal first,greedy,greedyKV
   if not target:return None
   call,was=state;out,physical=returned;key=target_key(case)
   require(runner.target_entries[key].runtime_validated is (was or phase!='first'),'original greedy natural cold/changed/restored flags')
   if phase=='first':first=physical
   if phase=='restored':
    require(torch.equal(first,physical),'full-M original production tokens restored exactly')
    greedy=out;greedyKV=query_KV(engine,call[2].slot_mapping.long())
   records.append(dict(phase=phase,key=repr(key),runtime_validated_before=was,
    runtime_validated_after=bool(runner.target_entries[key].runtime_validated),execution=asdict(runner.last_target_execution),
    full_physical_tokens=physical.cpu().tolist()))
  stage('target-greedy/'+phase+'/drain',drain)
 hidden=None;firstKV=None;hrecords=[]
 for phase in ('first','changed','restored'):
  def prepare_hidden():
   if not target:return None
   call=metadata(engine,case,work);key=target_key(case,hidden=True);entry=runner.target_entries.get(key)
   x=(call[0]+1)%151936 if phase=='changed' else call[0]
   return x,call[1],call[2],bool(entry and entry.runtime_validated)
  state=stage('target-hidden/'+phase+'/prepare',prepare_hidden)
  def native_hidden():
   if not target:return None
   value=runner.run_tree_hidden(state[0],state[1],state[2])
   return value.clone() # FIRST operation after original full-M hidden return.
  current=stage('target-hidden/'+phase+'/Native',native_hidden)
  def drain_hidden():
   nonlocal hidden,firstKV
   if not target:return None
   key=target_key(case,hidden=True);was=state[3]
   require(runner._v146_last_target_key==key and runner.target_entries[key].runtime_validated is (was or phase!='first'),
    'original hidden natural cold/changed/restored flags')
   now=query_KV(engine,state[2].slot_mapping.long())
   require(all(bool(torch.isfinite(v).all()) for v in (current,*now)),'full physical hidden/queryKV finite before the next TP lane')
   if phase=='first':hidden=current;firstKV=now
   if phase=='restored':require(matcher((hidden,),(current,)) and matcher(firstKV,now),'all physical hidden/all64 queryKV restored original matcher')
   hrecords.append(dict(phase=phase,key=repr(key),runtime_validated_before=was,
    runtime_validated_after=bool(runner.target_entries[key].runtime_validated),execution=asdict(runner.last_target_execution)))
  stage('target-hidden/'+phase+'/drain',drain_hidden)
 def final():
  if not target:return None
  candidateKV=query_KV(engine,state[2].slot_mapping.long())
  require(len(candidateKV)==128 and matcher(greedyKV,candidateKV)
   and prefix_evidence(engine,{i:v[:-1] for i,v in enumerate(bank.contexts)})==before,
   'greedy/hidden all64 KV and protected source prefix')
  return dict(output=greedy,physical_tokens=first,hidden=hidden,query_KV=candidateKV,
   report=dict(greedy=records,hidden=hrecords,all_physical_hidden=True,all64_query_KV=True,
    full_physical_greedy_cloned_immediately=True,production_greedy_M=case['M'],separate_hidden_counterpart_M=case['M'],M1_head_recompute=False))
 return stage('target-full-physical-summary',final)


def reference(engine,case,bank,work,candidate,*,stage):
 import torch
 target=not engine.is_draft
 def prepare():
  if not target:return None
  histories={i+32:v[:-1] for i,v in enumerate(bank.reference_contexts)}
  return histories,prefix_evidence(engine,histories),metadata(engine,case,work,reference=True)
 prepared=stage('target-reference/prepare',prepare)
 def model():
  if not target:return None
  returned=engine.model(*prepared[2]);return returned.clone() # Immediate original all-M return.
 hidden=stage('target-reference/original-model',model)
 def local_query():
  if not target:return None
  histories,before,call=prepared;kv=query_KV(engine,call[2].slot_mapping.long())
  require(prefix_evidence(engine,histories)==before and all(bool(torch.isfinite(v).all()) for v in (hidden,*kv)),
   'reference model/query/prefix local checks before the next TP head lane')
  return kv
 kv=stage('target-reference/query-and-finite',local_query)
 def head():
  if not target:return None
  returned=engine.model.compute_greedy_tokens(hidden,engine.draft_vocab_size);return returned.clone()
 physical=stage('target-reference/original-same-M-head',head)
 def drain():
  if not target:return None
  matcher=engine.graph_runner._target_outputs_match
  require(matcher((candidate['hidden'],),(hidden,)) and len(kv)==128 and matcher(candidate['query_KV'],kv)
   and torch.equal(candidate['physical_tokens'],physical),
   'same-M independent all physical hidden/full tokens/all64KV; reference never initialized from candidate KV')
  require(torch.equal(candidate['output']['target_query_token_ids'],physical[:case['Q']]),'real Q agrees with immediate full physical return')
  verdict=engine._verify_tree_outputs_host(work['rows'],work['plans'],physical[:case['Q']],4,engine.device)
  return dict(query_KV=kv,verdict_tokens=verdict.token_ids.cpu().tolist(),accepted=verdict.accepted_node_indices.cpu().tolist(),
   report=dict(independent_owners=list(range(32,64)),same_M=case['M'],same_S=case['S'],
    all_physical_hidden_original_matcher=True,all64_query_KV_original_matcher=True,all_physical_greedy_exact=True,
    candidate_KV_copied=False,real_query_tokens=physical[:case['Q']].cpu().tolist()))
 return stage('target-reference/drain-original-matcher',drain)

def commit_reference(engine,bank,work,reference,commit):
 import torch
 emitted=commit['committed_tokens'];accepted=commit['accepted_paths']
 require(emitted and len(emitted)==len(accepted)==32,'original all-owner actual commit')
 if not engine.is_draft:
  require([[int(t) for t in row if t>=0] for row in reference['verdict_tokens']]==emitted
   and reference['accepted']==accepted,'independent original verifier agrees before own reference compaction')
  before=prefix_evidence(engine,{i+32:v[:-1] for i,v in enumerate(bank.reference_contexts)})
  offset=0
  for i,(v,plan,path) in enumerate(zip(bank.reference_contexts,work['plans'],accepted,strict=True)):
   chosen=[0]+[1+node for node in path if node>=0]
   src=torch.tensor([offset+j for j in chosen],dtype=torch.long,device=engine.device)
   dest=_slots(engine,i+32,range(len(v)-1,len(v)-1+len(chosen)))
   for array,own in zip(_arrays(engine),reference['query_KV'],strict=True):array.index_copy_(0,dest,own.index_select(0,src))
   offset+=plan.candidate_budget+1
  require(prefix_evidence(engine,{i+32:v[:-1] for i,v in enumerate(bank.reference_contexts)})==before,'independent reference compaction preserves old prefixes')
 for i,(old,tokens) in enumerate(zip(bank.reference_contexts,emitted,strict=True)):
  require(tokens and commit['actual_logical_tokens'][i]==old+tokens,'only actual original verdict advances logical history')
 bank.contexts=[list(v) for v in commit['actual_logical_tokens']]
 bank.reference_contexts=[old+list(tokens) for old,tokens in zip(bank.reference_contexts,emitted,strict=True)]
 require(bank.contexts==bank.reference_contexts,'two histories advance from independently agreed original verdict')
 return dict(own_reference_query_KV_compacted=not engine.is_draft,draft_reference_KV_not_overwritten=True,
  histories_equal=True,candidate_KV_copied=False,lease=bank.guard())
