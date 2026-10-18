"""Original target gates with real firstN owners, genuine pad, fresh own pages."""
from dataclasses import asdict
from types import SimpleNamespace
from base205 import BASE,require
from scope_v146 import target_key
from bank205 import _slots,_arrays,query_KV,prefix_evidence
from adapters173 import adapted
adapted(BASE/'target_v146.py',globals(),kind='target')
_original_reference=reference

def frame(case,work,*,reference=False,changed=False):
 from target_layout import pad_target,_lengths
 from mapper173 import expected_geometry
 from types import FunctionType
 e=work['_engine'];n=case['N'];plans=list(work['plans']);roots=[v['token_ids'][-1]for v in work['source']['contexts']];rows=[list(v)for v in work['rows']]
 ids=list(range(16,16+n))if reference else list(work['source']['ids'])
 if changed:roots=[(r+1)%151936 for r in roots]
 if reference or case.get('old_control',False):
  # Independent old/same-M reference uses the original pad algorithm with
  # its exact case envelope and disjoint already-prefetched scratch pages.
  def envelope(a,q,f):
   require((a,q,f)==(n,5*n,'causal'),'exact finite actual source geometry')
   result=expected_geometry(a,q,f)if case.get('old_control',False)else(80,16,_lengths(a,q,80,16))
   require(result[:2]==(case['M'],case['S']),'same-M requested reference envelope')
   return result
  ns=dict(pad_target.__globals__,envelope=envelope)
  fn=FunctionType(pad_target.__code__,ns,'original_pad_reference')
  proxy=SimpleNamespace(config=e.config,_tree_graph_scratch_sequence_ids=tuple(range(492,527))if reference else e._tree_graph_scratch_sequence_ids)
  padded=fn(proxy,plans,roots,rows,ids,[],[])
 else:padded=e._pad_spec_rhythm_target_tree_graph(plans,roots,rows,ids,[],[])
 pp,rr,dd,ss,real=padded
 require(real==n and len(pp)==case['S']and sum(p.candidate_budget+1 for p in pp)==case['M']and ss[:n]==ids,'actual original pad retains true realN and owners')
 return pp,rr,dd,ss

def original_call(engine,case,work,*,changed=False):
 p,r,d,ids=frame(case,work,changed=changed);runner=engine.graph_runner;key=target_key(case)
 old=runner.run_target_greedy;had='run_target_greedy'in runner.__dict__;prior=runner.__dict__.get('run_target_greedy');saved=[]
 from candidate_phases_v132 import update_snapshot
 before=update_snapshot(engine)
 def immediate(*args,**kwargs):
  returned=old(*args,**kwargs);full=tuple(v.clone()for v in returned)
  require(len(full)==1 and tuple(full[0].shape)==(case['M'],),'immediate complete physical head')
  saved.append(full[0]);return returned
 runner.run_target_greedy=immediate
 try:out=engine.target_tree_forward(p,r,d,ids,return_logits=False,real_tree_count=case['N'])
 finally:
  if had:runner.run_target_greedy=prior
  else:runner.__dict__.pop('run_target_greedy',None)
 require(len(saved)==1 and out['used_aclgraph']and not out['used_tree_attention']and out['query_count']==case['M']
  and out['logical_query_count']==case['Q']and out['num_draft_tokens']==case['budgets']
  and runner._v146_last_target_key==key,'already-natural entry, actual realN route, no fake16')
 after=update_snapshot(engine);records=getattr(engine,'_v205_length_observations',None)
 if records is not None:
  current=dict(N=case['N'],M=case['M'],S=case['S'],key=repr(key),host_lengths=dict(q=[5]*case['N']+list(case['dummy_lengths']),KV=[v.prefix_len+v.candidate_budget+1 for v in p]),changed_root=changed)
  previous=getattr(runner,'_v205_previous_lengths',{}).get(key)
  current.update(previous_same_key=previous,before=before,after=after,delta={k:after[k]-before[k]for k in before},execution=asdict(runner.last_target_execution))
  require(all(v>=0 for v in current['delta'].values()),'original dynamic task-update nonnegative')
  records.append(current)
  if not hasattr(runner,'_v205_previous_lengths'):runner._v205_previous_lengths={}
  runner._v205_previous_lengths[key]=current['host_lengths']
 return out,saved[0]

def reference(engine,case,bank,work,candidate,*,stage):
 result=_original_reference(engine,case,bank,work,candidate,stage=stage)
 def actual_owners():
  if engine.is_draft:return None
  result['report']['independent_owners']=list(range(16,16+case['N']));result['report']['all128_queryKV_including_dummy']=True
  result['report']['unselected_real_reference_owners_protected']=True
  return result
 return stage('target-reference/truthful-selected-own-owners',actual_owners)

def commit_reference(engine,bank,work,reference,commit):
 import torch
 n=len(work['plans']);emitted=commit['committed_tokens'];accepted=commit['accepted_paths']
 require(len(emitted)==len(accepted)==n and 9<=n<=15 and len(bank.contexts)==len(bank.reference_contexts)==16,'actual subset commit in retained full16 bank')
 untouched=[list(v)for v in bank.contexts[n:]];untouched_ref=[list(v)for v in bank.reference_contexts[n:]]
 if not engine.is_draft:
  require([[int(t)for t in row if t>=0]for row in reference['verdict_tokens']]==emitted and reference['accepted']==accepted,'independent original BONUS verifier')
  before=prefix_evidence(engine,{i+16:v[:-1]for i,v in enumerate(bank.reference_contexts)})
  offset=0
  for i,(plan,path)in enumerate(zip(work['plans'],accepted,strict=True)):
   chosen=[0]+[1+j for j in path if j>=0];src=torch.tensor([offset+j for j in chosen],dtype=torch.long,device=engine.device)
   dest=_slots(engine,i+16,range(len(bank.reference_contexts[i])-1,len(bank.reference_contexts[i])-1+len(chosen)))
   for array,own in zip(_arrays(engine),reference['query_KV'],strict=True):array.index_copy_(0,dest,own.index_select(0,src))
   offset+=plan.candidate_budget+1
  require(prefix_evidence(engine,{i+16:v[:-1]for i,v in enumerate(bank.reference_contexts)})==before,'own compaction protects all16 prior prefixes')
 for i,tokens in enumerate(emitted):
  require(tokens and commit['actual_logical_tokens'][i]==bank.contexts[i]+tokens,'real verdict alone advances selected history')
  bank.contexts[i]=list(commit['actual_logical_tokens'][i]);bank.reference_contexts[i]+=list(tokens)
 require(bank.contexts[n:]==untouched and bank.reference_contexts[n:]==untouched_ref and bank.contexts==bank.reference_contexts,'unselected owners never advanced; independent histories equal')
 return dict(histories_equal=True,candidate_KV_copied=False,selected_N=n,reference_owner_offset=16,unselected_owners_preserved=True,own_reference_query_KV_compacted=not engine.is_draft,lease=bank.guard())

def old_control(engine,case,bank,work,stage):
 import torch
 target=not engine.is_draft
 old=dict(case,M=case['oldM'],S=case['oldS'],dummy_lengths=case['oldpad'],old_control=True)
 protected=stage('old128/all16-candidate-prefix-before',lambda:prefix_evidence(engine,{i:v[:-1]for i,v in enumerate(bank.contexts)})if target else None)
 prepared=stage('old128/prepare',lambda:metadata(engine,old,work)if target else None)
 value=stage('old128/actual-greedy',lambda:original_call(engine,old,work)if target else None)
 def candidate_finite():
  if not target:return None
  kv=query_KV(engine,prepared[2].slot_mapping.long())
  require(prefix_evidence(engine,{i:v[:-1]for i,v in enumerate(bank.contexts)})==protected and all(bool(torch.isfinite(v).all())for v in kv),'old candidate protects all16 prefixes and finite queryKV before reference TP call')
  return kv
 kv=stage('old128/immediate-query-KV',candidate_finite)
 own=stage('old128/reference-prepare',lambda:(metadata(engine,old,work,reference=True),prefix_evidence(engine,{i+16:v[:-1]for i,v in enumerate(bank.reference_contexts)}))if target else None)
 h=stage('old128/independent-full128-model',lambda:engine.model(*own[0]).clone()if target else None)
 def finite():
  if not target:return None
  refs=query_KV(engine,own[0][2].slot_mapping.long())
  require(all(bool(torch.isfinite(v).all())for v in(h,*refs))and prefix_evidence(engine,{i+16:v[:-1]for i,v in enumerate(bank.reference_contexts)})==own[1],'old own128 finite and untouched prefixes before TP head')
  return refs
 refs=stage('old128/independent-query-finite',finite)
 head=stage('old128/independent-full128-head',lambda:engine.model.compute_greedy_tokens(h,engine.draft_vocab_size).clone()if target else None)
 def finish():
  if not target:return None
  require(torch.equal(value[1],head)and engine.graph_runner._target_outputs_match(kv,refs),'old actual128 same-M head and all128 KV independent')
  q=case['Q'];verdict=engine._verify_tree_outputs_host(work['rows'],work['plans'],head[:q],4,engine.device)
  return dict(real=head[:q].cpu().tolist(),all128=head.cpu().tolist(),verdict_tokens=verdict.token_ids.cpu().tolist(),accepted=verdict.accepted_node_indices.cpu().tolist(),sameM_head_KV=True,new_hidden_key=False)
 return stage('old128/sameM-control-gate',finish)
