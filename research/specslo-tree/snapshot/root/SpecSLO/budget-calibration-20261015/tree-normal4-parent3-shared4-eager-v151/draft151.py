"""Actual original FastV5 B3 source and independent four-step PA reference.

General HOST grants are separate from real model steps and returned tensors.
All source/restore calls are fresh epochs, never seeded with candidate KV.
"""
import copy,time
from scope151 import require,digest,bucket
from originals151 import source_contexts
from finite151 import require_finite
import source151 as epoch


def parent_slots(e,owners,p):
 import torch
 return torch.cat([epoch.slots(e,i,range(p,p+3)) for i in owners])


def cpu_output(output):
 return {name:(value.tolist() if hasattr(value,'tolist') else
  [v.tolist() if hasattr(v,'tolist') else v for v in value] if isinstance(value,list) else copy.deepcopy(value))
  for name,value in output.items() if name in ('draft_token_ids','root_token_ids','eager_frontier_tokens','node_confidences','draft_confidence')}


def contexts(case,changed=False):
 source=source_contexts(case);out=[]
 for i,row in enumerate(source):
  end=case.prefix+(case.repair if i<case.normal else 0)+1
  tokens=list(row['token_ids'][:end])
  if changed:tokens[case.prefix:]=[(t+1)%151936 for t in tokens[case.prefix:]]
  out.append(dict(row,token_ids=tokens,context_length=end))
 return out


def fixture(e,case,changed=False,*,observe_parent=False):
 from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan as plan,pack_selected_tree_plan,tree_primary_path
 cs=contexts(case,changed) # source initialization is orchestrated by prepare() all4 lanes
 if not e.is_draft:return dict(contexts=cs)
 p=case.prefix;ids=list(range(case.count));eagerids=ids[case.normal:]
 parent_plans=[plan(2,4,p,4096,candidate_budget=case.parent_B) for _ in eagerids]
 parent_roots=[cs[i]['token_ids'][-1] for i in eagerids]
 if observe_parent:
  import observer151 as trace_observer
  parent,parent_observed=trace_observer.run(e,lambda:e.draft_tree_forward(parent_plans,parent_roots,eagerids))
  parent_KV=epoch.read(e,parent_slots(e,eagerids,p))
 else:
  parent=e.draft_tree_forward(parent_plans,parent_roots,eagerids);parent_observed=parent_KV=None
 plans=[];roots=[];catchups=[];sources={};parent_confidences={}
 for i in range(case.normal):
  tokens=cs[i]['token_ids'];root=tokens[-1];plans.append(plan(2,4,len(tokens)-1,4096,candidate_budget=case.normal_budgets[i]));roots.append(root)
  catchups.append(tokens[p:])
 for row,i in enumerate(eagerids):
  packed=pack_selected_tree_plan(parent_plans[row],range(case.parent_B));nodes=tuple(tree_primary_path(packed))
  require(nodes==tuple(range(case.parent_B)),'actual B3 parent primary path, never hardcoded B5 primary4')
  values=[int(parent['draft_token_ids'][row][j]) for j in range(case.parent_B)]
  child=e._spec_rhythm_tree_eager_plan(packed,type('LogicalPrefix',(),{'token_ids':cs[i]['token_ids']})(),3)
  require(child.prefix_len==p+len(nodes)+1,'original Native eager prefix follows actual parent length')
  plans.append(child);roots.append(parent_roots[row]);catchups.append(())
  sources[i]=(packed,values);parent_confidences[i]=sum(float(parent['node_confidences'][row][j]) for j in nodes)/len(nodes)
 return dict(contexts=cs,plans=plans,roots=roots,ids=ids,sources=sources,catchups=catchups,
  parent_output=cpu_output(parent),parent_confidences=parent_confidences,
  truthful_ledger=epoch.ledger(e),source_epoch=e._v151_source_epoch,source_lease=copy.deepcopy(e._v151_source_lease),parent_observed=parent_observed,parent_KV=parent_KV,
  parent_actual_execution=dict(query_count=parent['query_count'],graph_calls=parent['graph_calls'],model_calls=parent['model_calls'],
   actual_steps=3,actual_chain_M=bucket(case.eager),source_build_outside_mixed_generation_interval=True))


def invoke(e,f):
 return e.draft_tree_forward(f['plans'],f['roots'],f['ids'],eager_parent_sources=f['sources'],committed_catchup_token_ids=f['catchups'])


def PA(e,ids,positions,tokens,physical):
 import torch
 e._ensure_cache_capacity(list(ids),list(positions));x=torch.tensor(tokens,dtype=torch.long,device=e.device)
 pos,meta=e._prepare_attention_metadata(list(ids),list(positions),False)
 return e.graph_runner._pad_inputs(x,pos,meta,physical)


def top2(e,hidden):
 import torch
 probability=torch.softmax(e.model.compute_logits(hidden)[:,:e.draft_vocab_size].float(),dim=-1)
 confidence,tokens=probability.topk(2,dim=-1)
 return tokens.long()*1_000_001+torch.round(confidence.clamp(0,1)*1_000_000).long(),tokens[:,0]


def chain(e,ids,prefixes,roots,steps,physical):
 import torch
 x,_,_=PA(e,ids,prefixes,roots,physical);h=[];encoded=[]
 for step in range(steps):
  _,pos,meta=PA(e,ids,[p+step for p in prefixes],roots,physical)
  hidden=e.model(x,pos,meta);h.append(hidden.clone());value,x=top2(e,hidden);encoded.append(value)
 return torch.stack(encoded,1),h


def decode(encoded,real,steps):
 vals=encoded[:real].detach().cpu().tolist();rows=[];scores=[]
 for row in vals:
  rows.append([x[0]//1_000_001 for x in row]+[0]*(4-steps)+[x[1]//1_000_001 for x in row]+[0]*(4-steps))
  scores.append([x[0]%1_000_001/1_000_000 for x in row]+[0.]*(4-steps)+[x[1]%1_000_001/1_000_000 for x in row]+[0.]*(4-steps))
 return rows,scores


def independent(e,case,f):
 """Compute parent → missing suffix → frontier → chain from fresh real prefix."""
 from kv_ledger import record_overwrite
 p=case.prefix # independent fresh rebuild is performed by prepare() all4 before this draft-only callback
 eid=list(range(case.normal,case.count));prefix=epoch.protect(e,[p]*case.count)
 parent_encoded,parent_hidden=chain(e,eid,[p]*case.eager,[f['roots'][i] for i in eid],3,bucket(case.eager))
 parent_rows,parent_scores=decode(parent_encoded,case.eager,3)
 parent_KV=epoch.read(e,parent_slots(e,eid,p))
 ledger={i:dict(safe_before=p,tokens={}) for i in range(case.normal)}
 for j,i in enumerate(eid):
  require(parent_rows[j][:3]==f['sources'][i][1],'NEW reference actual parent B3 predictions differ')
  ledger[i]=dict(safe_before=p,tokens={p+k:t for k,t in enumerate([f['roots'][i],*parent_rows[j][:2]])})
 repair_hidden=None
 if case.repair:
  repair_call=PA(e,list(range(case.normal)),[p]*case.normal,[f['catchups'][i][0] for i in range(case.normal)],bucket(case.normal))
  repair_hidden=e.model(*repair_call).clone()
 for i in range(case.normal):ledger[i]['safe_before']=p+case.repair
 leaves=[row[2] for row in parent_rows];frontier_call=PA(e,eid,[p+3]*case.eager,leaves,bucket(case.eager))
 frontier_hidden=e.model(*frontier_call).clone();frontier_tokens=e.model.compute_greedy_tokens(frontier_hidden,e.draft_vocab_size)
 frontiers={i:int(frontier_tokens[j]) for j,i in enumerate(eid)}
 for i,leaf in zip(eid,leaves):record_overwrite(ledger[i],p+3,[leaf])
 prefixes=[int(v.prefix_len) for v in f['plans']];roots=[*f['roots'][:case.normal],*[frontiers[i] for i in eid]]
 encoded,hidden=chain(e,f['ids'],prefixes,roots,4,bucket(case.count));rows,scores=decode(encoded,case.count,4)
 for i,pos,root,row in zip(f['ids'],prefixes,roots,rows):
  written=[root,*row[:3]]
  if i<case.normal:ledger[i]=dict(safe_before=pos,tokens={pos+j:t for j,t in enumerate(written)})
  else:record_overwrite(ledger[i],pos,written)
 require(epoch.protect(e,[p]*case.count)==prefix,'independent chain wrote protected original prefix')
 return dict(rows=rows,scores=scores,encoded=encoded,hidden=hidden,repair_hidden=repair_hidden,
  frontier_hidden=frontier_hidden,frontier_tokens=frontier_tokens,frontiers=frontiers,roots=roots,ledger=ledger,
  source_epoch=e._v151_source_epoch,parent_hidden=parent_hidden,parent_encoded=parent_encoded,parent_KV=parent_KV)


def counts(case):return [4+case.repair]*case.normal+[8]*case.eager

def compare(e,case,actual,observed,actual_KV,actual_ledger,reference):
 import torch
 matcher=e.graph_runner._target_outputs_match
 expected_KV=epoch.logical_KV(e,[case.prefix]*case.count,counts(case))
 actual_hidden=[h for row in observed['draft'] for h in row['hidden']]+[row['hidden'] for kind in ('hidden','greedy') for row in observed[kind]]
 reference_hidden=[*reference['hidden'],reference['frontier_hidden']]+([reference['repair_hidden']] if case.repair else [])
 finite_actual=require_finite('mixed allphysical hidden/all28 written queryKV',actual_hidden,actual_KV)
 finite_reference=require_finite('independent mixed allphysical hidden/all28 written queryKV',reference_hidden,expected_KV)
 require(len(observed['hidden'])==int(bool(case.repair)) and len(observed['greedy'])==len(observed['draft'])==1
  ,'true FastV5 repair0/1 + frontier + FOUR step chain')
 if case.repair:require(matcher((observed['hidden'][0]['hidden'],),(reference['repair_hidden'],)),'independent complete physical repair hidden')
 require(matcher((observed['greedy'][0]['hidden'],),(reference['frontier_hidden'],))
  and matcher(observed['draft'][0]['hidden'],reference['hidden']),'independent full-physical frontier/all4 hidden')
 require(torch.equal(observed['draft'][0]['encoded'],reference['encoded']),'exact all-physical top2 confidence encoded')
 require(actual['draft_token_ids']==reference['rows'] and actual['root_token_ids']==reference['roots']
  and actual['eager_frontier_tokens']==reference['frontiers'],'exact actual row/root/frontier')
 require(all(torch.equal(v,torch.tensor(s)) for v,s in zip(actual['node_confidences'],reference['scores']))
  and torch.equal(actual['draft_confidence'],torch.tensor([sum(s)/8 for s in reference['scores']])),'actual FastV5 shared4 quantized confidence denominator8')
 require(actual_ledger==reference['ledger'] and matcher(actual_KV,expected_KV),
  'all28-layer actual written-query KV and independent exact ledger')
 keys=[]
 for kind in ('hidden','greedy','draft'):
  for row in observed[kind]:
   entry=(e.graph_runner.draft_entries if kind=='draft' else e.graph_runner.target_entries).get(row['key'])
   require(entry is not None and entry.runtime_validated,'original Native natural gate open')
   if kind=='draft':require(entry.validated_real_row_count>=case.count,'new real-row gate not inherited')
   keys.append(dict(kind=kind,key=repr(row['key']),runtime_validated=True,validated_real_row_count=getattr(entry,'validated_real_row_count',None)))
 return dict(independent_fresh_history=True,candidate_KV_used_as_reference=False,all_physical_hidden=True,
  exact_full_physical_encoded=True,all28_written_query_KV=True,exact_ledger=True,original_source_prefix_protected=True,
  shared_steps=4,return_slots_per_owner=8,real_generated_nodes=8*case.count,selected_candidates=case.selected,
  actual_native_keys=keys,reference_epoch=reference['source_epoch'],finite_actual_snapshots=finite_actual,finite_independent_snapshots=finite_reference)



def prepare(e,case,stage,changed=False):
 """All4 initialization lanes, draft-only math, independent fresh restore."""
 import torch
 import observer151
 cs=stage('source-contexts',lambda:contexts(case,changed))
 repairs=[case.repair]*32+[0]*32
 epoch.rebuild(e,cs,repairs,stage,role='draft')
 def parent_source():
  if not e.is_draft:return None
  return fixture(e,case,changed,observe_parent=True)
 f=stage('actual-parent3-source-and-immediate-observer',parent_source)
 before=stage('mixed-original-counter-before',lambda:dict(getattr(e,'_dynamic_tree_draft_counts',{})) if e.is_draft else None)
 def actual_call():
  if not e.is_draft:return None
  protected=epoch.protect(e,[128]*64);out,seen=observer151.run(e,lambda:invoke(e,f))
  return dict(output=out,observed=seen,KV=epoch.logical_KV(e,[128]*64,counts(case)),ledger=epoch.ledger(e),prefix=protected)
 actual=stage('actual-parent3-normal4-child3-shared4',actual_call)
 def actual_drain():
  if not e.is_draft:return None
  seen=actual['observed'];out=actual['output']
  require(len(seen['draft'])==1 and len(seen['draft'][0]['hidden'])==4 and tuple(seen['draft'][0]['encoded'].shape)==(64,4,2),
   'actual raw8/FOUR model steps; raw6 padded zeros cannot qualify')
  require(epoch.protect(e,[128]*64)==actual['prefix'],'actual source committed prefix remains unchanged')
  return dict(getattr(e,'_dynamic_tree_draft_counts',{}))
 after=stage('actual-raw8-four-step-finite-drain',actual_drain)
 epoch.rebuild(e,cs,repairs,stage,role='draft')
 ref=stage('own-fresh-parent3-frontier-shared4-independent',lambda:independent(e,case,f) if e.is_draft else None)
 def compare_all():
  if not e.is_draft:return None
  a=actual;proof=compare(e,case,a['output'],a['observed'],a['KV'],a['ledger'],ref)
  match=e.graph_runner._target_outputs_match;pt=f['parent_observed']
  require(len(pt['draft'])==1 and len(pt['draft'][0]['hidden'])==3
   and match(pt['draft'][0]['hidden'],ref['parent_hidden'])
   and torch.equal(pt['draft'][0]['encoded'],ref['parent_encoded']) and match(f['parent_KV'],ref['parent_KV']),
   'parent remains true B3 own3 hidden/encoded/all28KV; not new parent4 scope')
  proof['parent3_actual_independent_match']=True
  return proof
 proof=stage('own-allphysical-encoded-56KV-and-parent3-gate',compare_all)
 epoch.rebuild(e,cs,repairs,stage,role='draft')
 restored=stage('actual-whole-source-parent3-restore',lambda:fixture(e,case,changed) if e.is_draft else None)
 def restore():
  if not e.is_draft:return None
  output=invoke(e,restored)
  require(cpu_output(output)==cpu_output(actual['output']) and epoch.ledger(e)==actual['ledger']
   and e.graph_runner._target_outputs_match(actual['KV'],epoch.logical_KV(e,[128]*64,counts(case))),
   'entire actual parent3/shared4 source restored after own independent reference')
  return dict(case=case.__dict__,fixture=restored,mixed=cpu_output(output),numerical_proof=proof,
   generation_observation=dict(actual_shared_steps=4,real_raw_nodes=512,returned_slots=512,selected_normal=96,selected_eager=96,
    selected_total=192,original_counters_before=before,original_counters_after=after,
    original_counter_delta={k:after.get(k,0)-before.get(k,0) for k in set(before)|set(after)},raw6_padding_substitution=False),
   actual_source_lease=restored['source_lease'],controlled_grants_not_live_policy=True,adopted=False,Goodput_measured=False)
 trace=stage('actual-whole-shared4-restore-gate',restore)
 def share():
  import torch.distributed as dist
  packet=[trace if e.is_draft else None]
  dist.broadcast_object_list(packet,src=e.topology.draft_leader_rank,group=e.groups.verification_coordination_group)
  require(packet[0] is not None,'actual draft-produced source/typed rows must be shared')
  return packet[0]
 return stage('actual-source-and-raw8-proposal-metadata-all4',share)


def warm(e,case,stage):
 """Real unpublished F/T/F inputs open original parent3 and shared4 gates."""
 outputs=[]
 for index,changed in enumerate((False,True,False)):
  cs=stage('warm'+str(index)+'/contexts',lambda:contexts(case,changed))
  epoch.rebuild(e,cs,[case.repair]*32+[0]*32,lambda n,f:stage('warm'+str(index)+'/'+n,f),role='draft')
  def actual():
   if not e.is_draft:return None
   f=fixture(e,case,changed);return cpu_output(invoke(e,f))
  outputs.append(stage('warm'+str(index)+'/true-parent3-and-shared4',actual))
 stage('warm-whole-parent3-shared4-FTF',lambda:require(outputs[0]==outputs[2],'whole actual current parent3/shared4 F/T/F restoration') if e.is_draft else None)
 return dict(changed_inputs=[False,True,False],actual_model_steps=4,parent_model_steps=3,Native_flags_assigned=False)
