"""Actual original FastV5 B3 source and independent four-step PA reference.

General HOST grants are separate from real model steps and returned tensors.
All source/restore calls are fresh epochs, never seeded with candidate KV.
"""
import copy,time
from scope152 import require,digest,bucket
from reuse152 import source_contexts
from finite151 import require_finite
import source152 as epoch


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
  end=case.prefix+(case.repair_vector[i])+1
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
  import observer152 as trace_observer
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




def counts(case):return [4+case.repair_vector[i] for i in range(32)]+[8]*case.eager




def prepare(e,case,stage,changed=False):
 """All4 initialization lanes, draft-only math, independent fresh restore."""
 import torch
 import observer152
 cs=stage('source-contexts',lambda:contexts(case,changed))
 repairs=list(case.repair_vector)
 epoch.rebuild(e,cs,repairs,stage,role='draft')
 def parent_source():
  if not e.is_draft:return None
  return fixture(e,case,changed,observe_parent=True)
 f=stage('actual-parent3-source-and-immediate-observer',parent_source)
 before=stage('mixed-original-counter-before',lambda:dict(getattr(e,'_dynamic_tree_draft_counts',{})) if e.is_draft else None)
 def actual_call():
  if not e.is_draft:return None
  protected=epoch.protect(e,[128]*64);out,seen=observer152.run(e,lambda:invoke(e,f))
  return dict(output=out,observed=seen,KV=epoch.logical_KV(e,[128]*64,counts(case)),ledger=epoch.ledger(e),prefix=protected,TTU=copy.deepcopy(e._v152_last_mixed_TTU))
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
   generation_observation=dict(actual_shared_steps=4,real_raw_nodes=512,repair_count=case.repair_count,repair_owners=list(case.repair_owners),union_M=case.union_M,fusion_enabled=bool(case.repair_count),actual_TTU=actual['TTU'],dynamic_prefix_cost_measured=False,returned_slots=512,selected_normal=96,selected_eager=96,
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
  epoch.rebuild(e,cs,list(case.repair_vector),lambda n,f:stage('warm'+str(index)+'/'+n,f),role='draft')
  def actual():
   if not e.is_draft:return None
   f=fixture(e,case,changed);return cpu_output(invoke(e,f))
  outputs.append(stage('warm'+str(index)+'/true-parent3-and-shared4',actual))
 stage('warm-whole-parent3-shared4-FTF',lambda:require(outputs[0]==outputs[2],'whole actual current parent3/shared4 F/T/F restoration') if e.is_draft else None)
 return dict(changed_inputs=[False,True,False],actual_model_steps=4,parent_model_steps=3,Native_flags_assigned=False)

def _table_hash(e):
 import torch,hashlib
 return hashlib.sha256(e.cache_block_tables.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

def _TTU(e):
 runner=e.graph_runner
 values={k:v for k,v in e.graph_metrics().items() if type(v)is int and ('task_update' in k or 'pa_workspace' in k)}
 values['actual_task_update_skipped_replays']=runner.task_update_skip_replay_count
 entries={}
 for family,name in (('target','target_entries'),('draft','draft_entries')):
  for key,v in getattr(runner,name).items():
   entries[family+'/'+repr(key)]=dict(actual_seq_lengths_q=[list(t) for t in v.actual_seq_lengths_q],
    sequence_lens=[list(t) for t in v.sequence_lens],tasks=len(v.tasks),runtime_validated=bool(v.runtime_validated))
 return dict(counters=values,entries=entries)

def invoke(e,f):
 """Real union only inside mixed call. Parent/reference/next3 stay outside TTU window."""
 allocation=e.cache_allocation;tables=copy.deepcopy(allocation.block_tables);device_hash=_table_hash(e)
 require(f['source_epoch']==e._v151_source_epoch,'current source epoch before mixed execution')
 def lease_guard():
  require(e.cache_allocation is allocation and allocation.block_tables==tables
   and e._v151_source_epoch==f['source_epoch'] and _table_hash(e)==device_hash,
   'current real host/device allocation and source epoch lease')
 before=_TTU(e)
 out=e.draft_tree_forward(f['plans'],f['roots'],f['ids'],eager_parent_sources=f['sources'],
  committed_catchup_token_ids=f['catchups'],_repair_frontier_fusion_lease_guard=lease_guard)
 lease_guard();after=_TTU(e)
 require(set(before['counters'])==set(after['counters']) and all(after['counters'][k]>=v for k,v in before['counters'].items()),'actual TTU monotone separate from original21 discovery counters')
 e._v152_last_mixed_TTU=dict(before=before,after=after,delta={k:after['counters'][k]-v for k,v in before['counters'].items()},
  measurement='untimed actual mixed invocation only',prefix=128,dynamic_TTU_proved=False,dynamic_prefix_cost_measured=False,
  logical_draft_PA_uses_original_target_family=True)
 return out

def independent(e,case,f):
 """Own same-U recurrence; no old two-PA candidate KV initializes this oracle."""
 from kv_ledger import record_overwrite
 p=case.prefix;eid=list(range(32,64));baseline=epoch.protect(e,[p]*64)
 parent_encoded,parent_hidden=chain(e,eid,[p]*32,[f['roots'][i] for i in eid],3,bucket(32))
 parent_rows,parent_scores=decode(parent_encoded,32,3);parent_KV=epoch.read(e,parent_slots(e,eid,p))
 ledger={i:dict(safe_before=p,tokens={}) for i in range(32)}
 for j,i in enumerate(eid):
  require(parent_rows[j][:3]==f['sources'][i][1],'independent actual parent3 exact recursive predictions')
  ledger[i]=dict(safe_before=p,tokens={p+k:t for k,t in enumerate([f['roots'][i],*parent_rows[j][:2]])})
 leaves=[row[2] for row in parent_rows];rids=list(case.repair_owners)
 repair_hidden=None;union_hidden=None;union_call=None
 if rids:
  # Independent original PA producer on all genuine frontier+repair owners.
  # Here E=F=32: there is no interior frontier padding to emulate.
  # This does NOT call candidate plan_frame/build_union/run_union.
  ids=eid+rids;positions=[p+3]*32+[p]*len(rids)
  tokens=leaves+[f['catchups'][i][0] for i in rids]
  union_call=PA(e,ids,positions,tokens,case.union_M)
  union_hidden=e.model(*union_call).clone()
  frontier_tokens=e.model.compute_greedy_tokens(union_hidden[:32],e.draft_vocab_size).clone()
  frontier_hidden=None
 else:
  call=PA(e,eid,[p+3]*32,leaves,32);frontier_hidden=e.model(*call).clone()
  frontier_tokens=e.model.compute_greedy_tokens(frontier_hidden,e.draft_vocab_size).clone()
 for i in range(32):ledger[i]['safe_before']=p+case.repair_vector[i]
 frontiers={i:int(frontier_tokens[j]) for j,i in enumerate(eid)}
 for i,leaf in zip(eid,leaves,strict=True):record_overwrite(ledger[i],p+3,[leaf])
 prefixes=[int(v.prefix_len) for v in f['plans']];roots=[*f['roots'][:32],*[frontiers[i] for i in eid]]
 encoded,hidden=chain(e,f['ids'],prefixes,roots,4,64);rows,scores=decode(encoded,64,4)
 for i,pos,root,row in zip(f['ids'],prefixes,roots,rows,strict=True):
  written=[root,*row[:3]]
  if i<32:ledger[i]=dict(safe_before=pos,tokens={pos+j:t for j,t in enumerate(written)})
  else:record_overwrite(ledger[i],pos,written)
 require(epoch.protect(e,[p]*64)==baseline,'fresh union oracle preserves all original committed prefixes')
 return dict(rows=rows,scores=scores,encoded=encoded,hidden=hidden,repair_hidden=repair_hidden,
  frontier_hidden=frontier_hidden,union_hidden=union_hidden,union_call=union_call,frontier_tokens=frontier_tokens,
  frontiers=frontiers,roots=roots,ledger=ledger,source_epoch=e._v151_source_epoch,
  parent_hidden=parent_hidden,parent_encoded=parent_encoded,parent_KV=parent_KV,
  independent_union_PA=True,reference_model_M=case.union_M if rids else 32)

def compare(e,case,actual,observed,actual_KV,actual_ledger,reference):
 import torch
 matcher=e.graph_runner._target_outputs_match;ownKV=epoch.logical_KV(e,[128]*64,counts(case))
 actual_hidden=[h for row in observed['draft'] for h in row['hidden']]+[row['hidden'] for kind in ('hidden','greedy') for row in observed[kind]]
 reference_hidden=[*reference['hidden'],reference['union_hidden'] if case.repair_count else reference['frontier_hidden']]
 finite_actual=require_finite('actual union/prelude and fullphysical shared4 plus56KV',actual_hidden,actual_KV)
 finite_reference=require_finite('independent SAME-U fullphysical prelude/shared4 and56KV',reference_hidden,ownKV)
 if case.repair_count:
  require(len(observed['hidden'])==1 and not observed['greedy'] and len(observed['union_head'])==1,'one actual union hidden, outside F-head and4step chain')
  h=observed['hidden'][0];x,p,m=reference['union_call'];meta=h['metadata']
  require(h['hidden'].shape[0]==case.union_M and matcher((h['hidden'],),(reference['union_hidden'],)),'own same-U ALL physical hidden including genuine tail padding')
  require(torch.equal(meta['inputs'],x) and torch.equal(meta['positions'],p)
   and torch.equal(meta['slot_mapping'],m.slot_mapping) and torch.equal(meta['context_lens'],m.context_lens)
   and torch.equal(meta['block_tables'],m.block_tables) and meta['host_Q']==tuple(m.actual_seq_lengths_q)
   and meta['host_KV']==tuple(m.sequence_lens),'independent original union PA metadata actual owner/slot/context/padding')
  head=observed['union_head'][0]
  require(head['M']==32 and torch.equal(head['tokens'],reference['frontier_tokens']),'actual production head exactly F32, whole token output own same-U')
 else:
  require(not observed['hidden'] and len(observed['greedy'])==1,'R0 original unfused frontier fallback')
  require(matcher((observed['greedy'][0]['hidden'],),(reference['frontier_hidden'],)),'own original fallback complete physical frontier hidden')
 require(len(observed['draft'])==1 and len(observed['draft'][0]['hidden'])==4
  and tuple(observed['draft'][0]['encoded'].shape)==(64,4,2),'actual64x4x2 encoded; no shared3 ABI-padding substitution')
 require(matcher(observed['draft'][0]['hidden'],reference['hidden']) and torch.equal(observed['draft'][0]['encoded'],reference['encoded']),'own complete physical FOUR-step chain hidden and encoded512')
 require(actual['draft_token_ids']==reference['rows'] and actual['root_token_ids']==reference['roots'] and actual['eager_frontier_tokens']==reference['frontiers'],'own exact canonical8/root/frontier')
 require(all(torch.equal(v,torch.tensor(s)) for v,s in zip(actual['node_confidences'],reference['scores'],strict=True))
  and torch.equal(actual['draft_confidence'],torch.tensor([sum(s)/8 for s in reference['scores']])),'own exact raw score8 and denominator8 scalar confidence')
 require(actual_ledger==reference['ledger'] and len(actual_KV)==len(ownKV)==56 and matcher(actual_KV,ownKV),'own union history all56 written-query KV and exact lifecycle ledger')
 keys=[]
 for kind in ('hidden','greedy','draft'):
  for row in observed[kind]:
   entry=(e.graph_runner.draft_entries if kind=='draft' else e.graph_runner.target_entries).get(row['key'])
   require(entry is not None and entry.runtime_validated,'actual original natural Native entry gate')
   if kind=='draft':require(entry.validated_real_row_count>=64,'actual64 real-row qualification')
   keys.append(dict(kind=kind,key=repr(row['key']),runtime_validated=True,validated_real_row_count=getattr(entry,'validated_real_row_count',None)))
 return dict(independent_fresh_history=True,candidate_KV_used_as_reference=False,sameM_pass=True,
  independent_union_PA=True,all_physical_hidden=True,exact_full_physical_encoded=True,all28_written_query_KV=True,
  all56_written_query_KV=True,exact_ledger=True,original_source_prefix_protected=True,shared_steps=4,
  return_slots_per_owner=8,real_generated_nodes=512,selected_candidates=192,actual_native_keys=keys,
  repair_count=case.repair_count,repair_owners=list(case.repair_owners),union_M=case.union_M,head_M=32,
  reference_epoch=reference['source_epoch'],finite_actual_snapshots=finite_actual,finite_independent_snapshots=finite_reference,
  crossM_behavior_equal=None,crossM_not_evaluated_by_this_protocol_component=True,cost_advantage=None)
