"""True parent4, shared4 child3 and independent same-M complete source history."""
import copy
from base169 import V152,functions,require
from scope169 import bucket,digest
from provider169 import source_contexts
from finite151 import require_finite
from metadata169 import capture_actual,compare_fresh
import source169 as epoch
functions(V152/'draft152.py',globals(),names=('cpu_output','PA','top2','chain','decode','_TTU'))

def contexts(case,changed=False):
 out=[]
 for i,row in enumerate(source_contexts(case)):
  end=case.prefix+case.repair_vector[i]+1;tokens=list(row['token_ids'][:end])
  require(len(tokens)==end,'every actual normal/root source token exists')
  if changed:tokens[case.prefix:]=[(v+1)%151936 for v in tokens[case.prefix:]]
  out.append(dict(row,token_ids=tokens,context_length=end))
 return out

def parent_slots(e,owners,p):
 import torch
 return torch.cat([epoch.slots(e,i,range(p,p+4)) for i in owners])

def fixture(e,case,changed=False,*,observe_parent=False):
 from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan as plan,pack_selected_tree_plan,tree_primary_path
 cs=contexts(case,changed)
 if not e.is_draft:return dict(contexts=cs)
 import observer169
 eid=list(range(16,32));p=case.prefix
 parents=[plan(2,4,p,4096,candidate_budget=4) for _ in eid];roots=[cs[i]['token_ids'][-1] for i in eid]
 fn=lambda:e.draft_tree_forward(parents,roots,eid)
 parent,seen=observer169.run(e,fn) if observe_parent else (fn(),None)
 KV=epoch.read(e,parent_slots(e,eid,p)) if observe_parent else None
 plans=[];workroots=[];catchups=[];sources={};confidences={}
 for i in range(16):
  tokens=cs[i]['token_ids'];plans.append(plan(2,4,len(tokens)-1,4096,candidate_budget=4));workroots.append(tokens[-1]);catchups.append(tokens[p:])
 for row,i in enumerate(eid):
  packed=pack_selected_tree_plan(parents[row],range(4));path=tuple(tree_primary_path(packed))
  require(path==(0,1,2,3),'only genuine linear parent4')
  values=list(map(int,parent['draft_token_ids'][row][:4]))
  child=e._spec_rhythm_tree_eager_plan(packed,type('Prefix',(),{'token_ids':cs[i]['token_ids']})(),3)
  require(child.prefix_len==p+5,'original generic child prefix follows full parent4')
  plans.append(child);workroots.append(roots[row]);catchups.append(())
  sources[i]=(packed,values);confidences[i]=sum(float(v) for v in parent['node_confidences'][row][:4])/4
 return dict(contexts=cs,plans=plans,roots=workroots,ids=list(range(32)),sources=sources,catchups=catchups,
  parent_output=cpu_output(parent),parent_confidences=confidences,parent_observed=seen,parent_KV=KV,
  truthful_ledger=epoch.ledger(e),source_epoch=e._v151_source_epoch,source_lease=copy.deepcopy(e._v151_source_lease),
  parent_actual_execution=dict(actual_steps=4,actual_chain_M=16,query_count=parent['query_count'],graph_calls=parent['graph_calls'],model_calls=parent['model_calls']))

def invoke(e,f):
 require(f['source_epoch']==e._v151_source_epoch,'current actual parent4 source epoch')
 before=_TTU(e)
 output=e.draft_tree_forward(f['plans'],f['roots'],f['ids'],eager_parent_sources=f['sources'],committed_catchup_token_ids=f['catchups'])
 after=_TTU(e)
 require(set(before['counters'])==set(after['counters']) and all(after['counters'][k]>=v for k,v in before['counters'].items()),'actual TTU monotonic')
 e._v152_last_mixed_TTU=dict(before=before,after=after,delta={k:after['counters'][k]-v for k,v in before['counters'].items()},
  measurement='untimed actual parent4-prelude/shared4 call only',dynamic_TTU_proved=False,dynamic_prefix_cost_measured=False)
 return output

def counts(case):return [4+case.repair_vector[i] for i in range(16)]+[9]*16+[0]*32

def independent(e,case,f):
 from kv_ledger import record_overwrite
 p=case.prefix;eid=list(range(16,32));baseline=epoch.protect(e,[p]*64)
 encoded_parent,hidden_parent=chain(e,eid,[p]*16,[f['roots'][i] for i in eid],4,16)
 parent_rows,_=decode(encoded_parent,16,4);parent_KV=epoch.read(e,parent_slots(e,eid,p))
 ledger={i:dict(safe_before=p,tokens={}) for i in range(64)}
 for j,i in enumerate(eid):
  require(parent_rows[j][:4]==f['sources'][i][1],'fresh independent true parent4 predictions')
  ledger[i]['tokens']={p+k:t for k,t in enumerate([f['roots'][i],*parent_rows[j][:3]])}
 # Independent ORIGINAL PA on actual frontier+repair owners; no union builder.
 ids=eid+list(case.repair_owners);positions=[p+4]*16+[p]*case.repair_count;tokens=[r[3] for r in parent_rows]+[f['catchups'][i][0] for i in case.repair_owners]
 call=PA(e,ids,positions,tokens,case.union_M);union=e.model(*call).clone()
 frontier=e.model.compute_greedy_tokens(union[:16],e.draft_vocab_size).clone()
 frontiers={i:int(frontier[j]) for j,i in enumerate(eid)}
 for i,row in zip(eid,parent_rows,strict=True):record_overwrite(ledger[i],p+4,[row[3]])
 roots=[*f['roots'][:16],*[frontiers[i] for i in eid]];prefixes=[q.prefix_len for q in f['plans']]
 encoded,hidden=chain(e,list(range(32)),prefixes,roots,4,32);rows,scores=decode(encoded,32,4)
 for i,(prefix,root,row) in enumerate(zip(prefixes,roots,rows,strict=True)):
  written=[root,*row[:3]]
  if i<16:ledger[i]=dict(safe_before=prefix,tokens={prefix+k:t for k,t in enumerate(written)})
  else:record_overwrite(ledger[i],prefix,written)
 require(epoch.protect(e,[p]*64)==baseline,'independent parent4 history preserves all64 actual source prefixes')
 return dict(rows=rows,scores=scores,encoded=encoded,hidden=hidden,union_hidden=union,union_call=call,
  frontier_tokens=frontier,frontiers=frontiers,roots=roots,ledger=ledger,parent_hidden=hidden_parent,
  parent_encoded=encoded_parent,parent_KV=parent_KV,source_epoch=e._v151_source_epoch)

def compare(e,case,a,ref,f):
 import torch
 match=e.graph_runner._target_outputs_match;seen=a['observed'];out=a['output'];ownKV=epoch.logical_KV(e,[128]*64,counts(case))
 require(len(seen['draft'])==1,'one actual shared4 chain')
 if case.repair_count:
  require(len(seen['hidden'])==1 and not seen['greedy'] and len(seen['union_head'])==1,'one union hidden and actual F head')
  prelude=seen['hidden'][0];head=seen['union_head'][0]
 else:
  require(not seen['hidden'] and len(seen['greedy'])==1,'R0 retains original frontier greedy path')
  prelude=seen['greedy'][0];head=dict(M=16,tokens=prelude['tokens'])
 require(tuple(seen['draft'][0]['encoded'].shape)==(32,4,2) and len(seen['draft'][0]['hidden'])==4,'true allphysical shared4 raw8')
 finite_a=require_finite('actual parent4 union/shared4 all56KV',[prelude['hidden'],*seen['draft'][0]['hidden']],a['KV'])
 finite_r=require_finite('independent parent4 union/shared4 all56KV',[ref['union_hidden'],*ref['hidden']],ownKV)
 require(match((prelude['hidden'],),(ref['union_hidden'],)) and match(seen['draft'][0]['hidden'],ref['hidden'])
  and torch.equal(seen['draft'][0]['encoded'],ref['encoded']) and match(a['KV'],ownKV) and len(a['KV'])==len(ownKV)==56,
  'own allphysical SAME-U20/M32 hidden/encoded/all56 written KV')
 metadata=compare_fresh(e,case,a['metadata'],ref['union_call'])
 require(head['M']==16 and torch.equal(head['tokens'],ref['frontier_tokens']),'original F16 exact full frontier head')
 require(out['draft_token_ids']==ref['rows'] and out['root_token_ids']==ref['roots'] and out['eager_frontier_tokens']==ref['frontiers']
  and a['ledger']==ref['ledger'],'exact independent canonical8/root/frontier/ledger')
 require(all(torch.equal(v,torch.tensor(s)) for v,s in zip(out['node_confidences'],ref['scores'],strict=True))
  and torch.equal(out['draft_confidence'],torch.tensor([sum(s)/8 for s in ref['scores']])),'exact score8 and denominator8 confidence')
 parent=f['parent_observed']
 require(len(parent['draft'])==1 and len(parent['draft'][0]['hidden'])==4
  and match(parent['draft'][0]['hidden'],ref['parent_hidden']) and torch.equal(parent['draft'][0]['encoded'],ref['parent_encoded'])
  and match(f['parent_KV'],ref['parent_KV']),'fresh independent allphysical genuine parent4 M16 hidden/encoded/all56KV')
 for kind in ('draft','hidden','greedy'):
  for row in seen[kind]:
   entry=(e.graph_runner.draft_entries if kind=='draft' else e.graph_runner.target_entries)[row['key']]
   require(entry.runtime_validated is True,'actual current natural Native gate')
   if kind=='draft':require(entry.validated_real_row_count>=32,'true32 real-row Native qualification')
 from bank169 import tensor_hash
 return dict(sameM_pass=True,parent4_actual_independent_match=True,allphysical_union_hidden=True,
  allphysical_fourstep_hidden=True,full_encoded=True,all56_written_KV=True,exact_canonical8_scores_and_ledger=True,
  independent_metadata=metadata,finite_actual=finite_a,finite_reference=finite_r,actual_parent_B=4,child_B=3,
  parent_steps=4,shared_steps=4,M=32,U=case.union_M,F=16,R=case.repair_count,source_frontier_position=132,child_prefix=133,
  full_physical_encoded=seen['draft'][0]['encoded'].cpu().tolist(),reference_full_physical_encoded=ref['encoded'].cpu().tolist(),actual_canonical8=out['draft_token_ids'],
  reference_score8=ref['scores'],actual_score8=[v.cpu().tolist() for v in out['node_confidences']],actual_confidence=out['draft_confidence'].cpu().tolist(),actual_frontier=out['eager_frontier_tokens'],
  actual_root_tokens=out['root_token_ids'],reference_root_tokens=ref['roots'],
  actual_full_head_tokens=head['tokens'].cpu().tolist(),actual_parent_full_encoded=parent['draft'][0]['encoded'].cpu().tolist(),reference_parent_full_encoded=ref['parent_encoded'].cpu().tolist(),actual_parent_canonical8=f['parent_output']['draft_token_ids'],
  candidate_KV_sha256=[tensor_hash(v) for v in a['KV']],reference_KV_sha256=[tensor_hash(v) for v in ownKV],parent_candidate_KV_sha256=[tensor_hash(v) for v in f['parent_KV']],parent_reference_KV_sha256=[tensor_hash(v) for v in ref['parent_KV']],
  allphysical_hidden_sha256=[tensor_hash(prelude['hidden']),*[tensor_hash(v) for v in seen['draft'][0]['hidden']]],reference_allphysical_hidden_sha256=[tensor_hash(ref['union_hidden']),*[tensor_hash(v) for v in ref['hidden']]],crossM_behavior_equal=None,cost_advantage=None)

def prepare(e,case,stage,changed=False):
 import observer169
 cs=stage('source-contexts',lambda:contexts(case,changed));epoch.rebuild(e,cs,list(case.repair_vector),stage,role='draft')
 f=stage('actual-parent4-source',lambda:fixture(e,case,changed,observe_parent=True) if e.is_draft else None)
 def actual():
  if not e.is_draft:return None
  prefix=epoch.protect(e,[128]*64);out,seen=observer169.run(e,lambda:invoke(e,f))
  metadata=capture_actual(e,case,f,seen if case.repair_count else dict(hidden=seen['greedy']),PA)
  return dict(output=out,observed=seen,KV=epoch.logical_KV(e,[128]*64,counts(case)),ledger=epoch.ledger(e),prefix=prefix,metadata=metadata)
 a=stage('actual-parent4-frontier-shared4-snapshot',actual)
 epoch.rebuild(e,cs,list(case.repair_vector),stage,role='draft')
 ref=stage('fresh-independent-parent4-and-union-sameM',lambda:independent(e,case,f) if e.is_draft else None)
 proof=stage('complete-parent4-and-shared4-independent-gate',lambda:compare(e,case,a,ref,f) if e.is_draft else None)
 epoch.rebuild(e,cs,list(case.repair_vector),stage,role='draft')
 restored=stage('whole-actual-parent4-source-restore',lambda:fixture(e,case,changed) if e.is_draft else None)
 def restore():
  if not e.is_draft:return None
  out=invoke(e,restored)
  require(cpu_output(out)==cpu_output(a['output']) and epoch.ledger(e)==a['ledger']
   and e.graph_runner._target_outputs_match(a['KV'],epoch.logical_KV(e,[128]*64,counts(case)))
   and epoch.protect(e,[128]*64)==a['prefix'],'entire actual parent4/shared4 history restored before original target commit')
  return dict(case=case.__dict__,fixture=restored,mixed=cpu_output(out),numerical_proof=proof,
   actual_source_lease=restored['source_lease'],controlled_grants_not_live_policy=True,adopted=False,Goodput_measured=False,
   generation_observation=dict(parent_steps=4,actual_shared_steps=4,shared_M=32,union_M=case.union_M,head_M=16,real_raw_nodes=256,selected_normal=64,selected_eager=48,selected_total=112,raw6_padding_substitution=False))
 trace=stage('actual-whole-parent4-source-restoration-gate',restore)
 def share():
  import torch.distributed as dist
  packet=[trace if e.is_draft else None];dist.broadcast_object_list(packet,src=e.topology.draft_leader_rank,group=e.groups.verification_coordination_group)
  require(packet[0] is not None,'real current parent4 draft-produced rows shared')
  return packet[0]
 return stage('share-actual-parent4-typed-source',share)
