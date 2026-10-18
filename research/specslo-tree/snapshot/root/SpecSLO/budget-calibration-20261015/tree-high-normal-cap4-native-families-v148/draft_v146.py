"""Real variable-B FastV5 work, independent same-M eager PA and exact ABI."""
import copy
from scope_v146 import require,digest
from dual_bank_v146 import _arrays,_slots,prefix_evidence,query_KV

def plans_for(contexts,budgets):
 from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan
 return [cached_cpu_tree_speculation_plan(2,4,len(v)-1,4096,candidate_budget=b)
  for v,b in zip(contexts,budgets,strict=True)]

def work_plan(case,ordinal):
 from split_budget_v132 import DraftWorkPlan,DraftWorkLimit,TargetAdmissionLimit
 budgets=dict(enumerate(case['budgets']))
 return DraftWorkPlan(plan_id=ordinal,target_limit=TargetAdmissionLimit(),work_limit=DraftWorkLimit(),
  available_selected_candidates=192,requested_normal=tuple(range(32)),admitted_eager=(),
  normal_budgets=budgets,eager_budgets={},progress_gaps=dict.fromkeys(budgets,0.),eager_priorities={},
  caps=dict.fromkeys(budgets,4),needs=budgets,reserve_requested=0,reserve_used=0)

def reference_entries(bank):
 result=[]
 for i,values in enumerate(bank.reference_contexts):
  owner=i+32;p=len(values)-1;known=bank.reference_ledger[owner];start=p
  require(0<=known['safe_before']<=p,'reference known-KV prefix bound')
  for position in range(known['safe_before'],p):
   if known['tokens'].get(position)!=values[position]:start=position;break
  require(p-start<=1,'at most one truthful B4 deferred accepted token')
  result.append((owner,p,start,values[start:p+1]))
 return result

def eager_reference(engine,native,bank):
 import torch
 from types import SimpleNamespace
 from chain_metadata import build
 entries=reference_entries(bank);repair=any(len(v)>1 for _,_,_,v in entries)
 physical=native._next_linear_draft_graph_bucket(sum(len(v) for _,_,_,v in entries),128) if repair else 32
 require(physical==32 if not repair else physical in (36,40,44,48,64),'declared real grouped repair geometry')
 histories={i:bank.reference_contexts[i-32][:start] for i,_,start,_ in entries};before=prefix_evidence(engine,histories)
 ids=[];positions=[];tokens=[];roots=[]
 for owner,p,start,values in entries:
  ids.extend([owner]*len(values));positions.extend(range(start,p+1));tokens.extend(values);roots.append(len(tokens)-1)
 x=torch.tensor(tokens,dtype=torch.long,device=engine.device)
 pos,meta=engine._prepare_attention_metadata(ids,positions,False)
 x,pos,meta=native.NativeACLGraphRunner._pad_inputs(x,pos,meta,physical)
 require(not meta.use_fused_infer_attention and meta.context_lens.tolist()==[p+1 for p in positions]+[1]*(physical-len(tokens))
  and meta.slot_mapping.tolist()[len(tokens):]==[-1]*(physical-len(tokens)),'independent actual PA padding')
 first=engine.model(x,pos,meta).clone()
 selected=first
 if repair:
  selected=first.new_zeros((32,first.shape[-1]));selected.copy_(first.index_select(0,torch.tensor(roots,dtype=torch.long,device=engine.device)))
 def top2(hidden):
  logits=engine.model.compute_logits(hidden)[:,:engine.draft_vocab_size]
  probabilities=torch.softmax(logits.float(),dim=-1);confidence,token=probabilities.topk(2,dim=-1)
  return token.long()*1_000_001+torch.round(confidence.clamp(0,1)*1_000_000).long(),token[:,0]
 encoded,inputs=top2(selected);values=[encoded];tail=[]
 _,pp,mm=build(engine,[SimpleNamespace(prefix_len=p+1) for _,p,_,_ in entries],[i for i,_,_,_ in entries],[0]*32,3,32)
 for pos,meta in zip(pp,mm,strict=True):
  hidden=engine.model(inputs,pos,meta).clone();tail.append(hidden)
  encoded,inputs=top2(hidden);values.append(encoded)
 encoded=torch.stack(values,dim=1);pairs=encoded.cpu().tolist()
 rows=[[pair[0]//1_000_001 for pair in row]+[pair[1]//1_000_001 for pair in row] for row in pairs]
 scores=[[pair[0]%1_000_001/1_000_000 for pair in row]+[pair[1]%1_000_001/1_000_000 for pair in row] for row in pairs]
 ledger={owner:dict(safe_before=p,tokens={p+j:t for j,t in enumerate([values[-1]]+row[:3])})
  for (owner,p,_,values),row in zip(entries,rows,strict=True)}
 index=torch.cat([_slots(engine,owner,range(start,p+4)) for owner,p,start,_ in entries]);queries=query_KV(engine,index)
 require(prefix_evidence(engine,histories)==before,'independent draft preserves its valid prefix')
 require(all(bool(torch.isfinite(v).all()) for v in (first,*tail,*queries)),'independent full-physical finite')
 bank.reference_ledger=copy.deepcopy(ledger)
 return dict(first=first,tail=tuple(tail),encoded=encoded,rows=rows,scores=scores,ledger=ledger,
  query_KV=queries,starts=[v[2] for v in entries],repair=repair,first_physical=physical,
  repair_count=sum(len(v)-1 for _,_,_,v in entries))

def generate(engine,native,bank,case,ordinal,stage):
 import torch
 from draft_observer_v146 import actual_draft_observer,_used_KV
 source=stage('source',bank.source);plans=stage('variable-work-plans',lambda:plans_for(bank.contexts,case['budgets']))
 plan=stage('typed-controlled-grants',lambda:work_plan(case,ordinal))
 stage('typed-actual-work',lambda:plan.require_work(source['ids'],case['budgets'],[False]*32,plans))
 def candidate():
  if not engine.is_draft:return None
  engine._defer_tree_materialization=engine._defer_normal_tree_materialization=True
  before=copy.deepcopy(engine._dynamic_tree_known_KV);starts=[]
  for i,v in enumerate(bank.contexts):
   p=len(v)-1;start=p
   for j in range(before[i]['safe_before'],p):
    if before[i]['tokens'].get(j)!=v[j]:start=j;break
   starts.append(start)
  with actual_draft_observer(engine,source) as capture:
   output=engine.draft_tree_forward(plans,[v[-1] for v in bank.contexts],source['ids'],
    committed_catchup_token_ids=[v[-5:] for v in bank.contexts])
  require(capture['tail'] is not None,'actual four-step or root+three-step draft snapshot')
  repaired=output.get('execution_mode')=='normal-packed-root-seed-top2-v1'
  hidden=capture['tail']['hidden'];encoded=capture['tail']['encoded']
  if repaired:
   require(capture['first'] is not None and capture['seed'] is not None and len(hidden)==3,'actual root repair first/head/tail')
   first=capture['first']['hidden'];tail=hidden;encoded=torch.cat((capture['seed'][:,None,:],encoded),dim=1)
  else:
   require(capture['first'] is None and len(hidden)==4,'actual no-repair four-step inventory')
   first,tail=hidden[0],hidden[1:]
  return dict(output=output,first=first,tail=tail,encoded=encoded,
   query_KV=_used_KV(engine,source['ids'],starts,[len(v)+3 for v in bank.contexts]),
   ledger=copy.deepcopy(engine._dynamic_tree_known_KV),repair=repaired,
   controls=capture['controls'],starts=starts)
 actual=stage('actual-draft',candidate)
 reference=stage('independent-draft',lambda:eager_reference(engine,native,bank) if engine.is_draft else None)
 def compare():
  if not engine.is_draft:return None
  a,r=actual,reference;o=a['output'];matcher=engine.graph_runner._target_outputs_match
  require(a['repair']==r['repair'] and a['starts']==r['starts'] and o['draft_token_ids']==r['rows']
   and o['root_token_ids']==[v[-1] for v in bank.contexts] and o['num_draft_tokens']==case['budgets']
   and o['eager_frontier_tokens']=={} and tuple(a['encoded'].shape)==(32,4,2)
   and torch.equal(a['encoded'],r['encoded']) and matcher((a['first'],),(r['first'],))
   and matcher(tuple(a['tail']),r['tail']) and len(a['query_KV'])==56 and matcher(a['query_KV'],r['query_KV']),
   'same-M all physical draft hidden/encoded/all28queryKV and exact original output')
  for i,scores in enumerate(r['scores']):
   require(torch.equal(o['node_confidences'][i].cpu(),torch.tensor(scores,dtype=o['node_confidences'][i].dtype,device='cpu'))
    and a['ledger'][i]==r['ledger'][i+32],'exact full8 confidences/independent truthful ledger')
  require(torch.equal(o['draft_confidence'].cpu(),torch.tensor([sum(v)/8 for v in r['scores']],dtype=o['draft_confidence'].dtype,device='cpu')),'exact original mean8 confidence')
  return dict(all_physical_hidden_original_matcher=True,all28_query_KV_original_matcher=True,
   exact_full8_encoded=True,raw_encoded_shape=[32,4,2],raw_pool_nodes=256,actual_draft_model_stages=4,
   selected_C=case['C'],variable_num_draft_tokens=o['num_draft_tokens'],roots=o['root_token_ids'],
   raw_pool=o['draft_token_ids'],node_confidences=[v.cpu().tolist() for v in o['node_confidences']],
   mean_confidence=o['draft_confidence'].cpu().tolist(),encoded_sha256=digest(a['encoded'].cpu().tolist()),
   natural_missing_suffix_observed=r['repair'],natural_missing_suffix_rows=r['repair_count'],
   first_PA_physical=r['first_physical'],controls=a['controls'],independent_owners=list(range(32,64)),
   independent_history=True,source_KV_copied=False)
 proof=stage('independent-draft-original-matcher-and-exact-ABI',compare)
 def choose():
  if not engine.is_draft:return None
  from vllm_ascend.spec_decode.pearl.tree_budget import TreeCandidateRequest
  from backbone_selection import select
  from split_budget_v132 import refine_draft_groups
  output=actual['output'];requests={i:TreeCandidateRequest(parents=p.parent_indices.tolist(),token_ids=row,
   conditional_confidences=scores.cpu().tolist(),max_candidates=b,minimum_candidates=b)
   for i,p,row,scores,b in zip(source['ids'],plans,output['draft_token_ids'],output['node_confidences'],case['budgets'],strict=True)}
  selection=refine_draft_groups(requests,plan,select)
  selected=[list(selection.selected_indices[i]) for i in source['ids']]
  rows=[[row[j] for j in chosen] for row,chosen in zip(output['draft_token_ids'],selected,strict=True)]
  confidence=[sum(float(scores[j]) for j in chosen)/len(chosen) for scores,chosen in zip(output['node_confidences'],selected,strict=True)]
  selection.require_published({i:len(v) for i,v in zip(source['ids'],rows,strict=True)})
  return rows,selected,confidence
 chosen=stage('original-two-role-refine',choose)
 rows,selected,confidence=chosen if engine.is_draft else (None,None,None)
 packet=engine._exchange_spec_rhythm_tree_candidates(rows,plans,[None]*32 if engine.is_draft else None,
  confidence,selected_indices=selected,draft_compute_ms=0.)
 def split():
  nonlocal rows,selected,confidence
  from vllm_ascend.spec_decode.pearl.tree import pack_selected_tree_plan
  if not engine.is_draft:
   rows,frontier,confidence,selected,_=engine._split_tree_candidates(packet,plans)
   require(all(v is None for v in frontier),'original five-item packet, no eager')
  require([len(v) for v in rows]==case['budgets'] and selected==[list(range(b)) for b in case['budgets']],
   'actual selected chain and retained variable grants')
  return [pack_selected_tree_plan(p,i) for p,i in zip(plans,selected,strict=True)],rows,confidence,selected
 packed,rows,confidence,selected=stage('original-five-item-exchange-split',split)
 return dict(source=source,plans=packed,rows=rows,confidences=confidence,selected=selected,work_plan=plan,draft_proof=proof)
