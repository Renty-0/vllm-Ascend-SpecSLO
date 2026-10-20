"""Finite real new-usage representatives, independent same-U/full chain oracle.

Includes E3/F4/R1/U8 interior padding. Independent reference starts with one
ORIGINAL PA producer on E+R real rows and original tail pad, then reorders its
own dummy rows/metadata; it never calls the candidate union constructor.
"""
import copy
from contextlib import contextmanager
from dataclasses import replace
from base165 import require
from bank165 import Case,build_bank,build_parents,finite,tensor_hash

CASES=(Case('N1-E1-R1',1,1),Case('N1-E3-R1-interior-padding',1,3),Case('N32-E32-R32',32,32))
NORMAL_B=(4,)*19+(3,)+(2,)*5+(1,)*7

def independent_frame(e,bank,native,offset):
 import torch
 repairs=bank.parents['reference_repairs' if offset else 'repairs']
 frontiers=bank.parents['reference_frontiers' if offset else 'frontiers']
 E,R=len(frontiers),len(repairs);F=native._next_linear_draft_graph_bucket(E,64)
 U=native._next_linear_draft_graph_bucket(F+R,64)
 ids=[x[0] for x in frontiers]+[x[0] for x in repairs]
 positions=[x[1] for x in frontiers]+[x[1] for x in repairs]
 tokens=[x[2] for x in frontiers]+[x[2][0] for x in repairs]
 # Original PA prepares E+R real rows, then original Native pad supplies all
 # finite read-only dummy rows. Move its own tail dummies into the interior.
 x,p,m=bank.math['PA'](e,ids,positions,tokens,U)
 order=list(range(E))+list(range(E+R,E+R+F-E))+list(range(E,E+R))+list(range(E+R+F-E,U))
 require(len(order)==len(set(order))==U,'independent complete physical permutation')
 ix=torch.tensor(order,dtype=torch.long,device=e.device)
 m=replace(m,slot_mapping=m.slot_mapping.index_select(0,ix),context_lens=m.context_lens.index_select(0,ix),
  block_tables=m.block_tables.index_select(0,ix),actual_seq_lengths_q=tuple(range(1,F+R+1)),
  sequence_lens=tuple(m.sequence_lens[i] for i in order))
 return x.index_select(0,ix),p.index_select(0,ix),m

def metadata_record(frame):
 from metadata161 import snapshot
 return snapshot(*frame)

def actual_metadata_gate(e,bank,native,observed):
 from metadata161 import observed_snapshot,FIELDS
 expected=metadata_record(independent_frame(e,bank,native,0))
 actual=observed_snapshot(observed)
 checks={k:actual[k]==expected[k] for k in FIELDS}
 E,R=bank.case.eager,bank.case.normal;F=native._next_linear_draft_graph_bucket(E,64);U=native._next_linear_draft_graph_bucket(F+R,64)
 real_rows=list(range(E))+list(range(F,F+R));dummy=sorted(set(range(U))-set(real_rows))
 owners=list(range(R,R+E))+list(range(R));positions=[131]*E+[128]*R
 tables=e.cache_allocation.block_tables;slots=[tables[i][p//128]*128+p%128 for i,p in zip(owners,positions,strict=True)]
 own=dict(real_slots=[actual['slot_mapping'][i] for i in real_rows]==slots,
  real_tables=[actual['block_tables'][i] for i in real_rows]==[tables[i] for i in owners],
  dummy_slots=[actual['slot_mapping'][i] for i in dummy]==[-1]*len(dummy),
  logical_Q=actual['host_Q']==list(range(1,F+R+1)),
  logical_KV=actual['host_KV']==[132]*E+[1]*(F-E)+[129]*R+[1]*(U-F-R))
 result=dict(E=E,R=R,F=F,U=U,real_rows=real_rows,dummy_rows=dummy,field_equal=checks,own_lease=own,
  actual=actual,independent_original_PA_reordered=expected,independent_of_candidate_union_builder=True)
 e._v165_partial['actual_metadata']=result
 require(all(checks.values()) and all(own.values()),'allphysical actual own-lease original-PA metadata/real+interior+tail padding')
 return result

def query(bank,offset):
 import torch
 c=bank.case
 indices=[bank.kv['slots'](bank.e,offset+i,range(128,133 if i<c.normal else 136)) for i in range(c.count)]
 return bank.kv['read'](bank.e,torch.cat(indices))

@contextmanager
def observe_head(e):
 old=e.model.compute_greedy_tokens;had='compute_greedy_tokens' in vars(e.model);prior=vars(e.model).get('compute_greedy_tokens');seen=[]
 def head(hidden,vocabulary):
  value=old(hidden,vocabulary);seen.append(dict(M=hidden.shape[0],tokens=value.detach().clone()));return value
 e.model.compute_greedy_tokens=head
 try:yield seen
 finally:
  if had:e.model.compute_greedy_tokens=prior
  else:vars(e.model).pop('compute_greedy_tokens',None)

def fixture(e,bank):
 from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan as plan,pack_selected_tree_plan
 c=bank.case;normal=list(NORMAL_B) if c.normal==32 else [4]*c.normal
 require(sum(normal)<=96 and sum(normal)+3*c.eager<=192,'typed candidate roof in genuine mixed fixture')
 plans=[];roots=[];sources={};catchups=[]
 for i in range(c.normal):
  values=bank.contexts[i]['token_ids'];plans.append(plan(2,4,129,4096,candidate_budget=normal[i]));roots.append(values[-1]);catchups.append(values[128:])
 for row,i in enumerate(range(c.normal,c.count)):
  parent=pack_selected_tree_plan(plan(2,4,128,4096,candidate_budget=3),[0,1,2])
  child=e._spec_rhythm_tree_eager_plan(parent,type('Prefix',(),{'token_ids':bank.contexts[i]['token_ids']})(),3)
  plans.append(child);roots.append(bank.contexts[i]['token_ids'][-1]);catchups.append(())
  sources[i]=(parent,list(bank.parents['original_parent_rows'][row]))
 return plans,roots,list(c.candidate_ids),sources,catchups

def actual(e,bank,native,changed=False):
 import torch
 import observer165 as observer151
 plans,roots,ids,sources,catchups=fixture(e,bank)
 if changed:
  catchups=copy.deepcopy(catchups);catchups[0][0]=(catchups[0][0]+1)%e.draft_vocab_size
  sources=copy.deepcopy(sources)
  first=min(sources);p,row=sources[first];row[2]=(row[2]+1)%e.draft_vocab_size
 # Unpublished F/T/F control resets only the proven logical source ledger.
 # Source prefix/parent KV is unchanged; all future real queries are rewritten.
 e._dynamic_tree_known_KV=copy.deepcopy(bank._v165_source_ledger)
 with observe_head(e) as heads:
  out,seen=observer151.run(e,lambda:e.draft_tree_forward(plans,roots,ids,
   eager_parent_sources=sources,committed_catchup_token_ids=catchups))
 require(len(seen['hidden'])==len(seen['draft'])==1 and not seen['greedy'] and len(heads)==1,
  'one union hidden/F head and one genuine four-step graph')
 U=native._next_linear_draft_graph_bucket(native._next_linear_draft_graph_bucket(bank.case.eager,64)+bank.case.normal,64)
 M=native._next_linear_draft_graph_bucket(bank.case.count,64)
 require(seen['hidden'][0]['hidden'].shape[0]==U and len(seen['draft'][0]['hidden'])==4
  and tuple(seen['draft'][0]['encoded'].shape)==(M,4,2) and out['model_calls']==out['graph_calls']==2,
  'all physical union and real raw8 four-step source, reduced two total graph submits')
 return dict(output=out,seen=seen,heads=heads,KV=query(bank,0),ledger=copy.deepcopy(e._dynamic_tree_known_KV))

def own_reference(e,bank,native):
 import torch
 from kv_ledger import record_overwrite
 c=bank.case;F=native._next_linear_draft_graph_bucket(c.eager,64);M=native._next_linear_draft_graph_bucket(c.count,64)
 call=independent_frame(e,bank,native,64);hidden=e.model(*call).clone()
 ft=e.model.compute_greedy_tokens(hidden[:F],e.draft_vocab_size).clone()
 frontiers={i:int(ft[j]) for j,i in enumerate(range(c.normal,c.count))}
 roots=[bank.contexts[i]['token_ids'][-1] if i<c.normal else frontiers[i] for i in range(c.count)]
 prefixes=[129]*c.normal+[132]*c.eager
 encoded,chain=bank.math['chain'](e,list(c.reference_ids),prefixes,roots,4,M)
 rows,scores=bank.math['decode'](encoded,c.count,4)
 ledger={i:dict(safe_before=128,tokens={}) for i in range(c.count)}
 for row,i in enumerate(range(c.normal,c.count)):
  # Parent has its own exact allphysical hidden/encoded/KV gate above.
  parent=bank.parents['original_parent_rows'][row];root=bank.contexts[i]['token_ids'][-1]
  ledger[i]['tokens']={128:root,129:parent[0],130:parent[1]}
  record_overwrite(ledger[i],131,[parent[2]])
 for i,(prefix,root,row) in enumerate(zip(prefixes,roots,rows,strict=True)):
  written=[root,*row[:3]]
  if i<c.normal:ledger[i]=dict(safe_before=prefix,tokens={prefix+k:v for k,v in enumerate(written)})
  else:record_overwrite(ledger[i],prefix,written)
 return dict(union=hidden,head=ft,encoded=encoded,chain=chain,rows=rows,scores=scores,roots=roots,frontiers=frontiers,
  KV=query(bank,64),ledger=ledger,frame=call)

def compare(e,bank,native,a,r):
 import torch
 match=e.graph_runner._target_outputs_match;c=bank.case;seen=a['seen'];out=a['output']
 finite((*[seen['hidden'][0]['hidden']],*seen['draft'][0]['hidden'],*a['KV'],r['union'],*r['chain'],*r['KV']))
 require(match((seen['hidden'][0]['hidden'],),(r['union'],)) and match(seen['draft'][0]['hidden'],r['chain'])
  and torch.equal(seen['draft'][0]['encoded'],r['encoded']) and len(a['KV'])==len(r['KV'])==56 and match(a['KV'],r['KV']),
  'own same-U allphysical union/four-step hidden/encoded/all56 written queryKV')
 require(out['draft_token_ids']==r['rows'] and out['root_token_ids']==r['roots'] and out['eager_frontier_tokens']==r['frontiers']
  and all(torch.equal(v,torch.tensor(s)) for v,s in zip(out['node_confidences'],r['scores'],strict=True))
  and torch.equal(out['draft_confidence'],torch.tensor([sum(s)/8 for s in r['scores']]))
  and a['ledger']==r['ledger'],'whole canonical8/exact score8 and denominator8/independent ledger')
 F=native._next_linear_draft_graph_bucket(c.eager,64)
 require(a['heads'][0]['M']==F and torch.equal(a['heads'][0]['tokens'],r['head']),'original head exactly F, whole physical head output')
 bank.guard_full()
 return dict(sameM_pass=True,allphysical_union_hidden=True,allphysical_fourstep_hidden=True,full_encoded=True,all56_written_KV=True,
  exact_canonical8_scores_and_ledger=True,independent_parent_and_reference_pages=True,real_N=c.normal,real_E=c.eager,real_R=c.normal,
  F=F,U=r['union'].shape[0],M=r['encoded'].shape[0],head_M=F,model_calls=2,graph_calls=2,
  hidden_sha256=tensor_hash(seen['hidden'][0]['hidden']),encoded_sha256=tensor_hash(seen['draft'][0]['encoded']),
  reference_hidden_sha256=tensor_hash(r['union']),reference_encoded_sha256=tensor_hash(r['encoded']),
  full_physical_encoded=seen['draft'][0]['encoded'].cpu().tolist(),reference_full_physical_encoded=r['encoded'].cpu().tolist(),
  canonical8=out['draft_token_ids'],reference_canonical8=r['rows'],frontier_tokens=out['eager_frontier_tokens'],
  actual_score8=[v.cpu().tolist() for v in out['node_confidences']],reference_score8=r['scores'],actual_confidence=out['draft_confidence'].cpu().tolist(),
  actual_root_tokens=out['root_token_ids'],reference_root_tokens=r['roots'],
  actual_full_head_tokens=a['heads'][0]['tokens'].cpu().tolist(),reference_full_head_tokens=r['head'].cpu().tolist(),
  allphysical_KV_sha256=[tensor_hash(v) for v in a['KV']],reference_allphysical_KV_sha256=[tensor_hash(v) for v in r['KV']],
  reference_metadata=metadata_record(r['frame']),arbitrary_token_equivalence_claimed=False,
  teacher_forced_normal_suffix=True,genuine_parent3=True,target_commit_claimed=False)

def run_cases(e,native,stage):
 import sys
 helpers=dict(fast=sys.modules['fast_tree_draft_v5'],tail=sys.modules['tail_hidden_observer'])
 results=[]
 for c in CASES:
  lane=lambda n,f:stage(c.name+'/'+n,f)
  bank,lease=build_bank(e,c,False,lane)
  parent=lane('genuine-parent3-and-independent-reference',lambda:build_parents(e,native,helpers,bank,lane) if e.is_draft else None)
  def source():
   if e.is_draft:bank._v165_source_ledger=copy.deepcopy(e._dynamic_tree_known_KV)
  lane('capture-proven-parent-source-ledger',source);outputs=[]
  for i,change in enumerate((False,True,False)):
   outputs.append(lane('new-usage-FTF'+str(i),lambda:actual(e,bank,native,change) if e.is_draft else None))
  def actual_gate():
   if not e.is_draft:return None
   first,last=outputs[0],outputs[2];match=e.graph_runner._target_outputs_match
   require(first['output']['draft_token_ids']==last['output']['draft_token_ids'] and torch_equal(first['seen']['draft'][0]['encoded'],last['seen']['draft'][0]['encoded'])
    and match(first['KV'],last['KV']) and first['ledger']==last['ledger'],'actual original-input restoration after unpublished changed input')
   bank.guard_full()
   proof=actual_metadata_gate(e,bank,native,last['seen']['hidden'][0]['metadata'])
   padding=last['seen']['hidden'][0]['padding']
   proof['padding_diagnostic']=None if padding is None else {k:v for k,v in padding.items() if k!='prefix0_KV'}
   if padding is not None:proof['padding_diagnostic']['prefix0_KV_sha256']=[tensor_hash(v) for v in padding['prefix0_KV']]
   return proof
  metadata=lane('actual-allphysical-FTF-own-metadata-and-prefix',actual_gate)
  reference=lane('fresh-independent-same-U-reordered-PA-and-fourstep',lambda:own_reference(e,bank,native) if e.is_draft else None)
  proof=lane('own-complete-physical-numeric-canonical8-KV-ledger',lambda:compare(e,bank,native,outputs[-1],reference) if e.is_draft else None)
  results.append(dict(case=c.__dict__,lease=lease,parent=parent,metadata=metadata,numerical=proof))
  lane('release-both-independent-fixture-banks',e._release_cache)
 return results

def torch_equal(a,b):
 import torch
 return torch.equal(a,b)
